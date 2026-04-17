"""Orthogonal Gradient Projection using compression-defined subspaces.

After each task, identify the "consolidated" subspace (high bit-depth channels).
For subsequent tasks, project gradients to be orthogonal to the consolidated
subspace's representation. This prevents learning from interfering with
representations that compression has identified as important.

This is different from gradient scaling because:
- Scaling just slows down changes to important weights
- OGP prevents changes in the DIRECTIONS that matter for old tasks
- Random partitioning protects noise; learned partitioning protects signal
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from training.metrics import CLMetrics, evaluate_task, evaluate_all_tasks
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer


class OGPModel(nn.Module):
    """ResNet-18 with compression-defined OGP."""

    def __init__(self, num_classes_per_task, num_tasks, init_bit_depth=8.0):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, 0, 0.01)
            nn.init.constant_(h.bias, 0)

        self.importance = nn.ParameterDict()
        self._conv_map = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                safe = name.replace('.', '_')
                self.importance[safe] = nn.Parameter(
                    torch.full((module.out_channels,), init_bit_depth))
                self._conv_map[safe] = name

        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compute_compression_loss(self):
        total, count = 0.0, 0
        for safe, bits in self.importance.items():
            conv_name = self._conv_map[safe]
            for n, m in self.backbone.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    wpc = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    total += (bits.clamp(min=0) * wpc).sum()
                    count += m.weight.numel()
                    break
        return total / max(count, 1)

    def get_consolidated_mask(self, threshold_percentile=75):
        """Get per-channel mask of consolidated (high bit-depth) channels.

        Returns dict: safe_name -> boolean mask (True = consolidated)
        """
        all_bits = []
        for safe, bits in self.importance.items():
            all_bits.append(bits.clamp(min=0).detach().cpu())
        all_bits_cat = torch.cat(all_bits)

        threshold = np.percentile(all_bits_cat.numpy(), threshold_percentile)

        masks = {}
        for safe, bits in self.importance.items():
            masks[safe] = (bits.clamp(min=0).detach() >= threshold)
        return masks, threshold

    def project_gradients_ogp(self, consolidated_masks, task_id):
        """Project gradients orthogonal to consolidated channels.

        For each conv layer, zero out the gradient components in the
        directions corresponding to consolidated output channels.
        This prevents new task learning from modifying the representations
        that old tasks rely on.
        """
        for safe, mask in consolidated_masks.items():
            conv_name = self._conv_map[safe]
            for n, m in self.backbone.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    if m.weight.grad is not None:
                        # Zero gradient for consolidated channels entirely
                        m.weight.grad.data[mask] = 0
                    break

            # Also protect bit-depth params for consolidated channels
            if self.importance[safe].grad is not None:
                self.importance[safe].grad.data[mask] = 0

        # Protect other task heads
        for pname, param in self.named_parameters():
            if 'heads.' in pname and param.grad is not None:
                hi = int(pname.split('heads.')[1].split('.')[0])
                if hi != task_id:
                    param.grad.data.fill_(0)


def train_ogp_compression(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.01)
    consolidate_pct = config.get('consolidate_pct', 50)  # top X% channels consolidated
    replay_per_task = config.get('replay_per_task', 200)
    use_random_partition = config.get('use_random_partition', False)  # control: random partition

    classes_per_task = len(benchmark.tasks[0]['classes'])
    model = OGPModel(classes_per_task, num_tasks).to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    consolidated_masks = None  # No consolidation for first task

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        imp_params = list(model.importance.parameters())
        imp_ids = {id(p) for p in imp_params}
        other = [p for p in model.parameters() if id(p) not in imp_ids]

        opt = torch.optim.AdamW([
            {'params': other, 'lr': 1e-3, 'weight_decay': 5e-4},
            {'params': imp_params, 'lr': 0.5, 'eps': 1e-3, 'weight_decay': 0},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_loader))

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    Q = model.compute_compression_loss()
                    loss = loss + gamma * Q

                    if rb.size > 0:
                        buf = rb.sample(64)
                        if buf:
                            bx, by, bt = buf
                            bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                            for tid in bt.unique():
                                m = bt == tid
                                loss += F.cross_entropy(model(bx[m], task_id=tid.item()), by[m]) / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                # Apply OGP if we have consolidated masks from previous tasks
                if consolidated_masks is not None:
                    model.project_gradients_ogp(consolidated_masks, task_id)
                else:
                    # Just protect heads for first task
                    for pn, p in model.named_parameters():
                        if 'heads.' in pn and p.grad is not None:
                            hi = int(pn.split('heads.')[1].split('.')[0])
                            if hi != task_id: p.grad.data.fill_(0)

                scaler.step(opt)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # After task: consolidate top channels
        if use_random_partition:
            # Random control: randomly select same number of channels to consolidate
            masks, _ = model.get_consolidated_mask(consolidate_pct)
            for safe in masks:
                n_consolidated = masks[safe].sum().item()
                rand_mask = torch.zeros_like(masks[safe])
                indices = torch.randperm(masks[safe].shape[0])[:n_consolidated]
                rand_mask[indices] = True
                masks[safe] = rand_mask
            consolidated_masks = {k: v.to(device) for k, v in masks.items()}
        else:
            masks, threshold = model.get_consolidated_mask(consolidate_pct)
            consolidated_masks = {k: v.to(device) for k, v in masks.items()}

        n_cons = sum(m.sum().item() for m in consolidated_masks.values())
        n_total = sum(m.numel() for m in consolidated_masks.values())
        print(f"  Consolidated {n_cons}/{n_total} channels ({n_cons/n_total*100:.1f}%)")

        rb.add_task_samples(benchmark.sample_for_replay(task_id, replay_per_task))

        accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl.update(task_id, accs)
        for j, a in enumerate(accs):
            print(f"  Task {j}: {a*100:.1f}%")
        print(f"  Avg: {cl.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl.backward_transfer(task_id)*100:.2f}%")

    cl.print_matrix()
    return cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--consolidate_pct', type=int, default=50)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--use_random_partition', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)

    mode = "random_partition" if args.use_random_partition else "learned_partition"
    print(f"OGP-Compression ({mode}): gamma={args.gamma}, consolidate={args.consolidate_pct}%")

    cl = train_ogp_compression(benchmark, config, device)
    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'accuracy_matrix': cl.accuracy_matrix, 'avg_accuracy': avg, 'bwt': bwt},
               f'checkpoints/ogp_{mode}_c{args.consolidate_pct}.pt')


if __name__ == '__main__':
    main()
