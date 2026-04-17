"""Compression-weighted replay: weight replay samples by compression risk.

Each replay sample is stored with a snapshot of the bit-depths at storage time.
During replay, the loss for each sample is weighted by how much the bit-depths
have changed since storage — samples from tasks whose representations are being
compressed away get higher replay weight.

This is a fundamentally different use of compression: it doesn't change the
learning rate, it changes the DATA the model sees. Compression tells us
WHICH old knowledge is being lost, and we increase the replay signal for that.
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


class WeightedReplayBuffer:
    """Replay buffer that stores bit-depth snapshots with each sample."""

    def __init__(self, max_per_task=200):
        self.max_per_task = max_per_task
        self.data = []  # (x, y, task_id, bitdepth_snapshot)

    def add(self, samples, bitdepth_snapshot):
        """Add samples with current bit-depth snapshot."""
        for x, y, tid in samples:
            self.data.append((x, y, tid, bitdepth_snapshot.clone()))

    @property
    def size(self):
        return len(self.data)

    def sample(self, batch_size, current_bitdepths):
        """Sample and compute per-sample weights based on bit-depth change.

        Returns: (x, y, task_ids, weights) where weights reflect how much
        the bit-depths have changed since the sample was stored.
        """
        if self.size == 0:
            return None

        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        xs, ys, tids, weights = [], [], [], []

        for i in indices:
            x, y, tid, stored_bd = self.data[i]
            xs.append(x)
            ys.append(y)
            tids.append(tid)

            # Weight = how much bit-depths changed (L1 distance)
            change = (current_bitdepths - stored_bd).abs().mean().item()
            weights.append(1.0 + change)  # base weight 1 + change bonus

        xs = torch.stack(xs)
        ys = torch.tensor(ys, dtype=torch.long)
        tids = torch.tensor(tids, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize weights to mean 1
        weights = weights / weights.mean()

        return xs, ys, tids, weights


class CompModel(nn.Module):
    def __init__(self, num_classes_per_task, num_tasks, init_bd=8.0):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)])
        for h in self.heads:
            nn.init.normal_(h.weight, 0, 0.01)
            nn.init.constant_(h.bias, 0)

        self.importance = nn.ParameterDict()
        self._conv_map = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                safe = name.replace('.', '_')
                self.importance[safe] = nn.Parameter(torch.full((module.out_channels,), init_bd))
                self._conv_map[safe] = name

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compression_loss(self):
        total, count = 0.0, 0
        for safe, bits in self.importance.items():
            b = bits.clamp(min=0)
            conv_name = self._conv_map[safe]
            for n, m in self.backbone.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    total += (b * m.in_channels * m.kernel_size[0] * m.kernel_size[1]).sum()
                    count += m.weight.numel()
                    break
        return total / max(count, 1)

    def get_bitdepth_vector(self):
        """Get all bit-depths as a single flat vector."""
        return torch.cat([b.clamp(min=0).detach().cpu() for b in self.importance.values()])


def train_weighted_replay(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.01)
    replay_per_task = config.get('replay_per_task', 200)

    classes_per_task = len(benchmark.tasks[0]['classes'])
    model = CompModel(classes_per_task, num_tasks).to(device)
    rb = WeightedReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

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
                    Q = model.compression_loss()
                    loss = loss + gamma * Q

                    # Weighted replay
                    if rb.size > 0:
                        current_bd = model.get_bitdepth_vector()
                        buf = rb.sample(64, current_bd)
                        if buf is not None:
                            bx, by, bt, bw = buf
                            bx, by, bt, bw = bx.to(device), by.to(device), bt.to(device), bw.to(device)

                            # Compute per-sample weighted replay loss
                            replay_loss = torch.tensor(0.0, device=device)
                            for tid in bt.unique():
                                mask = bt == tid
                                r_logits = model(bx[mask], task_id=tid.item())
                                per_sample_loss = F.cross_entropy(r_logits, by[mask], reduction='none')
                                # Weight each sample by its compression change
                                weighted_loss = (per_sample_loss * bw[mask]).mean()
                                replay_loss += weighted_loss
                            loss = loss + replay_loss / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                # Protect heads
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

        # Store replay with current bit-depth snapshot
        samples = benchmark.sample_for_replay(task_id, replay_per_task)
        bd_snapshot = model.get_bitdepth_vector()
        rb.add(samples, bd_snapshot)

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
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)

    print(f"Compression-weighted replay: gamma={args.gamma}")
    cl = train_weighted_replay(benchmark, config, device)
    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'accuracy_matrix': cl.accuracy_matrix, 'avg_accuracy': avg, 'bwt': bwt},
               f'checkpoints/weighted_replay_g{args.gamma}.pt')


if __name__ == '__main__':
    main()
