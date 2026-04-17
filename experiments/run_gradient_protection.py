"""Gradient-based protection: use the compression gradient as the protection signal.

The gradient of the compression loss w.r.t. bit-depth tells us which weights
the network is TRYING to compress away. Protect those weights more during
replay — they're the ones at risk of being repurposed.

Key insight: d(compression_loss)/d(bit_depth_i) tells us:
- Large negative gradient: network wants to compress this weight (it's dispensable for current task)
- Near-zero gradient: weight is at equilibrium (useful for current task)

For CL: weights with large negative compression gradient are the ones being
repurposed from old tasks to new ones. These need the most replay protection.

This is fundamentally different from random scaling because random values have
no gradient — they carry no information about which weights are currently at risk.
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


class GradProtectionModel(nn.Module):
    """ResNet-18 with compression gradient-based protection."""

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

        # Importance parameters (bit-depths) — NOT used in forward pass
        self._conv_info = []
        self.importance = nn.ParameterDict()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                safe = name.replace('.', '_')
                self.importance[safe] = nn.Parameter(
                    torch.full((module.out_channels,), init_bit_depth))
                self._conv_info.append((name, safe, module.out_channels))

        # Running EMA of compression gradients (the protection signal)
        self.compression_grad_ema = {}
        for name, safe, n_ch in self._conv_info:
            self.compression_grad_ema[safe] = torch.zeros(n_ch)

        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compute_compression_loss(self):
        """Compute average bit-depth (compression loss)."""
        total, count = 0.0, 0
        for name, safe, n_ch in self._conv_info:
            b = self.importance[safe].clamp(min=0)
            for n, m in self.backbone.named_modules():
                if n == name and isinstance(m, nn.Conv2d):
                    wpc = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    total += (b * wpc).sum()
                    count += m.weight.numel()
                    break
        return total / max(count, 1)

    def update_compression_grad_ema(self, alpha=0.1):
        """Update EMA of compression gradients after backward pass."""
        for name, safe, n_ch in self._conv_info:
            if self.importance[safe].grad is not None:
                grad = self.importance[safe].grad.detach().cpu()
                self.compression_grad_ema[safe] = (
                    (1 - alpha) * self.compression_grad_ema[safe] + alpha * grad)

    def get_protection_signal(self, device):
        """Get protection factors based on compression gradient.

        Weights with large negative compression gradient = being compressed away
        = need MORE protection (lower effective LR).

        Protection = 1 / (1 + beta * max(0, -grad_ema))
        """
        factors = {}
        for name, safe, n_ch in self._conv_info:
            # Negative gradient = compression wants to reduce bit-depth = at risk
            risk = (-self.compression_grad_ema[safe]).clamp(min=0).to(device)
            factors[safe] = risk
        return factors

    def apply_gradient_protection(self, beta, task_id, device):
        """Scale backbone gradients inversely with compression risk."""
        risk_factors = self.get_protection_signal(device)

        for name, safe, n_ch in self._conv_info:
            for n, m in self.backbone.named_modules():
                if n == name and isinstance(m, nn.Conv2d):
                    if m.weight.grad is not None:
                        risk = risk_factors[safe]
                        # Normalize risk to [0, 1] range
                        if risk.max() > 0:
                            risk_norm = risk / risk.max()
                        else:
                            risk_norm = risk
                        scale = 1.0 / (1.0 + beta * risk_norm.view(-1, 1, 1, 1))
                        m.weight.grad.data *= scale.expand_as(m.weight.grad)
                    break

        # Also scale importance param gradients by their own risk
        for safe, bits in self.importance.items():
            if bits.grad is not None and safe in risk_factors:
                risk = risk_factors[safe]
                if risk.max() > 0:
                    risk_norm = risk / risk.max()
                else:
                    risk_norm = risk
                bits.grad.data *= 1.0 / (1.0 + beta * risk_norm)

        # Protect other task heads
        for pname, param in self.named_parameters():
            if 'heads.' in pname and param.grad is not None:
                hi = int(pname.split('heads.')[1].split('.')[0])
                if hi != task_id:
                    param.grad.data.fill_(0)


def train_gradient_protection(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.01)
    beta = config.get('beta', 5.0)
    replay_per_task = config.get('replay_per_task', 200)

    classes_per_task = len(benchmark.tasks[0]['classes'])
    model = GradProtectionModel(classes_per_task, num_tasks).to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Separate params
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
                    task_loss = F.cross_entropy(logits, y)

                    # Compression loss
                    Q = model.compute_compression_loss()
                    comp_loss = gamma * Q

                    loss = task_loss + comp_loss

                    # Replay
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

                # Update compression gradient EMA BEFORE applying protection
                model.update_compression_grad_ema(alpha=0.1)

                # Apply gradient-based protection
                model.apply_gradient_protection(beta, task_id, device)

                scaler.step(opt)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                # Show compression gradient magnitude
                avg_risk = np.mean([model.compression_grad_ema[s].abs().mean().item()
                                    for _, s, _ in model._conv_info])
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%, Risk={avg_risk:.4f}")

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
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)

    print(f"Gradient-based protection: gamma={args.gamma}, beta={args.beta}")
    cl = train_gradient_protection(benchmark, config, device)

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'accuracy_matrix': cl.accuracy_matrix, 'avg_accuracy': avg, 'bwt': bwt},
               f'checkpoints/grad_prot_g{args.gamma}_b{args.beta}.pt')


if __name__ == '__main__':
    main()
