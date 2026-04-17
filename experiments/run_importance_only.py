"""Compression-as-importance-only experiment.

Bit-depths are learned via compression loss, used for soft protection,
but the quantization function is REMOVED from the forward pass.
Weights are used at full precision. Model stays at 100% capacity.

This isolates the importance signal: any improvement over replay-only
can ONLY come from the learned importance driving better gradient scaling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from models.quantization import DifferentiableQuantizer, CompressionGranularity
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer
from data.der_buffer import DERBuffer


class ImportanceOnlyResNet18(nn.Module):
    """ResNet-18 with importance parameters (bit-depths) but NO quantization.

    Each conv layer has learnable bit-depth parameters that are optimized
    via a compression loss, but the forward pass uses full-precision weights.
    The bit-depths serve purely as importance signals for gradient scaling.
    """

    def __init__(self, num_classes_per_task=10, num_tasks=10, init_bit_depth=8.0):
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

        # Importance parameters (bit-depths) for each conv layer
        # These are optimized but DON'T affect the forward pass
        self.importance_params = nn.ModuleDict()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                n_channels = module.out_channels
                safe_name = name.replace('.', '_')
                self.importance_params[safe_name] = nn.Module()
                self.importance_params[safe_name].bit_depth = nn.Parameter(
                    torch.full((n_channels,), init_bit_depth))
                self.importance_params[safe_name].out_channels = n_channels
                self.importance_params[safe_name].conv_name = name

        self.num_tasks = num_tasks

        for head in self.heads:
            nn.init.normal_(head.weight, 0, 0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compute_avg_bitdepth(self):
        """Compute average bit-depth (compression loss) from importance params."""
        total_bits = 0.0
        total_weights = 0
        for name, module in self.importance_params.items():
            b = module.bit_depth.clamp(min=0)
            # Find corresponding conv to get weight shape
            conv_name = module.conv_name
            for n, m in self.backbone.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    weights_per_channel = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    total_bits += (b * weights_per_channel).sum()
                    total_weights += m.weight.numel()
                    break
        return total_bits / max(total_weights, 1)

    def get_channel_importance(self):
        """Get all channel bit-depths as importance signals."""
        result = {}
        for name, module in self.importance_params.items():
            result[name] = module.bit_depth.clamp(min=0).detach()
        return result

    def scale_gradients(self, beta, task_id):
        """Scale backbone gradients by importance (bit-depth)."""
        importance = self.get_channel_importance()

        for safe_name, imp_module in self.importance_params.items():
            conv_name = imp_module.conv_name
            bits = imp_module.bit_depth.clamp(min=0).detach()

            for n, m in self.backbone.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    if m.weight.grad is not None:
                        scale = 1.0 / (1.0 + beta * bits.view(-1, 1, 1, 1))
                        m.weight.grad.data *= scale.expand_as(m.weight.grad)
                    break

            # Scale importance param gradients too
            if imp_module.bit_depth.grad is not None:
                bit_scale = 1.0 / (1.0 + beta * bits)
                imp_module.bit_depth.grad.data *= bit_scale

        # Protect other task heads
        for pname, param in self.named_parameters():
            if 'heads.' in pname and param.grad is not None:
                head_idx = int(pname.split('heads.')[1].split('.')[0])
                if head_idx != task_id:
                    param.grad.data.fill_(0)


def train_importance_only(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.001)
    beta = config.get('beta', 1.0)
    replay_per_task = config.get('replay_per_task', 200)
    use_der = config.get('use_der', False)
    refresh_interval = config.get('refresh_interval', 0)  # 0 = no refresh

    freeze_importance = config.get('freeze_importance', False)

    model = ImportanceOnlyResNet18(
        100 // num_tasks, num_tasks).to(device)

    # Random scaling: randomize bit-depths and freeze them
    if freeze_importance:
        with torch.no_grad():
            for name, module in model.importance_params.items():
                module.bit_depth.uniform_(0, 8)  # random [0, 8]
                module.bit_depth.requires_grad = False

    if use_der:
        buffer = DERBuffer(max_per_task=replay_per_task)
    else:
        buffer = ReplayBuffer(max_per_task=replay_per_task)

    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Refresh importance if configured
        if refresh_interval > 0 and task_id > 0 and task_id % refresh_interval == 0:
            print(f"  Refreshing importance params to 8.0")
            with torch.no_grad():
                for name, module in model.importance_params.items():
                    module.bit_depth.fill_(8.0)

        # Separate importance params from backbone params
        imp_params = [p for p in model.importance_params.parameters() if p.requires_grad]
        imp_ids = {id(p) for p in imp_params}
        backbone_params = [p for p in model.parameters() if id(p) not in imp_ids and p.requires_grad]

        if imp_params:
            opt = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 1e-3, 'weight_decay': 5e-4},
                {'params': imp_params, 'lr': 0.5, 'eps': 1e-3, 'weight_decay': 0},
            ])
        else:
            # Random scaling: no importance params to optimize
            opt = torch.optim.AdamW(backbone_params, lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_loader))

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                    # Compression loss on importance params (NOT on actual weights)
                    Q = model.compute_avg_bitdepth()
                    loss = loss + gamma * Q

                    # Replay
                    if buffer.size > 0:
                        if use_der:
                            buf = buffer.sample(64)
                            if buf:
                                bx, by, blogits, bt = buf
                                bx, by, blogits, bt = bx.to(device), by.to(device), blogits.to(device), bt.to(device)
                                for tid in bt.unique():
                                    m = bt == tid
                                    cl_buf = model(bx[m], task_id=tid.item())
                                    loss = loss + F.cross_entropy(cl_buf, by[m]) / len(bt.unique())
                                    loss = loss + 0.5 * F.mse_loss(cl_buf, blogits[m]) / len(bt.unique())
                        else:
                            buf = buffer.sample(64)
                            if buf:
                                bx, by, bt = buf
                                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                                for tid in bt.unique():
                                    m = bt == tid
                                    loss = loss + F.cross_entropy(model(bx[m], task_id=tid.item()), by[m]) / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                # Soft protection using importance
                if beta > 0:
                    model.scale_gradients(beta, task_id)

                scaler.step(opt)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                importance = model.get_channel_importance()
                all_bits = torch.cat(list(importance.values()))
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%, "
                      f"Q={all_bits.mean():.3f}, range=[{all_bits.min():.2f}, {all_bits.max():.2f}]")

        # Store replay
        if replay_per_task > 0:
            if use_der:
                task_data = benchmark.tasks[task_id]
                indices = task_data['train_indices']
                rng = np.random.RandomState(task_id)
                sel = rng.choice(len(indices), min(replay_per_task, len(indices)), replace=False)
                model.eval()
                samples = []
                with torch.no_grad():
                    for i in sel:
                        x_raw, y_raw = benchmark.train_dataset_raw[indices[i]]
                        local_y = task_data['class_mapping'][y_raw]
                        logits = model(x_raw.unsqueeze(0).to(device), task_id=task_id).squeeze(0).cpu()
                        samples.append((x_raw, local_y, logits, task_id))
                model.train()
                buffer.add_task_samples(samples)
            else:
                samples = benchmark.sample_for_replay(task_id, replay_per_task)
                buffer.add_task_samples(samples)

        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl.update(task_id, all_accs)
        for j, a in enumerate(all_accs):
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
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--use_der', action='store_true')
    parser.add_argument('--refresh_interval', type=int, default=0,
                        help='Reset bit-depths every N tasks (0=never)')
    parser.add_argument('--random_scaling', action='store_true',
                        help='Use random fixed bit-depths instead of learned')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)

    # If random scaling, freeze importance params after random init
    if args.random_scaling:
        config['freeze_importance'] = True

    cl = train_importance_only(benchmark, config)

    name = f"imp_only_g{args.gamma}_b{args.beta}_r{args.replay_per_task}"
    if args.use_der:
        name += "_der"
    if args.refresh_interval > 0:
        name += f"_refresh{args.refresh_interval}"

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl.accuracy_matrix,
        'summary': cl.summary(),
    }, f'checkpoints/{name}.pt')


if __name__ == '__main__':
    main()
