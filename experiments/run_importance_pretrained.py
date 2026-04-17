"""Importance-only with pretrained backbone on ImageNet-R and CIFAR-100."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.der_buffer import DERBuffer
from data.replay_buffer import ReplayBuffer


class ImportanceOnlyPretrainedResNet18(nn.Module):
    """Pretrained ResNet-18 with importance params but no quantization."""

    def __init__(self, num_classes_per_task=20, num_tasks=10, init_bit_depth=8.0):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, 0, 0.01)
            nn.init.constant_(h.bias, 0)

        # Importance params for each conv layer
        self.importance_params = nn.ParameterDict()
        self._conv_names = {}
        for name, module in self.features.named_modules():
            if isinstance(module, nn.Conv2d):
                safe = name.replace('.', '_')
                self.importance_params[safe] = nn.Parameter(
                    torch.full((module.out_channels,), init_bit_depth))
                self._conv_names[safe] = name

        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compute_avg_bitdepth(self):
        total_bits, total_weights = 0.0, 0
        for safe, bits in self.importance_params.items():
            b = bits.clamp(min=0)
            conv_name = self._conv_names[safe]
            for n, m in self.features.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    wpc = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    total_bits += (b * wpc).sum()
                    total_weights += m.weight.numel()
                    break
        return total_bits / max(total_weights, 1)

    def scale_gradients(self, beta, task_id):
        for safe, bits in self.importance_params.items():
            b = bits.clamp(min=0).detach()
            conv_name = self._conv_names[safe]
            for n, m in self.features.named_modules():
                if n == conv_name and isinstance(m, nn.Conv2d):
                    if m.weight.grad is not None:
                        scale = 1.0 / (1.0 + beta * b.view(-1, 1, 1, 1))
                        m.weight.grad.data *= scale.expand_as(m.weight.grad)
                    break
            if bits.grad is not None:
                bits.grad.data *= 1.0 / (1.0 + beta * b)

        for pname, param in self.named_parameters():
            if 'heads.' in pname and param.grad is not None:
                hi = int(pname.split('heads.')[1].split('.')[0])
                if hi != task_id:
                    param.grad.data.fill_(0)


def train_importance_pretrained(model, benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.001)
    beta = config.get('beta', 1.0)
    replay_per_task = config.get('replay_per_task', 200)
    use_der = config.get('use_der', False)
    lr = config.get('lr', 1e-4)

    buffer = DERBuffer(max_per_task=replay_per_task) if use_der else ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        tl, vl = benchmark.get_task_dataloaders(task_id)

        imp_params = list(model.importance_params.parameters())
        imp_ids = {id(p) for p in imp_params}
        other = [p for p in model.parameters() if id(p) not in imp_ids and p.requires_grad]

        opt = torch.optim.AdamW([
            {'params': other, 'lr': lr, 'weight_decay': 5e-4},
            {'params': imp_params, 'lr': 0.1, 'eps': 1e-3, 'weight_decay': 0},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))

        for epoch in range(epochs):
            model.train()
            for batch in tl:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    Q = model.compute_avg_bitdepth()
                    loss = loss + gamma * Q

                    if buffer.size > 0:
                        if use_der:
                            buf = buffer.sample(64)
                            if buf:
                                bx, by, bl, bt = buf
                                bx, by, bl, bt = bx.to(device), by.to(device), bl.to(device), bt.to(device)
                                for tid in bt.unique():
                                    m = bt == tid
                                    cl_buf = model(bx[m], task_id=tid.item())
                                    loss += F.cross_entropy(cl_buf, by[m]) / len(bt.unique())
                                    loss += 0.5 * F.mse_loss(cl_buf, bl[m]) / len(bt.unique())
                        else:
                            buf = buffer.sample(64)
                            if buf:
                                bx, by, bt = buf
                                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                                for tid in bt.unique():
                                    m = bt == tid
                                    loss += F.cross_entropy(model(bx[m], task_id=tid.item()), by[m]) / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                model.scale_gradients(beta, task_id)
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc = evaluate_task(model, vl, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Store replay
        if replay_per_task > 0:
            task_data = benchmark.tasks[task_id]
            indices = task_data['train_indices']
            rng = np.random.RandomState(task_id)
            sel = rng.choice(len(indices), min(replay_per_task, len(indices)), replace=False)
            if use_der:
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
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar100', 'imagenet-r'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_der', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    if args.dataset == 'cifar100':
        from data.split_cifar100_pretrained import SplitCIFAR100Pretrained
        benchmark = SplitCIFAR100Pretrained(num_tasks=args.num_tasks, batch_size=args.batch_size, seed=args.seed)
        cpt = 100 // args.num_tasks
    else:
        from data.split_imagenet_r import SplitImageNetR
        benchmark = SplitImageNetR(num_tasks=args.num_tasks, batch_size=args.batch_size, seed=args.seed)
        cpt = benchmark.classes_per_task

    model = ImportanceOnlyPretrainedResNet18(cpt, args.num_tasks).to(device)
    config = vars(args)
    cl = train_importance_pretrained(model, benchmark, config, device)

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    name = f"imp_pretrained_{args.dataset}_t{args.num_tasks}"
    if args.use_der: name += "_der"
    torch.save({'accuracy_matrix': cl.accuracy_matrix, 'avg_accuracy': avg, 'bwt': bwt}, f'checkpoints/{name}.pt')


if __name__ == '__main__':
    main()
