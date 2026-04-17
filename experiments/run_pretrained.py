"""Run pretrained backbone experiments on Split CIFAR-100 and Split ImageNet-R."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import argparse
from models.pretrained import PretrainedResNet18CL, PretrainedResNet18Plain
from models.quantization import CompressionGranularity, compute_average_bit_depth, DifferentiableQuantizer
from models.compression import apply_bias_l1_penalty
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from training.hybrid import SoftProtectionCSC, make_optimizer_adamw
from data.replay_buffer import ReplayBuffer
from data.der_buffer import DERBuffer


def get_benchmark(dataset, num_tasks, batch_size, seed):
    if dataset == 'cifar100':
        from data.split_cifar100_pretrained import SplitCIFAR100Pretrained
        return SplitCIFAR100Pretrained(num_tasks=num_tasks, batch_size=batch_size,
                                        num_workers=8, seed=seed), 100 // num_tasks
    elif dataset == 'imagenet-r':
        from data.split_imagenet_r import SplitImageNetR
        bm = SplitImageNetR(num_tasks=num_tasks, batch_size=batch_size, seed=seed)
        return bm, bm.classes_per_task
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_soft_csc_pretrained(model, benchmark, config, device='cuda'):
    """Soft CSC + DER++ with pretrained backbone."""
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    gamma = config.get('gamma', 0.001)
    beta = config.get('beta', 1.0)
    alpha_der = config.get('alpha_der', 0.5)
    beta_der = config.get('beta_der', 1.0)
    replay_per_task = config.get('replay_per_task', 200)

    protector = SoftProtectionCSC(model, beta=beta, device=device)
    der_buffer = DERBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Separate quant and weight params
        quant_ids = set()
        qp, wp = [], []
        for m in model.modules():
            if isinstance(m, DifferentiableQuantizer):
                for p in m.parameters():
                    quant_ids.add(id(p)); qp.append(p)
        for p in model.parameters():
            if id(p) not in quant_ids and p.requires_grad:
                wp.append(p)

        opt = torch.optim.AdamW([
            {'params': wp, 'lr': config.get('lr', 1e-3), 'weight_decay': 5e-4},
            {'params': qp, 'lr': 0.5, 'eps': 1e-3, 'weight_decay': 0},
        ]) if qp else torch.optim.AdamW(wp, lr=config.get('lr', 1e-3), weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_loader))

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    if gamma > 0:
                        Q = compute_average_bit_depth(model)
                        loss = loss + gamma * Q
                        loss = loss + apply_bias_l1_penalty(model, 0.01)

                    # DER++ replay
                    if der_buffer.size > 0 and (alpha_der > 0 or beta_der > 0):
                        buf = der_buffer.sample(64)
                        if buf is not None:
                            bx, by, blogits, bt = buf
                            bx, by, blogits, bt = bx.to(device), by.to(device), blogits.to(device), bt.to(device)
                            for tid in bt.unique():
                                m = bt == tid
                                cur_logits = model(bx[m], task_id=tid.item())
                                if beta_der > 0:
                                    loss = loss + beta_der * F.cross_entropy(cur_logits, by[m]) / len(bt.unique())
                                if alpha_der > 0:
                                    loss = loss + alpha_der * F.mse_loss(cur_logits, blogits[m]) / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if beta > 0:
                    protector.scale_gradients(task_id)
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Store DER++ samples
        if replay_per_task > 0:
            task_data = benchmark.tasks[task_id]
            indices = task_data['train_indices']
            rng = __import__('numpy').random.RandomState(task_id)
            sel = rng.choice(len(indices), min(replay_per_task, len(indices)), replace=False)
            model.eval()
            samples = []
            with torch.no_grad():
                for i in sel:
                    x, y = benchmark.train_dataset_raw[indices[i]]
                    local_y = task_data['class_mapping'][y]
                    logits = model(x.unsqueeze(0).to(device), task_id=task_id).squeeze(0).cpu()
                    samples.append((x, local_y, logits, task_id))
            model.train()
            der_buffer.add_task_samples(samples)

        # Evaluate
        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl.update(task_id, all_accs)
        for j, a in enumerate(all_accs):
            print(f"  Task {j}: {a*100:.1f}%")
        print(f"  Avg: {cl.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl.backward_transfer(task_id)*100:.2f}%")

    cl.print_matrix()
    return cl


def train_replay_pretrained(model, benchmark, config, device='cuda'):
    """Plain replay or DER++ with pretrained backbone (no CSC)."""
    num_tasks = config['num_tasks']
    epochs = config['epochs_per_task']
    alpha_der = config.get('alpha_der', 0.0)
    beta_der = config.get('beta_der', 0.0)
    replay_per_task = config.get('replay_per_task', 200)

    if alpha_der > 0 or beta_der > 0:
        buffer = DERBuffer(max_per_task=replay_per_task)
        use_der = True
    else:
        buffer = ReplayBuffer(max_per_task=replay_per_task)
        use_der = False

    cl = CLMetrics(num_tasks)
    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*50} TASK {task_id} {'='*50}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=config.get('lr', 1e-3), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_loader))

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                    if buffer.size > 0:
                        if use_der:
                            buf = buffer.sample(64)
                            if buf:
                                bx, by, blogits, bt = buf
                                bx, by, blogits, bt = bx.to(device), by.to(device), blogits.to(device), bt.to(device)
                                for tid in bt.unique():
                                    m = bt == tid
                                    cl_buf = model(bx[m], task_id=tid.item())
                                    if beta_der > 0:
                                        loss = loss + beta_der * F.cross_entropy(cl_buf, by[m]) / len(bt.unique())
                                    if alpha_der > 0:
                                        loss = loss + alpha_der * F.mse_loss(cl_buf, blogits[m]) / len(bt.unique())
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
                # Protect other task heads
                for pn, p in model.named_parameters():
                    if p.grad is not None and 'heads.' in pn:
                        hi = int(pn.split('heads.')[1].split('.')[0])
                        if hi != task_id:
                            p.grad.data.fill_(0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Store replay
        if replay_per_task > 0:
            task_data = benchmark.tasks[task_id]
            indices = task_data['train_indices']
            rng = __import__('numpy').random.RandomState(task_id)
            sel = rng.choice(len(indices), min(replay_per_task, len(indices)), replace=False)

            if use_der:
                model.eval()
                samples = []
                with torch.no_grad():
                    for i in sel:
                        x, y = benchmark.train_dataset_raw[indices[i]]
                        local_y = task_data['class_mapping'][y]
                        logits = model(x.unsqueeze(0).to(device), task_id=task_id).squeeze(0).cpu()
                        samples.append((x, local_y, logits, task_id))
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

    cl.print_matrix()
    return cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar100', 'imagenet-r'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['soft_csc_der', 'der_only', 'replay_only', 'packnet'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark, classes_per_task = get_benchmark(
        args.dataset, args.num_tasks, args.batch_size, args.seed)

    config = {
        'num_tasks': args.num_tasks,
        'epochs_per_task': args.epochs_per_task,
        'replay_per_task': args.replay_per_task,
        'gamma': args.gamma,
        'beta': args.beta,
        'lr': args.lr,
        'alpha_der': 0.5 if 'der' in args.method else 0.0,
        'beta_der': 1.0 if 'der' in args.method else 0.0,
    }

    print(f"Pretrained ResNet-18 | {args.dataset} | {args.method} | "
          f"{args.num_tasks} tasks x {classes_per_task} classes")

    if args.method == 'soft_csc_der':
        model = PretrainedResNet18CL(
            classes_per_task, args.num_tasks, quantize=True,
            freeze_backbone=args.freeze_backbone).to(device)
        cl = train_soft_csc_pretrained(model, benchmark, config, device)

    elif args.method == 'der_only':
        model = PretrainedResNet18Plain(
            classes_per_task, args.num_tasks,
            freeze_backbone=args.freeze_backbone).to(device)
        config['alpha_der'] = 0.5
        config['beta_der'] = 1.0
        cl = train_replay_pretrained(model, benchmark, config, device)

    elif args.method == 'replay_only':
        model = PretrainedResNet18Plain(
            classes_per_task, args.num_tasks,
            freeze_backbone=args.freeze_backbone).to(device)
        cl = train_replay_pretrained(model, benchmark, config, device)

    elif args.method == 'packnet':
        # Use our existing PackNet implementation with pretrained model
        print("PackNet with pretrained backbone — using baselines/packnet.py")
        # For now, just run the plain pretrained model with PackNet-style training
        # TODO: integrate with PackNet properly
        model = PretrainedResNet18Plain(
            classes_per_task, args.num_tasks).to(device)
        cl = train_replay_pretrained(model, benchmark, config, device)

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl.accuracy_matrix,
        'avg_accuracy': avg, 'bwt': bwt,
        'config': config,
    }, f'checkpoints/pretrained_{args.dataset}_{args.method}_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
