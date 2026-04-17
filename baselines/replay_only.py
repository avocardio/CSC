"""Experience Replay baseline (no compression).

Standard replay: store a fixed buffer of examples from previous tasks,
mix them into training for new tasks. No compression or pruning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import argparse
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer
from baselines.finetune import SimpleResNet18


def train_replay(benchmark, config, device='cuda', model=None):
    num_tasks = config['num_tasks']
    epochs_per_task = config.get('epochs_per_task', 50)
    lr = config.get('lr', 1e-3)
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    alpha = config.get('alpha', 1.0)
    classes_per_task = len(benchmark.tasks[0]['classes'])

    if model is None:
        model = SimpleResNet18(classes_per_task, num_tasks).to(device)
    cl_metrics = CLMetrics(num_tasks)
    replay_buffer = ReplayBuffer(max_per_task=replay_per_task)

    for task_id in range(num_tasks):
        print(f"\n{'='*40} TASK {task_id} {'='*40}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))

        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(epochs_per_task):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                    if replay_buffer.size > 0:
                        replay_data = replay_buffer.sample(replay_batch_size)
                        if replay_data is not None:
                            rx, ry, rtids = replay_data
                            rx, ry, rtids = rx.to(device), ry.to(device), rtids.to(device)
                            replay_loss = torch.tensor(0.0, device=device)
                            for tid in rtids.unique():
                                mask = rtids == tid
                                r_logits = model(rx[mask], task_id=tid.item())
                                replay_loss += F.cross_entropy(r_logits, ry[mask])
                            replay_loss = replay_loss / len(rtids.unique())
                            loss = loss + alpha * replay_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Store replay
        samples = benchmark.sample_for_replay(task_id, replay_per_task)
        replay_buffer.add_task_samples(samples)

        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl_metrics.update(task_id, all_accs)
        for j, a in enumerate(all_accs):
            print(f"  Task {j}: {a*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")

    cl_metrics.print_matrix()
    return cl_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)
    cl_metrics = train_replay(benchmark, config)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
    }, f'checkpoints/replay_only_t{args.num_tasks}_r{args.replay_per_task}.pt')


if __name__ == '__main__':
    main()
