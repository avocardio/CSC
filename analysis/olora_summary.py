"""Aggregate Online-LoRA run logs (output_olora* dirs) into a method×seed table.

Each run's stdout log contains lines like:
    [Average accuracy till task10]  Acc@1: XX.XXXX  Acc@5: ...

We parse the LAST such line per log and aggregate by (method, seed) or
(method, lr, mean_thr, var_thr) for the reproduction sweep.
"""
from __future__ import annotations
import os, re, glob, argparse
import numpy as np
from collections import defaultdict


PAT = re.compile(r'\[Average accuracy till task\d+\]\s+Acc@1:\s+([\d.]+)')


def parse_final_acc(log_path: str) -> float | None:
    try:
        with open(log_path) as f:
            txt = f.read()
    except Exception:
        return None
    matches = PAT.findall(txt)
    return float(matches[-1]) if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='output_olora_repro',
                   help='Directory containing per-run logs (NAME.log) and subdirs.')
    args = p.parse_args()

    rows = []
    for log in sorted(glob.glob(os.path.join(args.root, '*.log'))):
        acc = parse_final_acc(log)
        if acc is None:
            continue
        name = os.path.basename(log).replace('.log', '')
        rows.append((name, acc))

    if not rows:
        print(f'No logs found in {args.root}')
        return

    # Try to group by config patterns
    print(f'\n=== {args.root} ===')
    print(f'{"config":50s}  acc@1')
    print('-' * 65)
    by_cfg = defaultdict(list)
    for name, acc in rows:
        # Extract seed if present
        m = re.search(r'_s(\d+)', name)
        if m:
            cfg = name[:m.start()]
            seed = int(m.group(1))
            by_cfg[cfg].append((seed, acc))
        else:
            by_cfg[name].append((None, acc))

    for cfg in sorted(by_cfg):
        seeds = by_cfg[cfg]
        accs = np.array([a for _, a in seeds])
        print(f'{cfg:50s}  {accs.mean():5.2f} ± {accs.std():.2f}  (n={len(accs)})')

    # Also print individual per-name lines for reproduction sweeps
    print(f'\nPer-run:')
    for name, acc in rows:
        print(f'  {name:50s}  {acc:5.2f}')


if __name__ == '__main__':
    main()
