#!/bin/bash
set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== PYTHON ==="
python --version
echo "=== MuJoCo Warp Benchmark on GH200 ==="
python -c "
import mujoco, warp as wp, mujoco_warp as mjw, time, torch
import numpy as np, os, metaworld

xml = os.path.dirname(metaworld.__file__) + '/assets/sawyer_xyz/sawyer_hammer.xml'
mjm = mujoco.MjModel.from_xml_path(xml)
for i in range(mjm.eq_data.shape[0]):
    if mjm.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
        mjm.eq_data[i] = np.array([0,0,0,0,0,0,-1,0,0,0,5.0])

wp.init()
with wp.ScopedDevice('cuda:0'):
    m = mjw.put_model(mjm)
    for nworld in [256, 1024, 4096, 8192, 16384]:
        d = mjw.make_data(mjm, nworld=nworld)
        for _ in range(10):
            mjw.step(m, d)
        wp.synchronize()
        t0 = time.time()
        for _ in range(500):
            mjw.step(m, d)
        wp.synchronize()
        elapsed = time.time() - t0
        total = nworld * 500
        print(f'nworld={nworld:6d}: {total/elapsed:>12,.0f} physics_sps ({elapsed:.3f}s)')

    # frame_skip=5 test
    d = mjw.make_data(mjm, nworld=4096)
    for _ in range(10):
        for _ in range(5):
            mjw.step(m, d)
    wp.synchronize()
    t0 = time.time()
    for _ in range(100):
        for _ in range(5):
            mjw.step(m, d)
    wp.synchronize()
    elapsed = time.time() - t0
    print(f'\nnworld=4096 frame_skip=5: {4096*100/elapsed:>12,.0f} control_sps')

    qpos = wp.to_torch(d.qpos)
    print(f'\nTorch interop: qpos shape={qpos.shape}, device={qpos.device}')
print('\nBenchmark complete!')
"
