#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -J data_prep
#SBATCH -o scripts/data_prep.%j.out
#SBATCH -e scripts/data_prep.%j.err

set -uo pipefail
DATA=/capstor/scratch/cscs/$USER/datasets
mkdir -p "$DATA"
cd "$DATA"

echo "=== Downloading ImageNet-R ==="
if [ ! -d imagenet-r ]; then
    curl -L --retry 3 -o imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar 2>&1 | tail -3
    tar xf imagenet-r.tar && rm imagenet-r.tar
fi
ls imagenet-r/ 2>/dev/null | head -3

echo "=== Downloading CUB-200 ==="
if [ ! -d CUB_200_2011 ]; then
    curl -L --retry 3 -o CUB.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz 2>&1 | tail -3
    [ -s CUB.tgz ] && tar xf CUB.tgz && rm CUB.tgz
fi
ls CUB_200_2011/ 2>/dev/null | head -3

echo "=== Downloading CORe50 ==="
if [ ! -d core50_128x128 ]; then
    curl -L --retry 3 -o core50.zip http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip 2>&1 | tail -3
    [ -s core50.zip ] && unzip -q core50.zip && rm core50.zip
fi
ls core50_128x128/ 2>/dev/null | head -3

echo "=== Downloading ImageNet-Sketch ==="
# Their datasets module uses Imagenet_Sketch_subset which expects a specific layout.
# Try the kaggle/HF mirrors:
if [ ! -d imagenet-sketch ]; then
    # Try HuggingFace sketch dataset
    pip install --quiet huggingface_hub 2>/dev/null || true
    python3 -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id='songweig/imagenet_sketch', repo_type='dataset',
                      local_dir='./imagenet-sketch', local_dir_use_symlinks=False)
    print('  HF sketch downloaded')
except Exception as e:
    print(f'  HF sketch failed: {e}')
" 2>&1 | tail -5
fi
ls imagenet-sketch/ 2>/dev/null | head -3

ls -la "$DATA"
echo "=== DONE ==="
