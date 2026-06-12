#!/bin/bash
#SBATCH --job-name=cxr_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cxr_%j.out
#SBATCH --error=logs/cxr_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cxr

cd ~/cxr_vit_shortcut_pilot

echo "=== ENV CHECK ==="
which python
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"
python -c "import timm, pydicom, sklearn, pandas, matplotlib; print('imports ok')"

echo "=== START RUN ==="
python src/cxr_interp.py \
  --data_dir ./rsna \
  --out_dir ./out/out_smoke \
  --epochs 1 \
  --subset 2000 \
  --batch 16
