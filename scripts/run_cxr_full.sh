#!/bin/bash
#SBATCH --job-name=cxr_full
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

python src/cxr_interp.py \
  --data_dir ./rsna \
  --out_dir ./out/out_full \
  --epochs 5 \
  --batch 32
