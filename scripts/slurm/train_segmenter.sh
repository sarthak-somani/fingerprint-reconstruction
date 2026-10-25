#!/bin/bash
# Fine-tune the SegFormer-B0 damage segmenter (20k steps, ~1-2 h). Needs the v2 train set. From the repo root:
#     mkdir -p logs && sbatch scripts/slurm/train_segmenter.sh
#SBATCH --job-name=seg_train
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seg_train_%j.out

set -euo pipefail
PY=${PY:-python}
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY segmenter/train.py "$@"
