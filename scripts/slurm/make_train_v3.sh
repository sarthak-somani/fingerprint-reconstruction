#!/bin/bash
# The v3 train set and every MSU-normalised copy, on CPUs. Run preprocessing 05-07 for v3 first. From the repo root:
#     mkdir -p logs && sbatch scripts/slurm/make_train_v3.sh
# Script 08 (3 damaged copies per crop and contaminant, ~113k samples), then script 10 (normalised copies
# of val, test and train).
#SBATCH --job-name=data_v3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/data_v3_%j.out

set -euo pipefail
PY=${PY:-python}
export FPR_DATA_VERSION=v3 OMP_NUM_THREADS=1

echo "== $(date)  script 08 (v3 train set)"
$PY preprocessing/08_make_train_set.py --workers 32
echo "== $(date)  script 10 (normalised copies)"
$PY preprocessing/10_normalize.py --workers 32
echo "== $(date)  done"
