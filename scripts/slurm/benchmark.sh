#!/bin/bash
# Time every method's repair on one A40 node (experiments/benchmark.py), so the costs are comparable.
#     mkdir -p logs && sbatch scripts/slurm/benchmark.sh                       # every experiment with test repairs
#     sbatch scripts/slurm/benchmark.sh E02_resshift E09_ccdf                  # only these
# Writes <experiment>/benchmark.json and experiments/benchmark.md; run_eval.py puts the cost in the report.
#SBATCH --job-name=benchmark
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_%j.out

set -euo pipefail
PY=${PY:-python}
export OMP_NUM_THREADS=8

nvidia-smi --query-gpu=name,memory.total --format=csv
if [ $# -gt 0 ]; then
    $PY experiments/benchmark.py --exp "$@"
else
    $PY experiments/benchmark.py
fi
