#!/bin/bash
# Re-evaluate experiments on test (run_eval.py, one leaderboard row each), then compare them all.
#     mkdir -p logs && sbatch scripts/slurm/eval_all.sh                        # every experiment with test repairs
#     sbatch scripts/slurm/eval_all.sh E04_msu_stft E09_ccdf                  # only these
#     sbatch --export=ALL,FPR_DATA_VERSION=v3 scripts/slurm/eval_all.sh E04_msu_stft E22_ccdf_aux_minutiae_cldice
# Needs an A40 / L40 node: NFIQ 2 needs libssl 1.1. Each evaluation holds logs/.run_eval.lock.
#SBATCH --job-name=eval_all
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_all_%j.out

set -euo pipefail
PY=${PY:-python}
EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then                  # every experiment whose test repairs exist for this data version
    mapfile -t EXPS < <($PY -c "
from fpr import experiments
for e in experiments.list_experiments():
    if any(experiments.reconstructions_dir(e, 'test').glob('*_rec.png')): print(e)")
fi
for exp in "${EXPS[@]}"; do
    echo "=== $exp $(date)"
    flock logs/.run_eval.lock $PY evaluation/run_eval.py --exp "$exp" --split test
done
$PY evaluation/compare_experiments.py --split test
