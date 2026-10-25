#!/bin/bash
# A CCDF re-sampling of an already trained model (no training): E21-E23 (E09's sampler on E06-E08's weights,
# configuration chosen on v2 val) and E24/E25 (E12's fixed configuration, nothing to select). From the repo root:
#     mkdir -p logs && sbatch scripts/slurm/resample_chain.sh E22_ccdf_aux_minutiae_cldice
# [v2 val sweep] -> repair test of each data version the experiment is reported on -> cost -> evaluate -> compare.
#SBATCH --job-name=resample
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/resample_%j.out

set -euo pipefail
EXP=${1:?usage: sbatch scripts/slurm/resample_chain.sh EXX_name}
PY=${PY:-python}
export OMP_NUM_THREADS=4
VERSIONS=$($PY -c "from fpr import experiments; print(experiments.load_params('$EXP').get('data_version', 'v2 v3'))")

if [ "$VERSIONS" = "v2 v3" ]; then
    echo "== $(date)  CCDF val sweep (v2)"
    FPR_DATA_VERSION=v2 $PY -m fpr.validation.ccdf_sweep --exp "$EXP"
fi
FPR_DATA_VERSION=v2 $PY experiments/benchmark.py --exp "$EXP"
for v in $VERSIONS; do
    echo "== $(date)  repair + evaluate $v test"
    FPR_DATA_VERSION=$v $PY experiments/run_experiment.py --exp "$EXP" --split test
    FPR_DATA_VERSION=$v flock logs/.run_eval.lock $PY evaluation/run_eval.py --exp "$EXP" --split test
    FPR_DATA_VERSION=$v flock logs/.run_eval.lock $PY evaluation/compare_experiments.py --split test
done
echo "== $(date)  done"
