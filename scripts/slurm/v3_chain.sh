#!/bin/bash
# A v3 ResShift experiment end to end on one GPU (E12-E16, E26). From the repo root, after the v3 data
# exists (preprocessing 05-07 with FPR_DATA_VERSION=v3, then scripts/slurm/make_train_v3.sh):
#     mkdir -p logs && sbatch scripts/slurm/v3_chain.sh E12_norm_ccdf_cldice_cv
# calibrate the null lams (once) -> train (resumes from last.pt) -> CCDF val sweep (v3 rule) -> repair test
# -> cost -> evaluate test -> v3 comparison. Needs an A40 / L40 for NFIQ 2 in the evaluation.
#SBATCH --job-name=v3_chain
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/v3_chain_%j.out

set -euo pipefail
EXP=${1:?usage: sbatch scripts/slurm/v3_chain.sh EXX_name}
PY=${PY:-python}
export FPR_DATA_VERSION=v3 OMP_NUM_THREADS=4
FOLDER=$($PY -c "from fpr import experiments; print(experiments.experiment_dir('$EXP'))")

nvidia-smi --query-gpu=name,memory.total --format=csv
if grep -q '"lam": null\|": null' "$FOLDER/params.json" && [ ! -f "$FOLDER/calibration.json" ]; then
    echo "== $(date)  calibrate"
    $PY experiments/train.py --exp "$EXP" --calibrate 20
fi
echo "== $(date)  train"
$PY experiments/train.py --exp "$EXP"
echo "== $(date)  CCDF val sweep"
$PY -m fpr.validation.ccdf_sweep_v3 --exp "$EXP"
echo "== $(date)  repair test"
$PY experiments/run_experiment.py --exp "$EXP" --split test
echo "== $(date)  cost"
$PY experiments/benchmark.py --exp "$EXP"
echo "== $(date)  evaluate test (waits for any other evaluation to finish)"
flock logs/.run_eval.lock $PY evaluation/run_eval.py --exp "$EXP" --split test
flock logs/.run_eval.lock $PY evaluation/compare_experiments.py --split test
echo "== $(date)  done"
