#!/bin/bash
# One GPU job for a whole experiment: train -> repair val -> quick val check (EER)
# -> repair test -> full evaluation on test (evaluation/run_eval.py: evaluate.py + leaderboard row).
# Submitted by scripts/slurm/submit_chain.sh EXX partition. Each step runs only if the one before succeeded.
# Evaluations hold logs/.run_eval.lock: run_eval.py files its output as "the one new folder in
# evaluation/results/", so two evaluations must never overlap.
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00

set -euo pipefail
EXP=${1:?usage: sbatch scripts/slurm/train_and_check.sh EXX_name}
PY=${PY:-python}
export OMP_NUM_THREADS=4

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== $(date)  train $EXP"
$PY experiments/train.py --exp "$EXP"
echo "== $(date)  repair val"
$PY experiments/run_experiment.py --exp "$EXP" --split val
echo "== $(date)  quick_check val"
$PY -c "from fpr.validation.quick_check import quick_check; print(quick_check('$EXP', 'val'))"
echo "== $(date)  repair test"
$PY experiments/run_experiment.py --exp "$EXP" --split test
echo "== $(date)  evaluate test (waits for any other evaluation to finish)"
flock logs/.run_eval.lock $PY evaluation/run_eval.py --exp "$EXP" --split test
echo "== $(date)  done"
