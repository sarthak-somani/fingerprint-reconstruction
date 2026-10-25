#!/bin/bash
# Submit one experiment as ONE SLURM job (train_and_check.sh): train -> repair val -> quick val check
# -> repair test -> full test evaluation (run_eval.py, leaderboard row). From the repo root:
#     bash scripts/slurm/submit_chain.sh E07_aux_minutiae_cldice a40        # partition, CPUs default 16
# Set PY to your Python (default: python on PATH). Log: logs/<EXX>_chain_<jobid>.out.
# Training resumes from checkpoints/last.pt if the job is resubmitted.
set -euo pipefail
EXP=${1:?usage: bash scripts/slurm/submit_chain.sh EXX_name partition [cpus]}
PARTITION=${2:?give a partition, e.g. a40}
CPUS=${3:-16}
cd "$(dirname "$0")/../.."                                         # repo root: the job's paths are relative
mkdir -p logs
job=$(sbatch --parsable --partition="$PARTITION" --qos="$PARTITION" --cpus-per-task="$CPUS" \
    --job-name="${EXP%%_*}_chain" --output="logs/${EXP%%_*}_chain_%j.out" \
    --export=ALL,PY="${PY:-python}" scripts/slurm/train_and_check.sh "$EXP")
echo "$EXP on $PARTITION ($CPUS CPUs): job $job (train -> val repair + check -> test repair + run_eval)"
