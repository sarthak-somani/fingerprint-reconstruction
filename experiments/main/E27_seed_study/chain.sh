#!/bin/bash
# E27 seed study on one GPU. From the repo root:
#     sbatch --job-name=E27_train_s1 experiments/main/E27_seed_study/chain.sh train 1   # E02 + E07 with training seed 1,
#                                          # both on this one GPU at once, then post 1 (sample + evaluate + analysis)
#     sbatch -c 16 --mem=64G -t 4:00:00 experiments/main/E27_seed_study/chain.sh post 0  # seed-0 checkpoints: sample + evaluate
#     sbatch experiments/main/E27_seed_study/chain.sh post all                           # whatever is missing, then the analysis
# "post" runs pipeline.py: CCDF repairs of v3 test with sampler seeds 0-2 for every finished checkpoint of those
# training seeds (3 at a time on the GPU), evaluate.py on each (3 at a time; only on an A40 / L40, NFIQ 2 needs
# libssl 1.1), then analysis.py. Every step skips what is already complete, so any stage can be resubmitted;
# training resumes from last.pt (the source trainers' own resume).
#SBATCH --job-name=E27_chain
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=experiments/main/E27_seed_study/logs/chain_%x_%j.out

set -uo pipefail
EXP=experiments/main/E27_seed_study
STAGE=${1:?stage: train or post}
SEED=${2:?training seed (0, 1, 2 or all)}
PY=${PY:-python}
export OMP_NUM_THREADS=4

echo "== $(date)  E27 $STAGE $SEED on $(hostname), job ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} CPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$STAGE" = train ]; then
    [ "$SEED" -ge 1 ] || { echo "train needs seed 1 or 2 (seed 0 = the original E02 / E07 runs)"; exit 1; }
    pids=()
    for SRC in E02_resshift E07_aux_minutiae_cldice; do       # both trainers share this GPU
        FPR_DATA_VERSION=v2 $PY $EXP/train.py --source $SRC --seed "$SEED" \
            > "$EXP/logs/train_${SRC}_s${SEED}_${SLURM_JOB_ID:-local}.out" 2>&1 &
        pids+=($!)
        echo "started $SRC seed $SEED (pid $!)"
    done
    ( sleep 900; nvidia-smi ) &                                # memory / utilisation with both running
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    echo "== $(date)  training finished (status $status)"
    for SRC in E02_resshift E07_aux_minutiae_cldice; do tail -n 2 "$EXP/checkpoints/${SRC}_s${SEED}/log.csv"; done
elif [ "$STAGE" != post ]; then
    echo "unknown stage $STAGE (train or post)"; exit 1
fi
FPR_DATA_VERSION=v3 $PY $EXP/pipeline.py post --train-seeds "$SEED"
echo "== $(date)  done"
