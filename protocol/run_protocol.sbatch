#!/bin/bash
#SBATCH --job-name=proto
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=88G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# The ONLY way to produce report numbers (see protocol/PROTOCOL.md). Submit from the repo root:
#   mkdir -p logs && source protocol/protocol.env
#   sbatch --qos=short --requeue $GPU_FINAL \
#     --export=ALL,MODEL=pi3_chunk,NAME=pi3_chunk_K,NS="1000",TIER=rtx6000,EXTRA_ENV="CH_SIZE=200 CH_OVERLAP=30 CH_KF=20" \
#     protocol/run_protocol.sbatch
# Smoke tests: $GPU_SMOKE and TIER=smoke_<tag> (never used in report tables). Optional: SCENES, HARNESS_ARGS.
source /etc/profile.d/modules.sh
export REPO=$SLURM_SUBMIT_DIR; source $REPO/protocol/protocol.env
source $CONDA_SH && conda activate $CONDA_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
: ${MODEL:?set MODEL} ${TIER:?set TIER} ${NS:?set NS}
NAME=${NAME:-$MODEL}; SCENES=${SCENES:-$DEV_SCENES}
verify_third_party || { echo "ABORT: third-party code differs from protocol/THIRD_PARTY.lock"; exit 2; }
cpu_inputs_ok $MODEL && export CAP_CPU_INPUTS=1 || export CAP_CPU_INPUTS=0
OUT=$RESULTS_ROOT/$TIER/$NAME; mkdir -p $OUT
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
DIRTY=$(git -C $REPO diff --quiet HEAD -- tools protocol 2>/dev/null || echo "-dirty")
cat > $OUT/run_config_${SLURM_JOB_ID}.json <<JSON
{"protocol": "$PROTOCOL_VERSION", "git_commit": "$(git -C $REPO rev-parse --short HEAD)$DIRTY", "user": "$USER",
 "job": "$SLURM_JOB_ID", "host": "$(hostname)", "gpu": "$GPU", "conda_env": "$CONDA_ENV", "tier": "$TIER",
 "model": "$MODEL", "name": "$NAME", "lengths": "$NS", "scenes": "$SCENES", "extra_env": "$EXTRA_ENV",
 "harness_args": "$HARNESS_ARGS", "CAP_DTYPE": "$CAP_DTYPE", "CAP_CPU_INPUTS": "$CAP_CPU_INPUTS",
 "CH_MEMORDER": "$CH_MEMORDER", "CH_ALIGN": "$CH_ALIGN", "date": "$(date -Iseconds)"}
JSON
echo "protocol v$PROTOCOL_VERSION | job $SLURM_JOB_ID on $(hostname) | $GPU | $MODEL as $NAME | Ns=$NS | $(date)"
for N in $NS; do
  echo "===== $NAME N=$N | $(date +%T)"
  env $EXTRA_ENV python $HARNESS --model $MODEL --ckpt_path $(ckpt_for $MODEL) --data_dir $DATA --gt_ply_dir $GT_PLY_DIR \
    --scenes $SCENES --input_frame $N --output_path $OUT $HARNESS_ARGS || echo "!!!!! $NAME N=$N exit=$?"
done
echo "DONE $(date)"
