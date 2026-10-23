#!/bin/bash
#SBATCH --job-name=golden
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Golden check: proves your setup reproduces the reference numbers. Run once per person and after ANY setup change,
# on the final-tier GPU, from the repo root:
#   mkdir -p logs && source protocol/protocol.env && sbatch --qos=short $GPU_FINAL protocol/golden.sbatch
# RECORD=1 (maintainers only, with a version bump) rewrites protocol/golden/expected.json instead of checking.
source /etc/profile.d/modules.sh
export REPO=$SLURM_SUBMIT_DIR; source $REPO/protocol/protocol.env
source $CONDA_SH && conda activate $CONDA_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
verify_third_party || { echo "GOLDEN FAILED: third-party code differs from protocol/THIRD_PARTY.lock"; exit 2; }
G=$REPO/results/_golden_${USER}_$SLURM_JOB_ID; mkdir -p $G
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "golden | $(hostname) | $GPU | $(date)"
for cfg in "pi3 X=0" "pi3_chunk CH_SIZE=25 CH_OVERLAP=5 CH_KF=3"; do
  set -- $cfg; m=$1; shift
  cpu_inputs_ok $m && export CAP_CPU_INPUTS=1 || export CAP_CPU_INPUTS=0
  env "$@" python $HARNESS --model $m --ckpt_path $(ckpt_for $m) --data_dir $DATA --gt_ply_dir $GT_PLY_DIR \
    --scenes scene0013_02 --input_frame 60 --output_path $G/$m --rerun || { echo "GOLDEN FAILED: $m crashed"; exit 1; }
done
python $REPO/tools/check_golden.py $G $REPO/protocol/golden/expected.json --gpu "$GPU" ${RECORD:+--record}
