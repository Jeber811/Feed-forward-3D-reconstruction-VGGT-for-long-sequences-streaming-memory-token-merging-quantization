#!/bin/bash
# GPU memory vs. number of input images for VGGT / FastVGGT / StreamVGGT, until OOM.
#
# Usage (on a GPU node):  bash mem_sweep.sh [scene]        default scene: scene0000_00 (5578 frames)
#
# Everything goes to logs/memsweep_<timestamp>/:
#   sweep.log                  full console output of this script
#   <model>.log                per-model python output
#   gpu_<model>.csv            nvidia-smi samples every 500 ms (timestamp, MiB used, GPU util %)
#   memory_vs_frames.csv       one row per measurement (all models)
#   memory_vs_frames.png       plot of the curves with the 24 GB limit

source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1    # live log output through tee
export PYTHONIOENCODING=utf-8  # node locale is latin-1; eval scripts print emoji
B=~/WorkSpace/cap6411/baseline
SCENE=${1:-scene0000_00}
SCENE_DIR=~/WorkSpace/cap6411/dataset/scannet/process_scannet/$SCENE

LOGDIR=$B/logs/memsweep_$(date +%Y%m%d_%H%M%S)
mkdir -p $LOGDIR
CSV=$LOGDIR/memory_vs_frames.csv
exec > >(tee -a $LOGDIR/sweep.log) 2>&1

echo "Memory sweep started $(date) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "Scene: $SCENE ($(ls $SCENE_DIR/color | wc -l) frames)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Logs:  $LOGDIR"

# sweep <model> <conda_env> <ckpt> [extra args...]
sweep() {
  local model=$1 env=$2 ckpt=$3
  shift 3
  echo
  echo "===== $model | $(date +%H:%M:%S) ====="
  conda activate $env
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 > $LOGDIR/gpu_${model}.csv &
  local smi_pid=$!
  local start=$SECONDS
  python $B/tools/mem_sweep.py --model $model --ckpt $ckpt --scene_dir $SCENE_DIR \
    --out_csv $CSV "$@" 2>&1 | tee $LOGDIR/${model}.log
  local status=${PIPESTATUS[0]}
  kill $smi_pid 2>/dev/null
  wait $smi_pid 2>/dev/null
  conda deactivate
  local smi_peak=$(awk -F', ' 'BEGIN{m=0} $2>m{m=$2} END{print m}' $LOGDIR/gpu_${model}.csv)
  echo "----- $model: exit=$status, wall=$((SECONDS - start))s, nvidia-smi peak=${smi_peak} MiB"
}

cd $B/FastVGGT
sweep vggt fastvggt $B/ckpt/model_tracker_fixed_e20.pt
sweep fastvggt fastvggt $B/ckpt/model_tracker_fixed_e20.pt
sweep streamvggt streamvggt $B/ckpt/streamvggt_checkpoints.pth --max_frames 2000 --log_every 10

echo
echo "===== results ====="
grep -h "\[RESULT\]" $LOGDIR/vggt.log $LOGDIR/fastvggt.log $LOGDIR/streamvggt.log
conda activate fastvggt
python $B/tools/plot_mem.py $CSV
conda deactivate
echo "Memory sweep finished $(date). Logs in $LOGDIR"
echo "MEMSWEEP DONE"
