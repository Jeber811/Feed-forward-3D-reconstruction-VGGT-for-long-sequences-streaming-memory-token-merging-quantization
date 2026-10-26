#!/bin/bash
# Run VGGT / FastVGGT / StreamVGGT on 2 ScanNet scenes with the same FastVGGT eval pipeline,
# tracking GPU memory two ways:
#   - PyTorch peak allocated/reserved per scene (memory.json, from eval_scannet_mem.py)
#   - nvidia-smi device memory sampled every 500 ms per run (gpu_<model>_N<frames>.csv)
#
# Usage (on a GPU node):  bash run_baselines.sh [input_frames ...]     default: 100
# Example:                bash run_baselines.sh 32 100 200
#
# Everything goes to logs/run_<timestamp>/:
#   run.log                      full console output of this script
#   <model>_N<frames>.log        per-run python output
#   gpu_<model>_N<frames>.csv    nvidia-smi samples (timestamp, MiB used, GPU util %)
#   results/<model>/...          eval outputs (metrics.json, memory.json, plots)
#   summary.csv                  one row per (model, frames, scene)

source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1    # live log output through tee
export PYTHONIOENCODING=utf-8  # node locale is latin-1; eval scripts print emoji
B=~/WorkSpace/cap6411/baseline
DATA=~/WorkSpace/cap6411/dataset/scannet/process_scannet
GT=/squash/scannetv2/scans
FRAMES=${@:-100}

LOGDIR=$B/logs/run_$(date +%Y%m%d_%H%M%S)
mkdir -p $LOGDIR
exec > >(tee -a $LOGDIR/run.log) 2>&1

echo "Run started $(date) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "Scenes: $(ls $DATA | tr '\n' ' ')"
echo "Frames: $FRAMES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Logs:   $LOGDIR"

# run_model <name> <conda_env> <eval_script> <frames> [extra args...]
run_model() {
  local name=$1 env=$2 script=$3 n=$4
  shift 4
  local log=$LOGDIR/${name}_N${n}.log
  local gpu_csv=$LOGDIR/gpu_${name}_N${n}.csv

  echo
  echo "===== $name | input_frame=$n | $(date +%H:%M:%S) ====="
  conda activate $env

  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 > $gpu_csv &
  local smi_pid=$!

  local start=$SECONDS
  python $script \
    --data_dir $DATA --gt_ply_dir $GT --num_scenes 2 --input_frame $n \
    --output_path $LOGDIR/results/$name "$@" 2>&1 | tee $log
  local status=${PIPESTATUS[0]}

  kill $smi_pid 2>/dev/null
  wait $smi_pid 2>/dev/null
  conda deactivate

  local smi_peak=$(awk -F', ' 'BEGIN{m=0} $2>m{m=$2} END{print m}' $gpu_csv)
  echo "----- $name N=$n: exit=$status, wall=$((SECONDS - start))s, nvidia-smi peak=${smi_peak} MiB"
  if grep -qi "out of memory" $log; then
    echo "----- $name N=$n: CUDA OUT OF MEMORY (see $log)"
  fi
}

cd $B/FastVGGT
for N in $FRAMES; do
  echo
  echo "########## input_frame=$N ##########"
  run_model vggt fastvggt eval/eval_scannet_mem.py $N \
    --ckpt_path $B/ckpt/model_tracker_fixed_e20.pt
  run_model fastvggt fastvggt eval/eval_scannet_mem.py $N --merging 0 \
    --ckpt_path $B/ckpt/model_tracker_fixed_e20.pt
  run_model streamvggt streamvggt eval/eval_scannet_streamvggt.py $N \
    --ckpt_path $B/ckpt/streamvggt_checkpoints.pth
done

echo
conda activate fastvggt
python $B/tools/summarize.py $LOGDIR
conda deactivate
echo "Run finished $(date). Logs in $LOGDIR"
echo "BASELINES DONE"
