#!/bin/bash
# Pose accuracy (ATE) + memory for all models on 2 ScanNet scenes, same frames for every model.
#
# New models (pi3, cut3r, spann3r, current official VGGT) also get an exact OOM search (--bisect),
# with memory rows appended to the main sweep CSV. Models swept before get ATE only; their memory
# rows from these runs go to <results>/mem_from_ate.csv so the earlier sweep curves stay as they were.
#
# Usage (on a GPU node):  bash run_all_ate.sh [model ...]     default: all, in the order below

source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
B=~/WorkSpace/cap6411/baseline
C=$B/ckpt
P=~/WorkSpace/cap6411/dataset/scannet/process_scannet
SCENES="$P/scene0000_00 $P/scene0013_02"
SWEEP=$B/logs/memsweep_20261005_155935
R=${ATE_DIR:-$B/logs/ate_$(date +%Y%m%d)}  # fixed by the batch job so a requeue resumes in the same dir
mkdir -p $R
exec > >(tee -a $R/run_all_ate.log) 2>&1

# model | conda env | checkpoint | memory csv | extra args
SPECS=(
  "pi3|pi3|$C/pi3_model.safetensors|$SWEEP/memory_vs_frames.csv|--bisect"
  "cut3r|cut3r|$C/cut3r_512_dpt_4_64.pth|$SWEEP/memory_vs_frames.csv|--bisect"
  "spann3r|spann3r|$C/spann3r/spann3r.pth|$SWEEP/memory_vs_frames.csv|--bisect"
  "vggt-current|vggt|$C/vggt_1b_model.pt|$SWEEP/memory_vs_frames.csv|--bisect"
  "vggt|fastvggt|$C/model_tracker_fixed_e20.pt|$R/mem_from_ate.csv|"
  "fastvggt|fastvggt|$C/model_tracker_fixed_e20.pt|$R/mem_from_ate.csv|"
  "streamvggt|streamvggt|$C/streamvggt_checkpoints.pth|$R/mem_from_ate.csv|"
  "vggt-official|quantvggt|$C/model_tracker_fixed_e20.pt|$R/mem_from_ate.csv|"
  "quantvggt|quantvggt|$C/model_tracker_fixed_e20.pt|$R/mem_from_ate.csv|"
)
ONLY=" $* "

echo "ATE runs started $(date) on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# All results must come from the same 24 GB card as the earlier sweeps
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
if [[ "$GPU" != "Quadro RTX 6000, 24576 MiB" ]]; then
  echo "ABORT: expected 'Quadro RTX 6000, 24576 MiB', got '${GPU:-no GPU}'"
  exit 1
fi
echo "Results: $R"

for spec in "${SPECS[@]}"; do
  IFS='|' read -r model env ckpt memcsv extra <<< "$spec"
  [ -n "$*" ] && [[ "$ONLY" != *" $model "* ]] && continue
  [ -e $R/.done_$model ] && { echo "$model already done, skipping"; continue; }
  echo
  echo "===== $model ($env) | $(date +%H:%M:%S) ====="
  conda activate $env
  python $B/tools/infer_poses.py --model $model --ckpt $ckpt --scenes $SCENES \
    --out_dir $R --mem_csv $memcsv $extra 2>&1 | tee $R/${model}.log
  rc=${PIPESTATUS[0]}; echo "----- $model: exit=$rc"; [ $rc -eq 0 ] && touch $R/.done_$model
  conda deactivate
done

echo
echo "===== scoring | $(date +%H:%M:%S) ====="
conda activate fastvggt
python $B/tools/score_ate.py $R --scenes $SCENES 2>&1 | tail -5
grep -h "\[RESULT\]" $R/*.log
conda deactivate
echo "ATE runs finished $(date)"
echo "ALL ATE DONE"
