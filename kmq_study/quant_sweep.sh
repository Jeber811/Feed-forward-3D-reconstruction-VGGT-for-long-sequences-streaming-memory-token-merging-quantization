#!/bin/bash
# Memory sweep for official VGGT and QuantVGGT (W4A4, released params), appended to an existing
# memsweep folder so they share memory_vs_frames.csv / .png with the other models.
#
# Usage (on a GPU node):  bash quant_sweep.sh <logs/memsweep_xxx> [scene]

source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1    # live log output through tee
export PYTHONIOENCODING=utf-8  # node locale is latin-1
B=~/WorkSpace/cap6411/baseline
D=$(realpath $1)
SCENE_DIR=~/WorkSpace/cap6411/dataset/scannet/process_scannet/${2:-scene0000_00}
CSV=$D/memory_vs_frames.csv
CKPT=$B/ckpt/model_tracker_fixed_e20.pt
FRAMES="25 50 100 150 200 250 300 350 400 500"
exec > >(tee -a $D/quant_sweep.log) 2>&1

echo "QuantVGGT sweep started $(date) on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
conda activate quantvggt
cd $B/FastVGGT

echo
echo "===== 1. check quantization is applied | $(date +%H:%M:%S) ====="
python $B/tools/check_quantvggt.py --ckpt $CKPT --scene_dir $SCENE_DIR --frames 8 || { echo "CHECK FAILED"; exit 1; }

echo
echo "===== 2. vggt-official | $(date +%H:%M:%S) ====="
python $B/tools/mem_sweep.py --model vggt-official --ckpt $CKPT --scene_dir $SCENE_DIR --out_csv $CSV \
  --frames $FRAMES 2>&1 | tee $D/vggt_official.log

echo
echo "===== 3. quantvggt | $(date +%H:%M:%S) ====="
python $B/tools/mem_sweep.py --model quantvggt --ckpt $CKPT --scene_dir $SCENE_DIR --out_csv $CSV \
  --frames $FRAMES 2>&1 | tee $D/quantvggt.log

echo
echo "===== results ====="
grep -h "\[RESULT\]" $D/vggt_official.log $D/quantvggt.log
python $B/tools/plot_mem.py $CSV
conda deactivate
echo "QuantVGGT sweep finished $(date)"
echo "QUANT SWEEP DONE"
