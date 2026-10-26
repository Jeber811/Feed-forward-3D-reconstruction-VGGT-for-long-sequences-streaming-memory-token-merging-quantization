#!/bin/bash
# Follow-up memory experiments, appended to an existing memsweep folder (new rows get their own labels):
#   1. check that token-chunked MLPs give the same VGGT predictions
#   2. streamvggt+expandable : StreamVGGT with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   3. vggt+mlpchunk         : VGGT with token-chunked block MLPs
# then redraws memory_vs_frames.png.
#
# Usage (on a GPU node):  bash memory_followups.sh <logs/memsweep_xxx> [scene]

source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1    # live log output through tee
export PYTHONIOENCODING=utf-8  # node locale is latin-1
B=~/WorkSpace/cap6411/baseline
D=$(realpath $1)
SCENE_DIR=~/WorkSpace/cap6411/dataset/scannet/process_scannet/${2:-scene0000_00}
CSV=$D/memory_vs_frames.csv
exec > >(tee -a $D/followups.log) 2>&1

echo "Follow-ups started $(date) on $(hostname), job ${SLURM_JOB_ID:-none}"
cd $B/FastVGGT

echo
echo "===== 1. check token-chunked MLP | $(date +%H:%M:%S) ====="
conda activate fastvggt
python $B/tools/check_mlp_chunk.py --ckpt $B/ckpt/model_tracker_fixed_e20.pt --scene_dir $SCENE_DIR --frames 16
conda deactivate

echo
echo "===== 2. streamvggt+expandable | $(date +%H:%M:%S) ====="
conda activate streamvggt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $B/tools/mem_sweep.py --model streamvggt \
  --label streamvggt+expandable --ckpt $B/ckpt/streamvggt_checkpoints.pth --scene_dir $SCENE_DIR \
  --out_csv $CSV --max_frames 2000 --log_every 10 2>&1 | tee $D/streamvggt_expandable.log
conda deactivate

echo
echo "===== 3. vggt+mlpchunk | $(date +%H:%M:%S) ====="
conda activate fastvggt
python $B/tools/mem_sweep.py --model vggt --label vggt+mlpchunk --mlp_chunk 65536 \
  --ckpt $B/ckpt/model_tracker_fixed_e20.pt --scene_dir $SCENE_DIR --out_csv $CSV \
  --frames 100 200 300 387 400 450 500 600 700 800 900 1000 1200 1500 2000 2>&1 | tee $D/vggt_mlpchunk.log

echo
echo "===== results ====="
grep -h "\[RESULT\]" $D/*.log | sort -u
python $B/tools/plot_mem.py $CSV
conda deactivate
echo "Follow-ups finished $(date)"
echo "FOLLOWUPS DONE"
