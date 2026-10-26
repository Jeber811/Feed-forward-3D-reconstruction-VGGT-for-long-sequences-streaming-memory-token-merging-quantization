#!/bin/bash
#SBATCH -J cut3r_and_grid
#SBATCH -p gpu
#SBATCH --account group4
#SBATCH --qos group4
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/cut3r_and_grid_%j.out
# 1) CUT3R baseline (transformers 5.x needs torch>=2.6; pin a 4.x that works with torch 2.5)
# 2) the 8-variant grid of compact/ on both scenes   3) score + plot everything
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
B=/home/zhyw86/WorkSpace/cap6411/baseline; P=$B/../dataset/scannet/process_scannet
S="$P/scene0000_00 $P/scene0013_02"
export ATE_DIR=$B/logs/ate_20261021
V=$B/logs/variants
cd $B
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ "$GPU" == "Quadro RTX 6000, 24576 MiB" ]] || { echo "ABORT: wrong GPU '$GPU'"; exit 1; }
echo "start $(date) on $(hostname)"

echo "===== cut3r: pin transformers ====="
conda activate cut3r
pip install -q "transformers==4.47.1" && python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
conda deactivate
bash run_all_ate.sh cut3r
echo "CUT3R STEP DONE"

echo "===== variant grid ====="
conda activate streamvggt
python compact/run_variant.py --variant all --scenes $S --out_dir $V
conda deactivate
echo "GRID STEP DONE"

echo "===== scoring + plots ====="
conda activate fastvggt
python tools/score_ate.py $ATE_DIR --scenes $S | tail -5
python tools/plot_results.py logs/memsweep_20261005_155935/memory_vs_frames.csv $ATE_DIR $ATE_DIR/baseline_results
python tools/score_ate.py $V --scenes $S | tail -5
echo "ALL GRID DONE"
