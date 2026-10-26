#!/bin/bash
#SBATCH -J ate_followup
#SBATCH -p gpu
#SBATCH --account group4
#SBATCH --qos group4
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH --requeue
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/ate_followup_%j.out
# 1) score the finished baselines  2) smoke-test compact/  3) CUT3R  4) re-score with CUT3R
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
B=/home/zhyw86/WorkSpace/cap6411/baseline; P=$B/../dataset/scannet/process_scannet
S="$P/scene0000_00 $P/scene0013_02"
export ATE_DIR=$B/logs/ate_20261021
cd $B
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ "$GPU" == "Quadro RTX 6000, 24576 MiB" ]] || { echo "ABORT: wrong GPU '$GPU'"; exit 1; }
echo "start $(date) on $(hostname)"
score() { conda activate fastvggt; python tools/score_ate.py $ATE_DIR --scenes $S 2>&1 | tail -40; conda deactivate; }
echo "===== scoring baselines ====="; score; echo "SCORING DONE"
echo "===== smoke test ====="; bash compact/smoke_test.sbatch; echo "SMOKE STEP DONE"
echo "===== cut3r ====="; bash run_all_ate.sh cut3r
echo "===== re-scoring ====="; score
echo "FOLLOWUP DONE"
