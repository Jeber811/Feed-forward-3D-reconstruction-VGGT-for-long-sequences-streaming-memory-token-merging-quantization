#!/bin/bash
#SBATCH -J compact_smoke
#SBATCH -p gpu
#SBATCH --qos preempt
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/compact_smoke_%j.out
# Smoke test: stream-full must match the original StreamVGGT; every letter must run.
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
B=/home/zhyw86/WorkSpace/cap6411/baseline; P=$B/../dataset/scannet/process_scannet
O=$B/logs/compact_smoke
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ "$GPU" == "Quadro RTX 6000, 24576 MiB" ]] || { echo "ABORT: wrong GPU '$GPU'"; exit 1; }
cd $B
conda activate streamvggt
python tools/infer_poses.py --model streamvggt --label streamvggt-original --ckpt ckpt/streamvggt_checkpoints.pth \
  --scenes $P/scene0013_02 --frames 50 --out_dir $O
python compact/run_variant.py --variant stream-full --scenes $P/scene0013_02 --frames 50 --out_dir $O
python compact/run_variant.py --variant stream stream+K stream+M stream+Q stream+K+M+Q \
  --set budget_mb=1024 --scenes $P/scene0013_02 --frames 50 200 --out_dir $O
conda activate fastvggt
python tools/score_ate.py $O --scenes $P/scene0013_02 | tail -20
echo SMOKE DONE
