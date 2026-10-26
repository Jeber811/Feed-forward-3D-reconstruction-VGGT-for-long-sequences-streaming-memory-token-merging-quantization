#!/bin/bash
#SBATCH -J scannet10_holdout
#SBATCH -p gpu
#SBATCH --account=group4
#SBATCH --qos=group4
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/scannet10_holdout_%j.out
set -euo pipefail
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fastvggt
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
unset SCANNET_FRAME_SELECTION
B=/home/zhyw86/WorkSpace/cap6411/baseline
P=$B/../dataset/scannet/process_scannet
cd "$B"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ "$GPU" == "Quadro RTX 6000, 24576 MiB" ]] || { echo "ABORT wrong GPU: $GPU"; exit 1; }
SCENES=()
for s in scene0010_00 scene0111_00 scene0188_00 scene0262_00 scene0390_00 scene0452_00 scene0533_00 scene0585_00 scene0629_00 scene0697_00; do
  test -f "$P/$s/.complete" || { echo "Missing validated scene $s"; exit 1; }
  SCENES+=("$P/$s")
done
echo "START $(date) on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
python compact/run_variant.py --variant chunk chunk+M chunk+K --scenes "${SCENES[@]}" --frames 2000 --out_dir "$B/logs/holdout10"
echo "HOLDOUT10 INFERENCE COMPLETE $(date)"
