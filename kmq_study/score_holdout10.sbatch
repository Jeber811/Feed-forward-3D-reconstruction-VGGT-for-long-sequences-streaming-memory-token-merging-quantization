#!/bin/bash
#SBATCH -J scannet10_score
#SBATCH -p normal
#SBATCH --account=group4
#SBATCH --qos=group4
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 06:00:00
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/scannet10_score_%j.out
set -euo pipefail
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate fastvggt
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=/home/zhyw86/WorkSpace/cap6411/baseline
P=$B/../dataset/scannet/process_scannet
cd "$B"
SCENES=()
for s in scene0010_00 scene0111_00 scene0188_00 scene0262_00 scene0390_00 scene0452_00 scene0533_00 scene0585_00 scene0629_00 scene0697_00; do SCENES+=("$P/$s"); done
python tools/score_ate.py logs/holdout10 --scenes "${SCENES[@]}"
/usr/bin/python tools/score_pointmaps.py --results logs/holdout10 --output-dir logs/holdout10 --n 2000 --models chunk chunk+M chunk+K --scenes "${SCENES[@]}"
echo "HOLDOUT10 SCORING COMPLETE $(date)"
