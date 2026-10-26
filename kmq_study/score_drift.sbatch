#!/bin/bash
#SBATCH -J drift_prefix1000
#SBATCH -p normal
#SBATCH -A group4
#SBATCH --qos=group4
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/drift_prefix1000_%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
B=/home/zhyw86/WorkSpace/cap6411/baseline
P=/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet
cd "$B"
/home/zhyw86/anaconda3/envs/fastvggt/bin/python tools/score_drift.py \
    --results logs/prefix_variants --data "$P" --output logs/drift_prefix1000 \
    --scenes scene0000_00 scene0013_02 scene0050_00 scene0249_00 scene0550_00 scene0621_00
echo "DRIFT JOB DONE $(date)"

