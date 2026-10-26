#!/bin/bash
#SBATCH -J pointmap2000
#SBATCH -p normal
#SBATCH -A group4
#SBATCH --qos=group4
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/pointmap2000_%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=/home/zhyw86/WorkSpace/cap6411/baseline
P=/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet
cd "$B"
/usr/bin/python tools/score_pointmaps.py \
    --results logs/variants --output-dir logs/pointmap_2000 --n 2000 \
    --scenes "$P"/scene0000_00 "$P"/scene0013_02 "$P"/scene0050_00 \
             "$P"/scene0249_00 "$P"/scene0550_00 "$P"/scene0621_00
echo "POINTMAP2000 JOB DONE $(date)"

