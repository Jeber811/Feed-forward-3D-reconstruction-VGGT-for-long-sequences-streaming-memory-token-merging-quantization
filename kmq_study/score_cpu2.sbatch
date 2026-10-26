#!/bin/bash
#SBATCH -J ate_score_cpu
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/ate_score_cpu_%j.out
# CPU only: ATE scoring of saved poses + combined plot/table. No model runs here.
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
B=/home/zhyw86/WorkSpace/cap6411/baseline; P=$B/../dataset/scannet/process_scannet
R=$B/logs/ate_20261021; V=$B/logs/variants
cd $B
conda activate fastvggt
echo "start $(date) on $(hostname)"
python tools/score_ate.py $R --scenes $P/scene0000_00 $P/scene0013_02 2>&1 | tail -40
python tools/plot_results.py logs/memsweep_20261005_155935/memory_vs_frames.csv $R $R/baseline_results
python tools/score_ate.py $V --scenes $P/scene0000_00 $P/scene0013_02 2>&1 | tail -3
echo "SCORE CPU DONE"
