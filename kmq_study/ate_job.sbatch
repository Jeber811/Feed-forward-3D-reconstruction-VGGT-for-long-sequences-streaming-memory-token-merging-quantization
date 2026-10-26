#!/bin/bash
#SBATCH -J ate_baselines
#SBATCH -p gpu
#SBATCH --account group4
#SBATCH --qos group4
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH --requeue
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/ate_job_%j.out
# Env setup (skips finished envs/checkpoints) + all ATE runs; requeues itself after preemption
# and resumes, skipping models already finished.
B=/home/zhyw86/WorkSpace/cap6411/baseline
cd $B
echo "start $(date) on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
bash setup_more_envs.sh 2>&1 | tee -a setup_more_envs.log
export ATE_DIR=$B/logs/ate_20261021
bash run_all_ate.sh
