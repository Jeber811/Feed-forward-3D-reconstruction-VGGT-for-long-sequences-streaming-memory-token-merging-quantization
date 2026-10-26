#!/bin/bash
#SBATCH -J ate_cut3r
#SBATCH -p gpu
#SBATCH --qos preempt
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH --requeue
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/ate_cut3r_%j.out
# CUT3R only (adapter import fix), into the same results dir as the main ATE job.
cd /home/zhyw86/WorkSpace/cap6411/baseline
export ATE_DIR=$PWD/logs/ate_20261021
bash run_all_ate.sh cut3r
