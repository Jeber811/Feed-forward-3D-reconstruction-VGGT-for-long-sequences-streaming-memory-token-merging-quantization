#!/bin/bash
#SBATCH --job-name=build_curope
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Builds croco's CUDA RoPE kernel for CUT3R / TTT3R / Spann3R (sm_75 RTX 6000 + sm_80/86 Ampere; NOT V100).
# nvcc 12.3 rejects gcc 13 -> gcc-11. Usage (repo root): sbatch scripts/build_curope.sbatch <third_party_dir>
set -euo pipefail
source /etc/profile.d/modules.sh && module load cuda/12.3
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh && conda activate /home/ja396977/.conda/envs/cap6411_env
TP=${1:?third_party dir}
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
for d in "$TP/CUT3R/src/croco/models/curope" "$TP/spann3r/croco/models/curope"; do
  (cd "$d" && rm -rf build && python setup.py build_ext --inplace)
done
ln -sf "$TP"/CUT3R/src/croco/models/curope/curope.cpython-310-x86_64-linux-gnu.so "$TP/TTT3R/src/croco/models/curope/"
echo "curope built"
