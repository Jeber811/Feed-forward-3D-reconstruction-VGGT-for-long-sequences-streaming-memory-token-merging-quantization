#!/bin/bash
# Creates conda envs for vggt, fastvggt, streamvggt (torch 2.3.1 + CUDA 12.1)
set -e
source ~/anaconda3/etc/profile.d/conda.sh
B=~/WorkSpace/cap6411/baseline
TORCH="torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"

echo "===== [1/3] vggt ====="
conda create -y -n vggt python=3.10
conda activate vggt
pip install $TORCH
pip install -r $B/vggt/requirements.txt
pip install -r $B/vggt/requirements_demo.txt
pip install -e $B/vggt
conda deactivate

echo "===== [2/3] fastvggt ====="
conda create -y -n fastvggt python=3.10
conda activate fastvggt
pip install $TORCH
pip install -r $B/FastVGGT/requirements.txt
conda deactivate

echo "===== [3/3] streamvggt ====="
conda create -y -n streamvggt python=3.11 cmake=3.14.0
conda activate streamvggt
pip install $TORCH
pip install -r $B/StreamVGGT/requirements.txt
conda install -y "llvm-openmp<16"
conda deactivate

echo "===== verify ====="
for e in vggt fastvggt streamvggt; do
  conda activate $e
  python -c "import torch; print(\"$e\", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  conda deactivate
done
echo "ALL ENVS DONE"
