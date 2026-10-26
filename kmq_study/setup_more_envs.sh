#!/bin/bash
# Conda envs for pi3, cut3r, spann3r (+ their Google Drive checkpoints and RoPE CUDA kernels).
# Run on a GPU node:  bash setup_more_envs.sh 2>&1 | tee setup_more_envs.log
set -e
source ~/anaconda3/etc/profile.d/conda.sh
B=~/WorkSpace/cap6411/baseline
C=$B/ckpt
CU121="--index-url https://download.pytorch.org/whl/cu121"
# skip an env that already exists and imports torch (makes this script safe to re-run)
need_env() { ! { [ -x ~/anaconda3/envs/$1/bin/python ] && ~/anaconda3/envs/$1/bin/python -c "import torch" 2>/dev/null; } || { echo "$1 env exists, skipping"; false; }; }

echo "===== [1/3] pi3 ====="
if need_env pi3; then
  conda create -y -q -n pi3 python=3.10
  conda activate pi3
  pip install -q torch==2.5.1 torchvision==0.20.1 $CU121
  pip install -q -r $B/Pi3/requirements.txt
  conda deactivate
fi

echo "===== [2/3] cut3r ====="
if need_env cut3r; then
  conda create -y -q -n cut3r python=3.11 cmake=3.14.0
  conda activate cut3r
  pip install -q torch==2.5.1 torchvision==0.20.1 $CU121
  pip install -q -r $B/CUT3R/requirements.txt
  pip install -q gdown
  conda deactivate
fi

echo "===== [3/3] spann3r ====="
if need_env spann3r; then
  conda create -y -q -n spann3r python=3.9 cmake=3.14.0
  conda activate spann3r
  pip install -q torch==2.3.1 torchvision==0.18.1 $CU121
  pip install -q -r $B/spann3r/requirements.txt
  conda deactivate
fi

echo "===== checkpoints from Google Drive ====="
conda activate cut3r
cd $C
[ -s cut3r_512_dpt_4_64.pth ] || gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O cut3r_512_dpt_4_64.pth
[ -s spann3r/spann3r.pth ] || gdown --folder "https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing" -O spann3r_gdrive
f=$(find $C/spann3r_gdrive -name spann3r.pth 2>/dev/null | head -1)
[ -n "$f" ] && [ ! -e $C/spann3r/spann3r.pth ] && ln -s "$(realpath "$f")" $C/spann3r/spann3r.pth
ls -laL $C/cut3r_512_dpt_4_64.pth $C/spann3r/*.pth 2>/dev/null || true
conda deactivate

echo "===== RoPE CUDA kernels (optional; repos fall back to PyTorch RoPE if this fails) ====="
set +e
if type module >/dev/null 2>&1; then module load cuda/12.1; fi
which nvcc && nvcc --version | tail -1
for spec in "cut3r:$B/CUT3R/src/croco/models/curope" "spann3r:$B/spann3r/croco/models/curope"; do
  env=${spec%%:*}; dir=${spec#*:}
  conda activate $env
  (cd $dir && python setup.py build_ext --inplace > build.log 2>&1) && echo "$env curope: built" || echo "$env curope: FAILED (see $dir/build.log), using PyTorch RoPE"
  conda deactivate
done

echo "===== verify ====="
for e in pi3 cut3r spann3r; do
  conda activate $e
  python -c "import torch; print('$e', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  conda deactivate
done
echo "ALL MORE ENVS DONE"
