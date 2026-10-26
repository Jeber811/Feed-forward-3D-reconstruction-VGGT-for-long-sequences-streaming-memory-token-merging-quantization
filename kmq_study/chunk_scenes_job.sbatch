#!/bin/bash
#SBATCH -J chunk_and_scenes
#SBATCH -p gpu
#SBATCH --account group4
#SBATCH --qos group4
#SBATCH --gres=gpu:turing:1
#SBATCH -C gmem24
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH --requeue
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/chunk_and_scenes_%j.out
# Every step skips work that is already on disk, so a requeue resumes where it stopped.
#  1) extract 4 more ScanNet scenes        2) chunk smoke test (1 chunk must equal VGGT*)
#  3) chunk grid on the 2 original scenes  4) baselines on the new scenes
#  5) key stream + chunk variants on the new scenes   6) score everything
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
B=/home/zhyw86/WorkSpace/cap6411/baseline; P=$B/../dataset/scannet/process_scannet
OLD="$P/scene0000_00 $P/scene0013_02"
NEWIDS="scene0050_00 scene0249_00 scene0550_00 scene0621_00"
NEW=$(for s in $NEWIDS; do printf "%s " $P/$s; done)
V=$B/logs/variants; NB=$B/logs/ate_newscenes
cd $B
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ "$GPU" == "Quadro RTX 6000, 24576 MiB" ]] || { echo "ABORT: wrong GPU '$GPU'"; exit 1; }
echo "start $(date) on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"

echo "===== 1) extract scenes ====="
conda activate fastvggt
for s in $NEWIDS; do
  [ -f $P/$s/.done ] && { echo "$s already extracted"; continue; }
  python tools/extract_sens.py /squash/scannetv2/scans/$s/$s.sens $P/$s && touch $P/$s/.done
done
echo "STEP 1 DONE"

echo "===== 2) chunk smoke ====="
python compact/run_variant.py --variant chunk chunk+K chunk+M chunk+Q --scenes $P/scene0013_02 --frames 100 300 \
  --out_dir $B/logs/chunk_smoke
python tools/score_ate.py $B/logs/chunk_smoke --scenes $P/scene0013_02 | tail -12
echo "STEP 2 DONE"

echo "===== 3) chunk grid, original scenes ====="
python compact/run_variant.py --variant all-chunk --scenes $OLD --out_dir $V
echo "STEP 3 DONE"

echo "===== 4) baselines, new scenes ====="
mkdir -p $NB
for spec in "vggt|fastvggt|model_tracker_fixed_e20.pt" "fastvggt|fastvggt|model_tracker_fixed_e20.pt" \
            "streamvggt|streamvggt|streamvggt_checkpoints.pth" "pi3|pi3|pi3_model.safetensors" \
            "cut3r|cut3r|cut3r_512_dpt_4_64.pth" "spann3r|spann3r|spann3r/spann3r.pth"; do
  IFS='|' read -r m env ck <<< "$spec"
  [ -e $NB/.done_$m ] && { echo "$m already done"; continue; }
  conda activate $env
  python tools/infer_poses.py --model $m --ckpt ckpt/$ck --scenes $NEW --out_dir $NB --mem_csv $NB/memory.csv \
    --frames 10 25 50 100 200 300 500 1000 2000 3000 4000 2>&1 | grep -vE "%\|" | tee $NB/$m.log \
    && touch $NB/.done_$m
done
echo "STEP 4 DONE"

echo "===== 5) variants, new scenes ====="
conda activate streamvggt
python compact/run_variant.py --variant stream stream+K stream+K+M stream+K+M+Q --scenes $NEW \
  --frames 100 300 1000 2000 --out_dir $V
conda activate fastvggt
python compact/run_variant.py --variant chunk chunk+K chunk+M chunk+K+M+Q --scenes $NEW \
  --frames 100 300 1000 2000 --out_dir $V
echo "STEP 5 DONE"

echo "===== 6) scoring ====="
python tools/score_ate.py $V --scenes $OLD $NEW | tail -3
python tools/score_ate.py $NB --scenes $NEW | tail -3
echo "ALL CHUNK+SCENES DONE"
