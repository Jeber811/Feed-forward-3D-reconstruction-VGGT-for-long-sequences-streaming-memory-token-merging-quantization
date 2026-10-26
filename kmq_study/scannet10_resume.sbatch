#!/bin/bash
#SBATCH -J scannet10_resume
#SBATCH -p normal
#SBATCH -A group4
#SBATCH --qos=group4
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o /home/zhyw86/WorkSpace/cap6411/baseline/logs/scannet10_resume_%j.out
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
B=/home/zhyw86/WorkSpace/cap6411/baseline
P=/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet
RAW=/squash/scannetv2/scans
cd "$B"
for scene in scene0010_00 scene0111_00 scene0188_00 scene0262_00 scene0390_00 scene0452_00 scene0533_00 scene0585_00 scene0629_00 scene0697_00; do
    src="$RAW/$scene/$scene.sens"
    final="$P/$scene"
    stage="$P/.incoming_$scene"
    if [[ -f "$final/.complete" ]]; then
        echo "[SKIP] $scene already complete"
        continue
    fi
    if [[ -e "$final" ]]; then
        echo "[STOP] Existing incomplete final target for $scene; inspect before resuming" >&2
        exit 1
    fi
    if [[ -e "$stage" ]]; then
        echo "[RESUME] $scene has a partial staging directory; the source archive will be replayed into it after the first job terminates"
    fi
    test -s "$src"
    echo "[START] $scene $(date)"
    /usr/bin/python tools/extract_sens.py "$src" "$stage"
    /usr/bin/python - "$stage" <<'PY'
import pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1])
ids={kind:{int(f.stem) for f in (p/kind).glob('*') if f.stem.isdigit()} for kind in ('color','depth','pose')}
if not ids['color'] or ids['color']!=ids['depth'] or ids['color']!=ids['pose']:
    raise SystemExit(f'incomplete extraction: '+str({k:len(v) for k,v in ids.items()}))
valid=0
for f in (p/'pose').glob('*.txt'):
    if np.isfinite(np.loadtxt(f)).all(): valid+=1
if valid<2000:
    raise SystemExit(f'only {valid} valid poses; expected >=2000')
(p/'.complete').write_text(f'frames={len(ids["color"])} valid_poses={valid}\n')
print(f'[VALID] {p.name}: frames={len(ids["color"])} valid_poses={valid}',flush=True)
PY
    mv "$stage" "$final"
    echo "[DONE] $scene $(date)"
done
echo "EXTRACT10 COMPLETE $(date)"

