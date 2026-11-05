"""v0.4 cross-tier check: the same configuration on the same scenes must give the same numbers on two final tiers.

Compares a run on the Newton tier (v100_cap24) against the identical run already recorded on the RTX 6000, scene by
scene, with the golden-check tolerances (accuracy) and MEM_REL (peak memory). Compute node only:
    srun -p normal -c 2 --mem=4G -t 00:05:00 python tools/crosscheck_tiers.py \\
        <rtx6000 run dir> <v100_cap24 run dir> --n 500 [--scenes ...]
The two runs must use the same model and EXTRA_ENV (checked from their run_config files). Exit 0 = CROSSCHECK PASS.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_golden import KEYS, MEM_REL  # noqa: E402  (same tolerances as the golden check)

SAME = ("model", "extra_env", "harness_args", "CAP_DTYPE", "CAP_CPU_INPUTS", "CH_MEMORDER", "CH_ALIGN")


def cfg(run: Path):
    c = sorted(run.glob("run_config_*.json"))
    if not c:
        sys.exit(f"{run}: no run_config_*.json")
    return json.loads(c[-1].read_text())


def scene_vals(run: Path, n: int, scene: str):
    d = run / f"input_frame_{n}" / scene
    mem, met = d / "memory.json", d / "metrics.json"
    if not (mem.exists() and met.exists()):
        return None
    m, x = json.loads(mem.read_text()), json.loads(met.read_text())
    return {**{k: x[k] for k in KEYS if k in x}, "peak_allocated_gb": m["peak_allocated_gb"],
            "num_frames": m.get("num_frames")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path, help="run dir on the reference tier (rtx6000)")
    ap.add_argument("new", type=Path, help="run dir on the new tier (v100_cap24)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scenes", nargs="*", help="default: the new run's scenes")
    a = ap.parse_args()
    cr, cn = cfg(a.ref), cfg(a.new)
    ok = True
    for k in SAME:
        if cr.get(k, "") != cn.get(k, ""):
            print(f"FAIL config {k}: ref={cr.get(k)!r} new={cn.get(k)!r}"); ok = False
    print(f"ref: {cr.get('tier')} {cr.get('gpu')} commit {cr.get('git_commit')} | "
          f"new: {cn.get('tier')} {cn.get('gpu')} commit {cn.get('git_commit')} cap={cn.get('CAP_MEM_CAP_GB')}")
    scenes = a.scenes or cn.get("scenes", "").split()
    for s in scenes:
        r, c = scene_vals(a.ref, a.n, s), scene_vals(a.new, a.n, s)
        if r is None or c is None:
            print(f"FAIL {s}: missing result ({'ref' if r is None else 'new'})"); ok = False; continue
        if r["num_frames"] != c["num_frames"]:
            print(f"FAIL {s}: frames ref={r['num_frames']} new={c['num_frames']}"); ok = False
        for k, (atol, rtol) in KEYS.items():
            if k not in r or k not in c:
                continue
            diff = abs(c[k] - r[k])
            good = diff <= atol or diff <= rtol * abs(r[k])
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {s} {k:12s} ref={r[k]:.5f} new={c[k]:.5f}")
        rm, cm = r["peak_allocated_gb"], c["peak_allocated_gb"]
        good = abs(cm - rm) <= MEM_REL * rm
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {s} peak_gb      ref={rm:.3f} new={cm:.3f}")
    print("CROSSCHECK PASS" if ok else "CROSSCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
