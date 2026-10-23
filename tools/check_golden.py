"""Compare a golden run (protocol/golden.sbatch) against protocol/golden/expected.json.

Passes when pose/geometry metrics match within tolerance (GPU nondeterminism only) and peak memory matches within
3% on the same GPU model. Exit code 0 = PASS, 1 = FAIL. --record writes the expected file instead (maintainers only).
"""
import argparse
import json
import sys
from pathlib import Path

SCENE, N = "scene0013_02", 60
KEYS = {  # metric: (absolute tolerance, relative tolerance) - pass if within either
    "ate": (2e-3, 0.02), "ate_cc": (2e-3, 0.02), "rpe_trans": (2e-3, 0.05),
    "rc_acc_med": (2e-3, 0.05), "rc_comp_med": (2e-3, 0.05), "rc_f5": (0.01, 0.02),
}
MEM_REL = 0.03


def collect(run_dir: Path):
    out = {}
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        d = model_dir / f"input_frame_{N}" / SCENE
        met, mem = json.loads((d / "metrics.json").read_text()), json.loads((d / "memory.json").read_text())
        out[model_dir.name] = {**{k: met[k] for k in KEYS if k in met}, "peak_allocated_gb": mem["peak_allocated_gb"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("expected", type=Path)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    got = collect(a.run_dir)
    if a.record:
        a.expected.parent.mkdir(parents=True, exist_ok=True)
        a.expected.write_text(json.dumps({"gpu": a.gpu, "scene": SCENE, "N": N, "models": got}, indent=2))
        print(f"RECORDED {a.expected}")
        return 0
    exp = json.loads(a.expected.read_text())
    same_gpu = exp["gpu"] == a.gpu
    if not same_gpu:
        print(f"WARNING: GPU {a.gpu!r} != reference {exp['gpu']!r}: memory not compared")
    ok = True
    for model, ref in exp["models"].items():
        cur = got.get(model)
        if cur is None:
            print(f"FAIL {model}: missing"); ok = False; continue
        for k, (atol, rtol) in KEYS.items():
            if k not in ref:
                continue
            diff = abs(cur[k] - ref[k])
            good = diff <= atol or diff <= rtol * abs(ref[k])
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {model:10s} {k:12s} ref={ref[k]:.5f} got={cur[k]:.5f}")
        if same_gpu:
            r, c = ref["peak_allocated_gb"], cur["peak_allocated_gb"]
            good = abs(c - r) <= MEM_REL * r
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {model:10s} peak_gb      ref={r:.3f} got={c:.3f}")
    print("GOLDEN PASS" if ok else "GOLDEN FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
