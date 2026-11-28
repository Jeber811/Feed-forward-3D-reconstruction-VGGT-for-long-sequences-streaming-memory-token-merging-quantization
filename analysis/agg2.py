import glob, json, math, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = Path.home() / "cap6411_b3/results/rtx6000"; out = {"drift": {}, "paired50": []}
print("|drift| (cm) from drift_*.json: n, mean |drift|, median, max")
for d in sorted(R.glob("jake_v04_*")):
    v = [100 * json.loads(f.read_text())["drift_last_minus_first"] for f in d.glob("drift_*.json")]
    if v:
        a = np.abs(v); out["drift"][d.name] = {"n": len(v), "abs_mean": a.mean(), "abs_med": float(np.median(a)), "max": a.max()}
        print(f"  {d.name:34s} n={len(v):2d}  {a.mean():6.2f}  {np.median(a):6.2f}  {a.max():6.2f}")
def L(name):
    o = {}
    for sd in glob.glob(f"{R}/{name}/input_frame_500/scene*"):
        m = json.loads(Path(sd, "metrics.json").read_text()); o[Path(sd).name] = (100*m["ate_cc"], 100*m["rc_acc_med"], m["rc_f5"])
    return o
runs = {v: L(f"jake_v04_g50_b1_b10_{v}") for v in ("none", "K", "KQ", "KMQ")}
rng = np.random.default_rng(0)
def sp(w, n): k = min(w, n-w); return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n)
print("\npi3 10 GB, 50 scenes, paired (b - a): mean [95% CI] better/n p")
for a, b in (("none","K"),("none","KQ"),("K","KQ"),("KQ","KMQ"),("none","KMQ")):
    sc = sorted(set(runs[a]) & set(runs[b]))
    for j, m in enumerate(("ATE-cc", "Acc", "F5")):
        d = np.array([runs[b][s][j] - runs[a][s][j] for s in sc]); bt = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
        lo, hi = np.percentile(bt, [2.5, 97.5]); better = int(((d > 0) if m == "F5" else (d < 0)).sum()); nz = int((d != 0).sum())
        sig = "*" if lo > 0 or hi < 0 else " "; f = ".3f" if m == "F5" else ".2f"
        out["paired50"].append({"a": a, "b": b, "metric": m, "mean": d.mean(), "ci": [lo, hi], "better": better, "n": len(sc), "p": sp(better, nz)})
        print(f"  {a:>4s} -> {b:4s} {m:6s} {d.mean():+{f}} [{lo:+{f}}, {hi:+{f}}]{sig} {better:2d}/{len(sc)}  p={sp(better, nz):.3g}")
Path(__file__).with_name("agg2.json").write_text(json.dumps(out, indent=1, default=float))
