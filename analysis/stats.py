"""Free statistics on existing results (no GPU): paired effects of K / M / Q on the 10 grid scenes with bootstrap 95% CIs
and a sign test, B1 vs B2, and absolute (unsigned) drift where trajectories exist on CRCV.
Run on a compute node:  srun -p normal -c 2 --mem=8G -t 00:20:00 ~/.venvs/slides/bin/python -I stats.py > stats.out
Writes stats.json next to this file."""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

H = os.path.expanduser
ZR = "/home/zhyw86/WorkSpace/cap6411/protocol_clone/results/rtx6000"
JR = [H("~/cap6411_res/results"), H("~/cap6411_mres/results")]
GRID = Path(H("~/cap6411_repo/protocol/scene_sets/scannet50_v03_grid10_N500.txt")).read_text().split()
sys.path.insert(0, H("~/cap6411_repo/tools"))
import drift_analysis  # noqa: E402  (reads GT poses from the shared ScanNet frames)

RNG = np.random.default_rng(0)
sys.stdout.reconfigure(encoding="utf-8")


def load(patterns, n=500):
    out = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            for sd in glob.glob(f"{d}/input_frame_{n}/scene*"):
                mp, tp = Path(sd, "memory.json"), Path(sd, "metrics.json")
                if not (mp.exists() and tp.exists()):
                    continue
                if json.loads(mp.read_text()).get("status") != "ok":
                    continue
                met = json.loads(tp.read_text())
                sc = Path(sd).name
                prev = out.get(sc)
                if prev is None or tp.stat().st_mtime > prev["mtime"]:
                    out[sc] = {"ate": 100 * met["ate_cc"], "acc": 100 * met["rc_acc_med"], "f5": met["rc_f5"],
                               "mtime": tp.stat().st_mtime, "dir": sd}
    return out


def J(name):
    return [f"{r}/v100_cap24/{name}" for r in JR]


def Z(name):
    return [f"{ZR}/{name}"]


runs = {}
for b in (6, 10, 16):
    for v in ("none", "K", "Q", "KQ", "M", "KM", "MQ", "KMQ"):
        r = load(J(f"jake_v04_grid_b1_b{b}_{v}"))
        if r:
            runs[f"pi3 {b}GB {v}"] = r
runs["vggt* none"] = load(Z("zhy_v03_chunk_N500_88G"))
for v in ("K", "Q", "KQ", "KM", "MQ", "KMQ"):
    runs[f"vggt* {v}"] = load(Z(f"zhy_v03_grid_chunk{v}_N500*"))
runs["vggt* M"] = load(Z("zhy_v03_chunkM_N500"))


def sign_p(wins, n):
    """two-sided exact sign test"""
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, metric):
    """effect of going from config a to config b (b - a), over scenes both have."""
    sc = [s for s in GRID if s in runs[a] and s in runs[b]]
    d = np.array([runs[b][s][metric] - runs[a][s][metric] for s in sc])
    boots = d[RNG.integers(0, len(d), (20000, len(d)))].mean(1)
    better = (d < 0) if metric != "f5" else (d > 0)
    nz = int((d != 0).sum())
    return {"n": len(sc), "mean": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "better": int(better.sum()), "p_sign": sign_p(int(better.sum()), nz) if nz else 1.0}


COMPARE = []
for b, base in ((10, "none"), (16, "none")):
    for v in ("K", "Q", "M", "KQ", "KMQ", "MQ", "KM"):
        COMPARE.append((f"pi3 {b}GB {base}", f"pi3 {b}GB {v}", f"π³ {b} GB: +{v}"))
for b in (10, 16):
    COMPARE.append((f"pi3 {b}GB KQ", f"pi3 {b}GB KMQ", f"π³ {b} GB: K+Q → K+M+Q"))
COMPARE.append(("pi3 6GB Q", "pi3 6GB KQ", "π³ 6 GB: Q → K+Q"))
COMPARE.append(("pi3 6GB KQ", "pi3 6GB KMQ", "π³ 6 GB: K+Q → K+M+Q"))
for v in ("K", "Q", "M", "KQ", "MQ", "KM", "KMQ"):
    COMPARE.append(("vggt* none", f"vggt* {v}", f"VGGT*: +{v}"))
COMPARE.append(("vggt* KQ", "pi3 6GB KQ", "VGGT* K+Q → π³ K+Q 6 GB"))
COMPARE.append(("vggt* none", "pi3 6GB KQ", "VGGT* chunk → π³ K+Q 6 GB"))

R = {"paired": [], "drift": {}}
print("Paired effects on the 10 grid scenes (b - a). Lower ATE/Acc and higher F5 are better; 'better' = scenes improved.")
print(f"{'comparison':32s} {'metric':4s} {'mean':>7s} {'95% CI':>18s} {'better':>7s} {'p(sign)':>8s}")
for a, b, lab in COMPARE:
    if a not in runs or b not in runs or not runs[a] or not runs[b]:
        print(f"{lab:32s} missing")
        continue
    for m in ("ate", "acc", "f5"):
        r = paired(a, b, m)
        sig = "*" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
        R["paired"].append({"label": lab, "a": a, "b": b, "metric": m, **r})
        fmt = ".3f" if m == "f5" else ".2f"
        print(f"{lab:32s} {m:4s} {r['mean']:+7{fmt}} [{r['ci'][0]:+7{fmt}}, {r['ci'][1]:+7{fmt}}]{sig:1s} "
              f"{r['better']:3d}/{r['n']:<3d} {r['p_sign']:8.3f}")

print("\nDrift on the 10 grid scenes from saved trajectories (cm): signed mean, |drift| mean, max |drift|")
for k, r in runs.items():
    vals = []
    for s in GRID:
        if s in r:
            t = Path(r[s]["dir"], "trajectory.npz")
            if t.exists():
                vals.append(100 * drift_analysis.analyse(t)["drift_last_minus_first"])
    if vals:
        v = np.array(vals)
        R["drift"][k] = {"n": len(v), "signed": float(v.mean()), "abs": float(np.abs(v).mean()), "max_abs": float(np.abs(v).max()),
                         "per_scene": vals}
        print(f"{k:16s} n={len(v):2d} signed {v.mean():+6.2f}  |drift| {np.abs(v).mean():5.2f}  max {np.abs(v).max():5.2f}")
Path(__file__).with_name("stats.json").write_text(json.dumps(R, indent=1))
print("wrote stats.json")
