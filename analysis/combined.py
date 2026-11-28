"""Combined B1 / B2 / baselines numbers from per-scene protocol JSON (read-only), matched scene sets.
Run on a compute node:  ~/.venvs/slides/bin/python combined.py  ->  combined.json + printed tables."""
import glob
import json
import os
from pathlib import Path

H = os.path.expanduser
ZR = "/home/zhyw86/WorkSpace/cap6411/protocol_clone/results/rtx6000"
JR = [H("~/cap6411_res/results"), H("~/cap6411_mres/results")]
B3 = H("~/cap6411_b3/results/rtx6000")  # InfiniteVGGT (B3) + pi3 10 GB on all 50 scenes, branch b3/infinitevggt
JB = lambda name: [f"{B3}/{name}"]
SS = Path(H("~/cap6411_repo/protocol/scene_sets"))
GRID = SS.joinpath("scannet50_v03_grid10_N500.txt").read_text().split()
COH = {n: SS.joinpath(f"scannet50_v03_N{n}.txt").read_text().split() for n in (300, 500, 1000, 2000)}


def load(patterns, n):
    """scene -> (memory, metrics, dir) for ok scenes; later mtime wins. Records disagreements."""
    out = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            for sd in glob.glob(f"{d}/input_frame_{n}/scene*"):
                mp, tp = Path(sd, "memory.json"), Path(sd, "metrics.json")
                if not (mp.exists() and tp.exists()):
                    continue
                mem = json.loads(mp.read_text())
                if mem.get("status") != "ok":
                    continue
                sc = Path(sd).name
                met = json.loads(tp.read_text())
                if sc in out and abs(out[sc][1]["ate_cc"] - met["ate_cc"]) > 1e-4:
                    print(f"  ! {sc} differs: {out[sc][2]} {out[sc][1]['ate_cc']:.4f} vs {d} {met['ate_cc']:.4f}")
                if sc not in out or tp.stat().st_mtime > Path(out[sc][2], f"input_frame_{n}", sc, "metrics.json").stat().st_mtime:
                    out[sc] = (mem, met, d)
    return out


def mean(rows, scenes):
    rs = [rows[s] for s in scenes if s in rows]
    if not rs:
        return None
    a = lambda k: sum(r[1][k] for r in rs) / len(rs)
    return {"n": len(rs), "ate_cm": 100 * a("ate_cc"), "acc_cm": 100 * a("rc_acc_med"), "f5": a("rc_f5"),
            "peak": max(r[0]["peak_allocated_gb"] for r in rs),
            "spf": sum(r[0]["inference_time_ms"] / 1000 / r[0]["num_frames"] for r in rs) / len(rs),
            "commits": sorted({json.loads(open(glob.glob(f"{r[2]}/run_config_*.json")[0]).read()).get("git_commit", "?")[:9]
                               for r in rs if glob.glob(f"{r[2]}/run_config_*.json")})}


def J(tier, name):
    return [f"{r}/{tier}/{name}" for r in JR]


Z = lambda *names: [f"{ZR}/{x}" for x in names]
B2 = {"chunk": Z("zhy_v03_chunk_N500_88G"), "M": Z("zhy_v03_chunkM_N500")}
for v in ("K", "Q", "KQ", "KM", "MQ", "KMQ"):
    B2[v] = Z(f"zhy_v03_grid_chunk{v}_N500*")
B2["Q"] = Z("zhy_v03_grid_chunkQ_N500*")

R = {"grid": {}, "long": {}, "timing": {}}
print("== N=500, 10 grid scenes ==")
rows = {}
for v, pats in B2.items():
    rows[f"B2 {v}"] = load(pats, 500)
for b in (6, 10, 16):
    for v in ("KQ", "KMQ", "none", "K", "Q", "M", "KM", "MQ"):
        if b == 6 and v in ("none", "K", "M", "KM"):
            continue
        rows[f"B1 {b}GB {v}"] = load(J("v100_cap24", f"jake_v04_grid_b1_b{b}_{v}"), 500)
rows["TTT3R"] = load(J("rtx6000", "jake_v04_ctrl_ttt3r_offload_N500"), 500)
rows["CUT3R"] = load(J("rtx6000", "jake_v04_ctrl_cut3r_offload_N500"), 500)
for k in (15, 110, 260):
    rows[f"kfinterp{k}"] = load(J("v100_cap24", f"jake_v04_ctrl_kfinterp{k}_N500"), 500)
rows["InfV 1.2M"] = load(JB("jake_v04_b3_infv_N500"), 500)  # full-cohort run, restricted to GRID by mean()
for b, tag in (("1.2M", ""), ("600k", "bud600k"), ("300k", "bud300k")):
    if tag:
        rows[f"InfV {b}"] = load(JB(f"jake_v04_b3_grid_infv_{tag}"), 500)
    rows[f"InfV {b} Q"] = load(JB(f"jake_v04_b3_grid_infv_{tag + '_' if tag else ''}q"), 500)
for k, r in rows.items():
    m = mean(r, GRID)
    R["grid"][k] = m
    if m:
        print(f"{k:16s} n={m['n']:2d} peak {m['peak']:6.2f}  ATE {m['ate_cm']:6.2f}  Acc {m['acc_cm']:5.2f}  F5 {m['f5']:.3f}  "
              f"spf {m['spf']:.3f} {m['commits']}")

print("== full cohorts ==")
for n in (300, 500, 1000, 2000):
    rows = {"B2 chunk": load(Z(f"zhy_v03_chunk_N{n}", f"zhy_v03_chunk_N{n}_b*") + (Z("zhy_v03_chunk_N500_88G") if n == 500 else []), n),
            "B2 chunk+M": load(Z(f"zhy_v03_chunkM_N{n}", f"zhy_v03_chunkM_N{n}_b*", f"zhy_v03_chunkMfix_N{n}_b*"), n),
            "TTT3R": load(J("rtx6000", f"jake_v04_ctrl_ttt3r_offload_N{n}"), n),
            "CUT3R": load(J("rtx6000", f"jake_v04_ctrl_cut3r_offload_N{n}"), n),
            "InfiniteVGGT": load(JB(f"jake_v04_b3_infv_N{n}"), n)}
    for b in (6, 10, 16):
        rows[f"B1 {b}GB KQ"] = load(J("v100_cap24", f"jake_v04_b1_b{b}_KQ_N{n}"), n)
    for k in (15, 110, 260):
        rows[f"kfinterp{k}"] = load(J("v100_cap24", f"jake_v04_ctrl_kfinterp{k}_N{n}"), n)
    if n == 300:  # unchunked models only fit 300 frames
        for k, d in (("full pi3", "pi3"), ("full VGGT*", "vggt_star"), ("full FastVGGT", "fastvggt")):
            rows[k] = load(J("v100_cap24", f"jake_v04_ctrl_{d}_N300"), n)
    have = [k for k, r in rows.items() if r]
    matched = [s for s in COH[n] if all(s in rows[k] for k in have)]
    print(f"-- N={n}: cohort {len(COH[n])}, matched across {have}: {len(matched)}")
    R["long"][n] = {}
    for k in have:
        full, mt = mean(rows[k], COH[n]), mean(rows[k], matched)
        R["long"][n][k] = {"cohort": full, "matched": mt}
        print(f"{k:12s} cohort n={full['n']:2d} ATE {full['ate_cm']:6.2f} F5 {full['f5']:.3f} peak {full['peak']:5.2f} | "
              f"matched n={mt['n']:2d} ATE {mt['ate_cm']:6.2f} F5 {mt['f5']:.3f}  {full['commits']}")

print("== pi3 + chunking, 10 GB, all 50 scenes at N=500 (paired stats: b3/agg2.json) ==")
R["g50"] = {}
for v in ("none", "K", "KQ", "KMQ"):
    m = mean(load(JB(f"jake_v04_g50_b1_b10_{v}"), 500), COH[500])
    R["g50"][v] = m
    print(f"{v:5s} n={m['n']:2d} peak {m['peak']:5.2f}  ATE {m['ate_cm']:5.2f}  Acc {m['acc_cm']:5.2f}  F5 {m['f5']:.3f}  {m['commits']}")

print("== timing runs (RTX 6000) ==")
TD = JR + [H("~/cap6411_repo/results"), H("~/cap6411_b3/results")]  # repo: K+M+Q timing (Nov 18); b3: InfiniteVGGT
for d in sorted(glob.glob(f"{ZR}/zhy_*timing") + [x for r in TD for x in glob.glob(f"{r}/rtx6000/*_timing")]):
    for sd in glob.glob(f"{d}/input_frame_*/scene*"):
        try:
            mem = json.loads(Path(sd, "memory.json").read_text())
        except Exception:
            continue
        if mem.get("status") == "ok":
            spf = mem["inference_time_ms"] / 1000 / mem["num_frames"]
            R["timing"][Path(d).name] = {"scene": Path(sd).name, "spf": spf, "peak": mem["peak_allocated_gb"],
                                          "gpu": mem.get("gpu")}
            print(f"{Path(d).name:45s} {Path(sd).name} {spf:.3f} s/f  peak {mem['peak_allocated_gb']:.2f}  {mem.get('gpu')}")
Path("combined.json").write_text(json.dumps(R, indent=1))
