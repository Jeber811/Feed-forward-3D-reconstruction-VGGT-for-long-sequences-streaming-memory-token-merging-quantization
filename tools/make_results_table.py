"""Build the combined results table from every protocol run in results/ (both people's). Compute node only:
    srun -p normal -c 2 --mem=4G -t 00:05:00 python tools/make_results_table.py [results] [--tier rtx6000] [--protocol 0.2]
Writes results/TABLE_<tier>.csv and .md for the current protocol version (TABLE_<tier>_v<ver> for another one): one row
per (run name, N, scene) + a mean row over scenes. Runs of different protocol versions are never mixed in one table.
"""
import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = (REPO / "protocol" / "protocol.env").read_text()
VERSION = re.search(r"^PROTOCOL_VERSION=(\S+)", ENV, re.M).group(1)
COMPATIBLE = set(re.search(r'^PROTOCOL_COMPATIBLE="([^"]+)"', ENV, re.M).group(1).split())  # v0.4: 0.3 + 0.4
COLS = ["ate", "ate_cc", "rpe_trans", "rc_acc_med", "rc_comp_med", "rc_f5", "rc_f10"]


def rows_for(tier_dir: Path, version: str):
    rows = []
    want = COMPATIBLE if version in COMPATIBLE else {version}
    for run in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
        cfgs = sorted(run.glob("run_config_*.json"))
        cfg = json.loads(cfgs[-1].read_text()) if cfgs else {}
        if cfg.get("protocol") not in want:
            continue
        user = cfg.get("user", "?")
        for nd in sorted(run.glob("input_frame_*"), key=lambda p: int(p.name.split("_")[-1])):
            n = int(nd.name.split("_")[-1])
            for sd in sorted(p for p in nd.iterdir() if p.is_dir()):
                mem_f, met_f = sd / "memory.json", sd / "metrics.json"
                if not mem_f.exists():
                    continue
                mem = json.loads(mem_f.read_text())
                met = json.loads(met_f.read_text()) if met_f.exists() else {}
                t = mem.get("inference_time_ms")
                rows.append({"tier": tier_dir.name, "run": run.name, "user": user, "N": n, "scene": sd.name, "status": mem.get("status"),
                             **{k: met.get(k) for k in COLS}, "peak_gb": mem.get("peak_allocated_gb"),
                             "s_per_frame": round(t / 1000 / mem["num_frames"], 4) if t and mem.get("num_frames") else None,
                             "gpu": mem.get("gpu")})
    return rows


def with_means(rows):
    out, groups = [], {}
    for r in rows:
        groups.setdefault((r["tier"], r["run"], r["N"]), []).append(r)
    for (tier, run, n), g in groups.items():
        out += g
        ok = [r for r in g if r["status"] == "ok"]
        if len(ok) > 1:
            m = {"tier": tier, "run": run, "user": g[0]["user"], "N": n, "scene": f"MEAN({len(ok)})", "status": "ok", "gpu": g[0]["gpu"]}
            for k in COLS + ["peak_gb", "s_per_frame"]:
                vals = [r[k] for r in ok if r[k] is not None]
                m[k] = round(sum(vals) / len(vals), 4) if vals else None
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=REPO / "results")
    ap.add_argument("--tier", default="rtx6000", help="tier, or comma-separated tiers for one combined table "
                    "(v0.4: rtx6000,v100_cap24 = all 24 GB results; s_per_frame is only comparable within a tier)")
    ap.add_argument("--protocol", default=VERSION, help="protocol version to tabulate (default: current)")
    a = ap.parse_args()
    tiers = a.tier.split(",")
    rows = with_means([r for t in tiers if (a.results / t).is_dir() for r in rows_for(a.results / t, a.protocol)])
    if not rows:
        print("no rows"); return
    keys = list(rows[0].keys())
    cur = a.protocol in COMPATIBLE
    stem = f"TABLE_{'+'.join(tiers)}" + ("" if cur else f"_v{a.protocol}")
    with open(a.results / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    (a.results / f"{stem}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {a.results}/{stem}.csv/.md ({len(rows)} rows, protocol v{a.protocol})")


if __name__ == "__main__":
    main()
