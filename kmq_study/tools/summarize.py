"""Collects metrics.json + memory.json + nvidia-smi peaks from a run_baselines.sh log dir into summary.csv.

Usage: python summarize.py <logs/run_xxx>
"""
import csv
import json
import sys
from pathlib import Path

METRICS = ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans"]
MEMORY = ["num_frames", "model_weights_gb", "peak_allocated_gb", "peak_reserved_gb", "inference_time_ms"]


def smi_peak_mib(gpu_csv):
    peak = 0
    if gpu_csv.exists():
        for line in gpu_csv.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                peak = max(peak, int(parts[1]))
    return peak


def main(logdir):
    logdir = Path(logdir)
    rows = []
    for model_dir in sorted((logdir / "results").glob("*")):
        for frame_dir in sorted(model_dir.glob("input_frame_*")):
            n = frame_dir.name.split("_")[-1]
            smi = smi_peak_mib(logdir / f"gpu_{model_dir.name}_N{n}.csv")
            for scene_dir in sorted(p for p in frame_dir.iterdir() if p.is_dir()):
                row = {"model": model_dir.name, "input_frame": n, "scene": scene_dir.name}
                mem_file, met_file = scene_dir / "memory.json", scene_dir / "metrics.json"
                mem = json.loads(mem_file.read_text()) if mem_file.exists() else {}
                met = json.loads(met_file.read_text()) if met_file.exists() else {}
                row.update({k: mem.get(k, "") for k in MEMORY})
                row["smi_peak_mib_run"] = smi
                row.update({k: round(float(met[k]), 4) if k in met else "" for k in METRICS})
                row["status"] = "ok" if met else ("no_metrics" if mem else "failed")
                rows.append(row)

    if not rows:
        print("No results found in", logdir)
        return
    out = logdir / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cols = ["model", "input_frame", "scene", "peak_allocated_gb", "smi_peak_mib_run",
            "inference_time_ms", "chamfer_distance", "ate", "are", "status"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    print("Saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
