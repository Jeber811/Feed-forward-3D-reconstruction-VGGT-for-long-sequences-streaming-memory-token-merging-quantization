"""Score every infer_poses.py output with FastVGGT's own trajectory metrics.

For each <results_dir>/<model>/<scene>/N<n>.npz: GT poses of the stored frame ids (relative to the
first valid frame, as FastVGGT's load_poses) are compared with the predicted first-frame-relative
world-to-camera poses via FastVGGT's eval_trajectory(..., align=True): Sim(3)-aligned ATE on
translations, plus ARE and RPE. This is the same call evaluate_scene_and_save makes for VGGT.

Writes <results_dir>/ate_results.csv (per scene) and ate_summary.csv (mean over scenes).
Run in the fastvggt env (needs evo):  python tools/score_ate.py <results_dir> --scenes S1 S2
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(os.path.dirname(TOOLS), "FastVGGT"))
from frames import Scene  # noqa: E402
try:
    import open3d  # noqa: F401  (eval_utils imports it for point-cloud metrics only)
except ImportError:  # e.g. no libEGL on CPU nodes; trajectory metrics don't need it
    import types
    sys.modules["open3d"] = types.ModuleType("open3d")
from vggt.utils.eval_utils import eval_trajectory  # noqa: E402
import evo.main_ape as main_ape  # noqa: E402
import evo.main_rpe as main_rpe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _align_wins(f):
    """FastVGGT passes align=True together with align_origin=True. evo < 1.3x applied Sim(3)
    alignment and ignored align_origin; evo 1.37 raises instead. Keep the old behaviour."""
    def wrapped(*a, align=False, align_origin=False, **k):
        return f(*a, align=align, align_origin=align_origin and not align, **k)
    return wrapped


main_ape.ape, main_rpe.rpe = _align_wins(main_ape.ape), _align_wins(main_rpe.rpe)

METRICS = ["ate", "are", "rpe_rot", "rpe_trans"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--scenes", nargs="+", required=True)
    args = p.parse_args()
    root = Path(args.results_dir)
    scenes = {Path(s).name: Scene(os.path.abspath(s)) for s in args.scenes}

    rows = []
    for npz in sorted(p for p in root.glob("*/*/N*.npz") if p.stem[1:].isdigit()):  # skip N*_pts.npz
        model, scene_name, n = npz.parent.parent.name, npz.parent.name, int(npz.stem[1:])
        d = np.load(npz)
        scene = scenes[scene_name]
        pose_idx = {f: i for i, f in enumerate(scene.pose_ids)}
        gt_c2w = scene.c2ws[[pose_idx[int(f)] for f in d["frame_ids"]]]
        stats, _, _ = eval_trajectory(d["w2c"], np.linalg.inv(gt_c2w), list(d["frame_ids"]), align=True)
        plt.close("all")  # eval_trajectory draws a figure per call
        row = dict(model=model, scene=scene_name, input_frame=n, num_frames=int(d["num_frames"]),
                   time_s=round(float(d["time_s"]), 2),
                   peak_allocated_gb=round(float(d["peak_allocated_gb"]), 3),
                   peak_reserved_gb=round(float(d["peak_reserved_gb"]), 3))
        row.update({k: round(float(stats[k]), 5) for k in METRICS})
        rows.append(row)
        print(f"{model:<14} {scene_name} N={n:<5} ATE {row['ate']:.4f}  ARE {row['are']:.3f}", flush=True)

    with open(root / "ate_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Mean over scenes, only where every scene has a result (matches "average of 2 scenes")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["input_frame"])].append(r)
    summary = []
    for (model, n), rs in sorted(groups.items()):
        if len(rs) < len(scenes):
            continue
        s = dict(model=model, input_frame=n, scenes=len(rs))
        for k in METRICS + ["time_s", "peak_allocated_gb"]:
            s[k] = round(float(np.mean([r[k] for r in rs])), 5)
        summary.append(s)
    with open(root / "ate_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {root / 'ate_results.csv'} and {root / 'ate_summary.csv'}")


if __name__ == "__main__":
    main()
