"""Per-frame camera-center error after one global Sim(3) on 1,000-frame prefixes."""
import argparse
import csv
from pathlib import Path
import numpy as np


def sim3(src, dst):
    x, y = src.astype(np.float64), dst.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ flip @ vt
    scale = np.sum(singular * np.diag(flip)) / np.mean(np.sum(xc * xc, axis=1))
    return scale, rot, my - scale * rot @ mx


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["chunk", "chunk+M"])
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, scene_summaries = [], []
    for scene_name in args.scenes:
        scene = args.data / scene_name
        reference_ids, gt = None, None
        for model in args.models:
            with np.load(args.results / model / scene_name / f"N{args.n}.npz") as d:
                ids = d["frame_ids"].astype(int)
                pred = np.linalg.inv(d["w2c"].astype(np.float64))[:, :3, 3]
                sizes = np.asarray(d["chunk_frames"], dtype=int)
                anchors = np.asarray(d["anchor_frames"], dtype=int)
            if len(ids) != args.n:
                raise ValueError(f"unexpected frame count: {model} {scene_name} {len(ids)}")
            if reference_ids is None:
                reference_ids = ids
                gt_poses = np.stack([np.loadtxt(scene / "pose" / f"{fid}.txt") for fid in ids])
                gt = (np.linalg.inv(gt_poses[0]) @ gt_poses)[:, :3, 3]
            elif not np.array_equal(ids, reference_ids):
                raise ValueError(f"different frame IDs: {model} {scene_name}")
            scale, rot, trans = sim3(pred, gt)
            error = np.linalg.norm(scale * (pred @ rot.T) + trans - gt, axis=1)
            starts = [0]
            seen = 0
            for i, (size, anchor) in enumerate(zip(sizes, anchors)):
                seen += int(size - anchor)
                if i < len(sizes) - 1:
                    starts.append(seen)
            if seen != len(ids):
                raise ValueError(f"chunk accounting failed: {model} {scene_name} {seen}")
            boundary_set = set(starts)
            for index, (fid, value) in enumerate(zip(ids, error)):
                rows.append(dict(model=model, scene=scene_name, index=index, frame_id=int(fid),
                                 error_m=round(float(value), 6), chunk_start=int(index in boundary_set)))
            jumps = [float(error[s:s+5].mean() - error[s-5:s].mean())
                     for s in starts[1:] if s >= 5 and s + 5 <= len(error)]
            summary = dict(model=model, scene=scene_name, n=len(ids),
                           early_quarter_mean_m=round(float(error[:250].mean()), 5),
                           late_quarter_mean_m=round(float(error[-250:].mean()), 5),
                           whole_mean_m=round(float(error.mean()), 5),
                           final_frame_error_m=round(float(error[-1]), 5),
                           mean_boundary_jump_m=round(float(np.mean(jumps)), 5),
                           scale=round(float(scale), 5))
            scene_summaries.append(summary)
            print(summary, flush=True)
    write_csv(args.output / "per_frame.csv", rows)
    write_csv(args.output / "per_scene.csv", scene_summaries)
    summaries = []
    for model in args.models:
        group = [r for r in scene_summaries if r["model"] == model]
        summaries.append(dict(model=model, scenes=len(group), **{
            key:round(float(np.mean([r[key] for r in group])), 5)
            for key in ("early_quarter_mean_m", "late_quarter_mean_m", "whole_mean_m",
                        "final_frame_error_m", "mean_boundary_jump_m")}))
    write_csv(args.output / "summary.csv", summaries)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    for model in args.models:
        curves = [np.array([r["error_m"] for r in rows if r["model"] == model and r["scene"] == scene])
                  for scene in args.scenes]
        mean = np.mean(curves, axis=0)
        smooth = np.convolve(np.pad(mean, (10, 10), mode="edge"), np.ones(21) / 21, mode="valid")
        plt.plot(np.arange(len(mean)), smooth, label=model)
    for index in [100] + list(range(190, args.n, 90)):
        plt.axvline(index, color="gray", alpha=0.10, linewidth=0.7)
    plt.xlabel("Chronological frame index")
    plt.ylabel("Mean camera-center error after one sequence Sim(3) (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "drift_curve.png", dpi=180)
    print("DRIFT SCORING COMPLETE", flush=True)


if __name__ == "__main__":
    main()

