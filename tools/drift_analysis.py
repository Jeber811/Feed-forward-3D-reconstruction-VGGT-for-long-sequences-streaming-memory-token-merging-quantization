"""Drift analysis from saved trajectories (protocol v0.2). Compute node only, protocol env:
    srun -p normal -c 2 --mem=8G -t 00:10:00 python tools/drift_analysis.py results/rtx6000/<name> [more runs ...]

For every <run>/input_frame_<N>/<scene>/trajectory.npz (predicted camera-to-world poses + ScanNet frame ids):
  - one global Sim(3) (Umeyama on camera centres) to the GT trajectory, then the camera-position error per frame;
  - error growth: mean error in each tenth of the sequence (first tenth vs last tenth = drift);
  - scale drift: Sim(3) scale fitted on each tenth separately, relative to the global scale.
Writes <run>/drift_<N>_<scene>.json and prints a summary. GT poses are read from the ScanNet frames on the cluster.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA", "/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet"))


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, D, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / ((X - mx) ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


def gt_centres(scene, frame_ids):
    out = []
    for f in frame_ids:
        p = np.loadtxt(DATA / scene / "pose" / f"{int(f)}.txt").reshape(4, 4)
        out.append(p[:3, 3])
    return np.asarray(out)


def analyse(traj: Path, bins=10):
    z = np.load(traj)
    X = z["c2w_est"][:, :3, 3].astype(np.float64)
    Y = gt_centres(traj.parent.name, z["frame_ids"])
    s, R, t = umeyama(X, Y)
    err = np.linalg.norm(s * X @ R.T + t - Y, axis=1)
    idx = np.array_split(np.arange(len(X)), bins)
    seg_err = [float(err[i].mean()) for i in idx]
    seg_scale = [float(umeyama(X[i], Y[i])[0] / s) if len(i) >= 3 else None for i in idx]
    return {"n": int(len(X)), "ate_rmse": float(np.sqrt((err ** 2).mean())), "err_per_tenth": seg_err,
            "drift_last_minus_first": seg_err[-1] - seg_err[0], "scale_rel_per_tenth": seg_scale,
            "scale_spread": float(np.nanmax(seg_scale) - np.nanmin(seg_scale))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    for run in a.runs:
        for traj in sorted(run.glob("input_frame_*/*/trajectory.npz")):
            n, scene = traj.parent.parent.name.split("_")[-1], traj.parent.name
            r = analyse(traj)
            (run / f"drift_{n}_{scene}.json").write_text(json.dumps(r, indent=2))
            print(f"{run.name:28s} N={n:>5s} {scene}  ATE-rmse={r['ate_rmse']:.3f}  "
                  f"drift(last-first tenth)={r['drift_last_minus_first']:+.3f} m  scale spread={r['scale_spread']:.3f}")


if __name__ == "__main__":
    main()
