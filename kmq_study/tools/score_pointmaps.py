"""Score saved 1/4-resolution ScanNet point maps at matching depth pixels.

This is a sampled, pixel-corresponded 3D point-map error after ONE Sim(3) fit per
scene and run. It is not the paper's mesh accuracy/completeness protocol.
Use the same frame and pixel sample for every variant in a scene. No GPU needed.
"""
import argparse
import csv
import struct
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = ["chunk", "chunk+K", "chunk+M", "chunk+K+M+Q", "stream", "stream+K", "stream+K+M"]


def sim3(src, dst):
    """One similarity mapping row-vector src points to dst points."""
    x, y = src.astype(np.float64), dst.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ flip @ vt
    scale = np.sum(singular * np.diag(flip)) / np.mean(np.sum(xc * xc, axis=1))
    trans = my - scale * (rot @ mx)
    return scale, rot, trans


def align(src, dst):
    keep = np.ones(len(src), dtype=bool)
    for _ in range(3):
        scale, rot, trans = sim3(src[keep], dst[keep])
        error = np.linalg.norm(scale * (src @ rot.T) + trans - dst, axis=1)
        keep = error <= np.quantile(error, 0.90)
    scale, rot, trans = sim3(src[keep], dst[keep])
    return np.linalg.norm(scale * (src @ rot.T) + trans - dst, axis=1), scale


def depth_shift(sens):
    with open(sens, "rb") as f:
        if struct.unpack("I", f.read(4))[0] != 4:
            raise ValueError(f"unexpected sens version: {sens}")
        name_len = struct.unpack("Q", f.read(8))[0]
        f.seek(name_len + 4 * 64 + 8 + 16, 1)
        return struct.unpack("f", f.read(4))[0]


def scene_reference(scene, frame_ids, shape, samples_per_frame, frames_to_score, raw_root):
    first_color = scene / "color" / f"{frame_ids[0]}.jpg"
    with Image.open(first_color) as im:
        color_w, color_h = im.size
    h, w = shape
    input_w = 518
    resized_h = round(color_h * (input_w / color_w) / 14) * 14
    input_h = min(resized_h, 518)
    if (h, w) != ((input_h + 3) // 4, (input_w + 3) // 4):
        raise ValueError(f"unexpected point-map shape {(h,w)} for {scene.name}")
    crop_top = (resized_h - input_h) // 2
    kc = np.loadtxt(scene / "intrinsic" / "intrinsic_color.txt")
    kd = np.loadtxt(scene / "intrinsic" / "intrinsic_depth.txt")
    for name in ("extrinsic_color.txt", "extrinsic_depth.txt"):
        e = np.loadtxt(scene / "intrinsic" / name)
        if not np.allclose(e, np.eye(4), atol=1e-5):
            raise ValueError(f"nonidentity {name} needs explicit registration: {scene.name}")
    shift = depth_shift(raw_root / scene.name / (scene.name + ".sens"))
    first_pose = np.loadtxt(scene / "pose" / f"{frame_ids[0]}.txt")
    first_inv = np.linalg.inv(first_pose)
    positions = np.linspace(0, len(frame_ids) - 1, min(frames_to_score, len(frame_ids)), dtype=int)
    references = []
    for pos in positions:
        fid = int(frame_ids[pos])
        rng = np.random.default_rng(fid + 1009)
        flat = rng.choice(h * w, min(samples_per_frame, h * w), replace=False)
        yy, xx = np.divmod(flat, w)
        uc = ((xx * 4 + 0.5) * color_w / input_w) - 0.5
        vc = ((yy * 4 + crop_top + 0.5) * color_h / resized_h) - 0.5
        ray_x = (uc - kc[0, 2]) / kc[0, 0]
        ray_y = (vc - kc[1, 2]) / kc[1, 1]
        ud = np.rint(kd[0, 0] * ray_x + kd[0, 2]).astype(int)
        vd = np.rint(kd[1, 1] * ray_y + kd[1, 2]).astype(int)
        with Image.open(scene / "depth" / f"{fid}.png") as im:
            depth = np.asarray(im)
        in_bounds = (ud >= 0) & (ud < depth.shape[1]) & (vd >= 0) & (vd < depth.shape[0])
        meters = np.zeros(len(flat), dtype=np.float64)
        meters[in_bounds] = depth[vd[in_bounds], ud[in_bounds]] / shift
        valid = in_bounds & (meters > 0.1) & (meters < 10.0)
        camera = np.stack((ray_x * meters, ray_y * meters, meters), axis=1)
        rel = first_inv @ np.loadtxt(scene / "pose" / f"{fid}.txt")
        world = camera @ rel[:3, :3].T + rel[:3, 3]
        references.append((pos, flat, valid, world))
    return references


def score_one(points_path, references):
    with np.load(points_path) as archive:
        points = archive["pts"]
    predicted, actual = [], []
    eligible = 0
    for pos, flat, valid_gt, gt in references:
        p = points[pos].reshape(-1, 3)[flat].astype(np.float64)
        good = valid_gt & np.isfinite(p).all(axis=1)
        eligible += int(valid_gt.sum())
        predicted.append(p[good])
        actual.append(gt[good])
    src, dst = np.concatenate(predicted), np.concatenate(actual)
    if len(src) < 500:
        raise ValueError(f"too few matched valid depth points: {len(src)}")
    error, scale = align(src, dst)
    return dict(valid_points=len(src), valid_fraction=round(len(src) / eligible, 5),
                mean_3d_m=round(float(np.mean(error)), 5),
                median_3d_m=round(float(np.median(error)), 5),
                p90_3d_m=round(float(np.quantile(error, 0.9)), 5),
                within_10cm=round(float(np.mean(error < 0.1)), 5),
                sim3_scale=round(float(scale), 5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--scenes", nargs="+", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("/squash/scannetv2/scans"))
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--sample-frames", type=int, default=64)
    p.add_argument("--pixels-per-frame", type=int, default=512)
    args = p.parse_args()
    output_dir = args.output_dir or args.results
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in args.scenes:
        nfile = f"N{args.n}.npz"
        ptsfile = f"N{args.n}_pts.npz"
        with np.load(args.results / args.models[0] / scene.name / nfile) as pose_data:
            ids = pose_data["frame_ids"].copy()
        with np.load(args.results / args.models[0] / scene.name / ptsfile) as pts_data:
            shape = pts_data["pts"].shape[1:3]
        reference = scene_reference(scene, ids, shape, args.pixels_per_frame,
                                    args.sample_frames, args.raw_root)
        for model in args.models:
            root = args.results / model / scene.name
            with np.load(root / nfile) as pose_data:
                if not np.array_equal(ids, pose_data["frame_ids"]):
                    raise ValueError(f"different frame IDs for {model} in {scene.name}")
            result = score_one(root / ptsfile, reference)
            row = dict(model=model, scene=scene.name, input_frame=args.n,
                       sampled_frames=len(reference), **result)
            rows.append(row)
            print(row, flush=True)
            with open(output_dir / f"pointmap_N{args.n}_results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = []
    for model in args.models:
        group = [r for r in rows if r["model"] == model]
        summary.append(dict(model=model, input_frame=args.n, scenes=len(group),
                            **{k:round(float(np.mean([r[k] for r in group])),5)
                               for k in ("mean_3d_m", "median_3d_m", "p90_3d_m", "within_10cm", "valid_fraction")}))
    with open(output_dir / f"pointmap_N{args.n}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("POINTMAP SCORING COMPLETE", flush=True)


if __name__ == "__main__":
    main()

