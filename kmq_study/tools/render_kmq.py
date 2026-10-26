"""Render matched ScanNet point-map reconstructions for the eight chunk K/M/Q variants.

Reads saved inference only. Each variant is aligned once to sampled ScanNet depth
with a robust Sim(3), as in tools/score_pointmaps.py. All panels use the same
frames, sampled pixels, colors, clipping bounds, and orthographic camera.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/home/zhyw86/WorkSpace/cap6411/baseline")
sys.path.insert(0, str(BASE / "tools"))
from score_pointmaps import scene_reference, sim3  # noqa: E402

METHODS = [
    ("chunk", "Base"), ("chunk+K", "K"), ("chunk+M", "M"),
    ("chunk+Q", "Q"), ("chunk+K+M", "K+M"),
    ("chunk+K+Q", "K+Q"), ("chunk+M+Q", "M+Q"),
    ("chunk+K+M+Q", "K+M+Q"),
]
BG = np.array([17, 25, 35], dtype=np.uint8)


def robust_fit(src, dst):
    keep = np.ones(len(src), dtype=bool)
    for _ in range(3):
        scale, rot, trans = sim3(src[keep], dst[keep])
        error = np.linalg.norm(scale * (src @ rot.T) + trans - dst, axis=1)
        keep = error <= np.quantile(error, 0.90)
    return sim3(src[keep], dst[keep])


def camera_basis(view):
    if view == "top":
        return np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
    sight = np.array([0.65, -0.75, 0.85], dtype=float)
    sight /= np.linalg.norm(sight)
    right = np.cross(np.array([0., 0., 1.]), sight)
    right /= np.linalg.norm(right)
    up = np.cross(sight, right)
    return right, up, sight


def project(points, center, basis):
    x = points - center
    return np.column_stack([x @ basis[0], x @ basis[1], x @ basis[2]])


def render(points, colors, center, basis, bounds, width=455, height=355):
    xyz = project(points, center, basis)
    lo, hi = bounds
    inside = np.isfinite(xyz).all(axis=1)
    inside &= (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0])
    inside &= (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
    inside &= (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
    clipped_fraction = 1.0 - inside.mean()
    xyz, colors = xyz[inside], colors[inside]
    x = np.clip(((xyz[:, 0] - lo[0]) / (hi[0] - lo[0]) * (width - 1)).astype(int), 0, width - 1)
    y = np.clip(((hi[1] - xyz[:, 1]) / (hi[1] - lo[1]) * (height - 1)).astype(int), 0, height - 1)
    pixel = y * width + x
    # Larger view-axis coordinate is closer to the orthographic camera.
    order = np.lexsort((-xyz[:, 2], pixel))
    p_sorted = pixel[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = p_sorted[1:] != p_sorted[:-1]
    image = np.broadcast_to(BG, (height * width, 3)).copy()
    image[p_sorted[first]] = colors[order[first]]
    return Image.fromarray(image.reshape(height, width, 3), "RGB"), clipped_fraction


def reference_and_colors(scene, frame_ids, point_shape, sample_frames, pixels_per_frame):
    refs = scene_reference(scene, frame_ids, point_shape, pixels_per_frame,
                           sample_frames, Path("/squash/scannetv2/scans"))
    h, w = point_shape
    all_colors, all_world, positions, flats, valids = [], [], [], [], []
    for pos, flat, valid, world in refs:
        fid = int(frame_ids[pos])
        with Image.open(scene / "color" / f"{fid}.jpg") as im:
            color = np.asarray(im.convert("RGB"))
        color = np.rint(255 * (color.astype(np.float32) / 255) ** 0.70).astype(np.uint8)
        ch, cw = color.shape[:2]
        resized_h = round(ch * (518 / cw) / 14) * 14
        crop_top = (resized_h - min(resized_h, 518)) // 2
        yy, xx = np.divmod(flat, w)
        uc = np.clip(np.rint((xx * 4 + 0.5) * cw / 518 - 0.5).astype(int), 0, cw - 1)
        vc = np.clip(np.rint((yy * 4 + crop_top + 0.5) * ch / resized_h - 0.5).astype(int), 0, ch - 1)
        all_colors.append(color[vc, uc])
        all_world.append(world)
        positions.append(pos)
        flats.append(flat)
        valids.append(valid)
    return (np.concatenate(all_world), np.concatenate(all_colors),
            positions, flats, np.concatenate(valids))


def load_predicted(path, positions, flats):
    with np.load(path) as archive:
        pts = archive["pts"]
    return np.concatenate([pts[pos].reshape(-1, 3)[flat] for pos, flat in zip(positions, flats)]).astype(np.float64)


def draw_scene(args, scene_name, view, metrics):
    scene = Path(args.data_root) / scene_name
    results = Path(args.results)
    with np.load(results / "chunk" / scene_name / "N2000.npz") as f:
        frame_ids = f["frame_ids"].copy()
    with np.load(results / "chunk" / scene_name / "N2000_pts.npz") as f:
        shape = f["pts"].shape[1:3]
    actual, colors, positions, flats, valid_gt = reference_and_colors(
        scene, frame_ids, shape, args.sample_frames, args.pixels_per_frame)
    reference = actual[valid_gt]
    reference_color = colors[valid_gt]
    center = np.median(reference, axis=0)
    basis = camera_basis(view)
    q = project(reference, center, basis)
    lo = np.quantile(q, 0.005, axis=0)
    hi = np.quantile(q, 0.995, axis=0)
    extent = hi - lo
    lo -= 0.12 * extent
    hi += 0.12 * extent
    # Keep the same metric scale horizontally and vertically in all panels.
    ratio = 455 / 355
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * ratio)
    midpoint = (lo + hi) / 2
    lo[0], hi[0] = midpoint[0] - span / 2, midpoint[0] + span / 2
    lo[1], hi[1] = midpoint[1] - span / (2 * ratio), midpoint[1] + span / (2 * ratio)
    bounds = lo, hi

    panels = [("ScanNet depth reference", reference, reference_color, None)]
    for method, label in METHODS:
        pose_path = results / method / scene_name / "N2000.npz"
        point_path = results / method / scene_name / "N2000_pts.npz"
        with np.load(pose_path) as f:
            if not np.array_equal(frame_ids, f["frame_ids"]):
                raise ValueError(f"Different frame IDs: {method} {scene_name}")
        predicted = load_predicted(point_path, positions, flats)
        good = valid_gt & np.isfinite(predicted).all(axis=1)
        scale, rot, trans = robust_fit(predicted[good], actual[good])
        mapped = scale * (predicted[good] @ rot.T) + trans
        error = np.linalg.norm(mapped - actual[good], axis=1)
        record = dict(scene=scene_name, method=method, view=view,
                      sampled_points=len(mapped), mean_3d_m=float(error.mean()),
                      median_3d_m=float(np.median(error)))
        metrics.append(record)
        panels.append((label, mapped, colors[good], record))

    panel_w, panel_h, pad = 455, 355, 20
    title_h, label_h = 74, 54
    canvas = Image.new("RGB", (3 * panel_w + 4 * pad,
                               title_h + 3 * (panel_h + label_h) + 4 * pad), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    draw.text((pad, 18), f"{scene_name} | 2,000 frames | {view} view", fill="#17212e", font=title_font)
    draw.text((pad, 48), "Same frames and camera; one scene-wide Sim(3) alignment; RGB brightened for display", fill="#4a5564", font=small_font)
    for index, (label, points, rgb, record) in enumerate(panels):
        col, row = index % 3, index // 3
        x = pad + col * (panel_w + pad)
        y = title_h + pad + row * (panel_h + label_h + pad)
        tile, clipped = render(points, rgb, center, basis, bounds, panel_w, panel_h)
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline="#b8c1cc", width=1)
        draw.text((x, y + panel_h + 5), label, fill="#17212e", font=font)
        if record is not None:
            note = f"sampled 3D error {record['mean_3d_m']:.3f} m; outside view {clipped:.0%}"
            draw.text((x, y + panel_h + 29), note, fill="#586575", font=small_font)
    output = Path(args.output) / f"{scene_name}_kmq_{view}.png"
    canvas.save(output, optimize=True)
    print(f"SAVED {output}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--views", nargs="+", default=["oblique", "top"])
    p.add_argument("--sample-frames", type=int, default=128)
    p.add_argument("--pixels-per-frame", type=int, default=1536)
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_name in args.scenes:
        for view in args.views:
            draw_scene(args, scene_name, view, rows)
    with open(Path(args.output) / "sampled_errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

