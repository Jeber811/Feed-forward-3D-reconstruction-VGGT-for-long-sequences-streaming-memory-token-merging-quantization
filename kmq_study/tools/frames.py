"""Frame selection shared by every model, without importing any model repo.

Copied from FastVGGT/vggt/utils/eval_utils.py (get_sorted_image_paths, load_poses,
build_frame_selection) so that all models see exactly the frames FastVGGT's ScanNet eval picks:
the first valid frame plus evenly spaced valid frames (valid = pose has no inf/NaN).
Only needs numpy, so it imports in every conda env.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_sorted_image_paths(images_dir: Path) -> List[Path]:
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_paths.extend(sorted(images_dir.glob(ext)))
    return image_paths


def load_poses(path: Path):
    """GT camera-to-world poses relative to the first valid frame, its original pose, and frame ids."""
    pose_files = sorted(path.glob("*.txt"), key=lambda x: int(x.stem))
    c2ws, ids = [], []
    for pose_file in pose_files:
        pose = np.array([float(x) for x in pose_file.read_text().strip().split()]).reshape(4, 4)
        if not (np.isinf(pose).any() or np.isnan(pose).any()):
            c2ws.append(pose)
            ids.append(int(pose_file.stem))
    c2ws = np.stack(c2ws)
    first_gt_pose = c2ws[0].copy()
    return np.linalg.inv(c2ws[0]) @ c2ws, first_gt_pose, np.array(ids)


def build_frame_selection(image_paths, available_pose_frame_ids, input_frame) -> Tuple[List[int], List[Path], List[int]]:
    all_image_frame_ids = [int(p.stem) for p in image_paths]
    valid_frame_ids = sorted(set(all_image_frame_ids) & set(available_pose_frame_ids))
    if os.getenv("SCANNET_FRAME_SELECTION", "uniform") == "prefix":
        selected = valid_frame_ids[:input_frame]
    elif len(valid_frame_ids) > input_frame:
        first_frame, remaining = valid_frame_ids[0], valid_frame_ids[1:]
        step = max(1, len(remaining) // (input_frame - 1))
        selected = [first_frame] + remaining[::step][: input_frame - 1]
    else:
        selected = valid_frame_ids
    id_to_path = {int(p.stem): p for p in image_paths}
    paths = [id_to_path[f] for f in selected if f in id_to_path]
    pose_idx = {f: i for i, f in enumerate(available_pose_frame_ids)}
    return selected, paths, [pose_idx[f] for f in selected if f in pose_idx]


class Scene:
    """One processed ScanNet scene (color/ + pose/), as written by tools/extract_sens.py."""

    def __init__(self, scene_dir):
        self.dir = Path(scene_dir)
        self.name = self.dir.name
        self.image_paths = get_sorted_image_paths(self.dir / "color")
        self.c2ws, self.first_gt_pose, self.pose_ids = load_poses(self.dir / "pose")

    def select(self, n):
        """frame_ids, image paths, GT c2ws (first-frame relative) for n evenly spaced frames."""
        ids, paths, pose_idx = build_frame_selection(self.image_paths, self.pose_ids, n)
        return ids, paths, self.c2ws[pose_idx]

    @property
    def num_valid(self):
        return len(set(int(p.stem) for p in self.image_paths) & set(self.pose_ids))


def symlink_dir(paths, root=None):
    """Temp folder of zero-padded symlinks to `paths`, so a repo's own folder loader reads exactly
    these frames in this order. Caller deletes it (shutil.rmtree)."""
    d = tempfile.mkdtemp(prefix="frames_", dir=root)
    for i, p in enumerate(paths):
        os.symlink(os.path.abspath(p), os.path.join(d, f"{i:05d}{Path(p).suffix}"))
    return d


def first_frame_w2c(c2w=None, w2c=None):
    """Predicted poses -> world-to-camera, re-expressed so frame 0 is the identity.

    This is the convention FastVGGT's eval_trajectory is given for VGGT (its extrinsics), and is
    compared there against inv(GT c2w) under Sim(3) alignment.
    """
    if w2c is None:
        w2c = np.linalg.inv(c2w)
    w2c = np.asarray(w2c, dtype=np.float64)
    return w2c @ np.linalg.inv(w2c[0])
