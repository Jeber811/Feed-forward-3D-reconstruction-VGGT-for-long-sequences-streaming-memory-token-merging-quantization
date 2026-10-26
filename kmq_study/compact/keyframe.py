"""K: keyframe selection -- which frames are kept in long-term memory.

Decided after a frame is processed, from its own predicted pose and depth (classic SLAM rule):
a frame is a keyframe if, relative to the last keyframe, the camera rotated more than
`kf_rot_deg`, or moved more than `kf_trans` x (median depth of the last keyframe), or
`kf_max_gap` frames passed. The depth ratio makes the threshold scale-free, since the model's
poses have no metric scale. Frame 0 is always the anchor.
"""
import numpy as np


class PoseKeyframes:
    def __init__(self, kf_rot_deg=15.0, kf_trans=0.15, kf_max_gap=30, **_):
        self.rot, self.trans, self.gap = kf_rot_deg, kf_trans, kf_max_gap
        self.last = None  # (frame index, camera-to-world, median depth)

    def __call__(self, i, w2c, median_depth):
        c2w = np.linalg.inv(w2c)
        if self.last is None:
            self.last = (i, c2w, median_depth)
            return True
        j, c2w_k, d_k = self.last
        rel = np.linalg.inv(c2w_k) @ c2w
        angle = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
        move = np.linalg.norm(rel[:3, 3]) / max(d_k, 1e-6)
        if angle > self.rot or move > self.trans or i - j >= self.gap:
            self.last = (i, c2w, median_depth)
            return True
        return False


class EveryFrame:
    """No K: every frame is a candidate for long-term memory (memory is then a sliding window)."""

    def __call__(self, i, w2c, median_depth):
        return True


def make(cfg):
    if cfg["keyframes"] is None:
        return EveryFrame()
    if cfg["keyframes"] == "pose":
        return PoseKeyframes(**cfg)
    raise ValueError(f"unknown keyframes policy {cfg['keyframes']}")
