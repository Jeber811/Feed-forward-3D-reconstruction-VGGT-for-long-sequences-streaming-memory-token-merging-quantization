"""Chunked VGGT* / FastVGGT: full attention inside a chunk, Sim(3) alignment between chunks.

Frames are processed in order, `chunk_size` at a time. Every chunk after the first starts with
`anchors` frames that are already placed in the global frame, followed by new frames:

    chunk 1:  f0 ... f99                         -> defines the global frame (f0 = identity)
    chunk 2:  [a1 ... a10] + f100 ... f189       -> a = anchors
    ...

The anchors appear in both the global map and the new chunk, pixel for pixel, so their point maps
give point correspondences directly; a Sim(3) (Umeyama, confidence-masked, two trimmed rounds)
maps the chunk into the global frame. Only poses and 1/4-resolution point maps are kept, so peak
memory is set by `chunk_size`, not by the sequence length.

    no K: anchors = the last frames processed          (plain overlapping chunks)
    K:    anchors = keyframes (pose-change rule), half the most recent ones and half spread evenly
          over all earlier keyframes, frame 0 included -> old parts of the scene stay in view
    M:    FastVGGT token merging inside each chunk (VGGT(merging=0))
    Q:    int8 aggregator weights (quant.py)
Training-free: the released VGGT* / FastVGGT checkpoint (model_tracker_fixed_e20.pt).
Runs in the fastvggt conda env.
"""
import os
import sys
import time

import numpy as np
import torch

from keyframe import PoseKeyframes
from quant import int8_weights_

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "FastVGGT"))
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.eval_utils import get_vgg_input_imgs, load_images_rgb  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

DTYPE = torch.float16  # Turing
GB = 2**30


def umeyama(src, dst):
    """Sim(3) with dst ~= s * R @ src + t.  src, dst: [N, 3] float64."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, rounds=2, max_pts=50000, seed=0):
    rng = np.random.default_rng(seed)
    if len(src) > max_pts:
        keep = rng.choice(len(src), max_pts, replace=False)
        src, dst = src[keep], dst[keep]
    s, R, t = umeyama(src, dst)
    for _ in range(rounds):
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        ok = res < 2.0 * np.median(res)
        s, R, t = umeyama(src[ok], dst[ok])
    return s, R, t


class ChunkVGGT:
    def __init__(self, ckpt, cfg):
        self.cfg = cfg
        model = VGGT(merging=cfg["merging"], merge_ratio=0.9)  # 0.9 = FastVGGT's eval setting
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        if cfg["weights"] == "int8":
            n = int8_weights_(model.aggregator)
            print(f"[Q] {n} aggregator linears -> int8", flush=True)
        elif cfg["weights"] != "fp16":
            raise ValueError(f"weights must be fp16|int8, got {cfg['weights']}")
        self.model = model.cuda().eval().to(DTYPE)  # int8 buffers are left as int8

    @torch.no_grad()
    def _forward(self, paths, stride):
        images, pw, ph = get_vgg_input_imgs(np.stack(load_images_rgb(paths)))
        self.model.update_patch_dimensions(pw, ph)
        with torch.cuda.amp.autocast(dtype=DTYPE):
            pred = self.model(images.cuda().to(DTYPE))
        hw = images.shape[-2:]
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], hw)
        extr, intr = extr[0].float(), intr[0].float()                    # [S,3,4], [S,3,3]
        depth = pred["depth"][0, :, ::stride, ::stride, 0].float()       # [S,h,w]
        conf = pred["depth_conf"][0, :, ::stride, ::stride].float()
        # unproject at the sampled pixel centres -> chunk-frame points
        h, w = depth.shape[1:]
        v, u = torch.meshgrid(torch.arange(h, device=depth.device) * stride,
                              torch.arange(w, device=depth.device) * stride, indexing="ij")
        x = (u[None] - intr[:, 0, 2, None, None]) / intr[:, 0, 0, None, None] * depth
        y = (v[None] - intr[:, 1, 2, None, None]) / intr[:, 1, 1, None, None] * depth
        cam = torch.stack([x, y, depth], -1)                              # [S,h,w,3]
        R, t = extr[:, :, :3], extr[:, :, 3]
        world = torch.einsum("sji,shwj->shwi", R, cam - t[:, None, None, :])  # R^T (X - t)
        w2c = np.tile(np.eye(4), (len(paths), 1, 1))
        w2c[:, :3] = extr.cpu().numpy()
        del pred
        return w2c, world.cpu().numpy(), conf.cpu().numpy(), depth.median(-1).values.median(-1).values.cpu().numpy()

    def _pick_anchors(self, done, keys):
        A = self.cfg["anchors"]
        if self.cfg["keyframes"] is None or len(keys) <= A:
            pool = keys if self.cfg["keyframes"] is not None else done
            return pool[-A:]
        recent = keys[-(A - A // 2):]
        older = keys[: -(A - A // 2)]
        spread = [older[int(round(i))] for i in np.linspace(0, len(older) - 1, A // 2)]
        return sorted(set(spread) | set(recent))

    def run(self, paths, pts_stride=4):
        S, C, A = len(paths), self.cfg["chunk_size"], self.cfg["anchors"]
        w2c_g = [None] * S
        pts_g, conf_g = [None] * S, [None] * S
        is_key = np.zeros(S, bool)
        kf = PoseKeyframes(**self.cfg) if self.cfg["keyframes"] else None
        keys, done = [], []
        stats = {k: [] for k in ("t", "peak_gb", "chunk_frames", "anchor_frames", "sim3_scale", "align_res")}
        t0 = time.time()
        nxt = 0
        while nxt < S:
            anchors = self._pick_anchors(done, keys) if done else []
            new = list(range(nxt, min(S, nxt + C - len(anchors))))
            ids = anchors + new
            w2c, pts, conf, med = self._forward([paths[i] for i in ids], pts_stride)
            if anchors:
                na = len(anchors)
                src, dst = pts[:na].reshape(-1, 3), np.stack([pts_g[i] for i in anchors]).reshape(-1, 3)
                c_loc, c_glb = conf[:na].reshape(-1), np.stack([conf_g[i] for i in anchors]).reshape(-1)
                ok = (c_loc > np.median(c_loc)) & (c_glb > np.median(c_glb))
                s, R, t = robust_sim3(src[ok].astype(np.float64), dst[ok].astype(np.float64))
                res = np.median(np.linalg.norm(s * src[ok] @ R.T + t - dst[ok], axis=1))
            else:
                s, R, t, res = 1.0, np.eye(3), np.zeros(3), 0.0
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = s * R, t
            for j, i in enumerate(ids):
                if i in anchors:
                    continue
                c2w = np.linalg.inv(w2c[j])
                c2w_g = np.eye(4)
                c2w_g[:3, :3] = R @ c2w[:3, :3]
                c2w_g[:3, 3] = s * R @ c2w[:3, 3] + t
                w2c_g[i] = np.linalg.inv(c2w_g)
                pts_g[i] = (pts[j] @ (s * R).T + t).astype(np.float32)
                conf_g[i] = conf[j]
                done.append(i)
                if kf is not None and kf(i, w2c_g[i], float(med[j]) * s):
                    keys.append(i)
                    is_key[i] = True
            nxt = new[-1] + 1
            torch.cuda.synchronize()
            for k, v in (("t", time.time() - t0), ("peak_gb", torch.cuda.max_memory_allocated() / GB),
                         ("chunk_frames", len(ids)), ("anchor_frames", len(anchors)),
                         ("sim3_scale", s), ("align_res", res)):
                stats[k].append(v)
        out = dict(w2c=np.stack(w2c_g), pts=np.stack(pts_g).astype(np.float16), conf=np.stack(conf_g).astype(np.float16),
                   is_key=is_key, evicted=0, memory_frames_at_end=np.array(keys if kf else done[-A:]))
        out.update({k: np.array(v) for k, v in stats.items()})
        # run_variant.py prints these; for chunks "memory" = the anchors carried into the next chunk
        out.update(mem_frames=stats["chunk_frames"], mem_long=stats["anchor_frames"], mem_mb=[0.0], mem_tokens=[0])
        return out
