"""ScanNet evaluation for any VGGT-class baseline, on the exact pipeline of the groupmate's
eval_scannet_mem.py (FastVGGT eval: same frame selection, 518-px inputs, Umeyama-aligned
ATE/ARE/RPE, Chamfer vs. *_vh_clean_2.ply, same peak-memory instrumentation).

Models (--model):
  vggt_star   FastVGGT codebase, merging off (VGGT* = keeps only the 4 DPT layers)
  fastvggt    FastVGGT codebase, token merging on (--merging 0 by default)
  vggt_orig   official facebookresearch/vggt code + vggt_1b_model.pt
  pi3         official Pi3 (permutation-equivariant, no reference frame)
  cut3r         official CUT3R, frame-by-frame recurrent state (forward_recurrent)
  cut3r_offload same computation, per-frame outputs moved to CPU, no state history (bounded GPU memory)
  spann3r       official Spann3R (bounded spatial memory), 224x224 centre crop, PnP-RANSAC poses as in its demo
  streamvggt  StreamVGGT (causal attention + KV cache)

Every model is wrapped to return VGGT-style {pose_enc, depth, depth_conf} so the reconstruction
and metric code is shared. Models whose confidence is not VGGT's (1 + exp) get conf = 1 + p,
which with the default --depth_conf_thresh 1.0 means "no filtering" for every model alike.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

P = Path(__file__).resolve().parents[1]
# Third-party model code and checkpoints: shared, pinned locations (protocol/protocol.env); defaults = this checkout.
TP = Path(os.environ.get("THIRD_PARTY", P / "repos"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", P / "ckpt"))
FAST = (TP / "FastVGGT").resolve()
sys.path.insert(0, str(FAST))

from vggt.utils.eval_utils import (  # noqa: E402  (FastVGGT copy)
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
    eval_trajectory,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


def to_pose_enc(c2w, K, hw):
    """c2w [S,4,4], K [S,3,3] -> VGGT pose_enc [1,S,9].

    Poses are first re-expressed relative to frame 0 (VGGT convention: world = first camera),
    because the Chamfer step maps predictions to GT world with the first GT pose and only a
    scale/center fit (no rotation). Pi3 predicts in an arbitrary frame, so this is required there.
    """
    c2w = c2w.float()
    c2w = torch.linalg.inv(c2w[:1]) @ c2w
    w2c = torch.linalg.inv(c2w)[:, :3, :]
    return extri_intri_to_pose_encoding(w2c[None], K.float()[None], image_size_hw=hw)


def focal_from_local_points(pts):
    """Least-squares fx, fy per frame from camera-frame points [S,H,W,3], principal point at center."""
    S, H, W, _ = pts.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=pts.device, dtype=torch.float32),
        torch.arange(W, device=pts.device, dtype=torch.float32),
        indexing="ij",
    )
    du, dv = u - W / 2, v - H / 2
    z = pts[..., 2].float()
    ok = z > 1e-6
    a = torch.where(ok, pts[..., 0].float() / z.clamp(min=1e-6), torch.zeros_like(z))
    b = torch.where(ok, pts[..., 1].float() / z.clamp(min=1e-6), torch.zeros_like(z))
    fx = (a * du).sum((1, 2)) / (a * a).sum((1, 2)).clamp(min=1e-12)
    fy = (b * dv).sum((1, 2)) / (b * b).sum((1, 2)).clamp(min=1e-12)
    K = torch.zeros(S, 3, 3, device=pts.device)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, W / 2, H / 2, 1
    return K


class _NoPatchMixin:
    def update_patch_dimensions(self, patch_width, patch_height):
        pass


class VGGTOrigWrapper(torch.nn.Module, _NoPatchMixin):
    """Official VGGT. Its package is also called `vggt`, so it is imported in isolation.

    The official `vggt/` has no __init__.py (namespace package), and a regular package anywhere on
    sys.path wins over a namespace portion, so FastVGGT must be *removed* from sys.path during the
    import, not just shadowed.
    """

    def __init__(self, ckpt):
        super().__init__()
        saved_mods = {k: v for k, v in sys.modules.items() if k == "vggt" or k.startswith("vggt.")}
        for k in saved_mods:
            del sys.modules[k]
        saved_path = list(sys.path)
        sys.path[:] = [str(TP / "vggt")] + [p for p in sys.path if Path(p).resolve() != FAST]
        try:
            from vggt.models.vggt import VGGT as OfficialVGGT

            src = sys.modules[OfficialVGGT.__module__].__file__  # before sys.modules is restored
        finally:
            sys.path[:] = saved_path
            for k in [k for k in sys.modules if k == "vggt" or k.startswith("vggt.")]:
                del sys.modules[k]
            sys.modules.update(saved_mods)
        print(f"[vggt_orig] VGGT class loaded from {src}", flush=True)
        assert "/repos/vggt/" in src, f"official VGGT import resolved to {src}"
        self.model = OfficialVGGT()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.cuda().eval()  # official recipe: fp32 weights + autocast

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        out = self.model(images.float())
        return {"pose_enc": out["pose_enc"], "depth": out["depth"], "depth_conf": out["depth_conf"]}


class Pi3Wrapper(torch.nn.Module, _NoPatchMixin):
    def __init__(self, ckpt):
        super().__init__()
        sys.path.insert(0, str(TP / "Pi3"))
        # Pi3 looks for croco's compiled RoPE as `models.curope` and otherwise falls back to a slow
        # PyTorch RoPE (its documented default). Opt-in reuse of the CUT3R-compiled kernel: PI3_CUROPE=1.
        if os.environ.get("PI3_CUROPE", "0") == "1":  # off: croco kernel needs (B,N,H,D)-contiguous tokens, Pi3 passes (B,H,N,D) -> "tokens are not contiguous"
            sys.path.insert(0, str(TP / "CUT3R" / "src" / "croco"))
        from pi3.models.pi3 import Pi3
        from pi3.models.layers import pos_embed as _pe

        print(f"[pi3] RoPE2D implementation: {_pe.RoPE2D.__module__}.{_pe.RoPE2D.__name__}", flush=True)
        from safetensors.torch import load_file

        self.model = Pi3()
        self.model.load_state_dict(load_file(ckpt), strict=True)
        self.model = self.model.cuda().eval()  # official recipe: fp32 weights + autocast

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        S, _, H, W = images.shape
        out = self.model(images.float()[None])
        # The harness calls us inside fp16 autocast; geometry post-processing must stay fp32.
        with torch.autocast("cuda", enabled=False):
            local = out["local_points"][0].float()  # [S,H,W,3] camera frame
            K = focal_from_local_points(local)
            conf = 1.0 + torch.sigmoid(out["conf"][0, ..., 0].float())
            pose_enc = to_pose_enc(out["camera_poses"][0].float(), K, (H, W))
        return {"pose_enc": pose_enc, "depth": local[..., 2:3][None], "depth_conf": conf[None]}


def weighted_sim3(X, Y, w, iters=5):
    """Robust weighted Umeyama: s, R, t with Y ~= s R X + t (float64). IRLS with Huber weights,
    delta = 2 x median residual, so outlier points from inconsistent predictions are down-weighted."""
    X, Y, w0 = X.double(), Y.double(), w.double()
    w = w0.clone()
    for _ in range(iters):
        wn = w / w.sum()
        mx, my = (wn[:, None] * X).sum(0), (wn[:, None] * Y).sum(0)
        Xc, Yc = X - mx, Y - my
        U, D, Vt = torch.linalg.svd((wn[:, None] * Yc).T @ Xc)
        E = torch.eye(3, dtype=X.dtype, device=X.device)
        if torch.det(U) * torch.det(Vt) < 0:
            E[2, 2] = -1
        R = U @ E @ Vt
        s = (D * E.diagonal()).sum() / (wn * (Xc ** 2).sum(1)).sum()
        t = my - s * R @ mx
        r = (Y - (s * X @ R.T + t)).norm(dim=1)
        delta = 2 * r.median().clamp(min=1e-9)
        w = w0 * torch.where(r <= delta, torch.ones_like(r), delta / r)
    return s.float(), R.float(), t.float(), float(r.median())


class ChunkWrapper(torch.nn.Module):
    """Training-free chunked inference with a keyframe memory (our method, week 2 prototype).
    Backbone "pi3" (pi3_chunk) or a VGGT*/FastVGGT model (vggt_chunk ~ VGGT-Long without loop closure;
    fastvggt_chunk = token merging x chunking). Env: CH_SIZE, CH_OVERLAP, CH_KF (fallback PI3_*), CH_ALIGN=irls|svd
    (svd = single weighted Umeyama as in SwiftVGGT, irls = Huber IRLS as in VGGT-Long).

    Frames are processed in temporal order in chunks of at most PI3_CHUNK frames. Each chunk = memory
    frames (already placed in the global frame) + new frames. Memory = the last PI3_OVERLAP processed
    frames (local overlap, VGGT-Long style) + PI3_KF keyframes spread evenly over the older history
    (long-range anchors). Pi3 is permutation-equivariant, so memory frames are ordinary set members,
    no reference frame. Each chunk is mapped to the global frame by a robust confidence-weighted
    Sim(3) fit between its memory frames' points and their stored global points; only the new
    frames' results are kept. Per-frame outputs live on CPU, so GPU memory is bounded by the chunk."""

    def __init__(self, backbone, ckpt=None, vggt_model=None):
        super().__init__()
        self.backbone = backbone
        self.inner = Pi3Wrapper(ckpt).model if backbone == "pi3" else vggt_model
        env = lambda k, d: int(os.environ.get("CH_" + k, os.environ.get("PI3_" + {"SIZE": "CHUNK"}.get(k, k), d)))
        self.chunk, self.overlap, self.n_kf = env("SIZE", 200), env("OVERLAP", 30), env("KF", 0)
        self.align = os.environ.get("CH_ALIGN", "irls")
        self.stride = 4  # pixel stride for the alignment points
        assert self.overlap >= 1 and self.chunk > self.overlap + self.n_kf and self.align in ("irls", "svd")
        print(f"[chunk] backbone={backbone} chunk={self.chunk} overlap={self.overlap} kf={self.n_kf} "
              f"align={self.align}", flush=True)
        self.diag = []

    def update_patch_dimensions(self, patch_width, patch_height):
        if self.backbone != "pi3":
            self.inner.update_patch_dimensions(patch_width, patch_height)

    @torch.no_grad()
    def _predict(self, x):
        """x [n,3,H,W] -> c2w [n,4,4], camera-frame points [n,H,W,3], weight in [0,1) [n,H,W], K [n,3,3] (fp32)."""
        n, _, H, W = x.shape
        if self.backbone == "pi3":
            out = self.inner(x.float()[None])
            with torch.autocast("cuda", enabled=False):
                loc = out["local_points"][0].float()
                return (out["camera_poses"][0].float(), loc, torch.sigmoid(out["conf"][0, ..., 0].float()),
                        focal_from_local_points(loc))
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        out = self.inner(x, image_paths=None)
        with torch.autocast("cuda", enabled=False):
            ext, K = pose_encoding_to_extri_intri(out["pose_enc"].float(), (H, W))
            w2c = torch.eye(4, device=x.device).repeat(n, 1, 1)
            w2c[:, :3, :] = ext[0].float()
            K = K[0].float()
            z = out["depth"][0, ..., 0].float()
            v, u = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                                  torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
            xs = (u[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None] * z
            ys = (v[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None] * z
            w = 1 - 1 / out["depth_conf"][0].float().clamp(min=1)  # VGGT conf = 1 + exp(.) >= 1
            return torch.linalg.inv(w2c), torch.stack([xs, ys, z], -1), w, K

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        S, _, H, W = images.shape
        c2w_g = torch.zeros(S, 4, 4)
        loc_g = torch.zeros(S, H, W, 3, dtype=torch.float16)  # camera-frame points, global scale
        conf_g = torch.zeros(S, H, W)
        K_g = torch.zeros(S, 3, 3)
        st = self.stride
        self.diag, done = [], 0
        while done < S:
            mem = []
            if done:
                recent = list(range(max(0, done - self.overlap), done))
                old = done - len(recent)
                kf = [] if old <= 0 or not self.n_kf else sorted(set(
                    np.linspace(0, old - 1, min(self.n_kf, old)).round().astype(int).tolist()))
                mem = (recent + kf) if os.environ.get("CH_MEMORDER") == "recent_first" else (kf + recent)
            new = list(range(done, min(S, done + self.chunk - len(mem))))
            idx = mem + new
            c2w, loc, cf, K = self._predict(images[idx].cuda())  # no-op unless inputs are kept on CPU
            with torch.autocast("cuda", enabled=False):
                if not mem:
                    s, R, t, res = torch.tensor(1.0), torch.eye(3), torch.zeros(3), 0.0
                    s, R, t = s.cuda(), R.cuda(), t.cuda()
                else:
                    m = len(mem)
                    Xc = torch.einsum("nij,nhwj->nhwi", c2w[:m, :3, :3], loc[:m, ::st, ::st]) + c2w[:m, None, None, :3, 3]
                    gm = c2w_g[mem].cuda()
                    lg = loc_g[mem][:, ::st, ::st].cuda().float()
                    Yg = torch.einsum("nij,nhwj->nhwi", gm[:, :3, :3], lg) + gm[:, None, None, :3, 3]
                    w = (cf[:m, ::st, ::st] * conf_g[mem][:, ::st, ::st].cuda()).reshape(-1)
                    keep = w >= w.median()
                    s, R, t, res = weighted_sim3(Xc.reshape(-1, 3)[keep], Yg.reshape(-1, 3)[keep], w[keep],
                                                 iters=5 if self.align == "irls" else 1)
                nn = len(new)
                cn = c2w[-nn:].clone()
                cn[:, :3, :3] = R @ cn[:, :3, :3]
                cn[:, :3, 3] = s * cn[:, :3, 3] @ R.T + t
                c2w_g[new] = cn.cpu()
                loc_g[new] = (s * loc[-nn:]).half().cpu()
                conf_g[new] = cf[-nn:].cpu()
                K_g[new] = K[-nn:].cpu()
            self.diag.append({"start": done, "n_mem": len(mem), "n_new": nn, "scale": round(float(s), 4),
                              "median_resid": round(res, 5)})
            print(f"[chunk] {self.diag[-1]}", flush=True)
            del c2w, loc, cf
            done = new[-1] + 1
        with torch.autocast("cuda", enabled=False):
            pose_enc = to_pose_enc(c2w_g.cuda(), K_g.cuda(), (H, W))
        return {"pose_enc": pose_enc, "depth": loc_g[..., 2:3].float()[None].to(images.device),
                "depth_conf": (1.0 + conf_g)[None].to(images.device)}


class Pi3KFInterpWrapper(Pi3Wrapper):
    """Simple baseline: Pi3 on PI3_KFN (default 300) evenly spaced keyframes only; poses of all other
    frames by SLERP/linear interpolation between neighbouring keyframes. Non-keyframes contribute no
    points (conf 0 -> dropped), so Chamfer uses keyframe geometry only."""

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        from scipy.spatial.transform import Rotation, Slerp

        S, _, H, W = images.shape
        n = min(int(os.environ.get("PI3_KFN", 300)), S)
        kf = np.unique(np.linspace(0, S - 1, n).round().astype(int))
        out = self.model(images[kf.tolist()].cuda().float()[None])
        with torch.autocast("cuda", enabled=False):
            loc = out["local_points"][0].float()
            K = focal_from_local_points(loc).cpu()
            conf = (1.0 + torch.sigmoid(out["conf"][0, ..., 0].float())).cpu()
            c2w_k = out["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        ii = np.arange(S)
        c2w = np.tile(np.eye(4), (S, 1, 1))
        c2w[:, :3, :3] = Slerp(kf, Rotation.from_matrix(c2w_k[:, :3, :3]))(ii).as_matrix()
        for d in range(3):
            c2w[:, d, 3] = np.interp(ii, kf, c2w_k[:, d, 3])
        near = kf[np.abs(ii[:, None] - kf[None]).argmin(1)]
        pos = {int(f): j for j, f in enumerate(kf)}
        Kall = K[[pos[int(f)] for f in near]]
        depth = torch.full((S, H, W), float("nan"))
        cfull = torch.zeros((S, H, W))
        depth[kf.tolist()] = loc[..., 2].cpu()
        cfull[kf.tolist()] = conf
        with torch.autocast("cuda", enabled=False):
            pose_enc = to_pose_enc(torch.from_numpy(c2w).float(), Kall, (H, W))
        dev = images.device  # CPU when CAP_CPU_INPUTS=1 (outputs stay off the GPU too)
        return {"pose_enc": pose_enc.to(dev), "depth": depth[None, ..., None].to(dev), "depth_conf": cfull[None].to(dev)}


class CUT3RWrapper(torch.nn.Module, _NoPatchMixin):
    """CUT3R at its native 512x384, run frame by frame (forward_recurrent), outputs resized back."""

    SIZE_HW = (384, 512)

    def __init__(self, ckpt, offload=False, ttt3r=False):
        super().__init__()
        # TTT3R (training-free, same CUT3R weights) ships a copy of CUT3R whose rollout can return attention maps
        repo = TP / ("TTT3R" if ttt3r else "CUT3R")
        sys.path.insert(0, str(repo / "src"))
        sys.path.insert(0, str(repo))
        # curope is built for sm_75/80/86 (scripts/setup_cut3r.sbatch). croco's PyTorch RoPE2D fallback is NOT usable:
        # CUT3R gives the pose token position -1, which the fallback's cos/sin table lookup rejects (device assert).
        assert torch.cuda.get_device_capability() >= (7, 5), "CUT3R/TTT3R need curope: run on Turing/Ampere, not V100"
        from dust3r.model import ARCroco3DStereo

        print(f"[cut3r] model code from {sys.modules[ARCroco3DStereo.__module__].__file__} ttt3r={ttt3r}", flush=True)
        self.model = ARCroco3DStereo.from_pretrained(ckpt).cuda().eval()  # official: fp32
        self.offload = offload or ttt3r
        self.ttt3r = ttt3r

    @torch.no_grad()
    def _run_offload(self, images, h, w):
        """Same per-frame computation as ARCroco3DStereo.forward_recurrent (image-only views,
        update=True, reset=False), but each frame's outputs go to CPU immediately and no per-frame
        state history is kept, so GPU memory is bounded by the recurrent state, not by S."""
        m = self.model
        assert m.pose_head_flag
        dev = torch.device("cuda")  # inputs may be in host RAM (CAP_CPU_INPUTS=1): moved one frame at a time
        shape = torch.tensor([[h, w]], device=dev)
        preds = []
        with torch.autocast("cuda", enabled=False):
            for i in range(images.shape[0]):
                xi = F.interpolate(images[i : i + 1].to(dev).float(), size=(h, w), mode="bilinear", align_corners=False) * 2 - 1
                img_out, pos_i, _ = m._encode_image(xi, shape)
                feat_i = img_out[-1]
                if i == 0:
                    state_feat, state_pos = m._init_state(feat_i, pos_i)
                    mem = m.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                    init_state_feat = state_feat.clone()
                glob_i = m._get_img_level_feat(feat_i)
                pose_feat_i = (m.pose_token.expand(feat_i.shape[0], -1, -1) if i == 0
                               else m.pose_retriever.inquire(glob_i, mem))
                pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=dev, dtype=pos_i.dtype)
                ones = torch.ones(1, dtype=torch.bool, device=dev)
                ro = m._recurrent_rollout(
                    state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i, init_state_feat,
                    img_mask=ones, reset_mask=~ones, update=ones, **({"return_attn": True} if self.ttt3r else {}),
                )
                new_state_feat, dec = ro[0], ro[1]
                mem = m.pose_retriever.update_mem(mem, glob_i, dec[-1][:, 0:1])
                if self.ttt3r and i > 0:
                    # TTT3R (model.py, model_update_type="ttt3r"): per-state-token learning rate =
                    # sigmoid(mean over layers, heads and image tokens of the state->image cross-attention)
                    lr = torch.sigmoid(torch.cat(ro[3], dim=0).mean(dim=(0, 1, 3)))[None, :, None]
                    state_feat = new_state_feat * lr + state_feat * (1 - lr)
                else:
                    state_feat = new_state_feat  # update_mask = 1, reset_mask = 0 in the official loop
                del ro
                d = m.dec_depth
                head_input = [dec[0].float(), dec[d * 2 // 4][:, 1:].float(),
                              dec[d * 3 // 4][:, 1:].float(), dec[d].float()]
                res = m._downstream_head(head_input, shape, pos=pos_i)
                preds.append({k: res[k].cpu() for k in ("pts3d_in_self_view", "conf_self", "camera_pose")})
                del img_out, dec, res, head_input
        return preds

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        from dust3r.inference import inference_recurrent

        S, _, H, W = images.shape
        h, w = self.SIZE_HW
        if self.offload:
            return self._finish(self._run_offload(images, h, w), S, H, W, h, w)
        x = F.interpolate(images.float(), size=(h, w), mode="bilinear", align_corners=False) * 2 - 1
        views = []
        for i in range(S):
            views.append({
                "img": x[i : i + 1],
                "ray_map": torch.full((1, 6, h, w), torch.nan, device=x.device),
                "true_shape": torch.tensor([[h, w]]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.eye(4, device=x.device)[None],
                "img_mask": torch.tensor([True], device=x.device),
                "ray_mask": torch.tensor([False], device=x.device),
                "update": torch.tensor([True], device=x.device),
                "reset": torch.tensor([False], device=x.device),
            })
        with torch.autocast("cuda", enabled=False):
            res, _ = inference_recurrent(views, self.model, "cuda", verbose=False)
        return self._finish(res["pred"], S, H, W, h, w)

    @staticmethod
    def _finish(preds, S, H, W, h, w):
        from dust3r.utils.camera import pose_encoding_to_camera
        from dust3r.post_process import estimate_focal_knowing_depth

        pts = torch.cat([p["pts3d_in_self_view"] for p in preds]).float()  # [S,h,w,3]
        conf = torch.cat([p["conf_self"] for p in preds]).float()  # [S,h,w]
        c2w = torch.cat([pose_encoding_to_camera(p["camera_pose"].clone()) for p in preds]).float()
        pp = torch.tensor([w // 2, h // 2]).float().repeat(S, 1)
        f = estimate_focal_knowing_depth(pts, pp, focal_mode="weiszfeld").float().reshape(S)
        K = torch.zeros(S, 3, 3)
        K[:, 0, 0], K[:, 1, 1] = f * W / w, f * H / h
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2, H / 2, 1
        depth = F.interpolate(pts[..., 2][:, None], size=(H, W), mode="nearest")[:, 0]
        conf = F.interpolate(conf[:, None], size=(H, W), mode="nearest")[:, 0]
        # CUT3R conf is already >= 1 (1 + exp), same convention as VGGT
        return {
            "pose_enc": to_pose_enc(c2w, K, (H, W)).cuda(),
            "depth": depth[None, ..., None].cuda(),
            "depth_conf": conf[None].cuda(),
        }


class Spann3RWrapper(torch.nn.Module, _NoPatchMixin):
    """Official Spann3R (DUSt3R backbone + bounded spatial memory), following demo.py exactly:
    fp32, 224x224 input (centred square crop of our 518x392 frame), focal from the first frame
    (Weiszfeld), per-frame pose by PnP-RANSAC on the predicted global points. Depth/conf are mapped
    back into the full 518x392 frame (NaN / conf 0 outside the crop, so those pixels are dropped)."""

    S = 224

    def __init__(self, ckpt):
        super().__init__()
        root = TP / "spann3r"
        sys.path.insert(0, str(root))
        from spann3r.model import Spann3R

        self.model = Spann3R(dus3r_name=str(CKPT_DIR / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"), use_feat=False)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model = self.model.cuda().eval()

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        import cv2
        from dust3r.post_process import estimate_focal_knowing_depth

        n, _, H, W = images.shape
        side = min(H, W)
        y0, x0 = (H - side) // 2, (W - side) // 2
        crop = images[:, :, y0 : y0 + side, x0 : x0 + side].float()
        x = F.interpolate(crop, size=(self.S, self.S), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        shape = torch.tensor([[self.S, self.S]], device=x.device)
        frames = [{"img": x[i : i + 1], "true_shape": shape} for i in range(n)]
        with torch.autocast("cuda", enabled=False):
            preds, _ = self.model.forward(frames)
            pts = torch.cat([p["pts3d" if i == 0 else "pts3d_in_other_view"] for i, p in enumerate(preds)]).float()
            conf = torch.cat([p["conf"] for p in preds]).float()  # [n,S,S], DUSt3R conf = 1 + exp(.) >= 1
        del preds
        pts_np = pts.cpu().numpy()
        pp = torch.tensor((self.S / 2, self.S / 2))
        f = float(estimate_focal_knowing_depth(pts[:1].cpu(), pp, focal_mode="weiszfeld").reshape(-1)[0])
        K224 = np.array([[f, 0, self.S / 2], [0, f, self.S / 2], [0, 0, 1]], np.float32)
        u, v = np.meshgrid(np.arange(self.S), np.arange(self.S))
        uv = np.stack((u, v), -1).reshape(-1, 2).astype(np.float32)
        c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        depth224 = np.full((n, self.S, self.S), np.nan, np.float32)
        for i in range(n):  # demo.py: cv2.solvePnPRansac on all pixels, zero distortion
            ok, rvec, tvec, _ = cv2.solvePnPRansac(pts_np[i].reshape(-1, 3).astype(np.float32), uv, K224,
                                                   np.zeros(4, np.float32))
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3], w2c[:3, 3] = R, tvec.reshape(3)
            c2w[i] = np.linalg.inv(w2c)
            depth224[i] = (pts_np[i].reshape(-1, 3) @ R.T + tvec.reshape(1, 3))[:, 2].reshape(self.S, self.S)
        d = F.interpolate(torch.from_numpy(depth224)[:, None], size=(side, side), mode="nearest")[:, 0]
        c = F.interpolate(conf.cpu()[:, None], size=(side, side), mode="nearest")[:, 0]
        depth = torch.full((n, H, W), float("nan"))
        cfull = torch.zeros((n, H, W))
        depth[:, y0 : y0 + side, x0 : x0 + side], cfull[:, y0 : y0 + side, x0 : x0 + side] = d, c
        s = side / self.S
        K = torch.tensor([[f * s, 0, W / 2], [0, f * s, H / 2], [0, 0, 1]], dtype=torch.float32).repeat(n, 1, 1)
        with torch.autocast("cuda", enabled=False):
            pose_enc = to_pose_enc(torch.from_numpy(c2w), K, (H, W))
        dev = images.device  # CPU when CAP_CPU_INPUTS=1 (outputs stay off the GPU too)
        return {"pose_enc": pose_enc.to(dev), "depth": depth[None, ..., None].to(dev), "depth_conf": cfull[None].to(dev)}


def recon_metrics(c2w_est, c2w_gt, pts, gt_pts, n_sample=200_000, seed=33):
    """Reconstruction metric replacing FastVGGT's saturated Chamfer (week 2).

    Predicted points are mapped to the GT world with the Sim(3) that aligns predicted camera centres to GT camera
    centres (Umeyama, as for ATE-cc) - no bounding-box scale fit, no 0.5 m clipping. Then, on random subsets of
    n_sample points: accuracy = pred->GT nearest-neighbour distance, completeness = GT->pred (mean and median, m),
    and precision/recall/F-score at 5 cm and 10 cm. GT = ScanNet vh_clean_2 mesh vertices."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    X, Y = c2w_est[:, :3, 3].astype(np.float64), c2w_gt[:, :3, 3].astype(np.float64)
    mx, my = X.mean(0), Y.mean(0)
    U, D, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / ((X - mx) ** 2).sum(1).mean()
    t = my - s * R @ mx
    P_ = np.concatenate([p.reshape(-1, 3) for p in pts if len(p)])
    P_ = P_[np.isfinite(P_).all(1)]
    P_ = P_[rng.choice(len(P_), min(n_sample, len(P_)), replace=False)].astype(np.float64)
    P_ = s * P_ @ R.T + t
    G = np.asarray(gt_pts, np.float64)
    G = G[rng.choice(len(G), min(n_sample, len(G)), replace=False)]
    d_acc = cKDTree(G).query(P_, workers=-1)[0]
    d_comp = cKDTree(P_).query(G, workers=-1)[0]
    out = {"rc_acc_mean": d_acc.mean(), "rc_acc_med": np.median(d_acc),
           "rc_comp_mean": d_comp.mean(), "rc_comp_med": np.median(d_comp), "rc_sim3_scale": s}
    for th in (0.05, 0.10):
        p, r = (d_acc < th).mean(), (d_comp < th).mean()
        out[f"rc_f{int(th * 100)}"] = 2 * p * r / max(p + r, 1e-12)
        out[f"rc_prec{int(th * 100)}"], out[f"rc_rec{int(th * 100)}"] = p, r
    return {k: float(v) for k, v in out.items()}


# Models whose wrappers move their own frames to the GPU (inputs may stay in host RAM, CAP_CPU_INPUTS=1).
CPU_INPUT_MODELS = ("pi3_chunk", "vggt_chunk", "fastvggt_chunk", "pi3_kfinterp", "cut3r_offload", "ttt3r_offload")


def infer_cpu_inputs(model, vgg_input, dtype, depth_conf_thresh):
    """Same as FastVGGT's infer_vggt_and_reconstruct, but the input frames stay in host RAM (CAP_CPU_INPUTS=1).
    The FastVGGT protocol stages every frame on the GPU (~3.4 MB/frame measured), which
    dominates peak memory for chunked models at thousands of frames. Only for wrappers that move their own chunks to
    the GPU (pi3_chunk / vggt_chunk / fastvggt_chunk / pi3_kfinterp). Same post-processing -> identical metrics."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(vgg_input.to(dtype), image_paths=None)
    torch.cuda.synchronize()
    t_ms = (time.time() - start) * 1000.0
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (vgg_input.shape[2], vgg_input.shape[3]))
    depth = pred["depth"][0].detach().float().cpu().numpy()
    depth[~(pred["depth_conf"][0].detach().float().cpu().numpy() >= depth_conf_thresh)] = np.nan
    ext_np, intr_np = ext[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    wp = unproject_depth_map_to_point_map(depth, ext_np, intr_np)
    pts = []
    for f in wp:
        p = f.reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        if len(p):
            pts.append(p)
    cams = np.tile(np.eye(4, dtype=ext_np.dtype), (len(ext_np), 1, 1))
    cams[:, :3, :] = ext_np
    return None, None, pts, None, list(cams), t_ms


def build_model(args, dtype):
    if args.model in ("vggt_star", "fastvggt", "vggt_chunk", "fastvggt_chunk"):
        from vggt.models.vggt import VGGT

        merging = None if args.model in ("vggt_star", "vggt_chunk") else args.merging
        m = VGGT(merging=merging, merge_ratio=args.merge_ratio, vis_attn_map=False)
        m.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"), strict=False)
        m = m.cuda().eval().to(dtype)
        return ChunkWrapper("vggt", vggt_model=m) if args.model.endswith("_chunk") else m
    if args.model == "pi3_chunk":
        return ChunkWrapper("pi3", ckpt=args.ckpt_path)
    if args.model == "vggt_orig":
        return VGGTOrigWrapper(args.ckpt_path)
    if args.model == "pi3":
        return Pi3Wrapper(args.ckpt_path)
    if args.model == "pi3_kfinterp":
        return Pi3KFInterpWrapper(args.ckpt_path)
    if args.model in ("cut3r", "cut3r_offload", "ttt3r_offload"):
        return CUT3RWrapper(args.ckpt_path, offload=args.model != "cut3r", ttt3r=args.model == "ttt3r_offload")
    if args.model == "spann3r":
        return Spann3RWrapper(args.ckpt_path)
    if args.model == "streamvggt":
        sys.path.insert(0, str(FAST / "eval"))
        from eval_scannet_streamvggt import StreamVGGTWrapper

        return StreamVGGTWrapper(args.ckpt_path)
    raise ValueError(args.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["vggt_star", "fastvggt", "vggt_orig", "pi3", "pi3_chunk", "pi3_kfinterp", "vggt_chunk",
                             "fastvggt_chunk", "ttt3r_offload",
                             "cut3r", "cut3r_offload", "streamvggt", "spann3r"])
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--gt_ply_dir", type=Path, default=Path("/squash/scannetv2/scans"))
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--input_frame", type=int, required=True)
    ap.add_argument("--output_path", type=Path, required=True)
    ap.add_argument("--merging", type=int, default=0)
    ap.add_argument("--merge_ratio", type=float, default=0.9)
    ap.add_argument("--depth_conf_thresh", type=float, default=1.0)
    ap.add_argument("--chamfer_max_dist", type=float, default=0.5)
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--max_eval_points", type=int, default=150_000_000,
                    help="thin points for the Chamfer step above this many (host-RAM guard; ATE unaffected)")
    ap.add_argument("--rerun", action="store_true",
                    help="recompute scenes that already have metrics.json (or a recorded OOM); default: skip them")
    args = ap.parse_args()
    torch.manual_seed(33)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if os.environ.get("CAP_DTYPE"):  # testing only, e.g. exercise the bf16 (Ampere) path on a Turing card
        dtype = getattr(torch, os.environ["CAP_DTYPE"])
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu} | dtype {dtype} | model {args.model} | N={args.input_frame}", flush=True)
    model = build_model(args, dtype)
    model_mem_gb = torch.cuda.memory_allocated() / 2**30
    print(f"[MEM] model weights: {model_mem_gb:.2f} GB", flush=True)

    all_scenes_metrics = {"scenes": {}, "average": {}}
    for scene in args.scenes:
        scene_dir = args.data_dir / scene
        out_dir = args.output_path / f"input_frame_{args.input_frame}" / scene
        if not args.rerun:
            prev = out_dir / "memory.json"
            done = (out_dir / "metrics.json").exists() or (
                prev.exists() and json.loads(prev.read_text()).get("status") == "oom")
            if done:
                print(f"[SKIP] {scene} N={args.input_frame}: result already recorded", flush=True)
                continue
        image_paths = get_sorted_image_paths(scene_dir / "color")
        poses_gt, first_gt_pose, avail = load_poses(scene_dir / "pose")
        frame_ids, image_paths, pose_idx = build_frame_selection(image_paths, avail, args.input_frame)
        c2ws = poses_gt[pose_idx]
        print(f"[SCENE] {scene}: {len(image_paths)} frames", flush=True)
        mem = {"scene": scene, "model": args.model, "gpu": gpu, "dtype": str(dtype),
               "num_frames": len(image_paths), "model_weights_gb": round(model_mem_gb, 3)}
        try:
            images = np.stack(load_images_rgb(image_paths))
            vgg_input, pw, ph = get_vgg_input_imgs(images)
            del images  # full-res frames are not needed after preprocessing (frees ~3.8 MB/frame of host RAM)
            model.update_patch_dimensions(pw, ph)
            torch.cuda.empty_cache()
            cpu_in = os.environ.get("CAP_CPU_INPUTS") == "1"
            if cpu_in:
                assert args.model in CPU_INPUT_MODELS, args.model
                mem["inputs_on"] = "cpu"

            def infer(x, paths):
                if cpu_in:
                    return infer_cpu_inputs(model, x, dtype, args.depth_conf_thresh)
                return infer_vggt_and_reconstruct(model, x, dtype, args.depth_conf_thresh, paths)

            # protocol v0.2: one small warm-up pass (CUDA kernels/allocator), excluded from time and peak memory
            infer(vgg_input[:4], image_paths[:4])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # CAP_TIMING_REPEATS=R (timing runs only): repeat the identical forward R times, report the median
            times = []
            for _ in range(max(1, int(os.environ.get("CAP_TIMING_REPEATS", "1")))):
                _, _, pts, _, cams, t_ms = infer(vgg_input, image_paths)
                times.append(t_ms)
            t_ms = float(np.median(times))
            mem.update({
                "status": "ok",
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                "inference_time_ms": round(t_ms, 1),
                "inference_time_ms_all": [round(t, 1) for t in times],
                "fps": round(len(image_paths) / (t_ms / 1000), 3),
            })
            if getattr(model, "diag", None):
                mem["chunks"] = model.diag
                mem["chunk_cfg"] = {"chunk": model.chunk, "overlap": model.overlap, "kf": model.n_kf}
            print(f"[MEM] {json.dumps(mem)}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "memory.json").write_text(json.dumps(mem, indent=2))
            # Host-RAM guard for very long runs: FastVGGT's Chamfer step stacks every predicted point in
            # float64 several times (~400M points at 2000 frames -> >80 GB). Above MAX_EVAL_POINTS, thin
            # each frame's points uniformly at random before that step. Poses/ATE/RPE are untouched;
            # Chamfer itself already random-samples 100k points, so this only thins earlier. The cap is
            # 150M points: only runs above ~740 full frames are thinned (none of the earlier results). Flagged.
            total_pts = sum(len(p) for p in pts)
            thinned = None
            if total_pts > args.max_eval_points:
                keep = args.max_eval_points / total_pts
                rng = np.random.default_rng(33)
                pts = [p[rng.random(len(p)) < keep] for p in pts]
                thinned = {"chamfer_points_total": int(total_pts), "chamfer_points_kept": int(sum(len(p) for p in pts))}
                print(f"[THIN] {scene}: {total_pts} -> {thinned['chamfer_points_kept']} points for Chamfer", flush=True)
            metrics = evaluate_scene_and_save(
                scene, c2ws, first_gt_pose, frame_ids, cams, pts, out_dir,
                args.gt_ply_dir, args.chamfer_max_dist, t_ms, bool(args.plot),
            )
            if metrics is not None:
                # FastVGGT's ATE aligns the translation column of world-to-camera matrices, which is
                # not the camera centre. Also report the standard metric: camera centres from
                # camera-to-world poses, Sim(3)-Umeyama aligned (same evo code path, same alignment).
                n = min(len(cams), len(c2ws))
                c2w_est = np.linalg.inv(np.asarray(cams)[:n])
                # protocol v0.2: keep the predicted trajectory for drift analysis (tools/drift_analysis.py).
                # Predictions only (model frame) - GT poses are ScanNet data and are re-read on the cluster.
                np.savez_compressed(out_dir / "trajectory.npz", c2w_est=c2w_est.astype(np.float32),
                                    frame_ids=np.asarray(frame_ids[:n]))
                cc, _, _ = eval_trajectory(c2w_est, c2ws[:n], frame_ids[:n], align=True)
                metrics.update({f"{k}_cc": float(v) for k, v in cc.items()})
                met_file = out_dir / "metrics.json"
                saved = json.loads(met_file.read_text())
                saved.update({f"{k}_cc": float(v) for k, v in cc.items()})
                try:
                    from vggt.utils.eval_utils import load_gt_pointcloud

                    # load_poses() re-expresses GT poses relative to frame 0; the GT mesh is in absolute ScanNet
                    # coordinates -> map GT poses back with first_gt_pose before the Sim(3) fit
                    rc = recon_metrics(c2w_est, first_gt_pose @ c2ws[:n], pts,
                                       load_gt_pointcloud(scene, args.gt_ply_dir)[0])
                    saved.update(rc)
                    metrics.update(rc)
                except Exception as e:  # never lose the pose metrics because of the reconstruction metric
                    print(f"[recon_metrics] {scene}: {type(e).__name__}: {e}", flush=True)
                if thinned:
                    saved.update(thinned)
                met_file.write_text(json.dumps(saved, indent=4))
                keep = ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms",
                        "ate_cc", "are_cc", "rpe_rot_cc", "rpe_trans_cc", "rc_acc_mean", "rc_comp_mean", "rc_f5", "rc_f10"]
                all_scenes_metrics["scenes"][scene] = {k: float(v) for k, v in metrics.items() if k in keep}
                print(f"[METRICS] {scene} {json.dumps(all_scenes_metrics['scenes'][scene])}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            mem.update({"status": "oom", "error": str(e).splitlines()[0],
                        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
            print(f"[MEM] {json.dumps(mem)}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "memory.json").write_text(json.dumps(mem, indent=2))
            torch.cuda.empty_cache()
        except Exception as e:  # keep going so one scene's failure doesn't hide the others
            import traceback

            traceback.print_exc()
            mem.setdefault("status", "error")
            mem["eval_error"] = f"{type(e).__name__}: {e}"
            print(f"[MEM] {json.dumps(mem)}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "memory.json").write_text(json.dumps(mem, indent=2))
            torch.cuda.empty_cache()
    if all_scenes_metrics["scenes"]:
        compute_average_metrics_and_save(all_scenes_metrics, args.output_path, args.input_frame)


if __name__ == "__main__":
    main()
