"""StreamVGGT with the compact memory: loads the released model and swaps its global-attention
KV cache for CompactMemory. Training-free: same weights, only what is remembered changes.

Per frame: aggregator (global attention reads memory) -> camera/depth/point heads -> pose and
depth decide the keyframe flag -> memory.commit. Outputs go to CPU every frame, so nothing but
the memory grows on the GPU. The camera head's own KV cache (1 token per frame per layer,
~64 KB/frame) is left as released.
"""
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

import keyframe as kf
from memory import CompactMemory
from quant import int8_weights_

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "StreamVGGT", "src"))
from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from streamvggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

DTYPE = torch.float16  # Turing; bf16 attention has no efficient kernel there
GB = 2**30
_NO_CACHE = (torch.empty(1, 1, 1, 0, 0), torch.empty(1, 1, 1, 0, 0))  # placeholder; see _install


def _attention_forward(memory, layer):
    """Replacement for streamvggt.layers.attention.Attention.forward of one global block."""

    def forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        q, k, v = q.to(DTYPE), k.to(DTYPE), v.to(DTYPE)
        stored = memory.read(layer)
        bias = None
        if stored is not None:
            mk, mv, mw = stored
            if mw is not None:  # merged tokens: +log(size) on their logits
                bias = torch.cat([mw, mw.new_zeros(N)]).view(1, 1, 1, -1).expand(B, self.num_heads, N, -1)
            k_all, v_all = torch.cat([mk, k], 2), torch.cat([mv, v], 2)
        else:
            k_all, v_all = k, v
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias)
        memory.stash(layer, k, v)
        out = self.proj(out.transpose(1, 2).reshape(B, N, C))
        return (out, _NO_CACHE) if use_cache else out

    return forward


class CompactStream:
    def __init__(self, ckpt, cfg):
        self.cfg = cfg
        model = StreamVGGT()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        agg = model.aggregator
        if cfg["weights"] == "int8":
            n = int8_weights_(agg)
            print(f"[Q] {n} aggregator linears -> int8", flush=True)
        elif cfg["weights"] != "fp16":
            raise ValueError(f"weights must be fp16|int8, got {cfg['weights']}")
        agg.half()  # int8 buffers are unaffected by .half()
        self.model = model.cuda().eval()
        self.n_layers = len(agg.global_blocks)
        self.n_special = agg.patch_start_idx

    def _install(self, memory):
        for i, blk in enumerate(self.model.aggregator.global_blocks):
            blk.attn.forward = types.MethodType(_attention_forward(memory, i), blk.attn)

    @torch.no_grad()
    def run(self, paths, pts_stride=4):
        """Returns dict: w2c [S,4,4], per-frame stats, downsampled point maps + confidence."""
        images = load_and_preprocess_images([str(p) for p in paths])  # CPU, [S,3,H,W]
        hw = images.shape[-2:]
        memory = CompactMemory(self.n_layers, self.cfg, self.n_special)
        self._install(memory)
        keyframes = kf.make(self.cfg)
        cam_kv = [None] * self.model.camera_head.trunk_depth
        out = {k: [] for k in ("w2c", "pts", "conf", "is_key", "t", "peak_gb", "mem_mb", "mem_frames",
                               "mem_long", "mem_tokens")}
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(images.shape[0]):
            img = images[i : i + 1].cuda()[None]  # [1,1,3,H,W]
            with torch.cuda.amp.autocast(dtype=DTYPE):
                # past_key_values only tells the aggregator "not the first frame" (picks the
                # non-first camera/register tokens); the real memory lives in `memory`.
                pkv = [None] * self.n_layers if i == 0 else [_NO_CACHE] * self.n_layers
                tokens, psi, _ = self.model.aggregator(img, past_key_values=pkv, use_cache=True, past_frame_idx=i)
            tokens = [t.float() for t in tokens]  # heads run in fp32, as released
            with torch.cuda.amp.autocast(enabled=False):
                pose_enc, cam_kv = self.model.camera_head(tokens, past_key_values_camera=cam_kv, use_cache=True)
                depth, _ = self.model.depth_head(tokens, images=img, patch_start_idx=psi)
                pts, conf = self.model.point_head(tokens, images=img, patch_start_idx=psi)
            del tokens
            extr, _ = pose_encoding_to_extri_intri(pose_enc[-1][:, :1], hw)
            w2c = np.eye(4)
            w2c[:3] = extr[0, 0].float().cpu().numpy()
            is_key = bool(keyframes(i, w2c, float(depth[0, 0].float().median())))
            memory.commit(i, is_key)

            s = pts_stride
            out["w2c"].append(w2c)
            out["pts"].append(pts[0, 0, ::s, ::s].half().cpu().numpy())
            out["conf"].append(conf[0, 0, ::s, ::s].half().cpu().numpy())
            out["is_key"].append(is_key)
            torch.cuda.synchronize()
            out["t"].append(time.time() - t0)
            out["peak_gb"].append(torch.cuda.max_memory_allocated() / GB)
            for k, v in memory.stats().items():
                out[k].append(v)
        out = {k: np.stack(v) if k in ("w2c", "pts", "conf") else np.array(v) for k, v in out.items()}
        out["evicted"] = memory.evicted
        out["memory_frames_at_end"] = np.array(memory.frames())
        return out
