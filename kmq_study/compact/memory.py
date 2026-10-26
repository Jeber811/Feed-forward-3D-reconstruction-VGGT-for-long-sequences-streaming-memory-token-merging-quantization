"""Bounded key/value memory for StreamVGGT's global attention (the base of every stream variant).

StreamVGGT caches the keys/values of EVERY past frame in each of its 24 global-attention layers,
so memory grows linearly with the frame count until OOM (84 frames on 24 GB). Here the current
frame attends to

    anchor (frame 0)  +  long-term frames  +  last `recent` frames  +  itself

and the stored keys/values are kept under `budget_mb`. Each frame is written once, after it has
been processed (commit), when its pose is known:
    K decides whether it may stay once it leaves the recent window (keyframe.py)
    M shrinks its tokens before storing                                (merge.py)
    Q shrinks the bytes per token                                      (quant.py)
Over budget, the oldest long-term frame is evicted first.

Keys are stored after RoPE. StreamVGGT's RoPE only encodes the 2D patch position, which is the
same in every frame, so this gives the same attention as re-applying RoPE to the cache, and it
lets merged tokens keep their own (averaged) positions.
"""
from dataclasses import dataclass, field

import torch

from merge import merge_frame
from quant import KVCodec

MB = 2**20


@dataclass
class Entry:
    frame: int
    kind: str                     # anchor | recent | long
    keyframe: bool
    layers: list = field(default_factory=list)  # per layer: (packed k, packed v, logw | None)
    nbytes: int = 0
    tokens: int = 0


class CompactMemory:
    def __init__(self, n_layers, cfg, n_special):
        self.n_layers, self.n_special = n_layers, n_special
        self.budget = cfg["budget_mb"] * MB
        self.recent = cfg["recent"]
        self.merge_ratio = cfg["merge_ratio"]
        self.codec = KVCodec(cfg["kv_dtype"])
        self.raw = KVCodec("fp16")
        self.compress_anchor = cfg["compress_anchor"]
        self.entries = []
        self.pending = [None] * n_layers
        self.evicted = 0

    # ---- used inside attention -----------------------------------------------------------
    def read(self, layer):
        """Stored keys, values [1, H, N, D] fp16 and logit bias [N] (None if nothing is merged)."""
        if not self.entries:
            return None
        ks, vs, ws = [], [], []
        for e in self.entries:
            k, v, w = e.layers[layer]
            codec = self.raw if (e.kind == "anchor" and not self.compress_anchor) else self.codec
            ks.append(codec.unpack(k))
            vs.append(codec.unpack(v))
            ws.append(w if w is not None else torch.zeros(ks[-1].shape[2], device=ks[-1].device, dtype=torch.half))
        merged = any(e.layers[layer][2] is not None for e in self.entries)
        return torch.cat(ks, 2), torch.cat(vs, 2), (torch.cat(ws) if merged else None)

    def stash(self, layer, k, v):
        self.pending[layer] = (k.half(), v.half())

    # ---- called once per frame, after the frame is processed -----------------------------
    def commit(self, frame, keyframe):
        anchor = not self.entries
        e = Entry(frame, "anchor" if anchor else "recent", keyframe)
        squeeze = not anchor or self.compress_anchor
        codec = self.codec if squeeze else self.raw
        for k, v in self.pending:
            w = None
            if squeeze and self.merge_ratio > 0:
                k, v, w = merge_frame(k, v, self.merge_ratio, self.n_special)
            pk, pv = codec.pack(k), codec.pack(v)
            e.layers.append((pk, pv, w))
            e.nbytes += codec.nbytes(pk) + codec.nbytes(pv) + (0 if w is None else w.numel() * 2)
        e.tokens = e.layers[0][0][0].shape[2] if isinstance(e.layers[0][0], tuple) else e.layers[0][0].shape[2]
        self.pending = [None] * self.n_layers
        self.entries.append(e)

        recent = [x for x in self.entries if x.kind == "recent"]
        if len(recent) > self.recent:          # oldest recent frame leaves the window
            old = recent[0]
            if old.keyframe:
                old.kind = "long"
            else:
                self.entries.remove(old)
                self.evicted += 1
        while self.nbytes > self.budget:        # over budget: evict the oldest long-term frame
            longs = [x for x in self.entries if x.kind == "long"]
            if not longs:
                break  # anchor + recent alone exceed the budget; nothing evictable
            self.entries.remove(longs[0])
            self.evicted += 1

    # ---- stats ----------------------------------------------------------------------------
    @property
    def nbytes(self):
        return sum(e.nbytes for e in self.entries)

    def stats(self):
        kinds = [e.kind for e in self.entries]
        return dict(mem_mb=self.nbytes / MB, mem_frames=len(self.entries),
                    mem_long=kinds.count("long"), mem_tokens=sum(e.tokens for e in self.entries))

    def frames(self):
        return [e.frame for e in self.entries]
