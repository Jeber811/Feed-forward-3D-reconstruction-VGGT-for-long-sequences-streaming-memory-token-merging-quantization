"""Q: quantization. Real memory savings (unlike the released QuantVGGT, which fake-quantizes).

KV storage: keys/values stored in memory as int8 or fp8 (e4m3) with one fp16 scale per
(token, head), dequantized to fp16 just before attention. 64 values + 1 scale per (token, head):
130 bytes in fp16 -> 66 bytes, i.e. 0.51x.

Weights: nn.Linear layers of the aggregator replaced by weight-only int8 (one scale per output
channel), dequantized one layer at a time inside forward. Saves ~1 byte/param of the aggregator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = getattr(torch, "float8_e4m3fn", None)
QMAX = {"int8": 127.0, "fp8": 448.0}


class KVCodec:
    """pack(x) -> stored object; unpack(stored) -> fp16 tensor; nbytes(stored)."""

    def __init__(self, kind="fp16"):
        if kind not in ("fp16", "int8", "fp8"):
            raise ValueError(f"kv_dtype must be fp16|int8|fp8, got {kind}")
        if kind == "fp8" and FP8 is None:
            raise RuntimeError("this torch has no float8_e4m3fn")
        self.kind = kind

    def pack(self, x):
        """x: [B, H, N, D] fp16."""
        if self.kind == "fp16":
            return x.half()
        scale = x.abs().amax(-1, keepdim=True).float().clamp_min(1e-8) / QMAX[self.kind]
        y = x.float() / scale
        y = y.round().clamp_(-127, 127).to(torch.int8) if self.kind == "int8" else y.to(FP8)
        return (y, scale.half())

    def unpack(self, s):
        if self.kind == "fp16":
            return s
        y, scale = s
        return y.half() * scale

    @staticmethod
    def nbytes(s):
        return sum(t.numel() * t.element_size() for t in (s if isinstance(s, tuple) else (s,)))


class Int8Linear(nn.Module):
    """Weight-only int8 linear: W = q * scale (per output channel)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0
        self.register_buffer("qweight", (w / scale).round().clamp_(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.half())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.half(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        w = self.qweight.half() * self.scale
        b = self.bias
        if x.dtype != w.dtype and not torch.is_autocast_enabled():
            w, b = w.to(x.dtype), None if b is None else b.to(x.dtype)
        return F.linear(x, w, b)


def int8_weights_(module):
    """Replace every nn.Linear inside `module` in place. Returns how many were replaced."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
            n += 1
        else:
            n += int8_weights_(child)
    return n
