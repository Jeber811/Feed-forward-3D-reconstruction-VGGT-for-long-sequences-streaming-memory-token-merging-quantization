"""Where does QuantVGGT's GPU memory go right after loading (before any frames)?

Splits torch.cuda.memory_allocated() into registered parameters, registered buffers, tensors stored
as plain module attributes (e.g. rotation matrices), and anything else still alive on the GPU.

Usage (run from baseline/FastVGGT): python ../tools/quantvggt_mem_breakdown.py --ckpt ...
"""
import argparse
import gc
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mem_sweep import GB, load_quantvggt_repo_model  # noqa: E402


def nbytes(t):
    return t.numel() * t.element_size()


def breakdown(model, label):
    gc.collect()
    total = torch.cuda.memory_allocated()
    seen = set()
    groups = defaultdict(int)

    def add(kind, t):
        if t.is_cuda and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            groups[f"{kind} ({str(t.dtype).replace('torch.', '')})"] += nbytes(t)

    for p in model.parameters():
        add("parameters", p)
    for b in model.buffers():
        add("buffers", b)
    attr_names = defaultdict(int)
    for m in model.modules():
        for name, v in vars(m).items():
            if isinstance(v, torch.Tensor) and not name.startswith("_"):
                if v.is_cuda and v.data_ptr() not in seen:
                    attr_names[name] += nbytes(v)
                add("plain tensor attributes", v)
    in_model = sum(groups.values())

    # Anything else alive on the GPU (e.g. loaded parameter dicts kept referenced)
    other = defaultdict(int)
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.data_ptr() not in seen:
                seen.add(obj.data_ptr())
                other[str(tuple(obj.shape))] += nbytes(obj)
        except Exception:
            pass

    print(f"\n== {label}: torch.cuda.memory_allocated = {total / GB:.2f} GB")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<40} {v / GB:7.2f} GB")
    for k, v in sorted(attr_names.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      attribute '{k}': {v / GB:.2f} GB")
    print(f"  {'other live GPU tensors':<40} {sum(other.values()) / GB:7.2f} GB")
    for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      shape {k}: {v / GB:.3f} GB")
    print(f"  {'unaccounted (allocator / freed-not-gc)':<40} "
          f"{(total - in_model - sum(other.values())) / GB:7.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    official = load_quantvggt_repo_model(args.ckpt, quantize=False)
    breakdown(official, "VGGT (QuantVGGT base), unquantized")
    del official
    gc.collect()
    torch.cuda.empty_cache()

    quant = load_quantvggt_repo_model(args.ckpt, quantize=True)
    breakdown(quant, "QuantVGGT W4A4")


if __name__ == "__main__":
    main()
