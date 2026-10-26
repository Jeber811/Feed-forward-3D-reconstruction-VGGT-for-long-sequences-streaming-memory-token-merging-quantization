"""Sanity check that QuantVGGT's W4A4 quantization is really applied before measuring memory.

  1. counts SmoothRotQuantizedLinear layers
  2. checks quantized weights lie on a 4-bit grid (<= 16 distinct values per output channel)
  3. compares predictions with official VGGT on a few frames (expect small, non-zero differences)

Usage (run from baseline/FastVGGT): python ../tools/check_quantvggt.py --ckpt ... --scene_dir ... [--frames 8]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mem_sweep import DTYPE, Frames, load_quantvggt_repo_model  # noqa: E402


def run(model, images):
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
        out = model(images.cuda())
    return {k: out[k].float().cpu() for k in ["pose_enc", "depth", "depth_conf"] if k in out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--frames", type=int, default=8)
    args = p.parse_args()

    frames = Frames(args.scene_dir)
    images = frames.load(frames.paths(args.frames))

    official = load_quantvggt_repo_model(args.ckpt, quantize=False)
    ref = run(official, images)
    del official
    torch.cuda.empty_cache()

    quant = load_quantvggt_repo_model(args.ckpt, quantize=True)
    qlayers = [(n, m) for n, m in quant.named_modules() if type(m).__name__ == "SmoothRotQuantizedLinear"]
    print(f"SmoothRotQuantizedLinear layers: {len(qlayers)}")
    if qlayers:
        for name, m in qlayers[:3]:
            w = m.linear.weight.data
            distinct = max(torch.unique(row).numel() for row in w[:64])
            print(f"  {name}: weight dtype {w.dtype}, max distinct values per channel (first 64) = {distinct}")
    new = run(quant, images)

    print(f"{args.frames} frames, dtype {DTYPE}: QuantVGGT W4A4 vs official VGGT")
    for k in ref:
        diff = (ref[k] - new[k]).abs()
        print(f"  {k:<11} mean |diff| {diff.mean().item():.3e}   mean |value| {ref[k].abs().mean().item():.3e}   "
              f"finite: {torch.isfinite(new[k]).all().item()}")


if __name__ == "__main__":
    main()
