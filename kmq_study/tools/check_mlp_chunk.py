"""Checks that token-chunked MLPs give the same VGGT predictions as the original blocks.

Usage (run from baseline/FastVGGT): python ../tools/check_mlp_chunk.py --ckpt ... --scene_dir ... [--frames 16]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from mem_sweep import DTYPE, Frames, chunk_block_mlps  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def run(model, images):
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
        out = model(images.cuda().to(DTYPE))
    keys = ["pose_enc", "depth", "depth_conf", "world_points"]
    return {k: out[k].float().cpu() for k in keys if k in out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--chunk", type=int, default=4096, help="small chunk so every block really is split")
    args = p.parse_args()

    frames = Frames(args.scene_dir)
    images = frames.load(frames.paths(args.frames))
    _, _, h, w = images.shape

    model = VGGT(merging=None)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    model = model.cuda().eval().to(DTYPE)
    model.update_patch_dimensions(w // 14, h // 14)

    ref = run(model, images)
    chunk_block_mlps(model.aggregator, args.chunk)
    new = run(model, images)

    print(f"{args.frames} frames, chunk {args.chunk} tokens, dtype {DTYPE}")
    for k in ref:
        diff = (ref[k] - new[k]).abs()
        scale = ref[k].abs().mean().item()
        print(f"  {k:<13} max |diff| {diff.max().item():.3e}   mean |diff| {diff.mean().item():.3e}   "
              f"mean |value| {scale:.3e}")


if __name__ == "__main__":
    main()
