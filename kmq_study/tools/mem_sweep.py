"""GPU memory vs. number of input images, until CUDA OOM.

VGGT / FastVGGT (all frames in one forward pass):
    tries increasing frame counts; after the first OOM, binary-searches between the last
    success and the failure down to --resolution frames.
StreamVGGT (one frame at a time with KV cache):
    a single streaming pass; logs memory after every frame until OOM or --max_frames.

Uses the same frame selection, preprocessing, checkpoint and bf16 autocast as eval_scannet.py.
Every measurement is appended to --out_csv immediately, so results survive a crash.

Usage (run from baseline/FastVGGT):
  python ../tools/mem_sweep.py --model vggt       --ckpt ../ckpt/model_tracker_fixed_e20.pt --scene_dir ... --out_csv ...
  python ../tools/mem_sweep.py --model fastvggt   --ckpt ../ckpt/model_tracker_fixed_e20.pt --scene_dir ... --out_csv ...
  python ../tools/mem_sweep.py --model streamvggt --ckpt ../ckpt/streamvggt_checkpoints.pth --scene_dir ... --out_csv ...
"""
import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

FASTVGGT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "FastVGGT"))
sys.path.insert(0, FASTVGGT_ROOT)
from vggt.utils.eval_utils import (  # noqa: E402
    build_frame_selection,
    get_sorted_image_paths,
    get_vgg_input_imgs,
    load_images_rgb,
    load_poses,
)

GB = 2**30
# bf16 on Ampere+, fp16 on older GPUs (no efficient bf16 attention kernel there); matches eval_scannet_mem.py
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
DEFAULT_FRAMES = list(range(50, 501, 50)) + list(range(600, 2001, 100))
FIELDS = ["model", "num_frames", "status", "allocated_gb", "peak_allocated_gb",
          "peak_reserved_gb", "time_s", "note"]


class Recorder:
    def __init__(self, path, model):
        self.path, self.model = Path(path), model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new:
            self.w.writeheader()

    def log(self, n, status, time_s, note=""):
        row = {
            "model": self.model,
            "num_frames": n,
            "status": status,
            "allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
            "time_s": round(time_s, 3),
            "note": note,
        }
        self.w.writerow(row)
        self.f.flush()
        print(f"[SWEEP] {row}", flush=True)


class AttnLogger:
    """Wraps F.scaled_dot_product_attention (used by both VGGT and StreamVGGT attention layers)
    and records the shape of every attention call: batch, heads, query length, key length.

    One CSV row per (num_frames, distinct shape), with the number of calls of that shape and the
    size the attention matrix would have if materialized (B * heads * q_len * k_len).

    With stub=True the attention itself is skipped (output = zeros), so shapes can be collected
    for frame counts that would not fit in memory.
    """

    FIELDS = ["model", "num_frames", "status", "attn", "calls", "batch", "heads", "q_len", "k_len",
              "head_dim", "tokens_per_frame", "matrix_elems", "matrix_gb_if_materialized"]

    def __init__(self, path, model, stub=False):
        self.path, self.model = Path(path), model
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()
        self.calls = {}
        self.tokens_per_frame = None
        orig = torch.nn.functional.scaled_dot_product_attention

        def wrapped(q, k, v, *a, **kw):
            key = (q.shape[0], q.shape[1], q.shape[2], k.shape[2], q.shape[3])
            self.calls[key] = self.calls.get(key, 0) + 1
            if stub:
                return torch.zeros_like(q)
            return orig(q, k, v, *a, **kw)

        torch.nn.functional.scaled_dot_product_attention = wrapped

    def reset(self):
        self.calls = {}

    def _label(self, b, q_len, k_len, n, streaming):
        if streaming:
            if q_len < self.tokens_per_frame // 2:
                return "camera_head"  # one pose token per frame, attending to cached pose tokens
            # Each new frame's tokens attend to the cached tokens of all frames so far
            return "global" if k_len > q_len else "frame"
        if b == n and n > 1:
            return "frame"  # each frame attends within itself
        if q_len == n:
            return "camera_head"  # one pose token per frame
        return "global"  # all frames' tokens attend to each other (fewer after FastVGGT merging)

    def log(self, n, status, streaming=False):
        for (b, h, q_len, k_len, d), calls in sorted(self.calls.items()):
            elems = b * h * q_len * k_len
            row = {
                "model": self.model, "num_frames": n, "status": status,
                "attn": self._label(b, q_len, k_len, n, streaming), "calls": calls,
                "batch": b, "heads": h, "q_len": q_len, "k_len": k_len, "head_dim": d,
                "tokens_per_frame": self.tokens_per_frame,
                "matrix_elems": elems,
                "matrix_gb_if_materialized": round(elems * torch.finfo(DTYPE).bits / 8 / GB, 3),
            }
            self.w.writerow(row)
            print(f"[ATTN] {row}", flush=True)
        self.f.flush()


class Frames:
    """Evenly spaced frames, picked and preprocessed exactly as eval_scannet.py does."""

    def __init__(self, scene_dir):
        scene_dir = Path(scene_dir)
        self.image_paths = get_sorted_image_paths(scene_dir / "color")
        _, _, self.pose_ids = load_poses(scene_dir / "pose")

    def paths(self, n):
        _, paths, _ = build_frame_selection(self.image_paths, self.pose_ids, n)
        return paths

    @staticmethod
    def load(paths, chunk=100):
        """[N, 3, H, W] in [0, 1]; loads full-res JPEGs in chunks to bound host RAM."""
        out = []
        for i in range(0, len(paths), chunk):
            imgs, _, _ = get_vgg_input_imgs(np.stack(load_images_rgb(paths[i : i + chunk])))
            out.append(imgs)
        return torch.cat(out)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def chunk_block_mlps(aggregator, chunk_tokens):
    """Token-chunked MLP for every frame/global block of the aggregator.

    norm2 -> MLP -> LayerScale -> residual is per-token, so running it over slices of tokens is
    mathematically identical to running it on the whole sequence; only the (tokens x 4C) hidden
    activation is limited to chunk_tokens rows at a time. The residual is added in place.
    Attention is unchanged. Inference only (in-place update).
    """
    import types

    def forward(self, x, pos=None, global_merging=None):
        norm1_output = self.norm1(x)
        attn_output = self.attn(norm1_output, pos=pos, global_merging=global_merging)
        del norm1_output
        x = x + self.ls1(attn_output)
        del attn_output
        flat = x.view(-1, x.shape[-1])
        for i in range(0, flat.shape[0], chunk_tokens):
            part = flat[i : i + chunk_tokens]
            part += self.ls2(self.mlp(self.norm2(part)))
        return x

    patched = 0
    for blocks in (aggregator.frame_blocks, aggregator.global_blocks):
        for blk in blocks:
            blk.forward = types.MethodType(forward, blk)
            patched += 1
    print(f"Token-chunked MLP ({chunk_tokens} tokens/chunk) in {patched} aggregator blocks", flush=True)


QUANTVGGT_ROOT = os.path.abspath(os.path.join(FASTVGGT_ROOT, os.pardir, "QuantVGGT"))


def load_quantvggt_repo_model(ckpt, quantize):
    """VGGT from the QuantVGGT repo, which is the official VGGT code.

    quantize=False: official VGGT.
    quantize=True:  QuantVGGT W4A4 with their released parameters, set up exactly as
                    evaluation/run_co3d.sh does (--dtype quarot_w4a4 --lwc --lac --resume_qs).
    Weights stay fp32 and inference runs under autocast, as in their run_co3d.py.

    Both repos ship a package named `vggt`, so FastVGGT's copy is moved out of sys.modules while
    QuantVGGT's is imported, then put back (the frame-loading helpers keep using FastVGGT's).
    QuantVGGT's vggt/ has no __init__.py (namespace package), and Python prefers a regular package
    anywhere on sys.path over a namespace package, so FastVGGT's root is also taken off sys.path.
    """
    is_vggt = lambda name: name == "vggt" or name.startswith("vggt.")  # noqa: E731
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if is_vggt(k)}
    saved_path = list(sys.path)
    sys.path[:] = [QUANTVGGT_ROOT, os.path.join(QUANTVGGT_ROOT, "evaluation")] + [
        p for p in saved_path if os.path.abspath(p or os.getcwd()) != FASTVGGT_ROOT]
    try:
        from vggt.models.vggt import VGGT as OfficialVGGT

        src = sys.modules[OfficialVGGT.__module__].__file__
        assert src.startswith(QUANTVGGT_ROOT), f"imported the wrong VGGT: {src}"
        print(f"Loaded VGGT class from {src}", flush=True)
        model = OfficialVGGT()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.cuda().eval()
        if quantize:
            from quarot.args_utils import get_config
            from quarot.utils import quarot_smooth_quant_model

            config = get_config()
            config.output_dir = os.path.join(QUANTVGGT_ROOT, "evaluation", "outputs")
            # exp_dir = outputs/w4a4/a44_quant_model_tracker_fixed_e20.pt_sym (their HF folder name)
            config.update_from_args(wbit=4, abit=4, not_smooth=False, not_rot=False, lwc=True, lac=True,
                                    rv=False, model_id=ckpt, exp_name="a44_quant")
            quarot_smooth_quant_model(config, model, None, wbit=4, abit=4, resume_qs=True,
                                      model_id=ckpt, exp_name="a44_quant")
            model = model.cuda()
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if is_vggt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    return model


def tokens_per_frame(model, h, w):
    # patch tokens + camera token + register tokens
    return (h // 14) * (w // 14) + getattr(model.aggregator, "patch_start_idx", 5)


def sweep_batch(args, rec, frames, attn):
    from vggt.models.vggt import VGGT

    # FastVGGT's VGGT (vggt/fastvggt) is cast to half precision, as in its eval_scannet.py;
    # the official/QuantVGGT code keeps fp32 weights and relies on autocast, as in its run_co3d.py.
    half_weights = args.model in ("vggt", "fastvggt")
    if half_weights:
        model = VGGT(merging=0 if args.model == "fastvggt" else None, merge_ratio=args.merge_ratio)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        model = model.cuda().eval().to(DTYPE)
    else:
        model = load_quantvggt_repo_model(args.ckpt, quantize=args.model == "quantvggt")
    if args.mlp_chunk:
        chunk_block_mlps(model.aggregator, args.mlp_chunk)
    free()
    if rec:
        rec.log(0, "weights", 0.0, "model loaded")

    def trial(n):
        paths = frames.paths(n)
        if len(paths) < n:
            return len(paths), "not_enough_frames"
        images = frames.load(paths)
        free()
        _, _, h, w = images.shape
        if hasattr(model, "update_patch_dimensions"):  # FastVGGT's VGGT only
            model.update_patch_dimensions(w // 14, h // 14)
        attn.tokens_per_frame = tokens_per_frame(model, h, w)
        attn.reset()
        out, start = None, time.time()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
                out = model(images.cuda().to(DTYPE) if half_weights else images.cuda())
            torch.cuda.synchronize()
            if rec:
                rec.log(n, "ok", time.time() - start)
            attn.log(n, "ok")
            return n, "ok"
        except torch.cuda.OutOfMemoryError:
            if rec:
                rec.log(n, "oom", time.time() - start)
            attn.log(n, "oom")  # attention calls made before the OOM
            return n, "oom"
        finally:
            del out
            free()

    if args.shapes_only:
        for n in args.frames:
            if trial(n)[1] == "not_enough_frames":
                break
        return

    last_ok, first_oom = 0, None
    for n in args.frames:
        _, status = trial(n)
        if status == "ok":
            last_ok = n
        elif status == "oom":
            first_oom = n
            break
        else:
            print(f"Scene has fewer than {n} frames, stopping")
            break

    # Binary search for the max frame count that fits
    while first_oom is not None and first_oom - last_ok > args.resolution:
        mid = (last_ok + first_oom) // 2
        _, status = trial(mid)
        if status == "ok":
            last_ok = mid
        else:
            first_oom = mid

    print(f"[RESULT] {args.model}: max frames that fit = {last_ok}, "
          f"first OOM = {first_oom if first_oom else 'none (within tested range)'}", flush=True)


def sweep_stream(args, rec, frames, attn):
    sys.path.insert(0, os.path.join(FASTVGGT_ROOT, os.pardir, "StreamVGGT", "src"))
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    free()
    rec.log(0, "weights", 0.0, "model loaded")

    paths = frames.paths(args.max_frames)
    n_total = len(paths)
    print(f"Streaming up to {n_total} frames", flush=True)

    # Same loop as StreamVGGT.inference(), with memory logged after every frame.
    # Outputs stay on the GPU, as they do in inference().
    agg = model.aggregator
    past_kv = [None] * agg.depth
    past_kv_cam = [None] * model.camera_head.trunk_depth
    outputs = []
    start = time.time()
    n = 0
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            for i in range(n_total):
                n = i + 1
                img = frames.load(paths[i : i + 1]).cuda()[None]  # [1, 1, 3, H, W]
                attn.tokens_per_frame = tokens_per_frame(model, img.shape[-2], img.shape[-1])
                attn.reset()
                tokens, patch_start_idx, past_kv = agg(img, past_key_values=past_kv, use_cache=True, past_frame_idx=i)
                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_kv_cam = model.camera_head(tokens, past_key_values_camera=past_kv_cam, use_cache=True)
                    depth, depth_conf = model.depth_head(tokens, images=img, patch_start_idx=patch_start_idx)
                    pts3d, pts3d_conf = model.point_head(tokens, images=img, patch_start_idx=patch_start_idx)
                outputs.append((pose_enc[-1][:, 0], depth[:, 0], depth_conf[:, 0], pts3d[:, 0], pts3d_conf[:, 0]))
                del tokens
                if n == 1 or n % args.log_every == 0 or n == n_total:
                    torch.cuda.synchronize()
                    rec.log(n, "ok", time.time() - start)
                    attn.log(n, "ok", streaming=True)  # attention shapes for this frame's step
        print(f"[RESULT] streamvggt: all {n_total} frames fit (no OOM within tested range)", flush=True)
    except torch.cuda.OutOfMemoryError:
        rec.log(n, "oom", time.time() - start)
        attn.log(n, "oom", streaming=True)
        print(f"[RESULT] streamvggt: max frames that fit = {n - 1}, first OOM = {n}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["vggt", "fastvggt", "streamvggt", "vggt-official", "quantvggt"],
                   help="vggt/fastvggt: FastVGGT's VGGT code; vggt-official/quantvggt: QuantVGGT repo "
                        "(official VGGT code, without / with their W4A4 quantization)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES, help="batch models: frame counts to try")
    p.add_argument("--resolution", type=int, default=10, help="batch models: binary-search precision (frames)")
    p.add_argument("--merge_ratio", type=float, default=0.9)
    p.add_argument("--max_frames", type=int, default=2000, help="streamvggt: stop after this many frames")
    p.add_argument("--log_every", type=int, default=10, help="streamvggt: log memory every N frames")
    p.add_argument("--attn_csv", default=None, help="default: attention_vs_frames.csv next to --out_csv")
    p.add_argument("--mlp_chunk", type=int, default=0,
                   help="batch models: run block MLPs over chunks of this many tokens (0 = off)")
    p.add_argument("--label", default=None, help="model name written to the CSVs (default: --model)")
    p.add_argument("--shapes_only", action="store_true",
                   help="batch models: only record attention shapes at --frames (attention stubbed out, "
                        "no memory rows, no OOM search)")
    args = p.parse_args()

    torch.manual_seed(33)
    print(f"GPU: {torch.cuda.get_device_name(0)}, dtype {DTYPE}, "
          f"{torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB total", flush=True)
    attn_csv = args.attn_csv or Path(args.out_csv).with_name("attention_vs_frames.csv")
    label = args.label or args.model
    attn = AttnLogger(attn_csv, label, stub=args.shapes_only)
    rec = None if args.shapes_only else Recorder(args.out_csv, label)
    frames = Frames(args.scene_dir)
    if args.model == "streamvggt":
        sweep_stream(args, rec, frames, attn)
    else:
        sweep_batch(args, rec, frames, attn)


if __name__ == "__main__":
    main()
