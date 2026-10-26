"""Run method variants on ScanNet scenes at increasing frame counts.

  python compact/run_variant.py --variant stream stream+K stream+K+M --scenes S1 S2
  python compact/run_variant.py --variant all --scenes S1 S2              # the 8 stream variants
  python compact/run_variant.py --variant all-chunk --scenes S1 S2        # the 8 chunk variants
  python compact/run_variant.py --variant stream+K+M --set budget_mb=4096 --scenes S1

Results, one folder per variant (scored by tools/score_ate.py like the baselines):
  <out_dir>/<variant>/config.json            the exact settings used
  <out_dir>/<variant>/<scene>/N<n>.npz       w2c, frame_ids, time_s, peak_*_gb, num_frames,
                                             + per frame: is_key, t, peak_gb, mem_mb, mem_frames, ...
  <out_dir>/<variant>/<scene>/N<n>_pts.npz   point maps (1/4 res) + confidence, for point-map metrics
Memory rows (first scene) are appended to --mem_csv in the mem_sweep format, labelled by variant.
Existing results are skipped (so a requeued job resumes); --redo reruns them.
Env: stream* variants run in the streamvggt env, chunk* variants in the fastvggt env -- one base
per call, since the two repos both ship a package named `vggt`.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(BASE, "tools")]
import variants  # noqa: E402
from frames import Scene, first_frame_w2c  # noqa: E402
from infer_poses import GB, MemCSV, free, is_oom  # noqa: E402

DEFAULT_FRAMES = [100, 300, 500, 1000, 2000]
CKPT = os.path.join(BASE, "ckpt", "streamvggt_checkpoints.pth")
CKPT_CHUNK = os.path.join(BASE, "ckpt", "model_tracker_fixed_e20.pt")  # VGGT* / FastVGGT


def run_one(model, scene, n, out_dir, mem, redo=False):
    ids, paths, _ = scene.select(n)
    d = Path(out_dir) / scene.name
    if not redo and (d / f"N{n}.npz").exists():
        print(f"[EXISTS] {scene.name} N={n}", flush=True)
        return "ok"
    free()
    start = time.time()
    try:
        r = model.run(paths)
        torch.cuda.synchronize()
    except Exception as e:
        if not is_oom(e):
            raise
        if mem:
            mem.log(n, "oom", time.time() - start)
        print(f"[OOM] {scene.name} N={n}", flush=True)
        free()
        return "oom"
    elapsed = time.time() - start
    if mem:
        mem.log(n, "ok", elapsed, "" if len(paths) == n else f"{len(paths)} valid frames")
    d.mkdir(parents=True, exist_ok=True)
    per_frame = {k: v for k, v in r.items() if k not in ("w2c", "pts", "conf", "evicted", "memory_frames_at_end")}
    np.savez(d / f"N{n}.npz", w2c=first_frame_w2c(w2c=r["w2c"]), frame_ids=np.array(ids), time_s=elapsed,
             peak_allocated_gb=torch.cuda.max_memory_allocated() / GB,
             peak_reserved_gb=torch.cuda.max_memory_reserved() / GB, num_frames=len(paths),
             evicted=r["evicted"], memory_frames_at_end=r["memory_frames_at_end"], **per_frame)
    np.savez(d / f"N{n}_pts.npz", pts=r["pts"], conf=r["conf"])
    print(f"[OK] {scene.name} N={n} ({len(paths)} frames) {elapsed:.1f}s ({len(paths) / elapsed:.2f} fps) "
          f"peak {torch.cuda.max_memory_allocated() / GB:.2f} GB | memory at end: {r['mem_frames'][-1]} frames "
          f"({r['mem_long'][-1]} long-term), {r['mem_mb'][-1]:.0f} MB | keyframes {int(r['is_key'].sum())}",
          flush=True)
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", nargs="+", required=True, help="names from compact/variants.py, or 'all'")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    p.add_argument("--out_dir", default=os.path.join(BASE, "logs", "variants"))
    p.add_argument("--mem_csv", default=None, help="default: <out_dir>/memory.csv")
    p.add_argument("--ckpt", default=None, help="default: StreamVGGT / VGGT* checkpoint by base")
    p.add_argument("--redo", action="store_true", help="rerun results that already exist")
    args = p.parse_args()
    names = {"all": variants.GRID, "all-chunk": variants.GRID_CHUNK}.get(args.variant[0], args.variant)
    bases = {variants.config(n)["base"].replace("stream-full", "stream") for n in names}
    if len(bases) > 1:
        p.error(f"one base per call (got {sorted(bases)}): stream* and chunk* need different envs")
    scenes = [Scene(os.path.abspath(s)) for s in args.scenes]
    mem_csv = args.mem_csv or os.path.join(args.out_dir, "memory.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    # imports the model repo; after arg parsing so --help is fast
    if bases == {"chunk"}:
        from chunk import ChunkVGGT as Model
        ckpt = args.ckpt or CKPT_CHUNK
    else:
        from stream import CompactStream as Model
        ckpt = args.ckpt or CKPT

    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    for name in names:
        cfg = variants.config(name, args.set)
        lab = variants.label(name, args.set)
        out_dir = Path(args.out_dir) / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
        print(f"\n===== {lab} | {time.strftime('%H:%M:%S')} =====\n{cfg}", flush=True)

        model = Model(ckpt, cfg)
        free()
        mem = MemCSV(mem_csv, lab)
        mem.log(0, "weights", 0.0, "model loaded")
        last_ok, first_oom = 0, None
        for n in args.frames:
            status = "ok"
            for i, scene in enumerate(scenes):
                prev = out_dir / scene.name / f"N{last_ok}.npz"
                if n > scene.num_valid and last_ok >= scene.num_valid and prev.exists():
                    # fewer valid frames than n: same input as the previous N (as infer_poses.py)
                    for suf in ("", "_pts"):
                        shutil.copy(prev.with_name(f"N{last_ok}{suf}.npz"), prev.with_name(f"N{n}{suf}.npz"))
                    print(f"[REUSE] {scene.name} N={n}: all {scene.num_valid} valid frames (= N{last_ok})", flush=True)
                    continue
                status = run_one(model, scene, n, out_dir, mem if i == 0 else None, args.redo)
                if status == "oom":
                    break
            if status == "oom":
                first_oom = n
                break
            last_ok = n
        print(f"[RESULT] {lab}: largest N that fits = {last_ok}, first OOM = {first_oom or 'none'}", flush=True)
        del model
        free()


if __name__ == "__main__":
    main()
