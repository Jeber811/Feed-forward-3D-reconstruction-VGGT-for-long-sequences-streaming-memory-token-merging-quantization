"""Find the largest number of frames each model can process on this GPU before CUDA OOM.

Same model wrappers, dtype, preprocessing and frame selection as eval_scannet_any.py (FastVGGT
`build_frame_selection` on one scene), but inference only (no metrics), so it can be repeated cheaply.
Schedule: ascending grid until the first OOM, then integer bisection between the last success and
the first failure, so `max_ok` is exact (first_oom = max_ok + 1).

Usage: python tools/max_frames_sweep.py --model pi3 --ckpt_path ... --data_dir ... \
         --scene scene0000_00 --out_csv results/sweep/pi3.csv [--grid 10 25 50 100 ...]
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scannet_any as H  # noqa: E402  (also puts FastVGGT on sys.path)

DEFAULT_GRID = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--scene", default="scene0000_00")
    ap.add_argument("--grid", type=int, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--merging", type=int, default=0)
    ap.add_argument("--merge_ratio", type=float, default=0.9)
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    gpu = torch.cuda.get_device_name(0)
    model = H.build_model(args, dtype)
    weights_gb = torch.cuda.memory_allocated() / 2**30

    scene_dir = args.data_dir / args.scene
    all_paths = H.get_sorted_image_paths(scene_dir / "color")
    _, _, avail = H.load_poses(scene_dir / "pose")
    n_valid = len(set(int(p.stem) for p in all_paths) & set(avail))
    grid = sorted({min(g, n_valid) for g in args.grid})
    cache = {}  # frame path -> preprocessed [3,H,W] tensor (CPU), so repeated trials don't re-decode JPEGs

    def frames(n):
        _, paths, _ = H.build_frame_selection(all_paths, avail, n)
        todo = [p for p in paths if p not in cache]
        if todo:
            imgs = H.load_images_rgb(todo)
            for p, im in zip(todo, imgs):
                t, pw, ph = H.get_vgg_input_imgs(np.asarray(im)[None])
                cache[p] = (t[0], pw, ph)
        x = torch.stack([cache[p][0] for p in paths])
        return x, cache[paths[0]][1], cache[paths[0]][2]

    rows = []
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: a requeued/preempted sweep reuses trials already in out_csv instead of re-running them.
    done = {}
    if args.out_csv.exists():
        prev = [r for r in csv.DictReader(open(args.out_csv)) if r.get("model") == args.model]
        rows.extend(prev)
        done = {int(r["num_frames"]): r["status"] == "ok" for r in prev}
        print(f"[RESUME] reusing {len(done)} recorded trials: {sorted(done)}", flush=True)

    def trial(n):
        if n in done:
            return done[n]
        x, pw, ph = frames(n)
        model.update_patch_dimensions(pw, ph)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"model": args.model, "gpu": gpu, "dtype": str(dtype).replace("torch.", ""),
               "scene": args.scene, "num_frames": n, "model_weights_gb": round(weights_gb, 3)}
        t0 = time.time()
        try:
            # identical to the start of FastVGGT's infer_vggt_and_reconstruct (the measured region)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                out = model(x.cuda().to(dtype), image_paths=None)
            torch.cuda.synchronize()
            row.update(status="ok", time_s=round(time.time() - t0, 2))
            del out
        except torch.cuda.OutOfMemoryError as e:
            row.update(status="oom", time_s="", error=str(e).splitlines()[0])
        row["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        del x
        gc.collect()
        torch.cuda.empty_cache()
        print("[SWEEP]", json.dumps(row), flush=True)
        rows.append(row)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        return row["status"] == "ok"

    last_ok, first_oom = 0, None
    for n in grid:
        if trial(n):
            last_ok = n
        else:
            first_oom = n
            break
    while first_oom is not None and first_oom - last_ok > 1:
        mid = (last_ok + first_oom) // 2
        if trial(mid):
            last_ok = mid
        else:
            first_oom = mid
    result = {"model": args.model, "gpu": gpu, "scene": args.scene, "max_ok": last_ok,
              "first_oom": first_oom, "scene_valid_frames": n_valid}
    print("[RESULT]", json.dumps(result), flush=True)
    args.out_csv.with_suffix(".result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
