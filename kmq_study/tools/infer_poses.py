"""Run one model on ScanNet scenes at increasing frame counts; save predicted poses + memory.

Every model sees the same frames (tools/frames.py = FastVGGT's selection) and is run with its own
repo's official preprocessing and inference call. Per (scene, N) it saves
    <out_dir>/<label>/<scene>/N<n>.npz : w2c (first-frame relative), frame_ids, time_s, peak_*_gb
which tools/score_ate.py scores. Memory rows for the first scene are appended to --mem_csv in the
same format as mem_sweep.py, so plot_mem.py can draw them.

After the first OOM, --bisect binary-searches the largest frame count that fits (first scene only).

Run inside the model's own conda env, e.g.
  python tools/infer_poses.py --model pi3 --ckpt ckpt/pi3_model.safetensors --scenes S1 S2 --out_dir ...
"""
import argparse
import csv
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
from frames import Scene, first_frame_w2c, symlink_dir  # noqa: E402

GB = 2**30
# bf16 on Ampere+, fp16 on older GPUs (no efficient bf16 attention kernel there); as in mem_sweep.py
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
DEFAULT_FRAMES = [10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000]


def homogeneous(extr):
    """[S, 3, 4] -> [S, 4, 4]"""
    out = np.tile(np.eye(4), (extr.shape[0], 1, 1))
    out[:, :3, :4] = extr
    return out


# --------------------------------------------------------------------------------------------
# Adapters: load(ckpt) once, run(paths) -> world-to-camera [S, 4, 4] (any frame; normalized later)
# --------------------------------------------------------------------------------------------
class FastVGGTFamily:
    """VGGT* (merging off) and FastVGGT (merging from block 0), FastVGGT's eval_scannet.py setup."""

    def __init__(self, merging):
        sys.path.insert(0, os.path.join(BASE, "FastVGGT"))
        self.merging = merging

    def load(self, ckpt):
        from vggt.models.vggt import VGGT
        from vggt.utils.eval_utils import get_vgg_input_imgs, load_images_rgb
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.prep = lambda paths: get_vgg_input_imgs(np.stack(load_images_rgb(paths)))
        self.decode = pose_encoding_to_extri_intri
        self.model = VGGT(merging=self.merging, merge_ratio=0.9)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        self.model = self.model.cuda().eval().to(DTYPE)

    def run(self, paths):
        images, pw, ph = self.prep(paths)
        self.model.update_patch_dimensions(pw, ph)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            pred = self.model(images.cuda().to(DTYPE))
        extr, _ = self.decode(pred["pose_enc"], images.shape[-2:])
        return homogeneous(extr[0].float().cpu().numpy())


class OfficialVGGT:
    """Current facebookresearch/vggt (pip -e in the `vggt` env); fp32 weights + autocast, as its demo."""

    def load(self, ckpt):
        sys.path.insert(0, os.path.join(BASE, "vggt"))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.prep, self.decode = load_and_preprocess_images, pose_encoding_to_extri_intri
        self.model = VGGT()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.cuda().eval()

    def run(self, paths):
        images = self.prep([str(p) for p in paths]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            pred = self.model(images)
        extr, _ = self.decode(pred["pose_enc"], images.shape[-2:])
        return homogeneous(extr[0].float().cpu().numpy())


class QuantVGGTRepo:
    """QuantVGGT repo model (older VGGT release), unquantized or W4A4 with released params."""

    def __init__(self, quantize):
        self.quantize = quantize

    def load(self, ckpt):
        sys.path.insert(0, TOOLS)
        from mem_sweep import load_quantvggt_repo_model  # also puts FastVGGT's vggt back in place
        from vggt.utils.load_fn import load_and_preprocess_images  # FastVGGT copy, same crop code
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.prep, self.decode = load_and_preprocess_images, pose_encoding_to_extri_intri
        self.model = load_quantvggt_repo_model(ckpt, quantize=self.quantize)

    run = OfficialVGGT.run


class StreamVGGTAdapter:
    """StreamVGGT.inference(): one frame at a time with KV cache, as its demo_gradio.py."""

    def load(self, ckpt):
        sys.path.insert(0, os.path.join(BASE, "StreamVGGT", "src"))
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.prep, self.decode = load_and_preprocess_images, pose_encoding_to_extri_intri
        self.model = StreamVGGT()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.cuda().eval()

    def run(self, paths):
        images = self.prep([str(p) for p in paths]).cuda()
        frames = [{"img": images[i : i + 1]} for i in range(images.shape[0])]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            out = self.model.inference(frames)
        pose_enc = torch.stack([r["camera_pose"].squeeze(0) for r in out.ress])[None]
        extr, _ = self.decode(pose_enc, images.shape[-2:])
        return homogeneous(extr[0].float().cpu().numpy())


class Pi3Adapter:
    """pi3 (original, not Pi3X), as example.py: fp32 weights + autocast; camera_poses are c2w."""

    def load(self, ckpt):
        sys.path.insert(0, os.path.join(BASE, "Pi3"))
        from pi3.models.pi3 import Pi3
        from pi3.utils.basic import load_images_as_tensor
        from safetensors.torch import load_file

        self.prep = load_images_as_tensor
        self.model = Pi3()
        self.model.load_state_dict(load_file(ckpt))
        self.model = self.model.cuda().eval()

    def run(self, paths):
        d = symlink_dir(paths)
        try:
            imgs = self.prep(d, interval=1, verbose=False).cuda()
        finally:
            shutil.rmtree(d)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            res = self.model(imgs[None])
        return np.linalg.inv(res["camera_poses"][0].float().cpu().numpy().astype(np.float64))


class CUT3RAdapter:
    """CUT3R demo.py: prepare_input(size=512) + inference(); runs fp32 (autocast off at inference)."""

    def load(self, ckpt):
        root = os.path.join(BASE, "CUT3R")
        sys.path.insert(0, root)
        os.chdir(root)  # demo.py / add_ckpt_path resolve paths relative to the repo
        from add_ckpt_path import add_path_to_dust3r

        # demo.py passes a checkpoint inside CUT3R/src, which puts src/ (its `dust3r` package) on
        # sys.path; ours lives in ckpt/, so point it at src/ explicitly
        add_path_to_dust3r(os.path.join(root, "src", os.path.basename(ckpt)))
        from demo import prepare_input
        from src.dust3r.inference import inference
        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.utils.camera import pose_encoding_to_camera

        self.prepare_input, self.inference, self.decode = prepare_input, inference, pose_encoding_to_camera
        self.model = ARCroco3DStereo.from_pretrained(ckpt).cuda().eval()

    def run(self, paths):
        views = self.prepare_input([str(p) for p in paths], [True] * len(paths), size=512, revisit=1, update=True)
        with torch.no_grad():
            outputs, _ = self.inference(views, self.model, "cuda", verbose=False)
        c2w = torch.cat([self.decode(p["camera_pose"].clone()).cpu() for p in outputs["pred"]], 0)
        return np.linalg.inv(c2w.numpy().astype(np.float64))


class Spann3RAdapter:
    """Spann3R demo.py (online mode, 224 res, kf_every=1 so all N frames are used); fp32.
    Spann3R predicts pointmaps, not poses: poses come from PnP-RANSAC on them, as in its demo."""

    def load(self, ckpt):
        root = os.path.join(BASE, "spann3r")
        sys.path.insert(0, root)
        from dust3r.post_process import estimate_focal_knowing_depth
        from spann3r.datasets import Demo
        from spann3r.model import Spann3R

        self.Demo, self.focal = Demo, estimate_focal_knowing_depth
        dust3r_ckpt = os.path.join(os.path.dirname(ckpt), "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        self.model = Spann3R(dus3r_name=dust3r_ckpt, use_feat=False).cuda()
        self.model.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
        self.model.eval()

    def run(self, paths):
        import cv2

        d = symlink_dir(paths)
        try:
            dataset = self.Demo(ROOT=d, resolution=224, full_video=True, kf_every=1)
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        finally:
            shutil.rmtree(d)
        for view in batch:
            view["img"] = view["img"].cuda(non_blocking=True)
        with torch.no_grad():
            preds, _ = self.model.forward(batch)

        _, H, W, _ = preds[0]["pts3d"].shape
        pp = torch.tensor((W / 2, H / 2))
        focal = self.focal(preds[0]["pts3d"].cpu(), pp, focal_mode="weiszfeld")
        K = np.eye(3)
        K[0, 0] = K[1, 1] = float(focal)
        K[:2, 2] = pp.numpy()
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pix = np.stack((u, v), axis=-1).reshape(-1, 2).astype(np.float32)
        w2cs = []
        for j in range(len(batch)):
            pts = preds[j]["pts3d" if j == 0 else "pts3d_in_other_view"].detach().cpu().numpy()[0]
            _, rvec, tvec, _ = cv2.solvePnPRansac(pts.reshape(-1, 3).astype(np.float32), pix,
                                                  K.astype(np.float32), np.zeros(4, np.float32))
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(-1)
            w2cs.append(w2c)
        return np.stack(w2cs)


ADAPTERS = {
    "vggt": lambda: FastVGGTFamily(merging=None),  # VGGT*
    "fastvggt": lambda: FastVGGTFamily(merging=0),
    "vggt-official": lambda: QuantVGGTRepo(quantize=False),  # QuantVGGT base (older VGGT release)
    "quantvggt": lambda: QuantVGGTRepo(quantize=True),
    "vggt-current": OfficialVGGT,  # current facebookresearch/vggt
    "streamvggt": StreamVGGTAdapter,
    "pi3": Pi3Adapter,
    "cut3r": CUT3RAdapter,
    "spann3r": Spann3RAdapter,
}


# --------------------------------------------------------------------------------------------
def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class MemCSV:
    FIELDS = ["model", "num_frames", "status", "allocated_gb", "peak_allocated_gb",
              "peak_reserved_gb", "time_s", "note"]

    def __init__(self, path, label):
        self.label, new = label, not Path(path).exists()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def log(self, n, status, time_s, note=""):
        row = dict(model=self.label, num_frames=n, status=status,
                   allocated_gb=round(torch.cuda.memory_allocated() / GB, 3),
                   peak_allocated_gb=round(torch.cuda.max_memory_allocated() / GB, 3),
                   peak_reserved_gb=round(torch.cuda.max_memory_reserved() / GB, 3),
                   time_s=round(time_s, 3), note=note)
        self.w.writerow(row)
        self.f.flush()
        print(f"[MEM] {row}", flush=True)


def trial(adapter, scene, n, out_dir, mem=None):
    """Run n frames of a scene; save npz on success. Returns 'ok' | 'oom'."""
    ids, paths, _ = scene.select(n)
    free()
    start = time.time()
    try:
        w2c = adapter.run(paths)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    except Exception as e:
        if not is_oom(e):
            raise
        elapsed = time.time() - start
        if mem:
            mem.log(n, "oom", elapsed)
        print(f"[OOM] {scene.name} N={n} ({len(paths)} frames)", flush=True)
        free()
        return "oom"
    if mem:
        mem.log(n, "ok", elapsed, "" if len(paths) == n else f"{len(paths)} valid frames")
    if out_dir is not None:
        out = Path(out_dir) / scene.name / f"N{n}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, w2c=first_frame_w2c(w2c=w2c), frame_ids=np.array(ids), time_s=elapsed,
                 peak_allocated_gb=torch.cuda.max_memory_allocated() / GB,
                 peak_reserved_gb=torch.cuda.max_memory_reserved() / GB, num_frames=len(paths))
        print(f"[OK] {scene.name} N={n} ({len(paths)} frames) {elapsed:.1f}s "
              f"peak {torch.cuda.max_memory_allocated() / GB:.2f} GB", flush=True)
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(ADAPTERS))
    p.add_argument("--label", default=None, help="name in outputs (default: --model)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scenes", nargs="+", required=True, help="processed scene dirs; memory rows use the first")
    p.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    p.add_argument("--out_dir", required=True, help="npz root (results go to <out_dir>/<label>/...)")
    p.add_argument("--mem_csv", default=None)
    p.add_argument("--bisect", action="store_true", help="after the first OOM, find the exact max on scene 1")
    args = p.parse_args()
    label = args.label or args.model
    ckpt = os.path.abspath(args.ckpt)
    scenes = [Scene(os.path.abspath(s)) for s in args.scenes]  # absolute: CUT3R's adapter chdirs
    out_dir = Path(args.out_dir).absolute() / label
    mem = MemCSV(os.path.abspath(args.mem_csv), label) if args.mem_csv else None

    print(f"GPU {torch.cuda.get_device_name(0)}, dtype {DTYPE}, model {args.model} ({label})", flush=True)
    adapter = ADAPTERS[args.model]()
    adapter.load(ckpt)
    free()
    if mem:
        mem.log(0, "weights", 0.0, "model loaded")

    last_ok, first_oom = 0, None
    for n in args.frames:
        status = "ok"
        for i, scene in enumerate(scenes):
            prev = out_dir / scene.name / f"N{last_ok}.npz"
            if n > scene.num_valid and last_ok >= scene.num_valid and prev.exists():
                # this scene has fewer valid frames than n: same input as the previous N, reuse it
                shutil.copy(prev, prev.with_name(f"N{n}.npz"))
                print(f"[REUSE] {scene.name} N={n}: all {scene.num_valid} valid frames (= N{last_ok})", flush=True)
                continue
            status = trial(adapter, scene, n, out_dir, mem if i == 0 else None)
            if status == "oom":
                break
        if status == "oom":
            first_oom = n
            break
        last_ok = n

    while args.bisect and first_oom is not None and first_oom - last_ok > 1:
        mid = (last_ok + first_oom) // 2
        if trial(adapter, scenes[0], mid, None, mem) == "ok":
            last_ok = mid
        else:
            first_oom = mid
    print(f"[RESULT] {label}: max frames that fit = {last_ok}, "
          f"first OOM = {first_oom if first_oom else 'none (within tested range)'}", flush=True)


if __name__ == "__main__":
    main()
