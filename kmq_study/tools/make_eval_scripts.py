"""Builds instrumented copies of FastVGGT/eval/eval_scannet.py (run from the baseline/ folder):

  eval_scannet_mem.py        VGGT / FastVGGT + per-scene GPU peak memory -> memory.json
  eval_scannet_streamvggt.py same, but with StreamVGGT (frame-by-frame, KV cache) as the model
"""
from pathlib import Path


def replace_once(src, old, new):
    assert src.count(old) == 1, f"expected exactly one match for: {old[:60]!r}"
    return src.replace(old, new)


src = Path("FastVGGT/eval/eval_scannet.py").read_text()

# --- Precision: bf16 only where the GPU supports it ---
# On pre-Ampere GPUs (e.g. Quadro RTX 6000 / Turing) there is no efficient bf16 SDPA kernel, so
# attention falls back to the math kernel and materializes the full attention matrix
# (16 GiB at 500 frames). fp16 uses the memory-efficient kernel there, as VGGT's own demo does.
src = replace_once(
    src,
    "    # Force use of bf16 data type\n    dtype = torch.bfloat16\n",
    "    # bf16 on Ampere+, fp16 on older GPUs (no efficient bf16 attention kernel there)\n"
    "    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n"
    '    print(f"Using dtype {dtype}")\n',
)
src = replace_once(src, "    model = model.to(torch.bfloat16)\n", "    model = model.to(dtype)\n")

# --- GPU memory instrumentation ---
src = replace_once(src, "import yaml\n", "import yaml\nimport json\n")
src = replace_once(
    src,
    "    model = model.to(dtype)\n",
    "    model = model.to(dtype)\n"
    "    model_mem_gb = torch.cuda.memory_allocated() / 2**30\n"
    '    print(f"[MEM] model weights: {model_mem_gb:.2f} GB")\n',
)
src = replace_once(
    src,
    "            # Inference + Reconstruction\n",
    "            torch.cuda.empty_cache()\n"
    "            torch.cuda.reset_peak_memory_stats()\n"
    "            # Inference + Reconstruction\n",
)
src = replace_once(
    src,
    '            print(f"Inference time: {inference_time_ms:.2f}ms")\n',
    '            print(f"Inference time: {inference_time_ms:.2f}ms")\n'
    "            mem = {\n"
    '                "scene": scene,\n'
    '                "num_frames": len(image_paths),\n'
    '                "model_weights_gb": round(model_mem_gb, 3),\n'
    '                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),\n'
    '                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),\n'
    '                "inference_time_ms": round(inference_time_ms, 1),\n'
    "            }\n"
    '            print(f"[MEM] {json.dumps(mem)}")\n'
    "            output_scene_dir.mkdir(parents=True, exist_ok=True)\n"
    '            (output_scene_dir / "memory.json").write_text(json.dumps(mem, indent=2))\n',
)
Path("FastVGGT/eval/eval_scannet_mem.py").write_text(src)

# --- StreamVGGT variant ---
start = src.index("    # Load VGGT model")
end_marker = "    model = model.to(dtype)\n"
end = src.index(end_marker) + len(end_marker)
src = src[:start] + (
    "    # Load StreamVGGT (causal attention + KV cache), wrapped to look like FastVGGT's VGGT\n"
    "    model = StreamVGGTWrapper(args.ckpt_path)\n"
) + src[end:]

wrapper = '''
STREAM_SRC = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "StreamVGGT", "src"))
if STREAM_SRC not in sys.path:
    sys.path.insert(0, STREAM_SRC)
from streamvggt.models.streamvggt import StreamVGGT


class StreamVGGTWrapper(torch.nn.Module):
    """Feeds frames one at a time through StreamVGGT and returns VGGT-style predictions."""

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = StreamVGGT()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)
        del ckpt
        self.model = self.model.cuda().eval()

    def update_patch_dimensions(self, patch_width, patch_height):
        pass

    @torch.no_grad()
    def forward(self, images, image_paths=None):
        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
        output = self.model.inference(frames)
        pose_enc = torch.stack([r["camera_pose"].squeeze(0) for r in output.ress])
        depth = torch.stack([r["depth"].squeeze(0) for r in output.ress])
        depth_conf = torch.stack([r["depth_conf"].squeeze(0) for r in output.ress])
        return {"pose_enc": pose_enc[None], "depth": depth[None], "depth_conf": depth_conf[None]}

'''
src = replace_once(src, '\nif __name__ == "__main__":', wrapper + '\nif __name__ == "__main__":')
Path("FastVGGT/eval/eval_scannet_streamvggt.py").write_text(src)
print("written eval_scannet_mem.py and eval_scannet_streamvggt.py")
