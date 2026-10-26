"""Extract a ScanNet .sens file into color/ (jpg), depth/ (16-bit png, mm), pose/ (4x4 c2w txt), intrinsic/.

Usage: python extract_sens.py <scene.sens> <out_dir> [--no-depth]
"""
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image


def main(sens_path, out_dir, export_depth=True):
    for sub in ["color", "pose", "intrinsic"] + (["depth"] if export_depth else []):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    with open(sens_path, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        assert version == 4, f"unexpected .sens version {version}"
        name_len = struct.unpack("Q", f.read(8))[0]
        f.read(name_len)
        intr = {}
        for key in ["intrinsic_color", "extrinsic_color", "intrinsic_depth", "extrinsic_depth"]:
            intr[key] = np.asarray(struct.unpack("f" * 16, f.read(64)), dtype=np.float32).reshape(4, 4)
        color_comp, depth_comp = struct.unpack("ii", f.read(8))
        color_w, color_h, depth_w, depth_h = struct.unpack("IIII", f.read(16))
        depth_shift = struct.unpack("f", f.read(4))[0]
        num_frames = struct.unpack("Q", f.read(8))[0]
        print(f"{num_frames} frames, color {color_w}x{color_h}, depth {depth_w}x{depth_h}, shift {depth_shift}")
        assert color_comp == 2, "expected jpeg color compression"
        assert depth_comp == 1, "expected zlib depth compression"

        for key, mat in intr.items():
            np.savetxt(os.path.join(out_dir, "intrinsic", key + ".txt"), mat, fmt="%f")

        for i in range(num_frames):
            c2w = np.asarray(struct.unpack("f" * 16, f.read(64)), dtype=np.float32).reshape(4, 4)
            f.read(16)  # color / depth timestamps
            color_size, depth_size = struct.unpack("QQ", f.read(16))
            color_bytes = f.read(color_size)
            depth_bytes = f.read(depth_size)

            with open(os.path.join(out_dir, "color", f"{i}.jpg"), "wb") as out:
                out.write(color_bytes)
            np.savetxt(os.path.join(out_dir, "pose", f"{i}.txt"), c2w, fmt="%f")
            if export_depth:
                depth = np.frombuffer(zlib.decompress(depth_bytes), dtype=np.uint16).reshape(depth_h, depth_w)
                Image.fromarray(depth).save(os.path.join(out_dir, "depth", f"{i}.png"))
            if i % 500 == 0:
                print(f"  frame {i}/{num_frames}", flush=True)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], export_depth="--no-depth" not in sys.argv)
