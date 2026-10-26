"""Plots how global attention grows with the number of frames, from mem_sweep.py's attention CSV.

Left:  global-attention key length (tokens every query attends to)
Right: size of one global-attention matrix if materialized (batch x heads x q_len x k_len)

VGGT / FastVGGT: all N frames at once -> q_len = k_len ~ N x tokens_per_frame -> matrix grows ~N^2.
StreamVGGT: one new frame per step -> q_len = tokens_per_frame, k_len = cache of all N frames -> ~N.

Usage: python plot_attn.py <attention_vs_frames.csv>     -> attention_vs_frames.png next to it
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "vggt": "VGGT (all frames at once)",
    "fastvggt": "FastVGGT (all frames, token merging)",
    "streamvggt": "StreamVGGT (per new frame, KV cache)",
}


def main(csv_path):
    csv_path = Path(csv_path)
    # model -> num_frames -> largest global-attention call at that frame count
    glob = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["attn"] != "global" or row["status"] != "ok":
                continue
            n, k_len, gb = int(row["num_frames"]), int(row["k_len"]), float(row["matrix_gb_if_materialized"])
            prev = glob[row["model"]].get(n)
            if prev is None or k_len > prev[0]:
                glob[row["model"]][n] = (k_len, gb, int(row["q_len"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    print(f"{'model':<11} {'frames':>6} {'q_len':>8} {'k_len':>8} {'matrix GB':>10}")
    for model in ["vggt", "fastvggt", "streamvggt"]:
        if model not in glob:
            continue
        pts = sorted(glob[model].items())
        xs = [n for n, _ in pts]
        ax1.plot(xs, [p[1][0] for p in pts], marker="o", markersize=3, label=LABELS[model])
        ax2.plot(xs, [p[1][1] for p in pts], marker="o", markersize=3, label=LABELS[model])
        for n, (k_len, gb, q_len) in pts:
            print(f"{model:<11} {n:>6} {q_len:>8} {k_len:>8} {gb:>10.3f}")

    ax1.set_xlabel("Number of frames loaded into GPU memory")
    ax1.set_ylabel("Global attention key length (tokens)")
    ax1.set_title("Tokens each query attends to")
    ax2.set_xlabel("Number of frames loaded into GPU memory")
    ax2.set_ylabel("Attention matrix size if materialized (GB)")
    ax2.set_title("Global attention matrix (batch x heads x q_len x k_len)")
    for ax in (ax1, ax2):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
