"""Plots peak GPU memory vs. number of frames from mem_sweep.py's CSV.

Left:  peak allocated (what the tensors need)
Right: peak reserved (what PyTorch holds, incl. fragmentation; this is what hits the GPU limit)
Variants such as "vggt+mlpchunk" use their base model's color with a dashed line.

Usage: python plot_mem.py <memory_vs_frames.csv>     -> memory_vs_frames.png next to it
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPU_LIMIT_GB = 24.0
BASE_COLORS = {"vggt": "tab:blue", "fastvggt": "tab:orange", "streamvggt": "tab:green",
               "vggt-official": "tab:purple", "quantvggt": "tab:red"}
# Distinct markers so curves that coincide (e.g. VGGT and FastVGGT peak reserved) stay visible
BASE_MARKERS = {"vggt": "o", "fastvggt": "s", "streamvggt": "^", "vggt-official": "D", "quantvggt": "X"}
# Legend names. "vggt-official" was run from the QuantVGGT repo, which forks an older VGGT release
# (caches all 24 layers); it is not the current facebookresearch/vggt code.
DISPLAY = {"vggt": "VGGT*", "fastvggt": "FastVGGT", "streamvggt": "StreamVGGT",
           "vggt-official": "VGGT (QuantVGGT base)", "quantvggt": "QuantVGGT W4A4"}


def style(model):
    base, _, variant = model.partition("+")
    return BASE_COLORS.get(base, "tab:gray"), ("--" if variant else "-"), BASE_MARKERS.get(base, "o")


def main(csv_path):
    csv_path = Path(csv_path)
    ok, oom = defaultdict(list), defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n = int(row["num_frames"])
            if row["status"] == "ok" and n > 0:
                ok[row["model"]].append((n, float(row["peak_allocated_gb"]), float(row["peak_reserved_gb"])))
            elif row["status"] == "oom":
                oom[row["model"]].append(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    order = list(BASE_COLORS)  # vggt, fastvggt, streamvggt; each base followed by its variants
    models = sorted(set(ok) | set(oom),
                    key=lambda m: (order.index(m.split("+")[0]) if m.split("+")[0] in order else len(order), m))
    for i, model in enumerate(models):
        color, ls, marker = style(model)
        pts = sorted(ok.get(model, []))
        max_ok = max((p[0] for p in pts), default=0)
        first_oom = min(oom[model]) if oom.get(model) else None
        if first_oom and model.startswith("streamvggt"):
            max_ok = first_oom - 1  # streaming adds one frame at a time; memory is only logged every N
        base, _, variant = model.partition("+")
        name = DISPLAY.get(base, base) + (f"+{variant}" if variant else "")
        label = f"{name} (max {max_ok}" + (f", OOM {first_oom})" if first_oom else ")")
        for ax, col in zip(axes, (1, 2)):
            # Later models are drawn on top with hollow markers so coinciding curves both show
            ax.plot([p[0] for p in pts], [p[col] for p in pts], color=color, linestyle=ls,
                    marker=marker, markersize=6 if i else 4, markerfacecolor="none" if i else color,
                    linewidth=1.5, alpha=0.9, zorder=2 + i, label=label)
            if first_oom:
                ax.axvline(first_oom, color=color, linestyle=":", alpha=0.6)
        print(f"{model}: max frames that fit = {max_ok}, first OOM = {first_oom or 'none in tested range'}")

    for ax, title in zip(axes, ("Peak allocated (what the tensors need)",
                                "Peak reserved (incl. fragmentation; hits the limit)")):
        ax.axhline(GPU_LIMIT_GB, color="gray", linestyle="--", label=f"{GPU_LIMIT_GB:.0f} GB GPU limit")
        ax.set_ylim(0, GPU_LIMIT_GB + 1)  # headroom so the limit line isn't hidden by the frame
        ax.set_xlim(left=0)
        ax.set_xlabel("Number of frames loaded into GPU memory")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Peak GPU memory (GB)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("GPU memory vs. frames on a 24 GB GPU (dotted line = first OOM, dashed = variant)")
    fig.tight_layout()
    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print("Saved", out)

    # Same figure with a log x-axis, so models that fit only a few frames are readable
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(1, None)
        ax.set_xlabel("Number of frames loaded into GPU memory (log scale)")
    out_log = csv_path.with_name(csv_path.stem + "_logx.png")
    fig.savefig(out_log, dpi=150)
    print("Saved", out_log)


if __name__ == "__main__":
    main(sys.argv[1])
