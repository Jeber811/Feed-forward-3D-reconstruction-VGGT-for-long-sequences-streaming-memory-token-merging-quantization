"""Variant grid figure + table: ATE per scene and peak memory vs. frames, with baseline references.

  rows    : (top) one technique on the bounded memory   (bottom) combinations
  columns : ATE scene0000_00 | ATE scene0013_02 | peak GPU memory (mean of scenes)

Usage: python plot_variants.py <variants ate_results.csv> <baseline ate_results.csv> <out_prefix>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed categorical order (validated palette); colour follows the variant in both rows
VARIANTS = {
    "stream": ("stream (bounded memory only)", "#2a78d6", "o"),
    "stream+K": ("+K keyframes", "#eb6834", "s"),
    "stream+M": ("+M token merging", "#1baf7a", "D"),
    "stream+Q": ("+Q int8", "#eda100", "^"),
    "stream+K+M": ("+K+M", "#e87ba4", "v"),
    "stream+K+Q": ("+K+Q", "#008300", "P"),
    "stream+M+Q": ("+M+Q", "#6250d6", "X"),
    "stream+K+M+Q": ("+K+M+Q", "#e34948", "*"),
}
ROWS = [("One technique on the bounded memory", ["stream", "stream+K", "stream+M", "stream+Q"]),
        ("Combinations", ["stream", "stream+K+M", "stream+K+Q", "stream+M+Q", "stream+K+M+Q"])]
# baselines drawn as recessive gray references
REFS = {"streamvggt": ("StreamVGGT (OOM 85)", "-", "x"), "spann3r": ("Spann3R", "--", None),
        "cut3r": ("CUT3R (OOM 356)", ":", None), "vggt": ("VGGT* (OOM 393)", "-.", None)}
SCENES = ["scene0000_00", "scene0013_02"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(var_csv, base_csv, out):
    v, b = pd.read_csv(var_csv), pd.read_csv(base_csv)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    for r, (title, names) in enumerate(ROWS):
        for c, what in enumerate(SCENES + ["mem"]):
            ax = axes[r, c]
            for m, (lab, ls, mk) in REFS.items():
                d = b[b.model == m]
                d = d.groupby("input_frame").mean(numeric_only=True) if what == "mem" else d[d.scene == what].set_index("input_frame")
                d = d.sort_index()
                y = d["peak_allocated_gb"] if what == "mem" else d["ate"]
                ax.plot(d.index, y, color="#9a9994", ls=ls, lw=1.4, marker=mk, ms=7, zorder=1,
                        label=f"baseline: {lab}")
            for m in names:
                lab, col, mk = VARIANTS[m]
                d = v[v.model == m]
                d = d.groupby("input_frame").mean(numeric_only=True) if what == "mem" else d[d.scene == what].set_index("input_frame")
                d = d.sort_index()
                if d.empty:
                    continue
                y = d["peak_allocated_gb"] if what == "mem" else d["ate"]
                base = m == "stream" and r == 1
                ax.plot(d.index, y, color=col, lw=1.2 if base else 2, marker=mk, ms=8 if mk != "*" else 11,
                        markeredgecolor="white", markeredgewidth=1.2, alpha=0.55 if base else 1,
                        label=lab, zorder=3)
            ax.set_xscale("log")
            if what == "mem":
                ax.axhline(23.5, color=MUTED, ls="--", lw=1)
                ax.text(11, 22.9, "24 GB card limit", fontsize=8, color=MUTED)
                ax.set_ylim(0, 25)
                ax.set_ylabel("Peak GPU memory allocated (GB)", color=INK)
            else:
                ax.set_ylim(0, 2.8 if what == "scene0000_00" else 0.72)
                ax.set_ylabel("ATE (m) — lower is better", color=INK)
            if r == 0:
                ax.set_title({"scene0000_00": "ATE — scene0000_00 (large room)",
                              "scene0013_02": "ATE — scene0013_02 (1106 valid frames)",
                              "mem": "Peak GPU memory (mean of 2 scenes)"}[what], fontsize=12, color=INK)
            if r == 1:
                ax.set_xlabel("Frames processed in one run (log scale)", color=INK)
            ax.grid(color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=MUTED)
        axes[r, 0].text(-0.2, 0.5, title, transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=12, color=INK, weight="bold")
        # one legend per row, outside the plots so it never covers a line
        h, l = axes[r, 0].get_legend_handles_labels()
        axes[r, 2].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.suptitle("StreamVGGT with a bounded 8 GB memory + K / M / Q — 24 GB Quadro RTX 6000, ScanNet "
                 "(gray = baselines)", fontsize=14, color=INK)
    fig.tight_layout(rect=(0.02, 0, 0.86, 0.96))
    fig.savefig(f"{out}.png", dpi=150)

    # table: ATE per scene at 300 / 1000 / 2000 (all frames for scene0013_02), peak memory, speed
    rows = []
    for m, (lab, _, _) in VARIANTS.items():
        d = v[v.model == m]
        if d.empty:
            continue
        row = {"Variant": m}
        for sc in SCENES:
            for n in (300, 1000, 2000):
                x = d[(d.scene == sc) & (d.input_frame == n)]
                row[f"ATE {sc[5:9]} N={n}"] = f"{x.ate.iloc[0]:.3f}" if len(x) else "—"
        big = d[d.input_frame == d.input_frame.max()]
        row["Peak GB"] = f"{d.peak_allocated_gb.max():.1f}"
        row["fps @ max N"] = f"{(big.num_frames / big.time_s).mean():.2f}"
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}_table.csv", index=False)
    with open(f"{out}_table.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r_ in t.iterrows():
            f.write("| " + " | ".join(str(x) for x in r_) + " |\n")
    print(open(f"{out}_table.md", encoding="utf-8").read())


if __name__ == "__main__":
    main(*sys.argv[1:4])
