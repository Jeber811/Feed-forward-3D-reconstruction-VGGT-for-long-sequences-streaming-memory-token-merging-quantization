"""Combined baseline figure + table: memory, pose error and max frames for every model.

  (a) peak GPU memory (allocated) vs. frames        <- memory_vs_frames.csv (sweeps + --bisect runs)
  (b) ATE vs. frames, mean of the scenes            <- ate_summary.csv (score_ate.py)
  (c) max frames before OOM                         <- memory_vs_frames.csv
and baseline_table.csv / .md with the same columns as the teammate's table.

Usage: python plot_results.py <memory_vs_frames.csv> <ate_dir> <out_prefix>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPU_GB = 24.0
# label in CSVs -> (display name, color, marker); order = legend / bar order
MODELS = {
    "vggt-current": ("VGGT (official)", "#2b7bd6", "o"),
    "vggt": ("VGGT*", "#ef6c33", "s"),
    "vggt+mlpchunk": ("VGGT* + chunked MLP", "#ef6c33", "s"),
    "fastvggt": ("FastVGGT", "#1aa67a", "D"),
    "pi3": ("π³", "#e8a400", "^"),
    "streamvggt": ("StreamVGGT", "#e0609c", "v"),
    "streamvggt+expandable": ("StreamVGGT + expandable", "#e0609c", "v"),
    "vggt-official": ("VGGT (QuantVGGT base)", "#8c6bb1", "P"),
    "quantvggt": ("QuantVGGT W4A4", "#c0392b", "X"),
    "cut3r": ("CUT3R", "#2e8b2e", "+"),
    "spann3r": ("Spann3R", "#4b3ba3", "o"),
}
STREAMING = ("streamvggt", "cut3r", "spann3r")  # frame-by-frame: max = first OOM - 1


def read_memory(path):
    ok, oom = defaultdict(dict), defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            n = int(r["num_frames"])
            if r["status"] == "ok" and n > 0:
                ok[r["model"]][n] = float(r["peak_allocated_gb"])  # later rows win
            elif r["status"] == "oom":
                oom[r["model"]].append(n)
    limits = {}
    for m in set(ok) | set(oom):
        first_oom = min(oom[m]) if oom[m] else None
        max_ok = max((n for n in ok[m] if first_oom is None or n < first_oom), default=0)
        if first_oom and m.split("+")[0] in STREAMING:
            max_ok = first_oom - 1
        limits[m] = (max_ok, first_oom)
    return ok, limits


def read_ate(ate_dir):
    ate = defaultdict(dict)
    path = Path(ate_dir) / "ate_summary.csv"
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                ate[r["model"]][int(r["input_frame"])] = float(r["ate"])
    return ate


def main(mem_csv, ate_dir, out_prefix):
    ok, limits = read_memory(mem_csv)
    ate = read_ate(ate_dir)
    models = [m for m in MODELS if m in limits or m in ate]

    fig, (ax_m, ax_a, ax_b) = plt.subplots(1, 3, figsize=(21, 6.5), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    for m in models:
        name, color, marker = MODELS[m]
        ls = "--" if "+" in m else "-"
        hollow = dict(markerfacecolor="none") if "+" in m else {}
        if m in ok:
            pts = sorted(ok[m].items())
            ax_m.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, marker=marker,
                      ms=5, lw=1.6, label=name, **hollow)
            if limits[m][1]:
                ax_m.plot(limits[m][1], GPU_GB - 0.5, marker="x", color=color, ms=11, mew=2.5)
        if m in ate and ate[m]:
            pts = sorted(ate[m].items())
            ax_a.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, marker=marker,
                      ms=5, lw=1.6, label=name, **hollow)
            ax_a.annotate(str(pts[-1][0]), pts[-1], textcoords="offset points", xytext=(6, 4), fontsize=8,
                          color="#444")

    ax_m.axhline(GPU_GB - 0.5, color="gray", ls="--", lw=1)
    ax_m.text(1.1, GPU_GB - 0.2, "24 GB card limit (23.5 GiB usable)", fontsize=9, color="#555")
    ax_m.set(xscale="log", xlim=(1, None), ylim=(0, GPU_GB + 2),
             xlabel="Frames processed in one run (log scale)", ylabel="Peak GPU memory allocated (GB) — lower is better",
             title="(a) Memory vs. frames  (× = first frame count that runs out of memory)")
    ax_a.set(xscale="log", yscale="log", xlabel="Frames processed in one run (log scale)",
             ylabel="ATE (m, log scale; mean of 2 scenes) — lower is better",
             title="(b) Pose error vs. frames, up to each model's last fitting count")

    bars = [(m, limits[m]) for m in models if m in limits]
    bars.sort(key=lambda b: b[1][0])
    y = range(len(bars))
    ax_b.barh(y, [b[1][0] for b in bars], color=[MODELS[b[0]][1] for b in bars],
              hatch=None, edgecolor="white")
    for i, (m, (mx, oom)) in enumerate(bars):
        ax_b.text(mx * 1.08, i, f"{mx:,}" + (f"  (OOM at {oom:,})" if oom else "  (no OOM)"), va="center", fontsize=9)
    ax_b.set_yticks(list(y), [MODELS[b[0]][0] for b in bars])
    ax_b.axvline(500, color="gray", ls="--", lw=1)
    ax_b.text(520, -0.6, "500-frame goal", fontsize=8, color="#555")
    ax_b.set(xscale="log", xlim=(1, max(b[1][0] for b in bars) * 12),
             xlabel="Max frames that fit on 24 GB (log scale) — higher is better",
             title="(c) Max frames before OOM")
    for ax in (ax_m, ax_a, ax_b):
        ax.grid(alpha=0.3, which="major")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax_m.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=10)
    fig.suptitle("Baselines on one 24 GB GPU (Quadro RTX 6000) — ScanNet, FastVGGT frame selection", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(f"{out_prefix}.png", dpi=150)
    print("Saved", f"{out_prefix}.png")

    # Table, same columns as the teammate's
    def ate_at(m, n):
        v = ate.get(m, {}).get(n)
        return f"{v:.3f}" if v is not None else ("out of memory" if limits.get(m, (0, None))[0] < n else "—")

    rows = []
    for m, (mx, oom) in sorted(((m, limits[m]) for m in models if m in limits), key=lambda b: b[1][0]):
        largest = max(ate.get(m, {}) or [0])
        rows.append({"Method": MODELS[m][0], "Max frames": mx, "Runs out of memory at": oom or "—",
                     "Peak GB at max": f"{ok[m].get(mx, max(ok[m].values(), default=0)):.1f}" if ok.get(m) else "—",
                     "ATE at largest tested N (m)": f"{ate[m][largest]:.3f} (N={largest})" if largest else "—",
                     "ATE at 300": ate_at(m, 300), "ATE at 1000": ate_at(m, 1000)})
    with open(f"{out_prefix}_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{out_prefix}_table.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
    print(open(f"{out_prefix}_table.md", encoding="utf-8").read())


if __name__ == "__main__":
    main(*sys.argv[1:4])
