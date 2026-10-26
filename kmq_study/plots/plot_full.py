"""Full results: our stream / chunk variants vs. baselines on 6 ScanNet scenes, 24 GB GPU.

Figure 1  <out>_per_scene.png : ATE vs. frames, one panel per scene
Figure 2  <out>_summary.png   : (a) mean ATE over the 6 scenes vs. frames  (b) peak memory vs. frames
                                (c) throughput at 1000 frames
Table     <out>_table.md/.csv : mean ATE at 100 / 300 / 1000 / 2000 frames, max frames, peak GB, fps

A method's mean at N is shown only if all 6 scenes have a result at N (no partial averages).

Usage: python plot_full.py <variants ate_results.csv> <baselines ate_results.csv (2 scenes)>
                           <baselines ate_results.csv (new scenes)> <out_prefix>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ours: validated categorical slots in fixed order
OURS = {
    "chunk": ("chunk (VGGT* chunks)", "#2a78d6", "o"),
    "chunk+M": ("chunk+M (FastVGGT chunks)", "#eb6834", "s"),
    "chunk+K": ("chunk+K", "#1baf7a", "D"),
    "chunk+K+M+Q": ("chunk+K+M+Q", "#eda100", "^"),
    "stream+K+M": ("stream+K+M (best stream)", "#e87ba4", "v"),
    "stream": ("stream (bounded memory)", "#6250d6", "X"),
}
# baselines: gray, told apart by dash + marker
BASE = {
    "vggt": ("VGGT* (OOM ~390)", "-", "o"),
    "fastvggt": ("FastVGGT (OOM ~310)", "--", "s"),
    "pi3": ("π³ (OOM ~265)", "-.", "^"),
    "cut3r": ("CUT3R (OOM ~355)", ":", "D"),
    "spann3r": ("Spann3R", (0, (5, 1, 1, 1)), "v"),
    "streamvggt": ("StreamVGGT (OOM ~85)", (0, (1, 1)), "x"),
}
SCENES = {"scene0000_00": "scene0000_00 · apartment", "scene0013_02": "scene0013_02 · (1106 frames)",
          "scene0050_00": "scene0050_00 · living room", "scene0249_00": "scene0249_00 · conference",
          "scene0550_00": "scene0550_00 · bathroom", "scene0621_00": "scene0621_00 · classroom"}
GRAY, INK, MUTED, GRID = "#8f8e89", "#0b0b0b", "#52514e", "#e4e3df"
TABLE_N = [100, 300, 1000, 2000]


def load(var_csv, base_csv, new_csv):
    d = pd.concat([pd.read_csv(var_csv), pd.read_csv(base_csv), pd.read_csv(new_csv)])
    return d[d.model.isin(list(OURS) + list(BASE))].drop_duplicates(["model", "scene", "input_frame"])


def full_mean(d, col):
    """Per method and N: mean of col over scenes, only where all 6 scenes are present."""
    g = d.groupby(["model", "input_frame"])
    m = g[col].mean()[g.scene.nunique() == len(SCENES)]
    return m.reset_index()


def style_ax(ax):
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)


def draw(ax, d, y, x="input_frame"):
    for m, (lab, ls, mk) in BASE.items():
        s = d[d.model == m].sort_values(x)
        if len(s):
            ax.plot(s[x], s[y], color=GRAY, ls=ls, lw=1.3, marker=mk, ms=5, label=f"baseline: {lab}", zorder=1)
    for m, (lab, col, mk) in OURS.items():
        s = d[d.model == m].sort_values(x)
        if len(s):
            ax.plot(s[x], s[y], color=col, lw=2, marker=mk, ms=7, markeredgecolor="white",
                    markeredgewidth=1, label=lab, zorder=3)


def main(var_csv, base_csv, new_csv, out):
    d = load(var_csv, base_csv, new_csv)

    # ---- figure 1: per scene
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    for ax, (sc, title) in zip(axes.flat, SCENES.items()):
        draw(ax, d[d.scene == sc], "ate")
        style_ax(ax)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_ylabel("ATE (m, log) — lower is better", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Frames processed in one run (log)", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.86, 0.5), fontsize=9, frameon=False)
    fig.suptitle("ATE per scene — ours (color) vs. baselines (gray), one 24 GB Quadro RTX 6000",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    fig.savefig(f"{out}_per_scene.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: summary over scenes
    ate, mem = full_mean(d, "ate"), full_mean(d, "peak_allocated_gb")
    d = d.assign(fps=d.num_frames / d.time_s)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 6.5), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    draw(a1, ate, "ate")
    style_ax(a1)
    a1.set_yscale("log")
    a1.set(title="(a) Mean ATE over 6 scenes", xlabel="Frames processed in one run (log)",
           ylabel="ATE (m, log) — lower is better")
    draw(a2, mem, "peak_allocated_gb")
    style_ax(a2)
    a2.axhline(23.5, color=MUTED, ls="--", lw=1)
    a2.text(11, 22.8, "24 GB card limit", fontsize=8, color=MUTED)
    a2.set(title="(b) Peak GPU memory (mean over scenes)", xlabel="Frames processed in one run (log)",
           ylabel="GB allocated", ylim=(0, 25))
    # (c) fps at the largest N each method reaches on every scene, up to 1000
    rows = []
    for m in list(OURS) + list(BASE):
        s = d[(d.model == m) & (d.input_frame <= 1000)]
        if s.empty:
            continue
        n = s.groupby("input_frame").scene.nunique()
        n = n[n == len(SCENES)].index.max()
        if not pd.isna(n):
            rows.append((m, n, s[s.input_frame == n].fps.mean()))
    rows.sort(key=lambda r: r[2])
    for i, (m, n, f) in enumerate(rows):
        ours = m in OURS
        a3.barh(i, f, color=OURS[m][1] if ours else GRAY, height=0.7)
        a3.text(f + 0.08, i, f"{f:.1f}  (N={n})", va="center", fontsize=9, color=INK)
    a3.set_yticks(range(len(rows)), [(OURS.get(m) or BASE[m])[0] for m, _, _ in rows])
    a3.set(title="(c) Speed at the largest N ≤ 1000 that fits", xlabel="frames / s — higher is better",
           xlim=(0, max(r[2] for r in rows) * 1.35))
    a3.spines[["top", "right"]].set_visible(False)
    a3.grid(axis="x", color=GRID, lw=0.8)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Summary over 6 ScanNet scenes — one 24 GB GPU", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(f"{out}_summary.png", dpi=150)
    plt.close(fig)

    # ---- table
    rows = []
    for m in list(OURS) + list(BASE):
        s = d[d.model == m]
        if s.empty:
            continue
        lab = (OURS.get(m) or BASE[m])[0]
        r = {"Method": lab, "Ours": "✓" if m in OURS else ""}
        for n in TABLE_N:
            x = ate[(ate.model == m) & (ate.input_frame == n)]
            r[f"mean ATE @{n}"] = f"{x.ate.iloc[0]:.3f}" if len(x) else ("OOM" if s.input_frame.max() < n else "—")
        full = s.groupby("input_frame").scene.nunique()
        full = full[full == len(SCENES)]
        r["max N on all 6 scenes"] = int(full.index.max()) if len(full) else "—"
        r["peak GB"] = f"{s.peak_allocated_gb.max():.1f}"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}_table.csv", index=False, encoding="utf-8")
    with open(f"{out}_table.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")
    print(open(f"{out}_table.md", encoding="utf-8").read())


if __name__ == "__main__":
    main(*sys.argv[1:5])
