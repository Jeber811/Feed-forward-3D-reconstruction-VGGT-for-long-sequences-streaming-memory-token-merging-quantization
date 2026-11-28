"""Figures for Jake's B1 slides. Every plotted number is read from protocol result JSON (means over scenes with
status ok), except the dev-scene tuning/smoke points (smoke tiers on Newton), which are copied from those runs' logs
and labelled as such. Run on a compute node:  ~/.venvs/slides/bin/python make_figs.py <results_root> [<m_results_root>]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path(sys.argv[1])                                  # repo results/ (rtx6000 + v100_cap24)
RM = Path(sys.argv[2]) if len(sys.argv) > 2 else None  # extra v100_cap24 root holding M grid runs (optional)
OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
NAVY, GREY, BAND = "#1B2A3A", "#6B6B6B", "#F1F4F7"
BLUE, ORANGE, GREEN, PINK, SKY, YELLOW, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#7B5EA7"
BUD = {6: GREEN, 10: BLUE, 16: ORANGE}
plt.rcParams.update({"font.size": 13, "text.color": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY,
                     "ytick.color": NAVY, "axes.edgecolor": "#C9CED4", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "figure.dpi": 100})
SUMMARY = {}


def run_mean(tier, name, n, root=None):
    d = (root or R) / tier / name / f"input_frame_{n}"
    rows = []
    for sd in sorted(p for p in d.iterdir() if p.is_dir()):
        mem = json.loads((sd / "memory.json").read_text())
        if mem.get("status") != "ok" or not (sd / "metrics.json").exists():
            continue
        met = json.loads((sd / "metrics.json").read_text())
        rows.append((mem, met))
    if not rows:
        raise SystemExit(f"no ok scenes in {d}")
    avg = lambda k: sum(m[k] for _, m in rows) / len(rows)
    r = {"n_scenes": len(rows), "ate_cc_cm": 100 * avg("ate_cc"), "acc_cm": 100 * avg("rc_acc_med"),
         "f5": avg("rc_f5"), "peak": max(mm["peak_allocated_gb"] for mm, _ in rows),
         "s_per_frame": sum(mm["inference_time_ms"] / 1000 / mm["num_frames"] for mm, _ in rows) / len(rows)}
    SUMMARY[f"{tier}/{name}/N{n}"] = {k: round(v, 4) for k, v in r.items()}
    return r


def title(fig, t, sub=None):
    fig.text(0.01, 0.97, t, fontsize=19, fontweight="bold", va="top")
    if sub:
        fig.text(0.01, 0.885, sub, fontsize=12, color=GREY, va="top")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1. protocol parity: golden peak memory RTX 6000 vs V100 (24 GB cap) + B2 cross-check -------------------------------
def fig_parity():
    rows = [("π³ (full)", 7.836, 7.836), ("π³ chunk (B1)", 6.412, 6.410), ("StreamVGGT", 13.834, 13.836),
            ("VGGT*", 4.993, 4.993), ("VGGT* chunk (B2), 3 scenes N=500", 6.882, 6.882)]
    fig, ax = plt.subplots(figsize=(9.3, 3.6))
    for i, (lab, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=BAND, zorder=0)
        ax.scatter(a, y, s=150, color=BLUE, zorder=3, label="Quadro RTX 6000 (24 GB)" if i == 0 else None)
        ax.scatter(b, y, s=150, marker="x", lw=2.5, color=ORANGE, zorder=4,
                   label="Tesla V100 capped to 24 GB" if i == 0 else None)
        ax.text(15.3, y, f"{a:.3f} / {b:.3f}", va="center", fontsize=12)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0, 15)
    ax.set_xlabel("Peak GPU memory (GB), golden reference config")
    ax.text(15.3, len(rows) - 0.35, "RTX / V100", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(axis="x", color="#E3E6EA")
    fig.text(0.01, -0.04, "All accuracy metrics also within golden tolerance on both GPUs. Source: golden jobs 395507 (RTX) / "
             "1165694 (V100); cross-check 1165695.", fontsize=10, color=GREY)
    save(fig, "f1_parity")


# 2. B1 schematic -------------------------------------------------------------------------------------------------------
def fig_b1_diagram():
    fig, ax = plt.subplots(figsize=(9.3, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
    ax.text(0, 40, "Video frames  →", fontsize=12, color=GREY, va="center")
    for i in range(40):
        c = "#D9DEE4"
        if 22 <= i < 25: c = SKY          # overlap = most recent frames of previous chunk
        if i in (3, 9, 15, 20): c = YELLOW  # keyframes = history anchors
        if i >= 25: c = BLUE              # new frames
        ax.add_patch(FancyBboxPatch((1 + i * 2.45, 31), 2.0, 5, boxstyle="round,pad=0.05,rounding_size=0.3", fc=c, ec="none"))
    ax.annotate("", xy=(99, 28.5), xytext=(1, 28.5), arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.text(30, 25.8, "already reconstructed (outputs kept in CPU RAM)", fontsize=11, color=GREY, ha="center")
    # current chunk box
    ax.add_patch(FancyBboxPatch((2, 3), 62, 16, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#EEF5FB", ec=BLUE, lw=1.6))
    ax.text(33, 16.3, "One π³ pass on the GPU: chunk of C frames", ha="center", fontsize=12.5, fontweight="bold")
    items = [(YELLOW, "K keyframes (history)"), (SKY, "overlap"), (BLUE, "new frames")]
    for j, (c, t) in enumerate(items):
        x0 = (6, 34, 47)[j]
        ax.add_patch(FancyBboxPatch((x0, 8.7), 2.4, 3.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc=c, ec="none"))
        ax.text(x0 + 3.4, 10.5, t, va="center", fontsize=11)
    ax.text(33, 5, "GPU memory set by C, not by sequence length", ha="center", fontsize=11, color=NAVY, style="italic")
    ax.add_patch(FancyArrowPatch((8, 30.5), (14, 19.6), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.2))
    ax.add_patch(FancyArrowPatch((64.5, 11), (72, 11), arrowstyle="-|>", mutation_scale=16, color=NAVY, lw=1.6))
    ax.add_patch(FancyBboxPatch((72.5, 3), 26, 16, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#FDF1EA", ec=ORANGE, lw=1.6))
    ax.text(85.5, 14.5, "Sim(3) stitch", ha="center", fontsize=12.5, fontweight="bold")
    ax.text(85.5, 9.5, "align chunk to the global\nframe on shared (overlap +\nkeyframe) points; robust IRLS", ha="center",
            va="center", fontsize=9.8)
    save(fig, "f2_b1_diagram")


# 3. Q on π³ ---------------------------------------------------------------------------------------------------------------
def fig_q():
    fig, ax = plt.subplots(figsize=(9.3, 3.7))
    groups, fp, i8 = ["Peak GPU memory", "Model weights on GPU"], [6.41, 3.572], [2.90, 1.476]
    x = range(len(groups))
    b1 = ax.bar([i - 0.19 for i in x], fp, 0.36, color=ORANGE, label="As released (FP32 weights + FP16 autocast)")
    b2 = ax.bar([i + 0.19 for i in x], i8, 0.36, color=BLUE, label="Q: INT8 weight-only (encoder + decoder)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08, f"{r.get_height():.2f} GB", ha="center", fontsize=12)
    ax.set_xticks(list(x), groups)
    ax.set_ylabel("GB"); ax.set_ylim(0, 7.6)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.text(1.62, 4.9, "Accuracy unchanged\nATE-cc 2.61 → 2.63 cm\nF@5cm 0.881 → 0.882", fontsize=12, va="top",
            bbox=dict(fc=BAND, ec="none", boxstyle="round,pad=0.5"))
    ax.set_xlim(-0.6, 2.4)
    fig.text(0.01, -0.04, "Golden config: π³ chunk C=25, 60 frames, scene0013_02; Newton V100 capped 24 GB (smoke tier, "
             "Nov 5).", fontsize=10, color=GREY)
    save(fig, "f3_q")


# 4. chunk-size tuning (dev scenes, smoke tier) ----------------------------------------------------------------------------
TUNE = {"FP16": [(25, 6.411), (30, 6.616), (110, 9.931), (120, 10.345), (240, 15.302)],
        "INT8 (Q)": [(45, 3.726), (60, 4.349), (80, 5.18), (150, 8.088), (180, 9.335), (280, 13.458), (300, 14.276)],
        "FP16 + M": [(100, 9.763), (105, 9.983), (220, 15.022), (230, 15.46), (240, 15.896)],
        "INT8 + M": [(70, 4.939), (75, 5.159), (80, 5.38), (170, 9.342), (175, 9.562), (180, 9.78), (280, 14.147),
                     (290, 14.279), (300, 14.657)]}
PICK = {"FP16": [110, 240], "INT8 (Q)": [80, 180, 300], "FP16 + M": [105, 240], "INT8 + M": [80, 180, 300]}


def fig_tune():
    fig, ax = plt.subplots(figsize=(9.3, 3.9))
    sty = {"FP16": (ORANGE, "o"), "INT8 (Q)": (BLUE, "o"), "FP16 + M": (ORANGE, "^"), "INT8 + M": (BLUE, "^")}
    for k, pts in TUNE.items():
        c, m = sty[k]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, m, color=c, ms=7, alpha=0.55 if "M" in k else 1, label=k, mfc="none" if "M" in k else c)
        for x, y in pts:
            if x in PICK[k]:
                ax.plot(x, y, m, color=c, ms=13, mfc="none", mew=2)
    for b in (6, 10, 16):
        ax.axhline(b, color=BUD[b], ls="--", lw=1.4)
        ax.text(358, b + 0.25, f"{b} GB budget", va="bottom", ha="right", color=BUD[b], fontsize=11.5, fontweight="bold")
    ax.annotate("FP16 cannot reach 6 GB:\nsmallest chunk (25) = 6.41 GB", xy=(25, 6.41), xytext=(45, 11.5), fontsize=11,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xlabel("Chunk size C (frames per π³ pass)"); ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_xlim(0, 360); ax.set_ylim(2, 17.5)
    ax.legend(frameon=False, ncol=4, fontsize=11, loc="upper left")
    ax.grid(color="#E3E6EA")
    fig.text(0.01, -0.04, "Circled = chosen (largest chunk under budget). Dev scenes scene0111_00/0188_00 (M: 0111_00), "
             "500 frames; peak is identical across scenes per config.", fontsize=10, color=GREY)
    save(fig, "f4_tune")


# 5. Tier B grid (10 scenes, N=500, v100_cap24) -------------------------------------------------------------------------------
GRID = {6: ["Q", "KQ", "MQ", "KMQ"], 10: ["none", "K", "Q", "KQ", "M", "KM", "MQ", "KMQ"],
        16: ["none", "K", "Q", "KQ", "M", "KM", "MQ", "KMQ"]}
LAB = {"none": "chunk", "K": "+K", "Q": "+Q", "KQ": "+K+Q", "M": "+M", "KM": "+K+M", "MQ": "+M+Q", "KMQ": "+K+M+Q"}


def grid_runs():
    out = {}
    for b, vs in GRID.items():
        for v in vs:
            nm = f"jake_v04_grid_b1_b{b}_{v}"
            for root in (R, RM):
                if root and (root / "v100_cap24" / nm).is_dir():
                    out[(b, v)] = run_mean("v100_cap24", nm, 500, root)
                    break
    return out


def fig_grid():
    g = grid_runs()
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 4.1), sharey=True, gridspec_kw={"width_ratios": [4, 8, 8]})
    for ax, b in zip(axes, (6, 10, 16)):
        vs = [v for v in GRID[b] if (b, v) in g]
        vals = [g[(b, v)]["ate_cc_cm"] for v in vs]
        best = min(vals)
        cols = [(PINK if "M" in v else BUD[b]) for v in vs]
        bars = ax.bar(range(len(vs)), vals, color=cols, alpha=0.9)
        for r, v in zip(bars, vs):
            r.set_edgecolor(NAVY if g[(b, v)]["ate_cc_cm"] == best else "none"); r.set_linewidth(2)
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08, f"{r.get_height():.2f}", ha="center", fontsize=9.5)
            ax.text(r.get_x() + r.get_width() / 2, 0.25, f"{g[(b, v)]['f5']:.2f}"[1:], ha="center", fontsize=8.5, color="white",
                    fontweight="bold")
        ax.set_xticks(range(len(vs)), [LAB[v] for v in vs], rotation=40, ha="right", fontsize=10.5)
        ax.set_title(f"{b} GB budget" + ("\n(FP16 does not fit)" if b == 6 else ""), fontsize=12.5, color=BUD[b],
                     fontweight="bold")
        ax.grid(axis="y", color="#E3E6EA")
    axes[0].set_ylabel("ATE-cc (cm)  ↓")
    fig.text(0.01, -0.07, "Bar label = ATE-cc; white label = F@5cm (↑). Outlined = lowest ATE-cc at that budget. Pink = with M. "
             "10 grid scenes, 500 frames, V100 capped 24 GB.", fontsize=10, color=GREY)
    fig.tight_layout()
    save(fig, "f5_grid")
    return g


# 6. Tier A (full cohorts) ----------------------------------------------------------------------------------------------------
NS = [300, 500, 1000, 2000]


def fig_tiera():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.3, 3.8))
    for b in (6, 10, 16):
        rs = [run_mean("v100_cap24", f"jake_v04_b1_b{b}_KQ_N{n}", n) for n in NS]
        lab = f"{b} GB  (peak {rs[0]['peak']:.2f} GB at every N)"
        a1.plot(range(4), [r["ate_cc_cm"] for r in rs], "o-", color=BUD[b], lw=2.2, label=lab)
        a2.plot(range(4), [r["f5"] for r in rs], "o-", color=BUD[b], lw=2.2)
    for a, yl in ((a1, "ATE-cc (cm)  ↓"), (a2, "F@5cm  ↑")):
        a.set_xticks(range(4), [f"{n}\n({s})" for n, s in zip(NS, (50, 50, 40, 13))], fontsize=11)
        a.set_xlabel("Frames per sequence (scenes)"); a.set_ylabel(yl); a.grid(color="#E3E6EA")
    a1.legend(frameon=False, fontsize=10.5, loc="upper left")
    a1.set_ylim(3.5, 7.2); a2.set_ylim(0.6, 0.82)
    fig.text(0.01, -0.06, "B1 = π³ chunk + K + Q. Full ScanNet cohorts; the 2,000-frame cohort is a different 13-scene set. "
             "V100 capped 24 GB.", fontsize=10, color=GREY)
    fig.tight_layout()
    save(fig, "f6_tiera")


# 7. controls + simple baseline ------------------------------------------------------------------------------------------------
def fig_controls():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.3, 3.5))
    kf = {6: 15, 10: 110, 16: 260}
    for b in (6, 10, 16):
        rb = [run_mean("v100_cap24", f"jake_v04_b1_b{b}_KQ_N{n}", n) for n in NS]
        rk = [run_mean("v100_cap24", f"jake_v04_ctrl_kfinterp{kf[b]}_N{n}", n) for n in NS]
        for a, k in ((a1, "ate_cc_cm"), (a2, "f5")):
            a.plot(range(4), [r[k] for r in rb], "o-", color=BUD[b], lw=2.2, label=f"B1 {b} GB" if a is a1 else None)
            a.plot(range(4), [r[k] for r in rk], "s--", color=BUD[b], lw=1.6, alpha=0.75, mfc="white",
                   label=f"Keyframe baseline {b} GB" if a is a1 else None)
    for m, c, lab in (("ttt3r_offload", GREY, "TTT3R (≈5–6 GB)"), ("cut3r_offload", "#A9A9A9", "CUT3R (≈4–6 GB)")):
        rr = [run_mean("rtx6000", f"jake_v04_ctrl_{m}_N{n}", n) for n in NS]
        a1.plot(range(4), [r["ate_cc_cm"] for r in rr], "^:", color=c, lw=1.5, label=lab)
        a2.plot(range(4), [r["f5"] for r in rr], "^:", color=c, lw=1.5)
    full = {"π³ full (17.8 GB)": ("jake_v04_ctrl_pi3_N300", PURPLE), "VGGT* full (16.3 GB)": ("jake_v04_ctrl_vggt_star_N300", YELLOW)}
    for lab, (nm, c) in full.items():
        r = run_mean("v100_cap24", nm, 300)
        a1.plot(0, r["ate_cc_cm"], "*", color=c, ms=15, label=lab); a2.plot(0, r["f5"], "*", color=c, ms=15)
    a1.set_yscale("log"); a1.set_ylabel("ATE-cc (cm, log)  ↓"); a2.set_ylabel("F@5cm  ↑")
    for a in (a1, a2):
        a.set_xticks(range(4), [str(n) for n in NS]); a.set_xlabel("Frames per sequence"); a.grid(color="#E3E6EA")
    a1.set_ylim(3, 120)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.06, "Keyframe baseline = π³ on 15/110/260 evenly spaced frames, poses of other frames interpolated "
             "(same peak budget). Full models OOM at 500. Means over 50/50/40/13 scenes.", fontsize=9.5, color=GREY)
    save(fig, "f7_controls")


# 8. speed (RTX 6000 timing runs) ----------------------------------------------------------------------------------------------
def fig_speed():
    pts = []  # (label, peak, s/frame, color, marker, label offset in points)
    pairs = [(6, ("Q", "KQ"), "INT8 (+Q, +K+Q)", (8, -14)), (10, ("none", "K"), "FP16 (chunk, +K)", (8, -12)),
             (10, ("Q", "KQ"), "INT8 (+Q, +K+Q)", (-150, 6)), (16, ("none", "K"), "FP16 (chunk, +K)", (8, -14)),
             (16, ("Q", "KQ"), "INT8 (+Q, +K+Q)", (-150, 4))]
    for b, vs, lab, off in pairs:  # each pair has identical memory and near-identical speed: one point, one label
        rs = [run_mean("rtx6000", f"jake_v04_b1_b{b}_{v}_N500_timing", 500) for v in vs]
        pts.append((f"B1 {b} GB {lab}", rs[0]["peak"], sum(r["s_per_frame"] for r in rs) / 2, BUD[b], "o", off))
    ctrl = [("π³ full @300", "pi3_N300", 300, PURPLE, (8, 6)), ("VGGT* full @300", "vggt_star_N300", 300, YELLOW, (8, 4)),
            ("FastVGGT @300", "fastvggt_N300", 300, "#8C6D31", (-40, 10)), ("TTT3R", "ttt3r_offload_N500", 500, GREY, (8, -12)),
            ("Keyframe baseline (110)", "kfinterp110_N2000", 2000, "#444444", (8, 4))]
    for lab, nm, n, c, off in ctrl:
        r = run_mean("rtx6000", f"jake_v04_ctrl_{nm}_timing", n)
        pts.append((lab, r["peak"], r["s_per_frame"], c, "D", off))
    fig, ax = plt.subplots(figsize=(9.3, 4.0))
    for lab, x, y, c, m, off in pts:
        ax.plot(x, y, m, color=c, ms=9)
        ax.annotate(f"{lab}  {y:.2f}s", (x, y), xytext=off, textcoords="offset points", fontsize=9.5,
                    color=c if c != GREY else NAVY)
    ax.set_yscale("log"); ax.set_xlim(0, 24); ax.set_xlabel("Peak GPU memory (GB)"); ax.set_ylabel("Seconds per frame (log)  ↓")
    ax.grid(color="#E3E6EA", which="both", lw=0.6)
    fig.text(0.01, -0.05, "Quadro RTX 6000, scene0000_00, median of 3 repeats (B1 at 500 frames). INT8 runs are slower per frame: "
             "per-layer dequantization + larger chunks.", fontsize=10, color=GREY)
    save(fig, "f8_speed")




# 9. M trade-off (grid, V100 capped 24 GB; speed ratios only, absolute speed is not the RTX 6000 timing protocol) ----------
def fig_m(g):
    fig, ax = plt.subplots(figsize=(9.3, 4.0))
    pairs = [(6, "KQ", "KMQ"), (10, "KQ", "KMQ"), (10, "K", "KM"), (16, "KQ", "KMQ"), (16, "K", "KM")]
    for b, a, m in pairs:
        if (b, a) not in g or (b, m) not in g:
            continue
        p, q = g[(b, a)], g[(b, m)]
        ax.annotate("", (q["s_per_frame"], q["ate_cc_cm"]), (p["s_per_frame"], p["ate_cc_cm"]),
                    arrowprops=dict(arrowstyle="->", color=BUD[b], lw=1.6, alpha=0.8))
        ax.plot(p["s_per_frame"], p["ate_cc_cm"], "o", color=BUD[b], ms=9)
        ax.plot(q["s_per_frame"], q["ate_cc_cm"], "^", color=PINK, ms=10, mec=BUD[b], mew=1.5)
        ax.annotate(f"{b} GB {LAB[a]}", (p["s_per_frame"], p["ate_cc_cm"]), xytext=(-6, -16) if a == "K" else (6, 6), ha="right" if a == "K" else "left",
                    textcoords="offset points", fontsize=9.5, color=BUD[b])
        ax.annotate(f"{b} GB {LAB[m]}  {p['s_per_frame'] / q['s_per_frame']:.1f}× faster", (q["s_per_frame"], q["ate_cc_cm"]), ha="right",
                    xytext=(-9, -4), textcoords="offset points", fontsize=9.5, color=BUD[b])
    ax.set_xscale("log"); ax.set_xlabel("Seconds per frame (log, V100 grid runs)  ←"); ax.set_ylabel("ATE-cc (cm)  ↓")
    ax.set_xlim(0.035, 1.6)
    ax.grid(color="#E3E6EA", which="both", lw=0.6)
    fig.text(0.01, -0.05, "Circle = without M, pink triangle = with M, same budget (M needs 0.05–0.6 GB more peak; 10 GB FP16 chunk "
             "110→105). 10 grid scenes, 500 frames.", fontsize=10, color=GREY)
    save(fig, "f9_m")


if __name__ == "__main__":
    fig_parity(); fig_b1_diagram(); fig_q(); fig_tune(); g = fig_grid(); fig_tiera(); fig_controls(); fig_speed(); fig_m(g)
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1))
    print("summary entries", len(SUMMARY))
