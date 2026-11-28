"""Slides 21-23 visuals from combined.json (written by combined.py). Metric names match the deck's metrics slide.
Run on a compute node:  ~/.venvs/slides/bin/python compare_figs.py  ->  figs/t1_vggt500.png, f12_length.png,
t3_streaming.png, plus t1_vggt500.csv / t3_streaming.csv (paste into Google Sheets -> copy -> paste into Slides)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = json.loads(Path("combined.json").read_text())
G, L, T = C["grid"], C["long"], C["timing"]
OUT = Path("figs")
NAVY, GREY = "#1B2A3A", "#6B6B6B"
FAM = {"rec": "#8C8C8C", "b2": "#0072B2", "b1": "#009E73", "kf": "#444444", "full": "#E69F00"}
TINT = {"b1": "#E6F4EF", "b2": "#E8F1F8", "rec": "#F2F2F2"}
plt.rcParams.update({"font.size": 13, "text.color": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY,
                     "ytick.color": NAVY, "axes.edgecolor": "#C9CED4", "axes.spines.top": False,
                     "axes.spines.right": False})
spf = lambda k: T[k]["spf"]
G50 = C["g50"]
FAM["infv"] = "#D55E00"
TINT["infv"] = "#FBEDE5"
VS = "VGGT* = official VGGT weights run in FastVGGT's memory-efficient code with merging off."
KMQ = "K = keyframe anchors (5 recent + 5 older), M = token merging, Q = int8 weights."


def draw_table(name, header, rows, widths, foot, figsize, fs=13):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    n = len(rows) + 1
    x = [0]
    for w in widths:
        x.append(x[-1] + w)
    W = x[-1]
    for i, h in enumerate(header):
        ax.text(x[i] + (0.01 if i == 0 else widths[i] / 2), n - 0.38, h, ha="left" if i == 0 else "center", va="center",
                fontsize=fs - 1, fontweight="bold", color=NAVY)
    ax.plot([0, W], [n - 1, n - 1], color=NAVY, lw=1.5)
    prev = None
    for r, (fam, cells, bold) in enumerate(rows):
        y = n - 1.5 - r
        if prev is not None and fam != prev:  # thin rule between model families
            ax.plot([0, W], [y + 0.5, y + 0.5], color="#C9CED4", lw=0.8)
        prev = fam
        if fam in TINT:
            ax.add_patch(plt.Rectangle((0, y - 0.5), W, 1, color=TINT[fam], zorder=0, lw=0))
        if fam == "span":
            ax.text(0.01, y, cells, ha="left", va="center", fontsize=fs - 1, color="#B07800", style="italic")
            continue
        ax.add_patch(plt.Rectangle((0.003, y - 0.32), 0.006, 0.64, color=FAM[fam], lw=0))
        for i, c in enumerate(cells):
            ax.text(x[i] + (0.016 if i == 0 else widths[i] / 2), y, c, ha="left" if i == 0 else "center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal", color=NAVY)
    ax.plot([0, W], [0, 0], color="#C9CED4", lw=1)
    ax.set_xlim(0, W)
    ax.set_ylim(-0.1, n + 0.25)
    fig.text(0.01, 0.0, foot, fontsize=fs - 3.5, color=GREY, va="top", wrap=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    with open(f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h.replace("\n", " ") for h in header])
        for fam, cells, _ in rows:
            w.writerow([cells] if fam == "span" else cells)
    print("wrote", name)


# ---- Slide 21: VGGT-family models at 500 frames, same 10 scenes ------------------------------------------------------
def r500(key, label, fam, sp, bold=False):
    m = G[key]
    return (fam, [label, f"{m['peak']:.1f}", f"{m['ate_cm']:.2f}", f"{m['acc_cm']:.2f}", f"{m['f5']:.2f}", sp], bold)


rows = [
    r500("B2 chunk", "VGGT* + chunking", "b2", f"{spf('zhy_v03_chunk_N500_timing'):.2f}"),
    r500("B2 KQ", "VGGT* + chunking + K + Q  (most accurate)", "b2", f"{spf('zhy_v03_chunk_N500_timing'):.2f}*"),
    r500("B2 MQ", "VGGT* + chunking + M + Q  (fastest)", "b2", f"{spf('zhy_v03_chunkM_N500_timing'):.2f}*"),
    r500("B1 6GB KQ", "π³ + chunking + K + Q, 6 GB budget", "b1", f"{spf('jake_v04_b1_b6_KQ_N500_timing'):.2f}", True),
    r500("B1 10GB KQ", "π³ + chunking + K + Q, 10 GB budget", "b1", f"{spf('jake_v04_b1_b10_KQ_N500_timing'):.2f}", True),
    r500("B1 10GB KMQ", "π³ + chunking + K + M + Q, 10 GB  (fastest)", "b1",
         f"{spf('jake_v04_b1_b10_KMQ_N500_timing'):.2f}"),
    r500("kfinterp110", "π³ on 110 keyframes (simple baseline)", "kf", "≈0.02‡"),
    r500("InfV 1.2M", "InfiniteVGGT  (streaming, default 1.2M budget)", "infv", f"{spf('jake_v04_b3_infv_N500_timing'):.2f}"),
    ("span", "Unchunked VGGT / VGGT* / FastVGGT / π³: out of memory at 500 frames on 24 GB", False),
    ("span", "All 50 scenes, 500 frames (π³ + chunking at the 10 GB budget; paired test vs. plain chunking)", False),
]


def r50(m, label, fam, sp, bold=False):
    return (fam, [label, f"{m['peak']:.1f}", f"{m['ate_cm']:.2f}", f"{m['acc_cm']:.2f}", f"{m['f5']:.2f}", sp], bold)


L500 = L["500"]
rows += [
    r50(L500["B2 chunk"]["cohort"], "VGGT* + chunking", "b2", f"{spf('zhy_v03_chunk_N500_timing'):.2f}"),
    r50(L500["InfiniteVGGT"]["cohort"], "InfiniteVGGT  (streaming, 1.2M budget)", "infv",
        f"{spf('jake_v04_b3_infv_N500_timing'):.2f}"),
    r50(G50["none"], "π³ + chunking, 10 GB", "b1", f"{spf('jake_v04_b1_b10_none_N500_timing'):.2f}"),
    r50(G50["K"], "π³ + chunking + K, 10 GB  (better in 40/50 scenes)", "b1", f"{spf('jake_v04_b1_b10_K_N500_timing'):.2f}"),
    r50(G50["KQ"], "π³ + chunking + K + Q, 10 GB  (better in 45/50)", "b1", f"{spf('jake_v04_b1_b10_KQ_N500_timing'):.2f}", True),
    r50(G50["KMQ"], "π³ + chunking + K + M + Q, 10 GB", "b1", f"{spf('jake_v04_b1_b10_KMQ_N500_timing'):.2f}"),
]
draw_table("t1_vggt500", ["VGGT-family model", "Peak GPU\nMemory (GiB) ↓", "ATE-cc\n(cm) ↓", "Acc\n(cm) ↓",
                          "F@5cm\n↑", "Speed\n(s/frame) ↓"], rows, [0.54, 0.16, 0.10, 0.09, 0.09, 0.13],
           f"Top: the same 10 ScanNet scenes, 500 frames each; bottom: all 50 scenes. Same evaluation code and metrics. "
           f"{KMQ} {VS} Speed: Quadro RTX 6000 (1 scene, median of 3). * measured on the same config without K/Q (they "
           "don't change VGGT* speed). ‡ measured at 2,000 frames. Paired over 50 scenes, ATE-cc vs. plain chunking: "
           "K −0.65 cm, K+Q −0.77 cm (both p<0.0001); adding M to K+Q costs +0.42 cm (95% CI excludes 0).", (13.5, 6.0))


# ---- Slide 22: ATE-cc vs sequence length, VGGT family only (full cohorts) -------------------------------------------
NS = [300, 500, 1000, 2000]
LINES = [("VGGT* + chunking", "B2 chunk", FAM["b2"], "D", 3.0),
         ("π³ + chunking + K + Q, 6 GB", "B1 6GB KQ", "#7FCBB0", "o", 3.0),
         ("π³ + chunking + K + Q, 10 GB", "B1 10GB KQ", FAM["b1"], "o", 4.0)]
# Broken y-axis: InfiniteVGGT (33-63 cm) sits in a short top panel so the chunked lines (4-12 cm) keep their scale.
fig, (axt, ax) = plt.subplots(2, 1, sharex=True, figsize=(7.4, 5.6), gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08})
iv = [L[str(n)]["InfiniteVGGT"]["cohort"] for n in NS]
axt.plot(NS, [m["ate_cm"] for m in iv], marker="s", color=FAM["infv"], lw=3.0, ms=8, zorder=3)
axt.annotate(f"InfiniteVGGT (streaming)\n{iv[-1]['ate_cm']:.1f} cm · {max(m['peak'] for m in iv):.1f} GiB",
             (2000, iv[-1]["ate_cm"]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=11,
             color=FAM["infv"])
axt.set_ylim(25, 70)
axt.set_yticks([30, 45, 60])
axt.grid(axis="y", color="#E3E6EA", lw=0.6)
axt.spines["bottom"].set_visible(False)
axt.tick_params(axis="x", length=0)
for a, y in ((axt, 0), (ax, 1)):  # diagonal break marks on the left spine
    a.plot([-0.012, 0.012], [y - 0.02, y + 0.02], transform=a.transAxes, color="#C9CED4", lw=1.2, clip_on=False)
for lab, k, c, m, lw in LINES:
    xs = [n for n in NS if k in L[str(n)]]
    ys = [L[str(n)][k]["cohort"]["ate_cm"] for n in xs]
    ax.plot(xs, ys, marker=m, color=c, lw=lw, ms=8, zorder=3)
    ax.annotate(f"{lab}\n{ys[-1]:.1f} cm · {L['2000'][k]['cohort']['peak']:.1f} GiB", (xs[-1], ys[-1]),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=11, color=c,
                fontweight="bold" if k.startswith("B1") else "normal")
for k, lab, c, dy in (("full pi3", "Unchunked π³", FAM["b1"], -1.6), ("full VGGT*", "Unchunked VGGT*", FAM["b2"], 3.0)):
    m = L["300"][k]["cohort"]
    ax.plot(300, m["ate_cm"], "*", color=FAM["full"], ms=17, mec="white", zorder=4)
    ax.annotate(f"{lab}: {m['ate_cm']:.1f} cm, {m['peak']:.1f} GiB\n(out of memory at 500)", (300, m["ate_cm"]),
                xytext=(380, m["ate_cm"] + dy), fontsize=10, color=c, va="center",
                arrowprops=dict(arrowstyle="->", color=FAM["full"], lw=1.1))
ax.set_xscale("log")
ax.set_xticks(NS, ["300\n(50 sc.)", "500\n(50)", "1000\n(40)", "2000\n(13)"])
ax.minorticks_off()
ax.set_xlim(270, 2150)
ax.set_ylim(0, 13)
ax.set_xlabel("Sequence length (frames)")
fig.supylabel("ATE-cc (cm)  ↓", x=0.03, fontsize=13, color=NAVY)
ax.grid(axis="y", color="#E3E6EA", lw=0.6)
fig.text(0.01, -0.07, "ScanNet; every method on the same scenes at each length. Labels: error at 2,000 frames and peak GPU "
         "memory (InfiniteVGGT: highest over all lengths). Note the broken y-axis. VGGT* chunking was not run at 300.",
         fontsize=9.5, color=GREY, wrap=True)
fig.savefig(OUT / "f12_length.png", dpi=200, bbox_inches="tight")
print("wrote f12_length")


# ---- Slide 23: streaming vs chunked, all four lengths ---------------------------------------------------------------
def r3(k, lab, fam, sp, b=False):
    have = [n for n in NS if k in L[str(n)]]
    pk = max(L[str(n)][k]["cohort"]["peak"] for n in have)
    ate = [f"{L[str(n)][k]['cohort']['ate_cm']:.1f}" if n in have else "—" for n in NS]
    return (fam, [lab, f"{pk:.1f}", *ate, f"{sp:.2f}"], b)


draw_table("t3_streaming", ["Model", "Peak GPU\nMemory (GiB) ↓", "ATE-cc ↓\n300", "ATE-cc ↓\n500", "ATE-cc ↓\n1,000",
                            "ATE-cc ↓\n2,000", "Speed\n(s/frame) ↓"],
           [r3("CUT3R", "CUT3R  (streaming)", "rec", spf("jake_v04_ctrl_cut3r_offload_N500_timing")),
            r3("TTT3R", "TTT3R  (streaming)", "rec", spf("jake_v04_ctrl_ttt3r_offload_N500_timing")),
            r3("InfiniteVGGT", "InfiniteVGGT  (streaming)", "infv", spf("jake_v04_b3_infv_N500_timing")),
            r3("B2 chunk", "VGGT* + chunking", "b2", spf("zhy_v03_chunk_N500_timing")),
            r3("B1 6GB KQ", "π³ + chunking + K + Q, 6 GB", "b1", spf("jake_v04_b1_b6_KQ_N500_timing"), True)],
           [0.30, 0.16, 0.10, 0.10, 0.10, 0.10, 0.12],
           "ATE-cc in cm at each sequence length (frames). ScanNet: 50 scenes at 300 and 500 frames, 40 at 1,000, 13 at 2,000 (same scenes for every model). Memory: "
           "highest peak over all lengths run. Speed: Quadro RTX 6000 at 500 frames. — = not run.", (12, 3.4))


# ---- InfiniteVGGT KV budget x precision (10 grid scenes, 500 frames) ------------------------------------------------
def r4(b, q):
    m = G[f"InfV {b}{' Q' if q else ''}"]
    return ("infv", [f"{b} tokens", "int8 (Q)" if q else "fp32 (default)", f"{m['peak']:.1f}", f"{m['ate_cm']:.1f}", f"{m['acc_cm']:.1f}",
                     f"{m['f5']:.2f}", f"{m['spf']:.2f}"], False)


ref = G["B1 6GB KQ"]
draw_table("t4_infv_budget", ["KV budget", "Weights", "Peak GPU\nMemory (GiB) ↓", "ATE-cc\n(cm) ↓", "Acc\n(cm) ↓", "F@5cm\n↑",
                              "Time\n(s/frame) ↓"],
           [r4(b, q) for b in ("1.2M", "600k", "300k") for q in (False, True)]
           + [("b1", ["π³ + chunking + K + Q, 6 GB", "int8 (Q)", f"{ref['peak']:.1f}", f"{ref['ate_cm']:.1f}", f"{ref['acc_cm']:.1f}",
                      f"{ref['f5']:.2f}", f"{spf('jake_v04_b1_b6_KQ_N500_timing'):.2f}*"], True)],
           [0.36, 0.13, 0.17, 0.10, 0.09, 0.09, 0.12],
           "Same 10 ScanNet scenes, 500 frames. InfiniteVGGT default budget = 1.2M KV tokens; compute is fp16 autocast in every row, Weights = storage. Int8 weights (4.7 → 2.2 GB) cut ~4.2 GiB of peak at every "
           "budget with no accuracy change; shrinking the budget costs accuracy. Last row: our 6 GB π³ config for "
           "reference. Time: mean over the 10 runs (Quadro RTX 6000); * RTX 6000 timing run (1 scene, median of 3).",
           (11.5, 3.6))
