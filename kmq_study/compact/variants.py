"""Every method variant, by name. Read this file to see exactly what a name means.

A name is  <base>+<letter>+<letter>...   (letters in any order: stream+M+K == stream+K+M)

  bases
    stream       StreamVGGT, one frame at a time. Each global-attention layer attends to a BOUNDED
                 memory (memory.py): anchor frame 0 + long-term frames + the last `recent` frames,
                 stored under `budget_mb`; oldest long-term frame is evicted first.
                 With no letters every frame enters long-term memory -> a sliding window.
                 Aggregator weights in fp16 (StreamVGGT's release keeps fp32 weights + autocast).
    stream-full  Same code, unlimited memory = the original StreamVGGT KV cache. Sanity check only:
                 matches `streamvggt` from the baseline runs (ATE 0.0801 vs 0.0801).
    chunk        VGGT* (full attention) on chunks of `chunk_size` frames (chunk.py). Each new chunk
                 also re-processes `anchors` frames that are already placed; their point maps give a
                 Sim(3) that puts the chunk into the global frame. With no letters the anchors are
                 the last frames processed (plain overlapping chunks).

  letters (what each adds on top of the base)
    K  keyframe selection   keyframe.py  stream: only keyframes enter long-term memory
                                         chunk:  anchors are keyframes, half recent and half spread
                                                 over the whole history (incl. frame 0)
    M  token merging        stream: merge.py, stored frames keep (1 - merge_ratio) of their tokens
                            chunk:  FastVGGT's token merging inside each chunk (= FastVGGT chunks)
    Q  quantization         quant.py     int8 aggregator weights; stream also stores K/V in int8
                                         (kv_dtype="fp8" = e4m3 storage; Turing has no fp8 math)

Every setting can be overridden from the command line:  --set budget_mb=4096 merge_ratio=0.75
"""

BASES = {
    "stream": dict(
        budget_mb=8192,       # bytes of stored keys/values (all 24 layers), excluding the current frame
        recent=8,             # last frames always kept (short-term context)
        keyframes=None,       # None: every frame goes to long-term memory (sliding window)
        merge_ratio=0.0,      # fraction of a stored frame's patch tokens merged away
        kv_dtype="fp16",      # fp16 | int8 | fp8
        weights="fp16",       # fp16 | int8   (aggregator only; heads stay as released)
        compress_anchor=False,  # keep frame 0 (the world frame) unmerged and unquantized
    ),
    "chunk": dict(
        chunk_size=100,       # frames per forward pass, anchors included
        anchors=10,           # already-placed frames re-processed in each chunk, for alignment
        keyframes=None,       # None: anchors = last frames processed
        merging=None,         # None: VGGT*; 0: FastVGGT merging from block 0
        weights="fp16",       # fp16 | int8   (aggregator only)
    ),
}
BASES["stream-full"] = dict(BASES["stream"], budget_mb=float("inf"), recent=10**9)

KF = dict(keyframes="pose", kf_rot_deg=15.0, kf_trans=0.15, kf_max_gap=30)
LETTERS = {
    "stream": {"K": KF, "M": dict(merge_ratio=0.5), "Q": dict(kv_dtype="int8", weights="int8")},
    "chunk": {"K": KF, "M": dict(merging=0), "Q": dict(weights="int8")},
}
LETTERS["stream-full"] = LETTERS["stream"]
ORDER = ["K", "M", "Q"]

# The grids run by default (run_variant.py --variant all | all-chunk)
GRID = ["stream", "stream+K", "stream+M", "stream+Q",
        "stream+K+M", "stream+K+Q", "stream+M+Q", "stream+K+M+Q"]
GRID_CHUNK = ["chunk", "chunk+K", "chunk+M", "chunk+Q",
              "chunk+K+M", "chunk+K+Q", "chunk+M+Q", "chunk+K+M+Q"]


def canonical(name):
    """'stream+M+K' -> 'stream+K+M' (letters in K, M, Q order), so result folders don't duplicate."""
    base, *letters = name.split("+")
    if base not in BASES:
        raise ValueError(f"unknown base '{base}', expected one of {list(BASES)}")
    bad = [x for x in letters if x not in LETTERS[base]]
    if bad:
        raise ValueError(f"unknown letter(s) {bad} in '{name}', expected {ORDER}")
    return "+".join([base] + [x for x in ORDER if x in letters])


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"none": None, "true": True, "false": False}.get(v.lower(), v)


def config(name, overrides=()):
    """Full settings dict for a variant name, plus 'key=value' overrides."""
    name = canonical(name)
    base, *letters = name.split("+")
    cfg = dict(BASES[base])
    for x in letters:
        cfg.update(LETTERS[base][x])
    for kv in overrides:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise ValueError(f"--set {k}: not a setting of {name} ({sorted(cfg)})")
        cfg[k] = parse_value(v)
    cfg["name"], cfg["base"] = name, base
    return cfg


def label(name, overrides=()):
    """Result folder name: the variant, plus any overrides (e.g. stream+K+M@budget_mb=4096)."""
    return canonical(name) + "".join(f"@{kv}" for kv in overrides)


if __name__ == "__main__":
    for n in GRID + GRID_CHUNK:
        print(f"{n:16s}", {k: v for k, v in config(n).items() if k not in ("name", "base")})
