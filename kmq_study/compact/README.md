# compact/ — StreamVGGT with a bounded memory, plus K / M / Q

Training-free: the released StreamVGGT weights, only what the model remembers changes.

## Variant names

`<base>+<letters>` — letters in any order, folder names always use K, M, Q order.

| Name | Bounded memory | K keyframes | M token merging | Q int8 |
|---|:-:|:-:|:-:|:-:|
| `stream-full` | — (original StreamVGGT cache) | | | |
| `stream` | ✓ (sliding window) | | | |
| `stream+K` | ✓ | ✓ | | |
| `stream+M` | ✓ | | ✓ | |
| `stream+Q` | ✓ | | | ✓ |
| `stream+K+M` | ✓ | ✓ | ✓ | |
| `stream+K+Q` | ✓ | ✓ | | ✓ |
| `stream+M+Q` | ✓ | | ✓ | ✓ |
| `stream+K+M+Q` | ✓ | ✓ | ✓ | ✓ |

The exact settings of every name are in [`variants.py`](variants.py) (`python compact/variants.py` prints them).
Override any setting with `--set key=value`; the result folder then gets `@key=value` appended.

## One file per idea

| File | What it is |
|---|---|
| `variants.py` | names → settings (the only file to read to know what a variant does) |
| `memory.py` | the bounded memory: anchor frame 0 + long-term frames + recent window, under `budget_mb` |
| `keyframe.py` | **K** — pose-change rule deciding which frames stay in long-term memory |
| `merge.py` | **M** — ToMe merging of the stored keys/values of each frame |
| `quant.py` | **Q** — int8 / fp8 storage of keys/values, int8 aggregator weights |
| `stream.py` | glue: loads StreamVGGT and swaps its KV cache for `memory.py` |
| `run_variant.py` | runner: variants × scenes × frame counts → `logs/variants/<variant>/` |

## Run (streamvggt env, 24 GB GPU)

```bash
python compact/run_variant.py --variant all --scenes $P/scene0000_00 $P/scene0013_02
python compact/run_variant.py --variant stream+K+M --set budget_mb=4096 --scenes $P/scene0000_00
python tools/score_ate.py logs/variants --scenes $P/scene0000_00 $P/scene0013_02   # fastvggt env
```
