# kmq_study — baselines + K / M / Q ablation (exploratory, pre-protocol)

Zengyan's baseline runs and the training-free K / M / Q study on a 24 GB Quadro RTX 6000 (CRCV c3-6 / c3-7).
These scripts predate `protocol/` and are **not** protocol runs: numbers produced here are exploratory and do not go
into report tables until they are rerun through `protocol/run_protocol.sbatch` (see `AGENTS.md`).
No data, checkpoints or logs are included. Paths are hard-coded to `/home/zhyw86/WorkSpace/cap6411/baseline`
(this folder = that directory, minus third-party checkouts, `ckpt/`, `logs/`).

## Variant names

`<base>+<letters>`; the exact settings of every name are in [`compact/variants.py`](compact/variants.py).

| Base | What it is |
|---|---|
| `stream` | StreamVGGT, one frame at a time, global attention over a **bounded** KV memory (8 GB): frame 0 + long-term frames + last 8 frames |
| `stream-full` | same code, unlimited memory (= original StreamVGGT; sanity check) |
| `chunk` | VGGT\* on 100-frame chunks; 10 anchor frames per chunk give a Sim(3) into the global frame |

| Letter | stream | chunk |
|---|---|---|
| **K** keyframes | only pose-change keyframes enter long-term memory | anchors = keyframes (half recent, half spread incl. frame 0) |
| **M** token merging | ToMe merging of stored K/V (50%) | FastVGGT merging inside each chunk |
| **Q** int8 | int8 K/V storage + int8 aggregator weights | int8 aggregator weights |

## Layout

| Path | What |
|---|---|
| `compact/` | the method: `variants.py`, `memory.py`, `keyframe.py`, `merge.py`, `quant.py`, `stream.py`, `chunk.py`, `run_variant.py` (see `compact/README.md`) |
| `tools/infer_poses.py` | baseline inference adapters: vggt (VGGT\*), fastvggt, vggt-official, vggt-current, quantvggt, streamvggt, pi3, cut3r, spann3r |
| `tools/score_ate.py` | ATE / ARE / RPE (evo, Sim(3)-aligned) for baselines and variants |
| `tools/score_drift.py`, `tools/score_pointmaps.py` | per-frame drift; sampled point-map error vs. ScanNet depth |
| `tools/mem_sweep.py`, `tools/plot_*.py`, `tools/render_kmq.py` | memory sweeps, figures, point-cloud renders |
| `tools/frames.py`, `tools/extract_sens.py` | ScanNet frame selection / extraction |
| `*.sbatch`, `*.sh` | the cluster jobs that produced the results (env setup, baselines, ATE, variant grids, scoring) |
| `plots/` | local plotting: `plot_full.py` (6-scene summary), `plot_variants.py` (stream grid) |
| `patches/FastVGGT_local.patch` | local fix in the FastVGGT checkout (input dtype follows autocast dtype; fp16 on Turing) |

## Run

```bash
# stream variants (streamvggt env), chunk variants (fastvggt env); one base per call
python compact/run_variant.py --variant all       --scenes $P/scene0000_00 $P/scene0013_02
python compact/run_variant.py --variant all-chunk --scenes $P/scene0000_00 $P/scene0013_02
python tools/score_ate.py logs/variants --scenes $P/scene0000_00 $P/scene0013_02
```

## Exploratory results (6 ScanNet scenes, mean ATE in m)

| | @100 | @300 | @1000 | @2000 | peak GB | fps |
|---|---|---|---|---|---|---|
| chunk | 0.187 | 0.380 | 0.219 | 0.208 | 6.9 | 2.2 |
| chunk+M | 0.173 | 0.330 | 0.193 | 0.241 | 8.7 | 5.0 |
| chunk+K | 0.187 | 0.297 | 0.406 | 0.596 | 6.9 | 2.2 |
| chunk+K+M+Q | 0.178 | 0.273 | 0.385 | 0.676 | 7.9 | 5.2 |
| stream+K+M | 0.706 | 0.911 | 1.025 | 1.117 | 17.0 | 1.5 |
| VGGT\* (OOM ~390) | 0.187 | 0.214 | OOM | OOM | 16.3 | 0.9 |
| FastVGGT (OOM ~310) | 0.175 | 0.187 | OOM | OOM | 21.8 | 3.3 |
| Spann3R | 1.238 | 1.527 | 1.887 | 1.737 | 10.9 | 5.1 |

Takeaways: chunking keeps VGGT\* accuracy to 2000 frames at flat memory; M stacks with chunks (~2.3× faster);
K stabilises the stream track but hurts chunk alignment at long N; the stream track is limited by StreamVGGT itself.
Known issues: chunk outlier at N=300 on scene0000_00 (sparse frames → poor anchor fit); stream+K+M+Q fragmentation
OOM at 2000 frames on scene0050_00.
