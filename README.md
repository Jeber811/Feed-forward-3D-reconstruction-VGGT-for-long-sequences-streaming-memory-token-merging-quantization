# Feed-forward 3D reconstruction (VGGT) for long sequences: streaming memory × token merging × quantization

CAP 6411 (Computer Vision Systems, UCF, Fall 2026), Topic 26. Jake Weber and Zengyan Wang.

**Goal:** make VGGT-class feed-forward reconstruction handle 500+ frames on one 24 GB GPU, and measure which
efficiency techniques stack. The techniques are bounded memory (chunked or streaming inference), keyframe anchors (K),
token merging (M) and real INT8 weight storage (Q). We measure pose and geometry accuracy against peak GPU memory and
seconds per frame.

**Start here:**
1. [`AGENTS.md`](AGENTS.md): the rules.
2. [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md): the evaluation protocol (v0.4) that every reported number follows.
3. The section [Reproduce the paper](#reproduce-the-paper) below.

## Main results (ScanNet, 500 frames, 10 grid scenes; full tables in the paper)

| Model | Peak GiB | ATE-cc (cm) | F@5cm | s/frame (RTX 6000) |
|---|---|---|---|---|
| Unchunked VGGT / VGGT\* / FastVGGT / π³ | out of memory at 500 frames on 24 GB | – | – | – |
| InfiniteVGGT (streaming, default 1.2M KV budget) | 15.2 | 50.96 | 0.21 | 0.54 |
| VGGT\* + chunking (B2) | 6.9 | 9.97 | 0.54 | 0.42 |
| π³ + chunking + K + Q, 6 GiB budget (B1) | 5.2 | 4.15 | 0.81 | 0.32 |
| π³ + chunking + K + Q, 10 GiB budget (B1) | 9.3 | 3.78 | 0.84 | 0.60 |
| π³ + chunking + K + M + Q, 10 GiB budget (B1, fastest) | 9.8 | 4.09 | 0.80 | 0.19 |
| π³ on 110 keyframes + pose interpolation (simple baseline) | 9.8 | 4.18 | 0.85 | ≈0.02 |

Peak memory of the chunked methods stays flat from 300 to 2,000 frames.

## Repository layout

| Path | Contents |
|---|---|
| `protocol/` | `protocol.env` (paths, scene sets, GPU requests, site detection), `run_protocol.sbatch` (the only way to produce a reported number), `golden.sbatch` + `golden/` (setup check), `THIRD_PARTY.lock` (pinned third-party commits), `scene_sets/` (fixed-N cohorts and the 10-scene grid) |
| `tools/eval_scannet_any.py` | One harness for every model: data loading, inference, peak-memory and timing measurement, metrics. Chunking, Sim(3) stitching, K/M/Q and the simple baseline live here |
| `tools/` | `validate_results.py`, `make_results_table.py`, `drift_analysis.py`, `check_golden.py`, `crosscheck_tiers.py`, `max_frames_sweep.py` |
| `analysis/` | Scripts that turn the result JSON into the paper's tables, statistics and figures (see step 5 below) |
| `results/<tier>/<run name>/` | Per-scene `memory.json`, `metrics.json`, `drift_*.json` and a `run_config_<job>.json` that records commit, GPU, scenes and every setting of the run. Tables: `results/TABLE_*.md` |
| `patches/`, `scripts/`, `env/` | Local fixes to third-party repos, setup scripts, pinned package versions |
| `kmq_study/` | Zengyan's early K/M/Q study scripts (B2 development) |

## Reproduce the paper

### 1. Setup
- **Hardware:** final numbers come from a Quadro RTX 6000 24 GB (tier `rtx6000`, CRCV cluster) or a Tesla V100-32GB
  capped to 24 GB in PyTorch (tier `v100_cap24`, UCF ARCC Newton). Speed numbers come only from RTX 6000 runs.
- **Environment:** torch 2.3.1+cu121, conda env pinned in `env/`. On CRCV, `protocol.env` already points to the shared
  env, third-party code and checkpoints.
- **Third-party code** (VGGT, FastVGGT, π³, StreamVGGT, InfiniteVGGT, CUT3R, TTT3R, …) at the commits in
  `protocol/THIRD_PARTY.lock`. Off-cluster: `bash scripts/setup_third_party.sh <dir>`, then
  `sbatch scripts/build_curope.sbatch <dir>`, then `export THIRD_PARTY=<dir>`.
- **Checkpoints:** the official releases, named as in `ckpt_for()` in `protocol/protocol.env`:
  - `pi3_model.safetensors`
  - `model_tracker_fixed_e20.pt` (VGGT\*/FastVGGT)
  - `vggt_1b_model.pt`
  - `streamvggt_checkpoints.pth` (StreamVGGT and InfiniteVGGT)
  - `cut3r_512_dpt_4_64.pth` (CUT3R/TTT3R)

  Set `CKPT_DIR` to their folder.
- **Data:** ScanNet v2 (not redistributed here). Frames are extracted per scene (`DATA` in `protocol.env`) and GT meshes
  come from the ScanNet scans (`GT_PLY_DIR`). Scene lists: `protocol/scene_sets/scannet50_v03_N{300,500,1000,2000}.txt`
  (50 / 50 / 40 / 13 scenes) and `scannet50_v03_grid10_N500.txt` (technique grid).

### 2. Check the setup (must print `GOLDEN PASS`)
```bash
mkdir -p logs && source protocol/protocol.env
sbatch $SUBMIT_OPTS $GPU_FINAL protocol/golden.sbatch
```

### 3. Run a configuration
Every run goes through `protocol/run_protocol.sbatch`. It needs:
- `MODEL`: a harness model.
- `NAME`: a unique run name, which becomes the results folder.
- `NS`: the frame count.
- `SCENES`: a scene list from `protocol.env`.
- `TIER`: the hardware tier.
- `EXTRA_ENV`: the settings in the table below.

Example: the B1 K+Q row at the 10 GiB budget on the grid scenes:
```bash
source protocol/protocol.env
sbatch $SUBMIT_OPTS $GPU_FINAL --export=ALL,MODEL=pi3_chunk,NAME=my_b1_b10_KQ,NS=500,SCENES="$GRID_SCENES_500",TIER=$FINAL_TIER,EXTRA_ENV="CH_SIZE=180 CH_OVERLAP=27 CH_KF=18 CAP_WEIGHTS=int8" protocol/run_protocol.sbatch
```
For full cohorts use `SCENES="$FINAL_SCENES_<N>"` with the matching `NS=<N>` (one N per job).

**Knobs:**
- `CH_SIZE` = chunk size, `CH_OVERLAP` = recent overlap frames, `CH_KF` = keyframe anchors (K). Set in every run by
  `protocol.env`: `CH_ALIGN=svd` (weighted Sim(3) stitching) and `CH_MEMORDER=recent_first`.
- `CAP_WEIGHTS=int8` = Q (INT8 weight-only storage).
- `CAP_MERGE=0` = M (token merging from the first global-attention block, ratio 0.9).
- `PI3_KFN` = keyframes for the simple baseline.
- `CAP_INFV_BUDGET` = InfiniteVGGT KV-cache budget in tokens.
- `CAP_TIMING_REPEATS=3` = timing run (median of 3; run name ends in `_timing`, one scene).

**Settings used in the paper.** Every run's exact settings are also recorded in
`results/<tier>/<name>/run_config_*.json`.

| Paper row | MODEL | EXTRA_ENV | Result folders (`results/…`) |
|---|---|---|---|
| B1 grid, 6 GiB: Q / K+Q | `pi3_chunk` | `CH_SIZE=80 CH_OVERLAP=20 CH_KF=0` / `CH_OVERLAP=12 CH_KF=8`, + `CAP_WEIGHTS=int8` | `v100_cap24/jake_v04_grid_b1_b6_{Q,KQ}` |
| B1 grid, 6 GiB: M+Q / K+M+Q | `pi3_chunk` | as Q / K+Q, + `CAP_MERGE=0` | `v100_cap24/jake_v04_grid_b1_b6_{MQ,KMQ}` |
| B1 grid, 10 GiB: none / K | `pi3_chunk` | `CH_SIZE=110 CH_OVERLAP=28 CH_KF=0` / `CH_SIZE=110 CH_OVERLAP=17 CH_KF=11` | `v100_cap24/jake_v04_grid_b1_b10_{none,K}` |
| B1 grid, 10 GiB: Q / K+Q | `pi3_chunk` | `CH_SIZE=180 CH_OVERLAP=45 CH_KF=0` / `CH_SIZE=180 CH_OVERLAP=27 CH_KF=18`, + `CAP_WEIGHTS=int8` | `v100_cap24/jake_v04_grid_b1_b10_{Q,KQ}` |
| B1 grid, 10 GiB: M / K+M | `pi3_chunk` | `CH_SIZE=105 CH_OVERLAP=27 CH_KF=0` / `CH_SIZE=105 CH_OVERLAP=16 CH_KF=11`, + `CAP_MERGE=0` | `v100_cap24/jake_v04_grid_b1_b10_{M,KM}` |
| B1 grid, 10 GiB: M+Q / K+M+Q | `pi3_chunk` | as Q / K+Q, + `CAP_MERGE=0` | `v100_cap24/jake_v04_grid_b1_b10_{MQ,KMQ}` |
| B1 grid, 16 GiB: none / K (M, K+M: + `CAP_MERGE=0`) | `pi3_chunk` | `CH_SIZE=240 CH_OVERLAP=59 CH_KF=0` / `CH_SIZE=240 CH_OVERLAP=35 CH_KF=24` | `v100_cap24/jake_v04_grid_b1_b16_{none,K,M,KM}` |
| B1 grid, 16 GiB: Q / K+Q (M+Q, K+M+Q: + `CAP_MERGE=0`) | `pi3_chunk` | `CH_SIZE=300 CH_OVERLAP=75 CH_KF=0` / `CH_SIZE=300 CH_OVERLAP=45 CH_KF=30`, + `CAP_WEIGHTS=int8` | `v100_cap24/jake_v04_grid_b1_b16_{Q,KQ,MQ,KMQ}` |
| B1 K+Q full cohorts, N = 300–2000 | `pi3_chunk` | the K+Q settings of each budget | `v100_cap24/jake_v04_b1_b{6,10,16}_KQ_N{300,500,1000,2000}` |
| π³ 10 GiB on all 50 scenes, N=500 | `pi3_chunk` | the 10 GiB none / K / K+Q / K+M+Q settings | `rtx6000/jake_v04_g50_b1_b10_{none,K,KQ,KMQ}` |
| Simple baseline (π³ keyframe interpolation) | `pi3_kfinterp` | `PI3_KFN=15` / `110` / `260` (6 / 10 / 16 GiB) | `v100_cap24/jake_v04_ctrl_kfinterp{15,110,260}_N*` |
| Unchunked models, N=300 | `pi3`, `vggt_star`, `fastvggt` | – | `v100_cap24/jake_v04_ctrl_{pi3,vggt_star,fastvggt}_N300` |
| Streaming: TTT3R / CUT3R | `ttt3r_offload`, `cut3r_offload` | – | `rtx6000/jake_v04_ctrl_{ttt3r,cut3r}_offload_N*` |
| Streaming: InfiniteVGGT | `infinitevggt` | default; grid: `CAP_INFV_BUDGET=600000` or `300000`, ± `CAP_WEIGHTS=int8` | `rtx6000/jake_v04_b3_infv_N*`, `rtx6000/jake_v04_b3_grid_infv_*` |
| Speed (all s/frame numbers) | same as the row | + `CAP_TIMING_REPEATS=3`, scene0000_00 | `rtx6000/*_timing` |
| Cross-site check (B2 config on V100) | `vggt_chunk` | `CH_SIZE=100 CH_OVERLAP=10 CH_KF=0` | `v100_cap24/xcheck_b2_chunk_N500` |

B2 (VGGT\* chunked, `MODEL=vggt_chunk`) runs were produced by Zengyan Wang with the same harness and protocol. Their
JSON is kept in her clone on the CRCV cluster (`/home/zhyw86/WorkSpace/cap6411/protocol_clone/results/rtx6000/zhy_v03_*`),
which `analysis/combined.py` reads.

### 4. Validate, tabulate, drift
Run on a compute node, not a login node:
```bash
python tools/validate_results.py --final-only                      # rejects runs from uncommitted code
python tools/make_results_table.py --tier rtx6000,v100_cap24         # results/TABLE_*.md
python tools/drift_analysis.py results/<tier>/<name>                 # writes drift_*.json (last tenth - first tenth error)
```

### 5. Paper tables, statistics and figures (`analysis/`)
These scripts read the result JSON above (paths at the top of each script point to our cluster clones; edit them to
your clone). They need only a small Python with numpy and matplotlib (check each script's imports):
- `combined.py`: means over matched scene sets for every paper table → `combined.json`.
- `compare_figs.py`: the comparison tables (500 frames, streaming vs length, InfiniteVGGT budget × int8) and the
  length plot → `figs/` + CSV.
- `stats.py`: paired per-scene effects on the 10 grid scenes (mean, 95% CI, sign test) and drift.
- `agg2.py`: 50-scene paired statistics and |drift|.
- `make_figs.py`, `drift_fig.py`: B1 figures (grid, budget tuning, Q, M speed/accuracy, cohorts, controls, drift).

## Third-party code and data
Third-party code (VGGT, FastVGGT, π³, CUT3R, TTT3R, Spann3R, StreamVGGT, InfiniteVGGT, …) is used under each project's
license at the commits in `protocol/THIRD_PARTY.lock`. Our local fixes are in `patches/`. ScanNet data, checkpoints and
point clouds are not redistributed here.
