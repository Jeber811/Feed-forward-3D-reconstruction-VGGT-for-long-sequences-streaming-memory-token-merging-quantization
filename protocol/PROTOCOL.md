# CAP 6411 Topic 26 — shared evaluation protocol (v0.1, DRAFT, Oct 23 2026)

Every number that goes into the report/slides is produced by `protocol/run_protocol.sbatch` (submitted from the repo root), which sources
`protocol/protocol.env` and calls one harness (`tools/eval_scannet_any.py`). Anything not produced this way is a
smoke test and is labelled as such. Changes: bump the version, add a line to the changelog, both sign off.

Owners: Jake (`ja396977`), Zilzil (`zhyw86`). Lives in the shared GitHub repo; changes via pull request.
Third-party code is the shared copy pinned in `THIRD_PARTY.lock` (verified at the start of every run); setups are proven equal by `protocol/golden.sbatch` (must print GOLDEN PASS). Rules for people and AI assistants: `AGENTS.md`.

## 1. Hardware + software
| Item | Setting |
|---|---|
| Final numbers | Quadro RTX 6000 24 GB (Turing, sm_75): `--partition=gpu --gres=gpu:turing:1 --constraint=gmem24` (c3-6/c3-7) |
| Smoke / unbounded references | A100 80 GB (`--gres=gpu:ampere:1 --constraint=gmem80`), labelled tier `a100_80g`; fp16 forced unless stated |
| Job shape | 1 GPU, 6 CPUs, `--mem=88G`, `--qos=short --requeue` (results are resumable) |
| Software | torch 2.3.1+cu121 (both of our envs), CUDA module 12.3; env name recorded per run |
| Never | c2-0, c2-1 (launch hangs), c3-2..c3-5 (11–16 GB cards / launch kills), V100 for CUT3R/TTT3R (no curope kernel) |

## 2. Data
| Item | Setting |
|---|---|
| Scenes | ScanNet v2 from `/squash/scannetv2` via the extracted frames in `/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet`; dev set = scene0000_00, scene0013_02; final set = the ScanNet-50 list (FastVGGT `eval/scannet_50.yaml`) subset agreed in protocol.env |
| Frame selection | FastVGGT `build_frame_selection` (first frame + uniform integer stride over valid-pose frames, capped at N) |
| Lengths N | 300, 500, 1000, 2000 (+ full scan for scene0000_00 = 5578) |
| Input | FastVGGT `get_vgg_input_imgs` → 518×392, values in [0,1]. Model-native adapters: CUT3R/TTT3R 512×384, Spann3R 224 centre crop; outputs resized back to 518×392 |

## 3. Precision
fp16 autocast on Turing (no bf16). Weights as each model's official recipe: fp32 + autocast for VGGT(official), π³,
CUT3R/TTT3R, Spann3R; fp16 weights for VGGT*/FastVGGT (FastVGGT recipe). Quantized variants state their storage dtype.

## 4. Memory (the budget axis)
- Metric: **total peak GPU memory allocated** = `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`
  right before inference, **including weights**. Also record `max_memory_reserved` (fragmentation) — never nvidia-smi.
- Inputs: kept in host RAM (`CAP_CPU_INPUTS=1`) for every method that does not need all frames on the GPU at once
  (chunked, streaming, recurrent, keyframe-interp). All-at-once models (VGGT, π³, FastVGGT, StreamVGGT-full) take
  their inputs on the GPU because they must. *(TODO: per-frame input path for CUT3R/TTT3R/Spann3R/StreamVGGT wrappers.)*
- Budgets: **6, 10, 16 GB** total peak. Each method tunes its own knob to stay under a budget:
  chunked → chunk size/overlap/#keyframes; streaming → `budget_mb`; keyframe-interp → #keyframes.
- Max frames: integer bisection with `tools/max_frames_sweep.py` (inference only), no `expandable_segments` unless
  a row is explicitly labelled with it.

## 5. Metrics + timing
- Poses: FastVGGT ATE/ARE/RPE (comparability with prior work) **and** camera-centre ATE-cc (Sim(3) on c2w; primary).
- Geometry: `rc_*` (Sim(3) from camera centres, no clipping): accuracy / completeness (mean, median, m) and
  F-score @5 cm / @10 cm. FastVGGT Chamfer kept only as a legacy column (saturated).
- Time: model forward incl. each method's own pose recovery/alignment; excludes data loading and metrics.
  Report seconds/frame. One warm-up pass before timed runs (TODO in run script).
- Every result folder: `results/<tier>/<name>/input_frame_<N>/<scene>/{memory,metrics}.json` + `run_config.json`
  (git commit, command, env vars, host, GPU, job id).

## 6. Methods (names used in tables)
Controls: vggt_orig, pi3, vggt_star, streamvggt, fastvggt, quantvggt (accuracy only), ttt3r_offload, cut3r, spann3r,
pi3_kfinterp (simple baseline, N keyframes per budget).
Bases: pi3_chunk (B1), vggt_chunk (B2), stream_capped (B3, Zilzil's compact — TODO: add as a harness model).
Technique grid on each base: {none, K, M, Q, K+M, K+Q, M+Q, K+M+Q}.

## 7. Sign-off
- [ ] Jake — date:
- [ ] Zilzil — date:

## Changelog
- v0.1 (Oct 23): repo version: location-independent scripts, THIRD_PARTY.lock check, golden check, run_config per run, validate_results.py + make_results_table.py. Draft from Jake's harness + week-1/2 practice; open items marked TODO; pending reconciliation of
  π³ / CUT3R / Spann3R max-frame differences between our week-1 tables.
