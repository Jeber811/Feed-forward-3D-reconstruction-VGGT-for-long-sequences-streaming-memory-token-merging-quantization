# CAP 6411 Topic 26 — shared evaluation protocol (v0.3 proposal, Oct 28 2026)

Every number that goes into the report/slides is produced by `protocol/run_protocol.sbatch` (submitted from the repo root), which sources
`protocol/protocol.env` and calls one harness (`tools/eval_scannet_any.py`). Anything not produced this way is a
smoke test and is labelled as such. Changes: bump the version, add a line to the changelog, both sign off.

Owners: Jake (`ja396977`), Zilzil (`zhyw86`). Lives in the shared GitHub repo; changes via pull request.
Third-party code is the shared copy pinned in `THIRD_PARTY.lock` (verified at the start of every run); setups are proven equal by `protocol/golden.sbatch` (must print GOLDEN PASS). Rules for people and AI assistants: `AGENTS.md`.

## 1. Hardware + software
| Item | Setting |
|---|---|
| Final numbers | Quadro RTX 6000 24 GB (Turing, sm_75): `--partition=gpu --gres=gpu:turing:1 --constraint=gmem24` (c3-6/c3-7) |
| Unbounded references | Tier `a100_ref` (`$GPU_REF`, A100 80 GB, fp16 forced): ONLY for unbounded controls that run out of memory on 24 GB (VGGT/π³ at 500–1000 frames); labelled "reference" in the report |
| Smoke tests | `$GPU_SMOKE`, tier `smoke_<tag>`; never in report tables |
| Low-bit hardware | No FP8 compute anywhere on the cluster and no bf16 on the RTX 6000: FP8 may only be a storage format; real low-bit compute = INT8. No FlashAttention-2 on Turing (absolute speed understates newer GPUs; memory unaffected) |
| Job shape | 1 GPU, 6 CPUs, `--mem=88G`, `--qos=short --requeue` (results are resumable) |
| Software | torch 2.3.1+cu121 (both of our envs), CUDA module 12.3; env name recorded per run |
| Never | c2-0, c2-1 (launch hangs), c3-2..c3-5 (11–16 GB cards / launch kills), V100 for CUT3R/TTT3R (no curope kernel) |

## 2. Data
| Item | Setting |
|---|---|
| Scenes | ScanNet v2 from `/squash/scannetv2` via the extracted frames in `/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet`; dev set = scene0111_00, scene0188_00; final candidate = the 50 scenes and usable-frame counts in `protocol/scene_sets/scannet50_v03.csv` (49 pinned FastVGGT ScanNet-50 IDs plus scene0010_00 in place of scene0150_00, which has only 430 usable frames) |
| Frame selection | FastVGGT `build_frame_selection` (first frame + uniform integer stride over valid-pose frames, capped at N) |
| Lengths N | 300, 500, 1000, 2000 (+ full scan for scene0000_00 = 5578). The committed fixed-N files contain 50, 50, 40, and 13 eligible scenes respectively; include only scenes with at least N distinct usable frames in each fixed-N aggregate. Never silently cap a scene in a fixed-N aggregate. Record actual selected-frame count per result and report cohort size beside every aggregate. |
| Input | FastVGGT `get_vgg_input_imgs` → 518×392, values in [0,1]. The active CUT3R adapter uses 512×384; outputs are resized back to 518×392. |

## 3. Precision
fp16 autocast on Turing (no bf16). Weights as each model's official recipe: fp32 + autocast for VGGT(official), π³,
CUT3R; fp16 weights for VGGT*/FastVGGT (FastVGGT recipe). Quantized variants state their storage dtype.

## 4. Memory (the budget axis)
- Metric: **total peak GPU memory allocated** = `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`
  right before inference, **including weights**. Also record `max_memory_reserved` (fragmentation) — never nvidia-smi.
- Inputs: kept in host RAM (`CAP_CPU_INPUTS=1`) for every method that does not need all frames on the GPU at once
  (chunked, streaming, recurrent, keyframe-interp). All-at-once models (VGGT, π³, FastVGGT, StreamVGGT-full) take
  their inputs on the GPU because they must. The optional `cut3r_offload` path loads frames one at a time from host RAM.
  The official-loop controls `cut3r` and `streamvggt` keep their official input handling (inputs staged on GPU,
  ~3.4 MB/frame) and are labelled so. **Every NEW bounded model (e.g. the streaming capped-cache base) must support
  `CAP_CPU_INPUTS=1`** (move its own frames to the GPU) and be added to `CPU_INPUT_MODELS` + `cpu_inputs_ok`.
- Budgets: **6, 10, 16 GB** total peak. Each method tunes its own knob to stay under a budget:
  chunked → chunk size/overlap/#keyframes; streaming → `budget_mb`; keyframe-interp → #keyframes.
- Max frames: integer bisection with `tools/max_frames_sweep.py` (inference only), no `expandable_segments` unless
  a row is explicitly labelled with it.

## 5. Metrics + timing
- Poses: FastVGGT ATE/ARE/RPE (comparability with prior work) **and** camera-centre ATE-cc (Sim(3) on c2w; primary).
- Geometry: `rc_*` (Sim(3) from camera centres, no clipping): accuracy / completeness (mean, median, m) and
  F-score @5 cm / @10 cm. FastVGGT Chamfer kept only as a legacy column (saturated).
- Time: model forward incl. each method's own pose recovery/alignment; excludes data loading and metrics.
  Report seconds/frame. v0.2: every run does a 4-frame warm-up pass first (excluded from time and peak memory).
  Timing claims come from dedicated timing runs: `EXTRA_ENV="... CAP_TIMING_REPEATS=3"`, one scene, NAME suffix
  `_timing`, median of 3 reported (`inference_time_ms`; all repeats in `inference_time_ms_all`); compare timings
  only between runs on the same node when possible.
- Drift (v0.2): every run saves `trajectory.npz` (predicted camera-to-world poses + frame ids; no GT data).
  `tools/drift_analysis.py` gives per-frame error after one global Sim(3), error per tenth of the sequence
  (last − first tenth = drift) and per-tenth Sim(3) scale (scale drift).
- Every result folder: `results/<tier>/<name>/input_frame_<N>/<scene>/{memory,metrics}.json` + `run_config.json`
  (git commit, command, env vars, host, GPU, job id).

## 6. Methods (names used in tables)
Controls: vggt_orig, pi3, vggt_star, streamvggt, fastvggt, quantvggt (accuracy only), cut3r,
pi3_kfinterp (simple baseline, N keyframes per budget).
Bases: pi3_chunk (B1), vggt_chunk (B2), stream_capped (B3, Zilzil's compact — TODO: add as a harness model).
Technique grid on each base: {none, K, M, Q, K+M, K+Q, M+Q, K+M+Q}.

Spann3R and TTT3R are excluded from the v0.3 active study and final-run matrix. Their earlier logs remain historical provenance; existing harness/checkpoint support and the legacy TTT3R golden configuration are retained to avoid changing unrelated code paths. The golden check is a setup control, not a final study result.

### Fixed-N final cohort submission
Use the matching committed scene file for each `NS` value; submit one N per job so the `SCENES` and `NS` match. For example, after sourcing `protocol/protocol.env`, set `SCENES="$FINAL_SCENES_500"` and `NS=500` in `--export` for a 500-frame job. `protocol/scene_sets/scannet50_v03.csv` records all 50 scene IDs and finite-pose usable counts. The preflight counted nonempty RGB/depth payloads and finite camera poses from raw `.sens`; every selected scene has at least 586 usable frames (maximum 5578). The 47 newly required extracts completed in Slurm array 397363 with validated `.complete` markers. Report fixed-N means only over the eligible file for that N, with its cohort count. Do not pool N=500 and N=2000 cohorts as if they share all 50 scenes.

## 7. Sign-off
- [ ] Jake — date: (v0.3)
- [ ] Zilzil — date: (v0.3)

## Changelog
- v0.3 proposal (Oct 28): 50-scene final cohort with scene0150_00 (430 usable frames) replaced by scene0010_00; disjoint development set; committed per-N eligibility files and cohort aggregation rule; Spann3R and TTT3R removed from the active method matrix. No final results until both owners sign off, merge, and a post-change golden check passes.
- v0.2 (Oct 23): 4-frame warm-up excluded from time/memory; CAP_TIMING_REPEATS timing runs (median); trajectory.npz +
  tools/drift_analysis.py; CPU-resident inputs for cut3r_offload/ttt3r_offload and required for new bounded models;
  a100_ref tier for unbounded references; golden check now covers pi3, pi3_chunk, vggt_star, streamvggt, ttt3r_offload;
  FP8/bf16/FlashAttention hardware limits stated.
- v0.1 (Oct 23): repo version: location-independent scripts, THIRD_PARTY.lock check, golden check, run_config per run, validate_results.py + make_results_table.py. Draft from Jake's harness + week-1/2 practice; open items marked TODO; pending reconciliation of
  π³ / CUT3R / Spann3R max-frame differences between our week-1 tables.
