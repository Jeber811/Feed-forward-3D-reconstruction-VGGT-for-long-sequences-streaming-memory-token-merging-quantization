# Rules for anyone (human or AI assistant) working in this repo

This is a two-person course project (CAP 6411, UCF, Fall 2026, Topic 26: long-sequence VGGT-class reconstruction on a
24 GB GPU). Both people's numbers must be directly comparable. **Read `protocol/PROTOCOL.md` before running anything.**

## Hard rules
1. **Report numbers come only from `protocol/run_protocol.sbatch`** with `TIER=rtx6000` on the Quadro RTX 6000
   (`$GPU_FINAL` from `protocol/protocol.env`). Anything else is a smoke test (`TIER=smoke_<tag>`) and never goes into
   report tables or slides as a final number.
2. **Before your first protocol run, and after any setup change, run the golden check** and make sure it prints
   `GOLDEN PASS`: `mkdir -p logs && source protocol/protocol.env && sbatch --qos=short $GPU_FINAL protocol/golden.sbatch`.
3. **Do not edit `protocol/protocol.env`, `protocol/THIRD_PARTY.lock`, `protocol/golden/` or the shared metric/memory
   code paths in `tools/eval_scannet_any.py` without both people agreeing.** Such a change bumps `PROTOCOL_VERSION`,
   adds a changelog line in `PROTOCOL.md`, and goes through a pull request. New models/variants may be added freely
   (new `--model` branch + `ckpt_for`/`cpu_inputs_ok` entries) as long as existing ones are unchanged. A new
   **bounded** model must keep its inputs in host RAM (`CAP_CPU_INPUTS=1`, listed in `CPU_INPUT_MODELS`) so its
   peak memory is comparable.
4. **Commit your code before a final run.** Runs from uncommitted `tools/` or `protocol/` changes are marked `-dirty`
   and rejected by `tools/validate_results.py`.
5. **CRCV cluster rules** (admin: Steven Dick): no Python or heavy work on the login node — use `sbatch`, or
   `srun -p normal -c 2 --mem=4G -t 00:05:00 ...` for small scripts; >= 4 CPU cores per GPU; use shared datasets and
   checkpoints (never re-download, never unzip big datasets); report anything broken; never use nodes c2-0, c2-1,
   c3-2..c3-5 (launch hangs / small GPUs); do not pip install into the shared conda env.
6. **Never commit data or weights**: no ScanNet frames/meshes (license forbids redistribution), no checkpoints, no
   point clouds, no logs. Only the small JSON results described below. This repo is public.

## How to add results
1. Run via `protocol/run_protocol.sbatch` (results land in `results/<tier>/<name>/...` inside YOUR clone, with a
   `run_config_<job>.json` recording commit, user, node, GPU and settings).
2. Validate on a compute node: `srun -p normal -c 2 --mem=4G -t 00:05:00 python tools/validate_results.py --final-only`.
3. Commit only `results/<tier>/<name>/**/*.json` on a branch named `results/<user>-<topic>` and open a pull request.
   Use a run `NAME` that is unique and descriptive (e.g. `pi3_chunk200_K`, `stream_cap1g_KMQ`), so two people never
   write the same folder.
4. After merge, rebuild the combined table: `srun ... python tools/make_results_table.py --tier rtx6000`.
5. Drift: `srun -p normal -c 2 --mem=8G -t 00:10:00 python tools/drift_analysis.py results/rtx6000/<name>` (uses the
   saved `trajectory.npz`; commit the `drift_*.json` it writes). Timing claims: separate runs with
   `CAP_TIMING_REPEATS=3` in `EXTRA_ENV`, NAME ending in `_timing` (see PROTOCOL.md §5).

## Layout
`protocol/` settings, run script, golden check, lock file · `tools/` harness + checks + table · `patches/` local
fixes to third-party repos (applied in the shared copy) · `scripts/` optional setup · `env/` pinned versions.
Shared, read-only cluster inputs (set in `protocol.env`): third-party code `/home/ja396977/cap6411/repos`,
checkpoints `/home/ja396977/cap6411/ckpt`, ScanNet frames `/home/zhyw86/WorkSpace/cap6411/dataset/scannet/process_scannet`,
GT meshes `/squash/scannetv2/scans`, conda env `/home/ja396977/.conda/envs/cap6411_env`.
