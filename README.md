# Feed-forward 3D reconstruction (VGGT) for long sequences: streaming memory × token merging × quantization

CAP 6411 (Computer Vision Systems, UCF, Fall 2026), Topic 26. Goal: make VGGT-class feed-forward reconstruction handle
500+ frames on one 24 GB GPU by combining bounded memory (chunked or streaming), keyframe selection, token merging and
real low-bit storage, measuring pose/geometry accuracy against peak memory and latency.

**Start here:** [`AGENTS.md`](AGENTS.md) (rules) → [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md) (the evaluation
protocol every reported number follows).

## Quick start (CRCV cluster)
```bash
git clone git@github.com:Jeber811/Feed-forward-3D-reconstruction-VGGT-for-long-sequences-streaming-memory-token-merging-quantization.git
cd Feed-forward-3D-reconstruction-VGGT-for-long-sequences-streaming-memory-token-merging-quantization
mkdir -p logs && source protocol/protocol.env
sbatch --qos=short $GPU_FINAL protocol/golden.sbatch          # must end with GOLDEN PASS (logs/golden_<job>.out)
sbatch --qos=short --requeue $GPU_FINAL \
  --export=ALL,MODEL=pi3_chunk,NAME=pi3_chunk200_K,NS="1000",TIER=rtx6000,EXTRA_ENV="CH_SIZE=200 CH_OVERLAP=30 CH_KF=20" \
  protocol/run_protocol.sbatch
```

## Methods in the harness (`tools/eval_scannet_any.py --model ...`)
Controls: `vggt_orig`, `pi3`, `vggt_star`, `fastvggt`, `streamvggt`, `cut3r`, `ttt3r_offload`, `spann3r`,
`pi3_kfinterp` (simple baseline). Bases: `pi3_chunk`, `vggt_chunk`, `fastvggt_chunk` (chunked, Sim(3)-stitched,
keyframe memory via `CH_*` env vars); streaming capped-cache variants to be added.

Third-party code (VGGT, FastVGGT, π³, CUT3R, TTT3R, Spann3R, StreamVGGT, …) is used under each project's license at the
commits in `protocol/THIRD_PARTY.lock`; local fixes are in `patches/`. ScanNet data is not redistributed here.
