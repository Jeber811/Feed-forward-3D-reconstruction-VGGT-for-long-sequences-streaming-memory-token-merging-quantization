#!/bin/bash
# Stage 0: re-run all controls through the protocol (2 dev scenes). Run from the repo root on the login node
# (it only calls sbatch):  bash scripts/submit_stage0_controls.sh <golden_job_id>
#  - a100_ref rows (unbounded controls that run out of memory on 24 GB): submitted now.
#  - rtx6000 rows: held with --dependency=afterok:<golden_job_id>, so they start once the golden reference is
#    recorded successfully (if the golden job fails, they never start: fix, re-record, resubmit).
#  - Not here (after GOLDEN PASS): max-frames bisection (week-1 maxima: VGGT 221, pi3 316, VGGT* 392, FastVGGT 306,
#    StreamVGGT 76, CUT3R 951, Spann3R 3389) and CAP_TIMING_REPEATS=3 timing runs.
set -euo pipefail
GOLDEN=${1:?usage: submit_stage0_controls.sh <golden_job_id>}
cd "$(dirname "$0")/.."
mkdir -p logs
source protocol/protocol.env
S="--qos=short --requeue --parsable"
sub() {  # sub <gpu flags> <tier> <model> <name> <Ns> [extra_env]
  local gpu=$1 tier=$2 model=$3 name=$4 ns=$5 extra=${6:-}
  local dep=""; [ "$tier" = rtx6000 ] && dep="--dependency=afterok:$GOLDEN"
  local id=$(sbatch $S $gpu $dep --job-name=c0_${name} \
    --export=ALL,MODEL=$model,NAME=$name,NS="$ns",TIER=$tier,EXTRA_ENV="$extra" protocol/run_protocol.sbatch)
  echo "$id $tier $name N=[$ns]"
}
echo "# a100_ref (start now)"
sub "$GPU_REF" a100_ref vggt_orig  vggt_orig  "300 500 1000"
sub "$GPU_REF" a100_ref pi3        pi3        "500 1000"
sub "$GPU_REF" a100_ref vggt_star  vggt_star  "500 1000"
sub "$GPU_REF" a100_ref fastvggt   fastvggt   "500 1000"
echo "# rtx6000 (after golden $GOLDEN)"
sub "$GPU_FINAL" rtx6000 vggt_orig     vggt_orig     "200"
sub "$GPU_FINAL" rtx6000 pi3           pi3           "300"
sub "$GPU_FINAL" rtx6000 vggt_star     vggt_star     "300"
sub "$GPU_FINAL" rtx6000 fastvggt      fastvggt      "300"
sub "$GPU_FINAL" rtx6000 streamvggt    streamvggt    "70"
sub "$GPU_FINAL" rtx6000 ttt3r_offload ttt3r_offload "300 500 1000 2000"
sub "$GPU_FINAL" rtx6000 cut3r         cut3r         "300 500"
sub "$GPU_FINAL" rtx6000 cut3r_offload cut3r_offload "300 500 1000 2000"  # Oct 27: replaces spann3r (dropped)
sub "$GPU_FINAL" rtx6000 pi3_kfinterp  kfinterp300   "1000 2000" "PI3_KFN=300"
sub "$GPU_FINAL" rtx6000 pi3_kfinterp  kfinterp100   "1000 2000" "PI3_KFN=100"
sub "$GPU_FINAL" rtx6000 pi3_kfinterp  kfinterp50    "1000 2000" "PI3_KFN=50"
