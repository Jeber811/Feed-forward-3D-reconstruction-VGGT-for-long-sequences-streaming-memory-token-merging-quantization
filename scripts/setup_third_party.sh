#!/bin/bash
# OPTIONAL: build your own copy of the third-party model code at the pinned commits (protocol/THIRD_PARTY.lock).
# On the CRCV cluster you do NOT need this: protocol.env points THIRD_PARTY at the shared, verified copy in
# /home/ja396977/cap6411/repos. Use this only off-cluster or if that copy becomes unavailable, then
#   export THIRD_PARTY=<dest>   (and re-run the golden check).
# Usage: bash scripts/setup_third_party.sh <dest_dir>      (git clone only; light enough for a login node)
#        then build the CUDA RoPE kernel on a compute node:  sbatch scripts/build_curope.sbatch <dest_dir>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:?usage: setup_third_party.sh <dest_dir>}
mkdir -p "$DEST"
while read -r name commit dsha url; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [ ! -d "$DEST/$name/.git" ]; then git clone -q "$url" "$DEST/$name"; fi
  git -C "$DEST/$name" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$DEST/$name" checkout -q "$commit"
  [ -f "$REPO/patches/$name.patch" ] && git -C "$DEST/$name" apply "$REPO/patches/$name.patch"
  got=$(git -C "$DEST/$name" diff | sha256sum | cut -c1-64)
  [ "$got" = "$dsha" ] && echo "ok   $name" || { echo "FAIL $name (diff hash $got != $dsha)"; exit 1; }
done < "$REPO/protocol/THIRD_PARTY.lock"
# TTT3R reuses CUT3R's compiled curope (same croco code); link after build_curope.sbatch has run.
echo "next: sbatch scripts/build_curope.sbatch $DEST"
