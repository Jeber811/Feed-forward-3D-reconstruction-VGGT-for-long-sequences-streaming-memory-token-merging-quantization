"""Check that every result folder under results/ was produced by the protocol. Run on a compute node:
    srun -p normal -c 2 --mem=4G -t 00:05:00 python tools/validate_results.py results [--final-only]

Rules (protocol v0.1): run_config present; protocol version current; code not dirty; fp16; inputs-on-CPU rule;
final tier `rtx6000` only on a Quadro RTX 6000; every scene/N has memory.json status ok + metrics.json.
Exit 1 if any FINAL-tier run violates a rule (smoke tiers only produce warnings).
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = (REPO / "protocol" / "protocol.env").read_text()
VERSION = re.search(r"^PROTOCOL_VERSION=(\S+)", ENV, re.M).group(1)
CPU_INPUT_MODELS = set(re.search(r"cpu_inputs_ok\(\) \{ case \$1 in ([^)]+)\)", ENV).group(1).split("|"))
# tiers whose runs may enter report tables -> required GPU. rtx6000 = main 24 GB results; a100_ref = unbounded
# controls that run out of memory on 24 GB (labelled "reference" in the report).
FINAL_TIERS = {"rtx6000": "Quadro RTX 6000", "a100_ref": "A100"}


def check_run(run_dir: Path):
    errs = []
    cfgs = sorted(run_dir.glob("run_config_*.json"))
    if not cfgs:
        return ["no run_config_*.json (not produced by protocol/run_protocol.sbatch)"]
    for f in cfgs:
        c = json.loads(f.read_text())
        tag = f.name
        if c.get("protocol") != VERSION:
            errs.append(f"{tag}: protocol {c.get('protocol')} != {VERSION}")
        if str(c.get("git_commit", "")).endswith("-dirty"):
            errs.append(f"{tag}: code was dirty")
        if c.get("CAP_DTYPE") != "float16":
            errs.append(f"{tag}: CAP_DTYPE {c.get('CAP_DTYPE')}")
        want_cpu = "1" if c.get("model") in CPU_INPUT_MODELS else "0"
        if c.get("CAP_CPU_INPUTS") != want_cpu:
            errs.append(f"{tag}: CAP_CPU_INPUTS {c.get('CAP_CPU_INPUTS')} != {want_cpu}")
        gpu_need = FINAL_TIERS.get(c.get("tier", ""))
        if gpu_need and gpu_need not in c.get("gpu", ""):
            errs.append(f"{tag}: final tier on {c.get('gpu')}")
        for n in str(c.get("lengths", "")).split():
            for s in str(c.get("scenes", "")).split():
                d = run_dir / f"input_frame_{n}" / s
                mem = d / "memory.json"
                if not mem.exists():
                    errs.append(f"{tag}: missing {d.relative_to(run_dir)}/memory.json")
                    continue
                st = json.loads(mem.read_text()).get("status")
                if st == "ok" and not (d / "metrics.json").exists():
                    errs.append(f"{tag}: {d.relative_to(run_dir)} ok but no metrics.json")
                elif st not in ("ok", "oom"):
                    errs.append(f"{tag}: {d.relative_to(run_dir)} status={st}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=REPO / "results")
    ap.add_argument("--final-only", action="store_true")
    a = ap.parse_args()
    bad_final = False
    for tier in sorted(p for p in a.results.iterdir() if p.is_dir() and not p.name.startswith("_")):
        final = tier.name in FINAL_TIERS
        if a.final_only and not final:
            continue
        for run in sorted(p for p in tier.iterdir() if p.is_dir()):
            errs = check_run(run)
            label = "FINAL" if final else "smoke"
            print(f"[{'OK ' if not errs else 'BAD'}] {label} {tier.name}/{run.name}")
            for e in errs:
                print(f"       - {e}")
            bad_final |= final and bool(errs)
    return 1 if bad_final else 0


if __name__ == "__main__":
    sys.exit(main())
