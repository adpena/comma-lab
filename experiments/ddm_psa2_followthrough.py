#!/usr/bin/env python3
"""Receipt-driven continuation of psa2's full lattice, pose, and singleton prices.

Every heavy child uses the canonical detached launcher. This is not an admission
or receiver implementation and cannot mark the full charter complete.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm_psa2_rgb_bias as p


def wait_receipt(path: Path) -> dict:
    while not path.exists():
        time.sleep(5)
    result = p.load(path)
    if result.get("schema") != "detached_local_process_done.v2" or result.get("rc") != 0:
        raise ValueError(f"child failed: {path}: {result}")
    return result


def child(name: str, args: list[str], sources: list[dict]) -> None:
    for source in sources:
        p.checked(source)
    receipt = p.REPO / ".omx/tmp/codex_runs" / (name + ".done")
    launch = p.STORE / "followthrough" / name
    if not receipt.exists() and not (launch / "launch_manifest.json").exists():
        print(json.dumps({"launch": name, "storage_before_heavy": p.capacity(32 << 20)}), flush=True)
        argv = [sys.executable, "tools/launch_detached_process.py", "--output-dir", str(launch),
                "--purpose", name, "--authority", "ddm_psa2 charter; receipt-sequenced scientific continuation",
                "--artifact-budget-gib", "0.1", "--done-receipt", name,
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--", sys.executable,
                "experiments/ddm_psa2_resolve_price.py", *args, "--resume-from", str(p.STORE)]
        result = subprocess.run(argv, cwd=p.REPO, text=True, capture_output=True, check=False)
        p.save(p.STORE / "followthrough" / (name + ".launch.json"),
               {"argv": argv, "rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode != 0:
            raise ValueError(f"launch refused: {name}, rc={result.returncode}")
    wait_receipt(receipt)
    print(json.dumps({"completed": name, "receipt": str(receipt)}), flush=True)


def harvest() -> None:
    root = p.STORE
    control = p.load(root / "CONTROL.json")
    mean = control["base_mean"]
    rows = []
    for pair in p.load(root / "SAMPLE.json")["pairs"]:
        folder = root / "pairs" / f"pair_{pair:03d}"
        search = p.load(folder / "SEARCH.json")
        resolve = p.load(folder / "RESOLVED.json")
        price = p.load(root / "prices" / f"pair_{pair:03d}/PRICE.json")
        for entry in price["artifacts"]:
            p.checked(entry)
        if search["vectors"] != 4096 or price["pairs"] != [pair]:
            raise ValueError("singleton measurement denominator differs")
        seg = -resolve["credit_cells"] * 100 / (600 * 384 * 512)
        pose = math.sqrt(10 * (mean + resolve["delta_d_pose"] / 600)) - math.sqrt(10 * mean)
        nets = {codec: seg + pose + price[f"delta_bytes_{codec}"] * 25 / 37545489 for codec in ("q11", "sm1")}
        codec = min(nets, key=nets.get)
        rows.append({"pair": pair, "credit_cells": resolve["credit_cells"], "bias_codes": resolve["bias_codes"],
                     "delta_pose": resolve["delta_d_pose"], "seg_delta_S": seg, "resolved_pose_delta_S": pose,
                     "delta_bytes_q11": price["delta_bytes_q11"], "delta_bytes_sm1": price["delta_bytes_sm1"],
                     "net_delta_S": nets, "cheapest_codec": codec,
                     "net_positive_nonzero_bias": nets[codec] < 0 and not resolve["zero_bias_control"],
                     "zero_bias_control": resolve["zero_bias_control"], "price": p.fact(root / "prices" / f"pair_{pair:03d}/PRICE.json")})
    p.save(root / "SINGLETONS.json", {"rows": rows, "n": 60,
           "k_net_positive_nonzero_bias": sum(row["net_positive_nonzero_bias"] for row in rows),
           "axis": p.AXIS, "score_claim": False, "charter_complete": False,
           "owed": ["all600 collateral census", "resolved-pose set admission and real price ladder",
                    "conditional public receiver realization and intent if the set nets"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != p.STORE.resolve():
        raise ValueError("wrong continuation store")
    paths = [Path(__file__), Path(p.__file__), p.REPO / "experiments/ddm_psa2_resolve_price.py"]
    start = p.STORE / "FOLLOWTHROUGH_BINDING.json"
    if not start.exists():
        p.save(start, {"sources": [p.fact(path) for path in paths], "charter_complete": False})
    sources = p.load(start)["sources"]
    for source in sources:
        p.checked(source)
    full_receipt = p.REPO / ".omx/tmp/codex_runs/ddm_psa2_search_full.done"
    print(json.dumps({"waiting_for": str(full_receipt), "next": "resolve then real singleton twins"}), flush=True)
    wait_receipt(full_receipt)
    pairs = p.load(p.STORE / "SAMPLE.json")["pairs"]
    if any(not (p.STORE / "pairs" / f"pair_{pair:03d}/SEARCH.json").exists() for pair in pairs):
        raise ValueError("full-search receipt without every completed pair")
    child("ddm_psa2_resolve60", ["resolve"], sources)
    for pair in pairs:
        child(f"ddm_psa2_price_{pair:03d}", ["price", "--pairs", str(pair), "--label", f"pair_{pair:03d}"], sources)
    harvest()
    print(json.dumps({"singletons": str(p.STORE / "SINGLETONS.json"), "charter_complete": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"continuation_failure": type(error).__name__, "reason": str(error)}), flush=True)
        raise
