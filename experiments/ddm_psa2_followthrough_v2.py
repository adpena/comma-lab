#!/usr/bin/env python3
"""Continue psa2 after the earlier waiter refuses its superseded pose-source pin.

The active RGB search is unchanged. No process is signalled; the old waiter is
allowed to reach its source guard and must stop before starting a pose child.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm_psa2_followthrough as flow

p = flow.p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != p.STORE.resolve():
        raise ValueError("wrong continuation store")
    root = p.REPO / ".omx/tmp/codex_runs"
    paths = [Path(__file__), Path(flow.__file__), Path(p.__file__), p.REPO / "experiments/ddm_psa2_resolve_price.py"]
    binding = p.STORE / "FOLLOWTHROUGH_V2_BINDING.json"
    if not binding.exists():
        p.save(binding, {"sources": [p.fact(x) for x in paths], "charter_complete": False})
    sources = p.load(binding)["sources"]
    for item in sources:
        p.checked(item)
    print(json.dumps({"waiting_for": str(root / "ddm_psa2_search_full.done"),
                      "then": "old waiter terminal source-pin refusal, then corrected pose replay"}), flush=True)
    flow.wait_receipt(root / "ddm_psa2_search_full.done")
    old_receipt = root / "ddm_psa2_followthrough.done"
    while not old_receipt.exists():
        time.sleep(5)
    old = p.load(old_receipt)
    old_log = p.STORE / "launch_followthrough/run.log"
    expected = "source or checkpoint drift: " + str(p.REPO / "experiments/ddm_psa2_resolve_price.py")
    if old.get("rc") == 0 or expected not in old_log.read_text():
        raise ValueError("old continuation did not refuse the expected superseded pose-source pin")
    supersession = p.STORE / "FOLLOWTHROUGH_SUPERSESSION.json"
    if not supersession.exists():
        if (p.STORE / "followthrough/ddm_psa2_resolve60/launch_manifest.json").exists():
            raise ValueError("old continuation unexpectedly launched a pose child")
        p.save(supersession, {"old_terminal_receipt": p.fact(old_receipt),
               "old_log": p.fact(old_log), "new_binding": p.fact(binding), "before_pose_child": True,
               "reason": "candidate-batch versus batch1 replay requires measured tolerance, not float equality"})
    else:
        for key in ("old_terminal_receipt", "old_log", "new_binding"):
            p.checked(p.load(supersession)[key])
    pairs = p.load(p.STORE / "SAMPLE.json")["pairs"]
    if any(not (p.STORE / "pairs" / f"pair_{pair:03d}/SEARCH.json").exists() for pair in pairs):
        raise ValueError("full-search receipt without60 complete searches")
    flow.child("ddm_psa2_resolve60", ["resolve"], sources)
    for pair in pairs:
        flow.child(f"ddm_psa2_price_{pair:03d}", ["price", "--pairs", str(pair), "--label", f"pair_{pair:03d}"], sources)
    flow.harvest()
    singletons = p.load(p.STORE / "SINGLETONS.json")
    rows = singletons["rows"]
    k = singletons["k_net_positive_nonzero_bias"]
    p.save(p.STORE / "SINGLETON_SUMMARY.json", {
        "source": p.fact(p.STORE / "SINGLETONS.json"), "n": 60, "k_net_positive": k,
        "median_credit_cells": statistics.median(row["credit_cells"] for row in rows),
        "median_delta_archive_bytes": statistics.median(
            row["delta_bytes_" + row["cheapest_codec"]] for row in rows),
        "fewer6_falsifier_fired": k < 6,
        "verdict_scope": "formulation: per-pair three-dof int4 RGB head bias on move52" if k < 6 else None,
        "next_disposition": "FOLDED receiver intent" if k < 6 else "QUEUED-WITH-A-FIRE-ORDER: real set admission",
        "axis": p.AXIS, "score_claim": False, "charter_complete": False,
        "owed": singletons["owed"],
    })
    print(json.dumps({"singletons": str(p.STORE / "SINGLETONS.json"), "charter_complete": False}), flush=True)


if __name__ == "__main__":
    main()
