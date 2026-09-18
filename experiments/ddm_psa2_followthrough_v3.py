#!/usr/bin/env python3
"""Continue psa2 after the RESUMED search completed (receipt rc 0, 60 pairs).

Why a v3: the first waiter (``ddm_psa2_followthrough``) and the full search it
waited on were both ended by MAIN's pause on 2026-09-16 (rc -15, see
``PAUSED_BY_MAIN.json``). The v2 waiter gated on that first waiter REFUSING a
superseded source pin, which never happened -- it was killed, not refused --
and the v2 process itself died before its guard. Nothing about the science
changed: the four source files still match the v2 binding's pins (asserted
below), the searches are complete, and the same resolve -> price -> harvest
chain applies. This script records that history and runs the chain.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
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
    binding = p.STORE / "FOLLOWTHROUGH_V3_BINDING.json"
    if not binding.exists():
        p.save(binding, {"sources": [p.fact(x) for x in paths], "charter_complete": False})
    sources = p.load(binding)["sources"]
    for item in sources:
        p.checked(item)
    # The science sources are pinned by the v2 binding; they must be unchanged.
    v2 = {item["path"]: item["sha256"] for item in p.load(p.STORE / "FOLLOWTHROUGH_V2_BINDING.json")["sources"]}
    for item in sources[1:]:
        if v2.get(item["path"]) != item["sha256"]:
            raise ValueError(f"science source drifted since the v2 binding: {item['path']}")
    resume_receipt = root / "ddm_psa2_search_resume.done.done"
    flow.wait_receipt(resume_receipt)
    pairs = p.load(p.STORE / "SAMPLE.json")["pairs"]
    missing = [pair for pair in pairs if not (p.STORE / "pairs" / f"pair_{pair:03d}/SEARCH.json").exists()]
    if len(pairs) != 60 or missing:
        raise ValueError(f"resume receipt without 60 complete searches: missing {missing}")
    supersession = p.STORE / "FOLLOWTHROUGH_V3_SUPERSESSION.json"
    if not supersession.exists():
        p.save(supersession, {
            "search_receipt": p.fact(resume_receipt),
            "paused_by_main": p.fact(p.STORE / "PAUSED_BY_MAIN.json"),
            "killed_full_search_receipt": p.fact(root / "ddm_psa2_search_full.done"),
            "killed_first_waiter_receipt": p.fact(root / "ddm_psa2_followthrough.done"),
            "v2_binding": p.fact(p.STORE / "FOLLOWTHROUGH_V2_BINDING.json"),
            "reason": "first waiter and full search ended rc -15 by MAIN's pause; v2 gated on a source-pin refusal that never occurred; sources unchanged",
        })
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
    print(json.dumps({"singletons": str(p.STORE / "SINGLETONS.json"), "k_net_positive": k,
                      "charter_complete": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"continuation_failure": type(error).__name__, "reason": str(error)}), flush=True)
        raise
