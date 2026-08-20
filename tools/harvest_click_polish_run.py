#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Harvest a Modal click-polish run (task #399 phase-2) with full custody.

Pulls the FunctionCall result (HARVEST OR LOSE — ~24h result-cache TTL), writes
candidate archive + sha256 + report + ledger + per-component deltas to
experiments/results/clickpolish_pr110_20260710/<run_id>/, and updates the
canonical modal call_id ledger.

Usage:
  .venv/bin/python tools/harvest_click_polish_run.py --call-id fc-... [--timeout 30]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT_ROOT = os.path.join(ROOT, "experiments/results/clickpolish_pr110_20260710")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    import modal

    fc = modal.FunctionCall.from_id(args.call_id)
    try:
        r = fc.get(timeout=args.timeout)
    except TimeoutError:
        print("STILL_RUNNING")
        return 2
    except Exception as e:  # report the remote failure verbatim
        print(f"REMOTE_FAILURE: {type(e).__name__}: {e}")
        return 1

    run_id = r.get("run_id", "unknown")
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    candidate = r.pop("candidate_archive_b", None)
    cand_sha = None
    if candidate:
        tmp = os.path.join(out_dir, "candidate_archive.zip.tmp")
        with open(tmp, "wb") as f:
            f.write(candidate)
        os.replace(tmp, os.path.join(out_dir, "candidate_archive.zip"))
        cand_sha = hashlib.sha256(candidate).hexdigest()
        assert cand_sha == r.get("candidate_sha256"), "candidate sha mismatch vs remote"
        assert len(candidate) == os.path.getsize(os.path.join(out_dir, "candidate_archive.zip"))

    from tac.deploy.modal.result_json import dump_modal_result_json

    if r.get("ledger_text"):
        with open(os.path.join(out_dir, "accepted_clicks_ledger.jsonl"), "w") as f:
            f.write(r["ledger_text"])
    if r.get("exact_eval", {}).get("report_text"):
        with open(os.path.join(out_dir, "report_cpu.txt"), "w") as f:
            f.write(r["exact_eval"]["report_text"])

    # Sister of the modal_harvest_poller defect: `default=str` turns any bytes
    # value into a Python repr that json.load cannot read. Popping the one known
    # bytes key above did not make the rest safe.
    dump_modal_result_json(os.path.join(out_dir, "harvest_result.json"), r)

    from tac.deploy.modal.call_id_ledger import update_call_id_outcome

    update_call_id_outcome(
        call_id=args.call_id,
        status="harvested",
        rc=0,
        elapsed_seconds=r.get("total_s"),
        archive_sha256=cand_sha,
        archive_bytes=r.get("candidate_bytes"),
        subagent_id="clickpolish-build",
        harvest_result={"status": r.get("status"), "run_id": run_id},
    )
    print(json.dumps({
        "status": r.get("status"),
        "run_id": run_id,
        "out_dir": out_dir,
        "candidate_sha256": cand_sha,
        "candidate_bytes": r.get("candidate_bytes"),
        "search": r.get("search", {}).get("best_row"),
        "microarch": r.get("microarch", {}).get("cpu_model"),
        "timings": {k: r.get(k) for k in ("gt_build_s", "search_s", "eval_s", "total_s")},
        "report_head": (r.get("exact_eval", {}).get("report_text") or "")[:400],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
