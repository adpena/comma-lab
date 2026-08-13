#!/usr/bin/env python3
"""Canonical detached Modal harvest poller (replaces per-dispatch hand-rolled pollers).

Two hand-rolled predecessors each shipped a defect (qs1: hardcoded paths; qs2: invalid
ledger status 'completed' — the valid vocabulary is in call_id_ledger.VALID_STATUSES).
This is the parameterized canonical form. Launch ONLY via
tools/launch_detached_process.py (the launch-guard blocks nohup/disown).

Usage:
  modal_harvest_poller.py --call-id fc-... --output-dir DIR [--result-name NAME]
                          [--deadline-s 10800] [--poll-s 60]

Writes DIR/<result-name> on success, DIR/poller.done | poller.failed markers, and marks
the call ledger 'harvested' (rc=0) or 'failed'.
"""
import argparse
import json
import pathlib
import sys
import time

REPO_SRC = "/Users/adpena/Projects/pact/src"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--result-name", default="MODAL_REMOTE_RESULT.json")
    ap.add_argument("--deadline-s", type=float, default=3 * 3600)
    ap.add_argument("--poll-s", type=float, default=60)
    args = ap.parse_args()

    sys.path.insert(0, REPO_SRC)
    from tac.deploy.modal.call_id_ledger import update_call_id_outcome

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / args.result_name

    import modal

    t0 = time.time()
    while time.time() - t0 < args.deadline_s:
        try:
            r = modal.functions.FunctionCall.from_id(args.call_id).get(timeout=5)
        except TimeoutError:
            time.sleep(args.poll_s)
            continue
        except Exception as e:  # terminal remote failure
            (out / "poller.failed").write_text(f"{type(e).__name__}: {e}\n")
            update_call_id_outcome(call_id=args.call_id, status="failed", rc=1,
                                   agent="MAIN", harvest_result={"error": str(e)[:500]})
            return 1
        result_path.write_text(json.dumps(r, indent=2, default=str))
        update_call_id_outcome(call_id=args.call_id, status="harvested", rc=0,
                               agent="MAIN",
                               harvest_result={"result_path": str(result_path)})
        (out / "poller.done").write_text("ok\n")
        return 0

    (out / "poller.failed").write_text(f"deadline {args.deadline_s:.0f}s exceeded\n")
    update_call_id_outcome(call_id=args.call_id, status="failed", rc=124, agent="MAIN",
                           harvest_result={"error": "poller deadline exceeded"})
    return 1


if __name__ == "__main__":
    sys.exit(main())
