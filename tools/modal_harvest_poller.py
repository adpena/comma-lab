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
from collections.abc import Callable
from typing import Any

REPO_SRC = "/Users/adpena/Projects/pact/src"

POLL_RESULT = "result"
POLL_REMOTE_FAILURE = "remote_failure"
POLL_DEADLINE = "deadline"


def poll_modal_call(
    *,
    call_id: str,
    deadline_s: float,
    poll_s: float,
    get_result: Callable[[float], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll one Modal call to a terminal provider response.

    This is the canonical polling loop used by both the legacy CLI below and
    ``tools/modal_endpoint_close.py``.  It deliberately performs no ledger or
    filesystem mutation: callers own terminal classification and custody.
    Keeping those effects outside the loop lets the endpoint closer close the
    claim ledger *before* the call-id ledger without forking polling logic.
    """

    if get_result is None:
        import modal

        call = modal.functions.FunctionCall.from_id(call_id)

        def get_result(timeout: float) -> Any:
            return call.get(timeout=timeout)

    started = monotonic()
    while monotonic() - started < deadline_s:
        try:
            result = get_result(5.0)
        except TimeoutError:
            remaining = deadline_s - (monotonic() - started)
            if remaining > 0 and poll_s > 0:
                sleep(min(poll_s, remaining))
            continue
        except Exception as exc:  # terminal remote failure
            return {
                "kind": POLL_REMOTE_FAILURE,
                "error_class": type(exc).__name__,
                "error": str(exc),
            }
        return {"kind": POLL_RESULT, "result": result}

    return {
        "kind": POLL_DEADLINE,
        "error_class": "PollDeadlineExceeded",
        "error": f"deadline {deadline_s:.0f}s exceeded",
    }


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

    outcome = poll_modal_call(
        call_id=args.call_id,
        deadline_s=args.deadline_s,
        poll_s=args.poll_s,
    )
    if outcome["kind"] == POLL_RESULT:
        r = outcome["result"]
        result_path.write_text(json.dumps(r, indent=2, default=str))
        update_call_id_outcome(
            call_id=args.call_id,
            status="harvested",
            rc=0,
            agent="MAIN",
            harvest_result={"result_path": str(result_path)},
        )
        (out / "poller.done").write_text("ok\n")
        return 0

    error = str(outcome["error"])
    (out / "poller.failed").write_text(f"{outcome['error_class']}: {error}\n")
    rc = 124 if outcome["kind"] == POLL_DEADLINE else 1
    update_call_id_outcome(
        call_id=args.call_id,
        status="failed",
        rc=rc,
        agent="MAIN",
        harvest_result={"error": error[:500]},
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
