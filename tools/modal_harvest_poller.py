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

# Field names the remote result actually uses for billed wall-clock and hardware.
# Verified against a live MODAL_REMOTE_RESULT.json (hv1 ep0634 T4 row):
#   modal_elapsed_seconds = 421.559061639 ; gpu_model = 'Tesla T4'
_ELAPSED_KEYS = ("modal_elapsed_seconds", "elapsed_seconds")
_GPU_KEYS = ("gpu_model", "gpu")


def _result_elapsed_seconds(result: Any) -> float | None:
    """Billed wall-clock from a Modal remote result, or None if absent.

    Returns None rather than a guess: an absent elapsed must stay visibly
    absent in the ledger so the spend reader can report its blind set instead
    of silently pricing a fabricated duration.
    """
    if not isinstance(result, dict):
        return None
    for key in _ELAPSED_KEYS:
        value = result.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def _result_gpu(result: Any) -> str | None:
    """Hardware string from a Modal remote result, or None if absent."""
    if not isinstance(result, dict):
        return None
    for key in _GPU_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
        # The remote result carries the two facts the spend ledger needs and
        # this poller has always dropped: MEASURED billed wall-clock and the
        # hardware it ran on. Without them 49 of 63 cap-window calls carried
        # neither cost nor elapsed, so the >=$20 envelope could only be
        # answered from memo prose. Record what is measured; never synthesise
        # cost from a rate table (the one row where a real cost_actual_usd and
        # an elapsed both exist proved a published-rate estimate 2.8x high).
        update_call_id_outcome(
            call_id=args.call_id,
            status="harvested",
            rc=0,
            agent="MAIN",
            elapsed_seconds=_result_elapsed_seconds(r),
            gpu=_result_gpu(r),
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
