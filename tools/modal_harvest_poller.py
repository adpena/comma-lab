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
import hashlib
import json
import pathlib
import re
import sys
import time
from collections.abc import Callable
from typing import Any

REPO_SRC = "/Users/adpena/Projects/pact/src"
REPO_ROOT = pathlib.Path(REPO_SRC).parent

# Where a scanner-visible scalar mirror of a harvested row is written.
# ``tac.frontier_scan.load_experiments_results_anchors`` globs
# ``experiments/results/*/contest_auth_eval*.json`` -- ONE directory deep, and
# the FILENAME must carry the ``contest_auth_eval`` stem. A mirror that misses
# either half is invisible, which is the whole defect this cures, so both are
# pinned here rather than left to a caller to remember.
MIRROR_DIR_REL = "experiments/results/modal_auth_eval_mirror"
MIRROR_NAME_TEMPLATE = "contest_auth_eval_{label}.json"

_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# gpu_model substring -> canonical QUALIFYING_HARDWARE token in tac.frontier_scan.
_GPU_SUBSTRATE_TOKENS: tuple[tuple[str, str], ...] = (
    ("t4", "linux_x86_64_t4"),
    ("a10g", "linux_x86_64_a10g"),
    ("a100", "linux_x86_64_a100"),
    ("4090", "linux_x86_64_4090"),
    ("h100", "linux_x86_64_h100"),
    ("l40s", "linux_x86_64_l40s"),
)

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


def _canonical_substrate(result: dict[str, Any]) -> str | None:
    """Canonical QUALIFYING_HARDWARE token for a harvested row, or None.

    DERIVED from the measured hardware fields, never hand-typed and never
    guessed. Returning None (rather than a plausible default) is deliberate:
    an anchor carrying a hardware substrate nobody measured is a false
    authority claim, and the frontier scanner would silently disqualify it
    anyway. A blocker marker is written instead so the gap stays visible.
    """
    if result.get("gpu_t4_match") is True:
        return "linux_x86_64_t4"
    model = result.get("gpu_model") or result.get("gpu") or ""
    lowered = str(model).lower()
    for token, substrate in _GPU_SUBSTRATE_TOKENS:
        if token in lowered:
            return substrate
    axis = str(result.get("score_axis") or "").lower()
    if "cpu" in axis:
        return "linux_x86_64_cpu"
    return None


def build_anchor_mirror(
    result: dict[str, Any],
    *,
    lane_id: str | None,
    source_receipt: pathlib.Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (mirror_payload, blocker). Every field is COPIED, never re-derived.

    The score is taken from ``score_recomputed_from_components`` and from
    nowhere else. ``final_score`` is the ROUNDED field (0.16 on the row that
    motivated this) and CLAUDE.md forbids reading it as a score; there is
    deliberately no fallback to it, because a fallback is how a rounded number
    becomes an anchor.
    """
    score = result.get("score_recomputed_from_components")
    if not isinstance(score, (int, float)):
        return None, "score_recomputed_from_components absent or non-numeric"
    sha = result.get("expected_archive_sha256") or result.get("archive_sha256")
    if not isinstance(sha, str) or not sha:
        return None, "expected_archive_sha256 absent"
    substrate = _canonical_substrate(result)
    if substrate is None:
        return None, f"no canonical hardware substrate derivable (gpu_model={result.get('gpu_model')!r})"

    payload: dict[str, Any] = {
        "schema": "modal_auth_eval_anchor_mirror.v1",
        "score": score,
        "score_axis": result.get("score_axis"),
        "evidence_grade": result.get("evidence_grade"),
        "archive_sha256": sha,
        "archive_size_bytes": result.get("archive_size_bytes")
        or result.get("expected_archive_size_bytes"),
        "avg_segnet_dist": result.get("avg_segnet_dist"),
        "avg_posenet_dist": result.get("avg_posenet_dist"),
        "n_samples": result.get("n_samples"),
        "hardware_substrate": substrate,
        "gpu_t4_match": result.get("gpu_t4_match"),
        "gpu_model_raw": result.get("gpu_model"),
        "lane_id": lane_id,
        "source_receipt": str(source_receipt),
        "source_receipt_sha256": hashlib.sha256(source_receipt.read_bytes()).hexdigest()
        if source_receipt.is_file()
        else None,
        "promotion_eligible": result.get("promotion_eligible"),
        "note": (
            "Scalar anchor mirror written at harvest time. The P0 payload stays "
            "at the harvest --output-dir; only scalars are mirrored here so "
            "tac.frontier_scan can see a row that landed on an SSD path outside "
            "experiments/results. Fields copied from the harvested result, never "
            "hand-typed; score is recomputed-from-components, never final_score."
        ),
    }
    return payload, None


def write_anchor_mirror(
    result: Any,
    *,
    source_receipt: pathlib.Path,
    label: str,
    lane_id: str | None,
    repo_root: pathlib.Path = REPO_ROOT,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Mirror a harvested row into the directory the frontier scanner reads.

    WHY (round-11 F1(b) class cure, 2026-08-18). ``fire_modal_auth_eval.py``
    harvests land wherever ``--output-dir`` points, and for large rows that is
    an SSD custody path under ``/Volumes/...`` per the disk-tier rules.
    ``tac.frontier_scan.load_experiments_results_anchors`` globs ONLY under
    ``experiments/results``. A real, paid, contest-CUDA row could therefore be
    complete on disk and invisible to the pointer -- which is exactly what
    happened to the keep01 row and had to be repaired by hand.

    The payload custody rule is unchanged: the bytes stay where --output-dir
    put them. This writes SCALARS plus a hash-pinned pointer back to the
    receipt, so the mirror can never become a competing custody claim.
    """
    if not isinstance(result, dict):
        return None
    mirror_dir = repo_root / MIRROR_DIR_REL
    safe = _LABEL_SAFE_RE.sub("-", label).strip("-") or "unlabeled"
    payload, blocker = build_anchor_mirror(
        result, lane_id=lane_id, source_receipt=source_receipt
    )
    if payload is None:
        # Make the gap DISCOVERABLE rather than only absent -- the same rule
        # POLLER_UNARMED.json follows. An anchor that silently was not written
        # is the signal loss this cure exists to end.
        if out_dir is not None:
            (out_dir / "MIRROR_UNWRITTEN.json").write_text(
                json.dumps(
                    {
                        "schema": "modal_auth_eval_anchor_mirror_unwritten.v1",
                        "blocker": blocker,
                        "source_receipt": str(source_receipt),
                        "consequence": (
                            "this row is NOT visible to tac.frontier_scan; the "
                            "pointer will not see it without a hand-made mirror"
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"MIRROR NOT WRITTEN: {blocker}")
        return None
    mirror_dir.mkdir(parents=True, exist_ok=True)
    path = mirror_dir / MIRROR_NAME_TEMPLATE.format(label=safe)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MIRROR: {path}")
    return path


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
    ap.add_argument("--lane-id", default=None, help="copied into the anchor mirror")
    ap.add_argument(
        "--mirror-label",
        default=None,
        help="filename stem for the scanner-visible mirror (default: output-dir name)",
    )
    ap.add_argument(
        "--no-anchor-mirror",
        action="store_true",
        help="skip the experiments/results anchor mirror (diagnostic rows only)",
    )
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
        if not args.no_anchor_mirror:
            write_anchor_mirror(
                r,
                source_receipt=result_path,
                label=args.mirror_label or out.name,
                lane_id=args.lane_id,
                out_dir=out,
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
