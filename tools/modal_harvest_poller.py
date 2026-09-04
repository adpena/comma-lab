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

    # The runtime tree is HALF the identity of a score (rv13 F4). It is copied,
    # never derived: the receipt records the tree the paid call actually ran.
    runtime_tree = result.get("expected_runtime_tree_sha256") or result.get("runtime_tree_sha256")
    runtime_tree = runtime_tree if isinstance(runtime_tree, str) and runtime_tree else None

    payload: dict[str, Any] = {
        "schema": "modal_auth_eval_anchor_mirror.v2",
        "score": score,
        "score_axis": result.get("score_axis"),
        "evidence_grade": result.get("evidence_grade"),
        "archive_sha256": sha,
        "archive_size_bytes": result.get("archive_size_bytes")
        or result.get("expected_archive_size_bytes"),
        # --- receiver identity (v2, rv13 F4) ---------------------------
        "runtime_tree_sha256": runtime_tree,
        "inflate_sh_rel": result.get("inflate_sh_rel"),
        "inflate_device_policy": result.get("inflate_device_policy"),
        "submission_dir_zip_sha256": result.get("submission_dir_zip_sha256"),
        "avg_segnet_dist": result.get("avg_segnet_dist"),
        "avg_posenet_dist": result.get("avg_posenet_dist"),
        "n_samples": result.get("n_samples"),
        "hardware_substrate": substrate,
        "gpu_t4_match": result.get("gpu_t4_match"),
        "gpu_model_raw": result.get("gpu_model"),
        "lane_id": lane_id,
        "measured_at_utc": result.get("measured_at_utc") or result.get("harvested_at_utc"),
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
            "hand-typed; score is recomputed-from-components, never final_score. "
            "v2 adds runtime_tree_sha256: a score is a function of (archive, "
            "runtime tree), and v1 keyed on the archive alone -- which let ONE "
            "archive sha carry two contradictory passing contest-CUDA scores "
            "(rv13 F4, measured)."
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


def backfill_runtime_custody(
    mirror_path: pathlib.Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upgrade one v1 anchor mirror to v2 by recovering its runtime-tree sha.

    This is RECOVERY, not repair-by-assertion. The value is read from the
    ``source_receipt`` the mirror already pins by sha256, and the receipt is
    re-hashed and checked against that pin BEFORE anything is copied. If the
    receipt is absent or its bytes have moved, the row is left exactly as it is
    and the reason is returned. Nothing is ever hand-typed, and a row whose
    custody cannot be PROVED keeps its honest ``runtime_custody=missing`` stamp
    rather than acquiring a plausible-looking one.

    Returns a verdict dict; never raises on a single bad row.
    """
    verdict: dict[str, Any] = {"path": str(mirror_path), "action": None, "reason": None}
    try:
        payload = json.loads(mirror_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        verdict.update(action="skipped", reason=f"unreadable mirror: {type(exc).__name__}")
        return verdict
    if not isinstance(payload, dict):
        verdict.update(action="skipped", reason="mirror is not a JSON object")
        return verdict
    if payload.get("runtime_tree_sha256"):
        verdict.update(action="already_pinned", reason="runtime_tree_sha256 already present")
        return verdict

    receipt_path_str = payload.get("source_receipt")
    pinned_sha = payload.get("source_receipt_sha256")
    if not receipt_path_str or not pinned_sha:
        verdict.update(action="refused", reason="mirror carries no hash-pinned source_receipt")
        return verdict
    receipt_path = pathlib.Path(receipt_path_str)
    if not receipt_path.is_file():
        verdict.update(action="refused", reason=f"source receipt not on disk: {receipt_path}")
        return verdict
    measured_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if measured_sha != pinned_sha:
        # The pin is the whole point. A receipt whose bytes moved is not the
        # receipt this row was written from, and copying from it would forge
        # custody rather than recover it.
        verdict.update(
            action="refused",
            reason=f"source receipt sha mismatch (pinned {pinned_sha[:16]}…, measured {measured_sha[:16]}…)",
        )
        return verdict
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        verdict.update(action="refused", reason=f"unreadable receipt: {type(exc).__name__}")
        return verdict

    runtime_tree = receipt.get("expected_runtime_tree_sha256") or receipt.get("runtime_tree_sha256")
    if not isinstance(runtime_tree, str) or not runtime_tree:
        verdict.update(action="refused", reason="receipt carries no expected_runtime_tree_sha256")
        return verdict

    updated = dict(payload)
    updated["schema"] = "modal_auth_eval_anchor_mirror.v2"
    updated["runtime_tree_sha256"] = runtime_tree
    for key in ("inflate_sh_rel", "inflate_device_policy", "submission_dir_zip_sha256"):
        if updated.get(key) is None and receipt.get(key) is not None:
            updated[key] = receipt.get(key)
    updated["runtime_custody_backfilled"] = (
        "recovered from the hash-pinned source_receipt (sha verified before copy); "
        "the v1 writer never emitted this field (rv13 F4)"
    )
    verdict.update(action="backfilled", runtime_tree_sha256=runtime_tree, reason=None)
    if not dry_run:
        mirror_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return verdict


def _harvest_outcome_facts(result: Any) -> dict[str, Any]:
    """The scored facts a harvested ledger row should carry, COPIED not derived.

    rv13 F7 (systemic half): the last 20 ``harvested`` rows all carried
    ``score: null``, ``score_axis: null``, ``archive_sha256: null``. The ledger
    recorded THAT a harvest happened and never WHAT it scored, so every spend or
    provenance question had to be answered from a memo instead of the ledger.

    ``score`` is read ONLY from ``score_recomputed_from_components``. There is
    deliberately no fallback to ``final_score`` -- that field is rounded to 2dp
    (0.16 on the live row) and CLAUDE.md forbids reading it as a score. A
    fallback is exactly how a rounded number becomes an anchor.
    """
    if not isinstance(result, dict):
        return {}
    facts: dict[str, Any] = {}
    score = result.get("score_recomputed_from_components")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        facts["score"] = float(score)
    for ledger_key, receipt_keys in (
        ("score_axis", ("score_axis",)),
        ("archive_sha256", ("expected_archive_sha256", "archive_sha256")),
        ("archive_bytes", ("archive_size_bytes", "expected_archive_size_bytes")),
        ("evidence_grade", ("evidence_grade",)),
    ):
        for key in receipt_keys:
            value = result.get(key)
            if value is not None:
                facts[ledger_key] = value
                break
    return facts


def canonical_terminal_claim_notes(result: Any, call_id: str) -> str:
    """The terminal claim-row note, in the shape the compliance checker binds on.

    THE DEFECT (2026-09-04, measured by ps1 against the staged PR #140 packet). This
    poller's note carried no shas at all, so ``pre_submission_compliance_check.py
    --contest-final`` red-flagged three checks — ``dispatch_claim_terminal_archive_sha_bound``,
    ``dispatch_claim_terminal_runtime_tree_sha_bound``, ``dispatch_claim_prior_active_row``
    — and MAIN cured them by hand-typing a second row. Typing 64 hex characters by hand
    is exactly the step that went wrong the same day: one archive sha was MISTYPED and
    needed a third, correcting row; and the fs2 row shipped a runtime tree sha
    TRUNCATED to 12 characters (``915d25f93ad6``), which still fails the checker because
    ``ARCHIVE_RUNTIME_SHA_BINDING_RE`` demands ``[0-9a-fA-F]{64}``.

    The checker matches on substring, so the requirement is exact and small: the FULL
    64-hex archive sha and the FULL 64-hex runtime-tree sha must appear in the row text
    as ``archive_sha256=<sha>`` / ``runtime_tree_sha256=<sha>``. Everything here is
    COPIED from the harvested receipt; nothing is re-derived and nothing is truncated.
    A field the receipt does not carry is simply absent — an absent sha is a visible
    gap the checker will red, while a plausible-looking one would be a forged binding.
    """

    if not isinstance(result, dict):
        return f"auto-closed by modal_harvest_poller at harvest; call {call_id} rc=0"
    parts: list[str] = []
    archive_sha = result.get("expected_archive_sha256") or result.get("archive_sha256")
    if isinstance(archive_sha, str) and archive_sha:
        parts.append(f"archive_sha256={archive_sha}")
    archive_bytes = result.get("archive_size_bytes") or result.get("expected_archive_size_bytes")
    if archive_bytes is not None:
        parts.append(f"archive_bytes={archive_bytes}")
    runtime_tree = result.get("expected_runtime_tree_sha256") or result.get("runtime_tree_sha256")
    if isinstance(runtime_tree, str) and runtime_tree:
        parts.append(f"runtime_tree_sha256={runtime_tree}")
    parts.append(f"call_id={call_id}")
    score = result.get("score_recomputed_from_components")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        parts.append(f"score={score}")
    pose = result.get("avg_posenet_dist")
    if pose is not None:
        parts.append(f"pose={pose}")
    seg = result.get("avg_segnet_dist")
    if seg is not None:
        parts.append(f"seg={seg}")
    n_samples = result.get("n_samples")
    if n_samples is not None:
        parts.append(f"n_samples={n_samples}")
    axis = result.get("score_axis")
    if axis:
        parts.append(f"axis={axis}")
    gpu = _result_gpu(result)
    if gpu:
        parts.append(f"gpu={gpu}")
    parts.append("auto-closed by modal_harvest_poller at harvest (canonical terminal row)")
    return " ".join(parts)


def _close_terminal_claim(
    *,
    lane_id: str | None,
    call_id: str,
    result: Any,
    out_dir: pathlib.Path,
    instance_job_id: str | None = None,
    repo_root: pathlib.Path = REPO_ROOT,
) -> str | None:
    """Append the terminal dispatch-claim row for a harvested fire.

    CLAUDE.md binds: *"When your dispatch completes (success or fail): append a
    terminal row with the same lane_id and instance/job_id ... Do not leave
    completed jobs as phantom active claims."* rv13 F7 measured the twelfth
    pointer move (to1) sitting at ``active_modal_auth_eval_spawning`` with the
    job long finished, because nothing on the harvest path ever closed it.

    Closing here rather than in a hand-edit is the no-manual-dispatch binding:
    the closer is armed at dispatch and MAIN only adjudicates. A close that
    fails NEVER fails the harvest -- the payload and the ledger row are already
    durable, and losing a real harvested row to a bookkeeping error would be a
    strictly worse outcome than a claim that needs one manual close. The failure
    is written to disk so it stays discoverable instead of only absent.
    """
    if not lane_id:
        return "CLAIM NOT CLOSED: no --lane-id passed; the claim ledger is lane-keyed"
    try:
        sys.path.insert(0, REPO_SRC)
        from tac.deploy.claims import DispatchClaimSpec, terminal_dispatch_claim

        passed = isinstance(result, dict) and result.get("passed") is True
        # The status prefix is what `pre_submission_compliance_check.py` reads to decide
        # whether a SUCCESSFUL exact-eval row closed the lane, and its two prefix tables
        # are per-axis (`completed_contest_cuda` / `completed_contest_cpu`). Hardcoding
        # cuda made every CPU-axis harvest close under a status the CPU checker does not
        # recognise, so the axis is DERIVED from the receipt the same way every other
        # fact here is. An axis the receipt does not state stays generic rather than
        # guessing, because a wrong axis label is the one thing worse than no label.
        axis = str(result.get("score_axis") or "").strip().lower() if isinstance(result, dict) else ""
        if not passed:
            status = "completed_modal_auth_eval_harvested_not_passed"
        elif axis == "contest_cuda":
            status = "completed_contest_cuda_exact_eval_harvested"
        elif axis == "contest_cpu":
            status = "completed_contest_cpu_exact_eval_harvested"
        else:
            status = "completed_modal_auth_eval_harvested"
        note = canonical_terminal_claim_notes(result, call_id)
        # The row must key on the SAME instance/job id the dispatch claim opened with,
        # or the compliance checker finds a terminal row with no active predecessor and
        # reds `dispatch_claim_prior_active_row`. The call_id fallback preserves the
        # pre-2026-09-04 behaviour for callers that do not pass one.
        job_id = instance_job_id or call_id
        terminal_dispatch_claim(
            repo_root=repo_root,
            spec=DispatchClaimSpec(
                lane_id=lane_id,
                instance_job_id=job_id,
                agent="MAIN",
                platform="modal",
                notes=note,
            ),
            status=status,
            notes=note,
        )
        return f"CLAIM CLOSED: {lane_id} / {job_id} -> {status}"
    except Exception as exc:  # never fail a real harvest on bookkeeping
        (out_dir / "CLAIM_CLOSE_FAILED.json").write_text(
            json.dumps(
                {
                    "schema": "modal_harvest_claim_close_failed.v1",
                    "lane_id": lane_id,
                    "call_id": call_id,
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:500],
                    "consequence": (
                        "the dispatch claim for this lane is still ACTIVE and will read as a "
                        "phantom; close it with tools/claim_lane_dispatch.py claim --force "
                        "--status completed_contest_cuda_exact_eval_harvested"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"CLAIM CLOSE FAILED ({type(exc).__name__}): {exc}"


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
        "--instance-job-id",
        default=None,
        help="instance/job id the dispatch claim was opened with; the terminal claim row "
        "keys on it so the compliance checker can see the active predecessor "
        "(default: the call_id, the pre-2026-09-04 behaviour)",
    )
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
    if REPO_SRC not in sys.path:
        sys.path.insert(0, REPO_SRC)
    from tac.deploy.modal.call_id_ledger import update_call_id_outcome
    from tac.deploy.modal.result_json import dump_modal_result_json

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
        # `json.dumps(r, ..., default=str)` wrote Python bytes-REPRS here: Modal
        # returns `artifacts` as dict[str, bytes], and `str(b"...")` produced a
        # receipt that raises JSONDecodeError for json.load. It damaged every
        # arm's MODAL_REMOTE_RESULT.json, the frontier's (ddm_jg5) included.
        # The canonical projection decodes UTF-8 bytes to text, base64s the
        # rest, and RECORDS which transform each path received.
        dump_modal_result_json(result_path, r)
        # The remote result carries the two facts the spend ledger needs and
        # this poller has always dropped: MEASURED billed wall-clock and the
        # hardware it ran on. Without them 49 of 63 cap-window calls carried
        # neither cost nor elapsed, so the >=$20 envelope could only be
        # answered from memo prose. Record what is measured; never synthesise
        # cost from a rate table (the one row where a real cost_actual_usd and
        # an elapsed both exist proved a published-rate estimate 2.8x high).
        #
        # ``lane_id`` is threaded here (rv13 F7). The ``dispatched`` row carries
        # it; the ``harvested`` row used to drop it, so a LANE-KEYED reconcile
        # could not see any completed fire and classified every one of them as a
        # "provable phantom". That is not hypothetical: ck2's terminal row states
        # `stale_superseded_reconciled_no_live_call` for a call that had already
        # harvested rc=0 under that exact lane 1h52m earlier, and two claims had
        # to be closed by hand with --force. The scored facts go in for the same
        # reason: the last 20 harvested rows recorded THAT a harvest happened and
        # never WHAT it scored.
        outcome_facts = _harvest_outcome_facts(r)
        update_call_id_outcome(
            call_id=args.call_id,
            status="harvested",
            rc=0,
            agent="MAIN",
            lane_id=args.lane_id,
            elapsed_seconds=_result_elapsed_seconds(r),
            gpu=_result_gpu(r),
            harvest_result={"result_path": str(result_path)},
            **outcome_facts,
        )
        if not args.no_anchor_mirror:
            write_anchor_mirror(
                r,
                source_receipt=result_path,
                label=args.mirror_label or out.name,
                lane_id=args.lane_id,
                out_dir=out,
            )
        close_note = _close_terminal_claim(
            lane_id=args.lane_id,
            call_id=args.call_id,
            result=r,
            out_dir=out,
            instance_job_id=args.instance_job_id,
        )
        if close_note:
            print(close_note)
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
