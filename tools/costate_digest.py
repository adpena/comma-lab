#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""costate_digest — the #247 costate controller as a CORE SENSE ORGAN, agent-native.

Operator NON-NEGOTIABLE 2026-07-07 (verbatim): "We must ensure the costate controller
does not require human or manual activation or use and that it is agent native and a
core sense organ and actuator the agent always knows about and uses where and how
optimal and appropriate."

ONE command that renders the controller's current SENSE+DECIDE state as a compact
(~25-line) agent-readable digest. It is AUTO-SURFACED (no memory or manual step needed):
  * SessionStart hook (.claude/settings.json) — stdout injected as session context;
  * the .claude/skills/witness-status skill (canonical check-in) runs it too;
  * the design doc for going deeper is printed in the footer.

READ-ONLY + FAIL-OPEN: every section degrades to "unavailable" — this tool NEVER
crashes a session start and NEVER writes anything. Target wall-clock < 5s.

═══ THE ACTUATION BOUNDARY (binding; do not misread "agent-native" as autonomy) ═══
AUTONOMOUS scope (no GO needed): advisory recommendations, lever-queue (duty-to-
measure) ranking, event-curriculum CONDITION INPUTS, and surfacing this digest.
OPERATOR-GO scope (CONTAINMENT non-negotiable, unchanged): heavy/paid launches,
stopping a live run, and ANY config change to a live run. The shadow controller
package (tac.witness_control) structurally cannot actuate (source-scan-tested).
═══════════════════════════════════════════════════════════════════════════════════

Usage:
  .venv/bin/python tools/costate_digest.py            # human digest
  .venv/bin/python tools/costate_digest.py --json     # machine-readable
  .venv/bin/python tools/costate_digest.py --session-start  # hook mode (always rc 0)
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))
# The historical witness artifact contract is imported lazily only inside the
# legacy fallback.  A complete DDM fleet never initializes that lineage.
_POINTER_JSON = _REPO / ".omx" / "state" / "canonical_frontier_pointer.json"
_ARM_NEXT_IF_RESUMED = _REPO / ".omx" / "state" / "codex_arm_queue.next_if_resumed.jsonl"
_DAG_GLOB = str(_REPO / ".omx" / "research" / "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md")
_DESIGN_DOC = ".omx/research/costate_controller_design_20260705.md"
_DISPATCH_DOC = ".omx/research/organ_regime_conditional_dispatch_436_20260711.md"
_REVIEW_COUNTER = _REPO / ".omx" / "state" / "review_counter.jsonl"
_DUTY_TOP_N = 6
_SHADOW_STALE_S = 2 * 3600.0

# --- SessionStart line budget (ddm_gh2, 2026-07-31) -------------------------
# MEASURED before this change: the session-start digest was 10,268 B (~2,570
# tokens), paid on EVERY session start AND every compaction, of which 6,085 B
# were DDM-* lines and 3,851 B the main-hot-state block.  The prefixes below
# carry ACCOUNTING / LAW / HISTORY detail: real signal, but not signal that
# changes MAIN's NEXT MOVE at t=0, and every byte of it is one `--full` away.
#
# Decision-relevance test applied per prefix: "would MAIN pick a different next
# action without this line?"  KEPT (not listed here): the pointer, the model
# stamp, live-run liveness/cadence, the duty-queue head, the schedulable
# allocation, blockers/validity/staleness, and the actuation BOUNDARY.
#
# FAIL-TOWARD-VISIBLE: this is a DROP-list, not a keep-list.  Any line whose
# prefix is not named here — including one a future author adds — stays VISIBLE
# by default, so the budget can never silently swallow new signal.  That is the
# opposite polarity from a keep-list and is deliberate (anti-forgetfulness).
_HOOK_DETAIL_PREFIXES = (
    "DDM-lambda",             # rho/NDCG model-fit accounting
    "DDM-arc",                # 15-row arc status board (mostly SETTLED)
    "DDM-band",               # regime narration
    "DDM-laws",               # measured laws (also carried in MEMORY.md)
    "DDM-joint",              # allocator table (head is re-emitted below)
    "DDM-parents",            # parent-band accounting
    "DDM-pending",            # commitment accounting
    "DDM-owners",             # 29-row owner queue (self-truncating already)
    "DDM-rv1",                # reactivation accounting
    "DDM-voi",                # VOI ranking (overlaps DDM-alloc)
    "DDM-campaign",           # campaign accounting + evidence roll-up
    "DDM-CO5",                # CO5 accounting
    "DDM-batch-local",        # batch-local telemetry detail
    "DDM-pose-watch",         # pose verdict-count detail
)
# Hot-state manifest: cap PER LINE, not just per line-COUNT.  The manifest holds
# free-text paragraphs (measured: 1,047 / 1,011 / 635 chars), so the existing
# max_lines=40 cap bounded nothing that mattered — a line budget on a file with
# mega-lines is a cap that looks binding and is not.
_HOOK_HOT_STATE_MAX_LINE_CHARS = 240
# section_ncde cache: {run_dir_str: (log_signature, line, data)} — keyed on run.log
# (mtime_ns, size) so a re-call within the same session reuses the fit instead of re-probing.
_NCDE_CACHE: dict[str, tuple[tuple, str | None, dict | None]] = {}
# section_verdict_trend cache: same run.log-signature keying as _NCDE_CACHE.
_VERDICT_TREND_CACHE: dict[str, tuple[tuple, str | None, dict | None]] = {}
_DDM_CAMPAIGN_RESULTS_ROOTS: tuple[Path, ...] = (
    Path("/Volumes/VertigoDataTier/pact/experiments/results"),
    Path("/Volumes/APDataStore/pact/experiments/results"),
    _REPO / "experiments" / "results",
)
_DDM_CAMPAIGN_OBSERVABILITY_SCHEMA = "ddm_campaign_run_observability.v1"
_DDM_CAMPAIGN_ROW_SCHEMA = "ddm_campaign_observability_row.v1"


def _ddm_pose_watch_deriver():
    """Load the leaf law without importing the heavyweight equation registry.

    The session-start digest must stay dependency-light.  Importing the
    ``tac.canonical_equations`` package registers the full equation fleet and
    initializes optional scientific dependencies, so load this exact source
    module directly while retaining one implementation of the law.
    """

    path = (
        _REPO
        / "src"
        / "tac"
        / "canonical_equations"
        / "ddm_pose_finish_engagement_watch_20260725.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_costate_ddm_pose_finish_engagement_watch_20260725",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical DDM pose-watch law: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.derive_pose_finish_engagement_watch


def _load_codex_arm_queue():
    """The arm-queue tool, for its retraction resolver. ONE implementation, not two.

    Loaded by path rather than re-implemented here: a second copy of the
    supersede-resolution rule would drift from the producer's, and the first
    symptom of that drift would be a retracted fire order quietly coming back.
    """
    path = _REPO / "tools" / "codex_arm_queue.py"
    spec = importlib.util.spec_from_file_location("_costate_codex_arm_queue", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load codex arm queue: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _last_jsonl_row(path: Path) -> dict | None:
    """Last parseable JSON row of a JSONL file (lenient; None on any failure)."""
    try:
        last = None
        with path.open("rb") as fh:
            for raw in fh:
                if raw.strip():
                    last = raw
        row = json.loads(last) if last else None
        return row if isinstance(row, dict) else None
    except Exception:
        return None


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


# ─────────────────────────── sections (each fail-open) ───────────────────────────
# The submittable original-work contest-CPU custody anchor (PR110-lineage recode). The pointer
# JSON's ``our_local_frontier_contest_cpu`` currently records the BORROWED PR128-on-PR110 bank
# (a lower but NON-SUBMISSION exact-CPU row); this anchor is the submittable frontier tracked in
# the committed council routing card + git-log ("pointer 0.19108 UNMOVED"). Used only as a
# fallback while the SoT's CPU frontier is the known bank (sha match); once a submittable row
# lands in the pointer JSON its sha will not match and the digest reads the JSON value directly.
_SUBMITTABLE_CONTEST_CPU_SCORE = 0.1910828242  # HISTORICAL_SCORE_LITERAL_OK: submittable-custody anchor, committed council routing card + git-log pointer 0.19108
_NON_SUBMISSION_BANK_SHA8 = "196acd18"  # PR128-on-PR110 borrowed NON-SUBMISSION bank (memory: banked 0.18804)


def section_pointer() -> tuple[str, dict]:
    """AMENDMENT 1 (means-as-ends firewall, operating manual §8.1): the END first.

    AMENDMENT 2 (ddm_co6, 2026-07-28): lead with the SUBMITTABLE original-work custody anchor +
    the effective score-to-beat bar, and label the pointer JSON's contest-CPU frontier as the
    borrowed NON-SUBMISSION bank when its archive sha matches. The prior line led with 0.18804 —
    the bank — presenting a borrowed non-submission row as our frontier ("everything below is
    means"), which conflates the defensive bank with submittable progress toward the goal.
    """
    try:
        d = json.loads(_POINTER_JSON.read_text())
        cpu = d.get("our_local_frontier_contest_cpu") or {}
        cpu_score = float(cpu["score"])
        since = str(cpu.get("measured_at_utc", ""))[:10] or "?"
        sha8 = str(cpu.get("archive_sha256", ""))[:8]
        bar = (d.get("effective_frontier") or {}).get("score")
        bar_s = f"{float(bar):g}" if isinstance(bar, (int, float)) else "?"
        is_bank = sha8.startswith(_NON_SUBMISSION_BANK_SHA8)
        if is_bank:
            submittable = _SUBMITTABLE_CONTEST_CPU_SCORE
            line = (
                f"POINTER submittable {submittable:.7g} [contest-CPU custody] UNMOVED · "
                f"effective bar {bar_s} · {cpu_score:.5f} = NON-SUBMISSION bank "
                f"(borrowed PR128-on-PR110, sha {sha8}, {since}) — everything below is means."
            )
        else:
            submittable = cpu_score  # the SoT CPU frontier is itself submittable
            line = (
                f"POINTER {cpu_score:.5f} [contest-CPU] UNMOVED since {since} · "
                f"effective bar {bar_s} — everything below is means."
            )
        return line, {
            "score": submittable,
            "submittable_contest_cpu": submittable,
            "effective_bar": bar,
            "local_cpu_frontier": {"score": cpu_score, "sha8": sha8, "since": since},
            "non_submission_bank": is_bank,
            "axis": "contest-CPU",
        }
    except Exception as exc:
        return (f"POINTER: unavailable ({type(exc).__name__}) — read .omx/state/canonical_frontier_pointer.json"), {
            "error": str(exc)
        }


def section_live_run() -> tuple[str, dict, Path | None]:
    """Live-run state via the canonical check-in tool (imported, not duplicated)."""
    try:
        import witness_checkin as wc

        procs = wc.find_trainer_procs()
        run_dir, proc, how = wc.pick_run_dir(procs, wc.RESULTS_DEFAULT)
        if run_dir is None or not run_dir.is_dir():
            return "live run: NONE found (no witness run dirs)", {"alive": False}, None
        status = wc.collect_status(run_dir, proc, wc.STALE_AFTER_S_DEFAULT)
        status["discovery"] = how
        return wc.human_line(status), status, run_dir
    except Exception as exc:
        return f"live run: unavailable ({type(exc).__name__}: {exc})", {"error": str(exc)}, None


def section_live_ddm() -> tuple[list[str], dict | None]:
    """Primary live DDM SENSE surface.

    A complete, schema-checked DDM receipt fleet supersedes the retired
    witness-training run lookup.  Partial or broken input stays fail-open and
    is never blended into a live claim.
    """
    try:
        from tac.ddm_costate_organ import (
            build_live_ddm_costate,
            digest_lines,
        )

        report = build_live_ddm_costate(repo_root=_REPO)
        return digest_lines(report), report
    except Exception as exc:
        return [f"DDM-LIVE unavailable ({type(exc).__name__}: {exc})"], {
            "available": False,
            "status": "FAIL_OPEN",
            "reason": f"{type(exc).__name__}: {exc}",
            "actuation": "NONE",
            "score_claim": False,
        }


def discover_latest_ddm_campaign_run(
    results_roots: tuple[Path, ...] | list[Path] | None = None,
) -> Path | None:
    """Find the latest structurally governed DDM campaign by directory mtime.

    The roots are stable storage tiers; the run name is never encoded here.
    Every candidate must first pass the canonical witness-run predicate and
    then the typed DDM run-identity schema check.
    """

    from tac import witness_run_artifacts as wra

    candidates: list[Path] = []
    for root in results_roots or _DDM_CAMPAIGN_RESULTS_ROOTS:
        if not root.is_dir():
            continue
        try:
            children = root.iterdir()
        except OSError:
            continue
        for child in children:
            if not wra.is_run_dir(child):
                continue
            identity = _last_jsonl_row(child / wra.DDM_RUN_IDENTITY_JSON)
            if identity is None:
                try:
                    identity = json.loads(
                        (child / wra.DDM_RUN_IDENTITY_JSON).read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
            if (
                isinstance(identity, dict)
                and identity.get("schema") == "ddm_joint_descent_run_identity.v1"
            ):
                candidates.append(child)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _campaign_schedule(run_dir: Path, receipt: dict) -> tuple[dict, dict]:
    """Load the sealed schedule from the final receipt or the launch ticket."""

    receipt_schedule = receipt.get("schedule")
    if isinstance(receipt_schedule, dict) and all(
        key in receipt_schedule
        for key in ("measured_seconds_per_step", "pose_finish_engage", "stages")
    ):
        return receipt_schedule, {
            "status": "FINAL_RECEIPT_COMPLETE_SCHEDULE",
            "final_receipt_path": str(run_dir / "full_run_receipt.json"),
        }
    launch = _json_object(run_dir / "launch_manifest.json")
    argv = launch.get("argv")
    if not isinstance(argv, list) or "--ticket" not in argv:
        raise ValueError("DDM launch manifest lacks typed ticket argv")
    ticket_index = argv.index("--ticket") + 1
    if ticket_index >= len(argv):
        raise ValueError("DDM launch manifest has an empty typed ticket argv")
    ticket = Path(str(argv[ticket_index]))
    if ticket.is_absolute():
        ticket_path = ticket
    else:
        ticket_path = (_REPO / ticket).resolve()
        ticket_path.relative_to(_REPO)
    typed = _json_object(ticket_path)
    ticket_schedule = (typed.get("semantic_program") or {}).get("full_run_schedule")
    if not isinstance(ticket_schedule, dict):
        raise ValueError("DDM typed ticket lacks full_run_schedule")
    ticket_comparator = (typed.get("execution_custody") or {}).get(
        "banked_r1_comparator"
    )
    if not isinstance(ticket_comparator, dict):
        raise ValueError("DDM typed ticket lacks banked R1 comparator custody")
    if receipt_schedule is not None and not isinstance(receipt_schedule, dict):
        raise ValueError("DDM final receipt schedule is not a mapping")
    # The final receipt intentionally carries only the runtime-consumed subset;
    # retain sealed projection/cadence fields from the ticket and overlay the
    # measured runtime copy where present.
    schedule = {**ticket_schedule, **(receipt_schedule or {})}
    return schedule, {
        "status": (
            "HASH_SEALED_TYPED_TICKET_PLUS_FINAL_RECEIPT"
            if receipt_schedule
            else "HASH_SEALED_TYPED_TICKET"
        ),
        "ticket_path": str(ticket_path),
        "banked_r1_comparator": ticket_comparator,
        "final_receipt_path": (
            str(run_dir / "full_run_receipt.json") if receipt_schedule else None
        ),
    }


def read_ddm_campaign_observability(run_dir: Path) -> dict:
    """Read one DDM run without writing it or invoking a scorer."""

    telemetry_paths = sorted((run_dir / "telemetry").glob("step*.json"))
    telemetry = [
        row
        for row in (_json_object(path) for path in telemetry_paths)
        if row.get("schema") == "ddm_joint_descent_full_run_step.v1"
    ]
    verdict_candidates: list[tuple[int, int, Path, dict]] = []
    for path in (run_dir / "verdicts").glob("*.json"):
        row = _json_object(path)
        if (
            row.get("schema") == "ddm_joint_descent_chunked_stage_verdict.v1"
            and row.get("num_pairs") == 600
            and isinstance(row.get("d_seg"), (int, float))
            and isinstance(row.get("d_pose"), (int, float))
        ):
            verdict_candidates.append(
                (int(row.get("global_step", 0)), path.stat().st_mtime_ns, path, row)
            )
    if not verdict_candidates:
        raise ValueError("DDM campaign has no exact n600 verdict")
    latest_step, _mtime, latest_verdict_path, latest_verdict = max(
        verdict_candidates,
        key=lambda item: (item[0], item[1]),
    )
    receipt_path = run_dir / "full_run_receipt.json"
    receipt = _json_object(receipt_path) if receipt_path.is_file() else {}
    schedule, schedule_source = _campaign_schedule(run_dir, receipt)
    stage_rows = schedule.get("stages") or []
    if not isinstance(stage_rows, list) or not stage_rows:
        raise ValueError("DDM schedule has no stages")
    total_steps = sum(int(stage["maximum_steps"]) for stage in stage_rows)
    verdict_interval = int(stage_rows[0]["verdict_interval_steps"])
    expected_seconds = float(schedule["measured_seconds_per_step"])

    step_seconds = [float(row["seconds"]) for row in telemetry]
    measured_cadence = sum(step_seconds) / len(step_seconds) if step_seconds else None
    accepted_checkpoints = sorted((run_dir / "checkpoints").glob("*_accepted_global*.npz"))
    geometry_events = sorted((run_dir / "telemetry").glob("geometry_*_cured.json"))
    local_dseg_delta = sum(
        float(row["final"]["d_seg"]) - float(row["initial"]["d_seg"])
        for row in telemetry
    )
    local_dpose_delta = sum(
        float(row["final"]["d_pose"]) - float(row["initial"]["d_pose"])
        for row in telemetry
    )
    pose_state = latest_verdict.get("pose_finish_engage_state")
    if not isinstance(pose_state, dict):
        pose_state = receipt.get("pose_finish_engage_state")
    if not isinstance(pose_state, dict):
        pose_state = {}
    exact_verdict_steps = pose_state.get("exact_verdict_steps") or []
    pose_gate = schedule.get("pose_finish_engage")
    if not isinstance(pose_gate, dict):
        raise ValueError("DDM schedule lacks typed pose-finish engage gate")
    comparator = receipt.get("banked_r1_comparator") or schedule_source.get(
        "banked_r1_comparator"
    )
    if not isinstance(comparator, dict) or not isinstance(
        comparator.get("score_contribution"), (int, float)
    ):
        raise ValueError("DDM campaign lacks banked R1 comparator score custody")
    fallback_contribution = float(comparator["score_contribution"])

    pose_watch = _ddm_pose_watch_deriver()(
        verdict_interval_steps=verdict_interval,
        ema_span=int(pose_gate["ema_span"]),
        hysteresis=int(pose_gate["hysteresis"]),
        settle_window=int(pose_gate["settle_window"]),
        observed_exact_verdicts=len(exact_verdict_steps),
        fallback_score_contribution=fallback_contribution,
    )
    remaining_steps = max(0, total_steps - latest_step)
    eta_seconds = (
        remaining_steps * measured_cadence
        if measured_cadence is not None
        else None
    )
    campaign_blocker = receipt.get("campaign_blocker")
    run_status = (
        "STOPPED_WITH_TYPED_BLOCKER"
        if campaign_blocker
        else "FINAL_RECEIPT_COMPLETE"
        if receipt
        else "RECEIPT_PENDING_NO_PROCESS_LIVENESS_CLAIM"
    )
    rows = [
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "latest_exact_n600_verdict",
            "epistemic_status": "MEASURED",
            "evidence_axis": latest_verdict.get("evidence_axis"),
            "global_step": latest_step,
            "d_seg": float(latest_verdict["d_seg"]),
            "d_pose": float(latest_verdict["d_pose"]),
            "source_path": str(latest_verdict_path),
            "score_claim": False,
        },
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "accepted_steps_and_cadence",
            "epistemic_status": "MEASURED",
            "accepted_step_count": len(accepted_checkpoints),
            "latest_accepted_checkpoint": (
                str(accepted_checkpoints[-1]) if accepted_checkpoints else None
            ),
            "telemetry_step_count": len(telemetry),
            "measured_seconds_per_step": measured_cadence,
            "sealed_seconds_per_step": expected_seconds,
            "measured_minus_sealed_seconds": (
                measured_cadence - expected_seconds
                if measured_cadence is not None
                else None
            ),
            "score_claim": False,
        },
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "cumulative_batch_local_trace",
            "epistemic_status": "ADVISORY_BATCH_LOCAL",
            "not_n600_verdict": True,
            "delta_d_seg_sum_final_minus_initial": local_dseg_delta,
            "delta_d_pose_sum_final_minus_initial": local_dpose_delta,
            "telemetry_step_count": len(telemetry),
            "score_claim": False,
        },
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "pose_finish_engagement_watch",
            "epistemic_status": "DERIVED_FROM_SEALED_GATE_CONSTANTS",
            "observed_classification": pose_state.get("classification"),
            "observed_exact_verdict_count": len(exact_verdict_steps),
            "observed_exact_verdict_steps": list(exact_verdict_steps),
            "watch": pose_watch,
            "score_claim": False,
        },
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "geometry_cure_events",
            "epistemic_status": "MEASURED",
            "event_count": len(geometry_events),
            "source_glob": "telemetry/geometry_*_cured.json",
            "score_claim": False,
        },
        {
            "schema": _DDM_CAMPAIGN_ROW_SCHEMA,
            "row_id": "schedule_endpoint_eta",
            "epistemic_status": "DERIVED_FROM_MEASURED_CADENCE_AND_SEALED_SCHEDULE",
            "current_global_step": latest_step,
            "sealed_total_steps": total_steps,
            "remaining_steps": remaining_steps,
            "measured_cadence_seconds_per_step": measured_cadence,
            "eta_seconds": eta_seconds,
            "eta_hours": eta_seconds / 3600.0 if eta_seconds is not None else None,
            "counterfactual_after_governed_stop": bool(campaign_blocker),
            "score_claim": False,
        },
    ]
    return {
        "schema": _DDM_CAMPAIGN_OBSERVABILITY_SCHEMA,
        "available": True,
        "status": run_status,
        "campaign_blocker": campaign_blocker,
        "run_dir": str(run_dir),
        "discovery": {
            "law": "filter with tac.witness_run_artifacts.is_run_dir; select max directory mtime_ns",
            "run_name_hardcoded": False,
            "selected_directory_mtime_ns": str(run_dir.stat().st_mtime_ns),
        },
        "schedule_source": schedule_source,
        "rows": rows,
        "actuation": "NONE",
        "execution_allowed": False,
        "score_claim": False,
        "main_landing_review_required": True,
    }


def section_ddm_campaign_run(
    results_roots: tuple[Path, ...] | list[Path] | None = None,
) -> tuple[list[str], dict]:
    """Typed DDM campaign observability for digest and machine consumers."""

    try:
        run_dir = discover_latest_ddm_campaign_run(results_roots)
        if run_dir is None:
            return ["DDM-run: unavailable (no canonical campaign run dir)"], {
                "schema": _DDM_CAMPAIGN_OBSERVABILITY_SCHEMA,
                "available": False,
                "status": "NO_CANONICAL_CAMPAIGN_RUN",
                "actuation": "NONE",
                "score_claim": False,
            }
        report = read_ddm_campaign_observability(run_dir)
        by_id = {row["row_id"]: row for row in report["rows"]}
        verdict = by_id["latest_exact_n600_verdict"]
        cadence = by_id["accepted_steps_and_cadence"]
        local = by_id["cumulative_batch_local_trace"]
        pose = by_id["pose_finish_engagement_watch"]
        eta = by_id["schedule_endpoint_eta"]
        measured_cadence = cadence["measured_seconds_per_step"]
        eta_hours = eta["eta_hours"]
        measured_cadence_text = (
            f"{measured_cadence:.3f}" if measured_cadence is not None else "pending"
        )
        eta_hours_text = f"{eta_hours:.2f}" if eta_hours is not None else "pending"
        lines = [
            (
                f"DDM-run: {report['status']} step {verdict['global_step']} "
                f"d_seg {verdict['d_seg']:.9f} d_pose {verdict['d_pose']:.9f} "
                "[macOS-CPU advisory exact n600]"
            ),
            (
                f"DDM-cadence: {cadence['accepted_step_count']} accepted | "
                f"{measured_cadence_text}s/step measured vs "
                f"{cadence['sealed_seconds_per_step']:.3f}s sealed | "
                f"{eta_hours_text}h counterfactual endpoint ETA"
            ),
            (
                f"DDM-batch-local: sum delta d_seg {local['delta_d_seg_sum_final_minus_initial']:+.9f} "
                f"over {local['telemetry_step_count']} steps "
                "[ADVISORY_BATCH_LOCAL; NOT n600]"
            ),
            (
                f"DDM-pose-watch: {pose['observed_classification']} "
                f"{pose['observed_exact_verdict_count']} exact verdicts; "
                f"conditional window verdict "
                f"{pose['watch']['candidate_engagement_verdict_index_one_based']}-"
                f"{pose['watch']['settled_engagement_verdict_index_one_based']} "
                f"(step {pose['watch']['candidate_engagement_global_step']}-"
                f"{pose['watch']['settled_engagement_global_step']})"
            ),
        ]
        return lines, report
    except Exception as exc:
        return [f"DDM-run: unavailable ({type(exc).__name__}: {exc})"], {
            "schema": _DDM_CAMPAIGN_OBSERVABILITY_SCHEMA,
            "available": False,
            "status": "FAIL_OPEN",
            "reason": f"{type(exc).__name__}: {exc}",
            "actuation": "NONE",
            "score_claim": False,
        }


def section_annulus(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Annulus/convergence headline from the run's annulus_live.jsonl (#333 SENSE)."""
    if run_dir is None:
        return None, None
    row = _last_jsonl_row(run_dir / "annulus_live.jsonl")
    if not row:
        return None, None
    try:
        ann = row.get("annulus") or {}
        lane = (ann.get("per_class_annulus_flip_frac") or {}).get("1")
        parts = [
            f"annulus: ep{row.get('epoch')} d_seg {ann.get('overall_d_seg'):.6f}",
            f"annulus mass share {100 * ann.get('annulus_flip_mass_share', 0):.1f}%",
        ]
        if lane is not None:
            parts.append(f"lane(cls1) flip {100 * float(lane):.1f}%")
        parts.append(f"[{row.get('seg_form')}, advisory]")
        return " | ".join(parts), {"epoch": row.get("epoch"), "annulus": ann}
    except Exception:
        return None, None


def section_shadow(run_dir: Path | None) -> tuple[list[str], dict | None]:
    """Latest shadow-observer DECIDE state (classification + pending recommendations)."""
    if run_dir is None:
        return [], None
    from tac import witness_run_artifacts as _wra

    path = run_dir / _wra.COSTATE_JSONL
    row = _last_jsonl_row(path)
    if row is None and (run_dir / "run.log").is_file():
        # No sidecar yet but telemetry exists: compute ONE read-only in-memory report.
        try:
            from tac.witness_control import build_shadow_report, load_run_inputs

            row = build_shadow_report(load_run_inputs(run_dir)).to_row()
        except Exception:
            row = None
    if not row:
        return ["costate-shadow: no rows yet (observer will populate costate_shadow.jsonl)"], None
    if not isinstance(row.get("costate_organ_v2"), dict):
        # Backward-compatible readback for pre-v2 persisted sidecars. This is an
        # in-memory additive field only; the sacred run directory is not rewritten.
        try:
            from tac.witness_control.costate_organ_v2 import aggregate_readback
            from tac.witness_control.shadow_controller import parse_launch_sh_flags

            flags = {}
            launch = run_dir / "launch.sh"
            if launch.is_file():
                flags = parse_launch_sh_flags(launch.read_text(errors="replace"))
            row = dict(row)
            row["costate_organ_v2"] = aggregate_readback(
                row.get("state") or {}, flags=flags, maturity="_dev")
        except Exception as exc:
            row = dict(row)
            row["costate_organ_v2"] = {
                "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}",
                "actuation": "NONE", "score_claim": False,
            }
    lines: list[str] = []
    try:
        age = max(0.0, time.time() - path.stat().st_mtime) if path.exists() else None
        cls = (row.get("classification") or {}).get("classification", "?")
        head = f"costate-shadow: ep{row.get('epoch')} class={str(cls).upper()}"
        if age is not None:
            head += f" ({_fmt_age(age)} old)"
        recs = row.get("recommendations") or []
        for r in recs[:2]:
            pd = r.get("predicted_dS")
            pd_s = f"{pd:+.4f}" if isinstance(pd, (int, float)) else "?"
            # P2 (design_philosophies_eightfold): EVERY comparison carries its noise floor. The rec's
            # predicted_dS_band IS the composed noise floor. No band -> cited Δ lacks a floor -> INSTANCE
            # only (never above advisory). A band spanning 0 -> Δ within noise -> INSTANCE. Else state Δ/floor.
            # P9 (proxies are poison, use the thing itself): the band gates predicted_dS in the SAME units
            # (ΔS) as the Δ — never a proxy-unit floor (the R7 category-error lesson).
            band = r.get("predicted_dS_band")
            if not (
                isinstance(band, (list, tuple)) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)
            ):
                floor_tag = " [INSTANCE — no noise floor]"
            elif band[0] <= 0.0 <= band[1]:
                floor_tag = f" [INSTANCE — Δ within noise floor [{band[0]:+.4f},{band[1]:+.4f}]]"
            else:
                floor_tag = f" (floor [{band[0]:+.4f},{band[1]:+.4f}])"
            lines.append(f"  rec: {r.get('action')} ΔS {pd_s}/{r.get('horizon_epochs')}ep{floor_tag}")
        if not recs:
            lines.append("  rec: (none identifiable)")
        factor = row.get("factorized_adjoint")
        if isinstance(factor, dict):
            fac = factor.get("factorization") or {}
            exact = fac.get("exact") or {}
            derived = fac.get("derived") or {}
            learned = factor.get("learned_residual") or {}
            decision = factor.get("decision") or {}
            zero = exact.get("certified_zero_weight_camera_frac")
            ratio = derived.get("road_lane_gain_only_lambda_ratio_vs_other_median")
            zero_s = f"{100 * zero:.1f}%" if isinstance(zero, (int, float)) else "?"
            ratio_s = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "?"
            lines.append(
                "  factorized-adjoint: " + str(factor.get("admission", "?"))
                + f" | EXACT head-rank={exact.get('head_rank', '?')} ker(A)-zero={zero_s}"
                + f" | DERIVED lambda_Road-Lane={ratio_s} other-major median"
                + f" | LEARNED={learned.get('n_parameters', '?')} shared scalars"
                + f" ({learned.get('amplitude_gate', 'not-fit')})")
            confidence = ((factor.get("recommendation_candidate") or {}).get("confidence")
                          or factor.get("validation_scope"))
            if confidence:
                lines.append("  factorized-confidence: " + str(confidence))
            if decision:
                pd = decision.get("predicted_dS")
                pd_s = f"{pd:+.4f}" if isinstance(pd, (int, float)) else "?"
                lines.append(
                    f"  factorized-DECIDE: DeltaS {pd_s}/{decision.get('horizon_epochs')}ep; "
                    + str(decision.get("why", "reason unavailable")))
        organ_v2 = row.get("costate_organ_v2")
        if isinstance(organ_v2, dict):
            debt = organ_v2.get("score_debt") or {}
            vis = organ_v2.get("visibility") or {}
            real = organ_v2.get("realizability") or {}
            app = organ_v2.get("apparatus") or {}
            debt_s = debt.get("total_s")
            debt_text = f"{debt_s:.6g}S" if isinstance(debt_s, (int, float)) else "unavailable"
            lines.append(
                "  costate-organ-v2: " + str(organ_v2.get("status", "?"))
                + f" | exact-gap={debt_text}"
                + f" | full-kernel-visible={vis.get('full_kernel_visible', '?')}"
                + f" | realization-design={real.get('design_anchor', '?')}"
                + f" | maturity={app.get('maturity', '_dev')}"
                + " | pair/site lambda=OWED")
            if app.get("bench_contaminated"):
                lines.append("  costate-organ-v2 apparatus REFUSED: "
                             + str(app.get("bench_reason")))
        events = row.get("event_advisories") or []
        if events and isinstance(events[0], dict):
            e = events[0]
            ep = ((e.get("morse_smale") or {}).get("derived") or {})
            ncde = e.get("ncde_344") or {}
            lines.append(
                "  event-intelligence: "
                + ("WARNING ACTIVE" if e.get("warning_active") else "next-boundary watch")
                + f" | critical pair={ep.get('critical_lambda_pair', '?')}"
                + f" lambda-ratio={ep.get('critical_lambda_ratio_vs_other_major_pair_median', '?')}"
                + f" | #344 fire={ncde.get('fire', False)} reason={ncde.get('reason', 'unavailable')}")
        lines.insert(0, head)
        if age is not None and age > _SHADOW_STALE_S:
            lines.append(f"  refresh: .venv/bin/python tools/costate_shadow_report.py --run-dir {run_dir} --write")
    except Exception as exc:
        lines = [f"costate-shadow: unavailable ({type(exc).__name__}: {exc})"]
    return lines, row


def format_ncde_line(row: dict) -> str:
    """Pure formatter for one Linear-NCDE hit->solve advisory row (#344). Unit-testable
    without touching real state. Surfaces: asymptote estimate + ETA-to-threshold +
    BASIN/HANDOFF verdict + the instrument-invalid guard state. Advisory-only wording."""
    ep = row.get("epoch")
    r2 = row.get("fit_r2")
    stable = row.get("stable")
    reason = str(row.get("reason", ""))
    fire = bool(row.get("fire"))
    asym = row.get("predicted_asymptote")  # log_d_seg space
    eta = row.get("eta_handoff_epochs")
    rem = row.get("remaining_descent_frac")
    invalid = "instrument-invalid" in reason or stable is False or (isinstance(r2, (int, float)) and r2 < 0.5)
    ep_s = f"{ep:.0f}" if isinstance(ep, (int, float)) else "?"
    r2_s = f"{r2:.2f}" if isinstance(r2, (int, float)) else "?"
    if invalid:
        tag = f"INSTRUMENT-INVALID (stable={stable}, r2={r2_s}) — no verdict"
    elif fire and reason.startswith("BASIN"):
        tag = "BASIN-FIRE (terminal SOLVE #341 admissible)"
    elif fire:
        tag = "HANDOFF-FIRE (#315 plateau-slope predicted soon)"
    else:
        tag = "NO-FIRE (still descending)"
    parts = [f"ncde-trajectory: ep{ep_s} d_seg {tag}"]
    detail: list[str] = []
    if not invalid:
        if isinstance(asym, (int, float)):
            detail.append(f"plateau d_seg~{math.exp(asym):.5f}")
        if isinstance(rem, (int, float)):
            detail.append(f"rem {100 * rem:.1f}% of travelled")
        if isinstance(eta, (int, float)):
            detail.append(f"ETA-handoff {eta:.0f}ep")
    head = " — ".join([parts[0], ", ".join(detail)]) if detail else parts[0]
    return head + " [advisory NON-PROMOTABLE]"


def section_ncde(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Linear-NCDE trajectory advisory (#344): the score-relevant d_seg hit->solve verdict
    for the live run. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is a
    tracked queue": read-only telemetry is not gate-able). Reuses the probe entry point
    (tools/ncde_trajectory_probe.run_probe) -- the math is NOT duplicated here. Fail-open
    (any error -> omit the row); omitted when d_seg verdict telemetry is too short (<8 pts,
    no advisory yet). Cheap (~0.01s measured) + cached by run.log signature."""
    if run_dir is None:
        return None, None
    try:
        log = run_dir / "run.log"
        if not log.is_file():
            return None, None
        st = log.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = _NCDE_CACHE.get(str(run_dir))
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        import ncde_trajectory_probe as ntp

        report = ntp.run_probe(run_dir, window=12, emit=False, do_backtest=False)
        adv = report.get("verdict_latest_advisory")
        if not adv:  # no d_seg advisory yet (short verdict telemetry) — omit, honestly
            _NCDE_CACHE[str(run_dir)] = (sig, None, None)
            return None, None
        line = format_ncde_line(adv)
        data = {"advisory": adv, "verdict_points": report.get("verdict_points"), "n_fires": report.get("n_fires")}
        _NCDE_CACHE[str(run_dir)] = (sig, line, data)
        return line, data
    except Exception:
        return None, None


def section_verdict_trend(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """VERDICT-TREND / TRAIN-VERDICT-DECOUPLING alarm (operator-catch 2026-07-09): the
    advisory verdict d_seg RISING while the train seg-loss descends — the false-green the
    scalar classifier misses. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is
    a tracked queue"). Fail-open (any error -> omit); omitted when there is no material
    rising trend. Cheap + cached by run.log signature (like section_ncde)."""
    if run_dir is None:
        return None, None
    try:
        log = run_dir / "run.log"
        if not log.is_file():
            return None, None
        st = log.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = _VERDICT_TREND_CACHE.get(str(run_dir))
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        from tac.witness_control import (
            format_verdict_trend_line,
            load_run_inputs,
            verdict_trend_alarm,
        )

        alarm = verdict_trend_alarm(load_run_inputs(run_dir).verdicts)
        if not alarm.fired():  # only surface when there is a real rising trend
            _VERDICT_TREND_CACHE[str(run_dir)] = (sig, None, None)
            return None, None
        line = format_verdict_trend_line(alarm)
        data = alarm.to_confound_alarm_row()
        _VERDICT_TREND_CACHE[str(run_dir)] = (sig, line, data)
        return line, data
    except Exception:
        return None, None


def section_pose_conditioning_gate(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """owed-1 POSE-FINISH CONDITIONING gate (SYNTHESIS_v3_v752 §A.4): surface the LOUD pose-DISENGAGED
    alarm (shipped banked R1 because the rolling-slope σ_min plateau never engaged) if present, else the
    latest conditioning-gate observer row. Read-only + score-neutral -> defaults ON. Fail-open (any
    error -> omit); omitted when the run has no pose-gate telemetry."""
    if run_dir is None:
        return None, None
    try:
        from tac.witness_control.sigma_min_plateau import format_gate_line, scan_run_for_pose_gate

        row = scan_run_for_pose_gate(run_dir)
        if not row:
            return None, None
        return format_gate_line(row), row
    except Exception:
        return None, None


def section_telemetry_binding(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """#404 P0 binding-vs-inert lever readback (tac.witness_control.telemetry_binding): amber
    grad-clip binding rate · chroma term share · pose-gate sensor liveness (the silent-crash class)
    · EMA-lag verdict-vs-live divergence · D27b terminal-band/solve-upon-basin trigger · TAIL
    endpoints. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is a tracked queue").
    Bounded tail read (cheap at SessionStart cadence); fail-open (any error -> omit)."""
    if run_dir is None:
        return None, None
    try:
        from tac.witness_control import telemetry_binding as _tb

        rows = _tb.load_run_rows(run_dir, tail_bytes=1_500_000)
        if not rows:
            return None, None
        audit = _tb.audit_rows(rows)
        return _tb.format_summary(audit), {
            k: audit.get(k) for k in ("amber", "chroma", "pose_gate", "ema_lag", "terminal_band")
        }
    except Exception:
        return None, None


def _duty_marker(r: dict) -> str:
    """Marker per ranked row: ~=un-built finding (missing wire) · ?=registered owed an estimate ·
    *=never-fired registered lever · ''=fired-but-unmeasured registered lever."""
    if not r.get("registered"):
        return "~"
    if r.get("est_delta_s") is None:
        return "?"
    return "*" if r.get("activation_state") == "never-fired" else ""


def _floor_marker(r: dict) -> str:
    """P8 floor marker (design_philosophies_eightfold): !FLOOR=target term AT its measured floor (buys ~0)
    · ^cap=est capped to the headroom-to-floor · ''=above floor / floor unmeasured. Score-neutral display."""
    st = r.get("floor_status")
    if st == "AT_FLOOR":
        return "!FLOOR"
    if st == "HEADROOM_CAPPED":
        return "^cap"
    return ""


def format_duty_to_measure_line(ranked: list[dict], top_n: int = _DUTY_TOP_N) -> str:
    """Pure formatter for the ranked duty-to-measure queue (unit-testable without touching real state).
    Renders each lever with its % of remaining descent and orphan marker; leads with the highest
    relative-value owed lever."""
    owed_n = sum(1 for r in ranked if r.get("in_duty_queue"))
    s_cur = ranked[0].get("s_current") if ranked else None
    tgt = ranked[0].get("s_target", 0.15) if ranked else 0.15

    def _cell(r: dict) -> str:
        pct = r.get("rel_sig_pct")
        tag = f" {pct:g}%" if pct is not None else " ?%"
        return f"{r['lever']}{_duty_marker(r)}{_floor_marker(r)}{tag}"

    top = ranked[:top_n]
    more = len(ranked) - len(top)
    anchor = f"pointer {s_cur:.5f}→{tgt:g}" if isinstance(s_cur, float) else "pointer unavailable"
    return (
        f"duty-to-measure ({owed_n} owed; ranked by % of remaining descent [P8 floor-aware], {anchor}; "
        f"*=never-fired ~=unbuilt ?=est-owed !FLOOR=at-floor ^cap=headroom-capped): "
        + ", ".join(_cell(r) for r in top)
        + (f" (+{more} more)" if more > 0 else "")
    )


def _live_term_current(annulus_data: dict | None) -> dict[str, float] | None:
    """Extract the live run's MEASURED current term value(s) for P8 floor-aware ranking. Today only
    ``d_seg`` is cleanly available (the annulus SENSE row's ``overall_d_seg``, an advisory MEASURED
    witness d_seg); d_pose/rate current-term stores are OWED (returned absent, honestly). Returns None
    when there is no live run (floors then surface as FLOOR_KNOWN_CURRENT_UNKNOWN — an owed measurement)."""
    if not annulus_data:
        return None
    try:
        d_seg = (annulus_data.get("annulus") or {}).get("overall_d_seg")
        if isinstance(d_seg, (int, float)) and d_seg == d_seg:  # not NaN
            return {"d_seg": float(d_seg)}
    except Exception:
        return None
    return None


def section_duty_to_measure(term_current: dict[str, float] | None = None) -> tuple[str, dict | None]:
    """Top-N owed DSL levers RANKED by relative significance — the fraction of the REMAINING descent
    to sub-0.15 each lever buys (est_delta_s / (pointer − 0.15)), **P8 floor-aware** (a lever whose
    target term is at its measured floor ranks ~0). This is the continual-learning fix for the recurring
    magnitude-dismissal bug: the CONTROLLER holds the value ranking, not the eyeball (CLAUDE.md "'Off' is
    a tracked queue" + "relative-not-absolute-significance-near-goal"). Markers: *=never-fired · ~=unbuilt
    · ?=owed-estimate · !FLOOR=at-floor · ^cap=headroom-capped.

    P1 (one-fact-one-store-one-key): the significance read is routed through
    ``duty_to_measure_ranked`` -> ``canonicalize_significance_keys`` (the ONLY significance-store reader);
    no raw-key read exists in the digest (audited 2026-07-09)."""
    try:
        from tac.witness_dsl.activation_ledger import duty_to_measure_ranked

        # reads the LIVE pointer + significance store (not hardcoded); term_current from live telemetry.
        ranked = duty_to_measure_ranked(term_current=term_current)
        line = format_duty_to_measure_line(ranked)
        s_cur = ranked[0].get("s_current") if ranked else None
        tgt = ranked[0].get("s_target", 0.15) if ranked else 0.15
        n_at_floor = sum(1 for r in ranked if r.get("floor_status") == "AT_FLOOR")
        n_capped = sum(1 for r in ranked if r.get("floor_status") == "HEADROOM_CAPPED")
        # The activation states in this ranking come from the ledger, so the ledger's BASIS
        # must travel with them (ddm_lr2, 2026-08-03). MEASURED: the ledger's only writer is
        # the RETIRED vehicle's launcher, so 31 governed live-vehicle launches produced zero
        # rows and every "*" never-fired marker above is an artifact of the writer, not a fact
        # about the lever. Printing the marker without the basis is how "178 never-fired" got
        # quoted as a backlog. Fail-open: a SENSE row must never break the digest.
        cov_d: dict | None = None
        try:
            from tac.witness_dsl.activation_ledger import ledger_coverage

            cov = ledger_coverage()
            cov_d = cov.to_dict()
            if cov.is_vacuous:
                line += (
                    f"\n  ⚠ activation states above are VACUOUS — {cov.vacuity_reason} "
                    f"(ledger covers {cov.levers_with_any_row}/{cov.known_levers} levers, "
                    f"last write {cov.last_write_utc})"
                )
        except Exception as exc:  # fail-open by contract: a SENSE row never breaks the digest
            line += f"\n  ⚠ activation-ledger basis unavailable ({type(exc).__name__})"
        return line, {
            "ranked_top": ranked[:_DUTY_TOP_N],
            "owed_registered": sum(1 for r in ranked if r.get("in_duty_queue")),
            "term_current": term_current,
            "n_at_floor": n_at_floor,
            "n_headroom_capped": n_capped,
            "s_current": s_cur,
            "s_target": tgt,
            "ledger_coverage": cov_d,
        }
    except Exception as exc:
        return f"duty-to-measure: unavailable ({type(exc).__name__}: {exc})", None


def section_orphaned_followons(days: int = 14) -> tuple[str, dict | None]:
    """Named cheap follow-ons the campaign has NOT drained — the #870 duty-to-measure surface.

    Sits BESIDE :func:`section_duty_to_measure` rather than beside a new registry: that function
    ranks owed DSL LEVERS (a build/activation debt keyed by factory name), this one ranks owed
    MEASUREMENTS an arm named in prose (a measurement debt keyed by memo+line). Same queue in
    kind, different key-space, and per CLAUDE.md P1 neither reads the other's store.

    Operator binding 2026-08-01: a named cheap follow-on has no budget, no scorer slot and no
    operator-GO between it and the answer, so it outranks grade 5 as debt.

    It reports STAGED separately from ORPHANED on purpose. STAGED means the runner exists but the
    row names no output artifact, so execution is NOT decidable from artifacts -- the honest
    answer, and the bucket a human should adjudicate first. Reporting it as ORPHANED would
    manufacture debt (``ddm_gd5``'s measured failure mode); folding it into UNKNOWN would hide it.
    """
    try:
        import datetime as _dt

        from tac.followon_ledger import ORPHANED, STAGED, audit, cache_age_s, summarise

        since = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=days)).date()
        # 6 h TTL: the SSD artifact walk is ~60 s cold, too slow for a SessionStart hook. The age
        # is RENDERED, never hidden — a cached answer must be legibly cached.
        paired, ledger = audit(since=since, cache_ttl_s=6 * 3600.0)
        counts = summarise(paired)
        age = cache_age_s()
        head = [
            f"{r.row_id} [{v.verdict}]"
            for r, v in paired
            if v.verdict in (ORPHANED, STAGED)
        ][:_DUTY_TOP_N]
        line = (
            f"orphaned-follow-ons({days}d): {counts[ORPHANED]} ORPHANED · "
            f"{counts[STAGED]} STAGED(adjudicate-first) · {counts['UNKNOWN']} UNKNOWN · "
            f"{counts['EXECUTED']} EXECUTED | {ledger.verdict} "
            f"{ledger.examined}/{ledger.declared_count} memos"
            + (f" | artifact-index {age / 60.0:.0f}m old" if age else "")
        )
        if head:
            line += " | " + " · ".join(head)
        return line, {
            "counts": counts,
            "scope": ledger.as_dict(),
            "adjudicate_first": head,
        }
    except Exception as exc:
        return f"orphaned-follow-ons: unavailable ({type(exc).__name__}: {exc})", None


def section_arm_next_if_resumed(
    path: Path | None = None,
    *,
    top_n: int = _DUTY_TOP_N,
) -> tuple[str, dict | None]:
    """Read the codex arm queue's durable NEXT_IF_RESUMED plan-of-record surface.

    This is a queue-side reader, not a second disposition engine. The broader
    orphaned-follow-on detector still decides EXECUTED/STAGED/UNKNOWN from
    memos and artifacts; this section makes fresh arm plans visible to the
    costate SENSE loop as soon as the keeper or harvest extractor writes them.

    RETRACTION-AWARE since ddm_sc3 (2026-08-16). Rows retracted as SUPERSEDED are
    EXCLUDED from ``latest`` -- a fire order whose admission bar has gone stale must
    not be served to a resuming arm as if it were live. They are never silently
    dropped: the count and the reasons ride on the line and in ``retraction_debt``,
    because a retraction is DEBT somebody must clear, and a shrinking queue that
    explains nothing reads as progress. AMEND_REQUIRED rows stay in ``latest``
    stamped with their notice, since hiding them would suppress the live follow-ons
    sitting beside the one stale clause.
    """
    path = _ARM_NEXT_IF_RESUMED if path is None else path
    try:
        loader_error: str | None = None
        try:
            queue = _load_codex_arm_queue()
            rows = queue.load_next_if_resumed(path)
            debt = queue.next_if_resumed_debt(path)
            superseded_n = debt["counts"][queue.RETRACTION_SUPERSEDED]
            amend_n = debt["counts"][queue.RETRACTION_AMEND_REQUIRED]
            reasons = [
                f"{r.get('name')}:{(r['retractions'][0].get('reason') or '')[:80]}"
                for r in debt["superseded"][-top_n:]
                if r["retractions"]
            ]
        except Exception as exc:  # fail-open, but LOUDLY -- never silently unfiltered
            loader_error = f"{type(exc).__name__}: {exc}"
            rows = [
                row
                for row in _read_jsonl_rows(path)
                if row.get("schema") == "codex_arm_queue.next_if_resumed.v1"
            ]
            superseded_n = amend_n = 0
            reasons = []
        counts: dict[str, int] = {}
        for row in rows:
            provenance = str(row.get("provenance") or "unknown")
            counts[provenance] = counts.get(provenance, 0) + 1
        head = [
            f"{row.get('name')}:{row.get('source_path')}:{row.get('line_start')}"
            for row in rows[-top_n:]
        ]
        line = (
            f"arm-next-if-resumed: {len(rows)} plan row(s) | "
            f"surface {_repo_rel(path)}"
        )
        if loader_error is not None:
            line += f" | RETRACTION FILTER OFF ({loader_error}) — rows may be stale"
        elif superseded_n or amend_n:
            line += f" | retracted: {superseded_n} superseded (hidden), {amend_n} amend-required"
        if counts:
            line += " | " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if head:
            line += " | latest " + " · ".join(head)
        return line, {
            "surface": _repo_rel(path),
            "rows": len(rows),
            "counts_by_provenance": counts,
            "latest": rows[-top_n:],
            "retraction_debt": {
                "superseded": superseded_n,
                "amend_required": amend_n,
                "reasons": reasons,
                "filter_available": loader_error is None,
                "loader_error": loader_error,
            },
        }
    except Exception as exc:
        return f"arm-next-if-resumed: unavailable ({type(exc).__name__}: {exc})", None


def section_factorized_sense(run_dir: Path | None) -> tuple[list[str], dict | None]:
    """Factorized SENSE rows (organ upgrades A+B, 2026-07-17): the exact rank-4/ker(A)
    duty ranking recomputed from the latest persisted margin snapshot (an ALTERNATIVE
    beside the statistical duty line — never a replacement), and the realization-vs-
    gradient regime read (sub-LSB fraction of remaining flip mass -> terminal-SOLVE
    admissibility).  Reads ONLY the .omx/state ledgers written by
    tools/costate_live_ingest.py / tac.witness_control.realization_regime (no decode or
    SegNet work at SessionStart — protects the <5s budget).  Read-only, score-neutral ->
    defaults ON; fail-open (omit on any error); staleness surfaced on each line."""
    lines: list[str] = []
    data: dict = {}
    prefix = run_dir.name if run_dir is not None else None
    try:
        from tac.witness_control.factorized_duty_ranking import (
            format_factorized_duty_line,
            rank_levers_from_summary_row,
        )

        snap_path = _REPO / ".omx" / "state" / "witness_factorized_snapshot.jsonl"
        row = None
        if snap_path.is_file():
            for ln in snap_path.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    cand = json.loads(ln)
                except Exception:
                    continue
                if isinstance(cand, dict) and (prefix is None or str(cand.get("run_ref", "")).startswith(prefix)):
                    row = cand
        if row:
            ranked = rank_levers_from_summary_row(row)
            age = None
            try:
                age = max(0.0, time.time() - snap_path.stat().st_mtime)
            except Exception:
                pass
            lines.append(format_factorized_duty_line(ranked, ema_epoch=row.get("ema_epoch"), age_s=age))
            data["factorized_duty"] = {"ranked_top": ranked[:6], "snapshot_row_ref": row.get("run_ref"),
                                       "ema_epoch": row.get("ema_epoch"), "n_flips": row.get("n_flips")}
    except Exception as exc:
        lines.append(f"factorized-duty: unavailable ({type(exc).__name__})")
    try:
        from tac.witness_control.realization_regime import REGIME_JSONL, format_regime_line, latest_regime_row

        rrow = latest_regime_row(run_prefix=prefix)
        if rrow is None and prefix is not None:
            rrow = latest_regime_row()  # fall back to the newest row, still labeled by run_ref
        if rrow:
            age = None
            try:
                age = max(0.0, time.time() - Path(REGIME_JSONL).stat().st_mtime)
            except Exception:
                pass
            line = format_regime_line(rrow, age_s=age)
            if prefix is not None and not str(rrow.get("run_ref", "")).startswith(prefix):
                line += f" (from run {rrow.get('run_ref')})"
            lines.append(line)
            data["realization_regime"] = {
                k: rrow.get(k) for k in ("run_ref", "ema_epoch", "sub_lsb_frac_mass_weighted",
                                         "regime", "terminal_solve_admissible", "n_pixels_vjp",
                                         "per_class")
            }
    except Exception as exc:
        lines.append(f"realization-regime: unavailable ({type(exc).__name__})")
    return lines, (data or None)


def _pool_marker(r: dict) -> str:
    """DSL-leg marker per pool row: '' = held by a DSL Lever factory · ~ = NOT a single-flag lever
    (tool-side / vehicle-level / unbuilt, carries dsl_na_reason). Mirrors the duty-line vocabulary."""
    return "" if r.get("dsl_lever") else "~"


def format_curriculum_pool_line(summary: dict, top_n: int = _DUTY_TOP_N) -> str:
    """Pure formatter for the curriculum-candidate pool (unit-testable without touching real state).
    Leads with production-fireable duty, then a separate research-only SENSE channel. Research rows
    are visible but never presented as DECIDE/fireable. ~ = not-a-DSL lever (tool/vehicle/unbuilt)."""
    counts = summary.get("counts", {})
    owed = summary.get("owed", 0)
    total = summary.get("total", 0)
    top = summary.get("top_fireable", [])[:top_n]
    more = summary.get("owed", 0) - len(top)
    research = summary.get("research_signals", [])[:top_n]
    research_more = len(summary.get("research_signals", [])) - len(research)

    def _cell(r: dict) -> str:
        leg = r.get("dsl_lever") or "N/A"
        return f"{r['candidate']}{_pool_marker(r)}[{r.get('status', '?')[:5]}·{leg}]"

    line = (
        f"curriculum-pool ({total} tracked; {owed} owed a fire; "
        f"{counts.get('built-never-fired', 0)} built-never-fired · "
        f"{counts.get('needs-build', 0)} needs-build · "
        f"{counts.get('reformulation-queue', 0)} reformulation-queue · "
        f"{counts.get('armed', 0)} armed; ~=not-a-DSL-lever): "
        + ", ".join(_cell(r) for r in top)
        + (f" (+{more} more owed)" if more > 0 else "")
    )
    if research:
        line += " | research-only SENSE (non-fireable): " + ", ".join(
            f"{row['candidate']}[{row.get('status', '?')}]" for row in research
        )
        if research_more > 0:
            line += f" (+{research_more} more research signals)"
    return line


def section_dsl_orphan_flags() -> tuple[str | None, dict | None]:
    """Surface the DSL-orphan debt (task #332): trainer argparse flags the DSL
    does not yet hold as a Lever factory, per lever_registry.completeness().unmapped.

    Per the "off is a tracked queue, never a forgotten default" non-negotiable +
    "DSL HOLDS every designed lever": completeness() ALREADY computes this, but it
    was only consumed by witness_autoconfig / convene — NOT surfaced in the
    always-shown digest, so the debt was queryable-if-you-remember, not a tracked
    queue. This SENSE row makes it visible + ranked-able so the controller
    remembers and the operator never has to. Read-only, score-neutral, fail-open.

    NOTE (anti-fake): this only COUNTS + names the debt; it does NOT auto-stub the
    flags (a generic knob is a lever only WITH swept intent — blind N-by-hand
    stubbing is the anti-pattern the discipline forbids). The fold path is the
    per-lever emit_stub_lever, done as designed, not from this row.

    ddm_rg5 (#825): ``completeness()`` ASTs ONLY ``curriculum_dsl.py``, so a flag held by a
    SIBLING lever module (fh1 / ph3_s10 / ax1 / constants_telemetry) was reported as an orphan
    that nobody holds. MEASURED: 9 of the 80 unmapped flags were FALSE ORPHANS (11%) — e.g.
    ``--muon-momentum``, ``--score-domain-loss``, ``--warm-start-weights-only``. The debt is now
    netted against the package-wide factory flag set before it is reported, and the false-orphan
    count is surfaced so the repair is auditable rather than silent."""
    try:
        from tac.witness_dsl import lever_registry as _lr

        comp = _lr.completeness()
        unmapped = getattr(comp, "unmapped", None)
        if unmapped is None and isinstance(comp, dict):
            unmapped = comp.get("unmapped")
        if not unmapped:
            return None, None
        n_raw = len(unmapped)
        try:  # net out flags held by a sibling module's Lever (fail-open: keep the raw debt)
            pkg_flags: set[str] = set()
            for fb in _lr.package_lever_factories():
                pkg_flags |= set(getattr(fb, "flags", ()) or ())
            unmapped = [u for u in unmapped if u not in pkg_flags]
        except Exception:
            pkg_flags = set()
        n_false = n_raw - len(unmapped)
        if not unmapped:
            return None, {"unmapped_count": 0, "unmapped_sample": [],
                          "false_orphans_netted": n_false}
        sample = list(unmapped)[:6]
        # The vehicle label is NOT decoration (ddm_lr2, 2026-08-03): this call scopes to the
        # RETIRED levelset trainer by default, and its reassuring coverage has been read as a
        # live-vehicle health number. A coverage count with no vehicle attached is unanchored.
        vehicle = getattr(comp, "vehicle_label", "[vehicle UNKNOWN]")
        line = (
            f"dsl-orphan {vehicle} ({len(unmapped)} trainer flag(s) NOT held by a DSL Lever "
            f"— #332 SoT debt; fold per-lever via emit_stub_lever ONLY with swept "
            f"intent, never N-by-hand"
            + (f"; {n_false} sibling-module false orphan(s) netted out" if n_false else "")
            + "): " + ", ".join(sample)
            + (f" (+{len(unmapped) - len(sample)} more)" if len(unmapped) > len(sample) else "")
        )
        return line, {"unmapped_count": len(unmapped), "unmapped_sample": sample,
                      "false_orphans_netted": n_false, "unmapped_count_single_module": n_raw}
    except Exception as exc:  # fail-open: a SENSE row must never break the digest
        return f"dsl-orphan: unavailable ({type(exc).__name__})", None


def section_curriculum_pool() -> tuple[str | None, dict | None]:
    """The CURRICULUM-CANDIDATE POOL as a tracked costate class (task #403; P0 orphan-class binding
    2026-07-10). A curriculum candidate in ANY form (stage / loss / init / preconditioning / data-order
    / averaging / solve-interleave / state-evolution) that is designed-or-built-but-never-fired is the
    SAME orphaned-signal class the lever ledger tracks — but is often a TOOL / stage / vehicle DOF, not a
    single-flag DSL lever, so it lives in the sibling ``curriculum_candidate_pool`` store. This SENSE row
    surfaces production-fireable rows beside a separate research-only SENSE channel, so NO curriculum
    candidate is orphaned and an unactivatable research finding never enters DECIDE. Read-only,
    score-neutral, fail-open (omit on any error)."""
    try:
        from tac.witness_dsl.curriculum_candidate_pool import pool_summary

        summary = pool_summary()
        if not summary.get("total"):
            return None, None
        line = format_curriculum_pool_line(summary)
        return line, {
            "total": summary["total"],
            "owed": summary["owed"],
            "counts": summary["counts"],
            "top_fireable": [
                {
                    "candidate": r.get("candidate"),
                    "status": r.get("status"),
                    "dsl_lever": r.get("dsl_lever"),
                    "dsl_na_reason": r.get("dsl_na_reason"),
                    "form_class": r.get("form_class"),
                    "owner": r.get("owner"),
                }
                for r in summary.get("top_fireable", [])
            ],
            "research_signals": [
                {
                    "candidate": r.get("candidate"),
                    "status": r.get("status"),
                    "research_only": r.get("research_only"),
                    "evidence_kind": r.get("evidence_kind"),
                    "authority_axis": r.get("authority_axis"),
                    "verdict_scope": r.get("verdict_scope"),
                    "activation_status": r.get("activation_status"),
                    "realized_speedup_factor": r.get("realized_speedup_factor"),
                    "derived_cost_reduction_fraction": r.get("derived_cost_reduction_fraction"),
                    "trusted_receipt_sha256": r.get("trusted_receipt_sha256"),
                    "blockers": r.get("blockers", []),
                }
                for r in summary.get("research_signals", [])
            ],
        }
    except Exception as exc:
        return f"curriculum-pool: unavailable ({type(exc).__name__}: {exc})", None


def section_deferral_ledger() -> tuple[str | None, dict | None]:
    """Operator-binding 2026-07-08 ("you deferred too much and now it's orphaned"): every
    deferral lives in .omx/state/deferral_ledger.md with a named trigger; this line makes
    the queue impossible to forget. Soft: absent file -> no line."""
    try:
        path = _REPO / ".omx" / "state" / "deferral_ledger.md"
        if not path.exists():
            return None, None
        rows = [ln for ln in path.read_text(errors="replace").splitlines() if ln.startswith("| D")]
        hot = [ln.split("|")[1].strip() for ln in rows if "SURFACED" in ln or "FIRING" in ln or "ARMED" in ln]
        line = (
            f"deferral-ledger: {len(rows)} open"
            + (f"; hot: {', '.join(hot)}" if hot else "")
            + " (.omx/state/deferral_ledger.md — every row has a named trigger)"
        )
        return line, {"open": len(rows), "hot": hot}
    except Exception as exc:
        return f"deferral-ledger: unavailable ({type(exc).__name__}: {exc})", None


# Schema-tolerant failure-ledger parsing. The ledger accumulated rows across several
# writer generations: the canonical harness_failure.v1 (failure_id + event=="resolution"),
# and older shapes using a bare "class"/"failure_class" key with terminal markers like
# event=="self_protected", status=="resolved", or a populated "resolution" field. A reader
# that keys only on failure_id and only recognises event=="resolution" collapses every
# legacy row into one phantom "?" class (unresolvable, since a None fid never resolves) AND
# hides genuinely-open legacy items behind that phantom. These helpers make the reader read
# the real state without mutating any historical row (append-only provenance preserved).
_LEDGER_CLASS_KEYS = ("failure_id", "failure_class", "class", "bug_class", "class_id")
_LEDGER_RESOLVED_EVENTS = {"resolution", "resolved", "self_protected"}
_LEDGER_RESOLVED_STATUS = {"resolved", "closed", "fixed"}


def _ledger_class_key(row: dict) -> str:
    """First populated class-identifier across writer generations, else '?'."""
    for k in _LEDGER_CLASS_KEYS:
        v = row.get(k)
        if v:
            return str(v)
    return "?"


def _ledger_row_is_resolution(row: dict) -> bool:
    """True if this row is a terminal/resolved marker under ANY writer generation.

    ``event`` is AUTHORITATIVE when present: the canonical v1 schema uses ``event``
    for lifecycle (opened/diagnosis/resolution), and a non-terminal v1 row may carry
    a ``resolution`` field describing the PLANNED fix — so the status/resolution-field
    fallbacks fire only for legacy rows that omit ``event`` entirely (else an
    ``event='opened'`` row with a resolution field would be falsely marked resolved).
    """
    event = str(row.get("event") or "").strip().lower()
    if event:
        return event in _LEDGER_RESOLVED_EVENTS
    for field in ("status", "causal_status"):
        if str(row.get(field) or "").strip().lower() in _LEDGER_RESOLVED_STATUS:
            return True
    res = row.get("resolution")
    return bool(isinstance(res, str) and res.strip())


def _summarize_failure_ledger_v2(rows: list[dict]) -> dict | None:
    """Canonical summary from the typed FailureEventV2 rows, when present.

    Once ``tools/migrate_harness_failure_ledger_v2.py --apply`` has run, every semantic
    class carries one ``harness_failure.v2`` row with a TYPED ``resolution_state`` — the
    reader no longer has to coalesce heterogeneous key names or infer closure from prose.
    Returns None when no V2 rows exist yet (fall back to the schema-tolerant path below).
    """
    if not any(isinstance(r, dict) and r.get("schema") == "harness_failure.v2" for r in rows):
        return None
    try:
        from tac import harness_failure_ledger as _hfl  # local import: soft dependency
    except Exception:
        return None
    latest: dict[str, dict] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("schema") == "harness_failure.v2" and r.get("class_id"):
            latest[str(r["class_id"])] = r  # rows are append-ordered; last wins
    resolved = set(_hfl.RESOLVED_STATES)
    unresolved = sorted(k for k, r in latest.items() if r.get("resolution_state") == "OPEN")
    not_closed = sorted(k for k, r in latest.items() if r.get("resolution_state") not in resolved)
    recurrent = sorted(k for k, r in latest.items() if int(r.get("recurrence_count") or 0) >= 1)
    return {"classes": len(latest), "unresolved": unresolved,
            "not_closed": not_closed, "recurrent": recurrent, "schema": "v2"}


def _summarize_failure_ledger(rows: list[dict]) -> dict:
    """Pure summary (class-count / unresolved / recurrent) over ledger rows.

    Prefers the canonical V2 typed summary when V2 rows exist. Otherwise falls back to the
    schema-tolerant legacy path:
      unresolved = classes whose LATEST row is not a resolution marker.
      recurrent  = classes with >=2 non-resolution rows.
    Schema-tolerant so no writer generation produces a phantom '?' class.
    """
    v2 = _summarize_failure_ledger_v2(rows)
    if v2 is not None:
        return v2
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        if isinstance(r, dict):
            by_id.setdefault(_ledger_class_key(r), []).append(r)
    unresolved = sorted(k for k, evs in by_id.items() if not _ledger_row_is_resolution(evs[-1]))
    recurrent = sorted(
        k for k, evs in by_id.items() if sum(1 for e in evs if not _ledger_row_is_resolution(e)) >= 2
    )
    return {"classes": len(by_id), "unresolved": unresolved, "recurrent": recurrent}


def section_failure_ledger() -> tuple[str | None, dict | None]:
    """Sibling SENSE input (soft: the ledger may not exist yet). Glob-matched so the
    sibling's chosen filename is picked up without a code change here."""
    try:
        hits = sorted(glob.glob(str(_REPO / ".omx" / "state" / "*failure*ledger*.jsonl")))
        if not hits:
            return None, None
        path = Path(hits[-1])
        rows: list[dict] = []
        for ln in path.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    row = json.loads(ln)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
        if not rows:
            return None, None
        # Schema-tolerant summary: class-key falls back across writer generations and the
        # resolved-signal recognises event/status/resolution markers, so no legacy row
        # collapses into a phantom "?" class (see _summarize_failure_ledger).
        summary = _summarize_failure_ledger(rows)
        classes = summary["classes"]
        unresolved = summary["unresolved"]
        recurrent = summary["recurrent"]
        line = (
            f"failure-ledger ({path.name}): {classes} class(es), "
            f"{len(unresolved)} unresolved, {len(recurrent)} recurrent"
        )
        if unresolved:
            line += f"; open: {', '.join(unresolved[:3])}"
        return line, {"path": str(path), "classes": classes, "unresolved": unresolved, "recurrent": recurrent}
    except Exception:
        return None, None


def section_schedule(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Planned-vs-actual DSL schedule position (sibling module — referenced, not built)."""
    for mod_name in ("dsl_schedule_readback", "dashboard_schedule_readback"):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        for fn_name in ("planned_vs_actual_summary", "schedule_position", "summary"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    out = fn(run_dir) if run_dir is not None else fn()
                    return f"schedule: {str(out)[:110]}", {"module": mod_name}
                except Exception:
                    continue
    return None, None


def section_verdict_scope_advisories() -> tuple[str | None, dict | None]:
    """Verdict-scope FM advisories (requirement R, advisory layer): suspected
    OVER-SCOPED negative verdicts flagged by the drift-detector's on-device-FM leg
    (tools/triality_drift_detector.py). The Stop hook can only show text when it
    blocks, so non-blocking advisories persist here; this line surfaces them.
    Read-only; omitted when none exist (fail-open None)."""
    try:
        p = _REPO / ".omx" / "state" / "verdict_scope_advisories.jsonl"
        if not p.exists():
            return None, None
        rows: list[dict] = []
        for ln in p.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
                if isinstance(row, dict) and row.get("advisory"):
                    rows.append(row)
            except Exception:
                continue
        if not rows:
            return None, None
        # recency window: a 14-day count keeps the line live (the file is append-only,
        # so an all-time count would grow monotonically and go stale-misleading).
        import datetime as _dt

        cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=14)

        def _fresh(r: dict) -> bool:
            try:
                return _dt.datetime.fromisoformat(str(r.get("ts", ""))) >= cutoff
            except Exception:
                return True  # unparsable ts → keep (fail toward visibility)

        fresh = [r for r in rows if _fresh(r)]
        if not fresh:
            return None, None
        last = str(fresh[-1].get("advisory", ""))[:110]
        line = (
            f"verdict-scope advisories: {len(fresh)} doc-flag(s) in the last 14d "
            f"(latest: {last}) — re-check declared scope vs evidence (req R)"
        )
        return line, {
            "count_14d": len(fresh),
            "count_all": len(rows),
            "recent": [r.get("advisory") for r in fresh[-3:]],
        }
    except Exception:
        return None, None


def section_resume_spine() -> tuple[str, dict | None]:
    """AMENDMENT 2: the compact resume spine (pointers, not content dumps)."""
    try:
        dag_files = sorted(glob.glob(_DAG_GLOB))
        feed = None
        if dag_files:
            for ln in reversed(Path(dag_files[-1]).read_text(errors="replace").splitlines()):
                if ln.startswith("## FEED-"):
                    feed = ln[3:].strip()
                    break
        feed_txt = (feed[:100] + "…") if feed and len(feed) > 100 else (feed or "none found")
        return (f"resume spine: MEMORY.md ⭐CURRENT-STATE → newest DAG {feed_txt}", {"newest_feed": feed})
    except Exception as exc:
        return f"resume spine: unavailable ({type(exc).__name__})", None


def section_active_convening() -> tuple[str | None, dict | None]:
    """Auto-surface any ACTIVE council/crucible orchestration so NO session (and no operator)
    must remember it exists. Reads the newest orchestration ledger's phase checkboxes + last
    log line; fail-open (omit if absent). Operator binding 2026-07-07: 'you are once again
    requiring me to have an insane memory' — the system holds the thread, not the human."""
    try:
        ledgers = sorted(glob.glob(".omx/research/*crucible*/ORCHESTRATION_LEDGER.md")) + sorted(
            glob.glob(".omx/research/*/ORCHESTRATION_LEDGER.md")
        )
        if not ledgers:
            return None, None
        txt = Path(ledgers[-1]).read_text(errors="replace")
        done = txt.count("- [x] P")
        total = done + txt.count("- [ ] P")
        # current phase = first unchecked
        phase = next(
            (ln.strip()[6:60] for ln in txt.splitlines() if ln.strip().startswith("- [ ] P")), "ALL PHASES DONE"
        )
        last_log = next((ln.strip() for ln in reversed(txt.splitlines()) if ln.strip().startswith("- 2026")), "")
        line = (
            f"ACTIVE CONVENING ({Path(ledgers[-1]).parent.name}): phases {done}/{total} done; "
            f"NEXT: {phase} | last: {last_log[:120]} | ledger: {ledgers[-1]}"
        )
        return line, {"ledger": ledgers[-1], "phases_done": done, "phases_total": total, "next_phase": phase}
    except Exception as exc:
        return f"active convening: unavailable ({type(exc).__name__})", None


def section_corpus_recall(conv: dict | None) -> tuple[list[str], dict | None]:
    """#346 RECALL-PUSH: if a convening is ACTIVE, surface the top corpus hits for its
    next phase — push, not pull (the apparatus writes better than it reads; this section
    makes the corpus speak at session start without anyone asking). Fail-open, ≤5 lines,
    time-bounded ≤2s via corpus_query's budget (newest units win under truncation)."""
    if not conv or not conv.get("next_phase"):
        return [], None
    try:
        import corpus_query

        result = corpus_query.run_query(str(conv["next_phase"]), top=3, max_seconds=2.0)
        hits = result.get("hits") or []
        if not hits:
            return [], None
        lines = ["the corpus knows (recall for the active convening's next phase):"]
        for h in hits[:3]:
            first = (h["lines"][0][:80] + "…") if h.get("lines") else ""
            ref = h["ref"] if len(h["ref"]) <= 90 else "…" + h["ref"][-89:]
            lines.append(f"  [{h['store']}] {ref} — {first}")
        return lines[:5], {"hits": hits, "truncated": result.get("truncated", False)}
    except Exception as exc:
        return [f"corpus recall: unavailable ({type(exc).__name__})"], None


def section_graph_memory() -> tuple[str | None, dict | None]:
    """#411 RECONSTRUCT-ON-DEMAND: surface the graph-memory (the DAG-as-graph) so
    recall-by-reconstruction is a STRUCTURAL session-start affordance, not grep-by-
    volition (the read side of L83 / the #346 retrieval-first nexus done as a graph).
    Read-only + fast (counts the cached graph's JSONL lines; NEVER rebuilds at session
    start). Score-neutral, fail-open."""
    try:
        from tac.graph_memory import cache_paths

        npath, epath = cache_paths()
        if not (npath.is_file() and epath.is_file()):
            return (
                "graph-memory: not yet built — reconstruct-on-demand via "
                "`tools/graph_memory_recall.py --rebuild` (the DAG-as-graph, #411)"
            ), None
        n_nodes = sum(1 for _ in npath.open(encoding="utf-8"))
        n_edges = sum(1 for _ in epath.open(encoding="utf-8"))
        # Usage counter (operator 2026-07-20 "are we making sure to use it as
        # much as possible?"): count recall_log rows in the last 24h so a
        # zero-recall day is LOUD here instead of silent (the 16.7h-stale-cache
        # incident). Fail-open.
        recalls_24h = 0
        try:
            import datetime as _dt
            import json as _json

            _cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=24)
            _log = npath.parent / "recall_log.jsonl"
            if _log.is_file():
                for _row in _log.open(encoding="utf-8"):
                    try:
                        _ts = _dt.datetime.fromisoformat(_json.loads(_row)["ts"])
                        if _ts >= _cutoff:
                            recalls_24h += 1
                    except Exception:
                        continue
        except Exception:
            pass
        usage = f"recalls 24h: {recalls_24h}" + (
            " ⚠ UNDER-USED — recall-before-decide is the discipline" if recalls_24h == 0 else ""
        )
        line = (
            f"graph-memory: {n_nodes} nodes / {n_edges} edges — {usage} — RECONSTRUCT "
            f'before grepping: `tools/graph_memory_recall.py "<query>"` (#411 DAG-as-graph)'
        )
        return line, {"nodes": n_nodes, "edges": n_edges, "recalls_24h": recalls_24h}
    except Exception as exc:
        return f"graph-memory: unavailable ({type(exc).__name__})", None


def section_costate_organ(run_dir: Path | None) -> tuple[list[str], dict | None]:
    """#426 λ-organ OBSERVATORY (Rudin/max-observability): which prototype regimes
    fired with what weights + uncertainty; the compounding organ-ledger summary
    (records/regimes/recommended architecture); the top PowerPlay acquisition rows.
    Read-only, numpy-only lenses (spread-gate discipline: no torch fits here),
    fail-open, bounded (<1s on verdict-cadence data)."""
    lines: list[str] = []
    data: dict = {}
    # (a) the compounding memory summary (triality ledger — cheap parse)
    try:
        from tac.witness_control.continual_costate import organ_summary

        s = organ_summary()
        data["memory"] = s
        if s["n_records"]:
            lines.append(
                f"λ-organ memory: {s['n_records']} trajectory record(s) / "
                f"{s['total_intervals']} intervals; {s['n_regimes']} regime(s) "
                f"[{', '.join(s['regimes'][:4])}{'…' if s['n_regimes'] > 4 else ''}]; "
                f"recommended arch {s['recommended_architecture']}"
            )
    except Exception as exc:
        lines.append(f"λ-organ memory: unavailable ({type(exc).__name__})")
    # (b) the live observatory + acquisition queue (needs a run dir with verdicts)
    if run_dir is None:
        return lines, data
    try:
        import numpy as np

        from tac.witness_control.control_alphabet import powerplay_acquisition
        from tac.witness_control.lambda_net import (
            RidgeSolveAdjoint,
            build_intervals,
            evaluate_lambda_field,
            fit_score_composition,
            lever_features,
            read_trajectory,
        )
        from tac.witness_control.prototype_router import PrototypeRouterLens

        traj = read_trajectory(run_dir)
        if traj.n_verdicts >= 3:
            comp = fit_score_composition(traj.verdicts)
            intervals = build_intervals(traj)
            phis = np.stack([lever_features(n) for n in traj.lever_names])
            lens = PrototypeRouterLens()
            lens.fit(intervals, phis)
            last = traj.verdicts[-1]
            x = np.concatenate(
                [
                    np.asarray(last["d_seg_by_class"], dtype=float),
                    [math.log(max(float(last.get("blob_bytes", 1.0)), 1.0))],
                ]
            )
            att = lens.attribute(x, float(last["epoch"]), comp.grad_s_wrt_state())
            data["observatory"] = {
                "explain": att.explain(),
                "fired": list(att.fired),
                "mixture_entropy": att.mixture_entropy,
            }
            lines.append("λ-organ observatory: " + att.explain())
            faith = lens.faithfulness_audit(x, intervals[-1].ctx, comp.grad_s_wrt_state(), lever_features("seg"))
            data["faithfulness"] = {k: faith[k] for k in ("faithful", "max_rel_gap", "rel_tol")}
            lines.append(
                f"λ-organ faithfulness: {'OK' if faith['faithful'] else 'DIVERGENT'} "
                f"(max_rel_gap {faith['max_rel_gap']:.3f} ≤ tol {faith['rel_tol']}; "
                "stated-vs-counterfactual, 2607.08046)"
            )
            m = RidgeSolveAdjoint()
            m.fit(intervals, phis)
            fld = evaluate_lambda_field(m, traj, comp, intervals, status="SPECULATIVE-UNTIL-BACKTESTED")
            ever = {n: fld.identified.get(n, False) for n in fld.per_lever}
            acq = powerplay_acquisition(fld.per_lever, ever_fired=ever)[:3]
            data["acquisition_top"] = [{"lever": r.lever, "acquisition": r.acquisition, "kind": r.kind} for r in acq]
            lines.append(
                "λ-organ acquisition (PowerPlay top-3): "
                + " · ".join(f"{r.lever}({r.kind.split('-')[0]},{r.acquisition:.2f})" for r in acq)
            )
            # (c) #436 regime-conditional SELF-DISPATCH — the per-state arbiter decision
            #     (advisory, past-only, NON-PROMOTABLE): current regime + which tool it
            #     dispatches (or PERSISTENCE), provenance-labeled to the measured verdict.
            from tac.witness_control.regime_dispatch import dispatch_for_trajectory

            dd = dispatch_for_trajectory(traj)
            data["dispatch"] = {
                "regime": dd.classification.regime,
                "tool": dd.tool,
                "deciding_signal": dd.classification.deciding_signal,
                "explain": dd.explain(),
                "artifact": _DISPATCH_DOC,
                "axis_tag": dd.axis_tag,
                "actuation": dd.actuation,
            }
            lines.append(
                f"organ-dispatch (#436, {dd.axis_tag}): regime={dd.classification.regime} "
                f"→ TOOL={dd.tool} [{dd.classification.deciding_signal}] "
                f"· actuation={dd.actuation} · {_DISPATCH_DOC}"
            )
    except Exception as exc:
        lines.append(f"λ-organ observatory: unavailable ({type(exc).__name__})")
    return lines, data


def _fm_regime_text(data: dict) -> str:
    """Best available current-regime text for the FM duty-relevance judgment: the organ
    dispatch regime → shadow classification → 'unknown'. Pure helper."""
    organ = data.get("costate_organ") or {}
    disp = (organ.get("dispatch") or {}) if isinstance(organ, dict) else {}
    if disp.get("regime"):
        return str(disp["regime"])
    shadow = data.get("shadow") or {}
    cls = (shadow.get("classification") or {}).get("classification") if isinstance(shadow, dict) else None
    return str(cls) if cls else "unknown"


def section_fm_advisory(run_dir: Path | None, data: dict) -> tuple[list[str], dict | None]:
    """Task #522: the on-device FM (fmtools) ADVISORY sense layer, surfaced as a compact
    section PRESENT ONLY WHEN THE fmtools VENV EXISTS (⇒ the digest is byte-identical without
    it). Consumes the ALREADY-COMPUTED digest ``data`` (shadow classification, annulus, duty,
    failure ledger) so no numeric work is redone; the FM adds a semantic second opinion at the
    four organ insertion points. ADVISORY ONLY: disagreement is a surfaced diagnostic, never an
    override; the P8 floor-aware duty order stays the base order. Fail-open (never breaks the
    digest); each sub-line rendered only when its inputs exist."""
    try:
        from tac import fm_advisory as _fm

        if not _fm.available():
            return [], {"available": False}
    except Exception as exc:  # import failure ⇒ absent, never a crash
        return [], {"available": False, "error": type(exc).__name__}

    lines: list[str] = ["fm-advisory (on-device FM · advisory · NON-PROMOTABLE):"]
    secdata: dict = {"available": True}
    try:
        cap = _fm.capability_report(timeout=8)
        if cap:
            secdata["capability_report"] = cap
            flags = [
                f"guided={'Y' if cap.get('supports_guided_generation') else 'n'}",
                f"tools={'Y' if cap.get('supports_tools') else 'n'}",
                f"stream={'Y' if cap.get('supports_streaming') else 'n'}",
                f"options={'Y' if cap.get('supports_generation_options') else 'n'}",
            ]
            avail = "Y" if cap.get("model_available") else "n"
            lines.append(
                "  capability: "
                f"sdk={cap.get('sdk_version') or '?'} available={avail} "
                f"backend={cap.get('backend') or '?'} "
                + " ".join(flags)
            )
        # (a) REGIME supplement — telemetry text from shadow classification + numeric hint from annulus.
        shadow = data.get("shadow") if isinstance(data.get("shadow"), dict) else None
        classification = (shadow or {}).get("classification")
        telemetry_texts: list = []
        if classification:
            telemetry_texts.append({k: classification.get(k) for k in ("classification", "phase_regime", "reason")})
        ann = data.get("annulus") if isinstance(data.get("annulus"), dict) else None
        hint = _fm.numeric_regime_hint(ann, classification)
        regime = _fm.regime_supplement(telemetry_texts, numeric_hint=hint) if telemetry_texts else None
        if regime:
            secdata["regime"] = regime
            agree = regime.get("agrees_with_numeric")
            tag = "AGREE" if agree is True else "DISAGREE" if agree is False else "—"
            lines.append(
                f"  regime: fm={regime.get('fm_regime')} vs numeric={regime.get('numeric_hint') or '?'} "
                f"[{tag}] · {str(regime.get('rationale') or '')[:80]}")
            if agree is False:
                lines.append(
                    f"  ⚠ regime DISAGREEMENT (advisory): FM reads '{regime.get('fm_regime')}' but the "
                    f"numeric hint is '{regime.get('numeric_hint')}' — re-check per-class dynamics (never an override)")
        # (b) EVENT-intelligence — notable events from the run.log (reuse the extractor).
        if run_dir is not None and (run_dir / "run.log").is_file():
            try:
                import dashboard_fm_events as _dfe

                evlines = (run_dir / "run.log").read_text(errors="replace").splitlines()
                events = _dfe.extract_notable_events(evlines, limit=6)
                ev = _fm.classify_events([e["line"] for e in events]) if events else None
                if ev:
                    secdata["events"] = ev
                    top = next((r for r in ev if r.get("event_class")), None)
                    lines.append(
                        f"  events: {sum(1 for r in ev if r.get('event_class'))} classified"
                        + (f" (e.g. {top.get('event_class')})" if top else ""))
            except Exception:
                pass
        # (c) DUTY-relevance secondary hint — top never-fired duty rows vs current regime.
        duty = data.get("duty_to_measure") if isinstance(data.get("duty_to_measure"), dict) else None
        ranked = (duty or {}).get("ranked_top") or []
        never_fired = [r for r in ranked if r.get("activation_state") == "never-fired"] or ranked
        rel = _fm.duty_relevance(never_fired, _fm_regime_text(data), top_k=4) if never_fired else None
        if rel:
            secdata["duty_relevance"] = rel
            cells = ", ".join(f"{r['lever']}={r.get('relevance') or '?'}" for r in rel[:4])
            lines.append(f"  duty-relevance (secondary hint; P8 order unchanged): {cells}")
        # (d) CONFOUND-alarm classing — recent unresolved failure rows vs known classes.
        fl = data.get("failure_ledger") if isinstance(data.get("failure_ledger"), dict) else None
        unresolved = (fl or {}).get("unresolved") or []
        if unresolved and fl.get("path"):
            try:
                import dashboard_fm_events as _dfe

                known = _dfe.known_failure_classes()
                # recent failure texts from the ledger tail
                texts: list = []
                for ln in Path(fl["path"]).read_text(errors="replace").splitlines()[-8:]:
                    if ln.strip():
                        try:
                            row = json.loads(ln)
                            if isinstance(row, dict) and row.get("event") != "resolution":
                                texts.append({k: row.get(k) for k in ("failure_id", "symptom", "event", "detail")})
                        except Exception:
                            continue
                conf = _fm.classify_confounds(texts, known) if (texts and known) else None
                if conf:
                    secdata["confounds"] = conf
                    hit = next((r for r in conf if r.get("matched_class")), None)
                    lines.append(
                        f"  confound-class: {sum(1 for r in conf if r.get('matched_class'))} matched"
                        + (f" (e.g. {hit.get('matched_class')})" if hit else ""))
            except Exception:
                pass
    except Exception as exc:  # advisory section never breaks the digest
        lines.append(f"  fm-advisory: unavailable ({type(exc).__name__})")
    if len(lines) == 1:
        lines.append("  (on-device FM present; no classifiable inputs this cycle)")
    return lines, secdata


def section_review_counter() -> tuple[str | None, dict | None]:
    """Open review-counter state (sibling ledger; soft — omit entirely if absent)."""
    row = _last_jsonl_row(_REVIEW_COUNTER) if _REVIEW_COUNTER.exists() else None
    if not row:
        return None, None
    try:
        # sibling schema review_counter.v1: surface_id / round_n / findings_count / verdict
        line = (
            f"review-counter: {row.get('surface_id', '?')} round {row.get('round_n', '?')} "
            f"findings {row.get('findings_count', '?')} verdict {row.get('verdict', '?')}"
        )
        return line[:130], row
    except Exception:
        return None, None


def section_chain_watchdog() -> tuple[str | None, dict | None]:
    """B4 chain-liveness SENSE row (p0_launcher_chain_durability_20260717): the LAST scan
    appended by tools/witness_chain_watchdog.py. Surfaces CHAIN_DEAD_NO_RECEIPT loudly (the
    silent-death alarm) and RUNNING_QUIET informatively (alive despite quiet logs — the
    20260716 phantom-death class). Fail-open; None when no scans exist yet."""
    try:
        row = _last_jsonl_row(_REPO / ".omx" / "state" / "witness_chain_watchdog.jsonl")
        if not row or not isinstance(row.get("verdicts"), list):
            return None, None
        verdicts = [v for v in row["verdicts"] if isinstance(v, dict)]
        dead = [v for v in verdicts if v.get("verdict") == "CHAIN_DEAD_NO_RECEIPT"]
        quiet = [v for v in verdicts if v.get("verdict") == "RUNNING_QUIET"]
        if dead:
            labels = ", ".join(str(v.get("label")) for v in dead)
            line = (f"chain-watchdog: ALARM CHAIN_DEAD_NO_RECEIPT [{labels}] — silent death; "
                    f"postmortem before relaunch (scan {row.get('ts')})")
        elif quiet:
            line = (f"chain-watchdog: {len(quiet)} chain(s) ALIVE-but-quiet (buffered logs "
                    f"!= death; scan {row.get('ts')}) — judge by tools/witness_chain_watchdog.py")
        elif verdicts:
            line = f"chain-watchdog: {len(verdicts)} chain(s) healthy (scan {row.get('ts')})"
        else:
            return None, None
        return line, {"scan_ts": row.get("ts"), "verdicts": verdicts}
    except Exception:  # SENSE row must never break the digest
        return None, None


def _fm_advisory_enabled(session_start: bool) -> bool:
    """Cost-gate for the #522 FM section (a genuine compute cost: on-device FM subprocesses,
    ~8s). Per CLAUDE.md "'Off' is a tracked queue": read-only-but-costly telemetry gates on
    compute cost with a RECORDED reason. Default ON for explicit check-ins (the agent asked);
    default OFF in the <5s SessionStart hot path. ``COSTATE_FM_ADVISORY`` (1/0) overrides both."""
    import os

    env = os.environ.get("COSTATE_FM_ADVISORY")
    if env is not None:
        return env not in ("0", "", "false", "False")
    return not session_start  # explicit call → on; session-start → off (protect <5s budget)


def _format_vehicle_routing_coverage(report: dict) -> str:
    """Render the owner-visible harvest denominator without hiding missing rows."""

    totals = report["totals"]
    lineages = report["lineages"]
    outstanding = sorted(
        (
            (name, int(counts["un_harvested"]))
            for name, counts in lineages.items()
            if int(counts["un_harvested"]) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    head = ",".join(f"{name}:{count}" for name, count in outstanding) or "none"
    return (
        "DDM-vehicle-harvest: "
        f"{totals['routed']}/{totals['artifacts']} routed artifacts; "
        f"harvested={totals['harvested']} un-harvested={totals['un_harvested']}; "
        f"largest gaps={head}"
    )


def section_vehicle_routing_coverage() -> tuple[str, dict | None]:
    """Read the canonical probe-outcomes extension; never write or block startup."""

    try:
        from tac.probe_outcomes_ledger import coverage

        report = coverage()
        return _format_vehicle_routing_coverage(report), report
    except Exception as exc:  # SENSE row must never break the digest
        return f"DDM-vehicle-harvest: unavailable ({type(exc).__name__})", None


# ─────────────────────────── assembly ───────────────────────────
def build_digest(*, include_fm: bool = True) -> tuple[list[str], dict]:
    t0 = time.time()
    lines: list[str] = []
    data: dict = {}

    ptr_line, data["pointer"] = section_pointer()
    lines.append(ptr_line)  # NEVER dropped (amendment 1)

    ddm_lines, data["ddm_costate_organ"] = section_live_ddm()
    ddm_live = bool(
        isinstance(data["ddm_costate_organ"], dict)
        and data["ddm_costate_organ"].get("available")
    )
    # Campaign-run observability is an independent read-only source.  Surface
    # it whether the broader DDM receipt fleet is complete or fail-open, so a
    # stopped campaign remains visible and is never mistaken for a live run.
    lines.extend(ddm_lines)
    campaign_run_lines, data["ddm_campaign_run_observability"] = (
        section_ddm_campaign_run()
    )
    lines.extend(campaign_run_lines)
    routing_line, data["vehicle_routing_coverage"] = (
        section_vehicle_routing_coverage()
    )
    lines.append(routing_line)
    if ddm_live:
        # The live DDM campaign is primary.  Do not even inspect the quarantined
        # witness run; compatibility keys remain explicit dominated markers.
        run_dir = None
        report = data["ddm_costate_organ"]
        from tac.ddm_campaign_costate import campaign_consumer_view

        campaign_digest = campaign_consumer_view(report["campaign"], "digest")
        campaign_duty = campaign_consumer_view(report["campaign"], "duty_queue")
        campaign_nag = campaign_consumer_view(report["campaign"], "activation_nag")
        data["ddm_campaign"] = campaign_digest
        data["ddm_campaign_activation_nag"] = campaign_nag
        data["live_run"] = {
            "alive": False,
            "status": "DOMINATED_BY_LIVE_DDM_RECEIPT_FLEET",
            "legacy_lookup_performed": False,
        }
        data["annulus"] = None
        data["shadow"] = {
            "status": "DOMINATED_STALE",
            "reason": report["legacy"]["reason"],
        }
        data["costate_organ_v2"] = {
            "status": "DOMINATED_STALE",
            "replacement": report["schema"],
            "actuation": "NONE",
            "score_claim": False,
        }
        data["ncde"] = report["instruments"]["ncde"]
        data["verdict_trend"] = None
        data["pose_conditioning_gate"] = report["instruments"]["pose_gate"]
        data["telemetry_binding"] = None
        data["duty_to_measure"] = {
            "ranked_top": [
                {
                    "lever": row["duty"],
                    "activation_state": "live-ddm-campaign-duty",
                    "why": row["reason"],
                }
                for row in campaign_duty["rows"]
            ],
            "state_digest": campaign_duty["state_digest"],
            "lambda_ranker": campaign_duty["lambda_ranker"],
            "activation_nag": campaign_duty["activation_nag"],
            **report["duties"],
            "legacy_duties": report["duties"],
        }
        data["factorized_sense"] = report["lambda"]["backtest"]
    else:
        # Historical fallback is allowed only when no complete live DDM source
        # fleet is available; partial DDM state is never mixed into it.
        live_line, data["live_run"], run_dir = section_live_run()
        lines.append(live_line)  # NEVER dropped

        ann_line, data["annulus"] = section_annulus(run_dir)
        if ann_line:
            lines.append(ann_line)

        shadow_lines, data["shadow"] = section_shadow(run_dir)
        lines.extend(shadow_lines)
        if isinstance(data["shadow"], dict):
            data["costate_organ_v2"] = data["shadow"].get("costate_organ_v2")
        else:
            try:
                from tac.witness_control.costate_organ_v2 import aggregate_readback

                data["costate_organ_v2"] = aggregate_readback({}, maturity="_dev")
                lines.append("costate-organ-v2: UNAVAILABLE_NO_VERDICT | pair/site lambda=OWED")
            except Exception as exc:
                data["costate_organ_v2"] = {
                    "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}",
                    "actuation": "NONE", "score_claim": False,
                }

        ncde_line, data["ncde"] = section_ncde(run_dir)
        if ncde_line:
            lines.append(ncde_line)

        vtrend_line, data["verdict_trend"] = section_verdict_trend(run_dir)
        if vtrend_line:
            lines.append(vtrend_line)

        posegate_line, data["pose_conditioning_gate"] = section_pose_conditioning_gate(run_dir)
        if posegate_line:
            lines.append(posegate_line)

        tbind_line, data["telemetry_binding"] = section_telemetry_binding(run_dir)
        if tbind_line:
            lines.append(tbind_line)

        # P8 floor-aware: feed the live run's MEASURED current d_seg (annulus SENSE) so at-floor levers rank ~0.
        duty_line, data["duty_to_measure"] = section_duty_to_measure(_live_term_current(data.get("annulus")))
        lines.append(duty_line)

        # #870 (2026-08-01): the SECOND duty queue — measurements arms NAMED in prose and nobody
        # ran. Beside the lever queue above, never inside it: different key-space (memo+line vs
        # factory name), same duty. Surfaced here because the operator's binding is that a $0
        # follow-on has nothing between it and the answer except somebody reading the memo.
        fo_line, data["orphaned_followons"] = section_orphaned_followons()
        lines.append(fo_line)

        arm_next_line, data["arm_next_if_resumed"] = section_arm_next_if_resumed()
        lines.append(arm_next_line)

        # A+B (2026-07-17): exact-factorized duty ranking (ALTERNATIVE beside the statistical
        # line above) + the realization-vs-gradient regime read — from persisted ledgers only.
        fact_lines, data["factorized_sense"] = section_factorized_sense(run_dir)
        lines.extend(fact_lines)

    if ddm_live:
        # Do not append the retired witness curriculum/deferral/schedule organ
        # below.  Compatibility keys are explicit dominated markers so API
        # consumers still receive a total schema without initializing the old
        # instrumentation line.
        dominated = {
            "status": "DOMINATED_STALE",
            "reason": "replaced by live DDM describe-line/joint-recursion state",
        }
        for key in (
            "curriculum_pool",
            "dsl_orphan",
            "failure_ledger",
            "chain_watchdog",
            "zero_work_arms",
            "deferral_ledger",
        ):
            data[key] = dict(dominated)
        data["schedule"] = report["scheduler"]
        data["verdict_scope"] = {
            "evidence_axis": report["evidence_axis"],
            "score_claim": report["score_claim"],
            "promotion_eligible": report["promotion_eligible"],
            "main_landing_review_required": report["main_landing_review_required"],
        }
        data["resume_spine"] = report["resume_state"]
        data["active_convening"] = None
        data["corpus_recall"] = []
        data["graph_memory"] = None
        data["costate_organ"] = report
        if include_fm:
            fm_lines, data["fm_advisory"] = section_fm_advisory(None, data)
            lines.extend(fm_lines)
        else:
            data["fm_advisory"] = {
                "available": None,
                "enabled": False,
                "reason": "compute-cost gate (fast path); set COSTATE_FM_ADVISORY=1",
            }
        rc_line, data["review_counter"] = section_review_counter()
        if rc_line:
            lines.append(rc_line)
        lines.append(
            "BOUNDARY: autonomous = DDM advisory ranking + re-derivation queue + this digest. "
            "Operator-GO = launches, run mutation, paid dispatch, or promotion."
        )
        lines.append(
            "deeper: .omx/research/codex_findings_ddm_costate_organ_elevation2_"
            "20260723T154610Z_codex.md"
        )
        data["wall_clock_s"] = round(time.time() - t0, 3)
        return lines, data

    # #403 P0: the curriculum-candidate pool (ANY form) as a tracked costate class, beside the lever queue.
    cpool_line, data["curriculum_pool"] = section_curriculum_pool()
    if cpool_line:
        lines.append(cpool_line)

    dslorphan_line, data["dsl_orphan"] = section_dsl_orphan_flags()
    if dslorphan_line:
        lines.append(dslorphan_line)

    fl_line, data["failure_ledger"] = section_failure_ledger()
    lines.append(fl_line or "failure-ledger: none yet (sibling SENSE input pending)")

    wd_line, data["chain_watchdog"] = section_chain_watchdog()
    if wd_line:
        lines.append(wd_line)

    # CLASS-4 (2026-07-17): surface subagent arms that registered but never advanced then went
    # silent (a SPEC_v10 arm died at ~15 tokens with no work, human-visible only). Detect-only.
    try:
        from tools.subagent_liveness import digest_line as _zwa_line
        zwa_line, data["zero_work_arms"] = _zwa_line()
        if zwa_line:
            lines.append(zwa_line)
    except Exception as _exc:  # never let the detector break the digest
        data["zero_work_arms"] = {"error": type(_exc).__name__}

    dl_line, data["deferral_ledger"] = section_deferral_ledger()
    if dl_line:
        lines.append(dl_line)

    sched_line, data["schedule"] = section_schedule(run_dir)
    lines.append(sched_line or "schedule: planned-vs-actual read-back pending (sibling module)")

    vs_line, data["verdict_scope"] = section_verdict_scope_advisories()
    if vs_line:
        lines.append(vs_line)

    spine_line, data["resume_spine"] = section_resume_spine()
    lines.append(spine_line)

    conv_line, data["active_convening"] = section_active_convening()
    if conv_line:
        lines.append(conv_line)

    recall_lines, data["corpus_recall"] = section_corpus_recall(data["active_convening"])
    lines.extend(recall_lines)

    gm_line, data["graph_memory"] = section_graph_memory()
    if gm_line:
        lines.append(gm_line)

    if ddm_live:
        data["costate_organ"] = data["ddm_costate_organ"]
    else:
        organ_lines, data["costate_organ"] = section_costate_organ(run_dir)
        lines.extend(organ_lines)

    # Task #522: on-device FM ADVISORY sense layer — PRESENT ONLY WHEN the fmtools venv exists
    # AND the compute-cost gate is enabled (⇒ byte-identical digest otherwise). Consumes the data
    # computed above (no numeric rework).
    if include_fm:
        fm_lines, data["fm_advisory"] = section_fm_advisory(run_dir, data)
        lines.extend(fm_lines)
    else:
        data["fm_advisory"] = {"available": None, "enabled": False,
                               "reason": "compute-cost gate (fast path); set COSTATE_FM_ADVISORY=1"}

    rc_line, data["review_counter"] = section_review_counter()
    if rc_line:
        lines.append(rc_line)

    lines.append(
        "BOUNDARY: autonomous = advisory recs · duty-to-measure ranking · curriculum "
        "condition inputs · this digest. Operator-GO = heavy/paid launches · run stops · "
        "live-config changes (CONTAINMENT)."
    )
    lines.append(f"deeper: {_DESIGN_DOC} (§2026-07-07 agent-native surfacing)")
    data["wall_clock_s"] = round(time.time() - t0, 3)
    return lines, data


def _model_identity_stamp() -> str:
    """MODEL-ROUTING stamp (operator 2026-07-31 "Just use opus for everything").

    SUPERSEDES the 2026-07-21 Fable-only rule, which is RETIRED. That banner demanded a
    Fable main-thread AND instructed a non-Fable agent to "do minimal safe work only" —
    under the current routing that is a stale gate that would halt a legitimate session,
    the exact "hook fucking us up" class. Detection is still worth keeping and is cheap:
    the hook cannot observe the live model, so it reads the saved client default and asks
    for a one-line self-declaration only when the model is NOT the expected class.
    Fail-open: never blocks a session."""
    default = "unknown"
    try:
        cfg = json.loads((Path.home() / ".claude" / "settings.json").read_text())
        default = str(cfg.get("model", "unknown"))
    except Exception:
        pass
    return (
        f"MODEL ROUTING (operator 2026-08-04): FABLE 5 on the MAIN thread ONLY; CODEX arms for "
        f"ALL subagent work (Claude subagents OFF incl. forks — the 08-04 quota catastrophe). "
        f"Spawn arms ONLY via tools/codex_arm_queue.py saturate --spawn (keeper mechanism — "
        f"hand-rolled codex exec dies at ~5-6min to the fleet launchd reaper). Client default: "
        f"{default!r}. Say so once if MAIN is NOT Fable-class, so a silent reroute is visible."
    )


def _compact_for_hook(lines: list[str]) -> list[str]:
    """Apply the SessionStart line budget: drop `_HOOK_DETAIL_PREFIXES` lines and
    replace them with ONE honest pointer naming what was suppressed and how to get
    it back.  Progressive disclosure, not deletion — nothing is destroyed, and the
    reader is told exactly what it is not seeing (a silent cut would be the very
    "hook fucking us up" class this budget exists to avoid).

    Order of kept lines is preserved.  Pure function; never raises."""
    kept: list[str] = []
    tags: list[str] = []
    for ln in lines:
        if ln.startswith(_HOOK_DETAIL_PREFIXES):
            # tag = the prefix up to ':' or '[' — "DDM-arc[07-28]: ..." -> "DDM-arc"
            tags.append(ln.split(":", 1)[0].split("[", 1)[0].strip())
        else:
            kept.append(ln)
    if tags:
        uniq = list(dict.fromkeys(tags))  # stable de-dupe
        kept.append(
            f"DDM-detail: {len(tags)} accounting/law lines suppressed for the hook budget "
            f"({', '.join(uniq)}) — full state: "
            f".venv/bin/python tools/costate_digest.py --full"
        )
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--session-start", action="store_true", help="hook mode: same digest, ALWAYS exits 0 (never blocks a session)"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="emit every line uncompacted (the pre-2026-07-31 session-start payload); "
        "the escape hatch the compacted hook digest points at",
    )
    args = ap.parse_args(argv)
    try:
        lines, data = build_digest(include_fm=_fm_advisory_enabled(args.session_start))
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True, default=str))
        else:
            # Compact ONLY in hook mode and ONLY when --full was not asked for.  An
            # explicit interactive `costate_digest.py` call is unchanged (full detail),
            # so no human workflow regresses; only the recurring hook payload shrinks.
            compact = args.session_start and not args.full
            if args.session_start:
                print("[costate-digest] controller SENSE+DECIDE state (auto-surfaced; tools/costate_digest.py):")
                print(_model_identity_stamp())
            print("\n".join(_compact_for_hook(lines) if compact else lines))
            # ADDITIVE, FAIL-OPEN (ddm_hw1, task #785): the MAIN retained-reasoning
            # manifest (compaction-cliff cure; ARC-AGI-3 retained-reasoning crosswalk).
            # Printed AFTER the digest so a compaction-recovered session re-loads the
            # live pids/arms/decisions/boundaries. Any failure prints NOTHING (never
            # blinds a session start) — the whole block is guarded and never raises.
            try:
                import main_hot_state as _mhs  # tools/ is on sys.path (line ~48)

                _block = _mhs.digest_block(
                    max_lines=40,
                    max_line_chars=None if not compact else _HOOK_HOT_STATE_MAX_LINE_CHARS,
                )
                if _block:
                    print(_block)
            except Exception:
                pass
        return 0
    except Exception as exc:  # fail-open: a broken digest must never crash a session
        print(f"[costate-digest] unavailable ({type(exc).__name__}: {exc})")
        return 0 if args.session_start else 1


if __name__ == "__main__":
    sys.exit(main())
