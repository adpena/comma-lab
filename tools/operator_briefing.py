#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator briefing: runs the dispatch trio in sequence — one command, full state.

Phases:
  1. Pre-dispatch (Pareto frontier + dispatch one-liners per non-dominated bits)
  2. Post-dispatch general (sorted view of every contest_auth_eval.json on disk)
  3. Post-dispatch apogee_intN (predicted-vs-actual reconciliation)

Use cases:
  - Start of session: see what's ready to dispatch + what's already landed
  - After a dispatch lands: see the new score in dashboard + reconciler verdict
  - Quick "where am I?" between deep work blocks

Usage:
  .venv/bin/python tools/operator_briefing.py                   # all 3 phases
  .venv/bin/python tools/operator_briefing.py --top 10           # cap dashboard rows
  .venv/bin/python tools/operator_briefing.py --skip-pareto      # only dashboard + reconciler
  .venv/bin/python tools/operator_briefing.py --json             # machine-readable composite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TOOLS = REPO_ROOT / "tools"
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

PARETO = TOOLS / "apogee_intN_pareto.py"
DASHBOARD = TOOLS / "score_dashboard.py"
RECONCILER = TOOLS / "predicted_vs_actual_reconciler.py"
CLAIM_DISPATCH = TOOLS / "claim_lane_dispatch.py"
CLOUD_PROVIDER_READINESS = TOOLS / "cloud_provider_readiness.py"
PROVIDER_READINESS_LATEST = REPO_ROOT / "experiments/results/cloud_provider_readiness_latest.json"
PR91_HPM1_READINESS = TOOLS / "audit_pr91_hpm1_readiness.py"
PR91_HPM1_RUNTIME_CONTRACT = TOOLS / "audit_pr91_hpm1_runtime_contract.py"
PR91_HPM1_READINESS_ARTIFACT = (
    REPO_ROOT / "experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json"
)
PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT = (
    REPO_ROOT / "experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json"
)
INVERSE_SCORER_CHAIN_SCAN_ROOT = REPO_ROOT / "experiments/results"
INVERSE_SCORER_CHAIN_MANIFEST_NAME = "inverse_scorer_cell_candidate_chain_manifest.json"
DISPATCH_CLAIMS = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
EXACT_READY_SCAN_ROOTS = (
    REPO_ROOT / "experiments" / "results",
    REPO_ROOT / ".omx" / "research",
)
MATERIALIZER_HANDOFF_SCAN_ROOTS = (
    REPO_ROOT / "experiments" / "results",
    REPO_ROOT / ".omx" / "research",
)
BYTE_SHAVING_ACQUISITION_SCAN_ROOTS = (
    REPO_ROOT / "experiments" / "results",
    REPO_ROOT / ".omx" / "research",
)
FRONTIER_FEEDBACK_SCAN_ROOTS = (
    REPO_ROOT / ".omx" / "research",
    REPO_ROOT / "experiments" / "results",
)
PR95_MLX_CONTROL_PROFILE_SCAN_ROOTS = (
    REPO_ROOT / "experiments" / "results",
    REPO_ROOT / ".omx" / "research",
)
BYTE_SHAVING_MATERIALIZER_CAMPAIGN_RUN_NAME = "materializer_campaign_run.json"
FRONTIER_FEEDBACK_CYCLE_REPORT_NAME = "frontier_rate_attack_feedback_cycle.json"
FRONTIER_FEEDBACK_REFRESH_REPORT_NAME = "feedback_refresh_report.json"
PR95_MLX_MATRIX_MANIFEST_NAME = "matrix_manifest.json"
FRONTIER_FEEDBACK_CYCLE_TOOL = "tools/run_frontier_rate_attack_feedback_cycle.py"
MATERIALIZER_EXACT_EVAL_CONSUMER_TOOL = "tools/build_materializer_exact_eval_consumer.py"
EXACT_READY_SCORE_AXIS_REPAIR_TOOL = "tools/repair_exact_ready_score_axis.py"
EXACT_READY_SUPPRESSION_MANIFEST = (
    REPO_ROOT / ".omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json"
)
L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH = (
    ".omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json"
)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    default_state_path,
    load_queue_definition,
)
from comma_lab.scheduler.experiment_queue_observer import (  # noqa: E402
    observe_experiment_queue,
)
from comma_lab.scheduler.queue_feedback_replan_policy import (  # noqa: E402
    build_queue_observation_recovery_plan,
)
from tac.authority_contract import apply_false_authority_contract  # noqa: E402
from tac.optimization.atw_v2_phase2_gate import (  # noqa: E402
    atw_v2_phase2_gate_status,
)
from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH,
    L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH,
    L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH,
    PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH,
    PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256,
    l5_v2_architecture_lock_packet,
    l5_v2_canonical_sideinfo_gate_evidence,
    l5_v2_dispatch_readiness,
)
from tac.optimization.l5_v2_paired_measurement_dispatch_plan import (  # noqa: E402
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH,
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA,
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH,
)
from tac.optimizer.exact_readiness import (  # noqa: E402
    ACTIVE_FLOOR_SCORE,
    as_bool,
    is_sha256,
    terminal_claim_result_conflicts,
)
from tac.optimizer.exact_ready_audit import (  # noqa: E402
    apply_suppression_manifest,
    audit_exact_ready_queues,
    discover_exact_ready_queues,
    load_suppression_manifest,
)
from tac.optimizer.exact_ready_axis_repair import (  # noqa: E402
    plan_exact_ready_score_axis_repairs_from_audit,
)
from tac.score_target_filter import (  # noqa: E402
    DEFAULT_SCORE_LOWERING_TARGET,
    decide_score_target_routing,
)

# Phase-1 supplementary lanes: pre-registered dispatches that don't fit the
# apogee_intN Pareto matrix but are operator-launchable now. Each entry is
# (lane_id, predicted_band, estimated_cost, council_priority, one-liner).
PHASE_1_SUPPLEMENTARY_LANES = [
    {
        "lane_id": "lane_pr106_latent_sidecar",
        "name": "PR106 + per-pair latent-correction sidecar (PR100 hnerv_lc_v2 pattern)",
        "predicted_band": (0.205, 0.208),
        "estimated_cost_usd": 0.60,
        "council_priority": 1,
        "max_dph": 0.30,
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_latent_sidecar.sh \\\n"
            "  --label lane_pr106_latent_sidecar \\\n"
            "  --predicted-band 0.205 0.208 \\\n"
            "  --estimated-cost 0.60 --council-priority 1 --max-dph 0.30 \\\n"
            "  --env PR106_LATENT_MODE=score_table --env PR106_LATENT_SCORE_TABLE_RESUME=1"
        ),
        "kaggle_bundle_tool": "tools/kaggle_build_pr106_latent_score_table.py",
        "kaggle_harvest_tool": "tools/harvest_kaggle_pr106_latent_score_table.py",
        "kaggle_kernel_slug": "adpena/comma-lab-pr106-latent-score-table",
    },
]


# Phase-1 exact-eval packets: candidates that have already passed local static
# custody checks and are waiting on an exact CUDA submit environment. The packet
# JSON is the command authority so operator briefing cannot drift from the
# claim/submit/harvest commands generated by the packet builder.
PHASE_1_EXACT_EVAL_PACKETS = [
    {
        "lane_id": "wr01_apply_pr106x_half",
        "name": "WR01 half-strength HNeRV latent-sidecar transform on PR106x",
        "packet_path": (
            "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/"
            "wr01_exact_eval_packet.json"
        ),
    },
    {
        "lane_id": "pr106_q10_151byte_brotli",
        "name": "PR106 low-level Brotli q10 151-byte rate candidate",
        "packet_path": (
            "experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/"
            "hnerv_lowlevel_exact_eval_packet.json"
        ),
    },
    {
        "lane_id": "pr106x_lgblock16_1byte_brotli",
        "name": "PR106x low-level Brotli lgblock16 one-byte rate candidate",
        "packet_path": (
            "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/"
            "hnerv_lowlevel_exact_eval_packet.json"
        ),
    },
    {
        "lane_id": "hnerv_hlm1_fixed_latent_recode_exact_eval",
        "name": "PR106 HDM4+HLM1 fixed-latent recode",
        "packet_path": (
            "experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/"
            "hlm1_exact_eval_packet.json"
        ),
    },
    {
        "lane_id": "hnerv_hlm1_xmember_exact_eval_20260514",
        "name": "PR106 HDM4+HLM1 xmember rate-only repack",
        "packet_path": (
            "experiments/results/pr106_r2_hdm4_hlm1_xmember_candidate_20260514_codex/"
            "hlm1_xmember_exact_eval_packet.json"
        ),
    },
]


# Phase-4 gated lanes: pre-registered dispatches that REQUIRE a prior empirical
# result before launch. Per docs/INDEX_score_aware_sidechannel_thread_20260504.md
# decision pipeline, sequential validation prevents wasting GPU spend on
# stacking lanes that interact unexpectedly. Each entry adds a `gate_condition`
# string operator must satisfy before running the one-liner.
PHASE_4_GATED_LANES = [
    {
        "lane_id": "lane_pr106_yshift_sidechannel",
        "name": "PR106 + per-frame Y-shift sidechannel (codex_metric_yshift SC01 mode-7 pattern)",
        "predicted_band": (0.2065, 0.2080),
        "estimated_cost_usd": 0.40,
        "council_priority": 2,
        "max_dph": 0.30,
        "gate_condition": (
            "DISPATCH ONLY IF lane_pr106_latent_sidecar lands < 0.20800 "
            "[contest-CUDA] (per docs/INDEX_score_aware_sidechannel_thread_20260504.md "
            "TICK 2). Verify via: `tools/score_dashboard.py --filter pr106_latent_sidecar`."
        ),
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_yshift_sidechannel.sh \\\n"
            "  --label lane_pr106_yshift_sidechannel \\\n"
            "  --predicted-band 0.2065 0.2080 \\\n"
            "  --estimated-cost 0.40 --council-priority 2 --max-dph 0.30 \\\n"
            "  --env PR106_YSHIFT_MODE=brute_force"
        ),
        "kaggle_bundle_tool": "tools/kaggle_build_pr106_yshift_score_table.py",
        "kaggle_harvest_tool": "tools/harvest_kaggle_pr106_yshift_score_table.py",
        "kaggle_kernel_slug": "adpena/comma-lab-pr106-yshift-score-table",
    },
    {
        "lane_id": "lane_pr106_lrl1_sidechannel",
        "name": "PR106 + per-frame LRL1 luma low-rank correction (codex_metric LRL1 mode-8 pattern, variant #6)",
        "predicted_band": (0.2050, 0.2065),
        "estimated_cost_usd": 0.50,
        "council_priority": 3,
        "max_dph": 0.30,
        "gate_condition": (
            "DISPATCH ONLY IF lane_pr106_yshift_sidechannel lands < 0.20650 [contest-CUDA] "
            "(per docs/INDEX_score_aware_sidechannel_thread_20260504.md TICK 3 — 3rd "
            "stack-on after both variants #1 latent_sidecar AND #3 yshift land empirically). "
            "Verify via: `tools/score_dashboard.py --filter pr106_yshift_sidechannel`."
        ),
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_lrl1_sidechannel.sh \\\n"
            "  --label lane_pr106_lrl1_sidechannel \\\n"
            "  --predicted-band 0.2050 0.2065 \\\n"
            "  --estimated-cost 0.50 --council-priority 3 --max-dph 0.30 \\\n"
            "  --env PR106_LRL1_MODE=brute_force --env PR106_LRL1_K=4"
        ),
    },
]


# Phase-5 composition lanes: meta-composition lanes that compose multiple
# pre-built sister-lane archives into a single dispatch. Gated on ALL the
# sister single-sidechannel lanes landing empirically. Per the score-aware
# sidechannel paradigm decision pipeline, composition lanes are the
# single-dispatch payoff after the sister gates pass.
PHASE_5_COMPOSITION_LANES = [
    {
        "lane_id": "lane_pr106_stacked",
        "name": "PR106 + meta-composition of all 3 score-aware sidechannels (latent + yshift + lrl1)",
        "predicted_band": (0.16, 0.20),
        "estimated_cost_usd": 0.40,
        "council_priority": 4,
        "max_dph": 0.30,
        "gate_condition": (
            "DISPATCH ONLY IF lane_pr106_lrl1_sidechannel + sisters all land < 0.20800 "
            "[contest-CUDA] empirically. Per tools/sidechannel_stack_predictor.py "
            "--bits 5 --all, the int4+full-stack predicted score is 0.163 "
            "(-0.046 vs PR106 0.20945). Composition lane is the single-dispatch "
            "payoff of all 3 sister lanes — verify via "
            "`tools/score_dashboard.py --filter pr106_lrl1_sidechannel`."
        ),
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_stacked.sh \\\n"
            "  --label lane_pr106_stacked \\\n"
            "  --predicted-band 0.16 0.20 \\\n"
            "  --estimated-cost 0.40 --council-priority 4 --max-dph 0.30 \\\n"
            "  --env STACKED_LATENT_ARCHIVE=<path/to/sister_latent_archive.zip> \\\n"
            "  --env STACKED_YSHIFT_ARCHIVE=<path/to/sister_yshift_archive.zip> \\\n"
            "  --env STACKED_LRL1_ARCHIVE=<path/to/sister_lrl1_archive.zip>"
        ),
    },
]


def _annotate_score_target_lanes(
    lanes: list[dict[str, object]],
    *,
    target_score: float,
    active_only: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lane in lanes:
        row = apply_false_authority_contract(
            dict(lane),
            preserve_dispatch_ready=True,
            reason="operator_briefing_planning_row_no_score_authority",
        )
        decision = decide_score_target_routing(
            row.get("predicted_band"),
            target_score=target_score,
            keep_unknown=False,
        )
        row["score_target_routing"] = decision.to_dict()
        dispatch_decision = _lane_dispatch_routing(row, decision_active=decision.active)
        row["dispatch_routing"] = dispatch_decision
        row["ready_for_operator_dispatch"] = bool(dispatch_decision["active"])
        row["ready_for_exact_eval_dispatch"] = (
            bool(row.get("ready_for_exact_eval_dispatch"))
            and bool(dispatch_decision["active"])
        )
        if not active_only or dispatch_decision["active"]:
            rows.append(row)
    return rows


_OPERATOR_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")


def _lane_dispatch_routing(
    lane: dict[str, object],
    *,
    decision_active: bool,
) -> dict[str, object]:
    """Return whether an operator row is actually dispatch-active.

    Score-target routing is intentionally narrow: it answers whether a predicted
    band could beat the target. Dispatch routing additionally enforces
    sequential gates and refuses one-liners with unresolved operator
    placeholders, so autopilot surfaces cannot present a planning row as an
    actionable dispatch.
    """

    blockers: list[str] = []
    if lane.get("gate_condition") and lane.get("gate_ready") is not True:
        configured = lane.get("gate_blockers")
        if isinstance(configured, list) and configured:
            blockers.extend(str(item) for item in configured)
        else:
            blockers.append("gate_condition_not_satisfied")
    one_liner = lane.get("one_liner")
    if isinstance(one_liner, str) and _OPERATOR_PLACEHOLDER_RE.search(one_liner):
        blockers.append("operator_one_liner_has_unresolved_placeholders")
    active = bool(decision_active) and not blockers
    if active:
        status = "dispatch_active"
        reason = "score target plausible and all operator dispatch gates are satisfied"
    elif not decision_active:
        status = "score_target_inactive"
        reason = "score-target routing is inactive"
    else:
        status = "dispatch_gate_blocked"
        reason = "score target plausible, but operator dispatch gate is blocked"
    return {
        "active": active,
        "status": status,
        "reason": reason,
        "blockers": blockers,
    }


def _hidden_above_target_summary(
    lanes: list[dict[str, object]],
    *,
    target_score: float,
) -> str:
    hidden = [
        row
        for row in _annotate_score_target_lanes(
            lanes,
            target_score=target_score,
            active_only=False,
        )
        if not bool(row["dispatch_routing"]["active"])
    ]
    if not hidden:
        return ""
    lane_ids = ", ".join(
        f"{row['lane_id']}[{row['dispatch_routing']['status']}]" for row in hidden
    )
    return (
        f"\n\n  hidden inactive/above target {target_score:.4f}: "
        f"{len(hidden)} row(s): {lane_ids}"
    )


def _score_target_line(lane: dict[str, object]) -> str:
    routing = lane.get("score_target_routing")
    if not isinstance(routing, dict):
        return ""
    dispatch = lane.get("dispatch_routing")
    dispatch_line = ""
    if isinstance(dispatch, dict):
        blockers = dispatch.get("blockers") or []
        blocker_text = f"; blockers={', '.join(str(item) for item in blockers)}" if blockers else ""
        dispatch_line = (
            f"\n    dispatch routing: {dispatch.get('status')} — "
            f"{dispatch.get('reason')}{blocker_text}"
        )
    return (
        f"\n    target routing: {routing.get('status')} — {routing.get('reason')}"
        f"{dispatch_line}"
    )


def _format_supplementary_lanes(
    *,
    target_score: float = DEFAULT_SCORE_LOWERING_TARGET,
    show_above_target: bool = False,
) -> str:
    lanes = _annotate_score_target_lanes(
        PHASE_1_SUPPLEMENTARY_LANES,
        target_score=target_score,
        active_only=not show_above_target,
    )
    lines = []
    for lane in lanes:
        lo, hi = lane["predicted_band"]
        lines.append(
            f"  • {lane['lane_id']} — {lane['name']}\n"
            f"    predicted [{lo:.4f}, {hi:.4f}]   est ${lane['estimated_cost_usd']:.2f}   "
            f"council priority {lane['council_priority']}\n"
            f"{_score_target_line(lane)}\n"
            f"    Operator one-liner:\n"
            f"      {lane['one_liner']}"
        )
    text = "\n\n".join(lines) if lines else f"  (none active below target {target_score:.4f})"
    if not show_above_target:
        text += _hidden_above_target_summary(PHASE_1_SUPPLEMENTARY_LANES, target_score=target_score)
    return text


def _packet_runtime_tree_sha256(packet: dict) -> str | None:
    for key in ("runtime_tree_sha256", "candidate_runtime_tree_sha256"):
        value = packet.get(key)
        if is_sha256(value):
            return str(value).lower()
    for key in ("runtime_manifest", "runtime_custody", "inflate_runtime_manifest"):
        nested = packet.get(key)
        if isinstance(nested, dict):
            value = nested.get("runtime_tree_sha256")
            if is_sha256(value):
                return str(value).lower()
    return None


def _load_exact_eval_packet(lane: dict) -> dict[str, object]:
    path = REPO_ROOT / str(lane["packet_path"])
    if not path.is_file():
        return {
            "lane_id": lane["lane_id"],
            "name": lane["name"],
            "packet_path": lane["packet_path"],
            "ready_for_submit": False,
            "blockers": ["missing_packet_json"],
            "missing_env": [],
            "archive_sha256": None,
            "archive_bytes": None,
            "commands": {},
        }
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "lane_id": lane["lane_id"],
            "name": lane["name"],
            "packet_path": lane["packet_path"],
            "ready_for_submit": False,
            "blockers": [f"malformed_packet_json:{exc.msg}"],
            "missing_env": [],
            "archive_sha256": None,
            "archive_bytes": None,
            "commands": {},
        }
    blockers = list(packet.get("blockers") or packet.get("submit_blockers") or [])
    archive_sha256 = packet.get("archive_sha256")
    runtime_tree_sha256 = _packet_runtime_tree_sha256(packet)
    runtime_changed = as_bool(packet.get("score_affecting_runtime_changed"))
    terminal_blockers = list(packet.get("terminal_exact_eval_evidence_blockers") or [])
    if isinstance(archive_sha256, str):
        live_terminal_blockers = terminal_claim_result_conflicts(
            str(lane["lane_id"]),
            archive_sha256,
            dispatch_claims_path=DISPATCH_CLAIMS,
            active_floor_score=ACTIVE_FLOOR_SCORE,
            runtime_tree_sha256=runtime_tree_sha256,
            score_affecting_runtime_changed=runtime_changed,
            block_runtime_mismatch_for_same_archive=True,
        )
        terminal_blockers.extend(
            blocker for blocker in live_terminal_blockers if blocker not in terminal_blockers
        )
    if (
        packet.get("dispatch_action") == "terminal_exact_eval_evidence_stop"
        and not terminal_blockers
    ):
        terminal_blockers.append("packet_dispatch_action_terminal_exact_eval_evidence_stop")
    if terminal_blockers:
        blockers.extend(
            blocker for blocker in terminal_blockers if blocker not in blockers
        )
    static_ready = bool(
        packet.get("preflight_ready")
        or packet.get("static_packet_ready")
        or packet.get("candidate_static_preflight_ready")
    )
    compliance_ok = bool(
        packet.get("compliance_ok")
        or packet.get("static_compliance_ok")
        or not packet.get("static_blockers")
    )
    artifacts = packet.get("artifacts")
    payload_diff_ready = bool(
        packet.get("payload_diff_ready")
        or packet.get("payload_section_diff_ready")
        or packet.get("linked_lowlevel_changes")
        or (
            isinstance(artifacts, dict)
            and bool(artifacts.get("payload_section_diff"))
        )
    )
    dry_run_ready = bool(
        packet.get("dry_run_ready")
        or packet.get("ready_for_exact_eval_dispatch_claim")
    )
    missing_env = list(packet.get("missing_env") or [])
    submit_gate_blockers: list[str] = []
    if not static_ready:
        submit_gate_blockers.append("static_preflight_not_ready")
    if not compliance_ok:
        submit_gate_blockers.append("static_compliance_not_ok")
    if not payload_diff_ready:
        submit_gate_blockers.append("payload_diff_not_ready")
    if not dry_run_ready:
        submit_gate_blockers.append("dry_run_not_ready")
    if missing_env:
        submit_gate_blockers.append("missing_submit_environment")
    blockers.extend(
        blocker for blocker in submit_gate_blockers if blocker not in blockers
    )
    commands = dict(packet.get("commands") or {})
    packet_suppressed_commands = dict(packet.get("suppressed_commands") or {})
    operator_next_steps = dict(packet.get("operator_next_steps") or {})
    repeat_dispatch_allowed = not terminal_blockers
    if not repeat_dispatch_allowed:
        commands = {}
        operator_next_steps = {
            "schema": "terminal_exact_eval_evidence_stop_v1",
            "reason": "terminal exact-eval evidence exists for this lane/archive; do not repeat-dispatch from operator briefing",
            "terminal_blockers": terminal_blockers,
            "steps": [
                {
                    "id": "review_terminal_cuda_result",
                    "dispatches_remote_gpu": False,
                    "purpose": "read the terminal exact-eval ledger and classify the measured candidate before any new dispatch",
                },
                {
                    "id": "choose_byte_different_successor_candidate",
                    "dispatches_remote_gpu": False,
                    "purpose": "resume only with a byte-different archive/runtime or a new lane claim that is not blocked by terminal evidence",
                },
            ],
        }
    ready_for_submit = (
        bool(packet.get("ready_for_submit"))
        and repeat_dispatch_allowed
        and not submit_gate_blockers
        and not blockers
    )
    return {
        "lane_id": lane["lane_id"],
        "name": lane["name"],
        "packet_path": lane["packet_path"],
        "ready_for_submit": ready_for_submit,
        "repeat_dispatch_allowed": repeat_dispatch_allowed,
        "dispatch_action": (
            "copy_safe_submit_after_gates"
            if ready_for_submit
            else "blocked_static_or_env_gates"
            if repeat_dispatch_allowed
            else "terminal_exact_eval_evidence_stop"
        ),
        "blockers": blockers,
        "terminal_exact_eval_evidence_blockers": terminal_blockers,
        "missing_env": missing_env,
        "submit_gate_blockers": submit_gate_blockers,
        "archive_sha256": archive_sha256,
        "runtime_tree_sha256": runtime_tree_sha256,
        "score_affecting_runtime_changed": runtime_changed,
        "archive_bytes": packet.get("archive_bytes"),
        "preflight_ready": static_ready,
        "compliance_ok": compliance_ok,
        "payload_diff_ready": payload_diff_ready,
        "dry_run_ready": dry_run_ready,
        "commands": commands,
        "suppressed_commands": (
            packet_suppressed_commands or dict(packet.get("commands") or {})
        )
        if not repeat_dispatch_allowed
        else packet_suppressed_commands,
        "operator_next_steps": operator_next_steps,
        "suppressed_operator_next_steps": dict(packet.get("operator_next_steps") or {})
        if not repeat_dispatch_allowed
        else {},
    }


def _exact_eval_packet_summaries() -> list[dict[str, object]]:
    return [_load_exact_eval_packet(lane) for lane in PHASE_1_EXACT_EVAL_PACKETS]


def _format_exact_eval_packets() -> str:
    lines = []
    for packet in _exact_eval_packet_summaries():
        commands = packet.get("commands") or {}
        next_steps = packet.get("operator_next_steps") or {}
        step_rows = next_steps.get("steps") if isinstance(next_steps, dict) else []
        step_ids = (
            ", ".join(str(step.get("id")) for step in step_rows if isinstance(step, dict))
            if isinstance(step_rows, list)
            else ""
        )
        blockers = packet.get("blockers") or []
        missing_env = packet.get("missing_env") or []
        state = "READY" if packet.get("ready_for_submit") else "BLOCKED"
        repeat_dispatch_allowed = bool(packet.get("repeat_dispatch_allowed", True))
        if not repeat_dispatch_allowed:
            claim_command = "(suppressed: terminal exact-eval evidence present)"
            submit_command = "(suppressed: terminal exact-eval evidence present)"
            harvest_command = "(suppressed: terminal exact-eval evidence present)"
        else:
            claim_command = commands.get("claim", "(missing)")
            submit_command = commands.get("submit", "(missing)")
            harvest_command = commands.get("harvest", "(missing)")
        lines.append(
            f"  • {packet['lane_id']} — {packet['name']}\n"
            f"    state {state}   bytes {packet.get('archive_bytes')}   "
            f"sha256 {packet.get('archive_sha256')}\n"
            f"    dispatch_action: {packet.get('dispatch_action', '<unset>')}\n"
            f"    static gates: preflight={packet.get('preflight_ready')} "
            f"compliance={packet.get('compliance_ok')} "
            f"payload_diff={packet.get('payload_diff_ready')} "
            f"dry_run={packet.get('dry_run_ready')}\n"
            f"    blockers: {', '.join(blockers) if blockers else '(none)'}\n"
            f"    missing env: {', '.join(missing_env) if missing_env else '(none)'}\n"
            f"    packet: {packet['packet_path']}\n"
            f"    Claim:\n"
            f"      {claim_command}\n"
            f"    Submit:\n"
            f"      {submit_command}\n"
            f"    Harvest:\n"
            f"      {harvest_command}\n"
            f"    Copy-safe next steps:\n"
            f"      {step_ids if step_ids else '(missing)'}"
        )
    return "\n\n".join(lines) if lines else "  (none)"


def _exact_ready_queue_audit() -> dict[str, object]:
    queues = discover_exact_ready_queues(
        repo_root=REPO_ROOT,
        scan_root=EXACT_READY_SCAN_ROOTS,
    )
    payload = audit_exact_ready_queues(
        queues,
        repo_root=REPO_ROOT,
        dispatch_claims_path=DISPATCH_CLAIMS,
        active_floor_score=ACTIVE_FLOOR_SCORE,
    )
    if EXACT_READY_SUPPRESSION_MANIFEST.is_file():
        try:
            return apply_suppression_manifest(
                payload,
                manifest=load_suppression_manifest(EXACT_READY_SUPPRESSION_MANIFEST),
                manifest_path=EXACT_READY_SUPPRESSION_MANIFEST,
                repo_root=REPO_ROOT,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            payload["suppression_manifest_error"] = str(exc)
    return payload


def _format_exact_ready_queue_audit() -> str:
    payload = _exact_ready_queue_audit()
    lines = [
        f"queues_scanned: {payload['queue_count']}",
        f"passed: {payload['passed']}",
        f"stale_ready_rows: {payload['stale_ready_row_count']}",
    ]
    if payload.get("stale_ready_row_count"):
        lines.append("blocked rows:")
        for queue in payload.get("queues", []):
            if not isinstance(queue, dict):
                continue
            for row in queue.get("stale_ready_rows", []):
                if not isinstance(row, dict):
                    continue
                blockers = ", ".join(str(b) for b in row.get("blockers", []))
                lines.append(
                    "  - "
                    f"{queue.get('queue_path')} :: {row.get('candidate_id')} "
                    f"lane={row.get('lane_id')} runtime={row.get('runtime_tree_sha256')} "
                    f"blockers={blockers}"
                )
    return "\n".join(lines)


def _exact_ready_score_axis_repair_summary() -> dict[str, object]:
    audit = _exact_ready_queue_audit()
    plan = plan_exact_ready_score_axis_repairs_from_audit(audit)
    rows = plan.get("rows")
    row_list = rows if isinstance(rows, list) else []
    repairable_rows = [
        row
        for row in row_list
        if isinstance(row, dict) and row.get("status") == "repairable"
    ]
    skipped_rows = [
        row
        for row in row_list
        if isinstance(row, dict) and row.get("status") != "repairable"
    ]
    skip_reasons = _unique_strings(
        [row.get("skip_reason") for row in skipped_rows if row.get("skip_reason")]
    )
    if repairable_rows:
        next_command = (
            "UTC=$(date -u +%Y%m%dT%H%M%SZ); "
            f".venv/bin/python {EXACT_READY_SCORE_AXIS_REPAIR_TOOL} "
            "--out-dir .omx/research/exact_ready_score_axis_repair_${UTC} "
            "--report-out .omx/research/exact_ready_score_axis_repair_${UTC}.json"
            " --write-repaired-queues"
        )
    else:
        next_command = f".venv/bin/python {EXACT_READY_SCORE_AXIS_REPAIR_TOOL} --help"
    return {
        "schema": "pact.exact_ready_score_axis_repair_summary.v1",
        "repair_tool": EXACT_READY_SCORE_AXIS_REPAIR_TOOL,
        "repair_tool_exists": (REPO_ROOT / EXACT_READY_SCORE_AXIS_REPAIR_TOOL).is_file(),
        "status": "REPAIRABLE" if repairable_rows else "NO_AXIS_ONLY_REPAIR",
        "reason": (
            f"{len(repairable_rows)} score-axis-only legacy row(s) can be copied "
            "through the reviewed repair tool"
            if repairable_rows
            else "no unresolved exact-ready row is blocked only by missing score axis"
        ),
        "queue_count": plan.get("queue_count"),
        "stale_ready_row_count": plan.get("stale_ready_row_count"),
        "repairable_or_repaired_count": plan.get("repairable_or_repaired_count"),
        "skipped_count": plan.get("skipped_count"),
        "automatic_mutation_count": plan.get("automatic_mutation_count"),
        "skip_reasons": skip_reasons[:8],
        "repairable_rows": repairable_rows[:8],
        "next_command": next_command,
        **_false_authority_fields(),
    }


def _format_exact_ready_score_axis_repairs() -> str:
    payload = _exact_ready_score_axis_repair_summary()
    lines = [
        f"status: {payload['status']} — {payload['reason']}",
        f"repair tool: {payload['repair_tool']} present={payload['repair_tool_exists']}",
        (
            "counts: "
            f"queues={payload['queue_count']} "
            f"stale_rows={payload['stale_ready_row_count']} "
            f"repairable={payload['repairable_or_repaired_count']} "
            f"skipped={payload['skipped_count']} "
            f"automatic_mutations={payload['automatic_mutation_count']}"
        ),
    ]
    repairable_rows = payload.get("repairable_rows")
    if isinstance(repairable_rows, list) and repairable_rows:
        lines.append("repairable rows:")
        for row in repairable_rows[:5]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  - "
                f"{row.get('source_queue_path')} :: {row.get('candidate_id')} "
                f"axis={row.get('score_axis')}"
            )
    skip_reasons = payload.get("skip_reasons")
    if isinstance(skip_reasons, list) and skip_reasons:
        lines.append("first skip reasons:")
        for reason in skip_reasons[:6]:
            lines.append(f"  - {reason}")
    lines.append("next command:")
    lines.append(f"  {payload['next_command']}")
    return "\n".join(lines)


def _false_authority_fields() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _materializer_handoff_paths(
    scan_roots: tuple[Path, ...] | None = None,
) -> list[Path]:
    if scan_roots is None:
        scan_roots = MATERIALIZER_HANDOFF_SCAN_ROOTS
    patterns = (
        "exact_readiness_bridge_report.json",
        "*/exact_readiness_bridge_report.json",
        "*/*/exact_readiness_bridge_report.json",
        "materializer_chain_exact_readiness_bridge_report_*.json",
        "*/materializer_chain_exact_readiness_bridge_report_*.json",
        "*/*/materializer_chain_exact_readiness_bridge_report_*.json",
        "consumer_report.json",
        "*/consumer_report.json",
        "*/*/consumer_report.json",
        "materializer_exact_eval_consumer*.json",
        "*/materializer_exact_eval_consumer*.json",
        "*/*/materializer_exact_eval_consumer*.json",
        "dispatch_plan.json",
        "*/dispatch_plan.json",
        "*/*/dispatch_plan.json",
        "materializer_exact_eval_dispatch_plan_*.json",
        "*/materializer_exact_eval_dispatch_plan_*.json",
        "*/*/materializer_exact_eval_dispatch_plan_*.json",
    )
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve(strict=False)
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                paths.append(path)
    return sorted(
        paths,
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )


def _unique_strings(values: list[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _queue_recovery_group_summaries(groups: object, *, limit: int = 3) -> list[str]:
    if not isinstance(groups, list):
        return []
    out: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        family = str(group.get("blocker_family") or "unknown")
        scope_kind = str(group.get("scope_kind") or "scope")
        scope_value = str(group.get("scope_value") or "unknown")
        count = _safe_int(group.get("count"))
        repeated = group.get("repeated") is True
        out.append(
            f"{family}:{scope_kind}={scope_value}:count={count}:repeated={repeated}"
        )
        if len(out) >= limit:
            break
    return out


def _row_exact_ready_queue_paths(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    return _unique_strings(
        [
            row.get("exact_ready_queue_path")
            for row in rows
            if isinstance(row, dict) and row.get("exact_ready_queue_path")
        ]
    )


def _row_values(rows: object, key: str) -> list[str]:
    if not isinstance(rows, list):
        return []
    return _unique_strings(
        [
            row.get(key)
            for row in rows
            if isinstance(row, dict) and row.get(key)
        ]
    )


def _row_values_where(rows: object, key: str, truthy_key: str) -> list[str]:
    if not isinstance(rows, list):
        return []
    return _unique_strings(
        [
            row.get(key)
            for row in rows
            if isinstance(row, dict) and row.get(truthy_key) is True and row.get(key)
        ]
    )


def _row_blockers(rows: object, *, limit: int = 12) -> list[str]:
    if not isinstance(rows, list):
        return []
    blockers: list[object] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_blockers = row.get("blockers")
        if isinstance(row_blockers, list):
            blockers.extend(row_blockers)
    return _unique_strings(blockers)[:limit]


def _materializer_handoff_summary_row(path: Path) -> dict[str, object]:
    payload = _load_json_file(path)
    base: dict[str, object] = {
        "path": _repo_rel(path),
        "sha256": _sha256_file(path) if path.is_file() else "",
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
        **_false_authority_fields(),
    }
    if "_error" in payload:
        return {
            **base,
            "kind": "unreadable",
            "schema": None,
            "authorized_candidate_count": 0,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "blockers": [str(payload["_error"])],
        }
    schema = payload.get("schema")
    if schema == "materializer_chain_exact_readiness_bridge_report.v1":
        rows = payload.get("rows")
        bridge_rows = rows if isinstance(rows, list) else []
        ready_count = _safe_int(payload.get("ready_candidate_count")) or sum(
            1
            for row in bridge_rows
            if isinstance(row, dict) and row.get("exact_ready_queue_written") is True
        )
        blocked_count = _safe_int(payload.get("blocked_candidate_count")) or sum(
            1
            for row in bridge_rows
            if isinstance(row, dict) and row.get("exact_ready_queue_written") is not True
        )
        return {
            **base,
            "kind": "bridge_report",
            "schema": schema,
            "ready_candidate_count": ready_count,
            "blocked_candidate_count": blocked_count,
            "row_count": len(bridge_rows),
            "exact_ready_queue_paths": _row_exact_ready_queue_paths(bridge_rows),
            "blockers": [],
        }
    if schema == "materializer_exact_eval_consumer.v1":
        consumer_rows = payload.get("rows")
        return {
            **base,
            "kind": "consumer_report",
            "schema": schema,
            "authorized_candidate_count": _safe_int(
                payload.get("authorized_candidate_count")
            ),
            "blocked_candidate_count": _safe_int(
                payload.get("blocked_candidate_count")
            ),
            "duplicate_candidate_count": _safe_int(
                payload.get("duplicate_candidate_count")
            ),
            "experiment_queue_id": str(payload.get("experiment_queue_id") or ""),
            "candidate_ids": _row_values(consumer_rows, "candidate_id"),
            "archive_sha256s": _row_values(consumer_rows, "archive_sha256"),
            "stable_identities": _row_values(consumer_rows, "stable_identity"),
            "runtime_content_tree_sha256s": _row_values(
                consumer_rows,
                "runtime_content_tree_sha256",
            ),
            "runtime_tree_sha256s": _row_values(consumer_rows, "runtime_tree_sha256"),
            "score_axes": _row_values(consumer_rows, "score_axis"),
            "authorized_stable_identities": _row_values_where(
                consumer_rows,
                "stable_identity",
                "authorized_for_paused_dry_run_queue",
            ),
            "blocked_stable_identities": _unique_strings(
                [
                    row.get("stable_identity") or row.get("candidate_id")
                    for row in consumer_rows
                    if isinstance(row, dict) and row.get("blockers")
                ]
                if isinstance(consumer_rows, list)
                else []
            ),
            "exact_ready_queue_paths": _row_exact_ready_queue_paths(consumer_rows),
            "hard_plan_blockers": [
                str(item) for item in payload.get("hard_plan_blockers", []) if str(item)
            ]
            if isinstance(payload.get("hard_plan_blockers"), list)
            else [],
            "blockers": _row_blockers(consumer_rows),
        }
    if schema == "experiment_queue.v1":
        experiments = payload.get("experiments")
        return {
            **base,
            "kind": "consumer_experiment_queue",
            "schema": schema,
            "experiment_queue_id": str(payload.get("queue_id") or payload.get("id") or ""),
            "experiment_count": len(experiments) if isinstance(experiments, list) else 0,
            "authorized_candidate_count": 0,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "blockers": [],
        }
    if schema == "materializer_exact_eval_dispatch_plan.v1":
        plan_blockers = payload.get("plan_blockers")
        hard_plan_blockers = payload.get("hard_plan_blockers")
        plan_rows = payload.get("rows")
        return {
            **base,
            "kind": "dispatch_plan",
            "schema": schema,
            "authorized_candidate_count": _safe_int(
                payload.get("authorized_candidate_count")
            ),
            "blocked_candidate_count": _safe_int(
                payload.get("blocked_candidate_count")
            ),
            "duplicate_candidate_count": _safe_int(
                payload.get("duplicate_candidate_count")
            ),
            "dispatch_mode": str(payload.get("dispatch_mode") or ""),
            "experiment_queue_id": str(payload.get("experiment_queue_id") or ""),
            "plan_blockers": [str(item) for item in plan_blockers if str(item)]
            if isinstance(plan_blockers, list)
            else [],
            "hard_plan_blockers": [str(item) for item in hard_plan_blockers if str(item)]
            if isinstance(hard_plan_blockers, list)
            else [],
            "blockers": _row_blockers(plan_rows),
        }
    return {
        **base,
        "kind": "ignored",
        "schema": schema if isinstance(schema, str) else None,
        "authorized_candidate_count": 0,
        "blocked_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "blockers": [f"unsupported_materializer_handoff_schema:{schema!r}"],
    }


def _materializer_row_recency(row: dict[str, object]) -> int:
    return _safe_int(row.get("mtime_ns"))


def _materializer_row_identity(row: dict[str, object]) -> str:
    kind = str(row.get("kind") or "")
    experiment_queue_id = str(row.get("experiment_queue_id") or "").strip()
    stable_identities = row.get("stable_identities")
    if isinstance(stable_identities, list) and stable_identities:
        stable = ",".join(str(item) for item in stable_identities if str(item))
        if stable:
            return f"{kind}:queue={experiment_queue_id}:stable={stable}"
    candidate_ids = row.get("candidate_ids")
    archive_sha256s = row.get("archive_sha256s")
    if isinstance(candidate_ids, list) and candidate_ids:
        candidates = ",".join(str(item) for item in candidate_ids if str(item))
        archives = (
            ",".join(str(item) for item in archive_sha256s if str(item))
            if isinstance(archive_sha256s, list)
            else ""
        )
        runtime_contents = row.get("runtime_content_tree_sha256s")
        runtime_content = (
            ",".join(str(item) for item in runtime_contents if str(item))
            if isinstance(runtime_contents, list)
            else ""
        )
        runtime_trees = row.get("runtime_tree_sha256s")
        runtime_tree = (
            ",".join(str(item) for item in runtime_trees if str(item))
            if isinstance(runtime_trees, list)
            else ""
        )
        score_axes = row.get("score_axes")
        score_axis = (
            ",".join(str(item) for item in score_axes if str(item))
            if isinstance(score_axes, list)
            else ""
        )
        return (
            f"{kind}:queue={experiment_queue_id}:candidates={candidates}"
            f":archives={archives}:runtime_content={runtime_content}"
            f":runtime_tree={runtime_tree}:score_axis={score_axis}"
        )
    exact_ready_paths = row.get("exact_ready_queue_paths")
    if isinstance(exact_ready_paths, list):
        joined = ",".join(str(path) for path in exact_ready_paths if str(path))
        if joined:
            return f"{kind}:queue={experiment_queue_id}:inputs={joined}"
    return f"{kind}:path={row.get('path') or ''}"


def _latest_materializer_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    latest_by_identity: dict[str, dict[str, object]] = {}
    for row in rows:
        identity = _materializer_row_identity(row)
        current = latest_by_identity.get(identity)
        if current is None or _materializer_row_recency(row) > _materializer_row_recency(
            current
        ):
            latest_by_identity[identity] = row
    return sorted(
        latest_by_identity.values(),
        key=_materializer_row_recency,
        reverse=True,
    )


def _materializer_exact_eval_consumer_next_command(
    *,
    bridge_report_paths: list[str],
    exact_ready_queue_paths: list[str],
) -> str:
    input_args: list[str] = []
    if bridge_report_paths:
        input_args = [f"--bridge-report {path}" for path in bridge_report_paths[:3]]
    elif exact_ready_queue_paths:
        input_args = [
            f"--exact-ready-queue {path}" for path in exact_ready_queue_paths[:3]
        ]
    else:
        return f".venv/bin/python {MATERIALIZER_EXACT_EVAL_CONSUMER_TOOL} --help"
    return (
        "UTC=$(date -u +%Y%m%dT%H%M%SZ); "
        f".venv/bin/python {MATERIALIZER_EXACT_EVAL_CONSUMER_TOOL} "
        f"{' '.join(input_args)} "
        "--consumer-report-out "
        ".omx/research/materializer_exact_eval_consumer_report_${UTC}.json "
        "--experiment-queue-out "
        ".omx/research/materializer_exact_eval_consumer_experiment_queue_${UTC}.json"
    )


def _materializer_exact_ready_handoff_summary() -> dict[str, object]:
    discovered_rows = [
        row
        for row in (
            _materializer_handoff_summary_row(path)
            for path in _materializer_handoff_paths()
        )
        if row.get("kind") != "ignored"
    ]
    bridge_rows = _latest_materializer_rows(
        [row for row in discovered_rows if row.get("kind") == "bridge_report"]
    )
    consumer_rows = _latest_materializer_rows(
        [row for row in discovered_rows if row.get("kind") == "consumer_report"]
    )
    consumer_queue_rows = _latest_materializer_rows(
        [row for row in discovered_rows if row.get("kind") == "consumer_experiment_queue"]
    )
    dispatch_rows = _latest_materializer_rows(
        [row for row in discovered_rows if row.get("kind") == "dispatch_plan"]
    )
    error_rows = [row for row in discovered_rows if row.get("kind") == "unreadable"]
    rows = sorted(
        [*bridge_rows, *consumer_rows, *consumer_queue_rows, *dispatch_rows, *error_rows],
        key=_materializer_row_recency,
        reverse=True,
    )
    bridge_report_paths = [str(row.get("path")) for row in bridge_rows]
    consumer_output_paths = [str(row.get("path")) for row in consumer_rows]
    consumer_output_paths.extend(str(row.get("path")) for row in consumer_queue_rows)
    exact_ready_queue_paths = _unique_strings(
        [
            path
            for row in [*bridge_rows, *consumer_rows]
            for path in (
                row.get("exact_ready_queue_paths", [])
                if isinstance(row.get("exact_ready_queue_paths"), list)
                else []
            )
        ]
    )
    consumer_authorized = sum(
        _safe_int(row.get("authorized_candidate_count")) for row in consumer_rows
    )
    unique_consumer_authorized = _unique_strings(
        [
            identity
            for row in consumer_rows
            for identity in (
                row.get("authorized_stable_identities", [])
                if isinstance(row.get("authorized_stable_identities"), list)
                else []
            )
        ]
    )
    if unique_consumer_authorized:
        consumer_authorized = len(unique_consumer_authorized)
    dispatch_authorized = sum(
        _safe_int(row.get("authorized_candidate_count")) for row in dispatch_rows
    )
    blocked_count = sum(
        _safe_int(row.get("blocked_candidate_count")) for row in consumer_rows
    )
    unique_consumer_blocked = _unique_strings(
        [
            identity
            for row in consumer_rows
            for identity in (
                row.get("blocked_stable_identities", [])
                if isinstance(row.get("blocked_stable_identities"), list)
                else []
            )
        ]
    )
    if unique_consumer_blocked:
        blocked_count = len(unique_consumer_blocked)
    blocked_count += sum(
        _safe_int(row.get("blocked_candidate_count")) for row in dispatch_rows
    )
    hard_blockers = [
        blocker
        for row in [*consumer_rows, *dispatch_rows]
        for blocker in (
            row.get("hard_plan_blockers")
            if isinstance(row.get("hard_plan_blockers"), list)
            else []
        )
    ]
    row_blockers = [
        blocker
        for row in [*consumer_rows, *dispatch_rows]
        for blocker in (
            row.get("blockers") if isinstance(row.get("blockers"), list) else []
        )
    ]
    if error_rows:
        status = "BLOCKED"
        reason = f"{len(error_rows)} materializer handoff artifact(s) failed JSON load"
    elif hard_blockers:
        status = "BLOCKED"
        reason = (
            f"{len(hard_blockers)} hard plan blocker(s) freeze materializer "
            "handoff authority"
        )
    elif consumer_authorized or dispatch_authorized:
        status = "READY"
        reason = (
            f"{consumer_authorized} consumer row(s), {dispatch_authorized} dispatch-plan "
            "row(s) authorized into paused queue handoffs"
        )
    elif bridge_rows or consumer_rows or dispatch_rows:
        status = "PENDING"
        reason = "materializer handoff artifacts exist but no authorized paused queue rows"
    else:
        status = "PENDING"
        reason = "no materializer exact-ready handoff artifacts found"
    return {
        "schema": "pact.materializer_exact_ready_handoff_summary.v1",
        "scan_roots": [_repo_rel(root) for root in MATERIALIZER_HANDOFF_SCAN_ROOTS],
        "consumer_tool": MATERIALIZER_EXACT_EVAL_CONSUMER_TOOL,
        "consumer_tool_exists": (
            REPO_ROOT / MATERIALIZER_EXACT_EVAL_CONSUMER_TOOL
        ).is_file(),
        "status": status,
        "reason": reason,
        "discoverability_status": "VISIBLE"
        if consumer_output_paths
        else (
            "NEXT_COMMAND_AVAILABLE"
            if bridge_report_paths or exact_ready_queue_paths
            else "NO_RECENT_INPUTS"
        ),
        "bridge_report_count": len(bridge_rows),
        "bridge_ready_candidate_count": sum(
            _safe_int(row.get("ready_candidate_count")) for row in bridge_rows
        ),
        "bridge_blocked_candidate_count": sum(
            _safe_int(row.get("blocked_candidate_count")) for row in bridge_rows
        ),
        "consumer_report_count": len(consumer_rows),
        "consumer_authorized_candidate_count": consumer_authorized,
        "consumer_blocked_candidate_count": len(unique_consumer_blocked)
        if unique_consumer_blocked
        else sum(_safe_int(row.get("blocked_candidate_count")) for row in consumer_rows),
        "consumer_duplicate_candidate_count": sum(
            _safe_int(row.get("duplicate_candidate_count")) for row in consumer_rows
        ),
        "consumer_experiment_queue_count": len(consumer_queue_rows),
        "dispatch_plan_count": len(dispatch_rows),
        "dispatch_plan_authorized_candidate_count": dispatch_authorized,
        "dispatch_plan_blocked_candidate_count": sum(
            _safe_int(row.get("blocked_candidate_count")) for row in dispatch_rows
        ),
        "dispatch_plan_duplicate_candidate_count": sum(
            _safe_int(row.get("duplicate_candidate_count")) for row in dispatch_rows
        ),
        "blocked_candidate_count": blocked_count,
        "hard_plan_blocker_count": len(hard_blockers),
        "top_blockers": _unique_strings(row_blockers + hard_blockers)[:12],
        "error_count": len(error_rows),
        "scanned_handoff_artifact_count": len(discovered_rows),
        "superseded_handoff_artifact_count": max(0, len(discovered_rows) - len(rows)),
        "recent_consumer_output_paths": _unique_strings(consumer_output_paths)[:5],
        "recent_bridge_report_paths": _unique_strings(bridge_report_paths)[:5],
        "recent_exact_ready_queue_paths": exact_ready_queue_paths[:5],
        "next_command": _materializer_exact_eval_consumer_next_command(
            bridge_report_paths=_unique_strings(bridge_report_paths),
            exact_ready_queue_paths=exact_ready_queue_paths,
        ),
        "latest_rows": rows[:8],
        **_false_authority_fields(),
    }


def _format_materializer_exact_ready_handoffs() -> str:
    payload = _materializer_exact_ready_handoff_summary()
    lines = [
        f"status: {payload['status']} — {payload['reason']}",
        f"discoverability: {payload['discoverability_status']}",
        f"consumer tool: {payload['consumer_tool']} present={payload['consumer_tool_exists']}",
        (
            "bridge reports: "
            f"{payload['bridge_report_count']} "
            f"(ready={payload['bridge_ready_candidate_count']} "
            f"blocked={payload['bridge_blocked_candidate_count']})"
        ),
        (
            "consumer reports: "
            f"{payload['consumer_report_count']} "
            f"(authorized={payload['consumer_authorized_candidate_count']} "
            f"blocked={payload['consumer_blocked_candidate_count']} "
            f"duplicates={payload['consumer_duplicate_candidate_count']})"
        ),
        f"consumer experiment queues: {payload['consumer_experiment_queue_count']}",
        (
            "dispatch plans: "
            f"{payload['dispatch_plan_count']} "
            f"(authorized={payload['dispatch_plan_authorized_candidate_count']} "
            f"blocked={payload['dispatch_plan_blocked_candidate_count']} "
            f"duplicates={payload['dispatch_plan_duplicate_candidate_count']})"
        ),
        (
            "artifact scan: "
            f"{payload['scanned_handoff_artifact_count']} found, "
            f"{payload['superseded_handoff_artifact_count']} superseded by newer "
            "queue-identity reports"
        ),
    ]
    latest = payload.get("latest_rows")
    if isinstance(latest, list) and latest:
        lines.append("latest handoffs:")
        for row in latest[:5]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  - "
                f"{row.get('kind')} {row.get('path')} "
                f"authorized={row.get('authorized_candidate_count', row.get('ready_candidate_count', 0))} "
                f"blocked={row.get('blocked_candidate_count', 0)}"
            )
    blockers = payload.get("top_blockers")
    if isinstance(blockers, list) and blockers:
        lines.append("first blockers:")
        for blocker in blockers[:8]:
            lines.append(f"  - {blocker}")
    outputs = payload.get("recent_consumer_output_paths")
    if isinstance(outputs, list) and outputs:
        lines.append("recent consumer outputs:")
        for path in outputs[:5]:
            lines.append(f"  - {path}")
    exact_ready_inputs = payload.get("recent_exact_ready_queue_paths")
    if isinstance(exact_ready_inputs, list) and exact_ready_inputs:
        lines.append("recent exact-ready inputs:")
        for path in exact_ready_inputs[:5]:
            lines.append(f"  - {path}")
    lines.append("next command:")
    lines.append(f"  {payload['next_command']}")
    return "\n".join(lines)


def _byte_shaving_acquisition_run_paths(
    scan_roots: tuple[Path, ...] | None = None,
) -> list[Path]:
    if scan_roots is None:
        scan_roots = BYTE_SHAVING_ACQUISITION_SCAN_ROOTS
    patterns = tuple(
        f"{'*/' * depth}{BYTE_SHAVING_MATERIALIZER_CAMPAIGN_RUN_NAME}"
        for depth in range(4)
    )
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve(strict=False)
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                paths.append(path)
    return sorted(
        paths,
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )


def _frontier_feedback_artifact_paths(
    filename: str,
    *,
    scan_roots: tuple[Path, ...] | None = None,
    max_depth: int = 5,
) -> list[Path]:
    if scan_roots is None:
        scan_roots = FRONTIER_FEEDBACK_SCAN_ROOTS
    patterns = tuple(f"{'*/' * depth}{filename}" for depth in range(max_depth))
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve(strict=False)
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                paths.append(path)
    return sorted(
        paths,
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )


def _authority_truthy(payload: dict[str, object]) -> list[str]:
    bad: list[str] = []
    for key in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
    ):
        if payload.get(key) is True:
            bad.append(key)
    return bad


def _frontier_feedback_refresh_row(path: Path) -> dict[str, object]:
    payload = _load_json_file(path)
    rel = _repo_rel(path)
    if payload.get("_error"):
        return {
            "kind": "frontier_feedback_refresh",
            "path": rel,
            "status": "ERROR",
            "error": payload["_error"],
            **_false_authority_fields(),
        }
    bad_authority = _authority_truthy(payload)
    eureka_payload = (
        payload.get("local_cpu_eureka_planning")
        if isinstance(payload.get("local_cpu_eureka_planning"), dict)
        else {}
    )
    bad_authority.extend(
        f"local_cpu_eureka_planning.{key}" for key in _authority_truthy(eureka_payload)
    )
    if bad_authority:
        return {
            "kind": "frontier_feedback_refresh",
            "path": rel,
            "status": "BLOCKED_AUTHORITY_LEAK",
            "blockers": [f"truthy_authority:{key}" for key in bad_authority],
            **_false_authority_fields(),
        }
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    commands = (
        payload.get("operator_commands")
        if isinstance(payload.get("operator_commands"), dict)
        else {}
    )
    cycle_command_parts = commands.get("run_frontier_feedback_cycle")
    eureka = eureka_payload
    queue_path = str(artifacts.get("dqs1_followup_queue") or "")
    retention = payload.get("retention_policy")
    retention_policy = retention if isinstance(retention, dict) else {}
    selected_candidate_ids = _first_list(payload, "selected_candidate_ids")
    eureka_hints = [
        row for row in eureka.get("planner_hints", []) if isinstance(row, dict)
    ]
    active_pairset_profiles = [
        row.get("pairset_acquisition_profile")
        for row in eureka_hints
        if isinstance(row.get("pairset_acquisition_profile"), dict)
        and row["pairset_acquisition_profile"].get("active") is True
    ]
    return {
        "kind": "frontier_feedback_refresh",
        "path": rel,
        "schema": str(payload.get("schema") or ""),
        "status": "READY_QUEUE" if queue_path else "BRIDGE_ONLY",
        "queue_path": queue_path,
        "queue_id": str(payload.get("queue_id") or ""),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_candidate_count": len(selected_candidate_ids),
        "selected_drop_many_candidate_count": sum(
            1 for candidate_id in selected_candidate_ids if "drop_many" in candidate_id
        ),
        "selected_geometry_candidate_count": sum(
            1 for candidate_id in selected_candidate_ids if "geometry" in candidate_id
        ),
        "materializer_feedback_payload_count": _safe_int(
            payload.get("materializer_feedback_payload_count")
        ),
        "dqs1_observation_count": _safe_int(payload.get("dqs1_observation_count")),
        "eureka_signal_count": _safe_int(eureka.get("signal_count")),
        "eureka_planner_hint_count": _safe_int(eureka.get("planner_hint_count")),
        "eureka_planner_hint_ids": [
            str(row.get("hint_id"))
            for row in eureka.get("planner_hints", [])
            if isinstance(row, dict) and str(row.get("hint_id") or "").strip()
        ],
        "eureka_pairset_profile_active": bool(active_pairset_profiles),
        "eureka_drop_many_counts": (
            active_pairset_profiles[0].get("drop_many_counts")
            if active_pairset_profiles
            else []
        ),
        "eureka_max_drop_many": (
            active_pairset_profiles[0].get("max_drop_many")
            if active_pairset_profiles
            else None
        ),
        "raw_retention_execute": bool(retention_policy.get("raw_retention_execute")),
        "raw_retention_action": retention_policy.get("raw_retention_action"),
        "mlx_retention_execute": bool(retention_policy.get("mlx_retention_execute")),
        "cycle_command": (
            " ".join(str(part) for part in cycle_command_parts)
            if isinstance(cycle_command_parts, list)
            else ""
        ),
        **_false_authority_fields(),
    }


def _frontier_feedback_cycle_row(path: Path) -> dict[str, object]:
    payload = _load_json_file(path)
    rel = _repo_rel(path)
    if payload.get("_error"):
        return {
            "kind": "frontier_feedback_cycle",
            "path": rel,
            "status": "ERROR",
            "error": payload["_error"],
            **_false_authority_fields(),
        }
    bad_authority = _authority_truthy(payload)
    if bad_authority:
        return {
            "kind": "frontier_feedback_cycle",
            "path": rel,
            "status": "BLOCKED_AUTHORITY_LEAK",
            "blockers": [f"truthy_authority:{key}" for key in bad_authority],
            **_false_authority_fields(),
        }
    initial = (
        payload.get("initial_refresh")
        if isinstance(payload.get("initial_refresh"), dict)
        else {}
    )
    harvest = (
        payload.get("harvest_signal")
        if isinstance(payload.get("harvest_signal"), dict)
        else {}
    )
    post = (
        payload.get("post_harvest_refresh")
        if isinstance(payload.get("post_harvest_refresh"), dict)
        else None
    )
    artifacts = (
        initial.get("artifacts") if isinstance(initial.get("artifacts"), dict) else {}
    )
    validate = (
        initial.get("queue_validate")
        if isinstance(initial.get("queue_validate"), dict)
        else {}
    )
    selected = (
        initial.get("selected_candidate_ids")
        if isinstance(initial.get("selected_candidate_ids"), list)
        else []
    )
    harvest_count = _safe_int(harvest.get("harvest_path_count"))
    post_artifacts = (
        post.get("artifacts")
        if isinstance(post, dict) and isinstance(post.get("artifacts"), dict)
        else {}
    )
    if harvest_count and isinstance(post, dict) and post_artifacts.get("dqs1_followup_queue"):
        status = "POST_HARVEST_QUEUE_READY"
    elif artifacts.get("dqs1_followup_queue") and validate.get("valid") is True:
        status = "READY_LOCAL_EXECUTION"
    elif artifacts.get("dqs1_followup_queue"):
        status = "READY_QUEUE_UNVALIDATED"
    else:
        status = "PENDING"
    return {
        "kind": "frontier_feedback_cycle",
        "path": rel,
        "schema": str(payload.get("schema") or ""),
        "status": status,
        "initial_queue_path": str(artifacts.get("dqs1_followup_queue") or ""),
        "initial_queue_valid": validate.get("valid") is True,
        "initial_selected_candidate_ids": selected,
        "initial_selected_candidate_count": len(selected),
        "harvest_path_count": harvest_count,
        "observation_jsonl": str(
            (
                harvest.get("observation_bundle")
                if isinstance(harvest.get("observation_bundle"), dict)
                else {}
            ).get("observation_jsonl")
            or ""
        ),
        "post_harvest_queue_path": str(post_artifacts.get("dqs1_followup_queue") or ""),
        **_false_authority_fields(),
    }


def _frontier_feedback_cycle_execute_command(latest: dict[str, object] | None) -> str:
    if not latest:
        return (
            f".venv/bin/python {FRONTIER_FEEDBACK_CYCLE_TOOL} "
            "--frontier-artifact-root .omx/research --candidate-limit 4"
        )
    initial_report = latest.get("initial_queue_path")
    if initial_report:
        return (
            f".venv/bin/python {FRONTIER_FEEDBACK_CYCLE_TOOL} "
            "--frontier-artifact-root .omx/research "
            "--candidate-limit 4 --execute-followup "
            "--max-candidates 4 --max-steps-per-candidate 8"
        )
    return f".venv/bin/python {FRONTIER_FEEDBACK_CYCLE_TOOL} --help"


def _frontier_feedback_cycle_summary() -> dict[str, object]:
    cycle_rows = [
        _frontier_feedback_cycle_row(path)
        for path in _frontier_feedback_artifact_paths(
            FRONTIER_FEEDBACK_CYCLE_REPORT_NAME
        )
    ]
    refresh_rows = [
        _frontier_feedback_refresh_row(path)
        for path in _frontier_feedback_artifact_paths(
            FRONTIER_FEEDBACK_REFRESH_REPORT_NAME
        )
    ]
    error_rows = [
        row
        for row in [*cycle_rows, *refresh_rows]
        if str(row.get("status") or "").startswith(("ERROR", "BLOCKED"))
    ]
    latest_cycle = cycle_rows[0] if cycle_rows else None
    latest_refresh = refresh_rows[0] if refresh_rows else None
    latest_eureka_refresh = next(
        (
            row
            for row in refresh_rows
            if row.get("eureka_pairset_profile_active")
            or int(row.get("selected_drop_many_candidate_count") or 0) > 0
        ),
        None,
    )
    if error_rows:
        status = "BLOCKED"
        reason = f"{len(error_rows)} frontier feedback artifact(s) failed safe loading"
    elif latest_cycle and latest_cycle.get("status") == "POST_HARVEST_QUEUE_READY":
        status = "POST_HARVEST_QUEUE_READY"
        reason = "latest cycle harvested local observations and emitted the next queue"
    elif latest_cycle and latest_cycle.get("status") == "READY_LOCAL_EXECUTION":
        status = "READY_LOCAL_EXECUTION"
        reason = "latest cycle emitted a validated DQS1 batch queue; run bounded local autopilot"
    elif latest_refresh and latest_refresh.get("queue_path"):
        status = "READY_CYCLE"
        reason = "feedback refresh emitted a DQS1 queue; run the cycle wrapper"
    else:
        status = "PENDING"
        reason = "no frontier feedback cycle or queue refresh artifact found"
    return {
        "schema": "pact.frontier_feedback_cycle_summary.v1",
        "scan_roots": [_repo_rel(root) for root in FRONTIER_FEEDBACK_SCAN_ROOTS],
        "cycle_tool": FRONTIER_FEEDBACK_CYCLE_TOOL,
        "cycle_tool_exists": (REPO_ROOT / FRONTIER_FEEDBACK_CYCLE_TOOL).is_file(),
        "status": status,
        "reason": reason,
        "cycle_report_count": len(cycle_rows),
        "refresh_report_count": len(refresh_rows),
        "ready_local_execution_count": sum(
            1 for row in cycle_rows if row.get("status") == "READY_LOCAL_EXECUTION"
        ),
        "post_harvest_queue_count": sum(
            1 for row in cycle_rows if row.get("status") == "POST_HARVEST_QUEUE_READY"
        ),
        "latest_cycle": latest_cycle or {},
        "latest_refresh": latest_refresh or {},
        "latest_eureka_refresh": latest_eureka_refresh or {},
        "error_count": len(error_rows),
        "next_command": _frontier_feedback_cycle_execute_command(latest_cycle),
        **_false_authority_fields(),
    }


def _format_frontier_feedback_cycle_summary() -> str:
    payload = _frontier_feedback_cycle_summary()
    lines = [
        f"status: {payload['status']} — {payload['reason']}",
        f"cycle tool: {payload['cycle_tool']} present={payload['cycle_tool_exists']}",
        (
            "artifacts: "
            f"cycles={payload['cycle_report_count']} "
            f"refreshes={payload['refresh_report_count']} "
            f"ready_local={payload['ready_local_execution_count']} "
            f"post_harvest={payload['post_harvest_queue_count']}"
        ),
    ]
    latest_cycle = payload.get("latest_cycle")
    if isinstance(latest_cycle, dict) and latest_cycle:
        lines.extend(
            [
                "latest cycle:",
                f"  path: {latest_cycle.get('path')}",
                f"  queue: {latest_cycle.get('initial_queue_path')}",
                f"  initial_selected: {latest_cycle.get('initial_selected_candidate_count')}",
                f"  harvest_paths: {latest_cycle.get('harvest_path_count')}",
                f"  post_queue: {latest_cycle.get('post_harvest_queue_path') or '-'}",
            ]
        )
    latest_refresh = payload.get("latest_refresh")
    if isinstance(latest_refresh, dict) and latest_refresh:
        lines.extend(
            [
                "latest refresh:",
                f"  path: {latest_refresh.get('path')}",
                f"  queue: {latest_refresh.get('queue_path') or '-'}",
                f"  selected: {latest_refresh.get('selected_candidate_count')}",
                f"  selected_drop_many: {latest_refresh.get('selected_drop_many_candidate_count', 0)}",
                f"  selected_geometry: {latest_refresh.get('selected_geometry_candidate_count', 0)}",
                f"  dqs1_observations: {latest_refresh.get('dqs1_observation_count')}",
                f"  eureka_signals: {latest_refresh.get('eureka_signal_count', 0)}",
                f"  eureka_hints: {latest_refresh.get('eureka_planner_hint_count', 0)}",
                f"  eureka_drop_many_counts: {latest_refresh.get('eureka_drop_many_counts', [])}",
                (
                    "  raw_retention: "
                    f"{'execute' if latest_refresh.get('raw_retention_execute') else 'plan'}"
                    f" {latest_refresh.get('raw_retention_action') or ''}".rstrip()
                ),
            ]
        )
    latest_eureka_refresh = payload.get("latest_eureka_refresh")
    if (
        isinstance(latest_eureka_refresh, dict)
        and latest_eureka_refresh
        and latest_eureka_refresh.get("path")
        != (latest_refresh.get("path") if isinstance(latest_refresh, dict) else None)
    ):
        lines.extend(
            [
                "latest eureka refresh:",
                f"  path: {latest_eureka_refresh.get('path')}",
                f"  queue: {latest_eureka_refresh.get('queue_path') or '-'}",
                f"  selected_drop_many: {latest_eureka_refresh.get('selected_drop_many_candidate_count', 0)}",
                f"  selected_geometry: {latest_eureka_refresh.get('selected_geometry_candidate_count', 0)}",
                f"  eureka_drop_many_counts: {latest_eureka_refresh.get('eureka_drop_many_counts', [])}",
            ]
        )
    lines.append("next command:")
    lines.append(f"  {payload['next_command']}")
    lines.append(
        "authority: planning/local only; exact CPU/CUDA auth still required before score, rank, promotion, or dispatch."
    )
    return "\n".join(lines)


def _repo_path_from_ref(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _pr95_mlx_control_profile_paths(
    scan_roots: tuple[Path, ...] | None = None,
) -> list[Path]:
    if scan_roots is None:
        scan_roots = PR95_MLX_CONTROL_PROFILE_SCAN_ROOTS
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.glob(f"**/{PR95_MLX_MATRIX_MANIFEST_NAME}"):
            resolved = path.resolve(strict=False)
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(
        paths,
        key=lambda item: item.stat().st_mtime_ns if item.exists() else 0,
        reverse=True,
    )


def _pr95_mlx_run_manifest_digest(path_ref: object) -> dict[str, object]:
    path = _repo_path_from_ref(path_ref)
    if path is None or not path.is_file():
        return {
            "run_manifest_path": _repo_rel(path) if path is not None else "",
            "run_manifest_exists": False,
            "runtime_consumption_proven": False,
            "training_loss_surface": "",
            "manifest_train_seconds": 0.0,
            "process_seconds": 0.0,
        }
    payload = _load_json_file(path)
    proof = (
        payload.get("runtime_consumption_proof")
        if isinstance(payload.get("runtime_consumption_proof"), dict)
        else {}
    )
    return {
        "run_manifest_path": _repo_rel(path),
        "run_manifest_exists": "_error" not in payload,
        "runtime_consumption_proven": (
            proof.get("runtime_consumption_proven") is True
        ),
        "training_loss_surface": str(payload.get("training_loss_surface") or ""),
        "manifest_train_seconds": _safe_float(
            payload.get("train_seconds")
            or payload.get("elapsed_train_seconds")
            or payload.get("manifest_train_seconds")
        ),
        "process_seconds": _safe_float(
            payload.get("process_seconds")
            or payload.get("elapsed_seconds")
            or payload.get("wall_seconds")
        ),
    }


def _pr95_mlx_control_profile_row(path: Path) -> dict[str, object]:
    payload = _load_json_file(path)
    base = {
        "kind": "pr95_mlx_control_profile",
        "path": _repo_rel(path),
        "sha256": _sha256_file(path) if path.is_file() else "",
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
        **_false_authority_fields(),
    }
    if "_error" in payload:
        return {
            **base,
            "status": "ERROR",
            "blockers": [str(payload["_error"])],
        }
    if payload.get("schema") != "pr95_hnerv_mlx_optimizer_matrix_queue.v1":
        return {
            **base,
            "status": "IGNORED",
            "schema": str(payload.get("schema") or ""),
            "blockers": ["not a PR95 MLX matrix manifest"],
        }
    bad_authority = _authority_truthy(payload)
    if bad_authority:
        return {
            **base,
            "status": "BLOCKED_AUTHORITY_LEAK",
            "schema": str(payload.get("schema") or ""),
            "blockers": [f"truthy_authority:{key}" for key in bad_authority],
        }
    plans = payload.get("plans") if isinstance(payload.get("plans"), list) else []
    run_digests = [
        _pr95_mlx_run_manifest_digest(row.get("run_manifest"))
        for row in plans
        if isinstance(row, dict)
    ]
    queue_path = _repo_path_from_ref(payload.get("queue_output"))
    state_path = default_state_path(REPO_ROOT, str(payload.get("queue_id") or ""))
    queue_observation: dict[str, object] = {}
    queue_exists = queue_path is not None and queue_path.is_file()
    if queue_exists and state_path.is_file():
        try:
            queue = load_queue_definition(queue_path)
            queue_observation = {
                **observe_experiment_queue(
                    queue,
                    state_path=state_path,
                    repo_root=REPO_ROOT,
                    tail_lines=0,
                    include_orphans=True,
                ),
                "operator_briefing_live_observation": True,
                **_false_authority_fields(),
            }
        except Exception as exc:  # pragma: no cover - depends on state corruption
            queue_observation = {
                "operator_briefing_live_observation": True,
                "healthy": False,
                "blockers": [
                    "operator_briefing_pr95_mlx_queue_observation_failed"
                ],
                "error": f"{type(exc).__name__}: {exc}",
                **_false_authority_fields(),
            }
    status_counts = (
        queue_observation.get("status_counts")
        if isinstance(queue_observation.get("status_counts"), dict)
        else {}
    )
    blocker_count = len(
        queue_observation.get("blockers")
        if isinstance(queue_observation.get("blockers"), list)
        else []
    )
    row_blockers = (
        queue_observation.get("blockers")
        if isinstance(queue_observation.get("blockers"), list)
        else []
    )
    if blocker_count:
        status = "BLOCKED_QUEUE_OBSERVATION"
    elif queue_observation.get("healthy") is True and _safe_int(
        status_counts.get("succeeded")
    ) >= len(plans) > 0:
        status = "EXECUTED_HEALTHY"
    elif queue_exists:
        status = "QUEUE_READY"
    else:
        status = "MANIFEST_ONLY"
    proven_count = sum(
        1 for row in run_digests if row.get("runtime_consumption_proven") is True
    )
    return {
        **base,
        "schema": str(payload.get("schema") or ""),
        "status": status,
        "queue_id": str(payload.get("queue_id") or ""),
        "queue_path": _repo_rel(queue_path) if queue_path is not None else "",
        "queue_exists": queue_exists,
        "queue_state_path": _repo_rel(state_path),
        "queue_state_exists": state_path.is_file(),
        "queue_observation": queue_observation,
        "queue_status_counts": status_counts,
        "queue_observation_blocker_count": blocker_count,
        "control_profile": str(payload.get("control_profile") or ""),
        "stage_indices": payload.get("stage_indices")
        if isinstance(payload.get("stage_indices"), list)
        else [],
        "plan_count": len(plans),
        "source_video_loss_surface": str(payload.get("source_video_loss_surface") or ""),
        "train_on_source_video_pairs": (
            payload.get("train_on_source_video_pairs") is True
        ),
        "prove_pr95_runtime_consumption": (
            payload.get("prove_pr95_runtime_consumption") is True
        ),
        "runtime_consumption_proven_count": proven_count,
        "runtime_consumption_missing_count": max(0, len(plans) - proven_count),
        "run_manifest_count": sum(
            1 for row in run_digests if row.get("run_manifest_exists") is True
        ),
        "loss_surfaces": _unique_strings(
            [
                str(row.get("training_loss_surface") or "")
                for row in run_digests
                if row.get("training_loss_surface")
            ]
        ),
        "manifest_train_seconds_sum": sum(
            _safe_float(row.get("manifest_train_seconds")) for row in run_digests
        ),
        "process_seconds_sum": sum(
            _safe_float(row.get("process_seconds")) for row in run_digests
        ),
        "blockers": row_blockers,
    }


def _pr95_mlx_control_profile_summary() -> dict[str, object]:
    rows = [
        row
        for row in (
            _pr95_mlx_control_profile_row(path)
            for path in _pr95_mlx_control_profile_paths()
        )
        if row.get("status") != "IGNORED"
    ]
    latest = rows[0] if rows else {}
    executed = [row for row in rows if row.get("status") == "EXECUTED_HEALTHY"]
    blocked = [
        row
        for row in rows
        if str(row.get("status") or "").startswith(("BLOCKED", "ERROR"))
    ]
    latest_status = str(latest.get("status") or "") if latest else ""
    if latest_status.startswith(("BLOCKED", "ERROR")):
        status = "BLOCKED"
        reason = "latest PR95 MLX profile failed safe loading or observation"
    elif executed and latest_status == "EXECUTED_HEALTHY":
        status = "EXECUTED_HEALTHY"
        reason = "latest PR95 MLX profile has healthy local queue observations"
    elif latest:
        status = str(latest.get("status") or "PENDING")
        reason = "latest PR95 MLX profile is visible but not fully observed healthy"
    else:
        status = "PENDING"
        reason = "no PR95 MLX matrix profile manifest found"
    return {
        "schema": "pact.pr95_mlx_control_profile_summary.v1",
        "scan_roots": [_repo_rel(root) for root in PR95_MLX_CONTROL_PROFILE_SCAN_ROOTS],
        "status": status,
        "reason": reason,
        "profile_count": len(rows),
        "executed_healthy_count": len(executed),
        "blocked_count": len(blocked),
        "latest_profile": latest,
        "latest_profiles": rows[:5],
        **_false_authority_fields(),
    }


def _format_pr95_mlx_control_profiles() -> str:
    payload = _pr95_mlx_control_profile_summary()
    lines = [
        f"status: {payload['status']} — {payload['reason']}",
        f"profiles: {payload['profile_count']} "
        f"executed_healthy={payload['executed_healthy_count']} "
        f"blocked={payload['blocked_count']}",
    ]
    latest = payload.get("latest_profile")
    if isinstance(latest, dict) and latest:
        lines.extend(
            [
                "latest profile:",
                f"  path: {latest.get('path')}",
                f"  queue: {latest.get('queue_path') or '-'}",
                f"  queue_status: {latest.get('status')}",
                f"  control_profile: {latest.get('control_profile') or '-'}",
                f"  stages: {latest.get('stage_indices')}",
                f"  source_loss: {latest.get('source_video_loss_surface') or '-'}",
                f"  plans: {latest.get('plan_count')} "
                f"runtime_proven={latest.get('runtime_consumption_proven_count')}",
                f"  queue_counts: {latest.get('queue_status_counts') or {}}",
            ]
        )
    lines.append(
        "authority: local MLX research-signal only; exact CPU/CUDA auth still required before score, rank, promotion, or dispatch."
    )
    return "\n".join(lines)


def _live_queue_observation_error(
    blocker: str,
    *,
    queue_path: Path | None,
    state_path: Path | None,
    error: str,
) -> dict[str, object]:
    return {
        "schema": "operator_briefing_live_queue_observation.v1",
        "observe_read_only": True,
        "operator_briefing_live_observation": True,
        "healthy": False,
        "blockers": [blocker],
        "blocker_count": 1,
        "status_counts": {},
        "ready_steps": [],
        "failed_steps": [],
        "definition_drift": {},
        "queue_path": _repo_rel(queue_path) if queue_path is not None else "",
        "state": _repo_rel(state_path) if state_path is not None else "",
        "error": error,
        **_false_authority_fields(),
    }


def _byte_shaving_live_queue_observation(
    queue_path_ref: object,
    state_path_ref: object,
) -> dict[str, object]:
    queue_path = _repo_path_from_ref(queue_path_ref)
    state_path = _repo_path_from_ref(state_path_ref)
    if queue_path is None:
        return {}
    if not queue_path.is_file():
        return _live_queue_observation_error(
            "operator_briefing_live_queue_definition_missing",
            queue_path=queue_path,
            state_path=state_path,
            error=f"queue definition missing: {_repo_rel(queue_path)}",
        )
    try:
        queue = load_queue_definition(queue_path)
        if state_path is None:
            state_path = default_state_path(REPO_ROOT, str(queue["queue_id"]))
        observation = observe_experiment_queue(
            queue,
            state_path=state_path,
            repo_root=REPO_ROOT,
            tail_lines=0,
            include_orphans=True,
        )
    except Exception as exc:  # pragma: no cover - precise type depends on state corruption
        return _live_queue_observation_error(
            "operator_briefing_live_queue_observation_failed",
            queue_path=queue_path,
            state_path=state_path,
            error=f"{type(exc).__name__}: {exc}",
        )
    return {
        **observation,
        "operator_briefing_live_observation": True,
        "queue_path": _repo_rel(queue_path),
        **_false_authority_fields(),
    }


def _first_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _byte_shaving_campaign_plan_digest(path_ref: object) -> dict[str, object]:
    path = _repo_path_from_ref(path_ref)
    if path is None:
        return {
            "plan_path": "",
            "plan_exists": False,
            "plan_error": "campaign run did not reference a plan path",
            "combination_count": 0,
            "operation_families": [],
            "top_combo": {},
            "materialization_bridge": {},
            "dispatch_blockers": [],
        }
    rel = _repo_rel(path)
    if not path.is_file():
        return {
            "plan_path": rel,
            "plan_exists": False,
            "plan_error": "referenced campaign plan is missing",
            "combination_count": 0,
            "operation_families": [],
            "top_combo": {},
            "materialization_bridge": {},
            "dispatch_blockers": [],
        }
    plan = _load_json_file(path)
    if "_error" in plan:
        return {
            "plan_path": rel,
            "plan_exists": False,
            "plan_error": str(plan["_error"]),
            "combination_count": 0,
            "operation_families": [],
            "top_combo": {},
            "materialization_bridge": {},
            "dispatch_blockers": [],
        }
    combos = _first_list(plan, "combination_ladder")
    top_combo = combos[0] if combos and isinstance(combos[0], dict) else {}
    families = _unique_strings(
        [
            family
            for combo in combos
            if isinstance(combo, dict)
            for family in (
                combo.get("operation_families")
                if isinstance(combo.get("operation_families"), list)
                else []
            )
        ]
    )
    blockers = _unique_strings(
        [
            *_first_list(plan, "dispatch_blockers"),
            *(
                top_combo.get("dispatch_blockers")
                if isinstance(top_combo.get("dispatch_blockers"), list)
                else []
            ),
        ]
    )
    bridge = plan.get("materialization_bridge")
    bridge_payload = bridge if isinstance(bridge, dict) else {}
    return {
        "plan_path": rel,
        "plan_exists": True,
        "plan_error": "",
        "campaign_id": plan.get("campaign_id"),
        "candidate_id": plan.get("candidate_id"),
        "combination_count": len(combos),
        "operation_families": families,
        "top_combo": {
            "combo_id": top_combo.get("combo_id"),
            "expected_score_gain": _safe_float(top_combo.get("expected_score_gain")),
            "unit_count": _safe_int(top_combo.get("unit_count")),
            "operation_families": top_combo.get("operation_families")
            if isinstance(top_combo.get("operation_families"), list)
            else [],
            "dispatch_blockers": top_combo.get("dispatch_blockers")
            if isinstance(top_combo.get("dispatch_blockers"), list)
            else [],
        },
        "materialization_bridge": {
            "next_gate": bridge_payload.get("next_gate"),
            "high_level_operation_compiler_required_count": _safe_int(
                bridge_payload.get("high_level_operation_compiler_required_count")
            ),
            "packet_ir_operation_set_count": _safe_int(
                bridge_payload.get("packet_ir_operation_set_count")
            ),
            "packet_ir_byte_closed_operation_count": _safe_int(
                bridge_payload.get("packet_ir_byte_closed_operation_count")
            ),
            "queue_consumable_packet_ir_operation_set_count": _safe_int(
                bridge_payload.get("queue_consumable_packet_ir_operation_set_count")
            ),
            "dispatch_blockers": _unique_strings(
                _first_list(bridge_payload, "dispatch_blockers")
            ),
        },
        "dispatch_blockers": blockers,
        **_false_authority_fields(),
    }


def _source_failure_diagnostics_from_payload(
    payload: dict[str, object],
    run_dir_path: Path,
) -> dict[str, object]:
    embedded = payload.get("queue_source_failure_diagnostics")
    if isinstance(embedded, dict):
        return embedded
    path = _repo_path_from_ref(payload.get("queue_source_failure_diagnostics_path"))
    if path is None:
        path = run_dir_path / "queue_source_failure_diagnostics.json"
    if not path.is_file():
        return {}
    loaded = _load_json_file(path)
    return loaded if "_error" not in loaded else {}


def _byte_shaving_acquisition_row(path: Path) -> dict[str, object]:
    payload = _load_json_file(path)
    base: dict[str, object] = {
        "path": _repo_rel(path),
        "sha256": _sha256_file(path) if path.is_file() else "",
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
        **_false_authority_fields(),
    }
    if "_error" in payload:
        return {
            **base,
            "kind": "unreadable",
            "schema": None,
            "status": "BLOCKED",
            "blockers": [str(payload["_error"])],
        }

    build = payload.get("build") if isinstance(payload.get("build"), dict) else {}
    worker = payload.get("worker") if isinstance(payload.get("worker"), dict) else {}
    queue_path = str(payload.get("queue_path") or "")
    state_path = str(payload.get("state_path") or "")
    embedded_observation = (
        payload.get("observation")
        if isinstance(payload.get("observation"), dict)
        else {}
    )
    live_observation = (
        _byte_shaving_live_queue_observation(queue_path, state_path)
        if queue_path
        else {}
    )
    observation = live_observation or embedded_observation
    drift = (
        observation.get("definition_drift")
        if isinstance(observation.get("definition_drift"), dict)
        else {}
    )
    status_counts = (
        observation.get("status_counts")
        if isinstance(observation.get("status_counts"), dict)
        else {}
    )
    commands = payload.get("commands") if isinstance(payload.get("commands"), list) else []
    failed_commands = [
        item
        for item in commands
        if isinstance(item, dict) and _safe_int(item.get("returncode")) != 0
    ]
    failed_steps = (
        observation.get("failed_steps")
        if isinstance(observation.get("failed_steps"), list)
        else []
    )
    observation_blockers = _unique_strings(
        observation.get("blockers")
        if isinstance(observation.get("blockers"), list)
        else []
    )
    authority_leaks = [
        field
        for field in _false_authority_fields()
        if payload.get(field) is True
        or (isinstance(build, dict) and build.get(field) is True)
    ]
    blockers = [
        f"campaign_run_authority_field_true:{field}" for field in authority_leaks
    ]
    if failed_commands:
        blockers.append(f"{len(failed_commands)} campaign command(s) failed")
    if _safe_int(worker.get("failure_count")):
        blockers.append(f"{worker.get('failure_count')} worker step(s) failed")
    blockers.extend(observation_blockers)
    if live_observation and str(live_observation.get("mode") or "") != "running":
        blockers.append(
            "experiment_queue_observation_mode_not_running:"
            f"{live_observation.get('mode') or '<unknown>'}"
        )
    if failed_steps:
        blockers.append(f"{len(failed_steps)} observed queue step(s) failed")
    drift_count = (
        _safe_int(drift.get("changed_step_count"))
        + _safe_int(drift.get("missing_step_count"))
        + _safe_int(drift.get("missing_hash_step_count"))
    )
    if drift_count:
        blockers.append(f"{drift_count} queue definition drift issue(s)")

    experiment_count = _safe_int(payload.get("experiment_count"))
    if blockers:
        status = "BLOCKED"
    elif queue_path and experiment_count:
        status = "READY_LOCAL_QUEUE"
    elif path.is_file():
        status = "PENDING"
    else:
        status = "PENDING"

    plan_digest = _byte_shaving_campaign_plan_digest(
        payload.get("plan") or payload.get("generated_campaign_plan_path")
    )
    if plan_digest.get("plan_error"):
        blockers.append(str(plan_digest["plan_error"]))
        if status == "READY_LOCAL_QUEUE":
            status = "BLOCKED"

    ready_steps = (
        observation.get("ready_steps")
        if isinstance(observation.get("ready_steps"), list)
        else []
    )
    local_mlx_ready_steps = [
        step
        for step in ready_steps
        if isinstance(step, dict) and step.get("resource_kind") == "local_mlx"
    ]
    local_cpu_ready_steps = [
        step
        for step in ready_steps
        if isinstance(step, dict) and step.get("resource_kind") == "local_cpu"
    ]
    executable_work_count = _safe_int(
        build.get("materializer_work_queue_executable_row_count")
    )
    blocked_work_count = _safe_int(
        build.get("materializer_work_queue_blocked_row_count")
    ) + _safe_int(build.get("blocked_row_count"))
    backlog_row_count = _safe_int(build.get("materializer_backlog_row_count"))
    total_work_count = executable_work_count + blocked_work_count
    executable_conversion_rate = (
        executable_work_count / total_work_count if total_work_count else 0.0
    )
    bridge = (
        plan_digest.get("materialization_bridge")
        if isinstance(plan_digest.get("materialization_bridge"), dict)
        else {}
    )
    feedback_policy = (
        payload.get("queue_feedback_replan_policy")
        if isinstance(payload.get("queue_feedback_replan_policy"), dict)
        else {}
    )
    feedback_action_summary = (
        feedback_policy.get("feedback_action_functional_summary")
        if isinstance(feedback_policy.get("feedback_action_functional_summary"), dict)
        else {}
    )
    recovery_plan = (
        payload.get("queue_observation_recovery_plan")
        if isinstance(payload.get("queue_observation_recovery_plan"), dict)
        else feedback_policy.get("queue_observation_recovery_plan")
    )
    recovery_plan_source = "payload"
    if not isinstance(recovery_plan, dict):
        recovery_plan = {}
        recovery_plan_source = "missing"
    if (
        not recovery_plan
        and live_observation
        and queue_path
        and state_path
        and live_observation.get("healthy") is not True
    ):
        recovery_plan = build_queue_observation_recovery_plan(
            live_observation,
            queue_path=queue_path,
            state_path=state_path,
            reason="operator briefing live queue health recovery",
        )
        recovery_plan_source = "live_queue_observation"
    run_dir_path = _repo_path_from_ref(payload.get("run_dir")) or path.parent
    adjacent_recovery_queue_path = run_dir_path / "queue_observation_recovery_queue.json"
    adjacent_recovery_state_path = run_dir_path / "queue_observation_recovery_queue.sqlite"
    adjacent_recovery_policy_path = run_dir_path / "queue_feedback_replan_policy.json"
    adjacent_recovery_plan_path = run_dir_path / "queue_observation_recovery_plan.json"
    adjacent_observation_path = run_dir_path / "queue_observation.json"
    adjacent_recovery_queue_exists = adjacent_recovery_queue_path.is_file()
    source_failure_diagnostics = _source_failure_diagnostics_from_payload(
        payload,
        run_dir_path,
    )
    source_failure_blockers = _unique_strings(
        source_failure_diagnostics.get("blockers")
        if isinstance(source_failure_diagnostics.get("blockers"), list)
        else []
    )
    source_failure_execution_blockers = _unique_strings(
        source_failure_diagnostics.get("recovery_queue_execution_blockers")
        if isinstance(
            source_failure_diagnostics.get("recovery_queue_execution_blockers"),
            list,
        )
        else []
    )
    source_failure_recovery_recommended = (
        source_failure_diagnostics.get("recovery_queue_execution_recommended")
        is not False
    )
    recovery_execution = (
        payload.get("queue_observation_recovery_execution")
        if isinstance(payload.get("queue_observation_recovery_execution"), dict)
        else {}
    )
    recovery_source_after = (
        recovery_execution.get("source_observation_after")
        if isinstance(recovery_execution.get("source_observation_after"), dict)
        else {}
    )
    post_recovery = (
        payload.get("post_recovery_feedback_replan")
        if isinstance(payload.get("post_recovery_feedback_replan"), dict)
        else {}
    )
    recovery_policy_blockers = _unique_strings(
        payload.get("queue_observation_recovery_policy_blockers")
        if isinstance(payload.get("queue_observation_recovery_policy_blockers"), list)
        else []
    )
    recovery_execution_blockers = _unique_strings(
        recovery_execution.get("blockers")
        if isinstance(recovery_execution.get("blockers"), list)
        else []
    )
    recovery_source_after_blockers = _unique_strings(
        recovery_source_after.get("blockers")
        if isinstance(recovery_source_after.get("blockers"), list)
        else []
    )
    row = {
        **base,
        "kind": "byte_shaving_materializer_campaign_run",
        "schema": payload.get("schema"),
        "status": status,
        "queue_id": str(payload.get("queue_id") or ""),
        "queue_path": queue_path,
        "state_path": state_path,
        "run_dir": str(payload.get("run_dir") or ""),
        "execute": payload.get("execute") is True,
        "live_queue_recovery_materialize_command": (
            ".venv/bin/python tools/materialize_byte_shaving_queue_recovery.py "
            f"--run-summary {base['path']} --write"
        )
        if recovery_plan_source == "live_queue_observation"
        and recovery_plan.get("recovery_required") is True
        and not adjacent_recovery_queue_exists
        else "",
        "queue_observation_source": "live"
        if live_observation
        else "embedded"
        if embedded_observation
        else "missing",
        "live_queue_observation_used": bool(live_observation),
        "live_queue_observation_healthy": (
            live_observation.get("healthy") is True if live_observation else None
        ),
        "live_queue_observation_mode": str(live_observation.get("mode") or "")
        if live_observation
        else "",
        "live_queue_observation_queue_sha256": str(
            live_observation.get("queue_sha256") or ""
        )
        if live_observation
        else "",
        "live_queue_observation_state_watermark": (
            live_observation.get("state_watermark")
            if isinstance(live_observation.get("state_watermark"), dict)
            else {}
        )
        if live_observation
        else {},
        "live_queue_observation_blockers": observation_blockers
        if live_observation
        else [],
        "live_queue_observation_blocker_count": len(observation_blockers)
        if live_observation
        else 0,
        "live_queue_observation_status_counts": status_counts
        if live_observation
        else {},
        "live_queue_observation_failed_step_count": len(failed_steps)
        if live_observation
        else 0,
        "high_level_action_source_count": _safe_int(
            payload.get("high_level_action_source_count")
        ),
        "experiment_count": experiment_count,
        "executable_work_count": executable_work_count,
        "blocked_work_count": blocked_work_count,
        "materializer_backlog_row_count": backlog_row_count,
        "executable_conversion_rate": executable_conversion_rate,
        "compiler_required_count": _safe_int(
            bridge.get("high_level_operation_compiler_required_count")
        ),
        "packet_ir_operation_set_count": _safe_int(
            bridge.get("packet_ir_operation_set_count")
        ),
        "queue_consumable_packet_ir_operation_set_count": _safe_int(
            bridge.get("queue_consumable_packet_ir_operation_set_count")
        ),
        "packet_ir_byte_closed_operation_count": _safe_int(
            bridge.get("packet_ir_byte_closed_operation_count")
        ),
        "exact_readiness_handoff_count": _safe_int(
            payload.get("exact_readiness_handoff_count")
        ),
        "queue_feedback_replan_ready": payload.get("queue_feedback_replan_ready") is True,
        "queue_feedback_replan_followup_queue_emitted": (
            payload.get("queue_feedback_replan_followup_queue_emitted") is True
        ),
        "queue_feedback_replan_followup_queue_path": str(
            payload.get("queue_feedback_replan_followup_queue_path") or ""
        ),
        "queue_feedback_replan_followup_blocker_count": len(
            payload.get("queue_feedback_replan_followup_queue_blockers")
            if isinstance(payload.get("queue_feedback_replan_followup_queue_blockers"), list)
            else []
        ),
        "queue_feedback_replan_followup_policy": str(
            payload.get("queue_feedback_replan_followup_policy") or ""
        ),
        "queue_feedback_replan_followup_policy_enabled": (
            payload.get("queue_feedback_replan_followup_policy_enabled") is True
        ),
        "queue_feedback_replan_followup_policy_blocker_count": len(
            payload.get("queue_feedback_replan_followup_policy_blockers")
            if isinstance(payload.get("queue_feedback_replan_followup_policy_blockers"), list)
            else []
        ),
        "queue_feedback_replan_followup_execution_requested": (
            payload.get("queue_feedback_replan_followup_execution_requested") is True
        ),
        "queue_feedback_replan_followup_executed": (
            payload.get("queue_feedback_replan_followup_executed") is True
        ),
        "queue_feedback_replan_followup_execution_success": (
            payload.get("queue_feedback_replan_followup_execution_success") is True
        ),
        "queue_feedback_replan_policy_path": str(
            payload.get("queue_feedback_replan_policy_path")
            or (
                _repo_rel(adjacent_recovery_policy_path)
                if adjacent_recovery_policy_path.is_file()
                else ""
            )
        ),
        "queue_feedback_replan_policy_decision": str(
            payload.get("queue_feedback_replan_policy_decision")
            or feedback_policy.get("decision")
            or ""
        ),
        "queue_feedback_replan_policy_should_continue": (
            payload.get("queue_feedback_replan_policy_should_continue") is True
            or feedback_policy.get("should_continue_feedback_loop") is True
        ),
        "queue_feedback_replan_candidate_widening_ready": (
            payload.get("queue_feedback_replan_candidate_widening_ready") is True
            or feedback_policy.get("ready_for_candidate_generation_widening") is True
        ),
        "queue_feedback_replan_dry_no_selected_cells": (
            feedback_action_summary.get("dry_no_selected_cells") is True
        ),
        "queue_feedback_replan_feedback_cell_count": _safe_int(
            feedback_action_summary.get("cell_count")
        ),
        "queue_feedback_replan_feedback_selected_count": _safe_int(
            feedback_action_summary.get("selected_count")
        ),
        "queue_feedback_replan_archive_delta_blocked_cell_count": _safe_int(
            feedback_action_summary.get(
                "materializer_archive_delta_blocked_cell_count"
            )
        ),
        "queue_feedback_candidate_widening_queue_path": str(
            payload.get("queue_feedback_candidate_widening_queue_path") or ""
        ),
        "queue_feedback_candidate_widening_queue_emitted": (
            payload.get("queue_feedback_candidate_widening_queue_emitted") is True
        ),
        "queue_feedback_candidate_widening_queue_blocker_count": len(
            payload.get("queue_feedback_candidate_widening_queue_blockers")
            if isinstance(
                payload.get("queue_feedback_candidate_widening_queue_blockers"),
                list,
            )
            else []
        ),
        "queue_feedback_candidate_actuation_planning_queue_path": str(
            payload.get("queue_feedback_candidate_actuation_planning_queue_path") or ""
        ),
        "queue_feedback_candidate_actuation_planning_queue_emitted": (
            payload.get("queue_feedback_candidate_actuation_planning_queue_emitted")
            is True
        ),
        "queue_feedback_candidate_actuation_planning_queue_blocker_count": len(
            payload.get("queue_feedback_candidate_actuation_planning_queue_blockers")
            if isinstance(
                payload.get(
                    "queue_feedback_candidate_actuation_planning_queue_blockers"
                ),
                list,
            )
            else []
        ),
        "queue_observation_path": str(
            payload.get("queue_observation_path")
            or (
                _repo_rel(adjacent_observation_path)
                if adjacent_observation_path.is_file()
                else ""
            )
        ),
        "queue_observation_recovery_plan_source": recovery_plan_source,
        "queue_observation_recovery_plan_path": str(
            payload.get("queue_observation_recovery_plan_path")
            or (_repo_rel(adjacent_recovery_plan_path) if adjacent_recovery_plan_path.is_file() else "")
        ),
        "queue_observation_recovery_queue_path": str(
            payload.get("queue_observation_recovery_queue_path")
            or (_repo_rel(adjacent_recovery_queue_path) if adjacent_recovery_queue_exists else "")
        ),
        "queue_observation_recovery_queue_state_path": str(
            payload.get("queue_observation_recovery_queue_state_path")
            or (
                _repo_rel(adjacent_recovery_state_path)
                if adjacent_recovery_queue_exists
                else ""
            )
        ),
        "queue_observation_recovery_queue_emitted": (
            payload.get("queue_observation_recovery_queue_emitted") is True
            or adjacent_recovery_queue_exists
        ),
        "queue_observation_recovery_queue_blocker_count": len(
            payload.get("queue_observation_recovery_queue_blockers")
            if isinstance(
                payload.get("queue_observation_recovery_queue_blockers"),
                list,
            )
            else []
        ),
        "queue_observation_recovery_policy_enabled": (
            payload.get("queue_observation_recovery_policy_enabled") is True
        ),
        "queue_observation_recovery_execution_requested": (
            payload.get("queue_observation_recovery_execution_requested") is True
        ),
        "queue_observation_recovery_executed": (
            payload.get("queue_observation_recovery_executed") is True
        ),
        "queue_observation_recovery_execution_success": (
            payload.get("queue_observation_recovery_execution_success") is True
        ),
        "queue_observation_recovery_grouped_blocker_count": _safe_int(
            recovery_plan.get("grouped_blocker_count")
        ),
        "queue_observation_recovery_repeated_group_count": _safe_int(
            recovery_plan.get("repeated_group_count")
        ),
        "queue_observation_recovery_top_groups": _queue_recovery_group_summaries(
            recovery_plan.get("grouped_blockers")
        ),
        "queue_observation_recovery_policy_blockers": recovery_policy_blockers,
        "queue_observation_recovery_execution_blockers": recovery_execution_blockers,
        "queue_observation_recovery_source_observation_healthy": (
            recovery_source_after.get("healthy") is True
            if recovery_source_after
            else None
        ),
        "queue_observation_recovery_source_observation_blockers": (
            recovery_source_after_blockers
        ),
        "queue_source_failure_diagnostics_path": str(
            payload.get("queue_source_failure_diagnostics_path")
            or source_failure_diagnostics.get("diagnostics_path")
            or ""
        ),
        "source_failure_diagnostic_count": _safe_int(
            payload.get("source_failure_diagnostic_count")
            if payload.get("source_failure_diagnostic_count") is not None
            else source_failure_diagnostics.get("diagnostic_count")
        ),
        "source_failure_non_rewindable_count": _safe_int(
            payload.get("source_failure_non_rewindable_count")
            if payload.get("source_failure_non_rewindable_count") is not None
            else source_failure_diagnostics.get(
                "non_rewindable_source_failure_count"
            )
        ),
        "source_failure_recovery_queue_execution_recommended": (
            payload.get("source_failure_recovery_queue_execution_recommended")
            is True
            if payload.get("source_failure_recovery_queue_execution_recommended")
            is not None
            else source_failure_recovery_recommended
        ),
        "source_failure_recovery_queue_execution_blockers": (
            source_failure_execution_blockers
        ),
        "source_failure_blockers": source_failure_blockers,
        "source_failure_requires_context_repair": (
            payload.get("source_failure_requires_context_repair") is True
            or source_failure_diagnostics.get("requires_context_repair") is True
        ),
        "source_failure_recommended_next_action": str(
            payload.get("source_failure_recommended_next_action")
            or source_failure_diagnostics.get("recommended_next_action")
            or ""
        ),
        "post_recovery_feedback_replan_triggered": (
            payload.get("post_recovery_feedback_replan_triggered") is True
            or post_recovery.get("triggered") is True
        ),
        "post_recovery_feedback_replan_attempted": (
            payload.get("post_recovery_feedback_replan_attempted") is True
            or post_recovery.get("attempted") is True
        ),
        "post_recovery_feedback_replan_artifacts_emitted": (
            payload.get("post_recovery_feedback_replan_artifacts_emitted") is True
            or post_recovery.get("artifacts_emitted") is True
        ),
        "post_recovery_feedback_replan_success": (
            payload.get("post_recovery_feedback_replan_success") is True
            or post_recovery.get("success") is True
        ),
        "post_recovery_feedback_replan_blocker_count": len(
            payload.get("post_recovery_feedback_replan_blockers")
            if isinstance(payload.get("post_recovery_feedback_replan_blockers"), list)
            else post_recovery.get("blockers")
            if isinstance(post_recovery.get("blockers"), list)
            else []
        ),
        "post_recovery_source_recovery_execution_success": (
            payload.get("post_recovery_source_recovery_execution_success") is True
            or post_recovery.get("source_recovery_execution_success") is True
        ),
        "post_recovery_queue_observation_path": str(
            payload.get("post_recovery_queue_observation_path")
            or post_recovery.get("queue_observation_path")
            or ""
        ),
        "post_recovery_queue_observation_recovery_plan_path": str(
            payload.get("post_recovery_queue_observation_recovery_plan_path")
            or post_recovery.get("queue_observation_recovery_plan_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_request_path": str(
            payload.get("post_recovery_queue_feedback_replan_request_path")
            or post_recovery.get("queue_feedback_replan_request_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_policy_path": str(
            payload.get("post_recovery_queue_feedback_replan_policy_path")
            or post_recovery.get("queue_feedback_replan_policy_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_policy_decision": str(
            payload.get("post_recovery_queue_feedback_replan_policy_decision")
            or post_recovery.get("queue_feedback_replan_policy_decision")
            or ""
        ),
        "post_recovery_queue_feedback_replan_policy_should_continue": (
            payload.get("post_recovery_queue_feedback_replan_policy_should_continue")
            is True
            or post_recovery.get("queue_feedback_replan_policy_should_continue")
            is True
        ),
        "post_recovery_queue_feedback_replan_followup_queue_path": str(
            payload.get("post_recovery_queue_feedback_replan_followup_queue_path")
            or post_recovery.get("queue_feedback_replan_followup_queue_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_followup_state_path": str(
            payload.get("post_recovery_queue_feedback_replan_followup_state_path")
            or post_recovery.get("queue_feedback_replan_followup_state_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_followup_queue_emitted": (
            payload.get("post_recovery_queue_feedback_replan_followup_queue_emitted")
            is True
            or post_recovery.get("queue_feedback_replan_followup_queue_emitted")
            is True
        ),
        "post_recovery_queue_feedback_replan_followup_queue_blocker_count": len(
            payload.get(
                "post_recovery_queue_feedback_replan_followup_queue_blockers"
            )
            if isinstance(
                payload.get(
                    "post_recovery_queue_feedback_replan_followup_queue_blockers"
                ),
                list,
            )
            else post_recovery.get("queue_feedback_replan_followup_queue_blockers")
            if isinstance(
                post_recovery.get("queue_feedback_replan_followup_queue_blockers"),
                list,
            )
            else []
        ),
        "post_recovery_queue_feedback_replan_followup_policy_enabled": (
            payload.get("post_recovery_queue_feedback_replan_followup_policy_enabled")
            is True
            or post_recovery.get("queue_feedback_replan_followup_policy_enabled")
            is True
        ),
        "post_recovery_queue_feedback_replan_followup_policy_blocker_count": len(
            payload.get("post_recovery_queue_feedback_replan_followup_policy_blockers")
            if isinstance(
                payload.get(
                    "post_recovery_queue_feedback_replan_followup_policy_blockers"
                ),
                list,
            )
            else post_recovery.get("queue_feedback_replan_followup_policy_blockers")
            if isinstance(
                post_recovery.get("queue_feedback_replan_followup_policy_blockers"),
                list,
            )
            else []
        ),
        "post_recovery_queue_feedback_replan_followup_execution_requested": (
            payload.get(
                "post_recovery_queue_feedback_replan_followup_execution_requested"
            )
            is True
            or post_recovery.get("queue_feedback_replan_followup_execution_requested")
            is True
        ),
        "post_recovery_queue_feedback_replan_followup_executed": (
            payload.get("post_recovery_queue_feedback_replan_followup_executed")
            is True
            or post_recovery.get("queue_feedback_replan_followup_executed") is True
        ),
        "post_recovery_queue_feedback_replan_followup_execution_success": (
            payload.get(
                "post_recovery_queue_feedback_replan_followup_execution_success"
            )
            is True
            or post_recovery.get("queue_feedback_replan_followup_execution_success")
            is True
        ),
        "post_recovery_queue_feedback_replan_followup_action_functional_path": str(
            payload.get(
                "post_recovery_queue_feedback_replan_followup_action_functional_path"
            )
            or post_recovery.get("queue_feedback_replan_followup_action_functional_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_continuation_queue_path": str(
            payload.get(
                "post_recovery_queue_feedback_replan_continuation_queue_path"
            )
            or post_recovery.get("queue_feedback_replan_continuation_queue_path")
            or ""
        ),
        "post_recovery_queue_feedback_replan_continuation_queue_state_path": str(
            payload.get(
                "post_recovery_queue_feedback_replan_continuation_queue_state_path"
            )
            or post_recovery.get(
                "queue_feedback_replan_continuation_queue_state_path"
            )
            or ""
        ),
        "post_recovery_queue_feedback_replan_continuation_queue_emitted": (
            payload.get(
                "post_recovery_queue_feedback_replan_continuation_queue_emitted"
            )
            is True
            or post_recovery.get("queue_feedback_replan_continuation_queue_emitted")
            is True
        ),
        "post_recovery_queue_feedback_replan_continuation_queue_blocker_count": len(
            payload.get(
                "post_recovery_queue_feedback_replan_continuation_queue_blockers"
            )
            if isinstance(
                payload.get(
                    "post_recovery_queue_feedback_replan_continuation_queue_blockers"
                ),
                list,
            )
            else post_recovery.get("queue_feedback_replan_continuation_queue_blockers")
            if isinstance(
                post_recovery.get(
                    "queue_feedback_replan_continuation_queue_blockers"
                ),
                list,
            )
            else []
        ),
        "queue_observation_recovery_required": (
            payload.get("queue_observation_recovery_required") is True
            or feedback_policy.get("queue_observation_recovery_required") is True
            or recovery_plan.get("recovery_required") is True
        ),
        "queue_observation_maintenance_recommended": (
            payload.get("queue_observation_maintenance_recommended") is True
            or feedback_policy.get("queue_observation_maintenance_recommended") is True
            or recovery_plan.get("maintenance_recommended") is True
        ),
        "queue_observation_recovery_action_count": _safe_int(
            recovery_plan.get("action_count")
        ),
        "queue_observation_required_action_count": _safe_int(
            recovery_plan.get("required_action_count")
        ),
        "queue_observation_maintenance_action_count": _safe_int(
            recovery_plan.get("maintenance_action_count")
        ),
        "ready_for_queue_health_recovery": (
            feedback_policy.get("ready_for_queue_health_recovery") is True
            or (
                recovery_plan_source == "live_queue_observation"
                and recovery_plan.get("recovery_required") is True
            )
        ),
        "operator_queue_state_mutation_required": (
            feedback_policy.get("operator_queue_state_mutation_required") is True
            or (
                recovery_plan_source == "live_queue_observation"
                and recovery_plan.get("recovery_required") is True
            )
        ),
        "queue_feedback_replan_policy_blocker_count": len(
            feedback_policy.get("blockers")
            if isinstance(feedback_policy.get("blockers"), list)
            else []
        ),
        "queue_feedback_replan_continuation_queue_path": str(
            payload.get("queue_feedback_replan_continuation_queue_path") or ""
        ),
        "queue_feedback_replan_continuation_queue_emitted": (
            payload.get("queue_feedback_replan_continuation_queue_emitted") is True
        ),
        "queue_feedback_replan_continuation_queue_blocker_count": len(
            payload.get("queue_feedback_replan_continuation_queue_blockers")
            if isinstance(
                payload.get("queue_feedback_replan_continuation_queue_blockers"),
                list,
            )
            else []
        ),
        "local_cpu_concurrency": _safe_int(build.get("local_cpu_concurrency")),
        "worker_max_parallel": _safe_int(worker.get("max_parallel")),
        "worker_execute": worker.get("execute") is True,
        "worker_stop_reason": str(worker.get("stop_reason") or ""),
        "worker_success_count": _safe_int(worker.get("success_count")),
        "worker_failure_count": _safe_int(worker.get("failure_count")),
        "queued_step_count": _safe_int(status_counts.get("queued")),
        "running_step_count": _safe_int(status_counts.get("running")),
        "succeeded_step_count": _safe_int(status_counts.get("succeeded")),
        "failed_step_count": _safe_int(status_counts.get("failed")),
        "ready_step_count": len(ready_steps),
        "local_mlx_ready_step_count": len(local_mlx_ready_steps),
        "local_cpu_ready_step_count": len(local_cpu_ready_steps),
        "plan": plan_digest,
        "blockers": _unique_strings(blockers),
    }
    return apply_false_authority_contract(
        row,
        preserve_dispatch_ready=False,
        reason="operator_briefing_high_level_byte_shaving_acquisition_no_score_authority",
    )


def _experiment_queue_command(
    queue_path: object,
    state_path: object,
    subcommand: str,
) -> str:
    state_arg = f" --state {state_path}" if state_path else ""
    return (
        f".venv/bin/python tools/experiment_queue.py --queue {queue_path}"
        f"{state_arg} {subcommand}"
    )


def _byte_shaving_acquisition_next_command(latest: dict[str, object] | None) -> str:
    if (
        latest
        and latest.get("post_recovery_queue_feedback_replan_continuation_queue_emitted")
        is True
        and latest.get("post_recovery_queue_feedback_replan_continuation_queue_path")
    ):
        return _experiment_queue_command(
            latest["post_recovery_queue_feedback_replan_continuation_queue_path"],
            latest.get("post_recovery_queue_feedback_replan_continuation_queue_state_path"),
            "init",
        )
    if (
        latest
        and latest.get("post_recovery_queue_feedback_replan_followup_queue_emitted")
        is True
        and latest.get("post_recovery_feedback_replan_success") is not True
        and latest.get("post_recovery_queue_feedback_replan_followup_queue_path")
    ):
        return _experiment_queue_command(
            latest["post_recovery_queue_feedback_replan_followup_queue_path"],
            latest.get("post_recovery_queue_feedback_replan_followup_state_path"),
            "init",
        )
    if (
        latest
        and latest.get("source_failure_recovery_queue_execution_recommended") is False
        and latest.get("queue_path")
    ):
        return _experiment_queue_command(
            latest["queue_path"],
            latest.get("state_path"),
            "observe --tail-lines 20",
        )
    if (
        latest
        and latest.get("queue_observation_recovery_queue_emitted") is True
        and latest.get("queue_observation_recovery_execution_success") is not True
        and latest.get("queue_observation_recovery_queue_path")
    ):
        return _experiment_queue_command(
            latest["queue_observation_recovery_queue_path"],
            latest.get("queue_observation_recovery_queue_state_path"),
            "init",
        )
    if latest and latest.get("live_queue_recovery_materialize_command"):
        return str(latest["live_queue_recovery_materialize_command"])
    if latest and latest.get("status") == "BLOCKED" and latest.get("queue_path"):
        return _experiment_queue_command(
            latest["queue_path"],
            latest.get("state_path"),
            "observe --tail-lines 20",
        )
    if latest and latest.get("queue_path"):
        return _experiment_queue_command(
            latest["queue_path"],
            latest.get("state_path"),
            "run-worker --execute --max-parallel 0",
        )
    return ".venv/bin/python tools/run_byte_shaving_materializer_campaign.py --help"


def _byte_shaving_acquisition_summary() -> dict[str, object]:
    rows = [
        _byte_shaving_acquisition_row(path)
        for path in _byte_shaving_acquisition_run_paths()
    ]
    latest = rows[0] if rows else None
    if latest is None:
        status = "PENDING"
        reason = "no high-level byte-shaving materializer campaign runs found"
    elif latest.get("status") == "BLOCKED":
        blockers = latest.get("blockers") if isinstance(latest.get("blockers"), list) else []
        status = "BLOCKED"
        reason = (
            f"latest campaign run is blocked by {len(blockers)} issue(s)"
            if blockers
            else "latest campaign run is blocked"
        )
    elif latest.get("status") == "READY_LOCAL_QUEUE":
        status = "READY_LOCAL_QUEUE"
        reason = (
            f"{latest.get('experiment_count', 0)} experiment(s) queued from "
            f"{latest.get('high_level_action_source_count', 0)} high-level source(s)"
        )
    elif latest.get("queue_feedback_candidate_actuation_planning_queue_emitted") is True:
        status = "NEEDS_RECEIVER_COMPILER"
        reason = (
            "widened inverse-action cells reached actuation planning; "
            "receiver/compiler transform is the next blocker"
        )
    elif latest.get("queue_feedback_replan_candidate_widening_ready") is True:
        status = "NEEDS_CANDIDATE_WIDENING"
        reason = (
            "latest feedback action surface has no selected materializer cells; "
            "refresh or widen inverse candidate generation"
        )
    else:
        status = "PENDING"
        reason = "latest campaign run has not produced an executable queue yet"

    total_executable = sum(_safe_int(row.get("executable_work_count")) for row in rows)
    total_blocked = sum(_safe_int(row.get("blocked_work_count")) for row in rows)
    total_work = total_executable + total_blocked
    return {
        "schema": "pact.byte_shaving_acquisition_summary.v1",
        "scan_roots": [_repo_rel(root) for root in BYTE_SHAVING_ACQUISITION_SCAN_ROOTS],
        "status": status,
        "reason": reason,
        "campaign_run_count": len(rows),
        "total_experiment_count": sum(_safe_int(row.get("experiment_count")) for row in rows),
        "total_executable_work_count": total_executable,
        "total_blocked_work_count": total_blocked,
        "total_compiler_required_count": sum(
            _safe_int(row.get("compiler_required_count")) for row in rows
        ),
        "total_packet_ir_operation_set_count": sum(
            _safe_int(row.get("packet_ir_operation_set_count")) for row in rows
        ),
        "total_queue_consumable_packet_ir_operation_set_count": sum(
            _safe_int(row.get("queue_consumable_packet_ir_operation_set_count"))
            for row in rows
        ),
        "total_packet_ir_byte_closed_operation_count": sum(
            _safe_int(row.get("packet_ir_byte_closed_operation_count")) for row in rows
        ),
        "total_exact_readiness_handoff_count": sum(
            _safe_int(row.get("exact_readiness_handoff_count")) for row in rows
        ),
        "queue_feedback_ready_count": sum(
            1 for row in rows if row.get("queue_feedback_replan_ready") is True
        ),
        "queue_feedback_followup_queue_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_followup_queue_emitted") is True
        ),
        "queue_feedback_followup_policy_enabled_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_followup_policy_enabled") is True
        ),
        "queue_feedback_followup_executed_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_followup_executed") is True
        ),
        "queue_feedback_followup_execution_success_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_followup_execution_success") is True
        ),
        "queue_feedback_policy_continue_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_policy_should_continue") is True
        ),
        "queue_feedback_candidate_widening_ready_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_candidate_widening_ready") is True
        ),
        "queue_feedback_dry_no_selected_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_dry_no_selected_cells") is True
        ),
        "queue_feedback_archive_delta_blocked_cell_count": sum(
            _safe_int(row.get("queue_feedback_replan_archive_delta_blocked_cell_count"))
            for row in rows
        ),
        "queue_feedback_candidate_widening_queue_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_candidate_widening_queue_emitted") is True
        ),
        "queue_feedback_candidate_actuation_planning_queue_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_candidate_actuation_planning_queue_emitted")
            is True
        ),
        "queue_observation_recovery_required_count": sum(
            1 for row in rows if row.get("queue_observation_recovery_required") is True
        ),
        "queue_observation_recovery_queue_count": sum(
            1
            for row in rows
            if row.get("queue_observation_recovery_queue_emitted") is True
        ),
        "source_failure_diagnostic_count": sum(
            _safe_int(row.get("source_failure_diagnostic_count")) for row in rows
        ),
        "source_failure_non_rewindable_count": sum(
            _safe_int(row.get("source_failure_non_rewindable_count")) for row in rows
        ),
        "source_failure_recovery_gated_count": sum(
            1
            for row in rows
            if row.get("source_failure_recovery_queue_execution_recommended")
            is False
        ),
        "source_failure_blocker_count": sum(
            len(row.get("source_failure_blockers"))
            for row in rows
            if isinstance(row.get("source_failure_blockers"), list)
        ),
        "queue_observation_recovery_executed_count": sum(
            1 for row in rows if row.get("queue_observation_recovery_executed") is True
        ),
        "queue_observation_recovery_execution_success_count": sum(
            1
            for row in rows
            if row.get("queue_observation_recovery_execution_success") is True
        ),
        "post_recovery_feedback_replan_count": sum(
            1
            for row in rows
            if row.get("post_recovery_feedback_replan_triggered") is True
        ),
        "post_recovery_feedback_replan_success_count": sum(
            1
            for row in rows
            if row.get("post_recovery_feedback_replan_success") is True
        ),
        "post_recovery_feedback_policy_continue_count": sum(
            1
            for row in rows
            if row.get("post_recovery_queue_feedback_replan_policy_should_continue")
            is True
        ),
        "post_recovery_feedback_followup_queue_count": sum(
            1
            for row in rows
            if row.get("post_recovery_queue_feedback_replan_followup_queue_emitted")
            is True
        ),
        "post_recovery_feedback_followup_executed_count": sum(
            1
            for row in rows
            if row.get("post_recovery_queue_feedback_replan_followup_executed")
            is True
        ),
        "post_recovery_feedback_followup_execution_success_count": sum(
            1
            for row in rows
            if row.get(
                "post_recovery_queue_feedback_replan_followup_execution_success"
            )
            is True
        ),
        "post_recovery_feedback_continuation_queue_count": sum(
            1
            for row in rows
            if row.get(
                "post_recovery_queue_feedback_replan_continuation_queue_emitted"
            )
            is True
        ),
        "queue_observation_recovery_grouped_blocker_count": sum(
            _safe_int(row.get("queue_observation_recovery_grouped_blocker_count"))
            for row in rows
        ),
        "queue_observation_recovery_repeated_group_count": sum(
            _safe_int(row.get("queue_observation_recovery_repeated_group_count"))
            for row in rows
        ),
        "queue_observation_maintenance_recommended_count": sum(
            1
            for row in rows
            if row.get("queue_observation_maintenance_recommended") is True
        ),
        "queue_observation_required_action_count": sum(
            _safe_int(row.get("queue_observation_required_action_count"))
            for row in rows
        ),
        "queue_observation_maintenance_action_count": sum(
            _safe_int(row.get("queue_observation_maintenance_action_count"))
            for row in rows
        ),
        "ready_for_queue_health_recovery_count": sum(
            1 for row in rows if row.get("ready_for_queue_health_recovery") is True
        ),
        "live_queue_observation_used_count": sum(
            1 for row in rows if row.get("live_queue_observation_used") is True
        ),
        "live_queue_observation_unhealthy_count": sum(
            1
            for row in rows
            if row.get("live_queue_observation_used") is True
            and row.get("live_queue_observation_healthy") is not True
        ),
        "live_queue_observation_blocker_count": sum(
            _safe_int(row.get("live_queue_observation_blocker_count"))
            for row in rows
        ),
        "queue_feedback_continuation_queue_count": sum(
            1
            for row in rows
            if row.get("queue_feedback_replan_continuation_queue_emitted") is True
        ),
        "overall_executable_conversion_rate": total_executable / total_work
        if total_work
        else 0.0,
        "local_mlx_ready_step_count": sum(
            _safe_int(row.get("local_mlx_ready_step_count")) for row in rows
        ),
        "latest_rows": rows[:5],
        "next_command": _byte_shaving_acquisition_next_command(latest),
        "observe_command": (
            _experiment_queue_command(
                latest[
                    "post_recovery_queue_feedback_replan_continuation_queue_path"
                ],
                latest.get(
                    "post_recovery_queue_feedback_replan_continuation_queue_state_path"
                ),
                "observe --tail-lines 20",
            )
            if (
                latest
                and latest.get(
                    "post_recovery_queue_feedback_replan_continuation_queue_emitted"
                )
                is True
                and latest.get(
                    "post_recovery_queue_feedback_replan_continuation_queue_path"
                )
            )
            else _experiment_queue_command(
                latest["post_recovery_queue_feedback_replan_followup_queue_path"],
                latest.get("post_recovery_queue_feedback_replan_followup_state_path"),
                "observe --tail-lines 20",
            )
            if (
                latest
                and latest.get("post_recovery_queue_feedback_replan_followup_queue_emitted")
                is True
                and latest.get("post_recovery_feedback_replan_success") is not True
                and latest.get("post_recovery_queue_feedback_replan_followup_queue_path")
            )
            else
            _experiment_queue_command(
                latest["queue_path"],
                latest.get("state_path"),
                "observe --tail-lines 20",
            )
            if (
                latest
                and latest.get("source_failure_recovery_queue_execution_recommended")
                is False
                and latest.get("queue_path")
            )
            else
            _experiment_queue_command(
                latest["queue_observation_recovery_queue_path"],
                latest.get("queue_observation_recovery_queue_state_path"),
                "observe --tail-lines 20",
            )
            if (
                latest
                and latest.get("queue_observation_recovery_queue_emitted") is True
                and latest.get("queue_observation_recovery_execution_success") is not True
                and latest.get("queue_observation_recovery_queue_path")
            )
            else _experiment_queue_command(
                latest["queue_path"],
                latest.get("state_path"),
                "observe --tail-lines 20",
            )
            if latest and latest.get("queue_path")
            else ""
        ),
        **_false_authority_fields(),
    }


def _format_byte_shaving_acquisition_summary() -> str:
    payload = _byte_shaving_acquisition_summary()
    lines = [
        "High-level inverse-steganalysis/action-surface campaign intake. "
        "This is local research queue authority, not score authority.",
        f"status: {payload['status']} — {payload['reason']}",
        (
            "runs: "
            f"{payload['campaign_run_count']} "
            f"experiments={payload['total_experiment_count']} "
            f"executable_work={payload['total_executable_work_count']} "
            f"blocked_work={payload['total_blocked_work_count']} "
            f"conversion={payload['overall_executable_conversion_rate']:.2%} "
            f"compiler_gaps={payload['total_compiler_required_count']} "
            f"packetir_sets={payload['total_packet_ir_operation_set_count']} "
            "packetir_queue_ready="
            f"{payload['total_queue_consumable_packet_ir_operation_set_count']} "
            f"exact_handoffs={payload['total_exact_readiness_handoff_count']} "
            f"feedback_ready={payload['queue_feedback_ready_count']} "
            f"feedback_queued={payload['queue_feedback_followup_queue_count']} "
            f"feedback_policy={payload['queue_feedback_followup_policy_enabled_count']} "
            f"feedback_executed={payload['queue_feedback_followup_executed_count']} "
            "feedback_success="
            f"{payload['queue_feedback_followup_execution_success_count']} "
            f"feedback_continue={payload['queue_feedback_policy_continue_count']} "
            "feedback_widen="
            f"{payload['queue_feedback_candidate_widening_ready_count']} "
            "feedback_dry="
            f"{payload['queue_feedback_dry_no_selected_count']} "
            "feedback_archive_delta_blocked_cells="
            f"{payload['queue_feedback_archive_delta_blocked_cell_count']} "
            "feedback_widen_queue="
            f"{payload['queue_feedback_candidate_widening_queue_count']} "
            "feedback_actuation_queue="
            f"{payload['queue_feedback_candidate_actuation_planning_queue_count']} "
            "queue_recovery_required="
            f"{payload['queue_observation_recovery_required_count']} "
            "queue_recovery_ready="
            f"{payload['ready_for_queue_health_recovery_count']} "
            "live_queue_observed="
            f"{payload['live_queue_observation_used_count']} "
            "live_queue_unhealthy="
            f"{payload['live_queue_observation_unhealthy_count']} "
            "live_queue_blockers="
            f"{payload['live_queue_observation_blocker_count']} "
            "queue_recovery_queued="
            f"{payload['queue_observation_recovery_queue_count']} "
            "source_failure_diagnostics="
            f"{payload['source_failure_diagnostic_count']} "
            "source_non_rewindable="
            f"{payload['source_failure_non_rewindable_count']} "
            "source_recovery_gated="
            f"{payload['source_failure_recovery_gated_count']} "
            "source_failure_blockers="
            f"{payload['source_failure_blocker_count']} "
            "queue_recovery_executed="
            f"{payload['queue_observation_recovery_executed_count']} "
            "queue_recovery_success="
            f"{payload['queue_observation_recovery_execution_success_count']} "
            "post_recovery_replan="
            f"{payload['post_recovery_feedback_replan_count']} "
            "post_recovery_replan_success="
            f"{payload['post_recovery_feedback_replan_success_count']} "
            "post_recovery_feedback_queued="
            f"{payload['post_recovery_feedback_followup_queue_count']} "
            "post_recovery_feedback_executed="
            f"{payload['post_recovery_feedback_followup_executed_count']} "
            "post_recovery_feedback_success="
            f"{payload['post_recovery_feedback_followup_execution_success_count']} "
            "post_recovery_continue="
            f"{payload['post_recovery_feedback_policy_continue_count']} "
            "post_recovery_continuation_queued="
            f"{payload['post_recovery_feedback_continuation_queue_count']} "
            "queue_recovery_groups="
            f"{payload['queue_observation_recovery_grouped_blocker_count']} "
            "queue_recovery_repeated_groups="
            f"{payload['queue_observation_recovery_repeated_group_count']} "
            "queue_maintenance="
            f"{payload['queue_observation_maintenance_recommended_count']} "
            "feedback_continuation_queued="
            f"{payload['queue_feedback_continuation_queue_count']} "
            f"local_mlx_ready_steps={payload['local_mlx_ready_step_count']}"
        ),
        f"score_claim: {payload['score_claim']}",
        f"ready_for_exact_eval_dispatch: {payload['ready_for_exact_eval_dispatch']}",
    ]
    latest_rows = payload.get("latest_rows")
    if isinstance(latest_rows, list) and latest_rows:
        lines.append("latest campaign runs:")
        for row in latest_rows[:3]:
            if not isinstance(row, dict):
                continue
            plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
            top_combo = (
                plan.get("top_combo")
                if isinstance(plan.get("top_combo"), dict)
                else {}
            )
            families = ", ".join(
                str(item) for item in plan.get("operation_families", [])[:4]
            ) if isinstance(plan.get("operation_families"), list) else ""
            lines.append(
                "  - "
                f"{row.get('path')} status={row.get('status')} "
                f"queue={row.get('queue_id') or '<none>'} "
                f"experiments={row.get('experiment_count', 0)} "
                f"conversion={row.get('executable_conversion_rate', 0.0):.2%} "
                f"compiler_gaps={row.get('compiler_required_count', 0)} "
                f"packetir_sets={row.get('packet_ir_operation_set_count', 0)} "
                "packetir_queue_ready="
                f"{row.get('queue_consumable_packet_ir_operation_set_count', 0)} "
                f"exact_handoffs={row.get('exact_readiness_handoff_count', 0)} "
                f"feedback_ready={row.get('queue_feedback_replan_ready') is True} "
                "feedback_queued="
                f"{row.get('queue_feedback_replan_followup_queue_emitted') is True} "
                "feedback_policy="
                f"{row.get('queue_feedback_replan_followup_policy') or '<none>'} "
                "feedback_executed="
                f"{row.get('queue_feedback_replan_followup_executed') is True} "
                "feedback_success="
                f"{row.get('queue_feedback_replan_followup_execution_success') is True} "
                "feedback_decision="
                f"{row.get('queue_feedback_replan_policy_decision') or '<none>'} "
                "feedback_continue="
                f"{row.get('queue_feedback_replan_policy_should_continue') is True} "
                "feedback_widening="
                f"{row.get('queue_feedback_replan_candidate_widening_ready') is True} "
                "feedback_dry="
                f"{row.get('queue_feedback_replan_dry_no_selected_cells') is True} "
                "feedback_archive_delta_blocked_cells="
                f"{row.get('queue_feedback_replan_archive_delta_blocked_cell_count', 0)} "
                "feedback_widen_queue="
                f"{row.get('queue_feedback_candidate_widening_queue_emitted') is True} "
                "feedback_actuation_queue="
                f"{row.get('queue_feedback_candidate_actuation_planning_queue_emitted') is True} "
                "queue_recovery_required="
                f"{row.get('queue_observation_recovery_required') is True} "
                "queue_recovery_ready="
                f"{row.get('ready_for_queue_health_recovery') is True} "
                "queue_observation_source="
                f"{row.get('queue_observation_source') or '<none>'} "
                "queue_recovery_plan_source="
                f"{row.get('queue_observation_recovery_plan_source') or '<none>'} "
                "live_queue_observed="
                f"{row.get('live_queue_observation_used') is True} "
                "live_queue_healthy="
                f"{row.get('live_queue_observation_healthy')} "
                "live_queue_mode="
                f"{row.get('live_queue_observation_mode') or '<none>'} "
                "live_queue_failed_steps="
                f"{row.get('live_queue_observation_failed_step_count', 0)} "
                "queue_recovery_queued="
                f"{row.get('queue_observation_recovery_queue_emitted') is True} "
                "source_failure_diagnostics="
                f"{row.get('source_failure_diagnostic_count', 0)} "
                "source_non_rewindable="
                f"{row.get('source_failure_non_rewindable_count', 0)} "
                "source_recovery_recommended="
                f"{row.get('source_failure_recovery_queue_execution_recommended')} "
                "queue_recovery_executed="
                f"{row.get('queue_observation_recovery_executed') is True} "
                "queue_recovery_success="
                f"{row.get('queue_observation_recovery_execution_success') is True} "
                "post_recovery_replan="
                f"{row.get('post_recovery_feedback_replan_triggered') is True} "
                "post_recovery_replan_success="
                f"{row.get('post_recovery_feedback_replan_success') is True} "
                "post_recovery_feedback_queued="
                f"{row.get('post_recovery_queue_feedback_replan_followup_queue_emitted') is True} "
                "post_recovery_feedback_executed="
                f"{row.get('post_recovery_queue_feedback_replan_followup_executed') is True} "
                "post_recovery_feedback_success="
                f"{row.get('post_recovery_queue_feedback_replan_followup_execution_success') is True} "
                "post_recovery_decision="
                f"{row.get('post_recovery_queue_feedback_replan_policy_decision') or '<none>'} "
                "post_recovery_continue="
                f"{row.get('post_recovery_queue_feedback_replan_policy_should_continue') is True} "
                "post_recovery_continuation_queued="
                f"{row.get('post_recovery_queue_feedback_replan_continuation_queue_emitted') is True} "
                "queue_recovery_groups="
                f"{row.get('queue_observation_recovery_grouped_blocker_count', 0)} "
                "queue_recovery_repeated_groups="
                f"{row.get('queue_observation_recovery_repeated_group_count', 0)} "
                "queue_maintenance="
                f"{row.get('queue_observation_maintenance_recommended') is True} "
                "queue_recovery_actions="
                f"{row.get('queue_observation_required_action_count', 0)} "
                "feedback_continuation_queued="
                f"{row.get('queue_feedback_replan_continuation_queue_emitted') is True} "
                f"ready_steps={row.get('ready_step_count', 0)} "
                f"local_mlx_ready={row.get('local_mlx_ready_step_count', 0)}"
            )
            top_groups = row.get("queue_observation_recovery_top_groups")
            if isinstance(top_groups, list) and top_groups:
                lines.append(
                    "    queue_recovery_top_groups="
                    + "; ".join(str(item) for item in top_groups[:3])
                )
            recovery_blockers = row.get("queue_observation_recovery_execution_blockers")
            if isinstance(recovery_blockers, list) and recovery_blockers:
                lines.append(
                    "    queue_recovery_execution_blockers="
                    + ", ".join(str(item) for item in recovery_blockers[:5])
                )
            policy_blockers = row.get("queue_observation_recovery_policy_blockers")
            if isinstance(policy_blockers, list) and policy_blockers:
                lines.append(
                    "    queue_recovery_policy_blockers="
                    + ", ".join(str(item) for item in policy_blockers[:5])
                )
            source_blockers = row.get(
                "queue_observation_recovery_source_observation_blockers"
            )
            if isinstance(source_blockers, list) and source_blockers:
                lines.append(
                    "    queue_recovery_source_blockers="
                    + ", ".join(str(item) for item in source_blockers[:5])
                )
            source_failure_blockers = row.get("source_failure_blockers")
            if isinstance(source_failure_blockers, list) and source_failure_blockers:
                lines.append(
                    "    source_failure_blockers="
                    + ", ".join(str(item) for item in source_failure_blockers[:5])
                )
            source_failure_recovery_blockers = row.get(
                "source_failure_recovery_queue_execution_blockers"
            )
            if (
                isinstance(source_failure_recovery_blockers, list)
                and source_failure_recovery_blockers
            ):
                lines.append(
                    "    source_failure_recovery_blockers="
                    + ", ".join(
                        str(item) for item in source_failure_recovery_blockers[:5]
                    )
                )
            live_blockers = row.get("live_queue_observation_blockers")
            if isinstance(live_blockers, list) and live_blockers:
                lines.append(
                    "    live_queue_blockers="
                    + ", ".join(str(item) for item in live_blockers[:5])
                )
            recovery_materialize = row.get("live_queue_recovery_materialize_command")
            if recovery_materialize:
                lines.append(f"    recovery_materialize={recovery_materialize}")
            if top_combo.get("combo_id"):
                lines.append(
                    "    top_combo: "
                    f"{top_combo.get('combo_id')} "
                    f"gain={top_combo.get('expected_score_gain', 0.0):.6g} "
                    f"units={top_combo.get('unit_count', 0)} "
                    f"ops={families or '-'}"
                )
            blockers = row.get("blockers")
            if isinstance(blockers, list) and blockers:
                lines.append("    blockers: " + "; ".join(str(v) for v in blockers[:4]))
    observe = payload.get("observe_command")
    if observe:
        lines.append("observe:")
        lines.append(f"  {observe}")
    lines.append("next local worker command:")
    lines.append(f"  {payload['next_command']}")
    return "\n".join(lines)


def _load_pr91_hpm1_readiness_artifact() -> dict[str, object]:
    readiness = _run_json(PR91_HPM1_READINESS)
    runtime = _run_json(PR91_HPM1_RUNTIME_CONTRACT)
    readiness_artifact = _load_json_file(PR91_HPM1_READINESS_ARTIFACT)
    runtime_artifact = _load_json_file(PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT)
    audit_errors = [
        str(payload["_error"])
        for payload in (readiness, runtime, readiness_artifact, runtime_artifact)
        if isinstance(payload.get("_error"), str)
    ]
    readiness_blockers = list(readiness.get("dispatch_blockers") or [])
    runtime_blockers = list(runtime.get("dispatch_blockers") or [])
    artifact_readiness_blockers = list(readiness_artifact.get("dispatch_blockers") or [])
    artifact_runtime_blockers = list(runtime_artifact.get("dispatch_blockers") or [])
    readiness_hash = _canonical_payload_hash(readiness)
    runtime_hash = _canonical_payload_hash(runtime)
    readiness_artifact_hash = _canonical_payload_hash(readiness_artifact)
    runtime_artifact_hash = _canonical_payload_hash(runtime_artifact)
    zip_report = {}
    if isinstance(readiness.get("member_x"), dict) and isinstance(readiness["member_x"].get("zip_report"), dict):
        zip_report = readiness["member_x"]["zip_report"]
    wire_contract = zip_report.get("wire_contract") if isinstance(zip_report, dict) else {}
    hpac_contract = runtime.get("hpac_device_contract")
    if not isinstance(hpac_contract, dict):
        hpac_contract = {}
    hpac_cpu_resolved = (
        hpac_contract.get("passed") is True
        and hpac_contract.get("status") == "resolved_cpu_only"
        and hpac_contract.get("resolved_device") == "cpu"
    )
    summary = (
        "PR91 static archive/member/HPM1 custody is visible, and HPAC device "
        "semantics are pinned CPU-only; HPM1 decode/reencode and sidecar-free "
        "runtime consumption remain blocked. This is not a score or dispatch artifact."
        if hpac_cpu_resolved
        else (
            "PR91 static archive/member/HPM1 custody is visible, but HPM1 decode/reencode "
            "and HPAC runtime device semantics remain blocked; this is not a score or "
            "dispatch artifact."
        )
    )
    next_patch = (
        "Recover full HPM1 decode/reencode parity, then prove sidecar-free runtime "
        "consumption before any dispatch."
        if hpac_cpu_resolved
        else (
            "Resolve HPAC CPU/CUDA device contract, recover full HPM1 decode/reencode "
            "parity, then prove sidecar-free runtime consumption before any dispatch."
        )
    )
    return {
        "kind": "pr91_hpm1_readiness_bundle",
        "name": "PR91 HPM1 categorical mask rate signal",
        "state": "AUDIT_ERROR_FAIL_CLOSED"
        if audit_errors
        else ("BLOCKED_CPU_CONTRACT_PINNED" if hpac_cpu_resolved else "BLOCKED_FAIL_CLOSED"),
        "evidence_grade": readiness.get("evidence_grade"),
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "audit_errors": audit_errors,
        "artifact_path": _repo_rel(PR91_HPM1_READINESS_ARTIFACT),
        "runtime_contract_artifact_path": _repo_rel(PR91_HPM1_RUNTIME_CONTRACT_ARTIFACT),
        "readiness_manifest_hash_self_consistent": _manifest_hash_self_consistent(readiness),
        "runtime_manifest_hash_self_consistent": _manifest_hash_self_consistent(runtime),
        "readiness_artifact_manifest_hash_self_consistent": _manifest_hash_self_consistent(
            readiness_artifact
        ),
        "runtime_artifact_manifest_hash_self_consistent": _manifest_hash_self_consistent(
            runtime_artifact
        ),
        "readiness_artifact_hash_matches_live": bool(readiness_hash)
        and readiness_hash == readiness_artifact_hash,
        "runtime_artifact_hash_matches_live": bool(runtime_hash)
        and runtime_hash == runtime_artifact_hash,
        "archive_custody_matches": (
            isinstance(readiness.get("source_archive"), dict)
            and readiness["source_archive"].get("matches_expected") is True
        ),
        "hpm1_mask_custody_matches": (
            isinstance(readiness.get("hpm1_mask_segment"), dict)
            and readiness["hpm1_mask_segment"].get("matches_expected") is True
        ),
        "zip_wire_contract_passed": isinstance(wire_contract, dict)
        and wire_contract.get("passed") is True,
        "ambient_device_call_count": runtime.get("ambient_device_call_count"),
        "contradiction_count": runtime.get("contradiction_count"),
        "dispatch_blockers": readiness_blockers + runtime_blockers,
        "artifact_dispatch_blockers": artifact_readiness_blockers + artifact_runtime_blockers,
        "summary": summary,
        "next_patch": next_patch,
    }


def _discover_inverse_scorer_cell_chain_manifests() -> list[Path]:
    candidates: list[Path] = []
    for pattern in (
        INVERSE_SCORER_CHAIN_MANIFEST_NAME,
        f"*/{INVERSE_SCORER_CHAIN_MANIFEST_NAME}",
        f"*/*/{INVERSE_SCORER_CHAIN_MANIFEST_NAME}",
    ):
        candidates.extend(INVERSE_SCORER_CHAIN_SCAN_ROOT.glob(pattern))
    return sorted(
        {path for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _inverse_scorer_cell_chain_readiness_artifacts(
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _discover_inverse_scorer_cell_chain_manifests()[:limit]:
        payload = _load_json_file(path)
        blockers = [
            str(item)
            for item in payload.get("readiness_blockers", [])
            if str(item)
        ]
        candidate_archive = payload.get("candidate_archive")
        if not isinstance(candidate_archive, dict):
            candidate_archive = {}
        next_gates = [
            str(item)
            for item in payload.get("next_required_gates", [])
            if str(item)
        ]
        has_error = bool(payload.get("_error"))
        receiver_ready = payload.get("receiver_contract_satisfied") is True
        parity_ready = payload.get("inflate_parity_satisfied") is True
        state = (
            "AUDIT_ERROR_FAIL_CLOSED"
            if has_error
            else (
                "BLOCKED_EXACT_AUTH_EVAL"
                if receiver_ready and parity_ready
                else "BLOCKED_PARITY_OR_RUNTIME"
            )
        )
        rows.append(
            {
                "kind": "inverse_scorer_cell_candidate_chain",
                "name": "IAS1 inverse-scorer candidate proof chain",
                "state": state,
                "artifact_path": _repo_rel(path),
                "artifact_sha256": _sha256_file(path) if path.is_file() else "",
                "candidate_archive": candidate_archive,
                "candidate_archive_sha256": payload.get("candidate_archive_sha256") or "",
                "candidate_archive_bytes": payload.get("candidate_archive_bytes"),
                "runtime_adapter_ready": payload.get("runtime_adapter_ready") is True,
                "receiver_proof_ready": payload.get("receiver_proof_ready") is True,
                "receiver_contract_satisfied": receiver_ready,
                "inflate_parity_satisfied": parity_ready,
                "candidate_runtime_adapter_blocker_cleared": (
                    payload.get("candidate_runtime_adapter_blocker_cleared") is True
                ),
                "full_frame_or_shell_parity_required": (
                    payload.get("full_frame_or_shell_parity_required") is True
                ),
                "next_required_gates": next_gates,
                "readiness_blockers": blockers,
                "dispatch_blockers": [
                    str(item)
                    for item in payload.get("dispatch_blockers", blockers)
                    if str(item)
                ],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "dispatch_attempted": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "audit_errors": [str(payload["_error"])] if has_error else [],
                "summary": (
                    "IAS1 descriptor receiver evidence is visible to the operator; "
                    "full-frame inflate parity and contest auth eval remain explicit gates."
                ),
                "next_patch": (
                    "Attach a strict inflate parity proof, then route the same archive/runtime "
                    "through claimed contest CPU/CUDA auth eval before any score language."
                ),
            }
        )
    return rows


def _non_dispatchable_readiness_artifacts() -> list[dict[str, object]]:
    return [
        _load_pr91_hpm1_readiness_artifact(),
        *_inverse_scorer_cell_chain_readiness_artifacts(),
    ]


def _format_non_dispatchable_readiness_artifacts() -> str:
    lines = []
    for artifact in _non_dispatchable_readiness_artifacts():
        if artifact.get("kind") == "inverse_scorer_cell_candidate_chain":
            blockers = ", ".join(str(item) for item in artifact["dispatch_blockers"])
            gates = ", ".join(str(item) for item in artifact["next_required_gates"])
            lines.append(
                f"  • {artifact['kind']} — {artifact['name']}\n"
                f"    state {artifact['state']}   ready_for_exact_eval_dispatch=false\n"
                f"    chain artifact: {artifact['artifact_path']}\n"
                f"    candidate sha: {artifact['candidate_archive_sha256']} "
                f"bytes={artifact['candidate_archive_bytes']}\n"
                f"    receiver: adapter={artifact['runtime_adapter_ready']} "
                f"proof={artifact['receiver_proof_ready']} "
                f"contract={artifact['receiver_contract_satisfied']}\n"
                f"    parity: satisfied={artifact['inflate_parity_satisfied']} "
                f"required={artifact['full_frame_or_shell_parity_required']}\n"
                f"    next gates: {gates if gates else '(none)'}\n"
                f"    blockers: {blockers if blockers else '(none)'}\n"
                f"    summary: {artifact['summary']}\n"
                f"    next patch: {artifact['next_patch']}"
            )
            continue
        blockers = ", ".join(str(item) for item in artifact["dispatch_blockers"])
        lines.append(
            f"  • {artifact['kind']} — {artifact['name']}\n"
            f"    state {artifact['state']}   ready_for_exact_eval_dispatch=false\n"
            f"    readiness artifact: {artifact['artifact_path']}\n"
            f"    runtime artifact: {artifact['runtime_contract_artifact_path']}\n"
            f"    artifact/live hashes: readiness={artifact['readiness_artifact_hash_matches_live']} "
            f"runtime={artifact['runtime_artifact_hash_matches_live']}\n"
            f"    custody: archive={artifact['archive_custody_matches']} "
            f"hpm1_mask={artifact['hpm1_mask_custody_matches']} "
            f"zip_wire={artifact['zip_wire_contract_passed']}\n"
            f"    runtime: ambient_device_calls={artifact['ambient_device_call_count']} "
            f"contradictions={artifact['contradiction_count']}\n"
            f"    blockers: {blockers if blockers else '(none)'}\n"
            f"    audit errors: {', '.join(artifact['audit_errors']) if artifact['audit_errors'] else '(none)'}\n"
            f"    summary: {artifact['summary']}\n"
            f"    next patch: {artifact['next_patch']}"
        )
    return "\n\n".join(lines) if lines else "  (none)"


def _format_gated_lanes(
    *,
    target_score: float = DEFAULT_SCORE_LOWERING_TARGET,
    show_above_target: bool = False,
) -> str:
    lanes = _annotate_score_target_lanes(
        PHASE_4_GATED_LANES,
        target_score=target_score,
        active_only=not show_above_target,
    )
    lines = []
    for lane in lanes:
        lo, hi = lane["predicted_band"]
        lines.append(
            f"  • {lane['lane_id']} — {lane['name']}\n"
            f"    predicted [{lo:.4f}, {hi:.4f}]   est ${lane['estimated_cost_usd']:.2f}   "
            f"council priority {lane['council_priority']}\n"
            f"{_score_target_line(lane)}\n"
            f"    GATE: {lane['gate_condition']}\n"
            f"    Operator one-liner (post-gate):\n"
            f"      {lane['one_liner']}"
        )
    text = "\n\n".join(lines) if lines else f"  (none active below target {target_score:.4f})"
    if not show_above_target:
        text += _hidden_above_target_summary(PHASE_4_GATED_LANES, target_score=target_score)
    return text


def _format_composition_lanes(
    *,
    target_score: float = DEFAULT_SCORE_LOWERING_TARGET,
    show_above_target: bool = False,
) -> str:
    lanes = _annotate_score_target_lanes(
        PHASE_5_COMPOSITION_LANES,
        target_score=target_score,
        active_only=not show_above_target,
    )
    lines = []
    for lane in lanes:
        lo, hi = lane["predicted_band"]
        lines.append(
            f"  • {lane['lane_id']} — {lane['name']}\n"
            f"    predicted [{lo:.4f}, {hi:.4f}]   est ${lane['estimated_cost_usd']:.2f}   "
            f"council priority {lane['council_priority']}\n"
            f"{_score_target_line(lane)}\n"
            f"    GATE: {lane['gate_condition']}\n"
            f"    Operator one-liner (post-gate):\n"
            f"      {lane['one_liner']}"
        )
    text = "\n\n".join(lines) if lines else f"  (none active below target {target_score:.4f})"
    if not show_above_target:
        text += _hidden_above_target_summary(PHASE_5_COMPOSITION_LANES, target_score=target_score)
    return text


def _run(script: Path, extra_args: list[str] | None = None) -> str:
    python = str(REPO_VENV_PYTHON) if REPO_VENV_PYTHON.is_file() else sys.executable
    args = [python, str(script)]
    if extra_args:
        args.extend(extra_args)
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return (
            f"(tool exited with code {proc.returncode}; stderr: "
            f"{proc.stderr.strip()[:200]})"
        )
    return proc.stdout


def _run_json(script: Path, extra_args: list[str] | None = None) -> dict:
    text = _run(script, extra_args or [])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_error": "non-JSON output", "_stdout": text[:500]}


def _dispatch_claim_summary() -> dict[str, object]:
    """Return the read-only dispatch-claim summary used to prevent duplicate evals."""
    return _run_json(CLAIM_DISPATCH, ["summary", "--format", "json", "--live-only"])


def _active_dispatch_claim_count(summary: dict[str, object]) -> int:
    try:
        return max(0, int(summary.get("active_count") or 0))
    except (TypeError, ValueError):
        return 0


def _dispatch_claim_historical_summary() -> dict[str, object]:
    """Return all-history claim hygiene without using it as a live dispatch blocker."""
    return _run_json(CLAIM_DISPATCH, ["summary", "--format", "json"])


def _latest_provider_readiness_artifact() -> Path | None:
    if PROVIDER_READINESS_LATEST.is_file():
        return PROVIDER_READINESS_LATEST
    candidates = sorted(
        (REPO_ROOT / "experiments/results").glob("cloud_provider_readiness_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _provider_readiness(refresh: bool = False) -> dict[str, object]:
    """Return cloud-provider status without implicit network checks.

    Normal operator briefing loads the latest provider-readiness artifact from
    disk. Live cloud CLI checks are intentionally opt-in via
    ``--refresh-provider-readiness`` so a status briefing cannot hang or mutate
    provider state.
    """

    if refresh:
        return _run_json(
            CLOUD_PROVIDER_READINESS,
            [
                "--output",
                str(PROVIDER_READINESS_LATEST),
                "--timeout-s",
                "8",
            ],
        )
    latest = _latest_provider_readiness_artifact()
    if latest is None:
        return {
            "_error": "no provider readiness artifact found",
            "next_command": (
                ".venv/bin/python tools/cloud_provider_readiness.py "
                "--output experiments/results/cloud_provider_readiness_latest.json"
            ),
        }
    payload = _load_json_file(latest)
    payload["artifact_path"] = _repo_rel(latest)
    return payload


def _format_provider_readiness(refresh: bool = False) -> str:
    payload = _provider_readiness(refresh=refresh)
    if payload.get("_error"):
        return (
            "Provider readiness unavailable; no remote dispatch should be inferred.\n"
            f"  error: {payload.get('_error')}\n"
            f"  next:  {payload.get('next_command')}"
        )
    providers = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    lines = [
        "Read-only provider readiness. This is not a dispatch or score claim.",
        f"  generated_at_utc: {payload.get('generated_at_utc', '<unknown>')}",
        f"  artifact:         {payload.get('artifact_path', _repo_rel(PROVIDER_READINESS_LATEST))}",
        f"  score_claim:      {payload.get('score_claim', False)}",
        f"  exact_dispatch:   {payload.get('ready_for_exact_eval_dispatch', False)}",
    ]
    for row in providers:
        if not isinstance(row, dict):
            continue
        blockers = ", ".join(str(x) for x in row.get("blockers", [])) or "-"
        lines.append(
            "  - "
            f"{row.get('provider', '<unknown>')}: "
            f"{row.get('status', '<unknown>')} "
            f"exact_cuda_now={bool(row.get('exact_cuda_evidence_allowed'))} "
            f"proxy_only={bool(row.get('proxy_only'))} "
            f"blockers={blockers}"
        )
    return "\n".join(lines)


def _codex_inbox_summary() -> dict[str, object]:
    from tac.codex_to_claude_inbox import inbox_summary

    try:
        return inbox_summary()
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _format_codex_inbox_summary() -> str:
    payload = _codex_inbox_summary()
    if payload.get("_error"):
        return f"Codex inbox unavailable: {payload['_error']}"
    return "\n".join(
        [
            "Codex to Claude inbox. This is an execution-loop coordination surface, not score authority.",
            f"  open_questions:         {payload.get('open_questions_count', 0)}",
            f"  expired_open_questions: {payload.get('expired_open_questions_count', 0)}",
            f"  oldest_open_age_hours:  {payload.get('open_questions_oldest_age_hours', 0.0)}",
            f"  relays:                 {payload.get('relays_count', 0)}",
        ]
    )


def _format_dispatch_claim_summary() -> str:
    summary = _dispatch_claim_summary()
    if summary.get("_error"):
        return (
            "Dispatch claim helper failed; refuse remote/eval dispatch until "
            f"claim state is readable. error={summary.get('_error')}"
        )
    active = summary.get("active") if isinstance(summary.get("active"), list) else []
    stale = (
        summary.get("stale_nonterminal")
        if isinstance(summary.get("stale_nonterminal"), list)
        else []
    )
    invalid_lane_id = (
        summary.get("invalid_lane_id")
        if isinstance(summary.get("invalid_lane_id"), list)
        else []
    )
    unparsable = (
        summary.get("unparsable_timestamp")
        if isinstance(summary.get("unparsable_timestamp"), list)
        else []
    )
    lines = [
        "Read-only claim state from tools/claim_lane_dispatch.py summary.",
        f"  active: {summary.get('active_count', len(active))}",
        f"  stale_nonterminal: {summary.get('stale_nonterminal_count', len(stale))}",
        f"  terminal_latest: {summary.get('terminal_latest_count', '<unknown>')}",
        f"  unparsable_timestamp: {summary.get('unparsable_timestamp_count', len(unparsable))}",
        f"  invalid_lane_id: {summary.get('invalid_lane_id_count', len(invalid_lane_id))}",
    ]
    if active:
        lines.append("  ACTIVE CONFLICT GUARD: do not dispatch duplicate active lanes.")
        for row in active:
            lines.append(
                "    - "
                f"lane_id={row.get('lane_id', '<missing>')} "
                f"job={row.get('instance_job_id', '<missing>')} "
                f"platform={row.get('platform', '<missing>')} "
                f"status={row.get('status', '<missing>')} "
                f"agent={row.get('agent', '<missing>')}"
            )
    if stale:
        lines.append("  STALE NONTERMINAL: close/classify before relaunching these lanes.")
        for row in stale:
            lines.append(
                "    - "
                f"lane_id={row.get('lane_id', '<missing>')} "
                f"job={row.get('instance_job_id', '<missing>')} "
                f"status={row.get('status', '<missing>')}"
            )
    if unparsable:
        lines.append("  UNPARSABLE TIMESTAMPS: repair live claim rows before any dispatch.")
        for row in unparsable:
            lines.append(
                "    - "
                f"lane_id={row.get('lane_id', '<missing>')} "
                f"job={row.get('instance_job_id', '<missing>')} "
                f"timestamp={row.get('timestamp_utc', '<missing>')} "
                f"status={row.get('status', '<missing>')}"
            )
    if invalid_lane_id:
        lines.append("  INVALID LANE IDS: repair claim ledger before any dispatch.")
        for row in invalid_lane_id:
            lines.append(
                "    - "
                f"lane_id={row.get('lane_id', '<missing>')} "
                f"job={row.get('instance_job_id', '<missing>')} "
                f"status={row.get('status', '<missing>')}"
            )
    if not active and not stale and not unparsable and not invalid_lane_id:
        lines.append("  No active, stale, unparsable, or invalid live claims.")
    historical = _dispatch_claim_historical_summary()
    if historical.get("_error"):
        lines.append(
            "  Historical claim hygiene: UNKNOWN — "
            f"{historical.get('_error')}"
        )
    else:
        historical_invalid = int(historical.get("invalid_lane_id_count", 0) or 0)
        historical_unparsable = int(
            historical.get("unparsable_timestamp_count", 0) or 0
        )
        if historical_invalid or historical_unparsable:
            lines.append(
                "  All-history claim hygiene: WARNING — live+archived rows "
                f"invalid_lane_id={historical_invalid} "
                f"unparsable_timestamp={historical_unparsable}; "
                "live blockers are listed above"
            )
        else:
            lines.append("  All-history claim hygiene: PASS")
    return "\n".join(lines)


def _load_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"{_repo_rel(path)} unreadable JSON: {exc}"}
    return payload if isinstance(payload, dict) else {"_error": f"{_repo_rel(path)} is not a JSON object"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_canonical_payload_hash(payload: dict[str, object]) -> str:
    manifest = payload.get("tool_run_manifest")
    if not isinstance(manifest, dict):
        return ""
    value = manifest.get("canonical_payload_without_tool_manifest_sha256")
    return value if isinstance(value, str) else ""


def _recomputed_canonical_payload_hash(payload: dict[str, object]) -> str:
    without_manifest = dict(payload)
    without_manifest.pop("tool_run_manifest", None)
    text = json.dumps(without_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_hash_self_consistent(payload: dict[str, object]) -> bool:
    embedded = _embedded_canonical_payload_hash(payload)
    return bool(embedded) and embedded == _recomputed_canonical_payload_hash(payload)


def _canonical_payload_hash(payload: dict[str, object]) -> str:
    return _embedded_canonical_payload_hash(payload) if _manifest_hash_self_consistent(payload) else ""


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _section(title: str, body: str) -> str:
    bar = "═" * len(title)
    return f"\n{bar}\n{title}\n{bar}\n\n{body}"


def _load_l5_v2_packetir_matrix() -> dict[str, object]:
    """Load the committed PR106 PacketIR matrix for L5-v2 operator routing."""

    path = REPO_ROOT / PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH
    base: dict[str, object] = {
        "path": PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH,
        "expected_artifact_sha256": PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256,
    }
    if not path.is_file():
        return {
            **base,
            "exists": False,
            "artifact_sha256": "",
            "load_blockers": ["pr106_packetir_candidate_matrix_missing"],
        }
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    load_blockers: list[str] = []
    if artifact_sha256 != PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256:
        load_blockers.append("l5_v2_packetir_matrix_artifact_sha_mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": [
                *load_blockers,
                f"pr106_packetir_candidate_matrix_json_invalid:{exc.msg}",
            ],
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": [
                *load_blockers,
                "pr106_packetir_candidate_matrix_not_object",
            ],
        }
    existing_blockers = [
        str(blocker) for blocker in payload.get("load_blockers", []) if str(blocker)
    ]
    return {
        **base,
        **payload,
        "exists": True,
        "artifact_sha256": artifact_sha256,
        "load_blockers": list(dict.fromkeys(load_blockers + existing_blockers)),
    }


def _load_l5_v2_section_entropy_matrix() -> dict[str, object]:
    """Load the planning-only L5-v2 PacketIR section entropy matrix."""

    path = REPO_ROOT / L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH
    base: dict[str, object] = {
        "path": L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH,
    }
    if not path.is_file():
        return {
            **base,
            "exists": False,
            "artifact_sha256": "",
            "load_blockers": ["l5_v2_packetir_section_entropy_matrix_missing"],
        }
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": [
                f"l5_v2_packetir_section_entropy_matrix_json_invalid:{exc.msg}"
            ],
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": ["l5_v2_packetir_section_entropy_matrix_not_object"],
        }
    return {
        **base,
        **payload,
        "exists": True,
        "artifact_sha256": artifact_sha256,
        "load_blockers": [
            str(blocker) for blocker in payload.get("matrix_blockers", []) if str(blocker)
        ],
    }


def _load_l5_v2_paired_measurement_dispatch_plan() -> dict[str, object]:
    """Load and sanity-check the planning-only L5-v2 paired dispatch plan."""

    path = REPO_ROOT / L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH
    base: dict[str, object] = {
        "path": L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH,
        "tool_path": L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH,
        "exists": False,
        "artifact_sha256": "",
        "load_blockers": [],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
    }
    if not path.is_file():
        return {
            **base,
            "load_blockers": ["l5_v2_paired_measurement_dispatch_plan_missing"],
        }
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": [
                f"l5_v2_paired_measurement_dispatch_plan_json_invalid:{exc.msg}"
            ],
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "exists": True,
            "artifact_sha256": artifact_sha256,
            "load_blockers": [
                "l5_v2_paired_measurement_dispatch_plan_not_object"
            ],
        }

    load_blockers: list[str] = []
    if payload.get("schema") != L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA:
        load_blockers.append("l5_v2_paired_measurement_dispatch_plan_schema_mismatch")
    for key in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "dispatch_attempted",
    ):
        if payload.get(key) is not False:
            load_blockers.append(
                f"l5_v2_paired_measurement_dispatch_plan_false_authority:{key}"
            )
    if payload.get("planning_only") is not True:
        load_blockers.append(
            "l5_v2_paired_measurement_dispatch_plan_planning_only_not_true"
        )
    if payload.get("paired_dispatch_tool") != "tools/dispatch_modal_paired_auth_eval.py":
        load_blockers.append(
            "l5_v2_paired_measurement_dispatch_plan_paired_tool_not_canonical"
        )
    source_schedule_path = str(
        payload.get("source_schedule_path") or L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH
    )
    source_schedule_sha256 = str(payload.get("source_schedule_sha256") or "")
    current_source_schedule_sha256 = ""
    if not source_schedule_sha256:
        load_blockers.append(
            "l5_v2_paired_measurement_dispatch_plan_source_schedule_sha256_missing"
        )
    source_schedule = Path(source_schedule_path)
    if source_schedule.is_absolute():
        load_blockers.append(
            "l5_v2_paired_measurement_dispatch_plan_source_schedule_path_absolute"
        )
        source_schedule_file = source_schedule
    else:
        source_schedule_file = REPO_ROOT / source_schedule
    if not source_schedule_file.is_file():
        load_blockers.append(
            "l5_v2_paired_measurement_dispatch_plan_source_schedule_missing"
        )
    else:
        current_source_schedule_sha256 = hashlib.sha256(
            source_schedule_file.read_bytes()
        ).hexdigest()
        if (
            source_schedule_sha256
            and source_schedule_sha256 != current_source_schedule_sha256
        ):
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_source_schedule_stale"
            )

    work_units = payload.get("work_units")
    if not isinstance(work_units, list):
        work_units = []
        load_blockers.append("l5_v2_paired_measurement_dispatch_plan_work_units_missing")
    for idx, row in enumerate(work_units):
        if not isinstance(row, dict):
            load_blockers.append(
                f"l5_v2_paired_measurement_dispatch_plan_work_unit_not_object:{idx}"
            )
            continue
        command = " ".join(str(part) for part in row.get("dispatch_command", []))
        measurement_id = row.get("measurement_id")
        if "tools/dispatch_modal_paired_auth_eval.py" not in command:
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_command_not_paired:"
                f"{measurement_id}"
            )
        if "experiments/modal_auth_eval.py" in command:
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_single_axis_cuda_leak:"
                f"{measurement_id}"
            )
        if "experiments/modal_auth_eval_cpu.py" in command:
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_single_axis_cpu_leak:"
                f"{measurement_id}"
            )
        if row.get("dispatch_command_executable") is not False:
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_executable_template:"
                f"{measurement_id}"
            )
        if row.get("standalone_active_claim_command") is not None:
            load_blockers.append(
                "l5_v2_paired_measurement_dispatch_plan_preclaim_leak:"
                f"{measurement_id}"
            )

    command_sample = [
        {
            "measurement_id": row.get("measurement_id"),
            "pair_group_id": row.get("pair_group_id"),
            "dispatch_command_template": row.get("dispatch_command_template"),
            "dispatch_command_executable": row.get("dispatch_command_executable"),
            "ready_for_operator_dispatch": row.get("ready_for_operator_dispatch"),
            "ready_for_provider_dispatch": row.get("ready_for_provider_dispatch"),
            "measurement_blockers_to_close": row.get("measurement_blockers_to_close"),
            "dispatch_blockers": row.get("dispatch_blockers"),
            "readiness_blockers": row.get("readiness_blockers"),
        }
        for row in work_units[:3]
        if isinstance(row, dict)
    ]
    return {
        **base,
        **payload,
        "exists": True,
        "artifact_sha256": artifact_sha256,
        "load_blockers": load_blockers,
        "source_schedule_path": source_schedule_path,
        "source_schedule_sha256": source_schedule_sha256,
        "current_source_schedule_sha256": current_source_schedule_sha256,
        "source_schedule_stale": (
            bool(source_schedule_sha256)
            and bool(current_source_schedule_sha256)
            and source_schedule_sha256 != current_source_schedule_sha256
        ),
        "work_unit_count": int(payload.get("work_unit_count") or 0),
        "ready_work_unit_count": int(payload.get("ready_work_unit_count") or 0),
        "command_sample": command_sample,
    }


def _l5_v2_frontier_readiness(
    dispatch_claim_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read-only L5-v2 frontier status for operator briefing.

    This is intentionally a visibility surface, not a dispatch actuator. The
    TT5L-first staircase action is the primary row; PacketIR/PR106 rows are
    optional stack evidence only. Every launch still needs lane claim,
    axis-specific runtime-tree custody, Modal recovery, and adversarial result
    review.
    """

    sideinfo_evidence = l5_v2_canonical_sideinfo_gate_evidence()
    gate_evidence = [sideinfo_evidence] if sideinfo_evidence is not None else None
    readiness = l5_v2_dispatch_readiness(gate_evidence=gate_evidence)
    architecture_lock_packet = l5_v2_architecture_lock_packet(repo_root=REPO_ROOT)
    atw_v2_gate_status = atw_v2_phase2_gate_status(repo_root=REPO_ROOT)
    matrix = _load_l5_v2_packetir_matrix()
    section_entropy_matrix = _load_l5_v2_section_entropy_matrix()
    paired_measurement_plan = _load_l5_v2_paired_measurement_dispatch_plan()
    matrix_blockers = [
        str(blocker) for blocker in matrix.get("load_blockers", []) if str(blocker)
    ]
    paired_measurement_plan_blockers = [
        str(blocker)
        for blocker in paired_measurement_plan.get("load_blockers", [])
        if str(blocker)
    ]
    if dispatch_claim_summary is None:
        dispatch_claim_summary = _dispatch_claim_summary()
    active_dispatch_claim_count = _active_dispatch_claim_count(dispatch_claim_summary)
    dispatch_blockers = list(matrix_blockers)
    if active_dispatch_claim_count:
        dispatch_blockers.append(
            f"blocked_active_dispatch_claims_present:{active_dispatch_claim_count}"
        )
    matrix_fresh = not matrix_blockers
    dispatch_targets_allowed = matrix_fresh and active_dispatch_claim_count == 0
    targets = [
        target
        for target in (
            matrix.get("next_exact_eval_targets", []) if dispatch_targets_allowed else []
        )
        if isinstance(target, dict)
    ]
    normalized_targets = [
        {
            "candidate_id": target.get("candidate_id"),
            "missing_axes": target.get("missing_axes"),
            "missing_axis": target.get("missing_axis"),
            "recommended_provider": target.get("recommended_provider"),
            "lane_id": target.get("lane_id"),
            "pair_group_id": target.get("pair_group_id"),
            "paired_dispatch_tool": target.get("paired_dispatch_tool"),
            "expected_runtime_tree_sha256_policy": target.get(
                "expected_runtime_tree_sha256_policy"
            ),
            "skip_axis_if_promotable_anchor_exists": target.get(
                "skip_axis_if_promotable_anchor_exists"
            ),
            "command_template": target.get("command_template"),
            "archive_path": target.get("archive_path"),
            "dispatch_status": target.get("dispatch_status"),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        for target in targets
    ]
    summarized_targets = normalized_targets[:5]
    blockers = [str(blocker) for blocker in readiness.get("blockers", [])]
    for section in ("packetir_stack_evidence", "pr106_stack_cell_candidates"):
        section_payload = readiness.get(section)
        if isinstance(section_payload, dict):
            blockers.extend(
                str(blocker)
                for blocker in section_payload.get("blockers", [])
                if str(blocker)
            )
    blockers.extend(dispatch_blockers)
    blockers.extend(paired_measurement_plan_blockers)
    blockers.extend(
        str(blocker)
        for blocker in paired_measurement_plan.get("blockers", [])
        if str(blocker)
    )
    target_count = len(normalized_targets)
    status_counts = matrix.get("status_counts") if isinstance(matrix, dict) else {}
    if not isinstance(status_counts, dict):
        status_counts = {}
    section_entropy_best = section_entropy_matrix.get("best_rate_positive_prototype")
    if not isinstance(section_entropy_best, dict):
        section_entropy_best = None
    section_entropy_best_adaptive = section_entropy_matrix.get("best_adaptive_prototype")
    if not isinstance(section_entropy_best_adaptive, dict):
        section_entropy_best_adaptive = None
    section_entropy_best_rate_positive_adaptive = section_entropy_matrix.get(
        "best_rate_positive_adaptive_prototype"
    )
    if not isinstance(section_entropy_best_rate_positive_adaptive, dict):
        section_entropy_best_rate_positive_adaptive = None
    section_entropy_best_derived_prefix_adaptive = section_entropy_matrix.get(
        "best_derived_prefix_adaptive_prototype"
    )
    if not isinstance(section_entropy_best_derived_prefix_adaptive, dict):
        section_entropy_best_derived_prefix_adaptive = None
    section_entropy_best_rate_positive_derived_prefix_adaptive = (
        section_entropy_matrix.get(
            "best_rate_positive_derived_prefix_adaptive_prototype"
        )
    )
    if not isinstance(
        section_entropy_best_rate_positive_derived_prefix_adaptive,
        dict,
    ):
        section_entropy_best_rate_positive_derived_prefix_adaptive = None
    tt5l_campaign = readiness.get("tt5l_campaign_readiness")
    if not isinstance(tt5l_campaign, dict):
        tt5l_campaign = {}
    asymptotic_payload = readiness.get("asymptotic_pursuit_candidates")
    if not isinstance(asymptotic_payload, dict):
        asymptotic_payload = {}
    asymptotic_candidates = [
        candidate
        for candidate in asymptotic_payload.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    asymptotic_next_action_status = [
        status
        for status in asymptotic_payload.get(
            "l5_v2_asymptotic_next_action_status",
            [],
        )
        if isinstance(status, dict)
    ]
    architecture_lock_packet_blockers = architecture_lock_packet.get(
        "architecture_lock_blockers",
    )
    if not isinstance(architecture_lock_packet_blockers, list):
        architecture_lock_packet_blockers = []
    next_non_pr106_l5_action = tt5l_campaign.get("next_non_pr106_l5_action")
    if not isinstance(next_non_pr106_l5_action, dict):
        next_non_pr106_l5_action = {}
    measurement_schedule_tool_path = str(
        tt5l_campaign.get("measurement_schedule_tool_path") or ""
    )
    measurement_schedule_artifact_path = str(
        tt5l_campaign.get("measurement_schedule_artifact_path") or ""
    )
    measurement_schedule_report_path = str(
        tt5l_campaign.get("measurement_schedule_report_path") or ""
    )
    return {
        "schema": "pact.l5_v2_frontier_readiness.v1",
        "subject_id": "time_traveler_l5_autonomy",
        "primary_staircase": "tt5l_first_non_pr106_l5_v2",
        "tt5l_campaign_readiness": tt5l_campaign,
        "next_non_pr106_l5_action": next_non_pr106_l5_action,
        "asymptotic_pursuit_candidate_count": len(asymptotic_candidates),
        "asymptotic_pursuit_candidates": asymptotic_candidates,
        "asymptotic_pursuit_candidate_sample": asymptotic_candidates[:3],
        "l5_v2_asymptotic_next_action_status": asymptotic_next_action_status,
        "measurement_schedule_tool_path": measurement_schedule_tool_path,
        "measurement_schedule_artifact_path": measurement_schedule_artifact_path,
        "measurement_schedule_report_path": measurement_schedule_report_path,
        "measurement_schedule_score_claim": False,
        "measurement_schedule_promotion_eligible": False,
        "measurement_schedule_ready_for_exact_eval_dispatch": False,
        "paired_measurement_dispatch_plan_tool_path": (
            paired_measurement_plan.get("tool_path", "")
        ),
        "paired_measurement_dispatch_plan_artifact_path": (
            paired_measurement_plan.get("path", "")
        ),
        "paired_measurement_dispatch_plan_exists": (
            paired_measurement_plan.get("exists") is True
        ),
        "paired_measurement_dispatch_plan_artifact_sha256": (
            paired_measurement_plan.get("artifact_sha256", "")
        ),
        "paired_measurement_dispatch_plan_source_schedule_sha256": (
            paired_measurement_plan.get("source_schedule_sha256", "")
        ),
        "paired_measurement_dispatch_plan_current_source_schedule_sha256": (
            paired_measurement_plan.get("current_source_schedule_sha256", "")
        ),
        "paired_measurement_dispatch_plan_source_schedule_stale": bool(
            paired_measurement_plan.get("source_schedule_stale")
        ),
        "paired_measurement_dispatch_plan_work_unit_count": int(
            paired_measurement_plan.get("work_unit_count") or 0
        ),
        "paired_measurement_dispatch_plan_ready_work_unit_count": int(
            paired_measurement_plan.get("ready_work_unit_count") or 0
        ),
        "paired_measurement_dispatch_plan_command_sample": (
            paired_measurement_plan.get("command_sample", [])
        ),
        "paired_measurement_dispatch_plan_score_claim": False,
        "paired_measurement_dispatch_plan_score_claim_valid": False,
        "paired_measurement_dispatch_plan_promotion_eligible": False,
        "paired_measurement_dispatch_plan_ready_for_exact_eval_dispatch": False,
        "paired_measurement_dispatch_plan_rank_or_kill_eligible": False,
        "paired_measurement_dispatch_plan_dispatch_attempted": False,
        "tt5l_sideinfo_effect_curve_allowed": bool(
            tt5l_campaign.get("sideinfo_effect_curve_allowed")
        ),
        "tt5l_sideinfo_effect_curve_artifact_valid": bool(
            tt5l_campaign.get("sideinfo_effect_curve_artifact_valid")
        ),
        "tt5l_architecture_lock_allowed": bool(
            tt5l_campaign.get("architecture_lock_allowed")
        ),
        "architecture_lock_packet_artifact_path": (
            L5_V2_ARCHITECTURE_LOCK_PACKET_ARTIFACT_PATH
        ),
        "architecture_lock_packet_report_path": (
            L5_V2_ARCHITECTURE_LOCK_PACKET_REPORT_PATH
        ),
        "architecture_lock_packet_allowed": bool(
            architecture_lock_packet.get("architecture_lock_allowed")
        ),
        "architecture_lock_packet_blockers": architecture_lock_packet_blockers,
        "atw_v2_phase2_gate_status": atw_v2_gate_status,
        "atw_v2_phase2_d4_verdict": atw_v2_gate_status.get("d4_verdict"),
        "atw_v2_phase2_dispatch_allowed": bool(
            atw_v2_gate_status.get("dispatch_allowed")
        ),
        "atw_v2_phase2_lift_allowed": bool(
            atw_v2_gate_status.get("phase2_lift_allowed")
        ),
        "atw_v2_phase2_next_action": atw_v2_gate_status.get("next_action"),
        "tt5l_first_anchor_timing_smoke_allowed": bool(
            tt5l_campaign.get("first_anchor_timing_smoke_allowed")
        ),
        "packetir_matrix_path": PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH,
        "packetir_matrix_exists": matrix.get("exists") is True,
        "packetir_matrix_artifact_sha256": matrix.get("artifact_sha256", ""),
        "packetir_matrix_expected_sha256": matrix.get("expected_artifact_sha256", ""),
        "packetir_matrix_dispatch_targets_suppressed": bool(dispatch_blockers),
        "packetir_section_entropy_matrix_path": section_entropy_matrix.get("path", ""),
        "packetir_section_entropy_matrix_exists": (
            section_entropy_matrix.get("exists") is True
        ),
        "packetir_section_entropy_matrix_artifact_sha256": (
            section_entropy_matrix.get("artifact_sha256", "")
        ),
        "packetir_section_entropy_profiled_candidate_count": int(
            section_entropy_matrix.get("profiled_candidate_count") or 0
        ),
        "packetir_section_entropy_prototype_row_count": int(
            section_entropy_matrix.get("prototype_row_count") or 0
        ),
        "packetir_section_entropy_rate_positive_prototype_row_count": int(
            section_entropy_matrix.get("rate_positive_prototype_row_count") or 0
        ),
        "packetir_section_entropy_best_rate_positive_prototype": section_entropy_best,
        "packetir_section_entropy_adaptive_prototype_row_count": int(
            section_entropy_matrix.get("adaptive_prototype_row_count") or 0
        ),
        "packetir_section_entropy_rate_positive_adaptive_prototype_row_count": int(
            section_entropy_matrix.get("rate_positive_adaptive_prototype_row_count")
            or 0
        ),
        "packetir_section_entropy_best_adaptive_prototype": (
            section_entropy_best_adaptive
        ),
        "packetir_section_entropy_best_rate_positive_adaptive_prototype": (
            section_entropy_best_rate_positive_adaptive
        ),
        "packetir_section_entropy_derived_prefix_adaptive_prototype_row_count": int(
            section_entropy_matrix.get("derived_prefix_adaptive_prototype_row_count")
            or 0
        ),
        "packetir_section_entropy_rate_positive_derived_prefix_adaptive_prototype_row_count": int(
            section_entropy_matrix.get(
                "rate_positive_derived_prefix_adaptive_prototype_row_count"
            )
            or 0
        ),
        "packetir_section_entropy_best_derived_prefix_adaptive_prototype": (
            section_entropy_best_derived_prefix_adaptive
        ),
        "packetir_section_entropy_best_rate_positive_derived_prefix_adaptive_prototype": (
            section_entropy_best_rate_positive_derived_prefix_adaptive
        ),
        "active_dispatch_claim_count": active_dispatch_claim_count,
        "dispatch_claim_gate_blocked": active_dispatch_claim_count > 0,
        "packetir_candidate_count": int(matrix.get("candidate_count") or 0),
        "packetir_status_counts": status_counts,
        "next_exact_eval_target_count": target_count,
        "next_exact_eval_targets": normalized_targets,
        "next_exact_eval_targets_sample": summarized_targets,
        "canonical_sideinfo_evidence_present": sideinfo_evidence is not None,
        "l5_ready_for_gate_probe_dispatch": bool(
            readiness.get("ready_for_gate_probe_dispatch")
        ),
        "l5_ready_for_score_or_rank_dispatch": bool(
            readiness.get("ready_for_score_or_rank_dispatch")
        ),
        "l5_ready_for_dispatch": bool(readiness.get("ready_for_dispatch")),
        "pr106_stack_cell_candidate_count": int(
            readiness.get("pr106_stack_cell_candidates", {}).get("candidate_count", 0)
            if isinstance(readiness.get("pr106_stack_cell_candidates"), dict)
            else 0
        ),
        "packetir_paired_candidate_count": int(
            readiness.get("packetir_stack_evidence", {}).get("paired_candidate_count", 0)
            if isinstance(readiness.get("packetir_stack_evidence"), dict)
            else 0
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "target_rows_are_fail_fast_only": True,
        "blockers": blockers,
        "recommendation": (
            "Advance the TT5L-first L5-v2 staircase action before optional "
            "PacketIR/PR106 stack evidence; no score/rank claim from this "
            "briefing surface."
        ),
    }


def _format_l5_v2_frontier_readiness() -> str:
    payload = _l5_v2_frontier_readiness()
    next_action = payload.get("next_non_pr106_l5_action")
    if not isinstance(next_action, dict):
        next_action = {}
    lines = [
        "L5-v2 frontier status (TT5L-first; read-only; no score claim):",
        f"  primary staircase:             {payload['primary_staircase']}",
        "  next non-PR106 L5 action:      "
        f"{next_action.get('action_id', 'missing')}",
        f"  TT5L side-info curve allowed:   {payload['tt5l_sideinfo_effect_curve_allowed']}",
        "  TT5L side-info curve artifact:  "
        f"{payload['tt5l_sideinfo_effect_curve_artifact_valid']}",
        f"  TT5L architecture lock allowed: {payload['tt5l_architecture_lock_allowed']}",
        f"  architecture lock packet:       {payload['architecture_lock_packet_artifact_path']}",
        f"  architecture lock packet ok:    {payload['architecture_lock_packet_allowed']}",
        f"  ATW v2 D4 verdict:              {payload['atw_v2_phase2_d4_verdict']}",
        f"  ATW v2 Phase-2 lift allowed:    {payload['atw_v2_phase2_lift_allowed']}",
        f"  TT5L timing smoke allowed:      {payload['tt5l_first_anchor_timing_smoke_allowed']}",
        f"  side-info proof present:        {payload['canonical_sideinfo_evidence_present']}",
        f"  L5 gate-probe dispatch ready:   {payload['l5_ready_for_gate_probe_dispatch']}",
        f"  L5 score/rank dispatch ready:   {payload['l5_ready_for_score_or_rank_dispatch']}",
        f"  exact dispatch authority:        {payload['ready_for_exact_eval_dispatch']}",
        f"  measurement schedule:           {payload['measurement_schedule_artifact_path']}",
        f"  measurement schedule tool:      {payload['measurement_schedule_tool_path']}",
        "  paired measurement plan:        "
        f"{payload['paired_measurement_dispatch_plan_artifact_path']}",
        "  paired measurement work units:  "
        f"{payload['paired_measurement_dispatch_plan_work_unit_count']}",
        "  paired measurement ready units: "
        f"{payload['paired_measurement_dispatch_plan_ready_work_unit_count']}",
        f"  optional PacketIR matrix:        {payload['packetir_matrix_path']}",
        f"  PacketIR candidates:             {payload['packetir_candidate_count']}",
        f"  PacketIR status counts:          {payload['packetir_status_counts']}",
        f"  section entropy matrix:          {payload['packetir_section_entropy_matrix_path']}",
        f"  charged prototype rows:          {payload['packetir_section_entropy_prototype_row_count']}",
        f"  rate-positive prototypes:        {payload['packetir_section_entropy_rate_positive_prototype_row_count']}",
        f"  adaptive prototype rows:         {payload['packetir_section_entropy_adaptive_prototype_row_count']}",
        f"  adaptive rate-positive rows:     {payload['packetir_section_entropy_rate_positive_adaptive_prototype_row_count']}",
        f"  derived-prefix adaptive rows:    {payload['packetir_section_entropy_derived_prefix_adaptive_prototype_row_count']}",
        "  derived-prefix rate-positive:    "
        f"{payload['packetir_section_entropy_rate_positive_derived_prefix_adaptive_prototype_row_count']}",
        f"  runtime-bound paired candidates: {payload['packetir_paired_candidate_count']}",
        f"  stack-cell candidates:           {payload['pr106_stack_cell_candidate_count']}",
        "  asymptotic candidate count:      "
        f"{payload['asymptotic_pursuit_candidate_count']}",
        f"  next exact-eval targets:         {payload['next_exact_eval_target_count']}",
        "",
        "  Asymptotic candidates:",
    ]
    asymptotic_sample = payload.get("asymptotic_pursuit_candidate_sample")
    if isinstance(asymptotic_sample, list) and asymptotic_sample:
        for candidate in asymptotic_sample:
            if not isinstance(candidate, dict):
                continue
            lines.append(
                "    - "
                f"{candidate.get('candidate_id')} -> "
                f"{candidate.get('effective_recommended_next_action_id') or candidate.get('recommended_next_action_id')} "
                f"[{candidate.get('recommended_next_action_status', 'pending')}] "
                f"(l1_present={candidate.get('l1_scaffold_present')}, "
                f"artifacts_all_present="
                f"{candidate.get('expected_first_artifacts_all_present')}, "
                f"l1_blockers={candidate.get('l1_build_blockers', [])})"
            )
            status = candidate.get("l5_v2_asymptotic_next_action_status")
            if isinstance(status, dict):
                next_prerequisite = status.get("next_prerequisite_status")
                if not isinstance(next_prerequisite, dict):
                    next_prerequisite = {}
                lines.append(
                    "      next-status: "
                    f"ledger={status.get('ledger_present')} "
                    f"registry={status.get('lane_registry_registered')} "
                    f"prereq={next_prerequisite.get('status')}"
                )
    else:
        lines.append("    - none")
    lines.extend(
        [
            "",
            "  Sample fail-fast targets:",
        ]
    )
    sample = payload.get("next_exact_eval_targets_sample")
    if isinstance(sample, list) and sample:
        for target in sample:
            if not isinstance(target, dict):
                continue
            lines.append(
                "    - "
                f"{target.get('lane_id')} [{target.get('missing_axis')}] "
                f"via {target.get('recommended_provider')} — "
                f"{target.get('dispatch_status')}"
            )
    else:
        lines.append("    - none")
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.append("")
        lines.append("  First blockers:")
        for blocker in blockers[:8]:
            lines.append(f"    - {blocker}")
    lines.append("")
    lines.append(f"Recommendation: {payload['recommendation']}")
    return "\n".join(lines)


# Phase-6 xray toolkit: surface the diagnostic tools landed 2026-05-09 so
# operators discover them without having to grep `tools/xray_*`. These are
# pure-CPU diagnostic tools — no GPU, no dispatch, no score claims.
XRAY_TOOLKIT = [
    {
        "tool": "tools/xray_archive_section_entropy_heatmap.py",
        "purpose": (
            "Per-section entropy density vs encoded-bytes — reveals which "
            "archive sections are saturated vs have recoverable headroom. "
            "Use BEFORE planning a new entropy-coder lane."
        ),
        "example": (
            ".venv/bin/python tools/xray_archive_section_entropy_heatmap.py \\\n"
            "  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\\n"
            "  --label pr101_gold"
        ),
    },
    {
        "tool": "tools/xray_per_pr_archive_layout_compare.py",
        "purpose": (
            "Side-by-side byte-section layout for N archives — surfaces "
            "SHARED structure vs DIVERGED bytes. Use to understand 'what "
            "bytes did SajayR/PR101 change vs rem2/PR103?'"
        ),
        "example": (
            ".venv/bin/python tools/xray_per_pr_archive_layout_compare.py \\\n"
            "  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\\n"
            "  --archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\\n"
            "  --label pr101 --label pr106"
        ),
    },
    {
        "tool": "tools/xray_per_tensor_saliency_heatmap.py",
        "purpose": (
            "Per-tensor saliency-vs-bytes — flags over-allocated tensors as "
            "next coarsening targets. Consumes a saliency dict from "
            "tac.score_gradient_param_saliency."
        ),
        "example": (
            ".venv/bin/python tools/xray_per_tensor_saliency_heatmap.py \\\n"
            "  --saliency-json reports/saliency_a1.json \\\n"
            "  --byte-map-json reports/per_tensor_bytes_a1.json \\\n"
            "  --bottom-n-percent 25"
        ),
    },
    {
        "tool": "tools/xray_inflate_op_cost_profiler.py",
        "purpose": (
            "Static AST-based op-cost catalog of inflate.py — finds "
            "per-channel mutation lines (the PR101→PR103 medal-delta "
            "pattern). Use BEFORE designing a new sidecar/inflate edit."
        ),
        "example": (
            ".venv/bin/python tools/xray_inflate_op_cost_profiler.py \\\n"
            "  --inflate-py experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py \\\n"
            "  --label pr101_canonical"
        ),
    },
    {
        "tool": "tools/xray_cpu_cuda_drift_per_arch_class.py",
        "purpose": (
            "Predict CPU score from CUDA score using per-architecture-class "
            "drift profile (HNeRV cluster R_pose=5.04). Use to decide "
            "WHETHER to spend on a [contest-CPU GHA] eval before paying."
        ),
        "example": (
            ".venv/bin/python tools/xray_cpu_cuda_drift_per_arch_class.py \\\n"
            "  --archive experiments/results/track4_sg_a1_t178000_20260509/archive.zip \\\n"
            "  --cuda-auth-eval-json experiments/results/.../contest_auth_eval.json \\\n"
            "  --label pr107_apogee"
        ),
    },
    {
        "tool": "tools/xray_paired_cpu_cuda_axis_delta.py",
        "purpose": (
            "Compare exact CPU and CUDA auth-eval artifacts for the same "
            "archive — quantifies component deltas, byte-equivalent axis gap, "
            "and whether rate-only polishing can plausibly fix the miss. Use "
            "AFTER both axes exist."
        ),
        "example": (
            ".venv/bin/python tools/xray_paired_cpu_cuda_axis_delta.py \\\n"
            "  --cpu-auth-eval-json experiments/results/.../cpu/contest_auth_eval.json \\\n"
            "  --cuda-auth-eval-json experiments/results/.../cuda/contest_auth_eval.json \\\n"
            "  --label candidate"
        ),
    },
    {
        "tool": "tools/xray_pair_component_errors.py",
        "purpose": (
            "Per-pair PoseNet/SegNet and frame0/frame1 pixel-error tails from "
            "an inflated raw output. Use before selector, film-grain, foveation, "
            "or repair work so component-moving edits target hard pairs instead "
            "of aggregate-score anecdotes."
        ),
        "example": (
            ".venv/bin/python tools/xray_pair_component_errors.py \\\n"
            "  --inflated-dir experiments/results/.../work/inflated \\\n"
            "  --device cpu \\\n"
            "  --label candidate \\\n"
            "  --output-dir experiments/results/pair_component_xray_candidate"
        ),
    },
    {
        "tool": "tools/xray_hardpair_hitlist.py",
        "purpose": (
            "Deterministic hard-pair priority list from pair-component XRay "
            "and paired CPU/CUDA axis deltas. Use to turn XRay diagnostics into "
            "selector, film-grain, foveation, and latent-repair hitlists while "
            "preserving false-authority status."
        ),
        "example": (
            ".venv/bin/python tools/xray_hardpair_hitlist.py \\\n"
            "  --pair-xray-json experiments/results/.../pair_component_xray.json \\\n"
            "  --paired-axis-artifact experiments/results/.../paired_axis_delta.md \\\n"
            "  --label candidate_hardpairs \\\n"
            "  --output-dir experiments/results/hardpair_hitlist_candidate"
        ),
    },
    {
        "tool": "tools/xray_substrate_classifier.py",
        "purpose": (
            "Classify archive payload grammar from magic bytes and layout "
            "signals, including cooperative-receiver, S2SBS/SABOR, and "
            "magic-codec packet classes. Use before routing an archive into "
            "a compiler or repack lane."
        ),
        "example": (
            ".venv/bin/python tools/xray_substrate_classifier.py \\\n"
            "  --archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\\n"
            "  --label pr106"
        ),
    },
    {
        "tool": "tools/xray_per_frame_difficulty_profile.py",
        "purpose": (
            "Per-frame difficulty and byte-allocation profile for finding "
            "hard frames, stable interiors, and candidate frame-local "
            "sidecar targets before a paid training or exact-eval pass."
        ),
        "example": (
            ".venv/bin/python tools/xray_per_frame_difficulty_profile.py \\\n"
            "  --auth-eval-json experiments/results/.../contest_auth_eval.json \\\n"
            "  --label candidate"
        ),
    },
    {
        "tool": "tools/master_gradient_xray.py",
        "purpose": (
            "Master-gradient anchor visualization — 5 canonical plot types "
            "(per-pair distribution, per-byte heatmap, cumulative-by-rank, "
            "cross-substrate correlation, Wyner-Ziv layer flow) + optional "
            "drift-vs-sensitivity scatter when --mps-drift-json is provided. "
            "Per Catalog #305 observability + Catalog #323 canonical "
            "Provenance: emits sister JSON sidecars + index.html landing "
            "page. Use to decide which downstream master-gradient consumer "
            "route to invoke for which archive. Lane "
            "lane_master_gradient_xray_viz_tool_20260519. Operator usage: "
            "docs/master_gradient_xray_usage.md."
        ),
        "example": (
            ".venv/bin/python tools/master_gradient_xray.py \\\n"
            "  --archive-sha 87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5 \\\n"
            "  --mps-drift-json .omx/state/mps_drift_granular_20260519T122700Z.json \\\n"
            "  --output-dir reports/master_gradient_xray/a1_baseline/"
        ),
    },
]


def _xray_toolkit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in XRAY_TOOLKIT:
        tool_path = REPO_ROOT / str(entry["tool"])
        rows.append(
            apply_false_authority_contract(
                {
                **entry,
                "tool_exists": tool_path.is_file(),
                "dispatch_blockers": ["diagnostic_tool_no_score_or_dispatch_authority"],
                },
                preserve_dispatch_ready=False,
                reason="xray_diagnostic_tool_no_score_or_dispatch_authority",
            )
        )
    return rows


def _format_xray_toolkit() -> str:
    lines = []
    for entry in _xray_toolkit_rows():
        exists = "present" if entry["tool_exists"] else "MISSING"
        lines.append(f"• `{entry['tool']}`")
        lines.append(
            "    "
            f"status: {exists}; score_claim=false; "
            "ready_for_exact_eval_dispatch=false"
        )
        lines.append(f"    {entry['purpose']}")
        lines.append("    Example:")
        for ln in entry["example"].splitlines():
            lines.append(f"      {ln}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _cooperative_receiver_solver_integration() -> dict[str, object]:
    """Return operator-visible hook counts for the unified solver stack.

    This is a briefing surface, not a dispatcher. It intentionally keeps every
    hook fail-closed so JSON consumers cannot turn research/planning manifests
    into score authority by omission.
    """

    try:
        from tac.optimization.cooperative_receiver_integration import (
            build_integration_manifest,
        )

        manifest = build_integration_manifest()
    except Exception as exc:
        return apply_false_authority_contract({
            "status": "BLOCKED",
            "reason": f"cooperative receiver integration manifest failed: {exc}",
            "dispatch_blockers": ["integration_manifest_build_failed"],
        }, preserve_dispatch_ready=False)

    autopilot_rows = manifest.get("autopilot_dispatch_hook", {}).get("rows") or []
    meta_rows = manifest.get("meta_lagrangian_hook", {}).get("rows") or []
    pareto_rows = manifest.get("pareto_constraint_hook", {}).get("rows") or []
    xray_rows = manifest.get("xray_hook", {}).get("cooperative_receiver_packet_grammars") or []
    magic_rows = manifest.get("magic_codec_hook", {}).get("entries") or []
    packet_grammars = (
        manifest.get("packet_compiler_hook", {}).get("cooperative_receiver_packet_grammars")
        or []
    )
    continual = manifest.get("continual_learning_hook", {})
    hooks_present = {
        "autopilot_dispatch_hook": bool(autopilot_rows),
        "meta_lagrangian_hook": bool(meta_rows),
        "pareto_constraint_hook": bool(pareto_rows),
        "continual_learning_hook": isinstance(continual, dict) and bool(continual),
        "xray_hook": bool(xray_rows),
        "magic_codec_hook": bool(magic_rows),
        "packet_compiler_hook": bool(packet_grammars),
    }
    missing_hooks = [name for name, present in hooks_present.items() if not present]
    return apply_false_authority_contract({
        "status": "READY" if not missing_hooks else "BLOCKED",
        "schema": manifest.get("schema"),
        "planning_only": manifest.get("planning_only") is True,
        "dispatch_blockers": ["planning_only_requires_byte_closed_exact_eval"],
        "missing_hooks": missing_hooks,
        "campaign_count": manifest.get("campaign_count"),
        "autopilot_rows": len(autopilot_rows),
        "meta_lagrangian_rows": len(meta_rows),
        "pareto_rows": len(pareto_rows),
        "continual_learning_posterior_update_allowed": bool(
            isinstance(continual, dict)
            and continual.get("posterior_update_allowed") is True
        ),
        "xray_grammars": len(xray_rows),
        "magic_codec_entries": len(magic_rows),
        "packet_compiler_grammars": len(packet_grammars),
        "canonical_packet_compiler": manifest.get("packet_compiler_hook", {}).get(
            "canonical_module"
        ),
    }, preserve_dispatch_ready=False)


def _format_cooperative_receiver_solver_integration() -> str:
    payload = _cooperative_receiver_solver_integration()
    lines = [
        f"status: {payload['status']}",
        f"score_claim: {payload['score_claim']}",
        f"ready_for_exact_eval_dispatch: {payload['ready_for_exact_eval_dispatch']}",
    ]
    if payload.get("reason"):
        lines.append(f"reason: {payload['reason']}")
    if payload.get("missing_hooks"):
        lines.append("missing_hooks: " + ", ".join(str(v) for v in payload["missing_hooks"]))
    lines.extend(
        [
            f"campaigns: {payload.get('campaign_count', 0)}",
            f"autopilot_rows: {payload.get('autopilot_rows', 0)}",
            f"meta_lagrangian_rows: {payload.get('meta_lagrangian_rows', 0)}",
            f"pareto_rows: {payload.get('pareto_rows', 0)}",
            "continual_learning_posterior_update_allowed: "
            f"{payload.get('continual_learning_posterior_update_allowed')}",
            f"xray_grammars: {payload.get('xray_grammars', 0)}",
            f"magic_codec_entries: {payload.get('magic_codec_entries', 0)}",
            f"packet_compiler_grammars: {payload.get('packet_compiler_grammars', 0)}",
            f"canonical_packet_compiler: {payload.get('canonical_packet_compiler', '')}",
        ]
    )
    return "\n".join(lines)


# Phase-7: constrained-coord-search status (added 2026-05-09).
# Surfaces the lane_pr101_bias_constrained_coord_search (sister subagent
# a8522fca) rollup so operators can see the 64-variant grid status without
# having to grep experiments/results/.
def _format_constrained_coord_search_status() -> str:
    """Locate latest constrained_coord_search rollup; format key metadata."""
    import json as _json

    glob_root = REPO_ROOT / "experiments" / "results"
    candidates = sorted(
        glob_root.glob("constrained_coord_search_pr101_bias_*/rollup.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ("(no constrained_coord_search rollups found yet — "
                "sister subagent a8522fca has not landed grid)")
    latest = candidates[0]
    try:
        data = _json.loads(latest.read_text())
    except Exception as exc:
        return f"(rollup parse failed: {exc} at {latest.relative_to(REPO_ROOT)})"
    rel = latest.relative_to(REPO_ROOT)
    lines = [
        f"Rollup: {rel}",
        f"  lane_id:                {data.get('lane_id', '<unknown>')}",
        f"  n_variants:             {data.get('n_variants', 0)}",
        f"  n_unique_inflates:      {data.get('n_unique_inflates', 0)}",
        f"  evidence_grade:         {data.get('evidence_grade', '<unset>')}",
        f"  build_timestamp_utc:    {data.get('build_timestamp_utc', '<unset>')}",
    ]
    anchor = data.get("regression_anchor") or {}
    if anchor:
        lines.append(f"  regression_anchor:      {anchor.get('value', '<unset>')} "
                     f"({anchor.get('tag', '<no-tag>')})")
        lines.append(f"  regression_delta:       {anchor.get('delta_vs_baseline', '<unset>')}")
    blockers = data.get("dispatch_blockers") or []
    if blockers:
        lines.append(f"  dispatch_blockers ({len(blockers)}):")
        for b in blockers:
            lines.append(f"    - {b}")
    grid_keys = list((data.get("grid") or {}).keys())
    if grid_keys:
        lines.append(f"  grid_dims:              {', '.join(grid_keys)}")
    return "\n".join(lines)


def _constrained_coord_search_readiness() -> dict[str, object]:
    """Classify Phase 7 without promoting prediction-only rollups to READY."""

    glob_root = REPO_ROOT / "experiments" / "results"
    candidates = sorted(
        glob_root.glob("constrained_coord_search_pr101_bias_*/rollup.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {
            "status": "PENDING",
            "reason": "sister subagent a8522fca grid not landed",
            "n_rollups": 0,
        }
    latest = candidates[0]
    rel = latest.relative_to(REPO_ROOT).as_posix()
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "reason": f"latest rollup parse failed: {type(exc).__name__} at {rel}",
            "n_rollups": len(candidates),
            "latest_rollup": rel,
        }
    blockers = data.get("dispatch_blockers")
    blocker_list = blockers if isinstance(blockers, list) else []
    evidence_grade = str(data.get("evidence_grade") or "").strip()
    evidence_lower = evidence_grade.lower()
    if blocker_list:
        return {
            "status": "BLOCKED",
            "reason": (
                f"{len(blocker_list)} dispatch blocker(s); latest rollup is not dispatch-ready"
            ),
            "n_rollups": len(candidates),
            "latest_rollup": rel,
            "evidence_grade": evidence_grade,
            "dispatch_blockers": [str(blocker) for blocker in blocker_list],
        }
    if "predicted" in evidence_lower or "proxy" in evidence_lower:
        return {
            "status": "PENDING",
            "reason": "prediction-only rollup; run M5/GHA or exact custody before dispatch",
            "n_rollups": len(candidates),
            "latest_rollup": rel,
            "evidence_grade": evidence_grade,
        }
    return {
        "status": "READY",
        "reason": f"{len(candidates)} rollup(s); top-5 to GHA next",
        "n_rollups": len(candidates),
        "latest_rollup": rel,
        "evidence_grade": evidence_grade,
    }


# Phase-8: dispatch readiness per phase 1-5+ (added 2026-05-09).
# Compact roll-up of which phases have actionable next-step output, so the
# operator can navigate "what should I dispatch next" without re-reading the
# whole briefing.
def _packet_is_terminal(packet: dict[str, object]) -> bool:
    return bool(packet.get("terminal_exact_eval_evidence_blockers")) or (
        packet.get("dispatch_action") == "terminal_exact_eval_evidence_stop"
    )


def _dispatch_readiness() -> dict[str, object]:
    """Structured per-phase dispatch readiness used by JSON and human output."""
    exact_ready_audit = _exact_ready_queue_audit()
    exact_ready_repairs = plan_exact_ready_score_axis_repairs_from_audit(
        exact_ready_audit
    )
    materializer_handoffs = _materializer_exact_ready_handoff_summary()
    stale_rows = int(exact_ready_audit.get("stale_ready_row_count") or 0)
    exact_packets = _exact_eval_packet_summaries()
    ready_packets = [
        packet
        for packet in exact_packets
        if packet.get("ready_for_submit") is True
        and not _packet_is_terminal(packet)
    ]
    terminal_packets = [
        packet
        for packet in exact_packets
        if _packet_is_terminal(packet)
    ]
    blocked_packets = [
        packet
        for packet in exact_packets
        if packet.get("ready_for_submit") is not True
        or packet.get("terminal_exact_eval_evidence_blockers")
    ]
    if stale_rows:
        phase1_status = "BLOCKED"
        phase1_reason = f"{stale_rows} exact-ready queue row(s) carry blockers"
    elif ready_packets:
        phase1_status = "READY"
        phase1_reason = f"{len(ready_packets)} exact-eval packet(s) ready"
    elif exact_packets and len(blocked_packets) == len(exact_packets):
        phase1_status = "BLOCKED"
        phase1_reason = "all exact-eval packets are blocked or terminal"
    else:
        phase1_status = "PENDING"
        phase1_reason = "no exact-eval packet is ready"
    phase1 = {
        "status": phase1_status,
        "reason": phase1_reason,
        "n_exact_eval_packets": len(exact_packets),
        "n_ready_exact_eval_packets": len(ready_packets),
        "n_terminal_exact_eval_packets": len(terminal_packets),
        "n_blocked_exact_eval_packets": len(blocked_packets),
        "stale_ready_row_count": stale_rows,
    }
    if stale_rows:
        queue_status = "BLOCKED"
        queue_reason = f"{stale_rows} row(s) carry terminal/live-custody blockers"
    else:
        queue_status = "READY"
        queue_reason = "no terminal/live-custody blockers"
    # Phase 4 / Phase 5: gated lanes — read PHASE_4_GATED_LANES count
    try:
        n_gated = len(PHASE_4_GATED_LANES)
    except NameError:
        n_gated = 0
    try:
        n_comp = len(PHASE_5_COMPOSITION_LANES)
    except NameError:
        n_comp = 0
    phase7 = _constrained_coord_search_readiness()
    byte_shaving_acquisition = _byte_shaving_acquisition_summary()
    frontier_feedback_cycle = _frontier_feedback_cycle_summary()
    pr95_mlx_profiles = _pr95_mlx_control_profile_summary()
    return {
        "schema": "pact.operator_dispatch_readiness.v1",
        "phase_1_exact_eval_packets": phase1,
        "phase_1_exact_ready_queue_hygiene": {
            "status": queue_status,
            "reason": queue_reason,
            "stale_ready_row_count": stale_rows,
        },
        "phase_1_exact_ready_score_axis_repairs": {
            "status": "REPAIRABLE"
            if int(exact_ready_repairs.get("repairable_or_repaired_count") or 0)
            else "NO_AXIS_ONLY_REPAIR",
            "reason": (
                f"{exact_ready_repairs.get('repairable_or_repaired_count')} row(s) "
                "are score-axis-only repair candidates"
            )
            if int(exact_ready_repairs.get("repairable_or_repaired_count") or 0)
            else "no unresolved exact-ready row is axis-only repairable",
            "repairable_or_repaired_count": exact_ready_repairs.get(
                "repairable_or_repaired_count"
            ),
        },
        "phase_1_materializer_exact_ready_handoffs": materializer_handoffs,
        "phase_4_gated_next_tick": {
            "status": "GATED",
            "n_lanes": n_gated,
            "reason": "check entry conditions before dispatch",
        },
        "phase_5_meta_composition": {
            "status": "TRACKED",
            "n_stacks": n_comp,
            "reason": "compose-stacks tracked",
        },
        "phase_7_constrained_coord_search": {
            **phase7,
        },
        "phase_6c_high_level_byte_shaving_acquisition": {
            "status": byte_shaving_acquisition["status"],
            "reason": byte_shaving_acquisition["reason"],
            "campaign_run_count": byte_shaving_acquisition["campaign_run_count"],
            "total_experiment_count": byte_shaving_acquisition[
                "total_experiment_count"
            ],
            "total_executable_work_count": byte_shaving_acquisition[
                "total_executable_work_count"
            ],
            "total_blocked_work_count": byte_shaving_acquisition[
                "total_blocked_work_count"
            ],
            "overall_executable_conversion_rate": byte_shaving_acquisition[
                "overall_executable_conversion_rate"
            ],
            "total_compiler_required_count": byte_shaving_acquisition[
                "total_compiler_required_count"
            ],
            "total_packet_ir_operation_set_count": byte_shaving_acquisition[
                "total_packet_ir_operation_set_count"
            ],
            "total_queue_consumable_packet_ir_operation_set_count": byte_shaving_acquisition[
                "total_queue_consumable_packet_ir_operation_set_count"
            ],
            "total_exact_readiness_handoff_count": byte_shaving_acquisition[
                "total_exact_readiness_handoff_count"
            ],
            "queue_feedback_ready_count": byte_shaving_acquisition[
                "queue_feedback_ready_count"
            ],
            "queue_feedback_followup_queue_count": byte_shaving_acquisition[
                "queue_feedback_followup_queue_count"
            ],
            "queue_observation_recovery_queue_count": byte_shaving_acquisition[
                "queue_observation_recovery_queue_count"
            ],
            "queue_observation_recovery_executed_count": byte_shaving_acquisition[
                "queue_observation_recovery_executed_count"
            ],
            "queue_observation_recovery_execution_success_count": byte_shaving_acquisition[
                "queue_observation_recovery_execution_success_count"
            ],
            "post_recovery_feedback_replan_count": byte_shaving_acquisition[
                "post_recovery_feedback_replan_count"
            ],
            "post_recovery_feedback_replan_success_count": byte_shaving_acquisition[
                "post_recovery_feedback_replan_success_count"
            ],
            "post_recovery_feedback_policy_continue_count": byte_shaving_acquisition[
                "post_recovery_feedback_policy_continue_count"
            ],
            "post_recovery_feedback_followup_queue_count": byte_shaving_acquisition[
                "post_recovery_feedback_followup_queue_count"
            ],
            "post_recovery_feedback_followup_executed_count": byte_shaving_acquisition[
                "post_recovery_feedback_followup_executed_count"
            ],
            "post_recovery_feedback_followup_execution_success_count": byte_shaving_acquisition[
                "post_recovery_feedback_followup_execution_success_count"
            ],
            "post_recovery_feedback_continuation_queue_count": byte_shaving_acquisition[
                "post_recovery_feedback_continuation_queue_count"
            ],
            "local_mlx_ready_step_count": byte_shaving_acquisition[
                "local_mlx_ready_step_count"
            ],
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
        },
        "phase_6d_frontier_feedback_cycle": {
            "status": frontier_feedback_cycle["status"],
            "reason": frontier_feedback_cycle["reason"],
            "cycle_report_count": frontier_feedback_cycle["cycle_report_count"],
            "refresh_report_count": frontier_feedback_cycle["refresh_report_count"],
            "ready_local_execution_count": frontier_feedback_cycle[
                "ready_local_execution_count"
            ],
            "post_harvest_queue_count": frontier_feedback_cycle[
                "post_harvest_queue_count"
            ],
            "next_command": frontier_feedback_cycle["next_command"],
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
        },
        "phase_6e_pr95_mlx_control_profiles": {
            "status": pr95_mlx_profiles["status"],
            "reason": pr95_mlx_profiles["reason"],
            "profile_count": pr95_mlx_profiles["profile_count"],
            "executed_healthy_count": pr95_mlx_profiles["executed_healthy_count"],
            "blocked_count": pr95_mlx_profiles["blocked_count"],
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
        },
        "recommendation": (
            "prefer M5 Max parallel coarse-rank ($0) before paid GHA promotion; "
            "reference Phase 6 xray toolkit for diagnosis"
        ),
    }


def _format_dispatch_readiness() -> str:
    """Roll up actionable next-step status across phases."""
    readiness = _dispatch_readiness()
    lines = ["Per-phase dispatch readiness (current session):"]
    phase1 = readiness["phase_1_exact_eval_packets"]
    lines.append(
        "  Phase 1 (exact-eval packets):               "
        f"{phase1['status']} — {phase1['reason']}"
    )
    queue = readiness["phase_1_exact_ready_queue_hygiene"]
    lines.append(
        "  Phase 1 exact-ready queue hygiene:          "
        f"{queue['status']} — {queue['reason']}"
    )
    repairs = readiness["phase_1_exact_ready_score_axis_repairs"]
    lines.append(
        "  Phase 1 exact-ready score-axis repairs:     "
        f"{repairs['status']} — {repairs['reason']}"
    )
    handoffs = readiness["phase_1_materializer_exact_ready_handoffs"]
    lines.append(
        "  Phase 1 materializer exact-ready handoffs:  "
        f"{handoffs['status']} — {handoffs['reason']}"
    )
    phase4 = readiness["phase_4_gated_next_tick"]
    lines.append(
        "  Phase 4 (gated next-tick):                  "
        f"{phase4['n_lanes']} lane(s) gated; {phase4['reason']}"
    )
    phase5 = readiness["phase_5_meta_composition"]
    lines.append(
        "  Phase 5 (meta-composition):                 "
        f"{phase5['n_stacks']} compose-stack(s) tracked"
    )
    phase6c = readiness["phase_6c_high_level_byte_shaving_acquisition"]
    lines.append(
        "  Phase 6c (high-level byte shaving queue):   "
        f"{phase6c['status']} — {phase6c['reason']}"
    )
    phase6d = readiness["phase_6d_frontier_feedback_cycle"]
    lines.append(
        "  Phase 6d (frontier feedback cycle):         "
        f"{phase6d['status']} — {phase6d['reason']}"
    )
    phase6e = readiness["phase_6e_pr95_mlx_control_profiles"]
    lines.append(
        "  Phase 6e (PR95 MLX control profiles):       "
        f"{phase6e['status']} — {phase6e['reason']}"
    )
    phase7 = readiness["phase_7_constrained_coord_search"]
    lines.append(
        "  Phase 7 (constrained-coord-search):         "
        f"{phase7['status']} — {phase7['reason']}"
    )
    lines.append("")
    lines.append(f"Recommendation: {readiness['recommendation']}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20,
                        help="Cap dashboard rows (default: 20).")
    parser.add_argument(
        "--target-score",
        type=float,
        default=DEFAULT_SCORE_LOWERING_TARGET,
        help=(
            "Default active-routing cutoff for predicted candidate rows "
            f"(default: {DEFAULT_SCORE_LOWERING_TARGET:.2f}). Rows whose "
            "predicted low cannot beat this remain historical unless "
            "--show-above-target is supplied."
        ),
    )
    parser.add_argument(
        "--show-above-target",
        action="store_true",
        help="Show predicted rows above --target-score in human-readable sections.",
    )
    parser.add_argument("--skip-pareto", action="store_true",
                        help="Skip Phase 1 (Pareto pre-dispatch matrix).")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip Phase 2 (general score dashboard).")
    parser.add_argument("--skip-reconciler", action="store_true",
                        help="Skip Phase 3 (apogee_intN predicted-vs-actual).")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip Phase 4 (gated next-tick lanes).")
    parser.add_argument("--skip-composition", action="store_true",
                        help="Skip Phase 5 (meta-composition lanes).")
    parser.add_argument("--skip-provider-readiness", action="store_true",
                        help="Skip cached cloud-provider readiness section.")
    parser.add_argument("--refresh-provider-readiness", action="store_true",
                        help="Run read-only cloud provider checks before rendering the briefing.")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable composite JSON output.")
    args = parser.parse_args(argv)

    # Verify delegated tools exist before building either text or JSON output.
    for tool in (PARETO, DASHBOARD, RECONCILER, CLAIM_DISPATCH):
        if not tool.is_file():
            print(f"FATAL: missing dependency tool {tool.relative_to(REPO_ROOT)}",
                  file=sys.stderr)
            return 2
    if args.refresh_provider_readiness and not CLOUD_PROVIDER_READINESS.is_file():
        print(
            f"FATAL: missing dependency tool {CLOUD_PROVIDER_READINESS.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 2

    if args.json:
        dispatch_claim_summary = _dispatch_claim_summary()
        out = {
            "target_score": args.target_score,
            "codex_inbox_summary": _codex_inbox_summary(),
            "dispatch_claim_summary": dispatch_claim_summary,
            "dispatch_claim_historical_summary": _dispatch_claim_historical_summary(),
            "dispatch_readiness": _dispatch_readiness(),
            "xray_tools": _xray_toolkit_rows(),
            "cooperative_receiver_solver_integration": (
                _cooperative_receiver_solver_integration()
            ),
            "byte_shaving_acquisition": _byte_shaving_acquisition_summary(),
            "frontier_feedback_cycle": _frontier_feedback_cycle_summary(),
            "pr95_mlx_control_profiles": _pr95_mlx_control_profile_summary(),
            "l5_v2_frontier_readiness": _l5_v2_frontier_readiness(
                dispatch_claim_summary=dispatch_claim_summary
            ),
        }
        if not args.skip_provider_readiness:
            out["provider_readiness"] = _provider_readiness(refresh=args.refresh_provider_readiness)
        out["exact_ready_queue_audit"] = _exact_ready_queue_audit()
        out["exact_ready_score_axis_repairs"] = (
            _exact_ready_score_axis_repair_summary()
        )
        out["materializer_exact_ready_handoffs"] = (
            _materializer_exact_ready_handoff_summary()
        )
        if not args.skip_pareto:
            out["pareto"] = _run_json(PARETO, ["--json"])
            out["supplementary_lanes"] = _annotate_score_target_lanes(
                PHASE_1_SUPPLEMENTARY_LANES,
                target_score=args.target_score,
                active_only=False,
            )
            out["active_supplementary_lanes"] = _annotate_score_target_lanes(
                PHASE_1_SUPPLEMENTARY_LANES,
                target_score=args.target_score,
                active_only=True,
            )
            out["exact_eval_packets"] = _exact_eval_packet_summaries()
            out["non_dispatchable_readiness_artifacts"] = _non_dispatchable_readiness_artifacts()
        if not args.skip_dashboard:
            out["dashboard"] = _run_json(DASHBOARD, ["--top", str(args.top), "--json"])
        if not args.skip_reconciler:
            out["reconciler"] = _run_json(RECONCILER, ["--json"])
        if not args.skip_gated:
            out["gated_lanes"] = _annotate_score_target_lanes(
                PHASE_4_GATED_LANES,
                target_score=args.target_score,
                active_only=False,
            )
            out["active_gated_lanes"] = _annotate_score_target_lanes(
                PHASE_4_GATED_LANES,
                target_score=args.target_score,
                active_only=True,
            )
        if not args.skip_composition:
            out["composition_lanes"] = _annotate_score_target_lanes(
                PHASE_5_COMPOSITION_LANES,
                target_score=args.target_score,
                active_only=False,
            )
            out["active_composition_lanes"] = _annotate_score_target_lanes(
                PHASE_5_COMPOSITION_LANES,
                target_score=args.target_score,
                active_only=True,
            )
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Human-readable composite
    parts: list[str] = ["OPERATOR BRIEFING — dispatch trio"]
    parts.append(_section(
        "Codex inbox — open design questions and relays",
        _format_codex_inbox_summary(),
    ))
    parts.append(_section(
        "Dispatch claim coordination — active lane guard",
        _format_dispatch_claim_summary(),
    ))
    if not args.skip_provider_readiness:
        parts.append(_section(
            "Cloud provider readiness — cached exact/proxy boundary",
            _format_provider_readiness(refresh=args.refresh_provider_readiness),
        ))
    if not args.skip_pareto:
        parts.append(_section(
            "Phase 1 — Pre-dispatch: apogee_intN Pareto frontier",
            _run(PARETO).strip(),
        ))
        parts.append(_section(
            "Phase 1 supplementary — pre-registered non-Pareto lanes",
            _format_supplementary_lanes(
                target_score=args.target_score,
                show_above_target=args.show_above_target,
            ),
        ))
        parts.append(_section(
            "Phase 1 exact-eval packets — static-clean CUDA candidates",
            _format_exact_eval_packets(),
        ))
        parts.append(_section(
            "Phase 1 exact-ready queues — terminal-evidence audit",
            _format_exact_ready_queue_audit(),
        ))
        parts.append(_section(
            "Phase 1 exact-ready queues — score-axis repair planner",
            _format_exact_ready_score_axis_repairs(),
        ))
        parts.append(_section(
            "Phase 1 materializer exact-ready handoffs — queue-owned bridge/consumer state",
            _format_materializer_exact_ready_handoffs(),
        ))
        parts.append(_section(
            "Phase 1 blocked readiness artifacts — non-dispatchable public frontier work",
            _format_non_dispatchable_readiness_artifacts(),
        ))
    if not args.skip_dashboard:
        parts.append(_section(
            f"Phase 2 — Post-dispatch (general): top {args.top} contest scores on disk",
            _run(DASHBOARD, ["--top", str(args.top)]).strip(),
        ))
    if not args.skip_reconciler:
        parts.append(_section(
            "Phase 3 — Post-dispatch (apogee_intN): predicted-vs-actual reconciliation",
            _run(RECONCILER).strip(),
        ))
    if not args.skip_gated:
        parts.append(_section(
            "Phase 4 — Gated next-tick lanes (sequential validation)",
            _format_gated_lanes(
                target_score=args.target_score,
                show_above_target=args.show_above_target,
            ),
        ))
    if not args.skip_composition:
        parts.append(_section(
            "Phase 5 — Meta-composition lanes (single-dispatch payoff of multi-sister stacks)",
            _format_composition_lanes(
                target_score=args.target_score,
                show_above_target=args.show_above_target,
            ),
        ))
    parts.append(_section(
        "Phase 6 — XRAY toolkit (diagnostic tools landed 2026-05-09)",
        _format_xray_toolkit(),
    ))
    parts.append(_section(
        "Phase 6b — Cooperative-receiver solver integration hooks",
        _format_cooperative_receiver_solver_integration(),
    ))
    parts.append(_section(
        "Phase 6c — High-level byte-shaving acquisition queue",
        _format_byte_shaving_acquisition_summary(),
    ))
    parts.append(_section(
        "Phase 6d — Frontier feedback cycle autopolicy",
        _format_frontier_feedback_cycle_summary(),
    ))
    parts.append(_section(
        "Phase 6e — PR95 MLX control profiles",
        _format_pr95_mlx_control_profiles(),
    ))
    parts.append(_section(
        "Phase 7 — Constrained-coord-search status (sister subagent a8522fca)",
        _format_constrained_coord_search_status(),
    ))
    parts.append(_section(
        "Phase 8 — Per-phase dispatch readiness (next actionable step)",
        _format_dispatch_readiness(),
    ))
    parts.append(_section(
        "Phase 9 — L5-v2 TT5L-first frontier readiness",
        _format_l5_v2_frontier_readiness(),
    ))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
