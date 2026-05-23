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
EXACT_READY_SUPPRESSION_MANIFEST = (
    REPO_ROOT / ".omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json"
)
L5_V2_PACKETIR_SECTION_ENTROPY_MATRIX_ARTIFACT_PATH = (
    ".omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json"
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
    return {
        "schema": "pact.operator_dispatch_readiness.v1",
        "phase_1_exact_eval_packets": phase1,
        "phase_1_exact_ready_queue_hygiene": {
            "status": queue_status,
            "reason": queue_reason,
            "stale_ready_row_count": stale_rows,
        },
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
            "l5_v2_frontier_readiness": _l5_v2_frontier_readiness(
                dispatch_claim_summary=dispatch_claim_summary
            ),
        }
        if not args.skip_provider_readiness:
            out["provider_readiness"] = _provider_readiness(refresh=args.refresh_provider_readiness)
        out["exact_ready_queue_audit"] = _exact_ready_queue_audit()
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
