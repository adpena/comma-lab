# SPDX-License-Identifier: MIT
"""Operator doctor plan for the TT5L Lightning route.

The route-unblock packet identifies the remaining provider blockers. This module
turns that packet into a machine-checkable doctor plan without contacting
Lightning or implying dispatch readiness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_tt5l_lightning_route_unblock_packet import (
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA,
)

L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA = (
    "l5_v2_tt5l_lightning_required_doctor_plan_v1"
)
L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_lightning_doctor_plan.py"
)
L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.json"
)
L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_REPORT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md"
)
L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_provider_dispatch": False,
    "dispatch_attempted": False,
    "provider_spend_attempted": False,
}
_AUTHORITY_FLAGS = {
    "planning_only": True,
    **_FALSE_AUTHORITY_FLAGS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _doctor_command(*, identity_flag: str) -> str:
    identity_value = "$LIGHTNING_SDK_USER" if identity_flag == "--user" else "$LIGHTNING_ORG"
    return (
        ".venv/bin/python scripts/launch_lightning_batch_job.py doctor "
        f"--json-out {L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH} "
        "--run-id l5_v2_tt5l_lightning_required_doctor_20260517 "
        "--strict "
        "--ssh-target \"$LIGHTNING_SSH_TARGET\" "
        "--require-ssh "
        "--remote-supply-chain "
        "--require-remote-supply-chain "
        "--repo-dir /teamspace/studios/this_studio/pact "
        "--python-bin .venv/bin/python "
        "--teamspace \"$LIGHTNING_TEAMSPACE\" "
        f"{identity_flag} \"{identity_value}\" "
        "--machine-inventory "
        "--require-machine-inventory "
        "--machine T4 "
        "--gpu-only"
    )


def build_l5_v2_tt5l_lightning_doctor_plan(
    *,
    route_packet: Mapping[str, Any],
    route_packet_path: str | Path,
    repo_root: str | Path,
    current_head_commit: str = "",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return a false-authority-safe operator doctor plan."""

    root = Path(repo_root).resolve()
    route_path = _resolve_repo_path(route_packet_path, root)
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    blockers: list[str] = []
    if route_packet.get("schema") != L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA:
        blockers.append("source_route_packet_schema_mismatch")
    for field, expected in _AUTHORITY_FLAGS.items():
        if route_packet.get(field) is not expected:
            blockers.append(f"source_route_packet_{field}_not_{str(expected).lower()}")
    source_artifact_blockers = route_packet.get("blockers")
    if isinstance(source_artifact_blockers, list) and source_artifact_blockers:
        blockers.append("source_route_packet_has_artifact_blockers")

    remaining_route_blockers = [
        str(blocker)
        for blocker in route_packet.get("remaining_blockers", [])
        if str(blocker)
    ]
    source_artifacts = (
        route_packet.get("source_artifacts")
        if isinstance(route_packet.get("source_artifacts"), Mapping)
        else {}
    )
    route_plan = (
        source_artifacts.get("sideinfo_lightning_paired_axis_plan")
        if isinstance(source_artifacts.get("sideinfo_lightning_paired_axis_plan"), Mapping)
        else {}
    )
    if route_plan.get("source_relevant_paths_match_current_head") is not True:
        blockers.append("source_route_packet_paired_axis_source_not_current")

    return {
        **_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA,
        "tool": L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_TOOL_PATH,
        "generated_at_utc": generated,
        "current_head_commit": current_head_commit,
        "source_route_packet": _repo_relative(route_path, root),
        "source_route_packet_sha256": _sha256_file(route_path) if route_path.is_file() else "",
        "source_route_packet_commit": str(route_packet.get("current_head_commit") or ""),
        "source_route_remaining_blockers": remaining_route_blockers,
        "doctor_output_path": L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH,
        "ready_for_operator_doctor": not blockers,
        "ready_for_non_dry_run_submit": False,
        "operator_environment_required": {
            "LIGHTNING_TEAMSPACE": {
                "required": True,
                "purpose": "Lightning teamspace for T4 machine inventory and batch jobs.",
            },
            "LIGHTNING_SSH_TARGET": {
                "required": True,
                "purpose": "SSH target for remote auth and remote supply-chain scan.",
            },
            "LIGHTNING_SDK_USER_or_LIGHTNING_ORG": {
                "required": True,
                "purpose": "Exactly one Lightning SDK owner identity.",
                "exclusive": True,
            },
            "LIGHTNING_STUDIO": {
                "required_after_doctor_for_submit": True,
                "purpose": "Studio name required by non-dry-run submit templates.",
            },
        },
        "identity_modes": [
            {
                "mode": "user",
                "required_env": ["LIGHTNING_TEAMSPACE", "LIGHTNING_SSH_TARGET", "LIGHTNING_SDK_USER"],
                "forbidden_env": ["LIGHTNING_ORG"],
                "doctor_command_template": _doctor_command(identity_flag="--user"),
            },
            {
                "mode": "org",
                "required_env": ["LIGHTNING_TEAMSPACE", "LIGHTNING_SSH_TARGET", "LIGHTNING_ORG"],
                "forbidden_env": ["LIGHTNING_SDK_USER"],
                "doctor_command_template": _doctor_command(identity_flag="--org"),
            },
        ],
        "doctor_required_checks": {
            "expected_status": "OK",
            "required_json_fields": [
                "schema_version",
                "tool",
                "recorded_at_utc",
                "checks",
                "status",
                "failed_checks",
            ],
            "required_checks": [
                "local_supply_chain",
                "ssh_auth",
                "remote_supply_chain",
                "machine_inventory",
            ],
            "check_pass_predicates": {
                "ssh_auth": "checks.ssh_auth.ok == true",
                "remote_supply_chain": "checks.remote_supply_chain.ok == true",
                "machine_inventory": "checks.machine_inventory.ok == true and machine_count > 0",
                "local_supply_chain": "checks.local_supply_chain.ok != false",
            },
        },
        "route_blocker_closure_map": {
            "LIGHTNING_TEAMSPACE missing": [
                "Set LIGHTNING_TEAMSPACE.",
                "Doctor must pass machine_inventory.",
            ],
            "LIGHTNING_SDK_USER or LIGHTNING_ORG missing": [
                "Choose exactly one identity mode.",
                "Doctor must pass machine_inventory with that owner.",
            ],
            "LIGHTNING_SSH_TARGET missing": [
                "Set LIGHTNING_SSH_TARGET.",
                "Doctor must pass ssh_auth and remote_supply_chain.",
            ],
            "Lightning machine inventory not checked": [
                "Doctor must pass machine_inventory for T4 with gpu_only.",
            ],
            "remote CUDA runtime not probed": [
                "Doctor must pass remote_supply_chain before submit.",
                "Non-dry-run submit still performs exact-eval remote preflight.",
            ],
            "source manifest not staged to remote Lightning workspace": [
                "Run each bundle cells[*].stage_source_manifest_command_template after doctor.",
            ],
            "active dispatch claims not created for non-dry-run cells": [
                "Run each bundle cells[*].claim_command after doctor and before submit.",
            ],
            "Lightning credits or quota not checked": [
                "Machine inventory must return at least one T4 candidate.",
                "Operator still owns account-credit confirmation before spend.",
            ],
        },
        "next_after_doctor_if_ok": [
            "Run per-cell stage_source_manifest_command_template from the execution bundle.",
            "Run per-cell claim_command before every non-dry-run submit.",
            "Replace all placeholders in non_dry_run_submit_command_template.",
            "Submit only after doctor JSON status is OK and all required checks pass.",
            "Harvest contest_auth_eval artifacts and terminal claims before any score claim.",
        ],
        "blockers": blockers,
    }


def l5_v2_tt5l_lightning_doctor_plan_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text for the doctor plan."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_lightning_doctor_plan_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render the TT5L Lightning doctor plan for operators."""

    lines = [
        "# L5 v2 TT5L Lightning required doctor plan",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Commit: `{payload.get('current_head_commit')}`",
        "",
        "This generated plan converts the TT5L route blockers into the exact "
        "Lightning doctor commands and JSON pass predicates. It does not run "
        "Lightning, submit jobs, claim score movement, or create dispatch claims.",
        "",
        "## Authority",
        "",
        "- `planning_only=true`",
        "- `score_claim=false`",
        "- `promotion_eligible=false`",
        "- `rank_or_kill_eligible=false`",
        "- `ready_for_exact_eval_dispatch=false`",
        "- `ready_for_provider_dispatch=false`",
        "- `dispatch_attempted=false`",
        "- `provider_spend_attempted=false`",
        "",
        "## Source",
        "",
        f"- Route packet: `{payload.get('source_route_packet')}`",
        f"- Route packet SHA-256: `{payload.get('source_route_packet_sha256')}`",
        f"- ready_for_operator_doctor: `{payload.get('ready_for_operator_doctor')}`",
        f"- blockers: `{payload.get('blockers', [])}`",
        "",
        "## Doctor Commands",
        "",
    ]
    for mode in payload.get("identity_modes", []):
        if not isinstance(mode, Mapping):
            continue
        lines.extend(
            [
                f"### {mode.get('mode')}",
                "",
                f"- required env: `{mode.get('required_env')}`",
                f"- forbidden env: `{mode.get('forbidden_env')}`",
                "",
                "```bash",
                str(mode.get("doctor_command_template") or ""),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Required Checks",
            "",
            f"- doctor output: `{payload.get('doctor_output_path')}`",
            "- expected `status`: `OK`",
            "- required checks: `local_supply_chain`, `ssh_auth`, "
            "`remote_supply_chain`, `machine_inventory`",
            "",
            "## Remaining Route Work",
            "",
        ]
    )
    for item in payload.get("next_after_doctor_if_ok", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_ARTIFACT_PATH",
    "L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_REPORT_PATH",
    "L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA",
    "L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_TOOL_PATH",
    "L5V2_TT5L_LIGHTNING_REQUIRED_DOCTOR_OUTPUT_PATH",
    "build_l5_v2_tt5l_lightning_doctor_plan",
    "l5_v2_tt5l_lightning_doctor_plan_json",
    "render_l5_v2_tt5l_lightning_doctor_plan_markdown",
]
