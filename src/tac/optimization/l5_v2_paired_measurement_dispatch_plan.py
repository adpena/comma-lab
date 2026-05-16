# SPDX-License-Identifier: MIT
"""Paired-axis dispatch plan for L5 v2 measurements.

This module turns the L5-v2 measurement schedule into concrete paired
CPU/CUDA work units. It is planning-only: it does not launch provider work,
does not write lane claims, and does not claim score movement. The paired
Modal dispatcher remains the only executable dispatch surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from tac.deploy.modal.paired_dispatch import (
    PAIRED_AUTH_EVAL_DISPATCH_TOOL,
    paired_auth_eval_dispatch_command_template,
)
from tac.deploy.modal.paired_dispatch_contract import (
    paired_auth_eval_dispatch_command_blockers,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_MEASUREMENT_SCHEDULE_SCHEMA,
)

L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA = (
    "l5_v2_paired_measurement_dispatch_plan_v1"
)
L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH = (
    "tools/build_l5_v2_paired_measurement_dispatch_plan.py"
)
L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH = (
    ".omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json"
)
L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH = (
    ".omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.md"
)

_REQUIRED_EXACT_AXES = ("contest_cpu", "contest_cuda")
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_PLACEHOLDER_ARCHIVE = "FILL_ARCHIVE_ZIP"
_PLACEHOLDER_ARCHIVE_SHA256 = "FILL_ARCHIVE_SHA256"
_PLACEHOLDER_SUBMISSION_DIR = "FILL_SUBMISSION_DIR"

_FALSE_AUTHORITY_FLAGS = {
    "planning_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
    "adjudication_required": True,
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("/", "_")
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = text.strip("_.:-")
    return text or "unknown"


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _as_mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _measurement_by_id(schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in _as_mapping_rows(schedule.get("measurements")):
        measurement_id = str(row.get("measurement_id") or "").strip()
        if measurement_id:
            out[measurement_id] = row
    return out


def _lane_id_base(measurement_id: str) -> str:
    return f"lane_l5_v2_{_slug(measurement_id)}"


def _pair_group_id(measurement_id: str) -> str:
    return f"pair_l5_v2_{_slug(measurement_id)}_cpu_cuda"


def _run_id(measurement_id: str) -> str:
    return f"l5_v2_{_slug(measurement_id)}_paired_measurement"


def _output_root(measurement_id: str) -> str:
    return f"experiments/results/l5_v2_probe/{_slug(measurement_id)}"


def _axis_output_dirs(*, measurement_id: str) -> dict[str, str]:
    run_id = _run_id(measurement_id)
    root = _output_root(measurement_id)
    return {
        "contest_cuda": f"{root}/modal_auth_eval/{run_id}_cuda",
        "contest_cpu": f"{root}/modal_auth_eval_cpu/{run_id}_cpu",
    }


def _row_measurement_blockers(row: Mapping[str, Any]) -> list[str]:
    return _as_text_list(row.get("blockers"))


def _dispatch_blockers(command: str) -> list[str]:
    blockers = [
        "requires_byte_closed_archive_path",
        "requires_archive_sha256",
        "requires_submission_dir_or_inflate_runtime",
        "requires_operator_execute_flag",
    ]
    blockers.extend(
        paired_auth_eval_dispatch_command_blockers(
            paired_dispatch_tool=PAIRED_AUTH_EVAL_DISPATCH_TOOL,
            command_template=command,
            require_command=True,
        )
    )
    return _dedupe(blockers)


def _paired_dispatch_command(*, measurement_id: str, candidate_id: str) -> list[str]:
    command = paired_auth_eval_dispatch_command_template(
        archive_path=_PLACEHOLDER_ARCHIVE,
        submission_dir=_PLACEHOLDER_SUBMISSION_DIR,
        lane_id_base=_lane_id_base(measurement_id),
        archive_sha256=_PLACEHOLDER_ARCHIVE_SHA256,
        execute=False,
        label=f"l5_v2_{_slug(candidate_id or measurement_id)}",
        run_id=_run_id(measurement_id),
        inflate_sh="inflate.sh",
        output_root=_output_root(measurement_id),
        gpu="T4",
        claim_agent="codex:l5_v2_paired_measurement_dispatch",
        claim_notes=f"l5_v2_paired_measurement:{_pair_group_id(measurement_id)}",
    )
    pair_index = command.index("--pair-group-id") + 1
    command[pair_index] = _pair_group_id(measurement_id)
    return command


def _harvest_commands(*, measurement_id: str) -> dict[str, str]:
    return {
        axis: f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {out_dir}"
        for axis, out_dir in _axis_output_dirs(measurement_id=measurement_id).items()
    }


def build_l5_v2_paired_measurement_dispatch_plan(
    *,
    schedule: Mapping[str, Any],
    schedule_path: str = "",
    schedule_sha256: str = "",
) -> dict[str, Any]:
    """Return a non-promotional paired-axis dispatch plan for active rows."""

    top_blockers: list[str] = []
    if schedule.get("schema") != L5V2_MEASUREMENT_SCHEDULE_SCHEMA:
        top_blockers.append("l5_v2_measurement_schedule_schema_mismatch")
    active_measurement_ids = _as_text_list(schedule.get("active_measurement_ids"))
    if not active_measurement_ids:
        top_blockers.append("l5_v2_active_measurement_ids_missing")

    rows = _measurement_by_id(schedule)
    work_units: list[dict[str, Any]] = []
    for measurement_id in active_measurement_ids:
        row = rows.get(measurement_id)
        if row is None:
            top_blockers.append(f"l5_v2_active_measurement_missing:{measurement_id}")
            continue

        required_axes = _as_text_list(row.get("required_axes")) or list(
            _REQUIRED_EXACT_AXES
        )
        missing_axes = [axis for axis in _REQUIRED_EXACT_AXES if axis not in required_axes]
        unknown_axes = [axis for axis in required_axes if axis not in _REQUIRED_EXACT_AXES]
        if missing_axes:
            top_blockers.append(
                f"l5_v2_measurement_missing_required_axes:{measurement_id}:"
                + ",".join(missing_axes)
            )
        if unknown_axes:
            top_blockers.append(
                f"l5_v2_unknown_measurement_axis:{measurement_id}:"
                + ",".join(unknown_axes)
            )

        candidate_id = str(row.get("candidate_id") or "").strip()
        pair_group_id = _pair_group_id(measurement_id)
        command = _paired_dispatch_command(
            measurement_id=measurement_id,
            candidate_id=candidate_id,
        )
        command_string = " ".join(command)
        measurement_blockers = _row_measurement_blockers(row)
        dispatch_blockers = _dispatch_blockers(command_string)
        top_blockers.extend(measurement_blockers)
        top_blockers.extend(dispatch_blockers)
        lanes = {
            "contest_cuda": f"{_lane_id_base(measurement_id)}_contest_cuda",
            "contest_cpu": f"{_lane_id_base(measurement_id)}_contest_cpu",
        }
        work_units.append(
            {
                **_FALSE_AUTHORITY_FLAGS,
                "measurement_id": measurement_id,
                "candidate_id": candidate_id,
                "lane_id": _lane_id_base(measurement_id),
                "lanes": lanes,
                "pair_group_id": pair_group_id,
                "required_axes": list(_REQUIRED_EXACT_AXES),
                "provider": "modal",
                "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
                "dispatch_command": command,
                "dispatch_command_template": command_string,
                "dispatch_command_executable": False,
                "claim_lifecycle_owner": (
                    "tools/dispatch_modal_paired_auth_eval.py and the per-axis "
                    "Modal auth-eval wrappers"
                ),
                "preclaim_forbidden": True,
                "standalone_active_claim_command": None,
                "axis_output_dirs": _axis_output_dirs(measurement_id=measurement_id),
                "harvest_commands": _harvest_commands(measurement_id=measurement_id),
                "measurement_blockers_to_close": measurement_blockers,
                "dispatch_blockers": dispatch_blockers,
            }
        )

    top_blockers = _dedupe(top_blockers)
    plan_id_payload = {
        "schema": schedule.get("schema"),
        "schedule_sha256": schedule_sha256 or _sha256_text(
            json.dumps(schedule, sort_keys=True, allow_nan=False)
        ),
        "active_measurement_ids": active_measurement_ids,
        "required_axes": list(_REQUIRED_EXACT_AXES),
    }
    plan = {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA,
        "tool": L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH,
        "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        "plan_id": "l5_v2_paired_measurement_dispatch_"
        + hashlib.sha256(_canonical_json_bytes(plan_id_payload)).hexdigest()[:16],
        "source_schedule_path": schedule_path,
        "source_schedule_sha256": schedule_sha256,
        "source_schedule_schema": str(schedule.get("schema") or ""),
        "active_rule": str(schedule.get("active_rule_id") or ""),
        "active_rule_id": str(schedule.get("active_rule_id") or ""),
        "active_measurement_ids": active_measurement_ids,
        "required_axes": list(_REQUIRED_EXACT_AXES),
        "work_unit_count": len(work_units),
        "ready_work_unit_count": 0,
        "work_units": work_units,
        "blockers": top_blockers,
    }
    return plan


def render_l5_v2_paired_measurement_dispatch_plan_markdown(
    plan: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing dispatch plan report."""

    lines = [
        "# L5 v2 paired measurement dispatch plan",
        "",
        f"- schema: `{plan.get('schema')}`",
        f"- plan_id: `{plan.get('plan_id')}`",
        f"- active_rule_id: `{plan.get('active_rule_id')}`",
        f"- work_unit_count: `{plan.get('work_unit_count')}`",
        f"- ready_work_unit_count: `{plan.get('ready_work_unit_count')}`",
        "- planning_only: `true`",
        "- score_claim: `false`",
        "- score_claim_valid: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- rank_or_kill_eligible: `false`",
        "- dispatch_attempted: `false`",
        "- adjudication_required: `true`",
        f"- blockers: `{plan.get('blockers')}`",
        "",
        "## Work Units",
    ]
    for row in _as_mapping_rows(plan.get("work_units")):
        lines.extend(
            [
                "",
                f"### {row.get('measurement_id')}",
                "",
                f"- candidate_id: `{row.get('candidate_id')}`",
                f"- lane_id: `{row.get('lane_id')}`",
                f"- lanes: `{row.get('lanes')}`",
                f"- pair_group_id: `{row.get('pair_group_id')}`",
                f"- required_axes: `{row.get('required_axes')}`",
                f"- paired_dispatch_tool: `{row.get('paired_dispatch_tool')}`",
                f"- dispatch_command_executable: `{row.get('dispatch_command_executable')}`",
                f"- claim_lifecycle_owner: `{row.get('claim_lifecycle_owner')}`",
                f"- measurement_blockers_to_close: `{row.get('measurement_blockers_to_close')}`",
                f"- dispatch_blockers: `{row.get('dispatch_blockers')}`",
                "- dispatch_command: "
                f"`{row.get('dispatch_command_template')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def dispatch_plan_json(plan: Mapping[str, Any]) -> str:
    """Return canonical JSON text for durable artifacts."""

    return json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH",
    "L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_REPORT_PATH",
    "L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA",
    "L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_TOOL_PATH",
    "build_l5_v2_paired_measurement_dispatch_plan",
    "dispatch_plan_json",
    "render_l5_v2_paired_measurement_dispatch_plan_markdown",
]
