# SPDX-License-Identifier: MIT
"""Compile SNeRV decoder-waterfill actions into explicit receiver modes."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import DECODER_SUBBANDS

SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA = "snerv_waterfill_mode_assignment.v1"
SNERV_TRAINED_LADDER_WATERFILL_SCHEMA = "snerv_trained_ladder_waterfill.v1"
PLANNING_AXIS = "[planning/control:false-authority]"
DEFAULT_RECEIVER_PACKET_ROOT = (
    "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets"
)
_GROUP_RE = re.compile(r"^decoder\.level(?P<level>\d+)\.(?P<subband>LH|HL|HH)\.kernel$")


class SnervWaterfillModeAssignmentError(ValueError):
    """Raised when a SNeRV waterfill report cannot be compiled safely."""


def build_snerv_waterfill_mode_assignment(
    waterfill_report: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Map waterfill selected actions to explicit SNeRV mixed decoder modes."""

    if waterfill_report.get("schema") != SNERV_TRAINED_LADDER_WATERFILL_SCHEMA:
        raise SnervWaterfillModeAssignmentError(
            f"expected {SNERV_TRAINED_LADDER_WATERFILL_SCHEMA} report"
        )
    rows = []
    blockers = [
        "mode_assignment_is_false_authority_until_receiver_replay_and_exact_eval"
    ]
    for row_index, row in enumerate(waterfill_report.get("rows") or ()):
        if not isinstance(row, Mapping):
            continue
        compiled = _compile_row(
            row,
            row_id=str(row.get("row_id") or f"snerv_waterfill_row_{row_index:04d}"),
            candidate_id=candidate_id or str(waterfill_report.get("candidate_id") or ""),
            source_report_blockers=_string_list(waterfill_report.get("blockers")),
        )
        rows.append(compiled)
        blockers.extend(compiled["blockers"])

    ready_probe_rows = [row for row in rows if row["ready_for_local_advisory_probe"]]
    export_rows = [row for row in rows if row["ready_for_receiver_mode_export"]]
    report = {
        "schema": SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA,
        "source_schema": waterfill_report.get("schema"),
        "family": "snerv",
        "axis_tag": PLANNING_AXIS,
        "authority": "false_authority_snerv_waterfill_mode_assignment_no_score_claim",
        "candidate_id": candidate_id or waterfill_report.get("candidate_id"),
        "row_count": len(rows),
        "local_advisory_probe_ready_row_count": len(ready_probe_rows),
        "receiver_mode_export_ready_row_count": len(export_rows),
        "rows": rows,
        "blockers": _ordered_unique(blockers),
        "next_actions": _next_actions(rows),
        **FALSE_AUTHORITY,
    }
    return report


def render_snerv_waterfill_mode_assignment_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing mode-assignment summary."""

    lines = [
        "# SNeRV waterfill mode assignment",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        "",
        "| row | modes | local probe | receiver export | blocker count |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report.get("rows") or ():
        lines.append(
            "| {row_id} | `{modes}` | {probe} | {export} | {blockers} |".format(
                row_id=row.get("row_id"),
                modes=row.get("mode_plan_cli_arg"),
                probe=str(bool(row.get("ready_for_local_advisory_probe"))).lower(),
                export=str(bool(row.get("ready_for_receiver_mode_export"))).lower(),
                blockers=len(row.get("blockers") or ()),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _compile_row(
    row: Mapping[str, Any],
    *,
    row_id: str,
    candidate_id: str,
    source_report_blockers: Sequence[str],
) -> dict[str, Any]:
    waterfill_plan = row.get("waterfill_plan")
    if not isinstance(waterfill_plan, Mapping):
        return _blocked_row(
            row_id=row_id,
            row=row,
            blockers=[
                *source_report_blockers,
                *_string_list(row.get("blockers")),
                "waterfill_plan_missing",
            ],
        )

    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    blockers = [
        *source_report_blockers,
        *_string_list(row.get("blockers")),
        *_string_list(waterfill_plan.get("blockers")),
    ]
    for action_row in waterfill_plan.get("rows") or ():
        if not isinstance(action_row, Mapping):
            continue
        group_name = str(action_row.get("group_name") or "")
        match = _GROUP_RE.match(group_name)
        if match is None:
            blockers.append(f"unrecognized_decoder_group_name:{group_name}")
            continue
        key = (int(match.group("level")), match.group("subband"))
        if key in parsed:
            blockers.append(
                f"duplicate_decoder_group:level{key[0]}.{key[1]}"
            )
            continue
        mode, mode_blockers = _mode_for_action(action_row)
        parsed[key] = {
            "group_name": group_name,
            "level": key[0],
            "subband": key[1],
            "selected_action": action_row.get("selected_action"),
            "selected_bits": action_row.get("selected_bits"),
            "receiver_mode": mode,
            "blockers": mode_blockers,
        }
        blockers.extend(mode_blockers)

    levels = _levels_from_parsed(parsed)
    modes: list[str] = []
    mode_rows: list[dict[str, Any]] = []
    for level in range(levels):
        for subband in DECODER_SUBBANDS:
            key = (level, subband)
            value = parsed.get(key)
            if value is None:
                blockers.append(f"decoder_mode_group_missing:level{level}.{subband}")
                continue
            mode_rows.append(value)
            modes.append(str(value["receiver_mode"]))

    complete = bool(modes) and len(modes) == levels * len(DECODER_SUBBANDS)
    unique_blockers = _ordered_unique(blockers)
    export_blockers = [
        blocker
        for blocker in unique_blockers
        if blocker
        not in {
            "contest_cpu_cuda_exact_eval_not_executed",
            "mode_assignment_is_false_authority_until_receiver_replay_and_exact_eval",
        }
    ]
    probe_command_argv = (
        _probe_command_argv(row_id=row_id, levels=levels, modes=modes)
        if complete
        else None
    )
    return {
        "row_id": row_id,
        "candidate_id": f"{candidate_id}:{row_id}" if candidate_id else row_id,
        "archive_sha256": row.get("archive_sha256_actual") or row.get("archive_sha256"),
        "decoder_payload_schema": row.get("decoder_payload_schema"),
        "decoder_precision_mode": row.get("decoder_precision_mode"),
        "levels": levels,
        "expected_mode_count": levels * len(DECODER_SUBBANDS),
        "mode_count": len(modes),
        "modes": modes,
        "mode_plan_cli_arg": ",".join(modes),
        "mode_histogram": dict(Counter(modes)),
        "mode_rows": mode_rows,
        "ready_for_local_advisory_probe": complete,
        "ready_for_receiver_mode_export": complete and not export_blockers,
        "probe_command_axis_tag": "[macOS-CPU advisory]",
        "probe_command_argv": probe_command_argv,
        "probe_receiver_packet_dir": (
            _probe_packet_dir(row_id) if complete else None
        ),
        "probe_command_hint": (
            " ".join(probe_command_argv) if probe_command_argv else None
        ),
        "blockers": unique_blockers,
        **FALSE_AUTHORITY,
    }


def _mode_for_action(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    action = str(row.get("selected_action") or "").strip().lower()
    bits = row.get("selected_bits")
    blockers: list[str] = []
    if action == "zero_rle" or bits == 0:
        return "zero", blockers
    if action in {"int2", "int4", "int8", "fp16"}:
        return action, blockers
    if action == "fp32_protect" or bits == 32:
        blockers.extend(
            [
                "mixed_decoder_modes_do_not_support_fp32",
                "fp32_protect_downgraded_to_fp16_requires_receiver_replay",
            ]
        )
        return "fp16", blockers
    blockers.append(f"selected_action_not_compilable:{action or 'missing'}")
    return "fp16", blockers


def _levels_from_parsed(parsed: Mapping[tuple[int, str], Mapping[str, Any]]) -> int:
    if not parsed:
        return 0
    max_level = max(level for level, _subband in parsed)
    return int(max_level) + 1


def _blocked_row(
    *,
    row_id: str,
    row: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "archive_sha256": row.get("archive_sha256_actual") or row.get("archive_sha256"),
        "decoder_payload_schema": row.get("decoder_payload_schema"),
        "decoder_precision_mode": row.get("decoder_precision_mode"),
        "levels": 0,
        "expected_mode_count": 0,
        "mode_count": 0,
        "modes": [],
        "mode_plan_cli_arg": "",
        "mode_histogram": {},
        "mode_rows": [],
        "ready_for_local_advisory_probe": False,
        "ready_for_receiver_mode_export": False,
        "probe_command_axis_tag": "[macOS-CPU advisory]",
        "probe_command_argv": None,
        "probe_receiver_packet_dir": None,
        "probe_command_hint": None,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _next_actions(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    actions = []
    if any("decoder_weight_saliency_missing_for_some_groups" in row.get("blockers", ()) for row in rows):
        actions.append("run_decoder_weight_saliency_or_vjp_before_mode_export")
    if any("fp32_protect_downgraded_to_fp16_requires_receiver_replay" in row.get("blockers", ()) for row in rows):
        actions.append("probe_fp16_protect_substitution_before_export")
    if any("full_video_coverage_missing" in row.get("blockers", ()) for row in rows):
        actions.append("rerun_on_full600_receiver_closed_snar_rows")
    if any(row.get("ready_for_local_advisory_probe") for row in rows):
        actions.append("run_local_receiver_decoded_mode_probe_with_cli_arg")
    return _ordered_unique(actions)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _probe_command_argv(*, row_id: str, levels: int, modes: Sequence[str]) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/probe_snerv_decoder_mode_assignments.py",
        "--levels",
        str(int(levels)),
        "--mode-plan",
        ",".join(str(mode) for mode in modes),
        "--receiver-packet-dir",
        _probe_packet_dir(row_id),
    ]


def _probe_packet_dir(row_id: str) -> str:
    return f"{DEFAULT_RECEIVER_PACKET_ROOT}/{_slug(row_id)}"


def _slug(value: str) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value)
    ).strip("_")
    return text or "row"


def load_snerv_waterfill_mode_assignment_source(path: str | Path) -> dict[str, Any]:
    """Load a JSON SNeRV waterfill report."""

    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


__all__ = [
    "SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA",
    "SnervWaterfillModeAssignmentError",
    "build_snerv_waterfill_mode_assignment",
    "load_snerv_waterfill_mode_assignment_source",
    "render_snerv_waterfill_mode_assignment_markdown",
]
