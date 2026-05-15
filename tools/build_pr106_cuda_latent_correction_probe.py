#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority PR106 CUDA latent-correction probe plan.

This is a deterministic planner, not an archive materializer. It combines
hard-pair XRay rows with paired CPU/CUDA axis evidence to choose a byte-budgeted
set of PR106 latent-correction probe targets. The future ``--materialize`` flag
is intentionally present but fails closed until runtime mutation, frame parity,
and paired exact CPU/CUDA eval custody are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any

TOOL = "tools/build_pr106_cuda_latent_correction_probe.py"
SCHEMA = "pr106_cuda_latent_correction_probe_plan_v1"
ARCHIVE_BYTES_DENOMINATOR = 37_545_489
PR100_RAW_HEADER_BYTES = 2
PR100_RAW_BYTES_PER_SELECTED_PAIR = 2
PR101_FIXED_MICROCODEC_SIDECAR_BYTES = 607
PR106_FORMAT0C_EXACT_RADIX_SIDECAR_BYTES = 511
LATENT_DIM = 28
DELTA_Q_VALUES = (-2, -1, 1, 2)
FALSE_AUTHORITY = {
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "contest_axis_claim": False,
    "dispatch_attempted": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_broad_waterfill_dispatch": False,
    "frontier_language_allowed": False,
    "paired_cpu_cuda_exact_eval_required": True,
}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value.replace(",", ""))
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load JSON {path}: {exc}") from exc


def _path_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _check_false_authority(payload: dict[str, Any], path: Path) -> None:
    authority = payload.get("authority")
    if isinstance(authority, dict):
        payload = {**payload, **authority}
    for key in (
        "score_claim",
        "score_claim_valid",
        "contest_axis_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
    ):
        if payload.get(key) is True:
            raise ValueError(f"{path} has {key}=true; refusing score-authority input")


def _score_terms(row: dict[str, Any]) -> tuple[float, float, float]:
    pose = _as_float(row.get("pose_score_contribution"))
    seg = _as_float(row.get("seg_score_contribution"))
    pose_dist = _as_float(row.get("pose_dist"))
    seg_dist = _as_float(row.get("seg_dist"))
    if pose is None and pose_dist is not None:
        pose = math.sqrt(max(0.0, 10.0 * pose_dist))
    if seg is None and seg_dist is not None:
        seg = 100.0 * seg_dist
    pose_value = pose or 0.0
    seg_value = seg or 0.0
    component = _as_float(row.get("component_score_no_rate"))
    if component is None:
        component = pose_value + seg_value
    return pose_value, seg_value, component


def _pair_source(path: Path, label: str | None, kind: str) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "label": label or path.stem,
        "kind": kind,
    }


def _pair_record_from_row(
    *,
    row: dict[str, Any],
    path: Path,
    label: str | None,
    kind: str,
) -> dict[str, Any]:
    pair_idx = _as_int(row.get("pair_idx", row.get("pair_index")))
    if pair_idx is None:
        raise ValueError(f"{path} row is missing integer pair_idx")
    pose, seg, component = _score_terms(row)
    tags = row.get("suggested_lane_tags")
    return {
        "pair_idx": pair_idx,
        "priority": _as_float(row.get("priority")),
        "priority_basis": row.get("priority_basis"),
        "dominant_component": row.get("dominant_component"),
        "axis_dominant_component": row.get("axis_dominant_component"),
        "pose_score_contribution": pose,
        "seg_score_contribution": seg,
        "component_score_no_rate": component,
        "byte_equivalent_component_mass": _as_float(row.get("byte_equivalent_component_mass")),
        "suggested_lane_tags": sorted(str(tag) for tag in tags) if isinstance(tags, list) else [],
        "sources": [_pair_source(path, label, kind)],
    }


def _merge_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["pair_idx"]), []).append(record)

    merged: list[dict[str, Any]] = []
    for pair_idx in sorted(grouped):
        rows = grouped[pair_idx]
        priority_values = [
            value
            for value in (_as_float(row.get("priority")) for row in rows)
            if value is not None
        ]
        component = max(float(row["component_score_no_rate"]) for row in rows)
        priority = max(priority_values) if priority_values else component
        tags = sorted({tag for row in rows for tag in row.get("suggested_lane_tags", [])})
        sources = sorted(
            {json.dumps(source, sort_keys=True) for row in rows for source in row["sources"]}
        )
        pose = max(float(row["pose_score_contribution"]) for row in rows)
        seg = max(float(row["seg_score_contribution"]) for row in rows)
        dominant_values = [
            str(row["dominant_component"])
            for row in rows
            if row.get("dominant_component") is not None
        ]
        axis_values = [
            str(row["axis_dominant_component"])
            for row in rows
            if row.get("axis_dominant_component") is not None
        ]
        merged.append(
            {
                "pair_idx": pair_idx,
                "priority": priority,
                "priority_basis": "hardpair_priority_or_component_tail",
                "dominant_component": dominant_values[0] if dominant_values else ("pose" if pose >= seg else "seg"),
                "axis_dominant_component": axis_values[0] if axis_values else None,
                "pose_score_contribution": pose,
                "seg_score_contribution": seg,
                "component_score_no_rate": component,
                "byte_equivalent_component_mass": max(
                    _as_float(row.get("byte_equivalent_component_mass")) or 0.0
                    for row in rows
                ),
                "suggested_lane_tags": tags,
                "sources": [json.loads(text) for text in sources],
            }
        )
    return merged


def load_pair_records(
    *,
    pair_hitlist_paths: list[Path],
    pair_xray_paths: list[Path],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in pair_hitlist_paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a hard-pair JSON object")
        _check_false_authority(payload, path)
        rows = payload.get("hitlist")
        if not isinstance(rows, list):
            raise ValueError(f"{path} is missing hitlist list")
        label = str(payload.get("label") or path.stem)
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path} hitlist[{idx}] must be an object")
            records.append(
                _pair_record_from_row(row=row, path=path, label=label, kind="hardpair_hitlist")
            )

    for path in pair_xray_paths:
        payload = _load_json(path)
        if isinstance(payload, dict):
            _check_false_authority(payload, path)
            label = str(payload.get("label") or path.stem)
            rows = payload.get("rows")
        elif isinstance(payload, list):
            label = path.stem
            rows = payload
        else:
            raise ValueError(f"{path} must contain a pair XRay object or row list")
        if not isinstance(rows, list):
            raise ValueError(f"{path} is missing rows list")
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path} rows[{idx}] must be an object")
            records.append(
                _pair_record_from_row(row=row, path=path, label=label, kind="pair_component_xray")
            )

    if not records:
        raise ValueError("at least one --pair-hitlist or --pair-xray-json input is required")
    return _merge_pair_records(records)


_NUM = r"([-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def _regex_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _as_float(match.group(1))


def _regex_word(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower().replace("posenet", "pose").replace("segnet", "seg")


def _byte_gap(target_gaps: Any, axis: str) -> int | None:
    if not isinstance(target_gaps, dict):
        return None
    row = target_gaps.get(axis)
    if not isinstance(row, dict):
        return None
    return _as_int(row.get("byte_gap_if_components_unchanged"))


def _axis_context_from_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    components = payload.get("components")
    if isinstance(components, dict):
        delta = components.get("delta_cuda_minus_cpu")
        if isinstance(delta, dict):
            return {
                "path": path.as_posix(),
                "kind": "paired_cpu_cuda_axis_delta",
                "classification": payload.get("classification"),
                "dominant_component": components.get("dominant_score_delta_component"),
                "score_delta_byte_equivalent": _as_float(components.get("score_delta_byte_equivalent")),
                "contest_cpu_byte_gap": _byte_gap(payload.get("target_gaps"), "contest_cpu"),
                "contest_cuda_byte_gap": _byte_gap(payload.get("target_gaps"), "contest_cuda"),
                "score_delta_cuda_minus_cpu": _as_float(delta.get("score_delta_cuda_minus_cpu")),
                "seg_delta": _as_float(delta.get("seg_score_contribution_delta")),
                "pose_delta": _as_float(delta.get("pose_score_contribution_delta")),
                "rate_delta": _as_float(delta.get("rate_score_contribution_delta")),
            }
    paired = payload.get("paired_axis_xray")
    if isinstance(paired, dict):
        return {
            "path": path.as_posix(),
            "kind": "selector_cuda_transfer_calibration",
            "classification": paired.get("classification"),
            "dominant_component": paired.get("dominant_score_delta_component"),
            "score_delta_byte_equivalent": _as_float(paired.get("score_delta_byte_equivalent")),
            "contest_cpu_byte_gap": _byte_gap(paired.get("target_gaps"), "contest_cpu"),
            "contest_cuda_byte_gap": _byte_gap(paired.get("target_gaps"), "contest_cuda"),
            "score_delta_cuda_minus_cpu": _as_float(paired.get("cuda_minus_cpu_score_delta")),
            "seg_delta": None,
            "pose_delta": None,
            "rate_delta": None,
        }
    return {
        "path": path.as_posix(),
        "kind": "generic_axis_artifact",
        "classification": payload.get("classification"),
        "dominant_component": payload.get("dominant_score_delta_component"),
        "score_delta_byte_equivalent": None,
        "contest_cpu_byte_gap": None,
        "contest_cuda_byte_gap": None,
        "score_delta_cuda_minus_cpu": None,
        "seg_delta": None,
        "pose_delta": None,
        "rate_delta": None,
    }


def _axis_context_from_markdown(path: Path, text: str) -> dict[str, Any]:
    dominant = _regex_word(r"dominant(?: score-delta)? component:\s*`?([A-Za-z0-9_-]+)`?", text)
    if dominant is None and "PoseNet contribution" in text:
        dominant = "pose"
    return {
        "path": path.as_posix(),
        "kind": "paired_axis_markdown",
        "classification": _regex_word(r"(?:classification|verdict):\s*`?([A-Za-z0-9_-]+)`?", text),
        "dominant_component": dominant,
        "score_delta_byte_equivalent": _regex_float(r"score-delta byte equivalent:\s*`?" + _NUM, text)
        or _regex_float(r"byte-equivalent CUDA gap[^:\n]*:\s*`?" + _NUM, text),
        "contest_cpu_byte_gap": _regex_float(r"\|\s*contest-CPU\s*\|\s*[-+0-9.eE]+\s*\|\s*[-+0-9.eE]+\s*\|\s*" + _NUM, text),
        "contest_cuda_byte_gap": _regex_float(r"\|\s*contest-CUDA\s*\|\s*[-+0-9.eE]+\s*\|\s*[-+0-9.eE]+\s*\|\s*" + _NUM, text),
        "score_delta_cuda_minus_cpu": _regex_float(r"total(?: CUDA-minus-CPU)? score delta:\s*`?" + _NUM, text),
        "seg_delta": _regex_float(r"seg(?:net)? contribution delta:\s*`?" + _NUM, text),
        "pose_delta": _regex_float(r"pose(?:net)? contribution delta:\s*`?" + _NUM, text),
        "rate_delta": _regex_float(r"rate contribution delta:\s*`?" + _NUM, text),
    }


def load_axis_context(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        _check_false_authority(payload, path)
        return _axis_context_from_json(path, payload)
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return _axis_context_from_markdown(path, path.read_text(encoding="utf-8"))
    raise ValueError(f"{path} must be JSON or Markdown/TXT")


def load_evidence_note(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        **_path_record(path),
        "kind": "pr106_format0c_evidence_note",
        "contest_cuda_score": _regex_float(r"\|\s*`\[contest-CUDA\]`\s*\|(?:[^|]*\|){2}\s*`?" + _NUM, text),
        "contest_cpu_score": _regex_float(r"\|\s*`\[contest-CPU\]`\s*\|(?:[^|]*\|){2}\s*`?" + _NUM, text),
        "archive_bytes": _regex_float(r"Archive bytes:\s*`?" + _NUM, text),
        "contains_paired_eval_requirement": "Do not infer either axis from the other" in text
        or "paired CPU/CUDA" in text,
    }


def _selected_payload_bytes(n_pairs: int) -> int:
    if n_pairs <= 0:
        return 0
    return PR100_RAW_HEADER_BYTES + PR100_RAW_BYTES_PER_SELECTED_PAIR * n_pairs


def select_candidate_pairs(
    pair_records: list[dict[str, Any]],
    *,
    byte_budget: int,
    max_pairs: int,
) -> list[dict[str, Any]]:
    if byte_budget < 0:
        raise ValueError(f"byte budget must be non-negative, got {byte_budget}")
    if max_pairs < 0:
        raise ValueError(f"max pairs must be non-negative, got {max_pairs}")
    ordered = sorted(
        pair_records,
        key=lambda row: (
            -float(row["priority"]),
            -float(row["component_score_no_rate"]),
            int(row["pair_idx"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        if len(selected) >= max_pairs:
            break
        next_count = len(selected) + 1
        if _selected_payload_bytes(next_count) > byte_budget:
            break
        selected.append(row)
    return selected


def _source_lessons() -> list[dict[str, Any]]:
    return [
        {
            "source": "PR100 hnerv_lc_v2",
            "lesson": "latent_correction_sidecar",
            "planner_use": "one selected per-pair latent dim plus int8 delta is the semantic repair unit",
        },
        {
            "source": "PR101 hnerv_ft_microcodec",
            "lesson": "fixed_microcodec_discipline",
            "planner_use": "fixed decoder/latent/sidecar byte sections and deterministic ordering are the reference discipline",
        },
        {
            "source": "PR103 hnerv_lc_ac",
            "lesson": "arithmetic_repack_after_component_movement",
            "planner_use": "range/arithmetic coding is a follow-up byte pass after CUDA component repair exists",
        },
        {
            "source": "PR106 format0C paired exact eval",
            "lesson": "cuda_pose_signal_not_cpu_frontier",
            "planner_use": "mine CUDA-favorable PoseNet signal while preserving paired CPU/CUDA axis separation",
        },
    ]


def _candidate_pair_plan(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "priority_rank": rank,
        "pair_idx": int(row["pair_idx"]),
        "priority": row["priority"],
        "priority_basis": row["priority_basis"],
        "dominant_component": row["dominant_component"],
        "axis_dominant_component": row["axis_dominant_component"],
        "pose_score_contribution": row["pose_score_contribution"],
        "seg_score_contribution": row["seg_score_contribution"],
        "component_score_no_rate": row["component_score_no_rate"],
        "planned_mode": "cuda_latent_dim28_delta_q_pm2_grid_then_format0c_selection",
        "latent_dim_count": LATENT_DIM,
        "delta_q_values": list(DELTA_Q_VALUES),
        "probe_modes_per_pair": LATENT_DIM * len(DELTA_Q_VALUES),
        "estimated_selected_correction_payload_bytes": PR100_RAW_BYTES_PER_SELECTED_PAIR,
        "suggested_lane_tags": row["suggested_lane_tags"],
        "sources": row["sources"],
    }


def build_plan(
    *,
    pair_records: list[dict[str, Any]],
    axis_contexts: list[dict[str, Any]],
    source_archive: Path,
    byte_budget: int,
    max_pairs: int,
    label: str,
    pair_hitlist_paths: list[Path],
    pair_xray_paths: list[Path],
    paired_axis_artifacts: list[Path],
    pr106_format0c_ledgers: list[Path],
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    source_archive = Path(source_archive)
    source_archive_record = _path_record(source_archive)
    if expected_source_sha256 is not None and source_archive_record["sha256"] != expected_source_sha256:
        raise ValueError(
            f"source archive SHA mismatch: got {source_archive_record['sha256']}, "
            f"expected {expected_source_sha256}"
        )

    selected = select_candidate_pairs(
        pair_records,
        byte_budget=byte_budget,
        max_pairs=max_pairs,
    )
    selected_payload_bytes = _selected_payload_bytes(len(selected))
    candidate_pairs = [
        _candidate_pair_plan(row, rank)
        for rank, row in enumerate(selected, start=1)
    ]

    input_records = {
        "pair_hitlists": [_path_record(path) for path in pair_hitlist_paths],
        "pair_xrays": [_path_record(path) for path in pair_xray_paths],
        "paired_axis_artifacts": [_path_record(path) for path in paired_axis_artifacts],
        "pr106_format0c_ledgers": [load_evidence_note(path) for path in pr106_format0c_ledgers],
        "source_archive": source_archive_record,
    }
    state_for_hash = {
        "schema": SCHEMA,
        "label": label,
        "byte_budget": byte_budget,
        "max_pairs": max_pairs,
        "inputs": input_records,
        "pair_records": pair_records,
        "axis_contexts": axis_contexts,
        "candidate_pairs": candidate_pairs,
    }
    from_state_hash = _sha256_bytes(
        json.dumps(state_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:16]

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "from_state_hash": from_state_hash,
        "label": label,
        "authority": {
            **FALSE_AUTHORITY,
            "notes": [
                "planner_only_from_xray_and_public_frontier_intake_artifacts",
                "no_archive_mutation_performed",
                "not_a_score_claim",
                "not_promotion_or_dispatch_authority",
                "paired_cpu_cuda_exact_eval_required_before_frontier_language",
            ],
        },
        "source_lessons": _source_lessons(),
        "inputs": input_records,
        "axis_contexts": axis_contexts,
        "selection_policy": {
            "sort_order": [
                "priority descending",
                "component_score_no_rate descending",
                "pair_idx ascending",
            ],
            "byte_budget_mode": "pr100_raw_dim_delta_planner_payload",
            "byte_formula": "0 if n=0 else 2 + 2*n",
            "latent_dim_count": LATENT_DIM,
            "delta_q_values": list(DELTA_Q_VALUES),
            "probe_modes_per_pair": LATENT_DIM * len(DELTA_Q_VALUES),
            "max_pairs": max_pairs,
        },
        "byte_accounting": {
            "archive_mutation_performed": False,
            "source_archive_bytes": source_archive_record["bytes"],
            "source_archive_sha256": source_archive_record["sha256"],
            "planner_byte_budget": byte_budget,
            "selected_pair_count": len(selected),
            "selected_pair_payload_bytes": selected_payload_bytes,
            "byte_budget_remaining": byte_budget - selected_payload_bytes,
            "rate_term_delta_if_added_without_replacing_bytes": (
                25.0 * selected_payload_bytes / ARCHIVE_BYTES_DENOMINATOR
            ),
            "pr100_raw_header_bytes": PR100_RAW_HEADER_BYTES if selected else 0,
            "pr100_raw_bytes_per_selected_pair": PR100_RAW_BYTES_PER_SELECTED_PAIR,
            "pr101_fixed_microcodec_reference_sidecar_bytes": PR101_FIXED_MICROCODEC_SIDECAR_BYTES,
            "pr106_format0c_exact_radix_reference_sidecar_bytes": PR106_FORMAT0C_EXACT_RADIX_SIDECAR_BYTES,
            "materialized_archive_bytes": None,
            "materialized_archive_sha256": None,
        },
        "candidate_pairs": candidate_pairs,
        "materialization": {
            "supported": False,
            "placeholder_flag": "--materialize",
            "placeholder_fails_closed": True,
            "archive_mutation_too_risky_for_this_hook": True,
            "blockers": [
                "cuda_in_loop_score_table_missing_for_selected_pair_modes",
                "runtime_decode_apply_mutation_not_implemented",
                "full_frame_same_runtime_parity_missing",
                "byte_closed_archive_not_emitted",
                "paired_exact_contest_cuda_and_contest_cpu_eval_missing",
            ],
        },
        "required_next_proofs": [
            "build CUDA-in-loop per-pair latent score table over the selected pair/mode grid",
            "materialize a byte-closed archive only after runtime decode/apply is implemented",
            "prove targeted correction bytes are consumed and produce intended frame changes",
            "run paired exact [contest-CUDA] and [contest-CPU] auth eval on the same archive/runtime",
            "allow frontier language only after both axes have custody and component recomputation",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    authority = plan["authority"]
    byte_accounting = plan["byte_accounting"]
    lines = [
        "# PR106 CUDA Latent-Correction Probe Plan",
        "",
        f"- schema: `{plan['schema']}`",
        f"- label: `{plan['label']}`",
        f"- from_state_hash: `{plan['from_state_hash']}`",
        f"- score_claim: `{str(authority['score_claim']).lower()}`",
        f"- promotion_eligible: `{str(authority['promotion_eligible']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(authority['ready_for_exact_eval_dispatch']).lower()}`",
        f"- frontier_language_allowed: `{str(authority['frontier_language_allowed']).lower()}`",
        "- paired CPU/CUDA exact eval required before frontier language: `true`",
        "",
        "## Byte Accounting",
        "",
        "| field | value |",
        "|---|---:|",
        f"| source archive bytes | {byte_accounting['source_archive_bytes']} |",
        f"| planner byte budget | {byte_accounting['planner_byte_budget']} |",
        f"| selected pairs | {byte_accounting['selected_pair_count']} |",
        f"| selected pair payload bytes | {byte_accounting['selected_pair_payload_bytes']} |",
        f"| byte budget remaining | {byte_accounting['byte_budget_remaining']} |",
        "",
        "## Candidate Pairs",
        "",
        "| rank | pair | priority | component | mode bytes | mode |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in plan["candidate_pairs"]:
        lines.append(
            f"| {row['priority_rank']} | {row['pair_idx']} | "
            f"{row['priority']:.12g} | {row['component_score_no_rate']:.12g} | "
            f"{row['estimated_selected_correction_payload_bytes']} | "
            f"`{row['planned_mode']}` |"
        )
    lines.extend(
        [
            "",
            "## Materialization",
            "",
            "- supported: `false`",
            "- placeholder flag: `--materialize`",
            "- placeholder fails closed: `true`",
            "",
            "## Required Next Proofs",
            "",
        ]
    )
    for proof in plan["required_next_proofs"]:
        lines.append(f"- {proof}")
    lines.extend(
        [
            "",
            "_Tag_: `[planner: false-authority PR106 CUDA latent-correction probe]`. "
            "This plan is not a score claim, not promotion evidence, and not dispatch authority.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-hitlist",
        "--hardpair-hitlist",
        action="append",
        type=Path,
        default=[],
        help="Path to hardpair_hitlist.json. Repeatable.",
    )
    parser.add_argument(
        "--pair-xray-json",
        action="append",
        type=Path,
        default=[],
        help="Path to pair_component_xray.json or JSON row list. Repeatable.",
    )
    parser.add_argument(
        "--paired-axis-artifact",
        "--axis-delta",
        action="append",
        type=Path,
        default=[],
        help="Paired CPU/CUDA axis artifact in JSON or Markdown. Repeatable.",
    )
    parser.add_argument(
        "--pr106-format0c-ledger",
        action="append",
        type=Path,
        default=[],
        help="PR106 format0C paired CPU/CUDA ledger Markdown. Repeatable.",
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--byte-budget", type=int, default=PR106_FORMAT0C_EXACT_RADIX_SIDECAR_BYTES)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--label", default="pr106_cuda_latent_correction_probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Reserved future archive mutation path; currently fails closed.",
    )
    return parser


def _rebuild_command(args: argparse.Namespace) -> str:
    parts = [".venv/bin/python", TOOL]
    for path in args.pair_hitlist:
        parts.extend(["--pair-hitlist", str(path)])
    for path in args.pair_xray_json:
        parts.extend(["--pair-xray-json", str(path)])
    for path in args.paired_axis_artifact:
        parts.extend(["--paired-axis-artifact", str(path)])
    for path in args.pr106_format0c_ledger:
        parts.extend(["--pr106-format0c-ledger", str(path)])
    parts.extend(["--source-archive", str(args.source_archive)])
    if args.expected_source_sha256:
        parts.extend(["--expected-source-sha256", args.expected_source_sha256])
    parts.extend(["--byte-budget", str(args.byte_budget)])
    parts.extend(["--max-pairs", str(args.max_pairs)])
    parts.extend(["--label", args.label])
    parts.extend(["--output-dir", str(args.output_dir)])
    return " \\\n  ".join(shlex.quote(part) for part in parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.materialize:
        print(
            "ERROR: --materialize is a reserved future archive-mutation path and "
            "currently fails closed; emit a planner-only report instead.",
            file=sys.stderr,
        )
        return 2
    try:
        pair_records = load_pair_records(
            pair_hitlist_paths=args.pair_hitlist,
            pair_xray_paths=args.pair_xray_json,
        )
        axis_contexts = [load_axis_context(path) for path in args.paired_axis_artifact]
        plan = build_plan(
            pair_records=pair_records,
            axis_contexts=axis_contexts,
            source_archive=args.source_archive,
            byte_budget=args.byte_budget,
            max_pairs=args.max_pairs,
            label=args.label,
            pair_hitlist_paths=args.pair_hitlist,
            pair_xray_paths=args.pair_xray_json,
            paired_axis_artifacts=args.paired_axis_artifact,
            pr106_format0c_ledgers=args.pr106_format0c_ledger,
            expected_source_sha256=args.expected_source_sha256,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "pr106_cuda_latent_correction_probe_plan.json"
    md_path = args.output_dir / "pr106_cuda_latent_correction_probe_plan.md"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    (args.output_dir / "rebuild_command.txt").write_text(_rebuild_command(args), encoding="utf-8")
    print(f"[pr106-cuda-latent-correction-probe] wrote {json_path}")
    print(f"[pr106-cuda-latent-correction-probe] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
