#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert CodecOp entropy-floor reports into planning-only atom rows.

This adapter bridges PR101 entropy-floor artifacts into the meta-Lagrangian /
Pareto planning surface without creating a score claim or dispatch-ready row.
The emitted rows are JSONL atom dictionaries accepted by
``tac.optimization.meta_lagrangian_allocator.build_atom_ledger`` and are marked
as proxy/planning evidence so exact-eval dispatch remains fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_NAME = "tools/codec_op_entropy_floor_to_ledger.py"
SUPPORTED_SCHEMAS = {
    "pr101_compression_floor_ladder.v2",
    "pr101_context_transform_floor_probe.v1",
}
DEFAULT_BLOCKERS = (
    "codec_op_entropy_floor_atom_is_planning_only",
    "entropy_floor_is_model_class_lower_bound_not_candidate_archive",
    "requires_joint_admm_codecop_materialization",
    "requires_byte_closed_archive_manifest_before_dispatch",
    "requires_noop_or_roundtrip_decode_proof",
    "requires_exact_cuda_auth_eval",
    "not_promotion_eligible",
)
DEFAULT_INTERACTION_ASSUMPTIONS = (
    "rate_only_entropy_floor_until_codec_op_materialized",
    "joint_admm_may_use_byte_delta_as_lower_bound_not_dispatch_candidate",
    "conflicts_with_materialized_codec_for_same_state_dict_without_supersession",
)
FLOOR_MODEL_KEYS = ("iid", "markov1", "markov2")


class EntropyFloorAdapterError(ValueError):
    """Raised when an entropy-floor report cannot be adapted fail-closed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _display_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    return str(path)


def _slug(value: Any, *, fallback: str = "row") -> str:
    text = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return slug[:120] or fallback


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique_ordered(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _optional_int(value: Any) -> int | None:
    if not _is_finite_number(value):
        return None
    number = float(value)
    if not number.is_integer():
        return None
    return int(number)


def _optional_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _load_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EntropyFloorAdapterError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise EntropyFloorAdapterError(f"{path}: expected JSON object report")
    schema = str(payload.get("schema") or "")
    if schema not in SUPPORTED_SCHEMAS:
        raise EntropyFloorAdapterError(f"{path}: unsupported schema {schema!r}")
    return payload


def _baseline_archive_bytes(payload: Mapping[str, Any]) -> tuple[int | None, str]:
    direct_keys = (
        "comparison_brotli_optuna_archive_bytes",
        "comparison_brotli_optuna_optimum_bytes",
        "reference_brotli_optuna_archive_bytes",
    )
    for key in direct_keys:
        value = _optional_int(payload.get(key))
        if value is not None:
            return value, key
    for row in payload.get("empirical_encoders") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("name") == "brotli_optuna_optimum":
            value = _optional_int(row.get("bytes_archive"))
            if value is not None:
                return value, "empirical_encoders.brotli_optuna_optimum.bytes_archive"
    return None, ""


def _planning_evidence_grade(payload: Mapping[str, Any]) -> str:
    grade = str(payload.get("evidence_grade") or "prediction").strip().lower()
    if not grade:
        grade = "prediction"
    if "planning" in grade:
        return grade
    return f"{grade}_planning"


def _confidence_for_source(payload: Mapping[str, Any], *, model_table_overhead_included: bool | None) -> float:
    grade = str(payload.get("evidence_grade") or "").strip().lower()
    if grade == "empirical":
        confidence = 0.35
    elif grade == "derivation":
        confidence = 0.25
    else:
        confidence = 0.10
    if model_table_overhead_included is False:
        confidence = min(confidence, 0.20)
    return confidence


def _source_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = [*DEFAULT_BLOCKERS, *_string_list(payload.get("dispatch_blockers"))]
    if payload.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("source_ready_for_exact_eval_dispatch_true_refused_by_adapter")
    if payload.get("score_claim") is True:
        blockers.append("source_score_claim_true_refused_by_adapter")
    if payload.get("charged_bits_changed") is True:
        blockers.append("source_charged_bits_changed_true_refused_by_adapter")
    if payload.get("score_affecting_payload_changed") is True:
        blockers.append("source_score_affecting_payload_changed_true_refused_by_adapter")
    return _unique_ordered(blockers)


def _base_atom(
    *,
    source_path: Path,
    source_sha256: str,
    payload: Mapping[str, Any],
    row_index: int,
    row_kind: str,
    row_label: str,
    family: str,
    pareto_scope: str,
    byte_delta: int,
    target_archive_bytes: int | None,
    target_payload_bytes: int | None,
    baseline_archive_bytes: int | None,
    baseline_archive_bytes_source: str,
    model_table_overhead_included: bool | None,
    repo_root: Path | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_schema = str(payload.get("schema") or "")
    input_sha = str(payload.get("input_state_dict_sha256") or "")
    blockers = _source_blockers(payload)
    if baseline_archive_bytes is None:
        blockers.append("missing_comparison_brotli_optuna_archive_bytes")
    if target_archive_bytes is None:
        blockers.append("missing_target_archive_bytes_estimate")
    if model_table_overhead_included is False:
        blockers.append("model_table_overhead_not_charged")
    atom = {
        "adapter": "codec_op_entropy_floor_to_ledger",
        "paradigm": "joint_admm_codec_op_entropy_floor",
        "atom_id": f"codec_op_entropy_floor:{_slug(source_path.stem)}:{_slug(row_label)}",
        "family": family,
        "family_group": "joint_admm_codec_op_entropy_floor",
        "pareto_scope": pareto_scope,
        "byte_delta": byte_delta,
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "confidence": _confidence_for_source(
            payload,
            model_table_overhead_included=model_table_overhead_included,
        ),
        "evidence_grade": _planning_evidence_grade(payload),
        "source_evidence_grade": str(payload.get("evidence_grade") or ""),
        "source_evidence_semantics": str(payload.get("evidence_semantics") or ""),
        "target_modes": ["planning_only"],
        "deployment_target": "desktop_research",
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "planning_only": True,
        "proxy_row": True,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "dispatch_attempted": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": _unique_ordered(blockers),
        "promotion_blockers": _unique_ordered(blockers),
        "interaction_assumptions": list(DEFAULT_INTERACTION_ASSUMPTIONS),
        "evidence_source_path": _display_path(source_path, repo_root),
        "evidence_source_sha256": source_sha256,
        "source_record_index": row_index,
        "source_record_kind": row_kind,
        "source_schema": source_schema,
        "source_tool": str(payload.get("tool") or ""),
        "input_state_dict": str(payload.get("input_state_dict") or ""),
        "input_state_dict_sha256": input_sha,
        "baseline_archive_bytes": baseline_archive_bytes,
        "baseline_archive_bytes_source": baseline_archive_bytes_source,
        "target_archive_bytes_estimate": target_archive_bytes,
        "target_payload_bytes_estimate": target_payload_bytes,
        "source_artifact_bytes": target_archive_bytes,
        "model_table_overhead_included": model_table_overhead_included,
        "readiness": {
            "ready_for_exact_eval_dispatch": False,
            "dispatchable": False,
            "score_claim": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "promotion_eligible": False,
            "promotable": False,
            "planning_only": True,
            "proxy_row": True,
        },
    }
    if extra:
        atom.update(dict(extra))
    return atom


def _floor_rows(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    baseline, baseline_source = _baseline_archive_bytes(payload)
    input_sha = str(payload.get("input_state_dict_sha256") or "unknown")
    rows: list[dict[str, Any]] = []
    for idx, floor in enumerate(payload.get("provable_floors") or []):
        if not isinstance(floor, Mapping):
            continue
        target_archive_bytes = _optional_int(floor.get("bytes_archive"))
        target_payload_bytes = _optional_int(floor.get("bytes_payload"))
        byte_delta = (
            target_archive_bytes - baseline
            if target_archive_bytes is not None and baseline is not None
            else 0
        )
        model_name = str(floor.get("name") or f"floor_{idx}")
        rows.append(
            _base_atom(
                source_path=source_path,
                source_sha256=source_sha256,
                payload=payload,
                row_index=idx,
                row_kind="provable_floor",
                row_label=model_name,
                family="codec_op_provable_entropy_floor",
                pareto_scope=f"pr101_decoder_entropy_floor:{input_sha[:12]}",
                byte_delta=byte_delta,
                target_archive_bytes=target_archive_bytes,
                target_payload_bytes=target_payload_bytes,
                baseline_archive_bytes=baseline,
                baseline_archive_bytes_source=baseline_source,
                model_table_overhead_included=bool(
                    floor.get("model_table_overhead_included") is True
                ),
                repo_root=repo_root,
                extra={
                    "entropy_floor_name": model_name,
                    "entropy_floor_bits": _optional_float(floor.get("bits")),
                    "entropy_floor_description": str(floor.get("description") or ""),
                },
            )
        )
    return rows


def _context_transform_rows(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    baseline, baseline_source = _baseline_archive_bytes(payload)
    input_sha = str(payload.get("input_state_dict_sha256") or "unknown")
    rows: list[dict[str, Any]] = []
    source_index = 0
    for transform in payload.get("transforms") or []:
        if not isinstance(transform, Mapping):
            continue
        transform_name = str(transform.get("transform") or f"transform_{source_index}")
        for model_name in FLOOR_MODEL_KEYS:
            target_archive_bytes = _optional_int(transform.get(f"{model_name}_archive_bytes"))
            target_payload_bytes = _optional_int(transform.get(f"{model_name}_payload_bytes"))
            explicit_delta = _optional_int(
                transform.get(f"delta_{model_name}_archive_vs_brotli_optuna")
            )
            byte_delta = (
                explicit_delta
                if explicit_delta is not None
                else (
                    target_archive_bytes - baseline
                    if target_archive_bytes is not None and baseline is not None
                    else 0
                )
            )
            rows.append(
                _base_atom(
                    source_path=source_path,
                    source_sha256=source_sha256,
                    payload=payload,
                    row_index=source_index,
                    row_kind="context_transform_floor",
                    row_label=f"{transform_name}:{model_name}",
                    family="codec_op_context_transform_entropy_floor",
                    pareto_scope=f"pr101_context_transform_entropy_floor:{input_sha[:12]}",
                    byte_delta=byte_delta,
                    target_archive_bytes=target_archive_bytes,
                    target_payload_bytes=target_payload_bytes,
                    baseline_archive_bytes=baseline,
                    baseline_archive_bytes_source=baseline_source,
                    model_table_overhead_included=False,
                    repo_root=repo_root,
                    extra={
                        "context_transform": transform_name,
                        "entropy_model": model_name,
                        "invertible_fixed_transform": bool(
                            transform.get("invertible_fixed_transform") is True
                        ),
                        "metadata_bytes_charged": _optional_int(
                            transform.get("metadata_bytes_charged")
                        ),
                        "n_streams": _optional_int(transform.get("n_streams")),
                        "n_symbols_total": _optional_int(transform.get("n_symbols_total")),
                    },
                )
            )
            source_index += 1
    return rows


def atoms_from_report(
    report_path: Path,
    *,
    repo_root: Path | None = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Return planning-only atom rows for one supported entropy-floor report."""

    payload = _load_report(report_path)
    source_sha256 = _sha256_file(report_path)
    schema = str(payload["schema"])
    if schema == "pr101_compression_floor_ladder.v2":
        return _floor_rows(
            payload,
            source_path=report_path,
            source_sha256=source_sha256,
            repo_root=repo_root,
        )
    if schema == "pr101_context_transform_floor_probe.v1":
        return _context_transform_rows(
            payload,
            source_path=report_path,
            source_sha256=source_sha256,
            repo_root=repo_root,
        )
    raise EntropyFloorAdapterError(f"{report_path}: unsupported schema {schema!r}")


def build_atom_rows(
    input_paths: Sequence[Path],
    *,
    repo_root: Path | None = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Adapt all input reports and reject duplicate atom IDs."""

    rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows.extend(atoms_from_report(path, repo_root=repo_root))
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        atom_id = str(row["atom_id"])
        if atom_id in seen:
            duplicates.add(atom_id)
        seen.add(atom_id)
    if duplicates:
        raise EntropyFloorAdapterError(
            "duplicate atom_id values: " + ", ".join(sorted(duplicates))
        )
    return rows


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _policy_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    planning_only_count = sum(1 for row in rows if row.get("planning_only") is True)
    proxy_row_count = sum(1 for row in rows if row.get("proxy_row") is True)
    score_claim_count = sum(1 for row in rows if row.get("score_claim") is True)
    score_affecting_payload_changed_count = sum(
        1 for row in rows if row.get("score_affecting_payload_changed") is True
    )
    charged_bits_changed_count = sum(1 for row in rows if row.get("charged_bits_changed") is True)
    dispatchable_count = sum(1 for row in rows if row.get("dispatchable") is True)
    ready_count = sum(1 for row in rows if row.get("ready_for_exact_eval_dispatch") is True)
    promotion_eligible_count = sum(
        1
        for row in rows
        if row.get("promotion_eligible") is True or row.get("promotable") is True
    )
    return {
        "row_count": row_count,
        "planning_only_atom_count": planning_only_count,
        "proxy_row_count": proxy_row_count,
        "score_claim_atom_count": score_claim_count,
        "score_affecting_payload_changed_atom_count": score_affecting_payload_changed_count,
        "charged_bits_changed_atom_count": charged_bits_changed_count,
        "dispatchable_atom_count": dispatchable_count,
        "ready_for_exact_eval_dispatch_atom_count": ready_count,
        "promotion_eligible_atom_count": promotion_eligible_count,
        "all_rows_planning_only": planning_only_count == row_count,
        "all_rows_proxy": proxy_row_count == row_count,
        "all_rows_non_score_claim": score_claim_count == 0,
        "all_rows_non_score_affecting_payload_changed": score_affecting_payload_changed_count == 0,
        "all_rows_no_charged_bits_changed": charged_bits_changed_count == 0,
        "all_rows_non_dispatchable": dispatchable_count == 0,
        "all_rows_not_ready_for_exact_eval_dispatch": ready_count == 0,
        "all_rows_non_promotable": promotion_eligible_count == 0,
        "fail_closed": (
            planning_only_count == row_count
            and proxy_row_count == row_count
            and score_claim_count == 0
            and score_affecting_payload_changed_count == 0
            and charged_bits_changed_count == 0
            and dispatchable_count == 0
            and ready_count == 0
            and promotion_eligible_count == 0
        ),
    }


def _row_label(row: Mapping[str, Any]) -> str:
    if row.get("context_transform") and row.get("entropy_model"):
        return f"{row['context_transform']}:{row['entropy_model']}"
    if row.get("entropy_floor_name"):
        return str(row["entropy_floor_name"])
    return str(row.get("atom_id") or "row")


def _best_negative_byte_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    negative_rows = sorted(
        (row for row in rows if int(row.get("byte_delta", 0)) < 0),
        key=lambda row: (int(row.get("byte_delta", 0)), str(row.get("atom_id") or "")),
    )
    out: list[dict[str, Any]] = []
    for row in negative_rows[:limit]:
        out.append(
            {
                "atom_id": str(row.get("atom_id") or ""),
                "label": _row_label(row),
                "family": str(row.get("family") or ""),
                "byte_delta": int(row.get("byte_delta", 0)),
                "target_archive_bytes_estimate": row.get("target_archive_bytes_estimate"),
                "target_payload_bytes_estimate": row.get("target_payload_bytes_estimate"),
                "baseline_archive_bytes": row.get("baseline_archive_bytes"),
                "planning_only": row.get("planning_only") is True,
                "proxy_row": row.get("proxy_row") is True,
                "dispatchable": row.get("dispatchable") is True,
                "ready_for_exact_eval_dispatch": row.get("ready_for_exact_eval_dispatch") is True,
                "promotion_eligible": row.get("promotion_eligible") is True,
            }
        )
    return out


def _input_report_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_paths: Sequence[Path],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in input_paths:
        display_path = _display_path(path, repo_root)
        source_rows = [row for row in rows if row.get("evidence_source_path") == display_path]
        schemas = sorted({str(row.get("source_schema") or "") for row in source_rows if row.get("source_schema")})
        sha256s = sorted(
            {str(row.get("evidence_source_sha256") or "") for row in source_rows if row.get("evidence_source_sha256")}
        )
        reports.append(
            {
                "path": display_path,
                "sha256": sha256s[0] if len(sha256s) == 1 else "",
                "schemas": schemas,
                "atom_count": len(source_rows),
                "negative_byte_delta_atom_count": sum(
                    1 for row in source_rows if int(row.get("byte_delta", 0)) < 0
                ),
            }
        )
    return reports


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    repo_root: Path | None = REPO_ROOT,
) -> dict[str, Any]:
    negative_count = sum(1 for row in rows if int(row.get("byte_delta", 0)) < 0)
    policy = _policy_counts(rows)
    return {
        "schema": "codec_op_entropy_floor_to_ledger.summary.v1",
        "tool": TOOL_NAME,
        "generated_at_unix_seconds": int(time.time()),
        "planning_only": True,
        "proxy_rows": True,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "promotion_eligible": False,
        "promotable": False,
        "dispatch_attempted": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "atom_count": len(rows),
        "negative_byte_delta_atom_count": negative_count,
        "input_paths": [_display_path(path, repo_root) for path in input_paths],
        "input_reports": _input_report_summaries(rows, input_paths=input_paths, repo_root=repo_root),
        "output_path": _display_path(output_path, repo_root),
        "dispatch_blockers": list(DEFAULT_BLOCKERS),
        "promotion_blockers": list(DEFAULT_BLOCKERS),
        "policy": policy,
        "policy_review": {
            "remote_gpu_dispatch": "not_applicable_no_training_eval_or_remote_gpu_job_dispatched",
            "lane_dispatch_claim_required": False,
            "rows_are_score_evidence": False,
            "rows_are_dispatch_candidates": False,
            "rows_are_promotion_candidates": False,
        },
        "best_negative_byte_deltas": _best_negative_byte_deltas(rows),
        "families": sorted({str(row.get("family")) for row in rows}),
        "pareto_scopes": sorted({str(row.get("pareto_scope")) for row in rows}),
    }


def build_ledger_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a durable human-readable ledger for the adapter run."""

    lines = [
        "# CodecOp Entropy-Floor Ledger Adapter - Worker G - 2026-05-07",
        "",
        "## Scope",
        "",
        "This ledger records the Worker G hardening pass for converting PR101 CodecOp "
        "entropy-floor reports into meta-Lagrangian planning atoms. It is not a score "
        "claim, dispatch claim, promotion record, or candidate archive manifest.",
        "",
        "## Inputs",
        "",
        "| path | sha256 | atoms | negative deltas | schemas |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for report in summary.get("input_reports") or []:
        if not isinstance(report, Mapping):
            continue
        sha = str(report.get("sha256") or "")
        schemas = ", ".join(str(schema) for schema in report.get("schemas") or [])
        lines.append(
            "| "
            f"{report.get('path', '')} | "
            f"{sha[:16]}... | "
            f"{report.get('atom_count', 0)} | "
            f"{report.get('negative_byte_delta_atom_count', 0)} | "
            f"{schemas} |"
        )

    policy = summary.get("policy") or {}
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- JSONL atom rows: `{summary.get('output_path', '')}`",
            f"- Atom count: `{summary.get('atom_count', 0)}`",
            f"- Negative byte-delta atom count: `{summary.get('negative_byte_delta_atom_count', 0)}`",
            "",
            "## Policy",
            "",
            f"- planning-only rows: `{policy.get('planning_only_atom_count', 0)}`",
            f"- proxy rows: `{policy.get('proxy_row_count', 0)}`",
            f"- score-claim rows: `{policy.get('score_claim_atom_count', 0)}`",
            f"- score-affecting payload changed rows: `{policy.get('score_affecting_payload_changed_atom_count', 0)}`",
            f"- charged bits changed rows: `{policy.get('charged_bits_changed_atom_count', 0)}`",
            f"- dispatchable rows: `{policy.get('dispatchable_atom_count', 0)}`",
            f"- exact-eval-ready rows: `{policy.get('ready_for_exact_eval_dispatch_atom_count', 0)}`",
            f"- promotion-eligible rows: `{policy.get('promotion_eligible_atom_count', 0)}`",
            f"- fail-closed policy: `{policy.get('fail_closed') is True}`",
            "",
            "No training, eval, or remote-GPU job was dispatched by this adapter run; no "
            "lane-dispatch claim is required for this planning-only ledger emission.",
            "",
            "## Best Negative Byte Deltas",
            "",
            "| atom | label | byte_delta | target_archive_bytes | policy |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("best_negative_byte_deltas") or []:
        if not isinstance(row, Mapping):
            continue
        policy_bits = "planning_only/proxy/non_dispatchable/non_promotable"
        lines.append(
            "| "
            f"`{row.get('atom_id', '')}` | "
            f"{row.get('label', '')} | "
            f"{row.get('byte_delta', 0)} | "
            f"{row.get('target_archive_bytes_estimate', '')} | "
            f"{policy_bits} |"
        )

    lines.extend(["", "## Blockers", ""])
    for blocker in summary.get("dispatch_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "## Row Custody Check",
            "",
            f"- rows inspected: `{len(rows)}`",
            f"- all rows non-promotable: `{policy.get('all_rows_non_promotable') is True}`",
            f"- all rows non-dispatchable: `{policy.get('all_rows_non_dispatchable') is True}`",
            f"- all rows not exact-eval-ready: `{policy.get('all_rows_not_ready_for_exact_eval_dispatch') is True}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Supported entropy-floor report JSON. May be passed more than once.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL atom ledger output path.")
    parser.add_argument(
        "--atoms-json-output",
        type=Path,
        default=None,
        help="Optional JSON-list output for tools/build_cross_paradigm_atom_ledger.py --atoms-json.",
    )
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary output.")
    parser.add_argument(
        "--ledger-md-output",
        type=Path,
        default=None,
        help="Optional durable markdown ledger summary output.",
    )
    parser.add_argument("--append", action="store_true", help="Append rows to --output instead of replacing it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_atom_rows(args.input, repo_root=REPO_ROOT)
    write_jsonl(rows, args.output, append=args.append)
    if args.atoms_json_output is not None:
        args.atoms_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.atoms_json_output.write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary = build_summary(
        rows,
        input_paths=args.input,
        output_path=args.output,
        repo_root=REPO_ROOT,
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.ledger_md_output is not None:
        args.ledger_md_output.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_md_output.write_text(
            build_ledger_markdown(summary, rows),
            encoding="utf-8",
        )
    print(
        f"wrote {len(rows)} planning atom rows to {args.output} "
        f"({summary['negative_byte_delta_atom_count']} negative byte-delta rows)"
    )
    if args.atoms_json_output is not None:
        print(f"wrote atoms JSON list to {args.atoms_json_output}")
    if args.summary_output is not None:
        print(f"wrote summary to {args.summary_output}")
    if args.ledger_md_output is not None:
        print(f"wrote ledger markdown to {args.ledger_md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
