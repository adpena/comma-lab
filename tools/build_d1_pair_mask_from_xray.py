#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a D1 per-pair overlay sign mask from pair-component xray reports.

This is a diagnostic-to-packet bridge. It does not score, dispatch, or promote
anything. It consumes baseline and D1 overlay per-pair component measurements
from ``tools/xray_pair_component_errors.py`` and emits a compact selector input
for ``tools/build_d1_overlay_policy_candidates.py --sign-policies pair_mask``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CONTEST_ARCHIVE_DENOMINATOR_BYTES = 37_545_489
_EVIDENCE_AXES = (
    "contest_cpu",
    "contest_cuda",
    "local_cpu_xray",
    "local_cuda_xray",
    "macos_cpu_advisory",
)
_EVIDENCE_AXIS_LABELS = {
    "contest_cpu": "[contest-CPU]",
    "contest_cuda": "[contest-CUDA]",
    "local_cpu_xray": "[local-CPU xray]",
    "local_cuda_xray": "[local-CUDA xray]",
    "macos_cpu_advisory": "[macOS-CPU advisory]",
}
_XRAY_AXIS_BY_DIAGNOSTIC_DEVICE = {
    "cpu": "local_cpu_xray",
    "cuda": "local_cuda_xray",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(payload: Any) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer; got {value!r}")
    return int(value)


def _require_positive_int(value: Any, *, field: str) -> int:
    out = _require_int(value, field=field)
    if out <= 0:
        raise ValueError(f"{field} must be > 0; got {out}")
    return out


def _require_finite_nonnegative_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite nonnegative number") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{field} must be finite and >= 0; got {value!r}")
    return out


def _pack_pair_sign_mask_bytes(signs: list[int]) -> bytes:
    if not signs:
        raise ValueError("pair_signs must not be empty")
    code_by_sign = {0: 0, 1: 1, -1: 2}
    out = bytearray()
    acc = 0
    nbits = 0
    for idx, raw in enumerate(signs):
        sign = _require_int(raw, field=f"pair_signs[{idx}]")
        if sign not in code_by_sign:
            raise ValueError(f"pair_signs[{idx}] must be -1, 0, or 1; got {sign}")
        acc |= code_by_sign[sign] << nbits
        nbits += 2
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc)
    return bytes(out)


def _pair_mask_custody(
    signs: list[int],
    *,
    measured_pairs: int,
    expected_pairs: int | None,
    partial_smoke_allowed: bool,
) -> dict[str, Any]:
    packed = _pack_pair_sign_mask_bytes(signs)
    packed_b85 = base64.b85encode(packed).decode("ascii")
    n_pairs = len(signs)
    full_expected_length = expected_pairs is not None and n_pairs == expected_pairs
    full_measured = full_expected_length and measured_pairs == n_pairs
    if full_measured:
        custody_scope = "full_measured_contest_selector"
    elif full_expected_length:
        custody_scope = "padded_full_length_unmeasured_pairs_disabled"
    elif partial_smoke_allowed:
        custody_scope = "partial_smoke_selector"
    else:
        custody_scope = "custom_length_selector"
    return {
        "custody_scope": custody_scope,
        "n_pairs": n_pairs,
        "measured_pairs": measured_pairs,
        "expected_pairs": expected_pairs,
        "full_expected_length": full_expected_length,
        "full_measured": full_measured,
        "partial_smoke_allowed": bool(partial_smoke_allowed),
        "active_pairs": sum(1 for value in signs if value != 0),
        "positive_pairs": sum(1 for value in signs if value > 0),
        "negative_pairs": sum(1 for value in signs if value < 0),
        "packed_raw_bytes": len(packed),
        "packed_raw_sha256": hashlib.sha256(packed).hexdigest(),
        "packed_base85_chars": len(packed_b85),
        "score_bearing_runtime_keys": ["pair_mask_b85", "pair_mask_n"],
        "pair_signs_sha256": _stable_json_sha256(signs),
    }


def _xray_axis_from_payload(payload: dict[str, Any], *, path: Path) -> str:
    raw_axis = payload.get("evidence_axis") or payload.get("score_axis")
    if isinstance(raw_axis, str) and raw_axis:
        normalized = raw_axis.strip().lower().replace("-", "_")
        if normalized in _EVIDENCE_AXES:
            return normalized
        raise ValueError(
            f"{path} declares unsupported evidence axis {raw_axis!r}; "
            f"expected one of {_EVIDENCE_AXES}"
        )

    evidence_grade = str(payload.get("evidence_grade", "")).strip().lower()
    device = str(payload.get("device", "")).strip().lower()
    if evidence_grade.startswith("diagnostic_pair_component_xray_"):
        suffix = evidence_grade.removeprefix("diagnostic_pair_component_xray_")
        axis = _XRAY_AXIS_BY_DIAGNOSTIC_DEVICE.get(suffix or device)
        if axis is not None:
            return axis
    if device in _XRAY_AXIS_BY_DIAGNOSTIC_DEVICE:
        return _XRAY_AXIS_BY_DIAGNOSTIC_DEVICE[device]

    raise ValueError(
        f"{path} does not declare a verifiable evidence axis; add evidence_axis "
        "or a diagnostic_pair_component_xray_<device> evidence_grade"
    )


def _xray_provenance(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    evidence_axis = _xray_axis_from_payload(payload, path=path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "label": payload.get("label"),
        "evidence_grade": payload.get("evidence_grade"),
        "device": payload.get("device"),
        "evidence_axis": evidence_axis,
        "evidence_axis_label": _EVIDENCE_AXIS_LABELS[evidence_axis],
        "n_pairs": payload.get("n_pairs"),
        "archive": payload.get("archive"),
    }


def _load_xray(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    schema = payload.get("schema")
    if schema != "pair_component_error_xray_v1":
        raise ValueError(
            f"{path} schema must be pair_component_error_xray_v1; got {schema!r}"
        )
    if payload.get("score_claim") is not False:
        raise ValueError(f"{path} must declare score_claim=false")
    provenance = _xray_provenance(path, payload)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain non-empty rows[]")
    out: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} rows[{idx}] is not an object")
        pair_idx = _require_int(
            row.get("pair_idx", row.get("pair_index")),
            field=f"{path} rows[{idx}] pair_idx",
        )
        if pair_idx in out:
            raise ValueError(f"{path} duplicate pair_idx={pair_idx}")
        _pose_seg(row)
        out[pair_idx] = row
    if "n_pairs" in payload:
        declared_pairs = _require_positive_int(
            payload.get("n_pairs"),
            field=f"{path} n_pairs",
        )
        if declared_pairs != len(out):
            raise ValueError(
                f"{path} n_pairs={declared_pairs} does not match rows={len(out)}"
            )
    elif provenance.get("n_pairs") is not None:
        raise ValueError(
            f"{path} n_pairs must be an integer when present; got "
            f"{provenance.get('n_pairs')!r}"
        )
    provenance["n_pairs"] = len(out)
    return out, provenance


def _pose_seg(row: dict[str, Any]) -> tuple[float, float]:
    return (
        _require_finite_nonnegative_float(row.get("pose_dist"), field="pose_dist"),
        _require_finite_nonnegative_float(row.get("seg_dist"), field="seg_dist"),
    )


def _component_from_means(mean_pose: float, mean_seg: float) -> float:
    return math.sqrt(10.0 * max(0.0, mean_pose)) + 100.0 * mean_seg


def _rate_score_from_bytes_delta(byte_delta: int) -> float:
    return 25.0 * float(byte_delta) / float(_CONTEST_ARCHIVE_DENOMINATOR_BYTES)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-xray", type=Path, required=True)
    parser.add_argument(
        "--positive-xray",
        type=Path,
        required=True,
        help="Pair xray for the +payload D1 overlay candidate.",
    )
    parser.add_argument(
        "--negative-xray",
        type=Path,
        help="Optional pair xray for the negated D1 overlay candidate.",
    )
    parser.add_argument(
        "--improvement-guard",
        type=float,
        default=0.0,
        help="Minimum per-pair component-score improvement required to enable a pair.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("waterfill_prefix", "independent"),
        default="waterfill_prefix",
        help=(
            "waterfill_prefix sorts pair actions by linearized improvement and "
            "chooses the prefix that minimizes the actual global component score; "
            "independent keeps every pair above --improvement-guard."
        ),
    )
    parser.add_argument(
        "--evidence-axis",
        choices=_EVIDENCE_AXES,
        required=True,
        help="Evidence axis for the pair xray inputs; never inferred.",
    )
    parser.add_argument(
        "--baseline-archive-bytes",
        type=int,
        help=(
            "Archive bytes for the same baseline represented by --baseline-xray. "
            "Use with --candidate-archive-bytes for full A/B rate accounting."
        ),
    )
    parser.add_argument(
        "--candidate-archive-bytes",
        type=int,
        help=(
            "Archive bytes for the D1 candidate that will carry the emitted mask. "
            "Use with --baseline-archive-bytes."
        ),
    )
    parser.add_argument(
        "--rate-from-candidate-manifest",
        type=Path,
        help=(
            "Preferred full A/B rate source. Reads archive_bytes plus "
            "source_base_archive_bytes from a D1 candidate_manifest.json emitted "
            "by tools/build_d1_overlay_policy_candidates.py."
        ),
    )
    parser.add_argument(
        "--allow-negative-archive-delta",
        action="store_true",
        help=(
            "Allow candidate archive bytes below baseline bytes. This is not "
            "expected for D1 sidecar selectors and requires a rationale."
        ),
    )
    parser.add_argument(
        "--negative-archive-delta-rationale",
        help="Required text when --allow-negative-archive-delta is used.",
    )
    parser.add_argument(
        "--incremental-rate-cost-bytes",
        type=int,
        help=(
            "Incremental selector bytes versus a named D1-family baseline. This "
            "is only valid when the xray baseline is the same D1 packet family; "
            "A1-relative selectors must use --baseline-archive-bytes and "
            "--candidate-archive-bytes."
        ),
    )
    parser.add_argument(
        "--incremental-baseline-label",
        help="Required rationale label when --incremental-rate-cost-bytes is used.",
    )
    parser.add_argument(
        "--max-active-pairs",
        type=int,
        help="Optional cap on active pairs after sorting by improvement.",
    )
    parser.add_argument(
        "--min-net-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum positive score improvement required after rate penalty. "
            "The default accepts any strictly score-lowering nonzero mask."
        ),
    )
    parser.add_argument(
        "--output-n-pairs",
        type=int,
        help=(
            "Optional final mask length. When larger than the xray pair count, "
            "pads unmeasured pairs as disabled zeros."
        ),
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=600,
        help=(
            "Contest pair count used when --output-n-pairs is omitted. The "
            "default pads shorter xray reports to 600 disabled pairs."
        ),
    )
    parser.add_argument(
        "--allow-partial-smoke",
        action="store_true",
        help=(
            "Do not pad to --expected-pairs when --output-n-pairs is omitted. "
            "Only valid for local smoke artifacts, not contest packets."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _load_rate_from_candidate_manifest(path: Path) -> tuple[int, int, dict[str, Any]]:
    payload = _read_json(path)
    archive_bytes = payload.get("archive_bytes")
    baseline_bytes = payload.get("source_base_archive_bytes")
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool):
        raise ValueError(
            f"{path} must contain integer archive_bytes"
        )
    if not isinstance(baseline_bytes, int) or isinstance(baseline_bytes, bool):
        raise ValueError(
            f"{path} must contain integer source_base_archive_bytes for "
            "manifest-sourced full-rate accounting"
        )
    archive_bytes = _require_positive_int(archive_bytes, field=f"{path} archive_bytes")
    baseline_bytes = _require_positive_int(
        baseline_bytes,
        field=f"{path} source_base_archive_bytes",
    )
    provenance = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rate_accounting_source": "candidate_manifest_compressed_archive_bytes_v1",
        "candidate_id": payload.get("candidate_id"),
        "archive_bytes": archive_bytes,
        "source_base_archive_bytes": baseline_bytes,
        "archive_sha256": payload.get("archive_sha256"),
        "source_d1_bin_sha256": payload.get("source_d1_bin_sha256"),
        "base_member_sha256": payload.get("base_member_sha256"),
    }
    return baseline_bytes, archive_bytes, provenance


def build_pair_sign_mask(
    *,
    baseline_rows: dict[int, dict[str, Any]],
    positive_rows: dict[int, dict[str, Any]],
    negative_rows: dict[int, dict[str, Any]] | None = None,
    improvement_guard: float = 0.0,
    selection_mode: str = "waterfill_prefix",
    rate_cost_bytes: int = 0,
    evidence_axis: str,
    rate_scope: str,
    baseline_archive_bytes: int | None = None,
    candidate_archive_bytes: int | None = None,
    rate_source_manifest: dict[str, Any] | None = None,
    allow_negative_archive_delta: bool = False,
    negative_archive_delta_rationale: str | None = None,
    incremental_baseline_label: str | None = None,
    xray_provenance: dict[str, Any] | None = None,
    expected_pairs: int | None = None,
    partial_smoke_allowed: bool = False,
    max_active_pairs: int | None = None,
    min_net_improvement: float = 0.0,
    output_n_pairs: int | None = None,
) -> dict[str, Any]:
    pair_indices = sorted(baseline_rows)
    if pair_indices != list(range(len(pair_indices))):
        raise ValueError("baseline pair rows must be contiguous and zero-based")
    if sorted(positive_rows) != pair_indices:
        raise ValueError("positive xray pair rows do not match baseline")
    if negative_rows is not None and sorted(negative_rows) != pair_indices:
        raise ValueError("negative xray pair rows do not match baseline")

    n_measured = len(pair_indices)
    if evidence_axis not in _EVIDENCE_AXES:
        raise ValueError(
            f"evidence_axis={evidence_axis!r} must be one of {_EVIDENCE_AXES}"
        )
    evidence_axis_label = _EVIDENCE_AXIS_LABELS[evidence_axis]
    if rate_scope not in {"archive_delta", "incremental"}:
        raise ValueError("rate_scope must be 'archive_delta' or 'incremental'")
    if rate_scope == "archive_delta":
        if baseline_archive_bytes is None or candidate_archive_bytes is None:
            raise ValueError(
                "archive_delta rate scope requires baseline_archive_bytes and "
                "candidate_archive_bytes"
            )
        if int(baseline_archive_bytes) <= 0 or int(candidate_archive_bytes) <= 0:
            raise ValueError(
                "archive_delta rate scope requires positive baseline and "
                "candidate archive bytes"
            )
        rate_cost_bytes = int(candidate_archive_bytes) - int(baseline_archive_bytes)
        if rate_cost_bytes < 0 and (
            not allow_negative_archive_delta or not negative_archive_delta_rationale
        ):
            raise ValueError(
                "negative archive deltas require allow_negative_archive_delta "
                "and negative_archive_delta_rationale"
            )
    else:
        if rate_cost_bytes < 0:
            raise ValueError(f"incremental rate_cost_bytes must be >= 0; got {rate_cost_bytes}")
        if not incremental_baseline_label:
            raise ValueError(
                "incremental rate scope requires incremental_baseline_label"
            )
    baseline_mean_pose = (
        sum(_pose_seg(baseline_rows[pair_idx])[0] for pair_idx in pair_indices)
        / n_measured
    )
    baseline_mean_seg = (
        sum(_pose_seg(baseline_rows[pair_idx])[1] for pair_idx in pair_indices)
        / n_measured
    )
    if baseline_mean_pose <= 0.0:
        raise ValueError("baseline mean pose distance must be > 0")
    pose_weight = 5.0 / math.sqrt(10.0 * baseline_mean_pose)
    seg_weight = 100.0
    baseline_pose_sum = sum(
        _pose_seg(baseline_rows[pair_idx])[0] for pair_idx in pair_indices
    )
    baseline_seg_sum = sum(
        _pose_seg(baseline_rows[pair_idx])[1] for pair_idx in pair_indices
    )
    baseline_component = _component_from_means(baseline_mean_pose, baseline_mean_seg)
    rate_penalty_score = _rate_score_from_bytes_delta(int(rate_cost_bytes))
    if not math.isfinite(float(improvement_guard)) or float(improvement_guard) < 0.0:
        raise ValueError(
            f"improvement_guard must be finite and >= 0; got {improvement_guard}"
        )
    if max_active_pairs is not None and max_active_pairs < 0:
        raise ValueError(f"max_active_pairs must be >= 0; got {max_active_pairs}")
    if min_net_improvement < 0:
        raise ValueError(
            f"min_net_improvement must be >= 0; got {min_net_improvement}"
        )

    guard = float(improvement_guard)
    potential_rows: list[dict[str, Any]] = []
    for pair_idx in pair_indices:
        base_pose, base_seg = _pose_seg(baseline_rows[pair_idx])
        base_objective = pose_weight * base_pose + seg_weight * base_seg
        choices = [(0, base_objective, baseline_rows[pair_idx])]
        pos_pose, pos_seg = _pose_seg(positive_rows[pair_idx])
        choices.append(
            (
                1,
                pose_weight * pos_pose + seg_weight * pos_seg,
                positive_rows[pair_idx],
            )
        )
        if negative_rows is not None:
            neg_pose, neg_seg = _pose_seg(negative_rows[pair_idx])
            choices.append(
                (
                    -1,
                    pose_weight * neg_pose + seg_weight * neg_seg,
                    negative_rows[pair_idx],
                )
            )
        sign, chosen_objective, row = min(
            choices, key=lambda item: (item[1], abs(item[0]))
        )
        improvement = base_objective - chosen_objective
        if sign != 0 and improvement > guard:
            selected_pose, selected_seg = _pose_seg(row)
            potential_rows.append(
                {
                    "pair_idx": pair_idx,
                    "sign": sign,
                    "baseline_objective": base_objective,
                    "selected_objective": chosen_objective,
                    "linearized_objective_improvement": improvement,
                    "selected_pose_dist": selected_pose,
                    "selected_seg_dist": selected_seg,
                    "pose_delta": selected_pose - base_pose,
                    "seg_delta": selected_seg - base_seg,
                }
            )

    mode = str(selection_mode)
    if mode not in {"waterfill_prefix", "independent"}:
        raise ValueError(
            f"selection_mode={selection_mode!r} must be waterfill_prefix or independent"
        )
    if max_active_pairs is not None:
        potential_rows = sorted(
            potential_rows,
            key=lambda row: (
                -float(row["linearized_objective_improvement"]),
                int(row["pair_idx"]),
            ),
        )[: int(max_active_pairs)]

    selected_by_pair: dict[int, dict[str, Any]] = {}
    selected_component = baseline_component
    selected_mean_pose = baseline_mean_pose
    selected_mean_seg = baseline_mean_seg
    best_prefix_size = 0
    best_component_prefix_size = 0
    best_component_no_rate_delta = 0.0
    best_net_delta = 0.0
    if mode == "independent":
        selected_by_pair = {int(row["pair_idx"]): row for row in potential_rows}
        selected_pose_sum = baseline_pose_sum + sum(
            float(row["pose_delta"]) for row in selected_by_pair.values()
        )
        selected_seg_sum = baseline_seg_sum + sum(
            float(row["seg_delta"]) for row in selected_by_pair.values()
        )
        selected_mean_pose = selected_pose_sum / n_measured
        selected_mean_seg = selected_seg_sum / n_measured
        selected_component = _component_from_means(selected_mean_pose, selected_mean_seg)
        best_component_no_rate_delta = selected_component - baseline_component
        best_component_prefix_size = len(selected_by_pair)
        best_net_delta = best_component_no_rate_delta + (
            rate_penalty_score if selected_by_pair else 0.0
        )
        if best_net_delta >= -float(min_net_improvement):
            selected_by_pair = {}
            selected_component = baseline_component
            selected_mean_pose = baseline_mean_pose
            selected_mean_seg = baseline_mean_seg
            best_net_delta = 0.0
    else:
        ranked = sorted(
            potential_rows,
            key=lambda row: (
                -float(row["linearized_objective_improvement"]),
                int(row["pair_idx"]),
            ),
        )
        pose_sum = baseline_pose_sum
        seg_sum = baseline_seg_sum
        best_rows: list[dict[str, Any]] = []
        best_component = baseline_component
        best_pose_mean = baseline_mean_pose
        best_seg_mean = baseline_mean_seg
        best_delta = 0.0
        for prefix_size, row in enumerate(ranked, start=1):
            pose_sum += float(row["pose_delta"])
            seg_sum += float(row["seg_delta"])
            pose_mean = pose_sum / n_measured
            seg_mean = seg_sum / n_measured
            component = _component_from_means(pose_mean, seg_mean)
            component_only_delta = component - baseline_component
            net_delta = component_only_delta + rate_penalty_score
            if component_only_delta < best_component_no_rate_delta:
                best_component_no_rate_delta = component_only_delta
                best_component_prefix_size = prefix_size
            if net_delta < best_delta:
                best_delta = net_delta
                best_rows = ranked[:prefix_size]
                best_component = component
                best_pose_mean = pose_mean
                best_seg_mean = seg_mean
                best_prefix_size = prefix_size
        if best_delta < -float(min_net_improvement):
            selected_by_pair = {int(row["pair_idx"]): row for row in best_rows}
            selected_component = best_component
            selected_mean_pose = best_pose_mean
            selected_mean_seg = best_seg_mean
            best_net_delta = best_delta

    signs: list[int] = [0] * n_measured
    selected_rows: list[dict[str, Any]] = []
    ranked_selected = sorted(
        selected_by_pair.values(),
        key=lambda row: (
            -float(row["linearized_objective_improvement"]),
            int(row["pair_idx"]),
        ),
    )
    rank_by_pair = {
        int(row["pair_idx"]): rank for rank, row in enumerate(ranked_selected, start=1)
    }
    for pair_idx, row in selected_by_pair.items():
        signs[pair_idx] = int(row["sign"])
    for pair_idx in sorted(selected_by_pair):
        row = dict(selected_by_pair[pair_idx])
        row["selection_rank"] = rank_by_pair[pair_idx]
        selected_rows.append(row)
    measured_pairs = len(signs)
    if output_n_pairs is not None:
        if output_n_pairs < measured_pairs:
            raise ValueError(
                f"output_n_pairs={output_n_pairs} is smaller than measured "
                f"pair count {measured_pairs}"
            )
        signs.extend([0] * (int(output_n_pairs) - measured_pairs))
    elif expected_pairs is not None and not partial_smoke_allowed:
        if expected_pairs < measured_pairs:
            raise ValueError(
                f"expected_pairs={expected_pairs} is smaller than measured "
                f"pair count {measured_pairs}"
            )
        signs.extend([0] * (int(expected_pairs) - measured_pairs))

    active_pairs = sum(1 for value in signs if value != 0)
    component_delta = selected_component - baseline_component
    total_delta_with_rate = component_delta + (
        rate_penalty_score if active_pairs else 0.0
    )
    pair_mask_custody = _pair_mask_custody(
        signs,
        measured_pairs=measured_pairs,
        expected_pairs=expected_pairs,
        partial_smoke_allowed=partial_smoke_allowed,
    )
    compressed_rate_accounting = {
        "schema": "d1_pair_mask_compressed_rate_accounting_v1",
        "scope": rate_scope,
        "source": (
            "candidate_manifest"
            if rate_source_manifest is not None
            else (
                "manual_full_archive_bytes"
                if rate_scope == "archive_delta"
                else "same_family_incremental_selector_bytes"
            )
        ),
        "evidence_axis": evidence_axis,
        "evidence_axis_label": evidence_axis_label,
        "denominator_bytes": _CONTEST_ARCHIVE_DENOMINATOR_BYTES,
        "formula": "25 * byte_delta / 37545489",
        "rate_cost_bytes": int(rate_cost_bytes),
        "rate_penalty_score": rate_penalty_score,
        "baseline_archive_bytes": baseline_archive_bytes,
        "candidate_archive_bytes": candidate_archive_bytes,
        "archive_byte_delta": (
            int(rate_cost_bytes) if rate_scope == "archive_delta" else None
        ),
        "incremental_rate_cost_bytes": (
            int(rate_cost_bytes) if rate_scope == "incremental" else None
        ),
        "incremental_baseline_label": incremental_baseline_label,
        "rate_source_manifest_sha256": (
            rate_source_manifest.get("sha256")
            if rate_source_manifest is not None
            else None
        ),
        "pair_mask_packed_raw_bytes": pair_mask_custody["packed_raw_bytes"],
        "pair_mask_packed_base85_chars": pair_mask_custody["packed_base85_chars"],
    }
    deterministic_provenance = {
        "schema": "d1_pair_mask_selector_deterministic_provenance_v1",
        "objective": "contest_score_linearized_at_baseline_mean_pose_v1",
        "evidence_axis": evidence_axis,
        "xray_provenance": xray_provenance,
        "pair_mask_custody": pair_mask_custody,
        "compressed_rate_accounting": compressed_rate_accounting,
        "selection_mode": mode,
        "selected_pairs": selected_rows,
        "pair_signs_sha256": pair_mask_custody["pair_signs_sha256"],
    }
    return {
        "schema": "d1_pair_sign_mask_from_xray_v1",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deterministic_provenance_sha256": _stable_json_sha256(
            deterministic_provenance
        ),
        "deterministic_provenance": deterministic_provenance,
        "objective": "contest_score_linearized_at_baseline_mean_pose_v1",
        "evidence_axis": evidence_axis,
        "evidence_axis_label": evidence_axis_label,
        "xray_provenance": xray_provenance,
        "selection_mode": mode,
        "pair_signs": signs,
        "pair_mask_custody": pair_mask_custody,
        "n_pairs": len(signs),
        "measured_pairs": measured_pairs,
        "active_pairs": active_pairs,
        "positive_pairs": sum(1 for value in signs if value > 0),
        "negative_pairs": sum(1 for value in signs if value < 0),
        "potential_pairs": len(potential_rows),
        "best_prefix_size": (
            best_prefix_size if mode == "waterfill_prefix" else active_pairs
        ),
        "best_component_prefix_size": (
            best_component_prefix_size
            if mode == "waterfill_prefix"
            else active_pairs
        ),
        "best_component_no_rate_delta": best_component_no_rate_delta,
        "max_active_pairs": max_active_pairs,
        "improvement_guard": guard,
        "rate_cost_bytes": int(rate_cost_bytes),
        "rate_scope": rate_scope,
        "compressed_rate_accounting": compressed_rate_accounting,
        "baseline_archive_bytes": baseline_archive_bytes,
        "candidate_archive_bytes": candidate_archive_bytes,
        "archive_byte_delta": (
            int(rate_cost_bytes) if rate_scope == "archive_delta" else None
        ),
        "incremental_rate_cost_bytes": (
            int(rate_cost_bytes) if rate_scope == "incremental" else None
        ),
        "incremental_baseline_label": incremental_baseline_label,
        "rate_source_manifest": rate_source_manifest,
        "allow_negative_archive_delta": bool(allow_negative_archive_delta),
        "negative_archive_delta_rationale": negative_archive_delta_rationale,
        "expected_pairs": expected_pairs,
        "partial_smoke_allowed": bool(partial_smoke_allowed),
        "rate_penalty_score": rate_penalty_score,
        "min_net_improvement": float(min_net_improvement),
        "baseline_mean_pose_dist": baseline_mean_pose,
        "baseline_mean_seg_dist": baseline_mean_seg,
        "selected_mean_pose_dist": selected_mean_pose,
        "selected_mean_seg_dist": selected_mean_seg,
        "pose_weight": pose_weight,
        "seg_weight": seg_weight,
        "predicted_component_no_rate_baseline": baseline_component,
        "predicted_component_no_rate_selected": selected_component,
        "predicted_component_no_rate_delta": component_delta,
        "predicted_total_delta_with_rate": total_delta_with_rate,
        "predicted_score_lowering_after_rate": total_delta_with_rate < 0.0,
        "selected_pairs": selected_rows,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline_rows, baseline_provenance = _load_xray(args.baseline_xray)
    positive_rows, positive_provenance = _load_xray(args.positive_xray)
    negative_rows: dict[int, dict[str, Any]] | None = None
    negative_provenance: dict[str, Any] | None = None
    if args.negative_xray is not None:
        negative_rows, negative_provenance = _load_xray(args.negative_xray)
    xray_provenance = {
        "baseline": baseline_provenance,
        "positive": positive_provenance,
        "negative": negative_provenance,
    }
    observed_axes = {
        str(item["evidence_axis"])
        for item in (baseline_provenance, positive_provenance, negative_provenance)
        if item is not None
    }
    if observed_axes != {str(args.evidence_axis)}:
        raise SystemExit(
            f"--evidence-axis={args.evidence_axis!r} does not match xray "
            f"provenance axes {sorted(observed_axes)}"
        )

    manifest_rate = args.rate_from_candidate_manifest is not None
    full_rate = (
        args.baseline_archive_bytes is not None
        or args.candidate_archive_bytes is not None
    )
    incremental_rate = args.incremental_rate_cost_bytes is not None
    rate_modes = sum(int(flag) for flag in (manifest_rate, full_rate, incremental_rate))
    if rate_modes != 1:
        raise SystemExit(
            "choose exactly one rate scope: "
            "--rate-from-candidate-manifest, or "
            "--baseline-archive-bytes + --candidate-archive-bytes, or "
            "--incremental-rate-cost-bytes + --incremental-baseline-label"
        )
    if full_rate and (
        args.baseline_archive_bytes is None or args.candidate_archive_bytes is None
    ):
        raise SystemExit(
            "--baseline-archive-bytes and --candidate-archive-bytes are required together"
        )
    if incremental_rate and not args.incremental_baseline_label:
        raise SystemExit(
            "--incremental-rate-cost-bytes requires --incremental-baseline-label"
        )
    rate_source_manifest = None
    baseline_archive_bytes = args.baseline_archive_bytes
    candidate_archive_bytes = args.candidate_archive_bytes
    if manifest_rate:
        baseline_archive_bytes, candidate_archive_bytes, rate_source_manifest = (
            _load_rate_from_candidate_manifest(args.rate_from_candidate_manifest)
        )
    rate_scope = "archive_delta" if (full_rate or manifest_rate) else "incremental"
    rate_cost_bytes = (
        int(candidate_archive_bytes) - int(baseline_archive_bytes)
        if rate_scope == "archive_delta"
        else int(args.incremental_rate_cost_bytes)
    )
    if args.allow_negative_archive_delta and not args.negative_archive_delta_rationale:
        raise SystemExit(
            "--allow-negative-archive-delta requires --negative-archive-delta-rationale"
        )
    output_n_pairs = args.output_n_pairs
    expected_pairs = _require_positive_int(
        int(args.expected_pairs),
        field="--expected-pairs",
    )
    if output_n_pairs is not None:
        output_n_pairs = _require_positive_int(
            int(output_n_pairs),
            field="--output-n-pairs",
        )
    result = build_pair_sign_mask(
        baseline_rows=baseline_rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        improvement_guard=float(args.improvement_guard),
        selection_mode=str(args.selection_mode),
        rate_cost_bytes=rate_cost_bytes,
        evidence_axis=str(args.evidence_axis),
        rate_scope=rate_scope,
        baseline_archive_bytes=baseline_archive_bytes,
        candidate_archive_bytes=candidate_archive_bytes,
        rate_source_manifest=rate_source_manifest,
        allow_negative_archive_delta=bool(args.allow_negative_archive_delta),
        negative_archive_delta_rationale=args.negative_archive_delta_rationale,
        incremental_baseline_label=args.incremental_baseline_label,
        xray_provenance=xray_provenance,
        expected_pairs=expected_pairs,
        partial_smoke_allowed=bool(args.allow_partial_smoke),
        max_active_pairs=args.max_active_pairs,
        min_net_improvement=float(args.min_net_improvement),
        output_n_pairs=output_n_pairs,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "output_json": str(args.output_json),
                "objective": result["objective"],
                "n_pairs": result["n_pairs"],
                "measured_pairs": result["measured_pairs"],
                "active_pairs": result["active_pairs"],
                "positive_pairs": result["positive_pairs"],
                "negative_pairs": result["negative_pairs"],
                "potential_pairs": result["potential_pairs"],
                "selection_mode": result["selection_mode"],
                "evidence_axis": result["evidence_axis"],
                "best_prefix_size": result["best_prefix_size"],
                "best_component_prefix_size": result["best_component_prefix_size"],
                "improvement_guard": result["improvement_guard"],
                "rate_cost_bytes": result["rate_cost_bytes"],
                "rate_scope": result["rate_scope"],
                "predicted_component_no_rate_delta": result[
                    "predicted_component_no_rate_delta"
                ],
                "predicted_total_delta_with_rate": result[
                    "predicted_total_delta_with_rate"
                ],
                "predicted_score_lowering_after_rate": result[
                    "predicted_score_lowering_after_rate"
                ],
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
