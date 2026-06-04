# SPDX-License-Identifier: MIT
"""Decoder-weight waterfilling for compact NeRV carriers.

This is a portable NumPy planning primitive for HiNeRV/SNeRV decoder weights.
It prices concrete tensor groups against the contest byte price and chooses the
cheapest quantization action whose saliency-weighted distortion proxy pays for
its rate savings. It is not score authority: measured scorer deltas and
receiver-closed archive replay must still fill the promotion gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.repo_io import sha256_file
from tac.substrates._shared.decoder_state_codec import (
    decoder_state_codec_stats,
    serialize_decoder_state_dict,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
    ORIGINAL_VIDEO_BYTES,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

NERV_DECODER_WEIGHT_WATERFILL_SCHEMA = "nerv_decoder_weight_waterfill.v1"
TRUSTED_RECEIVER_PROOF_STATUSES = frozenset(
    {
        "runtime_consumption_proof_ready",
        "receiver_proof_valid",
        "runtime_consumption_proof_passed",
    }
)
PLANNING_AXIS = "[planning/control]"
DEFAULT_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latent_embed",
    "blocks",
    "feature_grids",
    "head",
    "decoder",
    "injector",
)
DEFAULT_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latents",
    "codebook",
    "selector",
    "ema",
    "teacher",
    "student",
)
DEFAULT_ACTION_BITS: tuple[int, ...] = (0, 2, 4, 6, 7, 8, 16, 32)
SALIENCY_CALIBRATION_MODES: tuple[str, ...] = (
    "none",
    "max",
    "mean",
    "median",
    "rank",
)


class NervDecoderWeightWaterfillError(ValueError):
    """Raised when decoder-weight waterfill inputs are inconsistent."""


@dataclass(frozen=True)
class TensorGroup:
    """One decoder tensor group with concrete values and optional saliency."""

    name: str
    values: np.ndarray
    saliency: float | None = None


def build_nerv_decoder_weight_waterfill_plan(
    state_dict: Mapping[str, Any],
    *,
    saliency_by_name: Mapping[str, float] | None = None,
    saliency_calibration: Mapping[str, Any] | None = None,
    family: str = "hi_nerv",
    candidate_id: str | None = None,
    include_substrings: Sequence[str] = DEFAULT_INCLUDE_SUBSTRINGS,
    exclude_substrings: Sequence[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
    action_bits: Sequence[int] = DEFAULT_ACTION_BITS,
    byte_price_score_per_byte: float = CONTEST_BYTE_PRICE_SCORE,
    original_video_bytes: int = ORIGINAL_VIDEO_BYTES,
    zero_run_overhead_bytes: int = 2,
    decoder_state_codec_for_byte_calibration: str | None = None,
    full_video_coverage: bool = False,
    receiver_proof_status: str = "missing",
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority decoder-weight quantization plan.

    ``saliency_by_name`` should contain measured decoder-weight saliency from a
    scorer-loop trainer/VJP pass. Missing saliency keeps the row executable for
    byte accounting but blocks exact admission.
    """

    includes = _clean_tokens(include_substrings, name="include_substrings")
    excludes = tuple(str(token) for token in exclude_substrings if str(token))
    bits = _validated_action_bits(action_bits)
    if float(byte_price_score_per_byte) <= 0.0:
        raise NervDecoderWeightWaterfillError("byte price must be positive")
    if int(original_video_bytes) <= 0:
        raise NervDecoderWeightWaterfillError("original_video_bytes must be positive")
    if int(zero_run_overhead_bytes) < 0:
        raise NervDecoderWeightWaterfillError("zero_run_overhead_bytes must be >= 0")

    groups = _decoder_tensor_groups(
        state_dict,
        saliency_by_name=saliency_by_name or {},
        include_substrings=includes,
        exclude_substrings=excludes,
    )
    codec_calibration = _decoder_state_codec_byte_calibration(
        state_dict,
        groups,
        action_bits=bits,
        decoder_state_codec=decoder_state_codec_for_byte_calibration,
    )
    rows = [
        _row_for_group(
            group,
            family=str(family),
            candidate_id=candidate_id,
            action_bits=bits,
            byte_price_score_per_byte=float(byte_price_score_per_byte),
            zero_run_overhead_bytes=int(zero_run_overhead_bytes),
            codec_byte_calibration=codec_calibration["groups"].get(group.name),
            full_video_coverage=bool(full_video_coverage),
            receiver_proof_status=str(receiver_proof_status),
            archive_sha256=archive_sha256,
        )
        for group in groups
    ]
    blockers = []
    if not rows:
        blockers.append("decoder_weight_waterfill_no_decoder_groups_selected")
    if any(row["saliency"] is None for row in rows):
        blockers.append("decoder_weight_saliency_missing_for_some_groups")
    if not full_video_coverage:
        blockers.append("full_video_coverage_missing")
    if receiver_proof_status.lower() not in TRUSTED_RECEIVER_PROOF_STATUSES:
        blockers.append("receiver_proof_not_satisfied")
    if not archive_sha256:
        blockers.append("archive_sha256_missing")
    elif not _is_sha256_hex(archive_sha256):
        blockers.append("archive_sha256_invalid")
    blockers.append("contest_cpu_cuda_exact_eval_not_executed")

    section_value_rows = [_section_value_row(row) for row in rows]
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
            "family": str(family),
            "axis_tag": PLANNING_AXIS,
            "section_value_rows": section_value_rows,
            "blockers": blockers,
            "full_video_coverage": bool(full_video_coverage),
            "receiver_proof_status": str(receiver_proof_status),
            "archive_sha256": archive_sha256,
        },
        candidate_id=candidate_id,
    )
    return {
        "schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "family": str(family),
        "candidate_id": candidate_id,
        "axis_tag": PLANNING_AXIS,
        "authority": "false_authority_decoder_weight_waterfill_no_score_claim",
        "contest_byte_price_score_per_byte": float(byte_price_score_per_byte),
        "original_video_bytes": int(original_video_bytes),
        "selection_rule": (
            "choose argmin over zero/int2/int4/int8/fp16/fp32 of "
            "saliency_weighted_quantization_error + byte_delta * contest_byte_price"
        ),
        "include_substrings": list(includes),
        "exclude_substrings": list(excludes),
        "action_bits": list(bits),
        "zero_run_overhead_bytes": int(zero_run_overhead_bytes),
        "decoder_state_codec_for_byte_calibration": decoder_state_codec_for_byte_calibration,
        "decoder_state_codec_byte_calibration": codec_calibration["metadata"],
        "saliency_calibration": dict(saliency_calibration or {}),
        "full_video_coverage": bool(full_video_coverage),
        "receiver_proof_status": str(receiver_proof_status),
        "archive_sha256": archive_sha256,
        "group_count": len(rows),
        "total_baseline_fp32_bytes": int(sum(row["baseline_fp32_bytes"] for row in rows)),
        "total_selected_estimated_bytes": int(
            sum(row["selected_estimated_bytes"] for row in rows)
        ),
        "total_selected_byte_delta": int(sum(row["selected_byte_delta"] for row in rows)),
        "total_selected_delta_rate_score": float(
            sum(row["selected_delta_rate_score"] for row in rows)
        ),
        "total_selected_delta_nonrate_score_proxy": _sum_or_none(
            row["selected_delta_nonrate_score_proxy"] for row in rows
        ),
        "rows": rows,
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def load_state_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a state dictionary from an ``np.savez`` archive."""

    p = Path(path).expanduser().resolve(strict=False)
    if not p.exists():
        raise NervDecoderWeightWaterfillError(f"state npz does not exist: {p}")
    with np.load(p, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_state_npz_from_manifest(path: str | Path) -> dict[str, np.ndarray]:
    """Load a state NPZ only after verifying its bridge manifest SHA."""

    manifest_path = Path(path).expanduser().resolve(strict=False)
    if not manifest_path.is_file():
        raise NervDecoderWeightWaterfillError(
            f"state npz manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "framework_agnostic_npz_bridge_manifest.v1":
        raise NervDecoderWeightWaterfillError(
            "unsupported state npz manifest schema: "
            f"{manifest.get('schema')!r}"
        )
    if manifest.get("consumption_recommended") is not True:
        raise NervDecoderWeightWaterfillError(
            "state npz manifest is not consumption-recommended: "
            f"{manifest.get('blockers')}"
        )
    artifact = manifest.get("artifact_path")
    if not artifact:
        raise NervDecoderWeightWaterfillError("state npz manifest missing artifact_path")
    artifact_path = Path(str(artifact)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    artifact_path = artifact_path.resolve(strict=False)
    expected_sha = str(manifest.get("artifact_sha256") or "")
    if len(expected_sha) != 64:
        raise NervDecoderWeightWaterfillError(
            "state npz manifest missing 64-char artifact_sha256"
        )
    if not artifact_path.is_file():
        raise NervDecoderWeightWaterfillError(
            f"state npz artifact does not exist: {artifact_path}"
        )
    actual_sha = sha256_file(artifact_path)
    if actual_sha != expected_sha:
        raise NervDecoderWeightWaterfillError(
            "state npz artifact sha256 mismatch: "
            f"expected={expected_sha} actual={actual_sha}"
        )
    return load_state_npz(artifact_path)


def load_saliency_json(path: str | Path) -> dict[str, float]:
    """Load saliency as either ``{name: value}`` or row dictionaries."""

    p = Path(path).expanduser().resolve(strict=False)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        explicit = payload.get("saliency_by_name") or payload.get("global_saliency")
        if isinstance(explicit, Mapping):
            parsed = _saliency_from_mapping(explicit)
            if parsed:
                return parsed
        saliency_rows = payload.get("saliency_rows")
        if isinstance(saliency_rows, Sequence) and not isinstance(saliency_rows, (str, bytes)):
            parsed = _saliency_from_rows(saliency_rows)
            if parsed:
                return parsed
        rows = payload.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            parsed = _saliency_from_rows(rows)
            if parsed:
                return parsed
            parsed = _saliency_from_nested_row_maps(rows)
            if parsed:
                return parsed
        return {
            str(key): float(value)
            for key, value in payload.items()
            if _finite_float_or_none(value) is not None
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _saliency_from_rows(payload)
    raise NervDecoderWeightWaterfillError("unsupported saliency JSON shape")


def calibrate_saliency_by_name(
    saliency_by_name: Mapping[str, float],
    *,
    mode: str = "none",
    scale: float = 1.0,
    floor: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Normalize raw saliency proxies while preserving false-authority labels.

    Train-time gradient saliency and scorer-loss replay saliency can have
    different units. This helper makes that calibration explicit instead of
    hiding it in ad hoc JSON edits. ``mode="none"`` preserves values exactly;
    other modes keep only relative ordering/magnitude, then apply ``scale``.
    """

    mode = str(mode)
    if mode not in SALIENCY_CALIBRATION_MODES:
        raise NervDecoderWeightWaterfillError(
            "unsupported saliency calibration mode: "
            f"{mode!r}; expected one of {SALIENCY_CALIBRATION_MODES}"
        )
    if float(scale) <= 0.0:
        raise NervDecoderWeightWaterfillError("saliency scale must be positive")
    if float(floor) < 0.0:
        raise NervDecoderWeightWaterfillError("saliency floor must be >= 0")
    parsed = {
        str(name): max(0.0, float(value))
        for name, value in saliency_by_name.items()
        if _finite_float_or_none(value) is not None
    }
    if not parsed:
        return (
            {},
            {
                "schema": "nerv_decoder_weight_saliency_calibration.v1",
                "mode": mode,
                "scale": float(scale),
                "floor": float(floor),
                "input_count": 0,
                "output_count": 0,
                "blockers": ["saliency_calibration_no_finite_inputs"],
            },
        )
    if mode == "none":
        calibrated = {name: _apply_saliency_floor(value, floor) for name, value in parsed.items()}
        divisor = 1.0
    elif mode == "rank":
        ordered = sorted(parsed.items(), key=lambda item: (item[1], item[0]))
        denom = float(max(1, len(ordered)))
        calibrated = {
            name: _apply_saliency_floor(float(index + 1) / denom * float(scale), floor)
            for index, (name, _value) in enumerate(ordered)
        }
        divisor = None
    else:
        values = np.asarray(list(parsed.values()), dtype=np.float64)
        if mode == "max":
            divisor = float(np.max(values))
        elif mode == "mean":
            divisor = float(np.mean(values))
        else:
            divisor = float(np.median(values))
        if divisor <= 0.0:
            calibrated = {
                name: _apply_saliency_floor(0.0, floor) for name in parsed
            }
        else:
            calibrated = {
                name: _apply_saliency_floor(value / divisor * float(scale), floor)
                for name, value in parsed.items()
            }
    values_out = list(calibrated.values())
    return (
        calibrated,
        {
            "schema": "nerv_decoder_weight_saliency_calibration.v1",
            "mode": mode,
            "scale": float(scale),
            "floor": float(floor),
            "divisor": divisor,
            "input_count": len(parsed),
            "output_count": len(calibrated),
            "output_min": min(values_out) if values_out else None,
            "output_max": max(values_out) if values_out else None,
            "authority": "false_authority_saliency_proxy_calibration_no_score_claim",
            "blockers": [],
        },
    )


def render_nerv_decoder_weight_waterfill_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown summary."""

    lines = [
        "# NeRV decoder-weight waterfill",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Family: `{report.get('family')}`",
        f"Authority: `{report.get('authority')}`",
        "",
        "| group | action | fp32 bytes | selected bytes | delta rate | delta non-rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows", ()):
        lines.append(
            "| {group} | {action} | {fp32} | {selected} | {rate:.6f} | {nonrate} |".format(
                group=row["group_name"],
                action=row["selected_action"],
                fp32=row["baseline_fp32_bytes"],
                selected=row["selected_estimated_bytes"],
                rate=row["selected_delta_rate_score"],
                nonrate=(
                    "missing"
                    if row["selected_delta_nonrate_score_proxy"] is None
                    else f"{row['selected_delta_nonrate_score_proxy']:.6f}"
                ),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers", ()):
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _decoder_tensor_groups(
    state_dict: Mapping[str, Any],
    *,
    saliency_by_name: Mapping[str, float],
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> list[TensorGroup]:
    groups = []
    for name in sorted(str(key) for key in state_dict):
        if not any(token in name for token in include_substrings):
            continue
        if any(token in name for token in exclude_substrings):
            continue
        values = np.asarray(state_dict[name], dtype=np.float64)
        if values.size == 0 or not np.issubdtype(values.dtype, np.number):
            continue
        saliency = _saliency_for_name(name, saliency_by_name)
        groups.append(TensorGroup(name=name, values=values, saliency=saliency))
    return groups


def _row_for_group(
    group: TensorGroup,
    *,
    family: str,
    candidate_id: str | None,
    action_bits: Sequence[int],
    byte_price_score_per_byte: float,
    zero_run_overhead_bytes: int,
    full_video_coverage: bool,
    receiver_proof_status: str,
    archive_sha256: str | None,
    codec_byte_calibration: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    values = group.values.astype(np.float64, copy=False)
    numel = int(values.size)
    baseline_bytes = int(numel * 4)
    candidates = [
        _candidate_for_bits(
            values,
            bits=int(bits),
            saliency=group.saliency,
            baseline_bytes=baseline_bytes,
            byte_price_score_per_byte=byte_price_score_per_byte,
            zero_run_overhead_bytes=zero_run_overhead_bytes,
            codec_byte_calibration=(
                None
                if codec_byte_calibration is None
                else codec_byte_calibration.get(int(bits))
            ),
        )
        for bits in action_bits
    ]
    if group.saliency is None:
        selected = _required_candidate(candidates, bits=32)
    else:
        selected = min(
            candidates,
            key=lambda item: (
                float("inf")
                if item["delta_total_score_proxy"] is None
                else item["delta_total_score_proxy"],
                item["estimated_bytes"],
            ),
        )
    blockers = []
    if group.saliency is None:
        blockers.append("decoder_weight_group_saliency_missing")
    if not full_video_coverage:
        blockers.append("full_video_coverage_missing")
    if not archive_sha256:
        blockers.append("archive_sha256_missing")
    elif not _is_sha256_hex(archive_sha256):
        blockers.append("archive_sha256_invalid")
    if receiver_proof_status.lower() not in TRUSTED_RECEIVER_PROOF_STATUSES:
        blockers.append("receiver_proof_not_satisfied")
    return {
        "row_id": f"{family}_decoder_weight_waterfill:{group.name}",
        "family": family,
        "candidate_id": candidate_id,
        "group_name": group.name,
        "scope": "decoder_weight_group",
        "numel": numel,
        "shape": [int(dim) for dim in values.shape],
        "saliency": group.saliency,
        "value_stats": {
            "abs_max": float(np.max(np.abs(values))) if numel else 0.0,
            "abs_mean": float(np.mean(np.abs(values))) if numel else 0.0,
            "std": float(np.std(values)) if numel else 0.0,
        },
        "baseline_action": "fp32",
        "baseline_fp32_bytes": baseline_bytes,
        "selected_action": selected["action"],
        "selected_bits": selected["bits"],
        "selected_estimated_bytes": selected["estimated_bytes"],
        "selected_byte_delta": selected["byte_delta"],
        "selected_byte_delta_source": selected["byte_delta_source"],
        "selected_delta_rate_score": selected["delta_rate_score"],
        "selected_delta_nonrate_score_proxy": selected["delta_nonrate_score_proxy"],
        "selected_delta_total_score_proxy": selected["delta_total_score_proxy"],
        "candidate_actions": candidates,
        "full_video_coverage": bool(full_video_coverage),
        "receiver_proof_status": receiver_proof_status,
        "archive_sha256": archive_sha256,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _decoder_state_codec_byte_calibration(
    state_dict: Mapping[str, Any],
    groups: Sequence[TensorGroup],
    *,
    action_bits: Sequence[int],
    decoder_state_codec: str | None,
) -> dict[str, Any]:
    codec = None if decoder_state_codec is None else str(decoder_state_codec).strip()
    if not codec:
        return {
            "metadata": {
                "bound": False,
                "method": "analytic_group_byte_proxy",
            },
            "groups": {},
        }
    torch_state = _state_dict_to_torch(state_dict)
    baseline_blob = serialize_decoder_state_dict(torch_state, codec=codec)
    baseline_bytes = len(baseline_blob)
    groups_out: dict[str, dict[int, dict[str, Any]]] = {}
    for group in groups:
        group_rows: dict[int, dict[str, Any]] = {}
        for bits in action_bits:
            mutated = dict(torch_state)
            candidate_values = _candidate_values_for_bits(group.values, bits=int(bits))
            mutated[group.name] = torch.from_numpy(
                np.asarray(candidate_values, dtype=np.float32).copy()
            )
            candidate_blob = serialize_decoder_state_dict(mutated, codec=codec)
            group_rows[int(bits)] = {
                "decoder_state_codec_requested": codec,
                "baseline_decoder_blob_bytes": int(baseline_bytes),
                "candidate_decoder_blob_bytes": len(candidate_blob),
                "decoder_blob_byte_delta": int(len(candidate_blob) - baseline_bytes),
                "candidate_decoder_codec_emitted": decoder_state_codec_stats(
                    candidate_blob
                ).codec,
            }
        groups_out[group.name] = group_rows
    return {
        "metadata": {
            "bound": True,
            "method": "measured_whole_decoder_state_codec_delta",
            "decoder_state_codec_requested": codec,
            "baseline_decoder_blob_bytes": int(baseline_bytes),
            "baseline_decoder_codec_stats": decoder_state_codec_stats(
                baseline_blob
            ).as_dict(),
            "calibrated_group_count": len(groups_out),
        },
        "groups": groups_out,
    }


def _state_dict_to_torch(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        out[str(name)] = torch.from_numpy(
            np.asarray(value, dtype=np.float32).copy()
        )
    return out


def _candidate_for_bits(
    values: np.ndarray,
    *,
    bits: int,
    saliency: float | None,
    baseline_bytes: int,
    byte_price_score_per_byte: float,
    zero_run_overhead_bytes: int,
    codec_byte_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    numel = int(values.size)
    if bits == 0:
        action = "zero_rle"
        estimated_bytes = int(zero_run_overhead_bytes)
    elif bits == 32:
        action = "fp32_protect"
        estimated_bytes = baseline_bytes
    elif bits == 16:
        action = "fp16"
        estimated_bytes = int(numel * 2)
    else:
        action = f"int{bits}"
        estimated_bytes = int((numel * bits + 7) // 8)
    q = _candidate_values_for_bits(values, bits=bits)
    mse_sum = float(np.sum((values - q) ** 2))
    analytic_byte_delta = int(estimated_bytes - baseline_bytes)
    measured_delta = (
        None
        if codec_byte_calibration is None
        else int(codec_byte_calibration["decoder_blob_byte_delta"])
    )
    byte_delta = analytic_byte_delta if measured_delta is None else measured_delta
    byte_delta_source = (
        "analytic_group_proxy"
        if measured_delta is None
        else "measured_decoder_state_codec_whole_blob_delta"
    )
    delta_rate = float(byte_delta) * float(byte_price_score_per_byte)
    delta_nonrate = None if saliency is None else float(saliency) * mse_sum
    delta_total = None if delta_nonrate is None else float(delta_nonrate) + delta_rate
    return {
        "action": action,
        "bits": int(bits),
        "estimated_bytes": int(estimated_bytes),
        "byte_delta": int(byte_delta),
        "analytic_byte_delta": int(analytic_byte_delta),
        "byte_delta_source": byte_delta_source,
        "decoder_state_codec_calibration": (
            None if codec_byte_calibration is None else dict(codec_byte_calibration)
        ),
        "quantization_error_sum": mse_sum,
        "delta_rate_score": delta_rate,
        "delta_nonrate_score_proxy": delta_nonrate,
        "delta_total_score_proxy": delta_total,
    }


def _candidate_values_for_bits(values: np.ndarray, *, bits: int) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if bits == 0:
        return np.zeros_like(values64, dtype=np.float64)
    if bits == 32:
        return values64.copy()
    if bits == 16:
        return values64.astype(np.float16).astype(np.float64)
    return _symmetric_quantize(values64, bits=bits)


def _symmetric_quantize(values: np.ndarray, *, bits: int) -> np.ndarray:
    if bits < 1 or bits > 16:
        raise NervDecoderWeightWaterfillError("int quantization bits must be in [1, 16]")
    levels = max(1, (1 << (bits - 1)) - 1)
    scale = float(np.max(np.abs(values))) / float(levels)
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    clipped = np.clip(np.round(values / scale), -levels, levels)
    return clipped * scale


def _section_value_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "section_id": row["group_name"],
        "row_kind": "existing_section_cut",
        "family": row["family"],
        "scope": row["scope"],
        "byte_delta": row["selected_byte_delta"],
        "section_bytes": row["baseline_fp32_bytes"],
        "delta_nonrate_score": row["selected_delta_nonrate_score_proxy"],
        "axis_tag": PLANNING_AXIS,
        "receiver_proof_status": row["receiver_proof_status"],
        "full_video_coverage": row["full_video_coverage"],
        "archive_sha256": row["archive_sha256"],
        "blockers": row["blockers"],
        "selected_action": row["selected_action"],
        **FALSE_AUTHORITY,
    }


def _saliency_for_name(name: str, saliency_by_name: Mapping[str, float]) -> float | None:
    exact = _finite_float_or_none(saliency_by_name.get(name))
    if exact is not None:
        return exact
    best_value = None
    best_len = -1
    for key, value in saliency_by_name.items():
        token = str(key)
        if token and token in name and len(token) > best_len:
            parsed = _finite_float_or_none(value)
            if parsed is not None:
                best_value = parsed
                best_len = len(token)
    return best_value


def _saliency_from_rows(rows: Sequence[Any]) -> dict[str, float]:
    out = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = row.get("group_name") or row.get("name") or row.get("section_id")
        value = _first_present(row, ("saliency", "decoder_weight_saliency", "score_saliency"))
        parsed = _finite_float_or_none(value)
        if name is not None and parsed is not None:
            out[str(name)] = parsed
    return out


def _saliency_from_mapping(mapping: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in mapping.items()
        if _finite_float_or_none(value) is not None
    }


def _saliency_from_nested_row_maps(rows: Sequence[Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        nested = row.get("saliency_by_name")
        if isinstance(nested, Mapping):
            out.update(_saliency_from_mapping(nested))
    return out


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _apply_saliency_floor(value: float, floor: float) -> float:
    return max(float(floor), float(value))


def _validated_action_bits(bits: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(sorted({int(bit) for bit in bits}))
    if not parsed:
        raise NervDecoderWeightWaterfillError("action_bits must not be empty")
    if 32 not in parsed:
        raise NervDecoderWeightWaterfillError("action_bits must include 32")
    for bit in parsed:
        if bit < 0 or bit > 32:
            raise NervDecoderWeightWaterfillError("action bits must be in [0, 32]")
        if bit not in set(DEFAULT_ACTION_BITS):
            raise NervDecoderWeightWaterfillError(
                "supported action bits are 0, 2, 4, 6, 7, 8, 16, 32"
            )
    return parsed


def _clean_tokens(tokens: Sequence[str], *, name: str) -> tuple[str, ...]:
    out = tuple(str(token) for token in tokens if str(token))
    if not out:
        raise NervDecoderWeightWaterfillError(f"{name} must not be empty")
    return out


def _required_candidate(candidates: Sequence[Mapping[str, Any]], *, bits: int) -> dict[str, Any]:
    for candidate in candidates:
        if int(candidate["bits"]) == int(bits):
            return dict(candidate)
    raise NervDecoderWeightWaterfillError(f"required action bit-depth missing: {bits}")


def _finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _sum_or_none(values: Sequence[float | None] | Any) -> float | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return float(sum(float(value) for value in materialized))


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _is_sha256_hex(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


__all__ = [
    "DEFAULT_ACTION_BITS",
    "DEFAULT_EXCLUDE_SUBSTRINGS",
    "DEFAULT_INCLUDE_SUBSTRINGS",
    "NERV_DECODER_WEIGHT_WATERFILL_SCHEMA",
    "TRUSTED_RECEIVER_PROOF_STATUSES",
    "NervDecoderWeightWaterfillError",
    "TensorGroup",
    "build_nerv_decoder_weight_waterfill_plan",
    "calibrate_saliency_by_name",
    "load_saliency_json",
    "load_state_npz",
    "load_state_npz_from_manifest",
    "render_nerv_decoder_weight_waterfill_markdown",
]
