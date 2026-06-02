# SPDX-License-Identifier: MIT
"""Receiver-visible HiNeRV pruning, QuantNoise, and bitstream probes.

This module is the rate-side counterpart to the HiNeRV renderer/export path.
It deliberately performs real tensor transforms: magnitude pruning zeros decoder
weights, QuantNoise perturbs tensors on the codec grid, and the roundtrip probe
serializes through the same decoder-state codec consumed by HIV1 archives.

The output is local planning evidence only. It is useful to drive modelsize,
waterfill, and coder-QAT choices, but it is not scorer or promotion authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tac.score_geometry import CONTEST_REFERENCE_BYTES, RATE_COEFFICIENT
from tac.substrates._shared.decoder_state_codec import (
    decoder_state_codec_stats,
    deserialize_decoder_state_dict,
    serialize_decoder_state_dict,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    ADMIT,
    CUT,
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF = (
    "receiver_visible_prune_quantnoise_decoder_codec_roundtrip_v1"
)
HI_NERV_BITSTREAM_PREPARATION_SCHEMA = "hi_nerv_bitstream_preparation.v1"
HI_NERV_BITSTREAM_ROUNDTRIP_SCHEMA = "hi_nerv_bitstream_roundtrip_measurement.v1"
HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA = (
    "hi_nerv_bitstream_scorer_waterfill_selection.v1"
)
HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE = RATE_COEFFICIENT / CONTEST_REFERENCE_BYTES
HI_NERV_SUPPORTED_DECODER_CODECS: tuple[str, ...] = (
    "portfolio_auto",
    "int8_mixed",
    "int8_scale_bundled",
    "int4_mixed",
    "int4_scale_bundled",
    "int2_mixed",
    "int2_scale_bundled",
    "fp16_enveloped",
)


class HiNervBitstreamError(ValueError):
    """Raised when HiNeRV bitstream preparation is malformed."""


@dataclass(frozen=True)
class HinervBitstreamPreparation:
    """Prepared decoder state plus machine-readable transform metadata."""

    state_dict: dict[str, torch.Tensor]
    report: dict[str, Any]


def prepare_hi_nerv_decoder_bitstream_state(
    decoder_state_dict: Mapping[str, torch.Tensor],
    *,
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
) -> HinervBitstreamPreparation:
    """Apply real receiver-visible rate preparation to decoder weights.

    ``pruning_ratio`` is global magnitude pruning over floating decoder tensors.
    ``quant_noise_*`` adds deterministic uniform noise on the symmetric quantizer
    step used by the receiver codec.  Both transforms preserve tensor shapes and
    are recorded before the final archive codec serializes the state.
    """

    base = _clone_state(decoder_state_dict)
    pruned, pruning_report = apply_decoder_pruning(
        base,
        pruning_ratio=pruning_ratio,
    )
    prepared, quant_noise_report = apply_decoder_quant_noise(
        pruned,
        quant_bits=quant_noise_bits,
        noise_scale=quant_noise_scale,
        seed=quant_noise_seed,
    )
    report = {
        "schema": HI_NERV_BITSTREAM_PREPARATION_SCHEMA,
        "proof": HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF,
        "pruning": pruning_report,
        "quant_noise": quant_noise_report,
        "input_tensor_count": len(base),
        "output_tensor_count": len(prepared),
        "shape_preserved": _state_shapes(base) == _state_shapes(prepared),
        "zero_fraction_after_preparation": _zero_fraction(prepared),
        **FALSE_AUTHORITY,
    }
    return HinervBitstreamPreparation(state_dict=prepared, report=report)


def apply_decoder_pruning(
    decoder_state_dict: Mapping[str, torch.Tensor],
    *,
    pruning_ratio: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Globally prune the smallest-magnitude decoder weights."""

    ratio = float(pruning_ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise HiNervBitstreamError("pruning_ratio must be in [0, 1)")
    state = _clone_state(decoder_state_dict)
    candidates: list[torch.Tensor] = [
        tensor.detach().abs().reshape(-1).to(dtype=torch.float32, device="cpu")
        for name, tensor in state.items()
        if _is_prunable_tensor(name, tensor)
    ]
    total_prunable = int(sum(t.numel() for t in candidates))
    target_pruned = int(np.floor(ratio * total_prunable))
    threshold: float | None = None
    if target_pruned > 0 and candidates:
        values = torch.cat(candidates)
        threshold = float(torch.kthvalue(values, k=target_pruned).values.item())
        remaining_to_prune = target_pruned
        for name in sorted(state):
            tensor = state[name]
            if not _is_prunable_tensor(name, tensor):
                continue
            abs_tensor = tensor.detach().abs()
            below = abs_tensor < threshold
            equal = abs_tensor == threshold
            mask = below.clone()
            already = int(mask.sum().item())
            need = max(0, remaining_to_prune - already)
            if need and bool(equal.any()):
                equal_flat = equal.reshape(-1)
                selected = torch.nonzero(equal_flat, as_tuple=False).reshape(-1)[:need]
                flat_mask = mask.reshape(-1)
                flat_mask[selected] = True
                mask = flat_mask.reshape_as(mask)
            pruned_here = int(mask.sum().item())
            if pruned_here:
                tensor = tensor.clone()
                tensor[mask.to(device=tensor.device)] = 0.0
                state[name] = tensor
                remaining_to_prune -= pruned_here
            if remaining_to_prune <= 0:
                break
    actual_pruned = _zero_count(state) - _zero_count(decoder_state_dict)
    report = {
        "method": "global_magnitude_pruning",
        "pruning_ratio": ratio,
        "total_prunable_values": total_prunable,
        "target_pruned_values": target_pruned,
        "actual_new_zero_values": int(max(0, actual_pruned)),
        "threshold": threshold,
        "shape_preserved": _state_shapes(decoder_state_dict) == _state_shapes(state),
    }
    return state, report


def apply_decoder_quant_noise(
    decoder_state_dict: Mapping[str, torch.Tensor],
    *,
    quant_bits: int | None,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply deterministic QuantNoise on the symmetric receiver quant grid."""

    scale = float(noise_scale)
    if quant_bits is None or scale == 0.0:
        return _clone_state(decoder_state_dict), {
            "method": "disabled",
            "quant_bits": quant_bits,
            "noise_scale": scale,
            "seed": int(seed),
            "changed_tensor_count": 0,
            "max_abs_delta": 0.0,
        }
    bits = int(quant_bits)
    if bits not in (2, 4, 8):
        raise HiNervBitstreamError("quant_noise_bits must be one of 2, 4, 8")
    if scale < 0.0:
        raise HiNervBitstreamError("quant_noise_scale must be non-negative")

    rng = np.random.default_rng(int(seed))
    qmax = (1 << (bits - 1)) - 1
    out: dict[str, torch.Tensor] = {}
    max_delta = 0.0
    changed = 0
    for name, tensor in _clone_state(decoder_state_dict).items():
        if not torch.is_floating_point(tensor) or tensor.numel() == 0:
            out[name] = tensor
            continue
        arr = tensor.detach().to("cpu", dtype=torch.float32).numpy()
        abs_max = float(np.max(np.abs(arr))) if arr.size else 0.0
        if abs_max <= 0.0:
            out[name] = tensor
            continue
        step = abs_max / float(qmax)
        nonzero_mask = arr != 0.0
        noise = rng.uniform(-0.5, 0.5, size=arr.shape).astype(np.float32)
        delta = np.where(nonzero_mask, noise * (step * scale), 0.0).astype(
            np.float32,
            copy=False,
        )
        arr_noisy = arr + delta
        max_delta = max(max_delta, float(np.max(np.abs(delta))))
        changed += 1
        out[name] = torch.from_numpy(arr_noisy).to(dtype=tensor.dtype, device=tensor.device)
    return out, {
        "method": "uniform_symmetric_quant_grid_noise",
        "quant_bits": bits,
        "noise_scale": scale,
        "seed": int(seed),
        "changed_tensor_count": changed,
        "max_abs_delta": max_delta,
        "preserves_existing_zero_symbols": True,
        "shape_preserved": _state_shapes(decoder_state_dict) == _state_shapes(out),
    }


def measure_hi_nerv_decoder_bitstream_roundtrip(
    decoder_state_dict: Mapping[str, torch.Tensor],
    *,
    decoder_codecs: Sequence[str] = HI_NERV_SUPPORTED_DECODER_CODECS,
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
) -> dict[str, Any]:
    """Measure receiver codec bytes and decode error for a prepared state."""

    prepared = prepare_hi_nerv_decoder_bitstream_state(
        decoder_state_dict,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
    )
    rows = []
    for codec in _validate_codecs(decoder_codecs):
        blob = serialize_decoder_state_dict(prepared.state_dict, codec=codec)
        decoded = deserialize_decoder_state_dict(blob)
        rows.append(
            {
                "decoder_codec_requested": codec,
                "decoder_codec_emitted": decoder_state_codec_stats(blob).codec,
                "blob_bytes": len(blob),
                "codec_stats": decoder_state_codec_stats(blob).as_dict(),
                "roundtrip_error": _roundtrip_error(prepared.state_dict, decoded),
                "decoded_tensor_count": len(decoded),
                "shape_preserved": _state_shapes(prepared.state_dict)
                == _state_shapes(decoded),
                **FALSE_AUTHORITY,
            }
        )
    rows.sort(key=lambda row: int(row["blob_bytes"]))
    selection = select_hi_nerv_bitstream_codec_by_scorer_waterfill(
        rows,
        scorer_value_rows=None,
        baseline_codec=None,
    )
    return {
        "schema": HI_NERV_BITSTREAM_ROUNDTRIP_SCHEMA,
        "proof": HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF,
        "preparation": prepared.report,
        "decoder_codecs": list(_validate_codecs(decoder_codecs)),
        "rows": rows,
        "best_row": rows[0] if rows else None,
        "portfolio_selection": selection,
        "blockers": [
            "hi_nerv_bitstream_roundtrip_is_local_rate_distortion_evidence_only",
            "contest_cpu_cuda_exact_eval_not_executed",
            "score_sensitivity_replay_not_attached",
        ],
        **FALSE_AUTHORITY,
    }


def select_hi_nerv_bitstream_codec_by_scorer_waterfill(
    codec_rows: Sequence[Mapping[str, Any]],
    *,
    scorer_value_rows: Sequence[Mapping[str, Any]] | None = None,
    baseline_codec: str | None = None,
    candidate_id: str | None = None,
    archive_sha256: str | None = None,
    axis_tag: str = "[planning/control]",
    receiver_proof_status: str = "missing",
    full_video_coverage: bool = False,
) -> dict[str, Any]:
    """Select decoder codec rows by the contest Lagrangian when evidence exists.

    The rule is intentionally simple and exact for this section-level decision:
    a codec is admissible only when the measured non-rate scorer change plus
    the charged byte-rate change is negative.  Without scorer-value rows this
    returns a byte-only planning surface with an explicit blocker.
    """

    rows = [_as_codec_row(row) for row in codec_rows]
    if not rows:
        raise HiNervBitstreamError("at least one codec row is required")
    baseline = _resolve_baseline_codec_row(rows, baseline_codec=baseline_codec)
    value_by_codec = {
        _codec_key(row): row
        for row in (scorer_value_rows or ())
        if _codec_key(row)
    }
    section_value_rows = [
        _codec_section_value_row(
            row,
            baseline=baseline,
            scorer_value=value_by_codec.get(_codec_key(row)),
            candidate_id=candidate_id,
            archive_sha256=archive_sha256,
            axis_tag=axis_tag,
            receiver_proof_status=receiver_proof_status,
            full_video_coverage=full_video_coverage,
        )
        for row in rows
    ]
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA,
            "candidate_id": candidate_id,
            "family": "hi_nerv",
            "axis_tag": axis_tag,
            "archive_sha256": archive_sha256,
            "receiver_proof_status": receiver_proof_status,
            "full_video_coverage": bool(full_video_coverage),
            "section_value_rows": section_value_rows,
        }
    )
    decision_by_row_id = {
        str(row["row_id"]): row for row in byte_price_plan.get("decision_rows", ())
    }
    ranked: list[dict[str, Any]] = []
    for row in rows:
        codec_key = _codec_key(row)
        value = value_by_codec.get(codec_key)
        scorer_attached = value is not None
        delta_nonrate = (
            _float_value(value.get("delta_nonrate_score"), 0.0)
            if value is not None
            else None
        )
        byte_delta = int(row["blob_bytes"]) - int(baseline["blob_bytes"])
        rate_delta = float(byte_delta * HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE)
        net_delta = None if delta_nonrate is None else float(delta_nonrate + rate_delta)
        row_id = _codec_row_id(codec_key)
        decision_row = decision_by_row_id.get(row_id, {})
        economic_admissible = str(decision_row.get("economic_decision")) in {
            ADMIT,
            CUT,
        }
        authority_admissible = str(decision_row.get("decision")) in {ADMIT, CUT}
        ranked.append(
            {
                "row_id": row_id,
                "decoder_codec_requested": row["decoder_codec_requested"],
                "decoder_codec_emitted": row.get("decoder_codec_emitted"),
                "blob_bytes": int(row["blob_bytes"]),
                "baseline_blob_bytes": int(baseline["blob_bytes"]),
                "byte_delta_vs_baseline": int(byte_delta),
                "rate_score_delta_vs_baseline": rate_delta,
                "scorer_value_attached": scorer_attached,
                "delta_nonrate_score": delta_nonrate,
                "net_score_delta": net_delta,
                "waterfill_economic_admissible": bool(economic_admissible),
                "waterfill_admissible": bool(authority_admissible),
                "canonical_economic_decision": decision_row.get("economic_decision"),
                "canonical_decision": decision_row.get("decision"),
                "canonical_blockers": list(decision_row.get("blockers") or ()),
                "canonical_section_value_row": next(
                    (
                        dict(section_row)
                        for section_row in section_value_rows
                        if section_row["row_id"] == row_id
                    ),
                    {},
                ),
                "scorer_value_row": dict(value or {}),
                **FALSE_AUTHORITY,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["net_score_delta"] is None,
            float(row["net_score_delta"] if row["net_score_delta"] is not None else 0.0),
            int(row["blob_bytes"]),
        )
    )
    admissible = [row for row in ranked if row["waterfill_admissible"]]
    economic_admissible = [
        row for row in ranked if row["waterfill_economic_admissible"]
    ]
    blockers = [
        "hi_nerv_bitstream_selection_is_false_authority_until_full_replay",
        "contest_cpu_cuda_exact_eval_not_executed",
        *list(byte_price_plan.get("blockers") or ()),
    ]
    if not value_by_codec:
        blockers.append("hi_nerv_bitstream_scorer_value_replay_missing")
    if value_by_codec and not economic_admissible:
        blockers.append("hi_nerv_bitstream_no_codec_clears_contest_byte_price")
    return {
        "schema": HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA,
        "selection_rule": "admit iff delta_nonrate_score + byte_delta*25/N_ref < 0",
        "rate_score_per_byte": HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE,
        "baseline_codec": baseline["decoder_codec_requested"],
        "baseline_blob_bytes": int(baseline["blob_bytes"]),
        "scorer_value_replay_attached": bool(value_by_codec),
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "ranked_rows": ranked,
        "admissible_rows": admissible,
        "economic_admissible_rows": economic_admissible,
        "selected_row": admissible[0] if admissible else None,
        "selected_economic_row": (
            economic_admissible[0] if economic_admissible else None
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _codec_section_value_row(
    row: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    scorer_value: Mapping[str, Any] | None,
    candidate_id: str | None,
    archive_sha256: str | None,
    axis_tag: str,
    receiver_proof_status: str,
    full_video_coverage: bool,
) -> dict[str, Any]:
    codec_key = _codec_key(row)
    byte_delta = int(row["blob_bytes"]) - int(baseline["blob_bytes"])
    return {
        "row_id": _codec_row_id(codec_key),
        "section_id": f"hi_nerv_decoder_codec:{codec_key}",
        "row_kind": (
            "new_residual_or_sidecar"
            if byte_delta > 0
            else "existing_section_cut"
        ),
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "scope": "decoder_codec_replacement",
        "byte_delta": int(byte_delta),
        "section_bytes": abs(int(byte_delta)) or int(row["blob_bytes"]),
        "delta_nonrate_score": (
            None
            if scorer_value is None
            else _float_value(scorer_value.get("delta_nonrate_score"), 0.0)
        ),
        "axis_tag": axis_tag,
        "receiver_proof_status": receiver_proof_status,
        "full_video_coverage": bool(full_video_coverage),
        "archive_sha256": archive_sha256,
        "decoder_codec_requested": row.get("decoder_codec_requested"),
        "decoder_codec_emitted": row.get("decoder_codec_emitted"),
        "baseline_codec": baseline.get("decoder_codec_requested"),
        "baseline_blob_bytes": int(baseline["blob_bytes"]),
        "candidate_blob_bytes": int(row["blob_bytes"]),
        **FALSE_AUTHORITY,
    }


def _codec_row_id(codec_key: str) -> str:
    return f"hi_nerv_decoder_codec:{codec_key}"


def _validate_codecs(values: Iterable[str]) -> tuple[str, ...]:
    rows = tuple(str(value).strip().lower() for value in values if str(value).strip())
    bad = [value for value in rows if value not in HI_NERV_SUPPORTED_DECODER_CODECS]
    if bad:
        raise HiNervBitstreamError(f"unsupported HiNeRV decoder codec(s): {bad}")
    if not rows:
        raise HiNervBitstreamError("at least one decoder codec is required")
    return tuple(dict.fromkeys(rows))


def _as_codec_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if "blob_bytes" not in row:
        raise HiNervBitstreamError("codec row missing blob_bytes")
    codec = str(
        row.get("decoder_codec_requested")
        or row.get("decoder_codec")
        or row.get("codec")
        or ""
    ).strip().lower()
    if not codec:
        raise HiNervBitstreamError("codec row missing decoder codec id")
    return {
        **dict(row),
        "decoder_codec_requested": codec,
        "blob_bytes": int(row["blob_bytes"]),
    }


def _codec_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("decoder_codec_requested")
        or row.get("decoder_codec")
        or row.get("codec")
        or ""
    ).strip().lower()


def _resolve_baseline_codec_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_codec: str | None,
) -> Mapping[str, Any]:
    if baseline_codec:
        wanted = str(baseline_codec).strip().lower()
        for row in rows:
            if _codec_key(row) == wanted:
                return row
        raise HiNervBitstreamError(f"baseline codec not found: {baseline_codec!r}")
    for row in rows:
        if _codec_key(row) == "fp16_enveloped":
            return row
    return max(rows, key=lambda row: int(row["blob_bytes"]))


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().clone()
        for name, tensor in state.items()
    }


def _is_prunable_tensor(name: str, tensor: torch.Tensor) -> bool:
    lowered = str(name).lower()
    return (
        torch.is_floating_point(tensor)
        and tensor.numel() > 0
        and tensor.dim() >= 2
        and "norm" not in lowered
        and "bias" not in lowered
        and "gamma" not in lowered
    )


def _state_shapes(state: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {str(name): tuple(int(v) for v in tensor.shape) for name, tensor in state.items()}


def _zero_count(state: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for tensor in state.values():
        if torch.is_floating_point(tensor) or tensor.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            total += int((tensor.detach() == 0).sum().item())
    return total


def _zero_fraction(state: Mapping[str, torch.Tensor]) -> float:
    total = 0
    zeros = 0
    for tensor in state.values():
        if not torch.is_floating_point(tensor):
            continue
        total += int(tensor.numel())
        zeros += int((tensor.detach() == 0).sum().item())
    return float(zeros / total) if total else 0.0


def _roundtrip_error(
    reference: Mapping[str, torch.Tensor],
    decoded: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    missing = sorted(set(reference) - set(decoded))
    unexpected = sorted(set(decoded) - set(reference))
    max_abs = 0.0
    mean_abs_num = 0.0
    mean_abs_den = 0
    for name in sorted(set(reference) & set(decoded)):
        ref = reference[name].detach().to("cpu", dtype=torch.float32)
        got = decoded[name].detach().to("cpu", dtype=torch.float32)
        if tuple(ref.shape) != tuple(got.shape):
            missing.append(f"shape_mismatch:{name}")
            continue
        err = torch.abs(ref - got)
        if err.numel():
            max_abs = max(max_abs, float(err.max().item()))
            mean_abs_num += float(err.sum().item())
            mean_abs_den += int(err.numel())
    return {
        "max_abs": max_abs,
        "mean_abs": float(mean_abs_num / mean_abs_den) if mean_abs_den else 0.0,
        "missing": missing,
        "unexpected": unexpected,
        "tensor_count": len(reference),
    }


__all__ = [
    "HI_NERV_BITSTREAM_PREPARATION_SCHEMA",
    "HI_NERV_BITSTREAM_RATE_SCORE_PER_BYTE",
    "HI_NERV_BITSTREAM_ROUNDTRIP_SCHEMA",
    "HI_NERV_BITSTREAM_WATERFILL_SELECTION_SCHEMA",
    "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
    "HI_NERV_SUPPORTED_DECODER_CODECS",
    "HiNervBitstreamError",
    "HinervBitstreamPreparation",
    "apply_decoder_pruning",
    "apply_decoder_quant_noise",
    "measure_hi_nerv_decoder_bitstream_roundtrip",
    "prepare_hi_nerv_decoder_bitstream_state",
    "select_hi_nerv_bitstream_codec_by_scorer_waterfill",
]
