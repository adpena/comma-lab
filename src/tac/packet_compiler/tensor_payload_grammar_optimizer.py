# SPDX-License-Identifier: MIT
"""Substrate-agnostic tensor payload grammar optimizer.

This is the generic companion to the PR101 fixed-schema solver.  It prices the
same tested byte-map/coder portfolio for arbitrary exported tensors, but never
claims PR101 runtime compatibility.  Future MLX/HPRC/Z8/HNeRV/NeRV-family
exports can use this as the first rate gate before writing a substrate-specific
receiver adapter or byte-closed archive.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from tac.archive_byte_profile import contest_rate_term
from tac.packet_compiler.pr101_decoder_byte_maps import (
    VALID_BYTE_MAP_STRATEGIES,
    decode_byte_map,
    encode_byte_map,
)
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    DEFAULT_CODERS,
    FALSE_AUTHORITY_FIELDS,
    CoderName,
    empirical_shannon_floor_bytes,
    measure_payload_coder_candidate,
    payload_saturation_diagnostic,
)

TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA = "tensor_payload_grammar_optimizer.v1"
TENSOR_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA = "tensor_payload_grammar_candidate.v1"
TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
TENSOR_PAYLOAD_SOURCE_MANIFEST_SCHEMA = "tensor_payload_source_manifest.v1"

QuantizationMode = Literal["symmetric_int8_fp16_scale", "already_quantized_int8"]
ScaleDType = Literal["fp16", "fp32"]
StoragePermMode = Literal["identity", "identity-plus-exhaustive4"]


def quantize_tensor_symmetric_int8(
    values: Any,
    *,
    n_quant: int = 127,
) -> tuple[np.ndarray, float]:
    """Quantize numeric tensor values with symmetric int8 scale.

    The returned ``q_i8`` is exact integer payload input for the grammar solver;
    dequantization fidelity is intentionally not promoted to score authority.
    """

    if not 1 <= int(n_quant) <= 127:
        raise ValueError("n_quant must be in [1, 127]")
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("tensor values must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("tensor values must be finite")
    max_abs = float(np.max(np.abs(arr)))
    if max_abs <= 0.0:
        scale = 1.0
        q = np.zeros(arr.shape, dtype=np.int8)
    else:
        scale = max_abs / float(n_quant)
        q = np.rint(arr / scale)
        q = np.clip(q, -int(n_quant), int(n_quant)).astype(np.int8)
    return q, float(scale)


def measure_tensor_payload_candidates(
    q_i8: np.ndarray,
    *,
    tensor_index: int,
    tensor_name: str | None = None,
    scale: float = 1.0,
    scale_dtypes: Sequence[ScaleDType] = ("fp16",),
    storage_perm_mode: StoragePermMode = "identity-plus-exhaustive4",
    byte_maps: Sequence[str] = tuple(sorted(VALID_BYTE_MAP_STRATEGIES)),
    coders: Sequence[CoderName] = DEFAULT_CODERS,
    brotli_quality: int = 11,
) -> list[dict[str, Any]]:
    """Measure exact payload-coder candidates for one already-quantized tensor."""

    q = _validate_q_i8(q_i8)
    scale_dtype_values = _normalize_scale_dtypes(scale_dtypes)
    name = _tensor_name(tensor_name or f"tensor_{tensor_index}")
    perms = _candidate_storage_perms(q.shape, storage_perm_mode)
    candidate_rows: list[dict[str, Any]] = []
    for perm in perms:
        for raw_byte_map in byte_maps:
            byte_map = str(raw_byte_map)
            if byte_map not in VALID_BYTE_MAP_STRATEGIES:
                raise ValueError(f"unknown byte_map strategy: {byte_map!r}")
            mapped_values = _apply_storage_perm(q, perm).reshape(-1)
            payload_without_scale = encode_byte_map(
                mapped_values.astype(np.int8, copy=False),
                byte_map,
            )
            transform_ok = _byte_map_storage_roundtrip_ok(
                payload_without_scale,
                original=q,
                byte_map=byte_map,
                perm=perm,
            )
            for scale_dtype in scale_dtype_values:
                scale_tail = _scale_tail_bytes(scale, scale_dtype=scale_dtype)
                payload = payload_without_scale + scale_tail
                floor = empirical_shannon_floor_bytes(payload)
                for coder in coders:
                    measured = measure_payload_coder_candidate(
                        payload,
                        coder=coder,
                        brotli_quality=brotli_quality,
                        brotli_lgwin_sweep=False,
                    )
                    status = str(measured["status"])
                    roundtrip_exact = bool(transform_ok and measured["roundtrip_exact"])
                    if not transform_ok:
                        status = "transform_roundtrip_failed"
                    charged = int(measured["charged_bytes"])
                    candidate_rows.append(
                        {
                            "schema": TENSOR_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA,
                            "tensor_index": int(tensor_index),
                            "tensor_name": name,
                            "tensor_shape": [int(v) for v in q.shape],
                            "n_elements": int(q.size),
                            "byte_map": byte_map,
                            "storage_perm": _perm_label(perm),
                            "scale_dtype": scale_dtype,
                            "scale": float(scale),
                            "scale_tail_bytes": len(scale_tail),
                            "coder": coder,
                            "coder_params": measured["coder_params"],
                            "charged_bytes": charged,
                            "codec_payload_bytes": int(measured["codec_payload_bytes"]),
                            "side_info_bytes": int(measured["side_info_bytes"]),
                            "raw_payload_bytes": len(payload),
                            "empirical_shannon_floor_bytes": floor,
                            "coded_over_floor_ratio": None
                            if floor <= 0.0
                            else charged / floor,
                            "transform_roundtrip_exact": bool(transform_ok),
                            "codec_roundtrip_exact": bool(measured["roundtrip_exact"]),
                            "roundtrip_exact": roundtrip_exact,
                            "status": status,
                            "runtime_consumption_status": (
                                "generic_tensor_payload_receiver_required"
                            ),
                            "byte_accounting_scope": (
                                "isolated_tensor_payload_not_archive_authority"
                            ),
                            "axis_tag": "[planning-only byte-profile]",
                            "blockers": _candidate_blockers(status=status),
                            **FALSE_AUTHORITY_FIELDS,
                        }
                    )
    candidate_rows.sort(
        key=lambda row: (
            row["status"] != "ok",
            row["roundtrip_exact"] is False,
            int(row["charged_bytes"]),
            str(row["coder"]),
            str(row["byte_map"]),
            str(row["storage_perm"]),
            str(row["scale_dtype"]),
        )
    )
    return candidate_rows


def select_best_tensor_payload_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the smallest exact generic tensor payload candidate."""

    exact = [
        dict(row)
        for row in candidates
        if row.get("status") == "ok" and bool(row.get("roundtrip_exact"))
    ]
    if not exact:
        raise ValueError("no exact tensor payload candidate was produced")
    exact.sort(
        key=lambda row: (
            int(row["charged_bytes"]),
            str(row["coder"]),
            str(row["byte_map"]),
            str(row["storage_perm"]),
            str(row["scale_dtype"]),
        )
    )
    selected = dict(exact[0])
    selected["selected"] = True
    return selected


def solve_tensor_payload_grammar(
    tensors: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    n_quant: int = 127,
    scale_dtypes: Sequence[ScaleDType] = ("fp16",),
    storage_perm_mode: StoragePermMode = "identity-plus-exhaustive4",
    byte_maps: Sequence[str] = tuple(sorted(VALID_BYTE_MAP_STRATEGIES)),
    coders: Sequence[CoderName] = DEFAULT_CODERS,
    brotli_quality: int = 11,
    baseline_coder: CoderName = "brotli",
    campaign_id: str = "tensor_payload_grammar",
    source_kind: str = "state_dict_mapping",
) -> dict[str, Any]:
    """Solve independent per-tensor payload grammar selection.

    Inputs may be a mapping of ``name -> numeric array`` or a sequence of rows
    with either ``values`` or ``q_i8`` plus optional ``scale``.  The output is a
    fail-closed planning artifact; a substrate-specific receiver must consume
    the selected grammar before any archive or score claim is valid.
    """

    scale_dtype_values = _normalize_scale_dtypes(scale_dtypes)
    normalized = _normalize_tensor_rows(tensors, n_quant=n_quant)
    rows: list[dict[str, Any]] = []
    selected_total = 0
    baseline_total = 0
    floor_total = 0.0
    source_rows: list[dict[str, Any]] = []
    for index, tensor in enumerate(normalized):
        candidates = measure_tensor_payload_candidates(
            tensor["q_i8"],
            tensor_index=index,
            tensor_name=tensor["name"],
            scale=float(tensor["scale"]),
            scale_dtypes=scale_dtype_values,
            storage_perm_mode=storage_perm_mode,
            byte_maps=byte_maps,
            coders=coders,
            brotli_quality=brotli_quality,
        )
        selected = select_best_tensor_payload_candidate(candidates)
        baseline = _baseline_candidate(
            candidates,
            baseline_coder=baseline_coder,
            baseline_byte_map="zig",
            baseline_storage_perm="identity",
            baseline_scale_dtype=scale_dtype_values[0],
        )
        selected_total += int(selected["charged_bytes"])
        baseline_total += int(baseline["charged_bytes"])
        floor_total += float(selected["empirical_shannon_floor_bytes"])
        source_rows.append(
            {
                "tensor_index": index,
                "tensor_name": tensor["name"],
                "tensor_shape": [int(v) for v in tensor["q_i8"].shape],
                "n_elements": int(tensor["q_i8"].size),
                "quantization_mode": tensor["quantization_mode"],
                "scale": float(tensor["scale"]),
                "q_i8_sha256": hashlib.sha256(tensor["q_i8"].tobytes()).hexdigest(),
            }
        )
        rows.append(
            {
                "schema": "tensor_payload_grammar_row.v1",
                "tensor_index": index,
                "tensor_name": tensor["name"],
                "tensor_shape": [int(v) for v in tensor["q_i8"].shape],
                "n_elements": int(tensor["q_i8"].size),
                "quantization_mode": tensor["quantization_mode"],
                "selected": selected,
                "baseline": baseline,
                "candidate_count": len(candidates),
                "top_candidates": candidates[:8],
            }
        )
    ratio = None if floor_total <= 0.0 else selected_total / floor_total
    blockers = [
        "generic_tensor_payload_receiver_not_bound",
        "byte_closed_archive_not_materialized",
        "runtime_consumption_proof_missing",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    return {
        "schema": TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
        "campaign_id": campaign_id,
        "source_payload_manifest": {
            "schema": TENSOR_PAYLOAD_SOURCE_MANIFEST_SCHEMA,
            "source_kind": source_kind,
            "tensor_count": len(source_rows),
            "n_quant": int(n_quant),
            "tensors": source_rows,
            "blockers": [],
        },
        "tensor_count": len(rows),
        "n_quant": int(n_quant),
        "scale_dtypes": list(scale_dtype_values),
        "storage_perm_mode": storage_perm_mode,
        "byte_maps": list(byte_maps),
        "coders": list(coders),
        "baseline_coder": baseline_coder,
        "brotli_quality": int(brotli_quality),
        "byte_accounting": {
            "selected_isolated_tensor_bytes": int(selected_total),
            "baseline_isolated_tensor_bytes": int(baseline_total),
            "selected_saved_bytes_vs_baseline": int(baseline_total - selected_total),
            "empirical_shannon_floor_bytes": float(floor_total),
            "selected_over_floor_ratio": ratio,
            "isolated_tensor_payload_rate_term_not_archive_authority": contest_rate_term(
                selected_total
            ),
        },
        "saturation_diagnostic": payload_saturation_diagnostic(ratio),
        "planner_feedback": _planner_feedback(rows),
        "rows": rows,
        "blockers": blockers,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "contest_rate_bytes_authority": False,
            "ready_for_exact_eval_dispatch": False,
            "reason": (
                "generic tensor grammar choices need a substrate receiver and "
                "byte-closed archive replay before score authority"
            ),
        },
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def build_tensor_payload_optimizer_queue(
    report: Mapping[str, Any],
    *,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Convert a generic tensor grammar report into a queue-consumable surface."""

    if report.get("schema") != TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA:
        raise ValueError("expected tensor_payload_grammar_optimizer.v1 report")
    cid = campaign_id or str(report.get("campaign_id") or "tensor_payload_grammar")
    feedback = report.get("planner_feedback")
    if not isinstance(feedback, Mapping):
        raise ValueError("tensor payload report missing planner_feedback")
    hints = feedback.get("operation_hints")
    if not isinstance(hints, Sequence) or isinstance(hints, (str, bytes)):
        raise ValueError("tensor payload report planner_feedback.operation_hints missing")
    candidates: list[dict[str, Any]] = []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        saved = max(0, -int(hint.get("isolated_byte_delta_vs_baseline") or 0))
        candidates.append(
            {
                "schema": "optimizer_candidate_queue_row_v1",
                "candidate_id": f"{cid}:{hint['operation_id']}",
                "candidate_kind": "planning_only_tensor_payload_grammar",
                "status": "blocked_planning_signal_only",
                "target_kind": "tensor_payload_grammar",
                "operation_family": "tensor_payload_grammar_selection",
                "operation_families": ["tensor_payload_grammar_selection"],
                "operation_id": hint["operation_id"],
                "operation_params": dict(hint),
                "selected_operations": [dict(hint)],
                "candidate_saved_bytes": saved,
                "saved_bytes_scope": "isolated_tensor_payload_only_not_archive_authority",
                "predicted_delta_bytes": hint.get("isolated_byte_delta_vs_baseline"),
                "predicted_delta_bytes_scope": (
                    "isolated_tensor_payload_only_not_archive_authority"
                ),
                "runtime_consumption_status": (
                    "generic_tensor_payload_receiver_required"
                ),
                "consumer_payload": {
                    "selected_operations": [dict(hint)],
                    "byte_accounting_scope": (
                        "isolated_tensor_payload_not_archive_authority"
                    ),
                },
                "blockers": [
                    "generic_tensor_payload_receiver_not_bound",
                    "byte_closed_archive_not_materialized",
                    "runtime_consumption_proof_missing",
                    "full_frame_inflate_parity_missing",
                    "contest_cpu_cuda_exact_eval_not_executed",
                ],
                "axis_tag": "[planning-only byte-profile]",
                **FALSE_AUTHORITY_FIELDS,
            }
        )
    return {
        "schema": TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA,
        "campaign_id": cid,
        "source_schema": report.get("schema"),
        "producer": "tac.packet_compiler.tensor_payload_grammar_optimizer",
        "proof_scope": "planning_only_generic_tensor_payload_grammar_no_dispatch",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_k": [row for row in candidates if int(row["candidate_saved_bytes"]) > 0],
        "blockers": [
            "generic_tensor_payload_receiver_not_bound",
            "byte_closed_archive_replay_not_run",
        ],
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
        ],
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def _normalize_tensor_rows(
    tensors: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    n_quant: int,
) -> list[dict[str, Any]]:
    if isinstance(tensors, Mapping):
        iterable = [{"name": name, "values": values} for name, values in tensors.items()]
    elif isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
        iterable = list(tensors)
    else:
        raise ValueError("tensors must be a mapping or sequence of tensor rows")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(iterable):
        if not isinstance(item, Mapping):
            raise ValueError(f"tensors[{index}] must be an object")
        name = _tensor_name(str(item.get("name") or item.get("tensor_name") or ""))
        if name in seen:
            raise ValueError(f"duplicate tensor name: {name!r}")
        seen.add(name)
        if "q_i8" in item and item.get("q_i8") is not None:
            q = _validate_q_i8(np.asarray(item["q_i8"]))
            scale = float(item.get("scale", 1.0))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"tensor {name!r} scale must be finite and positive")
            quantization_mode: QuantizationMode = "already_quantized_int8"
        else:
            if "values" not in item:
                raise ValueError(f"tensor {name!r} must provide values or q_i8")
            q, scale = quantize_tensor_symmetric_int8(item["values"], n_quant=n_quant)
            quantization_mode = "symmetric_int8_fp16_scale"
        rows.append(
            {
                "name": name,
                "q_i8": q,
                "scale": float(scale),
                "quantization_mode": quantization_mode,
            }
        )
    if not rows:
        raise ValueError("at least one tensor is required")
    return rows


def _baseline_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_coder: CoderName,
    baseline_byte_map: str,
    baseline_storage_perm: str,
    baseline_scale_dtype: ScaleDType,
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in candidates
        if row.get("coder") == baseline_coder
        and row.get("byte_map") == baseline_byte_map
        and row.get("storage_perm") == baseline_storage_perm
        and row.get("scale_dtype") == baseline_scale_dtype
        and row.get("status") == "ok"
        and bool(row.get("roundtrip_exact"))
    ]
    if not matches:
        matches = [
            dict(row)
            for row in candidates
            if row.get("coder") == baseline_coder
            and row.get("status") == "ok"
            and bool(row.get("roundtrip_exact"))
        ]
    if not matches:
        raise ValueError(f"no exact baseline candidate for coder {baseline_coder!r}")
    matches.sort(key=lambda row: (int(row["charged_bytes"]), str(row["byte_map"])))
    out = dict(matches[0])
    out["baseline"] = True
    return out


def _planner_feedback(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hints: list[dict[str, Any]] = []
    for row in rows:
        selected = row.get("selected")
        baseline = row.get("baseline")
        if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
            continue
        selected_bytes = int(selected["charged_bytes"])
        baseline_bytes = int(baseline["charged_bytes"])
        hints.append(
            {
                "schema": "tensor_payload_grammar_operation_hint.v1",
                "operation_family": "tensor_payload_grammar_selection",
                "operation_id": (
                    f"tensor_{int(selected['tensor_index']):04d}_"
                    f"{selected['coder']}_{selected['byte_map']}_"
                    f"{selected['storage_perm']}_{selected['scale_dtype']}"
                ).replace(",", "_"),
                "tensor_index": int(selected["tensor_index"]),
                "tensor_name": str(selected["tensor_name"]),
                "byte_map": str(selected["byte_map"]),
                "storage_perm": str(selected["storage_perm"]),
                "scale_dtype": str(selected["scale_dtype"]),
                "coder": str(selected["coder"]),
                "coder_params": dict(selected.get("coder_params") or {}),
                "selected_charged_bytes": selected_bytes,
                "baseline_charged_bytes": baseline_bytes,
                "isolated_byte_delta_vs_baseline": selected_bytes - baseline_bytes,
                "runtime_consumption_status": (
                    "generic_tensor_payload_receiver_required"
                ),
                "queue_consumable": False,
                "queue_consumable_blockers": [
                    "generic_tensor_payload_receiver_not_bound",
                    "byte_closed_archive_replay_not_run",
                ],
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    return {
        "schema": "tensor_payload_grammar_planner_feedback.v1",
        "operation_hint_count": len(hints),
        "rate_positive_hint_count": sum(
            1 for row in hints if int(row["isolated_byte_delta_vs_baseline"]) < 0
        ),
        "posterior_update_hooks": [
            "generic_tensor_entropy_gap_by_substrate",
            "generic_tensor_byte_map_coder_selection",
            "receiver_adapter_value_by_tensor_family",
        ],
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
            "tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_queue",
        ],
        "operation_hints": hints,
    }


def _candidate_blockers(*, status: str) -> list[str]:
    blockers = [
        "isolated_tensor_measurement_not_archive_authority",
        "generic_tensor_payload_receiver_not_bound",
        "byte_closed_archive_not_materialized",
        "runtime_consumption_proof_missing",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if status != "ok":
        blockers.append("codec_candidate_not_usable")
    return blockers


def _normalize_scale_dtypes(values: Sequence[ScaleDType]) -> tuple[ScaleDType, ...]:
    if not values:
        raise ValueError("scale_dtypes must not be empty")
    out: list[ScaleDType] = []
    for raw in values:
        value = str(raw)
        if value not in {"fp16", "fp32"}:
            raise ValueError(f"unknown scale_dtype: {value!r}")
        out.append(value)  # type: ignore[arg-type]
    return tuple(dict.fromkeys(out))


def _scale_tail_bytes(scale: float, *, scale_dtype: ScaleDType) -> bytes:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if scale_dtype == "fp16":
        return np.array([float(scale)], dtype=np.float16).tobytes()
    if scale_dtype == "fp32":
        return np.array([float(scale)], dtype=np.float32).tobytes()
    raise ValueError(f"unknown scale_dtype: {scale_dtype!r}")


def _candidate_storage_perms(
    shape: tuple[int, ...],
    mode: StoragePermMode,
) -> tuple[tuple[int, ...] | None, ...]:
    if mode == "identity":
        return (None,)
    if mode == "identity-plus-exhaustive4":
        if len(shape) != 4:
            return (None,)
        import itertools

        identity = tuple(range(4))
        return (
            None,
            *tuple(
                tuple(int(v) for v in perm)
                for perm in itertools.permutations(range(4))
                if tuple(perm) != identity
            ),
        )
    raise ValueError(f"unknown storage_perm_mode: {mode!r}")


def _apply_storage_perm(
    q_i8: np.ndarray,
    perm: tuple[int, ...] | None,
) -> np.ndarray:
    if perm is None:
        return q_i8.copy()
    if q_i8.ndim != len(perm):
        raise ValueError(f"perm {perm!r} does not match tensor ndim {q_i8.ndim}")
    return np.transpose(q_i8, perm).copy()


def _invert_storage_perm(
    flat: np.ndarray,
    *,
    original_shape: tuple[int, ...],
    perm: tuple[int, ...] | None,
) -> np.ndarray:
    if perm is None:
        return flat.reshape(original_shape).copy()
    stored_shape = tuple(original_shape[i] for i in perm)
    inverse = tuple(int(v) for v in np.argsort(perm))
    return np.transpose(flat.reshape(stored_shape), inverse).copy()


def _byte_map_storage_roundtrip_ok(
    payload_without_scale: bytes,
    *,
    original: np.ndarray,
    byte_map: str,
    perm: tuple[int, ...] | None,
) -> bool:
    decoded = decode_byte_map(payload_without_scale, byte_map)
    restored = _invert_storage_perm(decoded, original_shape=original.shape, perm=perm)
    return bool(np.array_equal(restored, original))


def _validate_q_i8(q_i8: np.ndarray) -> np.ndarray:
    q = np.asarray(q_i8)
    if q.dtype != np.int8:
        raise ValueError(f"q_i8 dtype must be int8; got {q.dtype}")
    if q.size == 0:
        raise ValueError("q_i8 must be non-empty")
    return q


def _tensor_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("tensor name must be non-empty")
    return text


def _perm_label(perm: tuple[int, ...] | None) -> str:
    if perm is None:
        return "identity"
    return ",".join(str(int(v)) for v in perm)


__all__ = [
    "TENSOR_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA",
    "TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA",
    "TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA",
    "TENSOR_PAYLOAD_SOURCE_MANIFEST_SCHEMA",
    "build_tensor_payload_optimizer_queue",
    "measure_tensor_payload_candidates",
    "quantize_tensor_symmetric_int8",
    "select_best_tensor_payload_candidate",
    "solve_tensor_payload_grammar",
]
