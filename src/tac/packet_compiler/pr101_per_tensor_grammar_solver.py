# SPDX-License-Identifier: MIT
"""Per-tensor decoder-weight grammar solver for PR101/HNeRV-style packets.

This module is a deterministic, score-safe profiler.  It does not claim a
contest score and it does not emit a submission packet.  Its job is to turn the
manual PR101/PR103 decoder-weight grammar choices into reusable system
intelligence:

* per tensor byte-map and storage-permutation candidates;
* codec candidates over Brotli, raw LZMA1, canonical Huffman, and optional
  constriction range coding;
* empirical Shannon floor and coded/floor saturation diagnostics;
* a fail-closed report that says which choices still need receiver/runtime
  adapter work before they can become exact-ready materializers.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import itertools
import json
import lzma
import math
import struct
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import brotli
import numpy as np

from tac.archive_byte_profile import contest_rate_term
from tac.packet_compiler.pr101_decoder_byte_maps import (
    VALID_BYTE_MAP_STRATEGIES,
    decode_byte_map,
    encode_byte_map,
)
from tac.pr101_split_brotli_codec import (
    CONV4_STORAGE_PERMS,
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    LATENT_LZMA_FILTERS,
    N_QUANT,
    _quantize_tensor,
    decode_decoder_compact,
    decompress_brotli_streams,
    encode_decoder_compact,
    pack_brotli_stream,
)

PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA = "pr101_per_tensor_grammar_solver.v1"
PR101_TENSOR_GRAMMAR_CANDIDATE_SCHEMA = "pr101_tensor_grammar_candidate.v1"
PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA = "pr101_grouped_brotli_packet_grammar.v1"
PR101_GROUPED_DECODER_BLOB_MATERIALIZATION_SCHEMA = (
    "pr101_grouped_decoder_blob_materialization.v1"
)
PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA = (
    "pr101_grouped_archive_materialization.v1"
)
PR101_U32_RECEIVER_ADAPTER_SOURCE_SCHEMA = "pr101_u32_receiver_adapter_source.v1"
PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA = "pr101_u32_runtime_tree_materialization.v1"
PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
PR101_INNER_MEMBER_NAME = "x"
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_provider_dispatch": False,
    "dispatch_attempted": False,
}
StoragePermMode = Literal["identity", "pr101-plus-identity", "exhaustive-conv4"]
CoderName = Literal[
    "brotli",
    "lzma_raw",
    "canonical_huffman",
    "range_ac_empirical_hist_u16",
]


def measure_tensor_grammar_candidates(
    q_i8: np.ndarray,
    *,
    tensor_index: int,
    tensor_name: str | None = None,
    scale: float = 1.0,
    storage_perm_mode: StoragePermMode = "pr101-plus-identity",
    byte_maps: Sequence[str] = tuple(sorted(VALID_BYTE_MAP_STRATEGIES)),
    coders: Sequence[CoderName] = ("brotli", "lzma_raw", "canonical_huffman"),
    brotli_quality: int = 11,
    brotli_quality_values: Sequence[int] | None = None,
    brotli_lgwin_sweep: bool = False,
) -> list[dict[str, Any]]:
    """Measure all requested transform/coder candidates for one tensor.

    The input is already quantized int8.  Every ``status=="ok"`` candidate has
    been decoded back to the exact charged byte payload; every byte-map/storage
    transform has also been inverted back to ``q_i8``.  This prevents
    "metadata-only" fake codec branches.
    """

    q = _validate_q_i8(q_i8)
    name = tensor_name or f"tensor_{tensor_index}"
    perms = _candidate_storage_perms(q.shape, tensor_index, storage_perm_mode)
    quality_values = _brotli_quality_values(
        brotli_quality=brotli_quality,
        brotli_quality_values=brotli_quality_values,
    )
    candidates: list[dict[str, Any]] = []
    for perm in perms:
        for raw_byte_map in byte_maps:
            byte_map = str(raw_byte_map)
            if byte_map not in VALID_BYTE_MAP_STRATEGIES:
                raise ValueError(f"unknown byte_map strategy: {byte_map!r}")
            mapped = _build_transformed_tensor_payload(
                q,
                scale=scale,
                byte_map=byte_map,
                perm=perm,
            )
            transform_ok = _byte_map_storage_roundtrip_ok(
                mapped[:-2], original=q, byte_map=byte_map, perm=perm
            )
            floor = empirical_shannon_floor_bytes(mapped)
            for coder in coders:
                for quality in quality_values if coder == "brotli" else (brotli_quality,):
                    candidate = _measure_coder(
                        mapped,
                        coder=coder,
                        brotli_quality=quality,
                        brotli_lgwin_sweep=brotli_lgwin_sweep,
                    )
                    charged = int(candidate["charged_bytes"])
                    runtime_status = _runtime_consumption_status(
                        tensor_index=tensor_index,
                        byte_map=byte_map,
                        perm=perm,
                        coder=coder,
                    )
                    candidates.append(
                        {
                            "schema": PR101_TENSOR_GRAMMAR_CANDIDATE_SCHEMA,
                            "tensor_index": int(tensor_index),
                            "tensor_name": name,
                            "tensor_shape": [int(v) for v in q.shape],
                            "n_elements": int(q.size),
                            "byte_map": byte_map,
                            "storage_perm": _perm_label(perm),
                            "coder": coder,
                            "coder_params": candidate["coder_params"],
                            "charged_bytes": charged,
                            "codec_payload_bytes": int(candidate["codec_payload_bytes"]),
                            "side_info_bytes": int(candidate["side_info_bytes"]),
                            "raw_payload_bytes": len(mapped),
                            "empirical_shannon_floor_bytes": floor,
                            "coded_over_floor_ratio": (
                                None if floor <= 0.0 else charged / floor
                            ),
                            "transform_roundtrip_exact": bool(transform_ok),
                            "codec_roundtrip_exact": bool(candidate["roundtrip_exact"]),
                            "roundtrip_exact": bool(
                                transform_ok and candidate["roundtrip_exact"]
                            ),
                            "status": candidate["status"]
                            if transform_ok
                            else "transform_roundtrip_failed",
                            "runtime_consumption_status": runtime_status,
                            "byte_accounting_scope": (
                                "isolated_tensor_payload_not_grouped_pr101_archive"
                            ),
                            "axis_tag": "[planning-only byte-profile]",
                            "blockers": _candidate_blockers(runtime_status),
                            **FALSE_AUTHORITY_FIELDS,
                        }
                    )
    candidates.sort(
        key=lambda row: (
            row["status"] != "ok",
            row["roundtrip_exact"] is False,
            int(row["charged_bytes"]),
            str(row["coder"]),
            str(row["byte_map"]),
            str(row["storage_perm"]),
        )
    )
    return candidates


def select_best_tensor_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the smallest exact candidate from ``measure_tensor_grammar_candidates``."""

    exact = [
        dict(row)
        for row in candidates
        if row.get("status") == "ok" and bool(row.get("roundtrip_exact"))
    ]
    if not exact:
        raise ValueError("no exact tensor grammar candidate was produced")
    exact.sort(
        key=lambda row: (
            int(row["charged_bytes"]),
            str(row["coder"]),
            str(row["byte_map"]),
            str(row["storage_perm"]),
        )
    )
    selected = dict(exact[0])
    selected["selected"] = True
    return selected


def solve_state_dict_per_tensor_grammar(
    state_dict: Mapping[str, Any],
    *,
    n_quant: int = N_QUANT,
    storage_perm_mode: StoragePermMode = "pr101-plus-identity",
    coders: Sequence[CoderName] = ("brotli", "lzma_raw", "canonical_huffman"),
    brotli_quality: int = 11,
    brotli_quality_values: Sequence[int] | None = None,
    brotli_lgwin_sweep: bool = False,
    max_tensors: int | None = None,
    include_current_grouped_pr101: bool = True,
) -> dict[str, Any]:
    """Profile a full PR101 fixed-schema ``state_dict``.

    The selected per-tensor rows are isolated candidate measurements.  Brotli
    split-stream interactions are separately reported through
    ``current_grouped_pr101_brotli_bytes`` when requested.
    """

    schema = FIXED_STATE_SCHEMA if max_tensors is None else FIXED_STATE_SCHEMA[:max_tensors]
    rows: list[dict[str, Any]] = []
    for idx, (name, expected_shape) in enumerate(schema):
        if name not in state_dict:
            raise ValueError(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=n_quant)
        if tuple(qt.shape) != tuple(expected_shape):
            raise ValueError(
                f"shape mismatch for {name!r}: expected {expected_shape}, got {qt.shape}"
            )
        candidates = measure_tensor_grammar_candidates(
            qt.q_i8,
            tensor_index=idx,
            tensor_name=name,
            scale=qt.scale,
            storage_perm_mode=storage_perm_mode,
            coders=coders,
            brotli_quality=brotli_quality,
            brotli_quality_values=brotli_quality_values,
            brotli_lgwin_sweep=brotli_lgwin_sweep,
        )
        selected = select_best_tensor_candidate(candidates)
        current = _find_current_pr101_isolated_candidate(candidates, tensor_index=idx)
        rows.append(
            {
                "schema": "pr101_per_tensor_grammar_solver_row.v1",
                "tensor_index": idx,
                "tensor_name": name,
                "tensor_shape": list(expected_shape),
                "selected": selected,
                "current_pr101_isolated": current,
                "top_candidates": candidates[:8],
            }
        )

    total_selected = sum(int(row["selected"]["charged_bytes"]) for row in rows)
    total_floor = sum(
        float(row["selected"]["empirical_shannon_floor_bytes"]) for row in rows
    )
    current_isolated = sum(
        int(row["current_pr101_isolated"]["charged_bytes"])
        for row in rows
        if row["current_pr101_isolated"] is not None
    )
    current_grouped = None
    if include_current_grouped_pr101 and max_tensors is None:
        current_grouped = len(
            encode_decoder_compact(
                dict(state_dict),
                brotli_quality=brotli_quality,
            )
        )
    ratio = None if total_floor <= 0 else total_selected / total_floor
    return {
        "schema": PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA,
        "n_tensors": len(rows),
        "n_quant": int(n_quant),
        "storage_perm_mode": storage_perm_mode,
        "coders": list(coders),
        "brotli_quality": int(brotli_quality),
        "brotli_quality_values": list(
            _brotli_quality_values(
                brotli_quality=brotli_quality,
                brotli_quality_values=brotli_quality_values,
            )
        ),
        "brotli_lgwin_sweep": bool(brotli_lgwin_sweep),
        "partial_schema_sample": max_tensors is not None,
        "byte_accounting": {
            "selected_isolated_tensor_bytes": int(total_selected),
            "current_pr101_isolated_tensor_bytes": int(current_isolated),
            "current_grouped_pr101_brotli_bytes": current_grouped,
            "empirical_shannon_floor_bytes": total_floor,
            "selected_over_floor_ratio": ratio,
            "isolated_tensor_payload_rate_term_not_archive_authority": contest_rate_term(
                total_selected
            ),
            "isolated_savings_vs_current_pr101_bytes": int(
                current_isolated - total_selected
            ),
        },
        "saturation_diagnostic": _saturation_diagnostic(ratio),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "contest_rate_bytes_authority": False,
            "ready_for_exact_eval_dispatch": False,
            "reason": (
                "per-tensor codec choices are profiler output until a "
                "receiver/runtime adapter consumes the emitted grammar and a "
                "byte-closed archive replay passes"
            ),
        },
        "planner_feedback": _planner_feedback(rows),
        "rows": rows,
    }


def build_optimizer_candidate_queue_from_solver_report(
    report: Mapping[str, Any],
    *,
    campaign_id: str = "pr101_per_tensor_grammar_solver",
) -> dict[str, Any]:
    """Convert a solver report into a planning-only optimizer queue.

    This intentionally does **not** make per-tensor profiler winners exact-ready.
    PR101's current split-Brotli packet has grouped/window context, so isolated
    tensor winners are dispatch-blocked until a grouped packet compiler and
    receiver/runtime proof consume the selected grammar.
    """

    if report.get("schema") != PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA:
        raise ValueError("expected pr101_per_tensor_grammar_solver.v1 report")
    feedback = report.get("planner_feedback")
    if not isinstance(feedback, Mapping):
        raise ValueError("solver report missing planner_feedback")
    hints = feedback.get("operation_hints")
    if not isinstance(hints, Sequence) or isinstance(hints, (str, bytes)):
        raise ValueError("solver report planner_feedback.operation_hints missing")

    candidates: list[dict[str, Any]] = []
    for hint in hints:
        if not isinstance(hint, Mapping):
            continue
        delta = hint.get("isolated_byte_delta_vs_current_pr101")
        saved_bytes = 0
        if isinstance(delta, int | float) and delta < 0:
            saved_bytes = int(-delta)
        blockers = [
            "isolated_tensor_winner_not_grouped_packet_authority",
            "grouped_split_brotli_packet_selection_not_run",
            "receiver_adapter_not_emitted",
            "runtime_consumption_proof_missing",
            "full_frame_inflate_parity_missing",
            "byte_closed_archive_not_materialized",
        ]
        candidates.append(
            {
                "schema": "optimizer_candidate_queue_row_v1",
                "candidate_id": f"{campaign_id}:{hint['operation_id']}",
                "candidate_kind": "planning_only_pr101_tensor_grammar_hint",
                "status": "blocked_planning_signal_only",
                "target_kind": "decoder_weight_tensor_grammar",
                "operation_family": "pr101_per_tensor_grammar_selection",
                "operation_families": ["pr101_per_tensor_grammar_selection"],
                "operation_id": hint["operation_id"],
                "operation_params": {
                    "tensor_index": hint["tensor_index"],
                    "tensor_name": hint["tensor_name"],
                    "byte_map": hint["byte_map"],
                    "storage_perm": hint["storage_perm"],
                    "coder": hint["coder"],
                    "coder_params": hint.get("coder_params") or {},
                },
                "selected_operations": [dict(hint)],
                "candidate_saved_bytes": saved_bytes,
                "saved_bytes_scope": (
                    "isolated_tensor_payload_only_not_grouped_pr101_archive"
                ),
                "consumer_payload": {
                    "selected_operations": [dict(hint)],
                    "byte_accounting_scope": (
                        "isolated_tensor_payload_not_grouped_pr101_archive"
                    ),
                },
                "predicted_delta_bytes": delta,
                "predicted_delta_bytes_scope": (
                    "isolated_tensor_payload_only_not_archive_authority"
                ),
                "runtime_consumption_status": hint.get("runtime_consumption_status"),
                "consumer_surfaces": [
                    "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
                    "tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_queue",
                ],
                "blockers": blockers,
                "axis_tag": "[planning-only byte-profile]",
                **FALSE_AUTHORITY_FIELDS,
            }
        )
    return {
        "schema": PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA,
        "campaign_id": campaign_id,
        "source_schema": report.get("schema"),
        "source_partial_schema_sample": bool(report.get("partial_schema_sample")),
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "proof_scope": "planning_only_isolated_tensor_grammar_no_dispatch",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_k": candidates,
        "blockers": [
            "isolated_tensor_winner_not_grouped_packet_authority",
            "grouped_split_brotli_packet_selection_not_run",
            "receiver_adapter_not_emitted",
            "byte_closed_archive_replay_not_run",
        ],
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
        ],
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def solve_grouped_brotli_packet_grammar(
    state_dict: Mapping[str, Any],
    *,
    n_quant: int = N_QUANT,
    selected_transform_mode: Literal["stock_pr101", "best_brotli_per_tensor"] = "best_brotli_per_tensor",
    storage_perm_mode: StoragePermMode = "pr101-plus-identity",
    storage_order: Sequence[int] | None = None,
    exact_stream_count: int | None = len(DECODER_STREAM_ENDS),
    max_streams: int = len(DECODER_STREAM_ENDS),
    brotli_quality: int = 11,
    brotli_lgwin_sweep: bool = False,
    max_tensors: int | None = None,
) -> dict[str, Any]:
    """Solve the grouped split-Brotli packet layer for PR101 decoder weights.

    ``measure_tensor_grammar_candidates`` is intentionally isolated.  This
    function prices the next layer that actually matters for PR101-style
    packets: concatenate transformed tensor payloads in storage order, split
    them into Brotli windows, and solve the split-point DP on the real
    concatenated bytes.  The output is still not a submission packet; non-stock
    transform/order/split choices need a receiver adapter and full
    inflate/eval replay before dispatch.
    """

    if selected_transform_mode not in {"stock_pr101", "best_brotli_per_tensor"}:
        raise ValueError(f"unknown selected_transform_mode: {selected_transform_mode!r}")
    schema = FIXED_STATE_SCHEMA if max_tensors is None else FIXED_STATE_SCHEMA[:max_tensors]
    n_tensors = len(schema)
    if n_tensors <= 0:
        raise ValueError("schema must contain at least one tensor")
    order = _default_storage_order_for_n(n_tensors) if storage_order is None else tuple(int(v) for v in storage_order)
    _validate_storage_order(order, n_tensors=n_tensors)
    if exact_stream_count is not None:
        exact_stream_count = int(exact_stream_count)
        if not 1 <= exact_stream_count <= n_tensors:
            raise ValueError("exact_stream_count must be in [1, n_tensors]")
    if max_streams < 1:
        raise ValueError("max_streams must be >= 1")
    max_streams = min(int(max_streams), n_tensors)

    rows: list[dict[str, Any]] = []
    payloads_by_tensor: dict[int, bytes] = {}
    byte_maps: dict[int, str] = {}
    conv4_perms: dict[int, tuple[int, int, int, int]] = {}
    for idx, (name, expected_shape) in enumerate(schema):
        if name not in state_dict:
            raise ValueError(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=n_quant)
        if tuple(qt.shape) != tuple(expected_shape):
            raise ValueError(
                f"shape mismatch for {name!r}: expected {expected_shape}, got {qt.shape}"
            )
        if selected_transform_mode == "stock_pr101":
            byte_map = DECODER_BYTE_MAPS.get(idx, "zig")
            perm = CONV4_STORAGE_PERMS.get(idx)
            selected = None
        else:
            candidates = measure_tensor_grammar_candidates(
                qt.q_i8,
                tensor_index=idx,
                tensor_name=name,
                scale=qt.scale,
                storage_perm_mode=storage_perm_mode,
                coders=("brotli",),
                brotli_quality=brotli_quality,
                brotli_lgwin_sweep=brotli_lgwin_sweep,
            )
            selected = select_best_tensor_candidate(candidates)
            byte_map = str(selected["byte_map"])
            perm = _parse_perm_label(str(selected["storage_perm"]))
        payload = _build_transformed_tensor_payload(
            qt.q_i8,
            scale=qt.scale,
            byte_map=byte_map,
            perm=perm,
        )
        payloads_by_tensor[idx] = payload
        byte_maps[idx] = byte_map
        if perm is not None:
            if len(perm) != 4:
                raise ValueError(f"conv4 storage perm for tensor {idx} must be 4D")
            conv4_perms[idx] = tuple(int(v) for v in perm)  # type: ignore[assignment]
        rows.append(
            {
                "schema": "pr101_grouped_brotli_tensor_transform_row.v1",
                "tensor_index": idx,
                "tensor_name": name,
                "byte_map": byte_map,
                "storage_perm": _perm_label(perm),
                "payload_bytes": len(payload),
                "selected_isolated_charged_bytes": (
                    None if selected is None else int(selected["charged_bytes"])
                ),
            }
        )

    parts_by_storage = [payloads_by_tensor[idx] for idx in order]
    selected_partition = _solve_brotli_stream_partition(
        parts_by_storage,
        max_streams=max_streams,
        exact_stream_count=exact_stream_count,
        brotli_quality=brotli_quality,
        brotli_lgwin_sweep=brotli_lgwin_sweep,
    )
    selected_blob = _pack_parts_for_stream_ends(
        parts_by_storage,
        stream_ends=tuple(selected_partition["stream_ends"]),
        brotli_quality=brotli_quality,
        brotli_lgwin_sweep=brotli_lgwin_sweep,
    )
    raw_concat = b"".join(parts_by_storage)
    stream_roundtrip_exact = (
        decompress_brotli_streams(selected_blob, len(selected_partition["stream_ends"]))
        == raw_concat
    )

    current_stock_bytes = None
    current_stock_scope = "full_pr101_schema"
    if max_tensors is None and n_tensors == len(FIXED_STATE_SCHEMA):
        current_stock_bytes = len(
            encode_decoder_compact(
                dict(state_dict),
                brotli_quality=brotli_quality,
            )
        )
    else:
        current_stock_scope = "partial_schema_same_order_and_stream_count_not_pr101_authority"
        current_parts = _stock_payloads_in_order(dict(state_dict), order, n_quant=n_quant)
        current_blob = _pack_parts_for_stream_ends(
            current_parts,
            stream_ends=tuple(selected_partition["stream_ends"]),
            brotli_quality=brotli_quality,
            brotli_lgwin_sweep=brotli_lgwin_sweep,
        )
        current_stock_bytes = len(current_blob)

    delta = int(selected_partition["compressed_bytes"]) - int(current_stock_bytes)
    stock_runtime_compatible = (
        max_tensors is None
        and tuple(order) == tuple(DECODER_STORAGE_ORDER)
        and tuple(selected_partition["stream_ends"]) == tuple(DECODER_STREAM_ENDS)
        and _selected_maps_are_pr101_defaults(byte_maps)
        and _selected_conv4_perms_are_pr101_defaults(conv4_perms)
    )
    parser_roundtrip_status = "not_run_partial_schema"
    parser_roundtrip_exact = False
    if max_tensors is None and n_tensors == len(FIXED_STATE_SCHEMA):
        try:
            decoded = decode_decoder_compact(
                selected_blob,
                effective_byte_maps=byte_maps,
                derived_storage_order=tuple(order),
                derived_stream_ends=tuple(selected_partition["stream_ends"]),
                derived_conv4_perms=conv4_perms,
            )
            parser_roundtrip_exact = set(decoded) == {name for name, _ in FIXED_STATE_SCHEMA}
            parser_roundtrip_status = "passed" if parser_roundtrip_exact else "schema_name_mismatch"
        except Exception as exc:  # pragma: no cover - exercised by failure artifacts
            parser_roundtrip_status = f"failed:{type(exc).__name__}"

    blockers = [
        "byte_closed_archive_not_materialized",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not stock_runtime_compatible:
        blockers.extend(
            [
                "receiver_adapter_not_emitted",
                "runtime_consumption_proof_missing",
            ]
        )
    if not stream_roundtrip_exact:
        blockers.append("grouped_brotli_stream_roundtrip_failed")
    if max_tensors is not None:
        blockers.append("partial_schema_sample_not_archive_authority")

    return {
        "schema": PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA,
        "n_tensors": n_tensors,
        "n_quant": int(n_quant),
        "selected_transform_mode": selected_transform_mode,
        "storage_perm_mode": storage_perm_mode,
        "storage_order": list(order),
        "exact_stream_count": exact_stream_count,
        "max_streams": max_streams,
        "brotli_quality": int(brotli_quality),
        "brotli_lgwin_sweep": bool(brotli_lgwin_sweep),
        "byte_accounting": {
            "selected_grouped_brotli_bytes": int(selected_partition["compressed_bytes"]),
            "current_stock_pr101_grouped_bytes": int(current_stock_bytes),
            "current_stock_pr101_grouped_scope": current_stock_scope,
            "grouped_delta_bytes_vs_current_stock": delta,
            "grouped_saved_bytes_vs_current_stock": max(0, -delta),
            "grouped_rate_term_not_archive_authority": contest_rate_term(
                int(selected_partition["compressed_bytes"])
            ),
        },
        "partition": selected_partition,
        "adapter_params": {
            "effective_byte_maps": {str(k): v for k, v in sorted(byte_maps.items())},
            "derived_storage_order": list(order),
            "derived_stream_ends": list(selected_partition["stream_ends"]),
            "derived_conv4_perms": {
                str(k): list(v) for k, v in sorted(conv4_perms.items())
            },
        },
        "parser_roundtrip": {
            "stream_roundtrip_exact": bool(stream_roundtrip_exact),
            "state_dict_parser_roundtrip_status": parser_roundtrip_status,
            "state_dict_parser_roundtrip_exact": bool(parser_roundtrip_exact),
        },
        "runtime_consumption_status": (
            "stock_pr101_runtime" if stock_runtime_compatible else "tac_decode_decoder_compact_with_overrides_required"
        ),
        "stock_runtime_compatible": bool(stock_runtime_compatible),
        "rows": rows,
        "blockers": blockers,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "contest_rate_bytes_authority": False,
            "ready_for_exact_eval_dispatch": False,
            "reason": (
                "grouped Brotli bytes are packet-compiler measurements until "
                "a byte-closed archive and receiver proof consume the adapter params"
            ),
        },
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def build_grouped_optimizer_candidate_queue_from_report(
    report: Mapping[str, Any],
    *,
    campaign_id: str = "pr101_grouped_brotli_packet_grammar",
) -> dict[str, Any]:
    """Convert a grouped packet report into a fail-closed optimizer queue."""

    if report.get("schema") != PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA:
        raise ValueError("expected pr101_grouped_brotli_packet_grammar.v1 report")
    byte_accounting = report.get("byte_accounting")
    if not isinstance(byte_accounting, Mapping):
        raise ValueError("grouped report missing byte_accounting")
    saved = int(byte_accounting.get("grouped_saved_bytes_vs_current_stock") or 0)
    blockers = list(report.get("blockers") or [])
    candidate = {
        "schema": "optimizer_candidate_queue_row_v1",
        "candidate_id": f"{campaign_id}:grouped_brotli_packet",
        "candidate_kind": "planning_only_pr101_grouped_brotli_packet",
        "status": "blocked_planning_signal_only",
        "target_kind": "decoder_weight_grouped_packet_grammar",
        "operation_family": "pr101_grouped_brotli_packet_selection",
        "operation_families": ["pr101_grouped_brotli_packet_selection"],
        "operation_id": "pr101_grouped_brotli_packet_selection",
        "operation_params": {
            "storage_order": report.get("storage_order"),
            "partition": report.get("partition"),
            "adapter_params": report.get("adapter_params"),
            "selected_transform_mode": report.get("selected_transform_mode"),
        },
        "selected_operations": [
            {
                "operation_family": "pr101_grouped_brotli_packet_selection",
                "candidate_saved_bytes": saved,
                "adapter_params": report.get("adapter_params"),
            }
        ],
        "candidate_saved_bytes": saved,
        "saved_bytes_scope": "grouped_decoder_blob_payload_not_archive_authority",
        "predicted_delta_bytes": byte_accounting.get("grouped_delta_bytes_vs_current_stock"),
        "predicted_delta_bytes_scope": "grouped_decoder_blob_payload_only_not_archive_authority",
        "runtime_consumption_status": report.get("runtime_consumption_status"),
        "consumer_payload": {
            "adapter_params": report.get("adapter_params"),
            "byte_accounting_scope": "grouped_decoder_blob_payload_not_archive_authority",
        },
        "blockers": blockers,
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }
    return {
        "schema": PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA,
        "campaign_id": campaign_id,
        "source_schema": report.get("schema"),
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "proof_scope": "planning_only_grouped_packet_grammar_no_dispatch",
        "candidate_count": 1,
        "candidates": [candidate],
        "top_k": [candidate] if saved > 0 else [],
        "blockers": blockers,
        "consumer_surfaces": [
            "tac.optimization.byte_shaving_campaign.build_signal_surface_from_candidate_queue",
        ],
        "axis_tag": "[planning-only byte-profile]",
        **FALSE_AUTHORITY_FIELDS,
    }


def materialize_grouped_decoder_blob_from_report(
    state_dict: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    n_quant: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Materialize and prove the grouped decoder blob described by ``report``.

    This is the first receiver-facing proof surface after grouped planning.  It
    writes no archive and claims no score: the output is a byte-closed decoder
    section plus the adapter parameters a receiver would need to decode it.
    Full submission authority still requires archive splicing, ``inflate.sh``
    receiver consumption, full-frame parity/replay, and exact CPU/CUDA eval.
    """

    if report.get("schema") != PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA:
        raise ValueError("expected pr101_grouped_brotli_packet_grammar.v1 report")
    n_tensors = int(report.get("n_tensors") or 0)
    if n_tensors != len(FIXED_STATE_SCHEMA):
        raise ValueError("decoder blob materialization requires the full PR101 schema")
    quant = int(n_quant if n_quant is not None else report.get("n_quant", N_QUANT))
    adapter = report.get("adapter_params")
    partition = report.get("partition")
    if not isinstance(adapter, Mapping):
        raise ValueError("grouped report missing adapter_params")
    if not isinstance(partition, Mapping):
        raise ValueError("grouped report missing partition")
    byte_maps = _parse_adapter_byte_maps(adapter.get("effective_byte_maps"))
    storage_order = tuple(int(v) for v in _as_sequence(adapter.get("derived_storage_order")))
    stream_ends = tuple(int(v) for v in _as_sequence(adapter.get("derived_stream_ends")))
    conv4_perms = _parse_adapter_conv4_perms(adapter.get("derived_conv4_perms"))
    _validate_storage_order(storage_order, n_tensors=len(FIXED_STATE_SCHEMA))

    payloads_by_tensor: dict[int, bytes] = {}
    for idx, (name, expected_shape) in enumerate(FIXED_STATE_SCHEMA):
        if name not in state_dict:
            raise ValueError(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=quant)
        if tuple(qt.shape) != tuple(expected_shape):
            raise ValueError(
                f"shape mismatch for {name!r}: expected {expected_shape}, got {qt.shape}"
            )
        payloads_by_tensor[idx] = _build_transformed_tensor_payload(
            qt.q_i8,
            scale=qt.scale,
            byte_map=byte_maps.get(idx, DECODER_BYTE_MAPS.get(idx, "zig")),
            perm=conv4_perms.get(idx),
        )
    parts_by_storage = [payloads_by_tensor[idx] for idx in storage_order]
    blob = _pack_parts_for_stream_ends(
        parts_by_storage,
        stream_ends=stream_ends,
        brotli_quality=int(report.get("brotli_quality", 11)),
        brotli_lgwin_sweep=bool(report.get("brotli_lgwin_sweep", False)),
    )
    raw_concat = b"".join(parts_by_storage)
    stream_roundtrip_exact = decompress_brotli_streams(blob, len(stream_ends)) == raw_concat
    decoded = decode_decoder_compact(
        blob,
        effective_byte_maps=byte_maps,
        derived_storage_order=storage_order,
        derived_stream_ends=stream_ends,
        derived_conv4_perms=conv4_perms,
    )
    mismatches = _quantized_decode_mismatches(decoded, state_dict, n_quant=quant)
    quantized_exact = not mismatches
    sha = hashlib.sha256(blob).hexdigest()
    byte_accounting = report.get("byte_accounting")
    reported_bytes = (
        int(byte_accounting.get("selected_grouped_brotli_bytes"))
        if isinstance(byte_accounting, Mapping)
        and byte_accounting.get("selected_grouped_brotli_bytes") is not None
        else None
    )
    stock_runtime_compatible = bool(report.get("stock_runtime_compatible"))
    blockers = [
        "archive_zip_not_materialized",
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not stock_runtime_compatible:
        blockers.extend(["receiver_adapter_not_emitted", "runtime_consumption_proof_missing"])
    if not stream_roundtrip_exact:
        blockers.append("grouped_brotli_stream_roundtrip_failed")
    if not quantized_exact:
        blockers.append("decoded_quantized_state_dict_mismatch")
    if reported_bytes is not None and len(blob) != reported_bytes:
        blockers.append("materialized_decoder_blob_bytes_mismatch_report")
    manifest = {
        "schema": PR101_GROUPED_DECODER_BLOB_MATERIALIZATION_SCHEMA,
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "source_report_schema": report.get("schema"),
        "decoder_blob_bytes": len(blob),
        "decoder_blob_sha256": sha,
        "reported_grouped_brotli_bytes": reported_bytes,
        "byte_closed_decoder_blob_materialized": True,
        "adapter_params": {
            "effective_byte_maps": {str(k): v for k, v in sorted(byte_maps.items())},
            "derived_storage_order": list(storage_order),
            "derived_stream_ends": list(stream_ends),
            "derived_conv4_perms": {str(k): list(v) for k, v in sorted(conv4_perms.items())},
        },
        "proof": {
            "stream_roundtrip_exact": bool(stream_roundtrip_exact),
            "materialized_bytes_match_report": (
                None if reported_bytes is None else len(blob) == reported_bytes
            ),
            "state_dict_parser_roundtrip_exact": set(decoded) == {name for name, _ in FIXED_STATE_SCHEMA},
            "quantized_state_dict_exact": bool(quantized_exact),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:8],
        },
        "stock_runtime_compatible": stock_runtime_compatible,
        "runtime_consumption_status": report.get("runtime_consumption_status"),
        "ready_for_archive_packaging": bool(stream_roundtrip_exact and quantized_exact),
        "blockers": blockers,
        "axis_tag": "[decoder-blob-materialization-only]",
        **FALSE_AUTHORITY_FIELDS,
    }
    return blob, manifest


def materialize_grouped_archive_from_report(
    state_dict: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    source_archive_zip: bytes,
    n_quant: int | None = None,
    member_name: str = PR101_INNER_MEMBER_NAME,
    archive_layout: Literal["fixed_pr101", "u32_decoder_len_adapter"] = "fixed_pr101",
) -> tuple[bytes, dict[str, Any]]:
    """Build a deterministic PR101-shaped archive from a grouped decoder report.

    The function is intentionally stricter than a raw byte splicer.  It always
    preserves the source archive's latent and sidecar sections byte-for-byte and
    emits a single stored ZIP member, but it refuses authority unless the grouped
    decoder blob is compatible with PR101's fixed-offset stock runtime.  Non-stock
    grouped choices remain useful byte-closed artifacts for future receiver
    adapters, not exact-ready submissions.
    """

    decoder_blob, decoder_manifest = materialize_grouped_decoder_blob_from_report(
        state_dict,
        report,
        n_quant=n_quant,
    )
    source_layout = _read_pr101_source_archive_layout(
        source_archive_zip,
        expected_member_name=member_name,
    )
    source_inner = source_layout["inner_member_bytes"]
    if archive_layout not in {"fixed_pr101", "u32_decoder_len_adapter"}:
        raise ValueError(f"unknown PR101 grouped archive layout: {archive_layout!r}")
    source_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(source_inner)
    if archive_layout == "fixed_pr101":
        output_inner = decoder_blob + latent_blob + sidecar_blob
        decoder_blob_offset = 0
        latent_blob_offset = len(decoder_blob)
        sidecar_blob_offset = len(decoder_blob) + len(latent_blob)
    else:
        decoder_section_total = 4 + len(decoder_blob)
        output_inner = (
            struct.pack("<I", decoder_section_total)
            + decoder_blob
            + latent_blob
            + sidecar_blob
        )
        decoder_blob_offset = 4
        latent_blob_offset = decoder_section_total
        sidecar_blob_offset = decoder_section_total + len(latent_blob)
    archive_zip = _write_single_stored_zip_member(member_name, output_inner)
    output_layout = _read_pr101_source_archive_layout(
        archive_zip,
        expected_member_name=member_name,
    )

    fixed_offset_parse_safe = (
        archive_layout == "fixed_pr101"
        and len(decoder_blob) == DECODER_BLOB_LEN
        and bool(decoder_manifest.get("stock_runtime_compatible"))
    )
    u32_adapter_parse_safe = archive_layout == "u32_decoder_len_adapter"
    latent_preserved = hashlib.sha256(latent_blob).hexdigest() == hashlib.sha256(
        source_inner[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    ).hexdigest()
    sidecar_preserved = hashlib.sha256(sidecar_blob).hexdigest() == hashlib.sha256(
        source_inner[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]
    ).hexdigest()
    zip_roundtrip_exact = output_layout["inner_member_bytes"] == output_inner

    blockers = [
        "full_frame_inflate_parity_missing",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if archive_layout == "fixed_pr101" and not fixed_offset_parse_safe:
        blockers.append("stock_runtime_fixed_offset_decoder_blob_length_mismatch")
    if u32_adapter_parse_safe:
        blockers.append("receiver_runtime_source_not_emitted")
    if not bool(decoder_manifest.get("stock_runtime_compatible")):
        if archive_layout == "fixed_pr101":
            blockers.extend(["receiver_adapter_not_emitted", "runtime_consumption_proof_missing"])
        else:
            blockers.append("receiver_codec_constants_override_source_not_emitted")
    if not latent_preserved:
        blockers.append("latent_blob_not_preserved")
    if not sidecar_preserved:
        blockers.append("sidecar_blob_not_preserved")
    if not zip_roundtrip_exact:
        blockers.append("single_member_zip_roundtrip_failed")

    source_zip_sha = hashlib.sha256(source_archive_zip).hexdigest()
    output_zip_sha = hashlib.sha256(archive_zip).hexdigest()
    source_inner_sha = hashlib.sha256(source_inner).hexdigest()
    output_inner_sha = hashlib.sha256(output_inner).hexdigest()
    source_decoder_sha = hashlib.sha256(source_decoder).hexdigest()
    decoder_sha = hashlib.sha256(decoder_blob).hexdigest()
    latent_sha = hashlib.sha256(latent_blob).hexdigest()
    sidecar_sha = hashlib.sha256(sidecar_blob).hexdigest()
    manifest = {
        "schema": PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA,
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "source_report_schema": report.get("schema"),
        "archive_layout": archive_layout,
        "source_archive_zip_bytes": len(source_archive_zip),
        "source_archive_zip_sha256": source_zip_sha,
        "archive_zip_bytes": len(archive_zip),
        "archive_zip_sha256": output_zip_sha,
        "archive_zip_delta_bytes": len(archive_zip) - len(source_archive_zip),
        "inner_member_name": member_name,
        "inner_member_bytes": len(output_inner),
        "inner_member_sha256": output_inner_sha,
        "source_inner_member_bytes": len(source_inner),
        "source_inner_member_sha256": source_inner_sha,
        "decoder_blob_offset": decoder_blob_offset,
        "decoder_blob_bytes": len(decoder_blob),
        "decoder_blob_sha256": decoder_sha,
        "source_decoder_blob_bytes": len(source_decoder),
        "source_decoder_blob_sha256": source_decoder_sha,
        "latent_blob_offset": latent_blob_offset,
        "latent_blob_bytes": len(latent_blob),
        "latent_blob_sha256": latent_sha,
        "sidecar_blob_offset": sidecar_blob_offset,
        "sidecar_blob_bytes": len(sidecar_blob),
        "sidecar_blob_sha256": sidecar_sha,
        "fixed_offset_stock_runtime_parse_safe": bool(fixed_offset_parse_safe),
        "u32_decoder_len_adapter_parse_safe": bool(u32_adapter_parse_safe),
        "byte_closed_archive_zip_materialized": True,
        "decoder_materialization": decoder_manifest,
        "zip_proof": {
            "single_member": output_layout["zip_member_count"] == 1,
            "member_name_matches": output_layout["inner_member_name"] == member_name,
            "zip_stored": output_layout["zip_compress_type"] == zipfile.ZIP_STORED,
            "member_file_size_matches_inner": output_layout["zip_file_size"] == len(output_inner),
            "member_compress_size_matches_inner": output_layout["zip_compress_size"] == len(output_inner),
            "empty_extra": output_layout["zip_extra_len"] == 0,
            "empty_comment": output_layout["zip_comment_len"] == 0,
            "deterministic_timestamp": output_layout["zip_date_time"] == [1980, 1, 1, 0, 0, 0],
            "roundtrip_exact": bool(zip_roundtrip_exact),
        },
        "proof": {
            "decoder_blob_materialized": bool(
                decoder_manifest.get("byte_closed_decoder_blob_materialized")
            ),
            "decoder_stream_roundtrip_exact": bool(
                decoder_manifest.get("proof", {}).get("stream_roundtrip_exact")
                if isinstance(decoder_manifest.get("proof"), Mapping)
                else False
            ),
            "decoder_quantized_state_dict_exact": bool(
                decoder_manifest.get("proof", {}).get("quantized_state_dict_exact")
                if isinstance(decoder_manifest.get("proof"), Mapping)
                else False
            ),
            "latent_blob_preserved": bool(latent_preserved),
            "sidecar_blob_preserved": bool(sidecar_preserved),
            "fixed_offset_stock_runtime_parse_safe": bool(fixed_offset_parse_safe),
            "u32_decoder_len_adapter_parse_safe": bool(u32_adapter_parse_safe),
            "byte_closed_archive_zip_materialized": True,
        },
        "blockers": blockers,
        "axis_tag": "[archive-zip-materialization-only]",
        **FALSE_AUTHORITY_FIELDS,
    }
    return archive_zip, manifest


def build_u32_receiver_adapter_source_from_report(
    report: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Emit the parser adapter source for ``u32_decoder_len_adapter`` archives."""

    if report.get("schema") != PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA:
        raise ValueError("expected pr101_grouped_brotli_packet_grammar.v1 report")
    adapter = report.get("adapter_params")
    if not isinstance(adapter, Mapping):
        raise ValueError("grouped report missing adapter_params")
    byte_maps = _parse_adapter_byte_maps(adapter.get("effective_byte_maps"))
    storage_order = tuple(int(v) for v in _as_sequence(adapter.get("derived_storage_order")))
    stream_ends = tuple(int(v) for v in _as_sequence(adapter.get("derived_stream_ends")))
    conv4_perms = _parse_adapter_conv4_perms(adapter.get("derived_conv4_perms"))
    _validate_storage_order(storage_order, n_tensors=len(FIXED_STATE_SCHEMA))
    adapter_params = {
        "effective_byte_maps": {str(k): v for k, v in sorted(byte_maps.items())},
        "derived_storage_order": list(storage_order),
        "derived_stream_ends": list(stream_ends),
        "derived_conv4_perms": {str(k): list(v) for k, v in sorted(conv4_perms.items())},
    }
    adapter_json = json.dumps(adapter_params, sort_keys=True, separators=(",", ":"))
    source = f'''# SPDX-License-Identifier: MIT
"""Generated PR101 u32 decoder-length receiver adapter.

This adapter parses a PR101-family inner archive payload with layout:

    uint32_le decoder_section_total_bytes
    decoder_blob[4:decoder_section_total_bytes]
    latent_blob[decoder_section_total_bytes:decoder_section_total_bytes + {LATENT_BLOB_LEN}]
    sidecar_blob[decoder_section_total_bytes + {LATENT_BLOB_LEN}:]

It intentionally expects the caller to supply the codec functions from the
submission runtime so this generated file does not duplicate decoder code.
"""

LATENT_BLOB_LEN = {LATENT_BLOB_LEN}
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
ADAPTER_PARAMS_JSON = {adapter_json!r}


def _adapter_params():
    import json

    params = json.loads(ADAPTER_PARAMS_JSON)
    return {{
        "effective_byte_maps": {{int(k): v for k, v in params["effective_byte_maps"].items()}},
        "derived_storage_order": tuple(params["derived_storage_order"]),
        "derived_stream_ends": tuple(params["derived_stream_ends"]),
        "derived_conv4_perms": {{
            int(k): tuple(v) for k, v in params["derived_conv4_perms"].items()
        }},
    }}


def parse_archive_u32_decoder_len(
    archive_bytes,
    *,
    decode_decoder_compact,
    decode_latents_compact,
    apply_latent_sidecar,
):
    if len(archive_bytes) < 4:
        raise ValueError("archive too short for u32 decoder section header")
    section_total = int.from_bytes(archive_bytes[:4], "little")
    if section_total < 4 or section_total > len(archive_bytes):
        raise ValueError(f"bad decoder_section_total {{section_total}}")
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + LATENT_BLOB_LEN:]
    if not decoder_blob or len(latent_blob) != LATENT_BLOB_LEN:
        raise ValueError("bad PR101 u32 decoder-length archive layout")
    decoder_sd = decode_decoder_compact(decoder_blob, **_adapter_params())
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    meta = {{
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }}
    return decoder_sd, latents, meta
'''
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    manifest = {
        "schema": PR101_U32_RECEIVER_ADAPTER_SOURCE_SCHEMA,
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "source_report_schema": report.get("schema"),
        "receiver_adapter_source_sha256": sha,
        "receiver_adapter_source_bytes": len(source.encode("utf-8")),
        "archive_layout": "u32_decoder_len_adapter",
        "adapter_params": adapter_params,
        "runtime_consumption_status": "u32_decoder_len_adapter_source_emitted",
        "blockers": [
            "inflate_sh_integration_missing",
            "full_frame_inflate_parity_missing",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "axis_tag": "[receiver-adapter-source-only]",
        **FALSE_AUTHORITY_FIELDS,
    }
    return source, manifest


def build_u32_receiver_runtime_tree_from_report(
    report: Mapping[str, Any],
    *,
    codec_py_source: bytes | str,
    model_py_source: bytes | str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build a self-contained PR101 u32-adapter submission runtime tree.

    The returned mapping is ``rel_path -> bytes``.  It deliberately does not
    write to disk so callers can enforce their own storage/preflight policy.
    """

    adapter_source, adapter_manifest = build_u32_receiver_adapter_source_from_report(report)
    files = {
        "inflate.sh": _generated_u32_inflate_sh().encode("utf-8"),
        "inflate.py": _generated_u32_inflate_py().encode("utf-8"),
        "pr101_u32_adapter.py": adapter_source.encode("utf-8"),
        "src/codec.py": _source_bytes(codec_py_source),
        "src/model.py": _source_bytes(model_py_source),
    }
    file_rows = [
        {
            "rel_path": rel_path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executable": rel_path == "inflate.sh",
        }
        for rel_path, data in sorted(files.items())
    ]
    manifest = {
        "schema": PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA,
        "producer": "tac.packet_compiler.pr101_per_tensor_grammar_solver",
        "source_report_schema": report.get("schema"),
        "archive_layout": "u32_decoder_len_adapter",
        "file_count": len(file_rows),
        "files": file_rows,
        "receiver_adapter": adapter_manifest,
        "runtime_consumption_status": "u32_receiver_runtime_tree_materialized",
        "blockers": [
            "full_frame_inflate_parity_missing",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "axis_tag": "[receiver-runtime-tree-materialization-only]",
        **FALSE_AUTHORITY_FIELDS,
    }
    return files, manifest


def empirical_shannon_floor_bytes(payload: bytes | bytearray | memoryview) -> float:
    """Return the order-0 Shannon lower bound for a byte payload."""

    data = bytes(payload)
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    h = 0.0
    for count in counts.values():
        p = count / n
        h -= p * math.log2(p)
    return (h * n) / 8.0


def _measure_coder(
    payload: bytes,
    *,
    coder: CoderName,
    brotli_quality: int,
    brotli_lgwin_sweep: bool,
) -> dict[str, Any]:
    if coder == "brotli":
        return _measure_brotli(
            payload,
            quality=brotli_quality,
            lgwin_sweep=brotli_lgwin_sweep,
        )
    if coder == "lzma_raw":
        return _measure_lzma_raw(payload)
    if coder == "canonical_huffman":
        return _measure_canonical_huffman(payload)
    if coder == "range_ac_empirical_hist_u16":
        return _measure_range_ac(payload)
    raise ValueError(f"unknown coder: {coder!r}")


def _brotli_quality_values(
    *,
    brotli_quality: int,
    brotli_quality_values: Sequence[int] | None,
) -> tuple[int, ...]:
    values = (brotli_quality,) if brotli_quality_values is None else tuple(brotli_quality_values)
    if not values:
        raise ValueError("brotli_quality_values must not be empty")
    out: list[int] = []
    for value in values:
        q = int(value)
        if not 0 <= q <= 11:
            raise ValueError("brotli quality values must be in [0, 11]")
        out.append(q)
    return tuple(dict.fromkeys(out))


def _measure_brotli(
    payload: bytes,
    *,
    quality: int,
    lgwin_sweep: bool,
) -> dict[str, Any]:
    best_lgwin = None
    if lgwin_sweep:
        best = None
        for lgwin in range(10, 25):
            try:
                comp = brotli.compress(payload, quality=quality, lgwin=lgwin)
            except brotli.error:
                continue
            if best is None or len(comp) < len(best):
                best = comp
                best_lgwin = lgwin
        if best is None:
            best = pack_brotli_stream(payload, quality=quality)
    else:
        best = pack_brotli_stream(payload, quality=quality)
    try:
        decoded = brotli.decompress(best)
        ok = decoded == payload
        status = "ok" if ok else "codec_roundtrip_failed"
    except brotli.error:
        ok = False
        status = "codec_roundtrip_failed"
    return {
        "charged_bytes": len(best),
        "codec_payload_bytes": len(best),
        "side_info_bytes": 0,
        "coder_params": {
            "quality": int(quality),
            "lgwin": best_lgwin,
            "lgwin_sweep": bool(lgwin_sweep),
        },
        "roundtrip_exact": ok,
        "status": status,
    }


def _build_transformed_tensor_payload(
    q_i8: np.ndarray,
    *,
    scale: float,
    byte_map: str,
    perm: tuple[int, ...] | None,
) -> bytes:
    flat = _apply_storage_perm(q_i8, perm).reshape(-1)
    return encode_byte_map(flat.astype(np.int8, copy=False), byte_map) + (
        np.array([float(scale)], dtype=np.float16).tobytes()
    )


def _solve_brotli_stream_partition(
    parts_by_storage: Sequence[bytes],
    *,
    max_streams: int,
    exact_stream_count: int | None,
    brotli_quality: int,
    brotli_lgwin_sweep: bool,
) -> dict[str, Any]:
    n = len(parts_by_storage)
    if n == 0:
        raise ValueError("parts_by_storage must not be empty")
    if exact_stream_count is not None and not 1 <= exact_stream_count <= n:
        raise ValueError("exact_stream_count must be in [1, len(parts_by_storage)]")
    s_max = min(int(max_streams), n)
    if s_max <= 0:
        raise ValueError("max_streams must be >= 1")

    inf = float("inf")
    cost = np.full((n + 1, n + 1), inf, dtype=np.float64)
    raw_bytes = np.zeros((n + 1, n + 1), dtype=np.int64)
    for i in range(n):
        running = bytearray()
        for j in range(i + 1, n + 1):
            running.extend(parts_by_storage[j - 1])
            measured = _measure_brotli(
                bytes(running),
                quality=brotli_quality,
                lgwin_sweep=brotli_lgwin_sweep,
            )
            if measured["status"] != "ok":
                raise ValueError(f"brotli interval roundtrip failed for [{i},{j})")
            cost[i, j] = float(measured["charged_bytes"])
            raw_bytes[i, j] = len(running)

    dp = np.full((n + 1, s_max + 1), inf, dtype=np.float64)
    parent = np.full((n + 1, s_max + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for s in range(1, s_max + 1):
        for i in range(1, n + 1):
            for j in range(0, i):
                if not np.isfinite(dp[j, s - 1]):
                    continue
                candidate = dp[j, s - 1] + cost[j, i]
                if candidate < dp[i, s] or (
                    candidate == dp[i, s] and (parent[i, s] < 0 or j < parent[i, s])
                ):
                    dp[i, s] = candidate
                    parent[i, s] = j

    if exact_stream_count is None:
        candidates = [
            (float(dp[n, s]), s) for s in range(1, s_max + 1) if np.isfinite(dp[n, s])
        ]
        if not candidates:
            raise ValueError("no feasible brotli stream partition")
        _best_cost, best_s = min(candidates, key=lambda item: (item[0], item[1]))
    else:
        best_s = exact_stream_count
        if not np.isfinite(dp[n, best_s]):
            raise ValueError("no feasible exact brotli stream partition")

    starts: list[int] = []
    ends: list[int] = []
    i = n
    s = best_s
    while s > 0:
        j = int(parent[i, s])
        if j < 0:
            raise ValueError("failed to backtrack brotli stream partition")
        starts.append(j)
        ends.append(i)
        i = j
        s -= 1
    starts.reverse()
    ends.reverse()
    stream_rows = [
        {
            "stream_index": idx,
            "start": int(start),
            "end": int(end),
            "raw_bytes": int(raw_bytes[start, end]),
            "compressed_bytes": int(cost[start, end]),
        }
        for idx, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]
    return {
        "schema": "pr101_grouped_brotli_stream_partition.v1",
        "stream_count": int(best_s),
        "stream_ends": [int(v) for v in ends],
        "compressed_bytes": int(sum(row["compressed_bytes"] for row in stream_rows)),
        "raw_bytes": int(sum(row["raw_bytes"] for row in stream_rows)),
        "streams": stream_rows,
    }


def _pack_parts_for_stream_ends(
    parts_by_storage: Sequence[bytes],
    *,
    stream_ends: Sequence[int],
    brotli_quality: int,
    brotli_lgwin_sweep: bool,
) -> bytes:
    streams: list[bytes] = []
    start = 0
    for raw_end in stream_ends:
        end = int(raw_end)
        if not start < end <= len(parts_by_storage):
            raise ValueError("stream_ends must be strictly increasing and in range")
        window = b"".join(parts_by_storage[start:end])
        if brotli_lgwin_sweep:
            best: bytes | None = None
            for lgwin in range(10, 25):
                try:
                    comp = brotli.compress(window, quality=brotli_quality, lgwin=lgwin)
                except brotli.error:
                    continue
                if best is None or len(comp) < len(best):
                    best = comp
            if best is None:
                best = pack_brotli_stream(window, quality=brotli_quality)
            streams.append(best)
        else:
            streams.append(pack_brotli_stream(window, quality=brotli_quality))
        start = end
    if start != len(parts_by_storage):
        raise ValueError("stream_ends must end at len(parts_by_storage)")
    return b"".join(streams)


def _stock_payloads_in_order(
    state_dict: dict[str, Any],
    storage_order: Sequence[int],
    *,
    n_quant: int,
) -> list[bytes]:
    quantized = [
        _quantize_tensor(name, state_dict[name], n_quant=n_quant)
        for name, _shape in FIXED_STATE_SCHEMA[: max(storage_order) + 1]
    ]
    out: list[bytes] = []
    for idx in storage_order:
        qt = quantized[int(idx)]
        out.append(
            _build_transformed_tensor_payload(
                qt.q_i8,
                scale=qt.scale,
                byte_map=DECODER_BYTE_MAPS.get(int(idx), "zig"),
                perm=CONV4_STORAGE_PERMS.get(int(idx)),
            )
        )
    return out


def _measure_lzma_raw(payload: bytes) -> dict[str, Any]:
    try:
        comp = lzma.compress(
            payload,
            format=lzma.FORMAT_RAW,
            filters=LATENT_LZMA_FILTERS,
        )
        decoded = lzma.decompress(
            comp,
            format=lzma.FORMAT_RAW,
            filters=LATENT_LZMA_FILTERS,
        )
        ok = decoded == payload
        status = "ok" if ok else "codec_roundtrip_failed"
    except lzma.LZMAError:
        comp = b""
        ok = False
        status = "codec_unavailable_or_failed"
    return {
        "charged_bytes": len(comp),
        "codec_payload_bytes": len(comp),
        "side_info_bytes": 0,
        "coder_params": {"format": "lzma.FORMAT_RAW", "filters": "LATENT_LZMA_FILTERS"},
        "roundtrip_exact": ok,
        "status": status,
    }


def _measure_range_ac(payload: bytes) -> dict[str, Any]:
    try:
        from tac.pr103_arithmetic_codec import pack_ac_stream, unpack_ac_stream
    except Exception:
        return {
            "charged_bytes": 0,
            "codec_payload_bytes": 0,
            "side_info_bytes": 0,
            "coder_params": {"histogram": "empirical_u16_256_symbols"},
            "roundtrip_exact": False,
            "status": "codec_dependency_unavailable",
        }
    symbols = np.frombuffer(payload, dtype=np.uint8)
    hist = np.bincount(symbols, minlength=256).astype(np.int32)
    blob = pack_ac_stream(symbols, hist)
    decoded = unpack_ac_stream(blob, hist, int(symbols.size)).astype(np.uint8)
    ok = decoded.tobytes() == payload
    # Conservative standalone side info: raw uint16 histogram.  PR103's merged
    # path amortizes this better, so this candidate is a fail-closed upper bound.
    side_info = 256 * 2
    return {
        "charged_bytes": len(blob) + side_info,
        "codec_payload_bytes": len(blob),
        "side_info_bytes": side_info,
        "coder_params": {"histogram": "empirical_u16_256_symbols"},
        "roundtrip_exact": ok,
        "status": "ok" if ok else "codec_roundtrip_failed",
    }


def _measure_canonical_huffman(payload: bytes) -> dict[str, Any]:
    lengths = _huffman_code_lengths(payload)
    encoded = _canonical_huffman_encode(payload, lengths)
    decoded = _canonical_huffman_decode(encoded, lengths, len(payload))
    ok = decoded == payload
    side_info = 256
    return {
        "charged_bytes": len(encoded) + side_info,
        "codec_payload_bytes": len(encoded),
        "side_info_bytes": side_info,
        "coder_params": {"code_length_header_bytes": side_info},
        "roundtrip_exact": ok,
        "status": "ok" if ok else "codec_roundtrip_failed",
    }


def _huffman_code_lengths(payload: bytes) -> tuple[int, ...]:
    if not payload:
        return (0,) * 256
    counts = Counter(payload)
    counter = itertools.count()
    heap: list[tuple[int, int, dict[int, int]]] = [
        (count, next(counter), {symbol: 0}) for symbol, count in sorted(counts.items())
    ]
    if len(heap) == 1:
        lengths = [0] * 256
        lengths[next(iter(heap[0][2]))] = 1
        return tuple(lengths)
    heapq.heapify(heap)
    while len(heap) > 1:
        freq_a, _serial_a, tree_a = heapq.heappop(heap)
        freq_b, _serial_b, tree_b = heapq.heappop(heap)
        merged: dict[int, int] = {}
        for symbol, depth in tree_a.items():
            merged[symbol] = depth + 1
        for symbol, depth in tree_b.items():
            merged[symbol] = depth + 1
        heapq.heappush(heap, (freq_a + freq_b, next(counter), merged))
    lengths = [0] * 256
    for symbol, depth in heap[0][2].items():
        lengths[symbol] = int(depth)
    return tuple(lengths)


def _canonical_codes(lengths: Sequence[int]) -> dict[int, tuple[int, int]]:
    code = 0
    prev_len = 0
    out: dict[int, tuple[int, int]] = {}
    for symbol, length in sorted(
        ((idx, int(length)) for idx, length in enumerate(lengths) if int(length) > 0),
        key=lambda item: (item[1], item[0]),
    ):
        code <<= length - prev_len
        out[symbol] = (code, length)
        code += 1
        prev_len = length
    return out


def _canonical_huffman_encode(payload: bytes, lengths: Sequence[int]) -> bytes:
    codes = _canonical_codes(lengths)
    out = bytearray()
    acc = 0
    n_bits = 0
    for symbol in payload:
        code, length = codes[symbol]
        acc = (acc << length) | code
        n_bits += length
        while n_bits >= 8:
            n_bits -= 8
            out.append((acc >> n_bits) & 0xFF)
            acc &= (1 << n_bits) - 1 if n_bits else 0
    if n_bits:
        out.append((acc << (8 - n_bits)) & 0xFF)
    return bytes(out)


def _canonical_huffman_decode(
    encoded: bytes,
    lengths: Sequence[int],
    n_symbols: int,
) -> bytes:
    decode = {(length, code): symbol for symbol, (code, length) in _canonical_codes(lengths).items()}
    out = bytearray()
    code = 0
    length = 0
    for byte in encoded:
        for bit_idx in range(7, -1, -1):
            code = (code << 1) | ((byte >> bit_idx) & 1)
            length += 1
            symbol = decode.get((length, code))
            if symbol is not None:
                out.append(symbol)
                if len(out) == n_symbols:
                    return bytes(out)
                code = 0
                length = 0
    if len(out) != n_symbols:
        raise ValueError("truncated canonical Huffman payload")
    return bytes(out)


def _candidate_storage_perms(
    shape: tuple[int, ...],
    tensor_index: int,
    mode: StoragePermMode,
) -> tuple[tuple[int, ...] | None, ...]:
    if len(shape) != 4:
        return (None,)
    if mode == "identity":
        return (None,)
    if mode == "pr101-plus-identity":
        values: list[tuple[int, ...] | None] = [None]
        if tensor_index in CONV4_STORAGE_PERMS:
            values.append(CONV4_STORAGE_PERMS[tensor_index])
        return tuple(dict.fromkeys(values))
    if mode == "exhaustive-conv4":
        identity = tuple(range(4))
        return (
            None,
            *tuple(
                perm for perm in itertools.permutations(range(4)) if perm != identity
            ),
        )
    raise ValueError(f"unknown storage_perm_mode: {mode!r}")


def _default_storage_order_for_n(n_tensors: int) -> tuple[int, ...]:
    if n_tensors == len(FIXED_STATE_SCHEMA):
        return tuple(DECODER_STORAGE_ORDER)
    return tuple(idx for idx in DECODER_STORAGE_ORDER if idx < n_tensors)


def _validate_storage_order(order: Sequence[int], *, n_tensors: int) -> None:
    values = [int(v) for v in order]
    if sorted(values) != list(range(n_tensors)):
        raise ValueError(
            f"storage_order must be a permutation of range({n_tensors}); got {tuple(values)!r}"
        )


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


def _runtime_consumption_status(
    *,
    tensor_index: int,
    byte_map: str,
    perm: tuple[int, ...] | None,
    coder: str,
) -> str:
    default_map = DECODER_BYTE_MAPS.get(tensor_index, "zig")
    default_perm = CONV4_STORAGE_PERMS.get(tensor_index)
    if coder == "brotli" and byte_map == default_map and perm == default_perm:
        return "stock_pr101_runtime"
    if coder == "brotli":
        return "tac_decode_decoder_compact_with_overrides_required"
    return "new_receiver_adapter_required"


def _candidate_blockers(runtime_status: str) -> list[str]:
    blockers = [
        "isolated_tensor_measurement_not_grouped_archive_authority",
        "byte_closed_archive_not_materialized",
        "full_frame_inflate_parity_missing",
    ]
    if runtime_status != "stock_pr101_runtime":
        blockers.append(runtime_status)
        blockers.append("runtime_consumption_proof_missing")
    return blockers


def _find_current_pr101_isolated_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    tensor_index: int,
) -> dict[str, Any] | None:
    default_map = DECODER_BYTE_MAPS.get(tensor_index, "zig")
    default_perm = _perm_label(CONV4_STORAGE_PERMS.get(tensor_index))
    matches = [
        dict(row)
        for row in candidates
        if row.get("coder") == "brotli"
        and row.get("byte_map") == default_map
        and row.get("storage_perm") == default_perm
        and row.get("status") == "ok"
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: int(row["charged_bytes"]))
    return matches[0]


def _saturation_diagnostic(ratio: float | None) -> dict[str, Any]:
    if ratio is None:
        status = "floor_unavailable"
        action = "compute_empirical_floor_before_coder_work"
    elif ratio <= 1.02:
        status = "entropy_saturated"
        action = "stop_format_churn_on_this_tensor_family_without_new_substrate_signal"
    elif ratio <= 1.10:
        status = "weak_entropy_gap"
        action = "low_priority_runtime_adapter_only_if_already_needed_elsewhere"
    else:
        status = "unsaturated_entropy_gap"
        action = "build_receiver_adapter_for_selected_coder_and replay_byte_closed"
    return {
        "schema": "pr101_per_tensor_grammar_saturation_diagnostic.v1",
        "selected_over_floor_ratio": ratio,
        "status": status,
        "next_action": action,
    }


def _planner_feedback(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    operation_rows: list[dict[str, Any]] = []
    for row in rows:
        selected = row.get("selected")
        current = row.get("current_pr101_isolated")
        if not isinstance(selected, Mapping):
            continue
        current_bytes = int(current["charged_bytes"]) if isinstance(current, Mapping) else None
        selected_bytes = int(selected["charged_bytes"])
        operation_rows.append(
            {
                "schema": "pr101_per_tensor_grammar_packetir_operation_hint.v1",
                "operation_id": (
                    f"pr101_tensor_{int(selected['tensor_index']):02d}_"
                    f"{selected['coder']}_{selected['byte_map']}_{selected['storage_perm']}"
                ).replace(",", "_"),
                "tensor_index": int(selected["tensor_index"]),
                "tensor_name": str(selected["tensor_name"]),
                "byte_map": str(selected["byte_map"]),
                "storage_perm": str(selected["storage_perm"]),
                "coder": str(selected["coder"]),
                "coder_params": dict(selected.get("coder_params") or {}),
                "selected_charged_bytes": selected_bytes,
                "current_pr101_isolated_bytes": current_bytes,
                "isolated_byte_delta_vs_current_pr101": (
                    None if current_bytes is None else selected_bytes - current_bytes
                ),
                "runtime_consumption_status": str(selected["runtime_consumption_status"]),
                "queue_consumable": False,
                "queue_consumable_blockers": [
                    "receiver_adapter_not_emitted",
                    "byte_closed_archive_replay_not_run",
                ],
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    receiver_needed = [
        row
        for row in operation_rows
        if row["runtime_consumption_status"] != "stock_pr101_runtime"
    ]
    rate_positive = [
        row
        for row in operation_rows
        if row["isolated_byte_delta_vs_current_pr101"] is not None
        and int(row["isolated_byte_delta_vs_current_pr101"]) < 0
    ]
    return {
        "schema": "pr101_per_tensor_grammar_planner_feedback.v1",
        "operation_hint_count": len(operation_rows),
        "receiver_adapter_needed_count": len(receiver_needed),
        "isolated_rate_positive_hint_count": len(rate_positive),
        "top_receiver_adapter_targets": sorted(
            receiver_needed,
            key=lambda item: (
                item["isolated_byte_delta_vs_current_pr101"]
                if item["isolated_byte_delta_vs_current_pr101"] is not None
                else 0,
                item["tensor_index"],
            ),
        )[:8],
        "posterior_update_hooks": [
            {
                "schema": "pr101_tensor_grammar_entropy_gap_posterior_hook.v1",
                "consumer": "packet_compiler_or_autopilot_planner",
                "signal": "per_tensor_entropy_gap_and_receiver_adapter_value",
                "authority": "planning_only_until_byte_closed_receiver_replay",
            }
        ],
        "operation_hints": operation_rows,
    }


def _validate_q_i8(q_i8: np.ndarray) -> np.ndarray:
    q = np.asarray(q_i8)
    if q.dtype != np.int8:
        raise ValueError(f"q_i8 dtype must be int8; got {q.dtype}")
    if q.size == 0:
        raise ValueError("q_i8 must be non-empty")
    return q


def _perm_label(perm: tuple[int, ...] | None) -> str:
    if perm is None:
        return "identity"
    return ",".join(str(int(v)) for v in perm)


def _parse_perm_label(label: str) -> tuple[int, ...] | None:
    if label == "identity":
        return None
    try:
        values = tuple(int(item) for item in label.split(","))
    except ValueError as exc:
        raise ValueError(f"bad storage_perm label: {label!r}") from exc
    if sorted(values) != list(range(len(values))):
        raise ValueError(f"bad storage_perm label: {label!r}")
    return values


def _selected_maps_are_pr101_defaults(byte_maps: Mapping[int, str]) -> bool:
    for idx in range(len(FIXED_STATE_SCHEMA)):
        if byte_maps.get(idx, DECODER_BYTE_MAPS.get(idx, "zig")) != DECODER_BYTE_MAPS.get(idx, "zig"):
            return False
    return True


def _selected_conv4_perms_are_pr101_defaults(
    conv4_perms: Mapping[int, tuple[int, int, int, int]],
) -> bool:
    for idx, perm in CONV4_STORAGE_PERMS.items():
        if tuple(conv4_perms.get(idx, tuple(range(4)))) != tuple(perm):
            return False
    extra = set(conv4_perms) - set(CONV4_STORAGE_PERMS)
    return not extra


def _as_sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected sequence adapter parameter")
    return value


def _parse_adapter_byte_maps(value: Any) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise ValueError("adapter_params.effective_byte_maps must be a mapping")
    out: dict[int, str] = {}
    for raw_idx, raw_map in value.items():
        idx = int(raw_idx)
        byte_map = str(raw_map)
        if byte_map not in VALID_BYTE_MAP_STRATEGIES:
            raise ValueError(f"unknown adapter byte_map: {byte_map!r}")
        out[idx] = byte_map
    return out


def _parse_adapter_conv4_perms(value: Any) -> dict[int, tuple[int, int, int, int]]:
    if not isinstance(value, Mapping):
        raise ValueError("adapter_params.derived_conv4_perms must be a mapping")
    out: dict[int, tuple[int, int, int, int]] = {}
    for raw_idx, raw_perm in value.items():
        values = tuple(int(v) for v in _as_sequence(raw_perm))
        if sorted(values) != [0, 1, 2, 3]:
            raise ValueError(f"bad adapter conv4 perm for tensor {raw_idx!r}: {values!r}")
        out[int(raw_idx)] = values  # type: ignore[assignment]
    return out


def _quantized_decode_mismatches(
    decoded: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    *,
    n_quant: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, state_dict[name], n_quant=n_quant)
        scale = float(np.array([qt.scale], dtype=np.float16)[0])
        expected = qt.q_i8.astype(np.float32) * scale
        observed_raw = decoded.get(name)
        if observed_raw is None:
            mismatches.append({"tensor_name": name, "reason": "missing_decoded_tensor"})
            continue
        observed = observed_raw.detach().cpu().numpy() if hasattr(observed_raw, "detach") else np.asarray(observed_raw)
        if observed.shape != expected.shape:
            mismatches.append(
                {
                    "tensor_name": name,
                    "reason": "shape_mismatch",
                    "expected_shape": list(expected.shape),
                    "observed_shape": list(observed.shape),
                }
            )
            continue
        if not np.array_equal(observed.astype(np.float32, copy=False), expected):
            max_abs = float(np.max(np.abs(observed.astype(np.float32) - expected)))
            mismatches.append(
                {
                    "tensor_name": name,
                    "reason": "value_mismatch",
                    "max_abs_diff": max_abs,
                }
            )
    return mismatches


def _read_pr101_source_archive_layout(
    archive_zip: bytes,
    *,
    expected_member_name: str,
) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(
                f"PR101 archive must contain exactly one member; got {len(infos)}"
            )
        info = infos[0]
        if info.filename != expected_member_name:
            raise ValueError(
                f"PR101 archive member {info.filename!r} != expected {expected_member_name!r}"
            )
        member = zf.read(info.filename)
        return {
            "zip_member_count": len(infos),
            "inner_member_name": info.filename,
            "inner_member_bytes": member,
            "zip_compress_type": int(info.compress_type),
            "zip_file_size": int(info.file_size),
            "zip_compress_size": int(info.compress_size),
            "zip_date_time": [int(v) for v in info.date_time],
            "zip_extra_len": len(info.extra),
            "zip_comment_len": len(info.comment),
        }


def _split_pr101_inner_blob(inner: bytes) -> tuple[bytes, bytes, bytes]:
    minimum = DECODER_BLOB_LEN + LATENT_BLOB_LEN
    if len(inner) < minimum:
        raise ValueError(
            f"PR101 inner member length {len(inner)} < required minimum {minimum}"
        )
    decoder_blob = inner[:DECODER_BLOB_LEN]
    latent_blob = inner[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = inner[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]
    return decoder_blob, latent_blob, sidecar_blob


def _write_single_stored_zip_member(member_name: str, payload: bytes) -> bytes:
    if not member_name or member_name.endswith("/"):
        raise ValueError(f"invalid PR101 archive member name: {member_name!r}")
    buf = io.BytesIO()
    info = zipfile.ZipInfo(filename=member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(buf, "w") as zf:
        zf.comment = b""
        zf.writestr(info, payload)
    return buf.getvalue()


def _source_bytes(source: bytes | str) -> bytes:
    return source.encode("utf-8") if isinstance(source, str) else bytes(source)


def _generated_u32_inflate_sh() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  python "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


def _generated_u32_inflate_py() -> str:
    return '''#!/usr/bin/env python
"""Inflate PR101-family u32 decoder-length archive to raw uint8 RGB frames."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))

from codec import apply_latent_sidecar, decode_decoder_compact, decode_latents_compact
from model import HNeRVDecoder
from pr101_u32_adapter import parse_archive_u32_decoder_len


CAMERA_H, CAMERA_W = 874, 1164


def inflate(src_bin: str, dst_raw: str):
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta = parse_archive_u32_decoder_len(
        archive_bytes,
        decode_decoder_compact=decode_decoder_compact,
        decode_latents_compact=decode_latents_compact,
        apply_latent_sidecar=apply_latent_sidecar,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


def default_state_dict_output_path_hint() -> str:
    """Return the preferred bulky-output root for operator CLIs."""

    return "/Volumes/VertigoDataTier/pact/pr101_per_tensor_grammar_solver"


__all__ = [
    "PR101_GROUPED_ARCHIVE_MATERIALIZATION_SCHEMA",
    "PR101_GROUPED_BROTLI_PACKET_GRAMMAR_SCHEMA",
    "PR101_GROUPED_DECODER_BLOB_MATERIALIZATION_SCHEMA",
    "PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA",
    "PR101_TENSOR_GRAMMAR_CANDIDATE_SCHEMA",
    "PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA",
    "PR101_U32_RECEIVER_ADAPTER_SOURCE_SCHEMA",
    "PR101_U32_RUNTIME_TREE_MATERIALIZATION_SCHEMA",
    "CoderName",
    "StoragePermMode",
    "build_grouped_optimizer_candidate_queue_from_report",
    "build_optimizer_candidate_queue_from_solver_report",
    "build_u32_receiver_adapter_source_from_report",
    "build_u32_receiver_runtime_tree_from_report",
    "default_state_dict_output_path_hint",
    "empirical_shannon_floor_bytes",
    "materialize_grouped_archive_from_report",
    "materialize_grouped_decoder_blob_from_report",
    "measure_tensor_grammar_candidates",
    "select_best_tensor_candidate",
    "solve_grouped_brotli_packet_grammar",
    "solve_state_dict_per_tensor_grammar",
]
