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

import heapq
import itertools
import lzma
import math
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
    DECODER_BYTE_MAPS,
    FIXED_STATE_SCHEMA,
    LATENT_LZMA_FILTERS,
    N_QUANT,
    _quantize_tensor,
    encode_decoder_compact,
    pack_brotli_stream,
)

PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA = "pr101_per_tensor_grammar_solver.v1"
PR101_TENSOR_GRAMMAR_CANDIDATE_SCHEMA = "pr101_tensor_grammar_candidate.v1"
PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
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
        flat = _apply_storage_perm(q, perm).reshape(-1)
        for raw_byte_map in byte_maps:
            byte_map = str(raw_byte_map)
            if byte_map not in VALID_BYTE_MAP_STRATEGIES:
                raise ValueError(f"unknown byte_map strategy: {byte_map!r}")
            mapped = encode_byte_map(flat.astype(np.int8, copy=False), byte_map) + (
                np.array([float(scale)], dtype=np.float16).tobytes()
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


def default_state_dict_output_path_hint() -> str:
    """Return the preferred bulky-output root for operator CLIs."""

    return "/Volumes/VertigoDataTier/pact/pr101_per_tensor_grammar_solver"


__all__ = [
    "PR101_PER_TENSOR_GRAMMAR_SOLVER_SCHEMA",
    "PR101_TENSOR_GRAMMAR_CANDIDATE_SCHEMA",
    "PR101_TENSOR_GRAMMAR_OPTIMIZER_QUEUE_SCHEMA",
    "CoderName",
    "StoragePermMode",
    "build_optimizer_candidate_queue_from_solver_report",
    "default_state_dict_output_path_hint",
    "empirical_shannon_floor_bytes",
    "measure_tensor_grammar_candidates",
    "select_best_tensor_candidate",
    "solve_state_dict_per_tensor_grammar",
]
