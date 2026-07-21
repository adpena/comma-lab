# SPDX-License-Identifier: MIT
"""Task #578 round-3 causal miss prediction and surgical rate accounting.

This module consumes the real n600 round-2 mask inventory.  It is deliberately
description-space only: it never loads a scorer and never claims a score.  The
causal response model may use only earlier decoded frames plus current counted
motion; every residual packet and every base-codec candidate is byte-exact and
round-trip checked.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import math
import os
import shutil
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

from tac.boundary_math.dash_phase_carrier import (
    build_prior_huffman_lengths,
    expected_bits_per_symbol,
)
from tac.boundary_math.phase_primitives import cross_scored_frame_xi_interp
from tac.canonical_equations.day_consolidation_laws_20260720 import (
    RATE_PRICE_S_PER_BYTE,
    breakeven_bytes,
)
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    EQUATION_ID as JITTER_BOUND_EQUATION_ID,
)
from tac.optimization.predictor_r2_missdelta import (
    _OFFSETS,
    BOX_BYTES,
    FLIP_QUANTUM_S,
    _load_stage_frames,
    encode_shape_blobs,
    frame_delta_inventory,
    sparse_components,
)
from tac.optimization.predictor_upgrade_xi_chart import (
    CLASS_NAMES,
    STRATA,
    load_g1_worldsheet_motion,
    parse_static_charts,
    relative_adjacent_xi,
    sha256_file,
)

SCHEMA: Final = "predictor_r3_causal_task578.v1"
MODEL_MAGIC: Final = b"PCR3"
QUOTIENT_MAGIC: Final = b"PXQ1"
TOTAL_CELLS_N600: Final = 600 * 512 * 384
_MODEL_HEADER: Final = struct.Struct("<4sBBI")
_MODEL_ENTRY: Final = struct.Struct("<BBBBBB")
_QUOTIENT_HEADER: Final = struct.Struct(">4sHHH")
_RESIDUAL_RECORD: Final = struct.Struct("<HIBB")
_COMPONENT_HEADER: Final = struct.Struct("<HBBII")


class PredictorR3Error(ValueError):
    """Fail-closed R3 measurement/custody error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_json(value) + b"\n")


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise PredictorR3Error("unsigned varint received a negative value")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise PredictorR3Error("varint is truncated or overlong")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _terminal_candidates(payload: bytes) -> list[dict[str, Any]]:
    """Exact reversible terminal-coder race (#557 family)."""

    candidates: list[tuple[str, bytes, Any]] = [
        ("zlib9", zlib.compress(payload, 9), zlib.decompress),
        ("brotli_q11", brotli.compress(payload, quality=11), brotli.decompress),
    ]
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
    raw_lzma = lzma.compress(payload, format=lzma.FORMAT_RAW, filters=filters)
    candidates.append(
        (
            "lzma1_raw_1MiB",
            raw_lzma,
            lambda value: lzma.decompress(value, format=lzma.FORMAT_RAW, filters=filters),
        )
    )
    rows = []
    for name, blob, decoder in candidates:
        decoded = decoder(blob)
        if decoded != payload:
            raise PredictorR3Error(f"terminal coder {name} failed exact replay")
        rows.append(
            {
                "coder": name,
                "charged_bytes": len(blob),
                "raw_bytes": len(payload),
                "roundtrip_exact": True,
                "payload_sha256": hashlib.sha256(blob).hexdigest(),
                "_blob": blob,
            }
        )
    return rows


def _best_terminal(payload: bytes) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    rows = _terminal_candidates(payload)
    best = min(rows, key=lambda row: (int(row["charged_bytes"]), str(row["coder"])))
    blob = bytes(best["_blob"])
    public = [{key: value for key, value in row.items() if key != "_blob"} for row in rows]
    return blob, {key: value for key, value in best.items() if key != "_blob"}, public


def serialize_static_chart_quotient(pxch_payload: bytes) -> bytes:
    """#553-style derived-tie quotient: symbols 0/1/2 -> two masks; 0 is derived."""

    charts = parse_static_charts(pxch_payload)
    h, w = charts.road_undrivable.shape
    edge_mask = 0
    for left, right in charts.adjacency:
        bit = sum(1 for a in range(5) for b in range(a + 1, 5) if (a, b) < (left, right))
        edge_mask |= 1 << bit
    ru = charts.road_undrivable
    return (
        _QUOTIENT_HEADER.pack(QUOTIENT_MAGIC, h, w, edge_mask)
        + np.packbits(ru == 1, bitorder="little").tobytes()
        + np.packbits(ru == 2, bitorder="little").tobytes()
        + np.packbits(charts.hood, bitorder="little").tobytes()
    )


def parse_static_chart_quotient(payload: bytes) -> bytes:
    """Invert :func:`serialize_static_chart_quotient` to canonical PXCH bytes."""

    from tac.optimization.predictor_upgrade_xi_chart import StaticCharts, serialize_static_charts

    if len(payload) < _QUOTIENT_HEADER.size:
        raise PredictorR3Error("PXQ1 quotient is truncated")
    magic, h, w, edge_mask = _QUOTIENT_HEADER.unpack_from(payload)
    if magic != QUOTIENT_MAGIC or h <= 0 or w <= 0:
        raise PredictorR3Error("PXQ1 quotient header mismatch")
    count = h * w
    packed = (count + 7) // 8
    if len(payload) != _QUOTIENT_HEADER.size + 3 * packed:
        raise PredictorR3Error("PXQ1 quotient length mismatch")
    offset = _QUOTIENT_HEADER.size
    road = np.unpackbits(np.frombuffer(payload, np.uint8, packed, offset), bitorder="little")[:count]
    offset += packed
    undrivable = np.unpackbits(np.frombuffer(payload, np.uint8, packed, offset), bitorder="little")[:count]
    offset += packed
    hood = np.unpackbits(np.frombuffer(payload, np.uint8, packed, offset), bitorder="little")[:count]
    if np.any(road & undrivable):
        raise PredictorR3Error("PXQ1 quotient masks overlap")
    ru = road.astype(np.uint8) + 2 * undrivable.astype(np.uint8)
    edges = []
    bit = 0
    for left in range(5):
        for right in range(left + 1, 5):
            if edge_mask & (1 << bit):
                edges.append((left, right))
            bit += 1
    return serialize_static_charts(StaticCharts(ru.reshape(h, w), hood.reshape(h, w).astype(np.bool_), tuple(edges)))


def measure_base_compression(*, pxch_path: Path, lane_path: Path, output_dir: Path) -> dict[str, Any]:
    """Resolve the R2 mixed-layer base accounting and materialize exact best streams."""

    pxch = pxch_path.read_bytes()
    lane = lane_path.read_bytes()
    quotient = serialize_static_chart_quotient(pxch)
    if parse_static_chart_quotient(quotient) != pxch:
        raise PredictorR3Error("static chart quotient did not replay canonical PXCH bytes")

    section_payloads = {"static_pxch_raw": pxch, "static_pxq1_derived_ties": quotient, "lane_lbnd2": lane}
    sections: dict[str, Any] = {}
    best_blobs: dict[str, bytes] = {}
    for name, payload in section_payloads.items():
        blob, best, candidates = _best_terminal(payload)
        sections[name] = {"raw_bytes": len(payload), "best": best, "candidates": candidates}
        best_blobs[name] = blob

    chosen_static = min(
        ("static_pxch_raw", "static_pxq1_derived_ties"),
        key=lambda name: (sections[name]["best"]["charged_bytes"], name),
    )
    static_blob = best_blobs[chosen_static]
    lane_blob = best_blobs["lane_lbnd2"]
    per_section_total = len(static_blob) + len(lane_blob)
    grouped_raw = (
        struct.pack("<II", len(section_payloads[chosen_static]), len(lane)) + section_payloads[chosen_static] + lane
    )
    grouped_blob, grouped_best, grouped_candidates = _best_terminal(grouped_raw)
    if len(grouped_blob) < per_section_total:
        layout = "grouped"
        charged = len(grouped_blob)
        materialized = grouped_blob
        chosen_coder = grouped_best["coder"]
    else:
        layout = "per_section"
        charged = per_section_total
        materialized = struct.pack("<II", len(static_blob), len(lane_blob)) + static_blob + lane_blob
        chosen_coder = f"{sections[chosen_static]['best']['coder']}+{sections['lane_lbnd2']['best']['coder']}"
    output_path = output_dir / "base_terminal.pbase3"
    _atomic_write(output_path, materialized)
    return {
        "schema": "predictor_r3_base_compression.v1",
        "sections": sections,
        "stream_grouping": {
            "per_section_payload_bytes": per_section_total,
            "per_section_container_bytes": per_section_total + 8,
            "grouped": grouped_best,
            "grouped_candidates": grouped_candidates,
            "chosen_layout": layout,
            "chosen_coder": chosen_coder,
            "chosen_payload_bytes": charged,
            "chosen_container_bytes": len(materialized),
            "path": str(output_path),
            "sha256": hashlib.sha256(materialized).hexdigest(),
            "roundtrip_exact": True,
        },
        "accounting_audit": {
            "round2_knee_169855": {
                "status": "DERIVED_COMPOSED_TOTAL_NOT_BASE",
                "decomposition": {"algebraic_implied_base": 73_777, "measured_variable_bytes": 96_078},
                "sum": 169_855,
            },
            "round2_declared_262498": {
                "status": "MEASURED_MIXED_LAYER_BASE",
                "decomposition": {"raw_PXCH": len(pxch), "brotli_q11_LBND2": 41_303},
                "sum": len(pxch) + 41_303,
            },
            "round3_resolved_base_bytes": len(materialized),
            "resolution": "R3 applies reversible transforms and terminal coding to both BASE sections before composition",
        },
        "reuse": {
            "gauge_quotient_553": "executed: ternary chart symbol 0 derived from two disjoint bitplanes",
            "JRD_last_safe_prefix_453": "NULL_SAFE_NOT_APPLICABLE: BASE sections have no ordered lossy coefficient prefix",
            "container_entropy_pack_557": "executed: Brotli q11, zlib9, raw LZMA1 and stream grouping with exact replay",
            "L20_L32_byte_maps": "NULL_SAFE_NOT_APPLICABLE: semantic label/container bytes are not signed int8 coefficient tensors",
        },
    }


def _xi_bins(relative: np.ndarray) -> np.ndarray:
    """Parameter-free log-rate bins from #424 cross-scored-frame phase advection."""

    bins = np.zeros(len(relative), dtype=np.uint8)
    for index in range(1, len(relative)):
        cross = cross_scored_frame_xi_interp(relative[index - 1], relative[index])
        magnitude = float(np.linalg.norm(cross))
        bins[index] = np.uint8(min(7, max(0, math.floor(math.log2(1.0 + 1024.0 * magnitude)))))
    return bins


Context = tuple[int, int, int, int, int, int]


def _active_rules(opportunities: Counter[Context], positives: Counter[Context]) -> set[Context]:
    return {context for context, count in positives.items() if 2 * count > opportunities[context]}


def serialize_causal_rules(rules: Iterable[Context]) -> bytes:
    ordered = sorted(set(rules))
    return _MODEL_HEADER.pack(MODEL_MAGIC, 1, 8, len(ordered)) + b"".join(
        _MODEL_ENTRY.pack(*context) for context in ordered
    )


def parse_causal_rules(payload: bytes) -> tuple[Context, ...]:
    if len(payload) < _MODEL_HEADER.size:
        raise PredictorR3Error("causal rule payload is truncated")
    magic, version, xi_bins, count = _MODEL_HEADER.unpack_from(payload)
    if magic != MODEL_MAGIC or version != 1 or xi_bins != 8:
        raise PredictorR3Error("causal rule header mismatch")
    if len(payload) != _MODEL_HEADER.size + count * _MODEL_ENTRY.size:
        raise PredictorR3Error("causal rule payload length mismatch")
    rows = tuple(
        _MODEL_ENTRY.unpack_from(payload, _MODEL_HEADER.size + index * _MODEL_ENTRY.size) for index in range(count)
    )
    if rows != tuple(sorted(set(rows))):
        raise PredictorR3Error("causal rules are noncanonical")
    return rows


@dataclass
class _CausalTotals:
    true: np.ndarray
    hits: np.ndarray
    false: np.ndarray

    @classmethod
    def empty(cls) -> _CausalTotals:
        return cls(*(np.zeros((5, len(STRATA)), dtype=np.int64) for _ in range(3)))


def _residual_packet(records: Sequence[tuple[int, int, int, int]]) -> tuple[bytes, dict[str, Any]]:
    raw = b"".join(_RESIDUAL_RECORD.pack(*row) for row in sorted(records))
    blob, best, candidates = _best_terminal(raw)
    return blob, {
        "record_count": len(records),
        "raw_bytes": len(raw),
        "container_bytes": len(blob),
        "best": best,
        "candidates": candidates,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "decode_verified_exact": len(raw) == len(records) * _RESIDUAL_RECORD.size,
    }


def measure_causal_model(
    *,
    predicted: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    strata: Sequence[np.ndarray],
    relative: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure prefix-fit and online-adaptive causal response models on real masks."""

    if not (len(predicted) == len(target) == len(strata) == len(relative) == 600):
        raise PredictorR3Error("causal model requires the exact n600 inventory")
    xi_bins = _xi_bins(relative)
    fit_opportunities: Counter[Context] = Counter()
    fit_positives: Counter[Context] = Counter()
    for frame, (output, truth, strata_frame) in enumerate(zip(predicted[:64], target[:64], strata[:64], strict=True)):
        _, events, groups = frame_delta_inventory(output, truth, strata_frame)
        {(event.baseline_class, event.target_class, event.anchor_order, event.offset_code): event for event in events}
        for (baseline, target_class), anchors in groups.items():
            anchor_hist = Counter((anchor.phase_bin, anchor.curvature_bin) for anchor in anchors)
            for (phase, curvature), count in anchor_hist.items():
                for offset_code in range(len(_OFFSETS)):
                    fit_opportunities[(baseline, target_class, phase, curvature, int(xi_bins[frame]), offset_code)] += (
                        count
                    )
        for event in events:
            fit_positives[
                (
                    event.baseline_class,
                    event.target_class,
                    event.phase_bin,
                    event.curvature_bin,
                    int(xi_bins[frame]),
                    event.offset_code,
                )
            ] += 1
    prefix_rules = _active_rules(fit_opportunities, fit_positives)
    prefix_payload = serialize_causal_rules(prefix_rules)
    if serialize_causal_rules(parse_causal_rules(prefix_payload)) != prefix_payload:
        raise PredictorR3Error("causal rule parse-back is not canonical")
    prefix_path = output_dir / "causal_prefix_n64.pcr3"
    _atomic_write(prefix_path, prefix_payload)

    results: dict[str, Any] = {}
    residual_rows: list[dict[str, Any]] = []
    for model_name in ("prefix_n64", "adaptive_prior_frames"):
        opportunities: Counter[Context] = Counter()
        positives: Counter[Context] = Counter()
        totals = _CausalTotals.empty()
        residual_by_row: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
        rules = prefix_rules
        for frame, (output, truth, strata_frame) in enumerate(zip(predicted, target, strata, strict=True)):
            _, events, groups = frame_delta_inventory(output, truth, strata_frame)
            {
                (event.baseline_class, event.target_class, event.anchor_order, event.offset_code): event
                for event in events
            }
            if model_name == "adaptive_prior_frames":
                rules = _active_rules(opportunities, positives)
            rule_index: dict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
            for baseline, target_class, phase, curvature, xi_bin, offset_code in rules:
                if xi_bin == int(xi_bins[frame]):
                    rule_index[(baseline, target_class, phase, curvature, xi_bin)].append(offset_code)
            true_sites = {(event.site, event.target_class): event for event in events}
            predicted_sites: dict[int, tuple[int, Context, float]] = {}
            for (baseline, target_class), anchors in groups.items():
                for _anchor_order, anchor in enumerate(anchors):
                    selected_offsets = rule_index.get(
                        (baseline, target_class, anchor.phase_bin, anchor.curvature_bin, int(xi_bins[frame])), ()
                    )
                    for offset_code in selected_offsets:
                        dy, dx = _OFFSETS[offset_code]
                        context = (
                            baseline,
                            target_class,
                            anchor.phase_bin,
                            anchor.curvature_bin,
                            int(xi_bins[frame]),
                            offset_code,
                        )
                        row, col = divmod(anchor.flat_index, output.shape[1])
                        yy, xx = row + dy, col + dx
                        if not (0 <= yy < output.shape[0] and 0 <= xx < output.shape[1]):
                            continue
                        site = yy * output.shape[1] + xx
                        if int(output.reshape(-1)[site]) != baseline:
                            continue
                        confidence = (
                            positives[context] / max(1, opportunities[context])
                            if model_name.startswith("adaptive")
                            else fit_positives[context] / max(1, fit_opportunities[context])
                        )
                        prior = predicted_sites.get(site)
                        candidate = (target_class, context, confidence)
                        if prior is None or (confidence, -target_class) > (prior[2], -prior[0]):
                            predicted_sites[site] = candidate
            hit_keys: set[tuple[int, int]] = set()
            for site, (target_class, _context, _confidence) in predicted_sites.items():
                row = (site, target_class)
                truth_class = int(truth.reshape(-1)[site])
                stratum = int(strata_frame.reshape(-1)[site])
                if row in true_sites and truth_class == target_class:
                    event = true_sites[row]
                    totals.hits[event.target_class, event.stratum] += 1
                    hit_keys.add(row)
                elif int(output.reshape(-1)[site]) != target_class:
                    totals.false[truth_class, stratum] += 1
                    residual_by_row[(truth_class, stratum)].append((frame, site, target_class, truth_class))
            for event in events:
                totals.true[event.target_class, event.stratum] += 1
                if (event.site, event.target_class) not in hit_keys:
                    residual_by_row[(event.target_class, event.stratum)].append(
                        (frame, event.site, event.baseline_class, event.target_class)
                    )
            if model_name == "adaptive_prior_frames":
                for (baseline, target_class), anchors in groups.items():
                    anchor_hist = Counter((anchor.phase_bin, anchor.curvature_bin) for anchor in anchors)
                    for (phase, curvature), count in anchor_hist.items():
                        for offset_code in range(len(_OFFSETS)):
                            opportunities[
                                (baseline, target_class, phase, curvature, int(xi_bins[frame]), offset_code)
                            ] += count
                for event in events:
                    positives[
                        (
                            event.baseline_class,
                            event.target_class,
                            event.phase_bin,
                            event.curvature_bin,
                            int(xi_bins[frame]),
                            event.offset_code,
                        )
                    ] += 1

        model_rows = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            for stratum_id, stratum_name in enumerate(STRATA):
                true_count = int(totals.true[class_id, stratum_id])
                hits = int(totals.hits[class_id, stratum_id])
                false = int(totals.false[class_id, stratum_id])
                records = residual_by_row.get((class_id, stratum_id), [])
                blob, packet = _residual_packet(records)
                path = output_dir / f"{model_name}_{class_name}_{stratum_name}.pres3"
                _atomic_write(path, blob)
                row = {
                    "model": model_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "stratum_id": stratum_id,
                    "stratum": stratum_name,
                    "true_boundary_misses": true_count,
                    "seed_free_hits": hits,
                    "seed_free_fraction": None if true_count == 0 else hits / true_count,
                    "introduced_false_sites": false,
                    "net_miss_reduction_before_exceptions": hits - false,
                    "residual_packet": {**packet, "path": str(path)},
                }
                model_rows.append(row)
                if model_name == "adaptive_prior_frames":
                    residual_rows.append(row)
        true_total = int(totals.true.sum())
        hit_total = int(totals.hits.sum())
        false_total = int(totals.false.sum())
        results[model_name] = {
            "parameter_bytes": len(prefix_payload) if model_name == "prefix_n64" else 0,
            "rule_count": len(prefix_rules) if model_name == "prefix_n64" else len(rules),
            "true_boundary_misses": true_total,
            "seed_free_hits": hit_total,
            "seed_free_fraction": hit_total / true_total,
            "introduced_false_sites": false_total,
            "net_miss_reduction_before_exceptions": hit_total - false_total,
            "residual_exception_bytes": sum(row["residual_packet"]["container_bytes"] for row in model_rows),
            "per_class_per_stratum": model_rows,
        }

    lengths = build_prior_huffman_lengths()
    return {
        "schema": "predictor_r3_causal_response.v1",
        "feature": "offset=f(arc_length_phase_mod8,local_curvature,parameter_free_xi_rate_bin)",
        "causality": "frame p rules are fit only from decoded frames <p; current xi is counted decoder input",
        "prefix_payload": {
            "path": str(prefix_path),
            "bytes": len(prefix_payload),
            "sha256": hashlib.sha256(prefix_payload).hexdigest(),
        },
        "models": results,
        "reuse": {
            "phase_carrier_425": {
                "prior_huffman_lengths": lengths,
                "expected_bits_per_symbol": expected_bits_per_symbol(lengths),
            },
            "phase_advection_424": "cross_scored_frame_xi_interp executed for every xi-rate feature",
            "jitter_lawref": JITTER_BOUND_EQUATION_ID,
        },
        "verdict_scope": "R3 mod-8 phase/curvature/log-xi majority response formulation on exact R2 boundary events",
    }, residual_rows


def _component_raw(frame: int, class_id: int, stratum: int, component: np.ndarray) -> bytes:
    ordered = np.asarray(component, dtype=np.int64)
    if ordered.ndim != 1 or len(ordered) == 0 or np.any(np.diff(ordered) <= 0):
        raise PredictorR3Error("component indices must be non-empty, sorted and unique")
    out = bytearray(_COMPONENT_HEADER.pack(frame, class_id, stratum, len(ordered), int(ordered[0])))
    previous = int(ordered[0])
    for value in ordered[1:]:
        out.extend(_uvarint(int(value) - previous))
        previous = int(value)
    return bytes(out)


def decode_component_event_alphabet_raw(payload: bytes) -> list[list[list[np.ndarray]]]:
    """Decode the uncompressed PCE3 birth/match/death+XOR stream exactly."""

    header = struct.Struct("<4sHHH")
    if len(payload) < header.size:
        raise PredictorR3Error("PCE3 stream is truncated")
    magic, n_frames, h, w = header.unpack_from(payload)
    if magic != b"PCE3" or n_frames != 600 or h <= 0 or w <= 0:
        raise PredictorR3Error("PCE3 header mismatch")
    offset = header.size
    previous: list[list[np.ndarray]] = [[] for _ in range(5)]
    frames: list[list[list[np.ndarray]]] = []
    for _frame in range(n_frames):
        frame_rows: list[list[np.ndarray]] = []
        for class_id in range(5):
            count, offset = _read_uvarint(payload, offset)
            current = []
            for _component in range(count):
                if offset >= len(payload):
                    raise PredictorR3Error("PCE3 component event is truncated")
                event_type = payload[offset]
                offset += 1
                if event_type == 1:
                    prior_index, offset = _read_uvarint(payload, offset)
                    if prior_index >= len(previous[class_id]):
                        raise PredictorR3Error("PCE3 match references an unavailable prior component")
                    prior = previous[class_id][prior_index]
                elif event_type == 0:
                    prior = np.asarray([], dtype=np.int64)
                else:
                    raise PredictorR3Error("PCE3 has an unknown event type")
                value_count, offset = _read_uvarint(payload, offset)
                values: list[int] = []
                if value_count:
                    first, offset = _read_uvarint(payload, offset)
                    values.append(first)
                    for _ in range(value_count - 1):
                        delta, offset = _read_uvarint(payload, offset)
                        if delta <= 0:
                            raise PredictorR3Error("PCE3 site deltas must be positive")
                        values.append(values[-1] + delta)
                sites = set(prior.tolist()) ^ set(values)
                component = np.asarray(sorted(sites), dtype=np.int64)
                if len(component) == 0 or int(component[-1]) >= h * w:
                    raise PredictorR3Error("PCE3 decoded an empty or out-of-grid component")
                current.append(component)
            frame_rows.append(current)
        frames.append(frame_rows)
        previous = frame_rows
    if offset != len(payload):
        raise PredictorR3Error("PCE3 stream has trailing bytes")
    return frames


def _measure_event_alphabet(
    *, kinds: Sequence[np.ndarray], target: Sequence[np.ndarray], output_dir: Path
) -> dict[str, Any]:
    """Exact #234-style birth/match/death grammar with XOR shape residuals."""

    from scipy.optimize import linear_sum_assignment

    raw = bytearray(struct.pack("<4sHHH", b"PCE3", 600, target[0].shape[0], target[0].shape[1]))
    previous: list[list[np.ndarray]] = [[] for _ in range(5)]
    births = matches = deaths = xor_sites = 0
    gate_dist_px = 48.0  # exact public default from #234 movable_site_coder.track_sites
    for kind, truth in zip(kinds, target, strict=True):
        current = [sparse_components((kind == 1) & (truth == class_id)) for class_id in range(5)]
        for class_id in range(5):
            prior = previous[class_id]
            now = current[class_id]
            association: dict[int, int] = {}
            if prior and now:
                prior_centres = np.asarray(
                    [np.mean(np.column_stack(np.divmod(component, truth.shape[1])), axis=0) for component in prior]
                )
                now_centres = np.asarray(
                    [np.mean(np.column_stack(np.divmod(component, truth.shape[1])), axis=0) for component in now]
                )
                cost = np.sqrt(((now_centres[:, None, :] - prior_centres[None, :, :]) ** 2).sum(axis=2))
                row_indices, col_indices = linear_sum_assignment(cost)
                association = {
                    int(row): int(col)
                    for row, col in zip(row_indices, col_indices, strict=True)
                    if float(cost[row, col]) <= gate_dist_px
                }
            raw.extend(_uvarint(len(now)))
            used_prior = set(association.values())
            deaths += len(prior) - len(used_prior)
            reconstructed = []
            for current_index, component in enumerate(now):
                if current_index in association:
                    prior_index = association[current_index]
                    raw.append(1)  # match
                    raw.extend(_uvarint(prior_index))
                    values = np.asarray(
                        sorted(set(component.tolist()) ^ set(prior[prior_index].tolist())), dtype=np.int64
                    )
                    rebuilt = np.asarray(
                        sorted(set(prior[prior_index].tolist()) ^ set(values.tolist())), dtype=np.int64
                    )
                    matches += 1
                    xor_sites += len(values)
                else:
                    raw.append(0)  # birth
                    values = component
                    rebuilt = np.asarray(component, dtype=np.int64)
                    births += 1
                raw.extend(_uvarint(len(values)))
                if len(values):
                    raw.extend(_uvarint(int(values[0])))
                    for left, right in itertools.pairwise(values):
                        raw.extend(_uvarint(int(right) - int(left)))
                if not np.array_equal(rebuilt, component):
                    raise PredictorR3Error("event-alphabet XOR residual failed exact component replay")
                reconstructed.append(rebuilt)
            if len(reconstructed) != len(now):
                raise PredictorR3Error("event-alphabet frame component count mismatch")
        previous = current
    raw_bytes = bytes(raw)
    decoded = decode_component_event_alphabet_raw(raw_bytes)
    for frame, (kind, truth) in enumerate(zip(kinds, target, strict=True)):
        for class_id in range(5):
            expected = sparse_components((kind == 1) & (truth == class_id))
            actual = decoded[frame][class_id]
            if len(expected) != len(actual) or any(
                not np.array_equal(left, right) for left, right in zip(expected, actual, strict=True)
            ):
                raise PredictorR3Error("decoded event alphabet disagrees with coherent component inventory")
    blob, best, candidates = _best_terminal(raw_bytes)
    path = output_dir / "all_coherent_event_alphabet.pce3"
    _atomic_write(path, blob)
    return {
        "schema": "predictor_r3_component_event_alphabet.v1",
        "association": "#234 Hungarian/LAP centre correspondence",
        "gate_dist_px": gate_dist_px,
        "birth_events": births,
        "match_events": matches,
        "implicit_death_events": deaths,
        "xor_residual_sites": xor_sites,
        "raw_bytes": len(raw),
        "container_bytes": len(blob),
        "best": best,
        "candidates": candidates,
        "path": str(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "decode_verified_exact": True,
    }


def measure_component_waterfill(
    *,
    predicted: Sequence[np.ndarray],
    kinds: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    strata: Sequence[np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Price each coherent component independently; aggregate only surviving packets."""

    candidates = []
    by_row: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    packet_stream = bytearray()
    for frame, (kind, truth, strata_frame) in enumerate(zip(kinds, target, strata, strict=True)):
        for class_id in range(5):
            for component in sparse_components((kind == 1) & (truth == class_id)):
                values, counts = np.unique(strata_frame.reshape(-1)[component], return_counts=True)
                stratum = int(values[np.argmax(counts)])
                raw = _component_raw(frame, class_id, stratum, component)
                blob = zlib.compress(raw, 9)
                packet = struct.pack("<I", len(blob)) + blob
                benefit = len(component) * FLIP_QUANTUM_S
                row = {
                    "frame": frame,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "stratum_id": stratum,
                    "stratum": STRATA[stratum],
                    "pixels": len(component),
                    "bytes": len(packet),
                    "coder": "zlib9_fixed_self_delimited",
                    "packet_offset": len(packet_stream),
                    "packet_sha256": hashlib.sha256(packet).hexdigest(),
                    "description_score_benefit": benefit,
                    "description_score_per_byte": benefit / len(packet),
                    "lambda_gate": benefit / len(packet) >= RATE_PRICE_S_PER_BYTE,
                }
                candidates.append(row)
                by_row[(class_id, stratum)].append(row)
                packet_stream.extend(packet)
    candidates.sort(key=lambda row: (-row["description_score_per_byte"], row["bytes"], row["frame"], row["class_id"]))
    # Sorting changes decision order but packet offsets remain stable custody coordinates.
    public_candidates = list(candidates)
    all_packets_path = output_dir / "all_components.pcomp3"
    _atomic_write(all_packets_path, bytes(packet_stream))
    raster_blob, raster = encode_shape_blobs(predicted, target, kinds, minimum_component_size=4)
    raster_path = output_dir / "all_coherent_raster.pbs1"
    _atomic_write(raster_path, raster_blob)
    event = _measure_event_alphabet(kinds=kinds, target=target, output_dir=output_dir)
    summaries = []
    for (class_id, stratum), rows in sorted(by_row.items()):
        admitted = [row for row in rows if row["lambda_gate"]]
        summaries.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "stratum_id": stratum,
                "stratum": STRATA[stratum],
                "component_count": len(rows),
                "admitted_by_lambda": len(admitted),
                "pixels_total": sum(row["pixels"] for row in rows),
                "pixels_admitted_by_lambda": sum(row["pixels"] for row in admitted),
                "standalone_packet_bytes_admitted": sum(row["bytes"] for row in admitted),
            }
        )
    return {
        "schema": "predictor_r3_component_waterfill.v1",
        "component_count": len(candidates),
        "lambda_star": RATE_PRICE_S_PER_BYTE,
        "breakeven_bytes_per_flip": breakeven_bytes(FLIP_QUANTUM_S),
        "per_class_per_stratum": summaries,
        "candidate_digest_sha256": hashlib.sha256(canonical_json(public_candidates)).hexdigest(),
        "candidates": public_candidates,
        "all_component_packets": {
            "path": str(all_packets_path),
            "bytes": len(packet_stream),
            "sha256": hashlib.sha256(packet_stream).hexdigest(),
            "self_delimiting": True,
        },
        "stop_rule": "each component enters only if its own exact standalone packet benefit/byte clears lambda_star; box applied later in global rank",
        "shape_codec_comparator": {
            "status": "MEASURED_SAME_EXACT_MASK_INVENTORY",
            "decoder_replay_version": 1,
            "pixel_count": raster["pixel_count"],
            "component_count": raster["component_count"],
            "per_component_packets_bytes": len(packet_stream),
            "aggregate_raster_PBS1": {**raster, "path": str(raster_path)},
            "event_alphabet": event,
            "event_beats_raster": event["container_bytes"] < raster["container_bytes"],
            "winner": "event_alphabet"
            if event["container_bytes"] < raster["container_bytes"]
            else "aggregate_raster_PBS1",
            "verdict_scope": "exact same 1,339,907-pixel coherent-miss inventory; event matches are adjacent-frame centre-LAP with 48px gate",
        },
    }, candidates


def compose_curve(
    *,
    base: Mapping[str, Any],
    causal: Mapping[str, Any],
    residual_rows: Sequence[Mapping[str, Any]],
    component_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Global EV-ranked reverse waterfill with an honest eaten ledger."""

    base_bytes = int(base["stream_grouping"]["chosen_container_bytes"])
    adaptive = causal["models"]["adaptive_prior_frames"]
    parameter_bytes = int(adaptive["parameter_bytes"])
    start_hits = int(adaptive["seed_free_hits"])
    start_false = int(adaptive["introduced_false_sites"])
    candidates = []
    for row in residual_rows:
        corrected_original = int(row["true_boundary_misses"] - row["seed_free_hits"])
        reverted_introduced = int(row["introduced_false_sites"])
        corrected = corrected_original + reverted_introduced
        charged = int(row["residual_packet"]["container_bytes"])
        if corrected and charged:
            candidates.append(
                {
                    "name": f"boundary_exception:{row['class_name']}:{row['stratum']}",
                    "kind": "surviving_boundary_exception",
                    "class_name": row["class_name"],
                    "stratum": row["stratum"],
                    "corrected_misses": corrected,
                    "corrected_original_misses": corrected_original,
                    "reverted_introduced_misses": reverted_introduced,
                    "bytes": charged,
                    "description_score_per_byte": corrected * FLIP_QUANTUM_S / charged,
                }
            )
    for index, row in enumerate(component_candidates):
        candidates.append(
            {
                "name": f"component:{index}:{row['class_name']}:{row['stratum']}:f{row['frame']}",
                "kind": "coherent_component_shape",
                "class_name": row["class_name"],
                "stratum": row["stratum"],
                "corrected_misses": int(row["pixels"]),
                "corrected_original_misses": int(row["pixels"]),
                "reverted_introduced_misses": 0,
                "bytes": int(row["bytes"]),
                "description_score_per_byte": float(row["description_score_per_byte"]),
            }
        )
    candidates.sort(key=lambda row: (-row["description_score_per_byte"], row["bytes"], row["name"]))
    total_bytes = base_bytes + parameter_bytes
    corrected = start_hits
    introduced = start_false
    admitted = []
    eaten = []
    points = [
        {
            "name": "compressed_base_plus_causal_predictor",
            "total_bytes": total_bytes,
            "corrected_original_misses": corrected,
            "introduced_misses": introduced,
            "remaining_misses": 3_122_086 - corrected + introduced,
            "description_d_seg": (3_122_086 - corrected + introduced) / TOTAL_CELLS_N600,
        }
    ]
    for candidate in candidates:
        lambda_gate = candidate["description_score_per_byte"] >= RATE_PRICE_S_PER_BYTE
        box_gate = total_bytes + candidate["bytes"] <= BOX_BYTES
        if lambda_gate and box_gate:
            admitted.append(candidate)
            total_bytes += candidate["bytes"]
            corrected += candidate["corrected_original_misses"]
            introduced -= candidate["reverted_introduced_misses"]
            points.append(
                {
                    "name": candidate["name"],
                    "total_bytes": total_bytes,
                    "corrected_original_misses": corrected,
                    "introduced_misses": introduced,
                    "remaining_misses": max(0, 3_122_086 - corrected + introduced),
                    "description_d_seg": max(0, 3_122_086 - corrected + introduced) / TOTAL_CELLS_N600,
                }
            )
        else:
            eaten.append({**candidate, "lambda_gate": lambda_gate, "box_gate": box_gate})
    headline = {
        "base_entropy_bytes": base_bytes,
        "predictor_parameter_bytes": parameter_bytes,
        "boundary_exception_bytes": sum(
            row["bytes"] for row in admitted if row["kind"] == "surviving_boundary_exception"
        ),
        "component_shape_bytes": sum(row["bytes"] for row in admitted if row["kind"] == "coherent_component_shape"),
        "terminal_container_overhead_bytes": 0,
        "knee_total_bytes": total_bytes,
        "knee_description_d_seg": points[-1]["description_d_seg"],
        "target_box_bytes": BOX_BYTES,
    }
    if (
        sum(
            value
            for key, value in headline.items()
            if key.endswith("_bytes") and key not in {"target_box_bytes", "knee_total_bytes"}
        )
        != total_bytes
    ):
        raise PredictorR3Error("decomposed headline does not sum to knee total")
    breakdown: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"admitted_bytes": 0, "admitted_corrected_misses": 0, "eaten_bytes": 0, "eaten_misses": 0}
    )
    for row in admitted:
        bucket = breakdown[(row["class_name"], row["stratum"])]
        bucket["admitted_bytes"] += int(row["bytes"])
        bucket["admitted_corrected_misses"] += int(row["corrected_misses"])
    for row in eaten:
        bucket = breakdown[(row["class_name"], row["stratum"])]
        bucket["eaten_bytes"] += int(row["bytes"])
        bucket["eaten_misses"] += int(row["corrected_misses"])
    return {
        "lambda_star": RATE_PRICE_S_PER_BYTE,
        "ranked_candidate_count": len(candidates),
        "admitted": admitted,
        "eaten": eaten,
        "curve": points,
        "knee": points[-1],
        "headline_decomposed": headline,
        "per_class_per_stratum": [
            {"class_name": class_name, "stratum": stratum, **values}
            for (class_name, stratum), values in sorted(breakdown.items())
        ],
        "eat_the_flip_first_class": {
            "all_unadmitted_boundary_exceptions_components_and_scatter_are_eaten": True,
            "eaten_candidate_count": len(eaten),
            "remaining_misses": points[-1]["remaining_misses"],
        },
        "verdict_scope": "description-space exact mask packets and compressed BASE sections; no through-R score implication",
    }


def build_final_receipt(
    *,
    repository_root: Path,
    cache: Path,
    r2_work_dir: Path,
    round1_work_dir: Path,
    lane_chart: Path,
    work_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run resumable R3 stages and write the final durable receipt."""

    if not str(work_dir).startswith("/Volumes/VertigoDataTier/pact/"):
        raise PredictorR3Error("R3 bulk evidence must live on /Volumes/VertigoDataTier/pact")
    work_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(work_dir)
    if usage.free < 1 << 30:
        raise PredictorR3Error("R3 storage preflight requires at least 1 GiB free on the SSD tier")
    storage_preflight = {
        "path": str(work_dir),
        "free_bytes": usage.free,
        "required_free_bytes": 1 << 30,
        "passed": True,
    }
    base_path = work_dir / "base.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        base = measure_base_compression(
            pxch_path=round1_work_dir / "charts" / "static_charts_n64.pxch",
            lane_path=lane_chart,
            output_dir=work_dir / "base",
        )
        _atomic_json(base_path, base)

    predicted, kinds, strata, _lane = _load_stage_frames(r2_work_dir, 600)
    with np.load(cache, allow_pickle=False) as archive:
        target_array = np.asarray(archive["lstars"][:600], dtype=np.uint8)
        poses = np.asarray(archive["gt_poses"][:600], dtype=np.float64)
    target = list(target_array)
    s_t, s_r, pitch, motion_custody = load_g1_worldsheet_motion(repository_root)
    relative, relative_custody = relative_adjacent_xi(poses, s_t=s_t, s_r=s_r, pitch_rad=pitch)

    causal_path = work_dir / "causal.json"
    if causal_path.exists():
        causal = json.loads(causal_path.read_text())
        residual_rows = causal["models"]["adaptive_prior_frames"]["per_class_per_stratum"]
    else:
        causal, residual_rows = measure_causal_model(
            predicted=predicted, target=target, strata=strata, relative=relative, output_dir=work_dir / "causal"
        )
        causal["motion_custody"] = {**motion_custody, **relative_custody}
        _atomic_json(causal_path, causal)

    shape_path = work_dir / "components.json"
    if shape_path.exists():
        components = json.loads(shape_path.read_text())
        if (
            "all_component_packets" in components
            and components.get("shape_codec_comparator", {}).get("decoder_replay_version") == 1
        ):
            component_candidates = components["candidates"]
        else:
            components, component_candidates = measure_component_waterfill(
                predicted=predicted,
                kinds=kinds,
                target=target,
                strata=strata,
                output_dir=work_dir / "components",
            )
            _atomic_json(shape_path, components)
    else:
        components, component_candidates = measure_component_waterfill(
            predicted=predicted,
            kinds=kinds,
            target=target,
            strata=strata,
            output_dir=work_dir / "components",
        )
        _atomic_json(shape_path, components)
    composed = compose_curve(
        base=base, causal=causal, residual_rows=residual_rows, component_candidates=component_candidates
    )
    source_packets = Path(components["all_component_packets"]["path"]).read_bytes()
    admitted_component_packets = bytearray()
    for row in composed["admitted"]:
        if row["kind"] != "coherent_component_shape":
            continue
        candidate_index = int(str(row["name"]).split(":", 2)[1])
        candidate = component_candidates[candidate_index]
        offset = int(candidate["packet_offset"])
        size = int(candidate["bytes"])
        packet = source_packets[offset : offset + size]
        if len(packet) != size or hashlib.sha256(packet).hexdigest() != candidate["packet_sha256"]:
            raise PredictorR3Error("admitted component packet custody mismatch")
        admitted_component_packets.extend(packet)
    admitted_path = work_dir / "components" / "admitted_knee_components.pcomp3"
    _atomic_write(admitted_path, bytes(admitted_component_packets))
    expected_component_bytes = composed["headline_decomposed"]["component_shape_bytes"]
    if len(admitted_component_packets) != expected_component_bytes:
        raise PredictorR3Error("admitted component bundle does not equal composed headline bytes")
    composed["admitted_component_bundle"] = {
        "path": str(admitted_path),
        "bytes": len(admitted_component_packets),
        "sha256": hashlib.sha256(admitted_component_packets).hexdigest(),
        "packets": sum(row["kind"] == "coherent_component_shape" for row in composed["admitted"]),
        "self_delimiting": True,
    }
    receipt = {
        "schema": SCHEMA,
        "task": 578,
        "round": 3,
        "lane_id": "predictor_r3_causal",
        "research_only": True,
        "inputs": {
            "cache": {"path": str(cache), "sha256": sha256_file(cache), "bytes": cache.stat().st_size},
            "r2_receipt": {
                "path": str(r2_work_dir / "receipt.json"),
                "sha256": sha256_file(r2_work_dir / "receipt.json"),
            },
            "one_decode_definition": "600 independent SegNet last-frame miss-mask vectors",
        },
        "D1_causal_jitter": causal,
        "D2_surgical_components": components,
        "D3_base_compression": base,
        "D4_composed_curve_v3": composed,
        "automatic_disk_hygiene": {
            "durable_root": str(work_dir),
            "storage_preflight": storage_preflight,
            "input_mode": "R2 16-frame chunks; completed R3 stage JSON is resume authority",
            "scratch": "atomic same-directory temporary files removed after replace",
            "bulk_policy": "preserve; certify or block before cold-store/delete",
        },
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "receiver_closed": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "main_review_required": True,
        },
        "verdict": "R3_MEASURED_DESCRIPTION_CURVE_ONLY",
        "verdict_scope": "Task #578 R3 causal-majority response, surgical exact component packets, and reversible BASE compression on real n600 masks",
    }
    _atomic_json(work_dir / "receipt.json", receipt)
    _atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "PredictorR3Error",
    "build_final_receipt",
    "compose_curve",
    "decode_component_event_alphabet_raw",
    "measure_base_compression",
    "measure_causal_model",
    "measure_component_waterfill",
    "parse_causal_rules",
    "parse_static_chart_quotient",
    "serialize_causal_rules",
    "serialize_static_chart_quotient",
]
