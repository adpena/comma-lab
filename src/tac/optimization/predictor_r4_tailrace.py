# SPDX-License-Identifier: MIT
"""Task #578 R4 learned-tail three-way race.

The module reconstructs the physical R3 post-composition tail union-once and
measures a Rule-118-counted seed-conditioned cellular generator on the real
n64 prefix.  It is description-space research only: no scorer, receiver,
archive, or pointer mutation is present here.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
import re
import shutil
import struct
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

from tac.boundary_math.island_protection import identify_island_classes
from tac.boundary_math.lane_headstart import build_lane_headstart, rasterize_centerlines
from tac.canonical_equations.day_consolidation_laws_20260720 import (
    RATE_PRICE_S_PER_BYTE,
)
from tac.optimization.predictor_r2_missdelta import FLIP_QUANTUM_S, frame_delta_inventory
from tac.optimization.predictor_upgrade_xi_chart import CLASS_NAMES, STRATA

SCHEMA: Final = "predictor_r4_tailrace_task578.v2"
N64_SCHEMA: Final = "predictor_r4_tailrace_n64.v2"
CHECKPOINT_SCHEMA: Final = "predictor_r4_tailrace_checkpoint.v2"
RULE_MAGIC: Final = b"LTG4"
SCORER_RULE_MAGIC: Final = b"LTSR"
SEED_MAGIC: Final = b"LTS4"
MASK_MAGIC: Final = b"LTM4"
TOTAL_CELLS_N600: Final = 600 * 512 * 384
R3_KNEE_MISSES: Final = 1_898_681
R3_EATEN_LEDGER_MISSES: Final = 1_888_829
R2_SCATTERED_MISSES: Final = 9_852
R3_KNEE_BYTES: Final = 216_207
R3_BOX_BYTES: Final = 216_222
_RESIDUAL_RECORD: Final = struct.Struct("<HIBB")
_COMPONENT_HEADER: Final = struct.Struct("<HBBII")
_RULE: Final = struct.Struct("<4sBbbbBB")
_SCORER_RULE: Final = struct.Struct("<4sBBBB12b")
_BITSTREAM: Final = struct.Struct("<4sBHHHII")
_COMPONENT_NAME = re.compile(r"^component:(?P<index>[0-9]+):")
_STAGE_PLAN: Final = (
    ("warmup", 4, 0.25),
    ("band_fit", 12, 0.10),
    ("rate_polish", 6, 0.05),
)
_SEED_FACTORS: Final = (64, 32, 16, 8)


class PredictorR4Error(ValueError):
    """Fail-closed R4 custody, training, or accounting error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sources(repository_root: Path) -> dict[str, str]:
    """Bind a completed stage to every local source that defines its meaning."""

    relative_paths = (
        "tools/measure_predictor_upgrade_xi_chart.py",
        "src/tac/canonical_equations/day_consolidation_laws_20260720.py",
        "src/tac/canonical_equations/resize_full_kernel_structure_20260720.py",
        "src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py",
        "src/tac/boundary_math/island_protection.py",
        "src/tac/boundary_math/lane_headstart.py",
        "src/tac/boundary_math/prereq_surfaces.py",
        "src/tac/optimization/predictor_r2_missdelta.py",
        "src/tac/optimization/predictor_r3_causal.py",
        "src/tac/optimization/predictor_r4_tailrace.py",
        "src/tac/optimization/resize_full_kernel.py",
        "src/tac/scorer_exploits.py",
        ".omx/research/prereq_surfaces_flush_20260720/surface_2_rank4_prototype_bank.json",
    )
    sources = {}
    for relative_path in relative_paths:
        path = (repository_root / relative_path).resolve(strict=True)
        sources[relative_path] = sha256_file(path)
    return sources


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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredictorR4Error(f"{label} must be a mapping")
    return value


def _storage_preflight(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not str(resolved).startswith("/Volumes/VertigoDataTier/pact/evidence/predictor_r4_20260721/"):
        raise PredictorR4Error("R4 bulk evidence must remain on the designated Vertigo SSD root")
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    required = 1 << 30
    if usage.free < required:
        raise PredictorR4Error("R4 SSD preflight requires at least 1 GiB free")
    return {
        "status": "PASS",
        "path": str(resolved),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "cleanup": "atomic same-directory scratch removed; durable checkpoints preserved; no input deleted",
    }


def _stream_key(class_id: int, stratum_id: int) -> tuple[int, int]:
    if not 0 <= class_id < len(CLASS_NAMES) or not 0 <= stratum_id < len(STRATA):
        raise PredictorR4Error("stream class/stratum is out of range")
    return class_id, stratum_id


def _stream_id(class_id: int, stratum_id: int) -> str:
    return f"{CLASS_NAMES[class_id]}:{STRATA[stratum_id]}"


def build_d1_baseline_rows(
    r3: Mapping[str, Any],
    r2: Mapping[str, Any],
    *,
    physical_inventory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build exact n600 class-by-stratum literal/eaten generator bars."""

    r3_rows = {(row["class_name"], row["stratum"]): row for row in r3["D4_composed_curve_v3"]["per_class_per_stratum"]}
    r2_rows = {(int(row["class_id"]), str(row["stratum"])): row for row in r2["D1_miss_structure"]["n600"]["rows"]}
    physical_rows = (
        {row["stream"]: row for row in physical_inventory["per_stream"]} if physical_inventory is not None else {}
    )
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for stratum_id, stratum in enumerate(STRATA):
            r3_row = r3_rows.get(
                (class_name, stratum),
                {"admitted_bytes": 0, "admitted_corrected_misses": 0, "eaten_bytes": 0, "eaten_misses": 0},
            )
            r2_row = r2_rows[(class_id, stratum)]
            scattered = int(r2_row["kinds"]["scattered_incoherent"]["count"])
            ledger_sites = int(r3_row["eaten_misses"])
            physical_row = physical_rows.get(_stream_id(class_id, stratum_id))
            tail_sites = (
                int(physical_row["physical_tail_sites"]) if physical_row is not None else ledger_sites + scattered
            )
            literal_bytes = int(r3_row["eaten_bytes"])
            literal_equal_fidelity = (
                bool(physical_row["r3_eaten_candidate_union_equals_physical_tail"])
                if physical_row is not None
                else scattered == 0
            )
            eaten_score = tail_sites * FLIP_QUANTUM_S
            eaten_equivalent_bytes = eaten_score / RATE_PRICE_S_PER_BYTE
            entry_bar = (
                min(float(literal_bytes), eaten_equivalent_bytes) if literal_equal_fidelity else eaten_equivalent_bytes
            )
            rows.append(
                {
                    "stream": _stream_id(class_id, stratum_id),
                    "class_id": class_id,
                    "class_name": class_name,
                    "stratum_id": stratum_id,
                    "stratum": stratum,
                    "r3_admitted_literal_bytes": int(r3_row["admitted_bytes"]),
                    "r3_admitted_corrected_sites": int(r3_row["admitted_corrected_misses"]),
                    "r3_eaten_literal_bytes": literal_bytes,
                    "r3_eaten_candidate_sites": ledger_sites,
                    "r2_scattered_sites_without_r3_literal_candidate": scattered,
                    "physical_tail_sites": tail_sites,
                    "physical_tail_membership_sha256": (
                        physical_row["physical_tail_membership_sha256"] if physical_row is not None else None
                    ),
                    "r3_eaten_candidate_union_sites": (
                        int(physical_row["r3_eaten_candidate_union_sites"])
                        if physical_row is not None
                        else ledger_sites
                    ),
                    "tail_sites": tail_sites,
                    "literal_equal_fidelity_for_full_tail": literal_equal_fidelity,
                    "literal_status": "EXACT_R3_CANDIDATE_BYTES"
                    if literal_equal_fidelity
                    else "PARTIAL_R3_BYTES_SCATTER_ROUTE_ABSENT",
                    "eaten_score_cost": eaten_score,
                    "eaten_lambda_equivalent_bytes": eaten_equivalent_bytes,
                    "generator_strict_entry_bar_bytes": entry_bar,
                    "entry_rule": "generator exact bytes must be strictly less than this bar",
                }
            )
    if sum(row["r3_eaten_candidate_sites"] for row in rows) != R3_EATEN_LEDGER_MISSES:
        raise PredictorR4Error("R3 eaten class/stratum rows do not reproduce the delegated ledger")
    if sum(row["r2_scattered_sites_without_r3_literal_candidate"] for row in rows) != R2_SCATTERED_MISSES:
        raise PredictorR4Error("R2 scattered rows do not reproduce the delegated tail")
    if physical_inventory is None and sum(row["tail_sites"] for row in rows) != R3_KNEE_MISSES:
        raise PredictorR4Error("D1 rows do not reproduce the exact R3 knee debt")
    if physical_inventory is not None and sum(row["tail_sites"] for row in rows) != int(
        physical_inventory["physical_tail_site_count"]
    ):
        raise PredictorR4Error("D1 physical rows do not reproduce the union-once tail inventory")
    return rows


def _decompress_terminal(blob: bytes, coder: str) -> bytes:
    if coder == "zlib9":
        return zlib.decompress(blob)
    if coder == "brotli_q11":
        return bytes(brotli.decompress(blob))
    if coder == "lzma1_raw_1MiB":
        filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
        return lzma.decompress(blob, format=lzma.FORMAT_RAW, filters=filters)
    raise PredictorR4Error(f"unknown inherited R3 terminal coder: {coder}")


def _decode_boundary_records(row: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    packet = _require_mapping(row["residual_packet"], "R3 residual packet")
    path = Path(str(packet["path"])).resolve(strict=True)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != packet["sha256"]:
        raise PredictorR4Error("R3 residual packet SHA-256 mismatch")
    raw = _decompress_terminal(blob, str(packet["best"]["coder"]))
    if len(raw) != int(packet["raw_bytes"]) or len(raw) % _RESIDUAL_RECORD.size:
        raise PredictorR4Error("R3 residual packet decoded length mismatch")
    records = list(_RESIDUAL_RECORD.iter_unpack(raw))
    if len(records) != int(packet["record_count"]):
        raise PredictorR4Error("R3 residual packet record count mismatch")
    return records


def _read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise PredictorR4Error("component uvarint is truncated or overlong")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _decode_component(candidate: Mapping[str, Any], packet_stream: bytes) -> tuple[int, int, int, np.ndarray]:
    offset = int(candidate["packet_offset"])
    size = int(candidate["bytes"])
    packet = packet_stream[offset : offset + size]
    if len(packet) != size or hashlib.sha256(packet).hexdigest() != candidate["packet_sha256"]:
        raise PredictorR4Error("R3 component packet custody mismatch")
    if len(packet) < 4:
        raise PredictorR4Error("R3 component packet is truncated")
    compressed_size = struct.unpack_from("<I", packet)[0]
    if compressed_size != len(packet) - 4:
        raise PredictorR4Error("R3 component packet length prefix mismatch")
    raw = zlib.decompress(packet[4:])
    if len(raw) < _COMPONENT_HEADER.size:
        raise PredictorR4Error("R3 component raw record is truncated")
    frame, class_id, stratum_id, count, first = _COMPONENT_HEADER.unpack_from(raw)
    values = [first]
    cursor = _COMPONENT_HEADER.size
    for _ in range(count - 1):
        delta, cursor = _read_uvarint(raw, cursor)
        if delta <= 0:
            raise PredictorR4Error("R3 component site delta is not positive")
        values.append(values[-1] + delta)
    if cursor != len(raw) or len(values) != count:
        raise PredictorR4Error("R3 component raw record has trailing bytes or count drift")
    if (frame, class_id, stratum_id, count) != (
        int(candidate["frame"]),
        int(candidate["class_id"]),
        int(candidate["stratum_id"]),
        int(candidate["pixels"]),
    ):
        raise PredictorR4Error("R3 component packet metadata drift")
    return frame, class_id, stratum_id, np.asarray(values, dtype=np.int64)


def reconstruct_tail_sites(
    *,
    r3: Mapping[str, Any],
    r2_work_dir: Path,
    cache: Path,
    pair_count: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Reconstruct the physical post-R3 error set, union-once.

    R3's candidate ledger is additive, but its causal residual sites and
    component packets are not disjoint.  The receiver-equivalent target is:
    (adaptive residual records UNION original misses outside the causal event
    set) MINUS component sites actually corrected by admitted packets.
    """

    if pair_count not in (64, 600):
        raise PredictorR4Error("tail reconstruction is admitted only for n64 or n600")
    with np.load(cache, allow_pickle=False) as archive:
        truth = np.asarray(archive["lstars"][:pair_count], dtype=np.uint8)
    height, width = map(int, truth.shape[1:])
    residual_sites: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    non_event_sites: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    admitted_sites: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    eaten_candidate_sites: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    adaptive_rows = r3["D1_causal_jitter"]["models"]["adaptive_prior_frames"]["per_class_per_stratum"]
    if any(row["kind"] == "surviving_boundary_exception" for row in r3["D4_composed_curve_v3"]["admitted"]):
        raise PredictorR4Error("R4 target builder requires the measured R3 zero-boundary-admission knee")
    for row in adaptive_rows:
        key = _stream_key(int(row["class_id"]), int(row["stratum_id"]))
        selected = np.asarray(
            [
                frame * height * width + site
                for frame, site, _predicted_class, _truth_class in _decode_boundary_records(row)
                if frame < pair_count
            ],
            dtype=np.int64,
        )
        if len(selected):
            residual_sites[key].append(selected)
            eaten_candidate_sites[key].append(selected)

    chunk_records = []
    frame_cursor = 0
    for path in sorted((r2_work_dir / f"n{pair_count}" / "chunks").glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            predicted = np.asarray(archive["predicted"], dtype=np.uint8)
            strata = np.asarray(archive["strata"], dtype=np.uint8)
        if predicted.shape != strata.shape or predicted.shape[1:] != (height, width):
            raise PredictorR4Error("R2 chunk geometry mismatch during physical-tail reconstruction")
        stop = frame_cursor + len(predicted)
        if stop > pair_count:
            raise PredictorR4Error("R2 chunks exceed requested tail prefix")
        for local_index, (output, strata_frame) in enumerate(zip(predicted, strata, strict=True)):
            frame = frame_cursor + local_index
            _summary, events, _groups = frame_delta_inventory(output, truth[frame], strata_frame)
            event_sites = np.asarray([event.site for event in events], dtype=np.int64)
            outside_events = output.reshape(-1) != truth[frame].reshape(-1)
            outside_events[event_sites] = False
            for class_id in range(len(CLASS_NAMES)):
                for stratum_id in range(len(STRATA)):
                    local_sites = np.flatnonzero(
                        outside_events
                        & (truth[frame].reshape(-1) == class_id)
                        & (strata_frame.reshape(-1) == stratum_id)
                    ).astype(np.int64)
                    if len(local_sites):
                        non_event_sites[(class_id, stratum_id)].append(frame * height * width + local_sites)
        chunk_records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        frame_cursor = stop
    if frame_cursor != pair_count:
        raise PredictorR4Error("R2 chunks do not cover the requested physical-tail prefix")

    component_section = r3["D2_surgical_components"]
    packet_path = Path(component_section["all_component_packets"]["path"]).resolve(strict=True)
    packet_stream = packet_path.read_bytes()
    if hashlib.sha256(packet_stream).hexdigest() != component_section["all_component_packets"]["sha256"]:
        raise PredictorR4Error("R3 all-component stream SHA-256 mismatch")
    candidates = component_section["candidates"]
    for decision, destination in (
        (r3["D4_composed_curve_v3"]["admitted"], admitted_sites),
        (r3["D4_composed_curve_v3"]["eaten"], eaten_candidate_sites),
    ):
        for row in decision:
            if row["kind"] != "coherent_component_shape":
                continue
            match = _COMPONENT_NAME.match(str(row["name"]))
            if match is None:
                raise PredictorR4Error("R3 component decision name is noncanonical")
            candidate = candidates[int(match.group("index"))]
            if int(candidate["frame"]) >= pair_count:
                continue
            frame, class_id, stratum_id, local_sites = _decode_component(candidate, packet_stream)
            destination[(class_id, stratum_id)].append(frame * height * width + local_sites)

    def union(values: Sequence[np.ndarray]) -> np.ndarray:
        return np.unique(np.concatenate(values)) if values else np.asarray([], dtype=np.int64)

    # Component candidates are assigned to their majority stratum in R3's byte
    # ledger, but the physical target is classified by every site's true
    # stratum.  Subtract admitted component sites globally before rebuilding
    # the class x stratum rows; otherwise minority-stratum component pixels are
    # silently left in the physical tail.
    admitted_global = union([value for values in admitted_sites.values() for value in values])
    before_admission_global = union(
        [value for source in (residual_sites, non_event_sites) for values in source.values() for value in values]
    )
    admitted_present_global = np.intersect1d(before_admission_global, admitted_global, assume_unique=True)
    admitted_overcredit = len(admitted_global) - len(admitted_present_global)

    normalized: dict[tuple[int, int], np.ndarray] = {}
    per_stream = []
    for class_id in range(len(CLASS_NAMES)):
        for stratum_id in range(len(STRATA)):
            key = (class_id, stratum_id)

            residual = union(residual_sites[key])
            non_event = union(non_event_sites[key])
            admitted = union(admitted_sites[key])
            eaten_candidates = union(eaten_candidate_sites[key])
            before_admission = np.union1d(residual, non_event)
            physical = np.setdiff1d(before_admission, admitted_global, assume_unique=True)
            candidate_union = eaten_candidates
            normalized[key] = physical
            physical_sha = hashlib.sha256(physical.astype("<u8").tobytes()).hexdigest()
            candidate_sha = hashlib.sha256(candidate_union.astype("<u8").tobytes()).hexdigest()
            admitted_present = np.intersect1d(before_admission_global, admitted, assume_unique=True)
            per_stream.append(
                {
                    "stream": _stream_id(*key),
                    "class_id": class_id,
                    "stratum_id": stratum_id,
                    "adaptive_residual_sites": len(residual),
                    "adaptive_residual_records": sum(len(value) for value in residual_sites[key]),
                    "adaptive_residual_duplicate_records": sum(len(value) for value in residual_sites[key])
                    - len(residual),
                    "original_noncausal_event_sites": len(non_event),
                    "residual_noncausal_overlap_sites": len(residual) + len(non_event) - len(before_admission),
                    "admitted_component_candidate_sites": len(admitted),
                    "admitted_component_sites_actually_wrong_after_causal": len(admitted_present),
                    "admitted_component_overcredit_sites": len(admitted) - len(admitted_present),
                    "physical_tail_sites": len(physical),
                    "physical_tail_membership_sha256": physical_sha,
                    "r3_eaten_candidate_union_sites": len(candidate_union),
                    "r3_eaten_candidate_union_sha256": candidate_sha,
                    "r3_eaten_candidate_union_equals_physical_tail": np.array_equal(candidate_union, physical),
                }
            )
    if admitted_overcredit != sum(row["admitted_component_overcredit_sites"] for row in per_stream):
        raise PredictorR4Error("per-stream component overcredit does not reconcile the global union")
    causal_overlap = sum(row["residual_noncausal_overlap_sites"] for row in per_stream)
    residual_duplicates = sum(row["adaptive_residual_duplicate_records"] for row in per_stream)
    physical_total = sum(len(value) for value in normalized.values())
    if pair_count == 600 and physical_total != (
        R3_KNEE_MISSES + admitted_overcredit - causal_overlap - residual_duplicates
    ):
        raise PredictorR4Error(
            "physical n600 tail does not reconcile the R3 additive interactions: "
            f"physical={physical_total}, r3={R3_KNEE_MISSES}, "
            f"component_overcredit={admitted_overcredit}, causal_overlap={causal_overlap}, "
            f"residual_duplicates={residual_duplicates}"
        )
    custody = {
        "schema": "predictor_r4_physical_tail_inventory.v1",
        "pair_count": pair_count,
        "geometry": [pair_count, height, width],
        "r2_chunks": chunk_records,
        "r3_component_stream": {
            "path": str(packet_path),
            "bytes": len(packet_stream),
            "sha256": hashlib.sha256(packet_stream).hexdigest(),
        },
        "per_stream": per_stream,
        "physical_tail_site_count": physical_total,
        "r3_reported_additive_tail_site_count": R3_KNEE_MISSES if pair_count == 600 else None,
        "admitted_component_overcredit_site_count": admitted_overcredit,
        "causal_residual_noncausal_overlap_site_count": causal_overlap,
        "adaptive_residual_duplicate_record_count": residual_duplicates,
        "net_r3_additive_interaction_correction_sites": admitted_overcredit - causal_overlap - residual_duplicates,
        "target_membership_sha256": hashlib.sha256(
            b"".join(
                struct.pack("<BBI", *key, len(normalized[key])) + normalized[key].astype("<u8").tobytes()
                for key in sorted(normalized)
            )
        ).hexdigest(),
        "interaction_policy": "union physical error sites once; subtract only admitted sites still wrong after causal response",
    }
    return normalized, custody


def reconstruct_n64_tail_sites(
    *, r3: Mapping[str, Any], r2_work_dir: Path, cache: Path
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    return reconstruct_tail_sites(r3=r3, r2_work_dir=r2_work_dir, cache=cache, pair_count=64)


def _neighbour_count(state: np.ndarray) -> np.ndarray:
    padded = np.pad(state.astype(np.int16), ((0, 0), (1, 1), (1, 1)), mode="constant")
    total = np.zeros(state.shape, dtype=np.int16)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[:, dy : dy + state.shape[1], dx : dx + state.shape[2]]
    return total


def _coarse_seed(target: np.ndarray, factor: int) -> np.ndarray:
    frames, height, width = target.shape
    if factor < 1 or height % factor or width % factor:
        raise PredictorR4Error("seed factor must divide target geometry")
    return target.reshape(frames, height // factor, factor, width // factor, factor).any(axis=(2, 4))


def _expand_seed(seed: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(seed, factor, axis=1), factor, axis=2)


def _project_fixed_magnitude(weights: np.ndarray) -> None:
    nonzero = weights != 0.0
    weights[nonzero] = np.sign(weights[nonzero]) * np.maximum(np.abs(weights[nonzero]), 1.0)
    np.clip(weights, -127.0, 127.0, out=weights)


def _quantized_weights(weights: np.ndarray) -> np.ndarray:
    quantized = np.rint(np.clip(weights, -127.0, 127.0)).astype(np.int16)
    nonzero = quantized != 0
    quantized[nonzero] = np.sign(quantized[nonzero]) * np.maximum(np.abs(quantized[nonzero]), 1)
    return quantized


def _sample_features(seed_full: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_target = target.reshape(-1)
    positives = np.flatnonzero(flat_target)
    if len(positives) > 100_000:
        positions = np.linspace(0, len(positives) - 1, 100_000, dtype=np.int64)
        positives = positives[positions]
    desired_negative = max(1_024, min(100_000, 2 * max(1, len(positives))))
    total = len(flat_target)
    stride = max(1, total // desired_negative)
    negatives = np.arange(stride // 2, total, stride, dtype=np.int64)
    negatives = negatives[~flat_target[negatives]][:desired_negative]
    if len(negatives) < desired_negative:
        fallback = np.arange(0, total, max(1, stride // 2), dtype=np.int64)
        fallback = fallback[~flat_target[fallback]]
        negatives = np.unique(np.concatenate((negatives, fallback)))[:desired_negative]
    selected = np.concatenate((positives, negatives))
    neighbours = _neighbour_count(seed_full).reshape(-1)
    centre = seed_full.reshape(-1).astype(np.float32)
    features = np.column_stack(
        (centre[selected], neighbours[selected].astype(np.float32), np.ones(len(selected), dtype=np.float32))
    )
    labels = flat_target[selected].astype(np.float32)
    return features, labels


def _linear_logits(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 3 or weights.shape != (3,):
        raise PredictorR4Error("tail generator linear feature geometry mismatch")
    # The explicit three-term sum avoids host BLAS variability for this tiny rule.
    return features[:, 0] * weights[0] + features[:, 1] * weights[1] + features[:, 2] * weights[2]


def _generate_uint8(
    seed: np.ndarray, factor: int, weights: np.ndarray, iterations: int
) -> tuple[np.ndarray, dict[str, Any]]:
    state = _expand_seed(seed, factor).astype(np.bool_)
    quantized = _quantized_weights(weights)
    inert_at = None
    for step in range(iterations):
        neighbours = _neighbour_count(state)
        logits = quantized[0] * state.astype(np.int16) + quantized[1] * neighbours + quantized[2]
        updated = logits >= 0
        if np.array_equal(updated, state):
            inert_at = step + 1
            state = updated
            break
        state = updated
    emitted = state.astype(np.uint8) * np.uint8(255)
    parse_back = emitted >= np.uint8(128)
    if not np.array_equal(parse_back, state):
        raise PredictorR4Error("uint8 generator parse-back mismatch")
    return emitted, {
        "configured_iterations": iterations,
        "executed_iterations": inert_at or iterations,
        "rc7_inertness_halt": inert_at is not None,
        "halt_reason": "rc7_uint8_state_inert" if inert_at is not None else "iteration_cap_reached",
    }


def serialize_rule(weights: np.ndarray, *, iterations: int, seed_factor: int) -> bytes:
    quantized = _quantized_weights(weights)
    if quantized.shape != (3,) or np.any(quantized < -128) or np.any(quantized > 127):
        raise PredictorR4Error("tail generator weights are not exact int8")
    return _RULE.pack(RULE_MAGIC, 1, *(int(value) for value in quantized), iterations, seed_factor)


def parse_rule(payload: bytes) -> dict[str, Any]:
    if len(payload) != _RULE.size:
        raise PredictorR4Error("tail generator rule length mismatch")
    magic, version, centre, neighbour, bias, iterations, seed_factor = _RULE.unpack(payload)
    if magic != RULE_MAGIC or version != 1 or iterations < 1 or seed_factor not in _SEED_FACTORS:
        raise PredictorR4Error("tail generator rule header mismatch")
    return {
        "weights": np.asarray([centre, neighbour, bias], dtype=np.float32),
        "iterations": iterations,
        "seed_factor": seed_factor,
    }


def _rank4_label_direction(class_id: int) -> np.ndarray:
    """Return the exact 5-class/common-mode quotient direction for one label.

    This is only a coordinate-system identity.  It is not the frozen-head
    prototype or the missing receiver-coordinate pullback.
    """

    if not 0 <= class_id < len(CLASS_NAMES):
        raise PredictorR4Error("rank4 scorer-rule class is out of range")
    if class_id == 0:
        return np.full(4, -1, dtype=np.int16)
    direction = np.zeros(4, dtype=np.int16)
    direction[class_id - 1] = 1
    return direction


def serialize_scorer_rule(weights: np.ndarray, *, class_id: int, iterations: int, seed_factor: int) -> bytes:
    """Serialize a counted 4x3 int8 scorer-quotient weight form.

    The packet measures the requested weights bar.  Its spatial support is
    replayable, but it is deliberately ineligible for admission until the
    rank4 scorer coordinates have an exact uint8 RGB receiver pullback.
    """

    support_weights = _quantized_weights(weights)
    matrix = _rank4_label_direction(class_id)[:, None] * support_weights[None, :]
    if matrix.shape != (4, 3) or np.any(matrix < -128) or np.any(matrix > 127):
        raise PredictorR4Error("rank4 scorer-rule weights are not exact int8")
    return _SCORER_RULE.pack(
        SCORER_RULE_MAGIC,
        1,
        class_id,
        iterations,
        seed_factor,
        *(int(value) for value in matrix.reshape(-1)),
    )


def parse_scorer_rule(payload: bytes) -> dict[str, Any]:
    if len(payload) != _SCORER_RULE.size:
        raise PredictorR4Error("rank4 scorer-rule length mismatch")
    unpacked = _SCORER_RULE.unpack(payload)
    magic, version, class_id, iterations, seed_factor = unpacked[:5]
    if (
        magic != SCORER_RULE_MAGIC
        or version != 1
        or not 0 <= class_id < len(CLASS_NAMES)
        or iterations < 1
        or seed_factor not in _SEED_FACTORS
    ):
        raise PredictorR4Error("rank4 scorer-rule header mismatch")
    matrix = np.asarray(unpacked[5:], dtype=np.int16).reshape(4, 3)
    direction = _rank4_label_direction(class_id)
    pivot = int(np.flatnonzero(direction)[0])
    support_weights = matrix[pivot] * int(direction[pivot])
    if not np.array_equal(matrix, direction[:, None] * support_weights[None, :]):
        raise PredictorR4Error("rank4 scorer-rule is not a single-label quotient head")
    return {
        "class_id": class_id,
        "weights_bar": matrix,
        "support_weights": support_weights.astype(np.float32),
        "iterations": iterations,
        "seed_factor": seed_factor,
    }


def _serialize_bits(magic: bytes, bits: np.ndarray) -> bytes:
    value = np.ascontiguousarray(bits, dtype=np.bool_)
    if value.ndim != 3:
        raise PredictorR4Error("tail bitstream must have [frames,height,width] geometry")
    raw = np.packbits(value.reshape(-1), bitorder="little").tobytes()
    compressed = zlib.compress(raw, 9)
    return _BITSTREAM.pack(magic, 1, *map(int, value.shape), len(raw), len(compressed)) + compressed


def _parse_bits(payload: bytes, expected_magic: bytes) -> np.ndarray:
    if len(payload) < _BITSTREAM.size:
        raise PredictorR4Error("tail bitstream is truncated")
    magic, version, frames, height, width, raw_bytes, compressed_bytes = _BITSTREAM.unpack_from(payload)
    if magic != expected_magic or version != 1 or compressed_bytes != len(payload) - _BITSTREAM.size:
        raise PredictorR4Error("tail bitstream header mismatch")
    raw = zlib.decompress(payload[_BITSTREAM.size :])
    count = frames * height * width
    if len(raw) != raw_bytes or raw_bytes != (count + 7) // 8:
        raise PredictorR4Error("tail bitstream decoded length mismatch")
    return (
        np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[:count]
        .reshape(frames, height, width)
        .astype(np.bool_)
    )


def _polytope_warm_start(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Deterministic three-feature ridge solve used before the staged STE fit."""

    if features.ndim != 2 or features.shape[1] != 3 or labels.shape != (len(features),):
        raise PredictorR4Error("polytope warm-start feature geometry mismatch")
    target = labels.astype(np.float64) * 2.0 - 1.0
    work = features.astype(np.float64)
    gram = np.empty((3, 3), dtype=np.float64)
    rhs = np.empty(3, dtype=np.float64)
    for row in range(3):
        rhs[row] = np.sum(work[:, row] * target, dtype=np.float64)
        for column in range(3):
            gram[row, column] = np.sum(work[:, row] * work[:, column], dtype=np.float64)
    gram += np.eye(3, dtype=np.float64) * 1e-6

    # Fixed-order 3x3 Gaussian elimination avoids a host BLAS dependency.
    augmented = np.column_stack((gram, rhs))
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(float(augmented[row, column])))
        if abs(float(augmented[pivot, column])) < 1e-12:
            raise PredictorR4Error("polytope warm-start normal matrix is singular")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(3):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    weights = augmented[:, 3].astype(np.float32)
    peak = float(np.max(np.abs(weights)))
    if peak > 0.0:
        weights *= np.float32(4.0 / peak)
    _project_fixed_magnitude(weights)
    return weights


def _training_config_sha(
    *,
    target_sha256: str,
    factor: int,
    class_id: int,
    initialization_sha256: str,
    initialization_method: str,
    implementation_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": "predictor_r4_training_config.v1",
                "target_sha256": target_sha256,
                "factor": factor,
                "class_id": class_id,
                "initialization_sha256": initialization_sha256,
                "initialization_method": initialization_method,
                "implementation_sha256": implementation_sha256,
                "stage_plan": _STAGE_PLAN,
                "rare_class_prior": "task208_self_detected_islands_no_hardcoded_classes",
                "lane_geometry_prior": "openpilot_degree4_centerline_headstart",
                "ste": "saturation_aware_uint8_straight_through",
                "fixed_magnitude_floor": 1,
                "receiver": "exact_uint8_0_255_threshold_128",
                "seed": 20260721,
            }
        )
    ).hexdigest()


def _stage_checkpoint_path(output_dir: Path, stage_index: int, stage_name: str) -> Path:
    return output_dir / f"stage_{stage_index:02d}_{stage_name}.json"


def _train_factor(
    *,
    target: np.ndarray,
    factor: int,
    class_id: int,
    output_dir: Path,
    implementation_sha256: str,
    initialization_target: np.ndarray | None = None,
    initialization_method: str = "deterministic_per_stream_polytope_ridge_solve",
) -> dict[str, Any]:
    target_sha = hashlib.sha256(np.packbits(target.reshape(-1), bitorder="little").tobytes()).hexdigest()
    warm_target = target if initialization_target is None else np.asarray(initialization_target, dtype=np.bool_)
    if warm_target.shape != target.shape:
        raise PredictorR4Error("polytope initialization target geometry mismatch")
    initialization_sha = hashlib.sha256(np.packbits(warm_target.reshape(-1), bitorder="little").tobytes()).hexdigest()
    config_sha = _training_config_sha(
        target_sha256=target_sha,
        factor=factor,
        class_id=class_id,
        initialization_sha256=initialization_sha,
        initialization_method=initialization_method,
        implementation_sha256=implementation_sha256,
    )
    seed = _coarse_seed(target, factor)
    seed_full = _expand_seed(seed, factor)
    features, labels = _sample_features(seed_full, target)
    warm_seed = _coarse_seed(warm_target, factor)
    warm_features, warm_labels = _sample_features(_expand_seed(warm_seed, factor), warm_target)
    weights = _polytope_warm_start(warm_features, warm_labels)
    initial_weights = weights.copy()
    ema = weights.copy()
    checkpoints: list[dict[str, Any]] = []
    global_epoch = 0
    plateau_patience = max(1, math.ceil(math.log2(len(weights))))
    sampled_history: list[int] = []
    for stage_index, (stage_name, epochs, learning_rate) in enumerate(_STAGE_PLAN):
        checkpoint_path = _stage_checkpoint_path(output_dir, stage_index, stage_name)
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if (
                checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("config_sha256") != config_sha
                or checkpoint.get("implementation_sha256") != implementation_sha256
                or checkpoint.get("target_sha256") != target_sha
                or checkpoint.get("class_id") != class_id
                or checkpoint.get("stage_index") != stage_index
                or checkpoint.get("stage_name") != stage_name
                or checkpoint.get("stage_complete") is not True
            ):
                raise PredictorR4Error("R4 checkpoint refused config/source drift")
            weights = np.asarray(checkpoint["live_weights_fp32"], dtype=np.float32)
            ema = np.asarray(checkpoint["ema_weights_fp32"], dtype=np.float32)
            global_epoch = int(checkpoint["global_epoch"])
            sampled_history = [int(value) for value in checkpoint["sampled_realized_flip_history"]]
            checkpoints.append({"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)})
            continue
        halted = False
        for _epoch in range(epochs):
            quantized = _quantized_weights(weights).astype(np.float32)
            logits = _linear_logits(features, quantized)
            scaled = np.clip(logits / np.float32(4.0), -20.0, 20.0)
            probabilities = 1.0 / (1.0 + np.exp(-scaled))
            residual = probabilities - labels
            gradient = np.asarray(
                [np.sum(features[:, index] * residual, dtype=np.float64) for index in range(3)],
                dtype=np.float32,
            ) / np.float32(max(1, len(labels)))
            # The hard int8 forward and float surrogate derivative are the STE contract.
            weights -= np.float32(learning_rate) * gradient.astype(np.float32)
            _project_fixed_magnitude(weights)
            ema = np.float32(0.9) * ema + np.float32(0.1) * weights
            _project_fixed_magnitude(ema)
            global_epoch += 1
            sampled_prediction = _linear_logits(features, _quantized_weights(ema)) >= 0
            realized = int(np.count_nonzero(sampled_prediction & (labels > 0.5)))
            sampled_history.append(realized)
            if stage_name == "band_fit" and len(sampled_history) > plateau_patience:
                recent = sampled_history[-(plateau_patience + 1) :]
                if len(set(recent)) == 1:
                    halted = True
                    break
        emitted, continuation = _generate_uint8(seed, factor, ema, iterations=4)
        prediction = emitted >= 128
        true_positives = int(np.count_nonzero(prediction & target))
        false_positives = int(np.count_nonzero(prediction & ~target))
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "config_sha256": config_sha,
            "implementation_sha256": implementation_sha256,
            "target_sha256": target_sha,
            "class_id": class_id,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_complete": True,
            "global_epoch": global_epoch,
            "live_weights_fp32": weights.astype(float).tolist(),
            "ema_weights_fp32": ema.astype(float).tolist(),
            "initialization": {
                "method": initialization_method,
                "initialization_target_sha256": initialization_sha,
                "warm_start_weights_fp32": initial_weights.astype(float).tolist(),
                "task208_rare_class_protection_numerically_consumed": initialization_target is not None,
                "openpilot_degree4_lane_prior_numerically_consumed": initialization_target is not None,
            },
            "quantized_ema_weights_int8": _quantized_weights(ema).astype(int).tolist(),
            "sampled_realized_flip_history": sampled_history,
            "receiver_telemetry": {
                "authority": "exact_uint8_receiver_path_non_score",
                "target_sites": int(np.count_nonzero(target)),
                "realized_flip_count": true_positives,
                "realized_collateral_count": false_positives,
                "band_hit_rate": true_positives / max(1, int(np.count_nonzero(target))),
                **continuation,
            },
            "continuation": {
                "event": "rc7_sampled_realized_flip_plateau",
                "plateau_patience_epochs": plateau_patience,
                "halted_on_plateau": halted,
            },
        }
        _atomic_json(checkpoint_path, checkpoint)
        checkpoints.append({"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)})
        if halted and stage_name == "band_fit":
            # Rate polish still executes from this preserved receiver-realized state.
            pass
    emitted, continuation = _generate_uint8(seed, factor, ema, iterations=4)
    prediction = emitted >= 128
    rule_payload = serialize_rule(ema, iterations=4, seed_factor=factor)
    parsed = parse_rule(rule_payload)
    replay, _ = _generate_uint8(seed, factor, parsed["weights"], int(parsed["iterations"]))
    if not np.array_equal(replay, emitted):
        raise PredictorR4Error("quantized tail rule parse-back changed uint8 output")
    scorer_rule_payload = serialize_scorer_rule(
        ema,
        class_id=class_id,
        iterations=4,
        seed_factor=factor,
    )
    scorer_parsed = parse_scorer_rule(scorer_rule_payload)
    scorer_replay, _ = _generate_uint8(
        seed,
        factor,
        scorer_parsed["support_weights"],
        int(scorer_parsed["iterations"]),
    )
    if not np.array_equal(scorer_replay, emitted):
        raise PredictorR4Error("rank4 scorer-rule changed the exact support parse-back")
    seed_payload = _serialize_bits(SEED_MAGIC, seed)
    if not np.array_equal(_parse_bits(seed_payload, SEED_MAGIC), seed):
        raise PredictorR4Error("tail seed parse-back mismatch")
    exceptions = prediction ^ target
    exception_payload = _serialize_bits(MASK_MAGIC, exceptions)
    if not np.array_equal(_parse_bits(exception_payload, MASK_MAGIC), exceptions):
        raise PredictorR4Error("tail exception parse-back mismatch")
    corrected = prediction ^ _parse_bits(exception_payload, MASK_MAGIC)
    if not np.array_equal(corrected, target):
        raise PredictorR4Error("tail generator plus own exceptions lacks exact fidelity")
    return {
        "factor": factor,
        "config_sha256": config_sha,
        "target_sha256": target_sha,
        "checkpoints": checkpoints,
        "counted_weight_bytes": len(rule_payload),
        "scorer_rank4_counted_weight_bytes": len(scorer_rule_payload),
        "instance_seed_bytes": len(seed_payload),
        "own_exception_bytes": len(exception_payload),
        "exact_bytes": len(rule_payload) + len(seed_payload) + len(exception_payload),
        "scorer_rank4_exact_support_bytes": len(scorer_rule_payload) + len(seed_payload) + len(exception_payload),
        "weights_sha256": hashlib.sha256(rule_payload).hexdigest(),
        "scorer_rank4_weights_sha256": hashlib.sha256(scorer_rule_payload).hexdigest(),
        "instance_seed_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "own_exceptions_sha256": hashlib.sha256(exception_payload).hexdigest(),
        "quantized_weights": _quantized_weights(ema).astype(int).tolist(),
        "polytope_warm_start_weights": initial_weights.astype(float).tolist(),
        "polytope_warm_start": {
            "method": initialization_method,
            "initialization_target_sha256": initialization_sha,
            "openpilot_degree4_lane_prior_numerically_consumed": initialization_target is not None,
        },
        "scorer_rank4_weights_bar": scorer_parsed["weights_bar"].astype(int).tolist(),
        "scorer_rank4_contract": {
            "rank": 4,
            "class_count": 5,
            "fifth_common_mode_coordinate": 0,
            "coordinate": "exact_label_quotient_mod_common_mode",
            "support_parse_back_exact": True,
            "frozen_head_prototype_alignment": "BLOCKED_PROTOTYPE_ARRAYS_NOT_PRESENT_IN_SEALED_RECEIPT",
            "rank4_to_rgb_uint8_receiver_pullback": "ABSENT",
            "resize_full_kernel_projector": "AVAILABLE_FOR_SPATIAL_KERNEL_ONLY_NOT_SCORER_TO_RGB_PULLBACK",
            "eligible_for_admission": False,
        },
        "target_sites": int(np.count_nonzero(target)),
        "realized_true_positive_sites": int(np.count_nonzero(prediction & target)),
        "realized_false_positive_sites": int(np.count_nonzero(prediction & ~target)),
        "band_hit_rate": int(np.count_nonzero(prediction & target)) / max(1, int(np.count_nonzero(target))),
        "equal_fidelity_after_own_exceptions": True,
        "uint8_receiver": {"values": [0, 255], "threshold": 128, "parse_back_exact": True, **continuation},
    }


def _literal_payload(target: np.ndarray) -> bytes:
    payload = _serialize_bits(MASK_MAGIC, target)
    if not np.array_equal(_parse_bits(payload, MASK_MAGIC), target):
        raise PredictorR4Error("n64 literal target packet parse-back mismatch")
    return payload


def _build_generator_prior_inventory(*, repository_root: Path, cache: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Measure the reused #208/openpilot/rank4 initialization surfaces on n64."""

    with np.load(cache, allow_pickle=False) as archive:
        truth = np.asarray(archive["lstars"][:64], dtype=np.uint8)
    if truth.shape != (64, 384, 512):
        raise PredictorR4Error("generator-prior inventory requires the exact n64 cache geometry")

    islands = identify_island_classes(truth, n_classes=len(CLASS_NAMES))
    if islands.lane_cls is None:
        raise PredictorR4Error("task #208 did not self-detect the n64 lane island class")
    lane = build_lane_headstart(truth, degree=4, n_target_frames=600)
    if not lane.roundtrip_exact:
        raise PredictorR4Error("openpilot degree-4 lane prior failed its exact base/residual roundtrip")
    lane_base = np.stack(
        [rasterize_centerlines(frame, truth.shape[1], truth.shape[2]) for frame in lane.centerlines_per_frame]
    )

    receipt_path = (
        repository_root / ".omx/research/prereq_surfaces_flush_20260720/surface_2_rank4_prototype_bank.json"
    ).resolve(strict=True)
    rank4 = json.loads(receipt_path.read_text())
    if (
        rank4.get("schema") != "rank4_valid_cell_prototypes_v1"
        or rank4.get("rank") != 4
        or rank4.get("class_count") != 5
    ):
        raise PredictorR4Error("sealed rank4 prototype receipt is not the required 5-class rank-4 surface")

    upstream_root = repository_root / "upstream"
    segnet_weights = upstream_root / "models/segnet.safetensors"
    activation_extractor = (repository_root / "src/tac/scorer_exploits.py").resolve(strict=True)

    prior = {
        "schema": "predictor_r4_generator_prior_inventory.v1",
        "pair_count": 64,
        "task208_rare_class_protection": {
            "detection": {
                "lane_cls": islands.lane_cls,
                "movable_cls": islands.movable_cls,
                "island_classes": list(islands.island_classes),
                "n_classes": islands.n_classes,
                "evidence": [asdict(row) for row in islands.evidence],
            },
            "use": "self_detected_per_class_training_initialization; zero_shipped_dense_seed_bytes",
            "n64_scope_warning": (
                "Movable is not protected when the sealed detector does not select it on this prefix; "
                "no class index is hardcoded"
            ),
        },
        "openpilot_degree4_lane_prior": {
            "degree": lane.degree,
            "n_frames": lane.n_frames,
            "base_lane_dseg": lane.base_lane_dseg,
            "from_scratch_lane_dseg": lane.from_scratch_lane_dseg,
            "recovered_frac": lane.recovered_frac,
            "roundtrip_exact": lane.roundtrip_exact,
            "iou_mean": lane.iou_mean,
            "base_mask_sites": int(np.count_nonzero(lane_base)),
            "base_mask_sha256": hashlib.sha256(
                np.packbits(lane_base.reshape(-1), bitorder="little").tobytes()
            ).hexdigest(),
            "counted_if_shipped": lane.bytes,
            "use": ("numerical Lane polytope warm-start target only; no uncounted video-derived coefficients shipped"),
        },
        "scorer_rank4": {
            "sealed_receipt": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
            "rank": rank4["rank"],
            "class_count": rank4["class_count"],
            "frozen_weights_sha256": rank4["frozen_weights_sha256"],
            "quotient_basis_sha256": rank4["quotient_basis_sha256"],
            "prototype_sha256": rank4["prototype_sha256"],
            "rank4_reconstruction_maxabs_fp32": rank4["rank4_reconstruction_maxabs_fp32"],
            "fifth_common_mode_coordinate": 0,
            "prototype_array_custody": "ABSENT_FROM_SEALED_RECEIPT",
            "rank4_to_rgb_uint8_receiver_pullback": "ABSENT",
            "resize_full_kernel_projector": (
                "AVAILABLE_FOR_SPATIAL_KERNEL_NULLSPACE_ONLY_NOT_SCORER_COORDINATE_PULLBACK"
            ),
            "admission": "BLOCKED_RECEIVER_OPEN",
        },
        "upstream_weights_prior": {
            "directive": (
                "encode-side deeper-layer bases and intermediate activations are reusable; "
                "only shipped weight-derived constants may execute at decode"
            ),
            "worktree_custody": {
                "upstream_directory": str(upstream_root),
                "upstream_directory_present": upstream_root.is_dir(),
                "segnet_weights_expected_path": str(segnet_weights),
                "segnet_weights_present": segnet_weights.is_file(),
                "posenet_weights_enumerable": upstream_root.is_dir(),
            },
            "encode_side_activation_extractor": {
                "path": str(activation_extractor),
                "sha256": sha256_file(activation_extractor),
                "callable": "tac.scorer_exploits.preserve_skip_features",
                "execution_status": "BLOCKED_FROZEN_MODEL_BYTES_ABSENT_IN_DELEGATED_WORKTREE",
            },
            "deeper_layer_basis_arrays": "ABSENT_NO_LOCAL_WEIGHT_TENSORS",
            "intermediate_activation_custody": "ABSENT_NO_SCORER_FORWARD_AUTHORIZED_OR_RUN",
            "shipped_weight_derived_constants": "NOT_EMITTED_NO_ARRAY_CUSTODY_TO_SERIALIZE_AND_COUNT",
            "posenet_g1_tube_tightening": "OUT_OF_SCOPE_RECEIVER_SIBLING_AND_NO_LOCAL_WEIGHT_CUSTODY",
            "inflate_rule": "NO_SEGNET_OR_POSENET_LOAD_AT_INFLATE_TIME",
            "race_admission": "BLOCKED_LOCAL_WEIGHT_AND_ACTIVATION_CUSTODY",
            "verdict_scope": "this delegated worktree only; upstream-weight prior family remains open",
        },
        "authority": "MEASURED_n64_GEOMETRY_AND_SEALED_STRUCTURAL_PRIOR_NON_SCORE",
    }
    inventory = {**prior, "inventory_sha256": hashlib.sha256(canonical_json(prior)).hexdigest()}
    return inventory, lane_base


def _measure_stream(
    *,
    class_id: int,
    stratum_id: int,
    target_sites: np.ndarray,
    geometry: tuple[int, int, int],
    output_dir: Path,
    implementation_sha256: str,
    generator_priors: Mapping[str, Any],
    initialization_target: np.ndarray | None,
) -> dict[str, Any]:
    target = np.zeros(math.prod(geometry), dtype=np.bool_)
    target[target_sites] = True
    target = target.reshape(geometry)
    literal = _literal_payload(target)
    target_count = len(target_sites)
    literal_cost = RATE_PRICE_S_PER_BYTE * len(literal)
    eaten_cost = target_count * FLIP_QUANTUM_S
    entry_bar = min(float(len(literal)), eaten_cost / RATE_PRICE_S_PER_BYTE)
    architecture_rows = []
    for factor in _SEED_FACTORS:
        minimum_payload = _RULE.size + _BITSTREAM.size
        if minimum_payload >= entry_bar and architecture_rows:
            architecture_rows.append(
                {
                    "factor": factor,
                    "status": "PRUNED_BY_STRICT_ENTRY_BAR",
                    "minimum_known_bytes": minimum_payload,
                    "entry_bar_bytes": entry_bar,
                }
            )
            continue
        architecture_rows.append(
            {
                "status": "MEASURED",
                **_train_factor(
                    target=target,
                    factor=factor,
                    class_id=class_id,
                    output_dir=output_dir / f"factor_{factor}",
                    implementation_sha256=implementation_sha256,
                    initialization_target=initialization_target,
                    initialization_method=(
                        "task208_lane_openpilot_degree4_polytope_pretrain"
                        if initialization_target is not None
                        else "deterministic_per_stream_polytope_ridge_solve"
                    ),
                ),
            }
        )
    measured = [row for row in architecture_rows if row["status"] == "MEASURED"]
    if not measured:
        raise PredictorR4Error("architecture bar pruned every generator candidate")
    generator = min(measured, key=lambda row: (int(row["exact_bytes"]), int(row["factor"])))
    generator_cost = RATE_PRICE_S_PER_BYTE * int(generator["exact_bytes"])
    scorer_generator_cost = RATE_PRICE_S_PER_BYTE * int(generator["scorer_rank4_exact_support_bytes"])
    island_detection = generator_priors["task208_rare_class_protection"]["detection"]
    if class_id == int(island_detection["lane_cls"]):
        prior_role = "task208_rare_class_lane_plus_openpilot_degree4_geometry"
    elif island_detection["movable_cls"] is not None and class_id == int(island_detection["movable_cls"]):
        prior_role = "task208_rare_class_movable_geometry"
    else:
        prior_role = "per_stream_polytope_geometry"
    alternatives = [
        {
            "option": "literal_exceptions",
            "eligible_for_admission": True,
            "exact_bytes": len(literal),
            "delta_score": 0.0,
            "lagrangian_cost": literal_cost,
            "breakdown": {"literal_exception_bytes": len(literal)},
            "payload_sha256": hashlib.sha256(literal).hexdigest(),
        },
        {
            "option": "learned_generator",
            "eligible_for_admission": True,
            "exact_bytes": int(generator["exact_bytes"]),
            "delta_score": 0.0,
            "lagrangian_cost": generator_cost,
            "breakdown": {
                key: generator[key]
                for key in (
                    "counted_weight_bytes",
                    "instance_seed_bytes",
                    "own_exception_bytes",
                    "weights_sha256",
                    "instance_seed_sha256",
                    "own_exceptions_sha256",
                )
            },
        },
        {
            "option": "learned_generator_scorer_rank4",
            "eligible_for_admission": False,
            "admission_blocker": "RANK4_TO_RGB_UINT8_RECEIVER_PULLBACK_ABSENT",
            "exact_bytes": int(generator["scorer_rank4_exact_support_bytes"]),
            "delta_score": 0.0,
            "lagrangian_cost": scorer_generator_cost,
            "breakdown": {
                "counted_weight_bytes": int(generator["scorer_rank4_counted_weight_bytes"]),
                "instance_seed_bytes": int(generator["instance_seed_bytes"]),
                "own_exception_bytes": int(generator["own_exception_bytes"]),
                "weights_sha256": generator["scorer_rank4_weights_sha256"],
                "rank4": generator["scorer_rank4_contract"],
            },
            "measurement_scope": "COUNTED_WEIGHTS_BAR_AND_EXACT_SUPPORT_PARSEBACK_ONLY",
        },
        {
            "option": "eaten_flip",
            "eligible_for_admission": True,
            "exact_bytes": 0,
            "delta_score": eaten_cost,
            "lagrangian_cost": eaten_cost,
            "breakdown": {
                "positive_pixel_count": target_count,
                "exact_description_d_seg": target_count / TOTAL_CELLS_N600,
                "score_cost": eaten_cost,
            },
        },
    ]
    eligible = [row for row in alternatives if row["eligible_for_admission"]]
    costs = [float(row["lagrangian_cost"]) for row in eligible]
    minimum = min(costs)
    winner = eligible[costs.index(minimum)]["option"] if costs.count(minimum) == 1 else "NO_UNIQUE_WINNER"
    return {
        "stream": _stream_id(class_id, stratum_id),
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "stratum_id": stratum_id,
        "stratum": STRATA[stratum_id],
        "measurement_label": "MEASURED_DEVELOPMENT_PREFIX_n64",
        "equal_fidelity_target_sha256": hashlib.sha256(
            np.packbits(target.reshape(-1), bitorder="little").tobytes()
        ).hexdigest(),
        "target_sites": target_count,
        "generator_entry_bar_bytes": entry_bar,
        "generator_prior_role": prior_role,
        "generator_prior_inventory_sha256": generator_priors["inventory_sha256"],
        "architecture_rows": architecture_rows,
        "selected_generator": generator,
        "alternatives": alternatives,
        "winner": winner,
        "generator_clears_bar": winner == "learned_generator",
        "scorer_rank4_clears_weights_bar": int(generator["scorer_rank4_exact_support_bytes"]) < entry_bar,
        "scorer_rank4_admission_blocked": True,
        "verdict_scope": "real n64 R3-eaten plus R2-scattered description mask; not n600 or through-R",
    }


def run_n64_stage(
    *,
    repository_root: Path,
    cache: Path,
    r2_work_dir: Path,
    r3_receipt_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run or resume the real-prefix generator race and preserve every stage."""

    storage = _storage_preflight(work_dir)
    repository_root = repository_root.resolve(strict=True)
    implementation_sources = _implementation_sources(repository_root)
    implementation_sha = hashlib.sha256(canonical_json(implementation_sources)).hexdigest()
    r3_receipt_path = r3_receipt_path.resolve(strict=True)
    cache = cache.resolve(strict=True)
    r2_work_dir = r2_work_dir.resolve(strict=True)
    r3 = json.loads(r3_receipt_path.read_text())
    if r3.get("schema") != "predictor_r3_causal_task578.v1":
        raise PredictorR4Error("R4 requires the exact R3 receipt schema")
    sites, target_custody = reconstruct_n64_tail_sites(r3=r3, r2_work_dir=r2_work_dir, cache=cache)
    generator_priors, lane_initialization_target = _build_generator_prior_inventory(
        repository_root=repository_root, cache=cache
    )
    geometry = tuple(int(value) for value in target_custody["geometry"])
    source_config = {
        "implementation_sources": implementation_sources,
        "implementation_sha256": implementation_sha,
        "r3_receipt_sha256": sha256_file(r3_receipt_path),
        "r2_n64_receipt_sha256": sha256_file(r2_work_dir / "n64" / "receipt.json"),
        "cache_sha256": sha256_file(cache),
        "target_membership_sha256": target_custody["target_membership_sha256"],
        "generator_prior_inventory_sha256": generator_priors["inventory_sha256"],
        "lambda_star": RATE_PRICE_S_PER_BYTE,
        "flip_quantum_S": FLIP_QUANTUM_S,
    }
    source_config_sha = hashlib.sha256(canonical_json(source_config)).hexdigest()
    receipt_path = work_dir / "n64" / "receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior.get("source_config_sha256") != source_config_sha:
            raise PredictorR4Error("completed R4 n64 stage refused source/config drift")
        return prior
    races = []
    for class_id in range(len(CLASS_NAMES)):
        for stratum_id in range(len(STRATA)):
            races.append(
                _measure_stream(
                    class_id=class_id,
                    stratum_id=stratum_id,
                    target_sites=sites[(class_id, stratum_id)],
                    geometry=geometry,
                    output_dir=work_dir / "n64" / "streams" / f"c{class_id}_s{stratum_id}",
                    implementation_sha256=implementation_sha,
                    generator_priors=generator_priors,
                    initialization_target=(
                        lane_initialization_target
                        if class_id == int(generator_priors["task208_rare_class_protection"]["detection"]["lane_cls"])
                        else None
                    ),
                )
            )
    receipt = {
        "schema": N64_SCHEMA,
        "task": 578,
        "round": 4,
        "lane_id": "predictor_r4_tailrace",
        "research_only": True,
        "measurement_label": "MEASURED_DEVELOPMENT_PREFIX_n64",
        "source_config": source_config,
        "source_config_sha256": source_config_sha,
        "target_custody": target_custody,
        "generator_prior_inventory": generator_priors,
        "stream_races": races,
        "generator_winning_streams": [row["stream"] for row in races if row["generator_clears_bar"]],
        "scorer_rank4_weights_bar_clearing_streams": [
            row["stream"] for row in races if row["scorer_rank4_clears_weights_bar"]
        ],
        "training_contract": {
            "trainer_reuse": "s3.integer_plane_banded_trainer:uint8_ste+fixed_magnitude+realized_flip+rc7_inertness",
            "shared_rule_shape": "seed_conditioned_local_cellular_update_shared_across_64_frames",
            "initialization_reuse": {
                "polytope_pretrain": "deterministic_per_stream_three_feature_ridge_solve",
                "rare_class_protection": "task208_self_detected_islands_no_hardcoded_classes",
                "protected_class_ids": generator_priors["task208_rare_class_protection"]["detection"]["island_classes"],
                "lane_geometry": "openpilot_degree4_centerline_prior",
            },
            "scorer_rank4_form": {
                "counted_weights_bar_evaluated_per_stream": True,
                "support_parse_back_exact": True,
                "admission": "BLOCKED_RANK4_TO_RGB_UINT8_RECEIVER_PULLBACK_ABSENT",
            },
            "rule118": {
                "generic_generator_compute_is_free": True,
                "generator_weights_are_counted_payload": True,
                "instance_seeds_are_counted_payload": True,
                "own_exceptions_are_counted_payload": True,
                "training_or_launch_performed": True,
            },
            "per_stage_checkpoints_preserved": True,
            "resume_refuses_source_or_config_drift": True,
        },
        "automatic_disk_hygiene": storage,
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "main_review_required": True,
        },
        "verdict_scope": "real n64 prefix of the exact R3-eaten plus R2-scattered description tail",
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _governed_ticket(*, n64: Mapping[str, Any], work_dir: Path) -> dict[str, Any] | None:
    winners = list(n64["generator_winning_streams"])
    if not winners:
        return None
    config = {
        "schema": "predictor_r4_tailrace_n600_governed_ticket.v1",
        "lane_id": "predictor_r4_tailrace_n600",
        "parent_lane_id": "predictor_r4_tailrace",
        "content_lineage": "from_scratch_rule_on_our_exact_R3_R2_description_tail",
        "crux_alignment": "describe: learned-tail band only",
        "pair_count": 600,
        "chunk_size": 16,
        "streams": winners,
        "source_config_sha256": n64["source_config_sha256"],
        "resume_required": True,
        "per_stage_checkpoints_required": True,
        "storage_root": "/Volumes/VertigoDataTier/pact/evidence/predictor_r4_20260721/n600",
        "execution_route": "governed_launcher_only",
        "execution_allowed_by_this_receipt": False,
        "main_review_required_before_dispatch": True,
    }
    config_sha = hashlib.sha256(canonical_json(config)).hexdigest()
    ticket = {**config, "dsl_config_sha256": config_sha, "sealed": True}
    _atomic_json(work_dir / "governed_n600_ticket.json", ticket)
    return ticket


def build_final_receipt(
    *,
    repository_root: Path,
    cache: Path,
    r2_work_dir: Path,
    r2_receipt_path: Path,
    r3_receipt_path: Path,
    work_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compose D1-D4 after the resumable n64 measurement stage completes."""

    n64_path = work_dir / "n64" / "receipt.json"
    if not n64_path.is_file():
        raise PredictorR4Error("r4-final requires a completed r4-n64 receipt")
    n64 = json.loads(n64_path.read_text())
    r3 = json.loads(r3_receipt_path.read_text())
    r2 = json.loads(r2_receipt_path.read_text())
    repository_root = repository_root.resolve(strict=True)
    cache = cache.resolve(strict=True)
    r2_work_dir = r2_work_dir.resolve(strict=True)
    r2_receipt_path = r2_receipt_path.resolve(strict=True)
    r3_receipt_path = r3_receipt_path.resolve(strict=True)
    implementation_sources = _implementation_sources(repository_root)
    if n64.get("schema") != N64_SCHEMA or int(n64.get("target_custody", {}).get("pair_count", -1)) != 64:
        raise PredictorR4Error("r4-final requires the exact completed R4 n64 schema")
    if r2.get("schema") != "predictor_r2_missdelta_task578.v1":
        raise PredictorR4Error("r4-final requires the exact R2 receipt schema")
    if r3.get("schema") != "predictor_r3_causal_task578.v1":
        raise PredictorR4Error("r4-final requires the exact R3 receipt schema")
    prior_inventory = n64.get("generator_prior_inventory")
    if not isinstance(prior_inventory, Mapping) or not isinstance(prior_inventory.get("inventory_sha256"), str):
        raise PredictorR4Error("r4-final refused stale n64 generator-prior custody")
    expected_n64_source_config = {
        "implementation_sources": implementation_sources,
        "implementation_sha256": hashlib.sha256(canonical_json(implementation_sources)).hexdigest(),
        "r3_receipt_sha256": sha256_file(r3_receipt_path),
        "r2_n64_receipt_sha256": sha256_file(r2_work_dir / "n64" / "receipt.json"),
        "cache_sha256": sha256_file(cache),
        "target_membership_sha256": n64["target_custody"]["target_membership_sha256"],
        "generator_prior_inventory_sha256": prior_inventory["inventory_sha256"],
        "lambda_star": RATE_PRICE_S_PER_BYTE,
        "flip_quantum_S": FLIP_QUANTUM_S,
    }
    expected_n64_source_sha = hashlib.sha256(canonical_json(expected_n64_source_config)).hexdigest()
    if (
        n64.get("source_config") != expected_n64_source_config
        or n64.get("source_config_sha256") != expected_n64_source_sha
    ):
        raise PredictorR4Error("r4-final refused stale n64 source, config, or input custody")
    inventory_config = {
        "implementation_sources": implementation_sources,
        "implementation_sha256": hashlib.sha256(canonical_json(implementation_sources)).hexdigest(),
        "cache_sha256": sha256_file(cache),
        "r2_n600_receipt_sha256": sha256_file(r2_work_dir / "n600" / "receipt.json"),
        "r3_receipt_sha256": sha256_file(r3_receipt_path),
        "pair_count": 600,
        "interaction_policy": "union_once_then_subtract_admitted_if_still_wrong",
    }
    inventory_config_sha = hashlib.sha256(canonical_json(inventory_config)).hexdigest()
    inventory_path = work_dir / "n600_tail_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if inventory.get("source_config_sha256") != inventory_config_sha:
            raise PredictorR4Error("completed n600 physical-tail inventory refused source/config drift")
    else:
        _sites, inventory_body = reconstruct_tail_sites(
            r3=r3,
            r2_work_dir=r2_work_dir,
            cache=cache,
            pair_count=600,
        )
        inventory = {
            **inventory_body,
            "source_config": inventory_config,
            "source_config_sha256": inventory_config_sha,
        }
        _atomic_json(inventory_path, inventory)
    d1 = build_d1_baseline_rows(r3, r2, physical_inventory=inventory)
    ticket = _governed_ticket(n64=n64, work_dir=work_dir)
    generators = list(n64["generator_winning_streams"])
    n64_by_stream = {row["stream"]: row for row in n64["stream_races"]}
    per_stream_composition = [
        {
            "stream": row["stream"],
            "inherited_r3_admitted_literal_bytes": row["r3_admitted_literal_bytes"],
            "learned_generator_bytes": 0,
            "remaining_physical_tail_sites_eaten": row["physical_tail_sites"],
            "remaining_eaten_score_cost": row["eaten_score_cost"],
            "n64_gate_winner": n64_by_stream[row["stream"]]["winner"],
            "n64_generator_cleared": n64_by_stream[row["stream"]]["generator_clears_bar"],
            "n64_scorer_rank4_weights_bar_cleared": n64_by_stream[row["stream"]]["scorer_rank4_clears_weights_bar"],
            "scorer_rank4_admission": "BLOCKED_RANK4_TO_RGB_UINT8_RECEIVER_PULLBACK_ABSENT",
            "full_n600_route": "BLOCKED_PENDING_GOVERNED_MEASUREMENT"
            if n64_by_stream[row["stream"]]["generator_clears_bar"]
            else "R3_LITERAL_ADMISSIONS_PLUS_EATEN_PHYSICAL_TAIL",
            "reason": "R3 literal admissions remain settled; the remaining tail stays eaten unless the learned generator clears n64 and then wins a governed n600 race",
        }
        for row in d1
    ]
    r3_headline = r3["D4_composed_curve_v3"]["headline_decomposed"]
    if not generators:
        curve_v4 = {
            "status": "MEASURED_n64_NO_GENERATOR_ADMISSION_BYTES_UNCHANGED_INTERACTIONS_UNION_ONCE",
            "base_entropy_bytes": int(r3_headline["base_entropy_bytes"]),
            "causal_predictor_parameter_bytes": int(r3_headline["predictor_parameter_bytes"]),
            "admitted_literal_packet_bytes": int(r3_headline["component_shape_bytes"])
            + int(r3_headline["boundary_exception_bytes"]),
            "winning_generator_bytes": 0,
            "total_bytes": int(r3_headline["knee_total_bytes"]),
            "remaining_misses": int(inventory["physical_tail_site_count"]),
            "description_d_seg": int(inventory["physical_tail_site_count"]) / TOTAL_CELLS_N600,
            "box_bytes": int(r3_headline["target_box_bytes"]),
            "headroom_bytes": int(r3_headline["target_box_bytes"] - r3_headline["knee_total_bytes"]),
            "learned_tail_grammar": "ABSENT_DEFAULT",
            "global_lambda_star": RATE_PRICE_S_PER_BYTE,
            "r3_additive_reported_remaining_misses": int(r3["D4_composed_curve_v3"]["knee"]["remaining_misses"]),
            "r3_admitted_component_overcredit_sites": int(inventory["admitted_component_overcredit_site_count"]),
            "r3_causal_residual_noncausal_overlap_sites": int(
                inventory["causal_residual_noncausal_overlap_site_count"]
            ),
            "r3_adaptive_residual_duplicate_records": int(inventory["adaptive_residual_duplicate_record_count"]),
            "net_interaction_correction_sites": int(inventory["net_r3_additive_interaction_correction_sites"]),
            "interaction_correction": "MEASURED_UNION_ONCE",
            "per_stream_composition": per_stream_composition,
        }
        if curve_v4["total_bytes"] != R3_KNEE_BYTES or curve_v4["remaining_misses"] != (
            R3_KNEE_MISSES + int(inventory["net_r3_additive_interaction_correction_sites"])
        ):
            raise PredictorR4Error("no-generator v4 curve does not reconcile R3 interaction overcredit")
        n600_decision = "NOT_AUTHORIZED_n64_DID_NOT_CLEAR_STRICT_BAR"
    else:
        curve_v4 = {
            "status": "BLOCKED_PENDING_GOVERNED_n600_MEASUREMENT",
            "r3_base_knee_bytes": int(r3_headline["knee_total_bytes"]),
            "r3_additive_description_d_seg": float(r3_headline["knee_description_d_seg"]),
            "physical_union_once_description_d_seg": int(inventory["physical_tail_site_count"]) / TOTAL_CELLS_N600,
            "n64_generator_winning_streams": generators,
            "full_n600_bytes": None,
            "full_n600_description_d_seg": None,
            "learned_tail_grammar": "ABSENT_UNTIL_n600_STRICT_RACE",
            "global_lambda_star": RATE_PRICE_S_PER_BYTE,
            "per_stream_composition": per_stream_composition,
        }
        n600_decision = "SEALED_GOVERNED_TICKET_REQUIRES_MAIN_REVIEW"
    witness_bytes = 83_838
    witness_dseg = 0.003457972208658854
    witness_source_paths = (
        ".omx/research/duty_ticket_revision_ep725_fork_20260719_claude.md",
        ".omx/research/yhat_rd_ladder_20260719_codex.md",
    )
    witness_sources = [
        {
            "path": relative_path,
            "sha256": sha256_file(repository_root / relative_path),
        }
        for relative_path in witness_source_paths
    ]
    cross_line = {
        "policy": "NO_REPRESENTATION_LOYALTY",
        "trained_witness": {
            "epoch": 725,
            "d_seg": witness_dseg,
            "full_n600_archive_bytes": witness_bytes,
            "rate_term_from_exact_bytes": 25.0 * witness_bytes / 37_545_489.0,
            "delegated_rate_coordinate_approx": 0.118,
            "delegated_rate_coordinate_metric": "UNSPECIFIED_NOT_USED_AS_BYTE_AUTHORITY",
            "archive_sha256": "149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3",
            "custody_sources": witness_sources,
        },
        "predictor_r4": {
            "description_d_seg": int(inventory["physical_tail_site_count"]) / TOTAL_CELLS_N600,
            "description_bytes": int(r3_headline["knee_total_bytes"]),
            "receiver_closed": False,
        },
        "d_seg_gap_predictor_minus_witness": int(inventory["physical_tail_site_count"]) / TOTAL_CELLS_N600
        - witness_dseg,
        "verdict": "TRAINED_WITNESS_DOMINATES_THIS_R4_FORMULATION_ON_DSEG_AND_CUSTODIED_BYTES",
        "axis_warning": "R4 is description-space receiver-open; this comparison is representation steering, not a score",
    }
    receipt = {
        "schema": SCHEMA,
        "task": 578,
        "round": 4,
        "lane_id": "predictor_r4_tailrace",
        "research_only": True,
        "inputs": {
            "r2_receipt": {"path": str(r2_receipt_path), "sha256": sha256_file(r2_receipt_path)},
            "r3_receipt": {"path": str(r3_receipt_path), "sha256": sha256_file(r3_receipt_path)},
            "n64_receipt": {"path": str(n64_path), "sha256": sha256_file(n64_path)},
            "n600_physical_tail_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "implementation_sources": implementation_sources,
        },
        "D1_exact_stream_baselines": d1,
        "D2_n64_generator_races": {
            "measurement_label": n64["measurement_label"],
            "target_custody": n64["target_custody"],
            "generator_prior_inventory": n64["generator_prior_inventory"],
            "stream_races": n64["stream_races"],
            "generator_winning_streams": generators,
            "scorer_rank4_weights_bar_clearing_streams": n64["scorer_rank4_weights_bar_clearing_streams"],
            "verdict": (
                "n64 entry gate only; no n600 inference; rank4 weights-bar rows are receiver-open and cannot win"
            ),
        },
        "D3_composed_curve_v4": curve_v4,
        "D4_cross_line_comparison": cross_line,
        "n600_scale_decision": n600_decision,
        "governed_ticket": ticket,
        "automatic_disk_hygiene": {
            "durable_root": str(work_dir),
            "stage_checkpoints": "write-once per stream/factor/stage plus completed n64 receipt",
            "scratch": "atomic same-directory temporary files removed after replace",
            "deletion": "none",
        },
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "receiver_closed": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "main_review_required": True,
        },
        "verdict": "R4_MEASURED_n64_THREE_WAY_RACE_DESCRIPTION_ONLY",
        "verdict_scope": "small int8 seed-conditioned cellular rule on real n64 exact R3-eaten plus R2-scattered masks",
    }
    _atomic_json(work_dir / "receipt.json", receipt)
    _atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "PredictorR4Error",
    "build_d1_baseline_rows",
    "build_final_receipt",
    "parse_rule",
    "parse_scorer_rule",
    "reconstruct_n64_tail_sites",
    "run_n64_stage",
    "serialize_rule",
    "serialize_scorer_rule",
]
