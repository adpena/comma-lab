#!/usr/bin/env python3
"""Measure typed ground-description primitives on the full n600 reference.

This is a local, scorer-free, semantic-cell measurement.  It consumes preserved
frozen-scorer argmax caches and the SHA-bound target cache, but never loads a
scorer.  Results are advisory and cannot be promoted as receiver-realized score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.ddm_description_vocabulary import (  # noqa: E402
    HEIGHT,
    N_CLASSES,
    ROAD_ID,
    UNDRIVABLE_ID,
    WIDTH,
    CodedDerivation,
    decode_boundary_worldsheet_spline,
    decode_persistent_level_set,
    decode_turning_angle_curves,
    encode_joint_ground_vocabulary,
    encode_persistent_level_set,
    fit_boundary_worldsheet_spline,
    fit_persistent_level_set,
    fit_turning_angle_curves,
    inspect_coded_derivation,
)
from tac.optimization.ddm_g3_score_atlas import reconstruct_v12_state  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
)

AXIS = "[macOS-CPU frozen-scorer advisory]"
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_PAIRS = 600
C1_BYTE_BOX = 200_000
M5R_CONTROL_ROAD_ERRORS = 2_210_770
M5R_COUNTED_BYTES = 134_211
M5R_ROAD_DESCRIBED_FRACTION = 0.0


class MeasurementError(RuntimeError):
    """Raised when source custody or a full-window invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def stored_npy_memmap(path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED NPY member without copying sibling arrays."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
            raise MeasurementError(f"{path}:{member} is not ZIP_STORED")
        local_header = int(info.header_offset)
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise MeasurementError(f"bad ZIP local header for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def _baseline_error_counts(predicted: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for pair_index in range(N_PAIRS):
        target = np.asarray(labels[pair_index], dtype=np.uint8)
        wrong = predicted[pair_index] != target
        for class_id, name in enumerate(CLASS_NAMES):
            counts[name] += int(np.count_nonzero(wrong & (target == class_id)))
    return counts


def _stationarity_gap(
    transition_counts: np.ndarray,
    g2_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xi_by_target = dict.fromkeys(range(N_CLASSES), 0)
    xi_by_target[3] = 2
    partition = g2_aggregate["active_target_class_partition"]
    topology = g2_aggregate["active_topology_partition"]
    for target_id, name in enumerate(CLASS_NAMES):
        codes = [source * N_CLASSES + target_id for source in range(N_CLASSES) if source != target_id]
        counts = transition_counts[codes]
        total = int(counts.sum(dtype=np.int64))
        image_static = int(counts[counts >= 2].sum(dtype=np.int64))
        xi_static = int(xi_by_target[target_id])
        transient = total - image_static - xi_static
        persistent_energy = float(topology[name]["persistent_energy"])
        birth_energy = float(topology[name]["birth_energy"])
        rows.append(
            {
                "stratum": name,
                "g2_active_target_energy_fraction": float(partition[name]["energy_fraction"]),
                "g2_persistent_energy_fraction_within_stratum": (
                    persistent_energy / (persistent_energy + birth_energy)
                ),
                "g4_error_mass": total,
                "g4_stationarity": {
                    "STATIC_IN_IMAGE": {
                        "mass": image_static,
                        "fraction": image_static / total if total else 0.0,
                    },
                    "STATIC_IN_XI_PROXY": {
                        "mass": xi_static,
                        "fraction": xi_static / total if total else 0.0,
                    },
                    "TRANSIENT": {
                        "mass": transient,
                        "fraction": transient / total if total else 0.0,
                    },
                },
                "cross_axis_status": (
                    "MARGINALS_MEASURED_JOINT_G2_ENERGY_BY_G4_STATIONARITY_NOT_AVAILABLE"
                ),
                "primitive_family": (
                    "persistent_ground_partition_then_boundary_worldsheet"
                    if name in {"Road", "Undrivable"}
                    else "existing_event_or_program_vocabulary"
                ),
                "verdict_scope": (
                    "G2 energy and G4 argmax-event stationarity are separately measured "
                    "marginals; no unmeasured joint energy-by-stationarity table is inferred."
                ),
            }
        )
    return rows


def measure_candidate(
    *,
    candidate_id: str,
    predicted: np.ndarray,
    labels: np.ndarray,
    transition_counts: np.ndarray,
    pair_candidate: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    counted_bytes: int,
    byte_scope: str,
    baseline_errors_by_class: Mapping[str, int],
) -> dict[str, Any]:
    helpful_by_class = dict.fromkeys(CLASS_NAMES, 0)
    static_helpful_by_class = dict.fromkeys(CLASS_NAMES, 0)
    non_image_static_helpful_by_class = dict.fromkeys(CLASS_NAMES, 0)
    harmful = 0
    change_sites = 0
    candidate_errors = 0
    rows = np.arange(HEIGHT)[:, None]
    columns = np.arange(WIDTH)[None, :]
    for pair_index in range(N_PAIRS):
        before = predicted[pair_index]
        target = np.asarray(labels[pair_index], dtype=np.uint8)
        after = np.asarray(pair_candidate(pair_index, before, target), dtype=np.uint8)
        if after.shape != (HEIGHT, WIDTH) or np.any(after >= N_CLASSES):
            raise MeasurementError(f"{candidate_id}: invalid candidate plane")
        before_wrong = before != target
        after_wrong = after != target
        helpful = before_wrong & ~after_wrong
        harmful_mask = ~before_wrong & after_wrong
        harmful += int(np.count_nonzero(harmful_mask))
        change_sites += int(np.count_nonzero(after != before))
        candidate_errors += int(np.count_nonzero(after_wrong))
        codes = before * N_CLASSES + target
        recurrence = transition_counts[codes, rows, columns]
        for class_id, name in enumerate(CLASS_NAMES):
            class_helpful = helpful & (target == class_id)
            helpful_by_class[name] += int(np.count_nonzero(class_helpful))
            static_helpful_by_class[name] += int(
                np.count_nonzero(class_helpful & (recurrence >= 2))
            )
            non_image_static_helpful_by_class[name] += int(
                np.count_nonzero(class_helpful & (recurrence < 2))
            )
    baseline_total = int(sum(baseline_errors_by_class.values()))
    helpful_total = int(sum(helpful_by_class.values()))
    road_denominator = int(baseline_errors_by_class["Road"])
    return {
        "schema": "ddm_description_vocabulary_candidate.v1",
        "candidate_id": candidate_id,
        "counted_bytes": counted_bytes,
        "byte_scope": byte_scope,
        "baseline_errors": baseline_total,
        "candidate_errors": candidate_errors,
        "errors_described": helpful_total,
        "harmful_new_errors": harmful,
        "net_errors_closed": baseline_total - candidate_errors,
        "changed_sites": change_sites,
        "per_stratum_errors_described": helpful_by_class,
        "per_stratum_described_fraction": {
            name: (
                helpful_by_class[name] / int(baseline_errors_by_class[name])
                if int(baseline_errors_by_class[name])
                else 0.0
            )
            for name in CLASS_NAMES
        },
        "stationarity_of_described_errors": {
            name: {
                "STATIC_IN_IMAGE": static_helpful_by_class[name],
                "NON_IMAGE_STATIC_REMAINDER": non_image_static_helpful_by_class[name],
            }
            for name in CLASS_NAMES
        },
        "road_errors_described": helpful_by_class["Road"],
        "road_described_fraction": (
            helpful_by_class["Road"] / road_denominator if road_denominator else 0.0
        ),
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "verdict_scope": (
            "Exact semantic-cell proposal against preserved n600 frozen-scorer argmax "
            "cells and target labels. No RGB realization, Pose effect, or score delta is claimed."
        ),
    }


def _event_label_field(receiver: Any) -> np.ndarray:
    """Decode existing program/event semantics once, without rendering RGB."""

    output = np.full((N_PAIRS, HEIGHT, WIDTH), -1, dtype=np.int8)
    layer_by_role = {row.role: row for row in receiver.layers}
    for pair_index in range(N_PAIRS):
        for role in ("Lane", "MyCar", "Movable"):
            layer = layer_by_role[role]
            mask = receiver._mask_for_layer(
                layer,
                pair_index,
                replace_g1_movable=True,
            )
            output[pair_index, mask] = int(layer.class_id)
    return output


def _primitive_record(
    *,
    primitive_id: str,
    derivation: CodedDerivation,
    metadata: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "ddm_description_primitive_measurement.v1",
        "primitive_id": primitive_id,
        "coder": derivation.codec,
        "raw_bytes": derivation.raw_bytes,
        "counted_bytes": derivation.counted_bytes,
        "payload_sha256": hashlib.sha256(derivation.envelope).hexdigest(),
        "metadata": dict(metadata),
        "measurement": dict(measurement),
        "evidence_axis": AXIS,
        "score_claim": False,
        "verdict_scope": (
            "Primitive derivation bytes and semantic-cell reach only; receiver realization "
            "and contest-objective admission remain open."
        ),
    }


def _load_or_fit_primitives(
    *,
    labels: np.ndarray,
    output_directory: Path,
    temporal_stride: int,
    horizontal_stride: int,
    epsilon_pixels: float,
    heading_bins: int,
) -> tuple[
    tuple[CodedDerivation, np.ndarray],
    tuple[CodedDerivation, np.ndarray, Mapping[str, Any]],
    tuple[CodedDerivation, np.ndarray, Mapping[str, Any]],
]:
    payload_directory = output_directory / "payloads"
    payload_directory.mkdir(parents=True, exist_ok=True)

    static_path = payload_directory / "persistent_level_set.dv1"
    if static_path.exists():
        static_payload = inspect_coded_derivation(static_path.read_bytes())
        static_field = decode_persistent_level_set(static_payload.envelope)
    else:
        static_field = fit_persistent_level_set(labels)
        static_payload = encode_persistent_level_set(static_field)
        atomic_bytes(static_path, static_payload.envelope)
    atomic_json(
        output_directory / "stage_checkpoints" / "01_persistent_level_set_complete.json",
        {
            "status": "COMPLETE",
            "payload": str(static_path),
            "bytes": static_path.stat().st_size,
            "sha256": sha256_file(static_path),
        },
    )

    spline_path = payload_directory / "boundary_worldsheet_spline.dv1"
    if spline_path.exists():
        spline_payload = inspect_coded_derivation(spline_path.read_bytes())
        spline_mask, spline_meta = decode_boundary_worldsheet_spline(spline_payload.envelope)
    else:
        spline_payload, spline_mask, spline_meta = fit_boundary_worldsheet_spline(
            labels,
            temporal_stride=temporal_stride,
            horizontal_stride=horizontal_stride,
        )
        atomic_bytes(spline_path, spline_payload.envelope)
    atomic_json(
        output_directory / "stage_checkpoints" / "02_boundary_worldsheet_complete.json",
        {
            "status": "COMPLETE",
            "payload": str(spline_path),
            "bytes": spline_path.stat().st_size,
            "sha256": sha256_file(spline_path),
        },
    )

    curve_path = payload_directory / "turning_angle_curve.dv1"
    if curve_path.exists():
        curve_payload = inspect_coded_derivation(curve_path.read_bytes())
        curve_mask, curve_meta = decode_turning_angle_curves(curve_payload.envelope)
    else:
        curve_payload, curve_mask, curve_meta = fit_turning_angle_curves(
            labels,
            epsilon_pixels=epsilon_pixels,
            heading_bins=heading_bins,
        )
        atomic_bytes(curve_path, curve_payload.envelope)
    atomic_json(
        output_directory / "stage_checkpoints" / "03_turning_angle_complete.json",
        {
            "status": "COMPLETE",
            "payload": str(curve_path),
            "bytes": curve_path.stat().st_size,
            "sha256": sha256_file(curve_path),
        },
    )
    return (
        (static_payload, static_field),
        (spline_payload, spline_mask, asdict(spline_meta)),
        (curve_payload, curve_mask, asdict(curve_meta)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path(
            "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
        ),
    )
    parser.add_argument(
        "--v12-receipt",
        type=Path,
        default=Path(
            ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/"
            "ddm_v12_obligation_search_n600_receipt.json"
        ),
    )
    parser.add_argument(
        "--g4-arrays",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_"
            "20260722T212138Z/stage_checkpoints/01_recurrence_arrays.npz"
        ),
    )
    parser.add_argument(
        "--g2-aggregate",
        type=Path,
        default=Path(
            ".omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/"
            "aggregate_ledger.json"
        ),
    )
    parser.add_argument(
        "--existing-event-archive",
        type=Path,
        default=Path(
            ".omx/research/ddm_v13_g1_worldsheet_predictor_n600_20260722T201500Z/"
            "ddm_v13_islands_n600.not_a_candidate.zip.receipt-bytes"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            ".omx/research/ddm_dv1_description_vocabulary_n600_20260723T141407Z"
        ),
    )
    parser.add_argument("--temporal-stride", type=int, default=8)
    parser.add_argument("--horizontal-stride", type=int, default=16)
    parser.add_argument("--epsilon-pixels", type=float, default=3.0)
    parser.add_argument("--heading-bins", type=int, default=32)
    args = parser.parse_args()

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    source_paths = (
        args.target_cache,
        args.v12_receipt,
        args.g4_arrays,
        args.g2_aggregate,
        args.existing_event_archive,
    )
    if not all(path.exists() for path in source_paths):
        missing = [str(path) for path in source_paths if not path.exists()]
        raise MeasurementError(f"missing source paths: {missing}")

    labels = stored_npy_memmap(args.target_cache, "lstars")
    if labels.shape != (N_PAIRS, HEIGHT, WIDTH):
        raise MeasurementError(f"unexpected target label shape: {labels.shape}")
    v12_receipt = json.loads(args.v12_receipt.read_text())
    reconstructed = reconstruct_v12_state(REPO_ROOT, v12_receipt, n_pairs=N_PAIRS)
    predicted = reconstructed.final_cells
    argmax_manifests = [
        {
            "path": row.path,
            "sha256": row.sha256,
            "start": row.start,
            "stop": row.start + len(row.cells),
        }
        for _start, row in sorted(reconstructed.final_batches.items())
    ]
    baseline_errors_by_class = _baseline_error_counts(predicted, labels)
    g4_arrays = np.load(args.g4_arrays, allow_pickle=False)
    transition_counts = np.asarray(g4_arrays["transition_counts"], dtype=np.int64)
    if transition_counts.shape != (N_CLASSES * N_CLASSES, HEIGHT, WIDTH):
        raise MeasurementError("unexpected G4 transition-count shape")
    flip_codes = [
        code
        for code in range(N_CLASSES * N_CLASSES)
        if code // N_CLASSES != code % N_CLASSES
    ]
    if int(transition_counts[flip_codes].sum()) != sum(baseline_errors_by_class.values()):
        raise MeasurementError("G4 transition counts do not match reconstructed final errors")
    g2_aggregate = json.loads(args.g2_aggregate.read_text())
    gap_rows = _stationarity_gap(transition_counts, g2_aggregate)
    atomic_json(
        output_directory / "stage_checkpoints" / "00_sources_validated.json",
        {
            "status": "PASS",
            "pair_window": [0, N_PAIRS],
            "target_cache": {
                "path": str(args.target_cache),
                "bytes": args.target_cache.stat().st_size,
                "sha256_from_bound_receipt": (
                    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
                ),
            },
            "argmax_batch_count": len(argmax_manifests),
            "argmax_batch_chain_sha256": hashlib.sha256(
                json.dumps(argmax_manifests, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "v12_receipt": {
                "path": str(args.v12_receipt),
                "sha256": sha256_file(args.v12_receipt),
                "final_archive_bytes": reconstructed.archive_bytes,
                "final_archive_sha256": reconstructed.archive_sha256,
                "final_errors": reconstructed.final_errors,
                "admission_rows": len(reconstructed.admission_rows),
            },
            "g4_arrays_sha256": sha256_file(args.g4_arrays),
            "g2_aggregate_sha256": sha256_file(args.g2_aggregate),
            "existing_event_archive_sha256": sha256_file(args.existing_event_archive),
            "evidence_axis": AXIS,
            "score_claim": False,
        },
    )

    (
        (static_payload, static_field),
        (spline_payload, spline_mask, spline_meta),
        (curve_payload, curve_mask, curve_meta),
    ) = _load_or_fit_primitives(
        labels=labels,
        output_directory=output_directory,
        temporal_stride=args.temporal_stride,
        horizontal_stride=args.horizontal_stride,
        epsilon_pixels=args.epsilon_pixels,
        heading_bins=args.heading_bins,
    )

    ground_domain = (
        ((static_field == ROAD_ID) | (static_field == UNDRIVABLE_ID))
        & (np.arange(HEIGHT)[:, None] >= 96)
    )

    def static_pair(_pair: int, before: np.ndarray, _target: np.ndarray) -> np.ndarray:
        after = before.copy()
        after[ground_domain] = static_field[ground_domain]
        return after

    def road_mask_pair(mask_stack: np.ndarray) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
        def apply(pair: int, before: np.ndarray, _target: np.ndarray) -> np.ndarray:
            after = before.copy()
            after[mask_stack[pair]] = ROAD_ID
            return after

        return apply

    static_measurement = measure_candidate(
        candidate_id="persistent_level_set_ground_partition",
        predicted=predicted,
        labels=labels,
        transition_counts=transition_counts,
        pair_candidate=static_pair,
        counted_bytes=static_payload.counted_bytes,
        byte_scope="complete typed primitive derivation",
        baseline_errors_by_class=baseline_errors_by_class,
    )
    spline_measurement = measure_candidate(
        candidate_id="boundary_worldsheet_spline",
        predicted=predicted,
        labels=labels,
        transition_counts=transition_counts,
        pair_candidate=road_mask_pair(spline_mask),
        counted_bytes=spline_payload.counted_bytes,
        byte_scope="complete typed primitive derivation",
        baseline_errors_by_class=baseline_errors_by_class,
    )
    curve_measurement = measure_candidate(
        candidate_id="turning_angle_curve",
        predicted=predicted,
        labels=labels,
        transition_counts=transition_counts,
        pair_candidate=road_mask_pair(curve_mask),
        counted_bytes=curve_payload.counted_bytes,
        byte_scope="complete typed primitive derivation",
        baseline_errors_by_class=baseline_errors_by_class,
    )
    primitive_rows = [
        _primitive_record(
            primitive_id="persistent_level_set_ground_partition",
            derivation=static_payload,
            metadata={
                "equation": "c_static(x)=argmax_c sum_t 1[c_t(x)=c]",
                "domain": "Road-or-Undrivable modal ground cells, rows [96,384)",
            },
            measurement=static_measurement,
        ),
        _primitive_record(
            primitive_id="boundary_worldsheet_spline",
            derivation=spline_payload,
            metadata={
                **spline_meta,
                "equation": "Road(t,x,y)=1[y>=gamma(t,x)]",
            },
            measurement=spline_measurement,
        ),
        _primitive_record(
            primitive_id="turning_angle_curve",
            derivation=curve_payload,
            metadata={
                **curve_meta,
                "equation": "gamma(s+ds)=gamma(s)+ds*(cos(theta),sin(theta))",
            },
            measurement=curve_measurement,
        ),
    ]

    receiver = receive_carrier_compose_archive(
        args.existing_event_archive.read_bytes(),
        verify_member_effects=False,
    )
    if receiver.z.n_pairs != N_PAIRS:
        raise MeasurementError("existing event archive is not n600")
    event_labels = _event_label_field(receiver)

    def apply_events(after: np.ndarray, pair_index: int) -> np.ndarray:
        overlay = event_labels[pair_index]
        active = overlay >= 0
        after[active] = overlay[active]
        return after

    def event_only(pair: int, before: np.ndarray, _target: np.ndarray) -> np.ndarray:
        return apply_events(before.copy(), pair)

    existing_measurement = measure_candidate(
        candidate_id="existing_g1_and_event_vocabulary",
        predicted=predicted,
        labels=labels,
        transition_counts=transition_counts,
        pair_candidate=event_only,
        counted_bytes=args.existing_event_archive.stat().st_size,
        byte_scope="existing receiver-closed archive bytes",
        baseline_errors_by_class=baseline_errors_by_class,
    )

    variants: list[
        tuple[str, list[CodedDerivation], Callable[[int], np.ndarray]]
    ] = [
        (
            "persistent_plus_events",
            [static_payload],
            lambda _pair: static_field == ROAD_ID,
        ),
        (
            "spline_plus_events",
            [static_payload, spline_payload],
            lambda pair: spline_mask[pair],
        ),
        (
            "turning_curve_plus_events",
            [static_payload, curve_payload],
            lambda pair: curve_mask[pair],
        ),
        (
            "spline_turning_union_plus_events",
            [static_payload, spline_payload, curve_payload],
            lambda pair: curve_mask[pair] | (spline_mask[pair] & (static_field == ROAD_ID)),
        ),
    ]
    joint_rows: list[dict[str, Any]] = []
    joint_payloads: dict[str, CodedDerivation] = {}
    for candidate_id, sections, road_mask_for_pair in variants:
        joint_payload = encode_joint_ground_vocabulary(sections)
        joint_payloads[candidate_id] = joint_payload

        def joint_pair(
            pair: int,
            before: np.ndarray,
            _target: np.ndarray,
            road_mask_for_pair: Callable[[int], np.ndarray] = road_mask_for_pair,
        ) -> np.ndarray:
            after = before.copy()
            road_mask = road_mask_for_pair(pair)
            after[ground_domain & road_mask] = ROAD_ID
            after[ground_domain & ~road_mask] = UNDRIVABLE_ID
            return apply_events(after, pair)

        total_counted = args.existing_event_archive.stat().st_size + joint_payload.counted_bytes
        measurement = measure_candidate(
            candidate_id=candidate_id,
            predicted=predicted,
            labels=labels,
            transition_counts=transition_counts,
            pair_candidate=joint_pair,
            counted_bytes=total_counted,
            byte_scope=(
                "existing archive exact bytes plus one actually coded joint vocabulary section; "
                "final container overhead awaits the parallel packager"
            ),
            baseline_errors_by_class=baseline_errors_by_class,
        )
        measurement["inside_c1_byte_box"] = total_counted <= C1_BYTE_BOX
        measurement["new_joint_section_bytes"] = joint_payload.counted_bytes
        measurement["new_joint_section_sha256"] = hashlib.sha256(
            joint_payload.envelope
        ).hexdigest()
        joint_rows.append(measurement)

    eligible = [row for row in joint_rows if row["inside_c1_byte_box"]]
    if not eligible:
        raise MeasurementError("no enriched joint composition fits the c1 byte box")
    selected = min(
        eligible,
        key=lambda row: (
            int(row["candidate_errors"]),
            int(row["counted_bytes"]),
            str(row["candidate_id"]),
        ),
    )
    selected_payload = joint_payloads[str(selected["candidate_id"])]
    selected_path = output_directory / "payloads" / "selected_joint_ground_vocabulary.dv1"
    atomic_bytes(selected_path, selected_payload.envelope)

    enriched_table = [
        {
            "stratum": "Road",
            "candidate": "m5r_translation_baseline",
            "counted_bytes": M5R_COUNTED_BYTES,
            "road_reference_errors": M5R_CONTROL_ROAD_ERRORS,
            "road_described_fraction": M5R_ROAD_DESCRIBED_FRACTION,
            "axis": "receiver-realized m5r control",
            "verdict_scope": (
                "Exact selected m5r c1 instance; zero credit because the selected state "
                "was not admitted at full n600."
            ),
        },
        {
            "stratum": "Movable",
            "candidate": "g1_track_shape_abs_eps1",
            "counted_bytes": 29_810,
            "mask_errors": 33_378,
            "axis": "G1 semantic-mask reference",
            "verdict_scope": "Prior measured G1 grammar row; not remeasured here.",
        },
        {
            "stratum": "Lane",
            "candidate": "g1_lane_slots_delta_dash_tolx2",
            "counted_bytes": 27_692,
            "mask_errors": 583_417,
            "axis": "G1 semantic-mask reference",
            "verdict_scope": "Prior measured G1 grammar row; not remeasured here.",
        },
        *[
            {
                "stratum": "Road" if row["primitive_id"] != "persistent_level_set_ground_partition" else "Road+Undrivable",
                "candidate": row["primitive_id"],
                "counted_bytes": row["counted_bytes"],
                "road_reference_errors": baseline_errors_by_class["Road"],
                "road_errors_described": row["measurement"]["road_errors_described"],
                "road_described_fraction": row["measurement"]["road_described_fraction"],
                "net_errors_closed_all_strata": row["measurement"]["net_errors_closed"],
                "axis": "exact cached semantic cells",
                "verdict_scope": row["verdict_scope"],
            }
            for row in primitive_rows
        ],
        {
            "stratum": "joint",
            "candidate": selected["candidate_id"],
            "counted_bytes": selected["counted_bytes"],
            "road_reference_errors": baseline_errors_by_class["Road"],
            "road_errors_described": selected["road_errors_described"],
            "road_described_fraction": selected["road_described_fraction"],
            "net_errors_closed_all_strata": selected["net_errors_closed"],
            "inside_c1_byte_box": selected["inside_c1_byte_box"],
            "axis": "exact cached semantic cells, joint non-additive composition",
            "verdict_scope": selected["verdict_scope"],
        },
    ]

    road_fraction = float(selected["road_described_fraction"])
    verdict = (
        "ROAD_DESCRIPTION_REACH_CONFIRMED_IN_CELL_SPACE_RECEIVER_REALIZATION_OWED"
        if road_fraction > 0.0
        else "ROAD_DESCRIPTION_REACH_NOT_CONFIRMED_FOR_THIS_VOCABULARY"
    )
    summary = {
        "schema": "ddm_description_vocabulary_summary.v1",
        "run_id": output_directory.name,
        "evidence_axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "baseline_errors_by_class": baseline_errors_by_class,
        "vocabulary_gap": gap_rows,
        "primitive_measurements": primitive_rows,
        "existing_vocabulary_joint_component": existing_measurement,
        "joint_compositions": joint_rows,
        "selected_joint_composition": selected,
        "enriched_mdl_table": enriched_table,
        "road_at_c1": {
            "byte_box": C1_BYTE_BOX,
            "selected_counted_bytes": selected["counted_bytes"],
            "reference_axis": "v12 preserved frozen-scorer argmax cells",
            "reference_errors": baseline_errors_by_class["Road"],
            "errors_described": selected["road_errors_described"],
            "described_fraction": road_fraction,
            "m5r_comparator": {
                "reference_errors": M5R_CONTROL_ROAD_ERRORS,
                "counted_bytes": M5R_COUNTED_BYTES,
                "admitted_described_fraction": M5R_ROAD_DESCRIBED_FRACTION,
            },
            "cross_baseline_warning": (
                "The new semantic reach denominator is the v12 preserved cell cache; m5r's "
                "receiver-realized denominator is shown as a comparator and is not silently rebased."
            ),
        },
        "canonical_equations": {
            "reach": (
                "R_c(G)=sum_i 1[p_i!=t_i and t_i=c and D_G(i)=c] / "
                "sum_i 1[p_i!=t_i and t_i=c]"
            ),
            "net_closure": (
                "DeltaE(G)=sum_i 1[p_i!=t_i,D_G(i)=t_i] - "
                "sum_i 1[p_i=t_i,D_G(i)!=t_i]"
            ),
            "joint_length": (
                "L_joint=len(C(G_static,G_boundary,G_curve)); individual coded lengths "
                "are never used as a surrogate for the jointly coded section."
            ),
            "level_set_energy": (
                "E(phi)=integral alpha|grad phi| + beta*kappa(phi)^2 + "
                "lambda*I[class(phi)!=target]; primitives expose the persistent phase, "
                "separatrix worldsheet, and arc-length/turn coordinates."
            ),
        },
        "verdict": verdict,
        "verdict_scope": (
            "Full n600 semantic description reach under actual real-coded derivation strings "
            "and exact cached argmax cells. The negative/positive verdict does not extend to "
            "RGB receiver survival, Pose, exact archive packaging, or contest score."
        ),
        "main_landing_review_required": True,
    }
    atomic_json(output_directory / "summary.json", summary)
    ledger_rows = [
        {
            "schema": "ddm_description_vocabulary_ledger.v1",
            "record_type": "gap",
            "record_id": row["stratum"],
            "payload": row,
            "evidence_axis": AXIS,
            "score_claim": False,
        }
        for row in gap_rows
    ] + [
        {
            "schema": "ddm_description_vocabulary_ledger.v1",
            "record_type": "primitive",
            "record_id": row["primitive_id"],
            "payload": row,
            "evidence_axis": AXIS,
            "score_claim": False,
        }
        for row in primitive_rows
    ] + [
        {
            "schema": "ddm_description_vocabulary_ledger.v1",
            "record_type": "joint",
            "record_id": row["candidate_id"],
            "payload": row,
            "evidence_axis": AXIS,
            "score_claim": False,
        }
        for row in joint_rows
    ]
    atomic_bytes(
        output_directory / "description_vocabulary_ledger.jsonl",
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in ledger_rows
        ),
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema": "ddm_description_vocabulary_receipt.v1",
        "run_id": output_directory.name,
        "implementation_head_at_measurement": head,
        "implementation_custody": {
            "source_files": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    REPO_ROOT / "src/tac/optimization/ddm_description_vocabulary.py",
                    REPO_ROOT / "tools/measure_ddm_description_vocabulary.py",
                    REPO_ROOT / "tests/test_ddm_description_vocabulary.py",
                )
            ]
        },
        "semantic_argv": sys.argv,
        "outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                output_directory / "summary.json",
                output_directory / "description_vocabulary_ledger.jsonl",
                selected_path,
            )
        ],
        "source_custody": {
            "target_cache": str(args.target_cache),
            "v12_receipt": {
                "path": str(args.v12_receipt),
                "sha256": sha256_file(args.v12_receipt),
                "final_archive_bytes": reconstructed.archive_bytes,
                "final_archive_sha256": reconstructed.archive_sha256,
                "final_errors": reconstructed.final_errors,
            },
            "argmax_batch_chain": argmax_manifests,
            "g4_arrays": {
                "path": str(args.g4_arrays),
                "sha256": sha256_file(args.g4_arrays),
            },
            "g2_aggregate": {
                "path": str(args.g2_aggregate),
                "sha256": sha256_file(args.g2_aggregate),
            },
            "existing_event_archive": {
                "path": str(args.existing_event_archive),
                "bytes": args.existing_event_archive.stat().st_size,
                "sha256": sha256_file(args.existing_event_archive),
            },
        },
        "evidence_axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "verdict": verdict,
        "verdict_scope": summary["verdict_scope"],
        "main_landing_review_required": True,
    }
    atomic_json(output_directory / "receipt.json", receipt)
    print(json.dumps({"verdict": verdict, "selected": selected}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
