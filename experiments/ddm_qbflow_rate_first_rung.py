# SPDX-License-Identifier: MIT
"""Build and measure the scorer-free QBFLOW initialized rate-first rung.

The runner is intentionally limited to stages 00--02: preregistered selection,
initialized-object serialization, receiver/refusal closure, and the byte gate.
It never imports or dispatches a scorer, Metal, Modal, or a training runtime.
Every materialized model, latent, coder candidate, packet, archive, mutation,
and receiver output is retained below the charter-mandated APDataStore root.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ddm_qbflow_packet as packet
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARM = "ddm_qbflow_rate_first_rung"
STORE = Path("/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow")
SOURCE_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/retained/fields/"
    "decoded_tokens_instrumented.u8"
)
SOURCE_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
SPEC = REPO / ".omx/research/SPEC_ddm_qbflow_packet_schema_v1_20260827.md"
SPEC_SHA256 = "5405ccd499d14d28230874059e47d47f1f2818038519f1b27c97ed9377f132aa"
PACKET_MODULE = REPO / "experiments/ddm_qbflow_packet.py"
PACKET_MODULE_SHA256 = "cdf90d1a4d7d13001118f50a76692c04605f8e5ae9a7816c80f6e346160c7b9c"

N, H, W = 600, 384, 512
SEED = 20260827
COMPLETE_CAP_BYTES = 137_986
EXPLICIT_QBW2_FLOOR_BYTES = 188_860
SOURCE_INTERFACE_COUNT = 1_625_624
DSEG_ADMISSION_CEILING_AT_101150 = 0.00044667138915998396
OWN_VEHICLE_SCORE = 0.14811799921260607
OWN_VEHICLE_BYTES = 180_215
OWN_VEHICLE_SHA256 = "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"
RESERVE_BYTES = 8 * 1024**3
WORK_BYTES = 1 * 1024**3
EXPECTED_SELECTED_IDS = [
    4,
    31,
    49,
    52,
    62,
    90,
    100,
    113,
    128,
    148,
    173,
    179,
    186,
    187,
    214,
    236,
    256,
    260,
    268,
    278,
    326,
    328,
    341,
    352,
    368,
    382,
    444,
    456,
    483,
    508,
    563,
    573,
]


class QBFLOWBuildError(RuntimeError):
    """Fail-closed build or custody error."""


def canonical_json_bytes(value: Any) -> bytes:
    return packet.canonical_json_bytes(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise QBFLOWBuildError(f"resume payload drift: {path}")
        return file_fact(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_json_once(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes_once(path, canonical_json_bytes(value))


def atomic_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array_payload = io.BytesIO()
            np.lib.format.write_array(array_payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                array_payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return atomic_bytes_once(path, payload.getvalue())


def load_checkpoint(path: Path, schema: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("schema") != schema or value.get("complete") is not True:
        raise QBFLOWBuildError(f"invalid checkpoint: {path}")
    return value


def git_head(path: Path = REPO) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def storage_preflight() -> dict[str, Any]:
    rows = []
    for root in (
        Path("/Volumes/APDataStore/pact"),
        Path("/Volumes/VertigoDataTier/pact"),
        REPO,
    ):
        usage = shutil.disk_usage(root)
        rows.append(
            {
                "path": str(root),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    required = RESERVE_BYTES + WORK_BYTES
    if rows[0]["free_bytes"] < required:
        raise QBFLOWBuildError(f"APDataStore preflight refused: free={rows[0]['free_bytes']} required={required}")
    STORE.mkdir(parents=True, exist_ok=True)
    probe = STORE / f".write_probe.{os.getpid()}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"qbflow-write-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "schema": "ddm_qbflow_storage_preflight.v1",
        "selected_root": str(STORE),
        "waterfall": rows,
        "reserve_bytes": RESERVE_BYTES,
        "work_bytes": WORK_BYTES,
        "required_free_bytes": required,
        "pass": True,
    }


def source_field() -> np.memmap:
    if not SOURCE_FIELD.exists() or SOURCE_FIELD.stat().st_size != N * H * W:
        raise QBFLOWBuildError("source field missing or wrong size")
    if sha256_file(SOURCE_FIELD) != SOURCE_SHA256:
        raise QBFLOWBuildError("source field SHA-256 drifted")
    return np.memmap(SOURCE_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))


def road_lane_crack_count(field: np.ndarray) -> int:
    right_a = field[:, :-1]
    right_b = field[:, 1:]
    down_a = field[:-1, :]
    down_b = field[1:, :]
    right = ((right_a == 0) & (right_b == 1)) | ((right_a == 1) & (right_b == 0))
    down = ((down_a == 0) & (down_b == 1)) | ((down_a == 1) & (down_b == 0))
    return int(right.sum(dtype=np.int64) + down.sum(dtype=np.int64))


def all_interface_count(field: np.ndarray) -> int:
    right = field[:, :-1] != field[:, 1:]
    down = field[:-1, :] != field[1:, :]
    return int(right.sum(dtype=np.int64) + down.sum(dtype=np.int64))


def selection_rows(
    field: np.memmap,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    crack_counts = []
    total_interfaces = 0
    for pair_id in range(N):
        pair = np.asarray(field[pair_id])
        crack_counts.append(road_lane_crack_count(pair))
        total_interfaces += all_interface_count(pair)
    if total_interfaces != SOURCE_INTERFACE_COUNT:
        raise QBFLOWBuildError(f"source interface count drifted: {total_interfaces} != {SOURCE_INTERFACE_COUNT}")
    rng = np.random.Generator(np.random.PCG64(SEED))
    strata: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for block in range(10):
        members = list(range(block * 60, (block + 1) * 60))
        ordered = sorted(members, key=lambda pair: (crack_counts[pair], pair))
        take = 2 if block <= 5 else 1
        for half, population in (("low", ordered[:30]), ("high", ordered[30:])):
            chosen = sorted(int(value) for value in rng.choice(population, size=take, replace=False))
            stratum_id = f"block_{block:02d}_{half}"
            strata.append(
                {
                    "stratum_id": stratum_id,
                    "temporal_block": block,
                    "crack_half": half,
                    "population_size": len(population),
                    "sample_size": take,
                    "inclusion_probability": take / len(population),
                    "population_pair_ids": population,
                    "population_crack_count_min": min(crack_counts[pair] for pair in population),
                    "population_crack_count_max": max(crack_counts[pair] for pair in population),
                    "selected_pair_ids": chosen,
                }
            )
            for pair_id in chosen:
                selected.append(
                    {
                        "pair_id": pair_id,
                        "stratum_id": stratum_id,
                        "population_size": len(population),
                        "sample_size": take,
                        "inclusion_probability": take / len(population),
                        "road_lane_crack_count": crack_counts[pair_id],
                        "all_interface_count": all_interface_count(np.asarray(field[pair_id])),
                    }
                )
    selected.sort(key=lambda row: int(row["pair_id"]))
    actual_ids = [int(row["pair_id"]) for row in selected]
    if actual_ids != EXPECTED_SELECTED_IDS:
        raise QBFLOWBuildError(f"seeded selection drifted: {actual_ids}")
    return strata, selected, total_interfaces


def capacity_derivation() -> dict[str, Any]:
    parameter_shapes = packet.expected_param_shapes()
    count = sum(math.prod(shape) for shape in parameter_shapes.values())
    if count != 79_513:
        raise QBFLOWBuildError(f"architecture parameter count drifted: {count}")
    error_cells = N * H * W * DSEG_ADMISSION_CEILING_AT_101150
    return {
        "schema": "ddm_qbflow_capacity_derivation.v1",
        "complete_cap_bytes": COMPLETE_CAP_BYTES,
        "explicit_qbw2_floor_bytes": EXPLICIT_QBW2_FLOOR_BYTES,
        "source_cells": N * H * W,
        "source_interface_count": SOURCE_INTERFACE_COUNT,
        "interfaces_per_pair": SOURCE_INTERFACE_COUNT / N,
        "dseg_ceiling_at_101150_current_pose": DSEG_ADMISSION_CEILING_AT_101150,
        "cell_error_ceiling": error_cells,
        "cell_error_ceiling_per_pair": error_cells / N,
        "cell_error_ceiling_over_interfaces": error_cells / SOURCE_INTERFACE_COUNT,
        "quotient_allowance_bytes_per_interface": 84_910 / SOURCE_INTERFACE_COUNT,
        "all_class_interface_outputs": packet.N_INTERFACES,
        "along_tangent_frequencies": [8, 16, 24, 32],
        "flow_width_derivation": {
            "interface_frequency_phase_dof": 10 * 4 * 2,
            "boundary_conditioning_dof": packet.BOUNDARY_FEATURE_DIM,
            "flow_width": packet.FLOW_DIM,
        },
        "learned_parameter_count": count,
        "precision_scalar_counts": None,
        "status": "DERIVED_CAPACITY_PORTRAIT_NOT_DISTORTION_MEASUREMENT",
    }


def stage_00() -> dict[str, Any]:
    checkpoint = STORE / "stage_00_selection" / "STAGE_00_CHECKPOINT.json"
    resumed = load_checkpoint(checkpoint, "ddm_qbflow_stage_00_selection.v1")
    if resumed is not None:
        return resumed
    preflight = storage_preflight()
    if sha256_file(SPEC) != SPEC_SHA256:
        raise QBFLOWBuildError("frozen packet schema SHA-256 drifted")
    if sha256_file(PACKET_MODULE) != PACKET_MODULE_SHA256:
        raise QBFLOWBuildError("frozen packet module SHA-256 drifted")
    field = source_field()
    strata, selected, total_interfaces = selection_rows(field)
    capacity = capacity_derivation()
    architecture = packet.architecture_config(num_pairs=N, seed=SEED)
    config = {
        "schema": "ddm_qbflow_run_config.v1",
        "arm": ARM,
        "seed": SEED,
        "selection_mode": "PCG64_seeded_temporal_block_x_road_lane_crack_half_stratified_n32",
        "source_field": file_fact(SOURCE_FIELD),
        "packet_schema": file_fact(SPEC),
        "packet_module": file_fact(PACKET_MODULE),
        "architecture": architecture,
        "axis": "[macOS-CPU scorer-free advisory, initialized-untrained]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    config_fact = atomic_json_once(STORE / "CONFIG.json", config)
    receipt = {
        "schema": "ddm_qbflow_stage_00_selection.v1",
        "complete": True,
        "stage": "stage_00_selection",
        "storage_preflight": preflight,
        "source_field": file_fact(SOURCE_FIELD),
        "source_interface_count": total_interfaces,
        "seed": SEED,
        "strata": strata,
        "selected": selected,
        "capacity_derivation": capacity,
        "config": config_fact,
        "git_head_at_stage": git_head(),
    }
    atomic_json_once(checkpoint, receipt)
    return receipt


def retain_section_race(root: Path, section_id: int, raw: bytes) -> tuple[packet.EncodedSection, dict[str, Any]]:
    raw_fact = atomic_bytes_once(root / "raw.bin", raw)
    candidates = packet.encode_section_candidates(section_id, raw)
    rows = []
    for codec_name in sorted(candidates):
        candidate = candidates[codec_name]
        primary = atomic_bytes_once(root / f"payload.{codec_name}", candidate.payload)
        repeat_payload = packet.compress(codec_name, raw)
        repeat = atomic_bytes_once(root / f"payload.repeat.{codec_name}", repeat_payload)
        if primary["sha256"] != repeat["sha256"]:
            raise QBFLOWBuildError(f"section coder repeat drift: {root} {codec_name}")
        if packet.decompress(codec_name, candidate.payload) != raw:
            raise QBFLOWBuildError(f"section coder decode drift: {root} {codec_name}")
        rows.append(
            {
                "codec": codec_name,
                "payload": primary,
                "repeat": repeat,
            }
        )
    winner = packet.choose_section(candidates)
    result = {
        "schema": "ddm_qbflow_section_coder_race.v1",
        "section_id": section_id,
        "section_name": packet.SECTION_NAMES[section_id],
        "raw": raw_fact,
        "candidates": rows,
        "winner": {
            "codec": winner.codec_name,
            "payload_bytes": len(winner.payload),
            "payload_sha256": packet.sha256_bytes(winner.payload),
        },
    }
    atomic_json_once(root / "RACE.json", result)
    return winner, result


def retain_reset_record_race(root: Path, pair_id: int, raw_record: bytes) -> tuple[bytes, dict[str, Any]]:
    raw_fact = atomic_bytes_once(root / "record.raw.qbr", raw_record)
    candidates = packet.encode_reset_record(raw_record)
    rows = []
    for codec_name in sorted(candidates):
        primary = atomic_bytes_once(root / f"record.{codec_name}.qbrc", candidates[codec_name])
        repeat_payload = packet.encode_reset_record(raw_record)[codec_name]
        repeat = atomic_bytes_once(root / f"record.repeat.{codec_name}.qbrc", repeat_payload)
        if primary["sha256"] != repeat["sha256"]:
            raise QBFLOWBuildError(f"reset coder repeat drift for pair {pair_id}")
        if packet.decode_reset_record(candidates[codec_name]) != raw_record:
            raise QBFLOWBuildError(f"reset coder decode drift for pair {pair_id}")
        rows.append(
            {
                "codec": codec_name,
                "payload": primary,
                "repeat": repeat,
            }
        )
    winner_name, winner_payload = min(candidates.items(), key=lambda row: (len(row[1]), packet.CODEC_IDS[row[0]]))
    result = {
        "schema": "ddm_qbflow_reset_record_race.v1",
        "pair_id": pair_id,
        "raw": raw_fact,
        "candidates": rows,
        "winner": {
            "codec": winner_name,
            "bytes": len(winner_payload),
            "sha256": packet.sha256_bytes(winner_payload),
        },
    }
    atomic_json_once(root / "RACE.json", result)
    return winner_payload, result


def dequantized_latent(
    meta: Mapping[str, Any], boundary_codes: np.ndarray, interior_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.asarray(boundary_codes, dtype=np.float32) * np.float32(meta["boundary_scale"])
    interior = np.asarray(interior_codes, dtype=np.float32) * np.float32(meta["interior_scale"])
    return boundary, interior


def stage_01_02() -> dict[str, Any]:
    checkpoint = STORE / "stage_02_gate" / "STAGE_02_CHECKPOINT.json"
    resumed = load_checkpoint(checkpoint, "ddm_qbflow_stage_02_gate.v1")
    if resumed is not None:
        return resumed
    stage0 = stage_00()
    params_float = packet.initialize_params(SEED)
    boundary_float, interior_float = packet.initialize_latents(SEED, N)
    initialized_root = STORE / "stage_01_initialize_quantize"
    float_params_fact = atomic_npz_once(initialized_root / "initialized_float_params.npz", params_float)
    float_latents_fact = atomic_npz_once(
        initialized_root / "initialized_float_latents.npz",
        {"boundary": boundary_float, "interior": interior_float},
    )
    diagnostic_recovery_fact = atomic_json_once(
        initialized_root / "PRE_RUN_DEVELOPER_DIAGNOSTIC_RECOVERY.json",
        {
            "schema": "ddm_qbflow_pre_run_diagnostic_recovery.v1",
            "disposition": "RECOVERED_DETERMINISTIC_RECREATION_NO_MEASUREMENT",
            "incident": (
                "Before the schema freeze, a developer-only parameter-count diagnostic "
                "instantiated the seeded generic initialization without persisting it."
            ),
            "scientific_measurement_used": False,
            "rate_result_used": False,
            "recovery": {
                "seed": SEED,
                "packet_module": file_fact(PACKET_MODULE),
                "retained_initialized_float_params": float_params_fact,
                "retained_initialized_float_latents": float_latents_fact,
            },
        },
    )

    config_raw = canonical_json_bytes(packet.architecture_config(num_pairs=N, seed=SEED))
    model_raw = packet.encode_model(params_float)
    latent_meta_raw, boundary_codes, interior_codes = packet.encode_latent_meta(boundary_float, interior_float)
    full_latents_raw = packet.encode_latent_table(range(N), boundary_codes, interior_codes)
    selected_ids = [int(row["pair_id"]) for row in stage0["selected"]]
    selected_latents_raw = packet.encode_latent_table(selected_ids, boundary_codes, interior_codes)
    raw_facts = {
        "config": atomic_bytes_once(initialized_root / "config.raw.json", config_raw),
        "model": atomic_bytes_once(initialized_root / "model.raw.qbt", model_raw),
        "latent_meta": atomic_bytes_once(initialized_root / "latent_meta.raw.qbm", latent_meta_raw),
        "latents_n600": atomic_bytes_once(initialized_root / "latents_n600.raw.qbl", full_latents_raw),
        "latents_n32": atomic_bytes_once(initialized_root / "latents_n32.raw.qbl", selected_latents_raw),
    }

    precision_counts: dict[int, int] = {8: 0, 10: 0, 12: 0, 16: 0}
    for name, value in params_float.items():
        precision_counts[packet.precision_bits(name)] += int(value.size)
    model_params = packet.decode_model(model_raw)
    atomic_npz_once(initialized_root / "decoded_quantized_params.npz", model_params)
    latent_meta = packet.decode_latent_meta(latent_meta_raw)
    decoded_full_latents = packet.decode_latent_table(full_latents_raw)
    if sorted(decoded_full_latents) != list(range(N)):
        raise QBFLOWBuildError("full latent parse-back pair set mismatch")

    race_root = STORE / "stage_01_initialize_quantize" / "section_races"
    config_section, config_race = retain_section_race(race_root / "config", packet.SECTION_CONFIG, config_raw)
    model_section, model_race = retain_section_race(race_root / "model", packet.SECTION_MODEL, model_raw)
    meta_section, meta_race = retain_section_race(
        race_root / "latent_meta", packet.SECTION_LATENT_META, latent_meta_raw
    )
    full_latent_section, full_latent_race = retain_section_race(
        race_root / "latents_n600", packet.SECTION_LATENTS, full_latents_raw
    )
    n32_latent_section, n32_latent_race = retain_section_race(
        race_root / "latents_n32", packet.SECTION_LATENTS, selected_latents_raw
    )

    reset_rows = []
    selected_by_pair = {int(row["pair_id"]): row for row in stage0["selected"]}
    for pair_id in selected_ids:
        raw_record = packet.encode_latent_record(pair_id, boundary_codes[pair_id], interior_codes[pair_id])
        _winner_payload, race = retain_reset_record_race(
            STORE / "stage_01_initialize_quantize" / "reset_records" / f"pair_{pair_id:04d}",
            pair_id,
            raw_record,
        )
        selection = selected_by_pair[pair_id]
        reset_rows.append(
            {
                "pair_id": pair_id,
                "stratum_id": selection["stratum_id"],
                "population_size": selection["population_size"],
                "sample_size": selection["sample_size"],
                "ht_weight": selection["population_size"] / selection["sample_size"],
                "reset_record_bytes": race["winner"]["bytes"],
                "reset_record_sha256": race["winner"]["sha256"],
                "reset_record_codec": race["winner"]["codec"],
            }
        )

    shared_packet = packet.pack_packet([config_section, model_section, meta_section])
    n32_packet = packet.pack_packet([config_section, model_section, meta_section, n32_latent_section])
    full_packet = packet.pack_packet([config_section, model_section, meta_section, full_latent_section])
    packet_root = STORE / "stage_02_gate" / "retained"
    shared_fact = atomic_bytes_once(packet_root / "shared.qbf", shared_packet)
    n32_fact = atomic_bytes_once(packet_root / "n32.qbf", n32_packet)
    n32_repeat = atomic_bytes_once(packet_root / "n32.repeat.qbf", n32_packet)
    full_fact = atomic_bytes_once(packet_root / "full_n600.qbf", full_packet)
    full_repeat = atomic_bytes_once(packet_root / "full_n600.repeat.qbf", full_packet)
    if n32_fact["sha256"] != n32_repeat["sha256"]:
        raise QBFLOWBuildError("n32 packet repeat drift")
    if full_fact["sha256"] != full_repeat["sha256"]:
        raise QBFLOWBuildError("full packet repeat drift")

    shared_archive = packet.deterministic_archive(shared_packet)
    n32_archive = packet.deterministic_archive(n32_packet)
    full_archive = packet.deterministic_archive(full_packet)
    shared_archive_fact = atomic_bytes_once(packet_root / "shared.archive.zip", shared_archive)
    n32_archive_fact = atomic_bytes_once(packet_root / "n32.archive.zip", n32_archive)
    n32_archive_repeat = atomic_bytes_once(packet_root / "n32.archive.repeat.zip", n32_archive)
    full_archive_fact = atomic_bytes_once(packet_root / "archive.zip", full_archive)
    full_archive_repeat = atomic_bytes_once(packet_root / "archive.repeat.zip", full_archive)
    if n32_archive_fact["sha256"] != n32_archive_repeat["sha256"]:
        raise QBFLOWBuildError("n32 archive repeat drift")
    if full_archive_fact["sha256"] != full_archive_repeat["sha256"]:
        raise QBFLOWBuildError("full archive repeat drift")
    if packet.read_deterministic_archive(full_archive) != full_packet:
        raise QBFLOWBuildError("archive parse-back packet mismatch")

    decoded_packet = packet.decode_packet(full_packet)
    if set(decoded_packet.sections) != {
        packet.SECTION_CONFIG,
        packet.SECTION_MODEL,
        packet.SECTION_LATENT_META,
        packet.SECTION_LATENTS,
    }:
        raise QBFLOWBuildError("full packet section set mismatch")
    if decoded_packet.sections[packet.SECTION_MODEL] != model_raw:
        raise QBFLOWBuildError("model section parse-back mismatch")
    if decoded_packet.sections[packet.SECTION_LATENTS] != full_latents_raw:
        raise QBFLOWBuildError("latent section parse-back mismatch")

    mutation_rows = []
    for section_id in sorted(decoded_packet.sections):
        mutated = packet.mutate_counted_section(full_packet, section_id)
        mutation_fact = atomic_bytes_once(packet_root / "mutations" / f"section_{section_id}.mutated.qbf", mutated)
        try:
            packet.decode_packet(mutated)
        except packet.QBFLOWPacketError as exc:
            outcome = f"REFUSED:{type(exc).__name__}:{exc}"
        else:
            raise QBFLOWBuildError(f"section mutation was accepted: {section_id}")
        mutation_rows.append(
            {
                "section_id": section_id,
                "section_name": packet.SECTION_NAMES[section_id],
                "mutated_payload": mutation_fact,
                "outcome": outcome,
            }
        )

    forward_rows = []
    output_root = STORE / "stage_02_gate" / "receiver_outputs"
    for pair_id in selected_ids:
        boundary_decoded, interior_decoded = decoded_full_latents[pair_id]
        boundary_value, interior_value = dequantized_latent(latent_meta, boundary_decoded, interior_decoded)
        outputs = packet.reference_forward(
            model_params,
            boundary_value,
            interior_value,
            pair_id=pair_id,
            num_pairs=N,
            height=16,
            width=16,
        )
        output_fact = atomic_npz_once(output_root / f"pair_{pair_id:04d}_forward.npz", outputs)
        forward_rows.append(
            {
                "pair_id": pair_id,
                "output": output_fact,
                "signed_interface_shape": list(outputs["signed_interfaces"].shape),
                "class_logits_shape": list(outputs["class_logits"].shape),
                "rgb_pair_shape": list(outputs["rgb_pair"].shape),
                "pose12_shape": list(outputs["pose12"].shape),
            }
        )

    b_var_hat = sum(float(row["ht_weight"]) * int(row["reset_record_bytes"]) for row in reset_rows)
    b_hat = shared_archive_fact["bytes"] + math.ceil(b_var_hat)
    exact_full_bytes = full_archive_fact["bytes"]
    gate_clear = b_hat <= COMPLETE_CAP_BYTES and exact_full_bytes <= COMPLETE_CAP_BYTES
    disposition = "RATE_SHAPE_EXISTS_INITIALIZED_UNTRAINED" if gate_clear else "CLOSED_FIRST_RUNG_RATE_FORMULATION"
    result = {
        "schema": "ddm_qbflow_stage_02_gate.v1",
        "complete": True,
        "stage": "stage_02_gate",
        "arm": ARM,
        "axis": "[macOS-CPU scorer-free advisory, initialized-untrained]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "distortion_measured": False,
        "training_launched": False,
        "scorer_dispatched": False,
        "metal_dispatched": False,
        "modal_dispatched": False,
        "selection": {
            "seed": SEED,
            "selected_pair_ids": selected_ids,
            "strata": stage0["strata"],
        },
        "capacity_derivation": stage0["capacity_derivation"],
        "parameter_count": packet.parameter_count(params_float),
        "precision_scalar_counts": precision_counts,
        "retained_initialized_float_params": float_params_fact,
        "retained_initialized_float_latents": float_latents_fact,
        "pre_run_developer_diagnostic_recovery": diagnostic_recovery_fact,
        "retained_raw_payloads": raw_facts,
        "section_races": {
            "config": config_race,
            "model": model_race,
            "latent_meta": meta_race,
            "latents_n32": n32_latent_race,
            "latents_n600": full_latent_race,
        },
        "section_parseback": list(decoded_packet.section_facts),
        "reset_record_rows": reset_rows,
        "shared_packet": shared_fact,
        "n32_packet": n32_fact,
        "full_n600_packet": full_fact,
        "shared_archive": shared_archive_fact,
        "n32_archive": n32_archive_fact,
        "full_n600_archive": full_archive_fact,
        "b_var_hat": b_var_hat,
        "b_shared_archive": shared_archive_fact["bytes"],
        "b_hat_projected_complete_archive": b_hat,
        "exact_full_n600_archive_bytes": exact_full_bytes,
        "complete_cap_bytes": COMPLETE_CAP_BYTES,
        "projected_headroom_bytes": COMPLETE_CAP_BYTES - b_hat,
        "exact_full_headroom_bytes": COMPLETE_CAP_BYTES - exact_full_bytes,
        "mutation_rows": mutation_rows,
        "receiver_forward_rows": forward_rows,
        "parseback_pass": True,
        "repeat_identity_pass": True,
        "mutation_refusal_pass": True,
        "receiver_forward_pass": True,
        "gate_clear": gate_clear,
        "disposition": disposition,
        "boundaries": {
            "measured": (
                "initialized quantized tensor/latent bytes, real coder outputs, n32 HT reset "
                "projection, exact initialized n600 archive bytes, parse-back, repeat, mutations"
            ),
            "not_measured": (
                "training, RGB through R, SegNet, PoseNet, d_seg, d_pose, S, inflate runtime, contest CPU/CUDA"
            ),
        },
        "git_head_at_stage": git_head(),
        "upstream": {
            "git_head": git_head(REPO / "upstream"),
            "evaluate_sha256": sha256_file(REPO / "upstream/evaluate.py"),
        },
        "own_vehicle_frontier": {
            "score": OWN_VEHICLE_SCORE,
            "archive_bytes": OWN_VEHICLE_BYTES,
            "archive_sha256": OWN_VEHICLE_SHA256,
            "axis": "[contest-CUDA T4 n600]",
            "moved_by_this_arm": False,
        },
    }
    atomic_json_once(STORE / "RESULT.json", result)
    atomic_json_once(checkpoint, result)
    write_query_stores(result)
    write_fire_order(result)
    return result


def write_query_stores(result: Mapping[str, Any]) -> None:
    root = STORE / "query_stores"
    reset_lines = b"".join(canonical_json_bytes(row) for row in result["reset_record_rows"])
    mutation_lines = b"".join(canonical_json_bytes(row) for row in result["mutation_rows"])
    forward_lines = b"".join(canonical_json_bytes(row) for row in result["receiver_forward_rows"])
    atomic_bytes_once(root / "reset_record_facts.jsonl", reset_lines)
    atomic_bytes_once(root / "mutation_facts.jsonl", mutation_lines)
    atomic_bytes_once(root / "receiver_forward_facts.jsonl", forward_lines)


def write_fire_order(result: Mapping[str, Any]) -> dict[str, Any]:
    clear = bool(result["gate_clear"])
    disposition = "QUEUED-WITH-A-FIRE-ORDER" if clear else "FOLDED_BY_RATE_GATE"
    fire_order = {
        "schema": "ddm_qbflow_training_fire_order.v1",
        "disposition": disposition,
        "owner": "MAIN QBFLOW joint-training owner",
        "consumer_store": str(STORE),
        "candidate_archive": result["full_n600_archive"],
        "rate_gate": {
            "clear": clear,
            "projected_complete_bytes": result["b_hat_projected_complete_archive"],
            "exact_initialized_full_bytes": result["exact_full_n600_archive_bytes"],
            "cap_bytes": COMPLETE_CAP_BYTES,
        },
        "fire_trigger": (
            "MAIN consumes the committed rate verdict, confirms no duplicate active lane and no "
            "full-n600 scorer job, lands/reviews a real QBFLOW scorer-in-loop trainer consuming "
            "this exact packet ABI, and passes a live <=116 GiB memory/storage preflight"
            if clear
            else "none; rate gate refused the initialized object"
        ),
        "stages": [
            {
                "stage": "stage_03_joint_boundary_interior_birth",
                "pairs_per_chunk_max": 30,
                "objective": (
                    "joint realized-through-R Seg interface/RGB descent plus pose6 descent; "
                    "no fixed paint and no post-hoc pose"
                ),
                "checkpoint": "distinct atomic EMA checkpoint at birth end plus periodic saves",
            },
            {
                "stage": "stage_04_precision_waterfill_and_byteclose",
                "pairs_per_chunk_max": 30,
                "objective": (
                    "measure per-role receiver/scorer sensitivity, choose real precision options, "
                    "and retain every re-encoded checkpoint/archive"
                ),
                "checkpoint": "distinct atomic EMA and optimizer/resume state",
            },
            {
                "stage": "stage_05_same_budget_admission",
                "pairs_per_chunk_max": 30,
                "objective": (
                    "same seeded n32 QBFLOW versus discrete QBW1 control at identical complete "
                    "serialized byte budget; apply no2 section-5 d_pose and S_hat gates"
                ),
                "checkpoint": "retain frames before scoring, logits/argmax/pose6 and archives",
            },
        ],
        "hard_gates": {
            "memory_peak_bytes": 116 * 1024**3,
            "chunk_pairs_max": 30,
            "same_budget_qbw1_control_required": True,
            "dpose_hat_max": 1.25e-4,
            "s_hat_max_exclusive": 0.12,
            "complete_archive_max_bytes": COMPLETE_CAP_BYTES,
            "trained_checkpoint_must_reencode": True,
            "initialized_rate_does_not_transfer": True,
            "n600_or_contest_dispatch_authorized_now": False,
        },
    }
    return atomic_json_once(STORE / "SEALED_TRAINING_FIRE_ORDER.json", fire_order)


def custody_manifest() -> dict[str, Any]:
    manifest_path = STORE / "CUSTODY_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for fact in manifest["files"]:
            path = Path(fact["path"])
            if file_fact(path) != fact:
                raise QBFLOWBuildError(f"custody drift: {path}")
        return manifest
    files = []
    for path in sorted(STORE.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".part"):
            continue
        files.append(file_fact(path))
    manifest = {
        "schema": "ddm_qbflow_payload_custody_manifest.v1",
        "arm": ARM,
        "root": str(STORE),
        "file_count": len(files),
        "logical_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "command": ".venv/bin/python experiments/ddm_qbflow_rate_first_rung.py run",
        "config": file_fact(STORE / "CONFIG.json"),
        "result": file_fact(STORE / "RESULT.json"),
        "cleanup_policy": (
            "only success-only atomic .part files and the storage write probe are removed; "
            "material payloads are never deleted"
        ),
    }
    atomic_json_once(manifest_path, manifest)
    return manifest


def audit() -> dict[str, Any]:
    result = stage_01_02()
    manifest = custody_manifest()
    full_archive_path = Path(result["full_n600_archive"]["path"])
    full_archive = full_archive_path.read_bytes()
    full_packet = packet.read_deterministic_archive(full_archive)
    decoded = packet.decode_packet(full_packet)
    if len(decoded.sections) != 4:
        raise QBFLOWBuildError("audit packet section count mismatch")
    for row in result["mutation_rows"]:
        mutated = Path(row["mutated_payload"]["path"]).read_bytes()
        try:
            packet.decode_packet(mutated)
        except packet.QBFLOWPacketError:
            pass
        else:
            raise QBFLOWBuildError("audit accepted a retained mutation")
    audit_result = {
        "schema": "ddm_qbflow_postrun_audit.v1",
        "pass": True,
        "result": file_fact(STORE / "RESULT.json"),
        "custody_manifest": file_fact(STORE / "CUSTODY_MANIFEST.json"),
        "custody_file_count": manifest["file_count"],
        "custody_logical_bytes": manifest["logical_bytes"],
        "full_archive": file_fact(full_archive_path),
        "decoded_section_facts": list(decoded.section_facts),
        "projected_bytes_recomputed": (
            int(result["b_shared_archive"])
            + math.ceil(
                sum(float(row["ht_weight"]) * int(row["reset_record_bytes"]) for row in result["reset_record_rows"])
            )
        ),
        "exact_full_bytes_recomputed": full_archive_path.stat().st_size,
        "mutation_refusals_replayed": len(result["mutation_rows"]),
        "command": ".venv/bin/python experiments/ddm_qbflow_rate_first_rung.py audit",
    }
    if audit_result["projected_bytes_recomputed"] != result["b_hat_projected_complete_archive"]:
        raise QBFLOWBuildError("audit HT byte arithmetic mismatch")
    if audit_result["exact_full_bytes_recomputed"] != result["exact_full_n600_archive_bytes"]:
        raise QBFLOWBuildError("audit full archive byte mismatch")
    atomic_json_once(STORE / "POSTRUN_AUDIT.json", audit_result)
    return audit_result


def run() -> dict[str, Any]:
    stage_00()
    result = stage_01_02()
    custody_manifest()
    audit()
    supplemental = {
        "schema": "ddm_qbflow_supplemental_custody.v1",
        "custody_manifest": file_fact(STORE / "CUSTODY_MANIFEST.json"),
        "postrun_audit": file_fact(STORE / "POSTRUN_AUDIT.json"),
        "fire_order": file_fact(STORE / "SEALED_TRAINING_FIRE_ORDER.json"),
    }
    atomic_json_once(STORE / "SUPPLEMENTAL_CUSTODY.json", supplemental)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit", "manifest"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "run":
        result = run()
    elif args.action == "audit":
        result = audit()
    else:
        result = custody_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
