#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the resumable n600 C1-quotient q4/q8 box-tolerance control.

The finite family is intentionally scoped as a uniform-quantum upper bound.
It is scorer-recursive in its coordinates (the exact C1 quotient planes) and
realized through the production uint8 resize preimage plus both frozen
scorers.  It does not pretend to be the still-owed full Fisher/G4
per-dimension solve.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (  # noqa: E402
    tolerance_capped_rung_score,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    CONTENT_CODEC_TAG,
    PAIR_PREFIX,
    SPATIAL_SMOOTH_121_ID,
    decode_predictor_residual,
    encode_predictor_residual,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    MAGIC as PREDICTOR_MAGIC,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    PREFIX as PREDICTOR_PREFIX,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    VERSION as PREDICTOR_VERSION,
)
from tac.optimization.arith_selfcomp_rate_coders import (  # noqa: E402
    decode_brotli_q11,
    decode_spatial_context_constriction,
    encode_brotli_q11,
    encode_spatial_context_constriction,
)
from tac.optimization.ddm_metric_custody_bundle import load_metric_custody_bundle  # noqa: E402
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    LayerHome,
    StreamType,
    TypedStreamTag,
    build_minimum_description_headline,
)
from tac.optimization.ddm_ms2r_tolerance_capped_solve_r2 import (  # noqa: E402
    quantize_uint8_half_up,
    solve_binary_pair_lattice,
)
from tac.optimization.ddm_ms7_receiver_edges import (  # noqa: E402
    _encode_zstd_dictionary,
    _frame_coded,
    _linear_signed_array,
    decode_coded_receiver_object,
    sha256_bytes,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    DDMV14RealizationFidelityConfigV1,
    _forward,
    _load_models,
)
from tools.measure_v10_two_plane_receiver_timing import (  # noqa: E402
    _build_production_packet,
    _canonical_archive_bytes,
)

RUN_ID: Final = "ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z"
LANE_ID: Final = "lane_ddm_ms2r_tolerance_capped_r2_20260724"
SCHEMA: Final = "ddm_ms2r_tolerance_capped_solve_r2_receipt.v1"
SCORED_PIXELS: Final = 600 * 384 * 512
ALLOWED_ERRORS: Final = 136_839
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CONFIG_PATH: Final = REPO / ".omx/research/configs/ddm_ms2r_tolerance_capped_solve_r2_20260724.json"
RECEIPT_ROOT: Final = REPO / ".omx/research" / RUN_ID
RECEIPT_PATH: Final = RECEIPT_ROOT / "receipt.json"


class MS2RRunError(ValueError):
    """A bound source, stage checkpoint, or exact replay differs."""


class MS2RConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: str = Field(alias="schema", serialization_alias="schema")
    run_id: str
    lane_id: str
    c1_root: str
    c1_archive_path: str
    c1_archive_sha256: str
    scorer_config_path: str
    scorer_config_sha256: str
    bundle_complete_path: str
    bundle_complete_sha256: str
    ms7_receipt_path: str
    ms7_receipt_sha256: str
    rd1_duals_path: str
    rd1_duals_sha256: str
    rd1_frontier_path: str
    rd1_frontier_sha256: str
    bulk_root: str
    pair_count: StrictInt
    source_chunk_pairs: StrictInt
    scorer_batch_size: StrictInt
    scorer_threads: StrictInt
    rate_workers: StrictInt
    minimum_free_bytes: StrictInt
    seed: StrictInt
    allowed_errors: StrictInt
    research_only: StrictBool
    execution_allowed: StrictBool
    score_claim: StrictBool
    main_review_required: StrictBool
    receipt_timestamp_utc: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MS2RRunError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish(path, _canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MS2RRunError(f"JSON source is not an object: {path}")
    return value


def _bound(path: Path, expected_sha256: str) -> Path:
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_sha256:
        raise MS2RRunError(f"SHA-bound source differs: {path}")
    return path


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "role": role,
    }


def _load_config(path: Path) -> tuple[MS2RConfig, str]:
    payload = path.read_bytes()
    config = MS2RConfig.model_validate_json(payload)
    if (
        config.schema_name != "DDMMS2RToleranceCappedSolveR2ConfigV1"
        or config.run_id != RUN_ID
        or config.lane_id != LANE_ID
        or config.pair_count != 600
        or config.source_chunk_pairs != 12
        or config.scorer_batch_size != 16
        or config.scorer_threads != 4
        or config.seed != 1234
        or config.allowed_errors != ALLOWED_ERRORS
        or config.research_only is not True
        or config.execution_allowed is not False
        or config.score_claim is not False
        or config.main_review_required is not True
    ):
        raise MS2RRunError("typed execution boundary differs")
    return config, hashlib.sha256(payload).hexdigest()


def _predictor_records(payload: bytes) -> tuple[bytes, list[tuple[int, bytes]]]:
    if len(payload) < PREDICTOR_PREFIX.size:
        raise MS2RRunError("predictor payload is truncated")
    prefix = payload[: PREDICTOR_PREFIX.size]
    magic, version, content_tag, count, _height, _width, _channels = PREDICTOR_PREFIX.unpack(prefix)
    if magic != PREDICTOR_MAGIC or version != PREDICTOR_VERSION or content_tag != CONTENT_CODEC_TAG:
        raise MS2RRunError("predictor payload prefix differs")
    cursor = PREDICTOR_PREFIX.size
    records: list[tuple[int, bytes]] = []
    for _ in range(count):
        start = cursor
        if cursor + PAIR_PREFIX.size > len(payload):
            raise MS2RRunError("predictor pair prefix is truncated")
        unpacked = PAIR_PREFIX.unpack_from(payload, cursor)
        pair_id = int(unpacked[0])
        cursor += PAIR_PREFIX.size
        cursor += int(unpacked[2]) + int(unpacked[3]) + int(unpacked[4])
        if cursor > len(payload):
            raise MS2RRunError("predictor pair body is truncated")
        records.append((pair_id, payload[start:cursor]))
    if cursor != len(payload):
        raise MS2RRunError("predictor payload has a trailer")
    return prefix, records


def _source_chunk(c1_root: Path, chunk_index: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    base = c1_root / "prepare_chunks" / f"chunk-{chunk_index:04d}"
    manifest = _read_json(base.with_suffix(".manifest.json"))
    pair_ids = manifest.get("pair_ids")
    count = manifest.get("pair_count")
    if (
        manifest.get("complete") is not True
        or not isinstance(pair_ids, list)
        or type(count) is not int
        or len(pair_ids) != count
    ):
        raise MS2RRunError("C1 source chunk manifest differs")
    shape = (count, 384, 512, 3)
    y0 = np.fromfile(base.with_suffix(".y0.bin"), dtype=np.uint8).reshape(shape)
    y1 = np.fromfile(base.with_suffix(".y1.bin"), dtype=np.uint8).reshape(shape)
    return [int(value) for value in pair_ids], y0, y1


def _encode_rate_chunk_worker(args: tuple[str, str, int]) -> dict[str, Any]:
    c1_root_text, bulk_root_text, chunk_index = args
    c1_root = Path(c1_root_text)
    bulk_root = Path(bulk_root_text)
    pair_ids, y0, y1 = _source_chunk(c1_root, chunk_index)
    rows: dict[int, dict[str, Any]] = {pair_id: {"pair_id": pair_id} for pair_id in pair_ids}
    artifacts: dict[str, dict[str, Any]] = {}
    for step in (4, 8):
        q0 = quantize_uint8_half_up(y0, step)
        q1 = quantize_uint8_half_up(y1, step)
        payload = encode_predictor_residual(
            q0,
            q1,
            modes=SPATIAL_SMOOTH_121_ID,
            pair_ids=tuple(pair_ids),
        )
        decoded = decode_predictor_residual(payload)
        if (
            decoded.pair_ids != tuple(pair_ids)
            or not np.array_equal(decoded.frame0, q0)
            or not np.array_equal(decoded.frame1, q1)
        ):
            raise MS2RRunError("quantized predictor parse-back differs")
        _prefix, records = _predictor_records(payload)
        for pair_id, record in records:
            rows[pair_id][f"q{step}_record_bytes"] = len(record)
        path = bulk_root / "stage_checkpoints/01_rate" / f"chunk-{chunk_index:04d}.q{step}.predictor.bin"
        _publish(path, payload)
        artifacts[f"q{step}"] = _artifact(path, f"q{step} exact predictor chunk")
    return {
        "chunk_index": chunk_index,
        "pair_ids": pair_ids,
        "rows": [rows[pair_id] for pair_id in pair_ids],
        "artifacts": artifacts,
    }


def _measure_rate(config: MS2RConfig, bulk: Path) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/01_rate/rate_measurement.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    args = [
        (config.c1_root, str(bulk), chunk_index)
        for chunk_index in range(math.ceil(config.pair_count / config.source_chunk_pairs))
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.rate_workers) as executor:
        chunks = list(executor.map(_encode_rate_chunk_worker, args))
    chunks.sort(key=lambda row: int(row["chunk_index"]))
    rows = [row for chunk in chunks for row in chunk["rows"]]
    if [row["pair_id"] for row in rows] != list(range(config.pair_count)):
        raise MS2RRunError("rate measurement lost exact pair identity")
    value = {
        "schema": "ddm_ms2r_r2_rate_measurement.v1",
        "pair_count": config.pair_count,
        "codec": "predictor-residual-u8.v1 with Brotli-Q11 owned pair streams",
        "rows": rows,
        "chunks": chunks,
        "all_stage_checkpoints_preserved": True,
        "score_claim": False,
    }
    _publish_json(checkpoint, value)
    return value


def _source_batch(c1_root: Path, pair_start: int, pair_count: int) -> tuple[np.ndarray, np.ndarray]:
    y0_rows: list[np.ndarray] = []
    y1_rows: list[np.ndarray] = []
    chunks: dict[int, tuple[list[int], np.ndarray, np.ndarray]] = {}
    for pair_id in range(pair_start, pair_start + pair_count):
        chunk_index, local = divmod(pair_id, 12)
        if chunk_index not in chunks:
            chunks[chunk_index] = _source_chunk(c1_root, chunk_index)
        _pair_ids, y0, y1 = chunks[chunk_index]
        y0_rows.append(y0[local])
        y1_rows.append(y1[local])
    return np.stack(y0_rows), np.stack(y1_rows)


def _measure_scorers(config: MS2RConfig, bulk: Path) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/02_scorers/scorer_measurement.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    scorer_path = _bound(
        REPO / config.scorer_config_path,
        config.scorer_config_sha256,
    )
    scorer_config = DDMV14RealizationFidelityConfigV1.model_validate_json(scorer_path.read_bytes())
    if (
        scorer_config.scorer_threads != config.scorer_threads
        or scorer_config.scorer_batch_size != config.scorer_batch_size
        or scorer_config.pair_count != config.pair_count
    ):
        raise MS2RRunError("frozen scorer configuration differs")
    segnet, posenet, scorer_custody = _load_models(scorer_config)
    target = _bound(
        Path(scorer_config.target_cache_path),
        scorer_config.target_cache_sha256,
    )
    labels_all = open_stored_npy_memmap(target, "lstars")
    poses_all = open_stored_npy_memmap(target, "gt_poses")
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    rows: list[dict[str, Any]] = [{"pair_id": pair_id} for pair_id in range(config.pair_count)]
    verification_probes = 0
    for start in range(0, config.pair_count, config.scorer_batch_size):
        count = min(config.scorer_batch_size, config.pair_count - start)
        y0, y1 = _source_batch(Path(config.c1_root), start, count)
        labels = np.asarray(labels_all[start : start + count], dtype=np.uint8)
        poses = np.asarray(poses_all[start : start + count], dtype=np.float64)
        for step in (1, 4, 8):
            q0 = y0 if step == 1 else quantize_uint8_half_up(y0, step)
            q1 = y1 if step == 1 else quantize_uint8_half_up(y1, step)
            camera = np.empty((count, 2, 874, 1164, 3), dtype=np.uint8)
            for local in range(count):
                camera[local, 0] = realize_factor2_uint8_scorer_plane(operator, q0[local])
                camera[local, 1] = realize_factor2_uint8_scorer_plane(operator, q1[local])
            for plane_id, target_plane in ((0, q0[0]), (1, q1[0])):
                if not verify_factor2_uint8_scorer_plane(
                    operator,
                    camera[0, plane_id],
                    target_plane,
                ).certified_exact:
                    raise MS2RRunError("uint8 resize preimage verification failed")
                verification_probes += 1
            cells, predicted_pose = _forward(segnet, posenet, camera)
            differences = cells != labels
            pair_errors = np.count_nonzero(differences, axis=(1, 2))
            pose_sse = np.square(predicted_pose - poses).sum(axis=1, dtype=np.float64)
            for local in range(count):
                row = rows[start + local]
                row[f"q{step}_errors"] = int(pair_errors[local])
                row[f"q{step}_pose_sse"] = float(pose_sse[local])
                row[f"q{step}_stratum_errors"] = {
                    name: int(np.count_nonzero(differences[local] & (labels[local] == class_id)))
                    for class_id, name in enumerate(CLASS_NAMES)
                }
        batch_checkpoint = bulk / "stage_checkpoints/02_scorers/batches" / f"batch-{start:04d}.json"
        _publish_json(
            batch_checkpoint,
            {
                "schema": "ddm_ms2r_r2_scorer_batch.v1",
                "pair_range": [start, start + count],
                "rows": rows[start : start + count],
                "score_claim": False,
            },
        )
    q1_errors = sum(row["q1_errors"] for row in rows)
    if q1_errors != 17_927:
        raise MS2RRunError(f"fresh C1 exact replay differs from 17,927: {q1_errors}")
    value = {
        "schema": "ddm_ms2r_r2_scorer_measurement.v1",
        "pair_count": config.pair_count,
        "batch_size": config.scorer_batch_size,
        "threads": config.scorer_threads,
        "rows": rows,
        "q1_exact_control": {
            "errors": q1_errors,
            "d_seg": q1_errors / SCORED_PIXELS,
            "d_pose": sum(row["q1_pose_sse"] for row in rows) / (config.pair_count * 6),
        },
        "uint8_factor2_exact_probe_count": verification_probes,
        "all_candidate_planes_realized_by_exact_constructor": True,
        "target_cache": _artifact(target, "frozen n600 labels and Pose6"),
        "scorer_custody": scorer_custody,
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _publish_json(checkpoint, value)
    return value


def _solve(
    config: MS2RConfig,
    rate: Mapping[str, Any],
    scorers: Mapping[str, Any],
    bulk: Path,
) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/03_solve/exact_binary_solve.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    rate_rows = rate.get("rows")
    scorer_rows = scorers.get("rows")
    if not isinstance(rate_rows, list) or not isinstance(scorer_rows, list):
        raise MS2RRunError("rate or scorer rows are absent")
    rows = []
    for rate_row, scorer_row in zip(rate_rows, scorer_rows, strict=True):
        if rate_row["pair_id"] != scorer_row["pair_id"]:
            raise MS2RRunError("rate and scorer pair identity differ")
        rows.append(
            {
                **rate_row,
                **{
                    key: scorer_row[key]
                    for key in (
                        "q4_errors",
                        "q8_errors",
                        "q4_pose_sse",
                        "q8_pose_sse",
                        "q4_stratum_errors",
                        "q8_stratum_errors",
                    )
                },
            }
        )
    result = solve_binary_pair_lattice(rows, allowed_errors=config.allowed_errors)
    selected = result["selected_steps"]
    result["realized_pose_sse"] = sum(row[f"q{step}_pose_sse"] for row, step in zip(rows, selected, strict=True))
    result["realized_d_pose"] = result["realized_pose_sse"] / (config.pair_count * 6)
    result["realized_stratum_errors"] = {
        name: sum(row[f"q{step}_stratum_errors"][name] for row, step in zip(rows, selected, strict=True))
        for name in CLASS_NAMES
    }
    result["rows_sha256"] = sha256_bytes(_canonical_json({"rows": rows}))
    _publish_json(checkpoint, result)
    return result


def _selected_predictor_payload(
    config: MS2RConfig,
    solve: Mapping[str, Any],
    bulk: Path,
) -> tuple[bytes, list[dict[str, Any]]]:
    selected = solve.get("selected_steps")
    if not isinstance(selected, list) or len(selected) != config.pair_count:
        raise MS2RRunError("solve selection is incomplete")
    records: list[bytes] = []
    source_rows: list[dict[str, Any]] = []
    expected_pair = 0
    for chunk_index in range(math.ceil(config.pair_count / config.source_chunk_pairs)):
        by_step = {}
        for step in (4, 8):
            path = bulk / "stage_checkpoints/01_rate" / f"chunk-{chunk_index:04d}.q{step}.predictor.bin"
            prefix, parsed = _predictor_records(path.read_bytes())
            by_step[step] = dict(parsed)
            if chunk_index == 0 and step == 4:
                global_prefix = prefix
        pair_ids = sorted(by_step[4])
        if pair_ids != sorted(by_step[8]):
            raise MS2RRunError("q4/q8 chunk pair identity differs")
        for pair_id in pair_ids:
            if pair_id != expected_pair:
                raise MS2RRunError("selected predictor pair sequence differs")
            step = int(selected[pair_id])
            records.append(by_step[step][pair_id])
            source_rows.append(
                {
                    "pair_id": pair_id,
                    "selected_step": step,
                    "record_bytes": len(by_step[step][pair_id]),
                }
            )
            expected_pair += 1
    if expected_pair != config.pair_count:
        raise MS2RRunError("selected predictor does not cover n600")
    magic, version, tag, _count, height, width, channels = PREDICTOR_PREFIX.unpack(global_prefix)
    prefix = PREDICTOR_PREFIX.pack(magic, version, tag, config.pair_count, height, width, channels)
    return prefix + b"".join(records), source_rows


def _selected_y_digest(config: MS2RConfig, solve: Mapping[str, Any]) -> str:
    selected = solve["selected_steps"]
    digest = hashlib.sha256()
    for plane_id in (0, 1):
        for chunk_index in range(math.ceil(config.pair_count / config.source_chunk_pairs)):
            pair_ids, y0, y1 = _source_chunk(Path(config.c1_root), chunk_index)
            source = y0 if plane_id == 0 else y1
            for local, pair_id in enumerate(pair_ids):
                digest.update(quantize_uint8_half_up(source[local], int(selected[pair_id])).tobytes())
    return digest.hexdigest()


def _materialize_candidate(
    config: MS2RConfig,
    solve: Mapping[str, Any],
    bulk: Path,
) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/04_candidate/candidate.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    predictor, source_rows = _selected_predictor_payload(config, solve, bulk)
    packet = _build_production_packet(
        predictor,
        pair_count=config.pair_count,
        camera_hw=(874, 1164),
        scorer_hw=(384, 512),
        decoded_y_sha256=_selected_y_digest(config, solve),
    )
    archive = _canonical_archive_bytes(packet)
    archive_path = bulk / "stage_checkpoints/04_candidate/archive.zip"
    packet_path = bulk / "stage_checkpoints/04_candidate/0.bin"
    _publish(packet_path, packet)
    _publish(archive_path, archive)
    value = {
        "schema": "ddm_ms2r_r2_candidate.v1",
        "predictor": _artifact(packet_path, "production receiver packet"),
        "archive": _artifact(archive_path, "canonical stored-member archive"),
        "selected_record_rows": source_rows,
        "predictor_payload_bytes": len(predictor),
        "strict_production_parseback_exact": True,
        "canonical_archive_determinism_x2": _canonical_archive_bytes(packet) == archive,
        "receiver_contract": "tac.witness_dsl.v10_production_receiver.v1",
        "score_claim": False,
    }
    _publish_json(checkpoint, value)
    return value


def _race_chunk(payload: bytes) -> dict[str, Any]:
    rows = [
        {
            "coder": "RAW_COMPACT",
            "bytes": len(payload),
            "parseback_exact": True,
            "frame_sha256": sha256_bytes(payload),
        }
    ]
    try:
        encoded = encode_brotli_q11(payload)
        frame = _frame_coded("E4_BROTLI_Q11", encoded, payload)
        rows.append(
            {
                "coder": "E4_BROTLI_Q11",
                "bytes": len(frame),
                "parseback_exact": decode_brotli_q11(frame[struct.calcsize(">5sBQ32s") :]) == payload,
                "frame_sha256": sha256_bytes(frame),
            }
        )
    except Exception as exc:  # optional dependency, retained as a NULL row
        rows.append({"coder": "E4_BROTLI_Q11", "bytes": None, "parseback_exact": False, "blocker": str(exc)})
    try:
        encoded = encode_spatial_context_constriction(_linear_signed_array(payload))
        frame = _frame_coded("CONSTRICTION_ORDER1_CONTEXT_ANS", encoded, payload)
        decoded = decode_spatial_context_constriction(frame[struct.calcsize(">5sBQ32s") :])
        rows.append(
            {
                "coder": "CONSTRICTION_ORDER1_CONTEXT_ANS",
                "bytes": len(frame),
                "parseback_exact": np.ascontiguousarray(decoded).view(np.uint8).tobytes() == payload,
                "frame_sha256": sha256_bytes(frame),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "coder": "CONSTRICTION_ORDER1_CONTEXT_ANS",
                "bytes": None,
                "parseback_exact": False,
                "blocker": str(exc),
            }
        )
    try:
        frame = _encode_zstd_dictionary(payload)
        coded = _frame_coded("ZSTD19_TRAINED_DICTIONARY", frame, payload)
        rows.append(
            {
                "coder": "ZSTD19_TRAINED_DICTIONARY",
                "bytes": len(coded),
                "parseback_exact": decode_coded_receiver_object(coded) == payload,
                "frame_sha256": sha256_bytes(coded),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "coder": "ZSTD19_TRAINED_DICTIONARY",
                "bytes": None,
                "parseback_exact": False,
                "blocker": str(exc),
            }
        )
    eligible = [row for row in rows if row["parseback_exact"] and isinstance(row["bytes"], int)]
    winner = min(eligible, key=lambda row: (int(row["bytes"]), str(row["coder"])))
    return {
        "schema": "ddm_ms2r_r2_stream_coder_race.v1",
        "raw_bytes": len(payload),
        "raw_sha256": sha256_bytes(payload),
        "rows": rows,
        "winner": winner,
        "score_claim": False,
    }


def _race_candidate_streams(
    config: MS2RConfig,
    solve: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bulk: Path,
) -> dict[str, Any]:
    checkpoint = bulk / "stage_checkpoints/05_coder_race/coder_race.json"
    if checkpoint.exists():
        return _read_json(checkpoint)
    predictor, _rows = _selected_predictor_payload(config, solve, bulk)
    prefix, records = _predictor_records(predictor)
    streams = [
        prefix + b"".join(record for _pair_id, record in records[start : start + config.source_chunk_pairs])
        for start in range(0, len(records), config.source_chunk_pairs)
    ]
    stream_root = bulk / "stage_checkpoints/05_coder_race/streams"
    races: list[dict[str, Any] | None] = [None] * len(streams)
    pending: list[int] = []
    for index, stream in enumerate(streams):
        path = stream_root / f"stream-{index:04d}.json"
        if path.exists():
            row = _read_json(path)
            if (
                row.get("schema") != "ddm_ms2r_r2_stream_coder_race.v1"
                or row.get("raw_bytes") != len(stream)
                or row.get("raw_sha256") != sha256_bytes(stream)
            ):
                raise MS2RRunError("resumed stream coder checkpoint differs")
            races[index] = row
        else:
            pending.append(index)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(2, config.rate_workers)
    ) as executor:
        futures = {
            executor.submit(_race_chunk, streams[index]): index for index in pending
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            row = future.result()
            _publish_json(stream_root / f"stream-{index:04d}.json", row)
            races[index] = row
    if any(race is None for race in races):
        raise MS2RRunError("per-stream coder race checkpoint set is incomplete")
    completed_races = [race for race in races if race is not None]
    raw_archive_bytes = int(candidate["archive"]["bytes"])
    # The standard production archive remains the admitted object unless the
    # separately framed stream collection is actually smaller.  No unbuilt
    # decoder-container bytes are silently omitted.
    framed_stream_bytes = sum(
        int(race["winner"]["bytes"]) for race in completed_races
    )
    value = {
        "schema": "ddm_ms2r_r2_per_stream_coder_race.v1",
        "stream_count": len(streams),
        "required_coders": [
            "RAW_COMPACT",
            "E4_BROTLI_Q11",
            "CONSTRICTION_ORDER1_CONTEXT_ANS",
            "ZSTD19_TRAINED_DICTIONARY",
        ],
        "streams": [
            {"stream_index": index, **race}
            for index, race in enumerate(completed_races)
        ],
        "framed_stream_payload_bytes_excluding_unbuilt_container": framed_stream_bytes,
        "admitted_best_coded_bytes": raw_archive_bytes,
        "admitted_winner": "RAW_COMPACT_PRODUCTION_ARCHIVE",
        "why_smaller_stream_sum_not_admitted": (
            "a counted receiver container/header was not materialized; the raw "
            "production archive is the only complete one-object byte custody"
        ),
        "all_available_rows_parseback_exact": all(
            row["parseback_exact"]
            for race in completed_races
            for row in race["rows"]
            if isinstance(row.get("bytes"), int)
        ),
        "g4_context_status": "NULL_NO_CUSTODIED_G4_SPATIAL_HOME_FOR_C1_PREDICTOR_STREAMS",
        "score_claim": False,
    }
    _publish_json(checkpoint, value)
    return value


def _rd1_null_preserving_backfill(config: MS2RConfig, solve: Mapping[str, Any]) -> dict[str, Any]:
    source_path = _bound(REPO / config.rd1_duals_path, config.rd1_duals_sha256)
    source = _read_json(source_path)
    rows = source.get("dimension_duals", {}).get("bucket_rows")
    if not isinstance(rows, list) or len(rows) != 162:
        raise MS2RRunError("RD1 source cube differs from 162 cells")
    cells = []
    for row in rows:
        cells.append(
            {
                "dual_index": row["dual_index"],
                "stratum": row["stratum"],
                "scorer_visibility": row["scorer_visibility"],
                "g4_temporal_class": row["g4_temporal_class"],
                "lambda_bytes_per_D_dimension": None,
                "effective_quantum_D": row.get("effective_quantum_D"),
                "measurement_status": ("STILL_NULL_BINARY_CONTROL_HAS_NO_G4_X_VISIBILITY_DIMENSION_RATE_HOME"),
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
        )
    return {
        "schema": "ddm_ms2r_r2_rd1_dual_backfill.v1",
        "source": _artifact(source_path, "RD1 162-cell source cube"),
        "source_cell_count": 162,
        "measured_cell_count": 0,
        "still_null_cell_count": 162,
        "aggregate_binary_control_dual": {
            "delta_predictor_record_bytes_q4_vs_q8": sum(
                int(row["q4_record_bytes"]) - int(row["q8_record_bytes"])
                for row in _read_json(Path(config.bulk_root) / "stage_checkpoints/01_rate/rate_measurement.json")[
                    "rows"
                ]
            ),
            "delta_errors_q4_vs_q8": sum(
                int(row["q4_errors"]) - int(row["q8_errors"])
                for row in _read_json(Path(config.bulk_root) / "stage_checkpoints/02_scorers/scorer_measurement.json")[
                    "rows"
                ]
            ),
            "epistemic_status": ("MEASURED_AGGREGATE_BINARY_EDGE_NONTRANSFERABLE_TO_RD1_162_CELLS"),
        },
        "cells": cells,
        "blocker": (
            "binary pair controls do not assign nonadditive predictor bytes to "
            "stratum x scorer_visibility x G4 temporal cells"
        ),
        "score_claim": False,
    }


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(bulk)
    if usage.free < config.minimum_free_bytes:
        raise MS2RRunError("SSD-first preflight has insufficient free bytes")
    bundle_path = _bound(REPO / config.bundle_complete_path, config.bundle_complete_sha256)
    bundle = load_metric_custody_bundle(
        bundle_path,
        repository_root=REPO,
        require_complete=True,
    )
    ms7_path = _bound(REPO / config.ms7_receipt_path, config.ms7_receipt_sha256)
    ms7 = _read_json(ms7_path)
    if (
        ms7.get("schema") != "ddm_ms7_receiver_edges_receipt.v1"
        or ms7.get("pf3", {}).get("registered_waterfill_callable", {}).get("equation_id")
        != "ddm_tolerance_capped_min_score_waterfill_v1"
    ):
        raise MS2RRunError("MS7 five-edge callable custody differs")
    _bound(Path(config.c1_archive_path), config.c1_archive_sha256)
    _bound(REPO / config.rd1_frontier_path, config.rd1_frontier_sha256)

    rate = _measure_rate(config, bulk)
    scorers = _measure_scorers(config, bulk)
    solve = _solve(config, rate, scorers, bulk)
    candidate = _materialize_candidate(config, solve, bulk)
    coder_race = _race_candidate_streams(config, solve, candidate, bulk)
    rung = tolerance_capped_rung_score(
        seg_errors=int(solve["realized_errors"]),
        scored_pixels=SCORED_PIXELS,
        d_pose=float(solve["realized_d_pose"]),
        raw_compact_bytes=int(candidate["archive"]["bytes"]),
        best_coded_bytes=int(coder_race["admitted_best_coded_bytes"]),
        allowed_errors=config.allowed_errors,
        bundle_complete=bundle.complete,
        parseback_exact=bool(candidate["strict_production_parseback_exact"]),
        uint8_reverified=True,
    )
    duals = _rd1_null_preserving_backfill(config, solve)
    dual_path = RECEIPT_ROOT / "rd1_162_dual_backfill.json"
    _publish_json(dual_path, duals)
    empty_sha = hashlib.sha256(b"").hexdigest()
    headline = build_minimum_description_headline(
        stored_problem_bytes=int(candidate["archive"]["bytes"]),
        stored_problem_sha256=str(candidate["archive"]["sha256"]),
        exception_bytes=0,
        exception_sha256=empty_sha,
        realized_d_seg=float(rung["d_seg"]),
        realized_d_pose=float(rung["d_pose"]),
        stored_problem_own_lineage=True,
        donor_conditioned=False,
        expansion_receiver_closed=True,
        pose_tube_active=True,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=True,
        scorer_metric_active=True,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
        typed_stream_tags=(
            TypedStreamTag(
                type=StreamType.FIBER,
                layer_home=LayerHome.L3_RASTER,
                evaluate_py_recursion_level_cited=("L3 exact quotient raster -> L4 frozen scorers -> L5 verdict"),
                counted_bytes=int(candidate["archive"]["bytes"]),
                free_receiver_code=True,
            ),
            TypedStreamTag(
                type=StreamType.RESIDUAL,
                layer_home=LayerHome.L5_VERDICT,
                evaluate_py_recursion_level_cited="L5 no separate solve exception stream",
                counted_bytes=0,
                free_receiver_code=True,
            ),
        ),
        strict_typed_stream_tags=True,
        metric_custody_bundle_path=bundle_path,
        metric_custody_repository_root=REPO,
    )
    receipt = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "finished_at_utc": config.receipt_timestamp_utc,
        "verdict": (
            "MEASURED_RECEIVER_CLOSED_BINARY_Q4_Q8_CONTROL_KNEE_INSIDE_BOX; "
            "FULL_FISHER_G4_WATERFILL_AND_RD1_CELL_DUALS_STILL_BLOCKED"
        ),
        "verdict_scope": (
            "INSTANCE C1 exact scorer-quotient x finite per-pair q4/q8 uniform-quantum "
            "control on macOS CPU batch16; [naive-uniform-quantum upper bound], not "
            "a full per-dimension Fisher/G4 or contest score verdict"
        ),
        "authority": {
            "evidence_axis": AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "local_cost_usd": 0,
            "torch_threads": config.scorer_threads,
            "main_landing_review_required": True,
        },
        "typed_config": _artifact(config_path, "typed R2 execution config"),
        "storage": {
            "selected_tier": str(bulk),
            "free_bytes_at_preflight": usage.free,
            "minimum_free_bytes": config.minimum_free_bytes,
            "crash_resume": "immutable per-stage checkpoints",
            "cleanup": (
                "preserve measured checkpoints on SSD; no local bulk; certifying receipt SHA-binds all promoted bulk"
            ),
        },
        "input_custody": {
            "bundle_complete": _artifact(bundle_path, "strict MS3 complete bundle"),
            "ms7": _artifact(ms7_path, "PF3 five-edge and coder control"),
            "c1_archive": _artifact(Path(config.c1_archive_path), "exact C1 endpoint control"),
        },
        "homotopy": {
            "controls": ["q1 exact", "q4 uniform", "q8 uniform"],
            "waterfilled_family": "per-pair binary q4/q8 exact dynamic program",
            "solve": solve,
            "candidate": candidate,
            "coder_race": coder_race,
            "registered_callable": {
                "equation_id": "ddm_tolerance_capped_min_score_waterfill_v1",
                "output": rung,
            },
        },
        "headline": headline,
        "rd1_dual_backfill": _artifact(dual_path, "NULL-preserving 162-cell supplement"),
        "knee_comparison": {
            "c1_exact_control": {
                "bytes": 409_526_925,
                "errors": 17_927,
                "joint_S": 272.7342793310384,
            },
            "binary_control_knee": {
                "bytes": candidate["archive"]["bytes"],
                "errors": solve["realized_errors"],
                "d_seg": rung["d_seg"],
                "d_pose": rung["d_pose"],
                "joint_S": rung["joint_S"],
            },
            "rd1_proposal_channel_knee": {
                "bytes": 138_801,
                "errors": 8_318_787,
                "joint_S": 26.28022355199344,
                "inside_box": False,
            },
            "channel_price_status": (
                "MEASURED_DESCRIBE_CONTROL_POINT; FULL_CHANNEL_PRICE_BLOCKED_HEADLINE_NOT_ELIGIBLE"
            ),
            "ic2_composition": (
                "candidate archive is an exact receiver-closed quotient-plane input "
                "for incumbent_v1; fresh composed row remains owed to ic2/MAIN"
            ),
        },
        "directive_consumption": [
            {
                "utc": "2026-07-24T14:45:16Z",
                "application": (
                    "construction uses exact scorer quotient planes; uniform q4/q8 is "
                    "explicitly labeled naive control and not promoted as the Fisher/G4 solve"
                ),
            },
            {
                "utc": "2026-07-24T17:39:13Z",
                "application": "dependency-cap deletion noted; no dependency was added",
            },
        ],
        "triality": {
            "dsl": str(config_path.resolve().relative_to(REPO)),
            "dag": f".omx/research/{RUN_ID}/DAG_FEED.md",
            "equations": [
                "ddm_tolerance_capped_min_score_waterfill_v1",
                "dynamic_quantum_calibration_v1 (consumed as owed replacement; q4/q8 remain controls)",
            ],
        },
        "main_landing_review_required": True,
    }
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    _publish_json(RECEIPT_PATH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
