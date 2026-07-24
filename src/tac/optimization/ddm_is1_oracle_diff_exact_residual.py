# SPDX-License-Identifier: MIT
"""Exact n600 price floor for an oracle-diff raster correction stream.

This module deliberately prices the least-structured, fully reversible member
of the solve-as-oracle family.  It does not infer a price for SKELETON,
CONNECTION, or FIBER from historical proposal channels.  Each bounded stage
encodes the exact solved-minus-predictor uint8 scorer-plane delta, decodes it,
and proves predictor + correction == solved before checking the canonical
uint8 realization through the real resize operator.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tac.optimization.solve_diff_operator_mining import (
    AXIS,
    POINTER,
    FullResizeKernel,
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    canonical_json_bytes,
    load_sha256_checked_bytes,
    realize_solve_camera,
    sha256_file,
    storage_preflight,
)

SCHEMA = "ddm_is1_oracle_diff_exact_residual_config.v1"
STAGE_SCHEMA = "ddm_is1_oracle_diff_exact_residual_stage.v1"
RECEIPT_SCHEMA = "ddm_is1_oracle_diff_exact_residual_receipt.v1"
RUN_ID = "ddm_is1_oracle_diff_exact_residual_n600_20260724"
RECORD_MAGIC = b"IS1ODR1\0"
CONTAINER_MAGIC = b"IS1ODC1\0"
RECORD_PREFIX = struct.Struct(">8sIQ")
CONTAINER_PREFIX = struct.Struct(">8sI")


class OracleDiffPriceError(ValueError):
    """The exact correction price or its custody is malformed."""


class OracleDiffPriceConfigV1(BaseModel):
    """Strict local-only configuration for the exact residual price pass."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal[
        "ddm_is1_oracle_diff_exact_residual_config.v1"
    ] = Field(default=SCHEMA, alias="schema", serialization_alias="schema")
    run_id: Literal[
        "ddm_is1_oracle_diff_exact_residual_n600_20260724"
    ] = RUN_ID
    source_config_path: str
    source_config_sha256: str
    pair_count: Literal[600] = 600
    chunk_size: Literal[12] = 12
    coders: tuple[
        Literal["zlib9"],
        Literal["lzma0"],
    ] = ("zlib9", "lzma0")
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    archive_emitted: Literal[False] = False
    pointer_moved: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = POINTER
    evidence_axis: Literal[
        "[macOS-CPU frozen-scorer advisory]"
    ] = AXIS

    def source_config(self) -> SolveDiffMiningConfigV1:
        raw = load_sha256_checked_bytes(
            self.source_config_path,
            self.source_config_sha256,
        )
        source = SolveDiffMiningConfigV1.model_validate_json(raw)
        if (
            source.input_mode != "production"
            or source.pair_start != 0
            or source.pair_count != self.pair_count
            or source.chunk_size != self.chunk_size
        ):
            raise OracleDiffPriceError(
                "source config must be the production n600, chunk-12 G2 surface"
            )
        return source


@dataclass(frozen=True)
class EncodedCorrectionStage:
    """One self-describing exact correction record."""

    record: bytes
    header: Mapping[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _little_int16(delta: np.ndarray) -> np.ndarray:
    raw = np.asarray(delta)
    if raw.ndim != 5 or raw.shape[1:] != (2, 384, 512, 3):
        raise OracleDiffPriceError(
            "correction stage must have shape (pairs,2,384,512,3)"
        )
    if raw.dtype.kind not in "iu" or np.min(raw) < -255 or np.max(raw) > 255:
        raise OracleDiffPriceError(
            "correction stage must be integral and bounded to uint8 difference"
        )
    return np.ascontiguousarray(raw.astype("<i2", copy=False))


def encode_exact_correction_stage(
    delta: np.ndarray,
    *,
    pair_start: int,
    pair_stop: int,
) -> EncodedCorrectionStage:
    """Encode one exact scorer-plane correction with deterministic real coders."""

    typed = _little_int16(delta)
    if pair_start < 0 or pair_stop <= pair_start or pair_stop - pair_start != len(typed):
        raise OracleDiffPriceError("pair range does not match correction stage")
    raw = typed.tobytes(order="C")
    candidates = {
        "zlib9": zlib.compress(raw, level=9),
        "lzma0": lzma.compress(raw, preset=0),
    }
    coder = min(candidates, key=lambda name: (len(candidates[name]), name))
    payload = candidates[coder]
    header = {
        "schema": STAGE_SCHEMA,
        "pair_start": pair_start,
        "pair_stop": pair_stop,
        "shape": list(typed.shape),
        "dtype": "<i2",
        "coder": coder,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
    }
    header_bytes = canonical_json_bytes(header)
    record = (
        RECORD_PREFIX.pack(RECORD_MAGIC, len(header_bytes), len(payload))
        + header_bytes
        + payload
    )
    return EncodedCorrectionStage(record=record, header=header)


def decode_exact_correction_stage(record: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """Strictly parse and decode one exact correction record."""

    if len(record) < RECORD_PREFIX.size:
        raise OracleDiffPriceError("correction record is truncated")
    magic, header_bytes, payload_bytes = RECORD_PREFIX.unpack_from(record)
    if magic != RECORD_MAGIC:
        raise OracleDiffPriceError("correction record magic differs")
    expected = RECORD_PREFIX.size + header_bytes + payload_bytes
    if expected != len(record):
        raise OracleDiffPriceError("correction record length differs")
    header_start = RECORD_PREFIX.size
    header_stop = header_start + header_bytes
    try:
        header = json.loads(record[header_start:header_stop])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleDiffPriceError("correction record header is invalid") from exc
    required = {
        "schema",
        "pair_start",
        "pair_stop",
        "shape",
        "dtype",
        "coder",
        "raw_bytes",
        "raw_sha256",
        "payload_bytes",
        "payload_sha256",
    }
    if set(header) != required or header["schema"] != STAGE_SCHEMA:
        raise OracleDiffPriceError("correction record header schema differs")
    payload = record[header_stop:]
    if (
        header["payload_bytes"] != len(payload)
        or header["payload_sha256"] != _sha256(payload)
    ):
        raise OracleDiffPriceError("correction payload custody differs")
    if header["coder"] == "zlib9":
        raw = zlib.decompress(payload)
    elif header["coder"] == "lzma0":
        raw = lzma.decompress(payload)
    else:
        raise OracleDiffPriceError("correction coder is outside sealed vocabulary")
    shape = tuple(header["shape"])
    if (
        header["dtype"] != "<i2"
        or len(shape) != 5
        or shape[1:] != (2, 384, 512, 3)
        or shape[0] != header["pair_stop"] - header["pair_start"]
        or header["raw_bytes"] != len(raw)
        or header["raw_sha256"] != _sha256(raw)
    ):
        raise OracleDiffPriceError("decoded correction geometry/custody differs")
    result = np.frombuffer(raw, dtype="<i2").reshape(shape).copy()
    return header, result


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_checkpoint(
    path: Path,
    *,
    pair_start: int,
    pair_stop: int,
    config_sha256: str,
    module_sha256: str,
    tool_sha256: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleDiffPriceError(f"invalid stage checkpoint: {path}") from exc
    if (
        value.get("schema") != STAGE_SCHEMA
        or value.get("pair_start") != pair_start
        or value.get("pair_stop") != pair_stop
        or value.get("config_sha256") != config_sha256
        or value.get("module_sha256") != module_sha256
        or value.get("tool_sha256") != tool_sha256
        or value.get("exact_delta_parseback") is not True
        or value.get("real_r_uint8_identity") is not True
    ):
        raise OracleDiffPriceError(
            f"stage checkpoint cannot be resumed under current custody: {path}"
        )
    return value


def _stage_checkpoint(
    *,
    pair_start: int,
    pair_stop: int,
    encoded: EncodedCorrectionStage,
    max_r_abs_error: float,
    config_sha256: str,
    module_sha256: str,
    tool_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **encoded.header,
        "config_sha256": config_sha256,
        "module_sha256": module_sha256,
        "tool_sha256": tool_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "record_bytes": len(encoded.record),
        "record_sha256": _sha256(encoded.record),
        "exact_delta_parseback": True,
        "real_r_uint8_identity": True,
        "max_r_abs_error": max_r_abs_error,
        "evidence_axis": AXIS,
        "research_only": True,
        "score_claim": False,
    }


def _price_table(total_container_bytes: int) -> list[dict[str, Any]]:
    return [
        {
            "stream_type": "GAUGE",
            "layer_home": "L3_raster",
            "counted_bytes": 0,
            "measurement_status": "EXACT_BY_SCORER_PLANE_QUOTIENT_COORDINATE_CHOICE",
            "price_role": "STRUCTURAL_ZERO_NOT_AN_EMPIRICAL_EXCHANGE_RATE",
        },
        {
            "stream_type": "CONNECTION",
            "layer_home": None,
            "counted_bytes": None,
            "measurement_status": "NULL_NO_RECEIVER_CLOSED_ORACLE_DIFF_GENERATOR",
            "price_role": "UNMEASURED",
        },
        {
            "stream_type": "SKELETON",
            "layer_home": None,
            "counted_bytes": None,
            "measurement_status": "NULL_NO_RECEIVER_CLOSED_ORACLE_DIFF_GENERATOR",
            "price_role": "UNMEASURED",
        },
        {
            "stream_type": "FIBER",
            "layer_home": None,
            "counted_bytes": None,
            "measurement_status": "NULL_NO_RECEIVER_CLOSED_ORACLE_DIFF_GENERATOR",
            "price_role": "UNMEASURED",
        },
        {
            "stream_type": "RESIDUAL",
            "layer_home": "L3_raster",
            "counted_bytes": total_container_bytes,
            "measurement_status": "MEASURED_EXACT_REVERSIBLE_N600",
            "price_role": "TRUE_PRICE_UPPER_BOUND_BEFORE_REHOMING",
        },
    ]


def run_exact_oracle_diff_price(
    config: OracleDiffPriceConfigV1,
    output_root: str | Path,
    *,
    resume: bool,
    tool_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Measure the exact residual correction container in 50 resumable stages."""

    source = config.source_config()
    preflight = storage_preflight(source)
    root = Path(output_root)
    stages_root = root / "stage_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_file(config_path)
    module_sha256 = sha256_file(Path(__file__).resolve())
    tool_sha256 = sha256_file(tool_path)
    kernel = FullResizeKernel.build()
    context = None
    checkpoints: list[dict[str, Any]] = []
    for pair_start in range(0, config.pair_count, config.chunk_size):
        pair_stop = min(config.pair_count, pair_start + config.chunk_size)
        stage_path = stages_root / f"pairs_{pair_start:04d}_{pair_stop:04d}.json"
        if stage_path.exists():
            if not resume:
                raise OracleDiffPriceError(
                    f"stage exists; rerun with --resume: {stage_path}"
                )
            checkpoint = _read_checkpoint(
                stage_path,
                pair_start=pair_start,
                pair_stop=pair_stop,
                config_sha256=config_sha256,
                module_sha256=module_sha256,
                tool_sha256=tool_sha256,
            )
        else:
            if context is None:
                context = _open_production_inputs(source)
            pair_ids = tuple(range(pair_start, pair_stop))
            chunk = _load_production_inputs(
                context,
                source,
                pair_ids,
                kernel,
            )
            delta = (
                chunk.solved_planes.astype(np.int16)
                - chunk.predictor_planes.astype(np.int16)
            )
            encoded = encode_exact_correction_stage(
                delta,
                pair_start=pair_start,
                pair_stop=pair_stop,
            )
            _, decoded = decode_exact_correction_stage(encoded.record)
            reconstructed = chunk.predictor_planes.astype(np.int16) + decoded
            if (
                np.min(reconstructed) < 0
                or np.max(reconstructed) > 255
                or not np.array_equal(
                    reconstructed.astype(np.uint8),
                    chunk.solved_planes,
                )
            ):
                raise OracleDiffPriceError(
                    "predictor plus decoded correction differs from solved planes"
                )
            max_r_abs_error = 0.0
            for local in range(len(pair_ids)):
                for frame_index in (0, 1):
                    solved_plane = chunk.solved_planes[local, frame_index]
                    camera = realize_solve_camera(solved_plane, kernel)
                    realized = kernel.operator.apply(camera)
                    error = float(
                        np.max(
                            np.abs(
                                realized.astype(np.float64)
                                - solved_plane.astype(np.float64)
                            )
                        )
                    )
                    max_r_abs_error = max(max_r_abs_error, error)
                    if not np.array_equal(
                        np.rint(realized).astype(np.uint8),
                        solved_plane,
                    ):
                        raise OracleDiffPriceError(
                            "decoded correction fails real R/uint8 identity"
                        )
            checkpoint = _stage_checkpoint(
                pair_start=pair_start,
                pair_stop=pair_stop,
                encoded=encoded,
                max_r_abs_error=max_r_abs_error,
                config_sha256=config_sha256,
                module_sha256=module_sha256,
                tool_sha256=tool_sha256,
                source_hashes={
                    "solved_planes_receipt": source.solved_planes_receipt_sha256,
                    "predictor_archive": source.predictor_archive_sha256,
                    **chunk.source_hashes,
                },
            )
            _write_atomic_json(stage_path, checkpoint)
            del chunk, delta, decoded, reconstructed, encoded
        checkpoints.append(checkpoint)

    if (
        len(checkpoints) != 50
        or checkpoints[0]["pair_start"] != 0
        or checkpoints[-1]["pair_stop"] != 600
        or any(
            left["pair_stop"] != right["pair_start"]
            for left, right in pairwise(checkpoints)
        )
    ):
        raise OracleDiffPriceError("stage coverage is not exactly ordered n600")
    total_container_bytes = CONTAINER_PREFIX.size + sum(
        int(checkpoint["record_bytes"]) for checkpoint in checkpoints
    )
    stage_manifest = [
        {
            "pair_start": value["pair_start"],
            "pair_stop": value["pair_stop"],
            "coder": value["coder"],
            "payload_bytes": value["payload_bytes"],
            "record_bytes": value["record_bytes"],
            "record_sha256": value["record_sha256"],
            "checkpoint_path": str(
                stages_root
                / f"pairs_{value['pair_start']:04d}_{value['pair_stop']:04d}.json"
            ),
        }
        for value in checkpoints
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "evidence_axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "pair_count": 600,
        "frame_count": 1200,
        "stage_count": 50,
        "container_format": {
            "magic": CONTAINER_MAGIC.decode("ascii").rstrip("\0"),
            "prefix_bytes": CONTAINER_PREFIX.size,
            "record_magic": RECORD_MAGIC.decode("ascii").rstrip("\0"),
            "record_prefix_bytes": RECORD_PREFIX.size,
            "materialized": False,
            "role": "exact deterministic byte count from generated-and-parseback-verified stage records",
        },
        "exact_residual_container_bytes": total_container_bytes,
        "exact_residual_container_score_price": (
            25.0 * total_container_bytes / 37_545_489.0
        ),
        "type_layer_price_table": _price_table(total_container_bytes),
        "price_authority": {
            "historical_exchange_rates": "UPPER_BOUND_PROPOSAL_SEARCH_CHANNEL_ONLY",
            "this_exact_residual_row": "TRUE_PRICE_UPPER_BOUND_BEFORE_REHOMING",
            "gauge_row": "STRUCTURAL_ZERO_IN_THE_SELECTED_L3_COORDINATE_SYSTEM",
            "unmeasured_type_layer_rows": "NULL_NEVER_INFER",
        },
        "parseback": {
            "exact_delta_stage_count": 50,
            "real_r_uint8_identity_stage_count": 50,
            "max_r_abs_error": max(
                float(value["max_r_abs_error"]) for value in checkpoints
            ),
        },
        "source_custody": {
            "source_config_path": config.source_config_path,
            "source_config_sha256": config.source_config_sha256,
            "solved_planes_receipt_sha256": source.solved_planes_receipt_sha256,
            "predictor_archive_sha256": source.predictor_archive_sha256,
            "gt_cache_sha256": source.gt_cache_sha256,
        },
        "implementation_custody": {
            "config_sha256": config_sha256,
            "module_sha256": module_sha256,
            "tool_sha256": tool_sha256,
        },
        "storage_preflight": preflight,
        "stage_manifest": stage_manifest,
        "artifact_hygiene": {
            "scratch_created": [],
            "scratch_cleanup_status": "NO_SCRATCH_CREATED",
            "compressed_payloads_persisted": False,
            "rebuildable_from_sha_bound_sources": True,
        },
        "blockers": [
            "CONNECTION_ORACLE_DIFF_GENERATOR_NOT_RECEIVER_CLOSED",
            "SKELETON_ORACLE_DIFF_GENERATOR_NOT_RECEIVER_CLOSED",
            "FIBER_ORACLE_DIFF_GENERATOR_NOT_RECEIVER_CLOSED",
            "FIVE_TYPE_LAYER_REHOMING_669C_NOT_RUN",
            "NO_CANDIDATE_ARCHIVE",
            "NO_NEW_FROZEN_SCORER_INVOCATION",
        ],
        "verdict": "EXACT_RESIDUAL_TRUE_PRICE_RATE_DEAD_REHOMING_ROWS_NULL",
        "verdict_scope": (
            "FORMULATION: exact reversible L3_raster RESIDUAL correction of "
            "the V12 predictor to the solved n600 scorer planes; not a verdict "
            "on oracle-diff type/layer re-homing, training, family, or paradigm"
        ),
        "main_landing_review_required": True,
    }
    _write_atomic_json(root / "receipt.json", receipt)
    return receipt


__all__ = [
    "CONTAINER_MAGIC",
    "RECEIPT_SCHEMA",
    "RUN_ID",
    "EncodedCorrectionStage",
    "OracleDiffPriceConfigV1",
    "OracleDiffPriceError",
    "decode_exact_correction_stage",
    "encode_exact_correction_stage",
    "run_exact_oracle_diff_price",
]
