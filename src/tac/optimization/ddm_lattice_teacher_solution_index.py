# SPDX-License-Identifier: MIT
"""Dense-free, hash-verified index over the own-lineage G21 lattice teachers.

The selected MS2R packet contains 600 independently framed V10 predictor
records.  This module validates and reconstructs those records one at a time,
joins them to the 600 MS1 SENSE rows and the selected q4/q8 decisions, and
retains only hashes/accounting metadata.  Teacher pixels never enter the
returned index or a candidate payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np

from tac.codec.v10_predictor_residual import (
    AFFINE6,
    CONTENT_CODEC_TAG,
    MODE_IDS,
    PredictorMode,
    predict_plane,
)
from tac.codec.v10_predictor_residual import (
    MAGIC as PREDICTOR_MAGIC,
)
from tac.codec.v10_predictor_residual import (
    PAIR_PREFIX as PREDICTOR_PAIR_PREFIX,
)
from tac.codec.v10_predictor_residual import (
    PREFIX as PREDICTOR_PREFIX,
)
from tac.codec.v10_predictor_residual import (
    VERSION as PREDICTOR_VERSION,
)
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    PACKET_SCHEMA,
    PREDICTOR_RESIDUAL_Y_CODEC_ID,
    RECEIVER_CONTRACT_ID,
    SECTION_LENGTH,
    TIE_POLICY_ID,
)
from tac.witness_dsl.v10_production_receiver import (
    MAGIC as PRODUCTION_MAGIC,
)
from tac.witness_dsl.v10_production_receiver import (
    PREFIX as PRODUCTION_PREFIX,
)
from tac.witness_dsl.v10_production_receiver import (
    VERSION as PRODUCTION_VERSION,
)

INDEX_SCHEMA: Final = "ddm_lattice_teacher_solution_index.v1"
PAIR_SCHEMA: Final = "ddm_lattice_teacher_solution_index_pair.v1"
ASSET_SCHEMA: Final = "ddm_lattice_teacher_solution_index_asset.v1"
SENSE_PAIR_SCHEMA: Final = "ddm_min_description_lattice_sense_pair.v1"
SENSE_FACTORIZATION_SCHEMA: Final = "ddm_min_description_lattice_sense_factorization.v1"
MS2R_RECEIPT_SCHEMA: Final = "ddm_ms2r_tolerance_capped_solve_r2_receipt.v1"
MS2R_RECEIVER_CONTRACT: Final = "tac.witness_dsl.v10_production_receiver.v1"
EXPECTED_PAIR_COUNT: Final = 600
EXPECTED_Q4_COUNT: Final = 208
EXPECTED_Q8_COUNT: Final = 392
READ_CHUNK_BYTES: Final = 1 << 20


class LatticeTeacherIndexError(ValueError):
    """A teacher custody, packet, SENSE, or dense-free invariant failed."""


@dataclass(frozen=True, slots=True)
class TeacherAssetSpec:
    path: Path
    sha256: str
    bytes: int
    role: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise LatticeTeacherIndexError(f"{self.role} path must be absolute")
        _require_sha(self.sha256, f"{self.role}.sha256")
        _exact_int(self.bytes, f"{self.role}.bytes", minimum=0)
        _ascii(self.role, "asset role")


@dataclass(frozen=True, slots=True)
class PredictorRecordIndex:
    pair_id: int
    mode_id: str
    bootstrap_bytes: int
    descriptor_bytes: int
    residual_bytes: int
    bootstrap_sha256: str
    descriptor_sha256: str
    residual_sha256: str
    reconstructed_frame1_sha256: str


@dataclass(frozen=True, slots=True)
class PacketScan:
    packet_sha256: str
    packet_bytes: int
    production_header_sha256: str
    y_section_sha256: str
    decoded_two_plane_sha256: str
    pair_count: int
    height: int
    width: int
    channels: int
    records: tuple[PredictorRecordIndex, ...]


def _ascii(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise LatticeTeacherIndexError(f"{label} must be nonempty ASCII")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise LatticeTeacherIndexError(f"{label} must be an exact integer in range")
    return value


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LatticeTeacherIndexError(f"{label} must be a lowercase SHA-256")
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, expected_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
            total += len(chunk)
    if expected_bytes is not None and total != expected_bytes:
        raise LatticeTeacherIndexError(f"{path} byte length drifted while hashing")
    return digest.hexdigest()


def verify_asset(spec: TeacherAssetSpec) -> dict[str, object]:
    path = spec.path
    if not path.is_file() or path.is_symlink():
        raise LatticeTeacherIndexError(f"{spec.role} must be one regular non-symlink file")
    actual_bytes = path.stat().st_size
    if actual_bytes != spec.bytes:
        raise LatticeTeacherIndexError(
            f"{spec.role} byte custody drift: expected {spec.bytes}, observed {actual_bytes}"
        )
    actual_sha = sha256_file(path, expected_bytes=spec.bytes)
    if actual_sha != spec.sha256:
        raise LatticeTeacherIndexError(f"{spec.role} SHA custody drift: expected {spec.sha256}, observed {actual_sha}")
    return {
        "schema": ASSET_SCHEMA,
        "path": str(path),
        "role": spec.role,
        "bytes": spec.bytes,
        "sha256": spec.sha256,
        "verified": True,
        "payload_eligible": False,
    }


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise LatticeTeacherIndexError(f"{label} is truncated")
    return payload


def _hash_region(path: Path, *, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(remaining, READ_CHUNK_BYTES))
            if not chunk:
                raise LatticeTeacherIndexError("packet section truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _parse_production_envelope(
    packet_path: Path,
) -> tuple[Mapping[str, Any], str, int, int]:
    """Return header, header SHA, y-section offset, and y-section length."""

    with packet_path.open("rb") as handle:
        prefix = _read_exact(handle, PRODUCTION_PREFIX.size, "production packet prefix")
        magic, version, header_length = PRODUCTION_PREFIX.unpack(prefix)
        if magic != PRODUCTION_MAGIC or version != PRODUCTION_VERSION:
            raise LatticeTeacherIndexError("production packet magic/version drift")
        if not 0 < header_length <= 1 << 20:
            raise LatticeTeacherIndexError("production header length is outside its cap")
        header_bytes = _read_exact(handle, header_length, "production packet header")
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LatticeTeacherIndexError("production header is not valid JSON") from exc
        if not isinstance(header, Mapping) or canonical_json_bytes(header) != header_bytes:
            raise LatticeTeacherIndexError("production header is not canonical JSON")
        if (
            header.get("schema") != PACKET_SCHEMA
            or header.get("version") != PRODUCTION_VERSION
            or header.get("pair_count") != EXPECTED_PAIR_COUNT
            or header.get("receiver_contract_id") != RECEIVER_CONTRACT_ID
            or header.get("tie_policy_id") != TIE_POLICY_ID
            or header.get("frame0_policy_id") != DESCRIPTION_FRAME0_POLICY_ID
            or header.get("y_codec_id") != PREDICTOR_RESIDUAL_Y_CODEC_ID
            or header.get("residual_codec_id") is not None
            or header.get("section_count") != 2
            or header.get("launch_ready") is not False
            or header.get("score_claim") is not False
            or header.get("promotion_eligible") is not False
        ):
            raise LatticeTeacherIndexError("selected packet production contract drift")
        rows = header.get("sections")
        if not isinstance(rows, list) or len(rows) != 2:
            raise LatticeTeacherIndexError("selected packet section table drift")
        y_row = rows[0]
        policy_row = rows[1]
        if (
            not isinstance(y_row, Mapping)
            or y_row.get("section_id") != "y_description"
            or y_row.get("codec_id") != PREDICTOR_RESIDUAL_Y_CODEC_ID
            or y_row.get("video_derived") is not True
            or not isinstance(policy_row, Mapping)
            or policy_row.get("section_id") != "frame0_policy"
            or policy_row.get("codec_id") != DESCRIPTION_FRAME0_POLICY_ID
            or policy_row.get("byte_length") != 0
            or policy_row.get("video_derived") is not False
        ):
            raise LatticeTeacherIndexError("selected packet section role drift")
        framed_y_length = SECTION_LENGTH.unpack(_read_exact(handle, SECTION_LENGTH.size, "y section length"))[0]
        y_length = _exact_int(y_row.get("byte_length"), "y section byte_length", minimum=1)
        if framed_y_length != y_length:
            raise LatticeTeacherIndexError("y section framed/header length drift")
        y_offset = handle.tell()
        handle.seek(y_length, os.SEEK_CUR)
        framed_policy_length = SECTION_LENGTH.unpack(_read_exact(handle, SECTION_LENGTH.size, "policy section length"))[
            0
        ]
        if framed_policy_length != 0 or handle.read(1):
            raise LatticeTeacherIndexError("selected packet has nonempty policy or trailing bytes")
        if header.get("packet_bytes") != packet_path.stat().st_size:
            raise LatticeTeacherIndexError("selected packet total byte declaration drift")
        expected_y_sha = _require_sha(y_row.get("sha256"), "y section sha256")
        if _hash_region(packet_path, offset=y_offset, length=y_length) != expected_y_sha:
            raise LatticeTeacherIndexError("selected packet y section SHA drift")
        return header, sha256_bytes(header_bytes), y_offset, y_length


def _brotli_decompress(payload: bytes, *, expected_bytes: int, label: str) -> bytes:
    try:
        import brotli  # type: ignore[import-not-found]

        decoded = bytes(brotli.decompress(payload))
    except Exception as exc:
        raise LatticeTeacherIndexError(f"{label} Brotli decode failed") from exc
    if len(decoded) != expected_bytes:
        raise LatticeTeacherIndexError(f"{label} decoded byte geometry drift")
    return decoded


def _scan_predictor_records(
    packet_path: Path,
    *,
    y_offset: int,
    y_length: int,
    decoded_hasher: Any,
    hash_frame: int,
    collect_records: bool,
) -> tuple[tuple[PredictorRecordIndex, ...], tuple[int, int, int, int]]:
    """Strict-reconstruct one pair at a time and optionally retain metadata."""

    if hash_frame not in (0, 1):
        raise LatticeTeacherIndexError("hash_frame must be 0 or 1")
    rows: list[PredictorRecordIndex] = []
    with packet_path.open("rb") as handle:
        handle.seek(y_offset)
        prefix = _read_exact(handle, PREDICTOR_PREFIX.size, "predictor prefix")
        magic, version, codec_tag, pair_count, height, width, channels = PREDICTOR_PREFIX.unpack(prefix)
        if (
            magic != PREDICTOR_MAGIC
            or version != PREDICTOR_VERSION
            or codec_tag != CONTENT_CODEC_TAG
            or pair_count != EXPECTED_PAIR_COUNT
            or channels != 3
        ):
            raise LatticeTeacherIndexError("predictor prefix contract drift")
        plane_bytes = height * width * channels
        residual_decoded_bytes = plane_bytes * 2
        previous_pair = -1
        for pair_index in range(pair_count):
            raw_header = _read_exact(handle, PREDICTOR_PAIR_PREFIX.size, f"pair {pair_index} header")
            (
                pair_id,
                mode_value,
                bootstrap_length,
                descriptor_length,
                residual_length,
                bootstrap_sha,
                descriptor_sha,
                residual_sha,
                reconstructed_sha,
            ) = PREDICTOR_PAIR_PREFIX.unpack(raw_header)
            if pair_id != pair_index or pair_id <= previous_pair:
                raise LatticeTeacherIndexError("predictor pair order is not canonical 0..599")
            previous_pair = pair_id
            try:
                mode = PredictorMode(mode_value)
            except ValueError as exc:
                raise LatticeTeacherIndexError(f"pair {pair_id} has unknown predictor mode") from exc
            expected_descriptor_length = AFFINE6.size if mode is PredictorMode.AFFINE6_Q12 else 0
            if bootstrap_length < 1 or descriptor_length != expected_descriptor_length or residual_length < 1:
                raise LatticeTeacherIndexError(f"pair {pair_id} length/mode contract drift")
            bootstrap = _read_exact(handle, bootstrap_length, f"pair {pair_id} bootstrap")
            descriptor = _read_exact(handle, descriptor_length, f"pair {pair_id} descriptor")
            residual_payload = _read_exact(handle, residual_length, f"pair {pair_id} residual")
            if (
                hashlib.sha256(bootstrap).digest() != bootstrap_sha
                or hashlib.sha256(descriptor).digest() != descriptor_sha
                or hashlib.sha256(residual_payload).digest() != residual_sha
            ):
                raise LatticeTeacherIndexError(f"pair {pair_id} component hash custody failure")
            bootstrap_decoded = _brotli_decompress(
                bootstrap,
                expected_bytes=plane_bytes,
                label=f"pair {pair_id} bootstrap",
            )
            residual_decoded = _brotli_decompress(
                residual_payload,
                expected_bytes=residual_decoded_bytes,
                label=f"pair {pair_id} residual",
            )
            y0 = np.frombuffer(bootstrap_decoded, dtype=np.uint8).reshape(height, width, channels)
            residual = np.frombuffer(residual_decoded, dtype="<i2").reshape(height, width, channels)
            predictor = predict_plane(y0, mode, descriptor)
            reconstructed_i32 = predictor.astype(np.int32) + residual.astype(np.int32)
            if bool(np.any((reconstructed_i32 < 0) | (reconstructed_i32 > 255))):
                raise LatticeTeacherIndexError(f"pair {pair_id} reconstructs outside uint8")
            y1 = np.ascontiguousarray(reconstructed_i32.astype(np.uint8))
            y1_bytes = y1.tobytes(order="C")
            if hashlib.sha256(y1_bytes).digest() != reconstructed_sha:
                raise LatticeTeacherIndexError(f"pair {pair_id} reconstructed frame1 SHA drift")
            decoded_hasher.update(bootstrap_decoded if hash_frame == 0 else y1_bytes)
            if collect_records:
                rows.append(
                    PredictorRecordIndex(
                        pair_id=pair_id,
                        mode_id=MODE_IDS[mode],
                        bootstrap_bytes=bootstrap_length,
                        descriptor_bytes=descriptor_length,
                        residual_bytes=residual_length,
                        bootstrap_sha256=bootstrap_sha.hex(),
                        descriptor_sha256=descriptor_sha.hex(),
                        residual_sha256=residual_sha.hex(),
                        reconstructed_frame1_sha256=reconstructed_sha.hex(),
                    )
                )
        if handle.tell() != y_offset + y_length:
            raise LatticeTeacherIndexError("predictor record scan did not consume y section exactly")
    return tuple(rows), (pair_count, height, width, channels)


def scan_selected_packet(
    packet: TeacherAssetSpec,
) -> PacketScan:
    verify_asset(packet)
    header, header_sha, y_offset, y_length = _parse_production_envelope(packet.path)
    decoded_hasher = hashlib.sha256()
    records, geometry = _scan_predictor_records(
        packet.path,
        y_offset=y_offset,
        y_length=y_length,
        decoded_hasher=decoded_hasher,
        hash_frame=0,
        collect_records=True,
    )
    _, second_geometry = _scan_predictor_records(
        packet.path,
        y_offset=y_offset,
        y_length=y_length,
        decoded_hasher=decoded_hasher,
        hash_frame=1,
        collect_records=False,
    )
    if second_geometry != geometry:
        raise LatticeTeacherIndexError("predictor geometry changed between dense-free passes")
    declared_decoded_sha = _require_sha(header["sections"][0]["decoded_sha256"], "y section decoded_sha256")
    if decoded_hasher.hexdigest() != declared_decoded_sha:
        raise LatticeTeacherIndexError("dense-free two-plane reconstruction SHA drift")
    return PacketScan(
        packet_sha256=packet.sha256,
        packet_bytes=packet.bytes,
        production_header_sha256=header_sha,
        y_section_sha256=_require_sha(header["sections"][0]["sha256"], "y section sha256"),
        decoded_two_plane_sha256=declared_decoded_sha,
        pair_count=geometry[0],
        height=geometry[1],
        width=geometry[2],
        channels=geometry[3],
        records=records,
    )


def _load_json_asset(spec: TeacherAssetSpec) -> tuple[Mapping[str, Any], dict[str, object]]:
    custody = verify_asset(spec)
    try:
        value = json.loads(spec.path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LatticeTeacherIndexError(f"{spec.role} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise LatticeTeacherIndexError(f"{spec.role} must be a JSON object")
    return value, custody


def _selected_steps(receipt: Mapping[str, Any]) -> tuple[int, ...]:
    if receipt.get("schema") != MS2R_RECEIPT_SCHEMA:
        raise LatticeTeacherIndexError("MS2R receipt schema drift")
    authority = receipt.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("research_only") is not True
        or authority.get("execution_allowed") is not False
        or authority.get("score_claim") is not False
        or authority.get("promotion_eligible") is not False
        or authority.get("pointer_moved") is not False
    ):
        raise LatticeTeacherIndexError("MS2R receipt authority drift")
    homotopy = receipt.get("homotopy")
    solve = homotopy.get("solve") if isinstance(homotopy, Mapping) else None
    values = solve.get("selected_steps") if isinstance(solve, Mapping) else None
    if (
        not isinstance(values, list)
        or len(values) != EXPECTED_PAIR_COUNT
        or any(type(value) is not int or value not in (4, 8) for value in values)
    ):
        raise LatticeTeacherIndexError("MS2R selected-step population drift")
    steps = tuple(values)
    if steps.count(4) != EXPECTED_Q4_COUNT or steps.count(8) != EXPECTED_Q8_COUNT:
        raise LatticeTeacherIndexError("MS2R q4/q8 selected population count drift")
    if solve.get("q4_pair_count") != EXPECTED_Q4_COUNT or solve.get("q8_pair_count") != EXPECTED_Q8_COUNT:
        raise LatticeTeacherIndexError("MS2R selected-step declared counts drift")
    _require_sha(solve.get("rows_sha256"), "MS2R solve rows_sha256")
    return steps


def _validate_ms2r_packet_binding(
    receipt: Mapping[str, Any],
    *,
    packet: TeacherAssetSpec,
    records: Sequence[PredictorRecordIndex],
    steps: Sequence[int],
) -> None:
    homotopy = receipt.get("homotopy")
    candidate = homotopy.get("candidate") if isinstance(homotopy, Mapping) else None
    predictor = candidate.get("predictor") if isinstance(candidate, Mapping) else None
    selected_rows = candidate.get("selected_record_rows") if isinstance(candidate, Mapping) else None
    if (
        not isinstance(predictor, Mapping)
        or predictor.get("sha256") != packet.sha256
        or predictor.get("bytes") != packet.bytes
        or predictor.get("receiver_contract") is not None
        or candidate.get("receiver_contract") != MS2R_RECEIVER_CONTRACT
        or candidate.get("score_claim") is not False
        or not isinstance(selected_rows, list)
        or len(selected_rows) != EXPECTED_PAIR_COUNT
    ):
        raise LatticeTeacherIndexError("MS2R receipt selected-packet binding drift")
    for pair_id, (row, record, step) in enumerate(zip(selected_rows, records, steps, strict=True)):
        expected_record_bytes = (
            PREDICTOR_PAIR_PREFIX.size + record.bootstrap_bytes + record.descriptor_bytes + record.residual_bytes
        )
        if (
            not isinstance(row, Mapping)
            or row.get("pair_id") != pair_id
            or row.get("selected_step") != step
            or row.get("record_bytes") != expected_record_bytes
        ):
            raise LatticeTeacherIndexError(f"MS2R receipt selected record {pair_id} binding drift")


def _read_sense_rows(spec: TeacherAssetSpec) -> tuple[tuple[Mapping[str, Any], ...], dict[str, object]]:
    custody = verify_asset(spec)
    rows: list[Mapping[str, Any]] = []
    with spec.path.open("rb") as handle:
        for pair_id, line in enumerate(handle):
            if not line.endswith(b"\n"):
                raise LatticeTeacherIndexError("MS1 SENSE JSONL final/record newline missing")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LatticeTeacherIndexError(f"MS1 SENSE row {pair_id} is invalid JSON") from exc
            if (
                not isinstance(row, Mapping)
                or row.get("schema") != SENSE_PAIR_SCHEMA
                or row.get("pair_id") != pair_id
                or row.get("research_only") is not True
                or row.get("execution_allowed") is not False
                or row.get("promotion_eligible") is not False
                or row.get("score_claim") is not False
            ):
                raise LatticeTeacherIndexError(f"MS1 SENSE row {pair_id} authority/order drift")
            rows.append(row)
    if len(rows) != EXPECTED_PAIR_COUNT:
        raise LatticeTeacherIndexError("MS1 SENSE row count is not 600")
    return tuple(rows), custody


def _validate_ms1_bindings(
    receipt: Mapping[str, Any],
    *,
    sense_rows: TeacherAssetSpec,
    factorization: TeacherAssetSpec,
    factorization_payload: Mapping[str, Any],
) -> None:
    if (
        receipt.get("execution_allowed") is not False
        or receipt.get("research_only") is not True
        or receipt.get("score_claim") is not False
        or receipt.get("promotion_eligible") is not False
    ):
        raise LatticeTeacherIndexError("immutable MS1 receipt authority drift")
    sense = receipt.get("sense")
    pair_binding = sense.get("pair_jsonl") if isinstance(sense, Mapping) else None
    factor_binding = sense.get("factorization") if isinstance(sense, Mapping) else None
    summary = receipt.get("factorization_summary")
    if (
        not isinstance(pair_binding, Mapping)
        or pair_binding.get("bytes") != sense_rows.bytes
        or pair_binding.get("sha256") != sense_rows.sha256
        or not isinstance(factor_binding, Mapping)
        or factor_binding.get("bytes") != factorization.bytes
        or factor_binding.get("sha256") != factorization.sha256
        or not isinstance(summary, Mapping)
        or summary.get("pair_count") != EXPECTED_PAIR_COUNT
        or summary.get("admitted_factor_count") != factorization_payload.get("admitted_factor_count")
        or summary.get("matrix_sha256") != factorization_payload.get("matrix_sha256")
    ):
        raise LatticeTeacherIndexError("immutable MS1 SENSE/factorization binding drift")


def build_solution_index(
    *,
    ms2r_receipt: TeacherAssetSpec,
    selected_packet: TeacherAssetSpec,
    ms1_receipt: TeacherAssetSpec,
    ms1_sense_rows: TeacherAssetSpec,
    ms1_factorization: TeacherAssetSpec,
    additional_encoder_evidence: Sequence[TeacherAssetSpec] = (),
) -> dict[str, object]:
    """Build the real H0 index without retaining any decoded teacher plane."""

    ms2r, ms2r_custody = _load_json_asset(ms2r_receipt)
    steps = _selected_steps(ms2r)
    ms1, ms1_custody = _load_json_asset(ms1_receipt)
    if ms1.get("schema") != "ddm_min_description_lattice_solve_receipt.v1":
        raise LatticeTeacherIndexError("immutable MS1 receipt schema drift")
    sense_rows, sense_custody = _read_sense_rows(ms1_sense_rows)
    factorization, factor_custody = _load_json_asset(ms1_factorization)
    if (
        factorization.get("schema") != SENSE_FACTORIZATION_SCHEMA
        or factorization.get("pair_count") != EXPECTED_PAIR_COUNT
        or factorization.get("execution_allowed") is not False
        or factorization.get("research_only") is not True
        or factorization.get("score_claim") is not False
    ):
        raise LatticeTeacherIndexError("MS1 factorization authority/population drift")
    _validate_ms1_bindings(
        ms1,
        sense_rows=ms1_sense_rows,
        factorization=ms1_factorization,
        factorization_payload=factorization,
    )
    packet_scan = scan_selected_packet(selected_packet)
    _validate_ms2r_packet_binding(
        ms2r,
        packet=selected_packet,
        records=packet_scan.records,
        steps=steps,
    )
    pair_rows: list[dict[str, object]] = []
    leaves: list[str] = []
    for record, step, sense in zip(packet_scan.records, steps, sense_rows, strict=True):
        rate = sense.get("rate")
        if not isinstance(rate, Mapping):
            raise LatticeTeacherIndexError(f"sense row {record.pair_id} rate must be an object")
        sense_digest = sha256_bytes(canonical_json_bytes(sense))
        row: dict[str, object] = {
            "schema": PAIR_SCHEMA,
            "pair_id": record.pair_id,
            "selected_quantum": step,
            "predictor": {
                "mode_id": record.mode_id,
                "bootstrap_bytes": record.bootstrap_bytes,
                "descriptor_bytes": record.descriptor_bytes,
                "residual_bytes": record.residual_bytes,
                "bootstrap_sha256": record.bootstrap_sha256,
                "descriptor_sha256": record.descriptor_sha256,
                "residual_sha256": record.residual_sha256,
                "reconstructed_frame1_sha256": record.reconstructed_frame1_sha256,
            },
            "sense_row_sha256": sense_digest,
            "sense_summary": {
                "origin_sha256": _require_sha(sense.get("origin_sha256"), "sense origin_sha256"),
                "selected_sha256": _require_sha(sense.get("selected_sha256"), "sense selected_sha256"),
                "residual_sha256": _require_sha(sense.get("residual_sha256"), "sense residual_sha256"),
                "canonical_member_bytes": _exact_int(
                    rate.get("canonical_member_bytes"),
                    f"sense row {record.pair_id} canonical_member_bytes",
                ),
                "selected_residual_bytes": _exact_int(
                    rate.get("selected_residual_bytes"),
                    f"sense row {record.pair_id} selected_residual_bytes",
                ),
                "rate_delta_bytes": _exact_int(
                    rate.get("delta_bytes"),
                    f"sense row {record.pair_id} delta_bytes",
                    minimum=-(1 << 62),
                    maximum=(1 << 62) - 1,
                ),
            },
            "teacher_bytes_retained": False,
            "candidate_payload_eligible": False,
        }
        leaf = sha256_bytes(canonical_json_bytes(row))
        row["leaf_sha256"] = leaf
        leaves.append(leaf)
        pair_rows.append(row)
    extra_custody = [verify_asset(spec) for spec in additional_encoder_evidence]
    content_root = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": INDEX_SCHEMA,
                "pair_leaf_sha256": leaves,
                "factorization_sha256": ms1_factorization.sha256,
                "selected_packet_sha256": selected_packet.sha256,
                "ms1_receipt_sha256": ms1_receipt.sha256,
                "ms2r_receipt_sha256": ms2r_receipt.sha256,
            }
        )
    )
    return {
        "schema": INDEX_SCHEMA,
        "lane_id": "lane_g100_lattice_teacher_compaction_takeoff_20260727",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload_created": False,
        "historical_payload_reused": False,
        "dense_teacher_bytes_persisted": 0,
        "teacher_role": "encoder_evidence_only",
        "pair_count": EXPECTED_PAIR_COUNT,
        "q4_pair_count": steps.count(4),
        "q8_pair_count": steps.count(8),
        "content_root_sha256": content_root,
        "packet": {
            "sha256": packet_scan.packet_sha256,
            "bytes": packet_scan.packet_bytes,
            "production_header_sha256": packet_scan.production_header_sha256,
            "y_section_sha256": packet_scan.y_section_sha256,
            "decoded_two_plane_sha256": packet_scan.decoded_two_plane_sha256,
            "geometry": [
                packet_scan.pair_count,
                packet_scan.height,
                packet_scan.width,
                packet_scan.channels,
            ],
            "strict_pair_reconstruction": True,
            "decoder_peak_population_rows": 1,
        },
        "factorization": {
            "sha256": ms1_factorization.sha256,
            "matrix_sha256": _require_sha(factorization.get("matrix_sha256"), "factorization matrix_sha256"),
            "admitted_factor_count": _exact_int(
                factorization.get("admitted_factor_count"),
                "factorization admitted_factor_count",
            ),
            "proposal_only": True,
        },
        "assets": [
            ms2r_custody,
            ms1_custody,
            sense_custody,
            factor_custody,
            verify_asset(selected_packet),
            *extra_custody,
        ],
        "pairs": pair_rows,
        "continuation_blockers": [
            "CURRENT_G17_SELECTED_SOLUTION_TEACHER_OWED",
            "CURRENT_RECEIVER_ACTUATOR_FOREIGN_KEYS_OWED",
            "R10_DECODER_COMPUTABLE_FEATURE_RELAY_OWED",
            "POPULATION_GLOBAL_SAME_SOLUTION_CODEC_MEASUREMENT_OWED",
        ],
    }


__all__ = [
    "ASSET_SCHEMA",
    "EXPECTED_PAIR_COUNT",
    "INDEX_SCHEMA",
    "PAIR_SCHEMA",
    "LatticeTeacherIndexError",
    "PacketScan",
    "PredictorRecordIndex",
    "TeacherAssetSpec",
    "build_solution_index",
    "canonical_json_bytes",
    "scan_selected_packet",
    "sha256_file",
    "verify_asset",
]
