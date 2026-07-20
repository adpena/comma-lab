# SPDX-License-Identifier: MIT
"""Versioned, search-free receiver for the counted R1b2 extension grammar.

The receiver reconstructs the inherited C2 archive byte-for-byte, delegates
that archive to the hash-bound C2 decoder, and then consumes each R1b2 section
in a fixed order.  It contains no scorer, optimizer, target table, or search
loop.  The only video-derived state it accepts is present in the counted ZIP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
import zipfile
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from tac.boundary_math.integer_plane_emitter import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    factor2_operator,
)
from tac.boundary_math.integer_plane_emitter_byte_close import (
    decode_counted_archive,
    parse_counted_archive,
)
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryJointSolveError,
    decode_boundary_packet,
    selected_frame_features,
)
from tac.optimization.r1b3_producer_preflight import (
    R1B3ProducerError,
    decode_xi0_payload,
)
from tac.optimization.uint8_lattice_feasibility import (
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

ARCHIVE_SCHEMA: Final = "r1b2_counted_archive.v1"
RECEIVER_SCHEMA: Final = "r1b4_section_receiver.v1"
REPLAY_SCHEMA: Final = "r1b4_compact_replay.v1"
REPLAY_MAGIC: Final = b"R1K1"
MANIFEST_NAME: Final = "r1b2_manifest.json"
BOUNDARY_NAME: Final = "boundary_coordinate.bgj"
REPLAY_NAME: Final = "full_kernel_replay.r1k"
XI0_NAME: Final = "xi0.xi0"
EXTENSION_NAMES: Final = (MANIFEST_NAME, BOUNDARY_NAME, REPLAY_NAME, XI0_NAME)
APPLICATION_ORDER: Final = ("c2_base", BOUNDARY_NAME, XI0_NAME, REPLAY_NAME)
PAIR_COUNT: Final = 600
DECODE_LIMIT_SECONDS: Final = 1_800.0

_REPLAY_PREFIX: Final = struct.Struct("<4sII")
_REPLAY_ENTRY: Final = struct.Struct("<HBBHHBB")
_REPLAY_CRC: Final = struct.Struct("<I")
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "pair_count",
        "artifact_role",
        "base_archive_sha256",
        "base_archive_bytes",
        "base_zip_compressed_bytes",
        "base_zip_overhead_bytes",
        "base_sections",
        "sections",
        "source_manifest_hashes",
        "offline_full_kernel_selection",
        "receiver_search",
        "xi_coordinate_indices",
        "receiver_schema",
        "receiver_policy",
        "application_order",
        "final_output_assertion",
        "score_claim",
    }
)


class R1B4ReceiverError(ValueError):
    """Malformed archive, inactive section, or receiver custody violation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise R1B4ReceiverError("value is not canonical-JSON encodable") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def default_receiver_policy() -> dict[str, Any]:
    """Return the sealed, scorer-free xi[0] horizontal-warp policy."""

    return {
        "schema": "r1b4_receiver_policy.v1",
        "xi0_actuator": {
            "schema": "r1b4_xi0_integer_horizontal_translation.v1",
            "center": 31.0,
            "pixels_per_unit": 1.0,
            "maximum_absolute_pixels": 16,
            "rounding": "numpy_rint_ties_to_even",
            "fill": "edge_replication",
            "frame_index": 0,
        },
        "boundary_frame_index": 1,
        "replay_semantics": "ordered_absolute_camera_uint8_assignment_v1",
        "receiver_search_invocations": 0,
    }


def _validate_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "xi0_actuator",
        "boundary_frame_index",
        "replay_semantics",
        "receiver_search_invocations",
    }:
        raise R1B4ReceiverError("receiver policy fields mismatch")
    actuator = value["xi0_actuator"]
    if not isinstance(actuator, dict) or set(actuator) != {
        "schema",
        "center",
        "pixels_per_unit",
        "maximum_absolute_pixels",
        "rounding",
        "fill",
        "frame_index",
    }:
        raise R1B4ReceiverError("xi0 actuator policy fields mismatch")
    center = actuator["center"]
    gain = actuator["pixels_per_unit"]
    limit = actuator["maximum_absolute_pixels"]
    if (
        value["schema"] != "r1b4_receiver_policy.v1"
        or value["boundary_frame_index"] != 1
        or value["replay_semantics"] != "ordered_absolute_camera_uint8_assignment_v1"
        or value["receiver_search_invocations"] != 0
        or actuator["schema"] != "r1b4_xi0_integer_horizontal_translation.v1"
        or actuator["rounding"] != "numpy_rint_ties_to_even"
        or actuator["fill"] != "edge_replication"
        or actuator["frame_index"] != 0
        or isinstance(center, bool)
        or not isinstance(center, (int, float))
        or not math.isfinite(float(center))
        or isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not math.isfinite(float(gain))
        or float(gain) == 0.0
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= CAMERA_WIDTH
    ):
        raise R1B4ReceiverError("receiver policy sealed values mismatch")
    return value


@dataclass(frozen=True, order=True, slots=True)
class ReplayWrite:
    pair_index: int
    frame_index: int
    y: int
    x: int
    channel: int
    value: int

    def __post_init__(self) -> None:
        values = (
            self.pair_index,
            self.frame_index,
            self.y,
            self.x,
            self.channel,
            self.value,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise R1B4ReceiverError("replay coordinates and value must be integers")
        if not 0 <= self.pair_index < PAIR_COUNT:
            raise R1B4ReceiverError("replay pair index is outside [0,600)")
        if self.frame_index not in range(PLANE_COUNT):
            raise R1B4ReceiverError("replay frame index must be 0 or 1")
        if not 0 <= self.y < CAMERA_HEIGHT or not 0 <= self.x < CAMERA_WIDTH:
            raise R1B4ReceiverError("replay camera coordinate is out of range")
        if not 0 <= self.channel < RGB_CHANNELS or not 0 <= self.value <= 255:
            raise R1B4ReceiverError("replay channel/value is out of range")


def encode_replay_payload(writes: Sequence[ReplayWrite]) -> bytes:
    """Encode canonical ordered absolute camera-byte assignments."""

    rows = tuple(writes)
    if any(not isinstance(row, ReplayWrite) for row in rows):
        raise R1B4ReceiverError("replay rows must be ReplayWrite values")
    if tuple(sorted(rows)) != rows or len(set(rows)) != len(rows):
        raise R1B4ReceiverError("replay rows must be unique and canonically sorted")
    coordinates = [(row.pair_index, row.frame_index, row.y, row.x, row.channel) for row in rows]
    if len(coordinates) != len(set(coordinates)):
        raise R1B4ReceiverError("replay writes duplicate a camera coordinate")
    body = b"".join(
        _REPLAY_ENTRY.pack(
            row.pair_index,
            row.frame_index,
            row.channel,
            row.y,
            row.x,
            row.value,
            0,
        )
        for row in rows
    )
    header = canonical_json(
        {
            "schema": REPLAY_SCHEMA,
            "pair_count": PAIR_COUNT,
            "entry_count": len(rows),
            "entry_bytes": _REPLAY_ENTRY.size,
            "body_bytes": len(body),
            "body_sha256": _sha256(body),
            "search_required": False,
            "score_claim": False,
        }
    )
    prefix = _REPLAY_PREFIX.pack(REPLAY_MAGIC, len(header), len(body))
    crc = _REPLAY_CRC.pack(zlib.crc32(header + body) & 0xFFFFFFFF)
    return prefix + header + body + crc


def decode_replay_payload(payload: bytes) -> tuple[ReplayWrite, ...]:
    """Strictly decode a compact replay and require exact final-byte consumption."""

    if not isinstance(payload, bytes) or len(payload) < _REPLAY_PREFIX.size + _REPLAY_CRC.size:
        raise R1B4ReceiverError("compact replay is truncated")
    magic, header_size, body_size = _REPLAY_PREFIX.unpack_from(payload)
    if magic != REPLAY_MAGIC:
        raise R1B4ReceiverError("compact replay magic mismatch")
    expected = _REPLAY_PREFIX.size + header_size + body_size + _REPLAY_CRC.size
    if len(payload) != expected:
        raise R1B4ReceiverError("compact replay length mismatch or trailing bytes")
    header_start = _REPLAY_PREFIX.size
    body_start = header_start + header_size
    body_end = body_start + body_size
    header_bytes = payload[header_start:body_start]
    body = payload[body_start:body_end]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B4ReceiverError("compact replay header is invalid") from exc
    if not isinstance(header, dict) or canonical_json(header) != header_bytes:
        raise R1B4ReceiverError("compact replay header is noncanonical")
    entry_count = header.get("entry_count")
    if isinstance(entry_count, bool) or not isinstance(entry_count, int) or entry_count < 0:
        raise R1B4ReceiverError("compact replay entry count is invalid")
    expected_header = {
        "schema": REPLAY_SCHEMA,
        "pair_count": PAIR_COUNT,
        "entry_count": entry_count,
        "entry_bytes": _REPLAY_ENTRY.size,
        "body_bytes": entry_count * _REPLAY_ENTRY.size,
        "body_sha256": _sha256(body),
        "search_required": False,
        "score_claim": False,
    }
    if header != expected_header or len(body) != entry_count * _REPLAY_ENTRY.size:
        raise R1B4ReceiverError("compact replay sealed header values mismatch")
    (stored_crc,) = _REPLAY_CRC.unpack(payload[body_end:])
    if stored_crc != (zlib.crc32(header_bytes + body) & 0xFFFFFFFF):
        raise R1B4ReceiverError("compact replay CRC mismatch")
    rows: list[ReplayWrite] = []
    for offset in range(0, len(body), _REPLAY_ENTRY.size):
        pair, frame, channel, y, x, value, reserved = _REPLAY_ENTRY.unpack_from(body, offset)
        if reserved != 0:
            raise R1B4ReceiverError("compact replay reserved byte is nonzero")
        rows.append(ReplayWrite(pair, frame, y, x, channel, value))
    result = tuple(rows)
    if encode_replay_payload(result) != payload:
        raise R1B4ReceiverError("compact replay is noncanonical")
    return result


@dataclass(frozen=True, slots=True)
class ParsedR1B4Archive:
    path: Path
    archive_bytes: int
    archive_sha256: str
    base_members: tuple[tuple[str, bytes], ...]
    manifest: dict[str, Any]
    boundary_payload: bytes
    replay_payload: bytes
    replay_writes: tuple[ReplayWrite, ...]
    xi0_payload: bytes
    xi0_values: np.ndarray


def _zip_eof_offset(path: Path) -> int:
    size = path.stat().st_size
    tail_size = min(size, 65_557)
    with path.open("rb") as handle:
        handle.seek(size - tail_size)
        tail = handle.read()
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(tail):
        raise R1B4ReceiverError("archive has no terminal ZIP EOCD")
    comment_length = int.from_bytes(tail[marker + 20 : marker + 22], "little")
    return size - tail_size + marker + 22 + comment_length


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_zip(path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    if path.exists():
        raise R1B4ReceiverError(f"archive overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise R1B4ReceiverError(f"stale archive temporary requires review: {partial}")
    try:
        with zipfile.ZipFile(partial, "x", allowZip64=False) as archive:
            for name, payload in members:
                archive.writestr(
                    _zip_info(name),
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def _base_zip_metrics(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path, "r") as archive:
        compressed = sum(info.compress_size for info in archive.infolist())
    return compressed, path.stat().st_size - compressed


def _unsealed_assertion(*, pair_cap: int) -> dict[str, Any]:
    return {
        "status": "unsealed",
        "pair_cap": pair_cap,
        "decoded_bytes": pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS,
        "decoded_sha256": None,
    }


def build_r1b4_archive(
    *,
    base_archive: Path,
    boundary_payload: bytes,
    replay_payload: bytes,
    xi0_payload: bytes,
    source_manifest_hashes: Mapping[str, str],
    output: Path,
    artifact_role: str = "receiver_smoke_only",
    receiver_policy: Mapping[str, Any] | None = None,
    pair_cap: int = PAIR_COUNT,
) -> dict[str, Any]:
    """Assemble deterministic R1b4 bytes with an explicitly unsealed output assertion."""

    base = base_archive.expanduser().resolve(strict=True)
    parsed_base = parse_counted_archive(base)
    if not 2 <= pair_cap <= PAIR_COUNT:
        raise R1B4ReceiverError("pair_cap must be in [2,600]")
    if artifact_role not in {"receiver_smoke_only", "r1b2_candidate"}:
        raise R1B4ReceiverError("artifact role is invalid")
    if artifact_role == "r1b2_candidate" and pair_cap != PAIR_COUNT:
        raise R1B4ReceiverError("production candidate assertion must cover all 600 pairs")
    policy = _validate_policy(dict(receiver_policy or default_receiver_policy()))
    try:
        packet = decode_boundary_packet(boundary_payload)
        writes = decode_replay_payload(replay_payload)
        xi0 = decode_xi0_payload(xi0_payload)
    except (BoundaryJointSolveError, R1B3ProducerError) as exc:
        raise R1B4ReceiverError("R1b4 semantic section is invalid") from exc
    if packet.pair_count != PAIR_COUNT or xi0.shape != (PAIR_COUNT,):
        raise R1B4ReceiverError("R1b4 sections must cover exactly 600 pairs")
    if packet.scorer_height != SCORER_HEIGHT or packet.scorer_width != SCORER_WIDTH:
        raise R1B4ReceiverError("boundary packet scorer geometry mismatch")
    if not source_manifest_hashes or any(not _is_sha256(value) for value in source_manifest_hashes.values()):
        raise R1B4ReceiverError("source-manifest hashes must be nonempty SHA-256 strings")
    with zipfile.ZipFile(base, "r") as archive:
        base_members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    base_compressed, base_overhead = _base_zip_metrics(base)
    sections = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in (
            (BOUNDARY_NAME, boundary_payload),
            (REPLAY_NAME, replay_payload),
            (XI0_NAME, xi0_payload),
        )
    }
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "pair_count": PAIR_COUNT,
        "artifact_role": artifact_role,
        "base_archive_sha256": parsed_base.archive_sha256,
        "base_archive_bytes": parsed_base.archive_bytes,
        "base_zip_compressed_bytes": base_compressed,
        "base_zip_overhead_bytes": base_overhead,
        "base_sections": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in base_members
        },
        "sections": sections,
        "source_manifest_hashes": dict(source_manifest_hashes),
        "offline_full_kernel_selection": True,
        "receiver_search": False,
        "xi_coordinate_indices": [0],
        "receiver_schema": RECEIVER_SCHEMA,
        "receiver_policy": policy,
        "application_order": list(APPLICATION_ORDER),
        "final_output_assertion": _unsealed_assertion(pair_cap=pair_cap),
        "score_claim": False,
    }
    members = [
        *base_members,
        (MANIFEST_NAME, canonical_json(manifest)),
        (BOUNDARY_NAME, boundary_payload),
        (REPLAY_NAME, replay_payload),
        (XI0_NAME, xi0_payload),
    ]
    resolved_output = output.expanduser().resolve()
    _write_zip(resolved_output, members)
    parsed = parse_r1b4_archive(resolved_output)
    return {
        "archive": str(parsed.path),
        "archive_bytes": parsed.archive_bytes,
        "archive_sha256": parsed.archive_sha256,
        "artifact_role": artifact_role,
        "replay_entry_count": len(writes),
        "xi0_value_count": int(xi0.size),
        "output_assertion_status": "unsealed",
    }


def parse_r1b4_archive(path: Path) -> ParsedR1B4Archive:
    """Strictly parse the inherited C2 grammar and all four R1b2 members."""

    resolved = path.expanduser().resolve(strict=True)
    if _zip_eof_offset(resolved) != resolved.stat().st_size:
        raise R1B4ReceiverError("R1b4 archive carries trailing bytes")
    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(names) != len(set(names))
                or any(
                    info.is_dir()
                    or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or not _safe_member_name(info.filename)
                    for info in infos
                )
            ):
                raise R1B4ReceiverError("unsafe, duplicate, encrypted, or non-deflated archive member")
            if len(names) <= len(EXTENSION_NAMES) or names[-4:] != list(EXTENSION_NAMES):
                raise R1B4ReceiverError("R1b4 extension member order mismatch")
            members = [(info.filename, archive.read(info)) for info in infos]
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise R1B4ReceiverError("R1b4 archive ZIP parse failed") from exc
    payloads = dict(members)
    manifest_raw = payloads[MANIFEST_NAME]
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B4ReceiverError("R1b4 manifest is invalid") from exc
    if not isinstance(manifest, dict) or canonical_json(manifest) != manifest_raw:
        raise R1B4ReceiverError("R1b4 manifest is noncanonical")
    if set(manifest) != _MANIFEST_FIELDS:
        raise R1B4ReceiverError("R1b4 manifest fields mismatch")
    if (
        manifest["schema"] != ARCHIVE_SCHEMA
        or manifest["pair_count"] != PAIR_COUNT
        or manifest["artifact_role"] not in {"receiver_smoke_only", "r1b2_candidate"}
        or manifest["receiver_schema"] != RECEIVER_SCHEMA
        or manifest["receiver_search"] is not False
        or manifest["xi_coordinate_indices"] != [0]
        or manifest["application_order"] != list(APPLICATION_ORDER)
        or manifest["offline_full_kernel_selection"] is not True
        or manifest["score_claim"] is not False
    ):
        raise R1B4ReceiverError("R1b4 manifest sealed values mismatch")
    _validate_policy(manifest["receiver_policy"])
    base_members = tuple(members[: -len(EXTENSION_NAMES)])
    actual_base = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)} for name, payload in base_members
    }
    if manifest["base_sections"] != actual_base:
        raise R1B4ReceiverError("R1b4 inherited base-section custody mismatch")
    source_hashes = manifest["source_manifest_hashes"]
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        raise R1B4ReceiverError("R1b4 source-manifest hash custody is malformed")
    sections = manifest["sections"]
    if not isinstance(sections, dict) or set(sections) != {BOUNDARY_NAME, REPLAY_NAME, XI0_NAME}:
        raise R1B4ReceiverError("R1b4 section manifest fields mismatch")
    for name in (BOUNDARY_NAME, REPLAY_NAME, XI0_NAME):
        actual = {"bytes": len(payloads[name]), "sha256": _sha256(payloads[name])}
        if sections[name] != actual:
            raise R1B4ReceiverError(f"R1b4 section custody mismatch: {name}")
    try:
        packet = decode_boundary_packet(payloads[BOUNDARY_NAME])
        replay = decode_replay_payload(payloads[REPLAY_NAME])
        xi0 = decode_xi0_payload(payloads[XI0_NAME])
    except (BoundaryJointSolveError, R1B3ProducerError) as exc:
        raise R1B4ReceiverError("R1b4 semantic section parse failed") from exc
    if (
        packet.pair_count != PAIR_COUNT
        or packet.scorer_height != SCORER_HEIGHT
        or packet.scorer_width != SCORER_WIDTH
        or xi0.shape != (PAIR_COUNT,)
        or not np.all(np.isfinite(xi0))
    ):
        raise R1B4ReceiverError("R1b4 semantic section geometry mismatch")

    with tempfile.TemporaryDirectory(prefix="r1b4_parse_base_") as temp_name:
        base_path = Path(temp_name) / "base.zip"
        _write_zip(base_path, base_members)
        parsed_base = parse_counted_archive(base_path)
        if (
            parsed_base.archive_bytes != manifest["base_archive_bytes"]
            or parsed_base.archive_sha256 != manifest["base_archive_sha256"]
        ):
            raise R1B4ReceiverError("R1b4 reconstructed base archive custody mismatch")
        compressed, overhead = _base_zip_metrics(base_path)
        if (
            compressed != manifest["base_zip_compressed_bytes"]
            or overhead != manifest["base_zip_overhead_bytes"]
        ):
            raise R1B4ReceiverError("R1b4 base ZIP accounting mismatch")
    assertion = manifest["final_output_assertion"]
    if not isinstance(assertion, dict) or set(assertion) != {
        "status",
        "pair_cap",
        "decoded_bytes",
        "decoded_sha256",
    }:
        raise R1B4ReceiverError("R1b4 output assertion fields mismatch")
    pair_cap = assertion["pair_cap"]
    expected_bytes = pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    if (
        isinstance(pair_cap, bool)
        or not isinstance(pair_cap, int)
        or not 2 <= pair_cap <= PAIR_COUNT
        or assertion["decoded_bytes"] != expected_bytes
        or assertion["status"] not in {"unsealed", "sealed"}
        or (
            assertion["decoded_sha256"] is not None
            if assertion["status"] == "unsealed"
            else not _is_sha256(assertion["decoded_sha256"])
        )
    ):
        raise R1B4ReceiverError("R1b4 output assertion sealed values mismatch")
    if manifest["artifact_role"] == "r1b2_candidate" and pair_cap != PAIR_COUNT:
        raise R1B4ReceiverError("R1b2 candidate output assertion is not n600")
    return ParsedR1B4Archive(
        resolved,
        resolved.stat().st_size,
        sha256_file(resolved),
        base_members,
        manifest,
        payloads[BOUNDARY_NAME],
        payloads[REPLAY_NAME],
        replay,
        payloads[XI0_NAME],
        xi0,
    )


def seal_output_assertion(
    archive: Path,
    *,
    decoded_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Reseal a discovery decode hash without changing any actuation policy."""

    parsed = parse_r1b4_archive(archive)
    assertion = parsed.manifest["final_output_assertion"]
    if assertion["status"] != "unsealed":
        raise R1B4ReceiverError("only an unsealed archive may be output-sealed")
    decoded = decoded_path.expanduser().resolve(strict=True)
    if decoded.stat().st_size != assertion["decoded_bytes"]:
        raise R1B4ReceiverError("discovery decoded byte count does not match assertion")
    manifest = dict(parsed.manifest)
    manifest["final_output_assertion"] = {
        **assertion,
        "status": "sealed",
        "decoded_sha256": sha256_file(decoded),
    }
    members = [
        *parsed.base_members,
        (MANIFEST_NAME, canonical_json(manifest)),
        (BOUNDARY_NAME, parsed.boundary_payload),
        (REPLAY_NAME, parsed.replay_payload),
        (XI0_NAME, parsed.xi0_payload),
    ]
    resolved_output = output.expanduser().resolve()
    _write_zip(resolved_output, members)
    sealed = parse_r1b4_archive(resolved_output)
    return {
        "archive": str(sealed.path),
        "archive_bytes": sealed.archive_bytes,
        "archive_sha256": sealed.archive_sha256,
        "decoded_bytes": decoded.stat().st_size,
        "decoded_sha256": manifest["final_output_assertion"]["decoded_sha256"],
        "status": "sealed",
    }


def _project_camera_to_scorer(camera: np.ndarray) -> np.ndarray:
    operator = factor2_operator()
    numerator, denominator = operator.apply_numerators(camera)
    return np.clip(np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0).astype(np.uint8)


def _boundary_target(
    baseline: np.ndarray,
    *,
    features: np.ndarray,
    coefficients: np.ndarray,
    scale: float,
) -> np.ndarray:
    delta = features @ coefficients.astype(np.float64)
    target = baseline.astype(np.float64) + np.rint(delta * scale).reshape(
        SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS
    )
    return np.clip(target, 0.0, 255.0).astype(np.uint8)


def _translate_frame0(frame: np.ndarray, value: float, actuator: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    shift = int(
        np.rint((float(value) - float(actuator["center"])) * float(actuator["pixels_per_unit"]))
    )
    shift = max(-int(actuator["maximum_absolute_pixels"]), min(int(actuator["maximum_absolute_pixels"]), shift))
    if shift == 0:
        return frame.copy(), shift
    out = np.empty_like(frame)
    if shift > 0:
        out[:, shift:] = frame[:, :-shift]
        out[:, :shift] = frame[:, :1]
    else:
        width = -shift
        out[:, :-width] = frame[:, width:]
        out[:, -width:] = frame[:, -1:]
    return out, shift


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R1B4ReceiverError(f"receiver receipt overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise R1B4ReceiverError(f"stale receiver receipt temporary requires review: {partial}")
    partial_owned = False
    try:
        with partial.open("xb") as handle:
            partial_owned = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial_owned and partial.exists():
            partial.unlink()


def decode_r1b4_archive(
    *,
    archive: Path,
    base_decoder: Path,
    scratch_root: Path,
    output_raw: Path,
    receipt_path: Path,
    workers: int = 1,
    allow_unsealed_discovery: bool = False,
) -> dict[str, Any]:
    """Decode exact archive bytes, assert final custody, and atomically promote output."""

    started = time.monotonic()
    parsed = parse_r1b4_archive(archive)
    assertion = parsed.manifest["final_output_assertion"]
    pair_cap = int(assertion["pair_cap"])
    if assertion["status"] != "sealed" and not allow_unsealed_discovery:
        raise R1B4ReceiverError("production decode refuses an unsealed final-output assertion")
    output = output_raw.expanduser().resolve()
    receipt = receipt_path.expanduser().resolve()
    decoder = base_decoder.expanduser().resolve(strict=True)
    scratch = scratch_root.expanduser().resolve()
    if output.exists() or receipt.exists():
        raise R1B4ReceiverError("receiver output/receipt overwrite refused")
    scratch.mkdir(parents=True, exist_ok=True)
    policy = parsed.manifest["receiver_policy"]
    packet = decode_boundary_packet(parsed.boundary_payload)
    features = selected_frame_features(packet)
    replay_by_pair: dict[int, list[ReplayWrite]] = defaultdict(list)
    for row in parsed.replay_writes:
        if row.pair_index < pair_cap:
            replay_by_pair[row.pair_index].append(row)
    expected_shape = (pair_cap, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
    expected_bytes = int(np.prod(expected_shape))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise R1B4ReceiverError(f"stale receiver partial requires review: {partial}")
    boundary_changed_bytes = 0
    boundary_changed_pairs = 0
    xi0_changed_bytes = 0
    xi0_changed_pairs = 0
    xi0_shifts: list[int] = []
    replay_changed_bytes = 0
    replay_effective_entries = 0
    output_promoted = False
    receipt_written = False
    try:
        with tempfile.TemporaryDirectory(prefix="r1b4_receiver_", dir=scratch) as temp_name:
            temp = Path(temp_name)
            base_archive_path = temp / "base.zip"
            base_raw = temp / "base.raw"
            base_receipt = temp / "base_decode_receipt.json"
            _write_zip(base_archive_path, parsed.base_members)
            parsed_base = parse_counted_archive(base_archive_path)
            if parsed_base.archive_sha256 != parsed.manifest["base_archive_sha256"]:
                raise R1B4ReceiverError("decode-time reconstructed base archive hash drift")
            if sha256_file(decoder) != parsed_base.manifest["base_decoder_sha256"]:
                raise R1B4ReceiverError("pinned base decoder SHA-256 drift")
            base_decode = decode_counted_archive(
                archive=base_archive_path,
                base_decoder=decoder,
                scratch_root=temp,
                pair_cap=pair_cap,
                output_raw=base_raw,
                workers=workers,
            )
            _atomic_json(base_receipt, base_decode)
            base = np.memmap(base_raw, mode="r", dtype=np.uint8, shape=expected_shape)
            final = np.memmap(partial, mode="w+", dtype=np.uint8, shape=expected_shape)
            operator = factor2_operator()
            for pair_index in range(pair_cap):
                base_f0 = np.asarray(base[pair_index, 0])
                base_f1 = np.asarray(base[pair_index, 1])
                frame0, shift = _translate_frame0(
                    base_f0,
                    float(parsed.xi0_values[pair_index]),
                    policy["xi0_actuator"],
                )
                xi0_shifts.append(shift)
                xi_delta = int(np.count_nonzero(frame0 != base_f0))
                xi0_changed_bytes += xi_delta
                xi0_changed_pairs += int(xi_delta > 0)

                baseline_scorer = _project_camera_to_scorer(base_f1)
                target = _boundary_target(
                    baseline_scorer,
                    features=features,
                    coefficients=packet.coefficients[pair_index],
                    scale=float(packet.scales[pair_index]),
                )
                frame1 = realize_factor2_uint8_scorer_plane(operator, target)
                verification = verify_factor2_uint8_scorer_plane(operator, frame1, target)
                if not verification.numerator_exact or not verification.certified_exact:
                    raise R1B4ReceiverError(f"boundary factor-2 proof failed at pair {pair_index}")
                boundary_delta = int(np.count_nonzero(frame1 != base_f1))
                boundary_changed_bytes += boundary_delta
                boundary_changed_pairs += int(boundary_delta > 0)

                frames = [frame0, frame1]
                for row in replay_by_pair.get(pair_index, []):
                    before = int(frames[row.frame_index][row.y, row.x, row.channel])
                    frames[row.frame_index][row.y, row.x, row.channel] = row.value
                    if before != row.value:
                        replay_effective_entries += 1
                        replay_changed_bytes += 1
                final[pair_index, 0] = frames[0]
                final[pair_index, 1] = frames[1]
            final.flush()
            del final, base
            with partial.open("rb") as handle:
                os.fsync(handle.fileno())
            actual_bytes = partial.stat().st_size
            actual_sha = sha256_file(partial)
            if actual_bytes != expected_bytes or actual_bytes != assertion["decoded_bytes"]:
                raise R1B4ReceiverError("receiver final decoded byte count mismatch")
            if assertion["status"] == "sealed" and actual_sha != assertion["decoded_sha256"]:
                raise R1B4ReceiverError("receiver final decoded SHA-256 mismatch")
            if parsed.replay_writes and replay_effective_entries == 0:
                raise R1B4ReceiverError("nonempty compact replay was operationally inert")
            if xi0_changed_pairs == 0:
                raise R1B4ReceiverError("xi0 actuator was operationally inert")
            if (
                parsed.manifest["artifact_role"] == "r1b2_candidate"
                and boundary_changed_pairs == 0
            ):
                raise R1B4ReceiverError("production boundary packet was operationally inert")
            elapsed = time.monotonic() - started
            if elapsed > DECODE_LIMIT_SECONDS:
                raise R1B4ReceiverError("receiver decode exceeded 1800 seconds")
            os.replace(partial, output)
            output_promoted = True
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            result = {
                "schema": "r1b4_receiver_decode_receipt.v1",
                "archive": {
                    "path": str(parsed.path),
                    "bytes": parsed.archive_bytes,
                    "sha256": parsed.archive_sha256,
                    "artifact_role": parsed.manifest["artifact_role"],
                },
                "base_archive": {
                    "bytes": parsed_base.archive_bytes,
                    "sha256": parsed_base.archive_sha256,
                    "decoder_path": str(decoder),
                    "decoder_sha256": sha256_file(decoder),
                },
                "decoded": {
                    "path": str(output),
                    "bytes": actual_bytes,
                    "sha256": actual_sha,
                    "pair_cap": pair_cap,
                    "atomic_promotion": True,
                    "assertion_status": assertion["status"],
                    "assertion_verified": assertion["status"] == "sealed",
                },
                "section_consumption": {
                    MANIFEST_NAME: {
                        "policy_sha256": _sha256(canonical_json(policy)),
                        "policy_applied": True,
                        "final_output_assertion_applied": True,
                    },
                    BOUNDARY_NAME: {
                        "changed_pairs": boundary_changed_pairs,
                        "changed_bytes": boundary_changed_bytes,
                        "frame_indices": [1],
                        "factor2_exact": True,
                    },
                    REPLAY_NAME: {
                        "selected_entries_full_archive": len(parsed.replay_writes),
                        "selected_entries_executed_prefix": sum(len(rows) for rows in replay_by_pair.values()),
                        "effective_entries": replay_effective_entries,
                        "changed_bytes": replay_changed_bytes,
                        "zero_selection_admitted": len(parsed.replay_writes) == 0,
                    },
                    XI0_NAME: {
                        "decoded_value_count": int(parsed.xi0_values.size),
                        "changed_pairs": xi0_changed_pairs,
                        "changed_bytes": xi0_changed_bytes,
                        "frame_indices": [0],
                        "executed_shifts": xi0_shifts,
                    },
                },
                "receiver_search_invocations": 0,
                "decode_seconds": elapsed,
                "decode_limit_seconds": DECODE_LIMIT_SECONDS,
                "decode_limit_pass": elapsed <= DECODE_LIMIT_SECONDS,
                "base_decode": base_decode,
                "success_scratch_cleanup": {
                    "temporary_root": str(temp),
                    "base_raw_bytes": base_decode["decoded_raw_bytes_capped"],
                    "base_raw_sha256": base_decode["decoded_raw_sha256"],
                    "rebuildable_from_hash_bound_archive_and_decoder": True,
                    "delete_only_after_receiver_receipt_fsync": True,
                },
                "score_claim": False,
            }
            _atomic_json(receipt, result)
            receipt_written = True
        return result
    finally:
        if partial.exists():
            partial.unlink()
        if output_promoted and not receipt_written and output.exists():
            output.unlink()


__all__ = [
    "APPLICATION_ORDER",
    "ARCHIVE_SCHEMA",
    "BOUNDARY_NAME",
    "DECODE_LIMIT_SECONDS",
    "EXTENSION_NAMES",
    "MANIFEST_NAME",
    "PAIR_COUNT",
    "RECEIVER_SCHEMA",
    "REPLAY_NAME",
    "XI0_NAME",
    "ParsedR1B4Archive",
    "R1B4ReceiverError",
    "ReplayWrite",
    "build_r1b4_archive",
    "canonical_json",
    "decode_r1b4_archive",
    "decode_replay_payload",
    "default_receiver_policy",
    "encode_replay_payload",
    "parse_r1b4_archive",
    "seal_output_assertion",
    "sha256_file",
]
