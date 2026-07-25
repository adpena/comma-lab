# SPDX-License-Identifier: MIT
"""Exact seven-home lossless recoding for the DDM WF7 state object.

EV2's logical seven-home accounting partition is not itself a physical wire
format: its 270-byte lane-seed delta is spread across the lane member, manifest,
and central directory of the exact seeded state.  This module first closes that
boundary by recovering the seven non-overlapping physical homes of the sealed
134,211-byte state.  It then races the settled CC2 five-coder menu on each
whole home and emits one compact counted container whose receiver restores the
exact seeded-state bytes.

The container is deliberately state-scoped.  It is not an E4 contest packet
and does not claim a contest score.  A consumer must separately bind the
restored state to an admitted runtime/export surface.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from tac.optimization.arith_selfcomp_rate_coders import (
    decode_bellard_class_mixing,
    decode_g4_decoder_context,
    decode_spatial_context_arithmetic,
    decode_willems_ctw,
)
from tac.optimization.ddm_ms7_receiver_edges import race_counted_stream_contexts

SCHEMA: Final = "ddm_wf7_seven_home_stream_waterfill.v1"
CONTAINER_SCHEMA: Final = "ddm_wf7_seven_home_state_container.v1"
MAGIC: Final = b"DWF7"
VERSION: Final = 1
HOME_COUNT: Final = 7
RATE_DENOMINATOR_BYTES: Final = 37_545_489
STATE_BYTES: Final = 134_211
STATE_SHA256: Final = "3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd"
EXPECTED_MEMBERS: Final = (
    "manifest.json",
    "predictor.zip",
    "predict/movable_polygon_worldsheet.g1s",
    "render/receiver_realization.ddrp",
    "render/scorer_solved_templates.ddst",
    "predict/lane_periodic_programs.ddlp",
)
CENTRAL_HOME: Final = "__central_directory_and_eocd__"
EXPECTED_HOME_BYTES: Final = (3_379, 100_099, 29_878, 85, 151, 155, 464)
CODEC_NAMES: Final = (
    "RAW_CURRENT",
    "COUNTED_TINY_ARM_IFCE",
    "G4_FREE_DECODER_CONTEXT",
    "WILLEMS_CTW",
    "BELLARD_CLASS_MIXING",
)
CODEC_IDS: Final = {name: index for index, name in enumerate(CODEC_NAMES)}
CODEC_ID_BITS: Final = 3
CODEC_DIRECTORY_BYTES: Final = 3


class WF7Error(ValueError):
    """The state, home directory, codec frame, or parse-back contract differed."""


@dataclass(frozen=True, slots=True)
class PhysicalHome:
    """One exact non-overlapping byte home in the seeded state archive."""

    home_id: int
    name: str
    start: int
    stop: int
    payload: bytes

    @property
    def counted_bytes(self) -> int:
        return self.stop - self.start


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_uvarint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise WF7Error("uvarint input must be an exact nonnegative integer")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _decode_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(payload) and shift <= 28:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if payload[start:offset] != _encode_uvarint(value):
                raise WF7Error("noncanonical home-length uvarint")
            return value, offset
        shift += 7
    raise WF7Error("truncated or overlong home-length uvarint")


def _pack_codec_ids(codec_ids: tuple[int, ...]) -> bytes:
    if len(codec_ids) != HOME_COUNT or any(value < 0 or value >= len(CODEC_NAMES) for value in codec_ids):
        raise WF7Error("codec directory differs from the sealed seven-home menu")
    packed = sum(value << (index * CODEC_ID_BITS) for index, value in enumerate(codec_ids))
    return packed.to_bytes(CODEC_DIRECTORY_BYTES, "little")


def _unpack_codec_ids(payload: bytes) -> tuple[int, ...]:
    if len(payload) != CODEC_DIRECTORY_BYTES:
        raise WF7Error("codec directory byte count differs")
    packed = int.from_bytes(payload, "little")
    ids = tuple((packed >> (index * CODEC_ID_BITS)) & 0x7 for index in range(HOME_COUNT))
    if any(value >= len(CODEC_NAMES) for value in ids):
        raise WF7Error("codec directory contains an unsupported codec")
    if packed >> (HOME_COUNT * CODEC_ID_BITS):
        raise WF7Error("codec directory has nonzero reserved bits")
    return ids


def inspect_seeded_state(state_archive: bytes, *, require_sealed_identity: bool = True) -> tuple[PhysicalHome, ...]:
    """Return the seven exact outer byte homes of one seeded state archive."""

    state = bytes(state_archive)
    if require_sealed_identity and (len(state), sha256_bytes(state)) != (STATE_BYTES, STATE_SHA256):
        raise WF7Error("seeded state differs from the sealed 134211-byte identity")
    try:
        with zipfile.ZipFile(io.BytesIO(state), "r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != EXPECTED_MEMBERS:
                raise WF7Error("seeded-state member order differs")
            if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
                raise WF7Error("seeded-state outer members must remain ZIP_STORED")
            start_dir = archive.start_dir
            for info in infos:
                archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise WF7Error("seeded state is not a valid outer ZIP") from exc

    stops = [info.header_offset for info in infos[1:]] + [start_dir]
    homes = [
        PhysicalHome(
            home_id=index,
            name=info.filename,
            start=info.header_offset,
            stop=stop,
            payload=state[info.header_offset : stop],
        )
        for index, (info, stop) in enumerate(zip(infos, stops, strict=True))
    ]
    homes.append(
        PhysicalHome(
            home_id=len(homes),
            name=CENTRAL_HOME,
            start=start_dir,
            stop=len(state),
            payload=state[start_dir:],
        )
    )
    if require_sealed_identity and tuple(home.counted_bytes for home in homes) != EXPECTED_HOME_BYTES:
        raise WF7Error("seeded-state physical home sizes differ")
    if sum(home.counted_bytes for home in homes) != len(state):
        raise WF7Error("seeded-state physical homes do not conserve bytes")
    return tuple(homes)


def _decode_ifce(frame: bytes) -> bytes:
    try:
        return decode_spatial_context_arithmetic(frame).reshape(-1).view(np.uint8).tobytes()
    except ValueError as exc:
        raise WF7Error("IFCE frame failed exact parse-back") from exc


def _decode_frame(codec_id: int, frame: bytes) -> bytes:
    try:
        if codec_id == CODEC_IDS["RAW_CURRENT"]:
            return bytes(frame)
        if codec_id == CODEC_IDS["COUNTED_TINY_ARM_IFCE"]:
            return _decode_ifce(frame)
        if codec_id == CODEC_IDS["G4_FREE_DECODER_CONTEXT"]:
            return decode_g4_decoder_context(frame)
        if codec_id == CODEC_IDS["WILLEMS_CTW"]:
            return decode_willems_ctw(frame)
        if codec_id == CODEC_IDS["BELLARD_CLASS_MIXING"]:
            return decode_bellard_class_mixing(frame)
    except ValueError as exc:
        raise WF7Error("seven-home codec frame failed exact parse-back") from exc
    raise WF7Error(f"unsupported seven-home codec id: {codec_id}")


def serialize_candidate(codec_names: tuple[str, ...], frames: tuple[bytes, ...]) -> bytes:
    """Serialize seven already-framed homes with a compact counted directory."""

    if len(codec_names) != HOME_COUNT or len(frames) != HOME_COUNT:
        raise WF7Error("candidate requires exactly seven codec/frame rows")
    try:
        codec_ids = tuple(CODEC_IDS[name] for name in codec_names)
    except KeyError as exc:
        raise WF7Error(f"candidate selected an unsupported codec: {exc.args[0]}") from exc
    if any(not frame for frame in frames):
        raise WF7Error("candidate frames must be nonempty")
    directory = b"".join(_encode_uvarint(len(frame)) for frame in frames[:-1])
    return MAGIC + bytes((VERSION,)) + _pack_codec_ids(codec_ids) + directory + b"".join(frames)


def restore_candidate(candidate: bytes, *, require_sealed_identity: bool = True) -> tuple[bytes, dict[str, Any]]:
    """Decode one candidate and restore an exact seven-home seeded state."""

    payload = bytes(candidate)
    fixed_prefix = len(MAGIC) + 1 + CODEC_DIRECTORY_BYTES
    if len(payload) <= fixed_prefix or not payload.startswith(MAGIC) or payload[len(MAGIC)] != VERSION:
        raise WF7Error("candidate magic/version differs")
    codec_ids = _unpack_codec_ids(payload[len(MAGIC) + 1 : fixed_prefix])
    offset = fixed_prefix
    lengths: list[int] = []
    for _ in range(HOME_COUNT - 1):
        length, offset = _decode_uvarint(payload, offset)
        if length <= 0:
            raise WF7Error("candidate frame length must be positive")
        lengths.append(length)
    final_length = len(payload) - offset - sum(lengths)
    if final_length <= 0:
        raise WF7Error("candidate final frame is absent")
    lengths.append(final_length)

    frames: list[bytes] = []
    cursor = offset
    for length in lengths:
        stop = cursor + length
        if stop > len(payload):
            raise WF7Error("candidate frame directory overruns payload")
        frames.append(payload[cursor:stop])
        cursor = stop
    if cursor != len(payload):
        raise WF7Error("candidate has trailing bytes")
    raw_homes = tuple(_decode_frame(codec_id, frame) for codec_id, frame in zip(codec_ids, frames, strict=True))
    restored = b"".join(raw_homes)
    physical = inspect_seeded_state(restored, require_sealed_identity=require_sealed_identity)
    if tuple(home.payload for home in physical) != raw_homes:
        raise WF7Error("decoded frame boundaries differ from restored physical homes")
    return restored, {
        "schema": CONTAINER_SCHEMA,
        "candidate": {"bytes": len(payload), "sha256": sha256_bytes(payload)},
        "restored_state": {"bytes": len(restored), "sha256": sha256_bytes(restored)},
        "codec_names": [CODEC_NAMES[value] for value in codec_ids],
        "frame_lengths": lengths,
        "directory_bytes": offset,
        "home_count": HOME_COUNT,
        "exact_parseback": True,
    }


def build_best_candidate(state_archive: bytes) -> tuple[bytes, dict[str, Any]]:
    """Race all homes, materialize every winning frame, and prove exact restore."""

    state = bytes(state_archive)
    homes = inspect_seeded_state(state)
    rows: list[dict[str, Any]] = []
    codec_names: list[str] = []
    selected_frames: list[bytes] = []
    for home in homes:
        race, frames = race_counted_stream_contexts(home.payload)
        winner = race["winner"]
        codec = str(winner["codec"])
        frame = frames[codec]
        if len(frame) != int(winner["framed_bytes"]) or sha256_bytes(frame) != winner["frame_sha256"]:
            raise WF7Error(f"winner custody differs for home {home.name}")
        codec_names.append(codec)
        selected_frames.append(frame)
        rows.append(
            {
                "home_id": home.home_id,
                "home": home.name,
                "current_bytes": home.counted_bytes,
                "current_sha256": sha256_bytes(home.payload),
                "selected_codec": codec,
                "selected_frame_bytes": len(frame),
                "selected_frame_sha256": sha256_bytes(frame),
                "delta_bytes_before_joint_directory": len(frame) - home.counted_bytes,
                "delta_d_seg": 0.0,
                "delta_d_pose": 0.0,
                "distortion_authority": (
                    "DERIVED_EXACT_ZERO_FROM_MEASURED_FRAME_PARSEBACK_AND_BYTE_IDENTICAL_STATE_RESTORE"
                ),
                "all_five_arms_parseback_exact": all(row["parseback_exact"] for row in race["rows"]),
                "arms": race["rows"],
                "verdict_scope": (
                    "INSTANCE x exact seeded-state physical home; other stream shapes and lossy moves remain open"
                ),
            }
        )
    candidate = serialize_candidate(tuple(codec_names), tuple(selected_frames))
    restored, restoration = restore_candidate(candidate)
    if restored != state:
        raise WF7Error("best candidate did not restore the exact seeded state")
    payload_delta = sum(row["delta_bytes_before_joint_directory"] for row in rows)
    directory_bytes = restoration["directory_bytes"]
    total_delta = len(candidate) - len(state)
    if payload_delta + directory_bytes != total_delta:
        raise WF7Error("candidate payload and directory deltas do not reconcile")
    return candidate, {
        "schema": SCHEMA,
        "state": {"bytes": len(state), "sha256": sha256_bytes(state)},
        "candidate": {"bytes": len(candidate), "sha256": sha256_bytes(candidate)},
        "delta_bytes": total_delta,
        "delta_rate_score": 25.0 * total_delta / RATE_DENOMINATOR_BYTES,
        "delta_d_seg": 0.0,
        "delta_d_pose": 0.0,
        "strictly_joint_improving": total_delta < 0,
        "selected_negative_home_count": sum(row["delta_bytes_before_joint_directory"] < 0 for row in rows),
        "joint_directory_bytes": directory_bytes,
        "selected_payload_delta_bytes": payload_delta,
        "rows": rows,
        "restoration": restoration,
        "contest_packet_status": "STATE_CONTAINER_ONLY_E4_RUNTIME_BINDING_NOT_MATERIALIZED",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


__all__ = [
    "CONTAINER_SCHEMA",
    "EXPECTED_HOME_BYTES",
    "EXPECTED_MEMBERS",
    "HOME_COUNT",
    "SCHEMA",
    "STATE_BYTES",
    "STATE_SHA256",
    "PhysicalHome",
    "WF7Error",
    "build_best_candidate",
    "inspect_seeded_state",
    "restore_candidate",
    "serialize_candidate",
    "sha256_bytes",
]
