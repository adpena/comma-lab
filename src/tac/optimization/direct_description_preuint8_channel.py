# SPDX-License-Identifier: MIT
"""Counted camera-field Q8 correction wrapper for DDM pure-price A/Bs.

The nested receiver first produces its exact uint8 camera field.  This wrapper
then reconstructs that field at Q8 precision, adds counted template-periodic
and sparse Q8 corrections, performs deterministic ordered-dither rounding, and
returns uint8.  It is a receiver operation: no scorer, labels, logits, or
ground-truth state is present at decode.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_coupled_margin import (
    CoupledMarginReceiverV1,
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_entropy_priced_member import _sha256, _zip_stored
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.pure_priced_realized_objective import SOURCE_VIDEO_BYTES
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

ARCHIVE_SCHEMA: Final = "direct_description_preuint8_q8_channel_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_preuint8_q8_channel_receiver.v1"
BASE_MEMBER: Final = "base/ddm_v16_receiver.zip"
PROGRAM_MEMBER: Final = "render/preuint8_q8_program.ddq8"
MANIFEST_MEMBER: Final = "manifest.json"
PROGRAM_MAGIC: Final = b"DDQ81"
PROGRAM_VERSION: Final = 1
_HEADER: Final = struct.Struct(">5sBBHH")
_TEMPLATE_HEADER: Final = struct.Struct(">HHH")
_SPARSE: Final = struct.Struct(">HBHHhhh")
_DITHER_MODES: Final = {
    "off": 0,
    "bayer8": 1,
    "resize_null_sigma_delta": 2,
}
_DITHER_NAMES: Final = {value: key for key, value in _DITHER_MODES.items()}
_BINARY_ROUNDING_CHOICES: Final = np.asarray(
    [[(choice >> shift) & 1 for shift in range(4)] for choice in range(16)],
    dtype=np.int32,
)
_BAYER8: Final = np.asarray(
    [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ],
    dtype=np.int16,
)


@dataclass(frozen=True, slots=True, order=True)
class TemplateQ8CorrectionV1:
    """One Q8 patch correction for an existing placed template."""

    source_pair_id: int
    template_index: int
    delta_q8: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.source_pair_id <= 599 or not 0 <= self.template_index <= 65535:
            raise DirectDescriptionError("preuint8 template correction key is out of range")
        if not self.delta_q8 or len(self.delta_q8) % 3:
            raise DirectDescriptionError("preuint8 template correction has invalid RGB geometry")
        if any(isinstance(value, bool) or not isinstance(value, int) or not -32768 <= value <= 32767 for value in self.delta_q8):
            raise DirectDescriptionError("preuint8 template correction exceeds signed Q8")
        if not any(self.delta_q8):
            raise DirectDescriptionError("preuint8 template correction is identically zero")


@dataclass(frozen=True, slots=True, order=True)
class SparseQ8CorrectionV1:
    source_pair_id: int
    frame_index: int
    camera_y: int
    camera_x: int
    delta_q8: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not 0 <= self.source_pair_id <= 599 or self.frame_index not in (0, 1):
            raise DirectDescriptionError("preuint8 sparse correction key is out of range")
        if not 0 <= self.camera_y < CAMERA_H or not 0 <= self.camera_x < CAMERA_W:
            raise DirectDescriptionError("preuint8 sparse correction coordinate is out of range")
        if any(isinstance(value, bool) or not isinstance(value, int) or not -32768 <= value <= 32767 for value in self.delta_q8):
            raise DirectDescriptionError("preuint8 sparse correction exceeds signed Q8")
        if self.delta_q8 == (0, 0, 0):
            raise DirectDescriptionError("preuint8 sparse correction is identically zero")


@dataclass(frozen=True, slots=True)
class PreUint8Q8ProgramV1:
    templates: tuple[TemplateQ8CorrectionV1, ...] = ()
    sparse: tuple[SparseQ8CorrectionV1, ...] = ()
    dither_mode: str = "bayer8"
    dither_seed: int = 210

    def __post_init__(self) -> None:
        if self.dither_mode not in _DITHER_MODES:
            raise DirectDescriptionError("preuint8 dither mode is unsupported")
        if isinstance(self.dither_seed, bool) or not isinstance(self.dither_seed, int) or not 0 <= self.dither_seed <= 255:
            raise DirectDescriptionError("preuint8 dither seed is out of range")
        if tuple(sorted(self.templates)) != self.templates or tuple(sorted(self.sparse)) != self.sparse:
            raise DirectDescriptionError("preuint8 program records are not canonical-order")
        template_keys = {(row.source_pair_id, row.template_index) for row in self.templates}
        sparse_keys = {(row.source_pair_id, row.frame_index, row.camera_y, row.camera_x) for row in self.sparse}
        if len(template_keys) != len(self.templates) or len(sparse_keys) != len(self.sparse):
            raise DirectDescriptionError("preuint8 program contains duplicate keys")


def encode_preuint8_q8_program(program: PreUint8Q8ProgramV1) -> bytes:
    mode = _DITHER_MODES[program.dither_mode]
    body = bytearray(_HEADER.pack(PROGRAM_MAGIC, PROGRAM_VERSION, mode, program.dither_seed, len(program.templates)))
    body.extend(struct.pack(">H", len(program.sparse)))
    for row in program.templates:
        body.extend(_TEMPLATE_HEADER.pack(row.source_pair_id, row.template_index, len(row.delta_q8)))
        body.extend(struct.pack(f">{len(row.delta_q8)}h", *row.delta_q8))
    for row in program.sparse:
        body.extend(_SPARSE.pack(row.source_pair_id, row.frame_index, row.camera_y, row.camera_x, *row.delta_q8))
    return bytes(body)


def decode_preuint8_q8_program(payload: bytes) -> PreUint8Q8ProgramV1:
    if len(payload) < _HEADER.size + 2:
        raise DirectDescriptionError("preuint8 program header is truncated")
    magic, version, mode, seed, template_count = _HEADER.unpack_from(payload)
    if magic != PROGRAM_MAGIC or version != PROGRAM_VERSION or mode not in _DITHER_NAMES:
        raise DirectDescriptionError("preuint8 program header is invalid")
    cursor = _HEADER.size
    sparse_count = struct.unpack_from(">H", payload, cursor)[0]
    cursor += 2
    templates = []
    for _ in range(template_count):
        if cursor + _TEMPLATE_HEADER.size > len(payload):
            raise DirectDescriptionError("preuint8 template record is truncated")
        pair_id, template_index, count = _TEMPLATE_HEADER.unpack_from(payload, cursor)
        cursor += _TEMPLATE_HEADER.size
        size = count * 2
        if not count or cursor + size > len(payload):
            raise DirectDescriptionError("preuint8 template payload is truncated")
        values = struct.unpack_from(f">{count}h", payload, cursor)
        cursor += size
        templates.append(TemplateQ8CorrectionV1(pair_id, template_index, tuple(values)))
    sparse = []
    for _ in range(sparse_count):
        if cursor + _SPARSE.size > len(payload):
            raise DirectDescriptionError("preuint8 sparse record is truncated")
        pair_id, frame, y, x, dr, dg, db = _SPARSE.unpack_from(payload, cursor)
        cursor += _SPARSE.size
        sparse.append(SparseQ8CorrectionV1(pair_id, frame, y, x, (dr, dg, db)))
    if cursor != len(payload):
        raise DirectDescriptionError("preuint8 program has trailing bytes")
    result = PreUint8Q8ProgramV1(
        tuple(templates),
        tuple(sparse),
        _DITHER_NAMES[mode],
        seed,
    )
    if encode_preuint8_q8_program(result) != payload:
        raise DirectDescriptionError("preuint8 program parse/re-encode changed bytes")
    return result


def compile_preuint8_q8_archive(base_archive: bytes, program: PreUint8Q8ProgramV1) -> bytes:
    base = bytes(base_archive)
    receiver = receive_coupled_margin_archive(base)
    _validate_program(program, receiver)
    encoded = encode_preuint8_q8_program(program)
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "base": {"member": BASE_MEMBER, "bytes": len(base), "sha256": _sha256(base)},
        "program": {
            "member": PROGRAM_MEMBER,
            "bytes": len(encoded),
            "sha256": _sha256(encoded),
            "template_count": len(program.templates),
            "sparse_count": len(program.sparse),
            "dither_mode": program.dither_mode,
            "dither_seed": program.dither_seed,
        },
        "decode_boundary": "nested_uint8_camera_to_counted_q8_correction_then_deterministic_uint8",
        "source_video_bytes": SOURCE_VIDEO_BYTES,
        "scorer_present_at_decode": False,
        "ground_truth_argmax_present_at_decode": False,
        "score_claim": False,
    }
    members = {
        MANIFEST_MEMBER: json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        BASE_MEMBER: base,
        PROGRAM_MEMBER: encoded,
    }
    archive = _zip_stored(members)
    parsed, _ = parse_preuint8_q8_archive(archive)
    if parsed != members or _zip_stored(parsed) != archive:
        raise DirectDescriptionError("preuint8 archive parse/re-encode differs")
    return archive


def parse_preuint8_q8_archive(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if [row.filename for row in infos] != [MANIFEST_MEMBER, BASE_MEMBER, PROGRAM_MEMBER]:
                raise DirectDescriptionError("preuint8 archive member order differs")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("preuint8 ZIP metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("preuint8 archive ZIP is malformed") from exc
    try:
        manifest = json.loads(members[MANIFEST_MEMBER])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("preuint8 manifest is malformed") from exc
    program = decode_preuint8_q8_program(members[PROGRAM_MEMBER])
    expected_program = {
        "member": PROGRAM_MEMBER,
        "bytes": len(members[PROGRAM_MEMBER]),
        "sha256": _sha256(members[PROGRAM_MEMBER]),
        "template_count": len(program.templates),
        "sparse_count": len(program.sparse),
        "dither_mode": program.dither_mode,
        "dither_seed": program.dither_seed,
    }
    invalid = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() != members[MANIFEST_MEMBER]
        or manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("base") != {"member": BASE_MEMBER, "bytes": len(members[BASE_MEMBER]), "sha256": _sha256(members[BASE_MEMBER])}
        or manifest.get("program") != expected_program
        or manifest.get("decode_boundary") != "nested_uint8_camera_to_counted_q8_correction_then_deterministic_uint8"
        or manifest.get("source_video_bytes") != SOURCE_VIDEO_BYTES
        or manifest.get("scorer_present_at_decode") is not False
        or manifest.get("ground_truth_argmax_present_at_decode") is not False
        or manifest.get("score_claim") is not False
    )
    if invalid:
        raise DirectDescriptionError("preuint8 manifest custody differs")
    homes = tuple(
        {
            "name": row.filename,
            "payload_bytes": row.file_size,
            "sha256": _sha256(members[row.filename]),
        }
        for row in infos
    )
    if not 0 < start_dir < len(archive):
        raise DirectDescriptionError("preuint8 ZIP central directory is invalid")
    return members, homes


@dataclass(frozen=True, slots=True)
class PreUint8Q8ReceiverV1:
    archive: bytes
    base: CoupledMarginReceiverV1
    program: PreUint8Q8ProgramV1
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.base.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.base.pose6_codes

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        base_u8 = self.base.render_camera_pairs(indexes)
        q8 = base_u8.astype(np.int32) * 256
        source_to_local = {
            self.base.base.predictor.source_pair_start + pair_id: local
            for local, pair_id in enumerate(indexes)
        }
        placements = {
            (row.source_pair_id, row.template_index): row
            for row in self.base.program.placements
        }
        templates = self.base.base.scorer_solved_templates
        if templates is None:
            raise DirectDescriptionError("preuint8 receiver lost its inherited template bank")
        for row in self.program.templates:
            local = source_to_local.get(row.source_pair_id)
            if local is None:
                continue
            placement = placements[(row.source_pair_id, row.template_index)]
            template = templates.templates[row.template_index]
            delta = np.asarray(row.delta_q8, dtype=np.int32).reshape(
                template.patch_height, template.patch_width, 3
            )
            rows = (np.arange(CAMERA_H, dtype=np.intp) + placement.phase_y) % template.patch_height
            cols = (np.arange(CAMERA_W, dtype=np.intp) + placement.phase_x) % template.patch_width
            field = delta[rows[:, None], cols[None, :]]
            mask = self.base.base.template_camera_masks(
                (row.source_pair_id - self.base.base.predictor.source_pair_start,),
                template,
            )[0]
            q8[local, 0, mask] += field[mask]
            q8[local, 1, mask] += field[mask]
        for row in self.program.sparse:
            local = source_to_local.get(row.source_pair_id)
            if local is not None:
                q8[local, row.frame_index, row.camera_y, row.camera_x] += np.asarray(
                    row.delta_q8, dtype=np.int32
                )
        if self.program.dither_mode == "resize_null_sigma_delta":
            return _resize_null_sigma_delta_round_q8(q8)
        if self.program.dither_mode == "off":
            threshold = np.full((CAMERA_H, CAMERA_W), 128, dtype=np.int32)
        else:
            shift_y = self.program.dither_seed & 7
            shift_x = (self.program.dither_seed >> 3) & 7
            threshold = (
                _BAYER8[
                    (np.arange(CAMERA_H)[:, None] + shift_y) & 7,
                    (np.arange(CAMERA_W)[None, :] + shift_x) & 7,
                ].astype(np.int32)
                * 4
                + 2
            )
        rounded = np.floor_divide(q8 + threshold[None, None, :, :, None], 256)
        return np.ascontiguousarray(np.clip(rounded, 0, 255).astype(np.uint8))


@lru_cache(maxsize=1)
def _scorer_resize_operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=384,
        scorer_w=512,
    )


def _resize_null_sigma_delta_round_q8(q8: np.ndarray) -> np.ndarray:
    """Round Q8 blocks while minimizing exact scorer-resize numerator error.

    Each scorer cell has one disjoint 2x2 camera support. For every support
    containing a fractional Q8 value, this generalized block sigma-delta
    quantizer chooses among the adjacent integer lattice points and minimizes
    the exact separable bilinear numerator residual. The unavoidable residual
    is thereby pushed into the resize kernel as far as the local lattice
    permits. Blind coordinates retain ordinary nearest rounding.
    """

    raw = np.asarray(q8)
    if (
        raw.dtype.kind not in "iu"
        or raw.ndim < 3
        or raw.shape[-3:-1] != (CAMERA_H, CAMERA_W)
        or raw.shape[-1] < 1
    ):
        raise DirectDescriptionError(
            "resize-null sigma-delta requires integer [...,camera_h,camera_w,C] Q8"
        )
    clipped = np.clip(raw, 0, 255 * 256).astype(np.int32, copy=False)
    prefix = clipped.shape[:-3]
    channels = clipped.shape[-1]
    planes = clipped.reshape(-1, CAMERA_H, CAMERA_W, channels)
    floors = np.floor_divide(planes, 256)
    fractions = planes - floors * 256
    output = floors + (fractions >= 128)

    operator = _scorer_resize_operator()
    row_owner = np.full(CAMERA_H, -1, dtype=np.int32)
    col_owner = np.full(CAMERA_W, -1, dtype=np.int32)
    for index, support in enumerate(operator.row_supports):
        row_owner[np.asarray(support.indices, dtype=np.intp)] = index
    for index, support in enumerate(operator.col_supports):
        col_owner[np.asarray(support.indices, dtype=np.intp)] = index

    fractional = np.argwhere(fractions != 0)
    if fractional.size == 0:
        return np.ascontiguousarray(output.reshape(*prefix, CAMERA_H, CAMERA_W, channels).astype(np.uint8))
    owned_rows = row_owner[fractional[:, 1]]
    owned_cols = col_owner[fractional[:, 2]]
    owned = (owned_rows >= 0) & (owned_cols >= 0)
    if not np.any(owned):
        return np.ascontiguousarray(output.reshape(*prefix, CAMERA_H, CAMERA_W, channels).astype(np.uint8))

    keys = np.ravel_multi_index(
        (
            fractional[owned, 0],
            owned_rows[owned],
            owned_cols[owned],
            fractional[owned, 3],
        ),
        (planes.shape[0], operator.scorer_h, operator.scorer_w, channels),
    )
    unique = np.unique(keys)
    plane_ids, scorer_rows, scorer_cols, channel_ids = np.unravel_index(
        unique,
        (planes.shape[0], operator.scorer_h, operator.scorer_w, channels),
    )
    row_indices = np.asarray(
        [operator.row_supports[index].indices for index in scorer_rows],
        dtype=np.intp,
    )
    col_indices = np.asarray(
        [operator.col_supports[index].indices for index in scorer_cols],
        dtype=np.intp,
    )
    row_numerators = np.asarray(
        [operator.row_supports[index].numerators for index in scorer_rows],
        dtype=np.int64,
    )
    col_numerators = np.asarray(
        [operator.col_supports[index].numerators for index in scorer_cols],
        dtype=np.int64,
    )
    coefficients = (
        row_numerators[:, :, None] * col_numerators[:, None, :]
    ).reshape(-1, 4)

    chunk_size = 65_536
    choices = _BINARY_ROUNDING_CHOICES
    for start in range(0, unique.size, chunk_size):
        stop = min(start + chunk_size, unique.size)
        count = stop - start
        values = np.empty((count, 4), dtype=np.int32)
        for position, (row_offset, col_offset) in enumerate(
            ((0, 0), (0, 1), (1, 0), (1, 1))
        ):
            values[:, position] = planes[
                plane_ids[start:stop],
                row_indices[start:stop, row_offset],
                col_indices[start:stop, col_offset],
                channel_ids[start:stop],
            ]
        local_floors = np.floor_divide(values, 256)
        local_fractions = values - local_floors * 256
        candidates = local_floors[:, None, :] + choices[None, :, :]
        invalid = np.any(
            (choices[None, :, :] != 0) & (local_fractions[:, None, :] == 0),
            axis=2,
        )
        weights = coefficients[start:stop]
        target = np.sum(weights * values, axis=1, dtype=np.int64)
        candidate_numerators = np.sum(
            candidates.astype(np.int64) * weights[:, None, :],
            axis=2,
            dtype=np.int64,
        )
        residual = np.abs(candidate_numerators * 256 - target[:, None])
        residual[invalid] = np.iinfo(np.int64).max
        pixel_error = np.sum(
            np.square(candidates * 256 - values[:, None, :], dtype=np.int64),
            axis=2,
            dtype=np.int64,
        )
        best_residual = np.min(residual, axis=1)
        pixel_error[residual != best_residual[:, None]] = np.iinfo(np.int64).max
        selected = candidates[np.arange(count), np.argmin(pixel_error, axis=1)]
        for position, (row_offset, col_offset) in enumerate(
            ((0, 0), (0, 1), (1, 0), (1, 1))
        ):
            output[
                plane_ids[start:stop],
                row_indices[start:stop, row_offset],
                col_indices[start:stop, col_offset],
                channel_ids[start:stop],
            ] = selected[:, position]
    return np.ascontiguousarray(
        output.reshape(*prefix, CAMERA_H, CAMERA_W, channels).astype(np.uint8)
    )


def _validate_program(program: PreUint8Q8ProgramV1, receiver: CoupledMarginReceiverV1) -> None:
    bank = receiver.base.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("preuint8 program requires a template bank")
    placements = {(row.source_pair_id, row.template_index) for row in receiver.program.placements}
    for row in program.templates:
        if (row.source_pair_id, row.template_index) not in placements:
            raise DirectDescriptionError("preuint8 template correction lacks a base placement")
        template = bank.templates[row.template_index]
        if len(row.delta_q8) != template.patch_height * template.patch_width * 3:
            raise DirectDescriptionError("preuint8 template correction patch geometry differs")
    source_start = receiver.base.predictor.source_pair_start
    source_stop = source_start + receiver.z.n_pairs
    if any(not source_start <= row.source_pair_id < source_stop for row in program.sparse):
        raise DirectDescriptionError("preuint8 sparse correction is outside the base receiver")


def receive_preuint8_q8_archive(archive: bytes) -> PreUint8Q8ReceiverV1:
    members, _homes = parse_preuint8_q8_archive(archive)
    base = receive_coupled_margin_archive(members[BASE_MEMBER])
    program = decode_preuint8_q8_program(members[PROGRAM_MEMBER])
    _validate_program(program, base)
    custody = {
        "schema": RECEIVER_SCHEMA,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "base_archive_bytes": len(members[BASE_MEMBER]),
        "base_archive_sha256": _sha256(members[BASE_MEMBER]),
        "program_bytes": len(members[PROGRAM_MEMBER]),
        "program_sha256": _sha256(members[PROGRAM_MEMBER]),
        "template_count": len(program.templates),
        "sparse_count": len(program.sparse),
        "dither_mode": program.dither_mode,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
    }
    return PreUint8Q8ReceiverV1(bytes(archive), base, program, custody)


__all__ = [
    "ARCHIVE_SCHEMA",
    "BASE_MEMBER",
    "MANIFEST_MEMBER",
    "PROGRAM_MEMBER",
    "PreUint8Q8ProgramV1",
    "PreUint8Q8ReceiverV1",
    "SparseQ8CorrectionV1",
    "TemplateQ8CorrectionV1",
    "compile_preuint8_q8_archive",
    "decode_preuint8_q8_program",
    "encode_preuint8_q8_program",
    "parse_preuint8_q8_archive",
    "receive_preuint8_q8_archive",
]
