# SPDX-License-Identifier: MIT
"""Counted v16 placement and sparse-compensation extension of the v15 receiver.

The nested base archive is parsed and rendered by the established
``CarrierComposeReceiverV1``.  This module adds only two scorer-free receiver
operations after that render:

* select a counted phase for an existing scorer-solved RGB template on a named
  source pair; and
* add a counted sparse signed RGB correction at a camera-resolution site.

The scorer, logits, gradients, and ground-truth argmax table are encode-side
only.  Every video-derived placement and correction byte lives in the outer
archive and is included in exact ZIP-home accounting.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    EVIDENCE_AXIS,
    CarrierComposeReceiverV1,
    RowBandScorerTemplateV1,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_entropy_priced_member import _sha256, _zip_stored
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

ARCHIVE_SCHEMA: Final = "direct_description_v16_coupled_margin_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_v16_coupled_margin_receiver.v1"
BASE_MEMBER: Final = "base/ddm_v15_receiver.zip"
PROGRAM_MEMBER: Final = "render/coupled_margin_program.ddcm"
MANIFEST_MEMBER: Final = "manifest.json"
PROGRAM_MAGIC: Final = b"DDCM1"
PROGRAM_VERSION: Final = 1
_HEADER: Final = struct.Struct(">5sBHI")
_PLACEMENT: Final = struct.Struct(">HHBB")
_COMPENSATION: Final = struct.Struct(">HBHHbbb")
_MAX_PLACEMENTS: Final = 4096
_MAX_COMPENSATIONS: Final = 65535


@dataclass(frozen=True, slots=True, order=True)
class TemplatePlacementV1:
    """Pair-local phase selection for one inherited template."""

    source_pair_id: int
    template_index: int
    phase_y: int
    phase_x: int

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("source_pair_id", self.source_pair_id, 599),
            ("template_index", self.template_index, 65535),
            ("phase_y", self.phase_y, 255),
            ("phase_x", self.phase_x, 255),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise DirectDescriptionError(f"coupled placement {name} is out of range")


@dataclass(frozen=True, slots=True, order=True)
class SparseCameraCompensationV1:
    """One counted additive RGB correction applied after all templates."""

    source_pair_id: int
    frame_index: int
    camera_y: int
    camera_x: int
    delta_rgb: tuple[int, int, int]

    def __post_init__(self) -> None:
        if isinstance(self.source_pair_id, bool) or not isinstance(self.source_pair_id, int) or not 0 <= self.source_pair_id <= 599:
            raise DirectDescriptionError("compensation source_pair_id is out of range")
        if self.frame_index not in (0, 1):
            raise DirectDescriptionError("compensation frame_index must be 0 or 1")
        if not 0 <= self.camera_y < CAMERA_H or not 0 <= self.camera_x < CAMERA_W:
            raise DirectDescriptionError("compensation camera coordinate is out of range")
        if (
            len(self.delta_rgb) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or not -127 <= value <= 127 for value in self.delta_rgb)
            or self.delta_rgb == (0, 0, 0)
        ):
            raise DirectDescriptionError("compensation delta_rgb must be a nonzero int8-safe triple")


@dataclass(frozen=True, slots=True)
class CoupledMarginProgramV1:
    placements: tuple[TemplatePlacementV1, ...] = ()
    compensations: tuple[SparseCameraCompensationV1, ...] = ()

    def __post_init__(self) -> None:
        if len(self.placements) > _MAX_PLACEMENTS or len(self.compensations) > _MAX_COMPENSATIONS:
            raise DirectDescriptionError("coupled margin program exceeds bounded record count")
        if tuple(sorted(self.placements)) != self.placements or tuple(sorted(self.compensations)) != self.compensations:
            raise DirectDescriptionError("coupled margin program records are not canonical-order")
        placement_keys = {(row.source_pair_id, row.template_index) for row in self.placements}
        if len(placement_keys) != len(self.placements):
            raise DirectDescriptionError("coupled margin placement key is duplicated")
        compensation_keys = {
            (row.source_pair_id, row.frame_index, row.camera_y, row.camera_x)
            for row in self.compensations
        }
        if len(compensation_keys) != len(self.compensations):
            raise DirectDescriptionError("coupled margin compensation coordinate is duplicated")


def encode_coupled_margin_program(program: CoupledMarginProgramV1) -> bytes:
    body = bytearray(_HEADER.pack(PROGRAM_MAGIC, PROGRAM_VERSION, len(program.placements), len(program.compensations)))
    for row in program.placements:
        body.extend(_PLACEMENT.pack(row.source_pair_id, row.template_index, row.phase_y, row.phase_x))
    for row in program.compensations:
        body.extend(
            _COMPENSATION.pack(
                row.source_pair_id,
                row.frame_index,
                row.camera_y,
                row.camera_x,
                *row.delta_rgb,
            )
        )
    return bytes(body)


def decode_coupled_margin_program(payload: bytes) -> CoupledMarginProgramV1:
    if len(payload) < _HEADER.size:
        raise DirectDescriptionError("coupled margin program header is truncated")
    magic, version, placement_count, compensation_count = _HEADER.unpack_from(payload)
    if magic != PROGRAM_MAGIC or version != PROGRAM_VERSION:
        raise DirectDescriptionError("coupled margin program header is invalid")
    if placement_count > _MAX_PLACEMENTS or compensation_count > _MAX_COMPENSATIONS:
        raise DirectDescriptionError("coupled margin program count is out of range")
    expected = _HEADER.size + placement_count * _PLACEMENT.size + compensation_count * _COMPENSATION.size
    if len(payload) != expected:
        raise DirectDescriptionError("coupled margin program length differs from its counts")
    cursor = _HEADER.size
    placements = []
    for _ in range(placement_count):
        source_pair_id, template_index, phase_y, phase_x = _PLACEMENT.unpack_from(payload, cursor)
        cursor += _PLACEMENT.size
        placements.append(TemplatePlacementV1(source_pair_id, template_index, phase_y, phase_x))
    compensations = []
    for _ in range(compensation_count):
        source_pair_id, frame_index, camera_y, camera_x, dr, dg, db = _COMPENSATION.unpack_from(payload, cursor)
        cursor += _COMPENSATION.size
        compensations.append(
            SparseCameraCompensationV1(source_pair_id, frame_index, camera_y, camera_x, (dr, dg, db))
        )
    program = CoupledMarginProgramV1(tuple(placements), tuple(compensations))
    if encode_coupled_margin_program(program) != payload:
        raise DirectDescriptionError("coupled margin program parse/re-encode changed bytes")
    return program


def compile_coupled_margin_archive(
    base_archive: bytes,
    program: CoupledMarginProgramV1,
    *,
    verify_base_member_effects: bool = True,
) -> bytes:
    """Wrap exact v15 bytes and one counted program in a deterministic ZIP."""

    base = bytes(base_archive)
    receiver = receive_carrier_compose_archive(
        base,
        verify_member_effects=verify_base_member_effects,
    )
    _validate_program_against_receiver(program, receiver)
    encoded = encode_coupled_margin_program(program)
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "base": {"member": BASE_MEMBER, "bytes": len(base), "sha256": _sha256(base)},
        "program": {
            "member": PROGRAM_MEMBER,
            "bytes": len(encoded),
            "sha256": _sha256(encoded),
            "placement_count": len(program.placements),
            "sparse_compensation_count": len(program.compensations),
        },
        "decode_boundary": "v15_receiver_then_pair_phase_templates_then_sparse_camera_compensation",
        "scorer_present_at_decode": False,
        "ground_truth_argmax_present_at_decode": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    members = {
        MANIFEST_MEMBER: rfc8785_canonicalize(manifest),
        BASE_MEMBER: base,
        PROGRAM_MEMBER: encoded,
    }
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("coupled margin compiler is nondeterministic")
    parsed, _homes = parse_coupled_margin_archive(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("coupled margin archive parse/re-encode differs")
    return first


def parse_coupled_margin_archive(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if [row.filename for row in infos] != [MANIFEST_MEMBER, BASE_MEMBER, PROGRAM_MEMBER]:
                raise DirectDescriptionError("coupled margin archive member order differs")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("coupled margin ZIP metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("coupled margin archive ZIP is malformed") from exc
    try:
        manifest = json.loads(members[MANIFEST_MEMBER])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("coupled margin manifest is malformed") from exc
    program = decode_coupled_margin_program(members[PROGRAM_MEMBER])
    invalid = (
        rfc8785_canonicalize(manifest) != members[MANIFEST_MEMBER]
        or manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("base")
        != {"member": BASE_MEMBER, "bytes": len(members[BASE_MEMBER]), "sha256": _sha256(members[BASE_MEMBER])}
        or manifest.get("program")
        != {
            "member": PROGRAM_MEMBER,
            "bytes": len(members[PROGRAM_MEMBER]),
            "sha256": _sha256(members[PROGRAM_MEMBER]),
            "placement_count": len(program.placements),
            "sparse_compensation_count": len(program.compensations),
        }
        or manifest.get("decode_boundary")
        != "v15_receiver_then_pair_phase_templates_then_sparse_camera_compensation"
        or manifest.get("scorer_present_at_decode") is not False
        or manifest.get("ground_truth_argmax_present_at_decode") is not False
        or manifest.get("score_claim") is not False
        or manifest.get("evidence_axis") != EVIDENCE_AXIS
    )
    if invalid:
        raise DirectDescriptionError("coupled margin manifest custody differs")
    homes = []
    for info in infos:
        next_offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra) + info.compress_size
        homes.append(
            {
                "name": info.filename,
                "payload_bytes": info.file_size,
                "zip_home_bytes": next_offset - info.header_offset,
                "sha256": _sha256(members[info.filename]),
            }
        )
    central_bytes = len(archive) - start_dir
    if sum(int(row["zip_home_bytes"]) for row in homes) + central_bytes != len(archive):
        raise DirectDescriptionError("coupled margin ZIP home accounting does not close")
    return members, tuple(homes)


@dataclass(frozen=True, slots=True)
class CoupledMarginReceiverV1:
    archive: bytes
    base: CarrierComposeReceiverV1
    program: CoupledMarginProgramV1
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.base.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.base.pose6_codes

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        output = self.base.render_camera_pairs(indexes)
        source_to_local = {
            self.base.predictor.source_pair_start + pair_id: local_index
            for local_index, pair_id in enumerate(indexes)
        }
        templates = self.base.scorer_solved_templates
        if templates is None:
            raise DirectDescriptionError("coupled margin receiver lost its inherited template bank")
        for placement in self.program.placements:
            local_index = source_to_local.get(placement.source_pair_id)
            if local_index is None:
                continue
            template = templates.templates[placement.template_index]
            local_pair_id = placement.source_pair_id - self.base.predictor.source_pair_start
            mask = self.base.template_camera_masks((local_pair_id,), template)[0]
            field = _phase_template_field(template, placement.phase_y, placement.phase_x)
            output[local_index, 0, mask] = field[mask]
            output[local_index, 1, mask] = field[mask]
        for row in self.program.compensations:
            local_index = source_to_local.get(row.source_pair_id)
            if local_index is None:
                continue
            current = output[local_index, row.frame_index, row.camera_y, row.camera_x].astype(np.int16)
            output[local_index, row.frame_index, row.camera_y, row.camera_x] = np.clip(
                current + np.asarray(row.delta_rgb, dtype=np.int16), 0, 255
            ).astype(np.uint8)
        return np.ascontiguousarray(output)


def receive_coupled_margin_archive(
    archive: bytes,
    *,
    verify_base_member_effects: bool = True,
) -> CoupledMarginReceiverV1:
    members, homes = parse_coupled_margin_archive(archive)
    base = receive_carrier_compose_archive(
        members[BASE_MEMBER],
        verify_member_effects=verify_base_member_effects,
    )
    program = decode_coupled_margin_program(members[PROGRAM_MEMBER])
    _validate_program_against_receiver(program, base)
    custody = {
        "schema": RECEIVER_SCHEMA,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "base_archive_bytes": len(members[BASE_MEMBER]),
        "base_archive_sha256": _sha256(members[BASE_MEMBER]),
        "program_bytes": len(members[PROGRAM_MEMBER]),
        "program_sha256": _sha256(members[PROGRAM_MEMBER]),
        "placement_count": len(program.placements),
        "sparse_compensation_count": len(program.compensations),
        "zip_homes_close": sum(int(row["zip_home_bytes"]) for row in homes) < len(archive),
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return CoupledMarginReceiverV1(bytes(archive), base, program, custody)


def coupled_margin_byte_rows(archive: bytes) -> list[dict[str, Any]]:
    _members, homes = parse_coupled_margin_archive(archive)
    return [
        {
            "stratum": {
                MANIFEST_MEMBER: "outer_manifest",
                BASE_MEMBER: "inherited_v15_receiver",
                PROGRAM_MEMBER: "template_placements_plus_sparse_compensation",
            }[row["name"]],
            **row,
        }
        for row in homes
    ]


def _phase_template_field(template: RowBandScorerTemplateV1, phase_y: int, phase_x: int) -> np.ndarray:
    patch = template.patch()
    rows = (np.arange(CAMERA_H, dtype=np.intp) + phase_y) % template.patch_height
    cols = (np.arange(CAMERA_W, dtype=np.intp) + phase_x) % template.patch_width
    return np.ascontiguousarray(patch[rows[:, None], cols[None, :]])


def _validate_program_against_receiver(program: CoupledMarginProgramV1, receiver: CarrierComposeReceiverV1) -> None:
    bank = receiver.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("coupled margin program requires an inherited v15 template bank")
    source_start = receiver.predictor.source_pair_start
    source_stop = source_start + receiver.z.n_pairs
    for row in program.placements:
        if not source_start <= row.source_pair_id < source_stop or row.template_index >= len(bank.templates):
            raise DirectDescriptionError("coupled margin placement does not belong to the base receiver")
        template = bank.templates[row.template_index]
        if row.phase_y >= template.patch_height or row.phase_x >= template.patch_width:
            raise DirectDescriptionError("coupled margin placement phase exceeds its template")
    for row in program.compensations:
        if not source_start <= row.source_pair_id < source_stop:
            raise DirectDescriptionError("coupled margin compensation does not belong to the base receiver")


__all__ = [
    "ARCHIVE_SCHEMA",
    "BASE_MEMBER",
    "PROGRAM_MEMBER",
    "CoupledMarginProgramV1",
    "CoupledMarginReceiverV1",
    "SparseCameraCompensationV1",
    "TemplatePlacementV1",
    "compile_coupled_margin_archive",
    "coupled_margin_byte_rows",
    "decode_coupled_margin_program",
    "encode_coupled_margin_program",
    "parse_coupled_margin_archive",
    "receive_coupled_margin_archive",
]
