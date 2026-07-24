# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.optimization.ddm_rg1_receiver_grammar import (
    LANE_FIELDS,
    LaneProgramCoordinateV1,
    compile_rg1_receiver_grammar,
    decode_lane_program_coordinates,
    encode_lane_program_coordinates,
    parse_rg1_receiver_grammar,
    project_polygon_center,
    receive_rg1_receiver_grammar,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta

REPO = Path(__file__).resolve().parents[4]
V19C = REPO / (
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/"
    "ddm_v19c_final_n600.zip.receipt-bytes"
)
V19C_SHA256 = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"


def _v19c_carrier() -> bytes:
    from tac.optimization import direct_description_coupled_margin as coupled
    from tac.optimization import direct_description_preuint8_channel as preuint8

    archive = V19C.read_bytes()
    assert hashlib.sha256(archive).hexdigest() == V19C_SHA256
    pre_members, _ = preuint8.parse_preuint8_q8_archive(archive)
    coupled_members, _ = coupled.parse_coupled_margin_archive(
        pre_members[preuint8.BASE_MEMBER]
    )
    return coupled_members[coupled.BASE_MEMBER]


def test_all_24_lane_coordinate_ids_are_unique_and_roundtrip() -> None:
    rows = tuple(
        LaneProgramCoordinateV1(line, field, 1)
        for line in range(6)
        for field in LANE_FIELDS
    )
    assert len(rows) == 24
    assert len({row.actuator_id for row in rows}) == 24
    payload = encode_lane_program_coordinates(rows)
    assert decode_lane_program_coordinates(payload) == rows


def test_lane_packet_rejects_duplicate_address_and_crc_mutation() -> None:
    row = LaneProgramCoordinateV1(0, "width_bias_q8", 1)
    with pytest.raises(DirectDescriptionError, match="sorted, unique"):
        encode_lane_program_coordinates((row, row))
    payload = bytearray(encode_lane_program_coordinates((row,)))
    payload[-1] ^= 1
    with pytest.raises(DirectDescriptionError, match="CRC"):
        decode_lane_program_coordinates(bytes(payload))


def test_inactive_rg1_is_v19c_carrier_byte_identical() -> None:
    carrier = _v19c_carrier()
    assert compile_rg1_receiver_grammar(carrier) == carrier


def test_joint_streams_are_separate_and_typed() -> None:
    carrier = _v19c_carrier()
    archive = compile_rg1_receiver_grammar(
        carrier,
        lane_coordinates=(
            LaneProgramCoordinateV1(0, "dash_phase_origin_q8", 1),
        ),
        corrections=(LaneCoefficientDelta(0, 4, 3, 0.008202752098441124),),
    )
    members = parse_rg1_receiver_grammar(archive)
    assert "production/lane_program_coordinates.rg1lp" in members
    assert "correction/lane_chart_symbols.g2cs2" in members
    assert members["base/v13_v19c_carrier.zip"] == carrier


def test_joint_streams_change_receiver_without_mutating_base() -> None:
    carrier = _v19c_carrier()
    base = receive_rg1_receiver_grammar(carrier)
    archive = compile_rg1_receiver_grammar(
        carrier,
        lane_coordinates=(LaneProgramCoordinateV1(0, "width_bias_q8", 1),),
        corrections=(LaneCoefficientDelta(0, 4, 3, 0.008202752098441124),),
    )
    receiver = receive_rg1_receiver_grammar(archive)
    assert receiver.custody["composition_order_enforced"] is True
    assert receiver.custody["sealed_v13_v19c_mutated"] is False
    assert (base.render_camera_pairs((0,)) != receiver.render_camera_pairs((0,))).any()


def test_polygon_center_projection_is_explicit_bounded_and_fail_closed() -> None:
    relative = (-4, -1, 3)
    assert project_polygon_center(-20, relative, 16) == 4
    assert project_polygon_center(20, relative, 16) == 12
    assert project_polygon_center(8, relative, 16) == 8
    with pytest.raises(DirectDescriptionError, match="cannot fit"):
        project_polygon_center(0, (-8, 8), 16)
