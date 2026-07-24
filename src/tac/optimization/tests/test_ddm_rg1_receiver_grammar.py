# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.optimization import ddm_rg1_receiver_grammar as rg1
from tac.optimization.ddm_rg1_receiver_grammar import (
    LANE_FIELDS,
    LaneProgramCoordinateV1,
    RG2ReceiverV1,
    RG3ReceiverV1,
    RG3ResidualCoordinateV1,
    SkeletonAmplitudeCoordinateV1,
    compile_rg1_receiver_grammar,
    compile_rg2_receiver_grammar,
    compile_rg3_receiver_grammar,
    decode_lane_program_coordinates,
    decode_rg3_residual_coordinates,
    decode_skeleton_amplitude_coordinates,
    derive_rg3_class_birth_address,
    derive_rg3_finer_event_local_band,
    derive_rg3_fisher_margin_band,
    derive_skeleton_amplitude_row_band,
    encode_lane_program_coordinates,
    encode_rg3_residual_coordinates,
    encode_skeleton_amplitude_coordinates,
    parse_rg1_receiver_grammar,
    project_polygon_center,
    receive_rg1_receiver_grammar,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta

REPO = Path(__file__).resolve().parents[4]
V19C = REPO / (".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_final_n600.zip.receipt-bytes")
V19C_SHA256 = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"


def _v19c_carrier() -> bytes:
    from tac.optimization import direct_description_coupled_margin as coupled
    from tac.optimization import direct_description_preuint8_channel as preuint8

    archive = V19C.read_bytes()
    assert hashlib.sha256(archive).hexdigest() == V19C_SHA256
    pre_members, _ = preuint8.parse_preuint8_q8_archive(archive)
    coupled_members, _ = coupled.parse_coupled_margin_archive(pre_members[preuint8.BASE_MEMBER])
    return coupled_members[coupled.BASE_MEMBER]


class _FakeProfile:
    def colour_for(self, role: str) -> np.ndarray:
        value = {
            "UndrivableBoundary": (20, 20, 20),
            "Road": (40, 40, 40),
            "Lane": (80, 80, 80),
            "Movable": (120, 120, 120),
            "MyCar": (200, 200, 200),
        }[role]
        return np.asarray(value, dtype=np.uint8)


class _FakeCarrier:
    def __init__(self) -> None:
        self.predictor = SimpleNamespace(source_pair_start=0)
        self.z = SimpleNamespace(n_pairs=600)
        self.pose6_codes = np.zeros((600, 6), dtype=np.int16)
        self.layers = tuple(
            SimpleNamespace(role=role) for role in ("UndrivableBoundary", "Road", "Lane", "Movable", "MyCar")
        )
        self.realization_profile = _FakeProfile()
        self.scorer_solved_templates = None
        self.custody = {}
        self._masks = {
            "UndrivableBoundary": np.zeros((384, 512), dtype=bool),
            "Road": np.pad(
                np.ones((128, 320), dtype=bool),
                ((128, 128), (0, 192)),
            ),
            "Lane": np.zeros((384, 512), dtype=bool),
            "Movable": np.zeros((384, 512), dtype=bool),
            "MyCar": np.pad(
                np.ones((64, 128), dtype=bool),
                ((256, 64), (192, 192)),
            ),
        }

    def _mask_for_layer(
        self,
        layer: SimpleNamespace,
        pair_id: int,
        *,
        replace_g1_movable: bool,
    ) -> np.ndarray:
        assert 0 <= pair_id < 600
        assert replace_g1_movable is True
        return self._masks[layer.role].copy()

    def render_camera_pairs(self, pair_ids: tuple[int, ...]) -> np.ndarray:
        from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

        return np.zeros(
            (len(pair_ids), 2, CAMERA_H, CAMERA_W, 3),
            dtype=np.uint8,
        )

    def template_camera_masks(self, pair_ids: tuple[int, ...], template: object) -> np.ndarray:
        raise AssertionError("no template is used by this fixture")


def test_all_24_lane_coordinate_ids_are_unique_and_roundtrip() -> None:
    rows = tuple(LaneProgramCoordinateV1(line, field, 1) for line in range(6) for field in LANE_FIELDS)
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


def test_inactive_rg2_is_same_carrier_byte_identical() -> None:
    carrier = _v19c_carrier()
    assert compile_rg2_receiver_grammar(carrier) == carrier


def test_inactive_rg3_is_same_carrier_byte_identical() -> None:
    carrier = _v19c_carrier()
    assert compile_rg3_receiver_grammar(carrier) == carrier


def test_rg2_skeleton_packet_is_typed_unique_crc_bound_and_roundtrips() -> None:
    rows = (
        SkeletonAmplitudeCoordinateV1(
            14,
            0,
            4,
            "EVENT_LOCAL_BOUNDARY",
            "TRANSIENT",
            2,
            -1,
        ),
        SkeletonAmplitudeCoordinateV1(
            54,
            0,
            4,
            "PER_STRATUM_ROW_BAND",
            "STATIC_IN_IMAGE",
            3,
            1,
        ),
    )
    payload = encode_skeleton_amplitude_coordinates(rows)
    assert decode_skeleton_amplitude_coordinates(payload) == rows
    with pytest.raises(DirectDescriptionError, match="sorted, unique"):
        encode_skeleton_amplitude_coordinates((rows[0], rows[0]))
    mutated = bytearray(payload)
    mutated[-1] ^= 1
    with pytest.raises(DirectDescriptionError, match="CRC"):
        decode_skeleton_amplitude_coordinates(bytes(mutated))


def test_rg2_single_cell_amplitude_is_counted_and_receiver_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _FakeCarrier()
    band = derive_skeleton_amplitude_row_band(
        base,
        pair_index=54,
        class_a=0,
        class_b=4,
        family="PER_STRATUM_ROW_BAND",
    )
    coordinate = SkeletonAmplitudeCoordinateV1(
        54,
        0,
        4,
        "PER_STRATUM_ROW_BAND",
        "STATIC_IN_IMAGE",
        band,
        1,
    )
    monkeypatch.setattr(rg1, "receive_carrier_compose_archive", lambda *_args, **_kwargs: None)
    archive = compile_rg2_receiver_grammar(
        b"sealed-sha-bound-carrier",
        skeleton_amplitudes=(coordinate,),
    )
    assert archive != b"sealed-sha-bound-carrier"
    members = parse_rg1_receiver_grammar(archive)
    assert members["base/v13_v19c_carrier.zip"] == b"sealed-sha-bound-carrier"
    assert "production/skeleton_amplitude_coordinates.rg2sa" in members
    manifest = json.loads(members["manifest.json"])
    assert (
        manifest["skeleton_amplitude_productions"]["typed_stream_tag"]["layer_home"]
        == "L3_raster"
    )
    assert manifest["composition_order"][-1] == "execute evaluator-owned exact R"
    receiver = RG2ReceiverV1(
        archive=archive,
        base=base,
        skeleton_amplitudes=(coordinate,),
        custody={"skeleton_amplitude_typed_stream": "SKELETON/L3_raster"},
    )
    assert (base.render_camera_pairs((54,)) != receiver.render_camera_pairs((54,))).any()


def test_rg2_rejects_row_band_not_derived_from_bound_base() -> None:
    base = _FakeCarrier()
    band = derive_skeleton_amplitude_row_band(
        base,
        pair_index=54,
        class_a=0,
        class_b=4,
        family="PER_STRATUM_ROW_BAND",
    )
    coordinate = SkeletonAmplitudeCoordinateV1(
        54,
        0,
        4,
        "PER_STRATUM_ROW_BAND",
        "STATIC_IN_IMAGE",
        (band + 1) % 6,
        1,
    )
    receiver = RG2ReceiverV1(
        archive=b"fixture",
        base=base,
        skeleton_amplitudes=(coordinate,),
        custody={},
    )
    with pytest.raises(DirectDescriptionError, match="row-band/base binding"):
        receiver.render_camera_pairs((54,))


def test_rg3_packet_is_sorted_unique_crc_bound_and_roundtrips() -> None:
    rows = (
        RG3ResidualCoordinateV1(
            14,
            0,
            4,
            "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK",
            "TRANSIENT",
            2,
            1,
            -2,
        ),
        RG3ResidualCoordinateV1(
            54,
            0,
            4,
            "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK",
            "STATIC_IN_IMAGE",
            3,
            2,
            1,
        ),
    )
    payload = encode_rg3_residual_coordinates(rows)
    assert decode_rg3_residual_coordinates(payload) == rows
    assert rows[0].actuator_id == replace(rows[0], signed_quanta=2).actuator_id
    assert rows[0].actuator_id != replace(rows[0], signed_quanta=-1).actuator_id
    with pytest.raises(DirectDescriptionError, match="sorted, unique"):
        encode_rg3_residual_coordinates((rows[0], rows[0]))
    mutated = bytearray(payload)
    mutated[-1] ^= 1
    with pytest.raises(DirectDescriptionError, match="CRC"):
        decode_rg3_residual_coordinates(bytes(mutated))
    with pytest.raises(DirectDescriptionError, match="exactly one"):
        RG3ResidualCoordinateV1(
            14,
            0,
            4,
            "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION",
            "TRANSIENT",
            2,
            1,
            2,
        )


def test_rg3_receiver_geometry_and_fisher_derivations_are_surgical() -> None:
    base = _FakeCarrier()
    base._masks["Road"] = np.pad(
        np.ones((32, 80), dtype=bool),
        ((144, 208), (200, 232)),
    )
    base._masks["MyCar"] = np.pad(
        np.ones((32, 80), dtype=bool),
        ((144, 208), (280, 152)),
    )
    row_band = derive_skeleton_amplitude_row_band(
        base,
        pair_index=54,
        class_a=0,
        class_b=4,
        family="EVENT_LOCAL_BOUNDARY",
    )
    fine = derive_rg3_finer_event_local_band(
        base,
        pair_index=54,
        class_a=0,
        class_b=4,
        row_band=row_band,
    )
    assert row_band == 2
    assert fine == 1
    margins = np.full((384, 512), 20.0, dtype=np.float32)
    margins[160:176, 200:360] = 0.0
    assert (
        derive_rg3_fisher_margin_band(
            base,
            pair_index=54,
            class_a=0,
            class_b=4,
            row_band=2,
            margin_map=margins,
        )
        == 2
    )
    with pytest.raises(DirectDescriptionError, match="finite nonnegative"):
        derive_rg3_fisher_margin_band(
            base,
            pair_index=54,
            class_a=0,
            class_b=4,
            row_band=2,
            margin_map=np.full((384, 512), -1.0, dtype=np.float32),
        )


def test_rg3_class_birth_is_receiver_derived_counted_and_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _FakeCarrier()
    row_band, fine_band = derive_rg3_class_birth_address(base, pair_index=54)
    coordinate = RG3ResidualCoordinateV1(
        54,
        1,
        2,
        "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION",
        "STATIC_IN_IMAGE",
        row_band,
        fine_band,
        1,
    )
    monkeypatch.setattr(rg1, "receive_carrier_compose_archive", lambda *_args, **_kwargs: None)
    archive = compile_rg3_receiver_grammar(
        b"sealed-sha-bound-carrier",
        rg3_residuals=(coordinate,),
    )
    members = parse_rg1_receiver_grammar(archive)
    assert members["base/v13_v19c_carrier.zip"] == b"sealed-sha-bound-carrier"
    assert "production/residual_family_coordinates.rg3rf" in members
    manifest = json.loads(members["manifest.json"])
    rg3_manifest = manifest["rg3_residual_family_productions"]
    assert rg3_manifest["typed_stream_tag"]["layer_home"] == "L3_raster"
    assert "no scorer label" in rg3_manifest["fisher_source_policy"]
    receiver = RG3ReceiverV1(
        archive=archive,
        base=base,
        residuals=(coordinate,),
        custody={"rg3_residual_typed_stream": "SKELETON/L3_raster"},
    )
    baseline = base.render_camera_pairs((54,))
    realized = receiver.render_camera_pairs((54,))
    assert (baseline != realized).any()
    changed_colours = np.unique(realized.reshape(-1, 3), axis=0)
    assert [80, 80, 80] in changed_colours.tolist()
    assert [20, 20, 20] in changed_colours.tolist()


def test_joint_streams_are_separate_and_typed() -> None:
    carrier = _v19c_carrier()
    archive = compile_rg1_receiver_grammar(
        carrier,
        lane_coordinates=(LaneProgramCoordinateV1(0, "dash_phase_origin_q8", 1),),
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
