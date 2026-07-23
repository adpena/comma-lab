# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    compile_carrier_compose_archive,
)
from tac.optimization.direct_description_coupled_margin import (
    PROGRAM_MEMBER,
    CoupledMarginProgramV1,
    SparseCameraCompensationV1,
    TemplatePlacementV1,
    compile_coupled_margin_archive,
    coupled_margin_byte_rows,
    decode_coupled_margin_program,
    encode_coupled_margin_program,
    parse_coupled_margin_archive,
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_g1_worldsheet import encode_g1_movable_worldsheet
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.tests.test_direct_description_carrier_compose import _predictor


def _base() -> bytes:
    labels = np.zeros((64, 384, 512), dtype=np.int64)
    labels[:, 24:40, 32:48] = 3
    worldsheet, _metadata = encode_g1_movable_worldsheet(labels)
    profile = ReceiverRealizationProfileV1(
        role_rgb_u8=((0, 153, 0), (11, 3, 9), (51, 255, 204), (107, 0, 114), (63, 72, 63))
    )
    bank = ScorerSolvedTemplateBankV1(
        (
            RowBandScorerTemplateV1("Lane", "inner_boundary", 0, 384, 1, 1, bytes((51, 255, 204))),
            RowBandScorerTemplateV1(
                "Movable",
                "fill",
                0,
                384,
                2,
                2,
                bytes((10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)),
            ),
        )
    )
    return compile_carrier_compose_archive(
        _predictor(),
        worldsheet_g1_payload=worldsheet,
        realization_profile=profile,
        scorer_solved_templates=bank,
    )[0]


def test_program_codec_is_canonical_and_roundtrips() -> None:
    program = CoupledMarginProgramV1(
        placements=(TemplatePlacementV1(0, 1, 1, 1),),
        compensations=(SparseCameraCompensationV1(0, 1, 100, 200, (1, -2, 3)),),
    )
    encoded = encode_coupled_margin_program(program)
    assert decode_coupled_margin_program(encoded) == program
    with pytest.raises(DirectDescriptionError, match="duplicated"):
        CoupledMarginProgramV1(
            compensations=(
                SparseCameraCompensationV1(0, 1, 100, 200, (1, 0, 0)),
                SparseCameraCompensationV1(0, 1, 100, 200, (1, 0, 0)),
            )
        )


def test_outer_receiver_extends_v15_with_counted_phase_and_sparse_compensation() -> None:
    base = _base()
    program = CoupledMarginProgramV1(
        placements=(TemplatePlacementV1(0, 1, 1, 1),),
        compensations=(SparseCameraCompensationV1(0, 1, 100, 200, (1, -2, 3)),),
    )
    archive = compile_coupled_margin_archive(base, program)
    again = compile_coupled_margin_archive(base, program)
    assert archive == again
    members, homes = parse_coupled_margin_archive(archive)
    assert members[PROGRAM_MEMBER] == encode_coupled_margin_program(program)
    assert sum(int(row["zip_home_bytes"]) for row in homes) < len(archive)
    receiver = receive_coupled_margin_archive(archive)
    before = receiver.base.render_camera_pairs((0,))
    after = receiver.render_camera_pairs((0,))
    assert receiver.custody["scorer_weights_present"] is False
    assert receiver.custody["ground_truth_argmax_present"] is False
    assert after.shape == before.shape
    assert not np.array_equal(after, before)
    assert np.array_equal(
        after[0, 1, 100, 200].astype(np.int16),
        np.clip(before[0, 1, 100, 200].astype(np.int16) + np.asarray((1, -2, 3)), 0, 255),
    )
    rows = coupled_margin_byte_rows(archive)
    assert any(row["stratum"] == "template_placements_plus_sparse_compensation" for row in rows)


def test_outer_archive_mutation_refuses_or_changes_program() -> None:
    base = _base()
    program = CoupledMarginProgramV1(
        compensations=(SparseCameraCompensationV1(0, 1, 100, 200, (1, 0, 0)),)
    )
    archive = compile_coupled_margin_archive(base, program)
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        members = {row.filename: reader.read(row) for row in reader.infolist()}
    damaged = bytearray(members[PROGRAM_MEMBER])
    damaged[-1] ^= 1
    members[PROGRAM_MEMBER] = bytes(damaged)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as writer:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            writer.writestr(info, payload)
    with pytest.raises(DirectDescriptionError, match="manifest custody"):
        parse_coupled_margin_archive(output.getvalue())
