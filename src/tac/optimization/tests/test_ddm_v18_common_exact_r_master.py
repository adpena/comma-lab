# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_v18_common_exact_r_master import (
    PROFILE_MEMBER,
    compile_common_exact_r_master,
    parse_common_exact_r_master,
    receive_common_exact_r_master,
)
from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError

REPO_ROOT = Path(__file__).resolve().parents[4]
V12_ARCHIVE = (
    REPO_ROOT
    / ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/"
    "ddm_v12_obligation_n600_add0.not_a_candidate.zip.receipt-bytes"
)
V15_ARCHIVE = (
    REPO_ROOT
    / ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
    "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)


def _inputs() -> tuple[bytes, ReceiverRealizationProfileV1, object]:
    v15 = receive_carrier_compose_archive(V15_ARCHIVE.read_bytes())
    assert v15.realization_profile is not None
    assert v15.scorer_solved_templates is not None
    return V12_ARCHIVE.read_bytes(), v15.realization_profile, v15.scorer_solved_templates


def test_common_master_is_deterministic_strict_and_camera_renderable() -> None:
    base, profile, bank = _inputs()
    first = compile_common_exact_r_master(base, profile, bank)
    second = compile_common_exact_r_master(base, profile, bank)
    assert first == second
    members, _homes = parse_common_exact_r_master(first)
    assert members["base/ddm_v12_postsolve.zip"] == base
    receiver = receive_common_exact_r_master(first)
    camera = receiver.render_camera_pairs((0,))
    assert camera.shape == (1, 2, 874, 1164, 3)
    assert camera.dtype == np.uint8
    assert receiver.custody["postsolve_only"] is True
    assert receiver.custody["predict_productions_present"] is False


def test_common_master_refuses_predict_base() -> None:
    _base, profile, bank = _inputs()
    with pytest.raises(DirectDescriptionError, match="PREDICT"):
        compile_common_exact_r_master(V15_ARCHIVE.read_bytes(), profile, bank)


def test_common_master_refuses_mutated_profile_payload() -> None:
    base, profile, bank = _inputs()
    archive = compile_common_exact_r_master(base, profile, bank)
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        members = {row.filename: reader.read(row) for row in reader.infolist()}
    changed = bytearray(members[PROFILE_MEMBER])
    changed[-1] ^= 1
    members[PROFILE_MEMBER] = bytes(changed)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as writer:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            writer.writestr(info, payload)
    with pytest.raises(DirectDescriptionError):
        parse_common_exact_r_master(output.getvalue())
