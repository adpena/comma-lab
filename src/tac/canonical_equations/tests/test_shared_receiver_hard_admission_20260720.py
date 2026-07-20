# SPDX-License-Identifier: MIT
"""Triality tests for the R1 shared-receiver hard-admission law."""

from __future__ import annotations

import pytest

from tac.boundary_math.shared_receiver_admission import BLOCKER_ID
from tac.canonical_equations.shared_receiver_hard_admission_20260720 import (
    EQUATION_ID,
    build_shared_receiver_counted_spatial_hard_oracle_admission_v1,
    shared_receiver_hard_admission_certificate,
)


def _certificate(**overrides):
    values = {
        "n_pairs": 600,
        "archive_bytes": 286_000,
        "d_seg": 0.0003,
        "d_pose": 0.0001,
        "exact_archive": True,
        "archive_parseback_identical": True,
        "production_receiver": True,
        "through_r_authority": True,
        "hard_cpu_torch_oracle": True,
        "packet_mutation_changes_decoded": True,
        "scorer_free_spatial_rgb_pullback": True,
        "content_hashes_bound": True,
    }
    values.update(overrides)
    return shared_receiver_hard_admission_certificate(**values)


def test_structural_conjunction_cannot_self_confer_authority() -> None:
    result = _certificate()
    assert result["structural_conjunction"] is True
    assert result["trusted_contest_cpu_verifier_wired"] is False
    assert result["accepted"] is False
    assert result["status"] == BLOCKER_ID
    for predicate in (
        "exact_archive",
        "archive_parseback_identical",
        "production_receiver",
        "through_r_authority",
        "hard_cpu_torch_oracle",
        "packet_mutation_changes_decoded",
        "scorer_free_spatial_rgb_pullback",
        "content_hashes_bound",
    ):
        result = _certificate(**{predicate: False})
        assert result["structural_conjunction"] is False
        assert result["accepted"] is False
        assert result["status"] == BLOCKER_ID


def test_hard_admission_rejects_rate_distortion_and_type_laundering() -> None:
    assert _certificate(archive_bytes=286_681)["accepted"] is False
    assert _certificate(d_seg=0.000340)["accepted"] is False
    assert _certificate(n_pairs=599)["accepted"] is False
    with pytest.raises(ValueError, match="must be boolean"):
        _certificate(through_r_authority=1)
    with pytest.raises(ValueError, match="finite"):
        _certificate(d_seg=float("nan"))


def test_equation_builds_with_measured_dense_anchor() -> None:
    equation = build_shared_receiver_counted_spatial_hard_oracle_admission_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["blocker_id"] == BLOCKER_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["section_zip_bytes"] == 561_502_227
    assert anchor.empirical_output["through_r_authority"] is False
    assert anchor.empirical_output["d_seg"] is None
