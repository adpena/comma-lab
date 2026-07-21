# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lane_sdf_component import LaneLine
from tac.optimization.boundary_inverse_custody import (
    CURVELET_WIDTH,
    DICTIONARY_WIDTH,
    BoundaryInverseError,
    SparseInverseProgram,
    apply_sparse_program,
    authority_labels,
    chart_correction_domain,
    decode_program,
    deterministic_training_indices,
    dictionary_metadata,
    encode_program,
    fit_sparse_program,
    flip_accounting,
)


def _line(*, lateral: float = 0.0, dashed: bool = True) -> LaneLine:
    return LaneLine(
        centerline_coeffs=np.asarray([lateral], dtype=np.float64),
        halfwidth_coeffs=np.asarray([2.0], dtype=np.float64),
        dash_period_m=8.0 if dashed else 0.0,
        dash_phase_m=1.25 if dashed else 0.0,
        dash_duty=0.5,
        forward_range=(5.0, 100.0),
        n_pixels=100,
    )


def _program(*, mode: str = "generic_2d", bins: int = 1) -> SparseInverseProgram:
    qcoeff = np.zeros((bins, DICTIONARY_WIDTH), dtype=np.int8)
    qcoeff[:, 3] = 96
    qcoeff[:, CURVELET_WIDTH] = -7
    return SparseInverseProgram(
        coordinate_mode=mode,
        phase_bin_count=bins,
        qstep=1.0 / 64.0,
        threshold=0.2,
        qcoeff=qcoeff,
    )


def test_dictionary_is_genuine_mixed_finite_frame() -> None:
    metadata = dictionary_metadata()
    assert metadata["dictionary_width"] == DICTIONARY_WIDTH
    assert metadata["literal_polar_curvelet_columns"] == CURVELET_WIDTH
    assert metadata["compact_shearlet_columns"] > 0
    assert metadata["families"] == ["literal_polar_curvelet", "compact_shearlet"]
    assert metadata["forbidden_substitute"] == "Fourier"
    assert len(metadata["literal_polar_curvelet_atom_spec_sha256"]) == 64
    assert len(metadata["compact_shearlet_config_sha256"]) == 64


def test_program_context_arithmetic_roundtrip_is_exact_and_strict() -> None:
    program = _program(mode="dash_arc_phase", bins=4)
    first = encode_program(program)
    second = encode_program(program)
    assert first == second
    restored = decode_program(first)
    assert restored.coordinate_mode == "dash_arc_phase"
    assert restored.phase_bin_count == 4
    assert restored.qcoeff.dtype == np.int8
    assert np.array_equal(restored.qcoeff, program.qcoeff)
    assert encode_program(restored) == first
    with pytest.raises(BoundaryInverseError, match="truncated"):
        decode_program(first[:-1])
    with pytest.raises(BoundaryInverseError, match="trailing"):
        decode_program(first + b"x")


def test_phase_domain_is_structurally_distinct_and_requires_dash() -> None:
    generic = chart_correction_domain(
        [_line(dashed=False)],
        height=384,
        width=512,
        coordinate_mode="generic_2d",
        phase_bin_count=1,
    )
    phase_without_dash = chart_correction_domain(
        [_line(dashed=False)],
        height=384,
        width=512,
        coordinate_mode="dash_arc_phase",
        phase_bin_count=4,
    )
    phase = chart_correction_domain(
        [_line(dashed=True)],
        height=384,
        width=512,
        coordinate_mode="dash_arc_phase",
        phase_bin_count=4,
    )
    assert generic.flat_indices.size > 0
    assert phase_without_dash.flat_indices.size == 0
    assert phase.flat_indices.size > 0
    assert np.unique(phase.phase_bins).size > 1
    assert np.all(np.abs(phase.coords) <= 1.0 + 1e-6)
    assert not np.array_equal(generic.coords[:100], phase.coords[:100])


def test_sparse_solve_is_deterministic_and_screens_both_families() -> None:
    axis = np.linspace(-1.0, 1.0, 25, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    residual = np.where(xx.reshape(-1) > 0.2, 1, np.where(xx.reshape(-1) < -0.2, -1, 0))
    bins = (yy.reshape(-1) >= 0.0).astype(np.int16)
    kwargs = {
        "coords": coords,
        "phase_bins": bins,
        "residual": residual,
        "coordinate_mode": "dash_arc_phase",
        "phase_bin_count": 2,
        "atoms_per_bin": 4,
    }
    first = fit_sparse_program(**kwargs)
    second = fit_sparse_program(**kwargs)
    assert np.array_equal(first.program.qcoeff, second.program.qcoeff)
    assert first.selected_atom_ids_by_bin == second.selected_atom_ids_by_bin
    for selected in first.selected_atom_ids_by_bin:
        assert any(atom < CURVELET_WIDTH for atom in selected)
        assert any(atom >= CURVELET_WIDTH for atom in selected)


def test_sparse_solve_and_sampling_refuse_invalid_controls() -> None:
    coords = np.zeros((2, 2), dtype=np.float32)
    bins = np.zeros(2, dtype=np.int16)
    residual = np.asarray([-1, 1], dtype=np.int8)
    with pytest.raises(BoundaryInverseError, match="qstep"):
        fit_sparse_program(
            coords=coords,
            phase_bins=bins,
            residual=residual,
            coordinate_mode="generic_2d",
            phase_bin_count=1,
            atoms_per_bin=2,
            qstep=0.0,
        )
    with pytest.raises(BoundaryInverseError, match="sample caps"):
        deterministic_training_indices(residual, max_zero=-1)


def test_decoder_applies_same_selected_atoms_to_another_chart() -> None:
    program = decode_program(encode_program(_program()))
    baseline = np.zeros((384, 512), dtype=bool)
    first, first_stats = apply_sparse_program(baseline, [_line(lateral=0.0)], program)
    second, second_stats = apply_sparse_program(baseline, [_line(lateral=1.0)], program)
    assert first_stats["support_pixels"] > 0
    assert second_stats["support_pixels"] > 0
    assert np.count_nonzero(first) > 0
    assert np.count_nonzero(second) > 0
    assert not np.array_equal(first, second)


def test_flip_accounting_reports_eaten_and_remaining_residual() -> None:
    truth = np.asarray([[1, 1, 0, 0]], dtype=bool)
    baseline = np.asarray([[0, 1, 1, 0]], dtype=bool)
    corrected = np.asarray([[1, 0, 0, 0]], dtype=bool)
    row = flip_accounting(baseline, corrected, truth)
    assert row == {
        "pixels": 4,
        "changed": 3,
        "beneficial": 2,
        "harmful": 1,
        "remaining_false_negative": 1,
        "remaining_false_positive": 0,
    }


def test_authority_is_advisory_and_not_score() -> None:
    labels = authority_labels()
    assert labels["axis"] == "[macOS-CPU advisory]"
    assert labels["score_claim"] is False
    assert labels["promotion_eligible"] is False
    assert labels["mask_f1_is_dseg"] is False
    assert labels["waterfill_status_without_through_r_recovery"] == "FORMALIZATION_PENDING"
    assert "UNMOVED" in labels["pointer"]
