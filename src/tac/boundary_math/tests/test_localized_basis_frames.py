from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tac.boundary_math import localized_basis_frames as lbf


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_literal_family_has_fixed_80_column_order_and_new_identity() -> None:
    specs = lbf.deterministic_atom_specs()
    assert lbf.FAMILY == "literal_polar_curvelet"
    assert lbf.FEATURE_WIDTH == 80
    assert lbf.SCALING_WIDTH == 4
    assert lbf.DIRECTIONAL_WIDTH == 76
    assert len(specs) == 80
    assert [spec.column for spec in specs] == list(range(80))
    assert all(spec.kind == "scaling" for spec in specs[:4])
    assert all(spec.kind == "directional" for spec in specs[4:])
    assert lbf.scaling_column_mask().sum() == 4
    assert lbf.directional_column_mask().sum() == 76
    assert lbf.ATOM_SPEC_SHA256 == "48df53b84660396adc522fe966cb8e7c631c108332a3529eefe17ee9aaa44f6e"
    assert lbf.ATOM_SPEC_SHA256 != lbf.HISTORICAL_ATOM_SPEC_SHA256_EXPECTED_NONIDENTICAL
    assert lbf.module_source_sha256() != lbf.LOST_SOURCE_SHA256_NON_AUTHORIZING


def test_directional_order_scales_orientations_and_hash_ranked_translation_budget() -> None:
    specs = lbf.ATOM_SPECS[4:]
    assert [(j, sum(atom.scale == j for atom in specs)) for j in lbf.SCALES] == [
        (0, 4),
        (2, 16),
        (4, 56),
    ]
    for j, count in zip(lbf.SCALES, lbf.ORIENTATION_COUNTS, strict=True):
        orientations = {atom.orientation for atom in specs if atom.scale == j}
        assert orientations == set(range(count))
    order = [(atom.scale, atom.orientation) for atom in specs]
    assert order == sorted(order)

    max_residual = 0.0
    for atom in specs:
        assert atom.theta is not None and atom.scale is not None
        ct, st = math.cos(atom.theta), math.sin(atom.theta)
        recovered_normal = atom.center_x * ct + atom.center_y * st
        recovered_tangent = -atom.center_x * st + atom.center_y * ct
        max_residual = max(
            max_residual,
            abs(recovered_normal - atom.translation_normal_index * 2.0 ** (-atom.scale)),
            abs(recovered_tangent - atom.translation_tangent_index * 2.0 ** (-atom.scale / 2.0)),
        )
    assert max_residual < 2e-15
    assert any(atom.translation_normal_index != 0 for atom in specs)
    assert any(atom.translation_tangent_index != 0 for atom in specs)


def test_q1_is_partition_of_unity_inside_and_zero_outside() -> None:
    coords = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [0.2, -0.3], [1.1, 0.0]],
        dtype=np.float32,
    )
    q1 = lbf.localized_basis_features_numpy(coords)[:, :4]
    np.testing.assert_array_equal(q1[:4], np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(q1[:5].sum(axis=1), 1.0, atol=2e-7, rtol=0.0)
    np.testing.assert_array_equal(q1[5], 0.0)


def test_literal_polar_window_is_compact_even_hermitian_and_excludes_dc() -> None:
    q = np.arange(-160, 161, dtype=np.float32) * np.float32(0.5)
    xi_x, xi_y = np.meshgrid(q, q, indexing="xy")
    for j, n_orient, center, half in zip(
        lbf.SCALES,
        lbf.ORIENTATION_COUNTS,
        lbf.RADIAL_CENTERS,
        lbf.RADIAL_HALF_WIDTHS,
        strict=True,
    ):
        assert center - half > 0.0
        for orientation in range(n_orient):
            window = lbf.curvelet_frequency_window_numpy(
                xi_x, xi_y, scale=j, orientation=orientation
            )
            np.testing.assert_array_equal(window, window[::-1, ::-1])
            assert window[160, 160] == 0.0
            radial = np.hypot(xi_x, xi_y)
            assert np.max(window[(radial <= center - half) | (radial >= center + half)]) == 0.0


def test_directional_atoms_are_period_two_but_q1_is_not_falsely_relabelled_periodic() -> None:
    coords = np.asarray(
        [[-0.83, -0.41], [0.14, 0.72], [0.93, -0.95]], dtype=np.float32
    )
    reference = lbf.localized_basis_features_numpy(coords)
    shifted_x = lbf.localized_basis_features_numpy(coords + np.asarray([2.0, 0.0], np.float32))
    shifted_y = lbf.localized_basis_features_numpy(coords + np.asarray([0.0, 2.0], np.float32))
    np.testing.assert_allclose(reference[:, 4:], shifted_x[:, 4:], atol=8e-6, rtol=0.0)
    np.testing.assert_allclose(reference[:, 4:], shifted_y[:, 4:], atol=8e-6, rtol=0.0)
    assert not np.array_equal(reference[:, :4], shifted_x[:, :4])


def test_direct_and_alias_summed_inclusive_grid_agree_and_copy_periodic_endpoints() -> None:
    # Dyadic base-grid spacing is represented exactly in fp32, so this isolates
    # evaluator parity from non-dyadic coordinate quantization.
    height, width = 17, 17
    coords = lbf.inclusive_grid_coords(height, width)
    direct = lbf.localized_basis_features_numpy(coords)
    fft = lbf.localized_basis_features_grid_numpy(height, width)
    np.testing.assert_allclose(direct, fft, atol=8e-6, rtol=0.0)
    cube = fft.reshape(height, width, 80)
    np.testing.assert_array_equal(cube[0, :, 4:], cube[-1, :, 4:])
    np.testing.assert_array_equal(cube[:, 0, 4:], cube[:, -1, 4:])


def test_structural_proof_covers_aspect_localization_alignment_and_anti_fourier() -> None:
    receipt = lbf.genuine_frame_proof(diagnostic_grid_size=129)
    assert receipt.passed, receipt.to_dict()
    assert all(receipt.gates.values())
    aspects = dict(receipt.proof["measured_support_aspect_by_scale"])
    np.testing.assert_allclose(
        [aspects[0], aspects[2], aspects[4]], [1.0, 2.0, 4.0], atol=0.02, rtol=0.02
    )
    assert receipt.proof["directional_top10_energy_median"] > 0.45
    assert receipt.proof["direct_fft_max_abs_error"] < 8e-6
    assert receipt.proof["inclusive_endpoint_max_abs_error"] == 0.0
    assert "no continuum" in receipt.verdict_scope

    # A global plane wave has no spatial tail decay: its energy is present over
    # essentially the complete grid, unlike the compact-wedge inverse atoms.
    coords = lbf.inclusive_grid_coords(65, 65)
    global_plane_energy = np.cos(2.0 * np.pi * 7.0 * coords[:, 0]).astype(np.float64) ** 2
    top10 = np.sort(global_plane_energy)[-max(1, len(global_plane_energy) // 10) :].sum()
    assert top10 / global_plane_energy.sum() < receipt.proof["directional_top10_energy_median"]


def test_reimplementation_feature_hashes_are_deterministic_and_not_lost_source_custody() -> None:
    expected = {
        3: "a5e6c5450a459e3b5f1457cd7246f38e47069669ee5ef058d7ef91436597dc4d",
        17: "dc192dd1cb8f7f90decd39794183c33a636fdd370c1ce9400be8d32d117df860",
        33: "a97c32df556ae029b74ba7baded969e4459b38f45da15b8854a540ceeef3c033",
    }
    for size, digest in expected.items():
        features = lbf.localized_basis_features_numpy(lbf.inclusive_grid_coords(size, size))
        assert features.dtype == np.float32
        assert features.shape == (size * size, 80)
        assert _sha(features) == digest


def test_generated_inflate_contract_is_content_addressed_and_executable_source() -> None:
    source = lbf.inflate_embedded_numpy_source()
    compile(source, "<literal-polar-curvelet-generated-inflate>", "exec")
    contract = lbf.inflate_compile_contract()
    assert contract.compiled
    assert contract.entrypoint == "localized_basis_features_numpy"
    assert contract.atom_spec_sha256 == lbf.ATOM_SPEC_SHA256
    assert contract.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert contract.source_sha256 == lbf.basis_generated_source_sha256()
    assert not contract.lost_source_sha256_claimed
    assert "learned_basis_coefficients" in contract.counted_state
    assert lbf.basis_semantic_sha256() == lbf.basis_semantic_sha256()


def test_basis_program_config_binds_every_semantic_operator() -> None:
    taper_hash = lbf.basis_program_taper_config_sha256(strength=1.0, scale=0.0, floor=0.05)
    config = lbf.literal_basis_program_config(
        chart_enabled=True,
        chart_pose_dependency="counted_pose_carrier_xi",
        native_orientation_enabled=True,
        native_orientation_kappa=2.5,
        fixed_point_iteration_cap=6,
        taper_enabled=True,
        taper_train_config_sha256=taper_hash,
        aa_mode="supersample",
        aa_factor=2,
    )
    assert config.feature_width == 80
    assert config.atom_spec_sha256 == lbf.ATOM_SPEC_SHA256
    assert lbf.BasisProgramConfig.from_dict(config.to_dict()) == config
    assert lbf.BasisProgramConfig.from_dict(config.to_dict()).canonical_sha256() == (
        config.canonical_sha256()
    )
    changed = lbf.literal_basis_program_config(
        chart_enabled=True,
        chart_pose_dependency="counted_pose_carrier_xi",
        native_orientation_enabled=True,
        native_orientation_kappa=2.75,
        fixed_point_iteration_cap=6,
        taper_enabled=True,
        taper_train_config_sha256=taper_hash,
        aa_mode="supersample",
        aa_factor=2,
    )
    assert changed.canonical_sha256() != config.canonical_sha256()


def test_basis_program_config_refuses_unknown_hashes_and_unclosed_chart() -> None:
    with pytest.raises(ValueError, match="atom-spec hash drift"):
        lbf.BasisProgramConfig(atom_spec_sha256="0" * 64)
    with pytest.raises(ValueError, match="counted receiver dependency"):
        lbf.literal_basis_program_config(chart_enabled=True)
    with pytest.raises(ValueError, match=r"scalar|permits only"):
        lbf.literal_basis_program_config(aa_mode="ipe")
    with pytest.raises(ValueError, match="positive fixed-point cap"):
        lbf.literal_basis_program_config(native_orientation_enabled=True)


def test_mlx_parity_when_available_otherwise_soft_unavailable_receipt() -> None:
    receipt = lbf.mlx_parity_receipt(height=5, width=7)
    assert receipt["authority"] is False
    if receipt["status"] == "UNMEASURED_SOFT_UNAVAILABLE":
        assert receipt["reason"]
        pytest.skip(receipt["reason"])
    assert receipt["status"] == "MEASURED_PARITY"
    assert receipt["passed"]
    assert receipt["max_abs_error"] <= receipt["tolerance"]
