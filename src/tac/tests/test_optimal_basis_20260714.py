from __future__ import annotations

import pytest

from tac.canonical_equations.optimal_basis_selection_20260714 import (
    MEASURED_EP675_ROWS,
    build_optimal_basis_equal_budget_through_r_v1,
    select_basis_under_equal_budget,
)
from tac.witness_dsl.optimal_basis_20260714 import (
    BasisFamily,
    BasisLeverSpec,
    audit_legacy_polar_bank,
    basis_catalog,
    basis_metric_interface,
    inflate_compile_contract,
    lever_argv,
    v9_ideal_mod32_basis_ab_configs,
)


def test_candidate_catalog_covers_requested_families_without_guessed_dseg() -> None:
    rows = {row.family: row for row in basis_catalog()}
    assert set(rows) == set(BasisFamily)
    for family in (
        BasisFamily.WINDOWED_CURVELET,
        BasisFamily.COMPACT_SHEARLET,
        BasisFamily.STEERABLE_GABOR,
        BasisFamily.WAVELET,
        BasisFamily.HASH_GRID,
        BasisFamily.BSPLINE_RBF,
        BasisFamily.ZERNIKE_SPHERICAL,
        BasisFamily.LAPLACIAN_EIGEN,
        BasisFamily.NTK_OPTIMAL,
    ):
        assert rows[family].equal_budget_dseg is None


def test_legacy_curvelet_name_is_global_polar_fourier_by_structure() -> None:
    audit = audit_legacy_polar_bank()
    assert audit.representation_label == "polar_directional_fourier"
    assert audit.frequency_columns == 40
    assert audit.paired_feature_columns == 80
    assert audit.maximum_envelope_span < 1e-5
    assert not audit.spatially_localized
    assert not audit.has_translation_index
    assert not audit.has_spatial_window
    assert audit.has_parabolic_orientation_count


def test_measured_fallback_compiles_only_real_trainer_flags_and_inflate_ops() -> None:
    spec = BasisLeverSpec(family=BasisFamily.POLAR_DIRECTIONAL_FOURIER)
    lever = spec.compile_lever()
    argv = lever_argv(lever)
    assert "--no-self-orient" in argv
    contract = inflate_compile_contract(spec)
    assert contract.compiled
    assert contract.family is BasisFamily.POLAR_DIRECTIONAL_FOURIER
    assert contract.inflate_functions == ("_curvelet_B", "_curvelet_feats")


def test_default_basis_spec_is_explicit_legacy_control_without_behavior_loss() -> None:
    spec = BasisLeverSpec()
    assert spec.family is BasisFamily.LEGACY_FOURIER_AB_CONTROL
    overrides = spec.compile_lever().overrides
    assert overrides["--basis"] == "legacy_fourier_ab_control"
    assert overrides["--self-orient"] is False
    assert overrides["--bank-n-scales"] == 4


def test_self_oriented_reproduction_and_wired_windowed_treatment_compile() -> None:
    argv = lever_argv(BasisLeverSpec(
        family=BasisFamily.SELF_ORIENTED_FOURIER,
        freq_along=26.0,
    ).compile_lever())
    assert "--self-orient" in argv
    assert argv[argv.index("--freq-along") + 1] == "26.0"
    treatment = BasisLeverSpec(family=BasisFamily.WINDOWED_CURVELET).compile_lever()
    assert treatment.overrides == {"--basis": "windowed_curvelet"}


def test_siren_finer_compile_checks_periodic_inflate_activation() -> None:
    contract = inflate_compile_contract(BasisLeverSpec(family=BasisFamily.SIREN_FINER))
    assert contract.compiled
    assert "periodic_activation" in contract.train_functions
    assert "finer_bias_initialization" in contract.train_functions
    assert "_act" in contract.inflate_functions


def test_equal_budget_law_selects_off_and_refuses_missing_archive_custody() -> None:
    winner = select_basis_under_equal_budget(max_trainable_values=111_095)
    assert winner.family == "polar_directional_fourier_self_orient_off"
    assert winner.d_seg == 0.004244
    assert winner.trainable_values == 109_559
    with pytest.raises(ValueError, match="no n600 through-R basis row"):
        select_basis_under_equal_budget(MEASURED_EP675_ROWS, max_archive_bytes=200_000)
    equation = build_optimal_basis_equal_budget_through_r_v1()
    assert equation.equation_id == "optimal_basis_equal_budget_through_r_v1"
    assert "no archive-byte comparison" in equation.domain_of_validity["excluded_claims"]


def test_basis_metric_interface_is_non_owning() -> None:
    interface = basis_metric_interface(BasisFamily.WINDOWED_CURVELET)
    assert interface.family_id == "windowed_curvelet"
    assert interface.metric_id == "argmax_native_vjp_fidelity_v1"
    assert interface.state_receipt_schema == "reachable_decision_geometry_fidelity.v1"
    assert interface.selection_receipt_schema == "reachable_decision_preconditioner_selection.v1"
    assert interface.candidate_preconditioner == "winner_rival_margin_fisher_natural"
    assert interface.provider_module == "tac.scorer_surrogate.vjp_fidelity"
    assert interface.selection_status == "NO-VERDICT_DATA_CUSTODY"
    assert "pullback Gram" in interface.required_metric_quantity


def test_v9_mod32_basis_ab_is_explicit_typed_basis_only_but_not_fire_ready() -> None:
    pair = v9_ideal_mod32_basis_ab_configs(num_pairs=6, epochs=900)
    assert pair.control_basis == "legacy_fourier_ab_control"
    assert pair.treatment_basis == "windowed_curvelet"
    assert pair.differing_flags_excluding_out_dir == ("--basis",)
    assert pair.status == "BLOCKED_FAIL_CLOSED_BUILD_THEN_OPERATOR_GO"
    assert pair.launch_ready is False
    assert len(pair.composition_blockers) == 3
    blockers = " ".join(pair.composition_blockers)
    assert "render_aa=ipe" in blockers
    assert "dseg-aware taper" in blockers
    assert "invalidated spatial wave-packet" in blockers
    assert "literal polar-frequency-wedge" in blockers
    assert pair.control.dsl_levers.count("basis_family::legacy_fourier_ab_control") == 1
    assert pair.treatment.dsl_levers.count("basis_family::windowed_curvelet") == 1
