from __future__ import annotations

from dataclasses import replace

import pytest

from tac.boundary_math.compact_shearlet_frame import n_atoms as compact_shearlet_n_atoms
from tac.boundary_math.windowed_curvelet_frame import n_atoms as windowed_curvelet_n_atoms
from tac.witness_dsl.basis_control import (
    GENUINE_FRAME_FEATURE_WIDTH,
    genuine_frame_compact_shearlet_config,
    genuine_frame_equal_value_budget,
    genuine_frame_windowed_curvelet_config,
)
from tac.witness_dsl.lever_registry import lever_factories
from tac.witness_dsl.optimal_basis_20260714 import (
    BASIS_ABC_SCIENTIFIC_DECLARATION,
    BASIS_ABC_SCIENTIFIC_DECLARATION_SHA256,
    BasisFamily,
    BasisLeverSpec,
    inflate_compile_contract,
    v9_ideal_mod32_basis_abc_configs,
    validate_basis_abc_scientific_declaration,
)


def test_compact_shearlet_is_a_mapped_typed_basis_lever() -> None:
    spec = BasisLeverSpec(family=BasisFamily.COMPACT_SHEARLET)
    overrides = spec.compile_lever().overrides
    assert overrides["--basis"] == "compact_shearlet"
    assert overrides["--self-orient"] is False
    assert overrides["--bank-n-scales"] == 4
    contract = inflate_compile_contract(spec)
    assert contract.compiled
    assert "compact_shearlet_feats" in contract.train_functions
    assert "_compact_shearlet_feats" in contract.inflate_functions
    assert lever_factories()["CompactShearletBasis"] == frozenset(overrides)


def test_scientific_declaration_is_literal_sealed_and_mutation_refuses() -> None:
    receipt = validate_basis_abc_scientific_declaration()
    assert receipt["scientific_declaration_sha256"] == BASIS_ABC_SCIENTIFIC_DECLARATION_SHA256
    assert receipt["status"] == "RESEALED_SOURCE_AND_CONSUMER_CLOSED"
    mutated = tuple(dict(row) for row in BASIS_ABC_SCIENTIFIC_DECLARATION)
    mutated[0]["consumer"] = "invented"
    with pytest.raises(RuntimeError, match="scientific declaration seal mismatch"):
        validate_basis_abc_scientific_declaration(mutated)


def test_v9_three_arm_configs_have_one_basis_delta_and_gauge_custody() -> None:
    configs = v9_ideal_mod32_basis_abc_configs(num_pairs=600, epochs=3000)
    assert set(configs.pairwise_differing_flags_excluding_out_dir.values()) == {("--basis",)}
    expected = {
        "polar_fourier": "legacy_fourier_ab_control",
        "windowed_curvelet": "windowed_curvelet",
        "compact_shearlet": "compact_shearlet",
    }
    expected_ids = {
        row["family"]: row["config_id"] for row in BASIS_ABC_SCIENTIFIC_DECLARATION
    }
    for name, basis in expected.items():
        config = getattr(configs, name)
        assert config.name == expected_ids[basis]
        assert dict(config.to_trainer_flags())["--basis"] == basis
        basis_levers = [lever for lever in config.dsl_levers if lever.startswith("basis_family::")]
        assert basis_levers == [f"basis_family::{basis}"]
        declaration = config.dsl_program_manifest["genuine_frame_basis_abc"]
        gauge = config.dsl_program_manifest["affine_legendre_gauge_pair"]
        bijection = config.dsl_program_manifest["genuine_frame_config_provenance_bijection"]
        assert declaration["scientific_declaration_sha256"] == BASIS_ABC_SCIENTIFIC_DECLARATION_SHA256
        assert bijection["config_id"] == config.name
        assert bijection["family"] == basis
        assert bijection["lawref"] == "optimal_basis_equal_budget_through_r_v1"
        assert bijection["feature_width"] == 80
        assert bijection["self_orient"] is False
        assert bijection["equal_value_budget"]["decoder_values"] == 71_159
        assert bijection["equal_value_budget"]["per_pair_code_values"] == 38_400
        assert bijection["equal_value_budget"]["total_trainable_values"] == 109_559
        assert bijection["duplicate_long_flags"] == []
        assert (
            config.constants_manifest["optimal_basis_equal_budget_through_r"]["equation_id"]
            == "optimal_basis_equal_budget_through_r_v1"
        )
        assert gauge["divergence_abs_error"] <= gauge["tolerance"]
        assert gauge["action_abs_error"] <= gauge["tolerance"]
        assert gauge["custody"]["r_operator"]
        assert gauge["custody"]["xi_chart"]


def test_genuine_frame_factories_are_equal_width_and_budget() -> None:
    curvelet = genuine_frame_windowed_curvelet_config()
    shearlet = genuine_frame_compact_shearlet_config()
    assert 2 * windowed_curvelet_n_atoms(curvelet) == GENUINE_FRAME_FEATURE_WIDTH
    assert 2 * compact_shearlet_n_atoms(shearlet) == GENUINE_FRAME_FEATURE_WIDTH
    assert genuine_frame_equal_value_budget(num_pairs=600) == {
        "feature_width": 80,
        "hidden_dim": 96,
        "hidden_layers": 4,
        "mod_dim": 32,
        "pair_frame_code_rows": 1200,
        "in_projection": 7776,
        "film_projection": 25344,
        "hidden_stack": 37248,
        "sdf_head": 485,
        "texture_head": 291,
        "palette": 15,
        "decoder_values": 71159,
        "per_pair_code_values": 38400,
        "total_trainable_values": 109559,
    }


def test_launcher_resolves_all_three_registered_config_ids() -> None:
    from tools import launch_witness_run as launcher

    expected = {
        "v9_cgauge_ideal_mod32_basis_polar_fourier": "legacy_fourier_ab_control",
        "v9_cgauge_ideal_mod32_basis_windowed_curvelet": "windowed_curvelet",
        "v9_cgauge_ideal_mod32_basis_compact_shearlet": "compact_shearlet",
    }
    for config_id, family in expected.items():
        config = launcher.derive_named_config(
            config_id,
            "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
            num_pairs=8,
            epochs=3000,
            overfit=True,
        )
        assert config.name == config_id
        assert launcher.config_family(config) == config_id
        assert dict(config.to_trainer_flags())["--basis"] == family


def test_affine_gauge_pair_refuses_singular_transform() -> None:
    from tac.witness_dsl.affine_legendre_gauge_policy import (
        AffineLegendreGaugePolicyError,
        canonical_v9_affine_legendre_gauge_pair,
    )

    pair = canonical_v9_affine_legendre_gauge_pair()
    singular = replace(
        pair,
        transform=replace(
            pair.transform,
            matrix=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    )
    with pytest.raises(AffineLegendreGaugePolicyError, match="invertible"):
        singular.verify((0.8, -0.3, 0.2), (-0.1, 0.6, -0.4))
