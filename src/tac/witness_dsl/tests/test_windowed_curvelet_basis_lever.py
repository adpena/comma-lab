# SPDX-License-Identifier: MIT
"""Tests for the windowed-curvelet DSL basis lever + its canonical equation (task #502).

Key invariants: the baseline/default computation remains byte-identical under the explicit legacy
Fourier A/B-control identity, the deprecated ``polar_fourier`` alias normalizes to that identity,
the explicit treatment compiles the real ``--basis windowed_curvelet`` flag, the generated receiver
contract is present, registry completeness and the activation duty queue surface it, and the
equation remains registered.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig
from tac.canonical_equations.windowed_curvelet_parabolic_capacity_20260714 import (
    EQUATION_ID,
    build_windowed_curvelet_parabolic_capacity_v1,
    parabolic_sigma_pair,
    populate_windowed_curvelet_parabolic_capacity_equation,
)
from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers
from tac.witness_dsl.curriculum_dsl import BASELINE, build_real_trainer_parser, real_trainer_flags
from tac.witness_dsl.lever_registry import completeness, lever_factories
from tac.witness_dsl.optimal_basis_20260714 import (
    BasisFamily,
    BasisLeverSpec,
    inflate_compile_contract,
)
from tac.witness_dsl.windowed_curvelet_basis_lever_20260714 import (
    WIRE_STATUS_READY,
    WindowedCurveletBasisLeverSpec,
    WindowedCurveletWireNotReady,
    windowed_curvelet_basis_lever,
)


def test_lever_compiles_real_treatment_while_baseline_stays_default_off():
    lev = windowed_curvelet_basis_lever()
    assert lev.name == "basis_family::windowed_curvelet"
    assert lev.overrides == {"--basis": "windowed_curvelet"}
    assert "--basis" not in BASELINE.flag_dict()
    assert lev.epochs_delta == 0


def test_lever_never_invents_trainer_flags():
    lev = WindowedCurveletBasisLeverSpec().compile_lever()
    assert set(lev.overrides).issubset(set(real_trainer_flags()))


def test_deprecated_polar_alias_has_same_effective_parser_config_as_default():
    parser = build_real_trainer_parser()
    default = vars(parser.parse_args(["--out-dir", "x"]))
    explicit = vars(parser.parse_args(["--out-dir", "x", "--basis", "polar_fourier"]))
    assert explicit == default


def test_lever_carries_passing_localization_certificate():
    assert WindowedCurveletBasisLeverSpec().certificate_passes() is True


def test_owed_wire_is_nonempty_and_names_the_through_r_step():
    owed = WindowedCurveletBasisLeverSpec().owed_wire()
    assert len(owed) == 1
    joined = " ".join(owed).lower()
    assert "through-r" in joined and "d_seg" in joined
    assert "operator-go" in joined and "prepared_not_fired" in joined


def test_wire_status_has_op_parity_ready_but_measurement_owed():
    assert WindowedCurveletBasisLeverSpec().wire_status == WIRE_STATUS_READY


def test_lever_notes_keep_capacity_evidence_scoped_as_upper_bound():
    lev = windowed_curvelet_basis_lever()
    assert "UPPER-BOUND" in lev.notes
    assert "OWED" in lev.notes
    assert "certificate_passes=True" in lev.notes


def test_custom_config_refuses_until_checkpoint_and_receiver_serialize_it():
    cfg = WindowedCurveletConfig(n_scales=2, n_orient0=4, aniso=2.0)
    with pytest.raises(WindowedCurveletWireNotReady):
        windowed_curvelet_basis_lever(cfg)


def test_optimal_basis_spec_and_generated_inflate_contract_compile():
    spec = BasisLeverSpec(family=BasisFamily.WINDOWED_CURVELET)
    assert spec.compile_lever().overrides == {"--basis": "windowed_curvelet"}
    contract = inflate_compile_contract(spec)
    assert contract.compiled is True
    assert "_windowed_curvelet_feats" in contract.inflate_functions
    assert "windowed_curvelet_feats" in contract.train_functions


def test_completeness_and_activation_duty_queue_surface_lever(tmp_path):
    factories = lever_factories()
    assert factories["WindowedCurveletBasis"] == frozenset({"--basis"})
    assert factories["LegacyFourierABControl"] == frozenset({"--basis"})
    report = completeness()
    assert "--basis" in report.mapped
    assert "--basis" not in report.unmapped and "--basis" not in report.stale
    assert "WindowedCurveletBasis" in known_levers()
    assert "LegacyFourierABControl" in known_levers()
    assert "WindowedCurveletBasis" in duty_to_measure(path=tmp_path / "empty-ledger.jsonl")


def test_shadow_controller_dashboard_row_consumes_generic_duty_queue(tmp_path, monkeypatch):
    from tac.witness_control.shadow_controller import _duty_to_measure
    from tac.witness_dsl import activation_ledger

    monkeypatch.setattr(activation_ledger, "LEDGER_PATH", tmp_path / "empty-ledger.jsonl")
    rows = _duty_to_measure()
    row = next(r for r in rows if r["lever"] == "WindowedCurveletBasis")
    assert row["state"] == "never-fired"
    assert "OWED a measurement" in row["why"]


def test_equation_builds_with_three_anchors():
    eq = build_windowed_curvelet_parabolic_capacity_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 3
    ids = {a.anchor_id for a in eq.empirical_anchors}
    assert any("localization" in i for i in ids)
    assert any("capacity" in i for i in ids)
    assert any("parabolic" in i for i in ids)


def test_equation_callable_matches_primitive_law():
    """The equation's parabolic_sigma_pair must match the primitive's _sigma_pair law."""
    from tac.boundary_math.windowed_curvelet_frame import _sigma_pair
    cfg = WindowedCurveletConfig(w0=0.5, width_ratio=2.0, aniso=1.5, min_sigma=0.02)
    for j in range(4):
        eq_pair = parabolic_sigma_pair(j, w0=cfg.w0, width_ratio=cfg.width_ratio,
                                       aniso=cfg.aniso, min_sigma=cfg.min_sigma)
        prim_pair = _sigma_pair(cfg, j)
        assert np.allclose(eq_pair, prim_pair, rtol=1e-9)


def test_equation_registers_to_tmp_path(tmp_path):
    reg = tmp_path / "eq_registry.jsonl"
    lock = tmp_path / "eq_registry.lock"
    eq = populate_windowed_curvelet_parabolic_capacity_equation(
        path=reg, lock_path=lock, agent="test", subagent_id="test502"
    )
    assert eq.equation_id == EQUATION_ID
    assert reg.exists()
    assert EQUATION_ID in reg.read_text()


def test_parabolic_sigma_pair_rejects_bad_args():
    with pytest.raises(ValueError):
        parabolic_sigma_pair(-1)
    with pytest.raises(ValueError):
        parabolic_sigma_pair(0, aniso=0.5)
    with pytest.raises(ValueError):
        parabolic_sigma_pair(0, width_ratio=1.0)
