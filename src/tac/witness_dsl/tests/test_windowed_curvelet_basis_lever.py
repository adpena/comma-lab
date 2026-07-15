# SPDX-License-Identifier: MIT
"""Tests for the windowed-curvelet DSL basis lever + its canonical equation (task #502).

Key invariants: the lever is DEFAULT-OFF / byte-identical (empty overrides), never invents a
trainer flag, refuses a fake 'active' compile before the wire lands, carries a passing
localization certificate, and its canonical equation builds + registers (to a tmp path).
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
from tac.witness_dsl.curriculum_dsl import real_trainer_flags
from tac.witness_dsl.windowed_curvelet_basis_lever_20260714 import (
    WIRE_STATUS_OWED,
    WindowedCurveletBasisLeverSpec,
    WindowedCurveletWireNotReady,
    windowed_curvelet_basis_lever,
)


def test_lever_is_byte_identical_default_off():
    lev = windowed_curvelet_basis_lever()
    assert lev.name == "basis_family::windowed_curvelet"
    assert lev.overrides == {}  # empty -> perturbs no sealed config, changes no archive bytes
    assert lev.epochs_delta == 0


def test_lever_never_invents_trainer_flags():
    lev = WindowedCurveletBasisLeverSpec().compile_lever()
    assert set(lev.overrides).issubset(set(real_trainer_flags()))  # trivially: empty set


def test_enabled_without_wire_refuses_fake_active_compile():
    with pytest.raises(WindowedCurveletWireNotReady):
        WindowedCurveletBasisLeverSpec(enabled=True).compile_lever()


def test_lever_carries_passing_localization_certificate():
    assert WindowedCurveletBasisLeverSpec().certificate_passes() is True


def test_owed_wire_is_nonempty_and_names_the_through_r_step():
    owed = WindowedCurveletBasisLeverSpec().owed_wire()
    assert len(owed) >= 4
    joined = " ".join(owed).lower()
    assert "through-r" in joined and "d_seg" in joined
    assert "inflate" in joined and "trainer" in joined


def test_wire_status_is_owed_by_default():
    assert WindowedCurveletBasisLeverSpec().wire_status == WIRE_STATUS_OWED


def test_lever_notes_carry_measured_evidence():
    lev = windowed_curvelet_basis_lever()
    assert "envelope span" in lev.notes
    assert "OWED" in lev.notes
    assert "certificate_passes=True" in lev.notes


def test_lever_accepts_custom_config():
    cfg = WindowedCurveletConfig(n_scales=2, n_orient0=4, aniso=2.0)
    lev = windowed_curvelet_basis_lever(cfg)
    assert lev.overrides == {}


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
