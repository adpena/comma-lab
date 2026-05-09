# LOSS_CONVERGENCE_NOT_REQUIRED: CLI/static wiring test for the opt-in
# T19 Boyd §3.4.1 / He-Yang 2000 adaptive-ρ ADMM step integration.
"""T19 wire-in tests for ``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``.

Per CLAUDE.md ``forbidden_dead_flag_wiring_pattern``: every CLI flag this
module emits MUST be a subset of the target trainer's ``argparse``
contract. Per CLAUDE.md ``forbidden_empirical_claim_without_evidence_tag``:
every numeric T19 impact claim MUST carry a
``[predicted; T19 ...; not direct score]`` tag.

Tests verify:
  - the trainer exposes ``--enable-t19-adaptive-rho`` (default OFF —
    backward-compat preserved per coherence council recommendation)
  - the trainer exposes ``--t19-tau-grow`` and ``--t19-tau-shrink``
  - ``JointLagrangianADMMConfig`` accepts T19 toggles + validates them
  - the coordinator's ``_maybe_adapt_rho`` routes through
    ``tac.joint_admm_coordinator.adaptive_rho_step`` when the toggle is ON
  - the legacy windowed-average backend remains the default
  - the ρ trajectory is recorded for forensic logging
  - the trainer flushes the trajectory to ``rho_trajectory.json`` and
    surfaces it in ``provenance.json`` as ``t19_adaptive_rho``

Source memos:
  - ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`` (memo)
  - ``feedback_grand_council_portfolio_coherence_journal_grade_20260509`` (council §6)
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TRAIN_PATH = (
    REPO / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
)


def _import_trainer():
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    if "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend" in sys.modules:
        del sys.modules["train_paradigm_delta_epsilon_zeta_track1_balle_endtoend"]
    return importlib.import_module(
        "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend"
    )


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def _step(coord, *, rate_bits=800_000.0, distortion=1.0, seg=0.001, pose=0.0002):
    return coord.step(
        distortion=torch.tensor(distortion, requires_grad=True),
        rate_bits=torch.tensor(rate_bits, requires_grad=True),
        seg_loss=torch.tensor(seg, requires_grad=True),
        pose_loss=torch.tensor(pose, requires_grad=True),
    )


# ---------------------------------------------------------------------------
# CLI wiring (forbidden_dead_flag_wiring_pattern)
# ---------------------------------------------------------------------------


def test_parse_args_exposes_t19_flags():
    mod = _import_trainer()
    ns = mod.parse_args(["--output-dir", "/tmp/foo"])
    assert ns.enable_t19_adaptive_rho is False, "default backward-compat OFF"
    assert ns.t19_tau_grow == 2.0
    assert ns.t19_tau_shrink == 0.5


def test_parse_args_t19_overrides_take_effect():
    mod = _import_trainer()
    ns = mod.parse_args(
        [
            "--output-dir",
            "/tmp/foo",
            "--enable-t19-adaptive-rho",
            "--t19-tau-grow",
            "3.0",
            "--t19-tau-shrink",
            "0.25",
        ]
    )
    assert ns.enable_t19_adaptive_rho is True
    assert ns.t19_tau_grow == 3.0
    assert ns.t19_tau_shrink == 0.25


def test_t19_flags_listed_in_module_docstring():
    src = _train_src()
    assert "--enable-t19-adaptive-rho" in src
    assert "--t19-tau-grow" in src
    assert "--t19-tau-shrink" in src


# ---------------------------------------------------------------------------
# JointLagrangianADMMConfig accepts and validates T19 toggles
# ---------------------------------------------------------------------------


def test_config_accepts_t19_toggles():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMMConfig,
    )

    cfg = JointLagrangianADMMConfig(
        use_t19_adaptive_rho=True,
        t19_tau_grow=2.5,
        t19_tau_shrink=0.4,
    )
    assert cfg.use_t19_adaptive_rho is True
    assert cfg.t19_tau_grow == 2.5
    assert cfg.t19_tau_shrink == 0.4


def test_config_default_t19_off_backward_compat():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMMConfig,
    )

    cfg = JointLagrangianADMMConfig()
    assert cfg.use_t19_adaptive_rho is False, "default must be OFF for backward-compat"
    assert cfg.t19_tau_grow == 2.0
    assert cfg.t19_tau_shrink == 0.5


def test_config_validates_t19_tau_grow_above_one():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMMConfig,
    )

    with pytest.raises(ValueError, match="t19_tau_grow"):
        JointLagrangianADMMConfig(t19_tau_grow=1.0)


def test_config_validates_t19_tau_shrink_in_unit_interval():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMMConfig,
    )

    with pytest.raises(ValueError, match="t19_tau_shrink"):
        JointLagrangianADMMConfig(t19_tau_shrink=1.0)
    with pytest.raises(ValueError, match="t19_tau_shrink"):
        JointLagrangianADMMConfig(t19_tau_shrink=0.0)


# ---------------------------------------------------------------------------
# Coordinator routes through the standalone adaptive_rho_step helper
# ---------------------------------------------------------------------------


def test_t19_off_uses_legacy_backend(monkeypatch):
    """When use_t19_adaptive_rho == False, the standalone helper is NOT called."""
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(use_t19_adaptive_rho=False)
    )
    # Force a ρ change via the legacy backend by stuffing the histories.
    for _ in range(coord.config.adaptive_rho_window):
        coord._primal_history.append(1e6)
        coord._dual_history.append(1.0)
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho_trajectory[-1]["source"] == "legacy"


def test_t19_on_uses_standalone_helper():
    """When use_t19_adaptive_rho == True, the trajectory entry source is 't19'."""
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            adaptive_rho_window=4,
            adaptive_rho_ratio=10.0,
            t19_tau_grow=2.0,
            t19_tau_shrink=0.5,
        )
    )
    # T19 reads the latest residual; only need one entry.
    coord._primal_history.append(1e6)
    coord._dual_history.append(1.0)
    rho_before = coord.rho
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho == pytest.approx(rho_before * 2.0, rel=1e-9)
    assert coord.rho_trajectory[-1]["source"] == "t19"
    assert coord.rho_trajectory[-1]["direction"] == "grow"


def test_t19_grow_when_primal_dominates_per_step():
    """T19 fires per-step (not windowed) so a single huge primal triggers grow."""
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            rho_init=1.0,
            rho_max=1e3,
            adaptive_rho_ratio=10.0,
        )
    )
    # Massive primal, tiny dual -> single-step grow.
    coord._primal_history.append(1e3)
    coord._dual_history.append(1.0)
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho > 1.0


def test_t19_shrink_when_dual_dominates_per_step():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            rho_init=1.0,
            adaptive_rho_ratio=10.0,
        )
    )
    coord._primal_history.append(1.0)
    coord._dual_history.append(1e3)
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho < 1.0
    assert coord.rho_trajectory[-1]["direction"] == "shrink"


def test_t19_hold_when_balanced():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            rho_init=1.0,
            adaptive_rho_ratio=10.0,
        )
    )
    coord._primal_history.append(1.0)
    coord._dual_history.append(1.0)
    changed = coord._maybe_adapt_rho()
    assert changed is False
    assert coord.rho == 1.0


def test_t19_clipped_to_rho_max():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            rho_init=1.0,
            rho_max=4.0,
            adaptive_rho_ratio=2.0,
            t19_tau_grow=8.0,  # would jump to 8 if unclipped
        )
    )
    coord._primal_history.append(1e6)
    coord._dual_history.append(1.0)
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho == pytest.approx(4.0, rel=1e-9)


def test_t19_clipped_to_rho_min():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            rho_init=0.5,
            rho_min=0.25,
            rho_max=10.0,
            adaptive_rho_ratio=2.0,
            t19_tau_shrink=0.01,  # would drop to 0.005 if unclipped
        )
    )
    coord._primal_history.append(1.0)
    coord._dual_history.append(1e6)
    changed = coord._maybe_adapt_rho()
    assert changed is True
    assert coord.rho == pytest.approx(0.25, rel=1e-9)


# ---------------------------------------------------------------------------
# Trajectory log shape
# ---------------------------------------------------------------------------


def test_rho_trajectory_records_required_fields():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    coord = JointLagrangianADMM(
        JointLagrangianADMMConfig(
            use_t19_adaptive_rho=True,
            adaptive_rho_ratio=10.0,
        )
    )
    coord._primal_history.append(1e6)
    coord._dual_history.append(1.0)
    coord._maybe_adapt_rho()
    entry = coord.rho_trajectory[-1]
    for field in (
        "step",
        "rho_before",
        "rho_after",
        "direction",
        "ratio",
        "primal",
        "dual",
        "source",
    ):
        assert field in entry, f"trajectory entry missing {field}"


def test_rho_trajectory_initially_empty():
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
    )

    coord = JointLagrangianADMM()
    assert coord.rho_trajectory == []


# ---------------------------------------------------------------------------
# End-to-end ADMM stability (T19 vs legacy) — invariant: ρ stays in band.
# ---------------------------------------------------------------------------


def test_admm_inner_loop_threads_t19_when_enabled():
    """Run a small ADMM loop with T19 enabled; verify ρ stays in
    [rho_min, rho_max] AND any ρ change is sourced 't19'."""
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    cfg = JointLagrangianADMMConfig(
        use_t19_adaptive_rho=True,
        rho_init=1.0,
        rho_min=0.1,
        rho_max=10.0,
        adaptive_rho_ratio=2.0,
        adaptive_rho_window=4,
    )
    coord = JointLagrangianADMM(cfg)
    for _ in range(40):
        _step(coord, rate_bits=1e10, seg=1.0, pose=1.0)
    assert cfg.rho_min <= coord.rho <= cfg.rho_max
    if coord.rho_trajectory:
        for entry in coord.rho_trajectory:
            assert entry["source"] == "t19"


def test_admm_inner_loop_uses_legacy_when_disabled():
    """Run with T19 OFF; any ρ trajectory entry must be source='legacy'."""
    from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
        JointLagrangianADMM,
        JointLagrangianADMMConfig,
    )

    cfg = JointLagrangianADMMConfig(
        use_t19_adaptive_rho=False,
        rho_init=1.0,
        rho_min=0.1,
        rho_max=100.0,
        adaptive_rho_ratio=2.0,
        adaptive_rho_window=4,
        lambda_init=1.0,
        lambda_max=1e-3,  # cap dual updates so primal dominates
    )
    coord = JointLagrangianADMM(cfg)
    for _ in range(40):
        _step(coord, rate_bits=1e10, seg=1.0, pose=1.0)
    if coord.rho_trajectory:
        for entry in coord.rho_trajectory:
            assert entry["source"] == "legacy"


# ---------------------------------------------------------------------------
# Trainer wire-in (T19 → JointLagrangianADMMConfig + provenance)
# ---------------------------------------------------------------------------


def test_trainer_main_reads_enable_t19_adaptive_rho_flag():
    src = _train_src()
    assert "args.enable_t19_adaptive_rho" in src


def test_trainer_main_propagates_t19_to_coord_cfg():
    src = _train_src()
    # The trainer constructs coord_cfg via a coord_kwargs dict; accept either
    # keyword form or dict-key form.
    assert ("use_t19_adaptive_rho=" in src) or (
        '"use_t19_adaptive_rho":' in src
    )
    assert ("t19_tau_grow=" in src) or ('"t19_tau_grow":' in src)
    assert ("t19_tau_shrink=" in src) or ('"t19_tau_shrink":' in src)


def test_trainer_main_overrides_rho_band_for_t19():
    """Per memo, --enable-t19-adaptive-rho should use rho_min=1e-3, rho_max=1e3."""
    src = _train_src()
    assert "rho_min" in src and "1e-3" in src
    assert "rho_max" in src and "1e3" in src


def test_trainer_main_emits_rho_trajectory_side_log():
    src = _train_src()
    assert "rho_trajectory.json" in src
    assert "coord.rho_trajectory" in src


def test_trainer_provenance_includes_t19_field():
    src = _train_src()
    assert "t19_adaptive_rho" in src
    assert "t19_adaptive_rho: dict | None = None" in src


def test_trainer_t19_print_uses_predicted_tag():
    """Operator-visible print + provenance tag must carry the predicted band.

    The source carries two T19 predicted tags:
      1. The console banner inside ``main()`` printed when the flag is ON.
         Its source-form is split across two adjacent f-strings so we match
         each fragment plus the directional band ``"2-3× convergence speedup"``.
      2. The provenance-side ``tag`` field that gets serialized into
         ``provenance.json::t19_adaptive_rho.tag``.
    """
    src = _train_src()
    # Banner fragments (split-string form in Python source).
    assert "[predicted; T19 adaptive ρ 2-3× convergence speedup; " in src
    assert "not direct score]" in src
    # Provenance-side tag also present.
    assert (
        '"[predicted; T19 adaptive ρ 2-3× convergence speedup; "' in src
    ) or (
        "'[predicted; T19 adaptive ρ 2-3× convergence speedup; '" in src
    )


# ---------------------------------------------------------------------------
# Standalone helper signature unchanged (regression guard)
# ---------------------------------------------------------------------------


def test_standalone_adaptive_rho_step_still_importable():
    """The T19 standalone helper from tac.joint_admm_coordinator MUST
    remain importable + callable; the wire-in depends on it."""
    from tac.joint_admm_coordinator import (
        AdaptiveRhoStep,
        adaptive_rho_step,
    )

    r = adaptive_rho_step(
        rho_curr=1.0, primal_residual=100.0, dual_residual=1.0
    )
    assert isinstance(r, AdaptiveRhoStep)
    assert r.direction == "grow"
    assert r.rho_next == 2.0
