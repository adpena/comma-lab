"""Trainer wire-in tests for PR #95 binary-forensics replication.

Covers ``experiments/train_score_gradient_pr101_finetune.py`` AND
``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``
wire-ins of:

  * ``--enable-eval-roundtrip-in-training`` (Finding A)
  * ``--enable-differentiable-yuv6``         (Finding B)
  * ``--yuv6-mode {monkey_patch_global, tac_differentiable_routing, auto}``

References:
  * Source dossier: ``.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md``
  * Module under test: ``src/tac/differentiable_eval_roundtrip.py``
  * Probe: ``tools/probe_yuv6_differentiability_disambiguator.py``
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "upstream"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _upstream_available() -> bool:
    try:
        import frame_utils  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_UPSTREAM = _upstream_available()


# --------------------------------------------------------------------------- #
# Argparse subset (CLAUDE.md "NEVER invent CLI flags")                          #
# --------------------------------------------------------------------------- #


def test_score_gradient_trainer_argparse_includes_pr95_flags():
    """The score_gradient trainer must expose all 3 PR #95 replication flags."""
    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    sys.argv = ["x", "--help"]
    try:
        mod.parse_args()
    except SystemExit:
        pass
    # Re-parse to inspect the namespace.
    args = mod.parse_args.__wrapped__() if hasattr(mod.parse_args, "__wrapped__") else None
    # Easier path: use a mock invocation and check the namespace keys.
    sys.argv = ["x", "--output", "/tmp/_unused", "--smoke"]
    ns = mod.parse_args()
    assert hasattr(ns, "enable_eval_roundtrip_in_training")
    assert hasattr(ns, "enable_differentiable_yuv6")
    assert hasattr(ns, "yuv6_mode")
    # Defaults match the verified-working PR #95 recipe.
    assert ns.enable_eval_roundtrip_in_training is True
    assert ns.enable_differentiable_yuv6 is True
    assert ns.yuv6_mode == "auto"


def test_balle_trainer_argparse_includes_pr95_flags(tmp_path):
    """The balle trainer must expose all 3 PR #95 replication flags."""
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    ns = mod.parse_args(["--output-dir", str(tmp_path)])
    assert hasattr(ns, "enable_eval_roundtrip_in_training")
    assert hasattr(ns, "enable_differentiable_yuv6")
    assert hasattr(ns, "yuv6_mode")
    assert ns.enable_eval_roundtrip_in_training is True
    assert ns.enable_differentiable_yuv6 is True
    assert ns.yuv6_mode == "auto"


def test_score_gradient_trainer_disable_flag_inverts_default():
    """--disable-* flag must produce False namespace value."""
    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    sys.argv = [
        "x",
        "--output", "/tmp/_unused",
        "--smoke",
        "--disable-eval-roundtrip-in-training",
        "--disable-differentiable-yuv6",
    ]
    ns = mod.parse_args()
    assert ns.enable_eval_roundtrip_in_training is False
    assert ns.enable_differentiable_yuv6 is False


def test_balle_trainer_disable_flag_inverts_default(tmp_path):
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    ns = mod.parse_args([
        "--output-dir", str(tmp_path),
        "--disable-eval-roundtrip-in-training",
        "--disable-differentiable-yuv6",
    ])
    assert ns.enable_eval_roundtrip_in_training is False
    assert ns.enable_differentiable_yuv6 is False


def test_score_gradient_trainer_yuv6_mode_choices():
    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    for mode in ("monkey_patch_global", "tac_differentiable_routing", "auto"):
        sys.argv = ["x", "--output", "/tmp/_unused", "--smoke", "--yuv6-mode", mode]
        ns = mod.parse_args()
        assert ns.yuv6_mode == mode


def test_balle_trainer_yuv6_mode_choices(tmp_path):
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    for mode in ("monkey_patch_global", "tac_differentiable_routing", "auto"):
        ns = mod.parse_args(["--output-dir", str(tmp_path), "--yuv6-mode", mode])
        assert ns.yuv6_mode == mode


def test_score_gradient_trainer_invented_flag_rejected():
    """Invented (typoed) flag must be rejected by argparse — guards the
    ``forbidden_dead_flag_wiring_pattern`` rule for THIS surface."""
    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    sys.argv = ["x", "--output", "/tmp/_unused", "--smoke", "--enable-yuv6-monkey-patch-global"]
    with pytest.raises(SystemExit):
        mod.parse_args()


def test_balle_trainer_invented_flag_rejected(tmp_path):
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    with pytest.raises(SystemExit):
        mod.parse_args(["--output-dir", str(tmp_path), "--enable-yuv6-monkey-patch-global"])


# --------------------------------------------------------------------------- #
# Probe-disambiguator integration                                              #
# --------------------------------------------------------------------------- #


def test_score_gradient_trainer_resolves_auto_to_concrete_mode():
    """``_resolve_yuv6_mode_with_probe('auto')`` returns a concrete mode."""
    from tac.differentiable_eval_roundtrip import Yuv6RoutingMode

    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    resolved = mod._resolve_yuv6_mode_with_probe("auto")
    assert isinstance(resolved, Yuv6RoutingMode)
    assert resolved is not Yuv6RoutingMode.AUTO


def test_score_gradient_trainer_passthrough_mode_unchanged():
    """When --yuv6-mode is concrete, the probe is bypassed."""
    from tac.differentiable_eval_roundtrip import Yuv6RoutingMode

    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    assert mod._resolve_yuv6_mode_with_probe("monkey_patch_global") is Yuv6RoutingMode.MONKEY_PATCH_GLOBAL
    assert mod._resolve_yuv6_mode_with_probe("tac_differentiable_routing") is Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING


def test_balle_trainer_resolves_auto_to_concrete_mode():
    from tac.differentiable_eval_roundtrip import Yuv6RoutingMode

    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    resolved = mod._resolve_yuv6_mode_with_probe_t1("auto")
    assert isinstance(resolved, Yuv6RoutingMode)
    assert resolved is not Yuv6RoutingMode.AUTO


# --------------------------------------------------------------------------- #
# YUV6 monkey-patch activation                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_score_gradient_trainer_activate_monkey_patches_upstream():
    """``_activate_yuv6_mode(MONKEY_PATCH_GLOBAL, enabled=True)`` mutates frame_utils."""
    from tac.differentiable_eval_roundtrip import (
        Yuv6RoutingMode,
        differentiable_rgb_to_yuv6,
        unpatch_upstream_yuv6,
    )
    import frame_utils

    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    original = frame_utils.rgb_to_yuv6
    token = mod._activate_yuv6_mode(
        Yuv6RoutingMode.MONKEY_PATCH_GLOBAL, enabled=True
    )
    try:
        assert token is not None
        assert frame_utils.rgb_to_yuv6 is differentiable_rgb_to_yuv6
    finally:
        unpatch_upstream_yuv6(token)
    assert frame_utils.rgb_to_yuv6 is original


def test_score_gradient_trainer_activate_disabled_returns_none():
    """When --disable-differentiable-yuv6 is set, no patching occurs."""
    from tac.differentiable_eval_roundtrip import Yuv6RoutingMode

    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    token = mod._activate_yuv6_mode(
        Yuv6RoutingMode.MONKEY_PATCH_GLOBAL, enabled=False
    )
    assert token is None


# --------------------------------------------------------------------------- #
# Pose-gradient through YUV6 — flag ON vs OFF                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_pose_gradient_through_yuv6_nonzero_when_flag_on():
    """With --enable-differentiable-yuv6, gradient flows through YUV6."""
    from tac.differentiable_eval_roundtrip import (
        Yuv6RoutingMode,
        differentiable_rgb_to_yuv6,
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )

    token = patch_upstream_yuv6_globally()
    try:
        import frame_utils
        rgb = (torch.rand((1, 3, 32, 32)) * 255.0).requires_grad_(True)
        out = frame_utils.rgb_to_yuv6(rgb)
        out.sum().backward()
        assert rgb.grad is not None
        assert rgb.grad.abs().sum().item() > 0.0
    finally:
        unpatch_upstream_yuv6(token)


@pytest.mark.skipif(not _HAS_UPSTREAM, reason="upstream frame_utils unavailable")
def test_pose_gradient_through_yuv6_zero_when_flag_off():
    """Without the patch, upstream's @torch.no_grad path produces zero grad.

    This is the BUG the patch fixes — Aaron's PR #95 quote: "pose plateaued
    at 142 across 2500+ epochs" without this fix.
    """
    import frame_utils

    rgb = (torch.rand((1, 3, 32, 32)) * 255.0).requires_grad_(True)
    out = frame_utils.rgb_to_yuv6(rgb)
    # Cannot backward — out.grad_fn is None due to @torch.no_grad.
    assert out.grad_fn is None


# --------------------------------------------------------------------------- #
# eval_roundtrip-in-training callability                                        #
# --------------------------------------------------------------------------- #


def test_apply_eval_roundtrip_during_training_imports_in_score_gradient_trainer():
    """The trainer module must be importable; the canonical primitive
    must be available for any callsite that wants to apply it."""
    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    # The trainer ALREADY uses tac.training.simulate_eval_roundtrip in
    # train_one_step (line ~582). The canonical PR #95-faithful primitive
    # is available via tac.differentiable_eval_roundtrip for any future
    # ablation / migration.
    from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
    assert callable(apply_eval_roundtrip_during_training)


def test_apply_eval_roundtrip_during_training_imports_in_balle_trainer():
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    assert hasattr(mod, "apply_eval_roundtrip_during_training")
    from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
    assert mod.apply_eval_roundtrip_during_training is apply_eval_roundtrip_during_training


# --------------------------------------------------------------------------- #
# Backward-compat: defaults match pre-fix behavior of pixel-L1 trainer          #
# --------------------------------------------------------------------------- #


def test_score_gradient_trainer_roundtrip_flag_reaches_train_step_signature():
    """The argparse flag must not be dead; it reaches train/train_one_step."""
    import inspect

    mod = importlib.import_module("experiments.train_score_gradient_pr101_finetune")
    assert "enable_eval_roundtrip_in_training" in inspect.signature(
        mod.train_one_step
    ).parameters
    assert "enable_eval_roundtrip_in_training" in inspect.signature(
        mod.train
    ).parameters


def test_balle_trainer_eval_roundtrip_default_uses_canonical(monkeypatch):
    """Default path calls the PR #95-faithful canonical roundtrip primitive."""
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")
    called = {"value": False}

    def fake_roundtrip(x, **_kwargs):
        called["value"] = True
        return torch.zeros_like(x)

    monkeypatch.setattr(mod, "apply_eval_roundtrip_during_training", fake_roundtrip)
    torch.manual_seed(0)
    decoded = torch.rand(1, 2, 3, 384, 512) * 255.0
    target = torch.zeros_like(decoded)
    out = mod.eval_roundtrip_pixel_l1(decoded, target, noise_std=0.5)
    assert called["value"] is True
    assert out.dim() == 0
    assert torch.isfinite(out).item()
    assert float(out.item()) == 0.0


def test_balle_trainer_disable_roundtrip_skips_canonical(monkeypatch):
    """Ablation flag is real: disabled path must not call the roundtrip."""
    mod = importlib.import_module("experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend")

    def forbidden_roundtrip(_x, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("roundtrip should be disabled")

    monkeypatch.setattr(mod, "apply_eval_roundtrip_during_training", forbidden_roundtrip)
    decoded = torch.ones(1, 2, 3, 8, 8)
    target = torch.zeros_like(decoded)
    out = mod.eval_roundtrip_pixel_l1(
        decoded,
        target,
        noise_std=0.5,
        enable_eval_roundtrip_in_training=False,
    )
    assert out.item() == 1.0
