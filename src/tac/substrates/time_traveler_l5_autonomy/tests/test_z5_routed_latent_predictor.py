# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler -> Z5 sister routing primitive.

Per the 2026-05-14 grand-council reconvening Decision 1 (UNANIMOUS β
11/11) sister-substrate-consistency clause: Time-Traveler is a sibling
of Z5 (``z5_predictive_coding_world_model``); the sister-routing
adapter :class:`Z5RoutedLatentPredictor` keeps the latent-prediction
primitive consistent across the two L5-autonomy substrates so the
cathedral-autopilot composition logic (Catalog #227
``adjust_predicted_delta_for_composition_alpha``) sees the same
predictor kernel under stacking.

The Time-Traveler v1 substrate (``TimeTravelerSubstrate``) does NOT use
this adapter — the canonical SIREN renderer + Markov-1 ego-pose
dynamics + log-polar foveation grid is preserved per CLAUDE.md
"Forbidden premature KILL". The adapter is provided as a NEUTRAL
routing primitive so a future training run can probe SIREN-rendered
baseline vs Z5-routed predicted-residual variant without duplicating
Z5 plumbing.

Cross-references:
- Sister adapter on the C1 side:
  :class:`tac.substrates.c1_world_model_foveation.architecture.Z5RoutedWorldModel`
- Council ledger:
  ``.omx/research/grand_council_c1_post_probe_v2_reconvene_20260514.md``
"""

from __future__ import annotations

import pytest
import torch

from tac.substrates.time_traveler_l5_autonomy import (
    TimeTravelerSubstrate,
    Z5RoutedLatentPredictor,
)
from tac.substrates.z5_predictive_coding_world_model.architecture import (
    HierarchicalPredictor,
)


# =====================================================================
# 1. Z5RoutedLatentPredictor adapter exists + is wired correctly
# =====================================================================


def test_adapter_constructs_with_default_args() -> None:
    """Default args should construct a working predictor."""
    routed = Z5RoutedLatentPredictor(latent_dim=8)
    assert routed.latent_dim == 8
    assert routed.hidden_dim == 64
    assert routed.ego_motion_dim == 8
    assert routed.identity_predictor_mode is False


def test_adapter_wraps_hierarchical_predictor() -> None:
    """The adapter must wrap Z5's canonical HierarchicalPredictor (NOT a
    re-implementation in this module — sister-substrate consistency
    requires shared kernel)."""
    routed = Z5RoutedLatentPredictor(latent_dim=8)
    assert isinstance(routed.predictor, HierarchicalPredictor)


def test_adapter_predict_next_shape() -> None:
    """The predict_next contract returns ``(B, latent_dim)``."""
    routed = Z5RoutedLatentPredictor(latent_dim=8, ego_motion_dim=8)
    z_prev = torch.randn(4, 8)
    ego = torch.randn(4, 8)
    z_next = routed.predict_next(z_prev, ego)
    assert z_next.shape == (4, 8)


def test_adapter_rejects_wrong_latent_dim() -> None:
    routed = Z5RoutedLatentPredictor(latent_dim=8, ego_motion_dim=8)
    bad_z = torch.randn(2, 99)
    ego = torch.randn(2, 8)
    with pytest.raises(ValueError, match="latent_dim"):
        routed.predict_next(bad_z, ego)


def test_adapter_rejects_wrong_ego_motion_dim() -> None:
    routed = Z5RoutedLatentPredictor(latent_dim=8, ego_motion_dim=8)
    z_prev = torch.randn(2, 8)
    bad_ego = torch.randn(2, 99)
    with pytest.raises(ValueError, match="ego_motion_dim"):
        routed.predict_next(z_prev, bad_ego)


# =====================================================================
# 2. Mode arbitration matches Z5 directly
# =====================================================================


def test_predictive_world_model_mode_default() -> None:
    routed = Z5RoutedLatentPredictor(latent_dim=8)
    assert routed.identity_predictor_mode is False
    assert routed.predictor.identity_predictor is False


def test_identity_predictor_mode_opt_in() -> None:
    routed = Z5RoutedLatentPredictor(latent_dim=8, identity_predictor=True)
    assert routed.identity_predictor_mode is True
    assert routed.predictor.identity_predictor is True


def test_identity_predictor_returns_z_prev_unchanged() -> None:
    """In identity_predictor mode, predict_next should return z_prev."""
    routed = Z5RoutedLatentPredictor(latent_dim=8, identity_predictor=True)
    z_prev = torch.randn(2, 8)
    ego = torch.randn(2, 8)  # ignored in identity mode
    z_next = routed.predict_next(z_prev, ego)
    assert torch.allclose(z_next, z_prev, atol=1e-6)


def test_predictive_world_model_diverges_from_identity() -> None:
    """The predictive-world-model mode should produce a non-trivial
    transform of z_prev."""
    z_prev = torch.randn(2, 8)
    ego = torch.randn(2, 8)
    routed = Z5RoutedLatentPredictor(
        latent_dim=8, identity_predictor=False
    )
    z_next = routed.predict_next(z_prev, ego)
    # The predictor net applies a non-trivial transform; output should
    # not exactly equal z_prev (with high probability).
    assert not torch.allclose(z_next, z_prev, atol=1e-6)


# =====================================================================
# 3. Time-Traveler v1 substrate UNCHANGED (sister-routing is OPT-IN only)
# =====================================================================


def test_time_traveler_v1_substrate_does_not_use_routed_adapter() -> None:
    """Per CLAUDE.md "Forbidden premature KILL": Time-Traveler v1
    canonical SIREN renderer + Markov-1 ego-pose dynamics + log-polar
    foveation grid composition is preserved. The Z5-routed adapter is
    a NEUTRAL primitive available for future use, not a default
    substitution."""
    from tac.substrates.time_traveler_l5_autonomy import TimeTravelerConfig

    cfg = TimeTravelerConfig(num_pairs=4)
    substrate = TimeTravelerSubstrate(cfg)
    # No attribute named 'z5_routed_predictor' on the v1 substrate.
    assert not hasattr(substrate, "z5_routed_predictor")
    # The renderer is the canonical PredictiveRenderer (SIREN MLP), not
    # a Z5-routed wrapper.
    from tac.substrates.time_traveler_l5_autonomy.architecture import (
        PredictiveRenderer,
    )
    assert isinstance(substrate.renderer, PredictiveRenderer)


def test_routed_adapter_is_importable_from_package_init() -> None:
    """The sister-routing adapter must be exported at the package
    level so future trainers can import it without reaching into the
    submodule."""
    import tac.substrates.time_traveler_l5_autonomy as ttl5

    assert hasattr(ttl5, "Z5RoutedLatentPredictor")
    assert ttl5.Z5RoutedLatentPredictor is Z5RoutedLatentPredictor
