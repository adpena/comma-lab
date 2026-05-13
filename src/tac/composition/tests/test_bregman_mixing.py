"""Tests for tac.composition.bregman_mixing — Bregman-divergence mixing."""

from __future__ import annotations

import pytest
import torch

from tac.composition.bregman_mixing import (
    BREGMAN_MAGIC,
    BREGMAN_SCHEMA_VERSION,
    BregmanError,
    BregmanGenerator,
    BregmanMixer,
    BregmanMixerSpec,
    bregman_centroid,
    bregman_divergence,
    estimate_param_bytes,
)
from tac.composition.frontier_primitives import bregman_barycenter

# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_default_is_squared_euclidean() -> None:
    s = BregmanMixerSpec()
    assert s.generator == BregmanGenerator.SQUARED_EUCLIDEAN
    assert s.weights is None
    assert s.eps > 0


def test_spec_rejects_negative_eps() -> None:
    with pytest.raises(BregmanError, match="eps must be positive"):
        BregmanMixerSpec(eps=-1.0)


def test_spec_rejects_zero_eps() -> None:
    with pytest.raises(BregmanError, match="eps must be positive"):
        BregmanMixerSpec(eps=0.0)


def test_spec_rejects_nonfinite_public_floats() -> None:
    with pytest.raises(BregmanError, match="eps must be positive"):
        BregmanMixerSpec(eps=float("nan"))
    with pytest.raises(BregmanError, match="weights must be finite"):
        BregmanMixerSpec(weights=(0.5, float("inf")))


def test_spec_rejects_negative_weights() -> None:
    with pytest.raises(BregmanError, match="weights must be non-negative"):
        BregmanMixerSpec(weights=(0.5, -0.5))


def test_spec_rejects_empty_weights() -> None:
    with pytest.raises(BregmanError, match="weights must be non-empty"):
        BregmanMixerSpec(weights=())


def test_spec_rejects_zero_sum_weights() -> None:
    with pytest.raises(BregmanError, match="weights must sum to a positive value"):
        BregmanMixerSpec(weights=(0.0, 0.0))


def test_spec_mahalanobis_requires_metric() -> None:
    with pytest.raises(BregmanError, match="mahalanobis_metric is required"):
        BregmanMixerSpec(generator=BregmanGenerator.MAHALANOBIS)


def test_spec_mahalanobis_metric_must_be_positive() -> None:
    with pytest.raises(BregmanError, match="strictly positive"):
        BregmanMixerSpec(
            generator=BregmanGenerator.MAHALANOBIS,
            mahalanobis_metric=(1.0, 0.0, 2.0),
        )


def test_spec_mahalanobis_metric_must_be_nonempty() -> None:
    with pytest.raises(BregmanError, match="mahalanobis_metric must be non-empty"):
        BregmanMixerSpec(
            generator=BregmanGenerator.MAHALANOBIS,
            mahalanobis_metric=(),
        )


def test_spec_mahalanobis_metric_must_be_finite() -> None:
    with pytest.raises(BregmanError, match="finite"):
        BregmanMixerSpec(
            generator=BregmanGenerator.MAHALANOBIS,
            mahalanobis_metric=(1.0, float("nan")),
        )


# ---------------------------------------------------------------------------
# Divergence kernel
# ---------------------------------------------------------------------------


def test_squared_euclidean_divergence_matches_half_mse() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 0.0, 0.0])
    d = bregman_divergence(x, y, BregmanGenerator.SQUARED_EUCLIDEAN)
    assert torch.allclose(d, torch.tensor(0.5 * (1.0 + 4.0 + 9.0)))


def test_squared_euclidean_divergence_zero_at_x_eq_y() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    d = bregman_divergence(x, x.clone(), BregmanGenerator.SQUARED_EUCLIDEAN)
    assert torch.allclose(d, torch.tensor(0.0))


def test_kl_divergence_zero_at_x_eq_y() -> None:
    x = torch.tensor([0.5, 0.5])
    d = bregman_divergence(x, x.clone(), BregmanGenerator.KL)
    assert torch.allclose(d, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_nonnegative() -> None:
    x = torch.tensor([0.7, 0.3])
    y = torch.tensor([0.3, 0.7])
    d = bregman_divergence(x, y, BregmanGenerator.KL)
    assert float(d) > 0


def test_itakura_saito_zero_at_x_eq_y() -> None:
    x = torch.tensor([0.5, 1.0, 2.0])
    d = bregman_divergence(x, x.clone(), BregmanGenerator.ITAKURA_SAITO)
    assert torch.allclose(d, torch.tensor(0.0), atol=1e-5)


def test_itakura_saito_strictly_positive_for_unequal() -> None:
    x = torch.tensor([0.5, 1.0, 2.0])
    y = torch.tensor([1.0, 1.0, 1.0])
    d = bregman_divergence(x, y, BregmanGenerator.ITAKURA_SAITO)
    assert float(d) > 0


def test_mahalanobis_uses_diagonal_metric() -> None:
    x = torch.tensor([1.0, 1.0])
    y = torch.tensor([0.0, 0.0])
    metric = torch.tensor([1.0, 4.0])
    d = bregman_divergence(
        x,
        y,
        BregmanGenerator.MAHALANOBIS,
        mahalanobis_metric=metric,
    )
    # 0.5 * (1*1 + 4*1) = 2.5
    assert torch.allclose(d, torch.tensor(2.5))


def test_divergence_shape_mismatch_raises() -> None:
    x = torch.zeros(3)
    y = torch.zeros(4)
    with pytest.raises(BregmanError, match="shape mismatch"):
        bregman_divergence(x, y, BregmanGenerator.SQUARED_EUCLIDEAN)


# ---------------------------------------------------------------------------
# Centroid kernel
# ---------------------------------------------------------------------------


def test_centroid_squared_euclidean_is_arithmetic_mean() -> None:
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([2.0, 4.0])
    c = bregman_centroid([p1, p2], BregmanGenerator.SQUARED_EUCLIDEAN)
    assert torch.allclose(c, torch.tensor([1.0, 2.0]))


def test_centroid_kl_is_geometric_mean_on_simplex() -> None:
    p1 = torch.tensor([0.25, 0.75])
    p2 = torch.tensor([0.75, 0.25])
    c = bregman_centroid([p1, p2], BregmanGenerator.KL)
    # Geometric mean normalised: sqrt(0.25*0.75)=sqrt(0.1875); same for both.
    # So centroid should be uniform on simplex.
    assert torch.allclose(c, torch.tensor([0.5, 0.5]), atol=1e-5)


def test_centroid_itakura_saito_is_harmonic_mean() -> None:
    p1 = torch.tensor([2.0])
    p2 = torch.tensor([6.0])
    c = bregman_centroid([p1, p2], BregmanGenerator.ITAKURA_SAITO)
    # Harmonic mean: 2 / (1/2 + 1/6) = 2 / (2/3) = 3
    assert torch.allclose(c, torch.tensor([3.0]), atol=1e-5)


def test_centroid_weighted_squared_euclidean() -> None:
    p1 = torch.tensor([0.0])
    p2 = torch.tensor([1.0])
    w = torch.tensor([0.25, 0.75])
    c = bregman_centroid(
        [p1, p2], BregmanGenerator.SQUARED_EUCLIDEAN, weights=w
    )
    assert torch.allclose(c, torch.tensor([0.75]))


def test_centroid_conforms_to_frontier_bregman_barycenter() -> None:
    xs = (
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.75, 0.25]),
    )
    weights = torch.tensor([0.2, 0.8])
    assert torch.allclose(
        bregman_centroid(xs, BregmanGenerator.SQUARED_EUCLIDEAN, weights=weights),
        bregman_barycenter(xs, (0.2, 0.8), divergence="squared_euclidean"),
    )
    assert torch.allclose(
        bregman_centroid(xs, BregmanGenerator.KL, weights=weights),
        bregman_barycenter(xs, (0.2, 0.8), divergence="kl_forward"),
    )


def test_centroid_empty_raises() -> None:
    with pytest.raises(BregmanError, match="non-empty"):
        bregman_centroid([], BregmanGenerator.SQUARED_EUCLIDEAN)


def test_centroid_shape_mismatch_raises() -> None:
    p1 = torch.zeros(3)
    p2 = torch.zeros(4)
    with pytest.raises(BregmanError, match="shape"):
        bregman_centroid([p1, p2], BregmanGenerator.SQUARED_EUCLIDEAN)


def test_centroid_weights_shape_mismatch_raises() -> None:
    p1 = torch.zeros(3)
    p2 = torch.zeros(3)
    w = torch.tensor([0.5, 0.5, 0.5])
    with pytest.raises(BregmanError):
        bregman_centroid(
            [p1, p2], BregmanGenerator.SQUARED_EUCLIDEAN, weights=w
        )


# ---------------------------------------------------------------------------
# BregmanMixer class
# ---------------------------------------------------------------------------


def test_mixer_uniform_weights_match_unweighted_centroid() -> None:
    mixer = BregmanMixer(BregmanMixerSpec())
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([2.0, 2.0])
    c = mixer.mix([p1, p2])
    assert torch.allclose(c, torch.tensor([1.0, 1.0]))


def test_mixer_respects_explicit_weights() -> None:
    mixer = BregmanMixer(BregmanMixerSpec(weights=(0.25, 0.75)))
    p1 = torch.tensor([0.0])
    p2 = torch.tensor([1.0])
    c = mixer.mix([p1, p2])
    assert torch.allclose(c, torch.tensor([0.75]))


def test_mixer_weight_length_must_match_points() -> None:
    mixer = BregmanMixer(BregmanMixerSpec(weights=(0.5, 0.5)))
    with pytest.raises(BregmanError, match="weights length"):
        mixer.mix([torch.zeros(3)])


def test_mixer_divergence_round_trip() -> None:
    mixer = BregmanMixer(BregmanMixerSpec())
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([0.0, 0.0])
    d = mixer.divergence(x, y)
    assert float(d) > 0


def test_mixer_mahalanobis_divergence() -> None:
    spec = BregmanMixerSpec(
        generator=BregmanGenerator.MAHALANOBIS,
        mahalanobis_metric=(1.0, 4.0),
    )
    mixer = BregmanMixer(spec)
    x = torch.tensor([1.0, 1.0])
    y = torch.tensor([0.0, 0.0])
    d = mixer.divergence(x, y)
    assert torch.allclose(d, torch.tensor(2.5))


def test_mixer_serialize_deserialize_roundtrip_squared_euclidean() -> None:
    mixer = BregmanMixer(BregmanMixerSpec(weights=(0.5, 0.5)))
    blob = mixer.serialize_state()
    assert blob[:4] == BREGMAN_MAGIC
    restored = BregmanMixer.deserialize_state(blob)
    assert restored.spec.generator == BregmanGenerator.SQUARED_EUCLIDEAN
    assert restored.spec.weights == (0.5, 0.5)


def test_mixer_serialize_deserialize_roundtrip_mahalanobis() -> None:
    spec = BregmanMixerSpec(
        generator=BregmanGenerator.MAHALANOBIS,
        mahalanobis_metric=(1.0, 2.0, 3.0),
        weights=(0.3, 0.7),
    )
    mixer = BregmanMixer(spec)
    blob = mixer.serialize_state()
    restored = BregmanMixer.deserialize_state(blob)
    assert restored.spec.generator == BregmanGenerator.MAHALANOBIS
    assert restored.spec.mahalanobis_metric == (1.0, 2.0, 3.0)


def test_deserialize_rejects_bad_magic() -> None:
    with pytest.raises(BregmanError, match="bad magic"):
        BregmanMixer.deserialize_state(b"XXXX" + b"\x00" * 20)


def test_deserialize_rejects_unknown_schema_version() -> None:
    bad = BREGMAN_MAGIC + (BREGMAN_SCHEMA_VERSION + 99).to_bytes(2, "little") + b"\x00" * 20
    with pytest.raises(BregmanError, match="unsupported schema"):
        BregmanMixer.deserialize_state(bad)


def test_estimate_param_bytes_grows_with_weights() -> None:
    s1 = BregmanMixerSpec()
    s2 = BregmanMixerSpec(weights=(0.5, 0.5))
    assert estimate_param_bytes(s2) > estimate_param_bytes(s1)


def test_grad_flows_through_squared_euclidean() -> None:
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([0.0, 0.0])
    d = bregman_divergence(x, y, BregmanGenerator.SQUARED_EUCLIDEAN)
    d.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([1.0, 2.0]))


def test_grad_flows_through_kl() -> None:
    x = torch.tensor([0.6, 0.4], requires_grad=True)
    y = torch.tensor([0.5, 0.5])
    d = bregman_divergence(x, y, BregmanGenerator.KL)
    d.backward()
    assert x.grad is not None
    # Grad of KL wrt x: log(x/y).
    assert torch.all(torch.isfinite(x.grad))


def test_mix_composes_with_torch_dtype_preservation() -> None:
    mixer = BregmanMixer(BregmanMixerSpec())
    p1 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    p2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    c = mixer.mix([p1, p2])
    assert c.dtype == torch.float64
