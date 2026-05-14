# SPDX-License-Identifier: MIT
"""Unit tests for ``tac.film_time_varying`` (T15 scaffold)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tac.film_time_varying import (  # noqa: E402
    TimeVaryingFiLM,
    TimeVaryingFiLMConfig,
    TimeVaryingFiLMError,
    compute_per_pair_gamma_beta,
    time_varying_film_state_bytes,
)


# ---------------------------------------------------------------------------
# Test 1: Quantizr-canonical config matches eureka memo
# ---------------------------------------------------------------------------


def test_quantizr_canonical_config_matches_council():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t1")
    assert config.pose_dim == 6
    assert config.feature_channels == 64
    assert config.hidden_dim == 32
    assert config.activation == "relu"
    assert config.quantization == "fp4"


# ---------------------------------------------------------------------------
# Test 2: Config validation rejects invalid fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,err_token",
    [
        ({"pose_dim": 0}, "pose_dim"),
        ({"pose_dim": -1}, "pose_dim"),
        ({"feature_channels": 0}, "feature_channels"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"activation": "tanh"}, "activation"),
        ({"activation": ""}, "activation"),
        ({"quantization": "fp64"}, "quantization"),
        ({"label": ""}, "label"),
        ({"label": "  "}, "label"),
    ],
)
def test_config_validation_rejects(kwargs, err_token):
    base = dict(
        pose_dim=6,
        feature_channels=64,
        hidden_dim=32,
        activation="relu",
        quantization="fp4",
        label="t2",
    )
    base.update(kwargs)
    with pytest.raises(TimeVaryingFiLMError) as exc:
        TimeVaryingFiLMConfig(**base)
    assert err_token in str(exc.value)


# ---------------------------------------------------------------------------
# Test 3: Modulator forward shape
# ---------------------------------------------------------------------------


def test_modulator_forward_shape():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t3")
    modulator = TimeVaryingFiLM(config)
    pose_delta = torch.randn(8, 6)
    gamma, beta = modulator(pose_delta)
    assert gamma.shape == (8, 64)
    assert beta.shape == (8, 64)


# ---------------------------------------------------------------------------
# Test 4: Initial modulator is approximately identity (γ≈1, β≈0)
# ---------------------------------------------------------------------------


def test_modulator_initial_state_is_identity():
    """Per Quantizr convention: initial γ≈1, β≈0 so static FiLM is the baseline."""
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t4")
    modulator = TimeVaryingFiLM(config)
    pose_delta = torch.zeros(4, 6)
    gamma, beta = modulator(pose_delta)
    # With weight=0 init + bias initialized to (1s, 0s), γ should be 1
    # and β should be 0 at init regardless of input.
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: apply_film acts as identity at init
# ---------------------------------------------------------------------------


def test_apply_film_initial_is_identity():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t5")
    modulator = TimeVaryingFiLM(config)
    features = torch.randn(2, 64, 8, 8)
    pose_delta = torch.randn(2, 6)
    out = modulator.apply_film(features, pose_delta)
    # At init, γ=1, β=0 → output == input.
    assert torch.allclose(out, features, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: apply_film respects shape contracts
# ---------------------------------------------------------------------------


def test_apply_film_shape_validation():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t6", feature_channels=16)
    modulator = TimeVaryingFiLM(config)
    # Wrong feature channel count.
    with pytest.raises(TimeVaryingFiLMError) as exc:
        modulator.apply_film(torch.randn(2, 32, 4, 4), torch.randn(2, 6))
    assert "channel" in str(exc.value).lower()
    # Mismatched batch dims.
    with pytest.raises(TimeVaryingFiLMError) as exc:
        modulator.apply_film(torch.randn(2, 16, 4, 4), torch.randn(3, 6))
    assert "batch" in str(exc.value).lower()
    # Wrong feature rank.
    with pytest.raises(TimeVaryingFiLMError) as exc:
        modulator.apply_film(torch.randn(2, 16, 4), torch.randn(2, 6))
    assert "4D" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 7: Modulator forward rejects bad pose_delta shape
# ---------------------------------------------------------------------------


def test_modulator_forward_rejects_wrong_pose_dim():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t7")
    modulator = TimeVaryingFiLM(config)
    with pytest.raises(TimeVaryingFiLMError) as exc:
        modulator(torch.randn(2, 5))  # pose_dim mismatch
    assert "pose_delta" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 8: Byte cost matches closed form
# ---------------------------------------------------------------------------


def test_state_bytes_closed_form():
    """Param count: pose_dim*H + H + H*2C + 2C; bytes = params * bpp."""
    config = TimeVaryingFiLMConfig(
        pose_dim=6,
        feature_channels=64,
        hidden_dim=32,
        activation="relu",
        quantization="fp4",
        label="t8",
    )
    bytes_ = time_varying_film_state_bytes(config)
    # Layer 1: 6*32 + 32 = 224
    # Layer 2: 32*128 + 128 = 4224
    # Total = 4448 params * 0.5 bytes/param = 2224 bytes
    assert bytes_ == math.ceil((6 * 32 + 32 + 32 * 128 + 128) * 0.5)
    assert bytes_ == 2224


# ---------------------------------------------------------------------------
# Test 9: State bytes scales correctly with quantization
# ---------------------------------------------------------------------------


def test_state_bytes_scales_with_quantization():
    """FP4 (0.5 bpp) ≤ FP8 (1) ≤ FP16 (2) ≤ FP32 (4)."""
    base = dict(
        pose_dim=6,
        feature_channels=64,
        hidden_dim=32,
        activation="relu",
        label="t9",
    )
    bytes_fp4 = time_varying_film_state_bytes(
        TimeVaryingFiLMConfig(**base, quantization="fp4")
    )
    bytes_fp8 = time_varying_film_state_bytes(
        TimeVaryingFiLMConfig(**base, quantization="fp8")
    )
    bytes_fp16 = time_varying_film_state_bytes(
        TimeVaryingFiLMConfig(**base, quantization="fp16")
    )
    bytes_fp32 = time_varying_film_state_bytes(
        TimeVaryingFiLMConfig(**base, quantization="fp32")
    )
    assert bytes_fp4 < bytes_fp8 < bytes_fp16 < bytes_fp32
    assert bytes_fp8 == 2 * bytes_fp4
    assert bytes_fp16 == 2 * bytes_fp8
    assert bytes_fp32 == 2 * bytes_fp16


# ---------------------------------------------------------------------------
# Test 10: compute_per_pair_gamma_beta is the public alias
# ---------------------------------------------------------------------------


def test_compute_per_pair_gamma_beta_alias():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t10")
    modulator = TimeVaryingFiLM(config)
    pose = torch.randn(3, 6)
    g1, b1 = modulator(pose)
    g2, b2 = compute_per_pair_gamma_beta(modulator, pose)
    assert torch.equal(g1, g2)
    assert torch.equal(b1, b2)


# ---------------------------------------------------------------------------
# Test 11: Different pose deltas produce different (γ, β) after training
# ---------------------------------------------------------------------------


def test_modulator_responds_to_pose_after_training_step():
    """After ONE training step on a non-trivial loss, different pose deltas
    must produce different modulators (the whole point of T15)."""
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="t11")
    modulator = TimeVaryingFiLM(config)
    optimizer = torch.optim.Adam(modulator.parameters(), lr=1e-2)
    # Two distinct pose deltas; train so γ for input1 != γ for input2.
    pose1 = torch.zeros(1, 6)
    pose2 = torch.ones(1, 6)
    target_gamma1 = torch.ones(1, 64)
    target_gamma2 = torch.full((1, 64), 1.5)
    for _ in range(50):
        g1, _ = modulator(pose1)
        g2, _ = modulator(pose2)
        loss = ((g1 - target_gamma1) ** 2 + (g2 - target_gamma2) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    g1_final, _ = modulator(pose1)
    g2_final, _ = modulator(pose2)
    # The means should now differ — proving the modulator IS time-varying.
    assert not torch.allclose(g1_final, g2_final, atol=1e-2)


# ---------------------------------------------------------------------------
# Test 12: Factory rejects non-config args
# ---------------------------------------------------------------------------


def test_factory_rejects_dict_input():
    with pytest.raises(TimeVaryingFiLMError) as exc:
        TimeVaryingFiLM({"pose_dim": 6})  # type: ignore[arg-type]
    assert "TimeVaryingFiLMConfig" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 13: Public API surface complete
# ---------------------------------------------------------------------------


def test_public_api_complete():
    from tac import film_time_varying as mod

    expected = {
        "TimeVaryingFiLMConfig",
        "TimeVaryingFiLM",
        "compute_per_pair_gamma_beta",
        "time_varying_film_state_bytes",
        "TimeVaryingFiLMError",
    }
    assert set(mod.__all__) == expected
    for name in expected:
        assert hasattr(mod, name)
