"""Unit tests for ``tac.balle_nonlinear_transform`` (T18 scaffold)."""
from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")

from tac.balle_nonlinear_transform import (  # noqa: E402
    NonlinearTransformBlock,
    NonlinearTransformConfig,
    NonlinearTransformError,
    forward_transform,
    inverse_transform,
    transform_state_bytes,
)


# ---------------------------------------------------------------------------
# Test 1: He-Zheng canonical config
# ---------------------------------------------------------------------------


def test_he_zheng_canonical_config():
    config = NonlinearTransformConfig.he_zheng_canonical(label="t1")
    assert config.latent_dim == 192
    assert config.expansion_factor == 4
    assert config.num_hidden_layers == 3
    assert config.activation == "gelu"
    assert config.quantization == "fp16"


# ---------------------------------------------------------------------------
# Test 2: Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,err_token",
    [
        ({"latent_dim": 0}, "latent_dim"),
        ({"latent_dim": -1}, "latent_dim"),
        ({"expansion_factor": 0}, "expansion_factor"),
        ({"num_hidden_layers": 0}, "num_hidden_layers"),
        ({"activation": "tanh"}, "activation"),
        ({"quantization": "fp64"}, "quantization"),
        ({"label": ""}, "label"),
    ],
)
def test_config_rejects_invalid(kwargs, err_token):
    base = dict(
        latent_dim=192,
        expansion_factor=4,
        num_hidden_layers=3,
        activation="gelu",
        quantization="fp16",
        label="t2",
    )
    base.update(kwargs)
    with pytest.raises(NonlinearTransformError) as exc:
        NonlinearTransformConfig(**base)
    assert err_token in str(exc.value)


# ---------------------------------------------------------------------------
# Test 3: Forward shape preservation
# ---------------------------------------------------------------------------


def test_forward_shape_preservation():
    config = NonlinearTransformConfig(
        latent_dim=16, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t3",
    )
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(4, 16)
    z_t = block(z_e)
    assert z_t.shape == z_e.shape


# ---------------------------------------------------------------------------
# Test 4: Inverse shape preservation
# ---------------------------------------------------------------------------


def test_invert_shape_preservation():
    config = NonlinearTransformConfig(
        latent_dim=16, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t4",
    )
    block = NonlinearTransformBlock(config)
    z_t = torch.randn(4, 16)
    z_e = block.invert(z_t)
    assert z_e.shape == z_t.shape


# ---------------------------------------------------------------------------
# Test 5: Initial behaviour is identity (skip + zero last layer)
# ---------------------------------------------------------------------------


def test_initial_behaviour_is_identity():
    """Per docstring: at init, forward(z_e) == z_e (vanilla Ballé equivalent)."""
    config = NonlinearTransformConfig(
        latent_dim=16, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t5",
    )
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(4, 16)
    z_t = block(z_e)
    assert torch.allclose(z_t, z_e, atol=1e-6)
    z_e_recovered = block.invert(z_t)
    assert torch.allclose(z_e_recovered, z_e, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Forward rejects wrong latent dim
# ---------------------------------------------------------------------------


def test_forward_rejects_wrong_latent_dim():
    config = NonlinearTransformConfig(
        latent_dim=32, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t6",
    )
    block = NonlinearTransformBlock(config)
    with pytest.raises(NonlinearTransformError) as exc:
        block(torch.randn(2, 16))
    assert "latent_dim" in str(exc.value) or "32" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 7: Forward + invert chain after training drift → still bijective-ish
# ---------------------------------------------------------------------------


def test_invert_after_training_remains_bijective_ish():
    """After bumping weights, invert(forward(z)) should still recover z
    after a few training-like iterations (the network *can* be trained
    to be approximately invertible)."""
    config = NonlinearTransformConfig(
        latent_dim=8, expansion_factor=2, num_hidden_layers=1,
        activation="gelu", quantization="fp16", label="t7",
    )
    block = NonlinearTransformBlock(config)
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-2)
    # Train invert to be the inverse of forward.
    for _ in range(100):
        z_e = torch.randn(8, 8)
        z_t = block(z_e)
        z_e_back = block.invert(z_t)
        loss = torch.nn.functional.mse_loss(z_e_back, z_e)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # After training, invert should approximately recover.
    test_z = torch.randn(4, 8)
    recovered = block.invert(block(test_z))
    error = (recovered - test_z).abs().mean().item()
    assert error < 0.5, f"invert(forward(z)) error {error} too large"


# ---------------------------------------------------------------------------
# Test 8: forward_transform / inverse_transform helpers
# ---------------------------------------------------------------------------


def test_helper_functions_match_methods():
    config = NonlinearTransformConfig(
        latent_dim=16, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t8",
    )
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(4, 16)
    z_t1 = block(z_e)
    z_t2 = forward_transform(block, z_e)
    assert torch.equal(z_t1, z_t2)
    z_e_back1 = block.invert(z_t1)
    z_e_back2 = inverse_transform(block, z_t2)
    assert torch.equal(z_e_back1, z_e_back2)


# ---------------------------------------------------------------------------
# Test 9: state_bytes closed form
# ---------------------------------------------------------------------------


def test_state_bytes_closed_form():
    """Verify against analytical formula for fixed configuration."""
    config = NonlinearTransformConfig(
        latent_dim=8, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t9",
    )
    bytes_ = transform_state_bytes(config)
    # Per-MLP params:
    # Linear(8→16): 8*16 + 16 = 144
    # Linear(16→16): 16*16 + 16 = 272 (1 hidden layer beyond input)
    # Linear(16→8): 16*8 + 8 = 136
    # Per-MLP = 144 + 272 + 136 = 552
    # Two MLPs = 1104 params * 2 bytes (fp16) = 2208
    assert bytes_ == 2 * (8 * 16 + 16 + 16 * 16 + 16 + 16 * 8 + 8) * 2
    assert bytes_ == 2208


# ---------------------------------------------------------------------------
# Test 10: state_bytes scales with quantization
# ---------------------------------------------------------------------------


def test_state_bytes_scales_with_quantization():
    base = dict(
        latent_dim=8, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", label="t10",
    )
    fp4 = transform_state_bytes(NonlinearTransformConfig(**base, quantization="fp4"))
    fp8 = transform_state_bytes(NonlinearTransformConfig(**base, quantization="fp8"))
    fp16 = transform_state_bytes(NonlinearTransformConfig(**base, quantization="fp16"))
    fp32 = transform_state_bytes(NonlinearTransformConfig(**base, quantization="fp32"))
    assert fp4 < fp8 < fp16 < fp32
    assert fp32 == 2 * fp16 == 4 * fp8 == 8 * fp4


# ---------------------------------------------------------------------------
# Test 11: Forward + inverse use SEPARATE parameter sets per He-Zheng 2024
# ---------------------------------------------------------------------------


def test_forward_inverse_have_separate_parameters():
    """He-Zheng 2024 §3.2: weight-tying degrades RD frontier; verify NOT tied."""
    config = NonlinearTransformConfig(
        latent_dim=8, expansion_factor=2, num_hidden_layers=2,
        activation="gelu", quantization="fp16", label="t11",
    )
    block = NonlinearTransformBlock(config)
    fwd_params = list(block.forward_mlp.parameters())
    inv_params = list(block.inverse_mlp.parameters())
    # Verify no overlap in parameter ids.
    fwd_ids = {id(p) for p in fwd_params}
    inv_ids = {id(p) for p in inv_params}
    assert fwd_ids.isdisjoint(inv_ids)
    # Verify both have parameters.
    assert len(fwd_params) > 0
    assert len(inv_params) > 0


# ---------------------------------------------------------------------------
# Test 12: Factory rejects non-config
# ---------------------------------------------------------------------------


def test_factory_rejects_dict():
    with pytest.raises(NonlinearTransformError):
        NonlinearTransformBlock({"latent_dim": 192})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 13: Public API surface complete
# ---------------------------------------------------------------------------


def test_public_api_complete():
    from tac import balle_nonlinear_transform as mod

    expected = {
        "NonlinearTransformConfig",
        "NonlinearTransformBlock",
        "forward_transform",
        "inverse_transform",
        "transform_state_bytes",
        "NonlinearTransformError",
    }
    assert set(mod.__all__) == expected
    for name in expected:
        assert hasattr(mod, name)
