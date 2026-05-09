"""Unit tests for ``tac.shared_vq_codebook`` (T17 scaffold)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tac.shared_vq_codebook import (  # noqa: E402
    SharedCodebook,
    SharedCodebookConfig,
    SharedCodebookError,
    compute_codebook_perplexity,
    quantize_via_shared_codebook,
    shared_codebook_state_bytes,
)


# ---------------------------------------------------------------------------
# Test 1: van den Oord canonical config
# ---------------------------------------------------------------------------


def test_vandenoord_canonical_config():
    config = SharedCodebookConfig.vandenoord_canonical(label="t1")
    assert config.num_entries == 256
    assert config.entry_dim == 64
    assert config.ema_decay == 0.99
    assert config.quantization == "fp16"


# ---------------------------------------------------------------------------
# Test 2: Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,err_token",
    [
        ({"num_entries": 1}, "num_entries"),
        ({"num_entries": 0}, "num_entries"),
        ({"entry_dim": 0}, "entry_dim"),
        ({"ema_decay": 0.89}, "ema_decay"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"ema_decay": 1.001}, "ema_decay"),
        ({"ema_decay": float("nan")}, "finite"),
        ({"epsilon_laplace": 0.0}, "epsilon_laplace"),
        ({"epsilon_laplace": -1e-6}, "epsilon_laplace"),
        ({"quantization": "fp64"}, "quantization"),
        ({"label": ""}, "label"),
    ],
)
def test_config_rejects_invalid(kwargs, err_token):
    base = dict(
        num_entries=256,
        entry_dim=64,
        ema_decay=0.99,
        epsilon_laplace=1e-5,
        quantization="fp16",
        label="t2",
    )
    base.update(kwargs)
    with pytest.raises(SharedCodebookError) as exc:
        SharedCodebookConfig(**base)
    assert err_token in str(exc.value)


# ---------------------------------------------------------------------------
# Test 3: Codebook forward shape contracts
# ---------------------------------------------------------------------------


def test_codebook_forward_shapes():
    config = SharedCodebookConfig(
        num_entries=16, entry_dim=4, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t3",
    )
    cb = SharedCodebook(config)
    z_e = torch.randn(2, 5, 4)  # (B, N, D)
    z_q, indices, commit = cb(z_e)
    assert z_q.shape == z_e.shape
    assert indices.shape == (2, 5)
    assert commit.dim() == 0  # scalar
    assert (indices >= 0).all() and (indices < 16).all()


# ---------------------------------------------------------------------------
# Test 4: Quantization is closest-entry
# ---------------------------------------------------------------------------


def test_quantization_picks_nearest_entry():
    """A z_e exactly at a codebook entry must quantize to that entry."""
    config = SharedCodebookConfig(
        num_entries=4, entry_dim=2, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t4",
    )
    cb = SharedCodebook(config)
    # Force-set a known codebook (the buffer is mutable).
    with torch.no_grad():
        cb.codebook.copy_(torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]))
    # z_e exactly at entry 0 should quantize to entry 0.
    z_e = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    z_q, indices, _ = cb(z_e)
    assert indices.item() == 0
    assert torch.allclose(z_q, torch.tensor([[[1.0, 0.0]]]))


# ---------------------------------------------------------------------------
# Test 5: Straight-through gradient flows
# ---------------------------------------------------------------------------


def test_straight_through_gradient():
    """Gradient on z_q must propagate to z_e (encoder side)."""
    config = SharedCodebookConfig(
        num_entries=8, entry_dim=4, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t5",
    )
    cb = SharedCodebook(config)
    z_e = torch.randn(2, 4, requires_grad=True)
    z_q, _, _ = cb(z_e)
    loss = z_q.sum()
    loss.backward()
    assert z_e.grad is not None
    assert (z_e.grad.abs() > 0).any()


# ---------------------------------------------------------------------------
# Test 6: EMA update modifies codebook entries
# ---------------------------------------------------------------------------


def test_ema_update_modifies_codebook():
    config = SharedCodebookConfig(
        num_entries=8, entry_dim=4, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t6",
    )
    cb = SharedCodebook(config)
    initial_codebook = cb.codebook.clone()
    z_e = torch.randn(100, 4)
    _, indices, _ = cb(z_e)
    cb.update_ema(z_e, indices)
    # Codebook should have moved (at least the entries that were assigned).
    assert not torch.allclose(cb.codebook, initial_codebook, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: state_bytes closed form
# ---------------------------------------------------------------------------


def test_state_bytes_closed_form():
    config = SharedCodebookConfig.vandenoord_canonical(label="t7")
    bytes_ = shared_codebook_state_bytes(config)
    # 256 * 64 = 16384 params * 2 bytes/param (fp16) = 32768
    assert bytes_ == 256 * 64 * 2
    assert bytes_ == 32_768


# ---------------------------------------------------------------------------
# Test 8: Codebook perplexity diagnostic
# ---------------------------------------------------------------------------


def test_perplexity_uniform_max():
    """Uniform usage of N entries → perplexity ≈ N."""
    n_entries = 16
    # 1024 samples uniformly over 16 entries.
    indices = torch.arange(1024) % n_entries
    p = compute_codebook_perplexity(indices, n_entries)
    assert math.isclose(p, n_entries, rel_tol=1e-3)


def test_perplexity_collapse_one():
    """All samples at one entry → perplexity = 1."""
    n_entries = 16
    indices = torch.zeros(100, dtype=torch.long)
    p = compute_codebook_perplexity(indices, n_entries)
    assert math.isclose(p, 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 9: Public-alias helper matches forward
# ---------------------------------------------------------------------------


def test_quantize_via_shared_codebook_alias():
    config = SharedCodebookConfig(
        num_entries=8, entry_dim=4, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t9",
    )
    cb = SharedCodebook(config)
    torch.manual_seed(0)
    z_e = torch.randn(3, 4)
    z_q1, idx1, c1 = cb(z_e)
    torch.manual_seed(0)
    z_e2 = torch.randn(3, 4)
    z_q2, idx2, c2 = quantize_via_shared_codebook(cb, z_e2)
    assert torch.equal(z_q1, z_q2)
    assert torch.equal(idx1, idx2)
    assert torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# Test 10: Forward rejects wrong entry dim
# ---------------------------------------------------------------------------


def test_forward_rejects_wrong_entry_dim():
    config = SharedCodebookConfig(
        num_entries=8, entry_dim=4, ema_decay=0.99,
        epsilon_laplace=1e-5, quantization="fp16", label="t10",
    )
    cb = SharedCodebook(config)
    with pytest.raises(SharedCodebookError) as exc:
        cb(torch.randn(2, 5))  # wrong last dim
    assert "entry_dim" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 11: Factory rejects non-config
# ---------------------------------------------------------------------------


def test_factory_rejects_dict():
    with pytest.raises(SharedCodebookError):
        SharedCodebook({"num_entries": 256})  # noqa: type-arg


# ---------------------------------------------------------------------------
# Test 12: EMA decay 0.99 distinct from weight EMA 0.997
# ---------------------------------------------------------------------------


def test_ema_decay_default_99_not_997():
    """Per CLAUDE.md EMA exception: VQ-VAE codebooks use 0.99, not 0.997."""
    config = SharedCodebookConfig.vandenoord_canonical(label="t12")
    assert config.ema_decay == 0.99
    # Sanity: 0.997 must be REJECTABLE if the caller wants to enforce
    # weight-style EMA — but it's not invalid for codebooks (>= 0.9).
    config2 = SharedCodebookConfig(
        num_entries=8, entry_dim=4, ema_decay=0.997,
        epsilon_laplace=1e-5, quantization="fp16", label="t12b",
    )
    assert config2.ema_decay == 0.997  # accepted, but not the canon


# ---------------------------------------------------------------------------
# Test 13: Public API surface complete
# ---------------------------------------------------------------------------


def test_public_api_complete():
    from tac import shared_vq_codebook as mod

    expected = {
        "SharedCodebookConfig",
        "SharedCodebook",
        "quantize_via_shared_codebook",
        "shared_codebook_state_bytes",
        "compute_codebook_perplexity",
        "SharedCodebookError",
    }
    assert set(mod.__all__) == expected
    for name in expected:
        assert hasattr(mod, name)
