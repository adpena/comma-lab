"""Unit tests for tac.paradigm_delta_epsilon_zeta.balle_hyperprior."""
from __future__ import annotations

import pytest
import torch

from tac.paradigm_delta_epsilon_zeta.balle_hyperprior import (
    BalleHyperpriorConfig,
    BalleHyperpriorWrapper,
    _get_default_scale_table,
    build_balle_hyperprior,
)


def test_factory_builds_wrapper():
    wrapper = build_balle_hyperprior()
    assert isinstance(wrapper, BalleHyperpriorWrapper)


def test_default_config_y_channels():
    cfg = BalleHyperpriorConfig()
    assert cfg.y_channels == 28


def test_forward_returns_required_keys():
    wrapper = build_balle_hyperprior()
    y = torch.randn(8, 28)
    out = wrapper(y)
    assert {"y_hat", "rate_y_bits", "rate_z_bits", "rate_total_bits"}.issubset(out)


def test_forward_y_hat_shape_matches_input():
    wrapper = build_balle_hyperprior()
    y = torch.randn(8, 28)
    out = wrapper(y)
    assert out["y_hat"].shape == y.shape


def test_rate_total_bits_equals_sum():
    wrapper = build_balle_hyperprior()
    y = torch.randn(16, 28)
    out = wrapper(y)
    expected = out["rate_y_bits"] + out["rate_z_bits"]
    torch.testing.assert_close(out["rate_total_bits"], expected)


def test_rate_is_nonnegative():
    wrapper = build_balle_hyperprior()
    y = torch.randn(16, 28)
    out = wrapper(y)
    assert float(out["rate_total_bits"]) > 0.0


def test_aux_loss_returns_scalar_tensor():
    wrapper = build_balle_hyperprior()
    aux = wrapper.aux_loss()
    assert torch.is_tensor(aux)
    assert aux.numel() == 1


def test_compress_requires_update_call():
    wrapper = build_balle_hyperprior()
    y = torch.randn(8, 28)
    with pytest.raises(RuntimeError, match="compress.*before update"):
        wrapper.compress(y)


def test_decompress_requires_update_call():
    wrapper = build_balle_hyperprior()
    with pytest.raises(RuntimeError, match="decompress.*before update"):
        wrapper.decompress({"y_strings": [], "z_strings": [], "z_shape": [1, 1, 1, 1]})


def test_update_marks_cdf_built():
    wrapper = build_balle_hyperprior()
    assert wrapper._cdf_table_built is False
    wrapper.update(force=True)
    assert wrapper._cdf_table_built is True


def test_forward_supports_backprop_through_y():
    wrapper = build_balle_hyperprior()
    y = torch.randn(8, 28, requires_grad=True)
    out = wrapper(y)
    out["rate_total_bits"].backward()
    assert y.grad is not None


def test_n_parameters_returns_positive_count():
    wrapper = build_balle_hyperprior()
    n = wrapper.n_parameters
    assert n > 0


def test_default_scale_table_geometry():
    cfg = BalleHyperpriorConfig(scale_table_min=0.1, scale_table_max=10.0, scale_table_levels=8)
    table = _get_default_scale_table(cfg)
    assert table.numel() == 8
    assert float(table[0]) == pytest.approx(0.1, rel=1e-5)
    assert float(table[-1]) == pytest.approx(10.0, rel=1e-5)


def test_custom_y_channels_propagates():
    cfg = BalleHyperpriorConfig(y_channels=16, z_channels=8)
    wrapper = BalleHyperpriorWrapper(cfg)
    y = torch.randn(4, 16)
    out = wrapper(y)
    assert out["y_hat"].shape == (4, 16)
