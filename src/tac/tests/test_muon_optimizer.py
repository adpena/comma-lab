from __future__ import annotations

import pytest
import torch

from tac.optimization import (
    MuonOptimizer,
    partition_params_for_muon,
    zeropower_via_newtonschulz5,
)


def test_zeropower_preserves_shape_dtype_and_is_finite() -> None:
    grad = torch.randn(4, 3, dtype=torch.float32)

    out = zeropower_via_newtonschulz5(grad, steps=2)

    assert out.shape == grad.shape
    assert out.dtype == grad.dtype
    assert torch.isfinite(out).all()


def test_zeropower_rejects_1d_inputs() -> None:
    with pytest.raises(ValueError, match="requires ndim >= 2"):
        zeropower_via_newtonschulz5(torch.randn(4))


def test_partition_params_for_muon_keeps_stem_rgb_and_biases_on_adamw() -> None:
    class ToyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Linear(3, 4)
            self.hidden = torch.nn.Linear(4, 5)
            self.rgb_head = torch.nn.Linear(5, 3)

    model = ToyModule()

    muon_params, adamw_params = partition_params_for_muon(model)

    assert any(param is model.hidden.weight for param in muon_params)
    assert any(param is model.hidden.bias for param in adamw_params)
    assert any(param is model.stem.weight for param in adamw_params)
    assert any(param is model.rgb_head.weight for param in adamw_params)
    assert all(param.requires_grad for param in [*muon_params, *adamw_params])


def test_muon_optimizer_updates_2d_parameter_and_records_momentum() -> None:
    weight = torch.nn.Parameter(torch.eye(3))
    opt = MuonOptimizer([weight], lr=0.01, momentum=0.5, ns_steps=1, weight_decay=0.01)
    before = weight.detach().clone()
    weight.grad = torch.ones_like(weight)

    opt.step()

    assert not torch.equal(weight.detach(), before)
    assert "momentum_buffer" in opt.state[weight]
    assert opt.state[weight]["momentum_buffer"].shape == weight.shape


def test_muon_optimizer_returns_closure_loss() -> None:
    weight = torch.nn.Parameter(torch.ones(2, 2))
    opt = MuonOptimizer([weight], lr=0.01, ns_steps=1)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = weight.square().sum()
        loss.backward()
        return loss

    loss = opt.step(closure)

    assert loss is not None
    assert loss.item() == pytest.approx(4.0)
