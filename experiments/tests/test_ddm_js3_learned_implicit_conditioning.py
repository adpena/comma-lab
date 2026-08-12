from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
from torch.nn import functional

from experiments import ddm_js3_learned_implicit_conditioning as js3


def test_stage_steps_require_strict_progression() -> None:
    assert js3.parse_stage_steps("1,4,10") == (1, 4, 10)
    with pytest.raises(argparse.ArgumentTypeError):
        js3.parse_stage_steps("4,4")


def test_context_and_zero_init_are_receiver_identity() -> None:
    tokens = torch.zeros((2, js3.H, js3.W), dtype=torch.long)
    tokens[:, :, js3.W // 2 :] = 1
    pre_r = torch.full((2, 3, js3.H, js3.W), 127.5)
    context = js3.fixed_context(torch, functional, tokens, pre_r)
    assert context.shape == (2, js3.CHANNELS, js3.H, js3.W)
    assert int(context[:, 5:9].sum()) > 0
    model = js3.build_model(torch, functional, 4, 6.0, qat=True)
    assert torch.count_nonzero(model(context)) == 0


def test_delta_hinge_pushes_gt_margin_and_prices_collateral() -> None:
    logits = torch.zeros((1, 5, 2, 2), requires_grad=True)
    labels = torch.zeros((1, 2, 2), dtype=torch.long)
    baseline = torch.ones((1, 2, 2), dtype=torch.long)
    loss, terms = js3.delta_hinge_objective(torch, logits, labels, baseline)
    loss.backward()
    assert float(terms["repair_hinge"].detach()) == pytest.approx(js3.DELTA)
    assert float(logits.grad[0, 0].mean()) < 0.0


@pytest.mark.parametrize("mode", ["fp16", "int8"])
def test_real_brotli_module_roundtrip(mode: str) -> None:
    torch.manual_seed(js3.SEED)
    model = js3.build_model(torch, functional, 4, 6.0, qat=True)
    with torch.no_grad():
        model.head.weight.fill_(0.125)
    exported = js3.serialize_module(model, mode)
    decoded = js3.parse_module(exported.coded)
    assert exported.report["parseback_exact"] is True
    assert exported.report["brotli_q11_bytes"] == len(exported.coded)
    assert set(decoded) == set(model.state_dict())


def test_robust_metric_requires_signed_margin_beyond_delta() -> None:
    logits = np.zeros((1, 5, js3.H, js3.W), dtype=np.float32)
    base = np.zeros((1, js3.H, js3.W), dtype=np.uint8)
    gt = np.zeros_like(base)
    base[0, 0, 0] = 1
    logits[:, 0] = 1.0
    logits[0, 1, 0, 0] = 1.0 - js3.DELTA / 2
    fragile = js3.robust_metrics(logits, base, gt, np.array([600]))
    assert fragile["beneficial_flips"] == 1
    assert fragile["robust_beneficial_flips"] == 0
    logits[0, 1, 0, 0] = 1.0 - 2 * js3.DELTA
    robust = js3.robust_metrics(logits, base, gt, np.array([600]))
    assert robust["robust_beneficial_flips"] == 1
    assert robust["projected_n600_robust_delta_flips"] == -600
