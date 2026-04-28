"""Smoke tests for the Lane Ω-V2 Lagrangian QAT loop.

This is a STRUCTURAL smoke test, not a real QAT run — we exercise the
math primitives + bit-pressure mechanics on a tiny model to verify the
Lagrangian dynamics are set up correctly.

Pins:
  1. λ schedule is correct at boundary epochs.
  2. Rate penalty fires when mean bits > target.
  3. After a few optimizer steps with high λ + low target, mean bits
     decreases (verifies dual ascent direction).
  4. Bits stay in [1, 8] (parameterisation is robust).
  5. The QAT entrypoint argparse exposes the flags the launch script uses
     (CLAUDE.md NEVER-invent-flags rule, applied internally).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
QAT_SCRIPT = REPO_ROOT / "experiments" / "qat_omega_lagrangian.py"


# ── Load the QAT script as a module so we can call its helpers ──────────


@pytest.fixture(scope="module")
def qat_module():
    spec = importlib.util.spec_from_file_location("qat_omega_lagrangian", QAT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── λ schedule ───────────────────────────────────────────────────────────


def test_lambda_at_start_equals_lambda_start(qat_module):
    """At epoch 0 (before ramp), λ = lambda_start."""
    lam = qat_module._lambda_for_epoch(
        epoch=0, total_epochs=200,
        lambda_start=0.0, lambda_end=1.0, ramp_start_frac=0.3,
    )
    assert lam == 0.0


def test_lambda_pre_ramp_is_lambda_start(qat_module):
    """For epoch < total*ramp_start_frac, λ stays at lambda_start."""
    for epoch in (0, 30, 50, 59):
        lam = qat_module._lambda_for_epoch(
            epoch=epoch, total_epochs=200,
            lambda_start=0.0, lambda_end=1.0, ramp_start_frac=0.3,
        )
        assert lam == 0.0


def test_lambda_at_end_equals_lambda_end(qat_module):
    """At the final epoch, λ = lambda_end."""
    lam = qat_module._lambda_for_epoch(
        epoch=199, total_epochs=200,
        lambda_start=0.0, lambda_end=1.0, ramp_start_frac=0.3,
    )
    assert abs(lam - 1.0) < 1e-9


def test_lambda_monotonic(qat_module):
    """λ must be monotonically non-decreasing over epochs."""
    prev = -1.0
    for epoch in range(0, 200, 5):
        lam = qat_module._lambda_for_epoch(
            epoch=epoch, total_epochs=200,
            lambda_start=0.0, lambda_end=1.0, ramp_start_frac=0.3,
        )
        assert lam >= prev - 1e-9
        prev = lam


def test_lambda_constant_when_start_equals_end(qat_module):
    """Degenerate case: λ_start == λ_end → λ is constant."""
    for epoch in (0, 100, 199):
        lam = qat_module._lambda_for_epoch(
            epoch=epoch, total_epochs=200,
            lambda_start=0.5, lambda_end=0.5, ramp_start_frac=0.3,
        )
        assert lam == 0.5


# ── Bit-pressure mechanics ───────────────────────────────────────────────


def _build_tiny_model_with_swap():
    """A 3-layer toy CNN whose bulk layers are wrapped with
    LearnableBitConv2d. Used to exercise the Lagrangian bit-pressure
    without needing a full renderer + scorer."""
    from tac.learnable_bit_quant import (
        LearnableBitConv2d,
        compute_learnable_bit_rate_penalty,
        renderer_average_learnable_bits_per_weight,
    )

    model = nn.Sequential(
        LearnableBitConv2d(3, 8, 3, padding=1, init_bits=8.0),
        nn.ReLU(),
        LearnableBitConv2d(8, 8, 3, padding=1, init_bits=8.0),
        nn.ReLU(),
        LearnableBitConv2d(8, 3, 3, padding=1, init_bits=8.0),
    )
    return model


def test_high_lambda_drives_bits_down():
    """With high λ + low target, dual ascent should drive mean bits down
    after a few optimizer steps. This is the load-bearing dynamic of
    Lane Ω-V2 — if bits never decrease, the Lagrangian wiring is dead."""
    from tac.learnable_bit_quant import (
        compute_learnable_bit_rate_penalty,
        renderer_average_learnable_bits_per_weight,
    )

    torch.manual_seed(0)
    model = _build_tiny_model_with_swap()
    initial_bits = renderer_average_learnable_bits_per_weight(model)

    # Optimizer over bit_depth.raw only (we want to see bit pressure in
    # isolation, not entangled with weight updates).
    bits_params = [
        p for n, p in model.named_parameters() if n.endswith(".bit_depth.raw")
    ]
    optimizer = torch.optim.SGD(bits_params, lr=1.0)

    # 10 steps of pure rate-pressure (no scorer loss; just rate)
    for _ in range(10):
        pen = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=2.0, lambda_rate=10.0,
        )
        optimizer.zero_grad(set_to_none=True)
        pen.backward()
        optimizer.step()

    final_bits = renderer_average_learnable_bits_per_weight(model)
    assert final_bits < initial_bits, (
        f"high-λ rate pressure should drive bits down; "
        f"initial={initial_bits:.3f} final={final_bits:.3f}"
    )


def test_zero_lambda_keeps_bits_constant():
    """With λ=0, no rate pressure → bits do not change from rate alone."""
    from tac.learnable_bit_quant import (
        compute_learnable_bit_rate_penalty,
        renderer_average_learnable_bits_per_weight,
    )

    torch.manual_seed(0)
    model = _build_tiny_model_with_swap()
    initial_bits = renderer_average_learnable_bits_per_weight(model)

    bits_params = [
        p for n, p in model.named_parameters() if n.endswith(".bit_depth.raw")
    ]
    optimizer = torch.optim.SGD(bits_params, lr=1.0)
    for _ in range(5):
        pen = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=2.0, lambda_rate=0.0,
        )
        optimizer.zero_grad(set_to_none=True)
        pen.backward()
        optimizer.step()

    final_bits = renderer_average_learnable_bits_per_weight(model)
    # No grad → no update; bits unchanged within fp32 noise
    assert abs(final_bits - initial_bits) < 1e-4, (
        f"zero λ should leave bits unchanged; "
        f"initial={initial_bits:.4f} final={final_bits:.4f}"
    )


def test_bits_stay_in_valid_range_under_extreme_pressure():
    """Even with very high λ, bits should stay in [1, 8] (softplus + clamp)."""
    from tac.learnable_bit_quant import (
        compute_learnable_bit_rate_penalty,
        list_learnable_bit_layers,
    )

    torch.manual_seed(0)
    model = _build_tiny_model_with_swap()
    bits_params = [
        p for n, p in model.named_parameters() if n.endswith(".bit_depth.raw")
    ]
    optimizer = torch.optim.SGD(bits_params, lr=10.0)

    for _ in range(50):
        pen = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=1.0, lambda_rate=100.0,
        )
        optimizer.zero_grad(set_to_none=True)
        pen.backward()
        optimizer.step()

    for _name, layer in list_learnable_bit_layers(model):
        bits = layer.bit_depth.bits_used()
        assert (bits >= 1.0).all() and (bits <= 8.0).all()


# ── argparse coverage ────────────────────────────────────────────────────


def test_qat_argparse_exposes_required_flags():
    """The QAT script's argparse must expose every flag the launch
    bash script invokes. Mirror of the dead-flag scan inside the launch
    script itself, but enforced at import time so PR review can catch
    drift."""
    src = QAT_SCRIPT.read_text()
    flags = set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))
    # The flags that scripts/remote_lane_omega_v2_lagrangian.sh passes:
    required = {
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output-dir", "init-bits", "target-bits", "lambda-start",
        "lambda-end", "lambda-ramp-start-frac", "total-epochs",
        "lr", "bits-lr-scale", "noise-std", "seg-weight",
        "pose-weight", "device", "seed", "log-every", "hessian-init",
    }
    missing = required - flags
    assert not missing, f"qat_omega_lagrangian.py argparse missing: {missing}"


def test_qat_device_choices_exclude_mps(qat_module):
    """CLAUDE.md non-negotiable: --device cuda required, MPS forbidden.
    The argparse choices list must reflect this."""
    src = QAT_SCRIPT.read_text()
    # Find the --device argparse choices
    m = re.search(
        r'add_argument\(\s*"--device".*?choices=\[(.*?)\]',
        src, re.DOTALL,
    )
    assert m, "--device argument must declare choices"
    choices_text = m.group(1)
    assert '"mps"' not in choices_text and "'mps'" not in choices_text, (
        "--device MUST NOT include 'mps' as a choice (CLAUDE.md non-negotiable)"
    )
    assert '"cuda"' in choices_text or "'cuda'" in choices_text


def test_qat_default_eval_roundtrip_is_True():
    """Per CLAUDE.md non-negotiable: eval_roundtrip MUST default True
    everywhere. The QAT loop calls _scorer_loss(...) with
    eval_roundtrip=True hard-coded — verify that hard-coding is present."""
    src = QAT_SCRIPT.read_text()
    assert "eval_roundtrip=True" in src, (
        "QAT script must hard-code eval_roundtrip=True; "
        "if a flag is added later it MUST default True"
    )


# ── Strict-device check ─────────────────────────────────────────────────


def test_device_mps_raises(qat_module):
    """_device_strict('mps') must raise SystemExit."""
    with pytest.raises(SystemExit, match="mps"):
        qat_module._device_strict("mps")


def test_device_cuda_without_cuda_raises(qat_module):
    """If CUDA isn't available, --device cuda must raise (no silent fallback).
    Note: this test ALWAYS expects SystemExit on a no-CUDA host; on CUDA
    machines it returns a torch.device (which is also fine)."""
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="cuda"):
            qat_module._device_strict("cuda")
    else:
        d = qat_module._device_strict("cuda")
        assert d.type == "cuda"


# ── Hessian-init guard ──────────────────────────────────────────────────


def test_load_hessian_init_rejects_renderer_magic(qat_module, tmp_path):
    """The DEN-V2 trap: pickling a renderer .bin into --hessian-init must
    fail loudly, not silently use it."""
    bad = tmp_path / "fake_hessian.pt"
    bad.write_bytes(b"OMG1" + b"x" * 100)
    with pytest.raises(ValueError, match="renderer magic"):
        qat_module._load_hessian_init(str(bad))


def test_load_hessian_init_returns_none_for_none(qat_module):
    assert qat_module._load_hessian_init(None) is None


def test_load_hessian_init_missing_file_raises(qat_module, tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(SystemExit, match="does not exist"):
        qat_module._load_hessian_init(str(missing))
