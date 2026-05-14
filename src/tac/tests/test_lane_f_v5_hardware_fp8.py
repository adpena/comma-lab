# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lane F-V5 (hardware FP8 via torchao) — modules now landed. The skip wrapper
# is kept only as a defensive guard for environments without
# ``torch.float8_e4m3fn`` (PyTorch < 2.1).
import tac.quantization_fp8  # noqa: F401
from tac.renderer_export import export_hardware_fp8_checkpoint  # noqa: F401


def _mock_cuda_capability(monkeypatch: pytest.MonkeyPatch, cc: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cc)


def _require_float8() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable in this PyTorch build")


def _tiny_linear() -> nn.Linear:
    torch.manual_seed(7)
    return nn.Linear(4, 3, bias=False)


def test_hardware_capability_check_rejects_fp4_on_4090(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_capability(monkeypatch, (8, 9))

    from tac.quantization import assert_quantization_hardware_supported

    with pytest.raises(ValueError, match="FP4|fp4|10\\.0|Blackwell"):
        assert_quantization_hardware_supported("fp4", torch.device("cuda:0"))


def test_hardware_capability_check_accepts_fp8_on_4090(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda_capability(monkeypatch, (8, 9))

    from tac.quantization import (
        assert_quantization_hardware_supported,
        get_supported_quantization_modes,
    )

    modes = get_supported_quantization_modes(torch.device("cuda:0"))
    assert {"fp16", "bf16", "int8", "fp8"}.issubset(modes)
    assert "fp4" not in modes
    assert_quantization_hardware_supported("hardware_fp8", torch.device("cuda:0"))


def test_calibration_freezes_after_warmup() -> None:
    _require_float8()
    from tac.quantization_fp8 import HardwareFP8Quantizer

    quantizer = HardwareFP8Quantizer(momentum=0.95)
    quantizer.calibrate_from_warmup_batches([torch.ones(8, 4) * 2.0])
    first_scale = quantizer.activation_scale.clone()

    assert quantizer.calibration_frozen is True
    quantizer.calibrate_from_warmup_batches([torch.ones(8, 4) * 200.0])
    assert torch.equal(quantizer.activation_scale, first_scale)


def test_grad_weight_remains_high_precision() -> None:
    _require_float8()
    from tac.quantization_fp8 import HardwareFP8Quantizer

    quantizer = HardwareFP8Quantizer(momentum=0.95)
    model = quantizer.setup_module(_tiny_linear())
    x = torch.randn(5, 4, dtype=torch.bfloat16)
    y = torch.randn(5, 3, dtype=torch.bfloat16)

    out = model(x)
    loss = F.mse_loss(out.float(), y.float())
    loss.backward()

    weight_param = next(p for p in model.parameters() if p.requires_grad)
    assert weight_param.dtype is torch.bfloat16
    assert weight_param.grad is not None
    assert weight_param.grad.dtype is torch.bfloat16


def test_forward_grad_input_are_fp8() -> None:
    _require_float8()
    from tac.quantization_fp8 import HardwareFP8Quantizer

    quantizer = HardwareFP8Quantizer(momentum=0.95)
    model = quantizer.setup_module(_tiny_linear())
    x = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)

    out = model(x)
    out.float().sum().backward()

    assert quantizer.last_forward_quant_dtype is torch.float8_e4m3fn
    assert quantizer.last_grad_input_quant_dtype is torch.float8_e4m3fn


def test_export_load_roundtrip_fp8h_sentinel(tmp_path) -> None:
    _require_float8()
    from tac.renderer import AsymmetricPairGenerator
    from tac.renderer_export import (
        load_hardware_fp8_checkpoint,
    )

    torch.manual_seed(11)
    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=2,
        base_ch=4,
        mid_ch=4,
        motion_hidden=4,
        depth=1,
        pose_dim=0,
        use_dsconv=False,
        use_zoom_flow=False,
        padding_mode="zeros",
        use_dilation=False,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.25, 0.25)

    out_path = tmp_path / "renderer_fp8h.bin"
    nbytes = export_hardware_fp8_checkpoint(model, out_path)
    assert nbytes == out_path.stat().st_size
    assert out_path.read_bytes()[:4] == b"FP8H"

    loaded = load_hardware_fp8_checkpoint(out_path, device="cpu")
    original = model.state_dict()
    restored = loaded.state_dict()
    assert original.keys() == restored.keys()
    for name, value in original.items():
        if not torch.is_floating_point(value):
            continue
        assert torch.allclose(restored[name].float(), value.float(), atol=5e-2, rtol=2.5e-1), name


def test_t4_fallback_to_fp16(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _require_float8()
    from tac.renderer import AsymmetricPairGenerator

    inflate = pytest.importorskip("submissions.robust_current.inflate_renderer")

    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=2,
        base_ch=4,
        mid_ch=4,
        motion_hidden=4,
        depth=1,
        pose_dim=0,
        use_dsconv=False,
        use_zoom_flow=False,
        padding_mode="zeros",
        use_dilation=False,
    )
    out_path = tmp_path / "renderer_fp8h.bin"
    export_hardware_fp8_checkpoint(model, out_path)

    _mock_cuda_capability(monkeypatch, (7, 5))
    loaded = inflate._inline_load_fp8h(out_path.read_bytes(), device="cpu")
    captured = capsys.readouterr()

    assert getattr(loaded, "_fp8h_loaded_with_fallback_fp16", False) is True
    assert "WARNING" in captured.err
    assert "FP16" in captured.err


def test_smoke_train_50_batches() -> None:
    _require_float8()
    from tac.quantization_fp8 import HardwareFP8Quantizer

    torch.manual_seed(123)
    true_w = torch.tensor([[0.5, -0.25, 0.125, 0.75]], dtype=torch.float32)
    x = torch.randn(32, 4, dtype=torch.bfloat16)
    y = (x.float() @ true_w.t()).to(torch.bfloat16)

    quantizer = HardwareFP8Quantizer(momentum=0.95)
    quantizer.calibrate_from_warmup_batches([x])
    model = quantizer.setup_module(nn.Linear(4, 1, bias=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

    losses: list[float] = []
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred.float(), y.float())
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]


def test_profile_f_v5_hardware_fp8_is_importable() -> None:
    profiles = importlib.import_module("tac.profiles")
    profile = profiles.get_profile("f_v5_hardware_fp8_dilated_h64")

    assert profile["variant"] == "dilated"
    assert profile["hidden"] == 64
    assert profile["loss_mode"] == "standard"
    assert profile["quantization_mode"] == "hardware_fp8"
    assert profile["qat_warmup_batches"] == 50
