# SPDX-License-Identifier: MIT
"""#265 class-guard: custom Metal grouped-backward must FAIL SOFT off-GPU.

Bug class (order-of-check): the adapter choice (custom Metal kernels vs the
reference loop) is made at scorer-CONVERSION time from ``mx.default_device()``,
which can precede the trainer's FINAL device resolution
(``temporary_mlx_device(args.mlx_device)``). Before the fix, an adapter
installed under the process-default GPU crashed at VJP time under
``--mlx-device cpu`` with ``ValueError: [metal_kernel] Only supports the GPU.``
(reproduced 2026-07-08, task #265).

Guards here:
  1. VJP invoked on a non-GPU default device falls soft to the native MLX VJP
     (correct on CPU — the strided-grouped blowup is Metal-only) instead of
     crashing;
  2. the CPU-fallback gradient matches the trusted reference-adapter gradient;
  3. the fallback emits ONE loud warning (never silent — confound discipline);
  4. the GPU fast path is untouched (bit-identical to the CPU-fallback grad).

Sibling sweep (same order-of-check class, 2026-07-08): ``metal_fused_r_operator``
is fail-closed at the RESOLVED config (`--fused-r-kernel` refused unless
``--mlx-device gpu``; availability asserts are call-time) — not vulnerable. The
other #212 kernels (AA-SDF / margin / curvelet / clDice / island-birth) are
mx.compile or numpy paths (device-portable) or FLAGGED_NOT_BUILT.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from tac.local_acceleration import metal_grouped_conv_backward as mgcb  # noqa: E402
from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    MLXCustomKernelStridedGroupedConvAdapter,
    torch_conv2d_to_mlx_reference,
)


def _strided_depthwise_conv() -> torch.nn.Conv2d:
    torch.manual_seed(0)
    return torch.nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False)


def _input() -> mx.array:
    rng = np.random.default_rng(0)
    return mx.array(rng.standard_normal((1, 12, 12, 8)).astype(np.float32))


@pytest.fixture()
def _reset_warned_flag():
    old = mgcb._CPU_VJP_FALLBACK_WARNED
    mgcb._CPU_VJP_FALLBACK_WARNED = False
    yield
    mgcb._CPU_VJP_FALLBACK_WARNED = old


@pytest.fixture()
def _restore_device():
    old = mx.default_device()
    yield
    mx.set_default_device(old)


def test_custom_adapter_vjp_falls_soft_on_cpu_and_matches_reference(
    _reset_warned_flag, _restore_device, capsys
):
    conv = _strided_depthwise_conv()
    # install under whatever the process default is (the order-of-check hazard) …
    adapter = MLXCustomKernelStridedGroupedConvAdapter(conv)
    # … then resolve the FINAL device to cpu, as `--mlx-device cpu` does.
    mx.set_default_device(mx.Device(mx.cpu))
    x = _input()
    g = mx.grad(lambda x_: adapter(x_).sum())(x)
    mx.eval(g)  # pre-fix: ValueError "[metal_kernel] Only supports the GPU."
    ref = torch_conv2d_to_mlx_reference(conv)
    g_ref = mx.grad(lambda x_: ref(x_).sum())(x)
    mx.eval(g_ref)
    assert float(mx.abs(g - g_ref).max()) < 1e-6
    # loud, not silent (confound self-protection: no silent degradation)
    err = capsys.readouterr().err
    assert "falling back to the native MLX VJP" in err


def test_cpu_fallback_warns_exactly_once(_reset_warned_flag, _restore_device, capsys):
    conv = _strided_depthwise_conv()
    adapter = MLXCustomKernelStridedGroupedConvAdapter(conv)
    mx.set_default_device(mx.Device(mx.cpu))
    x = _input()
    for _ in range(2):
        g = mx.grad(lambda x_: adapter(x_).sum())(x)
        mx.eval(g)
    err = capsys.readouterr().err
    assert err.count("falling back to the native MLX VJP") == 1


@pytest.mark.skipif(
    not mgcb.metal_grouped_conv2d_backend_available()
    and mx.default_device().type != mx.gpu,
    reason="no Metal GPU default device on this host",
)
def test_gpu_fast_path_unaffected_and_matches_cpu_fallback(
    _reset_warned_flag, _restore_device
):
    conv = _strided_depthwise_conv()
    adapter = MLXCustomKernelStridedGroupedConvAdapter(conv)
    x = _input()
    mx.set_default_device(mx.Device(mx.cpu))
    g_cpu = mx.grad(lambda x_: adapter(x_).sum())(x)
    mx.eval(g_cpu)
    g_cpu_np = np.array(g_cpu)
    mx.set_default_device(mx.Device(mx.gpu))
    g_gpu = mx.grad(lambda x_: adapter(x_).sum())(x)
    mx.eval(g_gpu)
    assert float(np.abs(g_cpu_np - np.array(g_gpu)).max()) < 1e-5
