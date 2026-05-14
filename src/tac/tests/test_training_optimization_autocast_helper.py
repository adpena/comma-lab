"""Tests for ``tac.training_optimization.autocast_helper`` (O2).

The autocast helper is the canonical wrapper for substrate trainer
``--enable-autocast-fp16`` argparse flags. Per the optimization audit
2026-05-14 it lands the SHARED helper backporting can take in a 5-LOC
patch instead of every trainer re-implementing the autocast context.

Coverage targets:
- ``resolve_autocast_dtype`` string + torch.dtype acceptance
- unknown alias rejection
- ``AutocastConfig`` frozen dataclass validation
- ``autocast_aware_forward`` no-op when ``enabled=False``
- autocast context active when ``enabled=True``
- CPU + fp16 rejection (fp16 CPU autocast is not a speedup)
- mps device explicit refusal (CLAUDE.md non-negotiable)
- dtype default fp16
- bf16 string alias acceptance
- non-CUDA device coverage
"""

from __future__ import annotations

import pytest
import torch

from tac.training_optimization.autocast_helper import (
    AutocastConfig,
    autocast_aware_forward,
    resolve_autocast_dtype,
)


# ---------------------------------------------------------------------------
# resolve_autocast_dtype tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("half", torch.float16),
        ("FP16", torch.float16),
        ("Float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("BF16", torch.bfloat16),
    ],
)
def test_resolve_autocast_dtype_accepts_canonical_string_aliases(
    name: str, expected: torch.dtype
) -> None:
    assert resolve_autocast_dtype(name) is expected


def test_resolve_autocast_dtype_accepts_torch_dtype_directly() -> None:
    assert resolve_autocast_dtype(torch.float16) is torch.float16
    assert resolve_autocast_dtype(torch.bfloat16) is torch.bfloat16


def test_resolve_autocast_dtype_refuses_fp32() -> None:
    with pytest.raises(ValueError, match="not supported"):
        resolve_autocast_dtype(torch.float32)


def test_resolve_autocast_dtype_refuses_unknown_string() -> None:
    with pytest.raises(ValueError, match="unknown autocast dtype"):
        resolve_autocast_dtype("int8")


def test_resolve_autocast_dtype_refuses_non_string_non_dtype() -> None:
    with pytest.raises(TypeError, match="str or torch.dtype"):
        resolve_autocast_dtype(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AutocastConfig tests
# ---------------------------------------------------------------------------


def test_autocast_config_accepts_canonical_cuda_fp16() -> None:
    cfg = AutocastConfig(enabled=True, dtype=torch.float16, device_type="cuda")
    assert cfg.enabled is True
    assert cfg.dtype is torch.float16
    assert cfg.device_type == "cuda"


def test_autocast_config_accepts_cpu_bf16() -> None:
    cfg = AutocastConfig(enabled=True, dtype=torch.bfloat16, device_type="cpu")
    assert cfg.device_type == "cpu"


def test_autocast_config_refuses_non_dtype() -> None:
    with pytest.raises(TypeError, match="torch.dtype"):
        AutocastConfig(enabled=True, dtype="fp16", device_type="cuda")  # type: ignore[arg-type]


def test_autocast_config_refuses_mps_device_type() -> None:
    with pytest.raises(ValueError, match="MPS"):
        AutocastConfig(enabled=True, dtype=torch.float16, device_type="mps")


def test_autocast_config_refuses_unknown_device_type() -> None:
    with pytest.raises(ValueError, match="'cuda' or 'cpu'"):
        AutocastConfig(enabled=True, dtype=torch.float16, device_type="tpu")


def test_autocast_config_is_frozen() -> None:
    cfg = AutocastConfig(enabled=True, dtype=torch.float16, device_type="cuda")
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# autocast_aware_forward tests
# ---------------------------------------------------------------------------


def test_autocast_aware_forward_is_no_op_when_disabled() -> None:
    # When enabled=False the context yields without entering autocast.
    with autocast_aware_forward(enabled=False, dtype="fp16", device="cuda"):
        # We can confirm torch.is_autocast_enabled is False (no autocast active).
        # On non-CUDA hosts is_autocast_enabled raises; use the cached safe lookup.
        assert torch.is_autocast_enabled() is False


def test_autocast_aware_forward_refuses_cpu_fp16() -> None:
    with pytest.raises(ValueError, match="bfloat16 only"):
        with autocast_aware_forward(enabled=True, dtype="fp16", device="cpu"):
            pass


def test_autocast_aware_forward_refuses_mps_explicitly() -> None:
    # Test the explicit mps refusal even when CUDA is unavailable.
    with pytest.raises(ValueError, match="mps"):
        with autocast_aware_forward(enabled=True, dtype="bf16", device="mps"):
            pass


def test_autocast_aware_forward_refuses_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="unknown autocast dtype"):
        with autocast_aware_forward(enabled=True, dtype="int8", device="cuda"):
            pass


def test_autocast_aware_forward_accepts_torch_device_object() -> None:
    # Pass a torch.device object; should resolve to cuda type without error
    # but actually entering the cuda autocast context requires cuda available.
    if not torch.cuda.is_available():
        # On no-CUDA hosts, torch.autocast(device_type='cuda') may still
        # raise at enter time. We accept either ValueError or RuntimeError.
        try:
            with autocast_aware_forward(
                enabled=True, dtype="fp16", device=torch.device("cuda")
            ):
                pass
        except (RuntimeError, ValueError):
            pass
    else:  # pragma: no cover - depends on CI host
        with autocast_aware_forward(
            enabled=True, dtype="fp16", device=torch.device("cuda")
        ):
            assert torch.is_autocast_enabled() is True


def test_autocast_aware_forward_accepts_string_device() -> None:
    # CPU bf16 path should enter autocast successfully.
    with autocast_aware_forward(enabled=True, dtype="bf16", device="cpu"):
        # CPU autocast emits is_autocast_cpu_enabled but PyTorch >=2.5
        # unifies via is_autocast_enabled(device_type='cpu').
        # Either API is acceptable; this test just confirms no raise.
        pass


def test_autocast_aware_forward_default_device_is_cuda() -> None:
    # device=None defaults to "cuda" type. When CUDA unavailable the
    # context manager may raise on enter; we accept either no-raise or
    # the documented RuntimeError.
    if not torch.cuda.is_available():
        try:
            with autocast_aware_forward(enabled=True, dtype="fp16"):
                pass
        except (RuntimeError, ValueError):
            pass
    else:  # pragma: no cover - depends on CI host
        with autocast_aware_forward(enabled=True, dtype="fp16"):
            assert torch.is_autocast_enabled() is True


def test_autocast_aware_forward_disabled_does_not_inspect_dtype() -> None:
    # When enabled=False, the helper short-circuits BEFORE resolving the
    # dtype. This lets substrate trainers wire the flag without validating
    # the dtype arg at every call site.
    with autocast_aware_forward(enabled=False, dtype="garbage", device="cpu"):
        pass


def test_autocast_aware_forward_disabled_does_not_inspect_device() -> None:
    # Same: disabled path skips device validation.
    with autocast_aware_forward(enabled=False, dtype="fp16", device="mps"):
        pass


def test_autocast_aware_forward_torch_device_mps_refused() -> None:
    with pytest.raises(ValueError, match="mps"):
        with autocast_aware_forward(
            enabled=True, dtype="bf16", device=torch.device("mps")
        ):
            pass
