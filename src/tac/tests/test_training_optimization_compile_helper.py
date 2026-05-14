# SPDX-License-Identifier: MIT
"""Tests for ``tac.training_optimization.compile_helper`` (O3).

The compile helper is the canonical wrapper for substrate trainer
``--enable-torch-compile`` argparse flags. Per the optimization audit
2026-05-14 it lands the SHARED helper with graceful fallback so trainers
can opt in without each re-implementing the try/except boilerplate.

Coverage targets:
- ``CompileConfig`` dataclass validation
- ``compile_with_fallback`` is no-op when ``enabled=False``
- unknown mode rejection
- non-Module input rejection
- fallback-on-error path returns uncompiled model with warning
- ``fallback_on_error=False`` re-raises the original exception
- dynamic kwarg threaded through
- canonical modes accepted
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tac.training_optimization.compile_helper import (
    CompileConfig,
    compile_with_fallback,
)


class _TrivialModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# CompileConfig tests
# ---------------------------------------------------------------------------


def test_compile_config_accepts_canonical_default_mode() -> None:
    cfg = CompileConfig(
        enabled=True, mode="default", fallback_on_error=True, dynamic=None
    )
    assert cfg.enabled is True
    assert cfg.mode == "default"
    assert cfg.fallback_on_error is True
    assert cfg.dynamic is None


def test_compile_config_accepts_reduce_overhead_mode() -> None:
    cfg = CompileConfig(
        enabled=True,
        mode="reduce-overhead",
        fallback_on_error=False,
        dynamic=True,
    )
    assert cfg.mode == "reduce-overhead"
    assert cfg.fallback_on_error is False
    assert cfg.dynamic is True


def test_compile_config_accepts_max_autotune_mode() -> None:
    cfg = CompileConfig(
        enabled=True, mode="max-autotune", fallback_on_error=True, dynamic=False
    )
    assert cfg.mode == "max-autotune"
    assert cfg.dynamic is False


def test_compile_config_refuses_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        CompileConfig(
            enabled=True, mode="ludicrous-speed", fallback_on_error=True, dynamic=None
        )


def test_compile_config_refuses_non_bool_enabled() -> None:
    with pytest.raises(TypeError, match="bool"):
        CompileConfig(
            enabled="yes", mode="default", fallback_on_error=True, dynamic=None  # type: ignore[arg-type]
        )


def test_compile_config_refuses_non_bool_dynamic() -> None:
    with pytest.raises(TypeError, match="bool or None"):
        CompileConfig(
            enabled=True, mode="default", fallback_on_error=True, dynamic="auto"  # type: ignore[arg-type]
        )


def test_compile_config_is_frozen() -> None:
    cfg = CompileConfig(
        enabled=True, mode="default", fallback_on_error=True, dynamic=None
    )
    with pytest.raises(Exception):
        cfg.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compile_with_fallback tests
# ---------------------------------------------------------------------------


def test_compile_with_fallback_returns_unchanged_when_disabled() -> None:
    model = _TrivialModule()
    result = compile_with_fallback(model, enabled=False)
    assert result is model


def test_compile_with_fallback_refuses_non_module() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        compile_with_fallback("not a model", enabled=True)  # type: ignore[arg-type]


def test_compile_with_fallback_refuses_unknown_mode() -> None:
    model = _TrivialModule()
    with pytest.raises(ValueError, match="mode must be one of"):
        compile_with_fallback(model, enabled=True, mode="laser-go-pew-pew")


def test_compile_with_fallback_accepts_all_canonical_modes() -> None:
    model = _TrivialModule()
    # All three canonical modes should at least pass the validation stage.
    # The compile itself may fall back depending on PyTorch / inductor
    # availability on the test host, which is fine for this validation.
    for mode in ("default", "reduce-overhead", "max-autotune"):
        result = compile_with_fallback(
            model, enabled=True, mode=mode, fallback_on_error=True
        )
        assert isinstance(result, nn.Module)


def test_compile_with_fallback_returns_model_on_compile_failure() -> None:
    model = _TrivialModule()
    fake_exc = RuntimeError("inductor missing on this host")

    with patch(
        "tac.training_optimization.compile_helper.torch.compile",
        side_effect=fake_exc,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compile_with_fallback(
                model, enabled=True, mode="default", fallback_on_error=True
            )
        assert result is model
        assert any("torch.compile" in str(w.message) for w in caught)


def test_compile_with_fallback_reraises_when_fallback_disabled() -> None:
    model = _TrivialModule()
    fake_exc = RuntimeError("inductor missing on this host")
    with patch(
        "tac.training_optimization.compile_helper.torch.compile",
        side_effect=fake_exc,
    ):
        with pytest.raises(RuntimeError, match="inductor missing"):
            compile_with_fallback(
                model, enabled=True, mode="default", fallback_on_error=False
            )


def test_compile_with_fallback_threads_dynamic_kwarg() -> None:
    model = _TrivialModule()
    with patch(
        "tac.training_optimization.compile_helper.torch.compile",
        return_value=model,
    ) as mock_compile:
        compile_with_fallback(
            model, enabled=True, mode="default", dynamic=True, fallback_on_error=False
        )
        mock_compile.assert_called_once()
        # Verify dynamic kwarg was passed
        call_kwargs = mock_compile.call_args.kwargs
        assert call_kwargs.get("dynamic") is True


def test_compile_with_fallback_omits_dynamic_when_none() -> None:
    model = _TrivialModule()
    with patch(
        "tac.training_optimization.compile_helper.torch.compile",
        return_value=model,
    ) as mock_compile:
        compile_with_fallback(
            model, enabled=True, mode="default", dynamic=None, fallback_on_error=False
        )
        call_kwargs = mock_compile.call_args.kwargs
        assert "dynamic" not in call_kwargs


def test_compile_with_fallback_threads_mode_kwarg() -> None:
    model = _TrivialModule()
    with patch(
        "tac.training_optimization.compile_helper.torch.compile",
        return_value=model,
    ) as mock_compile:
        compile_with_fallback(
            model, enabled=True, mode="reduce-overhead", fallback_on_error=False
        )
        call_kwargs = mock_compile.call_args.kwargs
        assert call_kwargs.get("mode") == "reduce-overhead"


def test_compile_with_fallback_disabled_does_not_invoke_torch_compile() -> None:
    model = _TrivialModule()
    with patch(
        "tac.training_optimization.compile_helper.torch.compile"
    ) as mock_compile:
        result = compile_with_fallback(model, enabled=False)
        mock_compile.assert_not_called()
        assert result is model


def test_compile_with_fallback_disabled_short_circuits_before_mode_validation() -> (
    None
):
    # When enabled=False, mode validation is still performed (per the
    # safety design). The mode must be valid even if the helper is
    # disabled, because callers may toggle enabled mid-session.
    model = _TrivialModule()
    with pytest.raises(ValueError, match="mode must be one of"):
        compile_with_fallback(model, enabled=False, mode="garbage")


def test_compile_with_fallback_returns_torch_module_compatible() -> None:
    # Confirm the returned object responds to module methods.
    model = _TrivialModule()
    result = compile_with_fallback(model, enabled=False)
    assert isinstance(result, nn.Module)
    # Forward should still work on an uncompiled passthrough.
    x = torch.randn(1, 4)
    y = result(x)
    assert y.shape == (1, 4)
