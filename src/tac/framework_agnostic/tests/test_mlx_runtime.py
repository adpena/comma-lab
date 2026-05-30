# SPDX-License-Identifier: MIT
"""Canonical MLX runtime acquisition helper tests."""

from __future__ import annotations

import pytest

from tac.framework_agnostic import (
    BackendUnavailableError,
    is_mlx_runtime_available,
    mlx_array,
    mlx_eval,
    optional_mlx_runtime,
    require_mlx_core,
    require_mlx_runtime,
)


def test_optional_runtime_and_availability_agree():
    runtime = optional_mlx_runtime()
    assert is_mlx_runtime_available() is (runtime is not None)
    if runtime is not None:
        assert runtime.mx is not None
        assert runtime.nn is None
        assert runtime.optimizers is None
        assert runtime.utils is None


def test_require_core_fail_closed_or_returns_mlx_core():
    if optional_mlx_runtime() is None:
        with pytest.raises(BackendUnavailableError, match="MLX core is unavailable"):
            require_mlx_core()
        return

    mx = require_mlx_core()
    assert mx.__name__ == "mlx.core"


def test_requested_runtime_bundle_shape_when_available():
    if optional_mlx_runtime(nn=True, optimizers=True, utils=True) is None:
        pytest.skip("MLX runtime unavailable on this host")

    runtime = require_mlx_runtime(nn=True, optimizers=True, utils=True)
    assert runtime.mx.__name__ == "mlx.core"
    assert runtime.nn.__name__ == "mlx.nn"
    assert runtime.optimizers.__name__ == "mlx.optimizers"
    assert runtime.utils.__name__ == "mlx.utils"


def test_array_and_eval_wrappers_when_available():
    if optional_mlx_runtime() is None:
        pytest.skip("MLX runtime unavailable on this host")

    arr = mlx_array([1.0, 2.0, 3.0])
    mlx_eval(arr)
    assert tuple(arr.shape) == (3,)
