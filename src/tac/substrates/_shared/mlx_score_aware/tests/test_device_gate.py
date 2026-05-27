# SPDX-License-Identifier: MIT
"""Unit tests for the fail-closed MLX device gate."""
from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    is_mlx_available,
    require_mlx_for_harness,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")
no_mlx_only = pytest.mark.skipif(_MLX, reason="non-MLX host required")


def test_is_mlx_available_matches_import_state() -> None:
    assert is_mlx_available() == _MLX


def test_error_type_is_runtime_error_subclass() -> None:
    assert issubclass(MlxScoreAwareHarnessError, RuntimeError)


@mlx_only
def test_require_returns_mlx_core() -> None:
    mx = require_mlx_for_harness()
    assert hasattr(mx, "array")


@no_mlx_only
def test_require_fails_closed_off_apple_silicon() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="NO CPU/CUDA fallback"):
        require_mlx_for_harness()
