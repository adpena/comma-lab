# SPDX-License-Identifier: MIT
"""Fail-closed MLX device gate for the score-aware harness.

# AUTOCAST_FP16_WAIVED:MLX_harness_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive

Separation of concerns: this module owns ONLY the "is MLX available; if not,
fail closed" contract per CLAUDE.md "Forbidden device-selection defaults (the
MPS-fallback trap)" (Catalog #1) + the one-arg local-MPS-vs-modal switch
discipline (Catalog #317). There is NO silent CPU/CUDA fallback — the harness
is MLX-local ($0 M5 Max) by construction; on a non-MLX host the correct
behaviour is a clear diagnostic, NOT a paid-dispatch leak.

[verified-against: tac.substrates._shared.trainer_skeleton.device_or_die PyTorch CUDA sister]
"""
from __future__ import annotations

from typing import Any


class MlxScoreAwareHarnessError(RuntimeError):
    """Raised when the MLX score-aware harness cannot run faithfully.

    Shared error type across every harness sub-module so a caller can
    ``except MlxScoreAwareHarnessError`` for any harness-level failure
    (missing MLX, bad bundle, non-portable inflate, short video) without
    catching unrelated ``RuntimeError`` / ``ValueError``.
    """


def require_mlx_for_harness() -> Any:
    """Return the ``mlx.core`` module or fail closed on a non-MLX host.

    Per Catalog #1 + #317: NO silent CPU/CUDA fallback. The harness is
    MLX-local ($0 M5 Max) by construction; on a non-MLX host the correct
    behaviour is a clear diagnostic, NOT a paid-dispatch leak.

    Returns:
        the imported ``mlx.core`` module.

    Raises:
        MlxScoreAwareHarnessError: MLX is not importable.
    """
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - non-Apple CI.
        raise MlxScoreAwareHarnessError(
            "MLX score-aware harness requires MLX (Apple Silicon only). "
            "Install via `uv pip install mlx` or invoke from a macOS-ARM64 "
            "host. There is NO CPU/CUDA fallback (Catalog #1 + #317): the "
            "PyTorch sister path is `pact_nerv_full_main.py` / "
            "`trainer_skeleton.device_or_die` for CUDA dispatch."
        ) from exc
    return mx


def is_mlx_available() -> bool:
    """Return True iff MLX is importable in the current process (no raise)."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:  # pragma: no cover - non-Apple CI.
        return False
    return True


__all__ = [
    "MlxScoreAwareHarnessError",
    "is_mlx_available",
    "require_mlx_for_harness",
]
