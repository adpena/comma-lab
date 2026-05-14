# SPDX-License-Identifier: MIT
"""Canonical ``torch.compile`` wrapper for substrate trainers (Catalog #179).

Why this exists
---------------

REVIEW-OMNI Low NV3 audit 2026-05-13 + optimization-opportunities audit
2026-05-14 identified that 25/26 substrate trainers carry the file-level
waiver ``# TORCH_COMPILE_WAIVED:<reason>``. The audit estimates Inductor
compilation can deliver 1.5-2x speedup on the dominant scorer-forward
hot path (PoseNet FastViT-T12 + SegNet EfficientNet-B2 UNet). The fix is
a SHARED helper that wraps ``torch.compile`` with graceful fallback so
substrate trainers can opt in via the existing ``--enable-torch-compile``
argparse flag without each one re-implementing the try/except boilerplate.

Speedup estimate
----------------

[literature-extrapolation, PyTorch 2.1+ Inductor]: 1.5-2× total per-step
wall-clock on A100 / Ampere+. First-epoch overhead is ~5-15s (graph
capture + Inductor backend compile); steady-state second-epoch onward
hits the cached compiled graph.

Signal regression risk
----------------------

~1e-5 score drift expected; Inductor numerics are deterministic but not
bit-identical to eager mode. The contest auth eval still runs at eager
mode (inflate.py does not call torch.compile). Operator-routable per
the audit's probe-disambiguator hook: if any substrate observes >0
score regression with compile enabled, fall back via the existing
``fallback_on_error=True`` default.

Error handling
--------------

``torch.compile`` can fail at graph capture time for several reasons:

* Inductor backend missing on CPU-only environments (e.g., macOS dev loop)
* Dynamic shape mismatch in the substrate forward
* Missing Triton on non-CUDA platforms
* Specific operators not yet supported by Inductor

The canonical helper catches ``Exception`` (broad on purpose: Inductor
raises a variety of internal exceptions that are version-dependent) and
returns the uncompiled model with a warning. Caller-facing semantics:

* ``fallback_on_error=True`` (default) — warn + return uncompiled model;
  training continues, no hard fail. This matches the substrate-trainer
  convention of "compile is a speedup hint, not a correctness hint."
* ``fallback_on_error=False`` — re-raise the original exception. Useful
  for tests that want to assert compile WORKS rather than just attempts.

Per CLAUDE.md "Apples-to-apples evidence discipline" no score claim is
attached to this module. The helper is a pure speedup primitive with a
documented fallback mechanism.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch


__all__ = [
    "CompileConfig",
    "compile_with_fallback",
]


# Canonical Inductor compile modes per PyTorch 2.1+ docs:
#   "default"          — balanced; works for most models
#   "reduce-overhead"  — uses CUDA graphs; lower overhead at small batch sizes
#   "max-autotune"     — exhaustive autotuning; slowest compile, fastest run
# Substrate trainers should default to "default" unless they have measured
# that another mode helps the specific architecture.
_VALID_COMPILE_MODES: frozenset[str] = frozenset(
    {"default", "reduce-overhead", "max-autotune"}
)


@dataclass(frozen=True)
class CompileConfig:
    """Frozen configuration for ``compile_with_fallback``.

    Substrate trainers declare compile intent once at trainer init and
    reuse across the forward-wrap call site.
    """

    enabled: bool
    mode: str
    fallback_on_error: bool
    dynamic: bool | None  # None lets torch.compile pick; True/False forces

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"CompileConfig.enabled must be bool; got "
                f"{type(self.enabled).__name__}"
            )
        if self.mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"CompileConfig.mode must be one of {sorted(_VALID_COMPILE_MODES)}; "
                f"got {self.mode!r}"
            )
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise TypeError(
                f"CompileConfig.dynamic must be bool or None; got "
                f"{type(self.dynamic).__name__}"
            )


def compile_with_fallback(
    model: torch.nn.Module,
    *,
    enabled: bool = True,
    mode: str = "default",
    fallback_on_error: bool = True,
    dynamic: bool | None = None,
) -> torch.nn.Module:
    """Compile ``model`` with ``torch.compile``; fall back to uncompiled on error.

    Args:
        model: ``torch.nn.Module`` to wrap. Returned as-is when
            ``enabled=False``.
        enabled: When False, returns ``model`` unchanged. This is the
            canonical wiring for the ``--enable-torch-compile`` argparse
            flag.
        mode: Inductor compile mode. One of ``"default"``,
            ``"reduce-overhead"``, ``"max-autotune"``.
        fallback_on_error: When True (default), catch any exception from
            ``torch.compile`` and return the uncompiled model with a
            warning. When False, re-raise the original exception.
        dynamic: Forward to ``torch.compile(dynamic=...)``. ``None`` lets
            PyTorch choose (recommended for most substrate forwards which
            have fixed input shapes per epoch).

    Returns:
        Compiled model (when compile succeeded and enabled), OR the
        original uncompiled model (when ``enabled=False`` or compile
        failed with ``fallback_on_error=True``).

    Raises:
        ValueError: If ``mode`` is not a recognized Inductor compile mode.
        TypeError: If ``model`` is not a ``torch.nn.Module``.
        Exception: Re-raised from ``torch.compile`` when
            ``fallback_on_error=False``.

    Examples:
        Canonical substrate-trainer pattern::

            substrate = build_substrate(config).to(device)
            substrate = compile_with_fallback(
                substrate,
                enabled=args.enable_torch_compile,
                mode="default",
            )
            for step, batch in enumerate(loader):
                output = substrate(batch)  # compiled forward
                ...
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"compile_with_fallback expects torch.nn.Module; got "
            f"{type(model).__name__}"
        )

    if mode not in _VALID_COMPILE_MODES:
        raise ValueError(
            f"compile mode must be one of {sorted(_VALID_COMPILE_MODES)}; "
            f"got {mode!r}"
        )

    if not enabled:
        return model

    try:
        compile_kwargs: dict[str, Any] = {"mode": mode}
        if dynamic is not None:
            compile_kwargs["dynamic"] = dynamic
        return torch.compile(model, **compile_kwargs)
    except Exception as exc:  # pragma: no cover - depends on PyTorch version
        if not fallback_on_error:
            raise
        warnings.warn(
            f"torch.compile(mode={mode!r}) failed: {exc!s}; "
            "falling back to uncompiled model. Pass "
            "fallback_on_error=False to surface the original exception.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model
