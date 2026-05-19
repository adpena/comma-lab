# SPDX-License-Identifier: MIT
"""Strict fp32 matmul accumulation override for MPS-vs-CUDA convergence.

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`)
the MEDIUM-EV (2x) engineering correction targeting the Metal IEEE 754 strict
vs CUDA TF32 (19-bit vs 23-bit mantissa) accumulation gap. CUDA defaults TF32
on Ampere+; Metal MPS uses strict IEEE-754 fp32. Pinning CUDA to strict fp32
shrinks the cross-backend gap symmetrically.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 / #192 / #317: this is a
symmetric correction across BOTH backends; it does NOT promote MPS to a
score-truth axis. The contest scorer still runs on CUDA and the contest CPU
runner runs on Linux x86_64 — those axes are unchanged.

Cargo-cult-unwind per Catalog #303:
    - `torch.backends.mps.preferred_blas_library` was CITED in the task prompt
      as the canonical MPS BLAS override but does NOT exist in torch 2.11.0.
    - The canonical pre-2.11 MPS BLAS override path is the `MPS_PREFER_METAL`
      or `MPSGraphBLAS` env var family, OR (per Apple docs) leaving MPS to
      Accelerate via the default MPSGraph dispatcher. We surface the missing
      API in `MPS_BLAS_PREFERENCE_API_AVAILABLE` so callers can audit the
      structural absence without silently degrading.

Public API:
    - `enable_fp32_matmul_accumulation_strict(...)` — set strict fp32
       across CUDA + (optionally) MPS; return prior state for restoration.
    - `restore_fp32_matmul_accumulation_state(prior)` — undo.
    - `strict_fp32_matmul_accumulation()` — context manager.
    - `STRICT_FP32_FLAGS` — canonical flag tuple.
    - `MPS_BLAS_PREFERENCE_API_AVAILABLE` — runtime feature probe.

Sister of `tac.mps_diagnostic.drift_predictor` (slot 9; the predictor whose
predicted gap reduction this helper realizes) + `tac.mps_diagnostic.kahan_conv2d`
+ `tac.mps_diagnostic.pinned_softmax` (the highest-EV engineering corrections).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

__all__ = [
    "STRICT_FP32_FLAGS",
    "MPS_BLAS_PREFERENCE_API_AVAILABLE",
    "enable_fp32_matmul_accumulation_strict",
    "restore_fp32_matmul_accumulation_state",
    "strict_fp32_matmul_accumulation",
]

# Canonical fp32-strict flag tuple. Each entry is (module_path, attr_name).
# When the runtime attribute is missing (e.g. cuda backend on CPU-only build),
# the helper records `None` in the prior-state map and skips the set.
STRICT_FP32_FLAGS: tuple[tuple[str, str], ...] = (
    ("torch.backends.cuda.matmul", "allow_tf32"),
    ("torch.backends.cudnn", "allow_tf32"),
)

# Runtime probe per Catalog #229 PV: torch.backends.mps.preferred_blas_library
# was CITED in the slot 9 dispatch prompt but does NOT exist in torch 2.11.0.
# Future torch versions may expose it; the probe lets downstream code adapt.
MPS_BLAS_PREFERENCE_API_AVAILABLE: bool = hasattr(
    torch.backends, "mps"
) and hasattr(torch.backends.mps, "preferred_blas_library")


def _resolve_module_path(module_path: str) -> object | None:
    """Resolve a dotted module path like ``torch.backends.cuda.matmul``.

    Returns ``None`` if any intermediate attribute is missing (e.g. the
    machine does not have the cuda backend compiled in).
    """
    parts = module_path.split(".")
    if not parts:
        return None
    try:
        obj: object = __import__(parts[0])
    except ImportError:
        return None
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def enable_fp32_matmul_accumulation_strict(
    *,
    include_mps_blas_preference: bool = True,
    mps_blas_preference_value: str = "MPS_ACCELERATE",
) -> dict[str, bool | str | None]:
    """Enable strict fp32 matmul accumulation across CUDA + MPS backends.

    Sets (when the attribute is present at runtime):
    - ``torch.backends.cuda.matmul.allow_tf32 = False`` — CUDA strict fp32
      matmul (eliminates TF32 19-bit mantissa accumulation).
    - ``torch.backends.cudnn.allow_tf32 = False`` — cuDNN convolution strict
      fp32 (sister of the matmul flag for convolution kernels).
    - ``torch.backends.mps.preferred_blas_library = MPS_ACCELERATE`` —
      Apple Accelerate fp32 BLAS path (when API is present; SKIPPED with
      `prior_state["torch.backends.mps.preferred_blas_library"] = None` when
      absent so callers can audit the structural skip).

    Per slot 9 formalization: predicted drift reduction 2x for the matmul
    accumulation component of the per-pair MPS-vs-CUDA gap.

    Returns a prior-state map of ``"<dotted_path>.<attr>" → prior_value`` so
    callers can restore via :func:`restore_fp32_matmul_accumulation_state`.
    Entries whose attribute did not exist at call time hold ``None``.
    """
    prior_state: dict[str, bool | str | None] = {}

    for module_path, attr_name in STRICT_FP32_FLAGS:
        module = _resolve_module_path(module_path)
        key = f"{module_path}.{attr_name}"
        if module is None or not hasattr(module, attr_name):
            prior_state[key] = None
            continue
        prior_state[key] = getattr(module, attr_name)
        try:
            setattr(module, attr_name, False)
        except (AttributeError, RuntimeError):
            # Setting failed (e.g. read-only attribute on this torch build);
            # record the prior value and continue.
            pass

    if include_mps_blas_preference:
        key = "torch.backends.mps.preferred_blas_library"
        if MPS_BLAS_PREFERENCE_API_AVAILABLE:
            module = _resolve_module_path("torch.backends.mps")
            if module is not None and hasattr(module, "preferred_blas_library"):
                prior_state[key] = getattr(module, "preferred_blas_library")
                try:
                    setattr(module, "preferred_blas_library", mps_blas_preference_value)
                except (AttributeError, RuntimeError):
                    pass
            else:  # pragma: no cover - defensive
                prior_state[key] = None
        else:
            # API does not exist on this torch version; record the structural
            # absence so callers can audit.
            prior_state[key] = None

    return prior_state


def restore_fp32_matmul_accumulation_state(
    prior_state: dict[str, bool | str | None],
) -> None:
    """Restore matmul accumulation state captured by
    :func:`enable_fp32_matmul_accumulation_strict`.

    Skips keys whose recorded prior value is ``None`` (the attribute did not
    exist when state was captured, so there is nothing to restore).
    """
    for key, prior_value in prior_state.items():
        if prior_value is None:
            continue
        # Split "torch.backends.cuda.matmul.allow_tf32" -> ("torch.backends.cuda.matmul", "allow_tf32")
        module_path, _, attr_name = key.rpartition(".")
        if not module_path or not attr_name:
            continue
        module = _resolve_module_path(module_path)
        if module is None or not hasattr(module, attr_name):
            continue
        try:
            setattr(module, attr_name, prior_value)
        except (AttributeError, RuntimeError):
            pass


@contextmanager
def strict_fp32_matmul_accumulation(
    *,
    include_mps_blas_preference: bool = True,
    mps_blas_preference_value: str = "MPS_ACCELERATE",
) -> Iterator[None]:
    """Context manager that enables strict fp32 matmul accumulation inside
    the block and restores prior state on exit.

    Example::

        with strict_fp32_matmul_accumulation():
            output = model(input)  # MPS + CUDA backends both strict-fp32

    Per CLAUDE.md "MPS auth eval is NOISE": this is a symmetric correction
    across both backends; MPS remains non-promotable for score-truth claims.
    """
    prior_state = enable_fp32_matmul_accumulation_strict(
        include_mps_blas_preference=include_mps_blas_preference,
        mps_blas_preference_value=mps_blas_preference_value,
    )
    try:
        yield
    finally:
        restore_fp32_matmul_accumulation_state(prior_state)
