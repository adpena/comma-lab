# SPDX-License-Identifier: MIT
"""Canonical ``torch.autocast`` wrapper for substrate trainers (Catalog #172).

Why this exists
---------------

REVIEW-OMNI Medium NV2 (Hotz) audit 2026-05-12 + optimization-opportunities
audit 2026-05-14 identified that 25/26 substrate trainers carry the file-level
waiver ``# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport``.
Backporting the canonical pattern at every trainer is brittle because each
trainer's hot-loop is slightly different. The fix is a SHARED helper:

    from tac.training_optimization import autocast_aware_forward
    ...
    with autocast_aware_forward(enabled=args.enable_autocast_fp16, device=device):
        decoded_rt = substrate.forward(latents)
        seg_dist, pose_dist = score_pair_components(...)

The helper is a thin context manager around ``torch.autocast`` honoring two
rules:

1. CUDA-only opt-in (CPU autocast is BF16-only on PyTorch and slower than
   fp32 on most CPUs; skip the wrap on non-CUDA devices to avoid surprise).
2. FP16 by default with BF16 toggle (BF16 has wider exponent range; if
   autocast triggers fp16 underflow in any substrate, switch to BF16 via
   ``dtype`` kwarg WITHOUT GradScaler).

GradScaler is intentionally NOT wrapped here — adding it inside this helper
would require coordinated optimizer-step calls that mutate the caller's
hot loop. The reference T1 Balle trainer manages its own GradScaler when
needed; substrate trainers using BF16 do not need one.

Speedup estimate
----------------

[literature-extrapolation, PyTorch 2.5 docs + HuggingFace BF16 benchmarks]:
1.5-2× scorer forward on Ampere / Hopper (A100, 4090, H100). The scorer
forward (PoseNet FastViT-T12 + SegNet EfficientNet-B2 UNet) dominates
per-step time for most substrates; the speedup compounds with O1
(GT-scorer-cache) which removes the GT scorer forward entirely.

Signal regression risk
----------------------

<1e-5 score drift expected; BF16 has no fp16 underflow concern. The
contest auth eval still runs at fp32 (Catalog #178 TF32 enforcement is
matmul-only, not autocast). Empirical verification of zero score
regression is operator-routable per the audit's probe-disambiguator hook.

Per CLAUDE.md "Apples-to-apples evidence discipline" no score claim is
attached to this module. The helper is a pure speedup primitive.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch


__all__ = [
    "AutocastConfig",
    "autocast_aware_forward",
    "resolve_autocast_dtype",
]


@dataclass(frozen=True)
class AutocastConfig:
    """Frozen configuration for ``autocast_aware_forward``.

    Allows substrate trainers to declare autocast intent once at trainer
    init and reuse across the hot loop without re-parsing flags.
    """

    enabled: bool
    dtype: torch.dtype
    device_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(
                f"AutocastConfig.dtype must be torch.dtype, got "
                f"{type(self.dtype).__name__}"
            )
        if self.device_type not in {"cuda", "cpu"}:
            raise ValueError(
                f"AutocastConfig.device_type must be 'cuda' or 'cpu', got "
                f"{self.device_type!r}; mps autocast is forbidden per CLAUDE.md "
                "'MPS auth eval is NOISE' non-negotiable"
            )


def resolve_autocast_dtype(name: str | torch.dtype) -> torch.dtype:
    """Map a string flag value (``"fp16"``/``"bf16"``) or dtype to ``torch.dtype``.

    Accepted string aliases:

    * ``"fp16"`` / ``"float16"`` / ``"half"`` -> ``torch.float16``
    * ``"bf16"`` / ``"bfloat16"`` -> ``torch.bfloat16``
    """
    if isinstance(name, torch.dtype):
        if name not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                f"autocast dtype {name!r} not supported; use float16 or bfloat16"
            )
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"resolve_autocast_dtype expects str or torch.dtype; got "
            f"{type(name).__name__}"
        )
    canon = name.strip().lower()
    if canon in {"fp16", "float16", "half"}:
        return torch.float16
    if canon in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(
        f"unknown autocast dtype alias {name!r}; "
        "use 'fp16'/'float16'/'half' or 'bf16'/'bfloat16'"
    )


def _device_type_for(device: torch.device | str | None) -> str:
    """Resolve the autocast ``device_type`` argument from a device-like input."""
    if device is None:
        # Default to cuda; the helper is opt-in via enabled=False otherwise.
        return "cuda"
    if isinstance(device, str):
        # Strings can be "cuda", "cuda:0", "cpu", "mps", etc.
        return device.split(":", 1)[0]
    if isinstance(device, torch.device):
        return device.type
    raise TypeError(
        f"device must be torch.device, str, or None; got {type(device).__name__}"
    )


@contextmanager
def autocast_aware_forward(
    *,
    enabled: bool = True,
    dtype: torch.dtype | str = torch.float16,
    device: torch.device | str | None = None,
) -> Generator[None, None, None]:
    """Canonical ``torch.autocast`` context for substrate trainer hot loops.

    Wrap forward passes; gradients/optimizer/EMA stay in fp32 outside.

    Args:
        enabled: When False, the context is a no-op (yield without
            entering autocast). This is the canonical wiring for the
            ``--enable-autocast-fp16`` argparse flag.
        dtype: Target autocast dtype. ``torch.float16`` (default) for
            FP16-with-no-GradScaler use, ``torch.bfloat16`` if a
            substrate trips fp16 underflow on accumulator paths. String
            aliases ``"fp16"`` / ``"bf16"`` accepted (via
            :func:`resolve_autocast_dtype`).
        device: Active training device. Used to resolve the
            ``device_type`` argument for ``torch.autocast``. None defaults
            to ``"cuda"`` (the canonical substrate-trainer device).

    Yields:
        None. Use inside a ``with ... :`` block around forward calls.

    Examples:
        Canonical substrate-trainer hot-loop pattern::

            for step, (latents, targets, idx) in enumerate(loader):
                with autocast_aware_forward(
                    enabled=args.enable_autocast_fp16,
                    dtype="bf16" if args.autocast_dtype == "bf16" else "fp16",
                    device=device,
                ):
                    decoded = substrate(latents)
                    decoded_rt = apply_eval_roundtrip_during_training(decoded, ...)
                    seg_dist, pose_dist = score_pair_components(...)
                # gradients in fp32 outside the context
                total_loss = (seg_dist * 100.0 + (10.0 * pose_dist).sqrt()).float()
                total_loss.backward()
                optimizer.step()

    Raises:
        ValueError: If the resolved dtype is not float16 or bfloat16.
        TypeError: If dtype or device is the wrong runtime type.
    """
    if not enabled:
        # No-op context: substrate trainers can unconditionally wrap their
        # hot loop and toggle via the --enable-autocast-fp16 flag.
        yield
        return

    resolved_dtype = resolve_autocast_dtype(dtype)
    device_type = _device_type_for(device)

    # CPU autocast supports bfloat16 only (PyTorch 2.5 documented behavior).
    # FP16 on CPU is uncommon and slower than fp32 on most CPUs; raise a
    # clear error rather than silently downgrade.
    if device_type == "cpu" and resolved_dtype == torch.float16:
        raise ValueError(
            "CPU autocast supports bfloat16 only; got dtype=torch.float16. "
            "Switch to dtype='bf16' for CPU paths, or set enabled=False "
            "when device_type is 'cpu' (CPU autocast is rarely a speedup "
            "and substrate-trainer canonical device is cuda)."
        )

    # mps autocast is structurally forbidden per CLAUDE.md "MPS auth eval is
    # NOISE" non-negotiable. _device_type_for will return 'mps' for an mps
    # device; refuse to wrap.
    if device_type == "mps":
        raise ValueError(
            "autocast on device='mps' is forbidden per CLAUDE.md 'MPS auth "
            "eval is NOISE' non-negotiable. Substrate trainers must use "
            "cuda for full training; cpu autocast (bf16) is permitted for "
            "smoke-only paths."
        )

    if device_type not in {"cuda", "cpu"}:
        raise ValueError(
            f"autocast device_type must be 'cuda' or 'cpu'; got "
            f"{device_type!r}"
        )

    with torch.autocast(device_type=device_type, dtype=resolved_dtype):
        yield
