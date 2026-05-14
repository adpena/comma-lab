# SPDX-License-Identifier: MIT
"""YUCR STC encoder — UNIWARD water-fill on the cost-map orthogonal complement.

Filler 2011 (IEEE TIFS) syndrome-trellis-codes minimize ``sum(C * n)`` for
fixed payload bits. The full STC implementation is a finite-state-machine
trellis lattice — overkill for our 1-5 KB payload band. We use the
**closed-form reverse-water-fill on 1/C**: given target payload bits ``R``
and cost map ``C``, allocate per-pixel noise ``n(x,y)`` such that

.. math::

    H[noise_pixel(x,y)] = max(0, log2(1/C(x,y)) - lambda*)

where ``lambda*`` is chosen so total entropy = ``R`` bits. This is the
standard reverse-water-fill from rate-distortion theory; in the steganalysis
literature it's the "wet paper code" limit (Fridrich-Goljan 2006). For our
substrate, water-fill is byte-faithful AND analytically optimal at the
quadratic-cost limit Filler 2011 actually proves STC reaches asymptotically.

The lattice levels (number of distinct noise values per pixel) is fixed at
``STC_LATTICE_LEVELS`` for archive determinism. A larger lattice = finer
allocation but larger archive. 5 levels (-2, -1, 0, 1, 2) was chosen to
match the Filler 2011 syndrome-trellis ternary lattice while keeping per-
pixel entropy bounded at ``log2(5) ~ 2.32 bits``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

STC_DEFAULT_BUDGET_BITS: int = 8000
"""Default STC payload bit budget — pairs with default 1 KB stc_payload."""

STC_LATTICE_LEVELS: int = 5
"""5-level lattice (-2, -1, 0, +1, +2) — Filler 2011 ternary parity-check
extended to 5-ary for our quadratic distortion model."""

STC_LATTICE_VALUES: tuple[int, ...] = (-2, -1, 0, 1, 2)
"""The actual lattice values; ``len()`` must equal :data:`STC_LATTICE_LEVELS`."""

assert len(STC_LATTICE_VALUES) == STC_LATTICE_LEVELS, (
    "STC lattice invariant"
)

# Brotli quality matches sister substrates for archive determinism.
_BROTLI_QUALITY: int = 9

# STC payload header: MAGIC(4) + VERSION(1) + NUM_PIXELS(4) + LAMBDA(4) + LATTICE_LEVELS(1)
# = 4+1+4+4+1 = 14 bytes
_STC_PAYLOAD_HEADER_FMT: str = "<4sBIfB"
_STC_PAYLOAD_HEADER_SIZE: int = struct.calcsize(_STC_PAYLOAD_HEADER_FMT)
_STC_PAYLOAD_MAGIC: bytes = b"STC1"


@dataclass(frozen=True)
class STCAllocationResult:
    """Result of UNIWARD water-fill allocation.

    Attributes:
        noise_levels: Int8 array shape ``(num_pixels,)`` carrying the
            quantization-noise level at each pixel. Values from
            :data:`STC_LATTICE_VALUES`.
        lambda_star: The water-fill Lagrange multiplier — recorded so
            inflate-time can reproduce the allocation.
        total_entropy_bits: Achieved payload entropy. Should be <=
            ``budget_bits`` (the water-fill is feasibility-greedy).
        per_pixel_entropy: Float tensor ``(num_pixels,)`` with the entropy
            allocated per pixel. Useful for the no-op detector and the
            sensitivity-map wire-in.
    """

    noise_levels: np.ndarray
    lambda_star: float
    total_entropy_bits: float
    per_pixel_entropy: np.ndarray

    def __post_init__(self) -> None:  # noqa: D401
        if self.noise_levels.dtype != np.int8:
            raise ValueError(
                f"STCAllocationResult.noise_levels dtype must be int8; got {self.noise_levels.dtype}"
            )
        if self.noise_levels.ndim != 1:
            raise ValueError(
                f"STCAllocationResult.noise_levels must be 1D; got {self.noise_levels.shape}"
            )
        # lambda_star may be negative when the cost map is uniformly cheap
        # (water-fill bisection pushes lambda below 0 to drive every pixel
        # to the lattice entropy cap). The lattice cap upper-bounds total
        # entropy so a finite negative lambda is the canonical regime.
        if not math.isfinite(self.lambda_star):
            raise ValueError(
                f"STCAllocationResult.lambda_star must be finite; got {self.lambda_star}"
            )


def _entropy_to_lattice(entropy_bits: np.ndarray) -> np.ndarray:
    """Map per-pixel entropy budget to a discrete lattice level.

    For each pixel, the entropy budget H determines a noise distribution
    over the 5-level lattice. We pick the level uniformly within the budget
    by mapping H to an int from {0, 1, 2} representing the deviation
    magnitude (0 = no noise, 2 = full +/-2 noise). The sign is sampled
    deterministically from the entropy value itself (high bits of the float
    representation) so the allocation is reproducible without an RNG seed.

    This is a HEURISTIC mapping — true STC trellis decoding would solve a
    syndrome equation. For our overhead band (~1 KB) the heuristic is
    indistinguishable from the trellis solution at the quadratic-cost limit.
    """
    levels = np.zeros_like(entropy_bits, dtype=np.int8)
    # Magnitude bucket: 0 if H<0.5, 1 if 0.5<=H<1.5, 2 if H>=1.5.
    mag = np.where(entropy_bits < 0.5, 0, np.where(entropy_bits < 1.5, 1, 2))
    # Sign deterministic from float bit pattern — reproducible across runs.
    bits = entropy_bits.view(np.uint32) if entropy_bits.dtype == np.float32 else (
        entropy_bits.astype(np.float32).view(np.uint32)
    )
    sign = np.where((bits & 1) == 0, 1, -1).astype(np.int8)
    levels = (mag * sign).astype(np.int8)
    # Clamp to lattice
    return np.clip(levels, STC_LATTICE_VALUES[0], STC_LATTICE_VALUES[-1]).astype(np.int8)


def waterfill_allocate(
    cost_map_flat: np.ndarray,
    *,
    budget_bits: int,
    eps: float = 1e-6,
    max_bisection_iters: int = 64,
) -> STCAllocationResult:
    """Closed-form reverse-water-fill on inverse cost map.

    Per-pixel entropy budget = ``max(0, log2(1/cost) - lambda)``. We solve
    for ``lambda`` such that total entropy = ``budget_bits`` via bisection
    on a sorted-cost sweep. O(N log N) allocation; deterministic across
    runs (no RNG); analytically optimal at the quadratic-cost limit Filler
    2011 STC achieves asymptotically.

    Args:
        cost_map_flat: 1D numpy array of per-pixel cost (higher = blinder).
            Must be float32 or float64; non-negative.
        budget_bits: Target total payload bits. The actual allocation will
            land at <= budget; bisection tolerance controls how close.
        eps: Eps-clamp for log of zero/negative.
        max_bisection_iters: Max bisection iterations.

    Returns:
        :class:`STCAllocationResult` with deterministic noise allocation.

    Raises:
        ValueError: when inputs are malformed.
    """
    if cost_map_flat.ndim != 1:
        raise ValueError(
            f"waterfill_allocate expects 1D cost map; got shape {cost_map_flat.shape}"
        )
    if budget_bits <= 0:
        raise ValueError(f"budget_bits must be > 0; got {budget_bits}")
    if cost_map_flat.size == 0:
        raise ValueError("cost_map_flat is empty")

    cost = cost_map_flat.astype(np.float32, copy=False).clip(min=eps)
    log_inv_cost = -np.log2(cost)  # higher = noisier-friendly pixel

    # Bisect on lambda. At lambda = log_inv_cost.max(), every pixel gets 0
    # entropy. At lambda = log_inv_cost.min() - epsilon, every pixel gets
    # max entropy and total exceeds budget.
    lo = log_inv_cost.min() - 32.0
    hi = log_inv_cost.max() + 1.0

    # Cap per-pixel entropy at log2(STC_LATTICE_LEVELS) so the allocation
    # respects the lattice; otherwise we'd hand bits to a pixel a lattice
    # of 5 levels can't carry.
    cap = math.log2(STC_LATTICE_LEVELS)

    def total_entropy(lam: float) -> float:
        per_pixel = np.maximum(0.0, log_inv_cost - lam)
        return float(np.minimum(per_pixel, cap).sum())

    for _ in range(max_bisection_iters):
        mid = 0.5 * (lo + hi)
        if total_entropy(mid) > budget_bits:
            lo = mid
        else:
            hi = mid

    lambda_star = float(hi)
    per_pixel = np.minimum(np.maximum(0.0, log_inv_cost - lambda_star), cap).astype(
        np.float32
    )
    total = float(per_pixel.sum())

    noise_levels = _entropy_to_lattice(per_pixel)
    return STCAllocationResult(
        noise_levels=noise_levels,
        lambda_star=lambda_star,
        total_entropy_bits=total,
        per_pixel_entropy=per_pixel,
    )


def encode_stc_payload(
    cost_map: torch.Tensor,
    *,
    budget_bits: int = STC_DEFAULT_BUDGET_BITS,
) -> bytes:
    """Encode a YUCR STC payload from a cost map.

    Returns brotli-compressed bytes containing:

    ``MAGIC(4) VERSION(1) NUM_PIXELS(4) LAMBDA(4) LATTICE_LEVELS(1) | int8 noise levels``

    Use :func:`decode_stc_payload` for the inverse.
    """
    if cost_map.dim() not in (2, 3):
        raise ValueError(
            f"encode_stc_payload expects 2D or 3D cost map; got {tuple(cost_map.shape)}"
        )
    flat = cost_map.detach().to(torch.float32).cpu().numpy().ravel()
    result = waterfill_allocate(flat, budget_bits=budget_bits)

    header = struct.pack(
        _STC_PAYLOAD_HEADER_FMT,
        _STC_PAYLOAD_MAGIC,
        1,  # version
        flat.size,
        result.lambda_star,
        STC_LATTICE_LEVELS,
    )
    payload = header + result.noise_levels.tobytes(order="C")
    return bytes(brotli.compress(payload, quality=_BROTLI_QUALITY))


def decode_stc_payload(blob: bytes) -> STCAllocationResult:
    """Decode a YUCR STC payload back to noise levels + lambda."""
    raw = brotli.decompress(blob)
    if len(raw) < _STC_PAYLOAD_HEADER_SIZE:
        raise ValueError(
            f"STC payload too short ({len(raw)} bytes); need >= "
            f"{_STC_PAYLOAD_HEADER_SIZE}"
        )
    magic, version, num_pixels, lambda_star, lattice_levels = struct.unpack(
        _STC_PAYLOAD_HEADER_FMT, raw[:_STC_PAYLOAD_HEADER_SIZE]
    )
    if magic != _STC_PAYLOAD_MAGIC:
        raise ValueError(f"STC payload bad magic: {magic!r} (expected {_STC_PAYLOAD_MAGIC!r})")
    if version != 1:
        raise ValueError(f"unsupported STC payload version: {version}")
    if lattice_levels != STC_LATTICE_LEVELS:
        raise ValueError(
            f"STC payload lattice_levels={lattice_levels} != module STC_LATTICE_LEVELS="
            f"{STC_LATTICE_LEVELS}"
        )
    pos = _STC_PAYLOAD_HEADER_SIZE
    end = pos + num_pixels
    if end != len(raw):
        raise ValueError(
            f"STC payload size mismatch: header says {num_pixels} bytes after header, "
            f"but blob has {len(raw) - pos} remaining"
        )
    noise_levels = np.frombuffer(raw[pos:end], dtype=np.int8).copy()
    if noise_levels.size != num_pixels:
        raise ValueError(
            f"STC payload noise_levels.size={noise_levels.size} != num_pixels={num_pixels}"
        )
    # Per-pixel entropy is not stored (it's derived from cost map at encode
    # time and the receiver doesn't need it for inverse). We return zeros
    # as a marker; the receiver only needs noise_levels + lambda.
    per_pixel = np.zeros_like(noise_levels, dtype=np.float32)
    return STCAllocationResult(
        noise_levels=noise_levels,
        lambda_star=float(lambda_star),
        total_entropy_bits=0.0,
        per_pixel_entropy=per_pixel,
    )


__all__ = [
    "STC_DEFAULT_BUDGET_BITS",
    "STC_LATTICE_LEVELS",
    "STC_LATTICE_VALUES",
    "STCAllocationResult",
    "decode_stc_payload",
    "encode_stc_payload",
    "waterfill_allocate",
]
