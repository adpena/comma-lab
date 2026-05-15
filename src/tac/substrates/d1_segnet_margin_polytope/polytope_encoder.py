# SPDX-License-Identifier: MIT
"""D1 polytope encoder — Newton-step-distance reverse-water-fill on the margin map.

Per the deep-math memo §3.6, the safe-perturbation polytope at pixel
``(x, y)`` is

.. math::

    \\Pi_1(x, y) = \\{ \\delta \\in R^3 : \\|J_\\text{seg}(x, y) \\delta\\|_\\infty < m(x, y) \\}

where ``m(x, y)`` is the SegNet logit margin and ``J_seg(x, y)`` is the
per-pixel Jacobian of the SegNet logit w.r.t. the input RGB pixel. For
the constructive encoder, we approximate ``J_seg`` by its operator-norm
upper bound ``L`` so the per-pixel **safe-noise budget** becomes

.. math::

    B_\\text{safe}(x, y) = m(x, y) / L

and the constructive allocator places noise inside the polytope interior
via **reverse-water-fill on B_safe**: given a target payload bit budget
``R`` and the budget map ``B_safe``, allocate per-pixel noise ``n(x, y)``
such that

.. math::

    H[noise(x, y)] = \\min(\\log_2(L_\\text{lattice}), \\max(0, \\log_2(B_\\text{safe}(x, y)) - \\lambda^*))

where ``lambda*`` is chosen so total entropy = ``R`` bits. This is the
canonical reverse-water-fill from rate-distortion theory; in our setting
``B_safe`` plays the role of ``1/cost`` from the UNIWARD water-fill but
with a HARD upper-cap at the lattice entropy (perturbations cannot
exceed ``B_safe`` without flipping the SegNet argmax).

The lattice levels (number of distinct noise values per pixel) is fixed
at :data:`POLYTOPE_LATTICE_LEVELS` for archive determinism. 5 levels
(-2, -1, 0, 1, 2) match the YUCR STC lattice for composability + per-pixel
entropy bounded at ``log2(5) ~ 2.32 bits``.

The encoder is BYTE-DETERMINISTIC across runs (no RNG); sign is
derived from the per-pixel entropy float bit pattern so the allocation
reproduces from the cost map alone.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

POLYTOPE_DEFAULT_BUDGET_BITS: int = 8000
"""Default polytope payload bit budget — pairs with default 1 KB sidecar."""

POLYTOPE_LATTICE_LEVELS: int = 5
"""5-level lattice (-2, -1, 0, +1, +2) — matches YUCR STC for composability."""

POLYTOPE_LATTICE_VALUES: tuple[int, ...] = (-2, -1, 0, 1, 2)
"""Lattice values; ``len()`` must equal :data:`POLYTOPE_LATTICE_LEVELS`."""

assert len(POLYTOPE_LATTICE_VALUES) == POLYTOPE_LATTICE_LEVELS, (
    "POLYTOPE lattice invariant"
)

# Brotli quality matches sister substrates for archive determinism.
_BROTLI_QUALITY: int = 9

# Polytope payload header:
#   MAGIC(4) + VERSION(1) + NUM_PIXELS(4) + LAMBDA(4) + LATTICE_LEVELS(1)
#   + JACOBIAN_LIPSCHITZ(4) = 4+1+4+4+1+4 = 18 bytes
_POLYTOPE_PAYLOAD_HEADER_FMT: str = "<4sBIfBf"
_POLYTOPE_PAYLOAD_HEADER_SIZE: int = struct.calcsize(_POLYTOPE_PAYLOAD_HEADER_FMT)
_POLYTOPE_PAYLOAD_MAGIC: bytes = b"PLY1"


@dataclass(frozen=True)
class PolytopeAllocationResult:
    """Result of polytope reverse-water-fill allocation.

    Attributes:
        noise_levels: Int8 array shape ``(num_pixels,)`` carrying the
            quantization-noise level at each pixel. Values from
            :data:`POLYTOPE_LATTICE_VALUES`.
        lambda_star: Reverse-water-fill Lagrange multiplier — recorded so
            inflate-time can reproduce the allocation.
        jacobian_lipschitz: The operator-norm upper bound ``L`` used to
            convert margin -> safe budget (``B_safe = m / L``). Recorded
            so the receiver inverts deterministically.
        total_entropy_bits: Achieved payload entropy. Should be <=
            ``budget_bits`` (feasibility-greedy).
        per_pixel_entropy: Float tensor ``(num_pixels,)`` with the entropy
            allocated per pixel. Useful for the no-op detector + the
            sensitivity-map wire-in.
        polytope_interior_fraction: Fraction of pixels with ``B_safe > 0``
            (i.e. with non-singleton polytopes). Useful as a structural
            health signal — typical comma2k19 frames cluster around 0.8.
    """

    noise_levels: np.ndarray
    lambda_star: float
    jacobian_lipschitz: float
    total_entropy_bits: float
    per_pixel_entropy: np.ndarray
    polytope_interior_fraction: float

    def __post_init__(self) -> None:
        if self.noise_levels.dtype != np.int8:
            raise ValueError(
                f"PolytopeAllocationResult.noise_levels dtype must be int8; "
                f"got {self.noise_levels.dtype}"
            )
        if self.noise_levels.ndim != 1:
            raise ValueError(
                "PolytopeAllocationResult.noise_levels must be 1D; got "
                f"{self.noise_levels.shape}"
            )
        if not math.isfinite(self.lambda_star):
            raise ValueError(
                "PolytopeAllocationResult.lambda_star must be finite; "
                f"got {self.lambda_star}"
            )
        if not math.isfinite(self.jacobian_lipschitz) or self.jacobian_lipschitz <= 0:
            raise ValueError(
                "PolytopeAllocationResult.jacobian_lipschitz must be > 0; "
                f"got {self.jacobian_lipschitz}"
            )
        if not (0.0 <= self.polytope_interior_fraction <= 1.0):
            raise ValueError(
                "polytope_interior_fraction must be in [0, 1]; got "
                f"{self.polytope_interior_fraction}"
            )


def compute_safe_perturbation_budget(
    margin_map: torch.Tensor,
    *,
    jacobian_lipschitz: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-pixel Newton-step distance to nearest SegNet decision boundary.

    For pixel ``(x, y)`` with logit margin ``m(x, y)`` and SegNet Jacobian
    upper-bounded by operator-norm ``L``, the safe-perturbation budget is

    .. math::

        B_\\text{safe}(x, y) = m(x, y) / L

    By the structural geometry of argmax + the Lipschitz bound on the
    SegNet logit-vs-pixel map, perturbing the input RGB pixel by
    ``|delta| < B_safe(x, y)`` cannot flip the SegNet argmax.

    Args:
        margin_map: ``(H, W)`` or ``(B, H, W)`` float tensor of logit
            margins. MUST be non-negative.
        jacobian_lipschitz: ``L`` — the SegNet logit-vs-RGB-pixel Jacobian
            operator-norm upper bound. A typical scaled-RGB SegNet hits
            ``L ~ 10-50``; the value should be calibrated offline on a
            held-out frame batch and recorded in archive metadata.
        eps: Eps-clamp for div-by-zero.

    Returns:
        ``B_safe`` tensor with same shape as ``margin_map``.

    Raises:
        ValueError: when ``jacobian_lipschitz <= 0`` or ``margin_map``
            has negative values.
    """
    if jacobian_lipschitz <= 0:
        raise ValueError(
            f"jacobian_lipschitz must be > 0; got {jacobian_lipschitz}. "
            "L is the SegNet Jacobian operator-norm upper bound; a "
            "non-positive value would make the safe budget undefined."
        )
    if (margin_map.detach() < 0).any():
        raise ValueError(
            "margin_map has negative values — margin = top1 - top2 must "
            "be >= 0 by construction."
        )
    return (margin_map.clamp_min(0.0) / max(jacobian_lipschitz, eps)).contiguous()


def allocate_noise_within_polytope(
    safe_budget_flat: np.ndarray,
    *,
    budget_bits: int,
    jacobian_lipschitz: float,
    eps: float = 1e-6,
    max_bisection_iters: int = 64,
) -> PolytopeAllocationResult:
    """Reverse-water-fill on per-pixel safe-perturbation budget.

    Per-pixel entropy budget = ``max(0, log2(B_safe(x, y)) - lambda)``,
    capped at ``log2(POLYTOPE_LATTICE_LEVELS)``. We solve for ``lambda``
    such that total entropy = ``budget_bits`` via bisection. O(N log N);
    deterministic across runs (no RNG); analytically optimal at the
    quadratic-cost limit.

    Args:
        safe_budget_flat: 1D numpy array of per-pixel ``B_safe`` (Newton-step
            distance). Must be non-negative.
        budget_bits: Target total payload bits.
        jacobian_lipschitz: ``L`` recorded so the receiver inverts
            deterministically; this is the same value used to compute
            ``safe_budget_flat = margin / L``.
        eps: Eps-clamp.
        max_bisection_iters: Bisection iteration cap.

    Returns:
        :class:`PolytopeAllocationResult` with deterministic allocation.

    Raises:
        ValueError: when inputs are malformed.
    """
    if safe_budget_flat.ndim != 1:
        raise ValueError(
            "allocate_noise_within_polytope expects 1D safe_budget; got "
            f"shape {safe_budget_flat.shape}"
        )
    if budget_bits <= 0:
        raise ValueError(f"budget_bits must be > 0; got {budget_bits}")
    if safe_budget_flat.size == 0:
        raise ValueError("safe_budget_flat is empty")
    if jacobian_lipschitz <= 0:
        raise ValueError(
            f"jacobian_lipschitz must be > 0; got {jacobian_lipschitz}"
        )
    if (safe_budget_flat < 0).any():
        raise ValueError(
            "safe_budget_flat has negative values — B_safe = m / L must "
            "be >= 0 by construction."
        )

    safe_budget = safe_budget_flat.astype(np.float32, copy=False)

    # Polytope-interior fraction = pixels with strictly-positive safe budget.
    interior_mask = safe_budget > eps
    interior_fraction = float(interior_mask.mean()) if safe_budget.size > 0 else 0.0

    # log2(B_safe) is finite only for interior pixels. For boundary pixels
    # (B_safe = 0) we want zero entropy contribution; we set log_safe to
    # -inf there so the max(0, ...) clamp drives them to zero regardless
    # of lambda.
    safe_clipped = np.maximum(safe_budget, eps)
    log_safe = np.log2(safe_clipped)
    # Force boundary pixels to a very negative log_safe so they never get
    # entropy regardless of lambda.
    log_safe = np.where(interior_mask, log_safe, -1e9).astype(np.float32)

    cap = math.log2(POLYTOPE_LATTICE_LEVELS)

    # Bisect on lambda. At lambda very large -> every pixel gets 0 entropy.
    # At lambda very negative -> every interior pixel gets cap.
    finite_log = log_safe[np.isfinite(log_safe) & (log_safe > -1e8)]
    if finite_log.size == 0:
        # Pathological case: every pixel is boundary. No entropy allocatable.
        zero_levels = np.zeros(safe_budget.size, dtype=np.int8)
        per_pixel_zero = np.zeros(safe_budget.size, dtype=np.float32)
        return PolytopeAllocationResult(
            noise_levels=zero_levels,
            lambda_star=0.0,
            jacobian_lipschitz=float(jacobian_lipschitz),
            total_entropy_bits=0.0,
            per_pixel_entropy=per_pixel_zero,
            polytope_interior_fraction=interior_fraction,
        )

    lo = float(finite_log.min()) - 32.0
    hi = float(finite_log.max()) + 1.0

    def total_entropy(lam: float) -> float:
        per_pixel = np.maximum(0.0, log_safe - lam)
        return float(np.minimum(per_pixel, cap).sum())

    for _ in range(max_bisection_iters):
        mid = 0.5 * (lo + hi)
        if total_entropy(mid) > budget_bits:
            lo = mid
        else:
            hi = mid

    lambda_star = float(hi)
    per_pixel = np.minimum(
        np.maximum(0.0, log_safe - lambda_star), cap
    ).astype(np.float32)
    total = float(per_pixel.sum())

    noise_levels = _entropy_to_lattice(per_pixel)
    noise_levels = _clamp_lattice_to_safe_budget(noise_levels, safe_budget)
    return PolytopeAllocationResult(
        noise_levels=noise_levels,
        lambda_star=lambda_star,
        jacobian_lipschitz=float(jacobian_lipschitz),
        total_entropy_bits=total,
        per_pixel_entropy=per_pixel,
        polytope_interior_fraction=interior_fraction,
    )


def _entropy_to_lattice(entropy_bits: np.ndarray) -> np.ndarray:
    """Map per-pixel entropy budget to a 5-level lattice value.

    Magnitude bucket: 0 if H < 0.5, 1 if 0.5 <= H < 1.5, 2 if H >= 1.5.
    Sign deterministic from float bit pattern — reproducible without RNG.
    Final value clamped to ``[POLYTOPE_LATTICE_VALUES[0],
    POLYTOPE_LATTICE_VALUES[-1]]``.
    """
    mag = np.where(entropy_bits < 0.5, 0, np.where(entropy_bits < 1.5, 1, 2))
    bits = entropy_bits.view(np.uint32) if entropy_bits.dtype == np.float32 else (
        entropy_bits.astype(np.float32).view(np.uint32)
    )
    sign = np.where((bits & 1) == 0, 1, -1).astype(np.int8)
    levels = (mag * sign).astype(np.int8)
    return np.clip(
        levels, POLYTOPE_LATTICE_VALUES[0], POLYTOPE_LATTICE_VALUES[-1]
    ).astype(np.int8)


def _clamp_lattice_to_safe_budget(
    noise_levels: np.ndarray,
    safe_budget: np.ndarray,
) -> np.ndarray:
    """Clamp lattice magnitude to the integer safe budget per pixel."""
    if noise_levels.shape != safe_budget.shape:
        raise ValueError(
            "noise_levels/safe_budget shape mismatch: "
            f"{noise_levels.shape} != {safe_budget.shape}"
        )
    max_abs = np.floor(safe_budget.astype(np.float32) + 1e-6)
    max_abs = np.clip(max_abs, 0, max(abs(v) for v in POLYTOPE_LATTICE_VALUES))
    signed = noise_levels.astype(np.int16, copy=False)
    clipped_abs = np.minimum(np.abs(signed), max_abs.astype(np.int16))
    return (np.sign(signed).astype(np.int16) * clipped_abs).astype(np.int8)


def encode_polytope_payload(
    margin_map: torch.Tensor,
    *,
    jacobian_lipschitz: float,
    budget_bits: int = POLYTOPE_DEFAULT_BUDGET_BITS,
) -> bytes:
    """Encode a D1 polytope payload from a margin map + Jacobian Lipschitz.

    Returns brotli-compressed bytes containing:

    ``MAGIC(4) VERSION(1) NUM_PIXELS(4) LAMBDA(4) LATTICE_LEVELS(1) JACOBIAN_LIPSCHITZ(4)
    | int8 noise levels``

    Use :func:`decode_polytope_payload` for the inverse.

    Args:
        margin_map: ``(H, W)`` or ``(B, H, W)`` float tensor of logit
            margins.
        jacobian_lipschitz: ``L`` — SegNet Jacobian operator-norm upper
            bound used to convert margin -> safe budget.
        budget_bits: Bit budget for the polytope payload.

    Raises:
        ValueError: when inputs are malformed.
    """
    if margin_map.dim() not in (2, 3):
        raise ValueError(
            "encode_polytope_payload expects 2D or 3D margin map; got "
            f"{tuple(margin_map.shape)}"
        )
    safe_budget = compute_safe_perturbation_budget(
        margin_map, jacobian_lipschitz=jacobian_lipschitz
    )
    flat = safe_budget.detach().to(torch.float32).cpu().numpy().ravel()
    result = allocate_noise_within_polytope(
        flat,
        budget_bits=budget_bits,
        jacobian_lipschitz=jacobian_lipschitz,
    )

    header = struct.pack(
        _POLYTOPE_PAYLOAD_HEADER_FMT,
        _POLYTOPE_PAYLOAD_MAGIC,
        1,  # version
        flat.size,
        result.lambda_star,
        POLYTOPE_LATTICE_LEVELS,
        float(jacobian_lipschitz),
    )
    payload = header + result.noise_levels.tobytes(order="C")
    return bytes(brotli.compress(payload, quality=_BROTLI_QUALITY))


def decode_polytope_payload(blob: bytes) -> PolytopeAllocationResult:
    """Decode a D1 polytope payload back to noise levels + lambda + L."""
    raw = brotli.decompress(blob)
    if len(raw) < _POLYTOPE_PAYLOAD_HEADER_SIZE:
        raise ValueError(
            f"polytope payload too short ({len(raw)} bytes); need >= "
            f"{_POLYTOPE_PAYLOAD_HEADER_SIZE}"
        )
    (
        magic,
        version,
        num_pixels,
        lambda_star,
        lattice_levels,
        jacobian_lipschitz,
    ) = struct.unpack(
        _POLYTOPE_PAYLOAD_HEADER_FMT, raw[:_POLYTOPE_PAYLOAD_HEADER_SIZE]
    )
    if magic != _POLYTOPE_PAYLOAD_MAGIC:
        raise ValueError(
            f"polytope payload bad magic: {magic!r} "
            f"(expected {_POLYTOPE_PAYLOAD_MAGIC!r})"
        )
    if version != 1:
        raise ValueError(f"unsupported polytope payload version: {version}")
    if lattice_levels != POLYTOPE_LATTICE_LEVELS:
        raise ValueError(
            f"polytope payload lattice_levels={lattice_levels} != module "
            f"POLYTOPE_LATTICE_LEVELS={POLYTOPE_LATTICE_LEVELS}"
        )
    if jacobian_lipschitz <= 0:
        raise ValueError(
            f"polytope payload jacobian_lipschitz={jacobian_lipschitz} must be > 0"
        )
    pos = _POLYTOPE_PAYLOAD_HEADER_SIZE
    end = pos + num_pixels
    if end != len(raw):
        raise ValueError(
            f"polytope payload size mismatch: header says {num_pixels} "
            f"bytes after header, but blob has {len(raw) - pos} remaining"
        )
    noise_levels = np.frombuffer(raw[pos:end], dtype=np.int8).copy()
    if noise_levels.size != num_pixels:
        raise ValueError(
            f"polytope payload noise_levels.size={noise_levels.size} != "
            f"num_pixels={num_pixels}"
        )
    # per_pixel_entropy is not stored (derived from margin map at encode
    # time and the receiver doesn't need it for inverse). Return zeros as
    # a marker; the receiver only needs noise_levels + lambda + L.
    per_pixel = np.zeros_like(noise_levels, dtype=np.float32)
    # polytope_interior_fraction is similarly not in the wire format; we
    # set it to NaN-conservative value (recovered from noise count if the
    # receiver needs to estimate it).
    interior_fraction = float(
        (noise_levels != 0).mean()
    ) if noise_levels.size > 0 else 0.0
    return PolytopeAllocationResult(
        noise_levels=noise_levels,
        lambda_star=float(lambda_star),
        jacobian_lipschitz=float(jacobian_lipschitz),
        total_entropy_bits=0.0,
        per_pixel_entropy=per_pixel,
        polytope_interior_fraction=interior_fraction,
    )


__all__ = [
    "POLYTOPE_DEFAULT_BUDGET_BITS",
    "POLYTOPE_LATTICE_LEVELS",
    "POLYTOPE_LATTICE_VALUES",
    "PolytopeAllocationResult",
    "allocate_noise_within_polytope",
    "compute_safe_perturbation_budget",
    "decode_polytope_payload",
    "encode_polytope_payload",
]
