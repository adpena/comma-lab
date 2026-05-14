# SPDX-License-Identifier: MIT
"""Bregman-Divergence Mixing — composition/stacking primitive.

Bregman divergences (Bregman 1967) generalise squared-Euclidean distance to
any strictly convex generator ``φ : Ω → R``:

    D_φ(x, y) = φ(x) - φ(y) - ⟨∇φ(y), x - y⟩                            (B.1)

Different generators give different divergences:

* ``φ(x) = (1/2)||x||²`` →  squared Euclidean.
* ``φ(x) = Σ_i x_i log x_i``  → KL divergence (on the simplex).
* ``φ(x) = -Σ_i log x_i``  → Itakura-Saito (audio amplitude).
* ``φ(x) = (1/2) x^T Λ x``  → Mahalanobis (per-tensor metric).

The *Bregman centroid* of a set ``{x_k}`` with weights ``λ_k`` is the unique
minimiser of weighted divergence

    x* = arg min_x Σ_k λ_k D_φ(x_k, x)                                   (B.2)

For squared-Euclidean this is the weighted mean. For KL it is the geometric
mean (after normalising). For Itakura-Saito it is the harmonic mean.

Source memo:
``.omx/research/ancient_elder_polymath_research_20260513.md`` §7.2 IDEA OP-2
("Bregman-divergence mirror descent for entropy-regularized training") +
``.omx/research/ancient_elder_era_7_convex_optimization_20260513.md``.

The Bregman-mixing primitive composes K models (or K parameter slices) via
the Bregman centroid:

    W_mix = arg min_W Σ_k λ_k D_φ(W, W_k)                                (B.3)

For exponential-family parametrisations this has a closed form. The module
exposes :class:`BregmanMixer` that selects the appropriate centroid formula
from a small canonical generator enumeration and validates inputs at the
boundary.

Cross-references
----------------
- Sinkhorn-OT mixing (sister, ``tac.composition.sinkhorn_ot_mixing``).
- Wasserstein barycenter (a special Bregman with ``φ = (1/2)||·||²``,
  ``tac.composition.wbce_mera``).
- Stack-of-stacks integration: ``tac.composition.stack_of_stacks.compose``.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
This module produces a forward-pass module + serialisable spec; it does not
modify archive bytes by itself. Substrate integration must register a
parser-section manifest entry per CLAUDE.md Catalog #124 before any
``score_claim=True``. Until paired ``[contest-CUDA]`` + ``[contest-CPU]``
anchors land on a Bregman-equipped substrate, every result is
``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: the mixer is differentiable when the generator is smooth on
   the supplied support; trainer drives apply_eval_roundtrip + scorer
   gradient.
2. Export-first: :meth:`BregmanMixer.serialize_state` is deterministic.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 350 LOC.
8-13. Standard substrate concerns; not violated.
"""

from __future__ import annotations

import enum
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tac.composition.frontier_primitives import (
    BregmanDivergence as CanonicalBregmanDivergence,
)
from tac.composition.frontier_primitives import (
    bregman_barycenter as _canonical_bregman_barycenter,
)

BREGMAN_MAGIC = b"BRG1"
BREGMAN_SCHEMA_VERSION = 1

DEFAULT_EPS = 1e-8


class BregmanError(ValueError):
    """Raised when a BregmanMixer spec or input is invalid."""


class BregmanGenerator(enum.StrEnum):
    """Canonical Bregman generators supported by this primitive.

    The string values are stable wire-format tokens; do not rename.
    """

    SQUARED_EUCLIDEAN = "squared_euclidean"
    KL = "kl_divergence"
    ITAKURA_SAITO = "itakura_saito"
    MAHALANOBIS = "mahalanobis"


@dataclass(frozen=True)
class BregmanMixerSpec:
    """Specification for a Bregman-mixing composer.

    Args:
        generator: which canonical Bregman generator to use.
        weights: per-model weights ``λ_k`` (must be non-negative; need not
            sum to 1 — normalised internally). ``None`` → uniform.
        eps: numerical floor for KL/Itakura-Saito (avoid log(0) / 1/0).
        mahalanobis_metric: required for ``MAHALANOBIS`` generator; a 1-D
            tensor of POSITIVE per-coordinate weights ``Λ_ii`` (we
            restrict to diagonal metrics here; full matrices are a
            substrate-engineering responsibility).
    """

    generator: BregmanGenerator = BregmanGenerator.SQUARED_EUCLIDEAN
    weights: tuple[float, ...] | None = None
    eps: float = DEFAULT_EPS
    mahalanobis_metric: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.eps <= 0 or not math.isfinite(float(self.eps)):
            raise BregmanError(f"eps must be positive, got {self.eps}")
        if self.weights is not None:
            if len(self.weights) == 0:
                raise BregmanError("weights must be non-empty when provided")
            if any(not math.isfinite(float(w)) for w in self.weights):
                raise BregmanError("weights must be finite")
            if any(w < 0 for w in self.weights):
                raise BregmanError("weights must be non-negative")
            if sum(self.weights) <= 0:
                raise BregmanError("weights must sum to a positive value")
        if self.generator == BregmanGenerator.MAHALANOBIS:
            if self.mahalanobis_metric is None:
                raise BregmanError(
                    "mahalanobis_metric is required for MAHALANOBIS generator"
                )
            if len(self.mahalanobis_metric) == 0:
                raise BregmanError("mahalanobis_metric must be non-empty")
            if any(not math.isfinite(float(m)) for m in self.mahalanobis_metric):
                raise BregmanError("mahalanobis_metric entries must be finite")
            if any(m <= 0 for m in self.mahalanobis_metric):
                raise BregmanError(
                    "mahalanobis_metric entries must be strictly positive"
                )


# ---------------------------------------------------------------------------
# Divergence + centroid kernels
# ---------------------------------------------------------------------------


def bregman_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    generator: BregmanGenerator,
    *,
    eps: float = DEFAULT_EPS,
    mahalanobis_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute D_φ(x, y) element-wise then sum over all dimensions.

    Returns a 0-D scalar tensor with grad flowing to both inputs.
    """
    if eps <= 0 or not math.isfinite(float(eps)):
        raise BregmanError("eps must be positive and finite")
    if x.shape != y.shape:
        raise BregmanError(f"shape mismatch: {x.shape} vs {y.shape}")
    _require_finite_tensor(x, field_name="x")
    _require_finite_tensor(y, field_name="y")
    if generator == BregmanGenerator.SQUARED_EUCLIDEAN:
        return 0.5 * (x - y).pow(2).sum()
    if generator == BregmanGenerator.KL:
        # D_KL(x || y) for x, y on the simplex; we clamp at eps for stability.
        xc = x.clamp_min(eps)
        yc = y.clamp_min(eps)
        return (xc * (xc.log() - yc.log()) - xc + yc).sum()
    if generator == BregmanGenerator.ITAKURA_SAITO:
        # D_IS(x, y) = x/y - log(x/y) - 1 (Itakura-Saito 1968)
        xc = x.clamp_min(eps)
        yc = y.clamp_min(eps)
        ratio = xc / yc
        return (ratio - ratio.log() - 1.0).sum()
    if generator == BregmanGenerator.MAHALANOBIS:
        if mahalanobis_metric is None:
            raise BregmanError("mahalanobis_metric required")
        if mahalanobis_metric.shape != x.shape[-1:]:
            raise BregmanError(
                "mahalanobis_metric must be 1-D and align with x.shape[-1]"
            )
        _require_finite_tensor(mahalanobis_metric, field_name="mahalanobis_metric")
        if torch.any(mahalanobis_metric <= 0):
            raise BregmanError("mahalanobis_metric must be strictly positive")
        diff = x - y
        return 0.5 * (mahalanobis_metric * diff.pow(2)).sum()
    raise BregmanError(f"unknown generator: {generator}")


def bregman_centroid(
    points: Sequence[torch.Tensor],
    generator: BregmanGenerator,
    *,
    weights: torch.Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Return the weighted Bregman centroid of the supplied points.

    Closed-form for each canonical generator:
        SQUARED_EUCLIDEAN: weighted arithmetic mean.
        KL: weighted geometric mean (normalised).
        ITAKURA_SAITO: weighted harmonic mean.
        MAHALANOBIS: weighted arithmetic mean (diagonal metric).
    """
    if len(points) == 0:
        raise BregmanError("points must be non-empty")
    shape = points[0].shape
    for i, p in enumerate(points):
        if p.shape != shape:
            raise BregmanError(f"point {i} shape {p.shape} != {shape}")
        _require_finite_tensor(p, field_name=f"points[{i}]")
    canonical_weights = _weights_tensor_to_tuple(weights, len(points))
    canonical_divergence = _canonical_divergence(generator)
    if canonical_divergence is not None:
        try:
            return _canonical_bregman_barycenter(
                points,
                canonical_weights,
                divergence=canonical_divergence,
                eps=eps,
            )
        except ValueError as exc:
            raise BregmanError(str(exc)) from exc
    stack = torch.stack(list(points), dim=0)  # (K, *shape)
    if canonical_weights is None:
        w = torch.full(
            (stack.shape[0],),
            1.0 / stack.shape[0],
            dtype=stack.dtype,
            device=stack.device,
        )
    else:
        w = torch.tensor(
            canonical_weights,
            dtype=stack.dtype,
            device=stack.device,
        )
    w_view = w.view((-1,) + (1,) * (stack.dim() - 1))
    if generator in (BregmanGenerator.SQUARED_EUCLIDEAN, BregmanGenerator.MAHALANOBIS):
        return (w_view * stack).sum(dim=0)
    if generator == BregmanGenerator.KL:
        clamped = stack.clamp_min(eps)
        log_centroid = (w_view * clamped.log()).sum(dim=0)
        centroid = log_centroid.exp()
        # Normalise to the simplex along the last dim if it sums > 0.
        s = centroid.sum(dim=-1, keepdim=True)
        return torch.where(s > 0, centroid / s.clamp_min(eps), centroid)
    if generator == BregmanGenerator.ITAKURA_SAITO:
        clamped = stack.clamp_min(eps)
        inv = clamped.reciprocal()
        return (w_view * inv).sum(dim=0).reciprocal()
    raise BregmanError(f"unknown generator: {generator}")


def _require_finite_tensor(tensor: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise BregmanError(f"{field_name} must be torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise BregmanError(f"{field_name} must contain finite values")


def _weights_tensor_to_tuple(
    weights: torch.Tensor | None,
    expected_len: int,
) -> tuple[float, ...] | None:
    if weights is None:
        return None
    if weights.shape != (expected_len,):
        raise BregmanError(f"weights shape {weights.shape} must equal ({expected_len},)")
    if not torch.isfinite(weights).all():
        raise BregmanError("weights must be finite")
    vals = tuple(float(v) for v in weights.detach().cpu().tolist())
    if any(v < 0 for v in vals):
        raise BregmanError("weights must be non-negative")
    if sum(vals) <= 0:
        raise BregmanError("weights must sum to a positive value")
    total = sum(vals)
    return tuple(v / total for v in vals)


def _canonical_divergence(
    generator: BregmanGenerator,
) -> CanonicalBregmanDivergence | None:
    if generator == BregmanGenerator.SQUARED_EUCLIDEAN:
        return "squared_euclidean"
    if generator == BregmanGenerator.KL:
        return "kl_forward"
    return None


# ---------------------------------------------------------------------------
# BregmanMixer module
# ---------------------------------------------------------------------------


class BregmanMixer:
    """Compose K parameter tensors via Bregman centroid (Eq. B.3).

    The mixer is stateless across calls (the spec fully determines
    behaviour); construct once + reuse with different point sets.

    Example
    -------
    >>> import torch
    >>> from tac.composition.bregman_mixing import (
    ...     BregmanMixer, BregmanMixerSpec, BregmanGenerator,
    ... )
    >>> spec = BregmanMixerSpec(
    ...     generator=BregmanGenerator.SQUARED_EUCLIDEAN,
    ...     weights=(0.5, 0.5),
    ... )
    >>> mixer = BregmanMixer(spec)
    >>> w1 = torch.zeros(4)
    >>> w2 = torch.ones(4)
    >>> mixer.mix([w1, w2])
    tensor([0.5000, 0.5000, 0.5000, 0.5000])
    """

    def __init__(self, spec: BregmanMixerSpec) -> None:
        self.spec = spec
        self._mahal: torch.Tensor | None = None
        if spec.mahalanobis_metric is not None:
            self._mahal = torch.tensor(spec.mahalanobis_metric, dtype=torch.float32)

    def mix(self, points: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the weighted Bregman centroid of the supplied points."""
        if self.spec.weights is None:
            weights = None
        else:
            if len(self.spec.weights) != len(points):
                raise BregmanError(
                    f"weights length {len(self.spec.weights)} != "
                    f"points length {len(points)}"
                )
            weights = torch.tensor(
                self.spec.weights,
                dtype=points[0].dtype,
                device=points[0].device,
            )
        return bregman_centroid(
            points,
            self.spec.generator,
            weights=weights,
            eps=self.spec.eps,
        )

    def divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Bregman divergence D_φ(x, y) under the spec generator."""
        mahal = self._mahal
        if mahal is not None:
            mahal = mahal.to(dtype=x.dtype, device=x.device)
        return bregman_divergence(
            x,
            y,
            self.spec.generator,
            eps=self.spec.eps,
            mahalanobis_metric=mahal,
        )

    def serialize_state(self) -> bytes:
        """Deterministic serialisation of the spec.

        Format (little-endian):
            magic[4] = b"BRG1"
            schema_version: u16
            generator_token_len: u16
            generator_token: bytes  (ascii)
            num_weights: u16 (0 → uniform)
            weights: f32[num_weights]
            eps: f64
            mahal_len: u32 (0 → none)
            mahal: f32[mahal_len]
        """
        spec = self.spec
        token = spec.generator.value.encode("ascii")
        body = bytearray()
        body += BREGMAN_MAGIC
        body += struct.pack("<H", BREGMAN_SCHEMA_VERSION)
        body += struct.pack("<H", len(token))
        body += token
        if spec.weights is None:
            body += struct.pack("<H", 0)
        else:
            body += struct.pack("<H", len(spec.weights))
            for w in spec.weights:
                body += struct.pack("<f", float(w))
        body += struct.pack("<d", spec.eps)
        if spec.mahalanobis_metric is None:
            body += struct.pack("<I", 0)
        else:
            body += struct.pack("<I", len(spec.mahalanobis_metric))
            for m in spec.mahalanobis_metric:
                body += struct.pack("<f", float(m))
        return bytes(body)

    @classmethod
    def deserialize_state(cls, payload: bytes) -> BregmanMixer:
        """Inverse of :meth:`serialize_state`."""
        if len(payload) < 4 or payload[:4] != BREGMAN_MAGIC:
            raise BregmanError(f"bad magic: {payload[:4]!r}")
        off = 4
        (version,) = struct.unpack_from("<H", payload, off)
        off += 2
        if version != BREGMAN_SCHEMA_VERSION:
            raise BregmanError(f"unsupported schema version: {version}")
        (token_len,) = struct.unpack_from("<H", payload, off)
        off += 2
        token = payload[off : off + token_len].decode("ascii")
        off += token_len
        try:
            generator = BregmanGenerator(token)
        except ValueError as exc:
            raise BregmanError(f"unknown generator token: {token}") from exc
        (num_w,) = struct.unpack_from("<H", payload, off)
        off += 2
        if num_w == 0:
            weights: tuple[float, ...] | None = None
        else:
            weights = tuple(
                struct.unpack_from("<f", payload, off + 4 * i)[0]
                for i in range(num_w)
            )
            off += 4 * num_w
        (eps,) = struct.unpack_from("<d", payload, off)
        off += 8
        (mahal_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        if mahal_len == 0:
            mahal: tuple[float, ...] | None = None
        else:
            mahal = tuple(
                struct.unpack_from("<f", payload, off + 4 * i)[0]
                for i in range(mahal_len)
            )
            off += 4 * mahal_len
        return cls(
            BregmanMixerSpec(
                generator=generator,
                weights=weights,
                eps=eps,
                mahalanobis_metric=mahal,
            )
        )


def estimate_param_bytes(spec: BregmanMixerSpec) -> int:
    """Conservative byte estimate for serialisation (cap at 4 + spec body)."""
    base = 4 + 2 + 2 + len(spec.generator.value) + 2 + 8 + 4
    if spec.weights is not None:
        base += 4 * len(spec.weights)
    if spec.mahalanobis_metric is not None:
        base += 4 * len(spec.mahalanobis_metric)
    return base


__all__ = [
    "BREGMAN_MAGIC",
    "BREGMAN_SCHEMA_VERSION",
    "BregmanError",
    "BregmanGenerator",
    "BregmanMixer",
    "BregmanMixerSpec",
    "bregman_centroid",
    "bregman_divergence",
    "estimate_param_bytes",
]
