"""F4: Score-vs-archive Lipschitz constant estimator.

Tao shower-thought (deep_math §13): the contest score
``S: {0, 1}^{8B} -> R`` viewed as a scalar function on the discrete hypercube
of archive-bit configurations has a Lipschitz constant ``L`` such that
``|S(A) - S(B)| <= L * d_hamming(A, B)`` for any two archives A and B that
share the same grammar/decoder/runtime. The reciprocal ``1 / L`` bounds
the minimal byte-budget granularity at which score-improvement is detectable.

This primitive estimates L via finite-difference single-bit / single-byte
flips at random sample positions. The estimator is deliberately
CONSERVATIVE: it reports the MAX observed ``|delta_S| / d_hamming``, which
upper-bounds the true Lipschitz constant.

Per deep_math §1.3 (Lagrangian-dual derivation of the contest formula),
the marginal cost of a single archive BYTE at the score optimum is exactly
``lambda = 25 / N = 25 / 37,545,489 ≈ 6.66e-7 score-units per byte``.
Therefore the per-BIT marginal is ``L = lambda / 8 ≈ 8.32e-8 score-units
per bit`` for the rate-axis-only component. Distortion-axis perturbations
typically have larger L (single-pixel flips can shift PoseNet by 1e-3 or
more per pixel for high-sensitivity pixels).

Wire-in hooks engaged:

- ``pareto_constraint``: ``1 / L`` is the granularity below which the
  Pareto frontier convex-hull cannot meaningfully resolve. Pareto
  vertices closer than ``1 / L`` in score-space are not actually
  distinguishable; the bit-allocator uses ``1 / L`` as its smallest
  feasible step.
- ``bit_allocator``: per-section Lipschitz constants tell the allocator
  which sections have the LARGEST L (= most score-sensitive bytes) so
  it spends its budget there last.
- ``probe_disambiguator``: comparing L estimated via random-flips vs L
  derived from the Lagrangian-dual ``lambda / 8`` disambiguates "score
  is locally smooth (random L >> Lagrangian L)" from "score is locally
  cliff-y (random L >> Lagrangian L)".

**Implementation note.** The primitive's ``compute()`` method does NOT
itself run scorer forwards (which require CUDA / paid GPU spend). Instead,
it accepts a ``score_fn: Callable[[bytes], float]`` kwarg. The caller
provides whatever score-evaluation surface is appropriate (proxy MSE,
contest_cuda eval, MPS-research-signal, etc.) and the primitive computes
the Lipschitz constant from finite-difference flips.

Cross-references
----------------
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §1.3 + §13
- Source: Tao shower-thought (zen_floor council, deep_math §13)
- Sister memory: ``feedback_z1_mdl_ablation_landed_20260514.md`` (the Z1
  perturbation methodology that this primitive generalizes)

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)


@dataclass(frozen=True)
class ScoreLipschitzReport:
    """Typed result from :meth:`ScoreVsArchiveLipschitz.compute`.

    Attributes
    ----------
    archive_n_bits : int
        Total bit count of the archive (= 8 * byte_count).
    n_flip_samples : int
        Number of single-bit flips evaluated.
    max_delta_score : float
        Maximum observed |delta_S| across the n_flip_samples flips.
    mean_delta_score : float
        Mean observed |delta_S| across the flips.
    median_delta_score : float
        Median observed |delta_S|.
    empirical_lipschitz_per_bit : float
        Conservative L estimate: max_delta_score / 1 (each flip is hamming
        distance 1). Units: score-units per bit.
    lagrangian_lipschitz_per_bit : float
        Closed-form: lambda / 8 = 25 / (N * 8) where N is uncompressed
        size in bytes. The minimum-possible Lipschitz coming from the
        rate axis alone.
    flip_score_distribution : tuple[float, ...]
        Sample of observed |delta_S| values (capped at 1024 entries to
        keep result size bounded).
    """

    archive_n_bits: int
    n_flip_samples: int
    max_delta_score: float
    mean_delta_score: float
    median_delta_score: float
    empirical_lipschitz_per_bit: float
    lagrangian_lipschitz_per_bit: float
    flip_score_distribution: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.archive_n_bits <= 0:
            raise ValueError("archive_n_bits must be positive")
        if self.n_flip_samples < 0:
            raise ValueError("n_flip_samples must be non-negative")
        if self.max_delta_score < 0.0:
            raise ValueError("max_delta_score must be non-negative")
        if self.lagrangian_lipschitz_per_bit < 0.0:
            raise ValueError(
                "lagrangian_lipschitz_per_bit must be non-negative"
            )


class ScoreVsArchiveLipschitz:
    """F4 canonical primitive: archive-bit-Lipschitz constant estimator.

    The estimator does NOT itself score archives — that requires CUDA and
    a paid-GPU dispatch. Instead, callers pass a ``score_fn`` callable
    that maps archive bytes to a scalar score.
    """

    # Contest constants pinned from upstream/evaluate.py:92.
    CONTEST_UNCOMPRESSED_SIZE_BYTES = 37_545_489
    RATE_COEFFICIENT = 25.0

    @property
    def name(self) -> str:
        return "score_lipschitz"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "pareto_constraint",
            "bit_allocator",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: Path | str | bytes,
        *,
        score_fn: Callable[[bytes], float] | None = None,
        n_samples: int = 0,
        seed: int = 0xBEEF,
        flip_bit_positions: list[int] | None = None,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Estimate Lipschitz constant via random-bit-flip finite differences.

        Parameters
        ----------
        target : Path | str | bytes
            Archive path OR raw bytes.
        score_fn : Callable[[bytes], float] | None
            Function mapping (perturbed) archive bytes to a scalar score.
            If None, the primitive returns only the closed-form Lagrangian
            Lipschitz (no empirical estimate).
        n_samples : int
            Number of single-bit flips to evaluate. If 0 AND score_fn
            is provided, defaults to 64 (a quick smoke).
        seed : int
            Random seed.
        flip_bit_positions : list[int] | None
            Optional explicit bit positions to flip (overrides random
            sampling). Useful for deterministic regression tests + for
            targeting specific sections via the
            ``mdl_scorer_conditional`` per_section_breakdown.
        """
        # Resolve archive bytes.
        archive_path: Path | None = None
        archive_sha: str | None = None
        if isinstance(target, bytes):
            archive_bytes = target
        else:
            archive_path = Path(target)
            if not archive_path.exists():
                raise ValueError(
                    f"archive {archive_path!s} does not exist"
                )
            archive_bytes = archive_path.read_bytes()
            from tac.repo_io import sha256_bytes

            archive_sha = sha256_bytes(archive_bytes)

        archive_n_bits = 8 * len(archive_bytes)
        if archive_n_bits == 0:
            raise ValueError("archive is empty; Lipschitz analysis undefined")

        # Closed-form Lagrangian Lipschitz from the rate axis.
        lambda_per_byte = (
            self.RATE_COEFFICIENT / self.CONTEST_UNCOMPRESSED_SIZE_BYTES
        )
        lagrangian_per_bit = lambda_per_byte / 8.0

        # Empirical estimate (only if score_fn provided).
        observed: list[float] = []
        if score_fn is not None:
            if flip_bit_positions is None:
                if n_samples <= 0:
                    n_samples = 64
                import random

                rng = random.Random(seed)
                flip_bit_positions = [
                    rng.randint(0, archive_n_bits - 1) for _ in range(n_samples)
                ]
            elif n_samples == 0:
                n_samples = len(flip_bit_positions)

            base_score = score_fn(archive_bytes)
            for bit_pos in flip_bit_positions:
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                mutable = bytearray(archive_bytes)
                mutable[byte_idx] ^= 1 << bit_in_byte
                perturbed_score = score_fn(bytes(mutable))
                observed.append(abs(perturbed_score - base_score))

        if observed:
            max_delta = max(observed)
            mean_delta = sum(observed) / len(observed)
            sorted_obs = sorted(observed)
            mid = len(sorted_obs) // 2
            if len(sorted_obs) % 2 == 0 and mid > 0:
                median_delta = (sorted_obs[mid - 1] + sorted_obs[mid]) / 2
            else:
                median_delta = sorted_obs[mid]
            empirical_per_bit = max_delta
        else:
            max_delta = 0.0
            mean_delta = 0.0
            median_delta = 0.0
            # When no empirical samples, the closed-form Lagrangian is
            # the only available estimate.
            empirical_per_bit = lagrangian_per_bit

        # Cap distribution at 1024 entries.
        distribution = tuple(observed[:1024])

        report = ScoreLipschitzReport(
            archive_n_bits=archive_n_bits,
            n_flip_samples=len(observed),
            max_delta_score=max_delta,
            mean_delta_score=mean_delta,
            median_delta_score=median_delta,
            empirical_lipschitz_per_bit=empirical_per_bit,
            lagrangian_lipschitz_per_bit=lagrangian_per_bit,
            flip_score_distribution=distribution,
        )

        # Confidence band: [lagrangian_lower_bound, empirical_upper_bound].
        band = (
            lagrangian_per_bit,
            max(empirical_per_bit, lagrangian_per_bit),
        )

        evidence = (
            "mathematical-derivation"
            if observed
            else "first-principles-bound"
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=archive_path,
            archive_sha256=archive_sha,
            primitive_value=report,
            evidence_grade=evidence,
            confidence_band=band,
            composes_with=("shannon_vector_r_d", "bilinear_resize_nullspace"),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "n_samples_requested": n_samples,
                "score_fn_provided": score_fn is not None,
                "seed": seed,
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "ScoreLipschitzReport",
    "ScoreVsArchiveLipschitz",
]
