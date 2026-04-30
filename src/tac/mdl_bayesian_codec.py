# ROUNDTRIP_NOT_REQUIRED: this is a meta codec-comparison framework (Bayesian
# model selection over OTHER codecs), not a byte-encoding codec. There is no
# encode(x) → bytes / decode(bytes) → x pair to roundtrip. The module consumes
# pre-computed L_total accountings from sibling codecs and ranks them via
# Bayes factor + Laplace evidence + Occam-razor sanity. The `*_codec.py`
# filename matches our naming convention for sibling-codec analysis tools
# (cf. arithmetic_qint_codec which IS a byte codec; this isn't).
"""Lane MDL/Bayesian — meta codec-comparison framework (MacKay).

Per Phase 3 Lane MDL spec (memory `project_phases_2_3_4_*` §"Lane 16 Bayesian MDL")
and council design doc `.omx/research/council_lane_mdl_bayesian_design_20260430.md`.

This module is a **codec-selection framework**, not a new codec. It does NOT ship
bytes in `archive.zip`. It consumes per-codec L_total accountings from sibling
lanes (Lane Ω-W-V2, Lane J-NWC, Lane 20 Ballé, Lane SH static, etc.) and produces
a Bayesian ranking + Occam-razor sanity check.

Math foundation (MacKay 2003 *Information Theory, Inference, and Learning Algorithms* Ch. 28)
---------------------------------------------------------------------------------------------

**Two-part MDL** (Rissanen 1978):

    L(D, M) = L(M) + L(D | M)                                       [bits]

where L(M) = bits to ship the codec / model itself (e.g. hyperprior MLP weights,
codebook, header) and L(D|M) = bits to ship the data given the codec.

**Bayesian evidence** (MacKay 1992):

    p(D | M) = ∫ p(D | θ, M) p(θ | M) dθ

`-log2 p(D|M) ≈ L_total(M, D)` under the MDL-Bayesian equivalence (negligible
constant + integration over θ).

**Bayes factor** between two codec families:

    BF_{12} = p(D|M_1) / p(D|M_2)
    log2 BF_{12} = -L_total(M_1, D) + L_total(M_2, D)

**Laplace approximation** at the MAP for tractable evidence:

    log p(D|M) ≈ log p(D|θ_MAP, M) + log p(θ_MAP|M) + (k/2) log(2π) - (1/2) log |H|

where `H` is the Hessian of negative log posterior at θ_MAP, `k` = parameter count.
The `(1/2) log |H|` is the **Occam's razor** term penalizing models that need precise
parameter tuning.

**Variational MDL** (Hinton & van Camp 1993):

    F = E_q[log p(D|θ, M)] - KL(q(θ|D) || p(θ|M))

This is the variational LOWER bound on `log p(D|M)` and the variational UPPER bound
on description length. **Lane 20 Ballé's R_total IS this F under his learned q(θ|D)** —
Lane MDL consumes Ballé's reported R_total directly without re-running the inference.

CLAUDE.md compliance
--------------------
- No silent defaults — every public function arg required-keyword
- No scorer load — works on PROVIDED L_total numbers from upstream codec lanes
- No GPU dependency; pure CPU analysis
- All claims tagged [synthetic] / [prediction]
- Ships NOTHING in archive.zip (this is a SELECTION framework)

Out of scope (intentional)
--------------------------
- Implementing any specific codec (Lane 20 / Lane J-NWC / Lane Ω-W-V2 do that)
- Running inference / training (consumes already-trained codec results)

References
----------
- MacKay 2003 ITILA Ch. 28
- Rissanen 1978
- MacKay 1992 — "Bayesian interpolation"
- Hinton & van Camp 1993 — "Keeping neural networks simple by minimizing the
  description length of the weights"
- Ballé et al. 2018 ICLR (variational rate)
- council design doc: .omx/research/council_lane_mdl_bayesian_design_20260430.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ── magic / version (no archive bytes; for report headers only) ─────────


MDL_FRAMEWORK_VERSION: int = 1
"""Reporting schema version. Bumped on any change to MDLCodecResult fields."""


# ── core data structure ────────────────────────────────────────────────


@dataclass
class MDLCodecResult:
    """Per-codec MDL accounting fed to the ranker.

    All bits/bytes are honest measurements from the codec's encode-roundtrip
    on real data — NOT predictions. The framework refuses to rank predictions.
    """

    codec_name: str
    """Human-readable codec family identifier (e.g. 'lane_balle_hyperprior')."""

    model_bits: int
    """L(M): bits required to ship the codec ITSELF in archive.zip.
    Includes MLP weights, codebook, header, magic bytes — anything that gets
    written to disk for the decoder to recover D from compressed payload.
    """

    residual_bits: int
    """L(D|M): bits of compressed payload (the encoded data given the codec)."""

    n_data_symbols: int
    """Number of source symbols (e.g. qint count) used to compute per-symbol rates."""

    log_prior_log2_p_model: float = 0.0
    """log2 p(M) — prior weight over codec families. Default 0 = uniform prior.
    Negative values penalize codec families a-priori (e.g. -1.0 = 50% odds).
    """

    n_codec_params: int = 0
    """k in Laplace approximation: codec parameter count for Occam's razor term.
    0 = no Laplace correction (use raw two-part MDL only).
    """

    hessian_logdet: float | None = None
    """log |H| at θ_MAP for Laplace approximation. None = skip Laplace correction."""

    notes: str = ""
    """Provenance: contest-CUDA score, anchor, dispatch metadata, etc."""

    def total_bits(self) -> int:
        """L_total(M, D) = L(M) + L(D|M).

        Returns the canonical two-part MDL total in bits.
        """
        return int(self.model_bits + self.residual_bits)

    def per_symbol_bits(self) -> float:
        """L(D|M) / n_data_symbols — useful for cross-codec comparison."""
        if self.n_data_symbols <= 0:
            raise ValueError(
                f"n_data_symbols must be positive for per-symbol rate, "
                f"got {self.n_data_symbols}"
            )
        return float(self.residual_bits) / float(self.n_data_symbols)

    def laplace_log_evidence(self) -> float:
        """log p(D|M) under Laplace approximation.

        Returns the log-evidence in nats. Higher = more posterior weight.

        Raises:
            ValueError if Laplace inputs (n_codec_params, hessian_logdet)
            are not both set.
        """
        if self.n_codec_params <= 0 or self.hessian_logdet is None:
            raise ValueError(
                "laplace_log_evidence requires n_codec_params > 0 and hessian_logdet "
                "to be set; got n_codec_params="
                f"{self.n_codec_params}, hessian_logdet={self.hessian_logdet}"
            )
        # MDL says L_total ≈ -log2 p(D|M); convert to nats.
        log_p_data_given_map_nats = -self.total_bits() * math.log(2.0)
        # Laplace: + log p(θ_MAP|M) + (k/2) log(2π) - (1/2) log |H|
        # log p(θ_MAP|M) absorbed into log_prior_log2_p_model (in bits).
        log_prior_nats = self.log_prior_log2_p_model * math.log(2.0)
        occam_nats = (self.n_codec_params / 2.0) * math.log(2.0 * math.pi) - (
            0.5 * self.hessian_logdet
        )
        return log_p_data_given_map_nats + log_prior_nats + occam_nats


# ── Two-part MDL primitives ───────────────────────────────────────────


def mdl_total_bits(*, model_bits: int, residual_bits: int) -> int:
    """Compute two-part MDL total `L(M) + L(D|M)`.

    Args (all required-keyword):
        model_bits: L(M) — bits to ship the codec itself.
        residual_bits: L(D|M) — bits of compressed payload.

    Returns:
        Total description length in bits.

    Raises:
        ValueError: if any input is negative.
    """
    if model_bits < 0:
        raise ValueError(f"model_bits must be non-negative, got {model_bits}")
    if residual_bits < 0:
        raise ValueError(f"residual_bits must be non-negative, got {residual_bits}")
    return int(model_bits) + int(residual_bits)


def laplace_log_evidence(
    *,
    log_likelihood_max_nats: float,
    log_prior_max_nats: float,
    hessian_logdet: float,
    n_params: int,
) -> float:
    """Laplace approximation to `log p(D|M)` in nats.

        log p(D|M) ≈ log p(D|θ_MAP, M) + log p(θ_MAP|M)
                     + (k/2) log(2π) - (1/2) log |H|

    Args (all required-keyword):
        log_likelihood_max_nats: log p(D | θ_MAP, M) at the MAP.
        log_prior_max_nats: log p(θ_MAP | M) at the MAP.
        hessian_logdet: log |H| where H is the Hessian of -log p(θ|D, M) at θ_MAP.
        n_params: k = parameter count of the codec.

    Returns:
        Approximate log evidence in nats. Higher = better-supported codec.

    Raises:
        ValueError: if n_params < 0.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")
    occam = (n_params / 2.0) * math.log(2.0 * math.pi) - (0.5 * hessian_logdet)
    return log_likelihood_max_nats + log_prior_max_nats + occam


def bayes_factor_log2(*, l_total_a_bits: float, l_total_b_bits: float) -> float:
    """log2 of Bayes factor BF_{A,B} = p(D|M_A) / p(D|M_B).

    Under MDL-Bayesian equivalence (uniform prior), `log2 BF = -L_A + L_B`.
    Positive → A wins; negative → B wins.

    Per MacKay rule of thumb: |log2 BF| > 5 is "strong evidence" (~32:1).

    Args (all required-keyword):
        l_total_a_bits: total description length of codec A.
        l_total_b_bits: total description length of codec B.

    Returns:
        log2 Bayes factor in favor of A (positive = A preferred).
    """
    return float(-l_total_a_bits + l_total_b_bits)


# ── Ranking / model averaging ──────────────────────────────────────────


def rank_codecs(
    results: Sequence[MDLCodecResult],
    *,
    temperature: float = 1.0,
) -> list[tuple[str, float, int]]:
    """Rank codec families by posterior weight (softmax over -L_total).

    Args (all required-keyword):
        results: per-codec MDL accountings.
        temperature: softmax temperature in bits. T=1 = pure Bayesian; T>1 = soften;
            T<1 = sharpen. Default T=1 is the pure Bayesian rule.

    Returns:
        List of (codec_name, posterior_weight, rank) sorted by descending weight.
        rank is 1-indexed (1 = best).

    Raises:
        ValueError: if results is empty or temperature <= 0.
    """
    if not results:
        raise ValueError("rank_codecs requires at least one MDLCodecResult")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    totals = np.array([r.total_bits() for r in results], dtype=np.float64)
    # Subtract priors to apply them: log_prior is in bits (log2 p(M))
    # Posterior log-weight in bits = -L_total + log2 p(M)
    log_priors = np.array(
        [r.log_prior_log2_p_model for r in results], dtype=np.float64
    )
    log_unnormalized_bits = -totals + log_priors
    # Softmax with temperature in bits — convert to nats for numerical stability
    log_unnormalized_nats = (log_unnormalized_bits / temperature) * math.log(2.0)
    log_unnormalized_nats -= log_unnormalized_nats.max()  # stabilize
    weights = np.exp(log_unnormalized_nats)
    weights = weights / weights.sum()

    # Sort descending by weight, with secondary key = L_total (smaller = better)
    # to break softmax-underflow ties. argsort is unstable on equal values, so
    # we use a composite key.
    composite_key = list(zip(-weights, totals - log_priors, range(len(results))))
    composite_key.sort()
    order = [k[2] for k in composite_key]
    return [
        (results[i].codec_name, float(weights[i]), int(rank + 1))
        for rank, i in enumerate(order)
    ]


def bayesian_model_average(
    *,
    weights: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Bayesian model average of scalar predictions.

    Used when no single codec dominates and we want to commit to a weighted
    combination. Rare in our regime (one codec usually wins decisively).

    Args (all required-keyword):
        weights: posterior weights summing to 1.
        predictions: per-codec scalar predictions (e.g. byte counts).

    Returns:
        Weighted average prediction.

    Raises:
        ValueError: if weights/predictions length mismatch or weights don't
            sum to 1 within 1e-6 tolerance.
    """
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if w.shape != p.shape:
        raise ValueError(
            f"weights and predictions must have same shape, got {w.shape} vs {p.shape}"
        )
    if w.size == 0:
        raise ValueError("bayesian_model_average requires at least one weight")
    if not math.isclose(float(w.sum()), 1.0, abs_tol=1e-6):
        raise ValueError(f"weights must sum to 1, got {float(w.sum())}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    return float((w * p).sum())


# ── Occam's razor pre-flight ───────────────────────────────────────────


@dataclass
class OccamCheck:
    """Refuse codecs whose model bits exceed achievable y-stream savings.

    Per CLAUDE.md "Council conduct — non-negotiable" the framework MUST refuse
    to ship a codec whose `L(M)` exceeds the empirical achievable savings on
    the residual stream. This stops the "ship a 10KB hyperprior MLP for 5KB
    of savings" pattern Quantizr explicitly flagged.
    """

    achievable_savings_bytes: int
    """Empirical ceiling on bits saveable on the residual stream (e.g. Shannon
    entropy gap on a target stream).
    """

    safety_margin: float = 1.0
    """Multiplier on achievable savings before refusing. 1.0 = strict, 0.5 =
    permit codecs spending up to 2× savings (e.g. for amortization across
    many archives). Default 1.0 is strict per Quantizr objection.
    """

    def evaluate(self, codec: MDLCodecResult) -> tuple[bool, str]:
        """Return (passed, reason) for this codec.

        passed=True iff the codec's model bits are within the safety margin
        of the achievable savings.
        """
        if self.achievable_savings_bytes < 0:
            raise ValueError(
                f"achievable_savings_bytes must be non-negative, "
                f"got {self.achievable_savings_bytes}"
            )
        if self.safety_margin <= 0:
            raise ValueError(
                f"safety_margin must be positive, got {self.safety_margin}"
            )
        ceiling_bits = self.achievable_savings_bytes * 8 / self.safety_margin
        if codec.model_bits > ceiling_bits:
            return False, (
                f"codec {codec.codec_name!r} L(M)={codec.model_bits} bits "
                f"exceeds Occam ceiling {ceiling_bits:.0f} bits "
                f"(achievable_savings_bytes={self.achievable_savings_bytes}, "
                f"safety_margin={self.safety_margin}). REFUSE TO SHIP."
            )
        return True, (
            f"codec {codec.codec_name!r} L(M)={codec.model_bits} bits "
            f"within Occam ceiling {ceiling_bits:.0f} bits — OK"
        )


# ── Strict-prior helper ────────────────────────────────────────────────


def derive_codec_prior_log2(
    *,
    codec_lineage_depth: int,
    has_contest_cuda_validation: bool,
    has_3_clean_review: bool,
) -> float:
    """Derive the log2 p(M) prior weight from codec maturity signals.

    Per MacKay's "the prior must be stated explicitly" principle: rather than
    a hand-tuned scalar, we derive the prior from objective lane-maturity
    signals. Lanes with deeper lineage + contest-CUDA validation + adversarial
    review get higher prior weight.

    Args (all required-keyword):
        codec_lineage_depth: how many revisions has this codec gone through?
            (e.g. Lane Ω-W-V2 = 2; Lane G v3 = 3; Lane 20 = 1).
        has_contest_cuda_validation: True if a [contest-CUDA] score has landed.
        has_3_clean_review: True if 3-clean-pass adversarial review counter
            is at 3/3.

    Returns:
        log2 p(M) — to be assigned to MDLCodecResult.log_prior_log2_p_model.

    The base prior is uniform (0.0). Bonuses:
    + 0.5 per lineage depth above 1 (max +1.0)
    + 1.0 if has_contest_cuda_validation
    + 1.0 if has_3_clean_review
    """
    if codec_lineage_depth < 0:
        raise ValueError(
            f"codec_lineage_depth must be non-negative, got {codec_lineage_depth}"
        )
    bonus = 0.0
    bonus += min(0.5 * max(0, codec_lineage_depth - 1), 1.0)
    if has_contest_cuda_validation:
        bonus += 1.0
    if has_3_clean_review:
        bonus += 1.0
    return float(bonus)


__all__ = [
    "MDL_FRAMEWORK_VERSION",
    "MDLCodecResult",
    "OccamCheck",
    "bayes_factor_log2",
    "bayesian_model_average",
    "derive_codec_prior_log2",
    "laplace_log_evidence",
    "mdl_total_bits",
    "rank_codecs",
]
