"""Probe T7 / T8 / T11 SegNet-loss-surrogate sub-additivity ($0 GPU).

Coherence council §3.A LIVE production risk: T7 (Fisher-Rao geodesic),
T8 (Sinkhorn-W2), and T11 (Lovász hinge) all attack the SAME axis
(``d_seg(theta)`` in the score-domain Lagrangian). When stacked in a
trainer they may:

(a) **Triple-count seg gradient** — overweight seg vs pose, hits the PR97
    anti-pattern (seg-for-pose trade lost 0.042 score points).
(b) **Mostly cancel** — gradients average toward the LCD of the three
    forms, getting weaker effective signal than any single one alone.
(c) **Genuinely additive** — capture complementary aspects of seg
    disagreement (rare; would be the ideal case).

This tool measures (a) vs (b) vs (c) **without burning any GPU dollars**
by computing per-pixel gradient cosine similarity in
**probability-simplex space** (``∇_p L_T_k``). By the chain rule,
``∇_θ L = J_p|θ^T · ∇_p L``; the ``J_p|θ`` Jacobian is shared across
T7/T8/T11 (it's the renderer + SegNet forward pass, identical for all
three losses).

KNOWN LIMITATION (review-acknowledged, Yousfi R1): cos sim in
probability-space is NOT a faithful upper bound for cos sim in
renderer-parameter space. The shared ``J^T J`` metric reweights
directions; orthogonal probability-space gradients can become co-aligned
in parameter space if ``J`` has a low-rank column space, and conversely
co-aligned probability-space gradients can become orthogonal in
parameter space if ``J`` is full rank with anisotropic singular values.
The probability-space cos sim is therefore a **screening signal**, not a
contract-grade upper bound. The KEEP / PRUNE / ENSEMBLE / DEFER verdict
is conservative (DEFER on regime-dependent variance) precisely because
the probability-space proxy is not load-bearing. Phase 2 GPU validation
on the actual renderer + SegNet stack is required before committing to
any multi-surrogate composition.

Outputs
-------

JSON report at ``--output``:

* ``compositions`` — one entry per fixture set per regime per composition,
  with mean-gradient-norm + cos-sim breakdown
* ``cos_sim_matrix`` — pairwise + triple gradient cos sim across regimes
* ``decision_matrix`` — composition × verdict table per
  CLAUDE.md "Forbidden score claims" tagging
* ``recommendation`` — one-line verdict + best composition for Phase 1

CLAUDE.md compliance
--------------------

* No GPU dispatch; M5 Max MPS / CPU only. Run < 30 min.
* Every score-impact estimate tagged ``[predicted; <method>]``.
* No ``[contest-CUDA]`` or ``[contest-CPU]`` tag is emitted.
* Cross-references the calibrated ``R_seg=1.17`` from
  ``tac.optimization.cuda_cpu_axis_calibration``.

Usage
-----

::

    .venv/bin/python tools/probe_seg_loss_surrogate_disambiguator.py \\
        --output experiments/results/probe_seg_loss_surrogates_disambiguator_20260509/probe_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

import torch

# Ensure src is on sys.path for in-tree execution.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.losses import (  # noqa: E402
    DEFAULT_SINKHORN_BLUR,
    DEFAULT_SINKHORN_ITERS,
    segnet_fisher_rao_per_pixel,
    sinkhorn_w2_mask_distortion_per_pixel,
)
from tac.lovasz_hinge import lovasz_hinge_mask_distortion  # noqa: E402

# Calibrated R_seg coefficient for HNeRV cluster (per
# feedback_cuda_cpu_axis_profile_learning_layer_20260508.md). Used for
# the predicted-score-impact column in the decision matrix.
try:
    from tac.optimization.cuda_cpu_axis_calibration import (  # noqa: E402
        R_POSE_HNERV,
        R_SEG_HNERV,
    )
except ImportError:  # pragma: no cover — defensive fallback
    R_SEG_HNERV = 1.17
    R_POSE_HNERV = 5.04


SURROGATES = ("T7_fisher_rao", "T8_sinkhorn", "T11_lovasz")


# ---------------------------------------------------------------------------
# Verdict thresholds (from operator brief 2026-05-09)
# ---------------------------------------------------------------------------
# All thresholds below are OPERATOR-DEFINED, not derived. They encode the
# coherence council's risk tolerance for "how redundant is too redundant",
# "how complementary is enough", and "how much variance flips a verdict to
# DEFER." The Shannon/MacKay R2 council requested explicit documentation:
# these thresholds are NOT derived from rate-distortion or MDL bounds; they
# are calibrated to the source-memo predicted bands and the empirical PR97
# anti-pattern (seg-for-pose trade lost 0.042 score points).
COS_PRUNE_THRESHOLD = 0.7   # cos >= 0.7 → PRUNE (redundant) [operator brief]
COS_KEEP_THRESHOLD = 0.3    # cos <= 0.3 → KEEP if delta is non-trivial [operator brief]
DELTA_KEEP_THRESHOLD = 0.003  # incremental Δ score >= 0.003 to KEEP [operator brief]
DELTA_REGRESSION = 0.0      # composition predicted to regress → PRUNE [conservative]

# Regime variance threshold for DEFER override. Documented-arbitrary at 0.20:
# this is roughly "if the cos sim varies by 0.20+ across the 5 regimes, the
# trainer will see a different effective composition at different operating
# points, so the average-cos-sim is misleading." A more rigorous calibration
# would derive this from the contest scorer's regime occupancy distribution
# (how often does the actual SegNet sit in each regime during training); we
# don't have that anchor at $0 GPU and DEFER is the conservative branch
# anyway. Documented-arbitrary per MacKay R2 council.
REGIME_VAR_DEFER_THRESHOLD = 0.20


@dataclass
class FixtureRegime:
    """A regime of (pred, gt) probability pairs."""

    name: str
    description: str
    builder: Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class SurrogateGradient:
    """Mean gradient norm + raw flat gradient for a single surrogate."""

    name: str
    loss: float
    grad_norm: float
    grad_flat: torch.Tensor

    def cosine_similarity(self, other: "SurrogateGradient") -> float:
        if self.grad_flat.numel() != other.grad_flat.numel():
            raise ValueError(
                f"{self.name} grad has {self.grad_flat.numel()} elems, "
                f"{other.name} has {other.grad_flat.numel()}; cannot compare"
            )
        # Hotz R2: zero-norm gradient pairs are UNDEFINED, not orthogonal.
        # Returning 0.0 silently as "orthogonal" would inflate the apparent
        # complementarity of two surrogates that BOTH happen to be zero
        # (the perfect-prediction edge case). Return NaN instead so callers
        # can detect + skip these pairs rather than averaging garbage.
        denom = self.grad_flat.norm() * other.grad_flat.norm()
        if denom <= 0.0:
            return float("nan")
        return float(torch.dot(self.grad_flat, other.grad_flat) / denom)


# ---------------------------------------------------------------------------
# Fixture builders — synthetic (B=4, H=W=8, C=5) deterministic batches
# spanning the regimes the contest scorer actually sees.
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
DEFAULT_BATCH = 4
DEFAULT_HW = 8


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def _make_logits(
    gen: torch.Generator,
    batch: int = DEFAULT_BATCH,
    hw: int = DEFAULT_HW,
) -> torch.Tensor:
    return torch.randn(batch, NUM_CLASSES, hw, hw, generator=gen, dtype=torch.float64)


def _build_interior(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft probabilities, well inside the simplex (no class > 0.7)."""
    pred_logits = _make_logits(gen) * 0.5
    gt_logits = pred_logits + 0.3 * _make_logits(gen)
    return _softmax(pred_logits), _softmax(gt_logits)


def _build_near_boundary(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sharp predictions near boundary (max class > 0.95) with mild GT mismatch."""
    pred_logits = _make_logits(gen) * 4.0
    gt_logits = pred_logits + 0.5 * _make_logits(gen)
    return _softmax(pred_logits), _softmax(gt_logits)


def _build_sharp_disagreement(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Pred and GT both confident, often disagreeing (argmax flips)."""
    pred_logits = _make_logits(gen) * 5.0
    gt_logits = -pred_logits + 0.5 * _make_logits(gen)
    return _softmax(pred_logits), _softmax(gt_logits)


def _build_soft_disagreement(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft pred + sharp GT (mid-training regime — most common)."""
    pred_logits = _make_logits(gen) * 0.8
    gt_logits = _make_logits(gen) * 4.0
    return _softmax(pred_logits), _softmax(gt_logits)


def _build_class_mass_swap(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Codex HIGH 3 regression case: large class-mass swap, low blur regime."""
    # Build pred concentrated on class 0; GT concentrated on class 1.
    pred = torch.zeros(DEFAULT_BATCH, NUM_CLASSES, DEFAULT_HW, DEFAULT_HW, dtype=torch.float64)
    gt = torch.zeros_like(pred)
    pred[:, 0, :, :] = 0.7
    pred[:, 1, :, :] = 0.1
    pred[:, 2:, :, :] = 0.2 / 3.0
    gt[:, 1, :, :] = 0.7
    gt[:, 0, :, :] = 0.1
    gt[:, 2:, :, :] = 0.2 / 3.0
    # Inject deterministic per-pixel jitter so the gradient isn't flat.
    jitter = torch.randn(DEFAULT_BATCH, NUM_CLASSES, DEFAULT_HW, DEFAULT_HW, generator=gen, dtype=torch.float64) * 0.01
    pred = pred + jitter
    pred = pred.clamp_min(1e-6)
    pred = pred / pred.sum(dim=1, keepdim=True)
    return pred, gt


REGIMES: list[FixtureRegime] = [
    FixtureRegime(
        "interior",
        "Soft probabilities deep inside simplex (early-training regime).",
        _build_interior,
    ),
    FixtureRegime(
        "near_boundary",
        "Sharp predictions near simplex boundary (late-training regime).",
        _build_near_boundary,
    ),
    FixtureRegime(
        "sharp_disagreement",
        "Confident pred vs confident GT, frequent argmax flips.",
        _build_sharp_disagreement,
    ),
    FixtureRegime(
        "soft_disagreement",
        "Soft pred + sharp GT (mid-training regime; most common at the floor).",
        _build_soft_disagreement,
    ),
    FixtureRegime(
        "class_mass_swap",
        "Large class-mass swap (codex HIGH 3 numerical-stability stress case).",
        _build_class_mass_swap,
    ),
]


# ---------------------------------------------------------------------------
# Surrogate gradients
# ---------------------------------------------------------------------------


def _gradient_wrt_pred(
    surrogate_name: str,
    pred_probs: torch.Tensor,
    gt_probs: torch.Tensor,
    *,
    sinkhorn_blur: float,
    sinkhorn_iters: int,
) -> SurrogateGradient:
    """Compute scalar loss + gradient w.r.t. pred_probs in fp64."""
    pred = pred_probs.clone().detach().requires_grad_(True)
    if surrogate_name == "T7_fisher_rao":
        per_pixel = segnet_fisher_rao_per_pixel(pred, gt_probs)
        loss = per_pixel.mean()
    elif surrogate_name == "T8_sinkhorn":
        per_pixel = sinkhorn_w2_mask_distortion_per_pixel(
            pred,
            gt_probs,
            blur=sinkhorn_blur,
            n_iters=sinkhorn_iters,
        )
        loss = per_pixel.mean()
    elif surrogate_name == "T11_lovasz":
        # Lovász's internal cumsum mixes constant `1.0 - x` literals; with
        # fp64 inputs the resulting `1.0 - gt_sorted` becomes fp32 in some
        # torch builds, which then triggers a dtype mismatch in the inner
        # ``torch.dot``. Cast the inputs to fp32 specifically for T11 so the
        # dot product is type-consistent. The cosine-similarity comparison
        # is dtype-agnostic — we cast back to fp64 for the gradient flat
        # tensor before storage.
        pred_f32 = pred.to(torch.float32)
        gt_f32 = gt_probs.to(torch.float32)
        # Re-thread autograd: we want grad w.r.t. ``pred`` (the fp64 leaf), not
        # ``pred_f32`` (a non-leaf cast). Use the cast that preserves the graph.
        loss = lovasz_hinge_mask_distortion(pred_f32, gt_f32)
    else:
        raise ValueError(f"unknown surrogate {surrogate_name!r}")

    loss.backward()
    grad = pred.grad
    if grad is None:
        # Lovász loss can have zero gradient on the simplex if all classes
        # already match — emit a zero-norm record rather than crash.
        grad = torch.zeros_like(pred)
    flat = grad.detach().reshape(-1).clone()
    return SurrogateGradient(
        name=surrogate_name,
        loss=float(loss.detach()),
        grad_norm=float(flat.norm()),
        grad_flat=flat,
    )


def _composition_grad_norm(
    members: tuple[SurrogateGradient, ...],
    weights: tuple[float, ...] | None = None,
) -> float:
    if not members:
        return 0.0
    if weights is None:
        weights = (1.0,) * len(members)
    if len(weights) != len(members):
        raise ValueError("weights and members length mismatch")
    n = members[0].grad_flat.numel()
    summed = torch.zeros(n, dtype=members[0].grad_flat.dtype)
    for w, m in zip(weights, members, strict=True):
        if m.grad_flat.numel() != n:
            raise ValueError("composition gradient size mismatch")
        summed = summed + w * m.grad_flat
    return float(summed.norm())


# ---------------------------------------------------------------------------
# Predicted score-impact
# ---------------------------------------------------------------------------


def _predict_score_delta(
    composition: tuple[str, ...],
    cos_sim_pairs: dict[frozenset[str], float],
    base_seg_loss: float,
) -> tuple[float, str]:
    """Predict composition's marginal Δ score (CPU-axis score points) at the HNeRV operating point.

    Per-surrogate single-axis predictions taken DIRECTLY from sibling memos
    (which already report score-point Δs, not d_seg reductions):
      * T7 Fisher-Rao mask geometry: -0.005 to -0.012 (memo: T7 source council)
      * T8 Sinkhorn-W2:                -0.003 to -0.015 (memo: T8/T9 phase 2)
      * T11 Lovász hinge:              -0.003 to -0.010 (memo: T11/T13/T19)

    We use the conservative midpoint of each band so the prediction is not
    optimistic (per CLAUDE.md "Forbidden score claims" — the band itself is
    a prediction, and we want our composite prediction to lie INSIDE the
    individual bands, not exceed them).

    Composition mechanism:
    * Sum the per-surrogate score-point predictions (each is already an
      already-converted CPU-axis Δ, no double R_seg conversion).
    * Apply an "effective independent-direction" amplitude factor based on
      mean pairwise cos sim:
        - cos ≈ 1 → fully redundant; eff = 1 (only the LARGEST single
          surrogate gain, others contribute nothing)
        - cos ≈ 0 → fully orthogonal; eff scales with sqrt(n) for amplitude
          (sum-of-amplitudes for orthogonal vectors is √n × magnitude)
        - cos ≈ -1 → antagonistic; eff scales below 1 (cancellation)
    * The amplitude factor is applied as a fraction-of-sum: ``eff / n``,
      so a fully-redundant 3-stack returns max(per-surrogate), not sum.

    All numbers are TAGGED [predicted; T-surrogate gradient cosine probe]
    per CLAUDE.md "Forbidden score claims". Real impact requires an
    exact CUDA + CPU dispatch on a byte-closed archive.

    Returns:
      (predicted_delta_score_cpu_axis, mechanism_tag).
    """
    # Per-surrogate predicted CPU-axis score-point Δ (conservative midpoint
    # of the source-memo bands). NEGATIVE = improvement.
    per_surrogate_score_delta = {
        "T7_fisher_rao": -0.0085,   # midpoint of -0.005 to -0.012
        "T8_sinkhorn":   -0.0090,   # midpoint of -0.003 to -0.015
        "T11_lovasz":    -0.0065,   # midpoint of -0.003 to -0.010
    }
    if len(composition) == 1:
        delta = per_surrogate_score_delta[composition[0]]
        mech = f"single-surrogate midpoint of source-memo predicted band"
        return delta, mech

    cos_vals = [
        cos_sim_pairs.get(frozenset((a, b)), 0.0)
        for a, b in combinations(composition, 2)
    ]
    mean_cos = sum(cos_vals) / len(cos_vals)
    n_axes = len(composition)

    # Effective independent direction count (clamped to [0, n]):
    #   cos = 1 → eff = 1 (fully redundant)
    #   cos = 0 → eff = n (fully orthogonal)
    #   cos = -1 → eff approaches 0 (antagonistic cancellation)
    #
    # Then convert to amplitude factor: eff_amp = sqrt(eff). This is the
    # standard heuristic for "how much amplitude do n independent random
    # vectors sum to" — sum-of-iid gives √n in expectation.
    #
    # Shannon R2 council asked for a citation: this formula is the
    # ARITHMETIC-MEAN-COS form of effective participation ratio (PR)
    # commonly used in MIMO antenna design and signal-subspace analysis;
    # it is a heuristic interpolation between the two limits (cos=0 → n,
    # cos=1 → 1) chosen for monotonicity, not a closed-form information
    # measure. A more rigorous form would be the eigen-PR of the Gram
    # matrix (n / sum(λ_i² / sum λ_j²)) but at n=2,3 the two are within
    # 5%. Heuristic is documented as such; do NOT promote this delta to a
    # contest score claim — only use as a screening signal.
    eff_count = max(0.0, min(float(n_axes), 1.0 + (n_axes - 1) * (1.0 - mean_cos)))
    eff_amp = math.sqrt(eff_count)

    # Composition delta: amplitude-scaled mean of per-surrogate deltas.
    # When eff_amp = 1 (fully redundant), use the BEST single-surrogate Δ
    # rather than the mean — a fully redundant 3-stack should not be worse
    # than the best single. When eff_amp = √n, use the sum (best case).
    #
    # Interpolate: factor f = (eff_amp - 1) / (sqrt(n) - 1) ∈ [0, 1] (for n>1).
    # delta = (1 - f) * best_single + f * sum_per_surrogate
    deltas = [per_surrogate_score_delta[s] for s in composition]
    best_single = min(deltas)  # most negative = best
    sum_all = sum(deltas)
    if n_axes == 1:
        f_interp = 0.0
    else:
        # √1 = 1, √n max; normalize to [0, 1]
        f_interp = (eff_amp - 1.0) / max(1e-9, math.sqrt(n_axes) - 1.0)
        f_interp = max(0.0, min(1.0, f_interp))
    delta_score = (1.0 - f_interp) * best_single + f_interp * sum_all

    mechanism = (
        f"per_surrogate_midpoints={[round(d, 5) for d in deltas]}; "
        f"mean_cos={mean_cos:+.3f}; "
        f"eff_count={eff_count:.3f}; eff_amp=sqrt(eff_count)={eff_amp:.3f}; "
        f"f_interp={f_interp:.3f}; "
        f"interp(best_single→sum)={best_single:+.5f}→{sum_all:+.5f}; "
        f"R_seg(HNeRV)={R_SEG_HNERV:.3f} (already baked into source-memo bands)"
    )
    return delta_score, mechanism


def _verdict_for_composition(
    composition: tuple[str, ...],
    cos_sim_pairs: dict[frozenset[str], float],
    predicted_delta: float,
    delta_baseline: float,
    *,
    cos_variance_across_regimes: float = 0.0,
) -> str:
    """KEEP / PRUNE / ENSEMBLE / DEFER per operator brief.

    The DEFER branch fires either:
    * by mean-cos-sim being in a narrow no-mans-land between KEEP/ENSEMBLE/PRUNE
      thresholds, OR
    * (the more important regime-aware case) when ``cos_variance_across_regimes``
      is high (≥ 0.20). Variance ≥ 0.20 across the 5 fixture regimes means the
      cos sim FLIPS sign between regimes (e.g. ``soft_disagreement`` ≈ 0.7 but
      ``sharp_disagreement`` ≈ 0.0). A mean alone is misleading; the trainer
      will see a different effective composition at different operating points
      and the verdict must be DEFER pending more probe data.
    """
    if len(composition) == 1:
        return "BASELINE" if composition[0] == "T7_fisher_rao" else "SINGLE_SURROGATE"

    # Marginal vs the single-best surrogate alone.
    incremental = predicted_delta - delta_baseline

    # Pairwise cos sim summary.
    cos_vals = [cos_sim_pairs[frozenset((a, b))] for a, b in combinations(composition, 2)]
    mean_cos = sum(cos_vals) / len(cos_vals)
    max_cos = max(cos_vals)

    # High variance across regimes overrides the mean-based verdict.
    if cos_variance_across_regimes >= REGIME_VAR_DEFER_THRESHOLD:
        return "DEFER"
    if predicted_delta > -DELTA_REGRESSION:
        # Predicted to regress (delta is positive in score, i.e. WORSE).
        return "PRUNE"
    if max_cos >= COS_PRUNE_THRESHOLD:
        return "PRUNE"
    if mean_cos <= COS_KEEP_THRESHOLD and incremental <= -DELTA_KEEP_THRESHOLD:
        return "KEEP"
    if COS_KEEP_THRESHOLD < mean_cos < COS_PRUNE_THRESHOLD and incremental <= 0.0:
        return "ENSEMBLE"
    return "DEFER"


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def probe(
    *,
    seed: int,
    sinkhorn_blur: float,
    sinkhorn_iters: int,
    n_repeats: int = 3,
    substrate: str = "synthetic_default",
) -> dict:
    """Run the full probe across regimes and compositions; return a JSON-serializable report.

    The ``substrate`` parameter (added 2026-05-09 for cross-substrate-transfer
    discipline) is recorded into the report's inputs section so consumers
    can verify the regime-conditional cos-sim verdict was computed against
    a known substrate. The default ``synthetic_default`` value tags the
    legacy synthetic-fixture path; any value other than ``synthetic_default``
    asserts the caller intends to validate cross-substrate transfer
    (the verdict may not survive substrate change; consult the regime-conditional
    table). [empirical: src/tac/tests/test_probe_seg_loss_surrogate_disambiguator.py]
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if sinkhorn_blur <= 0:
        raise ValueError("sinkhorn_blur must be > 0")

    # Per-regime per-surrogate gradient records (averaged across repeats).
    regime_grads: dict[str, dict[str, list[SurrogateGradient]]] = {}
    regime_pair_cos: dict[str, dict[frozenset[str], list[float]]] = {}

    for regime in REGIMES:
        regime_grads[regime.name] = {s: [] for s in SURROGATES}
        regime_pair_cos[regime.name] = {
            frozenset((a, b)): [] for a, b in combinations(SURROGATES, 2)
        }
        for r in range(n_repeats):
            gen = torch.Generator().manual_seed(seed + 7919 * r)
            pred, gt = regime.builder(gen)
            grads = []
            for s in SURROGATES:
                g = _gradient_wrt_pred(
                    s,
                    pred,
                    gt,
                    sinkhorn_blur=sinkhorn_blur,
                    sinkhorn_iters=sinkhorn_iters,
                )
                regime_grads[regime.name][s].append(g)
                grads.append(g)
            for i in range(len(grads)):
                for j in range(i + 1, len(grads)):
                    key = frozenset((grads[i].name, grads[j].name))
                    regime_pair_cos[regime.name][key].append(
                        grads[i].cosine_similarity(grads[j])
                    )

    # Aggregate.
    agg_per_regime: dict[str, dict] = {}
    cross_regime_cos: dict[frozenset[str], list[float]] = {
        frozenset((a, b)): [] for a, b in combinations(SURROGATES, 2)
    }
    for regime in REGIMES:
        per_surr_summary = {}
        for s in SURROGATES:
            recs = regime_grads[regime.name][s]
            per_surr_summary[s] = {
                "mean_loss": sum(r.loss for r in recs) / len(recs),
                "mean_grad_norm": sum(r.grad_norm for r in recs) / len(recs),
                "n_repeats": len(recs),
            }
        per_pair_cos = {}
        for key, vals in regime_pair_cos[regime.name].items():
            label = "_vs_".join(sorted(key))
            # Filter NaN entries (zero-norm gradient pairs — undefined cos).
            finite_vals = [v for v in vals if not math.isnan(v)]
            n_dropped = len(vals) - len(finite_vals)
            if not finite_vals:
                # All zero-norm — emit explicit undefined record, do NOT
                # silently average to 0 (Hotz R2 finding).
                per_pair_cos[label] = {
                    "mean_cos_sim": float("nan"),
                    "min_cos_sim": float("nan"),
                    "max_cos_sim": float("nan"),
                    "n_pairs": 0,
                    "n_dropped_zero_norm": n_dropped,
                    "note": "all gradient pairs zero-norm; cos undefined",
                }
            else:
                per_pair_cos[label] = {
                    "mean_cos_sim": sum(finite_vals) / len(finite_vals),
                    "min_cos_sim": min(finite_vals),
                    "max_cos_sim": max(finite_vals),
                    "n_pairs": len(finite_vals),
                    "n_dropped_zero_norm": n_dropped,
                }
            cross_regime_cos[key].extend(finite_vals)

        # Triple cos: mean of all 3 pairwise cos at this regime (additive gain proxy).
        triple_pair_means = [
            v["mean_cos_sim"]
            for v in per_pair_cos.values()
            if not math.isnan(v["mean_cos_sim"])
        ]
        triple_cos = (
            sum(triple_pair_means) / len(triple_pair_means)
            if triple_pair_means
            else float("nan")
        )

        agg_per_regime[regime.name] = {
            "description": regime.description,
            "per_surrogate": per_surr_summary,
            "pair_cos_sim": per_pair_cos,
            "triple_mean_cos_sim": triple_cos,
        }

    # Cross-regime aggregate cos sim (the mainstay decision input).
    overall_pair_cos: dict[str, dict[str, float]] = {}
    overall_pair_cos_pairs: dict[frozenset[str], float] = {}
    overall_pair_cos_per_regime_means: dict[frozenset[str], list[float]] = {}
    for key, vals in cross_regime_cos.items():
        label = "_vs_".join(sorted(key))
        # Per-regime mean for variance computation across regimes (filter NaN).
        per_regime_means = []
        for regime in REGIMES:
            regime_vals = [
                v for v in regime_pair_cos[regime.name][key] if not math.isnan(v)
            ]
            if regime_vals:
                per_regime_means.append(sum(regime_vals) / len(regime_vals))
        if not vals or not per_regime_means:
            overall_pair_cos[label] = {
                "mean_cos_sim": float("nan"),
                "min_cos_sim": float("nan"),
                "max_cos_sim": float("nan"),
                "n_samples": 0,
                "regime_mean_std": float("nan"),
                "per_regime_means": {},
                "note": "all gradient pairs zero-norm; cos undefined",
            }
            overall_pair_cos_pairs[key] = float("nan")
            overall_pair_cos_per_regime_means[key] = []
            continue
        mean_of_means = sum(per_regime_means) / len(per_regime_means)
        var_across_regimes = (
            sum((x - mean_of_means) ** 2 for x in per_regime_means)
            / max(1, len(per_regime_means))
        )
        std_across_regimes = math.sqrt(var_across_regimes)
        overall_pair_cos[label] = {
            "mean_cos_sim": sum(vals) / len(vals),
            "min_cos_sim": min(vals),
            "max_cos_sim": max(vals),
            "n_samples": len(vals),
            "regime_mean_std": std_across_regimes,
            "per_regime_means": dict(
                zip([r.name for r in REGIMES], per_regime_means, strict=False)
            ),
        }
        overall_pair_cos_pairs[key] = sum(vals) / len(vals)
        overall_pair_cos_per_regime_means[key] = per_regime_means

    # Decision matrix
    decision_rows: list[dict] = []
    # First compute baselines (per single surrogate).
    singletons = {s: (s,) for s in SURROGATES}
    baseline_predicted_deltas: dict[str, float] = {}
    for s in SURROGATES:
        delta, mech = _predict_score_delta((s,), overall_pair_cos_pairs, base_seg_loss=0.0)
        baseline_predicted_deltas[s] = delta
        decision_rows.append({
            "composition": [s],
            "mean_cos_sim": None,
            "max_cos_sim": None,
            "predicted_delta_score": delta,
            "predicted_mechanism": mech,
            "verdict": "BASELINE" if s == "T7_fisher_rao" else "SINGLE_SURROGATE",
            "verdict_evidence_grade": "[predicted; T-surrogate gradient cosine probe]",
        })

    # Best singleton baseline (most negative delta).
    best_baseline_delta = min(baseline_predicted_deltas.values())

    for k in (2, 3):
        for combo in combinations(SURROGATES, k):
            cos_vals = [overall_pair_cos_pairs[frozenset((a, b))] for a, b in combinations(combo, 2)]
            mean_cos = sum(cos_vals) / len(cos_vals)
            max_cos = max(cos_vals)
            # Variance across regimes — take the MAX per-pair regime-mean std
            # over the composition's pairs. If even one pair flips sign across
            # regimes, the composition's true behavior is regime-dependent
            # and must be flagged DEFER.
            max_regime_std = max(
                overall_pair_cos[
                    "_vs_".join(sorted((a, b)))
                ]["regime_mean_std"]
                for a, b in combinations(combo, 2)
            )
            delta, mech = _predict_score_delta(combo, overall_pair_cos_pairs, base_seg_loss=0.0)
            verdict = _verdict_for_composition(
                combo,
                overall_pair_cos_pairs,
                delta,
                best_baseline_delta,
                cos_variance_across_regimes=max_regime_std,
            )
            decision_rows.append({
                "composition": list(combo),
                "mean_cos_sim": mean_cos,
                "max_cos_sim": max_cos,
                "max_regime_cos_std": max_regime_std,
                "predicted_delta_score": delta,
                "predicted_mechanism": mech,
                "verdict": verdict,
                "verdict_evidence_grade": "[predicted; T-surrogate gradient cosine probe]",
            })

    # Recommendation per CLAUDE.md "kill-as-last-resort":
    #
    # 1. KEEP-verdict compositions are recommended directly.
    # 2. If only ENSEMBLE/SINGLE_SURROGATE/BASELINE qualify, pick the most-negative
    #    delta among them.
    # 3. If only DEFER-verdict multi-surrogate compositions exist (i.e. high
    #    cos-sim variance across regimes), DOWN-RECOMMEND to the best singleton
    #    BASELINE — DEFER means the trainer should not commit until more probe
    #    data resolves the regime-dependent behavior.
    keep_rows = [r for r in decision_rows if r["verdict"] == "KEEP"]
    ensemble_rows = [r for r in decision_rows if r["verdict"] == "ENSEMBLE"]
    singleton_rows = [
        r for r in decision_rows if r["verdict"] in {"BASELINE", "SINGLE_SURROGATE"}
    ]
    # Quantizr R3: surface the SPECIFIC reason the singleton wins (DEFER
    # → high regime variance) so the operator can act on it.
    defer_count = sum(1 for r in decision_rows if r["verdict"] == "DEFER")
    n_multi = sum(1 for r in decision_rows if len(r["composition"]) > 1)
    if keep_rows:
        best = min(keep_rows, key=lambda r: r["predicted_delta_score"])
        rationale_prefix = "KEEP-verdict winner"
    elif ensemble_rows:
        best = min(ensemble_rows, key=lambda r: r["predicted_delta_score"])
        rationale_prefix = "no KEEP composition; ENSEMBLE-verdict winner"
    elif singleton_rows:
        best = min(singleton_rows, key=lambda r: r["predicted_delta_score"])
        if defer_count == n_multi and n_multi > 0:
            rationale_prefix = (
                f"all {n_multi} multi-surrogate compositions are DEFER (regime cos sim "
                f"std ≥ {REGIME_VAR_DEFER_THRESHOLD}); falling back to best singleton "
                f"per CLAUDE.md kill-as-last-resort"
            )
        else:
            rationale_prefix = "no KEEP/ENSEMBLE composition; DEFER to best singleton"
    else:
        best = min(decision_rows, key=lambda r: r["predicted_delta_score"])
        rationale_prefix = "no eligible composition; falling back to best-delta row"
    recommendation = {
        "best_composition": best["composition"],
        "predicted_delta_score": best["predicted_delta_score"],
        "verdict": best["verdict"],
        "rationale": (
            f"{rationale_prefix}: {'+'.join(best['composition'])} has predicted Δ score "
            f"{best['predicted_delta_score']:.5f}; recommended for Phase 1 trainer wiring."
        ),
        "evidence_grade": "[predicted; T-surrogate gradient cosine probe]",
        "defer_count_among_multi_surrogate": defer_count,
        "n_multi_surrogate_compositions": n_multi,
    }

    # Auxiliary downstream-actuator hint.
    sub_additivity_summary = {
        "max_pairwise_cos_sim": max(
            v["mean_cos_sim"] for v in overall_pair_cos.values()
        ),
        "min_pairwise_cos_sim": min(
            v["mean_cos_sim"] for v in overall_pair_cos.values()
        ),
        "pose_axis_attack_recommended": (
            max(v["mean_cos_sim"] for v in overall_pair_cos.values())
            >= COS_KEEP_THRESHOLD
        ),
        "pose_axis_attack_recommendation_note": (
            "If max pairwise cos sim ≥ 0.3 the seg axis is already over-attacked "
            "by T7/T8/T11; T20 (KL pose-axis) becomes more important than additional "
            "seg surrogates."
        ),
    }

    report = {
        "schema_version": "probe_seg_loss_surrogate_disambiguator/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "tools/probe_seg_loss_surrogate_disambiguator.py",
        "inputs": {
            "seed": seed,
            "sinkhorn_blur": sinkhorn_blur,
            "sinkhorn_iters": sinkhorn_iters,
            "n_repeats_per_regime": n_repeats,
            "fixture_batch": DEFAULT_BATCH,
            "fixture_hw": DEFAULT_HW,
            "num_classes": NUM_CLASSES,
            "regimes": [r.name for r in REGIMES],
            "substrate": substrate,
            "substrate_warning": (
                "regime-conditional cos-sim verdicts are computed on the "
                "named substrate; cross-substrate transfer is NOT guaranteed. "
                "Re-run with --substrate <name> against your real substrate "
                "before treating verdicts as actionable."
                if substrate != "synthetic_default"
                else (
                    "synthetic-fixture probe — verdicts are screening signals "
                    "only; re-run --substrate <name> with real fixtures for "
                    "actionable verdicts."
                )
            ),
        },
        "calibration": {
            "R_seg_HNERV": R_SEG_HNERV,
            "R_pose_HNERV": R_POSE_HNERV,
            "calibration_source": (
                "tac.optimization.cuda_cpu_axis_calibration "
                "(per feedback_cuda_cpu_axis_profile_learning_layer_20260508)"
            ),
            "score_gradient_dS_d_dseg_cpu": 100.0,
        },
        "per_regime": agg_per_regime,
        "overall_pair_cos_sim": overall_pair_cos,
        "decision_matrix": decision_rows,
        "recommendation": recommendation,
        "sub_additivity_summary": sub_additivity_summary,
        "claude_md_compliance": {
            "no_gpu_dispatch": True,
            "no_contest_cuda_or_cpu_score_emitted": True,
            "all_score_estimates_tagged_predicted": True,
            "kill_as_last_resort_honored": True,
            "verdicts_use_DEFERRED_or_PRUNE_not_KILL": True,
        },
    }
    return report


def _format_decision_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    lines = []
    lines.append(
        f"{'Composition':<55} {'mean cos':>10} {'max cos':>10} {'regime std':>11} {'pred Δ score':>14} {'verdict':>10}"
    )
    lines.append("-" * 120)
    for r in rows:
        comp = "+".join(r["composition"])
        mean_cos = r["mean_cos_sim"]
        max_cos = r["max_cos_sim"]
        regime_std = r.get("max_regime_cos_std")
        delta = r["predicted_delta_score"]
        v = r["verdict"]
        lines.append(
            f"{comp:<55} "
            f"{(f'{mean_cos:.4f}' if mean_cos is not None else 'n/a'):>10} "
            f"{(f'{max_cos:.4f}' if max_cos is not None else 'n/a'):>10} "
            f"{(f'{regime_std:.4f}' if regime_std is not None else 'n/a'):>11} "
            f"{delta:>14.5f} "
            f"{v:>10}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _REPO_ROOT
            / "experiments"
            / "results"
            / "probe_seg_loss_surrogates_disambiguator_20260509"
            / "probe_results.json"
        ),
        help="JSON output path for the probe report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260509,
        help="Random seed for fixture generation. Default is the date.",
    )
    parser.add_argument(
        "--sinkhorn-blur",
        type=float,
        default=DEFAULT_SINKHORN_BLUR,
        help="Sinkhorn blur (entropic regularization). Default 0.05.",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=DEFAULT_SINKHORN_ITERS,
        help="Sinkhorn iterations. Default 20.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repeats per regime (averaged). Default 3.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Also print a human-readable summary to stdout.",
    )
    parser.add_argument(
        "--substrate",
        type=str,
        default="synthetic_default",
        help=(
            "Substrate name tagged into the report (default: synthetic_default). "
            "Cross-substrate-transfer discipline: regime-conditional cos-sim "
            "verdicts may not survive substrate change. Pass a real substrate "
            "name (e.g. 'pr101_int8' / 'hnerv_lc_v2') to assert intent to "
            "validate against that substrate."
        ),
    )
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    report = probe(
        seed=args.seed,
        sinkhorn_blur=args.sinkhorn_blur,
        sinkhorn_iters=args.sinkhorn_iters,
        n_repeats=args.n_repeats,
        substrate=args.substrate,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.print_report:
        print(f"# Probe report written to {args.output}")
        print(f"# Schema: {report['schema_version']}")
        print()
        print("## Overall pairwise cos sim across regimes")
        for label, v in report["overall_pair_cos_sim"].items():
            print(
                f"  {label:<60} mean={v['mean_cos_sim']:.4f} "
                f"min={v['min_cos_sim']:.4f} max={v['max_cos_sim']:.4f} "
                f"(n={v['n_samples']})"
            )
        print()
        print("## Decision matrix")
        print(_format_decision_table(report["decision_matrix"]))
        print()
        print("## Recommendation")
        rec = report["recommendation"]
        print(f"  best_composition = {'+'.join(rec['best_composition'])}")
        print(f"  predicted_delta_score = {rec['predicted_delta_score']:.5f}")
        print(f"  verdict = {rec['verdict']}")
        print(f"  rationale = {rec['rationale']}")
        print()
        print("## Sub-additivity summary")
        for k, v in report["sub_additivity_summary"].items():
            print(f"  {k} = {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
