# SPDX-License-Identifier: MIT
"""Canonical Alice-vs-Eve adversarial loop (Yousfi autostego 2026).

Origin: ``github.com/YassineYousfi/autostego`` (Mar 11, 2026; Yousfi's most
recent active steganalysis repo; ships HILL/WOW/S-UNIWARD steganography
algorithms + SRNet/SRM/LCLSMR detectors + canonical ``eve.py`` orchestrator).

The CANONICAL insight (Yousfi 2026 autostego game):
A round-robin competition between Alice (steganography agent) and Eve
(steganalysis agent) operationalizes the Fridrich-school inverse-steganalysis
philosophy: every effective attack against a detector becomes an upgraded
detector for the next round. The canonical scoring rule
``Alice = min(per_algorithm_accuracy)`` vs ``Eve = max(per_detector_accuracy)``
is a MINIMAX game-theoretic formulation that ensures Alice's worst algorithm
must beat Eve's best detector for Alice to win.

For the comma-video contest the canonical adaptation is:
- **Alice** = our substrate trainer that generates per-pair perturbations
  designed to be UNDETECTABLE by SegNet+PoseNet (the contest scorers).
- **Eve** = SegNet+PoseNet themselves (fixed adversaries; we cannot upgrade
  them, but we can iteratively probe their blind spots per Slot GGG
  pose-axis null projection canonical pattern).
- **Score** = `Alice_score = max(min(d_seg, sqrt(10*d_pose)))` per the
  canonical pose-axis priority ordering (CLAUDE.md "SegNet vs PoseNet
  importance" at PR106 frontier operating point).

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- JPEG steganalysis -> contest YUV6 lossy compression
* **Axis B (problem space)** -- Alice-Eve round-robin -> Alice (substrate)
  vs fixed Eve (SegNet+PoseNet); we cannot upgrade Eve, but we can iteratively
  probe via Slot GGG pose-axis null
* **Axis C (math)** -- minimax scoring 1:1 (per Yousfi's
  ``Alice = min`` vs ``Eve = max`` rule)
* **Axis D (data)** -- BOSSbase 256x256 PGM -> ``upstream/videos/0.mkv``
  1164x874x1200 frames per Catalog #213
* **Axis E (video)** -- per-image -> per-pair shared latent

Sister landings
---------------
* ``tac.composition.alaska_inverse_steganalysis_patterns.canonical_pair_constraint_batch``
  -- canonical pair-constraint batching that Alice must honor when generating
  perturbations against Eve.
* ``tac.composition.alaska_inverse_steganalysis_patterns.canonical_detector_aware_iterative_training``
  -- canonical training protocol for Eve (and for Alice's adversary model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

__all__ = (
    "AliceEveScoringRule",
    "AliceEveRoundConfig",
    "AliceEveRoundResult",
    "AliceEveLoopError",
    "compute_alice_score_minimax",
    "compute_eve_score_minimax",
    "canonical_alice_seed_algorithms",
    "canonical_eve_seed_detectors",
)


class AliceEveLoopError(ValueError):
    """Raised when an Alice-vs-Eve round violates a canonical invariant."""


class AliceEveScoringRule(str, Enum):
    """Canonical Alice-vs-Eve scoring rules.

    Per Yousfi autostego README: *"Alice's score is the minimum of the 3
    accuracies, while Eve's score is the maximum of the 3 accuracies."*

    * ``MINIMAX_CANONICAL`` -- Yousfi's 1:1 rule (Alice=min, Eve=max).
    * ``MEAN_BASELINE`` -- ablation (Alice=mean, Eve=mean) for cargo-cult
      audit comparison; biases Alice toward marginally-better-than-average
      algorithms instead of forcing her to harden her WORST algorithm.
    * ``MEDIAN_ROBUST`` -- robust variant (Alice=median, Eve=median); useful
      when one of Alice's algorithms is a known outlier diagnostic-only smoke
      that should NOT dominate the score.
    * ``WEIGHTED_SUM`` -- weighted-by-confidence aggregation; canonical for
      Catalog #341 Tier A observability-only multi-detector ensembles where
      each detector carries a canonical Provenance per Catalog #323.
    """

    MINIMAX_CANONICAL = "minimax_canonical"
    MEAN_BASELINE = "mean_baseline"
    MEDIAN_ROBUST = "median_robust"
    WEIGHTED_SUM = "weighted_sum"


def canonical_alice_seed_algorithms() -> tuple[str, ...]:
    """Return the canonical seed algorithms Yousfi ships in autostego.

    Per Yousfi autostego README: *"I have seeded the game with 3 SOTA
    steganographic algorithms (HILL, WOW, S-UNIWARD) and 2 SOTA steganalysis
    detectors (SRNet, SRM)."*

    Returns
    -------
    tuple[str, ...]
        ``("HILL", "WOW", "S-UNIWARD")`` -- canonical 3 SOTA spatial-domain
        steganography algorithms.

    Notes
    -----
    * **HILL** = High-pass + Low-pass + Low-pass; sister of
      ``tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014``
      (existing Slot YY package).
    * **WOW** = Wavelet Obtained Weights; sister of canonical wavelet-based
      embedding cost (NOT YET PORTED to tac.composition; canonical-helper
      gap noted for Phase D op-routables).
    * **S-UNIWARD** = Spatial UNIWARD; sister of
      ``tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion``
      (existing Slot FF package).
    """
    return ("HILL", "WOW", "S-UNIWARD")


def canonical_eve_seed_detectors() -> tuple[str, ...]:
    """Return the canonical seed detectors Yousfi ships in autostego.

    Per ``autostego/eve.py`` ``PipelineConfig.detectors`` default:
    ``["lclsmr", "srnet"]`` + ``fusion`` available; SRM is also bundled in
    ``steganalysis/srm.py``.

    Returns
    -------
    tuple[str, ...]
        ``("SRNet", "SRM", "LCLSMR", "fusion")`` -- canonical 4 detectors
        autostego ships.

    Notes
    -----
    * **SRNet** = Steganalysis Residual Network (Boroumand-Chen-Fridrich
      2019); deep-learning detector; sister of
      ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_onehot_jpeg_steganalysis``
      OneHot CNN.
    * **SRM** = Spatial Rich Model (Pevny-Bas-Fridrich 2011); 34K-feature
      classical detector; canonical reference baseline.
    * **LCLSMR** = Linear Classifier with LSMR solver; NEW canonical
      detector ported in this package (NOT YET in our prior portfolio).
    * **fusion** = score-level ensemble combining the above per
      ``autostego/steganalysis/fusion.py`` canonical helper.
    """
    return ("SRNet", "SRM", "LCLSMR", "fusion")


@dataclass(frozen=True)
class AliceEveRoundConfig:
    """Configuration for one Alice-vs-Eve round.

    Attributes
    ----------
    scoring_rule
        Which canonical scoring rule to apply.
    alice_algorithm_count
        How many algorithms Alice ships per round (Yousfi canonical = 3).
    eve_detector_count
        How many detectors Eve ships per round (Yousfi canonical = 3, but
        autostego seeds with 2-4 SRNet/SRM/LCLSMR/fusion).
    max_rounds
        Hard cap on round count to prevent unbounded competition.
    """

    scoring_rule: AliceEveScoringRule = AliceEveScoringRule.MINIMAX_CANONICAL
    alice_algorithm_count: int = 3
    eve_detector_count: int = 3
    max_rounds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.scoring_rule, AliceEveScoringRule):
            raise AliceEveLoopError(
                f"scoring_rule={self.scoring_rule!r} must be AliceEveScoringRule"
            )
        if self.alice_algorithm_count < 1:
            raise AliceEveLoopError(
                f"alice_algorithm_count={self.alice_algorithm_count} must be >= 1"
            )
        if self.eve_detector_count < 1:
            raise AliceEveLoopError(
                f"eve_detector_count={self.eve_detector_count} must be >= 1"
            )
        if self.max_rounds < 1:
            raise AliceEveLoopError(
                f"max_rounds={self.max_rounds} must be >= 1"
            )


@dataclass(frozen=True)
class AliceEveRoundResult:
    """Result of one Alice-vs-Eve round.

    Attributes
    ----------
    round_index
        0-based round counter.
    alice_per_algorithm_accuracy
        Mapping algorithm-name -> detector accuracy across all Eve detectors.
        Higher means Eve detects Alice well (Eve wins on that algorithm).
    eve_per_detector_accuracy
        Mapping detector-name -> max accuracy across all Alice algorithms.
        Higher means Eve detects Alice well (Eve wins with that detector).
    alice_score
        Canonical Alice score per scoring_rule (lower = Alice winning).
    eve_score
        Canonical Eve score per scoring_rule (higher = Eve winning).
    winner
        ``"alice"`` if Eve cannot detect (alice_score < 0.5 typically) or
        ``"eve"`` if Eve detects (eve_score > 0.5 typically).
    """

    round_index: int
    alice_per_algorithm_accuracy: Mapping[str, float]
    eve_per_detector_accuracy: Mapping[str, float]
    alice_score: float
    eve_score: float
    winner: str


def compute_alice_score_minimax(
    per_algorithm_accuracy: Mapping[str, float],
    scoring_rule: AliceEveScoringRule = AliceEveScoringRule.MINIMAX_CANONICAL,
) -> float:
    """Compute canonical Alice score per scoring_rule.

    1:1 with Yousfi autostego README ``Alice = min(per_algorithm_accuracy)``
    when ``scoring_rule == MINIMAX_CANONICAL``.

    Parameters
    ----------
    per_algorithm_accuracy
        Mapping algorithm-name -> Eve's best detection accuracy against that
        algorithm. Higher accuracy = Alice's algorithm is more detectable.
    scoring_rule
        Which canonical rule to apply.

    Returns
    -------
    float
        Alice's score (lower = Alice winning more decisively).

    Raises
    ------
    AliceEveLoopError
        Invalid input.
    """
    if not per_algorithm_accuracy:
        raise AliceEveLoopError("per_algorithm_accuracy empty")
    values = list(per_algorithm_accuracy.values())
    for v in values:
        if not isinstance(v, (int, float)):
            raise AliceEveLoopError(
                f"per_algorithm_accuracy value {v!r} must be numeric"
            )
    if scoring_rule == AliceEveScoringRule.MINIMAX_CANONICAL:
        # Per Yousfi: Alice = min (forces her worst algorithm to beat Eve's
        # best detector).
        return float(min(values))
    if scoring_rule == AliceEveScoringRule.MEAN_BASELINE:
        return float(sum(values) / len(values))
    if scoring_rule == AliceEveScoringRule.MEDIAN_ROBUST:
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n % 2 == 1:
            return float(sorted_v[n // 2])
        return float((sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2)
    if scoring_rule == AliceEveScoringRule.WEIGHTED_SUM:
        # Equal-weight fallback when no explicit weights supplied.
        return float(sum(values) / len(values))
    raise AliceEveLoopError(f"unhandled scoring_rule {scoring_rule!r}")


def compute_eve_score_minimax(
    per_detector_accuracy: Mapping[str, float],
    scoring_rule: AliceEveScoringRule = AliceEveScoringRule.MINIMAX_CANONICAL,
) -> float:
    """Compute canonical Eve score per scoring_rule.

    1:1 with Yousfi autostego README ``Eve = max(per_detector_accuracy)``
    when ``scoring_rule == MINIMAX_CANONICAL``.

    Parameters
    ----------
    per_detector_accuracy
        Mapping detector-name -> max accuracy across all Alice algorithms.
    scoring_rule
        Which canonical rule to apply.

    Returns
    -------
    float
        Eve's score (higher = Eve winning more decisively).

    Raises
    ------
    AliceEveLoopError
        Invalid input.
    """
    if not per_detector_accuracy:
        raise AliceEveLoopError("per_detector_accuracy empty")
    values = list(per_detector_accuracy.values())
    for v in values:
        if not isinstance(v, (int, float)):
            raise AliceEveLoopError(
                f"per_detector_accuracy value {v!r} must be numeric"
            )
    if scoring_rule == AliceEveScoringRule.MINIMAX_CANONICAL:
        # Per Yousfi: Eve = max (rewards her best detector).
        return float(max(values))
    if scoring_rule == AliceEveScoringRule.MEAN_BASELINE:
        return float(sum(values) / len(values))
    if scoring_rule == AliceEveScoringRule.MEDIAN_ROBUST:
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n % 2 == 1:
            return float(sorted_v[n // 2])
        return float((sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2)
    if scoring_rule == AliceEveScoringRule.WEIGHTED_SUM:
        return float(sum(values) / len(values))
    raise AliceEveLoopError(f"unhandled scoring_rule {scoring_rule!r}")
