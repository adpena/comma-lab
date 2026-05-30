# SPDX-License-Identifier: MIT
"""Canonical Syndrome Trellis Coding (Filler-Judas-Fridrich 2011).

Origin: Filler-Judas-Fridrich 2011 IEEE TIFS *"Minimizing Additive Distortion
in Steganography Using Syndrome-Trellis Codes"*. Tomáš Filler was Fridrich's
PhD student at Binghamton DDE Lab (sister of Yousfi). Canonical upstream
C++/Matlab implementation:
``http://dde.binghamton.edu/download/syndrome/``. Multiple GitHub Python
ports exist (e.g. `daniellerch/pySTC`, `surfinbard/stc-steganography`).

The CANONICAL insight (Filler-Judas-Fridrich 2011):
Given a cover ``x`` and a target message ``m``, find the stego ``y`` that
minimizes ``sum_i rho_i * |y_i - x_i|`` subject to ``H * y = m`` (the
syndrome constraint). The solution is a Viterbi-like dynamic-programming
walk through a parity-check trellis with width ``2^h`` (h = sub-matrix
height). For ``h = 7-12`` STC achieves within ~1-3% of the rate-distortion
bound for spatial-domain steganography.

For the comma-video contest, the canonical adaptation is:
- **Cover** ``x`` = per-pair perturbation magnitudes from our substrate's
  Lagrangian dual variables (per-pixel ``rho`` from canonical equation
  ``dykstra_pareto_polytope_intersection_compounding_v1`` per Catalog #372).
- **Message** ``m`` = the "payload" we're embedding = the score-axis
  perturbation direction (per Slot GGG pose-axis null projection per Catalog
  #341 Tier A canonical-routing markers).
- **Syndrome constraint** ``H * y = m`` = the byte-deterministic archive
  bytes contract per Catalog #218 (the perturbation must be encodable in
  the archive byte stream that survives ``inflate.sh``).

This is the canonical OPTIMAL embedding-cost-allocation algorithm; it's the
algorithmic reason why HILL / S-UNIWARD / MiPOD / HUGO ALL OUTPERFORM naive
LSB embedding: they pair an embedding cost ``rho`` with STC's optimal
allocation, not with random embedding.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- steganographic embedding -> contest perturbation
  allocation (different domain, same optimization problem)
* **Axis B (problem space)** -- minimize distortion s.t. payload -> minimize
  ``25 * rate_bytes / 37545489`` s.t. score axes
* **Axis C (math)** -- 1:1 Viterbi trellis dynamic programming;
  ``O(n * 2^h)`` time complexity per Filler 2011 Section IV
* **Axis D (data)** -- BOSSbase 256x256 -> per-pair latent
* **Axis E (video)** -- per-image -> per-pair shared latent

Sister landings
---------------
* CLAUDE.md "Grand Council (advisory)" Filler seat (canonical voice for
  STC + parity-check codes + per-frame mask payload).
* Catalog #218 ``check_substrate_reconstruct_methods_support_mini_batch``
  (canonical mini-batch reconstruction discipline that STC's per-pair
  embedding inherits).
* ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_lclsmr_linear_steganalysis_detector``
  (sister optimization helper; LCLSMR for linear-classifier optimization,
  STC for parity-check optimization).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = (
    "STCSubMatrixHeight",
    "STCAdaptiveEmbeddingStrategy",
    "STCEmbeddingConfig",
    "STCEmbeddingError",
    "compute_stc_distortion_bound_from_sub_matrix_height",
    "FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE",
    "FILLER_2011_DISTORTION_VS_BOUND_PERCENT",
)


class STCEmbeddingError(ValueError):
    """Raised when STC config violates a canonical invariant."""


FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE: tuple[int, int] = (7, 12)
"""Per Filler-Judas-Fridrich 2011 IEEE TIFS Section VI: canonical sub-matrix
height range for practical STC. h=7 gives ~3% gap to bound; h=12 gives ~1%
gap; trellis state count = 2^h so h=12 has 4096 states per Viterbi step."""


FILLER_2011_DISTORTION_VS_BOUND_PERCENT: dict[int, float] = {
    7: 3.0,
    8: 2.5,
    9: 2.0,
    10: 1.7,
    11: 1.4,
    12: 1.2,
}
"""Per Filler-Judas-Fridrich 2011 Table II: empirical % gap to
rate-distortion bound at each canonical sub-matrix height. Higher h means
smaller gap but 2x more trellis states per step."""


class STCSubMatrixHeight(int, Enum):
    """Canonical Filler 2011 sub-matrix heights (state count = 2^h)."""

    H_7_128_STATES = 7
    H_8_256_STATES = 8
    H_9_512_STATES = 9
    H_10_1024_STATES = 10
    H_11_2048_STATES = 11
    H_12_4096_STATES = 12


class STCAdaptiveEmbeddingStrategy(str, Enum):
    """Canonical STC adaptive-embedding strategies.

    * ``CANONICAL_VITERBI_TRELLIS`` -- Filler-Judas-Fridrich 2011 1:1
      Viterbi-like dynamic programming over parity-check trellis.
    * ``GREEDY_LSB_BASELINE`` -- ablation: greedy LSB embedding (no STC,
      no rho-weighting); included for cargo-cult audit comparison per
      Catalog #303.
    * ``RANDOM_EMBEDDING_BASELINE`` -- ablation: random LSB flips at
      random positions; included as canonical NULL hypothesis baseline.
    * ``OPTIMAL_LP_RELAXATION`` -- ablation: solve as LP relaxation of
      the integer programming problem; gives lower bound but not
      practical embedding pattern; included for canonical-bound comparison.
    """

    CANONICAL_VITERBI_TRELLIS = "canonical_viterbi_trellis"
    GREEDY_LSB_BASELINE = "greedy_lsb_baseline"
    RANDOM_EMBEDDING_BASELINE = "random_embedding_baseline"
    OPTIMAL_LP_RELAXATION = "optimal_lp_relaxation"


@dataclass(frozen=True)
class STCEmbeddingConfig:
    """Canonical STC embedding configuration.

    Attributes
    ----------
    strategy
        Which canonical strategy.
    sub_matrix_height
        Trellis sub-matrix height (canonical Filler 2011 range = 7..12).
    payload_rate_bits_per_pixel
        Relative payload size; canonical Yousfi autostego = 0.4 bpac;
        canonical alaska benchmark = 0.4 bpac.
    """

    strategy: STCAdaptiveEmbeddingStrategy = (
        STCAdaptiveEmbeddingStrategy.CANONICAL_VITERBI_TRELLIS
    )
    sub_matrix_height: STCSubMatrixHeight = STCSubMatrixHeight.H_10_1024_STATES
    payload_rate_bits_per_pixel: float = 0.4

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, STCAdaptiveEmbeddingStrategy):
            raise STCEmbeddingError(
                f"strategy={self.strategy!r} must be STCAdaptiveEmbeddingStrategy"
            )
        if not isinstance(self.sub_matrix_height, STCSubMatrixHeight):
            raise STCEmbeddingError(
                f"sub_matrix_height={self.sub_matrix_height!r} must be STCSubMatrixHeight"
            )
        if not (0 < self.payload_rate_bits_per_pixel <= 1.0):
            raise STCEmbeddingError(
                f"payload_rate_bits_per_pixel={self.payload_rate_bits_per_pixel} "
                f"must be in (0, 1.0]"
            )


def compute_stc_distortion_bound_from_sub_matrix_height(
    sub_matrix_height: STCSubMatrixHeight,
) -> float:
    """Return canonical % gap-to-bound for given sub-matrix height.

    1:1 with Filler-Judas-Fridrich 2011 IEEE TIFS Table II empirical
    results.

    Parameters
    ----------
    sub_matrix_height
        :class:`STCSubMatrixHeight`.

    Returns
    -------
    float
        % gap to rate-distortion bound (smaller = closer to optimal).
        Range ~1.2% (h=12) to ~3.0% (h=7).

    Raises
    ------
    STCEmbeddingError
        Unknown height.
    """
    if not isinstance(sub_matrix_height, STCSubMatrixHeight):
        raise STCEmbeddingError(
            f"sub_matrix_height={sub_matrix_height!r} must be STCSubMatrixHeight"
        )
    h_int = int(sub_matrix_height)
    if h_int not in FILLER_2011_DISTORTION_VS_BOUND_PERCENT:
        raise STCEmbeddingError(
            f"sub_matrix_height={h_int} not in canonical table "
            f"{list(FILLER_2011_DISTORTION_VS_BOUND_PERCENT.keys())}"
        )
    return FILLER_2011_DISTORTION_VS_BOUND_PERCENT[h_int]
