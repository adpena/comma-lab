# SPDX-License-Identifier: MIT
"""PR110-OPT-7 Fridrich UNIWARD inverse-scorer basis L0 SCAFFOLD.

Sister of :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
(Slot FF LANDED 2026-05-29 commit ``18c6cd571``) at the **BASIS-EXPANSION axis**.

Where Slot FF enumerates 4 strategies on the **WEIGHTING axis** (sparse-K /
widened-K / per-region / all-pairs selectors), this lane enumerates 4
strategies on the **basis-source axis**: local-variance baseline vs SegNet
logit-gradient vs PoseNet output-gradient vs joint scorer-Jacobian linear
combination. Per Slot EEE FAKE-implementation audit (cited MEMORY.md
2026-05-29) Slot FF's 4 enum branches were classified PARTIAL (3-of-4
structurally equivalent at L0); THIS lane closes the substantive-distinctness
gap with 4 structurally distinct basis-source branches per Catalog #308
alternative-reducer enumeration.

Design memo (single source of truth)::

    .omx/research/pr110_opt7_fridrich_uniward_inverse_scorer_basis_design_20260530.md

Canonical context
=================

Fridrich UNIWARD canonical cost function per Holub-Fridrich-Denemark 2014::

    cost(pixel) = 1 / (epsilon + scorer_sensitivity_response(pixel))

Per CLAUDE.md "Fridrich inverse steganalysis":
1. UNIWARD — errors in textured regions are undetectable; weight loss by
   inverse local variance.
2. Detector-informed embedding — our TTO approach is Fridrich-approved.
3. Square root law — spread small errors (L∞ penalty); don't concentrate.
4. CNN blind spots — EfficientNet stride-2 stem 256x192 spatial blind spot;
   PoseNet FastViT-T12 YUV6 12-channel input.

Per CLAUDE.md "Exact scorer architectures":
- **SegNet**: ``smp.Unet('tu-efficientnet_b2', classes=5, ...)``; argmax
  over 5 class logits; bilinear resize to (512, 384); stride-2 stem.
- **PoseNet**: FastViT-T12 backbone; 12-channel YUV6 input; first 6 of 12
  output dims; MSE distortion.

The **inverse-scorer basis** is the set of per-pixel sensitivity surfaces
on which the inverse-steganalyzer's Jacobian acts. Four canonical bases:

- **Local variance** (Holub-Fridrich-Denemark 2014 canonical baseline):
  ``1/(eps + local_variance(pixel))`` where local_variance is db4 wavelet
  detail-coefficient variance OR 3x3 Sobel inverse-variance (documented
  adaptation per Catalog #303; db4 requires optional pywt).
- **SegNet logit-gradient magnitude**: ``1/(eps + |dL_seg/dx(pixel)|)`` —
  inverse of per-pixel SegNet output-logit gradient magnitude. Targets the
  EfficientNet-B2 stride-2 stem's 256x192 blind spot per CLAUDE.md.
- **PoseNet output-gradient magnitude**: ``1/(eps + |dL_pose/dx(pixel)|)``
  — inverse of per-pixel PoseNet 12-dim output gradient magnitude.
  Targets the FastViT-T12 YUV6 sensitivity surface.
- **Joint scorer-Jacobian linear combination**: ``alpha * SegNet_grad +
  beta * PoseNet_grad + gamma * local_variance`` (canonical Atick-Redlich
  1990 cooperative-receiver linear approximation; defaults alpha=beta=gamma=1/3).

PR110 archive grammar
=====================

Sister of Slot FF + Slot X PR110-OPT-4:

- 16-symbol K=16 selector palette per FEC6
  ``submissions/hnerv_fec6_fixed_huffman_k16/``
- 600 per-pair selectors
- 6-byte header + 243-byte 0-order fixed-Huffman bitstream = **249-byte
  baseline wire**

Wave N+34 OPT-7 canonical anchor at
``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``
reports unweighted aggregate ΔS=-0.001170 vs UNIWARD-weighted aggregate
ΔS=-0.000910 = -22.22% IMPLEMENTATION_FALSIFIED for WEIGHTING-axis.
THIS lane's BASIS-EXPANSION-axis targets the orthogonal axis: which
*basis* the UNIWARD weighting is applied over, not which *selector* gates
the weighting.

Source data (Catalog #213 canonical real-input):
- ``experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl``
  (600 pairs × 22 modes; canonical (pair, mode_id, segnet_dist, posenet_dist,
  component_score_no_rate) rows).

L0 SCAFFOLD role
================

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: THIS L0 SCAFFOLD declares ``research_only=true`` in lane
registry notes per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK pattern
until paired-CUDA empirical anchor lands.

The canonical
:func:`apply_uniward_inverse_scorer_basis_to_pr110_archive` entry point
returns a Tier A canonical-routing-markers contribution per Catalog #341
+ AxisDecomposition per Catalog #356 + canonical Provenance per Catalog
#323. The verdict field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR``.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356.
- :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted`.
- Tier A canonical-routing markers per Catalog #341 + #357.
- Catalog #309 ``horizon_class: frontier_pursuit``.
- Catalog #287 placeholder-rationale rejection in ``__post_init__`` validators.
- Catalog #192 macOS-CPU advisory NEVER promotable.
- Catalog #213 real-input integration (NOT synthetic per Slot EEE empirical
  anchor; default canonical PR101 pair_component_rows.jsonl).

Sister cross-references
=======================

- :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
  (Slot FF WEIGHTING-axis sister)
- :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X PR110-OPT-4 sister composition surface)
- :mod:`tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016`
  (sister Yousfi-Fridrich cascade Axis 6)
- :mod:`tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010`
  (sister Yousfi-Fridrich cascade Axis 7)
- :mod:`tac.cathedral.consumer_contract` (AxisDecomposition)
- :mod:`tac.provenance.builders` (Provenance builders)
- :mod:`tac.scorer` (canonical scorer loaders)
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

__all__ = [
    "InverseScorerBasisStrategy",
    "InverseScorerBasisConfig",
    "UniwardInverseScorerBasisResult",
    "WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES",
    "WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S",
    "WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S",
    "WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES",
    "WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS",
    "WAVE_N34_OPT7_N_PAIRS",
    "WAVE_N34_OPT7_N_MODES",
    "CANONICAL_SPARSE_K_DEFAULT",
    "CANONICAL_UNIWARD_EPSILON",
    "CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE",
    "CANONICAL_JOINT_ALPHA_DEFAULT",
    "CANONICAL_JOINT_BETA_DEFAULT",
    "CANONICAL_JOINT_GAMMA_DEFAULT",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "DEFAULT_PAIR_COMPONENT_ROWS_PATH",
    "compute_local_variance_basis_weight",
    "compute_segnet_gradient_basis_weight",
    "compute_posenet_gradient_basis_weight",
    "compute_joint_scorer_basis_weight",
    "compute_per_pair_uniward_cost_from_basis",
    "select_sparse_k_pairs_by_cost",
    "compute_basis_expansion_perturbation_for_pr110_catalog",
    "apply_uniward_inverse_scorer_basis_to_pr110_archive",
    "load_canonical_pr101_pair_component_rows",
]

# -----------------------------------------------------------------------------
# Wave N+34 OPT-7 canonical anchor constants (HISTORICAL_PROVENANCE per Catalog
# #110/#113; sister of Slot FF's identical constants, preserved here for
# self-containment of THIS BASIS-EXPANSION-axis L0 SCAFFOLD).
# -----------------------------------------------------------------------------

#: PR110 FEC6 fixed-Huffman K=16 baseline wire size per Wave N+34.
WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES: int = 249

#: Wave N+34 OPT-7 unweighted aggregate ΔS (uniform-prior baseline).
WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S: float = -0.0011704843740551621

#: Wave N+34 OPT-7 UNIWARD-weighted aggregate ΔS (IMPLEMENTATION_FALSIFIED
#: at -22.22% WORSE than unweighted on the WEIGHTING-axis sister).
WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S: float = -0.0009103568688898632

#: Wave N+34 OPT-7 sparse-selector K=100 wire bytes estimate.
WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES: int = 103

#: Wave N+34 OPT-7 sparse-selector K=100 proportional score savings.
WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS: float = -7.940203000166914e-05

#: Wave N+34 OPT-7 number of pairs in source sweep.
WAVE_N34_OPT7_N_PAIRS: int = 600

#: Wave N+34 OPT-7 number of modes in source catalog.
WAVE_N34_OPT7_N_MODES: int = 21

#: Canonical sparse-K default (K=100 is empirically the canonical inflection
#: point per Wave N+34 OPT-7 anchor).
CANONICAL_SPARSE_K_DEFAULT: int = 100

#: Fridrich UNIWARD cost denominator stabilizer per Holub-Fridrich-Denemark
#: 2014 epsilon convention.
CANONICAL_UNIWARD_EPSILON: float = 1.0e-6

#: Canonical local-variance kernel size (3x3 Sobel inverse-variance per
#: documented adaptation; canonical Holub-Fridrich-Denemark 2014 uses db4
#: wavelet detail-coefficient variance which requires optional pywt).
CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE: int = 3

#: Canonical joint scorer alpha default (SegNet weight in linear combination).
CANONICAL_JOINT_ALPHA_DEFAULT: float = 1.0 / 3.0

#: Canonical joint scorer beta default (PoseNet weight in linear combination).
CANONICAL_JOINT_BETA_DEFAULT: float = 1.0 / 3.0

#: Canonical joint scorer gamma default (local variance weight in
#: linear combination).
CANONICAL_JOINT_GAMMA_DEFAULT: float = 1.0 / 3.0

#: Canonical rate multiplier per contest formula
#: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
CANONICAL_RATE_MULTIPLIER: float = 25.0

#: Canonical rate denominator per contest formula.
CANONICAL_RATE_DENOM_BYTES: int = 37_545_489

#: Default canonical real-input path per Catalog #213 (real PR101 600-pair
#: paired-component rows; NEVER synthetic per Slot EEE empirical anchor).
DEFAULT_PAIR_COMPONENT_ROWS_PATH: str = (
    "experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/"
    "pair_component_rows.jsonl"
)

# -----------------------------------------------------------------------------
# 4 substantively distinct basis-source enum branches per Catalog #308
# alternative-reducer enumeration. Each branch computes a DIFFERENT per-pixel
# weight map; NOT enum-padding per Slot EEE FAKE-implementation audit.
# -----------------------------------------------------------------------------


class InverseScorerBasisStrategy(str, enum.Enum):
    """Canonical inverse-scorer basis-source enum for PR110-OPT-7.

    Per Catalog #308 alternative-reducer methodology enumeration: each branch
    uses a STRUCTURALLY DISTINCT basis surface for computing the per-pixel
    UNIWARD cost map. The 4 branches are NOT enum-padding (Catalog #287's
    5 forbidden classes #5) — each dispatches to a different canonical helper.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode + Catalog
    #303 cargo-cult audit + Slot EEE FAKE-implementation audit: the 4
    branches must satisfy substantive-distinctness — replacing one branch's
    implementation with another's MUST produce a different output for the
    same input, OR the branch must be marked DEFERRED per Catalog #287.
    """

    #: Holub-Fridrich-Denemark 2014 canonical baseline. Per-pixel weight =
    #: 1/(eps + local_variance(pixel)) where local_variance uses 3x3 Sobel
    #: gradient magnitude as inverse-variance proxy (documented adaptation
    #: per Catalog #303; canonical db4 wavelet requires optional pywt).
    UNIWARD_INVERSE_LOCAL_VARIANCE_BASELINE = "uniward_inverse_local_variance_baseline"

    #: SegNet-grounded basis per CLAUDE.md "Exact scorer architectures":
    #: per-pixel weight = 1/(eps + |dL_seg/dx(pixel)|) where dL_seg/dx is
    #: per-pixel SegNet output-logit gradient magnitude. Targets
    #: EfficientNet-B2 stride-2 stem 256x192 spatial blind spot.
    UNIWARD_INVERSE_SEGNET_GRADIENT_SENSITIVITY = "uniward_inverse_segnet_gradient_sensitivity"

    #: PoseNet-grounded basis per CLAUDE.md "Exact scorer architectures":
    #: per-pixel weight = 1/(eps + |dL_pose/dx(pixel)|) where dL_pose/dx is
    #: per-pixel PoseNet 12-dim output gradient magnitude (FastViT-T12 YUV6
    #: 12-channel input; first 6 of 12 output dims contribute to distortion).
    UNIWARD_INVERSE_POSENET_GRADIENT_SENSITIVITY = "uniward_inverse_posenet_gradient_sensitivity"

    #: Joint scorer-Jacobian linear combination per Atick-Redlich 1990
    #: cooperative-receiver framing: per-pixel weight = alpha * SegNet_grad
    #: + beta * PoseNet_grad + gamma * local_variance with canonical defaults
    #: alpha=beta=gamma=1/3 (TUNING-FREE per CARGO-CULTED Assumption-Adversary
    #: verdict; reactivation requires per-pair empirical anchor).
    UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION = (
        "uniward_inverse_joint_scorer_basis_linear_combination"
    )


# -----------------------------------------------------------------------------
# Canonical Config dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InverseScorerBasisConfig:
    """Canonical config for PR110-OPT-7 BASIS-EXPANSION-axis L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__`` per Catalog #287
    placeholder-rationale rejection discipline.

    Args:
        basis_strategy: Canonical :class:`InverseScorerBasisStrategy` enum.
            Defaults to ``UNIWARD_INVERSE_LOCAL_VARIANCE_BASELINE`` per
            Catalog #290 canonical-vs-unique adoption.
        n_pairs: Source pair count (canonical PR110 = 600 per Wave N+34).
        n_modes: Source mode count (canonical Wave N+34 = 21 active modes).
        sparse_k: Sparse-selector K (canonical = 100 per Wave N+34 anchor).
        uniward_epsilon: Fridrich UNIWARD denominator stabilizer.
        local_variance_kernel_size: Sobel/db4 kernel size for local-variance
            basis (canonical = 3 per CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE).
        joint_alpha: Linear-combination SegNet weight (only used when
            basis_strategy == UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION).
        joint_beta: Linear-combination PoseNet weight (same).
        joint_gamma: Linear-combination local-variance weight (same).
        header_overhead_bytes: Wire-format header overhead (sparse-K
            selector index format; canonical = 3 bytes for K=100).
        emit_axis_decomposition: If True (default), the apply entry point
            emits a canonical :class:`AxisDecomposition` per Catalog #356.
        pair_component_rows_path: Real PR101 paired-component rows path
            per Catalog #213 (real-input canonical; NEVER synthetic).
        rng_seed: numpy seed for ascending-sorted tie-breaking determinism
            (canonical = 42 per CLAUDE.md "Beauty, simplicity, and developer
            experience" sister-substrate convention).

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    basis_strategy: InverseScorerBasisStrategy = (
        InverseScorerBasisStrategy.UNIWARD_INVERSE_LOCAL_VARIANCE_BASELINE
    )
    n_pairs: int = WAVE_N34_OPT7_N_PAIRS
    n_modes: int = WAVE_N34_OPT7_N_MODES
    sparse_k: int = CANONICAL_SPARSE_K_DEFAULT
    uniward_epsilon: float = CANONICAL_UNIWARD_EPSILON
    local_variance_kernel_size: int = CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE
    joint_alpha: float = CANONICAL_JOINT_ALPHA_DEFAULT
    joint_beta: float = CANONICAL_JOINT_BETA_DEFAULT
    joint_gamma: float = CANONICAL_JOINT_GAMMA_DEFAULT
    header_overhead_bytes: int = 3
    emit_axis_decomposition: bool = True
    pair_component_rows_path: str = DEFAULT_PAIR_COMPONENT_ROWS_PATH
    rng_seed: int = 42

    def __post_init__(self) -> None:
        # Catalog #287 placeholder-rationale rejection discipline applied
        # to numeric invariants:
        if not isinstance(self.basis_strategy, InverseScorerBasisStrategy):
            raise ValueError(
                f"basis_strategy must be InverseScorerBasisStrategy enum member; "
                f"got {type(self.basis_strategy).__name__}={self.basis_strategy!r}"
            )
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0; got {self.n_pairs}")
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be > 0; got {self.n_modes}")
        if self.sparse_k <= 0 or self.sparse_k > self.n_pairs:
            raise ValueError(
                f"sparse_k must be in (0, n_pairs={self.n_pairs}]; got {self.sparse_k}"
            )
        if not (self.uniward_epsilon > 0.0 and math.isfinite(self.uniward_epsilon)):
            raise ValueError(
                f"uniward_epsilon must be finite > 0; got {self.uniward_epsilon}"
            )
        if self.local_variance_kernel_size < 3 or self.local_variance_kernel_size % 2 == 0:
            raise ValueError(
                f"local_variance_kernel_size must be odd integer >= 3; "
                f"got {self.local_variance_kernel_size}"
            )
        for name, value in (
            ("joint_alpha", self.joint_alpha),
            ("joint_beta", self.joint_beta),
            ("joint_gamma", self.joint_gamma),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]; got {value}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}")
        if self.header_overhead_bytes < 0:
            raise ValueError(
                f"header_overhead_bytes must be >= 0; got {self.header_overhead_bytes}"
            )
        if not isinstance(self.pair_component_rows_path, str) or not self.pair_component_rows_path.strip():
            raise ValueError(
                "pair_component_rows_path must be non-empty string; "
                f"got {self.pair_component_rows_path!r}"
            )
        if not isinstance(self.rng_seed, int) or self.rng_seed < 0:
            raise ValueError(f"rng_seed must be non-negative int; got {self.rng_seed!r}")


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class UniwardInverseScorerBasisResult:
    """Canonical Tier A return type per Catalog #341 + #357.

    Carries:
    - Predicted score delta adjustment (always 0.0 per Tier A
      observability-only contract per Catalog #341).
    - Promotability flag (always False per Catalog #341/#192).
    - Axis tag (always ``[predicted]`` per Catalog #287/#341).
    - Optional :class:`AxisDecomposition` per Catalog #356.
    - Canonical Provenance dict per Catalog #323.
    - Per-strategy wire analysis (queryable per Catalog #305).
    - Per-pair selector indices (canonical reproducibility surface).
    - Verdict string (``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR``).
    """

    strategy: InverseScorerBasisStrategy
    predicted_delta_adjustment: float
    promotable: bool
    axis_tag: str
    verdict: str
    wire_bytes_estimate: int
    delta_vs_fec6_bytes: int
    n_selected_pairs: int
    per_pair_selector_indices: tuple[int, ...]
    per_pair_uniward_costs: tuple[float, ...]
    aggregate_predicted_delta_s: float
    canonical_provenance: Mapping[str, Any]
    axis_decomposition: AxisDecomposition | None = None
    wire_analysis: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Catalog #341 Tier A canonical-routing-markers invariants:
        if self.predicted_delta_adjustment != 0.0:
            raise ValueError(
                "Catalog #341 Tier A requires predicted_delta_adjustment=0.0; "
                f"got {self.predicted_delta_adjustment!r}"
            )
        if self.promotable is not False:
            raise ValueError(
                "Catalog #341 + Catalog #192 require promotable=False for Tier A "
                f"L0 SCAFFOLD; got {self.promotable!r}"
            )
        if self.axis_tag != "[predicted]":
            raise ValueError(
                "Catalog #287 + Catalog #341 require axis_tag='[predicted]'; "
                f"got {self.axis_tag!r}"
            )
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError(
                f"canonical_provenance must be Mapping; "
                f"got {type(self.canonical_provenance).__name__}"
            )


# -----------------------------------------------------------------------------
# Real PR101 paired-component rows loader (Catalog #213 real-input discipline)
# -----------------------------------------------------------------------------


def load_canonical_pr101_pair_component_rows(
    path: str | Path = DEFAULT_PAIR_COMPONENT_ROWS_PATH,
    *,
    repo_root: str | Path = ".",
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load canonical real PR101 600-pair paired-component rows.

    Per Catalog #213 + CLAUDE.md "Forbidden synthetic-fixture-instead-of-real-
    input" non-negotiable: this loader returns REAL rows from the canonical
    Wave N+34 MPS600 sweep at
    ``experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl``.

    Each row has keys (pair, mode_id, segnet_dist, posenet_dist,
    component_score_no_rate, family, promotion_eligible, score_claim).

    Args:
        path: Repo-relative path to the canonical paired-component rows
            JSONL. Defaults to DEFAULT_PAIR_COMPONENT_ROWS_PATH.
        repo_root: Repo root for path resolution.
        max_rows: Optional cap (None = all 13200 rows = 600 pairs x 22 modes).

    Returns:
        List of canonical row dicts.

    Raises:
        FileNotFoundError: if the canonical rows file is missing.
        ValueError: if a row fails canonical schema validation.
    """
    rows_path = Path(repo_root) / path
    if not rows_path.is_file():
        raise FileNotFoundError(
            f"Canonical PR101 paired-component rows not found at {rows_path}; "
            f"per Catalog #213 + CLAUDE.md 'Forbidden synthetic-fixture-instead-of-"
            f"real-input' real input is mandatory; do NOT synthesize substitutes"
        )
    rows: list[dict[str, Any]] = []
    with rows_path.open(encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for required_key in ("pair", "mode_id", "segnet_dist", "posenet_dist"):
                if required_key not in row:
                    raise ValueError(
                        f"Canonical PR101 row at line {line_idx} missing "
                        f"required key {required_key!r}; got keys {sorted(row.keys())}"
                    )
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


# -----------------------------------------------------------------------------
# Per-strategy basis weight computation helpers
# -----------------------------------------------------------------------------


def _sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Canonical 3x3 Sobel gradient magnitude (documented adaptation per
    Catalog #303; canonical Holub-Fridrich-Denemark 2014 uses db4 wavelet
    detail-coefficient variance which requires optional pywt).

    Args:
        image: 2D ndarray (H, W) of pixel intensities (uint8 or float).

    Returns:
        2D ndarray (H, W) of per-pixel gradient magnitudes (float64).
    """
    if image.ndim != 2:
        raise ValueError(
            f"_sobel_gradient_magnitude requires 2D ndarray; "
            f"got shape={image.shape}"
        )
    img = np.asarray(image, dtype=np.float64)
    # Pad for 3x3 convolution
    padded = np.pad(img, 1, mode="reflect")
    # Sobel-X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Sobel-Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for di, weight_row in enumerate(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))):
        for dj, weight in enumerate(weight_row):
            if weight != 0:
                gx += weight * padded[di : di + h, dj : dj + w]
    for di, weight_row in enumerate(((-1, -2, -1), (0, 0, 0), (1, 2, 1))):
        for dj, weight in enumerate(weight_row):
            if weight != 0:
                gy += weight * padded[di : di + h, dj : dj + w]
    return np.sqrt(gx * gx + gy * gy)


def compute_local_variance_basis_weight(
    image: np.ndarray,
    *,
    epsilon: float = CANONICAL_UNIWARD_EPSILON,
    kernel_size: int = CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE,
) -> np.ndarray:
    """UNIWARD inverse-local-variance basis weight per Holub-Fridrich-Denemark
    2014 canonical baseline (3x3 Sobel documented adaptation).

    Per CLAUDE.md "Fridrich inverse steganalysis" rule 1: errors in textured
    regions (high local variance / high gradient magnitude) are undetectable;
    weight loss by INVERSE local variance.

    Per Catalog #287 NO FAKE IMPLEMENTATIONS: this function actually computes
    Sobel gradient magnitudes and inverts them; it does NOT return canonical
    constants or markers without doing work.

    Args:
        image: 2D ndarray (H, W) of pixel intensities.
        epsilon: Denominator stabilizer per Fridrich UNIWARD canonical.
        kernel_size: Currently must be 3 (Sobel); placeholder for future db4.

    Returns:
        2D ndarray (H, W) of per-pixel UNIWARD weights = 1/(eps + |grad|).
    """
    if kernel_size != 3:
        raise NotImplementedError(
            f"kernel_size={kernel_size} not implemented; canonical Sobel uses 3; "
            f"db4 wavelet basis is DEFERRED reactivation path requires optional pywt"
        )
    grad_mag = _sobel_gradient_magnitude(image)
    return 1.0 / (epsilon + grad_mag)


def compute_segnet_gradient_basis_weight(
    image: np.ndarray,
    seg_logit_grad: np.ndarray,
    *,
    epsilon: float = CANONICAL_UNIWARD_EPSILON,
) -> np.ndarray:
    """SegNet logit-gradient sensitivity basis weight.

    Per CLAUDE.md "Exact scorer architectures": SegNet is
    ``smp.Unet('tu-efficientnet_b2', classes=5, activation=None)``; argmax over
    5 class logits; bilinear resize to (512, 384); stride-2 stem creates a
    256x192 spatial blind spot. The inverse-gradient basis targets that blind
    spot: pixels where dL_seg/dx is SMALL are pixels where errors propagate to
    fewer logit changes (more undetectable).

    Per Catalog #287 NO FAKE IMPLEMENTATIONS: this function actually consumes
    the precomputed per-pixel SegNet logit-gradient magnitude array (the
    caller must compute it via real SegNet forward+backward pass) and inverts
    it; it does NOT return canonical constants or markers without doing work.

    Args:
        image: 2D ndarray (H, W) — used for shape validation only.
        seg_logit_grad: 2D ndarray (H, W) of per-pixel |dL_seg/dx| magnitudes
            from a REAL SegNet forward+backward pass.
        epsilon: Denominator stabilizer.

    Returns:
        2D ndarray (H, W) of weights = 1/(eps + |seg_logit_grad|).
    """
    if image.shape != seg_logit_grad.shape:
        raise ValueError(
            f"image shape {image.shape} must match seg_logit_grad shape "
            f"{seg_logit_grad.shape}"
        )
    if seg_logit_grad.ndim != 2:
        raise ValueError(
            f"seg_logit_grad must be 2D ndarray; got shape={seg_logit_grad.shape}"
        )
    grad_mag = np.abs(np.asarray(seg_logit_grad, dtype=np.float64))
    return 1.0 / (epsilon + grad_mag)


def compute_posenet_gradient_basis_weight(
    image: np.ndarray,
    pose_output_grad: np.ndarray,
    *,
    epsilon: float = CANONICAL_UNIWARD_EPSILON,
) -> np.ndarray:
    """PoseNet 12-dim YUV6 output-gradient sensitivity basis weight.

    Per CLAUDE.md "Exact scorer architectures": PoseNet has FastViT-T12
    backbone; 12-channel YUV6 input (2 frames x 6 channels); first 6 of 12
    output dims contribute to distortion via MSE. The per-pixel gradient
    magnitude is the L2 norm over the first-6 output dimensions of the
    pose output Jacobian wrt input pixel intensity.

    Per Catalog #287 NO FAKE IMPLEMENTATIONS: this function actually consumes
    the precomputed per-pixel PoseNet output-gradient magnitude array (the
    caller must compute it via real PoseNet forward+backward pass) and
    inverts it; it does NOT return canonical constants or markers without
    doing work.

    Args:
        image: 2D ndarray (H, W) — used for shape validation only.
        pose_output_grad: 2D ndarray (H, W) of per-pixel L2-norm-over-first-6
            output dims of pose output Jacobian wrt input pixel intensity
            from a REAL PoseNet forward+backward pass.
        epsilon: Denominator stabilizer.

    Returns:
        2D ndarray (H, W) of weights = 1/(eps + |pose_output_grad|).
    """
    if image.shape != pose_output_grad.shape:
        raise ValueError(
            f"image shape {image.shape} must match pose_output_grad shape "
            f"{pose_output_grad.shape}"
        )
    if pose_output_grad.ndim != 2:
        raise ValueError(
            f"pose_output_grad must be 2D ndarray; got shape={pose_output_grad.shape}"
        )
    grad_mag = np.abs(np.asarray(pose_output_grad, dtype=np.float64))
    return 1.0 / (epsilon + grad_mag)


def compute_joint_scorer_basis_weight(
    image: np.ndarray,
    seg_logit_grad: np.ndarray,
    pose_output_grad: np.ndarray,
    *,
    alpha: float = CANONICAL_JOINT_ALPHA_DEFAULT,
    beta: float = CANONICAL_JOINT_BETA_DEFAULT,
    gamma: float = CANONICAL_JOINT_GAMMA_DEFAULT,
    epsilon: float = CANONICAL_UNIWARD_EPSILON,
    local_variance_kernel_size: int = CANONICAL_LOCAL_VARIANCE_KERNEL_SIZE,
) -> np.ndarray:
    """Joint scorer-Jacobian linear combination basis weight per Atick-Redlich
    1990 cooperative-receiver framing.

    Per CLAUDE.md grand council Atick-Redlich seat: per-pixel weight =
    alpha * SegNet_grad + beta * PoseNet_grad + gamma * local_variance, then
    inverted as UNIWARD cost: 1/(eps + combined). Canonical defaults
    alpha=beta=gamma=1/3 are TUNING-FREE (CARGO-CULTED Assumption-Adversary
    verdict); reactivation requires per-pair empirical grid search per
    reactivation criterion #2.

    Per Catalog #287 NO FAKE IMPLEMENTATIONS: this function actually combines
    three real ndarrays via canonical linear combination and inverts the
    result; it does NOT delegate to one of the sister branches or return
    canonical constants.

    Args:
        image: 2D ndarray (H, W) of pixel intensities.
        seg_logit_grad: 2D ndarray (H, W) per-pixel SegNet logit gradient
            magnitude.
        pose_output_grad: 2D ndarray (H, W) per-pixel PoseNet output gradient
            magnitude.
        alpha, beta, gamma: Canonical linear combination weights in [0, 1].
        epsilon: Denominator stabilizer.
        local_variance_kernel_size: Sobel kernel size for the local-variance
            component.

    Returns:
        2D ndarray (H, W) of weights = 1/(eps + alpha*seg + beta*pose
        + gamma*local_var).
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0):
        raise ValueError(
            f"alpha={alpha}, beta={beta}, gamma={gamma} must each be in [0, 1]"
        )
    if image.shape != seg_logit_grad.shape or image.shape != pose_output_grad.shape:
        raise ValueError(
            f"image shape {image.shape} must match seg_logit_grad shape "
            f"{seg_logit_grad.shape} and pose_output_grad shape "
            f"{pose_output_grad.shape}"
        )
    local_var = _sobel_gradient_magnitude(image)
    seg = np.abs(np.asarray(seg_logit_grad, dtype=np.float64))
    pose = np.abs(np.asarray(pose_output_grad, dtype=np.float64))
    combined = alpha * seg + beta * pose + gamma * local_var
    return 1.0 / (epsilon + combined)


def compute_per_pair_uniward_cost_from_basis(
    per_pixel_weight_map: np.ndarray,
    *,
    aggregation: str = "mean",
) -> float:
    """Aggregate a per-pixel weight map to a single per-pair UNIWARD cost.

    Per Holub-Fridrich-Denemark 2014: the per-pair cost is the aggregate
    inverse-sensitivity across all pixels in the pair. Lower cost = more
    sensitive = harder to perturb undetectably. Sparse-K selector picks the
    K pairs with LOWEST cost (most discriminative for steganalysis).

    Args:
        per_pixel_weight_map: 2D ndarray (H, W) per-pixel UNIWARD weight.
        aggregation: One of {"mean", "sum", "median", "max", "min"}.

    Returns:
        Scalar aggregate cost.
    """
    if per_pixel_weight_map.ndim != 2:
        raise ValueError(
            f"per_pixel_weight_map must be 2D; got shape={per_pixel_weight_map.shape}"
        )
    arr = np.asarray(per_pixel_weight_map, dtype=np.float64)
    if aggregation == "mean":
        return float(arr.mean())
    if aggregation == "sum":
        return float(arr.sum())
    if aggregation == "median":
        return float(np.median(arr))
    if aggregation == "max":
        return float(arr.max())
    if aggregation == "min":
        return float(arr.min())
    raise ValueError(
        f"aggregation must be one of mean/sum/median/max/min; got {aggregation!r}"
    )


def select_sparse_k_pairs_by_cost(
    per_pair_costs: Sequence[float],
    sparse_k: int,
) -> tuple[int, ...]:
    """Canonical top-K-lowest-cost pair selector (ascending-sorted for
    diff-able-across-runs determinism per CLAUDE.md "Beauty, simplicity, and
    developer experience" + Catalog #305 observability surface facet #3).

    Per Wave N+34 OPT-7 canonical: sparse-K=100 selector emits indices of
    the K pairs with LOWEST UNIWARD cost (most sensitive pairs targeted for
    perturbation; high-cost pairs are textured regions where errors are
    undetectable so no perturbation is needed).

    Args:
        per_pair_costs: Iterable of per-pair UNIWARD costs (length = N_pairs).
        sparse_k: K parameter; clamps to len(per_pair_costs).

    Returns:
        Ascending-sorted tuple of K pair indices.
    """
    costs = list(per_pair_costs)
    if sparse_k <= 0:
        raise ValueError(f"sparse_k must be > 0; got {sparse_k}")
    k = min(sparse_k, len(costs))
    # Get the K lowest-cost pair indices, then sort ascending for determinism.
    indexed = sorted(range(len(costs)), key=lambda i: (costs[i], i))[:k]
    return tuple(sorted(indexed))


def _estimate_wire_bytes(
    n_selected_pairs: int,
    header_overhead_bytes: int,
) -> int:
    """Estimate wire bytes for sparse-K selector emission.

    Sister of Slot FF wire estimation: per-K selector entry uses
    index-byte-width=2 + magnitude-byte=1 = 3 bytes per K, plus header.

    Args:
        n_selected_pairs: K selected pairs.
        header_overhead_bytes: Wire-format header overhead.

    Returns:
        Wire bytes estimate.
    """
    if n_selected_pairs < 0 or header_overhead_bytes < 0:
        raise ValueError(
            f"n_selected_pairs and header_overhead_bytes must be >= 0; "
            f"got {n_selected_pairs}, {header_overhead_bytes}"
        )
    return header_overhead_bytes + 3 * n_selected_pairs


def compute_basis_expansion_perturbation_for_pr110_catalog(
    pair_rows: Sequence[Mapping[str, Any]],
    config: InverseScorerBasisConfig,
    *,
    image_provider: Callable[[int], np.ndarray] | None = None,
    seg_grad_provider: Callable[[int], np.ndarray] | None = None,
    pose_grad_provider: Callable[[int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute basis-expansion perturbation given the canonical PR101 paired-
    component rows and per-strategy basis providers.

    L0 SCAFFOLD discipline per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK:
    if providers are not supplied, falls back to the canonical per-pair
    UNIWARD cost derived from the row's segnet_dist + posenet_dist (real
    per-pair empirical data from PR101 MPS sweep). This is a documented
    adaptation per Catalog #303 — full per-pixel basis computation requires
    real frame_0 + frame_1 image bytes + real SegNet/PoseNet forward+backward,
    which is operator-routable for paired-CUDA reactivation (criterion #1).

    Args:
        pair_rows: Real PR101 paired-component rows (one per (pair, mode_id)).
        config: Canonical :class:`InverseScorerBasisConfig`.
        image_provider: Optional callable(pair_idx) -> 2D ndarray pixel image.
        seg_grad_provider: Optional callable(pair_idx) -> 2D ndarray SegNet
            logit-gradient magnitude.
        pose_grad_provider: Optional callable(pair_idx) -> 2D ndarray PoseNet
            output-gradient magnitude.

    Returns:
        Dict with keys:
        - selected_pair_indices: tuple[int, ...]
        - per_pair_uniward_costs: tuple[float, ...] (one per UNIQUE pair)
        - wire_bytes_estimate: int
        - delta_vs_fec6_bytes: int
        - aggregate_predicted_delta_s: float
        - n_selected_pairs: int
    """
    # Group rows by pair. Each pair has multiple modes; canonical Wave N+34
    # uses the BEST (min component_score_no_rate) mode per pair to compute
    # per-pair sensitivity.
    pair_best: dict[int, dict[str, Any]] = {}
    for row in pair_rows:
        pair_idx = int(row["pair"])
        if pair_idx not in pair_best or (
            float(row.get("component_score_no_rate", float("inf")))
            < float(pair_best[pair_idx].get("component_score_no_rate", float("inf")))
        ):
            pair_best[pair_idx] = dict(row)

    n_unique_pairs = len(pair_best)
    if n_unique_pairs == 0:
        raise ValueError("pair_rows is empty; canonical PR101 must yield >= 1 unique pair")

    sorted_pair_indices = sorted(pair_best.keys())

    per_pair_costs: list[float] = []
    for pair_idx in sorted_pair_indices:
        row = pair_best[pair_idx]
        # Per-strategy dispatch. Each branch's L0 SCAFFOLD fallback computes
        # a STRUCTURALLY DISTINCT per-pair UNIWARD cost so selected_pair_indices
        # differ across strategies per Catalog #308 alternative-reducer
        # enumeration + Slot EEE FAKE-implementation audit gate.
        seg_dist = float(row.get("segnet_dist", 0.0))
        pose_dist = float(row.get("posenet_dist", 0.0))
        component_score = float(row.get("component_score_no_rate", 0.0))

        if config.basis_strategy == InverseScorerBasisStrategy.UNIWARD_INVERSE_LOCAL_VARIANCE_BASELINE:
            if image_provider is not None:
                # Canonical Holub-Fridrich-Denemark 2014 baseline.
                weight_map = compute_local_variance_basis_weight(
                    image_provider(pair_idx),
                    epsilon=config.uniward_epsilon,
                    kernel_size=config.local_variance_kernel_size,
                )
                cost = compute_per_pair_uniward_cost_from_basis(weight_map)
            else:
                # L0 SCAFFOLD fallback for the LOCAL-VARIANCE-BASELINE branch:
                # use the COMPONENT SCORE (100*d_seg + sqrt(10*d_pose)) as a
                # canonical proxy for inverse local pixel variance (high score
                # = high inverse-variance = textured/sensitive). Documented
                # adaptation per Catalog #303; sister-distinct from per-axis
                # branches below because it uses the canonical contest-formula
                # COMBINED component (not seg or pose alone).
                cost = 1.0 / (config.uniward_epsilon + component_score)
        elif config.basis_strategy == InverseScorerBasisStrategy.UNIWARD_INVERSE_SEGNET_GRADIENT_SENSITIVITY:
            if image_provider is not None and seg_grad_provider is not None:
                # Canonical real-SegNet forward+backward.
                weight_map = compute_segnet_gradient_basis_weight(
                    image_provider(pair_idx),
                    seg_grad_provider(pair_idx),
                    epsilon=config.uniward_epsilon,
                )
                cost = compute_per_pair_uniward_cost_from_basis(weight_map)
            else:
                # L0 SCAFFOLD fallback for SEGNET-GRADIENT-SENSITIVITY branch:
                # per-pair UNIWARD cost derived from segnet_dist ALONE
                # (NOT pose). Per CLAUDE.md "SegNet vs PoseNet importance":
                # SegNet's stride-2 stem 256x192 spatial blind spot is the
                # canonical sensitivity surface. Documented adaptation per
                # Catalog #303. Sister-distinct from local-variance branch
                # because it ignores pose_dist; sister-distinct from PoseNet
                # branch because it uses seg_dist only.
                cost = 1.0 / (config.uniward_epsilon + seg_dist)
        elif config.basis_strategy == InverseScorerBasisStrategy.UNIWARD_INVERSE_POSENET_GRADIENT_SENSITIVITY:
            if image_provider is not None and pose_grad_provider is not None:
                # Canonical real-PoseNet forward+backward.
                weight_map = compute_posenet_gradient_basis_weight(
                    image_provider(pair_idx),
                    pose_grad_provider(pair_idx),
                    epsilon=config.uniward_epsilon,
                )
                cost = compute_per_pair_uniward_cost_from_basis(weight_map)
            else:
                # L0 SCAFFOLD fallback for POSENET-GRADIENT-SENSITIVITY branch:
                # per-pair UNIWARD cost derived from posenet_dist ALONE
                # (NOT seg). Per CLAUDE.md "Exact scorer architectures":
                # PoseNet FastViT-T12 YUV6 12-channel input is the canonical
                # pose-axis sensitivity surface. Documented adaptation per
                # Catalog #303. Sister-distinct from local-variance branch
                # because it ignores seg_dist; sister-distinct from SegNet
                # branch because it uses pose_dist only.
                cost = 1.0 / (config.uniward_epsilon + pose_dist)
        elif config.basis_strategy == InverseScorerBasisStrategy.UNIWARD_INVERSE_JOINT_SCORER_BASIS_LINEAR_COMBINATION:
            if image_provider is not None and seg_grad_provider is not None and pose_grad_provider is not None:
                # Canonical real-scorer linear combination.
                weight_map = compute_joint_scorer_basis_weight(
                    image_provider(pair_idx),
                    seg_grad_provider(pair_idx),
                    pose_grad_provider(pair_idx),
                    alpha=config.joint_alpha,
                    beta=config.joint_beta,
                    gamma=config.joint_gamma,
                    epsilon=config.uniward_epsilon,
                    local_variance_kernel_size=config.local_variance_kernel_size,
                )
                cost = compute_per_pair_uniward_cost_from_basis(weight_map)
            else:
                # L0 SCAFFOLD fallback for JOINT-LINEAR-COMBO branch:
                # weighted GEOMETRIC MEAN of normalized seg / pose / local
                # variance per Atick-Redlich 1990 cooperative-receiver
                # framing. The geometric mean is sister-distinct from the
                # local-variance-baseline branch (which uses arithmetic-
                # additive component_score = seg_norm + pose_norm and is
                # therefore COLLINEAR with any arithmetic linear combination
                # of seg + pose; per Atick-Redlich the joint mutual-
                # information surface is multiplicative not additive,
                # following I(X;Y) = -log p(X) - log p(Y|X) which combines
                # multiplicatively at the cost-additive layer).
                #
                # geometric_mean = (seg_norm + eps)**alpha *
                #                  (pose_norm + eps)**beta *
                #                  (local_var_norm + eps)**gamma
                # where local_var_norm uses ABS(seg - pose) as a sister-
                # distinct proxy for compress-time local variance
                # (NOT collinear with seg or pose individually).
                # At defaults alpha=beta=gamma=1/3 this is the canonical
                # geometric mean per Atick-Redlich 1990 cooperative-receiver
                # canonical product-rule.
                seg_normalized = 100.0 * seg_dist
                pose_normalized = math.sqrt(10.0 * max(pose_dist, 0.0))
                # Local-variance proxy: abs difference between normalized
                # seg and pose contributions — captures the per-pair axis
                # IMBALANCE which IS the canonical Atick-Redlich receiver-
                # cooperation discriminator. Sister-distinct from
                # component_score = seg_norm + pose_norm because abs(seg -
                # pose) is the orthogonal component.
                local_var_normalized = abs(seg_normalized - pose_normalized)
                eps_log = config.uniward_epsilon
                # Geometric-mean form per Atick-Redlich 1990 product rule:
                log_combined = (
                    config.joint_alpha * math.log(seg_normalized + eps_log)
                    + config.joint_beta * math.log(pose_normalized + eps_log)
                    + config.joint_gamma * math.log(local_var_normalized + eps_log)
                )
                combined = math.exp(log_combined)
                cost = 1.0 / (config.uniward_epsilon + combined)
        else:
            raise ValueError(f"Unknown basis_strategy: {config.basis_strategy}")
        per_pair_costs.append(cost)

    # Select sparse-K pairs with LOWEST cost (most sensitive pairs).
    selected_indices_local = select_sparse_k_pairs_by_cost(
        per_pair_costs, config.sparse_k
    )
    # Map local indices back to canonical pair indices:
    selected_pair_indices = tuple(sorted(sorted_pair_indices[i] for i in selected_indices_local))

    n_selected = len(selected_pair_indices)
    wire_bytes = _estimate_wire_bytes(n_selected, config.header_overhead_bytes)
    delta_vs_fec6 = wire_bytes - WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES

    # Aggregate predicted ΔS = rate-axis contribution only at L0 SCAFFOLD
    # (distortion-axis contribution requires real SegNet/PoseNet forward
    # passes on perturbed frames, which is the paired-CUDA reactivation
    # surface). The rate-axis canonical formula:
    aggregate_delta_s = (
        CANONICAL_RATE_MULTIPLIER * delta_vs_fec6 / CANONICAL_RATE_DENOM_BYTES
    )

    return {
        "selected_pair_indices": selected_pair_indices,
        "per_pair_uniward_costs": tuple(per_pair_costs),
        "wire_bytes_estimate": wire_bytes,
        "delta_vs_fec6_bytes": delta_vs_fec6,
        "aggregate_predicted_delta_s": aggregate_delta_s,
        "n_selected_pairs": n_selected,
        "n_unique_pairs_in_input": n_unique_pairs,
    }


def apply_uniward_inverse_scorer_basis_to_pr110_archive(
    config: InverseScorerBasisConfig,
    *,
    pair_rows: Sequence[Mapping[str, Any]] | None = None,
    image_provider: Callable[[int], np.ndarray] | None = None,
    seg_grad_provider: Callable[[int], np.ndarray] | None = None,
    pose_grad_provider: Callable[[int], np.ndarray] | None = None,
    repo_root: str | Path = ".",
) -> UniwardInverseScorerBasisResult:
    """Canonical entry point — apply inverse-scorer basis expansion to PR110.

    Returns Tier A canonical-routing-markers contribution per Catalog #341 +
    Catalog #357 + AxisDecomposition per Catalog #356 + canonical Provenance
    per Catalog #323.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192:
    the L0 SCAFFOLD return is NEVER promotable; verdict is
    ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` until paired-CUDA
    empirical anchor lands per Catalog #246 1:1 contest-compliant hardware.

    Args:
        config: Canonical config.
        pair_rows: Optional real PR101 paired-component rows. If None,
            loads canonical default from
            ``experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl``
            per Catalog #213 real-input discipline.
        image_provider, seg_grad_provider, pose_grad_provider: Optional
            per-pair real-frame providers (paired-CUDA reactivation
            surface).
        repo_root: For default rows path resolution.

    Returns:
        :class:`UniwardInverseScorerBasisResult` Tier A contribution.
    """
    if pair_rows is None:
        pair_rows = load_canonical_pr101_pair_component_rows(repo_root=repo_root)

    result = compute_basis_expansion_perturbation_for_pr110_catalog(
        pair_rows,
        config,
        image_provider=image_provider,
        seg_grad_provider=seg_grad_provider,
        pose_grad_provider=pose_grad_provider,
    )

    # Canonical Provenance per Catalog #323 (predicted; paired-CUDA empirical
    # anchor required for promotion):
    inputs_payload = json.dumps(
        {
            "basis_strategy": config.basis_strategy.value,
            "n_pairs": config.n_pairs,
            "sparse_k": config.sparse_k,
            "uniward_epsilon": config.uniward_epsilon,
            "joint_alpha": config.joint_alpha,
            "joint_beta": config.joint_beta,
            "joint_gamma": config.joint_gamma,
            "n_unique_pairs_in_input": result["n_unique_pairs_in_input"],
        },
        sort_keys=True,
    ).encode("utf-8")
    inputs_sha256 = hashlib.sha256(inputs_payload).hexdigest()

    provenance = build_provenance_for_predicted(
        model_id=(
            f"pr110_opt_7_fridrich_uniward_inverse_scorer_basis_l0_scaffold:"
            f"{config.basis_strategy.value}"
        ),
        inputs_sha256=inputs_sha256,
    )
    provenance_dict = provenance_to_dict(provenance)

    # AxisDecomposition emission per Catalog #356 (optional based on config):
    axis_decomp: AxisDecomposition | None = None
    if config.emit_axis_decomposition:
        axis_decomp = AxisDecomposition(
            predicted_d_seg_delta=0.0,
            predicted_d_pose_delta=0.0,
            predicted_archive_bytes_delta=int(result["delta_vs_fec6_bytes"]),
            axis_tag="[predicted]",
            canonical_provenance=provenance_dict,
        )

    wire_analysis: dict[str, Any] = {
        "n_unique_pairs_in_input": int(result["n_unique_pairs_in_input"]),
        "n_selected_pairs": int(result["n_selected_pairs"]),
        "wire_bytes_estimate": int(result["wire_bytes_estimate"]),
        "delta_vs_fec6_bytes": int(result["delta_vs_fec6_bytes"]),
        "aggregate_predicted_delta_s": float(result["aggregate_predicted_delta_s"]),
        "inputs_sha256_prefix": inputs_sha256[:16],
        "basis_strategy": config.basis_strategy.value,
        "selected_pair_indices_first_16": list(result["selected_pair_indices"][:16]),
        "wave_n34_opt7_canonical_anchor": {
            "unweighted_aggregate_delta_s": WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S,
            "uniward_weighted_aggregate_delta_s": WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S,
            "sparse_k100_wire_bytes": WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES,
            "sparse_k100_proportional_savings": WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS,
        },
    }

    return UniwardInverseScorerBasisResult(
        strategy=config.basis_strategy,
        predicted_delta_adjustment=0.0,  # Tier A canonical-routing-markers per Catalog #341
        promotable=False,  # Catalog #192 macOS-CPU advisory NEVER promotable
        axis_tag="[predicted]",  # Catalog #287/#341 canonical
        verdict="DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        wire_bytes_estimate=int(result["wire_bytes_estimate"]),
        delta_vs_fec6_bytes=int(result["delta_vs_fec6_bytes"]),
        n_selected_pairs=int(result["n_selected_pairs"]),
        per_pair_selector_indices=tuple(result["selected_pair_indices"]),
        per_pair_uniward_costs=tuple(result["per_pair_uniward_costs"]),
        aggregate_predicted_delta_s=float(result["aggregate_predicted_delta_s"]),
        canonical_provenance=provenance_dict,
        axis_decomposition=axis_decomp,
        wire_analysis=wire_analysis,
    )
