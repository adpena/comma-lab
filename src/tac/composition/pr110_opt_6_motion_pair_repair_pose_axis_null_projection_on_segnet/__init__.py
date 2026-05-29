# SPDX-License-Identifier: MIT
"""PR110-OPT-6 Motion-pair repair: pose-axis perturbation that null-projects on SegNet — L0 SCAFFOLD.

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
+ task #1318 PR110-OPT-6 + Slot RR cap≥3 parallel-cascade directive 2026-05-29
+ canonical sister of Slot MM canonical pose-axis null-byte exploitation finding
(canonical PARADIGM intact regardless of Slot QQ IMPLEMENTATION-LEVEL falsification
of Slot MM's quantitative ΔS=-0.021862 cross-substrate prediction).

Design memo (single source of truth)::

    .omx/research/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_design_20260529.md

Canonical context
=================

Fridrich-Yousfi inverse-steganalysis canonical pose-axis null-projection axis:
identify motion-pair perturbations to frame-1 (the canonical motion-pair successor
frame relative to frame-0) whose canonical:

* ``argmax(SegNet(frame_1_perturbed)) == argmax(SegNet(frame_1_baseline))`` — d_seg = 0.0
  (canonical SegNet-null axis; argmax invariant per CLAUDE.md "Exact scorer architectures")
* ``d_pose ∈ [1e-7, 1e-5]`` — canonical pose-axis carrier
  (canonical sister of OPT-12 PoseNet-null bottom-decile catalog 2026-05-26)

The canonical menu enumerates 4 candidate families per Catalog #308 alternative-reducer
enumeration:

* Single-pixel rolls frame-1: 8 modes ``(dx, dy) ∈ {-1, 0, +1}² \\ {(0,0)}``
* DCT-II sign basis frame-1: 16 modes (8 frequency bins × 2 amplitudes)
* Hadamard tile frame-1: 3 modes (Sylvester 8×8 amp{1,2,3})
* Gaussian noise frame-1: 16 modes (σ ∈ {0.5, 1.0, 1.5, 2.0} × seeds {1,2,3,4})

Total canonical menu: 43 frame-1 modes (canonical sister of OPT-1 87-candidate frame-0
catalog landed 2026-05-26 per ``.omx/research/pr110_opt_frame0_bundle_landed_20260526.md``).

PR110 archive grammar (sister of Slot X PR110-OPT-4 + Slot FF PR110-OPT-7 L0 SCAFFOLDs)::

* 16-symbol K=16 selector palette per FEC6 ``submissions/hnerv_fec6_fixed_huffman_k16/``
* 600 per-pair selectors (one selector per source-frame pair)
* 6-byte header + 243-byte 0-order fixed-Huffman bitstream = **249 byte baseline wire**

L0 SCAFFOLD role
================

THIS module serves the canonical dual role per Slot X + Slot FF sister-pattern template:

1. **Preserve the canonical Fridrich-Yousfi inverse-steganalysis pose-axis null-projection
   axis** as a queryable system surface so future widened-K + per-region + per-temporal-window
   probes can compare without re-deriving the canonical 4-family menu construction.

2. **Enumerate alternative basis-expansion methodologies** per Catalog #308 so the
   operator can route the next iteration through one of N=4 candidates (the canonical
   :class:`PoseAxisNullProjectionStrategy` enum).

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_pose_axis_null_projection_to_pr110_archive` entry point returns a Tier A
canonical-routing-markers contribution per Catalog #341: ``predicted_delta_adjustment=0.0``
+ ``promotable=False`` + ``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

* :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose, archive bytes)
  decomposition with canonical Provenance dict-form).
* :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted` (the L0 SCAFFOLD
  predicts via canonical OPT-12 PoseNet-null analog symmetry; empirical anchor
  required for promotion per Catalog #246 paired-CUDA RATIFICATION).
* Tier A canonical-routing markers per Catalog #341 + #357 (canonical predicted
  ``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag="[predicted]"``).
* HNeRV parity discipline L4 (≤200 LOC inflate budget; this L0 SCAFFOLD has zero
  inflate-time code — encoder-side only).
* HNeRV parity discipline L7 (bolt-on size budget ≤350 LOC; this L0 SCAFFOLD is
  ~340 LOC including docstring + canonical menu constants per CATHEDRAL-SMARTER-DESIGN-MEMO
  ≤250 effective implementation).
* Catalog #309 ``horizon_class: plateau_adjacent``.
* Catalog #311 ego-motion-conditioned non-negotiable (canonical pose-axis perturbations
  canonically tied to PoseNet's ego-motion-conditioned response per CLAUDE.md "Exact
  scorer architectures": PoseNet input is 2-frame YUV6 with first 6 pose dimensions
  measuring ego-motion).

Sister cross-references
=======================

* Slot X PR110-OPT-4 L0 SCAFFOLD (commit ``0eb7cb615``):
  ``src/tac/composition/pr110_opt_4_grouped_color_geometry_calibration/__init__.py``
* Slot FF PR110-OPT-7 L0 SCAFFOLD (commit ``0adecdc5b``):
  ``src/tac/composition/pr110_opt_7_uniward_inverse_scorer_basis_expansion/__init__.py``
* Slot LL canonical zero-byte bolt-on (commit ``febe...``):
  ``src/tac/codec/pr98_channel_balance_zero_byte_bolt_on/__init__.py``
* PR110-OPT-1/12/13 canonical bundle landed 2026-05-26:
  ``.omx/research/pr110_opt_frame0_bundle_landed_20260526.md``
* PR110 canonical sweep primitive: ``tools/frame_exploit_segnet_posenet_sweep.py``

Slot QQ canonical IMPLEMENTATION-LEVEL falsification disclaimer
================================================================

Per Slot QQ in-flight checkpoint 2026-05-29T13:33:40Z + Catalog #307 paradigm-vs-
implementation classification: Slot MM's cross-substrate quantitative ΔS=-0.021862
prediction (pr106 claimed 16,909 nulls + pr107 claimed 15,987 nulls) was empirically
IMPLEMENTATION-LEVEL FALSIFIED (pr106 actual 665 nulls; pr107 actual 612 nulls; 0
bytes in >=2KB runs in BOTH archives). Canonical equation #26 PARADIGM
(procedural codebook from seed compression savings) INTACT. THIS L0 SCAFFOLD does
NOT cite Slot MM's quantitative anchor as authoritative; predicted ΔS band derived
INDEPENDENTLY from canonical Fridrich-Yousfi inverse-steganalysis duality + canonical
OPT-12 PoseNet-null analog symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


# --- Canonical OPT-12 PoseNet-null analog constants (HARD-EARNED 2026-05-26) ----

# Per ``.omx/research/pr110_opt_frame0_bundle_landed_20260526.md`` § 4.1
# OPT-12 top-3 PoseNet-null at frame-0 (canonical d_seg = 0.0 confirmed)::
#
# | rank | mode_id                                  | abs_pose_delta | seg_delta |
# |-----:|------------------------------------------|---------------:|----------:|
# | 1    | frame0_widened_dct_u1_v2_amp_1           | 1.25e-7        | 0.0       |
# | 2    | frame0_widened_blue_chroma_amp_2         | 3.30e-7        | 0.0       |
# | 3    | frame0_widened_dct_u1_v2_amp_2           | 3.47e-7        | 0.0       |
#
# Of the 8 candidates in the canonical pose-null decile, 4 are
# ``frame0_dct_chroma`` (50%) and 3 are ``frame0_blue_chroma``-family (37.5%).
# Structured signed 8×8 chroma patterns dominate the canonical PoseNet-null axis.
# Canonical sister at frame-1 (SegNet-null axis) expected to mirror by canonical
# Fridrich-Yousfi inverse-steganalysis duality.
OPT12_POSENET_NULL_TYPICAL_ABS_POSE_DELTA: float = 1.25e-7
"""Canonical typical |d_pose| for OPT-12 PoseNet-null bottom-decile (frame-0 analog)."""

OPT12_POSENET_NULL_DOMINANT_FAMILY_FRACTION: float = 0.875
"""Canonical fraction of pose-null decile dominated by structured-signed-chroma
(DCT + blue-chroma) families (4+3 of 8 = 7/8 = 0.875)."""

# Per § 3.6 design memo Dykstra-feasibility check + canonical OPT-12 analog symmetry.
PREDICTED_SCORE_DELTA_BAND_LOWER: float = -0.0010
PREDICTED_SCORE_DELTA_BAND_UPPER: float = -0.0001
"""Canonical predicted ΔS band [-0.0010, -0.0001] per Fridrich-Yousfi inverse-
steganalysis duality + OPT-12 PoseNet-null analog symmetry. Conservative band
per Catalog #296 Dykstra-feasibility check + L7 bolt-on budget."""

# Per PR110 archive grammar (sister of Slot X + Slot FF L0 SCAFFOLDs).
FEC6_BASELINE_WIRE_BYTES: int = 249
"""Canonical FEC6 baseline wire bytes (6-byte header + 243-byte 0-order
fixed-Huffman bitstream over 600 per-pair K=16 selectors)."""

FEC6_BASELINE_ARCHIVE_SHA_PREFIX: str = "b7106c9bdbb8"
"""Canonical FEC6 baseline archive sha prefix per canonical frontier pointer
per Catalog #343 (CPU 0.19198533626623068)."""

PR110_NUM_PAIRS: int = 600
"""Canonical PR110 number of per-pair selectors."""

PR110_K_SYMBOLS: int = 16
"""Canonical PR110 K=16 selector palette size."""

# --- Canonical menu family counts per § 3.1 ---------------------------------

CANONICAL_PIXEL_ROLL_FRAME1_COUNT: int = 8
"""Canonical single-pixel rolls frame-1: (dx, dy) ∈ {-1, 0, +1}² \\ {(0,0)}."""

CANONICAL_DCT_CHROMA_FRAME1_COUNT: int = 16
"""Canonical DCT-II sign basis frame-1: 8 frequency bins × 2 amplitudes."""

CANONICAL_HADAMARD_TILE_FRAME1_COUNT: int = 3
"""Canonical Hadamard tile frame-1: Sylvester 8×8 amp{1,2,3}."""

CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT: int = 16
"""Canonical Gaussian noise frame-1: σ ∈ {0.5, 1.0, 1.5, 2.0} × seeds {1,2,3,4}."""

CANONICAL_FRAME1_MENU_TOTAL: int = (
    CANONICAL_PIXEL_ROLL_FRAME1_COUNT
    + CANONICAL_DCT_CHROMA_FRAME1_COUNT
    + CANONICAL_HADAMARD_TILE_FRAME1_COUNT
    + CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT
)
"""Canonical total frame-1 motion-pair perturbation menu count: 43 modes
(canonical sister of OPT-1 87-candidate frame-0 widened catalog)."""

# --- Canonical Slot MM IMPLEMENTATION-LEVEL FALSIFICATION disclaimer constants ---

SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED: bool = True
"""Per Slot QQ canonical empirical IMPLEMENTATION-LEVEL FALSIFICATION 2026-05-29T13:33:40Z:
Slot MM's quantitative ΔS=-0.021862 cross-substrate prediction MUST NOT be cited as
authoritative for PR110-OPT-6. Canonical equation #26 PARADIGM intact (procedural codebook
from seed compression savings); canonical Fridrich-Yousfi inverse-steganalysis pose-axis
null-projection PARADIGM intact regardless of Slot MM-specific overlay falsification."""

SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC: str = "2026-05-29T13:33:40Z"
"""Canonical Slot QQ in-flight checkpoint timestamp documenting empirical
IMPLEMENTATION-LEVEL FALSIFICATION of Slot MM's quantitative anchor."""

CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT: bool = True
"""Per Catalog #307 paradigm-vs-implementation classification: canonical Fridrich-Yousfi
inverse-steganalysis pose-axis null-projection PARADIGM intact regardless of any single
quantitative anchor's empirical verdict."""

# --- Canonical Catalog #344 equation candidate ID ----------------------------

CANONICAL_EQUATION_CANDIDATE_ID: str = (
    "pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1"
)
"""Canonical equation candidate ID per Catalog #344 (operator-decision-pending per
'iterate not force' until first paired-CUDA empirical anchor lands)."""

CANONICAL_ANTI_PATTERN_CANDIDATE_ID: str = (
    "pr110_opt_6_motion_pair_repair_segnet_null_axis_implementation_falsified_v1"
)
"""Canonical anti-pattern candidate ID per Catalog #344 sister discipline (registered
ONLY if empirical smoke FALSIFIES; per Catalog #307 IMPLEMENTATION-LEVEL classification;
PARADIGM intact)."""


# --- Canonical strategy enum per Catalog #308 alternative-reducer enumeration ----


class PoseAxisNullProjectionStrategy(str, Enum):
    """Canonical alternative-reducer enumeration per Catalog #308.

    Per design memo § 3.1 four canonical menu families serving the canonical
    Fridrich-Yousfi inverse-steganalysis pose-axis null-projection axis.
    """

    PER_PIXEL_ROLL = "per_pixel_roll"
    """Single-pixel rolls frame-1: 8 modes (dx, dy) in {-1, 0, +1}² \\ {(0,0)}.
    Canonical rationale: SegNet bilinear resize (512, 384) inverts subpixel shifts;
    argmax invariant under 1-pixel translation."""

    DCT_CHROMA_BASIS = "dct_chroma_basis"
    """DCT-II sign basis frame-1: 16 modes (8 frequency bins × 2 amplitudes).
    Canonical rationale: OPT-12 analog frame0_dct_chroma (u=1, v=2) achieved
    |d_pose|=1.25e-7 (200× smaller than baseline) with d_seg=0.0."""

    HADAMARD_TILE = "hadamard_tile"
    """Hadamard tile frame-1: 3 modes Sylvester 8×8 amp{1,2,3}. Canonical
    rationale: Hadamard tiles preserve EfficientNet stride-2 stem invariants."""

    GAUSSIAN_NOISE = "gaussian_noise"
    """Gaussian noise frame-1: 16 modes σ ∈ {0.5, 1.0, 1.5, 2.0} × seeds {1,2,3,4}.
    Canonical rationale: UNIWARD principle — noise in textured regions is canonical
    undetectable axis."""


# --- Canonical configuration dataclass per Catalog #287 ----------------------


@dataclass(frozen=True)
class MotionPairRepairPoseAxisNullProjectionConfig:
    """Canonical configuration for PR110-OPT-6 motion-pair repair pose-axis
    null-projection on SegNet per Catalog #287 + Catalog #323 canonical Provenance.
    """

    substrate_id: str
    """Canonical substrate identifier (e.g. ``hnerv_fec6_fixed_huffman_k16``)."""

    strategy: PoseAxisNullProjectionStrategy = PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL
    """Canonical strategy per Catalog #308 alternative-reducer enumeration."""

    d_seg_epsilon: float = 1e-9
    """Canonical d_seg tolerance for SegNet-null filter (canonical exact zero
    within floating-point epsilon)."""

    target_d_pose_lower: float = 1e-7
    """Canonical lower bound of d_pose carrier band per OPT-12 PoseNet-null analog."""

    target_d_pose_upper: float = 1e-5
    """Canonical upper bound of d_pose carrier band per L7 bolt-on budget."""

    emit_axis_decomposition: bool = True
    """Canonical per-axis AxisDecomposition emission per Catalog #356."""

    def __post_init__(self) -> None:
        """Validate canonical invariants per Catalog #287 + #323 canonical Provenance."""
        if not isinstance(self.substrate_id, str) or not self.substrate_id.strip():
            raise ValueError(
                "substrate_id must be non-empty string per Catalog #287 placeholder rejection"
            )
        if "<" in self.substrate_id and ">" in self.substrate_id:
            raise ValueError(
                f"substrate_id={self.substrate_id!r} is placeholder literal "
                "(rejected per Catalog #287 sister discipline)"
            )
        if not isinstance(self.strategy, PoseAxisNullProjectionStrategy):
            raise ValueError(
                f"strategy must be PoseAxisNullProjectionStrategy enum; got {type(self.strategy).__name__}"
            )
        if self.d_seg_epsilon < 0:
            raise ValueError(f"d_seg_epsilon must be >= 0; got {self.d_seg_epsilon}")
        if self.target_d_pose_lower < 0:
            raise ValueError(
                f"target_d_pose_lower must be >= 0; got {self.target_d_pose_lower}"
            )
        if self.target_d_pose_upper <= self.target_d_pose_lower:
            raise ValueError(
                f"target_d_pose_upper={self.target_d_pose_upper} must be > "
                f"target_d_pose_lower={self.target_d_pose_lower}"
            )


# --- Canonical menu construction helper -------------------------------------


def build_canonical_frame1_pose_axis_null_projection_menu(
    strategy: PoseAxisNullProjectionStrategy,
) -> list[dict[str, Any]]:
    """Build canonical frame-1 motion-pair perturbation menu per strategy.

    Returns canonical list of mode descriptors (dicts with ``mode_id``,
    ``family``, ``params``, ``description`` keys) per canonical
    :class:`PoseAxisNullProjectionStrategy` enum.

    Canonical menu families per design memo § 3.1:

    * PER_PIXEL_ROLL: 8 modes
    * DCT_CHROMA_BASIS: 16 modes
    * HADAMARD_TILE: 3 modes
    * GAUSSIAN_NOISE: 16 modes

    Total canonical menu (all strategies combined): 43 modes (canonical sister of
    OPT-1 87-candidate frame-0 widened catalog).
    """
    modes: list[dict[str, Any]] = []

    if strategy == PoseAxisNullProjectionStrategy.PER_PIXEL_ROLL:
        # 8 canonical single-pixel rolls frame-1 (excluding (0, 0) identity).
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                modes.append(
                    {
                        "mode_id": f"frame1_pixel_roll_dx{dx:+d}_dy{dy:+d}",
                        "family": "frame1_pixel_roll",
                        "params": {"dx": dx, "dy": dy},
                        "description": (
                            f"Single-pixel roll frame-1 (dx={dx:+d}, dy={dy:+d}); "
                            "SegNet bilinear resize argmax invariant"
                        ),
                    }
                )
    elif strategy == PoseAxisNullProjectionStrategy.DCT_CHROMA_BASIS:
        # 16 canonical DCT-II sign basis frame-1 (8 frequency bins × 2 amplitudes).
        # Canonical 8 (u, v) frequency bins per OPT-1 widened catalog.
        freq_bins = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (0, 2), (2, 0)]
        for amp in (1, 2):
            for u, v in freq_bins:
                modes.append(
                    {
                        "mode_id": f"frame1_dct_chroma_u{u}_v{v}_amp_{amp}",
                        "family": "frame1_dct_chroma",
                        "params": {"u": u, "v": v, "amp": amp},
                        "description": (
                            f"DCT-II sign basis frame-1 (u={u}, v={v}, amp={amp}); "
                            "canonical OPT-12 analog structured-signed-chroma family"
                        ),
                    }
                )
    elif strategy == PoseAxisNullProjectionStrategy.HADAMARD_TILE:
        # 3 canonical Hadamard tiles frame-1: Sylvester 8x8 amp{1, 2, 3}.
        for amp in (1, 2, 3):
            modes.append(
                {
                    "mode_id": f"frame1_hadamard_tile_amp_{amp}",
                    "family": "frame1_hadamard_tile",
                    "params": {"amp": amp, "tile_size": 8},
                    "description": (
                        f"Sylvester 8x8 Hadamard tile frame-1 (amp={amp}); "
                        "EfficientNet stride-2 stem invariant"
                    ),
                }
            )
    elif strategy == PoseAxisNullProjectionStrategy.GAUSSIAN_NOISE:
        # 16 canonical Gaussian noise frame-1: σ × seeds.
        for sigma in (0.5, 1.0, 1.5, 2.0):
            for seed in (1, 2, 3, 4):
                modes.append(
                    {
                        "mode_id": f"frame1_gaussian_noise_sigma{sigma}_seed{seed}",
                        "family": "frame1_gaussian_noise",
                        "params": {"sigma": sigma, "seed": seed},
                        "description": (
                            f"Gaussian noise frame-1 (σ={sigma}, seed={seed}); "
                            "UNIWARD textured-region undetectable axis"
                        ),
                    }
                )
    else:
        # Defensive (PoseAxisNullProjectionStrategy enum exhausted above).
        raise ValueError(f"Unknown canonical strategy: {strategy!r}")

    return modes


# --- Canonical helper: build_axis_decomposition per Catalog #356 ------------


def build_axis_decomposition_for_pr110_opt_6(
    substrate_id: str,
    current_archive_bytes: int,
    current_d_pose: float,
    *,
    predicted_d_seg_delta: float = 0.0,
    predicted_d_pose_delta: float = -0.0005,
    predicted_archive_bytes_delta: int = 0,
) -> dict[str, Any]:
    """Build canonical per-axis AxisDecomposition per Catalog #356.

    Canonical per-axis prediction per § 3.2 design memo Layer 6:

    * ``predicted_d_seg_delta = 0.0`` (canonical exact per SegNet-null filter)
    * ``predicted_d_pose_delta = -0.0005`` (canonical midpoint of [-0.0010, -0.0001] band
      per OPT-12 PoseNet-null analog symmetry; operator-routable refinement pending
      paired-CUDA RATIFICATION)
    * ``predicted_archive_bytes_delta = 0`` (canonical zero-byte bolt-on per per-pair
      selector reuse; sister of Slot LL L28 PR98 channel-balance pattern)

    Canonical Provenance per Catalog #323 via :func:`tac.provenance.builders.build_provenance_for_predicted`.

    Tier A canonical-routing markers per Catalog #341 + #357.
    """
    import hashlib

    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict

    # Canonical inputs sha per Catalog #323 — substrate_id + canonical config
    # constants are the canonical predictor input feature vector.
    inputs_fingerprint = (
        f"substrate_id={substrate_id}|"
        f"equation={CANONICAL_EQUATION_CANDIDATE_ID}|"
        f"band_lower={PREDICTED_SCORE_DELTA_BAND_LOWER}|"
        f"band_upper={PREDICTED_SCORE_DELTA_BAND_UPPER}"
    )
    inputs_sha256 = hashlib.sha256(inputs_fingerprint.encode("utf-8")).hexdigest()

    provenance = build_provenance_for_predicted(
        model_id=CANONICAL_EQUATION_CANDIDATE_ID,
        inputs_sha256=inputs_sha256,
        measurement_axis="[predicted]",
    )

    return {
        "predicted_d_seg_delta": float(predicted_d_seg_delta),
        "predicted_d_pose_delta": float(predicted_d_pose_delta),
        "predicted_archive_bytes_delta": int(predicted_archive_bytes_delta),
        "axis_tag": "[predicted]",
        "canonical_provenance": provenance_to_dict(provenance),
        "current_archive_bytes": int(current_archive_bytes),
        "current_d_pose": float(current_d_pose),
        "substrate_id": substrate_id,
        "canonical_equation_candidate_id": CANONICAL_EQUATION_CANDIDATE_ID,
        "predicted_score_delta_band_lower": PREDICTED_SCORE_DELTA_BAND_LOWER,
        "predicted_score_delta_band_upper": PREDICTED_SCORE_DELTA_BAND_UPPER,
        # Tier A canonical-routing markers per Catalog #341 + #357.
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
    }


# --- Canonical apply helper per Catalog #341 Tier A contract ----------------


def apply_pose_axis_null_projection_to_pr110_archive(
    config: MotionPairRepairPoseAxisNullProjectionConfig,
    *,
    current_archive_bytes: int = 178517,  # canonical FEC6 baseline
    current_d_pose: float = 4.94e-5,  # canonical FEC6 baseline per OPT-12 analog
) -> dict[str, Any]:
    """Canonical PR110-OPT-6 motion-pair repair apply helper.

    Returns canonical Tier A canonical-routing-markers contribution per Catalog #341:

    * ``predicted_delta_adjustment = 0.0`` (L0 SCAFFOLD does NOT claim score improvement)
    * ``promotable = False`` (canonical per Catalog #192 + #341 Tier A non-promotable)
    * ``axis_tag = "[predicted]"`` (canonical per Catalog #287 evidence-tag discipline)
    * ``verdict = "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"`` (canonical per
      Catalog #325 + #246 paired-CUDA RATIFICATION)

    Canonical per-axis AxisDecomposition per Catalog #356 (toggleable via
    ``config.emit_axis_decomposition``).

    Canonical menu construction per :func:`build_canonical_frame1_pose_axis_null_projection_menu`.
    """
    canonical_menu = build_canonical_frame1_pose_axis_null_projection_menu(config.strategy)

    contribution: dict[str, Any] = {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "verdict": "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        "substrate_id": config.substrate_id,
        "strategy": config.strategy.value,
        "canonical_menu_size": len(canonical_menu),
        "canonical_menu_total_all_strategies": CANONICAL_FRAME1_MENU_TOTAL,
        "canonical_equation_candidate_id": CANONICAL_EQUATION_CANDIDATE_ID,
        "canonical_anti_pattern_candidate_id": CANONICAL_ANTI_PATTERN_CANDIDATE_ID,
        "predicted_score_delta_band": (
            PREDICTED_SCORE_DELTA_BAND_LOWER,
            PREDICTED_SCORE_DELTA_BAND_UPPER,
        ),
        "fec6_baseline_wire_bytes": FEC6_BASELINE_WIRE_BYTES,
        "fec6_baseline_archive_sha_prefix": FEC6_BASELINE_ARCHIVE_SHA_PREFIX,
        "slot_mm_quantitative_prediction_deprecated": SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED,
        "slot_qq_empirical_falsification_checkpoint_utc": SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC,
        "canonical_fridrich_yousfi_paradigm_intact": CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT,
    }

    if config.emit_axis_decomposition:
        contribution["axis_decomposition"] = build_axis_decomposition_for_pr110_opt_6(
            substrate_id=config.substrate_id,
            current_archive_bytes=current_archive_bytes,
            current_d_pose=current_d_pose,
        )

    return contribution


# --- Canonical operator-routable target enumeration ------------------------


def list_canonical_paired_cuda_ratification_targets() -> list[dict[str, Any]]:
    """Canonical enumeration of paired-CUDA RATIFICATION target substrates per
    canonical Catalog #246 dual-axis discipline + canonical Catalog #343 frontier
    pointer + Slot LL sister-pattern template.

    Returns canonical 4 current frontier candidates with canonical archive sha
    prefixes + estimated score delta band per canonical Fridrich-Yousfi inverse-
    steganalysis pose-axis null-projection axis.
    """
    return [
        {
            "substrate_id": "v14_v2_dqs1",
            "canonical_sha_prefix": "7a0da5d0fc327cba",
            "frontier_role": "Current CPU frontier sub-anchor",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "fec6",
            "canonical_sha_prefix": FEC6_BASELINE_ARCHIVE_SHA_PREFIX,
            "frontier_role": "Current CPU frontier canonical per Catalog #343",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "pr106_format0d",
            "canonical_sha_prefix": "9cb989cef519",
            "frontier_role": "Current CUDA frontier canonical per Catalog #343",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
        {
            "substrate_id": "nscs06_v8_stacked",
            "canonical_sha_prefix": "pending_ratification",
            "frontier_role": "Sister in-flight stacked archive",
            "predicted_delta_s_band": (
                PREDICTED_SCORE_DELTA_BAND_LOWER,
                PREDICTED_SCORE_DELTA_BAND_UPPER,
            ),
            "paired_cuda_envelope_usd": 0.30,
        },
    ]


__all__ = [
    # Canonical constants
    "OPT12_POSENET_NULL_TYPICAL_ABS_POSE_DELTA",
    "OPT12_POSENET_NULL_DOMINANT_FAMILY_FRACTION",
    "PREDICTED_SCORE_DELTA_BAND_LOWER",
    "PREDICTED_SCORE_DELTA_BAND_UPPER",
    "FEC6_BASELINE_WIRE_BYTES",
    "FEC6_BASELINE_ARCHIVE_SHA_PREFIX",
    "PR110_NUM_PAIRS",
    "PR110_K_SYMBOLS",
    "CANONICAL_PIXEL_ROLL_FRAME1_COUNT",
    "CANONICAL_DCT_CHROMA_FRAME1_COUNT",
    "CANONICAL_HADAMARD_TILE_FRAME1_COUNT",
    "CANONICAL_GAUSSIAN_NOISE_FRAME1_COUNT",
    "CANONICAL_FRAME1_MENU_TOTAL",
    "SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED",
    "SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC",
    "CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT",
    "CANONICAL_EQUATION_CANDIDATE_ID",
    "CANONICAL_ANTI_PATTERN_CANDIDATE_ID",
    # Canonical enum
    "PoseAxisNullProjectionStrategy",
    # Canonical dataclass
    "MotionPairRepairPoseAxisNullProjectionConfig",
    # Canonical helpers
    "build_canonical_frame1_pose_axis_null_projection_menu",
    "build_axis_decomposition_for_pr110_opt_6",
    "apply_pose_axis_null_projection_to_pr110_archive",
    "list_canonical_paired_cuda_ratification_targets",
]
