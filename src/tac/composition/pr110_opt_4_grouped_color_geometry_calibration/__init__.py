# SPDX-License-Identifier: MIT
"""PR110-OPT-4 Grouped Color/Geometry Calibration with Cross-Pair Perturbation Reuse — L0 SCAFFOLD.

Per CATHEDRAL-SMARTER-DESIGN-MEMO + canonical 3-metric trichotomy RANK 3
+ TaskCreate #1316 + Slot X cap≥4 maintenance directive 2026-05-29.

Design memo (single source of truth):
``.omx/research/pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_design_20260529.md``

Wave N+34 analytical anchor (IMPLEMENTATION_FALSIFIED verdict):
``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``

Canonical context
=================

PR110 archive grammar (from FEC6 ``submissions/hnerv_fec6_fixed_huffman_k16/``):

- 16-symbol K=16 selector palette
- 600 per-pair selectors (one selector per source-frame pair)
- 6-byte header + 243-byte 0-order fixed-Huffman bitstream = **249 byte baseline wire**

Wave N+34 analytical investigation sourced 21 modes from the live PR101 paired-
component sweep (``experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl``)
and computed the grouped color/geometry calibration upper bounds:

- Shannon-coded grouped wire estimate: **258 bytes** (+9 WORSE than FEC6 249)
- Fixed-width grouped wire estimate: **383 bytes** (+134 WORSE than FEC6 249)
- 17 unique best-modes from 22-mode catalog over 600 pairs
- Shannon entropy: 3.3313203895039005 bits/pair
- Per-pair component delta_S: -0.0011704843740551626 (uniform across pairs)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog
#307: this is **IMPLEMENTATION-LEVEL FALSIFICATION** not paradigm-level.
The grouping paradigm (cross-pair perturbation reuse) remains
DEFERRED-PENDING-RESEARCH per Catalog #308 alternative-reducer enumeration.

L0 SCAFFOLD role
================

THIS module serves the canonical dual role:

1. **Preserve the Wave N+34 analytical primitive** as a queryable system surface
   so future widened-catalog probes can compare against the baked-in 17-group /
   21-mode counts without re-deriving them. The constants
   ``WAVE_N34_FEC6_BASELINE_WIRE_BYTES``, ``WAVE_N34_GROUPED_WIRE_BYTES_SHANNON``,
   ``WAVE_N34_GROUPED_WIRE_BYTES_FIXED``, ``WAVE_N34_N_MODES``,
   ``WAVE_N34_N_GROUPS``, ``WAVE_N34_N_PAIRS``,
   ``WAVE_N34_SHANNON_ENTROPY_BITS`` capture the Wave N+34 anchor verbatim.

2. **Enumerate alternative reducer methodologies** per Catalog #308 so the
   operator can route the next iteration through one of N≥4 candidates (NOT
   just the falsified Shannon-coded grouping). The enum
   :class:`GroupingStrategy` carries the 4 canonical alternatives.

L0 SCAFFOLD does NOT claim score improvement. The canonical
``apply_grouped_color_geometry_calibration_to_pr110_archive`` entry point
returns a Tier A canonical-routing-markers contribution per Catalog #341:
``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_ALTERNATIVE_REDUCER`` per Catalog #308.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose, archive
  bytes) decomposition with canonical Provenance dict-form).
- :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted` (the L0
  SCAFFOLD predicts via Wave N+34 analytical upper bound; empirical anchor
  required for promotion).
- Tier A canonical-routing markers per Catalog #341 + #357.
- HNeRV parity discipline L4 (≤200 LOC inflate budget; this L0 SCAFFOLD has
  no inflate-time code path — encoder-side only).
- HNeRV parity discipline L7 (bolt-on size budget ≤350 LOC; this L0 SCAFFOLD
  is ~250 LOC including the test fixtures' import surface).
- Catalog #309 ``horizon_class: plateau_adjacent``.

Sister cross-references
=======================

- :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
- :mod:`tac.provenance.builders` (canonical Provenance builders)
- :mod:`tac.provenance.validator.provenance_to_dict` (canonical dict-form)
- ``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``
- ``.omx/research/pr110_opt3_variant_b_markov_landed_20260526.md``
- ``.omx/research/pr110_opt3_variant_c_variable_k_escape_mechanism_landed_20260526.md``

Per Catalog #287 evidence tag discipline: the score deltas this module
returns are PREDICTED (Wave N+34 analytical upper bound), tagged
``[predicted]`` per Catalog #287/#341. Empirical paired-CUDA anchor required
before any score claim per CLAUDE.md "Apples-to-apples evidence discipline"
+ "Submission auth eval — BOTH CPU AND CUDA".
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

# -----------------------------------------------------------------------------
# Wave N+34 canonical anchor constants (analytical IMPLEMENTATION_FALSIFIED)
# -----------------------------------------------------------------------------

#: PR110 FEC6 fixed-Huffman K=16 baseline wire size in bytes per Wave N+34.
WAVE_N34_FEC6_BASELINE_WIRE_BYTES: int = 249

#: Wave N+34 grouped color/geometry calibration Shannon-coded wire estimate
#: (UPPER BOUND per analytical investigation; +9 WORSE than FEC6 baseline).
WAVE_N34_GROUPED_WIRE_BYTES_SHANNON: int = 258

#: Wave N+34 grouped color/geometry calibration fixed-width wire estimate
#: (UPPER BOUND per analytical investigation; +134 WORSE than FEC6 baseline).
WAVE_N34_GROUPED_WIRE_BYTES_FIXED: int = 383

#: Wave N+34 number of modes in the 22-mode catalog (one mode reserved for
#: "none" sentinel; 21 active modes).
WAVE_N34_N_MODES: int = 21

#: Wave N+34 number of unique best-modes (groups) observed across 600 pairs.
WAVE_N34_N_GROUPS: int = 17

#: Wave N+34 number of source pairs (canonical 600-pair PR101 sweep).
WAVE_N34_N_PAIRS: int = 600

#: Wave N+34 Shannon entropy in bits/pair (used for the analytical upper
#: bound; sister Markov 1st-order conditional entropy is 2.9402 bits/pair
#: per Variant B canonical anchor).
WAVE_N34_SHANNON_ENTROPY_BITS: float = 3.3313203895039005

#: Wave N+34 mean per-pair component delta_S (uniform across pairs because
#: the grouping reducer collapses per-pair signal into per-group menu).
WAVE_N34_MEAN_PER_PAIR_COMPONENT_DELTA_S: float = -0.0011704843740551626

#: Canonical rate multiplier per contest formula
#: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
CANONICAL_RATE_MULTIPLIER: float = 25.0

#: Canonical rate denominator per contest formula
#: (37,545,489 = total contest video byte count).
CANONICAL_RATE_DENOM_BYTES: int = 37_545_489

# -----------------------------------------------------------------------------
# Grouping strategy enum (Catalog #308 alternative-reducer enumeration)
# -----------------------------------------------------------------------------


class GroupingStrategy(str, enum.Enum):
    """Canonical grouping strategy for PR110-OPT-4 cross-pair perturbation reuse.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies; SHANNON_CODED is Wave N+34's
    falsified default; the other 3 are DEFERRED-PENDING-RESEARCH sister
    candidates per the design memo's reactivation criteria.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy may produce a different wire-bytes estimate; the canonical
    helper :func:`compute_grouped_wire_bytes` dispatches on this enum.
    """

    #: Wave N+34 Shannon-coded grouping (IMPLEMENTATION_FALSIFIED; +9 worse).
    SHANNON_CODED = "shannon_coded"

    #: Wave N+34 fixed-width grouping (IMPLEMENTATION_FALSIFIED; +134 worse).
    FIXED_WIDTH = "fixed_width"

    #: DEFERRED per Catalog #308: per-region grouping (per-pixel-region instead
    #: of per-pair-class; sister probe path per Catalog #277 wavelet hierarchy).
    PER_REGION = "per_region"

    #: DEFERRED per Catalog #308: per-temporal-window grouping (across N
    #: consecutive pairs instead of single pair; sister probe path per
    #: Catalog #277 multi-scale partition prior).
    PER_TEMPORAL_WINDOW = "per_temporal_window"


# -----------------------------------------------------------------------------
# Canonical Config dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedColorGeometryCalibrationConfig:
    """Canonical configuration for PR110-OPT-4 L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        grouping_strategy: Canonical :class:`GroupingStrategy` enum value.
            Defaults to ``SHANNON_CODED`` (Wave N+34 baseline) per the
            canonical-vs-unique decision per layer (Catalog #290).
        n_pairs: Source pair count (canonical PR110 = 600 per Wave N+34).
        n_modes: Source mode count in the perturbation catalog (canonical
            Wave N+34 = 21 active modes; sister widened-catalog probes use
            ≥40 modes).
        n_groups_hint: Optional grouping-cardinality hint; if None, the
            encoder auto-derives via the reducer for ``grouping_strategy``.
        header_overhead_bytes_per_group: Per-group menu overhead in bytes
            (canonical = 1 byte per group entry for symbol-index lookup
            table; sister widened-catalog probes use larger per-group menus).
        emit_axis_decomposition: If True (default), the
            ``apply_grouped_color_geometry_calibration_to_pr110_archive``
            entry point emits a canonical :class:`AxisDecomposition` per
            Catalog #356 for downstream Pareto polytope solver consumption.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    grouping_strategy: GroupingStrategy = GroupingStrategy.SHANNON_CODED
    n_pairs: int = WAVE_N34_N_PAIRS
    n_modes: int = WAVE_N34_N_MODES
    n_groups_hint: int | None = None
    header_overhead_bytes_per_group: int = 1
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.grouping_strategy, GroupingStrategy):
            raise ValueError(
                "grouping_strategy must be GroupingStrategy enum, got "
                f"{type(self.grouping_strategy).__name__}"
            )
        if not isinstance(self.n_pairs, int) or isinstance(self.n_pairs, bool):
            raise ValueError(
                f"n_pairs must be int, got {type(self.n_pairs).__name__}"
            )
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0, got {self.n_pairs}")
        if not isinstance(self.n_modes, int) or isinstance(self.n_modes, bool):
            raise ValueError(
                f"n_modes must be int, got {type(self.n_modes).__name__}"
            )
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {self.n_modes}")
        if self.n_groups_hint is not None:
            if (
                not isinstance(self.n_groups_hint, int)
                or isinstance(self.n_groups_hint, bool)
            ):
                raise ValueError(
                    "n_groups_hint must be int or None, got "
                    f"{type(self.n_groups_hint).__name__}"
                )
            if self.n_groups_hint <= 0:
                raise ValueError(
                    f"n_groups_hint must be > 0, got {self.n_groups_hint}"
                )
            if self.n_groups_hint > self.n_modes:
                raise ValueError(
                    f"n_groups_hint ({self.n_groups_hint}) must be <= "
                    f"n_modes ({self.n_modes})"
                )
        if (
            not isinstance(self.header_overhead_bytes_per_group, int)
            or isinstance(self.header_overhead_bytes_per_group, bool)
        ):
            raise ValueError(
                "header_overhead_bytes_per_group must be int, got "
                f"{type(self.header_overhead_bytes_per_group).__name__}"
            )
        if self.header_overhead_bytes_per_group < 0:
            raise ValueError(
                "header_overhead_bytes_per_group must be >= 0, got "
                f"{self.header_overhead_bytes_per_group}"
            )


# -----------------------------------------------------------------------------
# Canonical helpers
# -----------------------------------------------------------------------------


def _compute_grouping_signature(source_modes_per_pair: Sequence[int]) -> str:
    """Return sha256 hex digest over the source modes sequence.

    Used for: (a) Provenance ``source_sha256`` per Catalog #323;
    (b) deterministic reproducibility diff-able-across-runs facet per
    Catalog #305 observability surface.
    """
    payload = b",".join(str(int(m)).encode("ascii") for m in source_modes_per_pair)
    return hashlib.sha256(payload).hexdigest()


def _derive_group_count(
    source_modes_per_pair: Sequence[int],
    config: GroupedColorGeometryCalibrationConfig,
) -> int:
    """Derive grouping cardinality from source modes per the strategy.

    SHANNON_CODED / FIXED_WIDTH: uses unique mode count from source (Wave N+34
    reducer: "best mode per pair" → unique count = G_groups).
    PER_REGION / PER_TEMPORAL_WINDOW: DEFERRED — currently returns the
    hint or unique count as a placeholder; the canonical implementation
    requires per-region / per-temporal-window source signals (NOT just per-pair
    mode codes); empirical reactivation per Catalog #308 alternative-reducer
    enumeration.
    """
    if config.n_groups_hint is not None:
        return config.n_groups_hint
    unique = len(set(int(m) for m in source_modes_per_pair))
    return max(1, unique)


def compute_grouped_wire_bytes(
    source_modes_per_pair: Sequence[int],
    config: GroupedColorGeometryCalibrationConfig,
) -> dict[str, Any]:
    """Compute the grouped color/geometry calibration wire bytes estimate.

    Canonical entry point for PR110-OPT-4 L0 SCAFFOLD analytical primitive.
    Returns a dict with:

    - ``wire_bytes_estimate`` (int): total wire bytes (header + bitstream).
    - ``header_bytes`` (int): per-group menu overhead.
    - ``bitstream_bytes`` (int): coded bitstream size (Shannon-coded or
      fixed-width per ``config.grouping_strategy``).
    - ``n_groups`` (int): derived grouping cardinality.
    - ``shannon_entropy_bits`` (float): per-pair entropy used for the
      Shannon-coded estimate.
    - ``delta_vs_fec6_bytes`` (int): signed; positive = WORSE than FEC6 baseline.
    - ``grouping_strategy`` (str): canonical enum value.

    Args:
        source_modes_per_pair: list of per-pair mode codes (canonical Wave
            N+34 source = 600 pairs × 21 modes).
        config: canonical :class:`GroupedColorGeometryCalibrationConfig`.

    Returns:
        Dict with the canonical analytical primitive output.

    Raises:
        ValueError: if source modes length disagrees with ``config.n_pairs``.
    """
    if not isinstance(source_modes_per_pair, Sequence):
        raise ValueError(
            "source_modes_per_pair must be Sequence, got "
            f"{type(source_modes_per_pair).__name__}"
        )
    if len(source_modes_per_pair) != config.n_pairs:
        raise ValueError(
            f"source_modes_per_pair length ({len(source_modes_per_pair)}) "
            f"!= config.n_pairs ({config.n_pairs})"
        )

    n_groups = _derive_group_count(source_modes_per_pair, config)

    # Per-group histogram for Shannon entropy
    histogram: dict[int, int] = {}
    for mode in source_modes_per_pair:
        m_int = int(mode)
        histogram[m_int] = histogram.get(m_int, 0) + 1
    total = sum(histogram.values())
    shannon_entropy_bits: float = 0.0
    if total > 0 and len(histogram) > 1:
        for count in histogram.values():
            p = count / total
            if p > 0.0:
                shannon_entropy_bits -= p * math.log2(p)

    header_bytes = n_groups * config.header_overhead_bytes_per_group + 6  # +6 = FEC6 magic+n_pairs header sister

    if config.grouping_strategy == GroupingStrategy.SHANNON_CODED:
        # Shannon-coded: ceil(N_pairs * H(modes) / 8)
        bitstream_bits = max(
            0.0, float(config.n_pairs) * shannon_entropy_bits
        )
        bitstream_bytes = math.ceil(bitstream_bits / 8.0)
    elif config.grouping_strategy == GroupingStrategy.FIXED_WIDTH:
        # Fixed-width: N_pairs * ceil(log2(n_groups)) bits
        if n_groups <= 1:
            bits_per_symbol = 0
        else:
            bits_per_symbol = int(math.ceil(math.log2(n_groups)))
        bitstream_bytes = math.ceil(
            (float(config.n_pairs) * bits_per_symbol) / 8.0
        )
    elif config.grouping_strategy in (
        GroupingStrategy.PER_REGION,
        GroupingStrategy.PER_TEMPORAL_WINDOW,
    ):
        # DEFERRED-PENDING-RESEARCH per Catalog #308: placeholder uses
        # Shannon-coded as upper bound until per-region / per-temporal-window
        # source signals are landed; the canonical reactivation criteria
        # require widened source data per the design memo.
        bitstream_bits = max(
            0.0, float(config.n_pairs) * shannon_entropy_bits
        )
        bitstream_bytes = math.ceil(bitstream_bits / 8.0)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(
            f"unhandled grouping_strategy: {config.grouping_strategy}"
        )

    wire_bytes_estimate = int(header_bytes + bitstream_bytes)
    delta_vs_fec6 = int(wire_bytes_estimate - WAVE_N34_FEC6_BASELINE_WIRE_BYTES)

    return {
        "wire_bytes_estimate": wire_bytes_estimate,
        "header_bytes": int(header_bytes),
        "bitstream_bytes": int(bitstream_bytes),
        "n_groups": int(n_groups),
        "shannon_entropy_bits": float(shannon_entropy_bits),
        "delta_vs_fec6_bytes": delta_vs_fec6,
        "grouping_strategy": config.grouping_strategy.value,
    }


def apply_grouped_color_geometry_calibration_to_pr110_archive(
    source_modes_per_pair: Sequence[int],
    config: GroupedColorGeometryCalibrationConfig,
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply grouped calibration to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog #341
    + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-CUDA
    anchor required per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
    before any score claim.

    Per Wave N+34 IMPLEMENTATION_FALSIFIED verdict + Catalog #307
    paradigm-vs-implementation classification: the verdict field is
    ``DEFERRED_PENDING_ALTERNATIVE_REDUCER`` per Catalog #308 alternative-
    reducer methodology enumeration. Reactivation criteria pinned in the
    design memo.

    Args:
        source_modes_per_pair: list of per-pair mode codes (canonical Wave
            N+34 source = 600 pairs × 21 modes).
        config: canonical :class:`GroupedColorGeometryCalibrationConfig`.

    Returns:
        Dict with canonical Tier A contribution shape:

        - ``predicted_delta_adjustment`` (float): always 0.0 (Tier A
          observability-only per Catalog #341).
        - ``promotable`` (bool): always False (Catalog #341).
        - ``axis_tag`` (str): always "[predicted]" (Catalog #287).
        - ``predicted_axis_decomposition`` (dict): canonical
          :class:`AxisDecomposition` dict-form per Catalog #356 (if
          ``config.emit_axis_decomposition=True``); None otherwise.
        - ``wire_analysis`` (dict): output of
          :func:`compute_grouped_wire_bytes` for the input.
        - ``verdict`` (str): "DEFERRED_PENDING_ALTERNATIVE_REDUCER" per
          Catalog #308.
        - ``wave_n34_anchor`` (dict): Wave N+34 IMPLEMENTATION_FALSIFIED
          verdict citation for the canonical historical anchor.
    """
    wire = compute_grouped_wire_bytes(source_modes_per_pair, config)
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0
    seg_delta = 0.0  # L0 SCAFFOLD does not perturb frames

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.pr110_opt_4_grouped_color_geometry_calibration"
                ".apply_grouped_color_geometry_calibration_to_pr110_archive"
            ),
            inputs_sha256=_compute_grouping_signature(source_modes_per_pair),
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        )
        decomp = AxisDecomposition(
            predicted_d_seg_delta=seg_delta,
            predicted_d_pose_delta=pose_delta,
            predicted_archive_bytes_delta=archive_bytes_delta,
            axis_tag="[predicted]",
            canonical_provenance=provenance_to_dict(prov),
        )
        axis_decomp_payload = decomp.as_dict()

    return {
        # Tier A canonical-routing markers per Catalog #341 + #357
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        # Per-axis decomposition per Catalog #356 (Dim 3 Step 3.1)
        "predicted_axis_decomposition": axis_decomp_payload,
        # Analytical primitive output (queryable system surface)
        "wire_analysis": wire,
        # Catalog #308 verdict + Wave N+34 historical anchor
        "verdict": "DEFERRED_PENDING_ALTERNATIVE_REDUCER",
        "wave_n34_anchor": {
            "verdict": "IMPLEMENTATION_FALSIFIED",
            "shannon_coded_wire_bytes": WAVE_N34_GROUPED_WIRE_BYTES_SHANNON,
            "fixed_width_wire_bytes": WAVE_N34_GROUPED_WIRE_BYTES_FIXED,
            "fec6_baseline_wire_bytes": WAVE_N34_FEC6_BASELINE_WIRE_BYTES,
            "n_modes_in_source": WAVE_N34_N_MODES,
            "n_groups_in_source": WAVE_N34_N_GROUPS,
            "n_pairs_in_source": WAVE_N34_N_PAIRS,
            "shannon_entropy_bits": WAVE_N34_SHANNON_ENTROPY_BITS,
            "canonical_artifact_path": (
                ".omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json"
            ),
            "catalog_307_classification": (
                "IMPLEMENTATION_LEVEL_FALSIFICATION_PARADIGM_INTACT"
            ),
            "catalog_308_alternative_reducer_enumeration": [
                "widened_mode_catalog_>=40_modes",
                "per_region_grouping",
                "per_temporal_window_grouping",
                "composition_with_opt_7_uniward_sparse_selector",
            ],
        },
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "pr110_opt_4_grouped_color_geometry_calibration_cross_pair_perturbation_reuse_design_20260529.md"
        ),
        "horizon_class": "plateau_adjacent",
    }


__all__ = (
    "WAVE_N34_FEC6_BASELINE_WIRE_BYTES",
    "WAVE_N34_GROUPED_WIRE_BYTES_SHANNON",
    "WAVE_N34_GROUPED_WIRE_BYTES_FIXED",
    "WAVE_N34_N_MODES",
    "WAVE_N34_N_GROUPS",
    "WAVE_N34_N_PAIRS",
    "WAVE_N34_SHANNON_ENTROPY_BITS",
    "WAVE_N34_MEAN_PER_PAIR_COMPONENT_DELTA_S",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "GroupingStrategy",
    "GroupedColorGeometryCalibrationConfig",
    "compute_grouped_wire_bytes",
    "apply_grouped_color_geometry_calibration_to_pr110_archive",
)
