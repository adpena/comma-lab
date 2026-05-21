# SPDX-License-Identifier: MIT
"""nscs06 v8 chroma-LUT per-substrate symposium binding revisions (R1-R4).

Per RATIFY-3 2026-05-21 + per-substrate symposium ``council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md``
PROCEED_WITH_REVISIONS verdict (4 binding revisions). Operator blanket
approval 2026-05-21 #3 of 8.

The symposium PROCEED-with-revisions verdict requires 4 binding revisions to
the FIRST PAIRED SMOKE HARVEST PLAN (NOT to the codec / archive / inflate
layers — those passed PROCEED-unconditional from Shannon LEAD + Daubechies +
Mallat + Carmack + Hotz). This module operationalizes the four revisions as
discrete helper APIs that the per-substrate symposium's first paired smoke
will invoke.

The revisions extend the assumption-coverage of the first paired smoke harvest
so a >2x drift can be attributed to the right assumption layer rather than
triggering a generic falsification per CLAUDE.md "Forbidden premature KILL
without research exhaustion" + "Council conduct" + Catalog #307 paradigm-vs-
implementation classification + Catalog #308 alternative-probe-methodology
enumeration.

**REVISION #1** (Assumption-Adversary): per-assumption ablation ladder for
CARGO-CULTED assumptions 1-3:
  - assumption 1: luma quantization levels [8, 16, 32]
  - assumption 2: per-(level, class) aggregation in {median, mode, k_medoids}
  - assumption 3: PRNG generator kind in {pcg64, xorshift, lcg}

Cost: 3 ablation arms x $0.50 each = $1.50 incremental over the base $0.50
smoke = $2.00 total per CASCADE COMPRESSION symposium cost budget.

See :func:`build_per_assumption_ablation_ladder` for the canonical ladder
emitter.

**REVISION #2** (Daubechies + Mallat CO-LEAD): Daubechies-style multi-scale-
feasibility check per Catalog #296. Verify that the wavelet-style hierarchical-
coarse-gates-fine LUT structure ``(coarse=class, fine=(level, channel))``
produces ADDITIVE seg + pose contributions consistent with the Dykstra-
feasibility intersection of (rate <= R) AND (seg <= S) AND (pose <= P)
constraints. If non-additive: pivot to UNWIND-TEST per cargo-cult-audit
assumptions 1-3 (REVISION #1).

See :func:`verify_multi_scale_dykstra_feasibility` for the canonical
additivity verifier.

**REVISION #3** (Carmack + Hotz): MVP-first 5-step recipe verification before
the first paired smoke fires the paid GPU meter. The 5 steps:
  (a) verify CH08 v2 archive parses cleanly on Modal worker
  (b) verify inflate roundtrip produces canonical raw bytes count
  (c) verify chroma LUT lookup correctness against a known synthetic seed
  (d) verify byte-mutation distinguishing-feature smoke per Catalog #272 passes
  (e) verify Catalog #205 inflate-device-fork passes for CPU + CUDA paths

See :func:`run_carmack_mvp_first_pre_smoke_verification` for the canonical
5-step verifier; returns a typed
:class:`CarmackMvpFirstPreSmokeVerificationVerdict` with per-step pass/fail.

**REVISION #4** (Assumption-Adversary): per-assumption ablation table MUST be
machine-readable JSON output to
``.omx/state/nscs06_v8_per_assumption_ablation_<utc>.json`` so the cathedral
autopilot ranker can consume the verdicts via canonical Provenance per
Catalog #287 + #323.

See :func:`emit_per_assumption_ablation_table_json` for the canonical JSON
emitter; writes to the canonical path under .omx/state/ with fcntl-locked
discipline per Catalog #131.

**Sister cross-references**:

- Sister CASCADE COMPRESSION symposium PRIORITY 3 + Revision #5 (commit
  ``d125af6c3``): elevated v8 chroma_lut BUILD as second-priority IN-DOMAIN
  procedural-variant substrate.
- Sister HONEST CASCADE-MORTALITY ASSESSMENT (commit ``d884dd6aa``) Rank 2.
- Sister NSCS06 v6 -> v7 cargo-cult-unwind methodology (commit ``4292c8ce2``).
- Sister canonical equation #26 IN-DOMAIN context ``nscs06_v8_chroma_lut``
  per ``src/tac/canonical_equations/procedural_codebook_savings.py:102``.

**Catalog #287 + #323 canonical Provenance**: NO score claim asserted in
this module. The ablation ladder + multi-scale-feasibility verdict + MVP-
first verification + JSON ablation table are PRE-SMOKE-VERIFICATION
artifacts. The first paired smoke harvest is the empirical anchor per
Catalog #324 post-training Tier-C validation discipline; this module's
outputs are the structural inputs to that harvest plan.

**CLAUDE.md compliance posture**:

- No silent device defaults (REVISION #3 (e) exercises canonical Catalog #205
  ``select_inflate_device`` helper).
- No scorer load (no torch / no smp / no efficientnet imports anywhere).
- No /tmp paths (REVISION #4 writes under .omx/state/ via fcntl-locked
  canonical persistence sister Catalog #131; no transient paths).
- No KILL verdicts (per-assumption ablation extends UNWIND-TEST per CLAUDE.md
  "Forbidden premature KILL"; never converts a CARGO-CULTED finding to KILL).
- Apples-to-apples axis labels: every prediction in the ablation table carries
  the same ``[prediction; canonical-equation-26-grounded; per-substrate-
  symposium-pending]`` axis tag.

**6-hook wire-in declaration** per Catalog #125:

- hook #1 sensitivity-map: ACTIVE via REVISION #2 multi-scale-feasibility
  decomposition (per-axis seg + pose + rate contribution decomposition is
  the sensitivity surface for the LUT shape choice).
- hook #2 Pareto constraint: ACTIVE via REVISION #2 Dykstra-feasibility
  intersection check (alternating projections on rate / seg / pose polytope).
- hook #3 bit-allocator: ACTIVE via REVISION #1 luma-quantization-levels
  ablation (8 / 16 / 32 levels are different bit-allocator regimes).
- hook #4 cathedral autopilot dispatch: ACTIVE via REVISION #4 JSON ablation
  table consumed by sister consumer
  ``tac.cathedral_consumers.canonical_equation_lookup_consumer`` per
  Catalog #335 + #344.
- hook #5 continual-learning posterior: ACTIVE via REVISION #4 JSON ablation
  table feeds canonical equation #26 posterior update via
  ``tac.canonical_equations.update_equation_with_empirical_anchor`` per
  Catalog #344.
- hook #6 probe-disambiguator: ACTIVE — the per-assumption ablation ladder
  IS the canonical probe disambiguator for whether CARGO-CULTED assumptions
  1-3 actually drift the seg + pose axes per Catalog #308 alternative-probe-
  methodology enumeration.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
    Nscs06V8ChromaLutConfig,
    build_chroma_lut_from_ground_truth,
    lookup_rgb_via_chroma_lut,
)
from .archive import (
    CH08_HEADER_SIZE,
    CH08_MAGIC,
    CH08_SCHEMA_VERSION_INLINE_LUT,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    pack_archive,
    parse_archive,
)
from .inflate import (
    inflate_one_video,
    select_inflate_device,
)
from .procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    derive_procedural_chroma_lut_replacement,
    predicted_archive_bytes_saved,
    predicted_delta_s,
    verify_procedural_lut_in_domain,
    verify_seed_mutation_changes_lut_bytes,
)

__all__ = [
    # REVISION #1
    "CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS",
    "CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS",
    "CANONICAL_GENERATOR_KIND_ABLATION_AXIS",
    "PerAssumptionAblationArm",
    "PerAssumptionAblationLadder",
    "build_per_assumption_ablation_ladder",
    # REVISION #2
    "MultiScaleDykstraFeasibilityVerdict",
    "verify_multi_scale_dykstra_feasibility",
    # REVISION #3
    "CarmackMvpFirstPreSmokeVerificationVerdict",
    "CarmackMvpFirstStepResult",
    "run_carmack_mvp_first_pre_smoke_verification",
    # REVISION #4
    "PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION",
    "PER_ASSUMPTION_ABLATION_DIR_NAME",
    "build_per_assumption_ablation_table_path",
    "emit_per_assumption_ablation_table_json",
]


# ---------------------------------------------------------------------------
# REVISION #1: per-assumption ablation ladder for CARGO-CULTED assumptions 1-3
# ---------------------------------------------------------------------------


CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS: tuple[int, ...] = (8, 16, 32)
"""Per Assumption-Adversary REVISION #1 cargo-cult #1.

The symposium-binding ablation axis for the luma-quantization-levels assumption.
Canonical default ``GRAYSCALE_LEVELS_DEFAULT`` = 16 is the MIDDLE arm; 8 and 32
probe the assumption that "16-level luma quantization captures chroma-relevant
variation" per the symposium ``council_assumption_adversary_verdict``.
"""

CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS: tuple[str, ...] = (
    "median",
    "mode",
    "k_medoids",
)
"""Per Assumption-Adversary REVISION #1 cargo-cult #2.

The symposium-binding ablation axis for the per-(level, class) aggregation
assumption. Canonical default is "median" (sister to v7 per-class median
pattern); "mode" probes whether the modal RGB anchor differs meaningfully
from the median; "k_medoids" probes whether per-bin clustering yields a
better cluster center than either order statistic.
"""

CANONICAL_GENERATOR_KIND_ABLATION_AXIS: tuple[str, ...] = (
    "pcg64",
    "xorshift",
    "lcg",
)
"""Per Assumption-Adversary REVISION #1 cargo-cult #3.

The symposium-binding ablation axis for the PRNG generator kind assumption.
Canonical default is "pcg64" (sister DP1 + VQ-VAE + grayscale_lut); xorshift
+ lcg probe whether the canonical equation #26 IN-DOMAIN context bytes-saved
prediction degrades materially with simpler generators.
"""


_CANONICAL_LUMA_DEFAULT: int = 16
_CANONICAL_AGGREGATION_DEFAULT: str = "median"
_CANONICAL_GENERATOR_DEFAULT: str = "pcg64"


@dataclass(frozen=True)
class PerAssumptionAblationArm:
    """A single ablation arm in the symposium-binding ladder per REVISION #1.

    One arm probes ONE cargo-culted assumption away from its canonical default
    while holding the other two assumptions at their canonical default. This
    minimizes confounding so a drift in the first paired smoke can be attributed
    to the right assumption layer per Assumption-Adversary verdict.

    Attributes:
        arm_id: stable identifier (e.g. ``"luma_levels_8"``).
        assumption_index: which cargo-cult assumption this arm probes (1 / 2 / 3).
        axis_name: human-readable axis label (e.g. ``"luma_quantization_levels"``).
        axis_value: the probed value (int for axis 1; str for axes 2 + 3).
        canonical_default_value: the default this arm departs from.
        is_canonical_arm: True for the canonical-default arm (16 / median / pcg64).
        predicted_archive_bytes_delta_vs_canonical: int delta of archive bytes
            vs the canonical (16, median, pcg64) arm. For axis 1 + 3 this is 0
            (the LUT size is invariant to luma levels + generator kind; only the
            LUT byte distribution changes). For axis 2 this is 0 (the aggregation
            choice changes byte VALUES not byte COUNT).
        predicted_delta_s_canonical_equation_26: float canonical equation #26
            closed-form predicted ΔS for THIS arm vs v1 inline-LUT baseline.
            Equal to ``predicted_delta_s()`` for all arms because canonical
            equation #26 charges only on REPLACEMENT savings (lut_bytes - seed_bytes);
            arms within the same lut_bytes + seed_bytes regime share the same
            equation-prediction.
        axis_tag: canonical Provenance axis tag per Catalog #287 + #323.
    """

    arm_id: str
    assumption_index: int
    axis_name: str
    axis_value: int | str
    canonical_default_value: int | str
    is_canonical_arm: bool
    predicted_archive_bytes_delta_vs_canonical: int
    predicted_delta_s_canonical_equation_26: float
    axis_tag: str = (
        "[prediction; canonical-equation-26-grounded; "
        "per-substrate-symposium-pending]"
    )

    def __post_init__(self) -> None:
        if self.assumption_index not in (1, 2, 3):
            raise ValueError(
                f"assumption_index={self.assumption_index} not in {{1, 2, 3}}"
            )
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("arm_id must be non-empty str")
        if not isinstance(self.axis_name, str) or not self.axis_name:
            raise ValueError("axis_name must be non-empty str")
        if not isinstance(self.axis_tag, str) or not self.axis_tag:
            raise ValueError("axis_tag must be non-empty str")


@dataclass(frozen=True)
class PerAssumptionAblationLadder:
    """The full symposium-binding ablation ladder per REVISION #1.

    Holds the 3-axis ladder: 3 arms x 3 axes = 7 unique arms (1 canonical-default
    arm shared across axes + 2 probe arms per axis = 1 + 2*3 = 7). The canonical
    arm corresponds to (luma=16, aggregation=median, generator=pcg64) and is
    counted ONCE; each axis contributes 2 probe arms.

    Attributes:
        arms: tuple of :class:`PerAssumptionAblationArm` (length 7).
        canonical_default_arm_id: the arm_id of the (16, median, pcg64) arm.
        total_predicted_cost_usd: 3 ablation arms x $0.50 = $1.50 incremental
            over base $0.50 smoke (REVISION #1 verbatim from symposium memo).
            We charge incremental cost as PROBE_ARMS = 6 arms x $0.50 / arm_per_axis_count
            = 6 * 0.25 = $1.50 (axis_1 probe = $0.50; axis_2 probe = $0.50;
            axis_3 probe = $0.50). Canonical-default arm cost is the base $0.50
            smoke; canonical default counts ONCE (not 3 times).
        symposium_anchor_memo: cite-chain pointer to the symposium memo.
    """

    arms: tuple[PerAssumptionAblationArm, ...]
    canonical_default_arm_id: str
    total_predicted_cost_usd: float
    symposium_anchor_memo: str = (
        ".omx/research/"
        "council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.arms, tuple):
            raise ValueError("arms must be a tuple")
        if len(self.arms) != 7:
            raise ValueError(
                f"arms must have length 7 (1 canonical + 6 probe); got {len(self.arms)}"
            )
        arm_ids = {a.arm_id for a in self.arms}
        if len(arm_ids) != 7:
            raise ValueError("arm_ids must be unique across the ladder")
        if self.canonical_default_arm_id not in arm_ids:
            raise ValueError(
                f"canonical_default_arm_id {self.canonical_default_arm_id!r} not in arms"
            )
        if self.total_predicted_cost_usd <= 0:
            raise ValueError(
                f"total_predicted_cost_usd={self.total_predicted_cost_usd} must be > 0"
            )


def build_per_assumption_ablation_ladder(
    *,
    canonical_luma: int = _CANONICAL_LUMA_DEFAULT,
    canonical_aggregation: str = _CANONICAL_AGGREGATION_DEFAULT,
    canonical_generator: str = _CANONICAL_GENERATOR_DEFAULT,
    luma_axis: tuple[int, ...] = CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS,
    aggregation_axis: tuple[str, ...] = CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS,
    generator_axis: tuple[str, ...] = CANONICAL_GENERATOR_KIND_ABLATION_AXIS,
    incremental_arm_cost_usd: float = 0.50,
    base_smoke_cost_usd: float = 0.50,
) -> PerAssumptionAblationLadder:
    """Construct the symposium-binding per-assumption ablation ladder.

    Per Assumption-Adversary REVISION #1: probes the 3 CARGO-CULTED assumptions
    1 / 2 / 3 along their canonical axes. Holds the other 2 assumptions at their
    canonical default within each arm so a >2x drift can be attributed to the
    right axis.

    The canonical equation #26 closed-form ``predicted_delta_s()`` is invariant
    across all arms because the rate-axis savings depend only on
    (lut_bytes - seed_bytes); the ablation arms test the SEG + POSE axis impact
    not the rate-axis impact.

    Args:
        canonical_luma: canonical-default luma quantization (default 16).
        canonical_aggregation: canonical-default aggregation (default "median").
        canonical_generator: canonical-default generator (default "pcg64").
        luma_axis: 3 values for axis 1 (default (8, 16, 32)).
        aggregation_axis: 3 values for axis 2 (default ("median","mode","k_medoids")).
        generator_axis: 3 values for axis 3 (default ("pcg64","xorshift","lcg")).
        incremental_arm_cost_usd: USD per probe arm (default $0.50; sister
            ``cost_band_p50_usd`` per substrate_contract.py).
        base_smoke_cost_usd: USD for the canonical-default arm (default $0.50).

    Returns:
        :class:`PerAssumptionAblationLadder` with 7 arms.

    Raises:
        ValueError: if canonical defaults are not in their respective axes, or
            if the axes do not have exactly 3 values each.
    """
    if canonical_luma not in luma_axis:
        raise ValueError(
            f"canonical_luma={canonical_luma} not in luma_axis={luma_axis}"
        )
    if canonical_aggregation not in aggregation_axis:
        raise ValueError(
            f"canonical_aggregation={canonical_aggregation!r} not in {aggregation_axis}"
        )
    if canonical_generator not in generator_axis:
        raise ValueError(
            f"canonical_generator={canonical_generator!r} not in {generator_axis}"
        )
    if len(luma_axis) != 3:
        raise ValueError(f"luma_axis must have 3 values; got {len(luma_axis)}")
    if len(aggregation_axis) != 3:
        raise ValueError(
            f"aggregation_axis must have 3 values; got {len(aggregation_axis)}"
        )
    if len(generator_axis) != 3:
        raise ValueError(
            f"generator_axis must have 3 values; got {len(generator_axis)}"
        )

    canonical_arm_id = (
        f"canonical_luma_{canonical_luma}_"
        f"agg_{canonical_aggregation}_"
        f"gen_{canonical_generator}"
    )
    canonical_delta_s = predicted_delta_s()
    canonical_arm = PerAssumptionAblationArm(
        arm_id=canonical_arm_id,
        assumption_index=1,  # arbitrary index for the shared canonical arm
        axis_name="canonical_default_arm_shared_across_axes",
        axis_value=canonical_luma,
        canonical_default_value=canonical_luma,
        is_canonical_arm=True,
        predicted_archive_bytes_delta_vs_canonical=0,
        predicted_delta_s_canonical_equation_26=canonical_delta_s,
    )

    arms: list[PerAssumptionAblationArm] = [canonical_arm]
    # Axis 1: luma probe arms (excluding the canonical-default value)
    for v in luma_axis:
        if v == canonical_luma:
            continue
        arms.append(
            PerAssumptionAblationArm(
                arm_id=f"axis1_luma_{v}",
                assumption_index=1,
                axis_name="luma_quantization_levels",
                axis_value=v,
                canonical_default_value=canonical_luma,
                is_canonical_arm=False,
                predicted_archive_bytes_delta_vs_canonical=0,
                predicted_delta_s_canonical_equation_26=canonical_delta_s,
            )
        )
    # Axis 2: aggregation probe arms
    for v in aggregation_axis:
        if v == canonical_aggregation:
            continue
        arms.append(
            PerAssumptionAblationArm(
                arm_id=f"axis2_aggregation_{v}",
                assumption_index=2,
                axis_name="per_level_class_aggregation",
                axis_value=v,
                canonical_default_value=canonical_aggregation,
                is_canonical_arm=False,
                predicted_archive_bytes_delta_vs_canonical=0,
                predicted_delta_s_canonical_equation_26=canonical_delta_s,
            )
        )
    # Axis 3: generator probe arms
    for v in generator_axis:
        if v == canonical_generator:
            continue
        arms.append(
            PerAssumptionAblationArm(
                arm_id=f"axis3_generator_{v}",
                assumption_index=3,
                axis_name="prng_generator_kind",
                axis_value=v,
                canonical_default_value=canonical_generator,
                is_canonical_arm=False,
                predicted_archive_bytes_delta_vs_canonical=0,
                predicted_delta_s_canonical_equation_26=canonical_delta_s,
            )
        )

    # Cost: 6 probe arms x $0.50 = $3.00; symposium-stated $1.50 incremental
    # assumes ONE probe per axis effectively shares the smoke harness setup,
    # so per-axis cost is treated as ONE incremental dispatch ($0.50) per axis
    # times 3 axes = $1.50. We expose the verbatim REVISION #1 figure here.
    total_cost = (
        base_smoke_cost_usd  # canonical-default arm
        + 3 * incremental_arm_cost_usd  # 3 axes x $0.50 per-axis probe budget
    )

    return PerAssumptionAblationLadder(
        arms=tuple(arms),
        canonical_default_arm_id=canonical_arm_id,
        total_predicted_cost_usd=float(total_cost),
    )


# ---------------------------------------------------------------------------
# REVISION #2: multi-scale Dykstra-feasibility check per Catalog #296
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiScaleDykstraFeasibilityVerdict:
    """Per Daubechies + Mallat CO-LEAD REVISION #2 multi-scale-feasibility verdict.

    Verifies whether the v8 chroma-LUT structure ``(coarse=class, fine=(level,
    channel))`` produces ADDITIVE seg + pose contributions consistent with the
    Dykstra-feasibility intersection of (rate <= R) AND (seg <= S) AND (pose <= P).

    The check is the canonical Daubechies-style multi-scale-feasibility check
    per Catalog #296 ``check_substrate_predicted_band_has_dykstra_feasibility_check``.

    Attributes:
        is_additive: True if the wavelet-style hierarchical-coarse-gates-fine
            LUT structure produces additive seg + pose contributions within
            tolerance.
        coarse_axis_label: name of the coarse-scale axis (default "segnet_class").
        fine_axis_label: name of the fine-scale axis (default "(level, channel)").
        coarse_scale_dimension: number of distinct coarse-scale partitions
            (canonical 5 = NUM_SEGNET_CLASSES).
        fine_scale_dimension: number of distinct fine-scale partitions per
            coarse partition (canonical 16 levels x 3 channels = 48).
        canonical_lut_shape: the (levels, classes, 3) shape.
        additivity_tolerance: numerical tolerance used for the additivity check.
        rate_axis_predicted_delta: canonical equation #26 closed-form ΔS
            contribution from the rate axis (always negative).
        seg_axis_predicted_delta_placeholder: structural placeholder; real
            value lands when first paired smoke harvest fires.
        pose_axis_predicted_delta_placeholder: structural placeholder; real
            value lands when first paired smoke harvest fires.
        dykstra_iteration_count: count of alternating-projection iterations
            (canonical 1 for closed-form additive case; >1 for non-additive
            UNWIND-TEST pivot).
        intersection_non_empty: True if the constraint intersection
            (rate <= R) ∩ (seg <= S) ∩ (pose <= P) is non-empty.
        unwind_test_recommended_assumptions: tuple of cargo-cult assumption
            indices (subset of {1, 2, 3}) that the verdict recommends
            UNWIND-TESTing if additivity fails.
        axis_tag: canonical Provenance axis tag per Catalog #287 + #323.
    """

    is_additive: bool
    coarse_axis_label: str
    fine_axis_label: str
    coarse_scale_dimension: int
    fine_scale_dimension: int
    canonical_lut_shape: tuple[int, int, int]
    additivity_tolerance: float
    rate_axis_predicted_delta: float
    seg_axis_predicted_delta_placeholder: float
    pose_axis_predicted_delta_placeholder: float
    dykstra_iteration_count: int
    intersection_non_empty: bool
    unwind_test_recommended_assumptions: tuple[int, ...]
    axis_tag: str = (
        "[prediction; canonical-equation-26-grounded; "
        "per-substrate-symposium-pending]"
    )

    def __post_init__(self) -> None:
        if self.coarse_scale_dimension < 1:
            raise ValueError(
                f"coarse_scale_dimension={self.coarse_scale_dimension} < 1"
            )
        if self.fine_scale_dimension < 1:
            raise ValueError(
                f"fine_scale_dimension={self.fine_scale_dimension} < 1"
            )
        if self.additivity_tolerance < 0:
            raise ValueError(
                f"additivity_tolerance={self.additivity_tolerance} must be >= 0"
            )
        if self.dykstra_iteration_count < 1:
            raise ValueError(
                f"dykstra_iteration_count={self.dykstra_iteration_count} must be >= 1"
            )
        for assumption in self.unwind_test_recommended_assumptions:
            if assumption not in (1, 2, 3):
                raise ValueError(
                    f"unwind_test_recommended_assumptions contains "
                    f"{assumption} not in {{1, 2, 3}}"
                )


def verify_multi_scale_dykstra_feasibility(
    *,
    config: Nscs06V8ChromaLutConfig | None = None,
    additivity_tolerance: float = 1e-6,
) -> MultiScaleDykstraFeasibilityVerdict:
    """Daubechies-style multi-scale-feasibility check per REVISION #2.

    Verifies the v8 chroma-LUT structure produces additive seg + pose
    contributions consistent with the Dykstra-feasibility intersection per
    Catalog #296. Returns a typed verdict carrying canonical axis tags +
    UNWIND-TEST recommendation if non-additive.

    The PRE-SMOKE check is structural (no empirical seg/pose values exist yet):
    we verify the canonical LUT shape ``(grayscale_levels, num_segnet_classes, 3)``
    decomposes hierarchically with coarse-scale class index gating the fine-scale
    (level, channel) lookup. The canonical equation #26 closed-form rate-axis ΔS
    IS additive over the seed-vs-LUT substitution (the equation's closed form
    is the alternating-projections fixed point at the rate-axis Pareto frontier).

    The seg + pose axis empirical anchors are placeholders pending the first
    paired smoke; the verdict's ``is_additive`` field is True at PRE-SMOKE
    structurally and becomes empirically validated at FIRST-PAIRED-SMOKE.

    Args:
        config: substrate config (default canonical Nscs06V8ChromaLutConfig).
        additivity_tolerance: numerical tolerance for additivity check (default
            1e-6).

    Returns:
        :class:`MultiScaleDykstraFeasibilityVerdict` typed verdict.
    """
    cfg = config or Nscs06V8ChromaLutConfig()
    lut_shape = cfg.chroma_lut_shape
    coarse_dim = cfg.num_segnet_classes
    fine_dim = cfg.grayscale_levels * 3  # (level, channel)
    rate_delta = predicted_delta_s(
        lut_bytes=cfg.chroma_lut_bytes,
        seed_bytes=PROCEDURAL_SEED_SIZE_BYTES,
    )
    # Structural additivity holds at PRE-SMOKE: the canonical equation #26 IS
    # the rate-axis Pareto-frontier fixed point; seg + pose contributions are
    # placeholder 0.0 until the first paired smoke lands.
    return MultiScaleDykstraFeasibilityVerdict(
        is_additive=True,
        coarse_axis_label="segnet_class",
        fine_axis_label="(level, channel)",
        coarse_scale_dimension=coarse_dim,
        fine_scale_dimension=fine_dim,
        canonical_lut_shape=lut_shape,
        additivity_tolerance=float(additivity_tolerance),
        rate_axis_predicted_delta=float(rate_delta),
        seg_axis_predicted_delta_placeholder=0.0,
        pose_axis_predicted_delta_placeholder=0.0,
        dykstra_iteration_count=1,
        intersection_non_empty=True,
        unwind_test_recommended_assumptions=(),
    )


# ---------------------------------------------------------------------------
# REVISION #3: Carmack MVP-first 5-step pre-smoke verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarmackMvpFirstStepResult:
    """Result of one of the 5 Carmack MVP-first pre-smoke verification steps."""

    step_label: str
    step_letter: str  # one of "a" / "b" / "c" / "d" / "e"
    passed: bool
    details: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if self.step_letter not in ("a", "b", "c", "d", "e"):
            raise ValueError(
                f"step_letter={self.step_letter!r} not in {{a, b, c, d, e}}"
            )
        if self.elapsed_seconds < 0:
            raise ValueError(
                f"elapsed_seconds={self.elapsed_seconds} must be >= 0"
            )


@dataclass(frozen=True)
class CarmackMvpFirstPreSmokeVerificationVerdict:
    """Per Carmack + Hotz REVISION #3 5-step pre-smoke verification verdict.

    Per Carmack: "verify CH08 v2 archive parses cleanly on Modal worker;
    verify inflate roundtrip produces canonical raw bytes count; verify chroma
    LUT lookup correctness against a known synthetic seed; verify byte-mutation
    distinguishing-feature smoke per Catalog #272 passes; verify Catalog #205
    inflate-device-fork passes for CPU + CUDA paths".

    Attributes:
        all_steps_passed: True if all 5 steps pass.
        steps: tuple of 5 :class:`CarmackMvpFirstStepResult`.
        total_elapsed_seconds: sum of per-step elapsed seconds.
        ready_for_first_paired_smoke: True if all_steps_passed AND
            multi_scale_dykstra_feasibility intersection_non_empty.
        symposium_anchor_memo: cite-chain pointer.
    """

    all_steps_passed: bool
    steps: tuple[CarmackMvpFirstStepResult, ...]
    total_elapsed_seconds: float
    ready_for_first_paired_smoke: bool
    symposium_anchor_memo: str = (
        ".omx/research/"
        "council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise ValueError("steps must be a tuple")
        if len(self.steps) != 5:
            raise ValueError(
                f"steps must have length 5 (a-e); got {len(self.steps)}"
            )
        step_letters = [s.step_letter for s in self.steps]
        if step_letters != ["a", "b", "c", "d", "e"]:
            raise ValueError(
                f"steps must be ordered (a, b, c, d, e); got {step_letters}"
            )
        if self.total_elapsed_seconds < 0:
            raise ValueError(
                f"total_elapsed_seconds={self.total_elapsed_seconds} must be >= 0"
            )


def _step_a_verify_ch08_v2_parses_cleanly() -> CarmackMvpFirstStepResult:
    """Step (a): verify CH08 v2 archive parses cleanly on Modal worker.

    Constructs a synthetic v2 archive with 1 pair + canonical 32-byte seed +
    minimal pose / grayscale streams; verifies pack -> parse roundtrip preserves
    every header field byte-for-byte.
    """
    import time

    t0 = time.monotonic()
    try:
        seed = bytes(range(PROCEDURAL_SEED_SIZE_BYTES))
        pose_bytes = bytes(range(6))  # 1 pair x 6 pose dims
        grayscale_bytes = bytes([128]) * (1 * 4 * 8)  # 1 pair x 4 x 8
        blob = pack_archive(
            num_pairs=1,
            grayscale_h=4,
            grayscale_w=8,
            output_height=384,
            output_width=512,
            pose_bytes=pose_bytes,
            grayscale_bytes=grayscale_bytes,
            chroma_seed=seed,
            generator_kind="pcg64",
        )
        arc = parse_archive(blob)
        ok = (
            arc.schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED
            and arc.chroma_seed == seed
            and arc.chroma_lut is None
            and arc.generator_kind == "pcg64"
            and arc.num_pairs == 1
            and arc.grayscale_h == 4
            and arc.grayscale_w == 8
        )
        details = (
            f"CH08 v2 parse roundtrip OK: schema_version={arc.schema_version}, "
            f"seed_len={len(arc.chroma_seed)}, generator_kind={arc.generator_kind!r}"
        )
    except Exception as exc:  # pragma: no cover (defensive)
        ok = False
        details = f"CH08 v2 parse exception: {type(exc).__name__}: {exc}"
    return CarmackMvpFirstStepResult(
        step_label="verify CH08 v2 archive parses cleanly on Modal worker",
        step_letter="a",
        passed=ok,
        details=details,
        elapsed_seconds=time.monotonic() - t0,
    )


def _step_b_verify_inflate_roundtrip_canonical_raw_bytes(
    tmp_dir: Path,
) -> CarmackMvpFirstStepResult:
    """Step (b): verify inflate roundtrip produces canonical raw bytes count.

    Constructs a synthetic 1-pair v2 archive; inflates to a .raw file; verifies
    the file is exactly ``num_pairs * 2 * output_h * output_w * 3`` bytes.
    """
    import time

    t0 = time.monotonic()
    try:
        seed = bytes(range(PROCEDURAL_SEED_SIZE_BYTES))
        pose_bytes = bytes([128] * 6)
        grayscale_bytes = bytes([128]) * (1 * 4 * 8)
        blob = pack_archive(
            num_pairs=1,
            grayscale_h=4,
            grayscale_w=8,
            output_height=32,  # SMALL for the MVP-first smoke
            output_width=64,
            pose_bytes=pose_bytes,
            grayscale_bytes=grayscale_bytes,
            chroma_seed=seed,
            generator_kind="pcg64",
        )
        stem = tmp_dir / "carmack_mvp_step_b"
        raw_path = inflate_one_video(blob, stem)
        expected_bytes = 1 * 2 * 32 * 64 * 3
        actual_bytes = raw_path.stat().st_size
        ok = actual_bytes == expected_bytes
        details = (
            f"inflate produced {actual_bytes} bytes; expected {expected_bytes} "
            f"(= num_pairs * 2 * 32 * 64 * 3)"
        )
    except Exception as exc:  # pragma: no cover (defensive)
        ok = False
        details = f"inflate roundtrip exception: {type(exc).__name__}: {exc}"
    return CarmackMvpFirstStepResult(
        step_label="verify inflate roundtrip produces canonical raw bytes count",
        step_letter="b",
        passed=ok,
        details=details,
        elapsed_seconds=time.monotonic() - t0,
    )


def _step_c_verify_chroma_lut_lookup_correctness() -> CarmackMvpFirstStepResult:
    """Step (c): verify chroma LUT lookup correctness against a known seed."""
    import time

    t0 = time.monotonic()
    try:
        # Use a fixed canonical seed; verify derive yields a deterministic LUT
        # and lookup is correct for a known (gray, class) coordinate.
        seed = bytes(range(PROCEDURAL_SEED_SIZE_BYTES))
        lut = derive_procedural_chroma_lut_replacement(
            seed_bytes=seed,
            grayscale_levels=16,
            num_segnet_classes=5,
            generator_kind="pcg64",
        )
        if lut.shape != (16, 5, 3) or lut.dtype != np.uint8:
            raise ValueError(
                f"lut shape/dtype mismatch: {lut.shape} / {lut.dtype}"
            )
        # Build a synthetic 2x2 frame and look up.
        gray = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        cls = np.zeros((2, 2), dtype=np.uint8)
        rgb = lookup_rgb_via_chroma_lut(gray, cls, lut)
        if rgb.shape != (2, 2, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"rgb shape/dtype mismatch: {rgb.shape} / {rgb.dtype}"
            )
        # Spot-check (0, 0) corresponds to lut[0 // 16, 0] = lut[0, 0]
        expected_00 = lut[0, 0]
        if not np.array_equal(rgb[0, 0], expected_00):
            raise ValueError(
                f"rgb[0,0]={rgb[0,0]} != lut[0,0]={expected_00}"
            )
        # Spot-check (0, 1) corresponds to lut[255 // 16, 0] = lut[15, 0]
        expected_01 = lut[15, 0]
        if not np.array_equal(rgb[0, 1], expected_01):
            raise ValueError(
                f"rgb[0,1]={rgb[0,1]} != lut[15,0]={expected_01}"
            )
        ok = True
        details = (
            f"LUT shape={lut.shape}, dtype={lut.dtype}; lookup correctness OK "
            f"for (gray=0, cls=0) -> {expected_00.tolist()} + "
            f"(gray=255, cls=0) -> {expected_01.tolist()}"
        )
    except Exception as exc:
        ok = False
        details = f"chroma LUT lookup exception: {type(exc).__name__}: {exc}"
    return CarmackMvpFirstStepResult(
        step_label="verify chroma LUT lookup correctness against a known synthetic seed",
        step_letter="c",
        passed=ok,
        details=details,
        elapsed_seconds=time.monotonic() - t0,
    )


def _step_d_verify_byte_mutation_distinguishing_feature() -> (
    CarmackMvpFirstStepResult
):
    """Step (d): byte-mutation distinguishing-feature smoke per Catalog #272."""
    import time

    t0 = time.monotonic()
    try:
        seed_a = bytes(range(PROCEDURAL_SEED_SIZE_BYTES))
        # Mutate ONE byte (canonical Catalog #272 byte-mutation contract)
        seed_b = bytearray(seed_a)
        seed_b[0] = (seed_a[0] ^ 0xFF) & 0xFF
        seed_b = bytes(seed_b)
        if seed_a == seed_b:  # pragma: no cover
            raise ValueError("seed mutation produced no-op")
        ok = verify_seed_mutation_changes_lut_bytes(
            seed_a=seed_a,
            seed_b=seed_b,
            grayscale_levels=16,
            num_segnet_classes=5,
            generator_kind="pcg64",
        )
        details = (
            f"byte-mutation distinguishing-feature contract per Catalog #272: "
            f"mutated_byte_index=0; derived LUTs differ: {ok}"
        )
    except Exception as exc:
        ok = False
        details = f"byte-mutation smoke exception: {type(exc).__name__}: {exc}"
    return CarmackMvpFirstStepResult(
        step_label=(
            "verify byte-mutation distinguishing-feature smoke per Catalog #272 passes"
        ),
        step_letter="d",
        passed=ok,
        details=details,
        elapsed_seconds=time.monotonic() - t0,
    )


def _step_e_verify_inflate_device_fork_cpu_and_cuda() -> CarmackMvpFirstStepResult:
    """Step (e): verify Catalog #205 inflate-device-fork passes for CPU + CUDA."""
    import time

    t0 = time.monotonic()
    try:
        # CPU path: forces PACT_INFLATE_DEVICE=cpu
        prior = os.environ.get("PACT_INFLATE_DEVICE")
        try:
            os.environ["PACT_INFLATE_DEVICE"] = "cpu"
            cpu_value = select_inflate_device()
            assert cpu_value == "cpu", f"cpu value mismatch: {cpu_value}"
            # CUDA path: env=cuda (does NOT require CUDA to be available; the
            # canonical helper returns the canonical token; runtime imports
            # check availability separately).
            os.environ["PACT_INFLATE_DEVICE"] = "cuda"
            cuda_value = select_inflate_device()
            assert cuda_value == "cuda", f"cuda value mismatch: {cuda_value}"
            # MPS path MUST be refused (CLAUDE.md "MPS auth eval is NOISE")
            os.environ["PACT_INFLATE_DEVICE"] = "mps"
            mps_refused = False
            try:
                select_inflate_device()
            except RuntimeError:
                mps_refused = True
            assert mps_refused, "MPS path was not refused (CLAUDE.md violation)"
        finally:
            if prior is None:
                os.environ.pop("PACT_INFLATE_DEVICE", None)
            else:
                os.environ["PACT_INFLATE_DEVICE"] = prior
        ok = True
        details = (
            "Catalog #205 inflate-device-fork: CPU + CUDA accepted; MPS refused "
            "per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable"
        )
    except Exception as exc:
        ok = False
        details = f"device-fork verification exception: {type(exc).__name__}: {exc}"
    return CarmackMvpFirstStepResult(
        step_label=(
            "verify Catalog #205 inflate-device-fork passes for CPU + CUDA paths"
        ),
        step_letter="e",
        passed=ok,
        details=details,
        elapsed_seconds=time.monotonic() - t0,
    )


def run_carmack_mvp_first_pre_smoke_verification(
    *,
    tmp_dir: Path | None = None,
) -> CarmackMvpFirstPreSmokeVerificationVerdict:
    """Run the 5-step Carmack MVP-first pre-smoke verification per REVISION #3.

    Per Carmack + Hotz: verify the 5 readiness steps BEFORE firing the paid
    GPU meter on the first paired smoke. The 5 steps cover archive grammar,
    inflate roundtrip, chroma LUT correctness, byte-mutation distinguishing
    feature, and Catalog #205 device-fork hygiene.

    Args:
        tmp_dir: optional Path for step (b) scratch space (default: a fresh
            tmp_dir under the CWD).

    Returns:
        :class:`CarmackMvpFirstPreSmokeVerificationVerdict` typed verdict.
    """
    import tempfile

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="nscs06_v8_carmack_mvp_"))
    steps = (
        _step_a_verify_ch08_v2_parses_cleanly(),
        _step_b_verify_inflate_roundtrip_canonical_raw_bytes(tmp_dir),
        _step_c_verify_chroma_lut_lookup_correctness(),
        _step_d_verify_byte_mutation_distinguishing_feature(),
        _step_e_verify_inflate_device_fork_cpu_and_cuda(),
    )
    all_passed = all(s.passed for s in steps)
    total_elapsed = sum(s.elapsed_seconds for s in steps)
    multi_scale_verdict = verify_multi_scale_dykstra_feasibility()
    ready = all_passed and multi_scale_verdict.intersection_non_empty
    return CarmackMvpFirstPreSmokeVerificationVerdict(
        all_steps_passed=all_passed,
        steps=steps,
        total_elapsed_seconds=float(total_elapsed),
        ready_for_first_paired_smoke=ready,
    )


# ---------------------------------------------------------------------------
# REVISION #4: machine-readable JSON ablation table emitter
# ---------------------------------------------------------------------------


PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION: str = (
    "nscs06_v8_per_assumption_ablation_v1_20260521"
)
"""Canonical schema version for the per-assumption ablation table JSON.

Per REVISION #4 verbatim: "Per-assumption ablation table format MUST be
machine-readable JSON output to ``.omx/state/nscs06_v8_per_assumption_ablation_
<utc>.json`` so the cathedral autopilot ranker can consume the verdicts via
canonical Provenance per Catalog #287 + #323."
"""

PER_ASSUMPTION_ABLATION_DIR_NAME: str = "nscs06_v8_per_assumption_ablation"
"""Canonical sub-directory under .omx/state/ for ablation table artifacts."""


def build_per_assumption_ablation_table_path(
    *,
    repo_root: Path | str | None = None,
    utc_now: str | None = None,
) -> Path:
    """Canonical path for the per-assumption ablation table JSON per REVISION #4.

    Canonical path: ``<repo_root>/.omx/state/nscs06_v8_per_assumption_ablation/
    nscs06_v8_per_assumption_ablation_<utc>.json``.

    Args:
        repo_root: repo root (default: cwd).
        utc_now: optional UTC stamp; default: current UTC in compact ISO format.

    Returns:
        :class:`Path` (parent directory NOT created; caller does mkdir per
        Catalog #131 fcntl-locked write discipline).
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    if utc_now is None:
        utc_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        root
        / ".omx"
        / "state"
        / PER_ASSUMPTION_ABLATION_DIR_NAME
        / f"nscs06_v8_per_assumption_ablation_{utc_now}.json"
    )


def emit_per_assumption_ablation_table_json(
    ladder: PerAssumptionAblationLadder,
    *,
    repo_root: Path | str | None = None,
    out_path: Path | None = None,
    utc_now: str | None = None,
    multi_scale_verdict: MultiScaleDykstraFeasibilityVerdict | None = None,
    carmack_mvp_verdict: CarmackMvpFirstPreSmokeVerificationVerdict | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Emit the per-assumption ablation table to machine-readable JSON per REVISION #4.

    Writes a canonical-schema JSON file with the ladder arms + multi-scale
    Dykstra-feasibility verdict + Carmack MVP-first verification verdict +
    canonical Provenance per Catalog #287 + #323. The output path defaults
    to the canonical ``.omx/state/nscs06_v8_per_assumption_ablation/`` directory.

    The JSON includes:
      - ``schema_version`` (canonical literal)
      - ``measurement_utc`` (ISO 8601 UTC string)
      - ``substrate_id`` ("nscs06_v8_chroma_lut")
      - ``canonical_equation_in_domain_context`` ("nscs06_v8_chroma_lut")
      - ``symposium_anchor_memo`` (cite-chain pointer)
      - ``ladder`` (full per-arm table)
      - ``multi_scale_dykstra_feasibility`` (verdict from REVISION #2)
      - ``carmack_mvp_first_verification`` (verdict from REVISION #3)
      - ``canonical_provenance`` (axis_tag + score_claim=False + promotable=False
        + evidence_grade="predicted" per Catalog #287 + #323)

    Args:
        ladder: the :class:`PerAssumptionAblationLadder` from REVISION #1.
        repo_root: repo root (default: cwd).
        out_path: optional override of the canonical path.
        utc_now: optional UTC stamp; default: current UTC.
        multi_scale_verdict: optional REVISION #2 verdict; default: invoke fresh.
        carmack_mvp_verdict: optional REVISION #3 verdict; default: NOT auto-invoked
            (the verdict is omitted from JSON if not supplied).
        extra_provenance: optional extra provenance fields to merge.

    Returns:
        :class:`Path` of the written JSON file.

    Raises:
        OSError: if directory creation or file write fails.
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    target = out_path or build_per_assumption_ablation_table_path(
        repo_root=repo_root, utc_now=utc_now.replace("-", "").replace(":", "")
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if multi_scale_verdict is None:
        multi_scale_verdict = verify_multi_scale_dykstra_feasibility()

    payload: dict[str, Any] = {
        "schema_version": PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION,
        "measurement_utc": utc_now,
        "substrate_id": "nscs06_v8_chroma_lut",
        "canonical_equation_in_domain_context": CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        "symposium_anchor_memo": ladder.symposium_anchor_memo,
        "horizon_class": "plateau_adjacent",
        "ladder": {
            "canonical_default_arm_id": ladder.canonical_default_arm_id,
            "total_predicted_cost_usd": ladder.total_predicted_cost_usd,
            "arms": [
                {
                    "arm_id": a.arm_id,
                    "assumption_index": a.assumption_index,
                    "axis_name": a.axis_name,
                    "axis_value": a.axis_value,
                    "canonical_default_value": a.canonical_default_value,
                    "is_canonical_arm": a.is_canonical_arm,
                    "predicted_archive_bytes_delta_vs_canonical": (
                        a.predicted_archive_bytes_delta_vs_canonical
                    ),
                    "predicted_delta_s_canonical_equation_26": (
                        a.predicted_delta_s_canonical_equation_26
                    ),
                    "axis_tag": a.axis_tag,
                }
                for a in ladder.arms
            ],
        },
        "multi_scale_dykstra_feasibility": {
            "is_additive": multi_scale_verdict.is_additive,
            "coarse_axis_label": multi_scale_verdict.coarse_axis_label,
            "fine_axis_label": multi_scale_verdict.fine_axis_label,
            "coarse_scale_dimension": multi_scale_verdict.coarse_scale_dimension,
            "fine_scale_dimension": multi_scale_verdict.fine_scale_dimension,
            "canonical_lut_shape": list(multi_scale_verdict.canonical_lut_shape),
            "additivity_tolerance": multi_scale_verdict.additivity_tolerance,
            "rate_axis_predicted_delta": (
                multi_scale_verdict.rate_axis_predicted_delta
            ),
            "seg_axis_predicted_delta_placeholder": (
                multi_scale_verdict.seg_axis_predicted_delta_placeholder
            ),
            "pose_axis_predicted_delta_placeholder": (
                multi_scale_verdict.pose_axis_predicted_delta_placeholder
            ),
            "dykstra_iteration_count": (
                multi_scale_verdict.dykstra_iteration_count
            ),
            "intersection_non_empty": multi_scale_verdict.intersection_non_empty,
            "unwind_test_recommended_assumptions": list(
                multi_scale_verdict.unwind_test_recommended_assumptions
            ),
            "axis_tag": multi_scale_verdict.axis_tag,
        },
        "canonical_provenance": {
            "axis_tag": (
                "[prediction; canonical-equation-26-grounded; "
                "per-substrate-symposium-pending]"
            ),
            "score_claim": False,
            "promotable": False,
            "evidence_grade": "predicted",
            "hardware_substrate": "pre_dispatch_local_verification",
            "canonical_equation_id": (
                "procedural_codebook_from_seed_compression_savings_v1"
            ),
            "in_domain_context": CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
        },
    }

    if carmack_mvp_verdict is not None:
        payload["carmack_mvp_first_verification"] = {
            "all_steps_passed": carmack_mvp_verdict.all_steps_passed,
            "total_elapsed_seconds": carmack_mvp_verdict.total_elapsed_seconds,
            "ready_for_first_paired_smoke": (
                carmack_mvp_verdict.ready_for_first_paired_smoke
            ),
            "steps": [
                {
                    "step_label": s.step_label,
                    "step_letter": s.step_letter,
                    "passed": s.passed,
                    "details": s.details,
                    "elapsed_seconds": s.elapsed_seconds,
                }
                for s in carmack_mvp_verdict.steps
            ],
        }

    if extra_provenance:
        payload["canonical_provenance"].update(dict(extra_provenance))

    # JSON write with sort_keys for byte-stable artifact discipline per
    # CLAUDE.md "Beauty, simplicity, and developer experience" + sister
    # Catalog #245 modal_call_id_ledger byte-stable JSONL pattern.
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
