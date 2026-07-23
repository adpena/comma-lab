# SPDX-License-Identifier: MIT
"""Canonical paint-floor mechanism laws measured by ddm_pt1x (2026-07-23).

Three measured facts, one evaluator surface (n600, pinned camera-side
bicubic/uint8 R + real SegNet.preprocess_input bilinear + frozen SegNet):

1. Hard camera-placement mechanism bar PASSES: independent boundary survival
   wall = 1,387,404 / 5,152,536 = 0.269266 -> required recovery fraction
   1 - wall = 0.730734; observed placement recovery 117,055 / 136,673 =
   0.856460 >= required. The formulation still FAILS the secondary 0.0142
   box bar (d_seg 0.021980065240, signed residual +0.007780065240).
   verdict_scope: FORMULATION (hard camera placement); geometry family OPEN
   (honest fork MIXED_OR_PLACEMENT_FOLLOWUP, texture-diagnostic sites 7,233).
2. Global amplitude-statistics DOMINANCE: the 30-byte mean/variance match arm
   lands d_seg 0.008618884616 (1,016,725 errors) vs the PT1 flat-palette
   control 0.022448043823 (2,648,079 errors) = 61.6% error reduction for 30
   incremental counted bytes on the shared 68,464 B SDWL1 description —
   69,230x the contest rate dual. Disjoint operational attribution: BN/SE
   amplitude-statistics bucket 1,578,514 corrected sites vs combined
   placement 227,431 (~6.9x) vs texture 7,233. The prosody layer (doctrine
   point 3) is the measured dominant paint-floor mechanism.
3. Control-identity + target-custody corrections (premise falsifications):
   E2 chart+semantic paint (3,349,482 errors / 0.028393910726) is NOT the
   PT1 flat-palette control (2,648,079 / 0.022448043823; -701,403 renderer
   delta) — within-experiment deltas MUST use the PT1 control; the SHA-bound
   cached ``lstars`` remain the argmax authority (16-frame batch re-forward
   drifts 3 px total across n600 — batch-geometry drift reported separately,
   never silently substituted).

Anchors: pt1x measurement receipt SHA
83e06ef47027a4997e598c824c8bb36b2185d4225ca8fcfaf8a8568fbff4b4b9 (wall
b056030ed38ac36d2643b60929053e029cf6931d63b0649cef59451f57d9dee3), findings
.omx/research/codex_findings_ddm_pt1_continuous_paint_ceiling_execute_20260723T205028Z_codex.md
(merged main fbc4285618). All rows [macOS-CPU frozen-scorer advisory]:
score_claim=false, promotable=false.
"""

from __future__ import annotations

EQUATION_ID = "ddm_pt1_paint_mechanism_laws_v1"

# Measured constants (pt1x receipts, SHA-pinned above).
FRAME1_SCORER_SITES = 117_964_800  # 600 pairs x 512 x 384
SURVIVAL_WALL_SURVIVING = 1_387_404
SURVIVAL_WALL_TOTAL_BOUNDARY_SITES = 5_152_536
PLACEMENT_ATTRIBUTABLE_ERRORS = 136_673
PLACEMENT_RECOVERED_ERRORS = 117_055
PT1_CONTROL_ERRORS = 2_648_079
HARD_PLACEMENT_ERRORS = 2_592_874
ANALYTIC_BLEND_ERRORS = 2_470_714
STATS_ARM_ERRORS = 1_016_725
SPECTRUM_ARM_ERRORS = 1_034_847
STATS_ARM_DELTA_BYTES = 30
SPECTRUM_ARM_DELTA_BYTES = 186
BOX_BAR_DSEG = 0.0142
AMPLITUDE_STATS_BUCKET_SITES = 1_578_514
PLACEMENT_BUCKET_SITES = 227_431  # 117,055 in-band + 110,376 class-interaction
TEXTURE_BUCKET_SITES = 7_233


def dseg_from_errors(errors: int) -> float:
    """Frame-1 argmax d_seg from an error-site count over the n600 grid."""
    if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
        raise ValueError("errors must be a nonnegative integer site count")
    if errors > FRAME1_SCORER_SITES:
        raise ValueError("errors cannot exceed the n600 frame_1 site count")
    return errors / FRAME1_SCORER_SITES


def mechanism_bar_passes(recovered: int, attributable: int, wall_surviving: int, wall_total: int) -> bool:
    """Placement mechanism bar: recovered/attributable >= 1 - wall fraction."""
    for name, value in (
        ("recovered", recovered),
        ("attributable", attributable),
        ("wall_surviving", wall_surviving),
        ("wall_total", wall_total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer site count")
    if recovered > attributable or wall_surviving > wall_total:
        raise ValueError("recovered<=attributable and wall_surviving<=wall_total required")
    return (recovered / attributable) >= (1.0 - wall_surviving / wall_total)


def amplitude_dominance_ratio(stats_bucket_sites: int, placement_bucket_sites: int) -> float:
    """Ratio of amplitude-statistics corrected sites to combined placement sites.

    >1 means the prosody/amplitude mechanism dominates geometric placement in
    the disjoint operational attribution (measured ~6.94 on pt1x).
    """
    for name, value in (
        ("stats_bucket_sites", stats_bucket_sites),
        ("placement_bucket_sites", placement_bucket_sites),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer site count")
    return stats_bucket_sites / placement_bucket_sites
