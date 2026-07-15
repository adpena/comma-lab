"""Measured successor row for ``rate_law_ladder_v1`` owed measurables (2026-07-13).

This row does not overwrite the four-rung theory.  It anchors the scoped n600
operational codelength measurements for D36/D37, the fixed-stratum D38 splitting
derivation, and the D39 additive telemetry implementation.  It is not a contest
score claim and does not promote either empirical proxy beyond its named surface.
"""
from __future__ import annotations

EQUATION_ID = "rate_law_ladder_v2_measured"
PREDECESSOR_EQUATION_ID = "rate_law_ladder_v1"
AXIS = "[macOS-CPU codelength advisory; n600; score_claim=false]"

MEMO = ".omx/research/ladder_owed_measurables_20260713.md"
D36_RECEIPT = (
    "experiments/results/ladder_owed_measurables_20260713/"
    "d36_fiber_completeness_gap_n600.json"
)
D37_PREDECESSOR_RECEIPT = (
    "experiments/results/ladder_owed_measurables_20260713/"
    "d37_flip_conditional_mi_n600.json"
)
D37_RECEIPT = ".omx/research/v9_cgauge_symmetry_homotopy_n600_receipt_20260714.json"
D37_RECEIPT_SHA256 = "60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308"
D39_SPEC = ".omx/research/pact_causal_manifest_v1_event_marks_increment_spec_20260713.md"
D39_MANIFEST_IMPLEMENTATION = "src/tac/causal_manifest.py"
D39_PRODUCER_IMPLEMENTATION = "src/tac/witness_control/telemetry_producers.py"
D39_TEST_IMPLEMENTATION = "src/tac/tests/test_event_mark_telemetry.py"
D39_STATUS = "BUILT_RESUME_SAFE_OBSERVABILITY_ONLY"

ARCHIVE_BYTES = 83_430
RATE_DENOMINATOR_BYTES = 37_545_489

# D36: model-codelength upper bound for H(q_G(W)|U_proxy(W)).
D36_UNCONDITIONAL_CODE_BITS = 162_840
D36_CONDITIONAL_GAP_BITS = 147_616
D36_FOLD_SENSITIVITY_CI95_BITS = (146_913.960056988, 148_318.039943012)
D36_GAP_PERCENT_OF_ARCHIVE_RATE = 22.116744576291502
D36_CONDITIONAL_SAVING_BITS_BEFORE_MODEL = 15_224
D36_PREDICTOR_CHARGE_BYTES = 15_256
D36_NET_SAVING_BITS_AFTER_MODEL = -106_824

# D37: held-out codelength gain q00-q01.  The gross quantity estimates
# I(F;C|M,Qxi) on the completed empirical surface; the net quantity charges the
# explicitly assumed fixed-row table representation.
D37_MI_GROSS_BITS = 467_373.90888513427
D37_MI_GROSS_BITS_PER_BOUNDARY_PIXEL = 0.18318460696404312
D37_MI_TABLE_CHARGE_BYTES = 10_342
D37_MI_NET_BITS = 384_637.90888513427
D37_MI_NET_CI95_BITS = (373_674.7586229076, 395_236.54874890414)
D37_PHASE_AWARE_GROSS_BITS = 464_657.2100251259
D37_PHASE_AWARE_TABLE_CHARGE_BYTES = 56_552
D37_PHASE_AWARE_NET_BITS = 12_241.210025125882
D37_PHASE_AWARE_NET_CI95_BITS = (957.5520005738945, 23_180.749990467913)
D37_VERDICT = "RESIDUAL_NON_GAUGE_STRUCTURE_DETECTED__M_NOT_SUFFICIENT"
D37_VERDICT_SCOPE = "FORMULATION x V9_EMA_BEST_N600_EMPIRICAL_SURFACE"

# D38: after explicitly restricting to a fixed regular stratum and typing a
# strict action groupoid, the semidirect extension splits by h -> (1,h).
D38_LOCAL_EXTENSION = "K_sigma semidirect H_cov,sigma"
D38_LOCAL_OBSTRUCTION_CLASS = "neutral"
D38_LOCAL_IDEAL_TWIST_BITS = 0
D38_GLOBAL_EXTENSION_STATUS = (
    "TYPED_SCHEMA / EXACT_ARRAY_INSTANCE_GLUES / GLOBAL_RATE_DESCENT_UNBOUND"
)

REMAINING_OWED = (
    "exact_packed_class_conditional_contour_AB",
    "populate_Hcov_edge_tube_junction_atlas_and_charged_receiver_section",
)


def rate_term_for_bits(bits: float) -> float:
    """Convert counted bits to the contest rate term; no distortion authority implied."""
    return 25.0 * (float(bits) / 8.0) / RATE_DENOMINATOR_BYTES


def measured_gap_statement() -> str:
    return (
        "Operational n600 H(q_G|U_proxy) upper-bound codelength = 147,616 bits "
        "(22.116745% of the 83,430-byte archive rate); the 1,903-byte raw saving "
        "does not survive the conservative 15,256-byte predictor charge."
    )


def d38_split_statement() -> str:
    return (
        "Typed strict fixed-stratum semidirect extension: SPLIT, neutral factor set, "
        "R_twist^ideal=0; global H_cov gluing schema now typed and the exact-array "
        "instance glues, but action/coefficient/receiver-rate descent remains UNBOUND."
    )


__all__ = [
    "D36_CONDITIONAL_GAP_BITS",
    "D37_MI_GROSS_BITS",
    "D37_MI_NET_BITS",
    "D37_RECEIPT_SHA256",
    "D38_LOCAL_IDEAL_TWIST_BITS",
    "D39_STATUS",
    "EQUATION_ID",
    "PREDECESSOR_EQUATION_ID",
    "d38_split_statement",
    "measured_gap_statement",
    "rate_term_for_bits",
]
