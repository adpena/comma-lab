# SPDX-License-Identifier: MIT
"""Canonical equation: TROPNNC DENSE-TRUNK EXACT-Δd_seg REDUCTION IS EMPTY (#311, MEASURED NEGATIVE).

The #284 chapters place the witness argmax partition as a Laguerre power diagram at low softmax
temperature; a weight change leaving every Laguerre cell boundary unchanged is a byte cut at zero
Δd_seg (the tropical-skeleton rate lever, TropNNC / Misiakos et al.). This law records the MEASURED
verdict for the frozen v752 witness: at its ACTUAL deploy operating point (hosc β=1.0, ω=1.0, softmax
τ=1.0 — fully soft, NOT the low-τ max-plus limit) the trunk is DENSE and admits ZERO exactly-d_seg-
invariant structured reduction. The certificate over-predicts removability because τ=1 is far from the
tropical limit; the n-scale SegNet-argmax equality is the sole authority and REJECTS at k=1.

MEASURED (VERIFIED_VIA_EMPIRICAL_ANCHOR):
  * DENSE-TRUNK diagnostic (probe render of final-layer activations, spread of pairs): min per-unit
    activation std ~0.026, EVERY out_sdf column norm > 0.8, ZERO dead units, max final-layer pairwise
    |corr| 0.959 (0 pairs > 0.99 -> no merge redundancy). The 96-wide trunk is fully utilised.
  * EXACT-PRESERVATION screen (render baseline + each mean-compensated reduced witness -> torch R ->
    frozen CPU-torch SegNet argmax; ACCEPT k iff ALL pairs' argmax BIT-IDENTICAL to baseline):
    n24 -> 0/24 (k=1, k=2); n96 -> 0/96 (k=1, k=2); n600 -> 0/600 COMPLETE (k=1; mean flip 1.249e-2,
    max 5.218e-2, 1839s). bytes_saved_at_exact_Δd_seg=0 = 0.
  * Raw structural byte potentials (exact int8+brotli, NON-admissible): baseline trunk blob 82,706 B;
    k1 -> 81,905 (-801), k2 -> 81,139 (-1,567), k4 -> 79,654 (-3,052) — the bytes the lever WOULD cut
    IF any k preserved the partition; none does.

VERDICT_SCOPE = FORMULATION (this checkpoint's fully-soft β=1/τ=1 operating point + dense 96-wide
trunk), NOT FAMILY. REACTIVATION: re-measure on a genuinely low-τ / high-β (annealed, saturated)
checkpoint where hosc saturation manufactures dead/dominated units and the Laguerre-boundary-
invariance argument becomes tight; the apparatus (tac.boundary_math.tropnnc_witness_reduction +
tools/witness_apply_pass.py _tropnnc stage) is BUILT and fires via --fire-scorer-stages.

means != ends: a rate MEANS that produced NO admissible cut. [macOS-CPU advisory] NON-PROMOTABLE;
pointer 0.19108282 UNMOVED. DSL leg = the #311 apply-pass stage (tools/witness_apply_pass.py);
DAG = FEED-tropnnc311; memo = .omx/research/tropnnc_311_20260712T010936Z.md.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "tropnnc_dense_trunk_exact_dseg_reduction_empty_v1"

_UTC = "2026-07-12T01:09:36Z"
_ADVISORY = "[macOS-CPU advisory]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO = ".omx/research/tropnnc_311_20260712T010936Z.md"
_MODULE = "src/tac/boundary_math/tropnnc_witness_reduction.py"

# --- DENSE-TRUNK diagnostic (probe) ---
_MIN_ACT_STD = 0.026
_MIN_OUT_SDF_COL_NORM = 0.80
_N_DEAD_UNITS = 0
_MAX_FINAL_PAIRWISE_CORR = 0.9594

# --- EXACT-PRESERVATION screen (SegNet argmax equality reduced-vs-baseline) ---
_N24_K1_EQUAL = 0
_N24_K2_EQUAL = 0
_N96_K1_EQUAL = 0
_N96_K2_EQUAL = 0
_N96_PAIRS = 96
_MEAN_FLIP_K1 = 9.9e-3
_BYTES_SAVED_AT_EXACT_DSEG0 = 0

# --- raw structural byte potentials (NON-admissible; the counterfactual) ---
_BASELINE_TRUNK_BLOB = 82_706
_K1_POTENTIAL_BYTES = 801
_K2_POTENTIAL_BYTES = 1_567
_K4_POTENTIAL_BYTES = 3_052


def build_tropnnc_dense_trunk_exact_reduction_empty_v1() -> CanonicalEquation:
    """Build the TropNNC dense-trunk negative law: the measured empty exact-reduction set on the
    fully-soft v752 witness trunk."""

    anchor = EmpiricalAnchor(
        anchor_id="tropnnc_dense_trunk_exact_reduction_empty_20260712",
        measurement_utc=_UTC,
        inputs={
            "checkpoint": "levelset_v752_baseline_20260710T185913Z/levelset_witness_ema_BEST.npz",
            "operating_point": "hosc beta=1.0 omega=1.0, softmax tau=1.0 (fully soft; NOT low-tau max-plus)",
            "reduction": (
                "mean-compensated uniform-width structured neuron prune ranked by tropical influence "
                "std(activation)*||downstream weight||; k units dropped per output layer (width H-k)"
            ),
            "accept_gate": (
                "render baseline + reduced witness -> torch R (bicubic->uint8) -> frozen CPU-torch "
                "SegNet argmax; ACCEPT k iff ALL pairs' argmax bit-identical to baseline (Δd_seg==0)"
            ),
            "mechanism": _MODULE,
            "apply_pass_stage": "tools/witness_apply_pass.py::ApplyPass._tropnnc",
            "tests": "src/tac/boundary_math/tests/test_tropnnc_witness_reduction.py (17 passing)",
        },
        predicted_output={
            "tropical_certificate_hypothesis": (
                "at low tau the argmax is a Laguerre power diagram; low-influence / dead / duplicate "
                "units are removable at zero Δd_seg"
            ),
        },
        empirical_output={
            "dense_trunk": {
                "min_activation_std": _MIN_ACT_STD,
                "min_out_sdf_col_norm": _MIN_OUT_SDF_COL_NORM,
                "n_dead_units": _N_DEAD_UNITS,
                "max_final_layer_pairwise_corr": _MAX_FINAL_PAIRWISE_CORR,
            },
            "exact_preservation_screen": {
                "n24_k1_argmax_equal": _N24_K1_EQUAL,     # 0/24
                "n24_k2_argmax_equal": _N24_K2_EQUAL,     # 0/24
                "n96_k1_argmax_equal": _N96_K1_EQUAL,     # 0/96
                "n96_k2_argmax_equal": _N96_K2_EQUAL,     # 0/96
                "n96_pairs": _N96_PAIRS,
                "mean_argmax_flip_frac_k1": _MEAN_FLIP_K1,
                "n600_k1_argmax_equal": 0,               # 0/600 COMPLETE (the n-scale authority)
                "n600_pairs": 600,
                "n600_mean_argmax_flip_frac_k1": 1.249e-2,
                "n600_max_argmax_flip_frac_k1": 5.218e-2,
                "n600_artifact": "experiments/results/tropnnc_311_work/n600_final.out "
                                 "(1839s GT-free reduced-vs-baseline SegNet-argmax equality)",
            },
            "bytes_saved_at_exact_dseg0": _BYTES_SAVED_AT_EXACT_DSEG0,   # 0
            "raw_structural_byte_potentials_NON_admissible": {
                "baseline_trunk_blob_bytes": _BASELINE_TRUNK_BLOB,
                "k1_potential_bytes": _K1_POTENTIAL_BYTES,
                "k2_potential_bytes": _K2_POTENTIAL_BYTES,
                "k4_potential_bytes": _K4_POTENTIAL_BYTES,
            },
            "verdict": (
                "the fully-soft (tau=1) DENSE 96-wide v752 trunk admits ZERO exactly-d_seg-invariant "
                "structured reduction: even the single least-influential unit per layer (mean-folded) "
                "flips the SegNet argmax on 100% of screened pairs (0/24, 0/96, 0/600). The tropical "
                "certificate over-predicts removability because tau=1 is far from the max-plus limit "
                "and the trunk carries no dead/dominated/duplicate unit surviving the uint8+argmax "
                "tolerance. bytes_saved_at_exact_Δd_seg=0 = 0."
            ),
            "verdict_scope": (
                "FORMULATION — this checkpoint's fully-soft operating point + dense trunk; NOT FAMILY. "
                "REACTIVATE on a low-tau/high-beta annealed (saturated) checkpoint where the Laguerre-"
                "boundary-invariance argument becomes tight; the apparatus is built and ready."
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="mean_compensated_structured_prune + render->R->frozen_cpu_torch_segnet_argmax_equality",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "re-run tools/witness_apply_pass.py --fire-scorer-stages --eval-pairs 600 on a low-tau/"
                "high-beta annealed checkpoint; re-measure whether any k preserves the SegNet argmax"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "TropNNC dense-trunk exact-Δd_seg reduction is EMPTY: the fully-soft (β=1/τ=1) 96-wide v752 "
            "witness trunk admits NO exactly-d_seg-invariant structured neuron prune — every k≥1 flips "
            "the SegNet argmax on 100% of screened pairs (0/24, 0/96, 0/600 n600-COMPLETE); bytes_saved_at_exact_Δd_seg=0 = 0"
        ),
        one_line_summary=(
            "dense τ=1 witness trunk: 0 bytes at exact Δd_seg=0 (0/24, 0/96, 0/600 argmax-equal). "
            "FORMULATION-scoped; reactivate on low-τ. Pointer UNMOVED."
        ),
        latex_form=(
            r"\tau=1,\ \min_j \mathrm{std}(h_j)\approx0.026,\ \#\{\mathrm{dead}\}=0 \Rightarrow "
            r"\nexists\,k\!\ge\!1:\ \mathrm{argmax}\,\mathrm{Seg}(R(\hat f_k))=\mathrm{argmax}\,\mathrm{Seg}(R(f))\ \forall\,\text{pairs}"
        ),
        python_callable_module_path="tac.boundary_math.tropnnc_witness_reduction:build_reduction_plan",
        domain_of_validity={
            "vehicle": ["level_set_witness"],
            "checkpoint": "levelset_v752_baseline (hosc beta=1, omega=1, softmax tau=1; dense 96-wide trunk)",
            "labels": "n24 + n96 + n600 SegNet argmax (reduced-vs-baseline equality; n600 COMPLETE 0/600)",
            "verdict_scope": (
                "FORMULATION — the empty exact-reduction set is a property of the fully-soft operating "
                "point + dense trunk, NOT of the tropical-skeleton paradigm; reactivate on low-tau"
            ),
            "scope": "structured trunk reduction RATE lever; measured NEGATIVE (0 admissible bytes)",
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
        },
        units_in={
            "reduction_k": "units_dropped_per_layer",
            "operating_point": "softmax_tau_and_hosc_beta",
        },
        units_out={
            "bytes_saved_at_exact_dseg0": "bytes",
            "pairs_argmax_equal": "pairs",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"measured_empty_set_n24_n96": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/witness_apply_pass.py",   # #311 stage consumes the reduction + reports the verdict
            _MEMO,
        ),
        canonical_producers=(
            "tac.boundary_math.tropnnc_witness_reduction",   # the reduction + byte accounting + ranking
            "tools/witness_apply_pass.py",                   # the exact-preservation screen (_tropnnc_screen)
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "re-measure on a low-tau/high-beta annealed checkpoint via the built apply-pass stage"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_tropnnc_dense_trunk_exact_reduction_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the TropNNC dense-trunk negative law (#311).

    EQUATIONS leg of FEED-tropnnc311. DSL leg = the #311 apply-pass stage
    (tools/witness_apply_pass.py::ApplyPass._tropnnc). Mechanism =
    tac.boundary_math.tropnnc_witness_reduction."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_tropnnc_dense_trunk_exact_reduction_empty_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="tropnnc_dense_trunk_exact_dseg_reduction_empty_v1 (equations leg of #311; MEASURED "
              "NEGATIVE: dense fully-soft β=1/τ=1 witness trunk admits 0 bytes at exact Δd_seg=0, "
              "0/24 + 0/96 argmax-equal at k=1,2; FORMULATION-scoped, reactivate on low-τ; advisory "
              "NON-PROMOTABLE, pointer UNMOVED)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_tropnnc_dense_trunk_exact_reduction_empty_v1",
    "populate_tropnnc_dense_trunk_exact_reduction_equation",
]
