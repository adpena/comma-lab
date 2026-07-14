# SPDX-License-Identifier: MIT
"""Canonical throughput-frontier laws for exact reduction and certified precision.

This module is deliberately registration-inert. It builds six research-only
equations and exposes one aggregate populator whose registry and lock paths must
be supplied explicitly. Five laws are analytic and therefore carry no empirical
anchors. The support-closure law has one hash-pinned source-inspection anchor for
the settled frozen-SegNet dependency-closure receipt; it is not a runtime or
score measurement.
"""

from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EXACT_REDUCTION_EQUATION_ID = "exact_commutative_reduction_reorder_invariance_v1"
ARGMAX_CERTIFICATE_EQUATION_ID = "interval_argmax_enclosure_certificate_v1"
PRECISION_WATERFILL_EQUATION_ID = "certified_layer_precision_waterfill_v1"
SUPPORT_CLOSURE_EQUATION_ID = "segnet_exact_dependency_closure_flop_ceiling_v1"
ORDINAL_MARGIN_EQUATION_ID = "top1_ordinal_margin_minimality_v1"
SIGMA_METRIC_EQUATION_ID = "multiphase_sigma_metric_closure_gamma_admissibility_v1"
EQUATION_IDS = (
    EXACT_REDUCTION_EQUATION_ID,
    ARGMAX_CERTIFICATE_EQUATION_ID,
    PRECISION_WATERFILL_EQUATION_ID,
    ORDINAL_MARGIN_EQUATION_ID,
    SIGMA_METRIC_EQUATION_ID,
    SUPPORT_CLOSURE_EQUATION_ID,
)

MEMO = ".omx/research/throughput_frontier_math_20260714T015118Z.md"
DERIVATION_UTC = "2026-07-14T01:49:24Z"
EXTENSION_DERIVATION_UTC = "2026-07-14T02:42:05Z"
SUPPORT_RECEIPT_UTC = "2026-07-13T16:22:11Z"
THEORY_AXIS = (
    "[DERIVED analytic law; NumPy/Python reference; research-only MEANS; "
    "no score/promotion authority]"
)
SUPPORT_AXIS = (
    "[source-inspected architecture proof + macOS-CPU advisory n600 coverage; "
    "NON-PROMOTABLE MEANS]"
)

SUPPORT_RECEIPT = (
    "experiments/results/cheapen_real95_tilehalo_fp16_20260713/"
    "tile_halo_receipt.json"
)
SUPPORT_RECEIPT_SHA256 = (
    "b9f264166fea40224966c1902065eebd3fb34949750f87d7fd020e963bb99465"
)


def _theory_provenance(
    *, reactivation_criteria: str, captured_at_utc: str = DERIVATION_UTC
):
    return build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=reactivation_criteria,
        measurement_axis=THEORY_AXIS,
        hardware_substrate="backend_free_numpy_python_reference",
        captured_at_utc=captured_at_utc,
    )


def _common_domain(
    *,
    domain: str,
    verdict_scope: str,
    authority: str,
    req_r: str,
    distinct_from: tuple[str, ...],
) -> dict[str, object]:
    """Return the mandatory no-false-authority domain fields."""

    return {
        "research_only": True,
        "domain": domain,
        "verdict_scope": verdict_scope,
        "authority": authority,
        "req_R": req_r,
        "distinct_from": distinct_from,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def build_exact_commutative_reduction_reorder_invariance_v1() -> CanonicalEquation:
    """Build the minimal algebraic condition for reorder-invariant reduction."""

    provenance = _theory_provenance(
        reactivation_criteria=(
            "bind an actual target reduction to a proved range bound, then pass real-n600 "
            "cross-process target-device bit comparison and deterministic finalization"
        )
    )
    return CanonicalEquation(
        equation_id=EXACT_REDUCTION_EQUATION_ID,
        name="Exact commutative reduction is invariant to execution reordering",
        one_line_summary=(
            "A reduction is reorder-invariant when every reachable partial sum is represented "
            "exactly in a commutative monoid and one deterministic finalization follows."
        ),
        latex_form=(
            r"(\mathcal A,\oplus,0)\ \mathrm{commutative},\ "
            r"E(\bigoplus_i x_i)\ \mathrm{exact}\Rightarrow "
            r"q(\bigoplus_{i=1}^{n}x_i)=q(\bigoplus_{i=1}^{n}x_{\pi(i)});\quad "
            r"|x_i|\le A,\ nA\le2^{w-1}-1,\ "
            r"w_{\min}=1+\lceil\log_2(nA+1)\rceil"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:"
            "fixed_width_reduction_certificate"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "bounded exact fixed-point/integer reductions, or injective CRT reductions, "
                "with no overflow, saturation, rounded intermediate, or nondeterministic finalizer"
            ),
            verdict_scope=(
                "THEOREM over declared bounded reductions; failure of one width/scale is an "
                "INSTANCE, not a fixed-point, RNS, accumulator, or throughput-family negative"
            ),
            authority=THEORY_AXIS,
            req_r=(
                "for deployment, prove the actual op/tensor fan-in and absolute-sum bound on real "
                "n600 0.mkv states, then bit-compare all outputs across fresh target-device processes"
            ),
            distinct_from=(
                "decode_determinism_integer_arithmetic_v1: decode discipline plus one measured "
                "Q15 resize-adjoint instance; this law supplies the minimal range condition",
                "witness_fp_reorder_transform_bit_identity_wall_v1: measured floating-point "
                "reorder wall, not a sufficient exact-accumulation theorem",
            ),
        ),
        units_in={
            "max_abs_term_A": "integer_accumulator_units",
            "fan_in_n": "term_count",
            "accumulator_bits_w": "signed_bits",
        },
        units_out={
            "sum_abs_bound": "integer_accumulator_units",
            "minimum_signed_bits": "signed_bits",
            "reorder_invariant": "boolean",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_throughput_frontier_math",
            "tools.probe_pythagorean_exact_arithmetic_bitident",
        ),
        canonical_producers=(
            "tac.local_acceleration.throughput_frontier_math",
            "tools.probe_throughput_frontier_math",
        ),
        provenance=provenance,
    )


def build_interval_argmax_enclosure_certificate_v1() -> CanonicalEquation:
    """Build the strict interval-separation certificate for argmax identity."""

    provenance = _theory_provenance(
        reactivation_criteria=(
            "produce sound per-class logit error enclosures for every real-n600 pixel through "
            "the actual quantized graph and R path; equality remains uncertified"
        )
    )
    return CanonicalEquation(
        equation_id=ARGMAX_CERTIFICATE_EQUATION_ID,
        name="Interval enclosure certificate for argmax preservation",
        one_line_summary=(
            "The reference winner is certified only when its lower logit bound is strictly above "
            "every competitor upper bound; with uniform error epsilon, margin must exceed 2 epsilon."
        ),
        latex_form=(
            r"a=\arg\max_c z_c,\quad "
            r"z_a-e_a>\max_{c\ne a}(z_c+e_c)\Rightarrow "
            r"\arg\max_c\widetilde z_c=a;\quad "
            r"e_c=\varepsilon\Rightarrow z_{(1)}-z_{(2)}>2\varepsilon"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:certify_argmax_intervals"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "finite reference logits with sound non-negative componentwise class absolute "
                "error bounds that include every upstream approximation and rounding contribution"
            ),
            verdict_scope=(
                "PER-PIXEL CERTIFICATE; an uncertified pixel or equality is UNKNOWN, not an argmax "
                "flip and not a negative on fixed-point, mixed precision, or tropical families"
            ),
            authority=THEORY_AXIS,
            req_r=(
                "a load-bearing zero-flip claim requires all real n600 0.mkv pixels after actual R, "
                "sound graph-wide classwise enclosures, deterministic tie semantics, and exact "
                "NumPy-fp32/reference-winner custody"
            ),
            distinct_from=(
                "scalar_top1_top2_margin_is_exact_distance_to_flip_v1: scalar margin geometry; "
                "this law composes unequal classwise implementation-error enclosures",
                "segnet_margin_trust_region_v1: operational trust-region policy; this law is the "
                "strict local certificate and grants no training or score authority",
            ),
        ),
        units_in={
            "reference_logits": "logit_units",
            "class_abs_error_bound": "logit_units",
        },
        units_out={
            "robust_margin": "logit_units",
            "certified": "boolean_per_pixel",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_throughput_frontier_math",
            "tac.bit_allocator",
        ),
        canonical_producers=(
            "tac.local_acceleration.throughput_frontier_math",
            "tools.probe_throughput_frontier_math",
        ),
        provenance=provenance,
    )


def build_certified_layer_precision_waterfill_v1() -> CanonicalEquation:
    """Build the certified error-budget/cost precision allocation law."""

    provenance = _theory_provenance(
        reactivation_criteria=(
            "measure target-device layer-option costs and derive sound layer error contributions "
            "on real n600 states; solve the discrete frontier and validate the composed certificate"
        )
    )
    return CanonicalEquation(
        equation_id=PRECISION_WATERFILL_EQUATION_ID,
        name="Certified layer-precision error-budget waterfill",
        one_line_summary=(
            "Choose one measured precision option per layer to minimize charged cost under a sound "
            "additive logit-error budget; the KKT bit formula is only a continuous initializer."
        ),
        latex_form=(
            r"\min_{o_\ell\in\mathcal O_\ell}\sum_\ell C_{\ell o_\ell}\ "
            r"\mathrm{s.t.}\ \sum_\ell E_{\ell o_\ell}\le\varepsilon;\quad "
            r"E_\ell(b)=a_\ell2^{-b}\Rightarrow "
            r"b_\ell=\left[\log_2\!\left(\lambda\ln2\,a_\ell/c_\ell\right)\right]"
            r"_{b_\ell^{\min}}^{b_\ell^{\max}}"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:"
            "solve_discrete_precision_waterfill"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "finite per-layer precision menus with sound additively composable class-logit "
                "error bounds and measured non-negative costs on one declared hardware substrate"
            ),
            verdict_scope=(
                "PRECISION-OPTION ASSIGNMENT within the supplied menus/bounds/cost surface; "
                "infeasibility is an INSTANCE, not a mixed-precision or integer-compute negative"
            ),
            authority=THEORY_AXIS,
            req_r=(
                "real n600 0.mkv per-layer error/cost table, actual R-aware final argmax certificate, "
                "worst-pair reporting, and a matched fully charged target-device wall-time replay"
            ),
            distinct_from=(
                "witness_measured_reverse_waterfill_v1: archive-byte/distortion allocation across "
                "witness tensors; this law allocates compute precision under a logit certificate",
                "pose_spd_cone_waterfill_rate_v1: pose/rate geometry, not per-layer scorer-compute "
                "precision or certified class-logit error",
            ),
        ),
        units_in={
            "error_bound": "absolute_class_logit_units",
            "measured_cost": "declared_target_device_cost_units",
            "bits": "integer_bits",
            "error_budget": "absolute_class_logit_units",
        },
        units_out={
            "selected_precision": "one_option_per_layer",
            "total_error_bound": "absolute_class_logit_units",
            "total_measured_cost": "declared_target_device_cost_units",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_throughput_frontier_math",
            "tac.bit_allocator",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=(
            "tac.local_acceleration.throughput_frontier_math",
            "tools.probe_throughput_frontier_math",
        ),
        provenance=provenance,
    )


def build_top1_ordinal_margin_minimality_v1() -> CanonicalEquation:
    """Build the minimal decision-order law behind the recos-adjacent probe."""

    provenance = _theory_provenance(
        captured_at_utc=EXTENSION_DERIVATION_UTC,
        reactivation_criteria=(
            "run a matched real-n600 CE versus zero-margin winner-rival hinge trajectory from "
            "identical EMA weights and measure per-class all/hard/easy convergence rates"
        ),
    )
    return CanonicalEquation(
        equation_id=ORDINAL_MARGIN_EQUATION_ID,
        name="Top-1 ordinal margin is the minimal argmax order constraint",
        one_line_summary=(
            "Full class-order concordance implies top-1 identity but is not necessary; the "
            "minimal loss debt is target versus strongest rival, with zero the decision boundary."
        ),
        latex_form=(
            r"\pi(z)=\pi(\tilde z)\Rightarrow\arg\max z=\arg\max\tilde z,\quad "
            r"\arg\max z=\arg\max\tilde z\not\Rightarrow\pi(z)=\pi(\tilde z);\qquad "
            r"\ell_{\rm ord}(z,y)=[m-(z_y-\max_{c\ne y}z_c)]_+,\ m=0"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:"
            "ordinal_top1_concordance_diagnostic"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "finite multiclass logits with a deterministic smallest-index tie rule; "
                "the zero-margin hinge grants no positive robustness radius or class balance"
            ),
            verdict_scope=(
                "THEOREM for argmax decision minimality; a real-n600 CE/hinge A/B decides only "
                "the arithmetic-loss INSTANCE, not ordinal-loss or geometric-boundary families"
            ),
            authority=THEORY_AXIS,
            req_r=(
                "matched real n600 0.mkv arms, identical EMA initialization/seed/order/optimizer/"
                "geometry, actual R, per-class convergence versus updates and wall time, and "
                "separate hard/easy geometry strata"
            ),
            distinct_from=(
                "interval_argmax_enclosure_certificate_v1: implementation-error proof for a "
                "fixed winner; this law compares training objectives and order constraints",
                "recos similarity: full ordinal ranking is sufficient but overconstrains "
                "loser-versus-loser permutations irrelevant to d_seg",
            ),
        ),
        units_in={
            "logits": "logit_units",
            "target_class": "class_index",
            "margin": "logit_units",
        },
        units_out={
            "top1_preserved": "boolean_per_pixel",
            "full_ordinal_concordance": "boolean_per_pixel",
            "winner_rival_hinge": "logit_units_per_pixel",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=EXTENSION_DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_ordinal_perclass_convergence",
            "tac.witness_control.perclass_verdict",
        ),
        canonical_producers=(
            "tac.local_acceleration.throughput_frontier_math",
            "tools.probe_ordinal_perclass_convergence",
        ),
        provenance=provenance,
    )


def build_multiphase_sigma_metric_closure_gamma_admissibility_v1() -> CanonicalEquation:
    """Build the metric condition and wetting relaxation for pair tensions."""

    provenance = _theory_provenance(
        captured_at_utc=EXTENSION_DERIVATION_UTC,
        reactivation_criteria=(
            "compare all-ones, triangle-valid, and metric-closed sigma arms from matched real-n600 "
            "states; separately add normal-dependent densities before claiming Wulff anisotropy"
        ),
    )
    return CanonicalEquation(
        equation_id=SIGMA_METRIC_EQUATION_ID,
        name="Multiphase pair tensions require metric closure for Gamma admissibility",
        one_line_summary=(
            "A scalar pair-tension perimeter is lower-semicontinuous only under all triangle "
            "inequalities; violations relax by intermediate-phase wetting to shortest-path costs."
        ),
        latex_form=(
            r"E_\sigma(\chi)=\frac12\sum_{i,j}\sigma_{ij}\mathcal H^{d-1}"
            r"(\partial^*E_i\cap\partial^*E_j),\quad "
            r"\sigma_{ik}\le\sigma_{ij}+\sigma_{jk};\qquad "
            r"\bar\sigma_{ik}=\min_{i=i_0,\ldots,i_r=k}\sum_{q=0}^{r-1}"
            r"\sigma_{i_q i_{q+1}}"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:"
            "multiphase_surface_tension_metric_certificate"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "finite symmetric positive scalar class-pair surface tensions; scalar sigma_ij "
                "multiplies Euclidean perimeter and does not encode interface-normal anisotropy"
            ),
            verdict_scope=(
                "THEOREM for lower-semicontinuity/metric relaxation; a triangle violation kills "
                "only that matrix INSTANCE, not pairwise perimeter, Wulff, Finsler, or sigma families"
            ),
            authority=THEORY_AXIS,
            req_r=(
                "matched real n600 0.mkv all-ones versus triangle-valid/metric-closed sigma arms, "
                "same CE and length weight, actual R, per-class all/hard/easy convergence, and "
                "thin-lane/island birth telemetry"
            ),
            distinct_from=(
                "junction_young_angle_sigma_fit_v1: measured pair-tension estimator; this law "
                "checks whether its matrix defines an admissible multiphase energy",
                "orientation-dependent Wulff/Finsler perimeter: requires phi_ij(normal), not only "
                "one scalar per class pair",
            ),
        ),
        units_in={
            "sigma_ij": "positive_relative_surface_tension",
            "interface_measure": "pixels_or_continuum_length",
        },
        units_out={
            "triangle_violation": "relative_surface_tension",
            "metric_closure": "relative_surface_tension",
            "gamma_limit_metric_admissible": "boolean",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=EXTENSION_DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_sigma_ccprime_gamma_limit",
            "tac.boundary_math.length_sigma",
        ),
        canonical_producers=(
            "tac.local_acceleration.throughput_frontier_math",
            "tools.probe_sigma_ccprime_gamma_limit",
        ),
        provenance=provenance,
    )


def _support_receipt_provenance():
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=SUPPORT_RECEIPT,
        reactivation_criteria=(
            "re-open exact spatial teacher skipping only after the frozen scorer architecture "
            "changes or a reformulation explicitly preserves all global-SE dependencies"
        ),
        measurement_axis=SUPPORT_AXIS,
        hardware_substrate="macos_arm64_cpu_plus_frozen_segnet_source_inspection",
        captured_at_utc=SUPPORT_RECEIPT_UTC,
    )
    if provenance.source_sha256 != SUPPORT_RECEIPT_SHA256:
        raise ValueError(
            "frozen-SegNet dependency-closure receipt is missing or changed; "
            "refuse stale source-inspection authority"
        )
    return provenance


def build_segnet_exact_dependency_closure_flop_ceiling_v1() -> CanonicalEquation:
    """Build exact dependency-closed FLOP accounting plus its source anchor."""

    provenance = _support_receipt_provenance()
    anchor = EmpiricalAnchor(
        anchor_id="task494_segnet_forward_dependency_closure_source_inspection_20260714",
        measurement_utc=SUPPORT_RECEIPT_UTC,
        inputs={
            "receipt_schema": "cheapen_real95_tile_halo_exactness.v1",
            "receipt_sha256": SUPPORT_RECEIPT_SHA256,
            "frozen_scorer": "tu-efficientnet_b2 SMP U-Net SegNet at 384x512",
            "n_pairs_coverage": 600,
            "requested_boundary_area_fraction": 0.04736597696940104,
        },
        predicted_output={
            "global_dependency_implies_closed_source_fraction": 1.0,
            "exact_sparse_forward_speedup_upper_bound": 1.0,
        },
        empirical_output={
            "evidence_kind": "SOURCE_INSPECTION_PLUS_SETTLED_RECEIPT",
            "global_squeeze_excite_blocks": 23,
            "local_halo_px_without_global_closure": 685,
            "local_receptive_field_px": 1311,
            "exact_dependency": "FULL_FRAME_GLOBAL",
            "exact_source_area_fraction": 1.0,
            "ideal_exact_speedup_upper_bound": 1.0,
            "exact_on_tiles_verified": False,
            "execution_status": "STRUCTURALLY_REFUSED_BEFORE_EXECUTION",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=SUPPORT_RECEIPT,
        measurement_method=(
            "hash-pinned source inspection of the settled phase-aware receptive-field and "
            "global-SqueezeExcite dependency receipt; no sparse-forward timing was executed"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=SUPPORT_CLOSURE_EQUATION_ID,
        name="Frozen-SegNet exact dependency closure fixes the forward FLOP ceiling",
        one_line_summary=(
            "Charge FLOPs on dependency-closed support, not the requested output mask; 23 global-SE "
            "blocks close this frozen SegNet to the full frame, so exact spatial speedup is at most 1x."
        ),
        latex_form=(
            r"F_{\mathrm{closed}}=\sum_\ell F_\ell\,"
            r"\mu(\operatorname{cl}_{G_\ell}M_\ell),\quad "
            r"S_{\mathrm{exact}}\le F_{\mathrm{dense}}/F_{\mathrm{closed}};\quad "
            r"\forall p_{\rm out},\ \operatorname{Dep}(p_{\rm out})=\Omega\Rightarrow "
            r"\mu(\operatorname{cl}_{G}M)=1\Rightarrow S_{\mathrm{exact}}\le1"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.throughput_frontier_math:"
            "support_closure_flop_accounting"
        ),
        domain_of_validity=_common_domain(
            domain=(
                "exact finite input-crop spatial forward for the frozen tu-efficientnet_b2 SMP "
                "U-Net SegNet at 384x512, with all global-SE dependencies and local halos charged"
            ),
            verdict_scope=(
                "FORMULATION: exact input-crop tile-with-halo forward for this frozen scorer; not "
                "a negative on stale-SE, decoder-only, local-student, cotangent, or surrogate families"
            ),
            authority=SUPPORT_AXIS,
            req_r=(
                "any reformulation must pass real n600 0.mkv exact logit/argmax bit comparison through "
                "actual R and matched-device fully charged timing; approximate training paths require "
                "explicit full-costate/optimizer-regret gates"
            ),
            distinct_from=(
                "sparse_adjoint_mask_error_and_se_support_closure_v1: backward output-cotangent "
                "mask error and low-rank law; this equation is exact forward source-support FLOPs",
                "the 4.7366pct boundary statistic: numerical decision concentration is not graph "
                "dependency closure and does not itself imply a forward-compute saving",
            ),
        ),
        units_in={
            "dense_flops": "multiply_add_operations",
            "requested_active_fraction": "unit_fraction",
            "closed_active_fraction": "unit_fraction",
        },
        units_out={
            "dependency_closed_flops": "multiply_add_operations",
            "dependency_closed_speedup_upper_bound": "dimensionless_ratio",
            "closure_tax_flops": "multiply_add_operations",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"source_inspected_exact_speedup_ceiling": 0.0},
        last_calibration_utc=DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_throughput_frontier_math",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=(
            "tac.local_acceleration.tile_halo_exactness",
            "tools.probe_tile_halo_exactness_n600",
        ),
        provenance=provenance,
    )


def build_throughput_frontier_math_equations() -> tuple[CanonicalEquation, ...]:
    """Build all six equations in their deterministic registration order."""

    equations = (
        build_exact_commutative_reduction_reorder_invariance_v1(),
        build_interval_argmax_enclosure_certificate_v1(),
        build_certified_layer_precision_waterfill_v1(),
        build_top1_ordinal_margin_minimality_v1(),
        build_multiphase_sigma_metric_closure_gamma_admissibility_v1(),
        build_segnet_exact_dependency_closure_flop_ceiling_v1(),
    )
    if tuple(equation.equation_id for equation in equations) != EQUATION_IDS:
        raise AssertionError("throughput-frontier equation order or membership drifted")
    return equations


def populate_throughput_frontier_math_equations(
    *,
    path: str | Path,
    lock_path: str | Path,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> tuple[CanonicalEquation, ...]:
    """Append exactly six events to caller-supplied registry paths.

    No default live-registry path is accepted. Tests use a temporary ledger; a
    reviewed caller may later pass the canonical path explicitly.
    """

    from tac.canonical_equations.registry import register_canonical_equation

    registry_path = Path(path)
    registry_lock_path = Path(lock_path)
    equations = build_throughput_frontier_math_equations()
    for equation in equations:
        register_canonical_equation(
            equation,
            path=registry_path,
            lock_path=registry_lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "Task #494 throughput-frontier math; research_only; MEANS; "
                "no score/promotion authority"
            ),
        )
    return equations


__all__ = [
    "ARGMAX_CERTIFICATE_EQUATION_ID",
    "EQUATION_IDS",
    "EXACT_REDUCTION_EQUATION_ID",
    "MEMO",
    "ORDINAL_MARGIN_EQUATION_ID",
    "PRECISION_WATERFILL_EQUATION_ID",
    "SIGMA_METRIC_EQUATION_ID",
    "SUPPORT_CLOSURE_EQUATION_ID",
    "SUPPORT_RECEIPT_SHA256",
    "build_certified_layer_precision_waterfill_v1",
    "build_exact_commutative_reduction_reorder_invariance_v1",
    "build_interval_argmax_enclosure_certificate_v1",
    "build_multiphase_sigma_metric_closure_gamma_admissibility_v1",
    "build_segnet_exact_dependency_closure_flop_ceiling_v1",
    "build_throughput_frontier_math_equations",
    "build_top1_ordinal_margin_minimality_v1",
    "populate_throughput_frontier_math_equations",
]
