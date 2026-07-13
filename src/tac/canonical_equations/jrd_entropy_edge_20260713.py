# SPDX-License-Identifier: MIT
"""Canonical JRD reuse law: constrained post-hoc edge, dormant priors, and routing.

The exact result is narrower than the informal phrase "entropy edge": on the
named PR110 archive, exhaustive uniform/Laplace prefix cuts produced no
component-safe exact-ZIP improvement.  That measured constrained optimum routes
new rate pressure into training-time entropy/structure objectives.  It does not
prove every coefficient codec family dead.
"""

from __future__ import annotations

from collections.abc import Iterable

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "jrd_component_safe_entropy_edge_stop_v1"

_UTC = "2026-07-13T03:45:00Z"
_MEMO = ".omx/research/jrd_reusable_priors_harvest_20260713.md"
_CURVES = "experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/section_precision_response_curves.json"
_PR110_RECEIPT = "experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/measurement_receipt.json"
_HARVEST_RECEIPT = ".omx/research/jrd_reusable_priors_harvest_20260713.json"
_ADVISORY = "[macOS-CPU advisory] NON-PROMOTABLE"


def component_safe_posthoc_gain_bytes(baseline_bytes: int, admissible_candidate_bytes: Iterable[int]) -> int:
    """Exact positive byte gain over the component-safe admissible set."""

    baseline = int(baseline_bytes)
    candidates = [int(value) for value in admissible_candidate_bytes]
    if baseline < 0 or any(value < 0 for value in candidates):
        raise ValueError("archive byte counts must be nonnegative")
    return max([0, *(baseline - value for value in candidates)])


def kkt_mass_initialization(bias_coefficients: int, weight_coefficients: int) -> dict[str, float]:
    """Mass-only initializer for bit-budget measurement, not an allocation verdict."""

    bias = int(bias_coefficients)
    weight = int(weight_coefficients)
    if bias < 0 or weight < 0 or bias + weight == 0:
        raise ValueError("coefficient masses must be nonnegative with positive total")
    total = float(bias + weight)
    return {"bias": bias / total, "weight": weight / total}


def build_jrd_component_safe_entropy_edge_stop_v1() -> CanonicalEquation:
    n1_anchor = EmpiricalAnchor(
        anchor_id="jrd_pr110_per_tensor_response_screen_n1_20260713",
        measurement_utc=_UTC,
        inputs={
            "archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
            "eval_pairs": 1,
            "families": ["uniform", "laplace_dead_zone"],
            "planes_per_tensor_family": 8,
            "authority": "n=1 SCREEN ONLY; never load-bearing",
        },
        predicted_output={
            "use": "rank n600 measurements; do not actuate precision",
            "operator_warm_start_hypothesis": {
                "bias_planes": [5, 6],
                "weight_planes": [3, 4],
                "evidence_label": "ASSUMED, not established dataset-wide by this JSON",
            },
        },
        empirical_output={
            "response_rows": 448,
            "zero_delta_dseg_rows": 33,
            "zero_delta_dseg_positive_dpose_rows": 29,
            "zero_delta_dseg_unique_sections": [
                "refine.1.bias",
                "rgb_0.bias",
                "rgb_0.weight",
            ],
            "zero_delta_dseg_byte_shrinking_rows": 22,
            "component_safe_and_byte_shrinking_rows": 0,
            "verdict_scope": "INSTANCE-SCREEN: first pair only; no n600 tensor safety verdict",
        },
        residual=0.0,
        source_artifact=_CURVES,
        measurement_method="n1_exact_r_component_split_response_curve_rederivation",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_CURVES,
            reactivation_criteria=(
                "real-GT n600 NumPy-fp32 exact-R response curves confirm the section ordering "
                "and a receiver-closed exact archive strictly reduces bytes"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64_cpu",
        ),
    )
    n600_anchor = EmpiricalAnchor(
        anchor_id="jrd_pr110_uniform_laplace_component_safe_null_n600_20260713",
        measurement_utc=_UTC,
        inputs={
            "archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
            "eval_pairs": 600,
            "decoder_tensors": 28,
            "signed_int8_coefficients": 228_958,
            "families": ["uniform", "laplace_dead_zone"],
            "planes_per_tensor_family": 8,
        },
        predicted_output={
            "posthoc_gain_law": "max over component-safe q of max(B0-B(q),0)",
        },
        empirical_output={
            "archive_bytes": 177_169,
            "archive_bytes_saved": 0,
            "component_safe_proposals_reaching_composition": 0,
            "bias_coefficients": 2_446,
            "weight_coefficients": 226_512,
            "derived_bias_mass_fraction": 2_446 / 228_958,
            "derived_weight_mass_fraction": 226_512 / 228_958,
            "positive_repeat_noise_floor": {"d_seg": 0.0, "d_pose": 0.0},
            "fifo_parse_back_passes": 2,
            "verdict": "NULL_EXACT_PR110_POSTHOC_UNIFORM_LAPLACE",
            "verdict_scope": (
                "FORMULATION x INSTANCE: named PR110 bytes plus post-hoc uniform/Laplace "
                "prefix cuts; not a family kill for training-time or learned conditional codecs"
            ),
        },
        residual=0.0,
        source_artifact=_HARVEST_RECEIPT,
        measurement_method="n600_component_safe_exact_zip_null_rederivation",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_HARVEST_RECEIPT,
            reactivation_criteria=(
                "a structurally different witness-integrated training-time codec produces an "
                "n600 component-safe receiver-closed exact archive with fewer bytes"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="JRD component-safe post-hoc entropy-edge stop and training-time rate route",
        one_line_summary=(
            "PR110 post-hoc uniform/Laplace gain is exactly 0 B; keep n=1 ranks dormant and route rate pressure into training-time structure."
        ),
        latex_form=(
            r"G_{post}=\max_{q\in\mathcal A}[B(\theta)-B(q)]_+=0;\quad "
            r"\mathcal L=100d_{seg}+\sqrt{10d_{pose}}+\beta\widehat R(\theta)+\gamma\Omega_{struct};\quad "
            r"U'_g(b_g^*)=\mu"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.jrd_entropy_edge_20260713:component_safe_posthoc_gain_bytes"
        ),
        domain_of_validity={
            "measured_negative_scope": (
                "FORMULATION x INSTANCE: exact PR110 archive and exhaustive post-hoc uniform/Laplace int8 prefix cuts"
            ),
            "not_killed": [
                "training-time MDL or entropy penalties (task #242)",
                "training-time latent-structure regularization (task #110)",
                "low-tau structure-induced TropNNC reactivation (task #311)",
                "learned conditional or non-prefix witness codecs",
            ],
            "prior_activation_gate": (
                "real-GT n600, NumPy-fp32 bit-identical receiver, exact R, separate d_seg/d_pose, "
                "positive-repeat noise floor, exact archive bytes"
            ),
            "operator_plane_ranges": "ASSUMED warm-start only: bias 5-6, weight 3-4",
            "measurement_axis": _ADVISORY,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "archive_bytes": "exact_ZIP_bytes",
            "coefficient_budget": "stored_int8_coefficients",
            "d_seg": "frozen_SegNet_exact_R_distortion",
            "d_pose": "frozen_PoseNet_exact_R_distortion",
        },
        units_out={
            "component_safe_posthoc_gain": "exact_ZIP_bytes",
            "mass_initialization": "dimensionless_fraction",
            "route": "typed_policy_token",
        },
        empirical_anchors=(n1_anchor, n600_anchor),
        predicted_vs_empirical_residual={
            "n1_curve_rederivation": 0.0,
            "n600_exact_null_rederivation": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.jrd_priors",
            ".omx.research.sub015_DAG_jrd_reusable_priors_20260713",
            "task#110",
            "task#242",
            "task#311",
            "task#336_route_only",
        ),
        canonical_producers=(
            "tools.probe_jrd_pr110_coefficient_prefix",
            "tac.packet_compiler.jrd_pr110_coefficient_prefix",
            _HARVEST_RECEIPT,
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "register a superseding anchor after an n600 witness-integrated exact-byte response "
                "measurement or a structurally different learned codec"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64_cpu",
        ),
    )


def populate_jrd_component_safe_entropy_edge_stop_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append this law through the canonical fcntl-locked registry writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_jrd_component_safe_entropy_edge_stop_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "FEED-jrd-priors: n1 screen dormant; exact PR110 post-hoc uniform/Laplace gain 0 B; "
            "route training-time rate structure to #110/#242/#311 and instrument ownership to #336"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_jrd_component_safe_entropy_edge_stop_v1",
    "component_safe_posthoc_gain_bytes",
    "kkt_mass_initialization",
    "populate_jrd_component_safe_entropy_edge_stop_v1",
]
