# SPDX-License-Identifier: MIT
"""Canonical equation: master-gradient null-space byte fraction (v1).

Builder for ``master_gradient_null_space_byte_fraction_v1``, the formal
predictor for the *fraction of bytes in an archive whose master-
gradient is in the joint null-space of all score axes (seg, pose,
rate)*. Sister of ``builtins`` builders per Catalog #344 canonical-
equations-and-models registry.

Empirical anchor: fec6 frontier OP3-V3 [contest-CUDA T4] anchor
``a1afce29...`` measured 16,292 / 178,417 = 9.13% joint null bytes on
the canonical 600-pair extraction at operating point
``d_seg=0.001, d_pose=0.003817, rate=0.004755, score=0.4175``.

Producer/consumer wiring per CLAUDE.md "Subagent coherence-by-default"
6-hook discipline:

- Producer: ``tools/probe_null_byte_master_gradient.py``
- Consumer #1: ``tac.cathedral_consumers.null_byte_codebook_candidate_consumer``
  (Tier A observability-only routing per Catalog #341)
- Consumer #2: ``tac.procedural_codebook_generator`` (PROPOSED — Q5
  follow-on per ``procedural_codebook_generator_null_exploit_design_20260520.md``)
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)


def build_master_gradient_null_space_byte_fraction_v1() -> CanonicalEquation:
    """Equation: per-archive null-space byte fraction = |{b : max_a |dS/d_b^a| < epsilon}| / N.

    Per Catalog #318 (canonical helper ``tac.master_gradient``) +
    Catalog #344 (canonical equations registry) +
    ``.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
    Q4 verdict (null-space exploitation = STRUCTURALLY COMPLIANT
    contest-compliance path — REDUCES bytes inside archive.zip;
    rate term moves the correct direction; no maintainer-rejection
    grounds).

    The empirical anchor on the fec6 frontier (`6bae0201fb08...`) is the
    [contest-CUDA T4] OP3-V3 measurement (anchor sha `a1afce29...`).
    """
    anchor = EmpiricalAnchor(
        anchor_id="fec6_frontier_op3v3_contest_cuda_t4_null_byte_count_20260520",
        measurement_utc="2026-05-20T17:38:02Z",
        inputs={
            "archive_family": "pr101_fec6_fixed_huffman_k16",
            "scored_archive_sha256": "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            "scored_archive_bytes": 178517,
            "inner_member_bytes": 178417,
            "n_pairs_used": 600,
            "operating_point": {
                "d_seg": 0.001,
                "d_pose": 0.00381654,
                "rate": 0.004755,
                "score": 0.4175,
            },
            "measurement_axis": "[contest-CUDA]",
            "measurement_hardware": "linux_x86_64_t4_modal",
        },
        predicted_output={
            "null_byte_count_lower_bound": 0,
            "null_byte_count_upper_bound": 178417,
        },
        empirical_output={
            "null_byte_count": 16292,
            "null_byte_fraction": 0.09131416849291267,
            "seg_axis_zero_count": 16638,
            "pose_axis_zero_count": 16292,
            "rate_axis_zero_count": 178417,
            "section_breakdown": {
                "OUTER_MAGIC_null_fraction": 1.0,
                "source_len_hdr_null_fraction": 1.0,
                "source_payload_null_fraction": 0.09,
                "selector_len_hdr_null_fraction": 1.0,
                "selector_payload_null_fraction": 1.0,
            },
        },
        residual=0.0,  # this anchor IS the first empirical observation; no prior prediction
        source_artifact=".omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy",
        measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian_600pair",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy",
            reactivation_criteria="master_gradient_null_space_byte_fraction_v1_initial_anchor",
            measurement_axis="[contest-CUDA]",
            hardware_substrate="linux_x86_64_t4_modal",
        ),
    )
    return CanonicalEquation(
        equation_id="master_gradient_null_space_byte_fraction_v1",
        name="Master-gradient null-space byte fraction (per archive)",
        one_line_summary=(
            "Fraction of bytes whose master-gradient is below epsilon across all "
            "3 score axes; candidates for procedural-codebook replacement."
        ),
        latex_form=(
            r"\rho_{\text{null}}(A, \epsilon) = "
            r"\frac{|\{b : \max_{a \in \{\text{seg},\text{pose},\text{rate}\}} "
            r"|\partial S / \partial A_b^a| < \epsilon\}|}{|A|}"
        ),
        python_callable_module_path=(
            "tools.probe_null_byte_master_gradient:probe_null_bytes"
        ),
        domain_of_validity={
            "archive_families": ["pr101_fec6_fixed_huffman_k16", "pr101_lc_v2", "a1_finetuned"],
            "epsilon_range": [1e-12, 1e-6],
            "n_pairs_range": [8, 600],
            "measurement_axes": ["[contest-CUDA]", "[contest-CPU]", "[macOS-CPU advisory]"],
        },
        units_in={
            "archive_bytes": "uint8_inner_member_bytes_count",
            "epsilon": "float_score_axis_sensitivity_threshold",
            "n_pairs": "int_evaluator_pairs_used",
        },
        units_out={
            "null_byte_count": "int_byte_indices_count",
            "null_byte_fraction": "dimensionless_ratio_in_0_1",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "fec6_frontier_op3v3_contest_cuda_t4": 0.0,
        },
        last_calibration_utc="2026-05-20T17:38:02Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.null_byte_codebook_candidate_consumer",
            "tac.procedural_codebook_generator",  # PROPOSED per Q5 follow-on (memo Top-3 #2)
            "tac.master_gradient",
        ),
        canonical_producers=(
            "tools/probe_null_byte_master_gradient.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="null_space_byte_fraction_predictor.v1",
            inputs_sha256="a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc",
            measurement_axis="[contest-CUDA]",
            hardware_substrate="linux_x86_64_t4_modal",
        ),
    )
