# SPDX-License-Identifier: MIT
"""Initial population of canonical equations from 2026-05-19 session findings.

Per operator NON-NEGOTIABLE 2026-05-19 verbatim: *"we need to formalize
all of this and canonicalize and operationalize because I am afraid we
are learning but if we don't have systems of equations and models and
such we are just gaining tribal knowledge"*.

The 6 initial equations:

1. ``brotli_cascade_bounded_per_stream_v1`` — PR101 commit 0696a1488
2. ``mps_drift_architecture_class_dependent_v1`` — slot 16 commit 65db9f570
3. ``per_byte_leverage_uniformly_distributed_v1`` — convergent multi-signal
4. ``per_pair_master_gradient_score_impact_taylor_v1`` — slot 9 commit ab7f8f7e2
5. ``master_gradient_locality_violation_by_codec_v1`` — slot 15+17+18
6. ``canonical_frontier_pointer_v1`` — slot 14 commit 023a2374f

Each equation carries Provenance per Catalog #323 and an initial
empirical anchor backed by a canonical artifact path. Calling
``populate_initial_equations(path=...)`` is idempotent — equations are
keyed by ``equation_id`` and the registry's APPEND-ONLY semantics mean
re-running this helper appends additional ``registered`` events (which
is the canonical re-registration trail).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_ON_RESIDUAL_DRIFT,
)
from tac.canonical_equations.registry import (
    register_canonical_equation,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)
from tac.provenance.contract import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)


# Canonical anchor SHA placeholder for design-only equations whose
# empirical anchor is a documented memo path rather than an archive sha.
_DESIGN_ANCHOR_SHA_PLACEHOLDER = "0" * 64


def _design_provenance(model_id: str) -> Provenance:
    """Build a PREDICTED Provenance for design-only equation registration."""
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=_DESIGN_ANCHOR_SHA_PLACEHOLDER,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


def _empirical_research_sidecar_provenance(
    sidecar_path: str,
    measurement_axis: str = "[predicted]",
    hardware_substrate: str = "unknown",
) -> Provenance:
    """Build a RESEARCH_SIDECAR Provenance for empirical anchor memos."""
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar_path,
        reactivation_criteria="canonical_equations_registry_initial_population",
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
    )


def build_brotli_cascade_bounded_per_stream_v1() -> CanonicalEquation:
    """Equation 1: PR101 brotli cascade is bounded per stream container.

    Source: commit 0696a1488 "codex op7 iteration items 3+4: post-brotli-
    decompress grain + MPS-axis probe + multiple-passes deterministic".
    PR101 decoder_blob = 7 brotli streams = 229,014 decompressed bytes;
    cascade is structurally bounded to a single stream (only stream 2
    differs between op7 baseline + candidate).
    """
    anchor = EmpiricalAnchor(
        anchor_id="pr101_op7_stream_2_only_differs_anchor_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "archive_family": "pr101",
            "decoder_blob_streams_count": 7,
            "decompressed_bytes_total": 229014,
        },
        predicted_output={"differing_streams_count_upper_bound": 7},
        empirical_output={"differing_streams_count": 1, "differing_stream_index": 2},
        residual=0.0,  # prediction was upper-bound; empirical is within bound
        source_artifact=".omx/research/master_gradient_post_brotli_decompress_landing_20260519.md",
        measurement_method="pr101_op7_diff",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/master_gradient_post_brotli_decompress_landing_20260519.md"
        ),
    )
    return CanonicalEquation(
        equation_id="brotli_cascade_bounded_per_stream_v1",
        name="Brotli decompression cascade is bounded per stream container",
        one_line_summary=(
            "For brotli-coded archives, a single raw-byte mutation cascade is bounded "
            "by the stream container it lives in (NOT the full archive)."
        ),
        latex_form=r"\text{cascade\_range}(b, A) = |\text{stream}_i| \text{ where } b \in \text{stream}_i",
        python_callable_module_path=(
            "tac.master_gradient_post_decompress_multi_archive:classify_cascade_severity_for_codec"
        ),
        domain_of_validity={
            "codec_family": ["brotli"],
            "archive_families": ["pr101"],
            "expected_stream_count_range": [1, 64],
        },
        units_in={
            "byte_index": "int_offset_in_compressed_payload",
            "archive_payload": "bytes",
        },
        units_out={"cascade_range_bytes": "int_decompressed_bytes_count"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"pr101_op7_diff": 0.0},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.master_gradient_post_brotli_decompress",
            "tac.master_gradient_post_decompress_multi_archive",
            "tac.cathedral_consumers.per_byte_sensitivity_consumer",
        ),
        canonical_producers=(
            "tools/master_gradient_xray.py",
            "tac.master_gradient_iterative_refinement",
        ),
        provenance=_design_provenance("brotli_cascade_predictor.v1"),
    )


def build_mps_drift_architecture_class_dependent_v1() -> CanonicalEquation:
    """Equation 2: MPS-vs-CUDA drift is architecture-class dependent, not universal.

    Source: commit 65db9f570 "mps_diagnostic: land 3 engineering corrections
    (Kahan Conv2d + pinned softmax + fp32 matmul)" + slot 16 TinyRenderer
    falsification (predicted 30x combined reduction; empirical 1.00x NO-OP).
    The corrections are PROPERLY ENGINEERED but NO-OP for architecture
    classes that lack the noise sources they target (e.g., a 12K-param
    Conv2d-heavy renderer with zero softmax has no softmax noise to fix).
    """
    anchor_tinyrenderer = EmpiricalAnchor(
        anchor_id="tinyrenderer_mps_drift_falsification_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "architecture_class": "tinyrenderer_phase_b",
            "param_count": 12000,
            "conv2d_count": 4,
            "softmax_count": 0,
            "matmul_depth": 1,
        },
        predicted_output={"drift_reduction_factor": 30.0},
        empirical_output={"drift_reduction_factor": 1.00},
        residual=30.0,  # predicted 30x off
        source_artifact=".omx/research/mps_drift_predictor_falsification_tinyrenderer_20260519.md",
        measurement_method="tinyrenderer_phase_b_paired_mps_cuda",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/mps_drift_predictor_falsification_tinyrenderer_20260519.md",
            measurement_axis="MPS-research-signal",
            hardware_substrate="m5_max_mps",
        ),
    )
    anchor_pre_existing = EmpiricalAnchor(
        anchor_id="phase_b_aggregate_mps_drift_0p072pct_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "architecture_class": "tinyrenderer_phase_b_aggregate",
            "aggregate_axis": "pre_correction",
        },
        predicted_output={"aggregate_drift_pct": 1.0},
        empirical_output={"aggregate_drift_pct": 0.072},
        residual=0.928,
        source_artifact=".omx/state/mps_research_signal_manifest.jsonl",
        measurement_method="phase_b_aggregate",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/state/mps_research_signal_manifest.jsonl",
            measurement_axis="MPS-research-signal",
            hardware_substrate="m5_max_mps",
        ),
    )
    return CanonicalEquation(
        equation_id="mps_drift_architecture_class_dependent_v1",
        name="MPS-vs-CUDA drift reduction is a function of architecture features",
        one_line_summary=(
            "MPS drift corrections are NO-OP for architectures lacking the noise sources "
            "they target; calibrate per architecture class via predict_drift(features)."
        ),
        latex_form=(
            r"\Delta_{\text{drift}}(\theta_{\text{arch}}) = f(N_{\text{conv2d}}, "
            r"N_{\text{softmax}}, D_{\text{accumulation}})"
        ),
        python_callable_module_path="tac.mps_diagnostic.drift_predictor:predict_drift",
        domain_of_validity={
            "param_count_range": [1000, 100_000_000],
            "architecture_classes": [
                "tinyrenderer",
                "segnet_class_pending",
                "posenet_class_pending",
                "renderer_class_pending",
            ],
            "noise_sources_modeled": ["conv2d_accumulation", "softmax_numerics", "matmul_fp16"],
        },
        units_in={
            "param_count": "int_parameters",
            "conv2d_count": "int_layer_count",
            "softmax_count": "int_layer_count",
            "matmul_depth": "int_layer_depth",
        },
        units_out={"drift_reduction_factor": "dimensionless_ratio_predicted_to_empirical"},
        empirical_anchors=(anchor_tinyrenderer, anchor_pre_existing),
        predicted_vs_empirical_residual={
            "tinyrenderer_phase_b_paired_mps_cuda": 30.0,
            "phase_b_aggregate": 0.928,
        },
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_RESIDUAL_DRIFT,
        canonical_consumers=(
            "tac.cathedral_consumers.mps_viable_prescreen_consumer",
            "tac.cathedral_consumers.mps_diagnostic_consumer",
            "tac.engineered_corrections.kahan_summation",
            "tac.engineered_corrections.softmax_with_epsilon",
            "tac.engineered_corrections.fp32_matmul",
            "tools/operator_authorize.py",
        ),
        canonical_producers=(
            "tac.mps_diagnostic.drift_predictor",
            "tools/audit_mps_research_signal.py",
        ),
        provenance=_design_provenance("mps_drift_predictor.v1"),
    )


def build_per_byte_leverage_uniformly_distributed_v1() -> CanonicalEquation:
    """Equation 3: top-K byte leverage scales near-linearly with K for entropy-coded archives.

    Source: convergent multi-signal — slot 2 per-pair CV=2.6% + slot 8
    per-byte 91% non-zero + slot 10 top-1% leverage 6.4% + slot 15 PR101
    stream-2 only-differs. The four orthogonal signals converge on
    "sensitivity is broadly distributed; substrate-class shifts dominate
    per-byte optimization".
    """
    anchor = EmpiricalAnchor(
        anchor_id="per_byte_leverage_top_1pct_6p4pct_pr101_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "archive_family": "pr101",
            "total_bytes": 154147,
            "top_k_percent": 1.0,
        },
        predicted_output={"top_k_leverage_ratio": 0.01},  # uniform-Pareto prediction
        empirical_output={"top_k_leverage_ratio": 0.064},
        residual=5.4,  # 6.4x uniform; mild Pareto concentration
        source_artifact=".omx/research/per_byte_leverage_uniform_distribution_audit_20260519.md",
        measurement_method="per_byte_top_k_pr101",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/per_byte_leverage_uniform_distribution_audit_20260519.md"
        ),
    )
    return CanonicalEquation(
        equation_id="per_byte_leverage_uniformly_distributed_v1",
        name=(
            "Top-K byte leverage scales near-linearly with K (not Pareto-concentrated) "
            "for entropy-coded archives"
        ),
        one_line_summary=(
            "Per-byte optimization saturates quickly; substrate-class shifts dominate per-byte "
            "edits on entropy-coded archives (PR101 top-1% leverage = 6.4%)."
        ),
        latex_form=r"\text{top\_k\_leverage}(k, N) \approx k/N + \epsilon",
        python_callable_module_path="tac.canonical_equations.builtins:uniform_leverage_predictor",
        domain_of_validity={
            "archive_families": ["pr101", "pr106", "a1", "dp1"],
            "codec_families": ["brotli", "arithmetic", "huffman_static"],
            "k_percent_range": [0.1, 50.0],
        },
        units_in={"k_percent": "float_percentage_of_total_bytes"},
        units_out={"leverage_ratio": "float_fraction_of_total_score_impact"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"per_byte_top_k_pr101": 5.4},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.per_byte_sensitivity_consumer",
            "tac.master_gradient_iterative_refinement",
        ),
        canonical_producers=(
            "tools/master_gradient_xray.py",
        ),
        provenance=_design_provenance("uniform_leverage_predictor.v1"),
    )


def uniform_leverage_predictor(k_percent: float, total_bytes: int = 154147) -> float:
    """Canonical callable for equation #3 (uniform-leverage prediction).

    Returns predicted top-K leverage ratio. The empirical refinement in
    anchor `per_byte_leverage_top_1pct_6p4pct_pr101_20260519` shows the
    empirical ratio is approximately 6.4x the uniform prediction for
    PR101 top-1%; this implementation returns the uniform baseline so the
    residual is auditable.
    """
    if k_percent < 0 or k_percent > 100:
        raise ValueError(f"k_percent must be in [0, 100], got {k_percent}")
    return k_percent / 100.0


def build_per_pair_master_gradient_score_impact_taylor_v1() -> CanonicalEquation:
    """Equation 4: per-pair score impact via Taylor first-order + Cauchy-Schwarz.

    Source: commit ab7f8f7e2 slot 9. The canonical Bayesian-experimental-
    design lens for "is drift in the score-relevant subspace or the
    nullspace?". Provides both the first-order Taylor predictor
    `δS_p ≈ Σ g_{i,p} · d_i` and the Cauchy-Schwarz upper bound
    `|δS_p| ≤ ‖g_p‖₂ · ‖d‖₂`.
    """
    anchor = EmpiricalAnchor(
        anchor_id="pr101_mps_axis_probe_cauchy_schwarz_bound_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "archive_family": "pr101",
            "pair_count": 600,
            "drift_norm_l2": 0.072,
        },
        predicted_output={"upper_bound_delta_s_l2": 0.072},
        empirical_output={"observed_delta_s_max_abs": 0.012},
        residual=0.06,  # observed well below upper bound (NULLSPACE_VIABLE)
        source_artifact=".omx/research/mps_axis_probe_pr101_cauchy_schwarz_20260519.md",
        measurement_method="pr101_mps_axis_probe",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/mps_axis_probe_pr101_cauchy_schwarz_20260519.md",
            measurement_axis="MPS-research-signal",
            hardware_substrate="m5_max_mps",
        ),
    )
    return CanonicalEquation(
        equation_id="per_pair_master_gradient_score_impact_taylor_v1",
        name="Per-pair score impact via Taylor first-order + Cauchy-Schwarz bound",
        one_line_summary=(
            "For drift d in master-gradient g-space: delta_S_p ~= sum_i g_{i,p} * d_i; "
            "|delta_S_p| <= ||g_p||_2 * ||d||_2 (Cauchy-Schwarz)."
        ),
        latex_form=(
            r"\delta S_p \approx \sum_i g_{i,p} \cdot d_i "
            r"\quad \text{with} \quad |\delta S_p| \leq \|g_p\|_2 \cdot \|d\|_2"
        ),
        python_callable_module_path="tac.mps_diagnostic.drift_predictor:cauchy_schwarz_upper_bound",
        domain_of_validity={
            "drift_norm_range_l2": [0.001, 1.0],
            "pair_count_range": [1, 10000],
            "archive_families": ["pr101", "pr106", "a1", "dp1"],
        },
        units_in={
            "master_gradient": "tensor[N_params, N_pairs]_float",
            "drift_vector": "tensor[N_params]_float",
        },
        units_out={
            "delta_s_per_pair": "tensor[N_pairs]_float",
            "upper_bound_l2": "float_score_units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"pr101_mps_axis_probe": 0.06},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.master_gradient_iterative_refinement",
            "tac.mps_diagnostic.drift_predictor",
            "tac.cathedral_consumers.per_pair_pareto_envelope_consumer",
        ),
        canonical_producers=(
            "tools/master_gradient_xray.py",
            "tac.master_gradient",
        ),
        provenance=_design_provenance("per_pair_taylor_first_order.v1"),
    )


def build_master_gradient_locality_violation_by_codec_v1() -> CanonicalEquation:
    """Equation 5: master-gradient locality violation for entropy-coded archives.

    Source: slot 15 PR101 op7 SegNet+0.0014 unexpected regression + slot 17
    class-level fix in master_gradient_post_decompress_multi_archive +
    slot 18 grain-comparison visualization. Raw-byte master-gradient
    violates locality for entropy-coded archives; post-decompress grain
    is the canonical basis.
    """
    anchor = EmpiricalAnchor(
        anchor_id="pr101_op7_segnet_regression_locality_violation_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "archive_family": "pr101",
            "codec_family": "brotli",
            "grain": "raw_byte",
        },
        predicted_output={"segnet_delta_local_first_order": 0.0},
        empirical_output={"segnet_delta_post_decompress": 0.0014},
        residual=0.0014,
        source_artifact=".omx/research/master_gradient_post_brotli_decompress_landing_20260519.md",
        measurement_method="pr101_op7_segnet_delta",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/master_gradient_post_brotli_decompress_landing_20260519.md"
        ),
    )
    return CanonicalEquation(
        equation_id="master_gradient_locality_violation_by_codec_v1",
        name=(
            "Raw-byte master-gradient violates locality for entropy-coded archives; "
            "post-decompress is the canonical basis"
        ),
        one_line_summary=(
            "For entropy-coded archives: g_raw(b) != g_post_decompress(b); "
            "use post-decompress grain for per-byte sensitivity work."
        ),
        latex_form=(
            r"g_{\text{raw}}(b) \neq g_{\text{post-decompress}}(b) "
            r"\text{ when } b \in \text{entropy-coded section}"
        ),
        python_callable_module_path=(
            "tac.master_gradient_post_decompress_multi_archive:classify_cascade_severity_for_codec"
        ),
        domain_of_validity={
            "codec_families": ["brotli", "arithmetic", "huffman_static", "lzma"],
            "archive_families": ["pr101", "pr106", "a1", "dp1", "hdm8", "pr107"],
            "cascade_severity": ["BOUNDED", "UNBOUNDED"],
        },
        units_in={
            "codec_token": "str_codec_family_name",
            "byte_offset": "int_offset_in_compressed_payload",
        },
        units_out={"cascade_severity": "str_BOUNDED_or_UNBOUNDED"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"pr101_op7_segnet_delta": 0.0014},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.master_gradient",
            "tac.master_gradient_post_decompress_multi_archive",
            "tac.cathedral_consumers.per_byte_sensitivity_consumer",
        ),
        canonical_producers=(
            "tools/master_gradient_xray.py",
            "tac.master_gradient_iterative_refinement",
        ),
        provenance=_design_provenance("master_gradient_locality_classifier.v1"),
    )


def build_canonical_frontier_pointer_v1() -> CanonicalEquation:
    """Equation 6: canonical frontier pointer model.

    Source: slot 14 commit 023a2374f. Frontier scores are pointer-only;
    hardcoded literals in CLAUDE.md / MEMORY.md drift. The canonical
    source-of-truth is computed via `argmax_anchor score(anchor)` over
    the qualifying-hardware x qualifying-axis cross product.
    """
    anchor = EmpiricalAnchor(
        anchor_id="local_frontier_0p19205_contest_cpu_gha_linux_x86_64_20260519",
        measurement_utc="2026-05-19T00:00:00Z",
        inputs={
            "axis": "[contest-CPU]",
            "hardware": "linux_x86_64_cpu",
        },
        predicted_output={"frontier_score": 0.19205},
        empirical_output={"frontier_score": 0.19205},
        residual=0.0,  # pointer-only; tautological by design
        source_artifact=".omx/state/canonical_frontier_pointer.json",
        measurement_method="gha_linux_x86_64_cpu_eval",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/state/canonical_frontier_pointer.json"
        ),
    )
    return CanonicalEquation(
        equation_id="canonical_frontier_pointer_v1",
        name="Frontier scores are pointer-only; hardcoded literals drift",
        one_line_summary=(
            "F = argmax_anchor score(anchor) over qualifying hardware x qualifying axis; "
            "consult tac.canonical_frontier_pointer instead of hardcoded literals."
        ),
        latex_form=(
            r"F = \text{argmax}_{\text{anchor}} \text{score}(\text{anchor}) "
            r"\text{ over qualifying hardware} \times \text{qualifying axis}"
        ),
        python_callable_module_path=(
            "tac.canonical_frontier_pointer:load_canonical_frontier_pointer_lenient"
        ),
        domain_of_validity={
            "qualifying_hardware": [
                "linux_x86_64_cpu",
                "linux_x86_64_t4",
                "linux_x86_64_a10g",
                "linux_x86_64_a100",
                "linux_x86_64_4090",
                "linux_x86_64_h100",
                "linux_x86_64_l40s",
            ],
            "qualifying_axes": ["[contest-CPU]", "[contest-CUDA]"],
        },
        units_in={"axis": "str_axis_tag", "hardware": "str_substrate_token"},
        units_out={"frontier_score": "float_contest_score"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"gha_linux_x86_64_cpu_eval": 0.0},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "reports/latest.md",
            "CLAUDE.md",
            "tools/cathedral_autopilot_autonomous_loop.py",
        ),
        canonical_producers=(
            "tac.canonical_frontier_pointer",
            "tools/scan_best_anchor_per_axis.py",
        ),
        provenance=_design_provenance("canonical_frontier_pointer.v1"),
    )


def build_main_thread_spawn_pv_gap_pre_catalog_376_extension_v1() -> CanonicalEquation:
    """Equation: main-thread spawn-decision PV gap predicts STAND_DOWN rate.

    Per Wave N+25 OPERATOR-CRITIQUE-DRIVEN AUDIT memo op-routable #6 +
    operator NON-NEGOTIABLE 2026-05-28 "ensure no signal loss".

    Mathematical predicate: P(STAND_DOWN | N_in_flight_sisters) >=
    N_in_flight_sisters / S_total_substrates when NOT
    Catalog_376_PV_invoked_at_main_thread. Empirical: today's 4
    STAND_DOWNs (PR111 Slot 4 RESUME / Z4 Wave N+23 / Cascade A FEC10
    Wave N+24 / paper review Wave N+13.5) all surfaced post-spawn at
    STAGING/COMMIT surfaces rather than at SPAWN time. The PARENT-side
    spawn-decision PV gap is empirical.

    Anchor 1 (THIS Wave N+25 audit memo): 4-of-N spawn mandates today
    produced STAND_DOWN where N >= 5 active subagents. P_STAND_DOWN >= 0.8
    under the unstructured-PV condition; the canonical Catalog #378
    structural enforcement is the canonical_unwind_path.
    """
    anchor = EmpiricalAnchor(
        anchor_id="wave_n25_audit_4_stand_downs_today_20260528",
        measurement_utc="2026-05-28T22:22:43Z",
        inputs={
            "n_in_flight_sisters": 5,
            "catalog_376_pv_invoked_at_main_thread": False,
            "session_duration_hours": 24,
        },
        predicted_output={"stand_down_count_lower_bound": 1, "stand_down_count_upper_bound": 4},
        empirical_output={"stand_down_count": 4, "stand_down_events": [
            "pr111_slot_4_resume_20260528",
            "z4_wave_n23_atick_redlich_20260528",
            "cascade_a_fec10_wave_n24_20260528",
            "paper_review_wave_n135_20260528",
        ]},
        residual=0.0,  # empirical 4 is within predicted band [1, 4]
        source_artifact=".omx/research/operator_critique_existing_work_audit_20260528T222243Z.md",
        measurement_method="wave_n25_stand_down_incident_count",
        provenance=_empirical_research_sidecar_provenance(
            ".omx/research/operator_critique_existing_work_audit_20260528T222243Z.md"
        ),
    )
    return CanonicalEquation(
        equation_id="main_thread_spawn_pv_gap_pre_catalog_376_extension_v1",
        name="Main-thread agent-spawn-decision PV gap predicts STAND_DOWN rate",
        one_line_summary=(
            "P(STAND_DOWN | N_in_flight_sisters) >= N_in_flight_sisters / S_total_substrates "
            "when main-thread does NOT invoke Catalog #376 verify_head_state_before_main_thread_spawn."
        ),
        latex_form=(
            r"P(\text{STAND\_DOWN}_{\text{new\_spawn}} | N_{\text{in\_flight\_sisters}}) "
            r"\geq \frac{N_{\text{in\_flight\_sisters}}}{S_{\text{total\_substrates}}} "
            r"\text{ when } \neg \text{Catalog\_378\_PV\_invoked\_at\_main\_thread}"
        ),
        python_callable_module_path=(
            "tac.discipline_anti_pattern_guards.main_thread_spawn_decision_pv_guard:"
            "verify_head_state_before_main_thread_spawn"
        ),
        domain_of_validity={
            "apparatus_state": "in_flight_sister_count_geq_2",
            "spawn_context": "main_thread_parent_agent_Agent_tool_call",
            "pv_check_status": "catalog_378_not_invoked_at_main_thread",
        },
        units_in={
            "n_in_flight_sisters": "int_count",
            "catalog_376_pv_invoked_at_main_thread": "bool",
            "session_duration_hours": "int_hours",
        },
        units_out={
            "stand_down_count_lower_bound": "int_count",
            "stand_down_count_upper_bound": "int_count",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"wave_n25_stand_down_incident_count": 0.0},
        last_calibration_utc="2026-05-28T22:22:43Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.preflight:check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state",
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        canonical_producers=(
            "tac.discipline_anti_pattern_guards.main_thread_spawn_decision_pv_guard:verify_head_state_before_main_thread_spawn",
        ),
        provenance=_design_provenance("main_thread_spawn_pv_gap_pre_catalog_376_extension.v1"),
    )


def build_all_initial_equations() -> list[CanonicalEquation]:
    """Return the 6 initial canonical equations as a list (no registry write).

    NOTE: ``main_thread_spawn_pv_gap_pre_catalog_376_extension_v1`` is
    registered separately via the Wave N+25 OP6 landing path (NOT in
    this 6-equation initial population to preserve historical schema
    stability). Use ``populate_main_thread_spawn_pv_gap_equation()`` to
    register it idempotently."""
    return [
        build_brotli_cascade_bounded_per_stream_v1(),
        build_mps_drift_architecture_class_dependent_v1(),
        build_per_byte_leverage_uniformly_distributed_v1(),
        build_per_pair_master_gradient_score_impact_taylor_v1(),
        build_master_gradient_locality_violation_by_codec_v1(),
        build_canonical_frontier_pointer_v1(),
    ]


def populate_main_thread_spawn_pv_gap_equation(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent population of the Wave N+25 OP6 canonical equation.

    Per CLAUDE.md "Canonical equations + models registry" non-negotiable:
    every empirical finding memo claiming a STAND_DOWN-rate prediction
    MUST be backed by a registered canonical equation. THIS helper
    registers the equation idempotently (APPEND-ONLY via
    ``tac.canonical_equations.register_canonical_equation``).

    The registration appends a new ``registered`` event each time it
    runs; the latest-row-wins query semantics in ``query_equations``
    ensure consumers see the most recent payload.
    """
    eq = build_main_thread_spawn_pv_gap_pre_catalog_376_extension_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="wave_n25_op6_main_thread_spawn_pv_gap_extension_20260528",
    )
    return eq


def populate_initial_equations(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> list[CanonicalEquation]:
    """Idempotent population of the 6 initial canonical equations.

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132):
    APPEND-ONLY — re-running this helper appends new ``registered`` events.
    The latest-row-wins query semantics in
    ``query_equations`` ensure consumers see the most recent payload.
    """
    out = []
    for eq in build_all_initial_equations():
        register_canonical_equation(
            eq,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="initial_population_2026_05_19",
        )
        out.append(eq)
    return out


__all__ = [
    "build_brotli_cascade_bounded_per_stream_v1",
    "build_mps_drift_architecture_class_dependent_v1",
    "build_per_byte_leverage_uniformly_distributed_v1",
    "build_per_pair_master_gradient_score_impact_taylor_v1",
    "build_master_gradient_locality_violation_by_codec_v1",
    "build_canonical_frontier_pointer_v1",
    "build_main_thread_spawn_pv_gap_pre_catalog_376_extension_v1",
    "build_all_initial_equations",
    "populate_initial_equations",
    "populate_main_thread_spawn_pv_gap_equation",
    "uniform_leverage_predictor",
]
