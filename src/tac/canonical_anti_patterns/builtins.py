# SPDX-License-Identifier: MIT
"""Initial population of canonical anti-patterns from CLAUDE.md FORBIDDEN_PATTERNS.

Per operator NON-NEGOTIABLE 2026-05-28 verbatim: *"learning anti-patterns is
upser important too for compounding continual learning, like the canonical
equations bu netgative and a higher layer of abstraction"*.

The 12 initial anti-patterns (per design memo §"Initial canonical population"):

Compounding-order:
 1. lzma_on_already_brotli_saturated_compounding_v1
 2. quantize_then_svd_corrupted_low_rank_v1
 3. fp4_packed_without_qat_cos_collapse_v1
 4. brotli_plus_lzma_chained_anti_pattern_v1

Diagnosis:
 5. cross_paradigm_test_without_per_axis_decomposition_v1
 6. predicted_band_from_random_init_tier_c_v1
 7. rank_1_problem_spec_synergy_tautology_v1

Provenance:
 8. phantom_score_directory_naming_lie_v1
 9. transient_tmp_path_in_persisted_artifact_v1

Data-source:
10. source_selector_inherited_predicted_score_mean_v1

Observability:
11. silent_no_spawn_modal_dispatch_v1

Rigor-loss:
12. docstring_overstatement_without_evidence_tag_v1

Each anti-pattern carries Provenance per Catalog #323 (PREDICTED grade
because anti-patterns are CLASS-level predictions of future recurrences,
not promotable score claims). Calling ``populate_initial_anti_patterns()``
is idempotent — the registry's APPEND-ONLY semantics mean re-running this
helper appends additional ``anti_pattern_registered`` events (the canonical
re-registration trail).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    PARADIGM_COMPOUNDING_ORDER,
    PARADIGM_DATA_SOURCE,
    PARADIGM_DIAGNOSIS,
    PARADIGM_DISCIPLINE,
    PARADIGM_OBSERVABILITY,
    PARADIGM_PROVENANCE,
    PARADIGM_QUANTIZATION,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_LOW,
    SEVERITY_OBSERVED_MEDIUM,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)
from tac.provenance.contract import Provenance


# Canonical placeholder SHA for design-only provenance.
_DESIGN_PROV_SHA = "0" * 64
_DESIGN_LANDING_UTC = "2026-05-28T00:00:00Z"


def _design_provenance(anti_pattern_id: str) -> Provenance:
    """Build a PREDICTED Provenance for design-only anti-pattern registration.

    Anti-patterns are CLASS-level predictions of future recurrences; they
    are never promotable score claims. The PREDICTED grade + non-promotable
    invariants are enforced at construction by ``build_provenance_for_predicted``.
    """
    return build_provenance_for_predicted(
        model_id=f"canonical_anti_patterns.builtins.{anti_pattern_id}",
        inputs_sha256=_DESIGN_PROV_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_DESIGN_LANDING_UTC,
    )


# -----------------------------------------------------------------------------
# Compounding-order anti-patterns (#1-#4)
# -----------------------------------------------------------------------------


def build_lzma_on_already_brotli_saturated_compounding_v1() -> AntiPattern:
    """Anti-pattern #1: LZMA over already-brotli'd bytes saturates ~1.001.

    Empirical anchor: CASCADE_SATURATION sister verdict commit d78401444 +
    decoder compression analysis commit 44d12e75d. Source: V3 RE-RUN
    commit b01232473.
    """
    return AntiPattern(
        anti_pattern_id="lzma_on_already_brotli_saturated_compounding_v1",
        description=(
            "Chaining LZMA over an already-brotli-compressed byte stream "
            "yields a saturated ratio (~1.001). The two entropy coders "
            "operate on overlapping redundancy domains so the second pass "
            "produces near-zero additional savings while adding decode-side "
            "complexity + format-coupling risk."
        ),
        forbidden_pattern_predicate=(
            "compression_pipeline.contains(brotli) AND "
            "compression_pipeline.contains(lzma) AND "
            "lzma.order > brotli.order"
        ),
        falsification_band={
            "lzma_after_brotli_ratio_lo": 0.999,
            "lzma_after_brotli_ratio_hi": 1.005,
        },
        recurrence_conditions=(
            "compression_ops list contains both brotli and lzma tokens",
            "stack proposes lzma_q9 after brotli_q11",
            "decoder pipeline chains lzma decode after brotli decode",
        ),
        canonical_source_anchor=(
            "commit:d78401444 CASCADE_SATURATION verdict + commit:44d12e75d "
            "decoder compression analysis + commit:b01232473 V3 RE-RUN; "
            "CLAUDE.md FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Choose ONE high-quality entropy coder (brotli q=11 standalone) "
            "OR replace LZMA with an ORTHOGONAL coder (ANS over distinct "
            "symbol space). Do NOT chain LZMA after brotli."
        ),
        canonical_producers=(
            "tools/audit_compression_pipeline_saturation.py",
            "experiments/build_*.py (any compounding-stack builder)",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/dykstra_pareto_solver/ (Wave N+2 Slot 1 integration)",
        ),
        paradigm_class=PARADIGM_COMPOUNDING_ORDER,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance("lzma_on_already_brotli_saturated_compounding_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_quantize_then_svd_corrupted_low_rank_v1() -> AntiPattern:
    """Anti-pattern #2: SVD applied to already-quantized tensor; rank corrupted."""
    return AntiPattern(
        anti_pattern_id="quantize_then_svd_corrupted_low_rank_v1",
        description=(
            "Applying SVD (low-rank decomposition) to a tensor that has "
            "ALREADY been quantized (int8 / FP4 / etc.) corrupts the "
            "rank-N residual because quantization noise dominates the "
            "low-rank energy structure. The canonical compound order is "
            "SVD FIRST (lossless rank reduction) -> quantization SECOND "
            "(per-factor int8 or FP4) -> entropy coding THIRD."
        ),
        forbidden_pattern_predicate=(
            "compound_stack.contains(svd) AND "
            "compound_stack.contains(quantize) AND "
            "svd.order > quantize.order"
        ),
        falsification_band={
            "post_quantize_svd_residual_rank_loss_lo": 0.30,
            "post_quantize_svd_residual_rank_loss_hi": 0.85,
        },
        recurrence_conditions=(
            "quantization_ops precede svd / low_rank in compound stack",
            "stack proposes int8 quantize then truncated SVD",
            "compound stack lists fp4 then low_rank decomposition",
        ),
        canonical_source_anchor=(
            "Wave N+1 compound stacking analysis; CLAUDE.md FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Canonical order: SVD/low-rank FIRST (lossless rank reduction "
            "preserves residual structure) -> quantization (int8 OR FP4 "
            "per-factor) SECOND -> entropy coding THIRD."
        ),
        canonical_producers=(
            "experiments/build_compound_quantize_svd_*.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/dykstra_pareto_solver/ (Wave N+2 Slot 1 integration)",
        ),
        paradigm_class=PARADIGM_COMPOUNDING_ORDER,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance("quantize_then_svd_corrupted_low_rank_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_fp4_packed_without_qat_cos_collapse_v1() -> AntiPattern:
    """Anti-pattern #3: FP4 packed nibbles deployment without QAT; cos<0.999."""
    return AntiPattern(
        anti_pattern_id="fp4_packed_without_qat_cos_collapse_v1",
        description=(
            "Deploying FP4 packed nibbles (E2M1 unsigned codebook) WITHOUT "
            "first running Quantization-Aware Training (QAT) results in cos "
            "similarity dropping below the canonical 0.999 threshold against "
            "the FP32 reference. The substrate's distillation_gap explodes "
            "and the downstream scorer signal degrades catastrophically."
        ),
        forbidden_pattern_predicate=(
            "quantization_ops.contains(fp4_packed) AND NOT "
            "training_pipeline.includes_qat_finetune_pass"
        ),
        falsification_band={
            "post_fp4_packed_cos_similarity_lo": 0.90,
            "post_fp4_packed_cos_similarity_hi": 0.998,
        },
        recurrence_conditions=(
            "substrate trainer emits fp4_packed archive without prior QAT pass",
            "compound stack lists fp4 quantization without LSQ/QAT step",
            "deployment recipe declares fp4_packed without training.qat_epochs > 0",
        ),
        canonical_source_anchor=(
            "decoder compression analysis Scenario A; CLAUDE.md 'QAT pipeline "
            "— non-negotiable for FP4 deployment' section"
        ),
        canonical_unwind_path=(
            "Follow the 5-step QAT pipeline per CLAUDE.md: (1) train float "
            "first with eval_roundtrip + noise + EMA + hinge loss; (2) freeze "
            "BN stats; (3) insert per-channel FP4 fake-quant on weights + "
            "per-tensor on activations; (4) fine-tune 20% of original epochs "
            "at 0.1x LR (LSQ lr=0.01x base); (5) export 4 bits/param."
        ),
        canonical_producers=(
            "experiments/train_substrate_*_fp4_packed.py",
            "experiments/qat_finetune.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_QUANTIZATION,
        severity=SEVERITY_CRITICAL,
        provenance=_design_provenance("fp4_packed_without_qat_cos_collapse_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_brotli_plus_lzma_chained_anti_pattern_v1() -> AntiPattern:
    """Anti-pattern #4: brotli + LZMA chained (sister of #1; broader scope)."""
    return AntiPattern(
        anti_pattern_id="brotli_plus_lzma_chained_anti_pattern_v1",
        description=(
            "Compounding entropy coders that operate on similar redundancy "
            "domains saturate at the joint entropy floor. brotli and LZMA "
            "both implement LZ77+arithmetic; chaining them adds decode "
            "complexity + format-coupling risk for near-zero additional "
            "compression. The broader sister of anti-pattern #1 (which "
            "fires specifically when LZMA is AFTER brotli)."
        ),
        forbidden_pattern_predicate=(
            "compression_pipeline.contains(brotli) AND "
            "compression_pipeline.contains(lzma)"
        ),
        falsification_band={
            "brotli_lzma_joint_ratio_lo": 0.998,
            "brotli_lzma_joint_ratio_hi": 1.010,
        },
        recurrence_conditions=(
            "compression_ops list contains both brotli AND lzma tokens",
            "decoder pipeline includes both brotli and lzma decompressors",
            "compound stack proposes brotli + lzma in any order",
        ),
        canonical_source_anchor=(
            "Sister of commit:d78401444 CASCADE_SATURATION; CLAUDE.md "
            "FORBIDDEN_PATTERNS; design memo §Initial canonical population"
        ),
        canonical_unwind_path=(
            "Choose ORTHOGONAL entropy coders (e.g. ANS over distinct "
            "symbol space + brotli over text-redundancy domain) OR single "
            "high-quality (brotli q=11) standalone."
        ),
        canonical_producers=(
            "experiments/build_*.py (any builder declaring brotli + lzma)",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/dykstra_pareto_solver/ (Wave N+2 Slot 1 integration)",
        ),
        paradigm_class=PARADIGM_COMPOUNDING_ORDER,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance("brotli_plus_lzma_chained_anti_pattern_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Diagnosis anti-patterns (#5-#7)
# -----------------------------------------------------------------------------


def build_cross_paradigm_test_without_per_axis_decomposition_v1() -> AntiPattern:
    """Anti-pattern #5: cross-paradigm test without per-axis decomposition.

    Empirical anchor: V3 RE-RUN commit b01232473. Seg-axis dominance gets
    attributed to substrate when it is shared-decoder artifact (PR97 anti-
    pattern -0.042 receipt).
    """
    return AntiPattern(
        anti_pattern_id="cross_paradigm_test_without_per_axis_decomposition_v1",
        description=(
            "Dispatching a cross-paradigm substrate WITHOUT enabling the "
            "Catalog #356 per-axis decomposition causes seg-axis dominance "
            "to be misattributed to the substrate when it is actually a "
            "SHARED-DECODER artifact. PR97 anti-pattern empirical receipt: "
            "0.042 score lost by trading pose for seg unwittingly because "
            "scalar score signal hid axis attribution."
        ),
        forbidden_pattern_predicate=(
            "candidate.cross_paradigm=true AND "
            "candidate.per_axis_decomposition_active=false"
        ),
        falsification_band={
            "axis_attribution_error_lo": 0.020,
            "axis_attribution_error_hi": 0.080,
        },
        recurrence_conditions=(
            "cross-paradigm substrate dispatched without per-axis decomposition",
            "candidate stack_spec has per_axis_decomposition_active=false",
            "ranker consumes scalar predicted_delta without AxisDecomposition",
            "Catalog #356 GAP FIX not active for this candidate",
        ),
        canonical_source_anchor=(
            "commit:b01232473 V3 RE-RUN empirical anchor; CLAUDE.md 'SegNet "
            "vs PoseNet importance — operating-point dependent' section; "
            "Catalog #356 per-axis decomposition canonical contract"
        ),
        canonical_unwind_path=(
            "Enable Catalog #356 per-axis decomposition (post-commit "
            "92a39dc62) BEFORE cross-paradigm test. The AxisDecomposition "
            "contract exposes (seg, pose, archive_bytes) deltas so axis "
            "attribution is structural rather than tribal-knowledge."
        ),
        canonical_producers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral/consumer_contract.py (AxisDecomposition)",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/dykstra_pareto_solver/ (Wave N+2 Slot 1 integration)",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "cross_paradigm_test_without_per_axis_decomposition_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_predicted_band_from_random_init_tier_c_v1() -> AntiPattern:
    """Anti-pattern #6: predicted band derived from RANDOM_INIT (pre-training).

    Empirical anchor: C6 IBPS 22× miss; Catalog #324 self-protection.
    """
    return AntiPattern(
        anti_pattern_id="predicted_band_from_random_init_tier_c_v1",
        description=(
            "Predicted band derived from Tier-C density measurement on "
            "RANDOM_INIT (pre-training) archive misses post-training reality "
            "by up to 22× (C6 IBPS empirical anchor). The random-init "
            "spectrum does not predict the trained spectrum because the "
            "post-training spectrum is the substrate's actual learned signal "
            "distribution; pre-training Tier-C measures a different "
            "distribution entirely."
        ),
        forbidden_pattern_predicate=(
            "recipe.predicted_band_source IN {random_init, pre_training} AND "
            "recipe.predicted_band_validation_status NOT IN "
            "{validated_post_training, pending_post_training}"
        ),
        falsification_band={
            "post_training_band_miss_ratio_lo": 5.0,
            "post_training_band_miss_ratio_hi": 25.0,
        },
        recurrence_conditions=(
            "recipe declares predicted_band derived from RANDOM_INIT archive",
            "recipe.predicted_band_source token mentions pre_training",
            "Tier-C density measured before training pass landed",
            "recipe predicted_band_validation_status missing or pending_post_training without reactivation criteria",
        ),
        canonical_source_anchor=(
            "C6 IBPS empirical anchor (22× miss); Catalog #324 STRICT "
            "preflight gate; CLAUDE.md 'PREDICTED BAND FROM RANDOM-INIT' "
            "forbidden pattern"
        ),
        canonical_unwind_path=(
            "Re-measure Tier-C density POST-TRAINING on the actual learned "
            "archive bytes OR declare predicted_band_validation_status="
            "pending_post_training with explicit reactivation criteria per "
            "Catalog #324."
        ),
        canonical_producers=(
            ".omx/operator_authorize_recipes/*.yaml",
        ),
        canonical_consumers=(
            "tools/operator_authorize.py",
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_CRITICAL,
        provenance=_design_provenance("predicted_band_from_random_init_tier_c_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_rank_1_problem_spec_synergy_tautology_v1() -> AntiPattern:
    """Anti-pattern #7: rank-1 problem-spec synergy tautology.

    Empirical anchor: paradox half 2 rigor review commit 21014faa7.
    """
    return AntiPattern(
        anti_pattern_id="rank_1_problem_spec_synergy_tautology_v1",
        description=(
            "Multi-op synergy measurement built on a RANK-1 operator-gradient "
            "matrix returns synergy ≈ 0 for ANY input. This is an arithmetic "
            "tautology, NOT an empirical property. Per the paradox half 2 "
            "rigor review (commit 21014faa7): _build_multiop_problem_spec "
            "makes all operator gradients (seg,pose,0)×leverage_i so cosine "
            "distance is 0.0 for any pair. The sweep was structurally "
            "incapable of measuring synergy."
        ),
        forbidden_pattern_predicate=(
            "problem_spec.operator_gradient_matrix.rank == 1 AND "
            "downstream_metric == 'synergy'"
        ),
        falsification_band={
            "rank_1_synergy_measured_lo": -0.001,
            "rank_1_synergy_measured_hi": 0.001,
        },
        recurrence_conditions=(
            "problem_spec builds per-operator gradients as (axis_seg, axis_pose, 0)*leverage",
            "synergy measurement reports identically ~0 across all operator pairs",
            "operator gradient matrix has rank 1 (single dominant direction)",
            "_build_multiop_problem_spec construction uses uniform per-axis projection",
        ),
        canonical_source_anchor=(
            "commit:21014faa7 paradox half 2 rigor review; CLAUDE.md "
            "FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Give operators distinct per-axis gradients from per-pair cell "
            "footprints via Catalog #356 AxisDecomposition (each operator "
            "carries its OWN seg/pose/byte per-pair gradient signature). "
            "Assert operator_gradient_matrix rank > 1 via regression test "
            "BEFORE running synergy sweep."
        ),
        canonical_producers=(
            "tools/build_multi_op_synergy_problem_spec.py",
            "src/tac/sensitivity_map/*",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance("rank_1_problem_spec_synergy_tautology_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Provenance anti-patterns (#8-#9)
# -----------------------------------------------------------------------------


def build_phantom_score_directory_naming_lie_v1() -> AntiPattern:
    """Anti-pattern #8: filename claims a device/contract content does not match.

    Empirical anchor: Z3 v2 FULL Modal A100; Catalog #249 self-protection.
    """
    return AntiPattern(
        anti_pattern_id="phantom_score_directory_naming_lie_v1",
        description=(
            "Output file/directory name claims a device/scope/contract that "
            "the contents do NOT match (e.g. contest_auth_eval_cuda.json "
            "containing CPU eval results). Operators + downstream consumers "
            "trust the filename; phantom-score directories silently leak "
            "non-promotable signals into promotion paths. Z3 v2 FULL Modal "
            "A100 commit empirical anchor."
        ),
        forbidden_pattern_predicate=(
            "artifact.filename.contains_device_token(cuda|cpu|mps) AND "
            "artifact.metadata.device != artifact.filename.device_token"
        ),
        falsification_band={
            "device_mismatch_rate_lo": 0.05,
            "device_mismatch_rate_hi": 0.50,
        },
        recurrence_conditions=(
            "build_manifest filename references _cuda but metadata says cpu",
            "submission directory named _gpu but contains MPS-PROXY results",
            "contest_auth_eval_cuda.json contains evidence_grade=mps_proxy",
            "phantom-score artifact in build manifest with mismatched device",
        ),
        canonical_source_anchor=(
            "Z3 v2 FULL Modal A100 incident; Catalog #249 self-protection; "
            "CLAUDE.md FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Filename MUST match the metadata that generated it OR be "
            "device-agnostic. Use canonical naming convention: <substrate>_"
            "<axis-grade>_<utc>.json where axis-grade matches the artifact's "
            "evidence_grade exactly."
        ),
        canonical_producers=(
            "experiments/contest_auth_eval.py",
            "experiments/build_*.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #249 STRICT gate)",
        ),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_CRITICAL,
        provenance=_design_provenance("phantom_score_directory_naming_lie_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_transient_tmp_path_in_persisted_artifact_v1() -> AntiPattern:
    """Anti-pattern #9: /tmp/<path> cited as durable evidence.

    Empirical anchor: lane_pr106_stacked; Catalog #220 sister.
    """
    return AntiPattern(
        anti_pattern_id="transient_tmp_path_in_persisted_artifact_v1",
        description=(
            "/tmp/<path> cited as durable evidence in lane registry, dispatch "
            "claim, commit message, or build manifest. /tmp is wiped on "
            "reboot + per-CI-runner; persisted artifacts that reference /tmp "
            "paths cannot be reproduced. lane_pr106_stacked + Catalog #220 + "
            "multiple sister incidents."
        ),
        forbidden_pattern_predicate=(
            "persisted_artifact.body.contains_path_starting_with('/tmp/') AND "
            "artifact_category IN {lane_registry, dispatch_claim, build_manifest, "
            "commit_message}"
        ),
        falsification_band={
            "tmp_reference_in_persisted_artifact_lo": 0.0,
            "tmp_reference_in_persisted_artifact_hi": 0.10,
        },
        recurrence_conditions=(
            "lane registry evidence path starts with /tmp/",
            "dispatch claim body references /tmp/<artifact>",
            "build manifest archive_artifact_path under /tmp/",
            "commit message body cites /tmp path as evidence",
        ),
        canonical_source_anchor=(
            "lane_pr106_stacked empirical anchor; Catalog #220 sister; "
            "CLAUDE.md FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Use canonical durable paths: experiments/results/<lane_id>_<utc>/ "
            "for build artifacts; .omx/state/ for ledgers; .omx/research/ for "
            "analyses; .omx/tmp/ ONLY for explicitly ephemeral scratch (and "
            "never cite that in a persisted artifact)."
        ),
        canonical_producers=(
            "experiments/build_*.py",
            "tools/*.py (any tool writing persisted artifacts)",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #220 sister gate)",
        ),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance("transient_tmp_path_in_persisted_artifact_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Data-source anti-patterns (#10)
# -----------------------------------------------------------------------------


def build_source_selector_inherited_predicted_score_mean_v1() -> AntiPattern:
    """Anti-pattern #10: empirical interaction matrix from inherited predicted_score_mean.

    Empirical anchor: DQS1 drop-many BUILD-1 verdict 2026-05-25.
    """
    return AntiPattern(
        anti_pattern_id="source_selector_inherited_predicted_score_mean_v1",
        description=(
            "Empirical interaction matrix populated from a predicted_score_mean "
            "field that inherits from the source SELECTOR rather than being "
            "measured per-pair empirically. The interaction matrix then "
            "concentrates 100% of computed values at a single arithmetic "
            "artifact and the autopilot routes against a phantom signal. "
            "DQS1 drop-many BUILD-1 verdict 2026-05-25 empirical anchor."
        ),
        forbidden_pattern_predicate=(
            "interaction_matrix.populated_from='predicted_score_mean' AND "
            "predicted_score_mean.derivation_path.includes(source_selector)"
        ),
        falsification_band={
            "matrix_value_concentration_at_artifact_lo": 0.95,
            "matrix_value_concentration_at_artifact_hi": 1.00,
        },
        recurrence_conditions=(
            "interaction matrix derived from source-selector inherited values",
            "predicted_score_mean field traced back to selector heuristic",
            "100% of matrix values concentrate at single arithmetic artifact",
            "no paired CPU exact-eval ledger backing the interaction values",
        ),
        canonical_source_anchor=(
            "DQS1 drop-many BUILD-1 verdict 2026-05-25; CLAUDE.md "
            "FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Populate interaction matrix from paired CPU exact-eval ledger "
            "via Modal CPU dispatch (~$1.20-2 per Catalog #246) OR use "
            "drop-many greedy heuristic without interaction matrix dependency."
        ),
        canonical_producers=(
            "tools/build_*_interaction_matrix.py",
            "tools/dqs1_drop_many_*.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DATA_SOURCE,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "source_selector_inherited_predicted_score_mean_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Observability anti-patterns (#11)
# -----------------------------------------------------------------------------


def build_silent_no_spawn_modal_dispatch_v1() -> AntiPattern:
    """Anti-pattern #11: Modal dispatch sys.exit BEFORE fn.spawn(); orphan paid GPU.

    Empirical anchor: STC v2 5th consecutive silent-no-spawn; Catalog #360.
    """
    return AntiPattern(
        anti_pattern_id="silent_no_spawn_modal_dispatch_v1",
        description=(
            "Modal dispatch sys.exit(...) FATAL path BEFORE fn.spawn() queues "
            "a task; no canonical ledger row + no recovery dump; harvester-"
            "invisible per CLAUDE.md 'Modal .spawn() HARVEST OR LOSE' non-"
            "negotiable. STC v2 dispatched 5 consecutive times silently "
            "before Catalog #360 self-protection landed."
        ),
        forbidden_pattern_predicate=(
            "experiments/modal_*.py::main() contains sys.exit(...) BEFORE "
            "fn.spawn() AND does NOT call register_pre_spawn_fatal()"
        ),
        falsification_band={
            "orphaned_paid_dispatch_rate_lo": 0.05,
            "orphaned_paid_dispatch_rate_hi": 0.60,
        },
        recurrence_conditions=(
            "Modal dispatch sys.exit fires before fn.spawn() reaches network",
            "no canonical ledger row for the would-be dispatch",
            "harvester scan shows no record for paid GPU spend",
            "STC v2-style silent-no-spawn dispatch pattern",
            "modal_dispatch_pre_spawn_path=true in stack_spec",
        ),
        canonical_source_anchor=(
            "STC v2 5th consecutive silent-no-spawn anchor; Catalog #360 "
            "STRICT preflight gate; CLAUDE.md 'Modal .spawn() HARVEST OR LOSE'"
        ),
        canonical_unwind_path=(
            "Route through tac.deploy.modal.call_id_ledger."
            "register_pre_spawn_fatal(...) per Catalog #360 helper which "
            "writes pre_spawn_fatal event to canonical ledger BEFORE sys.exit. "
            "Every FATAL path in main() becomes observable + recoverable."
        ),
        canonical_producers=(
            "experiments/modal_*.py",
            "tools/operator_authorize.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #360 STRICT gate)",
        ),
        paradigm_class=PARADIGM_OBSERVABILITY,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance("silent_no_spawn_modal_dispatch_v1"),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Rigor-loss anti-patterns (#12)
# -----------------------------------------------------------------------------


def build_docstring_overstatement_without_evidence_tag_v1() -> AntiPattern:
    """Anti-pattern #12: empirical claim without [empirical:] / [contest-CUDA] / [predicted] tag.

    Empirical anchor: Lane PD 49% docstring vs 18.5% empirical.
    """
    return AntiPattern(
        anti_pattern_id="docstring_overstatement_without_evidence_tag_v1",
        description=(
            "Empirical claim ('saves N%' / 'improves N%' / 'beats baseline' / "
            "'verified') in docstring/report/script without adjacent "
            "[empirical:<artifact>] / [contest-CUDA] / [contest-CPU] / "
            "[predicted] / [advisory only] axis tag. Lane PD docstring stated "
            "49% savings; empirical regression test caught 18.5%. Without the "
            "axis tag, future agents cannot distinguish predicted-from-model "
            "from empirically-measured claims."
        ),
        forbidden_pattern_predicate=(
            "docstring.contains_quantitative_claim AND NOT "
            "docstring.has_adjacent_axis_tag"
        ),
        falsification_band={
            "untagged_claim_overstatement_ratio_lo": 1.5,
            "untagged_claim_overstatement_ratio_hi": 4.0,
        },
        recurrence_conditions=(
            "docstring contains 'saves N%' without axis tag",
            "report claims 'improves N%' without [empirical:<artifact>]",
            "script summary states 'beats baseline' without [contest-CUDA]",
            "memo claims numeric improvement without canonical evidence grade",
        ),
        canonical_source_anchor=(
            "Lane PD docstring 49% vs empirical 18.5%; Catalog #287 STRICT "
            "preflight gate; CLAUDE.md FORBIDDEN_PATTERNS"
        ),
        canonical_unwind_path=(
            "Tag every claim with canonical axis token per Catalog #287/#323: "
            "[empirical:<artifact-path>] for measured / [contest-CUDA] for "
            "contest-CUDA archive / [contest-CPU] for contest-CPU archive / "
            "[predicted] for model prediction / [advisory only] for non-"
            "promotable advisory."
        ),
        canonical_producers=(
            "experiments/*.py",
            "tools/*.py",
            "reports/*.md",
            ".omx/research/*.md",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/preflight.py (Catalog #287 STRICT gate)",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "docstring_overstatement_without_evidence_tag_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Discipline anti-patterns (#13-#14) — per Slot 2 Wave N+7 audit 2026-05-28
# -----------------------------------------------------------------------------


def build_subagent_spawn_without_head_state_premise_verification_v1() -> AntiPattern:
    """Anti-pattern #13: spawning Agent without HEAD-state PV check.

    Empirical anchors (2 receipts):
      * Wave N+5 Slot 1 Compound C STAND_DOWN (predecessor commit ``e61ea93b0``):
        the parent agent dispatched the subagent without first reading
        ``git log --oneline -30`` + ``git status``; predecessor had already
        landed; subagent re-discovered the landed work + stood down with
        landing memo trail.
      * Wave N+5 Slot 2 framework_agnostic STAND_DOWN this session: same
        class — Agent dispatched into a working tree with sister-subagent's
        uncommitted edits visible at spawn time; sister landed cleanly via
        commit ``5d38bf9df`` later, but the spawn-time visibility was a
        signal to either coordinate or stand down BEFORE spawn.

    Canonical unwind path per Catalog #229 (premise-verification-before-edit):
    ALWAYS run ``git log --oneline -30`` + ``git status`` + check sister
    landing memos BEFORE invoking Agent spawn so the parent agent decides
    against an empirically-verified HEAD state instead of a stale prior.
    """
    return AntiPattern(
        anti_pattern_id="subagent_spawn_without_head_state_premise_verification_v1",
        description=(
            "Parent agent invoked Agent spawn without first running "
            "git log + git status + sister-landing-memo check. The subagent "
            "subsequently discovers the work is already landed and STANDS_DOWN "
            "(or worse: collides with a sister subagent's in-flight edits). "
            "Wasted compute + audit-trail noise; canonical Catalog #229 "
            "premise-verification-before-edit was bypassed at the spawn "
            "decision boundary."
        ),
        forbidden_pattern_predicate=(
            "Agent.spawn(...) called WITHOUT preceding git log --oneline -30 "
            "AND git status AND sister-landing-memo PV per Catalog #229"
        ),
        falsification_band={
            # Each STAND_DOWN is +1 wasted-subagent-spawn count; the band
            # tracks observed recurrences-per-session.
            "stand_down_recurrences_per_session_lo": 1.0,
            "stand_down_recurrences_per_session_hi": 4.0,
        },
        recurrence_conditions=(
            "parent agent spawns Agent without git log review in same turn",
            "subagent reports STAND_DOWN due to already-landed predecessor",
            "subagent encounters sister-subagent uncommitted edits in shared tree",
            "Catalog #340 sister-checkpoint guard fires on the subagent's commit attempt",
        ),
        canonical_source_anchor=(
            "Wave N+5 Slot 1 Compound C STAND_DOWN (predecessor commit "
            "e61ea93b0) + Wave N+5 Slot 2 framework_agnostic STAND_DOWN; "
            "Catalog #229 premise-verification-before-edit; Catalog #340 "
            "sister-checkpoint guard"
        ),
        canonical_unwind_path=(
            "Before Agent spawn: (1) run `git log --oneline -30` to verify "
            "HEAD state; (2) run `git status` to detect sister-subagent "
            "in-flight edits; (3) read sister landing memos under "
            ".omx/research/*landed_<YYYYMMDD>.md to detect already-landed "
            "work; (4) only THEN decide spawn vs STAND_DOWN. Per Catalog #229: "
            "premise verification IS the discipline."
        ),
        canonical_producers=(
            "parent_agent_dispatch_layer",
            "Agent_tool_invocation_callsites",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "tools/subagent_checkpoint.py (predecessor-resume helper)",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "subagent_spawn_without_head_state_premise_verification_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_predecessor_working_tree_uncommitted_handoff_v1() -> AntiPattern:
    """Anti-pattern #14: predecessor subagent leaves uncommitted edits.

    Empirical anchor: Wave N+5 Slot 2 framework_agnostic STAND_DOWN this
    session — predecessor's working-tree edits to multiple sister files were
    uncommitted at spawn boundary; successor inherited a dirty working tree
    that complicated PV. Resolution: predecessor's canonical-serializer
    auto-commit at ``5d38bf9df`` landed the framework_agnostic primitives
    cleanly with proper attribution, restoring the working tree to a
    well-defined HEAD state.

    Canonical unwind path: every subagent commits its incremental edits via
    the canonical serializer at session-end checkpoint OR posts
    SUPERSESSION-PENDING in its STAND_DOWN memo so the successor knows
    the handoff is uncommitted. The post-spawn HEAD verification step in
    the canonical Catalog #229 PV flow surfaces the gap.
    """
    return AntiPattern(
        anti_pattern_id="predecessor_working_tree_uncommitted_handoff_v1",
        description=(
            "Predecessor subagent's STAND_DOWN or completion left files "
            "uncommitted in the shared working tree; successor inherits a "
            "dirty state that is hard to distinguish from sister-subagent "
            "in-flight collision (Catalog #314 absorption-pattern). The "
            "canonical fix is post-spawn HEAD verification + canonical-"
            "serializer auto-commit by the predecessor."
        ),
        forbidden_pattern_predicate=(
            "subagent SUBAGENT_TERMINATE without canonical-serializer "
            "commit AND working-tree has predecessor-owned edits"
        ),
        falsification_band={
            # Implementation-inefficiency cost is small (forensic re-investigation
            # at handoff) but recurrent.
            "wasted_pv_minutes_per_recurrence_lo": 2.0,
            "wasted_pv_minutes_per_recurrence_hi": 10.0,
        },
        recurrence_conditions=(
            "git status shows uncommitted modifications at subagent spawn time",
            "predecessor STAND_DOWN memo does not declare SUPERSESSION-PENDING",
            "Catalog #117 last-50-commit serializer-log shows no predecessor commit",
            "successor's PV cannot disambiguate uncommitted-handoff vs sister-collision",
        ),
        canonical_source_anchor=(
            "Wave N+5 Slot 2 framework_agnostic STAND_DOWN; resolved at "
            "predecessor canonical-serializer auto-commit 5d38bf9df; "
            "Catalog #117 + Catalog #229 + Catalog #314"
        ),
        canonical_unwind_path=(
            "Every subagent at SUBAGENT_TERMINATE / STAND_DOWN MUST commit "
            "its incremental edits via `tools/subagent_commit_serializer.py "
            "--message <subject> --files <files> "
            "--expected-content-sha256 <file>=<post-edit-sha>` OR explicitly "
            "declare SUPERSESSION-PENDING in its STAND_DOWN memo so successor "
            "knows the handoff is uncommitted. Catalog #117 forbids bare "
            "git commit; #229 forces PV; #314 detects absorption post-hoc."
        ),
        canonical_producers=(
            "any subagent at SUBAGENT_TERMINATE / STAND_DOWN boundary",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "tools/subagent_commit_serializer.py",
            "src/tac/preflight.py (Catalog #314 absorption-pattern gate)",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_LOW,
        provenance=_design_provenance(
            "predecessor_working_tree_uncommitted_handoff_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Diagnosis anti-patterns (#15) — Wyner-Ziv prefix-Y FALSIFICATION 2026-05-28
# -----------------------------------------------------------------------------


def build_wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1() -> AntiPattern:
    """Anti-pattern #15: Wyner-Ziv prefix-detector density on decoder-state-dict.

    Empirical anchor: ``.omx/research/wyner_ziv_pipeline_stage_codec_l1_long_
    mlx_600pair_landed_20260528.md`` commit ``6f5eabf30`` — L1 LONG MLX
    measurement on real PR101 fp16 decoder state_dict bytes
    (sha256=``79b804d9a5839eb3`` / 457916 B) yielded maximum
    Y-derivable-prefix density **0.000218%** across all 4 canonical Y sources
    (Comma2k19 / ImageNet / torch_defaults / math_constants) — **4 orders
    of magnitude below 1% threshold** per op-routable #4. Per Catalog #307:
    IMPLEMENTATION-LEVEL falsification (PARADIGM Wyner 1976 R(D|Y) INTACT).

    Canonical unwind path per Catalog #311 Atick-Tishby-Wyner triple:
    per-pair PoseNet-output Y derivation. Y is no longer a fixed canonical
    source but a per-pair pose tensor shipped IN the archive (decoder
    reproduces it deterministically via the pose-axis side-info; pre-
    computed per-pair to avoid Catalog #6 strict-scorer-rule at inflate
    time).
    """
    return AntiPattern(
        anti_pattern_id="wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1",
        description=(
            "Wyner-Ziv prefix-detector applied to a decoder-state-dict byte "
            "form yields ~0% Y-derivable-prefix density vs canonical Y "
            "sources. The byte distribution of fp16/fp32 tensor weights is "
            "entropy-flat — there is no common prefix to extract via a "
            "byte-prefix detector. The IMPLEMENTATION (prefix-detector at "
            "the state_dict surface) is empirically falsified; the PARADIGM "
            "(Wyner 1976 R(D|Y) with decoder-side side-info per Catalog #311 "
            "Atick-Tishby-Wyner triple) is INTACT and the canonical unwind "
            "path is per-pair PoseNet-output Y derivation."
        ),
        forbidden_pattern_predicate=(
            "wyner_ziv_layer.intercept_location == STATE_DICT_SERIALIZATION "
            "AND wyner_ziv_layer.side_info_source IN canonical_4_sources "
            "AND base_substrate_bytes_form IN {raw_fp16, raw_fp32, torch_save}"
        ),
        falsification_band={
            # Empirical: 0.000218% max density across 4 canonical Y sources
            # on PR101 fp16 state_dict (457916 B). Falsification band tracks
            # the observed density-percent on this byte form.
            "max_density_percent_lo": 0.0,
            "max_density_percent_hi": 0.01,
            "threshold_percent_per_op_routable_4": 1.0,
        },
        recurrence_conditions=(
            "wyner_ziv_pipeline_stage_codec measuring density on raw_fp16 state_dict",
            "wyner_ziv_pipeline_stage_codec measuring density on raw_fp32 state_dict",
            "wyner_ziv_pipeline_stage_codec measuring density on torch_save state_dict",
            "prefix-detector applied to any entropy-flat tensor-weight byte stream",
            "all 4 canonical Y sources yield density << 1% threshold",
        ),
        canonical_source_anchor=(
            "commit:6f5eabf30 Wyner-Ziv L1 LONG MLX 600-PAIR landing; sister "
            ".omx/research/wyner_ziv_pipeline_stage_codec_l1_long_mlx_"
            "600pair_landed_20260528.md; Catalog #311 Atick-Tishby-Wyner "
            "triple; Catalog #307 paradigm-vs-implementation classification"
        ),
        canonical_unwind_path=(
            "Per-pair PoseNet-output Y derivation per Catalog #311 "
            "Atick-Tishby-Wyner triple (Op-routable #5 in sister landing "
            "memo). Y is a per-pair pose tensor (6-dim per Wyner 1976 "
            "R(D|Y)) shipped IN the archive as decoder-side side-info; "
            "decoder reproduces Y deterministically at inflate time WITHOUT "
            "loading scorer per Catalog #6 strict-scorer-rule. Pre-compute "
            "per-pair pose at compress time via PoseNet on M5 Max OR via "
            "an Atick-Redlich ego-motion-conditioned deterministic stand-in. "
            "Sister reactivation paths: non-prefix Y derivation (substring "
            "overlap); cross-substrate composition Y (FEC6 archive bytes "
            "as Y for PR101). Per CLAUDE.md 'Forbidden premature KILL': "
            "DEFERRED-PENDING-research, NOT killed."
        ),
        canonical_producers=(
            "experiments/train_substrate_wyner_ziv_pipeline_stage_codec.py",
            "src/tac/substrates/wyner_ziv_pipeline_stage_codec/trainer.py",
            "src/tac/codec/wyner_ziv_layer.py",
        ),
        canonical_consumers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
            "src/tac/cathedral_consumers/wyner_ziv_pipeline_stage_codec/",
            "src/tac/canonical_equations/registry.py (canonical equation #344 anchor)",
        ),
        paradigm_class=PARADIGM_DIAGNOSIS,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_DESIGN_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# -----------------------------------------------------------------------------
# Population helpers
# -----------------------------------------------------------------------------


def build_all_initial_anti_patterns() -> list[AntiPattern]:
    """Return the 15 initial canonical anti-patterns as a list (no registry write).

    Wave N+7 Slot 2 2026-05-28 expansion: added #13/#14 discipline anti-patterns
    and #15 Wyner-Ziv prefix-Y diagnosis anti-pattern per operator directive
    'ensure all negative findings audited' + Wyner-Ziv FALSIFICATION
    reactivation criteria.
    """
    return [
        build_lzma_on_already_brotli_saturated_compounding_v1(),
        build_quantize_then_svd_corrupted_low_rank_v1(),
        build_fp4_packed_without_qat_cos_collapse_v1(),
        build_brotli_plus_lzma_chained_anti_pattern_v1(),
        build_cross_paradigm_test_without_per_axis_decomposition_v1(),
        build_predicted_band_from_random_init_tier_c_v1(),
        build_rank_1_problem_spec_synergy_tautology_v1(),
        build_phantom_score_directory_naming_lie_v1(),
        build_transient_tmp_path_in_persisted_artifact_v1(),
        build_source_selector_inherited_predicted_score_mean_v1(),
        build_silent_no_spawn_modal_dispatch_v1(),
        build_docstring_overstatement_without_evidence_tag_v1(),
        build_subagent_spawn_without_head_state_premise_verification_v1(),
        build_predecessor_working_tree_uncommitted_handoff_v1(),
        build_wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1(),
    ]


def populate_initial_anti_patterns(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> list[AntiPattern]:
    """Idempotent population of the 12 initial canonical anti-patterns.

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132):
    APPEND-ONLY — re-running this helper appends new
    ``anti_pattern_registered`` events. The latest-row-wins query
    semantics in ``query_anti_patterns`` ensure consumers see the most
    recent payload.
    """
    out: list[AntiPattern] = []
    for ap in build_all_initial_anti_patterns():
        register_anti_pattern(
            ap,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
        )
        out.append(ap)
    return out


__all__ = [
    "build_all_initial_anti_patterns",
    "build_brotli_plus_lzma_chained_anti_pattern_v1",
    "build_cross_paradigm_test_without_per_axis_decomposition_v1",
    "build_docstring_overstatement_without_evidence_tag_v1",
    "build_fp4_packed_without_qat_cos_collapse_v1",
    "build_lzma_on_already_brotli_saturated_compounding_v1",
    "build_phantom_score_directory_naming_lie_v1",
    "build_predecessor_working_tree_uncommitted_handoff_v1",
    "build_predicted_band_from_random_init_tier_c_v1",
    "build_quantize_then_svd_corrupted_low_rank_v1",
    "build_rank_1_problem_spec_synergy_tautology_v1",
    "build_silent_no_spawn_modal_dispatch_v1",
    "build_source_selector_inherited_predicted_score_mean_v1",
    "build_subagent_spawn_without_head_state_premise_verification_v1",
    "build_transient_tmp_path_in_persisted_artifact_v1",
    "build_wyner_ziv_prefix_y_density_decoder_state_dict_surface_v1",
    "populate_initial_anti_patterns",
]
