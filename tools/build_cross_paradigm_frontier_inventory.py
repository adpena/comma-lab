#!/usr/bin/env python3
"""Build a deterministic cross-paradigm frontier inventory.

The output is an orchestration artifact, not a score ledger. It separates exact
evidence from planning/proposal surfaces and makes stack/replacement candidates
discoverable without launching GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.frontier_rows import (  # noqa: E402
    FRONTIER_ROW_FIELDS,
    FRONTIER_ROW_SCHEMA,
    build_frontier_row,
)
from tac.geometry_feedback_readiness import (  # noqa: E402
    GEOMETRY_FEEDBACK_ROADMAP_KEYS,
    build_geometry_feedback_runtime_contract,
)
from tac.repo_io import json_text  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_ACTION_CLASS = "research_review_before_local_patch"
ACTION_CLASS_BY_KEY = {
    "hnerv_pr103_pr106_ac_repack_runtime_closure": "maintain_exact_eval_anchor_and_pivot",
    "hnerv_lowlevel_brotli_repack": "exact_eval_or_promote_measured_rate_candidate",
    "hnerv_per_tensor_context_entropy": "reduce_entropy_model_overhead",
    "hnerv_wavelet_wr01_apply": "claim_exact_eval_packet_after_static_gate",
    "categorical_qma9_clade_spade_openpilot": "build_byte_closed_categorical_candidate",
    "cmg3_predictive_mask_grammar": "close_runtime_decoder_fixture",
    "lapose_motion_atom_allocator": "calibrate_planning_signal_and_attach_archive_consumer",
    "meta_lagrangian_cross_paradigm_allocator": "attach_byte_closed_manifest_gate",
    "sensitivity_omega_w_v3": "replace_stub_sensitivity_with_certified_cuda_artifact",
    "joint_admm_balle_arithmetic_stack": "wire_jcsp_submission_runtime_consumer",
    "telescopic_foveation_field": "charge_runtime_geometry_consumer_contract",
    "raft_radial_openpilot_pose": "emit_pose_disagreement_readiness_artifact",
    "selfcompress_mdl_fp4_tto": "prove_deterministic_export_and_inflate_closure",
}
PRIORITY_TIER_BY_KEY = {
    "categorical_qma9_clade_spade_openpilot": 10,
    "joint_admm_balle_arithmetic_stack": 20,
    "hnerv_per_tensor_context_entropy": 30,
    "telescopic_foveation_field": 40,
    "lapose_motion_atom_allocator": 50,
    "hnerv_wavelet_wr01_apply": 60,
    "sensitivity_omega_w_v3": 70,
    "selfcompress_mdl_fp4_tto": 80,
    "hnerv_lowlevel_brotli_repack": 90,
    "raft_radial_openpilot_pose": 90,
    "cmg3_predictive_mask_grammar": 100,
    "meta_lagrangian_cross_paradigm_allocator": 110,
    "hnerv_pr103_pr106_ac_repack_runtime_closure": 900,
}


@dataclass(frozen=True)
class InventoryRow:
    key: str
    title: str
    paradigms: tuple[str, ...]
    role: str
    status: str
    evidence_grade: str
    stackability: str
    replacement_potential: str
    code_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    next_patch: str
    blockers: tuple[str, ...]


STATIC_ROWS: tuple[InventoryRow, ...] = (
    InventoryRow(
        key="hnerv_pr103_pr106_ac_repack_runtime_closure",
        title="PR103 AC decoder repack inside PR106 envelope",
        paradigms=("alpha_mask_payload", "entropy_coding", "public_frontier_replay"),
        role="current_exact_rate_anchor",
        status="exact_cuda_a++_anchor_promoted",
        evidence_grade="A++ contest T4 exact CUDA plus contest-final compliance",
        stackability=(
            "high: replaces the PR106 decoder section with PR103 arithmetic-coded bytes "
            "while preserving fixed PR106 latents; should compose before scorer-changing atoms"
        ),
        replacement_potential="low: rate-only public-frontier repack, not a representation replacement",
        code_paths=(
            "src/tac/pr103_arithmetic_codec.py",
            "src/tac/pr103_pr106_runtime_closure.py",
            "submissions/pr103_pr106_final_runtime/inflate.py",
            "submissions/pr103_pr106_final_runtime/inflate.sh",
            "experiments/build_pr103_repacked_archive.py",
            "tools/prove_pr103_pr106_runtime_closure.py",
            "tools/prove_pr103_pr106_final_runtime_packet.py",
        ),
        evidence_paths=(
            "experiments/results/pr103_repack_pr106_standalone_20260507/manifest.json",
            "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json",
            "experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json",
            "experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.static.json",
            "experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json",
            "experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive_manifest.json",
            ".omx/research/pr103_pr106_runtime_closure_20260507_codex.md",
            ".omx/research/hnerv_pr103_lc_ac_schema_frontier_20260507_codex.md",
            ".omx/research/pr103_pr106_ac_repack_exact_eval_20260507_codex.md",
        ),
        next_patch=(
            "Use strict formula score 0.2089810755823297 at 185578 bytes as "
            "the current A++ HNeRV rate anchor (report-reconstructed score "
            "0.20898105277982337), feed it into Pareto/meta-Lagrangian "
            "calibration, and require future rate-only HNeRV candidates to "
            "beat this byte floor or stack cleanly before spending exact-eval "
            "wall clock."
        ),
        blockers=("completed anchor; next score movement requires a new lower-byte or scorer-changing candidate",),
    ),
    InventoryRow(
        key="hnerv_lowlevel_brotli_repack",
        title="HNeRV low-level brotli repack",
        paradigms=("alpha_mask_payload", "entropy_coding", "archive_packing"),
        role="stacker_rate_only",
        status="exact_evaluated_pr106x_rate_frontier",
        evidence_grade="A++ exact CUDA for PR106x; empirical for PR106 q10 rebuild until exact eval",
        stackability="high: raw-equivalent section recode should stack before scorer-changing atoms",
        replacement_potential="low: rate-only improvement, not a representation replacement",
        code_paths=(
            "src/tac/hnerv_lowlevel_packer.py",
            "src/tac/hnerv_section_repack.py",
            "tools/build_hnerv_lowlevel_repack_candidate.py",
            "tools/audit_hnerv_section_candidate_diff.py",
        ),
        evidence_paths=(
            "experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json",
            "experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/manifest.json",
            "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/result.json",
            ".omx/research/hnerv_decoder_recode_pr106_20260506_codex.md",
        ),
        next_patch=(
            "Promote only exact-evaluated archive SHAs; surface PR106x lgblock16 -1B and "
            "PR106 q10 as local archive candidates until candidate-specific preflight, lane "
            "claim, and exact CUDA auth eval land."
        ),
        blockers=("requires exact eval for each rebuilt archive SHA",),
    ),
    InventoryRow(
        key="hnerv_per_tensor_context_entropy",
        title="HNeRV per-tensor/context entropy recode",
        paradigms=("alpha_mask_payload", "entropy_coding"),
        role="stacker_rate_only",
        status="shared_context_fixture_landed_still_byte_negative",
        evidence_grade="empirical entropy-floor plus parity fixture profile",
        stackability="medium-high: could stack with HNeRV raw-equivalent section recodes",
        replacement_potential="low: recodes current decoder representation",
        code_paths=(
            "src/tac/hnerv_decoder_recode.py",
            "src/tac/arithmetic_qint_codec.py",
            "src/tac/optimization/entropy_codec_gap_audit.py",
            "src/tac/lossless/range_coder.py",
            "src/tac/lossless/frequency_coder.py",
            "tools/audit_entropy_codec_gap.py",
        ),
        evidence_paths=("experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json",),
        next_patch=(
            "Cluster or codebook-share HDC2 context tables; HDC2 cut PR106x penalty from "
            "+96,671B to +51,103B but remains byte-negative."
        ),
        blockers=(
            "HDC1/HDC2 parity fixtures are raw-equal but still byte-negative versus source brotli",
            "requires deterministic decoder runtime before any exact eval",
        ),
    ),
    InventoryRow(
        key="hnerv_wavelet_wr01_apply",
        title="HNeRV WR01 wavelet apply transform",
        paradigms=("alpha_mask_payload", "wavelet", "latent_repair"),
        role="stacker_scorer_changing",
        status="archive_candidate_preflighted",
        evidence_grade="empirical archive candidate; exact CUDA pending",
        stackability="medium: stack after raw-equivalent rate recodes, before sidechannel stacks",
        replacement_potential="low-medium: local residual transform, not full representation replacement",
        code_paths=(
            "src/tac/hnerv_wavelet_residual.py",
            "src/tac/hnerv_wavelet_apply_transform.py",
            "src/tac/hnerv_wavelet_apply_gate.py",
            "src/tac/hnerv_wavelet_compress_time_harness.py",
            "tools/build_hnerv_wavelet_apply_transform_candidate.py",
            "tools/build_hnerv_wavelet_compress_time_harness.py",
            "tools/build_wr01_exact_eval_packet.py",
        ),
        evidence_paths=(
            "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json",
            "experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json",
        ),
        next_patch=(
            "Use the compress-time harness to emit a byte-custody-backed WR01 atom plan, "
            "then exact CUDA only after lane claim and operator approval."
        ),
        blockers=("changes decoded output", "tiny byte win means component drift dominates EV"),
    ),
    InventoryRow(
        key="categorical_qma9_clade_spade_openpilot",
        title="Categorical QMA9 plus CLADE/SPADE/openpilot labels",
        paradigms=("alpha_mask_payload", "categorical_masks", "openpilot_priors"),
        role="replacement_or_mask_stacker",
        status="byte_closed_local_candidate_artifact_landed_blocked_on_parity",
        evidence_grade=(
            "local byte-closed PR91 HPM1 payload candidate plus deterministic "
            "archive-readiness audit; decode/reencode and runtime parity blocked"
        ),
        stackability=(
            "medium: strongest as mask grammar or class-conditioned residual layer; must not duplicate "
            "HNeRV decoder bytes"
        ),
        replacement_potential="high for mask stream if learned/predictive grammar avoids CMG2 collapse",
        code_paths=(
            "src/tac/categorical_compression_contract.py",
            "src/tac/categorical_candidate_readiness.py",
            "src/tac/categorical_candidate_plan.py",
            "src/tac/categorical_candidate_runtime_skeleton.py",
            "src/tac/categorical_openpilot_mask_prior_contract.py",
            "src/tac/categorical_payload_candidate.py",
            "src/tac/pr91_hpm1_readiness.py",
            "src/tac/pr91_hpm1_runtime_contract.py",
            "src/tac/qma9_range_mask_contract.py",
            "src/tac/qma9_run_grammar.py",
            "src/tac/qma9_alt_grammar.py",
            "src/tac/mask_grayscale_lut.py",
            "src/tac/semantic_quantization.py",
            "src/tac/contrib/diffusion_renderer.py",
            "src/tac/openpilot_seeding.py",
            "tools/audit_categorical_compression_contract.py",
            "tools/audit_categorical_candidate_readiness.py",
            "tools/audit_pr91_hpm1_readiness.py",
            "tools/audit_pr91_hpm1_runtime_contract.py",
            "tools/build_categorical_candidate_fixture.py",
            "tools/build_categorical_candidate_payload.py",
        ),
        evidence_paths=(
            ".omx/research/qma9_range_mask_deconstruction_20260503_codex.md",
            ".omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md",
            ".omx/research/categorical_byte_closed_payload_candidate_20260506_codex.md",
            "experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json",
            "experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json",
            "experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json",
            "experiments/results/categorical_openpilot_payload_candidate_20260506_codex/summary.json",
            "experiments/results/categorical_openpilot_payload_candidate_20260506_codex/readiness.json",
            ".omx/research/pr91_hpm1_phase_major_failure_classification_20260507_codex.json",
            ".omx/research/pr91_hpm1_submitted_prefix_token_recovery_tile_major_20260507_codex.json",
            ".omx/research/pr91_hpm1_next_row_suffix_scan_tile_major_20260507_codex.json",
        ),
        next_patch=(
            "Use the PR91/HPM1 phase-major and tile-major failure classifications to "
            "recover encoder-side probability/range-state contract drift or earlier "
            "context/order drift; then prove full decode/reencode parity before "
            "replacing the runtime skeleton with a charged consumer."
        ),
        blockers=(
            "raw lossless class wrapping was byte-regressive",
            "CMG2 exact evals showed PoseNet collapse",
            "CLADE/SPADE/openpilot priors need charged archive consumption",
            "local PR91 HPM1 payload candidate lacks full 600-frame decode/reencode parity",
            "phase-major reference row is argmax but stream still fails after 15989 symbols",
            "tile-major submitted prefix reaches 8274 symbols but fails at frame 0 group 12 symbol 210",
            "same-group next-coordinate suffix scan tested 1134 remaining rows and found 0 decodable",
            "prior probability/context or range-state grammar remains unrecovered",
            "runtime consumer is a charged fail-closed skeleton, not a decoder",
            "PR91 HPAC device contract is ambient/contradictory in public runtime sources",
        ),
    ),
    InventoryRow(
        key="cmg3_predictive_mask_grammar",
        title="CMG3 predictive row-span mask grammar",
        paradigms=("alpha_mask_payload", "categorical_masks", "foveation"),
        role="replacement_mask_stream",
        status="empirical_byte_probe_needs_runtime_decoder",
        evidence_grade="empirical byte probe only",
        stackability="medium: can feed residual/categorical/foveated atoms if decoder is closed",
        replacement_potential="high for C067-style mask stream; unknown for HNeRV frontier",
        code_paths=(
            "experiments/probe_predictive_mask_grammar.py",
            "experiments/build_cmg3_rowspan_candidate.py",
            "experiments/plan_cmg3_pixel_lagrangian_atoms.py",
            "experiments/build_cmg3_adaptive_runs_candidate.py",
        ),
        evidence_paths=(
            ".omx/research/shannon_floor_claim_matrix_20260430_codex.md",
            ".omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md",
        ),
        next_patch="Close a deterministic runtime decoder and exact archive fixture before ranking.",
        blockers=("existing CMG2 repairs are A-negative", "probe excludes decoder/archive overhead"),
    ),
    InventoryRow(
        key="lapose_motion_atom_allocator",
        title="LA-POSE motion atom allocator",
        paradigms=("la_pose", "openpilot_priors", "meta_lagrangian", "pose"),
        role="proposal_allocator",
        status="planning_chain_hardened_non_rankable_global_allocations",
        evidence_grade="diagnostic CUDA response allocated to pairs; planning only",
        stackability="high as allocator feedback for pose, foveation, categorical, and sidechannel atoms",
        replacement_potential="medium if it produces a charged motion decoder or pose-sidecar replacement",
        code_paths=(
            "src/tac/analysis/lapose_lite_inputs.py",
            "src/tac/analysis/lapose_motion_atoms.py",
            "src/tac/analysis/lapose_motion_evidence.py",
            "tools/build_lapose_lite_inputs_from_pair_metrics.py",
            "tools/build_lapose_motion_atom_manifest.py",
        ),
        evidence_paths=(
            "docs/runbooks/analysis_optimization_package_map.md",
            ".omx/research/geometry_feedback_runtime_consumer_contract_20260506_codex.md",
        ),
        next_patch=(
            "Keep labeled as LA-Pose-inspired until a paper-faithful inverse-dynamics encoder and "
            "pose head exist; add class/openpilot manifests, calibrate confidence, and require a "
            "charged archive consumer before dispatch."
        ),
        blockers=("current outputs are planning only", "confidence thresholds need calibration evidence"),
    ),
    InventoryRow(
        key="meta_lagrangian_cross_paradigm_allocator",
        title="Meta-lagrangian cross-paradigm allocator",
        paradigms=("meta_lagrangian", "all_paradigms"),
        role="canonical_ranker",
        status="field_acquisition_ranker_landed_planning_only",
        evidence_grade="derivation/planning with deterministic acquisition artifact",
        stackability=(
            "high: common row schema compares rate, pose, seg, class, hard-pair support, "
            "Pareto/KKT readiness, and Bayesian acquisition pressure"
        ),
        replacement_potential="none directly; selects concrete stack/replacement builders",
        code_paths=(
            "src/tac/optimization/meta_lagrangian_allocator.py",
            "src/tac/optimization/field_equation_planner.py",
            "src/tac/optimization/bayesian_experimental_design.py",
            "tools/build_field_equation_plan.py",
        ),
        evidence_paths=(
            ".omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md",
            ".omx/research/field_acquisition_ranking_20260507_codex.md",
            "reports/cross_paradigm_atom_ledger_v3_20260506.json",
        ),
        next_patch=(
            "Feed every paradigm into field_acquisition_ranking, then promote only rows with "
            "Pareto/KKT readiness, byte-closed archive manifests, and explicit Volterra/"
            "interaction assumptions."
        ),
        blockers=(
            "planner assumes additive expected deltas until exact stacked archives land",
            "live v3 atom ledger currently has zero design-ready rows",
            "high-acquisition planning rows still need byte-closed archive consumers",
        ),
    ),
    InventoryRow(
        key="sensitivity_omega_w_v3",
        title="Sensitivity-aware Omega-W-V3",
        paradigms=("beta_sensitivity", "quantization", "waterfill"),
        role="stacker_quantization",
        status="guarded_needs_real_sensitivity",
        evidence_grade="planning/diagnostic until certified sensitivity artifact",
        stackability="medium-high after HNeRV source archive identity is locked",
        replacement_potential="low-medium: quantization policy, not representation replacement",
        code_paths=(
            "src/tac/sensitivity_map.py",
            "src/tac/component_sensitivity_artifact.py",
            "src/tac/neural_weight_codec_sensitivity.py",
            "src/tac/owv3_sensitivity_weighted.py",
            "tools/dispatch_dryrun_omega_w_v3.py",
            "experiments/repack_pr106_with_water_filling.py",
        ),
        evidence_paths=(
            ".omx/research/component_sensitivity_map_certification_20260501_codex.md",
            ".omx/research/nwcs_beta_encoding_loop_greenup_20260507_codex.md",
        ),
        next_patch=(
            "Attach the deterministic NWCS stream manifest to a byte-closed archive/container "
            "candidate, then replace all-ones/stub sensitivity producers with certified CUDA/"
            "component artifacts."
        ),
        blockers=("stub sensitivity must fail closed", "component collapse risk"),
    ),
    InventoryRow(
        key="joint_admm_balle_arithmetic_stack",
        title="Joint ADMM plus Balle hyperprior plus arithmetic stack",
        paradigms=("gamma_joint_codec", "entropy_coding", "hyperprior"),
        role="stack_or_replacement_orchestrator",
        status="byte_closed_jcsp_member_landed_runtime_consumption_blocked",
        evidence_grade="empirical byte-closed JCSP archive member plus runtime-loader parity",
        stackability="high in architecture, unproven in exact archive",
        replacement_potential="medium-high if it can replace HNeRV sections with a smaller scorer-stable stack",
        code_paths=(
            "src/tac/joint_codec_stack_orchestrator.py",
            "src/tac/jcsp_stream_builder.py",
            "src/tac/joint_admm_coordinator.py",
            "src/tac/balle_hyperprior_codec.py",
            "src/tac/arithmetic_qint_codec.py",
            "src/tac/stack_compositions.py",
            "experiments/pipeline.py",
            "submissions/robust_current/jcsp_runtime_bridge.py",
            "tools/audit_arithmetic_qint_optimality.py",
            "tools/build_joint_stack_noop_manifest.py",
            "src/tac/entropy_archive.py",
        ),
        evidence_paths=(
            "docs/score_aware_sidechannel_paradigm_20260504.md",
            "experiments/results/joint_stack_noop_manifest_20260506_codex/manifest.json",
            ".omx/research/joint_stack_noop_manifest_20260506_codex.md",
            ".omx/research/jcsp_runtime_parity_hardening_20260506_codex.md",
        ),
        next_patch=(
            "Wire submissions/robust_current to decode/consume jcsp.bin and emit contest outputs, "
            "then claim a lane before exact CUDA auth eval."
        ),
        blockers=(
            "submission runtime detects but refuses jcsp.bin consumption",
            "Balle hyperprior stream codecs must be instantiated and charged for non-fixture model streams",
            "side information must be charged",
            "no lane dispatch claim",
            "no exact CUDA auth eval for stacked JCSP archive",
            "individual components do not imply score composability",
        ),
    ),
    InventoryRow(
        key="telescopic_foveation_field",
        title="Telescopic foveation field",
        paradigms=("foveation", "openpilot_priors", "categorical_masks", "pose"),
        role="scorer_weighted_proposal_or_replacement",
        status="historical_mixed_needs_qfaithful_successor_contract",
        evidence_grade="planning plus historical negatives",
        stackability="medium: best as atom-ranking/foveation prior before charged runtime warp",
        replacement_potential="medium if runtime consumes charged geometry without PoseNet collapse",
        code_paths=(
            "src/tac/hyperbolic_foveation.py",
            "src/tac/foveation_readiness.py",
            "tools/audit_hyperbolic_foveation_readiness.py",
            "experiments/plan_cmg3_pixel_lagrangian_atoms.py",
            "experiments/preflight_qfaithful_successor_geometry_contract.py",
            "src/tac/raft_radial_pose.py",
        ),
        evidence_paths=(
            ".omx/research/lane_g_v3_stacking_skunkworks_20260428.md",
            ".omx/research/all_scores_forensic_audit_20260430.md",
            ".omx/research/geometry_feedback_runtime_consumer_contract_20260506_codex.md",
        ),
        next_patch=(
            "Run charged foveation-params readiness audit, then keep foveation as ranking feedback "
            "until a runtime consumer passes geometry preflight and exact component gates."
        ),
        blockers=("configured-but-unconsumed foveation is a known fail-closed preflight class",),
    ),
    InventoryRow(
        key="raft_radial_openpilot_pose",
        title="RAFT/radial/openpilot pose basis",
        paradigms=("pose", "openpilot_priors", "la_pose"),
        role="proposal_or_pose_sidecar_replacement",
        status="prototype_readiness_needed",
        evidence_grade="planning/prototype",
        stackability="medium: can seed la-pose atoms, foveation centers, or pose-sidecar search",
        replacement_potential="medium if charged pose stream beats current pose bytes/components",
        code_paths=(
            "src/tac/raft_pose.py",
            "src/tac/raft_radial_pose.py",
            "src/tac/openpilot_seeding.py",
            "src/tac/openpilot_features.py",
        ),
        evidence_paths=(
            ".omx/research/council_lane_raft_radial_pose_design_20260430.md",
            ".omx/research/geometry_feedback_runtime_consumer_contract_20260506_codex.md",
        ),
        next_patch="Emit deterministic pose-disagreement and runtime-consumption readiness artifacts.",
        blockers=("RAFT full-frame jobs historically OOM at naive settings", "small pose errors can dominate score"),
    ),
    InventoryRow(
        key="selfcompress_mdl_fp4_tto",
        title="Self-compress NN plus MDL/Bayesian plus FP4/TTO",
        paradigms=("delta_epsilon_zeta", "self_compress", "mdl", "tto"),
        role="replacement_renderer_or_decoder",
        status="pending_integration",
        evidence_grade="planning/prototype",
        stackability="medium-low until deterministic export/inflate closure exists",
        replacement_potential="high if it can replace HNeRV renderer with smaller scorer-stable decoder",
        code_paths=(
            "src/tac/self_compressing_nn.py",
            "src/tac/mdl_bayesian_codec.py",
            "src/tac/joint_renderer_scorer_finetune.py",
            "src/tac/tto.py",
            "src/tac/fp4_quantize.py",
        ),
        evidence_paths=(".omx/research/council_lane_mdl_bayesian_design_20260430.md",),
        next_patch="Require deterministic export manifest, inflate budget proof, and no scorer load at inflate.",
        blockers=("runtime/export closure risk", "TTO must not rely on sidecars outside archive"),
    ),
)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _path_status(paths: tuple[str, ...], *, repo_root: Path) -> dict[str, Any]:
    existing = [path for path in paths if (repo_root / path).exists()]
    missing = [path for path in paths if not (repo_root / path).exists()]
    return {
        "existing": existing,
        "missing": missing,
        "existing_count": len(existing),
        "missing_count": len(missing),
    }


def _score_snapshot(row: InventoryRow, *, repo_root: Path) -> dict[str, Any] | None:
    for relpath in row.evidence_paths:
        if not (
            relpath.endswith("contest_auth_eval.adjudicated.json")
            or relpath.endswith("pre_submission_compliance.contest_final.json")
        ):
            continue
        payload = _load_json(repo_root / relpath)
        if not isinstance(payload, dict):
            continue
        auth_eval = payload.get("auth_eval") if isinstance(payload.get("auth_eval"), dict) else {}
        auth_record = (
            auth_eval.get("record") if isinstance(auth_eval.get("record"), dict) else {}
        )
        if auth_record:
            checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
            failed_checks = [
                str(check.get("name"))
                for check in checks
                if isinstance(check, dict) and check.get("passed") is not True
            ]
            strict_formula = (
                auth_eval.get("strict_formula")
                if isinstance(auth_eval.get("strict_formula"), dict)
                else {}
            )
            anchor_proof = (
                auth_eval.get("anchor_proof")
                if isinstance(auth_eval.get("anchor_proof"), dict)
                else {}
            )
            score = strict_formula.get("score", auth_record.get("score"))
            archive_bytes = strict_formula.get(
                "archive_bytes",
                auth_record.get("archive_bytes"),
            )
            seg_dist = strict_formula.get(
                "avg_segnet_dist",
                auth_record.get("avg_segnet_dist"),
            )
            pose_dist = strict_formula.get(
                "avg_posenet_dist",
                auth_record.get("avg_posenet_dist"),
            )
            return {
                "path": relpath,
                "compliance_passed": payload.get("passed") is True,
                "compliance_failed_checks": failed_checks,
                "compliance_check_count": len(checks),
                "score": score,
                "report_reconstructed_score": auth_record.get("score"),
                "score_basis": strict_formula.get("basis"),
                "score_delta_vs_report_reconstruction": strict_formula.get(
                    "score_delta_vs_report_reconstruction"
                ),
                "anchor_proof_schema": anchor_proof.get("schema"),
                "archive_bytes": archive_bytes,
                "archive_sha256": auth_record.get("archive_sha256"),
                "seg_dist": seg_dist,
                "pose_dist": pose_dist,
            }
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        score = payload.get(
            "score_recomputed_from_components",
            payload.get("canonical_score", payload.get("final_score", payload.get("score"))),
        )
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
        return {
            "path": relpath,
            "score": score,
            "archive_bytes": archive.get(
                "bytes",
                payload.get("archive_bytes", payload.get("archive_size_bytes")),
            ),
            "archive_sha256": archive.get(
                "sha256",
                payload.get("archive_sha256", provenance.get("archive_sha256")),
            ),
            "seg_dist": components.get("seg_dist", payload.get("seg_dist", payload.get("avg_segnet_dist"))),
            "pose_dist": components.get(
                "pose_dist",
                payload.get("pose_dist", payload.get("avg_posenet_dist")),
            ),
        }
    return None


def _action_class(row: InventoryRow) -> str:
    return ACTION_CLASS_BY_KEY.get(row.key, DEFAULT_ACTION_CLASS)


def _priority_tier(row: InventoryRow) -> int:
    return PRIORITY_TIER_BY_KEY.get(row.key, 999)


def _frontier_action_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for row in rows:
        queue_row = {
            "key": row["key"],
            "priority_tier": row["priority_tier"],
            "action_class": row["action_class"],
            "role": row["role"],
            "paradigms": row["paradigms"],
            "status": row["status"],
            "next_patch": row["next_patch"],
            "blockers": row["blockers"],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if row.get("geometry_feedback_contract") is not None:
            queue_row["geometry_feedback_contract"] = row["geometry_feedback_contract"]
        queue.append(queue_row)
    queue.sort(
        key=lambda item: (
            int(item["priority_tier"]),
            str(item["action_class"]),
            str(item["key"]),
        )
    )
    return queue


def build_inventory(*, repo_root: Path) -> dict[str, Any]:
    rows = []
    for row in STATIC_ROWS:
        code_status = _path_status(row.code_paths, repo_root=repo_root)
        evidence_status = _path_status(row.evidence_paths, repo_root=repo_root)
        geometry_contract = _geometry_feedback_contract(row)
        blockers = _row_blockers(row, geometry_contract)
        score_snapshot = _score_snapshot(row, repo_root=repo_root)
        score_evidence_rankable = score_snapshot is not None and row.evidence_grade.startswith("A++")
        exact_anchor_rankable = (
            score_evidence_rankable
            and row.key == "hnerv_pr103_pr106_ac_repack_runtime_closure"
        )
        row_payload = {
            "key": row.key,
            "title": row.title,
            "paradigms": list(row.paradigms),
            "role": row.role,
            "status": row.status,
            "evidence_grade": row.evidence_grade,
            "stackability": row.stackability,
            "replacement_potential": row.replacement_potential,
            "code_paths": list(row.code_paths),
            "evidence_paths": list(row.evidence_paths),
            "path_audit": {
                "code": code_status,
                "evidence": evidence_status,
            },
            "score_snapshot": score_snapshot,
            "next_patch": row.next_patch,
            "blockers": blockers,
            "action_class": _action_class(row),
            "priority_tier": _priority_tier(row),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }
        row_payload["frontier_row"] = build_frontier_row(
            source_tool="tools/build_cross_paradigm_frontier_inventory.py",
            key=row.key,
            candidate_id=row.key,
            title=row.title,
            family=row.key,
            family_group=row.key,
            pareto_scope=row.key,
            paradigms=row.paradigms,
            role=row.role,
            status=row.status,
            evidence_grade=row.evidence_grade,
            action_class=row_payload["action_class"],
            priority_tier=row_payload["priority_tier"],
            score_claim=False,
            dispatch_attempted=False,
            candidate_static_preflight_ready=False,
            ready_for_exact_eval_dispatch=False,
            pareto_eligible=exact_anchor_rankable,
            pareto_frontier=exact_anchor_rankable,
            score_evidence_rankable=score_evidence_rankable,
            score_evidence_path=(score_snapshot or {}).get("path", ""),
            exact_score=(score_snapshot or {}).get("score"),
            archive_bytes=(score_snapshot or {}).get("archive_bytes"),
            archive_sha256=(score_snapshot or {}).get("archive_sha256", ""),
            planning_priority_rankable=exact_anchor_rankable,
            blockers=blockers,
            next_required_proof=(
                "candidate_specific_archive_manifest",
                "lane_dispatch_claim",
                "exact_cuda_auth_eval",
            ),
            next_patch=row.next_patch,
            code_paths=row.code_paths,
            evidence_paths=row.evidence_paths,
        )
        if geometry_contract is not None:
            row_payload["geometry_feedback_contract"] = geometry_contract
        rows.append(row_payload)
    role_counts = Counter(row["role"] for row in rows)
    paradigm_counts = Counter(paradigm for row in rows for paradigm in row["paradigms"])
    action_class_counts = Counter(row["action_class"] for row in rows)
    missing_code = sum(row["path_audit"]["code"]["missing_count"] for row in rows)
    missing_evidence = sum(row["path_audit"]["evidence"]["missing_count"] for row in rows)
    action_queue = _frontier_action_queue(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/build_cross_paradigm_frontier_inventory.py",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "research_grade_standard": {
            "external_method_rule": (
                "Do not claim 1:1 paper implementation unless the repo has the paper's core "
                "model, training objective, inputs, outputs, and evaluation contract."
            ),
            "contest_native_rule": (
                "Contest-native methods must carry typed contracts, deterministic manifests, "
                "no-op controls, byte-closed archive consumption, and exact CUDA custody before "
                "promotion."
            ),
        },
        "row_count": len(rows),
        "frontier_row_schema": FRONTIER_ROW_SCHEMA,
        "frontier_row_fields": list(FRONTIER_ROW_FIELDS),
        "frontier_row_count": len(rows),
        "frontier_rows": [row["frontier_row"] for row in rows],
        "role_counts": dict(sorted(role_counts.items())),
        "paradigm_counts": dict(sorted(paradigm_counts.items())),
        "action_class_counts": dict(sorted(action_class_counts.items())),
        "missing_code_path_count": missing_code,
        "missing_evidence_path_count": missing_evidence,
        "frontier_action_queue": action_queue,
        "frontier_action_queue_count": len(action_queue),
        "rows": rows,
        "dispatch_blockers": [
            "inventory_only",
            "geometry_feedback_requires_charged_runtime_consumer",
            "requires_candidate_specific_archive_manifest",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _geometry_feedback_contract(row: InventoryRow) -> dict[str, Any] | None:
    if row.key not in GEOMETRY_FEEDBACK_ROADMAP_KEYS:
        return None
    return build_geometry_feedback_runtime_contract(
        lane_key=row.key,
        paradigms=row.paradigms,
        role=row.role,
        evidence_grade=row.evidence_grade,
    )


def _row_blockers(
    row: InventoryRow,
    geometry_contract: dict[str, Any] | None,
) -> list[str]:
    blockers = list(row.blockers)
    if geometry_contract is not None:
        for blocker in geometry_contract["dispatch_blockers"]:
            if blocker not in blockers:
                blockers.append(blocker)
    return blockers


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Paradigm Frontier Inventory",
        "",
        "Inventory-only orchestration artifact. It does not claim scores or unlock dispatch.",
        "",
        f"- row_count: `{payload['row_count']}`",
        f"- missing_code_path_count: `{payload['missing_code_path_count']}`",
        f"- missing_evidence_path_count: `{payload['missing_evidence_path_count']}`",
        "",
        "| key | tier | action | role | paradigms | status | stackability | replacement | next patch | blockers |",
        "|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(row['key'])}`",
                    str(row["priority_tier"]),
                    f"`{_md(row['action_class'])}`",
                    f"`{_md(row['role'])}`",
                    "<br>".join(f"`{_md(item)}`" for item in row["paradigms"]),
                    _md(row["status"]),
                    _md(row["stackability"]),
                    _md(row["replacement_potential"]),
                    _md(row["next_patch"]),
                    "<br>".join(_md(item) for item in row["blockers"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_inventory(repo_root=args.repo_root)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        sys.stdout.write(json_text(payload) if args.format == "json" else render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
