# SPDX-License-Identifier: MIT
"""Source-derived PR95 distortion-practice guard for NeRV launch rows.

This guard is deliberately narrower than the full PR95 stack-binding matrix:
it only checks practices that directly affect scorer-domain distortion and
renderer collapse risk before a HiNeRV/SNeRV local-MLX row is admitted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.pr95_stack_binding_requirements import FALSE_AUTHORITY
from tac.contest_eval_contract import build_score_allocation_contract

SCHEMA = "pr95_distortion_practices_guard.v1"
SOURCE_INVENTORY_SCHEMA = "pr95_distortion_source_inventory.v1"
PRACTICE_ROW_SCHEMA = "pr95_distortion_practice_row.v1"
PAYLOAD_GUARD_SCHEMA = "pr95_distortion_practices_payload_guard.v1"
TELEMETRY_CONTRACT_SCHEMA = "pr95_evaluate_scorer_domain_telemetry_contract.v1"
PRACTICE_DAG_SCHEMA = "pr95_distortion_practices_dag.v1"
STAGE_DAG_SCHEMA = "pr95_eight_stage_optimization_dag.v1"
AXIS_TRACE_CONTRACT_SCHEMA = "pr95_distortion_axis_trace_contract.v1"
SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA = "pr95_scorer_atom_actuator_contract.v1"
POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA = "pr95_posenet_marginal_telemetry_contract.v1"

PR95_SOURCE_REL = Path(
    "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon"
)
UPSTREAM_REL = Path("upstream")
RUNNER_REL = Path("tools/run_compact_renderer_mlx_spine_runner.py")


@dataclass(frozen=True)
class Practice:
    practice_id: str
    title: str
    why_it_matters: str
    source_check_ids: tuple[str, ...]


PRACTICES: tuple[Practice, ...] = (
    Practice(
        practice_id="official_non_overlapping_seq2_pair_geometry",
        title="official non-overlapping seq_len=2 pair geometry",
        why_it_matters=(
            "upstream evaluate.py scores 600 non-overlapping 2-frame samples; "
            "overlapping pair construction changes PoseNet targets and can hide "
            "collapsed temporal geometry."
        ),
        source_check_ids=(
            "upstream_frame_utils_seq_len_2",
            "upstream_evaluate_seq_len_shape_assert",
            "pr95_score_streams_non_overlapping_pairs",
        ),
    ),
    Practice(
        practice_id="scorer_preprocess_eval_roundtrip_yuv6",
        title="PoseNet YUV6 plus uint8 eval-roundtrip STE",
        why_it_matters=(
            "PR95 trains through the camera-resize/rounding scorer surface and "
            "PoseNet consumes two-frame YUV6; skipping either lets a renderer fit "
            "RGB-looking proxies while collapsing scorer inputs."
        ),
        source_check_ids=(
            "upstream_posenet_uses_yuv6_pair",
            "upstream_rgb_to_yuv6_is_no_grad",
            "pr95_training_eval_roundtrip_ste",
            "runner_hinerv_eval_roundtrip_metadata",
        ),
    ),
    Practice(
        practice_id="dual_component_real_scorer_pressure",
        title="both real SegNet last-frame and PoseNet pair pressure",
        why_it_matters=(
            "upstream SegNet scores only the last frame by hard argmax while "
            "PoseNet scores pair motion; single-component pressure can look "
            "healthy while the other component collapses."
        ),
        source_check_ids=(
            "upstream_segnet_last_frame",
            "upstream_segnet_argmax_distortion",
            "pr95_losses_include_seg_margin_and_pose",
        ),
    ),
    Practice(
        practice_id="official_evaluate_archive_byte_price",
        title="official evaluate.py archive.zip byte price",
        why_it_matters=(
            "PR95 optimization prices bytes through archive.zip size over the "
            "source-video denominator; nominal modelsize or inflated raw bytes "
            "are not acceptable substitutes for the charged packet."
        ),
        source_check_ids=("upstream_evaluate_archive_byte_price",),
    ),
    Practice(
        practice_id="scorer_domain_telemetry_contract",
        title="SegNet argmax/occupancy and PoseNet YUV6 telemetry contract",
        why_it_matters=(
            "A row can carry PR95-looking weights and curriculum flags while "
            "still missing the telemetry that proves scorer-domain controls "
            "actuate. Long runs must fail closed if SegNet frame-1 argmax/"
            "occupancy or PoseNet YUV6-pair metrics disappear."
        ),
        source_check_ids=(
            "upstream_segnet_last_frame",
            "upstream_segnet_argmax_distortion",
            "upstream_posenet_uses_yuv6_pair",
        ),
    ),
    Practice(
        practice_id="posenet_marginal_vjp_telemetry_contract",
        title="PoseNet frontier marginal and VJP telemetry contract",
        why_it_matters=(
            "evaluate.py prices PoseNet as sqrt(10*d_pose), whose local marginal "
            "5/sqrt(10*d_pose) grows as d_pose shrinks. Long runs must not treat "
            "PoseNet as a late cleanup axis or accept raw-MSE-only telemetry."
        ),
        source_check_ids=(
            "upstream_posenet_uses_yuv6_pair",
            "pr95_losses_include_seg_margin_and_pose",
        ),
    ),
    Practice(
        practice_id="family_local_scorer_atom_actuator_contract",
        title="family-local scorer-atom actuator contract",
        why_it_matters=(
            "PR95's 28-D per-pair latents matched the evaluator atom. HiNeRV "
            "needs grid/head/pair-adapter controls, while SNeRV needs MFU/HFR/TUB "
            "and LF/HF incidence controls; cross-family actuator evidence is not "
            "launch authority."
        ),
        source_check_ids=(
            "pr95_losses_include_seg_margin_and_pose",
            "upstream_posenet_uses_yuv6_pair",
        ),
    ),
    Practice(
        practice_id="pr95_staged_qat_coder_curriculum",
        title="PR95 staged curriculum with QAT and coder pressure",
        why_it_matters=(
            "PR95 did not run one homogeneous proxy loss: it staged CE, margin "
            "losses, QAT, C1a entropy pressure, sigma tightening, and final "
            "optimizer polish so distortion stayed attached to the charged packet."
        ),
        source_check_ids=(
            "pr95_eight_stage_curriculum_present",
            "pr95_qat_and_c1a_present",
            "pr95_stage8_muon_present",
        ),
    ),
    Practice(
        practice_id="archive_parseback_distortion_axis_trace",
        title="live/fake-quant/parse-back/inflate/evaluate trace contract",
        why_it_matters=(
            "PR95 selected archive-parsed candidates, not live tensors. "
            "HiNeRV/SNeRV launch rows need the same axis chain so live scorer "
            "movement cannot hide fake-quant, packet parse-back, inflate, or "
            "official evaluate.py divergence."
        ),
        source_check_ids=(
            "upstream_evaluate_archive_byte_price",
            "pr95_training_eval_roundtrip_ste",
        ),
    ),
)

PRACTICE_DAG_EDGES: Mapping[str, tuple[str, ...]] = {
    "official_non_overlapping_seq2_pair_geometry": (),
    "official_evaluate_archive_byte_price": (),
    "scorer_preprocess_eval_roundtrip_yuv6": (
        "official_non_overlapping_seq2_pair_geometry",
    ),
    "dual_component_real_scorer_pressure": (
        "scorer_preprocess_eval_roundtrip_yuv6",
    ),
    "scorer_domain_telemetry_contract": (
        "dual_component_real_scorer_pressure",
    ),
    "posenet_marginal_vjp_telemetry_contract": (
        "scorer_domain_telemetry_contract",
    ),
    "family_local_scorer_atom_actuator_contract": (
        "dual_component_real_scorer_pressure",
        "posenet_marginal_vjp_telemetry_contract",
    ),
    "pr95_staged_qat_coder_curriculum": (
        "dual_component_real_scorer_pressure",
        "official_evaluate_archive_byte_price",
        "scorer_domain_telemetry_contract",
        "posenet_marginal_vjp_telemetry_contract",
        "family_local_scorer_atom_actuator_contract",
    ),
    "archive_parseback_distortion_axis_trace": (
        "pr95_staged_qat_coder_curriculum",
        "official_evaluate_archive_byte_price",
    ),
}

PRACTICE_DAG_LAYER_METADATA: Mapping[str, Mapping[str, str]] = {
    "official_non_overlapping_seq2_pair_geometry": {
        "layer": "geometry",
        "math_surface": "600 non-overlapping seq_len=2 samples",
        "architecture_surface": "frame sampler and pair teacher construction",
        "stabilization_role": "prevents PoseNet target aliasing",
    },
    "official_evaluate_archive_byte_price": {
        "layer": "rate_lagrangian",
        "math_surface": "25 * archive_zip_bytes / source_video_bytes",
        "architecture_surface": "archive.zip packet grammar and byte oracle",
        "stabilization_role": "keeps optimizer pressure tied to charged bytes",
    },
    "scorer_preprocess_eval_roundtrip_yuv6": {
        "layer": "scorer_manifold",
        "math_surface": "resize + uint8 STE + two-frame YUV6 PoseNet input",
        "architecture_surface": "differentiable scorer input preprocessing",
        "stabilization_role": "keeps gradients on the upstream evaluate.py manifold",
    },
    "dual_component_real_scorer_pressure": {
        "layer": "distortion_lagrangian",
        "math_surface": "100*d_seg + sqrt(10*d_pose)",
        "architecture_surface": "real SegNet last-frame and PoseNet pair teachers",
        "stabilization_role": "prevents single-component collapse",
    },
    "scorer_domain_telemetry_contract": {
        "layer": "actuator_observability",
        "math_surface": "argmax occupancy plus YUV6 geometry metrics",
        "architecture_surface": "telemetry and checkpoint selection contract",
        "stabilization_role": "fails closed when a requested loss is not consumed",
    },
    "posenet_marginal_vjp_telemetry_contract": {
        "layer": "pose_score_marginal",
        "math_surface": "d/d(d_pose) sqrt(10*d_pose) = 5/sqrt(10*d_pose)",
        "architecture_surface": "direct-live PoseNet VJP and marginal telemetry",
        "stabilization_role": "keeps low-d_pose frontier pressure visible",
    },
    "family_local_scorer_atom_actuator_contract": {
        "layer": "local_controllability",
        "math_surface": "one controllable actuator per scorer atom and incidence axis",
        "architecture_surface": "HiNeRV grid/head adapter or SNeRV MFU/HFR/TUB/LF-HF binding",
        "stabilization_role": "prevents global-only credit assignment collapse",
    },
    "pr95_staged_qat_coder_curriculum": {
        "layer": "optimization_schedule",
        "math_surface": "CE -> margins -> QAT -> C1a -> sigma -> Muon polish",
        "architecture_surface": "PR95-faithful 8-stage Muon+AdamW curriculum",
        "stabilization_role": "stages convergence before byte-closed promotion",
    },
    "archive_parseback_distortion_axis_trace": {
        "layer": "authority_axis_chain",
        "math_surface": "same score trend across live, QAT, packet, inflate, evaluate",
        "architecture_surface": "archive parse-back and exact inflate/evaluate trace",
        "stabilization_role": "rejects live-only distortion wins before long runs",
    },
}

PR95_STAGE_DAG_ROWS: tuple[dict[str, Any], ...] = (
    {
        "stage_index": 1,
        "stage_id": "stage1_v328_ce",
        "epochs": 3000,
        "loss_family": "ce",
        "qat_active": False,
        "c1a_lambda": 0.0,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": [],
        "required_signals": ["pr95_curriculum"],
    },
    {
        "stage_index": 2,
        "stage_id": "stage2_v331_softplus",
        "epochs": 5650,
        "loss_family": "tau_softplus",
        "qat_active": False,
        "c1a_lambda": 0.0,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": ["stage1_v328_ce"],
        "required_signals": ["pr95_curriculum"],
    },
    {
        "stage_index": 3,
        "stage_id": "stage3_v332_smooth",
        "epochs": 1500,
        "loss_family": "smooth_disagreement",
        "qat_active": False,
        "c1a_lambda": 0.0,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": ["stage2_v331_softplus"],
        "required_signals": ["pr95_curriculum"],
    },
    {
        "stage_index": 4,
        "stage_id": "stage4_v332_qat",
        "epochs": 500,
        "loss_family": "smooth_disagreement",
        "qat_active": True,
        "c1a_lambda": 0.0,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": ["stage3_v332_smooth"],
        "required_signals": ["pr95_curriculum", "coder_qat"],
    },
    {
        "stage_index": 5,
        "stage_id": "stage5_c1a_l7",
        "epochs": 9000,
        "loss_family": "l7_softplus",
        "qat_active": True,
        "c1a_lambda": 0.01,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": ["stage4_v332_qat"],
        "required_signals": ["pr95_curriculum", "coder_qat", "c1a_entropy"],
    },
    {
        "stage_index": 6,
        "stage_id": "stage6_lambda_sweep",
        "epochs": 2000,
        "loss_family": "l7_softplus",
        "qat_active": True,
        "c1a_lambda": 0.02,
        "sigma": 0.2,
        "uses_muon": False,
        "depends_on": ["stage5_c1a_l7"],
        "required_signals": ["pr95_curriculum", "coder_qat", "c1a_entropy"],
    },
    {
        "stage_index": 7,
        "stage_id": "stage7_sigma_sweep",
        "epochs": 3000,
        "loss_family": "l7_softplus",
        "qat_active": True,
        "c1a_lambda": 0.02,
        "sigma": 0.1,
        "uses_muon": False,
        "depends_on": ["stage6_lambda_sweep"],
        "required_signals": ["pr95_curriculum", "coder_qat", "c1a_entropy"],
    },
    {
        "stage_index": 8,
        "stage_id": "stage8_muon_finetune",
        "epochs": 5000,
        "loss_family": "l7_softplus",
        "qat_active": True,
        "c1a_lambda": 0.02,
        "sigma": 0.1,
        "uses_muon": True,
        "depends_on": ["stage7_sigma_sweep"],
        "required_signals": [
            "pr95_curriculum",
            "coder_qat",
            "c1a_entropy",
            "muon_adamw_partition",
            "muon_stage8_only",
        ],
    },
)


def build_pr95_evaluate_scorer_domain_telemetry_contract(
    family: str,
) -> dict[str, Any]:
    """Return the launch-row telemetry contract for PR95 distortion readiness."""

    family_key = str(family).strip().lower().replace("-", "_") or "unknown"
    if family_key == "hinerv":
        family_key = "hi_nerv"
    return {
        "schema": TELEMETRY_CONTRACT_SCHEMA,
        "family": family_key,
        "source": "upstream/evaluate.py+modules.py+frame_utils.py",
        "segnet_scored_frame_index": 1,
        "segnet_frame_scope": "last_frame_only",
        "segnet_distortion": "mean(argmax(gt_logits)!=argmax(candidate_logits))",
        "segnet_last_frame_argmax_metric_names": [
            f"dual_ascent_metric__{family_key}_segnet_last_frame_distill",
            "loss_part_segnet_direct_live_argmax_disagreement",
            "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement",
        ],
        "segnet_argmax_occupancy_metric_names": [
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
            "loss_part_segnet_direct_live_candidate_any_occupied_class_fraction",
            ("loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction"),
            "selection_health_segnet_direct_live_candidate_occupied_class_fraction",
            "post_export_receiver_segnet_candidate_occupied_class_fraction",
        ],
        "argmax_occupancy_gate_required": True,
        "posenet_scored_frame_indices": [0, 1],
        "posenet_input_domain": "two RGB frames -> resize -> YUV6 pair",
        "posenet_yuv6_pair_metric_names": [
            f"dual_ascent_metric__{family_key}_posenet_yuv6_pair_distill",
            ("loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_min_std_ratio"),
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair_mse",
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta_mse",
        ],
        "posenet_marginal_metric_names": [
            "loss_part_pose_direct_live_raw_mse",
            "loss_part_pose_direct_live_score_term",
            "loss_part_pose_direct_live_score_marginal_wrt_raw_mse",
            "loss_part_pose_direct_live_vjp_norm_by_group",
        ],
        "pose_geometry_gate_required": True,
        "pose_marginal_gate_required": True,
        "required_metric_groups": [
            "segnet_last_frame_argmax",
            "segnet_argmax_occupancy",
            "posenet_yuv6_pair",
            "posenet_marginal_vjp",
        ],
        "fail_closed_on_missing_metrics": True,
        **FALSE_AUTHORITY,
    }


def build_pr95_posenet_marginal_telemetry_contract(family: str) -> dict[str, Any]:
    """Return the PoseNet marginal/VJP telemetry contract for long-run readiness."""

    family_key = str(family).strip().lower().replace("-", "_") or "unknown"
    if family_key == "hinerv":
        family_key = "hi_nerv"
    score_allocation = build_score_allocation_contract()
    return {
        "schema": POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA,
        "family": family_key,
        "source": "upstream/evaluate.py score term sqrt(10*d_pose)",
        "pose_score_term": "sqrt(10*d_pose)",
        "pose_marginal_formula": score_allocation["posenet"]["derivative_wrt_d_pose"],
        "pose_marginal_increases_as_d_pose_decreases": True,
        "required_telemetry": [
            "pose_direct_live_raw_mse",
            "pose_direct_live_score_term",
            "pose_direct_live_score_marginal_wrt_raw_mse",
            "pose_direct_live_vjp_norm_by_group",
            "pose_direct_live_yuv6_pair_std",
            "pose_direct_live_yuv6_pair_temporal_delta_std",
            "mlx_torch_posenet_forward_parity",
        ],
        "acceptance_policy": {
            "pose_is_not_late_cleanup_axis": True,
            "fail_closed_on_missing_direct_live_pose_marginal": True,
            "fail_closed_on_zero_or_nan_pose_vjp": True,
            "long_run_admission_requires_pose_marginal_telemetry": True,
        },
        **FALSE_AUTHORITY,
    }


def build_pr95_scorer_atom_actuator_contract(family: str) -> dict[str, Any]:
    """Return the family-specific scorer-atom actuator contract."""

    family_key = str(family).strip().lower().replace("-", "_") or "unknown"
    if family_key == "hinerv":
        family_key = "hi_nerv"
    if family_key == "snerv":
        family_actuators = [
            "official_mfu_hfr_tub_source_forward_parity",
            "tub_output2_segnet_last_frame_binding",
            "lf_posenet_yuv6_pair_carrier",
            "hf_segnet_boundary_margin_carrier",
            "pair_conditioned_mfu_hfr_tub_adapter",
        ]
        control_surface = "SNeRV MFU/HFR/TUB plus LF/HF scorer-incidence controls"
    else:
        family_actuators = [
            "hierarchical_grid_saliency",
            "output_head_target_region_bias",
            "target_region_waterfill",
            "pair_local_film_or_latent_adapter",
        ]
        control_surface = "HiNeRV hierarchical grid/output-head plus pair-adapter controls"
    return {
        "schema": SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA,
        "family": family_key,
        "source": "PR95 28-D per-pair latent scorer-atom control",
        "control_surface": control_surface,
        "common_scorer_atoms": [
            "600_non_overlapping_pairs",
            "segnet_last_frame_argmax_regions",
            "posenet_two_frame_yuv6_pair_motion",
            "archive_section_value_per_byte",
        ],
        "family_actuators": family_actuators,
        "acceptance_policy": {
            "family_specific_actuators_are_not_interchangeable": True,
            "pair_local_smoke_required_before_long_run": True,
            "cross_family_evidence_rejected": True,
            "actuator_must_report_grad_norm_by_group": True,
        },
        **FALSE_AUTHORITY,
    }


def build_pr95_distortion_axis_trace_contract(family: str) -> dict[str, Any]:
    """Return the scorer-axis trace contract required before long campaigns."""

    family_key = str(family).strip().lower().replace("-", "_") or "unknown"
    if family_key == "hinerv":
        family_key = "hi_nerv"
    return {
        "schema": AXIS_TRACE_CONTRACT_SCHEMA,
        "family": family_key,
        "source": (
            "PR95 archive parse-back selection plus upstream/evaluate.py "
            "DistortionNet score axes"
        ),
        "required_axes": [
            "live_forward",
            "fakequant_forward",
            "archive_parseback",
            "inflate_replay",
            "official_evaluate_py",
        ],
        "axis_order_is_dependency_order": True,
        "acceptance_policy": {
            "accepted_update_must_reduce_worst_region_debt": True,
            "live_only_improvement_is_false_authority": True,
            "fakequant_score_delta_must_be_bounded_before_stage5": True,
            "parseback_score_delta_must_be_bounded_before_stage6": True,
            "inflate_and_official_evaluate_required_before_promotion": True,
            "fail_closed_on_axis_divergence": True,
        },
        "required_metrics": [
            "score_live",
            "score_fakequant",
            "score_parseback",
            "score_inflate",
            "score_official_evaluate",
            "d_seg_live",
            "d_seg_parseback",
            "d_pose_live",
            "d_pose_parseback",
            "pose_direct_live_score_marginal_wrt_raw_mse",
            "pose_direct_live_vjp_norm_by_group",
            "worst_region_score_debt",
            "worst_region_margin_p50",
            "worst_region_margin_p90",
            "pose_worst_pair",
            "archive_bytes_by_section",
            "grad_norm_by_group",
        ],
        "stage_gates": [
            {
                "stage": "class_birth",
                "pass_condition": (
                    "target classes have nonzero direct-live support and "
                    "worst-region score debt decreases on consecutive evals"
                ),
            },
            {
                "stage": "margin_crossing",
                "pass_condition": "worst-region target margin median crosses zero",
            },
            {
                "stage": "argmax_disagreement",
                "pass_condition": "actual argmax disagreement decreases",
            },
            {
                "stage": "fakequant_survival",
                "pass_condition": "fake-quant score delta remains bounded",
            },
            {
                "stage": "archive_parseback_survival",
                "pass_condition": "parse-back score remains close to live score",
            },
            {
                "stage": "pose_marginal_vjp",
                "pass_condition": (
                    "direct-live PoseNet marginal telemetry is finite and at "
                    "least one trainable parameter group has nonzero VJP norm"
                ),
            },
            {
                "stage": "late_byte_and_optimizer_pressure",
                "pass_condition": (
                    "section byte duals, entropy pressure, and Muon activate "
                    "only after earlier gates pass"
                ),
            },
        ],
        **FALSE_AUTHORITY,
    }


class PR95DistortionPracticesGuardError(ValueError):
    """Raised when a PR95 distortion guard input is malformed."""


def build_pr95_distortion_source_inventory(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the source evidence inventory used by distortion-practice rows."""

    repo = Path(repo_root).expanduser().resolve(strict=False)
    pr95 = repo / PR95_SOURCE_REL
    runner = repo / RUNNER_REL
    records: list[dict[str, Any]] = []
    blockers: list[str] = []

    def add_record(
        *,
        rel_path: Path,
        check_id: str,
        required_tokens: Sequence[str],
    ) -> None:
        path = repo / rel_path
        record = _source_record(path, repo_root=repo)
        token_rows = []
        for token in required_tokens:
            token_rows.append(
                {
                    "token": token,
                    "present": bool(record.get("exists")) and token in record.get("text", ""),
                }
            )
        passed = bool(record.get("exists")) and all(row["present"] for row in token_rows)
        if not passed:
            blockers.append(f"source_check_failed:{check_id}")
        record.update(
            {
                "check_id": check_id,
                "required_tokens": token_rows,
                "passed": passed,
            }
        )
        record.pop("text", None)
        records.append(record)

    add_record(
        rel_path=UPSTREAM_REL / "frame_utils.py",
        check_id="upstream_frame_utils_seq_len_2",
        required_tokens=("seq_len = 2", "seq_buf = []"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "evaluate.py",
        check_id="upstream_evaluate_seq_len_shape_assert",
        required_tokens=(
            "TensorVideoDataset",
            "[seq_len, camera_size[1], camera_size[0], 3]",
            "DistortionNet().eval()",
        ),
    )
    add_record(
        rel_path=UPSTREAM_REL / "evaluate.py",
        check_id="upstream_evaluate_archive_byte_price",
        required_tokens=(
            "compressed_size = (args.submission_dir / 'archive.zip').stat().st_size",
            "uncompressed_size = sum(file.stat().st_size",
            "rate = compressed_size / uncompressed_size",
            "score = 100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate",
        ),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_posenet_uses_yuv6_pair",
        required_tokens=("IN_CHANS = 6 * 2", "rgb_to_yuv6(x)", "b (t c) h w"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "frame_utils.py",
        check_id="upstream_rgb_to_yuv6_is_no_grad",
        required_tokens=("@torch.no_grad()", "def rgb_to_yuv6"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_segnet_last_frame",
        required_tokens=("x = x[:, -1, ...]", "SegNet"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_segnet_argmax_distortion",
        required_tokens=("out1.argmax(dim=1)", "out2.argmax(dim=1)"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "score.py",
        check_id="pr95_score_streams_non_overlapping_pairs",
        required_tokens=("gt_pairs_iter_state['prev'] = None", "compute_distortion"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "stages" / "common.py",
        check_id="pr95_training_eval_roundtrip_ste",
        required_tokens=(
            "F.interpolate(up, size=(384, 512)",
            "decoded_clamped.round()",
            "distortion_net.preprocess_input(decoded_bhwc)",
        ),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "stages" / "common.py",
        check_id="pr95_archive_contains_latents",
        required_tokens=(
            "HNeRVDecoder(latent_dim=28",
            "latents = nn.Parameter(torch.randn(n_pairs, 28",
            "decoder(latents[idx])",
        ),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "losses.py",
        check_id="pr95_losses_include_seg_margin_and_pose",
        required_tokens=(
            "ce_seg_loss",
            "smooth_disagreement_seg_loss",
            "l7_softplus_seg_loss",
            "def pose_loss",
        ),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "losses.py",
        check_id="pr95_qat_and_c1a_present",
        required_tokens=("cat_entropy_v2", "fake_quantize", "apply_qat"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "stages" / "stage8_muon_finetune.py",
        check_id="pr95_stage8_muon_present",
        required_tokens=("use_muon=True", "muon_lr", "Muon"),
    )
    add_record(
        rel_path=RUNNER_REL,
        check_id="runner_hinerv_eval_roundtrip_metadata",
        required_tokens=(
            "_hi_nerv_eval_roundtrip_ste_metadata",
            "eval_roundtrip_ste_enabled",
            "pose_student_input_preprocess",
        ),
    )

    stage_paths = sorted((pr95 / "src" / "stages").glob("stage*.py"))
    expected_stage_names = {
        "stage1_v328_ce.py",
        "stage2_v331_softplus.py",
        "stage3_v332_smooth.py",
        "stage4_v332_qat.py",
        "stage5_c1a_l7.py",
        "stage6_lambda_sweep.py",
        "stage7_sigma_sweep.py",
        "stage8_muon_finetune.py",
    }
    present_stage_names = {path.name for path in stage_paths}
    eight_stage_passed = expected_stage_names.issubset(present_stage_names)
    if not eight_stage_passed:
        blockers.append("source_check_failed:pr95_eight_stage_curriculum_present")
    records.append(
        {
            "schema": "pr95_distortion_source_record.v1",
            "check_id": "pr95_eight_stage_curriculum_present",
            "path": (PR95_SOURCE_REL / "src" / "stages").as_posix(),
            "exists": (pr95 / "src" / "stages").is_dir(),
            "sha256": _combined_sha256(stage_paths),
            "bytes": sum(path.stat().st_size for path in stage_paths if path.is_file()),
            "required_stage_files": sorted(expected_stage_names),
            "present_stage_files": sorted(present_stage_names),
            "passed": eight_stage_passed,
        }
    )

    check_passed = {str(record.get("check_id")): bool(record.get("passed")) for record in records}
    practice_source_rows = []
    for practice in PRACTICES:
        missing = [check_id for check_id in practice.source_check_ids if check_passed.get(check_id) is not True]
        practice_source_rows.append(
            {
                "schema": "pr95_distortion_practice_source_row.v1",
                "practice_id": practice.practice_id,
                "title": practice.title,
                "source_check_ids": list(practice.source_check_ids),
                "source_ready": not missing,
                "missing_source_check_ids": missing,
            }
        )

    inventory = {
        "schema": SOURCE_INVENTORY_SCHEMA,
        "repo_root": repo.as_posix(),
        "pr95_source_dir": (repo / PR95_SOURCE_REL).as_posix(),
        "upstream_dir": (repo / UPSTREAM_REL).as_posix(),
        "runner_path": runner.as_posix(),
        "source_records": records,
        "practice_source_rows": practice_source_rows,
        "source_ready": not blockers,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    inventory["sha256"] = _payload_sha256(inventory)
    return inventory


def build_pr95_distortion_practices_row_guard(
    row: Mapping[str, Any],
    *,
    repo_root: str | Path,
    source_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed PR95 distortion-practice guard for one launch row."""

    if not isinstance(row, Mapping):
        raise PR95DistortionPracticesGuardError("row must be a mapping")
    inventory = (
        dict(source_inventory)
        if isinstance(source_inventory, Mapping)
        else build_pr95_distortion_source_inventory(repo_root)
    )
    command = _command_list(row)
    family = str(row.get("family") or _arg_value(command, "--execute-family") or "unknown")
    row_id = str(row.get("id") or row.get("row_id") or row.get("candidate_id") or "")
    required = family in {"hi_nerv", "snerv"}
    source_by_practice = {
        str(item.get("practice_id")): item for item in _mapping_list(inventory.get("practice_source_rows"))
    }

    practice_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for practice in PRACTICES:
        source_row = source_by_practice.get(practice.practice_id, {})
        source_ready = source_row.get("source_ready") is True
        observed, observed_evidence = _observe_practice(
            practice.practice_id,
            family=family,
            row=row,
            command=command,
            source_inventory=inventory,
        )
        passed = (not required) or (source_ready and observed)
        blocker = None
        if required and not passed:
            blocker = f"{family}_pr95_distortion_{practice.practice_id}_missing"
            blockers.append(blocker)
        practice_rows.append(
            {
                "schema": PRACTICE_ROW_SCHEMA,
                "practice_id": practice.practice_id,
                "title": practice.title,
                "required_for_family": required,
                "source_ready": source_ready,
                "observed": observed,
                "passed": passed,
                "blocker": blocker,
                "observed_evidence": observed_evidence,
                "source_check_ids": list(practice.source_check_ids),
                "why_it_matters": practice.why_it_matters,
            }
        )

    if required and inventory.get("source_ready") is not True:
        blockers.insert(0, "pr95_distortion_source_inventory_incomplete")

    practice_dag = _build_practice_dag(
        family=family,
        required=required,
        practice_rows=practice_rows,
    )
    stage_dag = _build_stage_dag(row=row, command=command, family=family)
    if required and stage_dag.get("all_required_stage_signals_observed") is not True:
        blockers.extend(_string_list(stage_dag.get("blockers")))
    passed_count = sum(1 for item in practice_rows if item["passed"])
    guard = {
        "schema": SCHEMA,
        "family": family,
        "row_id": row_id,
        "required_for_family": required,
        "command": command,
        "source_inventory_schema": inventory.get("schema"),
        "source_inventory_sha256": inventory.get("sha256"),
        "required_practice_count": len(PRACTICES) if required else 0,
        "passed_practice_count": passed_count if required else 0,
        "launch_allowed": bool(
            not blockers and practice_dag.get("all_nodes_green") is True
        ),
        "practice_rows": practice_rows,
        "practice_dag": practice_dag,
        "optimization_stage_dag": stage_dag,
        "dag_blockers": _dedupe(
            [
                *_string_list(practice_dag.get("blockers")),
                *_string_list(stage_dag.get("blockers")),
            ]
        ),
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    guard["sha256"] = _payload_sha256(guard)
    return guard


def build_pr95_distortion_practices_payload_guard(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build row guards for a verdict, queue, report, or single row payload."""

    if not isinstance(payload, Mapping):
        raise PR95DistortionPracticesGuardError("payload must be a mapping")
    inventory = build_pr95_distortion_source_inventory(repo_root)
    rows = _extract_candidate_rows(payload)
    row_guards = [
        build_pr95_distortion_practices_row_guard(
            row,
            repo_root=repo_root,
            source_inventory=inventory,
        )
        for row in rows
    ]
    blockers = [blocker for guard in row_guards for blocker in _string_list(guard.get("blockers"))]
    if not rows:
        blockers.append("pr95_distortion_guard_no_candidate_rows_found")
    if inventory.get("source_ready") is not True:
        blockers.append("pr95_distortion_source_inventory_incomplete")
    out = {
        "schema": PAYLOAD_GUARD_SCHEMA,
        "source_inventory": inventory,
        "candidate_row_count": len(rows),
        "row_guards": row_guards,
        "launch_allowed": not blockers,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    out["sha256"] = _payload_sha256(out)
    return out


def render_pr95_distortion_practices_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for guard artifacts."""

    lines = [
        "# PR95 Distortion Practices Guard",
        "",
        f"Schema: `{payload.get('schema')}`",
        f"Launch allowed: `{payload.get('launch_allowed')}`",
        f"Candidate rows: `{payload.get('candidate_row_count', 1)}`",
        "",
        "## Practices",
        "",
    ]
    row_guards = _mapping_list(payload.get("row_guards"))
    if not row_guards and payload.get("schema") == SCHEMA:
        row_guards = [payload]
    for guard in row_guards:
        lines.append(f"### `{guard.get('row_id') or guard.get('family')}`")
        for row in _mapping_list(guard.get("practice_rows")):
            lines.append(f"- `{row.get('practice_id')}` passed=`{row.get('passed')}` observed=`{row.get('observed')}`")
        lines.append("")
    lines.append("## Blockers")
    lines.append("")
    blockers = _string_list(payload.get("blockers"))
    if not blockers and row_guards:
        blockers = [blocker for guard in row_guards for blocker in _string_list(guard.get("blockers"))]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in _dedupe(blockers))
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _build_practice_dag(
    *,
    family: str,
    required: bool,
    practice_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_id = {str(row.get("practice_id")): row for row in practice_rows}
    node_green: dict[str, bool] = {}
    nodes: list[dict[str, Any]] = []
    blockers: list[str] = []
    first_failed_nodes: list[str] = []

    for practice in PRACTICES:
        practice_id = practice.practice_id
        row = rows_by_id.get(practice_id, {})
        depends_on = list(PRACTICE_DAG_EDGES.get(practice_id, ()))
        missing_prerequisites = [
            dep for dep in depends_on if node_green.get(dep) is not True
        ]
        practice_passed = row.get("passed") is True
        green = (not required) or (practice_passed and not missing_prerequisites)
        node_green[practice_id] = green
        metadata = dict(PRACTICE_DAG_LAYER_METADATA.get(practice_id, {}))
        dependency_blocker = None
        if required and practice_passed and missing_prerequisites:
            dependency_blocker = (
                f"{family}_pr95_dag_{practice_id}_blocked_by_"
                + "_and_".join(missing_prerequisites)
            )
            blockers.append(dependency_blocker)
        if required and not green and not missing_prerequisites:
            first_failed_nodes.append(practice_id)
        nodes.append(
            {
                "schema": "pr95_distortion_practice_dag_node.v1",
                "practice_id": practice_id,
                "depends_on": depends_on,
                "missing_prerequisites": missing_prerequisites,
                "source_ready": row.get("source_ready") is True,
                "observed": row.get("observed") is True,
                "practice_passed": practice_passed,
                "green": green,
                "status": (
                    "not_required"
                    if not required
                    else "green"
                    if green
                    else "blocked_by_prerequisite"
                    if missing_prerequisites
                    else "missing"
                ),
                "practice_blocker": row.get("blocker"),
                "dependency_blocker": dependency_blocker,
                **metadata,
            }
        )

    all_nodes_green = (not required) or all(node["green"] for node in nodes)
    return {
        "schema": PRACTICE_DAG_SCHEMA,
        "family": family,
        "required_for_family": required,
        "node_count": len(nodes),
        "all_nodes_green": all_nodes_green,
        "first_failed_practice_ids": first_failed_nodes,
        "nodes": nodes,
        "blockers": _dedupe(blockers),
        "policy": {
            "dependency_status_is_launch_relevant": True,
            "flat_practice_rows_are_not_sufficient_without_green_dag": True,
            "math_geometry_rate_and_optimization_layers_are_ordered": True,
        },
        **FALSE_AUTHORITY,
    }


def _build_stage_dag(
    *,
    row: Mapping[str, Any],
    command: Sequence[str],
    family: str,
) -> dict[str, Any]:
    signals = _stage_observed_signals(row=row, command=command, family=family)
    stage_green: dict[str, bool] = {}
    nodes: list[dict[str, Any]] = []
    first_failed: list[str] = []
    blockers: list[str] = []
    for stage in PR95_STAGE_DAG_ROWS:
        stage_id = str(stage["stage_id"])
        depends_on = [str(item) for item in stage.get("depends_on", [])]
        missing_prerequisites = [
            dep for dep in depends_on if stage_green.get(dep) is not True
        ]
        missing_signals = [
            str(signal)
            for signal in stage.get("required_signals", [])
            if signals.get(str(signal)) is not True
        ]
        green = not missing_prerequisites and not missing_signals
        stage_green[stage_id] = green
        if not green and not missing_prerequisites:
            first_failed.append(stage_id)
        if missing_signals:
            blockers.append(
                f"{family}_pr95_stage_dag_{stage_id}_missing_"
                + "_and_".join(missing_signals)
            )
        nodes.append(
            {
                "schema": "pr95_eight_stage_optimization_dag_node.v1",
                **{
                    key: value
                    for key, value in stage.items()
                    if key not in {"depends_on", "required_signals"}
                },
                "depends_on": depends_on,
                "required_signals": list(stage.get("required_signals", [])),
                "missing_prerequisites": missing_prerequisites,
                "missing_signals": missing_signals,
                "green": green,
                "status": (
                    "green"
                    if green
                    else "blocked_by_prerequisite"
                    if missing_prerequisites
                    else "missing_signals"
                ),
            }
        )
    return {
        "schema": STAGE_DAG_SCHEMA,
        "family": family,
        "source": (
            "experiments/results/public_pr95_intake_20260504_codex/"
            "profile_pr95_hnerv_muon_intake.md"
        ),
        "canonical_stage_count": len(PR95_STAGE_DAG_ROWS),
        "canonical_total_epochs": sum(
            int(stage["epochs"]) for stage in PR95_STAGE_DAG_ROWS
        ),
        "observed_signals": signals,
        "all_required_stage_signals_observed": all(
            bool(node["green"]) for node in nodes
        ),
        "first_failed_stage_ids": first_failed,
        "nodes": nodes,
        "blockers": _dedupe(blockers),
        "policy": {
            "stage8_muon_depends_on_prior_qat_and_c1a": True,
            "stage_dag_is_observability_not_score_authority": True,
        },
        **FALSE_AUTHORITY,
    }


def _stage_observed_signals(
    *,
    row: Mapping[str, Any],
    command: Sequence[str],
    family: str,
) -> dict[str, bool]:
    hinerv_policy = _arg_value(command, "--hi-nerv-optimizer-policy")
    snerv_curriculum_opted_out = _has_flag(
        command,
        "--no-snerv-score-aware-long-training-pr95-faithful-curriculum",
    )
    snerv_curriculum_flag = _has_flag(
        command,
        "--snerv-score-aware-long-training-pr95-faithful-curriculum",
    )
    family_key = str(family).strip().lower().replace("-", "_")
    optimizer = _arg_value(command, "--optimizer-kind") or _arg_value(
        command,
        "--snerv-score-aware-long-training-optimizer",
    )
    snerv_muon_policy = _arg_value(
        command,
        "--snerv-score-aware-long-training-pr95-muon-policy",
    )
    pr95_curriculum = bool(
        hinerv_policy == "pr95_curriculum"
        or snerv_curriculum_flag
        or (family_key == "snerv" and not snerv_curriculum_opted_out)
        or _deep_truthy(
            row,
            {
                "pr95_faithful_curriculum_enabled",
                "pr95_staged_curriculum",
                "native_mlx_pr95_curriculum_bound",
            },
        )
    )
    coder_qat = bool(_has_flag(command, "--coder-aware-qat"))
    c1a_entropy = bool(_positive_float_arg(command, "--coder-qat-c1a-entropy-weight"))
    muon_partition = bool(
        str(optimizer or "") in {"pact_muon_adamw", "muon", "pr95_8stage_muon_adamw"}
        or hinerv_policy == "pr95_curriculum"
        or _deep_truthy(
            row,
            {
                "muon_adamw_partition",
                "muon_adamw_partition_bound",
                "native_mlx_muon_adamw_partition_bound",
            },
        )
    )
    muon_stage8_only = bool(
        hinerv_policy == "pr95_curriculum"
        or snerv_muon_policy in {None, "", "faithful_stage8_only"}
        or _deep_truthy(
            row,
            {
                "faithful_stage8_only",
                "stage8_muon_depends_on_prior_qat_and_c1a",
            },
        )
    )
    return {
        "pr95_curriculum": pr95_curriculum,
        "coder_qat": coder_qat,
        "c1a_entropy": c1a_entropy,
        "muon_adamw_partition": muon_partition,
        "muon_stage8_only": muon_stage8_only,
    }


def _observe_practice(
    practice_id: str,
    *,
    family: str,
    row: Mapping[str, Any],
    command: Sequence[str],
    source_inventory: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if practice_id == "official_non_overlapping_seq2_pair_geometry":
        num_pairs = _positive_int_arg(command, "--num-pairs")
        batch_pairs = _positive_int_arg(command, "--batch-pairs") or _positive_int_arg(
            command, "--snerv-score-aware-long-training-batch-pairs"
        )
        evidence = []
        if num_pairs:
            evidence.append(f"command_num_pairs={num_pairs}")
        if batch_pairs:
            evidence.append(f"command_batch_pairs={batch_pairs}")
        return bool(num_pairs and batch_pairs), evidence

    if practice_id == "scorer_preprocess_eval_roundtrip_yuv6":
        evidence = []
        pose_weight = _positive_float_arg(command, "--pose-distillation-weight")
        if pose_weight:
            evidence.append(f"pose_distillation_weight={pose_weight:g}")
        explicit_snerv = _has_flag(command, "--snerv-score-aware-long-training-eval-roundtrip-ste")
        snerv_opted_out = _has_flag(
            command,
            "--no-snerv-score-aware-long-training-eval-roundtrip-ste",
        )
        default_snerv = family == "snerv" and not snerv_opted_out
        if explicit_snerv:
            evidence.append("snerv_eval_roundtrip_ste_flag")
        if default_snerv and not explicit_snerv:
            evidence.append("snerv_eval_roundtrip_ste_default_true")
        hinerv_runner_support = _source_check_passed(source_inventory, "runner_hinerv_eval_roundtrip_metadata")
        if family == "hi_nerv" and hinerv_runner_support:
            evidence.append("hinerv_runner_eval_roundtrip_metadata_source_verified")
        payload_declares = _deep_truthy(
            row,
            {
                "eval_roundtrip_ste_enabled",
                "eval_roundtrip_ste_attached",
                "native_mlx_eval_roundtrip_ste_bound",
            },
        )
        if payload_declares:
            evidence.append("payload_eval_roundtrip_truthy")
        observed = bool(pose_weight) and (
            explicit_snerv
            or default_snerv
            or payload_declares
            or (family == "hi_nerv" and hinerv_runner_support)
        )
        return observed, evidence

    if practice_id == "dual_component_real_scorer_pressure":
        seg_weight = _positive_float_arg(command, "--segnet-distillation-weight") or _positive_float_arg(
            command, "--segnet-direct-live-distillation-weight"
        )
        pose_weight = _positive_float_arg(command, "--pose-distillation-weight")
        device = _arg_value(command, "--distillation-device")
        evidence = []
        if seg_weight:
            evidence.append(f"segnet_distillation_weight={seg_weight:g}")
        if pose_weight:
            evidence.append(f"pose_distillation_weight={pose_weight:g}")
        if device:
            evidence.append(f"distillation_device={device}")
        return bool(seg_weight and pose_weight and device), evidence

    if practice_id == "official_evaluate_archive_byte_price":
        binding = _first_mapping(row, ("upstream_evaluate_score_binding",))
        rate = binding.get("rate") if isinstance(binding, Mapping) else None
        rate = rate if isinstance(rate, Mapping) else {}
        price = _positive_float_value(rate.get("rate_price_per_archive_byte"))
        denominator = _positive_int_value(rate.get("canonical_denominator_bytes"))
        raw_not_denominator = _positive_int_value(rate.get("raw_output_shape_bytes_are_not_rate_denominator"))
        archive_authority = str(rate.get("archive_authority") or "")
        hard_byte_ceiling = _positive_int_arg(command, "--hard-byte-ceiling") or (
            _positive_int_value(row.get("hard_byte_ceiling"))
        )
        archive_bound = "archive.zip" in archive_authority and ".stat()" in archive_authority
        evidence = []
        if price:
            evidence.append(f"rate_price_per_archive_byte={price:.12g}")
        if denominator:
            evidence.append(f"canonical_denominator_bytes={denominator}")
        if raw_not_denominator:
            evidence.append(f"raw_output_shape_bytes_are_not_rate_denominator={raw_not_denominator}")
        if archive_authority:
            evidence.append(f"archive_authority={archive_authority}")
        if hard_byte_ceiling:
            evidence.append(f"command_hard_byte_ceiling={hard_byte_ceiling}")
        return bool(
            binding and price and denominator and raw_not_denominator and archive_bound and hard_byte_ceiling
        ), evidence

    if practice_id == "scorer_domain_telemetry_contract":
        contract = _first_mapping(
            row,
            (
                "pr95_evaluate_scorer_domain_telemetry_contract",
                "scorer_domain_telemetry_contract",
            ),
        )
        evidence = []
        schema_ok = contract.get("schema") == TELEMETRY_CONTRACT_SCHEMA
        if contract.get("schema"):
            evidence.append(f"telemetry_contract_schema={contract.get('schema')}")
        fail_closed = contract.get("fail_closed_on_missing_metrics") is True
        if fail_closed:
            evidence.append("fail_closed_on_missing_metrics")
        segnet_frame_ok = _int_value(contract.get("segnet_scored_frame_index")) == 1
        if segnet_frame_ok:
            evidence.append("segnet_scored_frame_index=1")
        posenet_frames_ok = _int_sequence(contract.get("posenet_scored_frame_indices")) == [0, 1]
        if posenet_frames_ok:
            evidence.append("posenet_scored_frame_indices=[0,1]")
        segnet_argmax_metrics = _string_list(contract.get("segnet_last_frame_argmax_metric_names"))
        segnet_occupancy_metrics = _string_list(contract.get("segnet_argmax_occupancy_metric_names"))
        posenet_metrics = _string_list(contract.get("posenet_yuv6_pair_metric_names"))
        has_segnet_argmax = _contains_any_substring(
            segnet_argmax_metrics,
            ("argmax", "segnet_last_frame_distill"),
        )
        has_segnet_occupancy = contract.get("argmax_occupancy_gate_required") is True and _contains_any_substring(
            segnet_occupancy_metrics,
            ("occupied_class_fraction", "occupancy"),
        )
        has_posenet_yuv6 = contract.get("pose_geometry_gate_required") is True and _contains_any_substring(
            posenet_metrics, ("posenet_yuv6", "yuv6_pair")
        )
        if has_segnet_argmax:
            evidence.append("segnet_argmax_metrics=" + ",".join(sorted(segnet_argmax_metrics)[:4]))
        if has_segnet_occupancy:
            evidence.append("segnet_occupancy_metrics=" + ",".join(sorted(segnet_occupancy_metrics)[:4]))
        if has_posenet_yuv6:
            evidence.append("posenet_yuv6_metrics=" + ",".join(sorted(posenet_metrics)[:4]))
        return bool(
            schema_ok
            and fail_closed
            and segnet_frame_ok
            and posenet_frames_ok
            and has_segnet_argmax
            and has_segnet_occupancy
            and has_posenet_yuv6
        ), evidence

    if practice_id == "pr95_staged_qat_coder_curriculum":
        evidence = []
        hinerv_policy = _arg_value(command, "--hi-nerv-optimizer-policy")
        snerv_opted_out = _has_flag(
            command,
            "--no-snerv-score-aware-long-training-pr95-faithful-curriculum",
        )
        snerv_curriculum = _has_flag(
            command,
            "--snerv-score-aware-long-training-pr95-faithful-curriculum",
        ) or (family == "snerv" and not snerv_opted_out)
        coder_qat = _has_flag(command, "--coder-aware-qat")
        c1a_weight = _positive_float_arg(command, "--coder-qat-c1a-entropy-weight")
        optimizer = _arg_value(command, "--optimizer-kind") or _arg_value(
            command, "--snerv-score-aware-long-training-optimizer"
        )
        if hinerv_policy:
            evidence.append(f"hi_nerv_optimizer_policy={hinerv_policy}")
        if snerv_curriculum:
            evidence.append("snerv_pr95_faithful_curriculum_flag")
        if coder_qat:
            evidence.append("coder_aware_qat_flag")
        if c1a_weight:
            evidence.append(f"coder_qat_c1a_entropy_weight={c1a_weight:g}")
        if optimizer:
            evidence.append(f"optimizer={optimizer}")
        curriculum = (
            hinerv_policy == "pr95_curriculum"
            or snerv_curriculum
            or _deep_truthy(row, {"pr95_faithful_curriculum_enabled", "pr95_staged_curriculum"})
        )
        return bool(curriculum and coder_qat and c1a_weight), evidence

    if practice_id == "archive_parseback_distortion_axis_trace":
        contract = _first_mapping(
            row,
            (
                "pr95_distortion_axis_trace_contract",
                "distortion_axis_trace_contract",
            ),
        )
        evidence = []
        schema_ok = contract.get("schema") == AXIS_TRACE_CONTRACT_SCHEMA
        if contract.get("schema"):
            evidence.append(f"axis_trace_contract_schema={contract.get('schema')}")
        axes = _string_list(contract.get("required_axes"))
        required_axes = {
            "live_forward",
            "fakequant_forward",
            "archive_parseback",
            "inflate_replay",
            "official_evaluate_py",
        }
        axes_ok = required_axes.issubset(set(axes))
        if axes_ok:
            evidence.append("required_axes=" + ",".join(sorted(required_axes)))
        policy = contract.get("acceptance_policy")
        policy = policy if isinstance(policy, Mapping) else {}
        fail_closed = policy.get("fail_closed_on_axis_divergence") is True
        live_false_authority = (
            policy.get("live_only_improvement_is_false_authority") is True
        )
        parseback_gate = (
            policy.get("parseback_score_delta_must_be_bounded_before_stage6")
            is True
        )
        if fail_closed:
            evidence.append("fail_closed_on_axis_divergence")
        if live_false_authority:
            evidence.append("live_only_improvement_is_false_authority")
        if parseback_gate:
            evidence.append("parseback_score_delta_must_be_bounded_before_stage6")
        stage_gates = _mapping_list(contract.get("stage_gates"))
        gate_names = {str(gate.get("stage")) for gate in stage_gates}
        gates_ok = {
            "class_birth",
            "margin_crossing",
            "argmax_disagreement",
            "fakequant_survival",
            "archive_parseback_survival",
            "pose_marginal_vjp",
            "late_byte_and_optimizer_pressure",
        }.issubset(gate_names)
        if gates_ok:
            evidence.append("stage_gates=" + ",".join(sorted(gate_names)))
        return bool(
            schema_ok
            and axes_ok
            and fail_closed
            and live_false_authority
            and parseback_gate
            and gates_ok
        ), evidence

    if practice_id == "posenet_marginal_vjp_telemetry_contract":
        contract = _first_mapping(
            row,
            (
                "pr95_posenet_marginal_telemetry_contract",
                "posenet_marginal_telemetry_contract",
            ),
        )
        evidence = []
        schema_ok = contract.get("schema") == POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA
        if contract.get("schema"):
            evidence.append(f"pose_marginal_contract_schema={contract.get('schema')}")
        formula_ok = str(contract.get("pose_marginal_formula") or "") == (
            "5/sqrt(10*d_pose)"
        )
        if formula_ok:
            evidence.append("pose_marginal_formula=5/sqrt(10*d_pose)")
        required = set(_string_list(contract.get("required_telemetry")))
        required_ok = {
            "pose_direct_live_raw_mse",
            "pose_direct_live_score_term",
            "pose_direct_live_score_marginal_wrt_raw_mse",
            "pose_direct_live_vjp_norm_by_group",
            "mlx_torch_posenet_forward_parity",
        }.issubset(required)
        if required_ok:
            evidence.append("required_pose_marginal_and_vjp_telemetry")
        policy = contract.get("acceptance_policy")
        policy = policy if isinstance(policy, Mapping) else {}
        fail_closed = (
            policy.get("fail_closed_on_missing_direct_live_pose_marginal") is True
            and policy.get("fail_closed_on_zero_or_nan_pose_vjp") is True
            and policy.get("long_run_admission_requires_pose_marginal_telemetry")
            is True
        )
        if fail_closed:
            evidence.append("pose_marginal_fail_closed_policy")
        return bool(schema_ok and formula_ok and required_ok and fail_closed), evidence

    if practice_id == "family_local_scorer_atom_actuator_contract":
        contract = _first_mapping(
            row,
            (
                "pr95_scorer_atom_actuator_contract",
                "scorer_atom_actuator_contract",
            ),
        )
        evidence = []
        schema_ok = contract.get("schema") == SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA
        if contract.get("schema"):
            evidence.append(f"actuator_contract_schema={contract.get('schema')}")
        family_ok = str(contract.get("family") or "") == family
        if family_ok:
            evidence.append(f"actuator_family={family}")
        atoms = set(_string_list(contract.get("common_scorer_atoms")))
        atoms_ok = {
            "600_non_overlapping_pairs",
            "segnet_last_frame_argmax_regions",
            "posenet_two_frame_yuv6_pair_motion",
        }.issubset(atoms)
        if atoms_ok:
            evidence.append("common_scorer_atoms_bound")
        actuators = set(_string_list(contract.get("family_actuators")))
        if family == "snerv":
            family_actuators_ok = {
                "official_mfu_hfr_tub_source_forward_parity",
                "tub_output2_segnet_last_frame_binding",
                "lf_posenet_yuv6_pair_carrier",
                "hf_segnet_boundary_margin_carrier",
            }.issubset(actuators)
        else:
            family_actuators_ok = {
                "hierarchical_grid_saliency",
                "output_head_target_region_bias",
                "target_region_waterfill",
                "pair_local_film_or_latent_adapter",
            }.issubset(actuators)
        if family_actuators_ok:
            evidence.append("family_specific_actuators_bound")
        policy = contract.get("acceptance_policy")
        policy = policy if isinstance(policy, Mapping) else {}
        policy_ok = (
            policy.get("family_specific_actuators_are_not_interchangeable") is True
            and policy.get("pair_local_smoke_required_before_long_run") is True
            and policy.get("cross_family_evidence_rejected") is True
            and policy.get("actuator_must_report_grad_norm_by_group") is True
        )
        if policy_ok:
            evidence.append("family_local_actuator_fail_closed_policy")
        return bool(
            schema_ok and family_ok and atoms_ok and family_actuators_ok and policy_ok
        ), evidence

    return False, []


def _source_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    exists = path.is_file()
    text = path.read_text(encoding="utf-8") if exists else ""
    try:
        rel = path.relative_to(repo_root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return {
        "schema": "pr95_distortion_source_record.v1",
        "path": rel,
        "exists": exists,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if exists else None,
        "bytes": len(text.encode("utf-8")) if exists else 0,
        "text": text,
    }


def _extract_candidate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = _mapping_list(payload.get("selected_local_mlx_experiments"))
    if direct:
        return direct
    queue = payload.get("experiment_queue")
    if isinstance(queue, Mapping):
        experiments = _mapping_list(queue.get("experiments"))
        rows = [row for row in experiments if _command_list(row)]
        if rows:
            return rows
    experiments = _mapping_list(payload.get("experiments"))
    if experiments:
        rows = [row for row in experiments if _command_list(row)]
        if rows:
            return rows
    selected_rows = _mapping_list(payload.get("selected_rows"))
    if selected_rows:
        rows = [row for row in selected_rows if _command_list(row)]
        if rows:
            return rows
    if _command_list(payload):
        return [payload]
    return []


def _command_list(row: Mapping[str, Any]) -> list[str]:
    for key in ("command", "command_argv", "argv"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
    steps = row.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in steps:
            if isinstance(step, Mapping):
                value = step.get("command")
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return [str(item) for item in value]
    return []


def _arg_value(command: Sequence[str], flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return flag in command


def _positive_int_arg(command: Sequence[str], flag: str) -> int | None:
    value = _arg_value(command, flag)
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float_arg(command: Sequence[str], flag: str) -> float | None:
    value = _arg_value(command, flag)
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _positive_float_value(value: Any) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _positive_int_value(value: Any) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int_value(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _int_sequence(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out = []
    for item in value:
        parsed = _int_value(item)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def _first_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    for key in keys:
        item = value.get(key)
        if isinstance(item, Mapping):
            return item
    launch_authority = value.get("launch_authority_contract")
    if isinstance(launch_authority, Mapping):
        for key in keys:
            item = launch_authority.get(key)
            if isinstance(item, Mapping):
                return item
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        for source in (metadata, metadata.get("source_selected_row")):
            if not isinstance(source, Mapping):
                continue
            for key in keys:
                item = source.get(key)
                if isinstance(item, Mapping):
                    return item
            nested_authority = source.get("launch_authority_contract")
            if isinstance(nested_authority, Mapping):
                for key in keys:
                    item = nested_authority.get(key)
                    if isinstance(item, Mapping):
                        return item
    return {}


def _contains_any_substring(values: Sequence[str], needles: Sequence[str]) -> bool:
    return any(needle in value for value in values for needle in needles)


def _source_check_passed(inventory: Mapping[str, Any], check_id: str) -> bool:
    for record in _mapping_list(inventory.get("source_records")):
        if record.get("check_id") == check_id:
            return record.get("passed") is True
    return False


def _deep_truthy(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys and item is True:
                return True
            if _deep_truthy(item, keys):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_deep_truthy(item, keys) for item in value)
    return False


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _combined_sha256(paths: Sequence[Path]) -> str | None:
    h = hashlib.sha256()
    count = 0
    for path in paths:
        if not path.is_file():
            continue
        count += 1
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest() if count else None


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    import json

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "AXIS_TRACE_CONTRACT_SCHEMA",
    "PAYLOAD_GUARD_SCHEMA",
    "POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA",
    "PRACTICES",
    "PRACTICE_DAG_SCHEMA",
    "PRACTICE_ROW_SCHEMA",
    "SCHEMA",
    "SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA",
    "SOURCE_INVENTORY_SCHEMA",
    "STAGE_DAG_SCHEMA",
    "TELEMETRY_CONTRACT_SCHEMA",
    "PR95DistortionPracticesGuardError",
    "build_pr95_distortion_axis_trace_contract",
    "build_pr95_distortion_practices_payload_guard",
    "build_pr95_distortion_practices_row_guard",
    "build_pr95_distortion_source_inventory",
    "build_pr95_evaluate_scorer_domain_telemetry_contract",
    "build_pr95_posenet_marginal_telemetry_contract",
    "build_pr95_scorer_atom_actuator_contract",
    "render_pr95_distortion_practices_markdown",
]
