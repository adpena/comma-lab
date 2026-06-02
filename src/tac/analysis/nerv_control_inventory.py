# SPDX-License-Identifier: MIT
"""Inventory exploitable NeRV-family controls and their scorer bindings.

This is a false-authority control map for HiNeRV/SNeRV/HNeRV/SR-NeRV/RNeRV
synergy work. It does not choose a contest candidate. It records which knobs
exist, which local surfaces should consume them, and which missing bindings
prevent the knobs from becoming score-lowering work.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.analysis.nerv_modelsize_ladder import (
    SCORER_ONLY_OBJECTIVE_AUTHORITY,
    build_nerv_modelsize_ladder,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

NERV_CONTROL_INVENTORY_SCHEMA = "nerv_control_inventory.v1"


def build_nerv_control_inventory(
    *,
    focus_families: Iterable[str] = ("hi_nerv", "snerv"),
    repo_root: str | Path | None = None,
    hinerv_archive_size_ladder_report: Mapping[str, Any] | None = None,
    hinerv_archive_ladder_waterfill_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable map of NeRV controls and required bindings."""

    focus = tuple(
        dict.fromkeys(str(item).strip() for item in focus_families if str(item).strip())
    )
    controls = [
        row
        for row in _control_rows()
        if not focus or row["applies_to"] == "cross_stack" or row["applies_to"] in focus
    ]
    gaps = _binding_gaps(controls)
    status_counts = Counter(str(row["binding_status"]) for row in controls)
    return {
        "schema": NERV_CONTROL_INVENTORY_SCHEMA,
        "focus_families": list(focus),
        "authority": "false_authority_control_inventory_no_score_claim",
        "objective_authority": SCORER_ONLY_OBJECTIVE_AUTHORITY,
        "upstream_sources_checked": _upstream_sources(),
        "rate_constraint": _rate_constraint(),
        "runner_spend_rule": _runner_spend_rule(),
        "stack_transfer_matrix": _stack_transfer_matrix(),
        "local_binding_surfaces": _local_binding_surfaces(),
        "modelsize_ladder": build_nerv_modelsize_ladder(focus_families=focus),
        "measured_archive_size_ladders": _measured_archive_size_ladders(
            hinerv_archive_size_ladder_report=hinerv_archive_size_ladder_report,
            focus_families=focus,
        ),
        "decoder_weight_waterfill_reports": _decoder_weight_waterfill_reports(
            hinerv_archive_ladder_waterfill_report=(
                hinerv_archive_ladder_waterfill_report
            ),
            focus_families=focus,
        ),
        "control_rows": controls,
        "binding_gap_rows": gaps,
        "status_counts": dict(sorted(status_counts.items())),
        "recommended_next_work_orders": _recommended_work_orders(gaps),
        "runner_policy": _runner_policy(controls),
        "anti_patterns_guarded": _anti_patterns_guarded(),
        "implementation_sweep": build_nerv_design_implementation_sweep(
            repo_root=repo_root,
            focus_families=focus,
        ),
        **FALSE_AUTHORITY,
    }


def render_nerv_control_inventory_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing version of the control inventory."""

    lines = [
        "# NeRV control inventory",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        "",
        "## Spend Rule",
        "",
        str(report.get("runner_spend_rule", {}).get("rule")),
        "",
        "## Controls",
        "",
        "| control | applies to | binding status | missing binding count |",
        "|---|---|---|---|",
    ]
    for row in report.get("control_rows", []):
        lines.append(
            "| {control} | {applies} | {status} | {count} |".format(
                control=row["control_id"],
                applies=row["applies_to"],
                status=row["binding_status"],
                count=len(row.get("missing_bindings") or []),
            )
        )
    lines.extend(["", "## Recommended Work Orders", ""])
    for work_order in report.get("recommended_next_work_orders", []):
        lines.append(
            f"- `{work_order['work_order_id']}` from "
            f"`{work_order['source_gap']['control_id']}`"
        )
    sweep = report.get("implementation_sweep")
    if isinstance(sweep, Mapping):
        lines.extend(["", "## Implementation Sweep", ""])
        for row in sweep.get("stack_rows", []):
            lines.append(
                f"- `{row['family']}`: `{row['overall_status']}` "
                f"({len(row.get('blocking_gaps') or [])} blocking gaps)"
            )
    ladder = report.get("modelsize_ladder")
    if isinstance(ladder, Mapping):
        lines.extend(["", "## Model-Size Ladder", ""])
        for row in ladder.get("family_rows", []):
            lines.append(
                f"- `{row['family']}`: {len(row.get('ladder_rows') or [])} rows, "
                f"{len(row.get('marginal_gates') or [])} marginal gates"
            )
    measured_ladders = report.get("measured_archive_size_ladders")
    if isinstance(measured_ladders, Mapping):
        lines.extend(["", "## Measured Archive Size Ladders", ""])
        for family, ladder_row in measured_ladders.items():
            lines.append(
                f"- `{family}`: `{ladder_row.get('status')}` "
                f"({ladder_row.get('row_count', 0)} rows)"
            )
    lines.extend(["", "## Sources", ""])
    for source in report.get("upstream_sources_checked", []):
        lines.append(f"- [{source['id']}]({source['url']}): {source['control_signal']}")
    lines.append("")
    return "\n".join(lines)


def build_nerv_design_implementation_sweep(
    *,
    repo_root: str | Path | None,
    focus_families: Iterable[str] = ("hi_nerv", "snerv"),
) -> dict[str, Any]:
    """Audit HiNeRV/SNeRV implementation/design surfaces for false completeness."""

    focus = tuple(
        dict.fromkeys(str(item).strip() for item in focus_families if str(item).strip())
    )
    if repo_root is None:
        return {
            "schema": "nerv_design_implementation_sweep.v1",
            "status": "repo_root_not_supplied",
            "focus_families": list(focus),
            "stack_rows": [],
            "design_memo_index": {},
            "blockers": ["repo_root_not_supplied_for_implementation_sweep"],
            **FALSE_AUTHORITY,
        }
    root = Path(repo_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        return {
            "schema": "nerv_design_implementation_sweep.v1",
            "status": "repo_root_missing",
            "repo_root": root.as_posix(),
            "focus_families": list(focus),
            "stack_rows": [],
            "design_memo_index": {},
            "blockers": ["repo_root_missing_for_implementation_sweep"],
            **FALSE_AUTHORITY,
        }
    families = [
        family for family in ("hi_nerv", "snerv") if not focus or family in focus
    ]
    stack_rows = [_implementation_stack_row(root, family) for family in families]
    memo_index = {
        family: _memo_index(root, _memo_terms(family))
        for family in families
    }
    blockers = _unique(
        [
            blocker
            for row in stack_rows
            for blocker in row.get("blocking_gaps", [])
        ]
        + [
            "implementation_sweep_is_false_authority",
            "online_sources_are_research_context_not_score_authority",
        ]
    )
    return {
        "schema": "nerv_design_implementation_sweep.v1",
        "status": "implementation_sweep_completed_false_authority",
        "repo_root": root.as_posix(),
        "focus_families": families,
        "stack_rows": stack_rows,
        "design_memo_index": memo_index,
        "online_research_sources": _upstream_sources(),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _measured_archive_size_ladders(
    *,
    hinerv_archive_size_ladder_report: Mapping[str, Any] | None,
    focus_families: Iterable[str],
) -> dict[str, Any]:
    focus = {str(family) for family in focus_families}
    rows: dict[str, Any] = {}
    if hinerv_archive_size_ladder_report is not None and (
        not focus or "hi_nerv" in focus
    ):
        report = hinerv_archive_size_ladder_report
        rows["hi_nerv"] = {
            "schema": report.get("schema"),
            "status": "measured_archive_bytes_available_false_authority",
            "report_path": report.get("report_path"),
            "output_dir": report.get("output_dir"),
            "decoder_codec": report.get("decoder_codec"),
            "required_allocator_bindings": list(
                report.get("required_allocator_bindings") or ()
            ),
            "selection_rule": report.get("selection_rule"),
            "row_count": int(report.get("row_count", 0) or 0),
            "archive_rows": [
                {
                    "row_id": row.get("row_id"),
                    "archive_bytes": row.get("archive_bytes"),
                    "archive_sha256": row.get("archive_sha256"),
                    "archive_path": row.get("archive_path"),
                    "runtime_consumption_proof_ready": row.get(
                        "runtime_consumption_proof_ready"
                    ),
                    "blockers": list(row.get("blockers") or ()),
                }
                for row in report.get("archive_rows", ())
                if isinstance(row, Mapping)
            ],
            "marginal_archive_gates": list(
                report.get("marginal_archive_gates") or ()
            ),
            "blockers": list(report.get("blockers") or ()),
            **FALSE_AUTHORITY,
        }
    return rows


def _decoder_weight_waterfill_reports(
    *,
    hinerv_archive_ladder_waterfill_report: Mapping[str, Any] | None,
    focus_families: Iterable[str],
) -> dict[str, Any]:
    focus = {str(family) for family in focus_families}
    rows: dict[str, Any] = {}
    if hinerv_archive_ladder_waterfill_report is not None and (
        not focus or "hi_nerv" in focus
    ):
        report = hinerv_archive_ladder_waterfill_report
        rows["hi_nerv"] = {
            "schema": report.get("schema"),
            "status": "decoder_weight_waterfill_rows_available_false_authority",
            "report_path": report.get("report_path"),
            "row_count": int(report.get("row_count", 0) or 0),
            "section_value_row_count": len(report.get("section_value_rows") or ()),
            "full_video_coverage": bool(report.get("full_video_coverage")),
            "byte_price_plan": report.get("byte_price_plan"),
            "waterfill_rows": [
                {
                    "row_id": row.get("row_id"),
                    "archive_bytes": row.get("archive_bytes"),
                    "archive_sha256": row.get("archive_sha256"),
                    "state_npz_artifact_sha256": row.get("state_npz_artifact_sha256"),
                    "waterfill_summary": row.get("waterfill_summary"),
                    "blockers": list(row.get("blockers") or ()),
                }
                for row in report.get("rows", ())
                if isinstance(row, Mapping)
            ],
            "blockers": list(report.get("blockers") or ()),
            **FALSE_AUTHORITY,
        }
    return rows


def _control_rows() -> list[dict[str, Any]]:
    return [
        _row(
            "hi_nerv_hierarchical_capacity",
            "hi_nerv",
            upstream="HiNeRV-S/M/L config capacity; hierarchical positional encodings",
            local="latent_dim_coarse/mid/fine, embed_dim, decoder_channels, injection blocks",
            scorer="score-aware MLX full_main + cache quality gate + scorer prefilter",
            allocator=(
                "modelsize_budget_plan + nerv_decoder_weight_waterfill over "
                "measured decoder saliency"
            ),
            archive="export_hi_nerv_mlx_archive + decoder_codec",
            status="partially_wired_needs_measured_ladder",
            missing=[
                "measured_hi_nerv_modelsize_budget_ladder",
                "decoder_weight_saliency_replay_for_hi_nerv_archive_rows",
                "decoder_weight_saliency_into_trainer",
                "cache_quality_gate_required_before_profile_or_spend",
            ],
        ),
        _row(
            "hi_nerv_bitstream_quantization",
            "hi_nerv",
            upstream="HiNeRV --bitstream-q evaluates compressed bitstreams",
            local="decoder_codec int8/int4/int2 plus coder_aware_qat quant bits",
            scorer="MLX replay and section value profile",
            allocator="contest byte price from modelsize_budget_plan",
            archive="compact decoder codec sweep",
            status="partially_wired_needs_hi_nerv_codec_sweep_replay",
            missing=["hi_nerv_decoder_codec_sweep_full600_replay"],
        ),
        _row(
            "hi_nerv_sr_resolution_axis",
            "hi_nerv",
            upstream="SR-NeRV principle: encode low internal resolution and super-resolve",
            local="output_height/output_width currently 384x512 scorer size; SR mirror exists",
            scorer="sr_nerv_resolution_axis_mirror + MLX scorer cache quality",
            allocator="resolution dead-zone before per-pixel saliency spend",
            archive="do not store high-frequency output detail unless scorer-visible",
            status="design_knob_needs_trained_sr_receiver",
            missing=["trained_scorer_preserving_sr_receiver_for_hi_nerv"],
        ),
        _row(
            "sr_nerv_lowres_receiver_axis",
            "cross_stack",
            upstream="SR-NeRV uses low-detail INR reconstruction plus SR to improve embedding efficiency",
            local="sr_nerv_resolution_axis_mirror and scorer cache quality gate",
            scorer="contest scorer downsample mirror; reject SR detail that scorer cannot see",
            allocator="resolution-axis dead-zone before atom-level waterfilling",
            archive="receiver-generated SR, not stored high-frequency payload",
            status="design_knob_needs_receiver_closed_training",
            missing=["receiver_closed_scorer_preserving_sr_axis_for_hi_nerv_and_snerv"],
        ),
        _row(
            "hi_nerv_inverse_steg_decoder_weight_fit",
            "hi_nerv",
            upstream="HNeRV content amortizes into decoder weights",
            local="hinerv_latent_linf_allocation, score_exact_saliency, carrier_training_plan",
            scorer="P18 SegNet last-frame saliency + P19 PoseNet Fisher",
            allocator=(
                "nerv_decoder_weight_waterfill protects weights unless measured "
                "full-video saliency prices a cut"
            ),
            archive="QAT/coder-aware regularized trained decoder bytes",
            status="not_wired_into_real_trainer",
            missing=[
                "decoder_weight_vjp_or_saliency_proxy_in_hi_nerv_full_main",
                "decoder_weight_waterfill_plan_for_hi_nerv_full_main",
            ],
        ),
        _row(
            "snerv_frequency_split",
            "snerv",
            upstream="SNeRV DWT LF/HF split, HFR, MFU, temporal TUB",
            local="snerv_step_map_coder, SNeRV packet/runtime proofs, LF/HF advisories",
            scorer="score_exact_saliency + hprc_saliency_rd_allocation",
            allocator="G3 adjoint pushes pixel saliency into wavelet/LF/HF domains",
            archive="SNAR packet + compact LF residual grammar",
            status="partially_wired_cpu_advisory_mlx_missing",
            missing=["mlx_native_snerv_train_export", "receiver_closed_learned_hfr"],
        ),
        _row(
            "snerv_lf_modelsize_and_stepmap",
            "snerv",
            upstream="SNeRV controls enc_strds/dec_strds, fc_dim, num_blocks, emb_size",
            local="step-map waterfill, LF predictor profiles, decoder mode probes",
            scorer="cache quality gate + full-video scorer replay",
            allocator=(
                "modelsize_budget_plan + nerv_decoder_weight_waterfill for decoder "
                "weights + waterfilling over LF/HF atoms"
            ),
            archive="deterministic receiver-side LF generator or symbolic residual grammar",
            status="needs_representation_change",
            missing=[
                "snerv_measured_modelsize_ladder",
                "decoder_weight_waterfill_plan_for_snerv_receiver_rows",
                "learned_lf_generator_byte_collapse",
            ],
        ),
        _row(
            "snerv_pose_guarded_hf_restoration",
            "snerv",
            upstream="SNeRV HFR restores high-frequency detail from compact LF features",
            local="snerv_pose_guarded_decoder_gate, score-weighted decoder fit smokes",
            scorer="PoseNet Fisher over both frames and SegNet last-frame only",
            allocator="frame_0 pose-only vs frame_1 seg+pose asymmetry",
            archive="HF restoration should be receiver-generated, not stored as floats",
            status="advisory_only_needs_scorer_loop_training",
            missing=["snerv_scorer_loop_decoder_qat_full_video"],
        ),
        _row(
            "hnerv_modelsize_control",
            "cross_stack",
            upstream="HNeRV/NeRV use architecture size, pruning, quant_bit, model parameter bpp",
            local="PR95 HNeRV MLX + modelsize_budget_plan",
            scorer="PR95 control replay + MLX/PyTorch render parity",
            allocator="fixed contest byte price; spend size only if marginal distortion drops enough",
            archive="PR95-style byte-closed archive/runtime",
            status="control_baseline_available",
            missing=["feed_pr95_hnerv_measured_ladder_into_all_carrier_plans"],
        ),
        _row(
            "rnerv_config_optimizer",
            "cross_stack",
            upstream="RNeRV-style per-video configuration search",
            local="runner family configs, bounded runner, implementation readiness",
            scorer="timing smoke + cache gate + scorer replay gates",
            allocator="acquisition rule over modelsize, codec, SR, saliency controls",
            archive="only promote receiver-closed archive/runtime rows",
            status="planner_missing",
            missing=["runnable_rnerv_style_config_search_over_hi_nerv_snerv_controls"],
        ),
        _row(
            "ffnerv_flow_temporal_redundancy",
            "cross_stack",
            upstream="FFNeRV injects flow information and compact convolutions for temporal redundancy",
            local="pose/flow side-channel probes, RAFT/ego-motion priors, scorer-visible motion groups",
            scorer="PoseNet pair sensitivity plus SegNet last-frame sensitivity",
            allocator="only spend flow/support tokens when byte-priced score drop pays",
            archive="flow/support must be receiver-generated, tiny-tokenized, or entropy-coded",
            status="not_wired_into_compact_carrier_training",
            missing=["byte_priced_flow_or_pose_support_tokens_for_hi_nerv_snerv"],
        ),
        _row(
            "nervplusplus_decoder_efficiency_blocks",
            "cross_stack",
            upstream="NeRV++ improves decoder capacity with separable residual blocks and skip layers",
            local="decoder architecture candidates for HiNeRV/SNeRV MLX trainers",
            scorer="cache gate plus modelsize_budget_plan before full-video training",
            allocator="architecture changes must beat byte price after quantization and pruning",
            archive="decoder block bytes must pass intN/coder-aware archive section proof",
            status="architecture_candidate_needs_rate_priced_ladder",
            missing=["rate_priced_nervplusplus_block_ablation_for_hi_nerv_snerv"],
        ),
        _row(
            "vq_c3_cool_chic_latent_codebook",
            "cross_stack",
            upstream=(
                "VQ-NeRV, C3, Cool-Chic, and NVRC expose discrete latent grids, "
                "codebooks, entropy models, and learned quantization controls"
            ),
            local="PACT-NeRV/VQ section profiler, HPRC packet spine, compact decoder codec sweeps",
            scorer="full-video MLX section value by decoder/codebook/index/residual",
            allocator="codebook and index bytes survive only if value-per-byte beats contest byte price",
            archive="all codebooks, entropy models, indices, and synthesis weights inside archive.zip",
            status="partially_wired_needs_full_video_section_pricing",
            missing=[
                "codebook_entropy_model_bytes_charged_in_spine",
                "full_video_section_value_for_vq_codebook_indices",
                "demote_codebook_sections_without_positive_total_value",
            ],
        ),
        _row(
            "inverse_steganalysis_saliency_stack",
            "cross_stack",
            upstream="detector-aware allocation rather than perceptual allocation",
            local=(
                "score_exact_saliency, inverse_steganalysis_linf_vs_l2_gate, "
                "joint_p18_p19_waterfill, segnet_boundary_marginals"
            ),
            scorer="SegNet boundary/logit margin + PoseNet Fisher + exact byte price",
            allocator="waterfill sparse atoms, zero dead zones, protect scorer-visible coefficients",
            archive="int8/fp16 only for protected atoms; int4/int2/zero/RLE elsewhere",
            status="available_needs_carrier_domain_binding",
            missing=["push_saliency_into_hi_nerv_weight_groups_and_snerv_wavelet_groups"],
        ),
        _row(
            "master_gradient_xray_stack",
            "cross_stack",
            upstream="none; repo-native diagnostic score-response tensor",
            local="master_gradient, xray tools, Venn composition, sensitivity maps",
            scorer="component deltas and CPU/CUDA drift by pair/frame/byte group",
            allocator="rank controls by measured marginal score response",
            archive="operator plans must include grammar-aware packet proof",
            status="available_needs_nerv_control_consumer",
            missing=["nerv_control_inventory_consumer_for_master_gradient_rows"],
        ),
        _row(
            "full_video_vjp_master_gradient_authority",
            "cross_stack",
            upstream="exact full-video constrained variational solve, not stochastic minibatch promotion",
            local="master_gradient anchors, MLX scorer port, joint P18/P19 waterfill",
            scorer="all 600 pairs reduce into one SegNet/PoseNet scalar action before any update",
            allocator="bundle/trust-region and hard archive projection decide accepted mutations",
            archive="gradient proposals are advisory until receiver replay proves changed bytes",
            status="available_needs_exact_reduced_bundle_consumer",
            missing=[
                "full_video_vjp_bundle_as_budget_spend_prerequisite",
                "gradient_bundle_to_modelsize_and_section_allocator",
                "hard_archive_projection_line_search_for_compact_carriers",
            ],
        ),
        _row(
            "bitmask_and_zero_packing",
            "cross_stack",
            upstream="standard entropy coding and sparse neural compression practice",
            local="bitmask, packed-zero, intN codec sweeps, section value profiling",
            scorer="section neutralization replay and cache quality gates",
            allocator="protect high-value atoms; RLE/zero low-value atoms",
            archive="compact decoder codec sweep + arithmetic/range/ANS candidates",
            status="available_needs_family_specific_packet_layout",
            missing=["hi_nerv_snerv_grouped_intN_zero_run_packet_layout"],
        ),
        _row(
            "receiver_exact_custody_gate",
            "cross_stack",
            upstream="contest archive.zip plus deterministic inflate.sh contract",
            local="archive_candidate contract, paired auth-eval, receiver proof tools",
            scorer="contest auth eval sees only archive bytes and inflated frames",
            allocator="exact budget cannot be spent without archive/runtime/content-tree custody",
            archive="single charged packet spine with hashes, argv, env, cleanup, and false-authority flags",
            status="available_must_remain_hard_gate",
            missing=[
                "all_compact_carrier_emitters_on_shared_archive_bound_contract",
                "bounded_runner_refuses_non_contract_candidate_paths",
            ],
        ),
    ]


def _implementation_stack_row(root: Path, family: str) -> dict[str, Any]:
    specs = _implementation_specs(family)
    category_rows = [_implementation_category_row(root, family, spec) for spec in specs]
    official_feature_rows = _official_feature_rows(family)
    blocking_gaps = _unique(
        [
            gap
            for row in category_rows
            for gap in row.get("blocking_gaps", [])
        ]
        + [
            row["gap_id"]
            for row in official_feature_rows
            if row["local_binding_status"] != "implemented_or_receiver_proven"
        ]
    )
    status = (
        "local_implementation_has_blocking_gaps_no_method_negative"
        if blocking_gaps
        else "local_implementation_ready_for_full_video_value_pricing"
    )
    return {
        "family": family,
        "overall_status": status,
        "category_rows": category_rows,
        "official_feature_rows": official_feature_rows,
        "blocking_gaps": blocking_gaps,
        "source_fidelity_rule": (
            "bad scores from non-OSS-faithful adapters are config or wiring "
            "signals until official feature parity, tiny forward parity, and "
            "receiver byte grammar pass"
        ),
        "score_lowering_rule": (
            "after source-fidelity blockers close, train with scorer-aware loss "
            "and admit bytes only through measured value-per-byte"
        ),
        **FALSE_AUTHORITY,
    }


def _implementation_category_row(
    root: Path,
    family: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    files = [_presence(root, path) for path in spec["required_files"]]
    missing_files = [row["path"] for row in files if not row["present"]]
    marker_rows = _marker_rows(root, spec.get("marker_files", ()))
    marker_hits = [
        hit
        for row in marker_rows
        for hit in row.get("hits", [])
        if hit.get("severity") in {"blocking", "warning"}
    ]
    blocking_gaps = list(spec.get("intrinsic_gaps", ()))
    blocking_gaps.extend(
        f"{family}_{spec['category']}_missing_file:{path}" for path in missing_files
    )
    blocking_gaps.extend(hit["gap_id"] for hit in marker_hits if hit["severity"] == "blocking")
    status = _category_status(spec, missing_files=missing_files, marker_hits=marker_hits)
    return {
        "family": family,
        "category": spec["category"],
        "status": status,
        "required_files": files,
        "marker_rows": marker_rows,
        "missing_files": missing_files,
        "blocking_gaps": _unique(blocking_gaps),
        "next_action": spec["next_action"],
        "oss_or_design_reference": spec["reference"],
        **FALSE_AUTHORITY,
    }


def _implementation_specs(family: str) -> list[dict[str, Any]]:
    common_exact = {
        "category": "receiver_exact_custody",
        "required_files": [
            "src/tac/substrates/hprc/archive_candidate.py",
            "src/tac/submission_packet/archive_grammar.py",
            "src/tac/submission_packet/paired_auth_eval.py",
        ],
        "marker_files": [],
        "reference": "AGENTS.md archive-custody and exact-axis non-negotiables",
        "intrinsic_gaps": [
            f"{family}_full600_receiver_proven_candidate_missing",
            f"{family}_same_axis_pr95_control_missing",
        ],
        "next_action": (
            "emit archive-bound contract rows and receiver proof for every "
            "full-coverage survivor before exact gating"
        ),
    }
    if family == "hi_nerv":
        return [
            {
                "category": "oss_source_fidelity",
                "required_files": [
                    "src/tac/substrates/hi_nerv/architecture.py",
                    "src/tac/substrates/hi_nerv/mlx_renderer.py",
                    "src/tac/substrates/hi_nerv/archive.py",
                    "src/tac/substrates/hi_nerv/inflate.py",
                    "src/tac/analysis/nerv_top_priority_stack_seam.py",
                ],
                "marker_files": [
                    "src/tac/substrates/hi_nerv/mlx_renderer.py",
                    "tools/run_compact_renderer_mlx_spine_runner.py",
                ],
                "reference": "HiNeRV official hierarchy, patch/frame, pruning, quantization, bitstream-q",
                "intrinsic_gaps": [
                    "hi_nerv_official_symbol_parity_map_missing",
                    "hi_nerv_tiny_forward_parity_against_oss_missing",
                    "hi_nerv_bitstream_q_receiver_roundtrip_missing",
                ],
                "next_action": (
                    "build official-symbol parity map and tiny forward parity "
                    "before interpreting bad local scores as method evidence"
                ),
            },
            {
                "category": "score_aware_training",
                "required_files": [
                    "src/tac/substrates/hi_nerv/score_aware_loss.py",
                    "src/tac/substrates/_shared/mlx_score_aware/carrier_training_plan.py",
                    "src/tac/analysis/hinerv_latent_linf_allocation.py",
                    "tools/run_compact_renderer_mlx_spine_runner.py",
                ],
                "marker_files": [
                    "tools/run_compact_renderer_mlx_spine_runner.py",
                    "src/tac/substrates/hi_nerv/score_aware_loss.py",
                ],
                "reference": "PR95 stage/optimizer discipline plus P18/P19 score-aware decoder-weight fitting",
                "intrinsic_gaps": [
                    "hi_nerv_long_real_teacher_scoreaware_training_missing",
                    "hi_nerv_decoder_weight_saliency_not_bound_to_full_main",
                ],
                "next_action": (
                    "make long HiNeRV runs use real scorer teachers, cache "
                    "quality gate, modelsize plan, and coder-aware QAT by default"
                ),
            },
            {
                "category": "modelsize_and_codec",
                "required_files": [
                    "src/tac/substrates/_shared/mlx_score_aware/modelsize_budget_plan.py",
                    "src/tac/analysis/nerv_decoder_weight_waterfill.py",
                    "tools/build_nerv_decoder_weight_waterfill_plan.py",
                    "src/tac/substrates/hi_nerv/archive.py",
                    "src/tac/substrates/hi_nerv/archive_candidate.py",
                ],
                "marker_files": ["src/tac/substrates/hi_nerv/archive.py"],
                "reference": "HNeRV modelsize and HiNeRV pruning/quantization codec pipeline",
                "intrinsic_gaps": [
                    "hi_nerv_measured_modelsize_budget_ladder_missing",
                    "hi_nerv_decoder_weight_saliency_replay_missing",
                    "hi_nerv_grouped_intN_zero_run_packet_layout_missing",
                ],
                "next_action": (
                    "sweep modelsize/width/latent/codec ladders under hard byte "
                    "ceilings, then build decoder-weight waterfill plans from "
                    "full-video saliency before selecting by byte price"
                ),
            },
            common_exact,
        ]
    if family == "snerv":
        return [
            {
                "category": "oss_source_fidelity",
                "required_files": [
                    "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/dwt.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/archive.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/inflate.py",
                    "src/tac/analysis/nerv_top_priority_stack_seam.py",
                ],
                "marker_files": [
                    "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
                ],
                "reference": "SNeRV official DWT LF/HF, MFU, HFR, SNeRV-T temporal extension",
                "intrinsic_gaps": [
                    "snerv_official_symbol_parity_map_missing",
                    "snerv_tiny_forward_parity_against_oss_missing",
                    "snerv_quantized_checkpoint_payload_replay_missing",
                ],
                "next_action": (
                    "bind official MFU/HFR/TUB features or explicitly block them "
                    "before treating local inverse-steg carrier as source-faithful"
                ),
            },
            {
                "category": "score_aware_training",
                "required_files": [
                    "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
                    "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
                    "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
                    "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
                ],
                "marker_files": [
                    "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
                    "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
                ],
                "reference": "SNeRV HFR/MFU plus scorer-loop QAT and P18/P19 pose guard",
                "intrinsic_gaps": [
                    "snerv_scorer_loop_decoder_qat_full_video_missing",
                    "snerv_mlx_native_train_export_missing",
                ],
                "next_action": (
                    "replace bounded smoke-only QAT with full-video MLX-native "
                    "trained HFR/MFU export before promotion work"
                ),
            },
            {
                "category": "frequency_and_packet_codec",
                "required_files": [
                    "src/tac/analysis/snerv_step_map_coder.py",
                    "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/allocation.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py",
                ],
                "marker_files": [
                    "src/tac/analysis/snerv_step_map_coder.py",
                    "src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py",
                ],
                "reference": "SNeRV spectral split plus waterfilled intN/zero/RLE packet grammar",
                "intrinsic_gaps": [
                    "snerv_measured_modelsize_ladder_missing",
                    "snerv_receiver_closed_learned_hfr_missing",
                ],
                "next_action": (
                    "make mixed LF/HF packet modes receiver-decoded and priced by "
                    "full-video section value before another scalar sweep"
                ),
            },
            common_exact,
        ]
    return []


def _official_feature_rows(family: str) -> list[dict[str, Any]]:
    if family == "hi_nerv":
        rows = [
            (
                "official_hierarchical_feature_grid_encoding",
                "partial_local_three_scale_latent_pyramid_not_official_feature_grid",
            ),
            (
                "official_patch_mode_frame_mode_equivalence",
                "missing_patch_frame_equivalence_proof",
            ),
            (
                "official_fast_3d_hierarchical_upsampling",
                "missing_official_3d_upsampling_parity",
            ),
            (
                "official_config_family_size_sweeps",
                "missing_measured_config_family_ladder",
            ),
            (
                "official_pruning_quant_noise_quant_ste_stack",
                "missing_prune_quant_noise_qste_bitstream_roundtrip",
            ),
            (
                "official_torchac_or_equivalent_integer_bitstream_codec",
                "missing_integer_bitstream_q_roundtrip",
            ),
        ]
    elif family == "snerv":
        rows = [
            (
                "official_encoder_decoder_stride_stack",
                "missing_official_stride_stack_parity",
            ),
            (
                "official_haar_dwt_idwt_low_high_frequency_reconstruction",
                "partial_native_dwt_present_hfr_receiver_not_source_faithful",
            ),
            (
                "official_multi_resolution_fusion_blocks",
                "missing_mfu_blocks",
            ),
            (
                "official_high_frequency_restoration_heads",
                "missing_official_hfr_heads",
            ),
            (
                "official_temporal_extension_snerv_t",
                "missing_snerv_t_temporal_path_or_no_go",
            ),
            (
                "official_modelsize_fc_dim_budget_binding",
                "missing_measured_fc_dim_modelsize_ladder",
            ),
            (
                "official_quant_model_embedding_payload_accounting",
                "missing_quant_payload_receiver_replay",
            ),
        ]
    else:
        rows = []
    return [
        {
            "feature_id": feature,
            "gap_id": f"{family}_{gap}",
            "local_binding_status": "missing_or_partial",
            "score_interpretation": (
                "bad local scores are implementation/config signals until this "
                "official feature is either source-faithfully implemented or "
                "explicitly blocked by a receiver-closed proof"
            ),
            **FALSE_AUTHORITY,
        }
        for feature, gap in rows
    ]


def _category_status(
    spec: Mapping[str, Any],
    *,
    missing_files: list[str],
    marker_hits: list[dict[str, Any]],
) -> str:
    if missing_files:
        return "missing_required_local_surface"
    if any(hit["severity"] == "blocking" for hit in marker_hits):
        return "local_surface_contains_blocking_fake_or_partial_markers"
    if spec.get("intrinsic_gaps"):
        return "present_but_incomplete_or_not_source_faithful"
    if marker_hits:
        return "present_with_review_warnings"
    return "present_no_known_blocker_in_sweep"


def _presence(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    return {
        "path": rel,
        "present": path.exists(),
        "kind": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
    }


def _marker_rows(root: Path, rels: Iterable[str]) -> list[dict[str, Any]]:
    return [_scan_markers(root, rel) for rel in rels]


def _scan_markers(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    if not path.is_file():
        return {"path": rel, "present": False, "hits": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    marker_specs = (
        (
            "NotImplementedError",
            "blocking",
            "notimplemented_marker_requires_explicit_blocker",
        ),
        (
            "placeholder",
            "blocking",
            "placeholder_marker_requires_implementation_or_blocker",
        ),
        (
            "scoreaware_trainer_pending",
            "blocking",
            "scoreaware_trainer_pending_requires_real_training_path",
        ),
        (
            "allow_mock_scorer_teacher",
            "blocking",
            "mock_scorer_teacher_escape_hatch_requires_false_authority_guard",
        ),
        ("smoke", "warning", "smoke_marker_not_promotion_authority"),
        ("proxy", "warning", "proxy_marker_requires_false_authority"),
        ("random_signed", "warning", "random_search_mode_requires_measured_successor"),
        ("default=0.0", "warning", "zero_default_requires_config_justification"),
    )
    hits = []
    for needle, severity, gap_id in marker_specs:
        count = text.count(needle)
        if count:
            hits.append(
                {
                    "needle": needle,
                    "count": count,
                    "severity": severity,
                    "gap_id": f"{rel}:{gap_id}",
                }
            )
    return {"path": rel, "present": True, "hits": hits}


def _memo_terms(family: str) -> tuple[str, ...]:
    if family == "hi_nerv":
        return ("hinerv", "hi_nerv", "hnerv", "sr_nerv", "rnerv", "modelsize")
    if family == "snerv":
        return ("snerv", "SNeRV", "rnerv", "sr_nerv", "modelsize")
    return (family,)


def _memo_index(root: Path, terms: Iterable[str]) -> dict[str, Any]:
    research = root / ".omx" / "research"
    if not research.is_dir():
        return {"memo_count": 0, "memos": [], "blocker": "research_dir_missing"}
    lowered_terms = tuple(term.lower() for term in terms)
    matches = []
    for path in sorted(research.glob("*")):
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        if any(term in lower for term in lowered_terms):
            matches.append(path.relative_to(root).as_posix())
    rows = [_memo_row(root, rel) for rel in matches]
    return {
        "memo_count": len(rows),
        "memos": matches,
        "memo_rows": rows,
        "memo_paths_are_complete": True,
        "truncated": False,
    }


def _memo_row(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    data = path.read_bytes()
    return {
        "path": rel,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _rate_constraint() -> dict[str, Any]:
    return {
        "constraint_id": "fixed_contest_byte_price",
        "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
        "implication": (
            "large NeRV-family capacity is admissible only when measured non-rate "
            "improvement per added archive byte exceeds the fixed contest byte price"
        ),
        "required_large_model_escape_hatches": [
            "quantization_aware_training",
            "coder_aware_regularization",
            "weight_or_wavelet_group_ablation",
            "waterfilled_int8_int4_int2_zero_allocation",
            "packed_zero_run_or_entropy_coded_packet_layout",
            "receiver_generated_low_resolution_or_symbolic_components",
        ],
        **FALSE_AUTHORITY,
    }


def _runner_spend_rule() -> dict[str, Any]:
    return {
        "schema": "nerv_control_runner_spend_rule.v1",
        "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
        "rule": (
            "admit a control only when measured delta_nonrate_score + "
            "contest_byte_price_score_per_byte * delta_archive_bytes < 0 on "
            "the matching evidence axis; MLX rows can route training budget "
            "but never claim score or promotion"
        ),
        "exact_axis_required_for_score_claim": True,
        **FALSE_AUTHORITY,
    }


def _stack_transfer_matrix() -> list[dict[str, Any]]:
    return [
        {
            "from_family": "PR95/HNeRV",
            "to_families": ["hi_nerv", "snerv"],
            "transfers": [
                "modelsize-budgeted decoder bytes",
                "PR95-style staged optimizer/curriculum",
                "quantized decoder and latent packet grammar",
                "same-axis control replay discipline",
            ],
            "guard": "source/runtime parity before beat claims or method negatives",
        },
        {
            "from_family": "SR-NeRV",
            "to_families": ["hi_nerv", "snerv"],
            "transfers": [
                "low internal resolution",
                "trained scorer-preserving upsample",
                "resolution-axis byte dead-zone",
            ],
            "guard": "SR bytes are charged; interpolation-only probes stay advisory",
        },
        {
            "from_family": "RNeRV",
            "to_families": ["hi_nerv", "snerv"],
            "transfers": [
                "per-video component search",
                "training-time versus size-quality accounting",
                "carrier configuration as an acquisition surface",
            ],
            "guard": "configuration search must produce trained archive-bound rows",
        },
        {
            "from_family": "VQ-NeRV/C3/Cool-Chic/NVRC",
            "to_families": ["hi_nerv", "snerv"],
            "transfers": [
                "discrete latent/codebook grammar",
                "hierarchical latent grids",
                "learned entropy model",
                "end-to-end quantization/rate loss",
            ],
            "guard": "codebook, entropy-model, index, and synthesis bytes are all charged",
        },
    ]


def _row(
    control_id: str,
    applies_to: str,
    *,
    upstream: str,
    local: str,
    scorer: str,
    allocator: str,
    archive: str,
    status: str,
    missing: list[str],
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "applies_to": applies_to,
        "upstream_control": upstream,
        "local_control_surface": local,
        "scorer_binding": scorer,
        "allocator_binding": allocator,
        "archive_runtime_binding": archive,
        "binding_status": status,
        "missing_bindings": missing,
        **FALSE_AUTHORITY,
    }


def _binding_gaps(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for row in controls:
        for missing in row["missing_bindings"]:
            gaps.append(
                {
                    "gap_id": missing,
                    "control_id": row["control_id"],
                    "applies_to": row["applies_to"],
                    "blocks_score_lowering": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return gaps


def _recommended_work_orders(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = [
        "cache_quality_gate_required_before_profile_or_spend",
        "measured_hi_nerv_modelsize_budget_ladder",
        "decoder_weight_saliency_replay_for_hi_nerv_archive_rows",
        "decoder_weight_vjp_or_saliency_proxy_in_hi_nerv_full_main",
        "decoder_weight_waterfill_plan_for_hi_nerv_full_main",
        "mlx_native_snerv_train_export",
        "snerv_measured_modelsize_ladder",
        "decoder_weight_waterfill_plan_for_snerv_receiver_rows",
        "full_video_section_value_for_vq_codebook_indices",
        "full_video_vjp_bundle_as_budget_spend_prerequisite",
        "push_saliency_into_hi_nerv_weight_groups_and_snerv_wavelet_groups",
        "runnable_rnerv_style_config_search_over_hi_nerv_snerv_controls",
        "all_compact_carrier_emitters_on_shared_archive_bound_contract",
    ]
    gap_index = {str(row["gap_id"]): row for row in gaps}
    return [
        {
            "work_order_id": gap_id,
            "source_gap": gap_index[gap_id],
            "authority": "implementation_or_measurement_only_no_score_claim",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        for gap_id in priority
        if gap_id in gap_index
    ]


def _local_binding_surfaces() -> dict[str, list[str]]:
    return {
        "scorer_and_saliency": [
            "src/tac/analysis/score_exact_saliency.py",
            "src/tac/analysis/hprc_saliency_rd_allocation.py",
            "src/tac/analysis/segnet_boundary_marginals.py",
            "src/tac/optimization/joint_p18_p19_waterfill.py",
            "src/tac/analysis/inverse_steganalysis_linf_vs_l2_gate.py",
        ],
        "xray_and_master_gradient": [
            "src/tac/master_gradient.py",
            "src/tac/master_gradient_consumers.py",
            "src/tac/master_gradient_mlx_pipeline.py",
            "tools/build_pair_frame_scorer_geometry_lattice.py",
            "tools/cpu_cuda_xray_loader_drift.py",
            "tools/xray_hardpair_hitlist.py",
        ],
        "nerv_carriers": [
            "tools/run_compact_renderer_mlx_spine_runner.py",
            "src/tac/substrates/hi_nerv",
            "src/tac/analysis/snerv_step_map_coder.py",
            "src/tac/analysis/sr_nerv_resolution_axis_mirror.py",
            "src/tac/local_acceleration/pr95_hnerv_mlx.py",
        ],
        "section_value_and_codebook": [
            "tools/profile_pact_nerv_selector_v3_mlx_section_value.py",
            "src/tac/substrates/pact_nerv_vq",
            "src/tac/analysis/hnerv_packet_sections.py",
            "src/tac/analysis/scorer_conditional_mdl.py",
            "src/tac/analysis/nerv_decoder_weight_waterfill.py",
            "src/tac/analysis/hinerv_archive_ladder_waterfill.py",
            "tools/build_nerv_decoder_weight_waterfill_plan.py",
            "tools/build_hinerv_archive_ladder_waterfill.py",
        ],
        "receiver_and_exact_custody": [
            "src/tac/substrates/hprc/archive_candidate.py",
            "src/tac/submission_packet/archive_grammar.py",
            "src/tac/submission_packet/paired_auth_eval.py",
            "tools/run_contest_oracle_batch.py",
        ],
        "newly_required_gates": [
            "src/tac/analysis/mlx_cache_quality_gate.py",
            "src/tac/substrates/_shared/mlx_score_aware/modelsize_budget_plan.py",
        ],
    }


def _upstream_sources() -> list[dict[str, str]]:
    return [
        {
            "id": "hnerv_official",
            "url": "https://github.com/haochen-rye/HNeRV",
            "control_signal": "architecture size, pruning, quantization, model bpp",
        },
        {
            "id": "hinerv_official",
            "url": "https://github.com/hmkx/HiNeRV",
            "control_signal": "HiNeRV S/M/L configs, patch size, bitstream-q",
        },
        {
            "id": "hinerv_paper",
            "url": "https://arxiv.org/abs/2306.09818",
            "control_signal": "hierarchical encodings plus pruning and quantization pipeline",
        },
        {
            "id": "snerv_official",
            "url": "https://github.com/qwertja/SNeRV",
            "control_signal": "DWT LF/HF, enc/dec strides, fc_dim, emb_size, temporal extension",
        },
        {
            "id": "snerv_paper",
            "url": "https://arxiv.org/abs/2501.01681",
            "control_signal": "spectral split, HFR/MFU/TUB controls",
        },
        {
            "id": "sr_nerv_paper",
            "url": "https://arxiv.org/abs/2505.00046",
            "control_signal": "low-detail INR representation plus super-resolution",
        },
        {
            "id": "vq_nerv_paper",
            "url": "https://arxiv.org/abs/2403.12401",
            "control_signal": "shallow residual/inter-frame VQ codebook controls",
        },
        {
            "id": "rnerv_paper",
            "url": "https://arxiv.org/abs/2506.24127",
            "control_signal": "per-video design and training search over NeRV components",
        },
        {
            "id": "ffnerv_paper",
            "url": "https://arxiv.org/abs/2212.12294",
            "control_signal": "flow-guided temporal redundancy and compact convolutional architecture",
        },
        {
            "id": "nervplusplus_paper",
            "url": "https://arxiv.org/abs/2402.18305",
            "control_signal": "separable residual decoder blocks and skip-layer capacity",
        },
        {
            "id": "c3_paper",
            "url": "https://arxiv.org/abs/2312.02753",
            "control_signal": "overfitted hierarchical latents and entropy-model bytes",
        },
        {
            "id": "cool_chic_docs",
            "url": "https://orange-opensource.github.io/Cool-Chic/encoding/architecture.html",
            "control_signal": "hierarchical latent grids and autoregressive entropy model",
        },
        {
            "id": "nvrc_paper",
            "url": "https://arxiv.org/abs/2409.07414",
            "control_signal": "end-to-end neural representation quantization and entropy coding",
        },
    ]


def _runner_policy(controls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "nerv_control_runner_policy.v1",
        "bounded_runner_must_select_from_inventory_rows": True,
        "non_contract_candidates_allowed_only_as_migration_work": True,
        "score_authority": "contest CPU/CUDA exact replay only",
        "mlx_authority": "training and acquisition prefilter only",
        "required_controls_before_long_training_spend": [
            "hnerv_modelsize_control",
            "hi_nerv_hierarchical_capacity",
            "snerv_lf_modelsize_and_stepmap",
            "inverse_steganalysis_saliency_stack",
            "receiver_exact_custody_gate",
        ],
        "currently_unresolved_controls": [
            row["control_id"]
            for row in controls
            if row["binding_status"]
            not in {"control_baseline_available", "available_must_remain_hard_gate"}
        ],
        **FALSE_AUTHORITY,
    }


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _anti_patterns_guarded() -> list[dict[str, str]]:
    return [
        {
            "anti_pattern": "cheap_unfit_carrier_overclaim",
            "guard": "modelsize capacity is useful only after scorer-faithful full-video fit and value-per-byte replay",
        },
        {
            "anti_pattern": "visual_quality_objective",
            "guard": "controls are priced by SegNet/PoseNet/rate, not human visual fidelity",
        },
        {
            "anti_pattern": "residual_sidecar_bloat",
            "guard": "residual/codebook sections survive only when delta_nonrate plus rate cost is negative",
        },
        {
            "anti_pattern": "proxy_or_mlx_score_promotion",
            "guard": "FALSE_AUTHORITY is attached until receiver custody and exact-axis replay pass",
        },
        {
            "anti_pattern": "oss_drift_or_fake_adapter",
            "guard": "HiNeRV/SNeRV method negatives require OSS feature parity and tiny forward parity first",
        },
        {
            "anti_pattern": "orphaned_research_signal",
            "guard": "every research source maps to a local binding surface or explicit missing binding",
        },
    ]


__all__ = [
    "NERV_CONTROL_INVENTORY_SCHEMA",
    "build_nerv_control_inventory",
    "build_nerv_design_implementation_sweep",
    "render_nerv_control_inventory_markdown",
]
