# SPDX-License-Identifier: MIT
"""Bounded-runner contract for compact HPRC representation spines."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.campaign import HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA
from tac.substrates.hprc.mlx_prefilter_coverage import (
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
    mlx_profile_full_video_scope,
    mlx_profile_has_full_video_coverage,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT
from tac.substrates.hprc.spine_acquisition import HPRC_SPINE_ACQUISITION_REPORT_SCHEMA

HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA = "hprc_spine_bounded_runner_plan.v1"
HPRC_SPINE_COMPACT_BASE_SWEEP_ROW_SCHEMA = "hprc_spine_compact_base_sweep_row.v1"
HPRC_SPINE_SECTION_VALUE_ROW_SCHEMA = "hprc_spine_section_value_row.v1"
HPRC_SPINE_SECTION_VALUE_PROFILE_WORK_ORDER_SCHEMA = (
    "hprc_spine_section_value_profile_work_order.v1"
)
HPRC_SPINE_SECTION_CUT_MATERIALIZER_WORK_ORDER_SCHEMA = (
    "hprc_spine_section_cut_materializer_work_order.v1"
)
HPRC_SPINE_PROJECTION_GAP_REPAIR_WORK_ORDER_SCHEMA = (
    "hprc_spine_projection_gap_repair_work_order.v1"
)
HPRC_SPINE_COMPACT_DECODER_CODEC_SWEEP_WORK_ORDER_SCHEMA = (
    "hprc_spine_compact_decoder_codec_sweep_work_order.v1"
)
_PROJECTION_GAP_STRUCTURAL_SECTIONS = frozenset(
    ("decoder_qw", "codebooks_q", "latents_rc")
)
_SECTION_VALUE_PROFILERS: dict[str, dict[str, Any]] = {
    "pact_nerv_vq_pvq": {
        "tool": "tools/profile_pact_nerv_vq_mlx_section_value.py",
        "sections": ("decoder_qw", "codebooks_q", "selectors_rc", "residual_rc"),
    },
    "pact_nerv_selector_v3_psv3": {
        "tool": "tools/profile_pact_nerv_selector_v3_mlx_section_value.py",
        "sections": ("decoder_qw", "latents_rc", "selectors_rc", "residual_rc"),
    },
    "pact_nerv_selector_v4_psv4": {
        "tool": "tools/profile_pact_nerv_selector_v4_mlx_section_value.py",
        "sections": ("decoder_qw", "latents_rc", "selectors_rc", "residual_rc"),
    },
}
_SECTION_CUT_MATERIALIZERS: dict[str, dict[str, Any]] = {
    "pact_nerv_vq_pvq": {
        "tool": "tools/materialize_pact_nerv_vq_section_cut_candidate.py",
        "sections": ("decoder_qw", "codebooks_q", "selectors_rc"),
    },
    "pact_nerv_selector_v4_psv4": {
        "tool": "tools/materialize_pact_nerv_selector_v4_section_cut_candidate.py",
        "sections": ("latents_rc", "selectors_rc"),
    },
}
_SECTION_VALUE_PROFILE_STORAGE_WATERFALL = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
)
_COMPACT_DECODER_CODEC_SWEEP_PORTFOLIO = (
    "portfolio_auto",
    "int8_mixed",
    "int8_scale_bundled",
    "int4_mixed",
    "int4_scale_bundled",
    "int2_mixed",
    "int2_scale_bundled",
    "fp16_enveloped",
)


def build_spine_bounded_runner_plan(
    *,
    acquisition_report_path: str | Path,
    repo_root: str | Path = ".",
    upstream_dir: str | Path | None = None,
    mlx_profile_paths: list[str | Path] | tuple[str | Path, ...] = (),
    receiver_proof_report_paths: list[str | Path] | tuple[str | Path, ...] = (),
    exact_gate_report_paths: list[str | Path] | tuple[str | Path, ...] = (),
    hprc_queue_followup_report_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Build the one-contract runner plan for compact-base and residual work.

    The plan is deliberately fail-closed.  It may rank and route local work from
    MLX evidence, but all emitted rows still require receiver proof and contest
    CPU/CUDA exact authority before score or promotion claims.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    upstream_root = _resolve(upstream_dir or "upstream", base=root)
    acquisition_path = _resolve(acquisition_report_path, base=root)
    acquisition_report = _load_json_object(acquisition_path)
    if acquisition_report.get("schema") != HPRC_SPINE_ACQUISITION_REPORT_SCHEMA:
        raise ValueError(
            "acquisition_report_path must point to "
            f"{HPRC_SPINE_ACQUISITION_REPORT_SCHEMA}"
        )

    mlx_profiles = [_load_profile(path, root=root) for path in mlx_profile_paths]
    receiver_proofs = [
        _load_receiver_proof(path, root=root) for path in receiver_proof_report_paths
    ]
    exact_reports = [_load_exact_report(path, root=root) for path in exact_gate_report_paths]
    hprc_queue_followups = [
        _load_hprc_queue_followup(path, root=root)
        for path in hprc_queue_followup_report_paths
    ]
    section_evidence = _index_section_evidence(mlx_profiles)
    receiver_proof_index = _index_receiver_proofs(receiver_proofs)
    exact_index = _index_exact_reports(exact_reports)
    hprc_queue_followup_index = _index_hprc_queue_followups(hprc_queue_followups)
    compact_base_rows = [
        _compact_base_sweep_row(
            acquisition_row=row,
            ceiling_result=ceiling_result,
            receiver_proof_index=receiver_proof_index,
            exact_index=exact_index,
            hprc_queue_followup_index=hprc_queue_followup_index,
        )
        for row in _rows(acquisition_report, "rows")
        for ceiling_result in _rows(row, "ceiling_results")
    ]
    section_value_rows = [
        _section_value_row(
            acquisition_row=row,
            section=row_section,
            section_evidence=section_evidence,
        )
        for row in _rows(acquisition_report, "rows")
        for row_section in _rows(row, "section_rows")
    ]
    section_value_profile_work_orders = _section_value_profile_work_orders(
        acquisition_rows=_rows(acquisition_report, "rows"),
        section_value_rows=section_value_rows,
        repo_root=root,
        upstream_dir=upstream_root,
    )
    section_cut_materializer_work_orders = _section_cut_materializer_work_orders(
        acquisition_rows=_rows(acquisition_report, "rows"),
        section_value_rows=section_value_rows,
        repo_root=root,
    )
    projection_gap_repair_work_orders = _projection_gap_repair_work_orders(
        acquisition_rows=_rows(acquisition_report, "rows"),
        section_value_rows=section_value_rows,
        mlx_profiles=mlx_profiles,
        repo_root=root,
        upstream_dir=upstream_root,
    )
    residual_rows = [
        row
        for row in section_value_rows
        if row["section_name"] == "residual_rc"
        or "residual" in row["section_role"]
    ]
    residual_candidate_rows = _residual_candidate_value_rows(mlx_profiles)
    runner_rows = _choose_runner_rows(compact_base_rows=compact_base_rows)
    blockers = _plan_blockers(
        compact_base_rows=compact_base_rows,
        section_value_rows=section_value_rows,
        mlx_profiles=mlx_profiles,
        projection_gap_repair_work_orders=projection_gap_repair_work_orders,
    )
    return {
        "schema": HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "upstream_dir": upstream_root.as_posix(),
        "acquisition_report_path": acquisition_path.as_posix(),
        "acquisition_report_sha256": _sha256_file(acquisition_path),
        "hard_byte_ceilings": acquisition_report.get("hard_byte_ceilings", []),
        "mlx_profile_paths": [item["path"] for item in mlx_profiles],
        "receiver_proof_report_paths": [item["path"] for item in receiver_proofs],
        "exact_gate_report_paths": [item["path"] for item in exact_reports],
        "hprc_queue_followup_report_paths": [
            item["path"] for item in hprc_queue_followups
        ],
        "hprc_queue_followup_signal_rows": _hprc_queue_followup_signal_rows(
            hprc_queue_followups
        ),
        "compact_base_sweep_rows": compact_base_rows,
        "section_value_rows": section_value_rows,
        "section_value_profile_work_orders": section_value_profile_work_orders,
        "section_cut_materializer_work_orders": section_cut_materializer_work_orders,
        "projection_gap_repair_work_orders": projection_gap_repair_work_orders,
        "residual_token_admission_rows": [*residual_rows, *residual_candidate_rows],
        "selected_runner_rows": runner_rows,
        "runner_policy": {
            "schema": "hprc_spine_bounded_runner_policy.v1",
            "custody_surface": "hprc_representation_spine_projection_only",
            "base_sweep": (
                "every PR95/HNeRV/RNeRV/PACT-NeRV/VQ/selector candidate must "
                "emit the spine projection, then receiver proof, then full-video "
                "MLX scorer replay, then exact gate"
            ),
            "section_value_rule": (
                "section and residual bytes are admitted only when measured "
                "delta_nonrate + charged_rate_cost < 0; missing MLX evidence "
                "routes to replay; cut recommendations route to byte-closed "
                "materializers, never promotion"
            ),
            "projection_gap_rule": (
                "decoder/codebook/latent sections whose removal improves "
                "full-video MLX objective indicate archive-projection or "
                "training-capacity mismatch; route to direct-model-vs-archive "
                "repair before exact spend"
            ),
            "residual_rule": (
                "VQ/HPRC/Z8 residual tokens are blocked unless the measured "
                "non-rate improvement pays for their archive bytes"
            ),
            "authority": (
                "MLX rows are advisory; receiver proof and exact CPU/CUDA "
                "authority are mandatory before score or dispatch promotion"
            ),
            "stop_conditions": [
                "better_receiver_proven_archive_bound_candidate",
                "precise_exact_axis_blocker",
                "durable_negative_evidence_with_posterior_demotion",
            ],
        },
        "posterior_update_hooks": _posterior_update_hooks(
            compact_base_rows=compact_base_rows,
            section_value_rows=section_value_rows,
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def write_spine_bounded_runner_plan(
    *,
    output_path: str | Path,
    plan: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic HPRC spine bounded-runner plan."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _compact_base_sweep_row(
    *,
    acquisition_row: dict[str, Any],
    ceiling_result: dict[str, Any],
    receiver_proof_index: dict[str, dict[str, Any]],
    exact_index: dict[str, dict[str, Any]],
    hprc_queue_followup_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    family = str(acquisition_row.get("family") or "unknown")
    ceiling = int(ceiling_result.get("ceiling_bytes") or 0)
    coverage = acquisition_row.get("coverage") if isinstance(acquisition_row.get("coverage"), dict) else {}
    coverage_valid = coverage.get("valid_for_base_comparison") is True
    fits_ceiling = ceiling_result.get("fits") is True
    source = (
        acquisition_row.get("source_archive")
        if isinstance(acquisition_row.get("source_archive"), dict)
        else {}
    )
    receiver_proof = _lookup_receiver_proof(
        acquisition_row=acquisition_row,
        source=source,
        receiver_proof_index=receiver_proof_index,
    )
    receiver_proof_passed = (
        receiver_proof is not None
        and receiver_proof.get("runtime_consumption_proof_passed") is True
    )
    exact_report = exact_index.get(str(acquisition_row.get("projection_manifest_path") or ""))
    hprc_followup = _lookup_hprc_queue_followup(
        acquisition_row=acquisition_row,
        source=source,
        hprc_queue_followup_index=hprc_queue_followup_index,
    )
    hprc_followup_blockers = _hprc_queue_followup_demoting_blockers(hprc_followup)
    if not coverage_valid:
        action = "train_or_scale_to_full_coverage_emit_spine_then_receiver_proof"
        route_status = "blocked_until_full_video_coverage"
    elif not fits_ceiling:
        action = "shrink_or_recode_compact_base_under_ceiling"
        route_status = "queued_for_compact_base_byte_sweep"
    else:
        action = "receiver_proof_then_full_video_mlx_replay_then_exact_gate"
        route_status = "queued_for_receiver_proof_and_replay"
    if exact_report is not None and exact_report.get("ready_for_exact_eval_dispatch") is True:
        route_status = "exact_dispatchable_after_lane_claim"
    if hprc_followup_blockers:
        action = (
            "route_to_native_pose_geometry_or_predictive_hprc_redesign_"
            "before_cpu_replay"
        )
        route_status = "demoted_by_hprc_queue_followup"
    return {
        "schema": HPRC_SPINE_COMPACT_BASE_SWEEP_ROW_SCHEMA,
        "runner_row_id": f"{family}:{ceiling}",
        "family": family,
        "projection_manifest_path": acquisition_row.get("projection_manifest_path"),
        "effective_archive_bytes": acquisition_row.get("effective_archive_bytes"),
        "effective_rate_term": acquisition_row.get("effective_rate_term"),
        "ceiling_bytes": ceiling,
        "fits_ceiling": fits_ceiling,
        "excess_bytes": ceiling_result.get("excess_bytes"),
        "slack_bytes": ceiling_result.get("slack_bytes"),
        "coverage_valid_for_base_comparison": coverage_valid,
        "declared_pairs": coverage.get("declared_pairs"),
        "required_pairs": coverage.get("required_pairs", CONTEST_PAIR_COUNT),
        "action": action,
        "route_status": route_status,
        "required_spine_projection": True,
        "requires_receiver_proof": True,
        "receiver_proof_observed": receiver_proof is not None,
        "receiver_proof_passed": receiver_proof_passed,
        "receiver_proof_summary": _receiver_proof_summary(receiver_proof),
        "requires_full_video_mlx_replay": True,
        "requires_exact_gate": True,
        "exact_gate_observed": exact_report is not None,
        "exact_gate_summary": None if exact_report is None else exact_report.get("exact_axis_gate"),
        "hprc_queue_followup_observed": hprc_followup is not None,
        "hprc_queue_followup_summary": _hprc_queue_followup_summary(hprc_followup),
        "requires_architecture_redesign_before_replay": bool(hprc_followup_blockers),
        "blockers": _dedupe(
            [
                *([] if coverage_valid else ["declared_pair_coverage_below_full_video"]),
                *([] if fits_ceiling else ["candidate_exceeds_hard_byte_ceiling"]),
                *hprc_followup_blockers,
                *(
                    []
                    if receiver_proof_passed
                    else [
                        "receiver_proof_failed"
                        if receiver_proof is not None
                        else "receiver_proof_not_attached"
                    ]
                ),
                *(
                    []
                    if exact_report is not None
                    else ["exact_gate_not_yet_attached"]
                ),
                "contest_cpu_cuda_exact_eval_not_executed",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _section_value_row(
    *,
    acquisition_row: dict[str, Any],
    section: dict[str, Any],
    section_evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    section_name = str(section.get("name") or "")
    section_role = str(section.get("role") or "")
    byte_count = int(section.get("bytes") or 0)
    rate_cost = contest_rate_term(byte_count)
    evidence_rows = section_evidence.get(section_name, [])
    evidence_rows = [
        row
        for row in evidence_rows
        if _section_evidence_matches_acquisition(
            evidence=row,
            acquisition_row=acquisition_row,
        )
    ]
    best_evidence = _best_section_evidence(evidence_rows)
    full_video_evidence_present = any(
        _section_evidence_has_full_video_scope(row) for row in evidence_rows
    )
    coverage = (
        acquisition_row.get("coverage")
        if isinstance(acquisition_row.get("coverage"), dict)
        else {}
    )
    coverage_valid = coverage.get("valid_for_base_comparison") is True
    projection_only_metadata = section_name in {
        "rdo_plan",
        "manifest_json",
        "receiver_state",
    }
    observed_removal_delta_nonrate = None
    observed_removal_delta_total = None
    observed_archive_bytes_removed = None
    measured_marginal_status = None
    section_spend_recommendation = "not_priced"
    if projection_only_metadata:
        admission_status = "projection_contract_metadata_not_candidate_runtime_spend"
        delta_nonrate = 0.0
        admission_delta = 0.0
        evidence_status = "metadata_contract_no_mlx_replay_required"
        requires_replay = False
        blockers = ["contest_cpu_cuda_exact_eval_not_executed"]
        section_spend_recommendation = "metadata_no_runtime_spend"
    elif not coverage_valid:
        admission_status = "blocked_until_full_video_coverage_before_section_pricing"
        delta_nonrate = None
        admission_delta = None
        evidence_status = "not_required_until_full_video_coverage"
        requires_replay = False
        blockers = [
            "declared_pair_coverage_below_full_video",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
        section_spend_recommendation = "wait_for_full_video_coverage"
    elif best_evidence is None:
        admission_status = "blocked_until_full_video_mlx_section_value_replay"
        delta_nonrate = None
        admission_delta = None
        evidence_status = "missing"
        requires_replay = True
        blockers = [
            "full_video_mlx_section_value_replay_missing",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
        section_spend_recommendation = "run_full_video_section_value_replay"
    elif not full_video_evidence_present:
        admission_status = "blocked_until_full_video_mlx_section_value_replay"
        delta_nonrate = None
        admission_delta = None
        evidence_status = "sampled_mlx_advisory_requires_full_video_replay"
        requires_replay = True
        blockers = [
            "full_video_mlx_section_value_replay_missing",
            "sampled_mlx_section_value_replay_not_budget_authority",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
        section_spend_recommendation = "rerun_as_full_video_section_value_replay"
    else:
        delta_nonrate = best_evidence["presence_delta_nonrate"]
        admission_delta = float(delta_nonrate) + rate_cost
        evidence_status = "measured_mlx_advisory"
        observed_removal_delta_nonrate = best_evidence.get(
            "observed_removal_delta_nonrate"
        )
        observed_removal_delta_total = best_evidence.get(
            "observed_removal_delta_total_mlx_advisory"
        )
        observed_archive_bytes_removed = best_evidence.get(
            "archive_bytes_removed_vs_baseline"
        )
        measured_marginal_status = best_evidence.get("marginal_status")
        if admission_delta < 0.0:
            admission_status = "admit_section_bytes_for_receiver_proof"
            section_spend_recommendation = (
                "protect_section_bytes_measured_value_exceeds_rate_price"
            )
        elif section_name == "residual_rc" or "residual" in section_role:
            admission_status = "demote_or_block_residual_tokens"
            section_spend_recommendation = (
                "demote_residual_bytes_measured_value_below_rate_price"
            )
        elif (
            observed_removal_delta_total is not None
            and float(observed_removal_delta_total) < 0.0
        ):
            admission_status = "cut_section_bytes_for_receiver_proof"
            section_spend_recommendation = (
                "cut_section_bytes_measured_removal_improves_objective"
            )
        else:
            admission_status = "protect_or_shrink_by_smaller_recode_only"
            section_spend_recommendation = (
                "recode_smaller_only_if_exact_replay_preserves_distortion"
            )
        requires_replay = False
        blockers = ["contest_cpu_cuda_exact_eval_not_executed"]
    return {
        "schema": HPRC_SPINE_SECTION_VALUE_ROW_SCHEMA,
        "family": acquisition_row.get("family"),
        "projection_manifest_path": acquisition_row.get("projection_manifest_path"),
        "section_name": section_name,
        "section_role": section_role,
        "section_bytes": byte_count,
        "rate_cost": rate_cost,
        "evidence_status": evidence_status,
        "measured_presence_delta_nonrate": delta_nonrate,
        "measured_removal_delta_nonrate": observed_removal_delta_nonrate,
        "measured_removal_delta_total_mlx_advisory": observed_removal_delta_total,
        "measured_archive_bytes_removed_vs_baseline": observed_archive_bytes_removed,
        "measured_marginal_status": measured_marginal_status,
        "admission_objective_delta": admission_delta,
        "admission_rule": "measured_delta_nonrate + rate_cost < 0",
        "admission_status": admission_status,
        "section_spend_recommendation": section_spend_recommendation,
        "evidence_rows": evidence_rows,
        "requires_receiver_proof": True,
        "requires_full_video_mlx_replay": requires_replay,
        "requires_exact_gate": True,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _section_value_profile_work_orders(
    *,
    acquisition_rows: list[dict[str, Any]],
    section_value_rows: list[dict[str, Any]],
    repo_root: Path,
    upstream_dir: Path,
) -> list[dict[str, Any]]:
    rows_by_projection: dict[str, list[dict[str, Any]]] = {}
    for row in section_value_rows:
        key = str(row.get("projection_manifest_path") or "")
        rows_by_projection.setdefault(key, []).append(row)

    work_orders: list[dict[str, Any]] = []
    for acquisition_row in acquisition_rows:
        projection = str(acquisition_row.get("projection_manifest_path") or "")
        missing_rows = [
            row
            for row in rows_by_projection.get(projection, [])
            if row.get("requires_full_video_mlx_replay") is True
        ]
        if not missing_rows:
            continue
        work_order = _section_value_profile_work_order(
            acquisition_row=acquisition_row,
            missing_rows=missing_rows,
            repo_root=repo_root,
            upstream_dir=upstream_dir,
        )
        if work_order is not None:
            work_orders.append(work_order)
    work_orders.sort(
        key=lambda row: (
            row["status"],
            str(row.get("source_payload_kind") or ""),
            str(row.get("projection_manifest_path") or ""),
        )
    )
    return work_orders


def _section_value_profile_work_order(
    *,
    acquisition_row: dict[str, Any],
    missing_rows: list[dict[str, Any]],
    repo_root: Path,
    upstream_dir: Path,
) -> dict[str, Any] | None:
    source = (
        acquisition_row.get("source_archive")
        if isinstance(acquisition_row.get("source_archive"), dict)
        else {}
    )
    payload_kind = str(
        acquisition_row.get("representation_source_payload_kind")
        or acquisition_row.get("source_payload_kind")
        or source.get("source_payload_kind")
        or source.get("kind")
        or ""
    )
    profiler = _SECTION_VALUE_PROFILERS.get(payload_kind)
    if profiler is None:
        return None
    archive_path = _source_archive_path(source)
    archive_sha256 = source.get("archive_zip_sha256") or source.get("sha256")
    archive_bytes = source.get("archive_zip_bytes") or source.get("bytes")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    sections = _profile_work_order_sections(
        profiler_sections=profiler["sections"],
        missing_rows=missing_rows,
    )
    output_dir = _profile_work_order_output_dir(
        acquisition_row=acquisition_row,
        archive_sha256=archive_sha256,
    )
    blockers: list[str] = []
    if not archive_path:
        blockers.append("source_archive_zip_path_missing_for_section_value_profile")
    if not sections:
        blockers.append("no_supported_runtime_sections_missing_value_profile")
    status = (
        "blocked_section_value_profile_work_order"
        if blockers
        else "queued_for_full_video_mlx_section_value_profile"
    )
    argv = [
        ".venv/bin/python",
        str(profiler["tool"]),
        "--archive",
        archive_path or "<missing-archive.zip>",
        "--projection-manifest",
        projection or "<missing-projection-manifest.json>",
        "--output-dir",
        output_dir,
        "--repo-root",
        repo_root.as_posix(),
        "--upstream-dir",
        upstream_dir.as_posix(),
        "--sections",
        *sections,
        "--max-pairs",
        str(CONTEST_PAIR_COUNT),
        "--window-pairs",
        "25",
        "--scorer-batch-pairs",
        "1",
        "--device",
        "gpu",
        "--allow-large-tensor-cache",
    ]
    return {
        "schema": HPRC_SPINE_SECTION_VALUE_PROFILE_WORK_ORDER_SCHEMA,
        "work_order_id": _profile_work_order_id(
            projection_manifest_path=projection,
            archive_sha256=archive_sha256,
            payload_kind=payload_kind,
        ),
        "family": acquisition_row.get("family"),
        "source_payload_kind": payload_kind,
        "projection_manifest_path": projection,
        "archive_zip_path": archive_path,
        "archive_zip_sha256": archive_sha256,
        "archive_zip_bytes": archive_bytes,
        "tool": profiler["tool"],
        "profile_tool": profiler["tool"],
        "sections": sections,
        "profile_sections": sections,
        "upstream_dir": upstream_dir.as_posix(),
        "missing_section_names": [
            str(row.get("section_name") or "") for row in missing_rows
        ],
        "argv": argv,
        "shell_command": " ".join(_shell_quote_arg(arg) for arg in argv),
        "preferred_output_dir": output_dir,
        "storage_waterfall": list(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL),
        "cleanup_contract": {
            "schema": "ssd_first_section_value_profile_cleanup_contract.v1",
            "artifact_tier": "ssd_preferred",
            "large_artifacts_under_output_dir": True,
            "delete_policy": (
                "success_scratch_only; profile JSON, argv, archive hashes, "
                "cache reports, and MLX response manifests retained"
            ),
            "no_signal_loss": True,
        },
        "provenance_requirements": [
            "archive_zip_path_bytes_sha256",
            "projection_manifest_path_sha256",
            "tool_argv",
            "repo_root",
            "upstream_dir",
            "reference_cache_manifest",
            "false_authority_flags",
        ],
        "status": status,
        "blockers": _dedupe(
            [
                *blockers,
                "macos_mlx_section_value_profile_is_advisory_not_score_authority",
                "contest_cpu_cuda_exact_eval_not_executed",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _profile_work_order_sections(
    *,
    profiler_sections: tuple[str, ...],
    missing_rows: list[dict[str, Any]],
) -> list[str]:
    wanted = {
        str(row.get("section_name") or "")
        for row in missing_rows
        if str(row.get("section_name") or "")
    }
    return [section for section in profiler_sections if section in wanted]


def _source_archive_path(source: dict[str, Any]) -> str | None:
    for key in ("archive_zip_path", "archive_path", "path"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _profile_work_order_output_dir(
    *,
    acquisition_row: dict[str, Any],
    archive_sha256: Any,
) -> str:
    family = str(acquisition_row.get("family") or "unknown")
    payload_kind = str(acquisition_row.get("representation_source_payload_kind") or "unknown")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    fingerprint_source = f"{projection}\n{archive_sha256 or ''}\n{payload_kind}"
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    safe_name = _safe_path_token(f"{family}_{payload_kind}_{fingerprint}")
    return (
        Path(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL[0])
        / "hprc_section_value_profiles"
        / safe_name
    ).as_posix()


def _profile_work_order_id(
    *,
    projection_manifest_path: str,
    archive_sha256: Any,
    payload_kind: str,
) -> str:
    fingerprint_source = f"{projection_manifest_path}\n{archive_sha256 or ''}\n{payload_kind}"
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    return f"section_value_profile:{_safe_path_token(payload_kind)}:{fingerprint}"


def _section_cut_materializer_work_orders(
    *,
    acquisition_rows: list[dict[str, Any]],
    section_value_rows: list[dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows_by_projection: dict[str, list[dict[str, Any]]] = {}
    for row in section_value_rows:
        if row.get("admission_status") != "cut_section_bytes_for_receiver_proof":
            continue
        key = str(row.get("projection_manifest_path") or "")
        rows_by_projection.setdefault(key, []).append(row)

    work_orders: list[dict[str, Any]] = []
    for acquisition_row in acquisition_rows:
        projection = str(acquisition_row.get("projection_manifest_path") or "")
        cut_rows = rows_by_projection.get(projection, [])
        if not cut_rows:
            continue
        work_order = _section_cut_materializer_work_order(
            acquisition_row=acquisition_row,
            cut_rows=cut_rows,
            repo_root=repo_root,
        )
        if work_order is not None:
            work_orders.append(work_order)
    work_orders.sort(
        key=lambda row: (
            row["status"],
            str(row.get("source_payload_kind") or ""),
            str(row.get("projection_manifest_path") or ""),
        )
    )
    return work_orders


def _section_cut_materializer_work_order(
    *,
    acquisition_row: dict[str, Any],
    cut_rows: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any] | None:
    payload_kind = str(acquisition_row.get("representation_source_payload_kind") or "")
    materializer = _SECTION_CUT_MATERIALIZERS.get(payload_kind)
    if materializer is None:
        return None
    source = (
        acquisition_row.get("source_archive")
        if isinstance(acquisition_row.get("source_archive"), dict)
        else {}
    )
    archive_path = _source_archive_path(source)
    archive_sha256 = source.get("archive_zip_sha256") or source.get("sha256")
    archive_bytes = source.get("archive_zip_bytes") or source.get("bytes")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    allowed_sections = set(materializer["sections"])
    sections = [
        section
        for section in materializer["sections"]
        if any(row.get("section_name") == section for row in cut_rows)
    ]
    unsupported_sections = sorted(
        {
            str(row.get("section_name") or "")
            for row in cut_rows
            if str(row.get("section_name") or "") not in allowed_sections
        }
    )
    evidence_profile_paths = sorted(
        {
            str(evidence.get("profile_path") or "")
            for row in cut_rows
            for evidence in _rows(row, "evidence_rows")
            if str(evidence.get("profile_path") or "")
        }
    )
    profile_path = evidence_profile_paths[0] if len(evidence_profile_paths) == 1 else ""
    output_dir = _section_cut_work_order_output_dir(
        acquisition_row=acquisition_row,
        archive_sha256=archive_sha256,
        sections=sections,
    )
    blockers: list[str] = []
    if not archive_path:
        blockers.append("source_archive_zip_path_missing_for_section_cut_materializer")
    if not profile_path:
        blockers.append("single_full_video_profile_path_missing_for_section_cut")
    if not sections:
        blockers.append("no_supported_sections_to_cut")
    if unsupported_sections:
        blockers.append("unsupported_cut_sections_require_new_materializer")
    status = (
        "blocked_section_cut_materializer_work_order"
        if blockers
        else "queued_for_byte_closed_section_cut_materializer"
    )
    argv = [
        ".venv/bin/python",
        str(materializer["tool"]),
        "--archive",
        archive_path or "<missing-archive.zip>",
        "--profile",
        profile_path or "<missing-full-video-profile.json>",
        "--output-dir",
        output_dir,
        "--repo-root",
        repo_root.as_posix(),
        "--sections",
        *sections,
        "--run-receiver-proof",
        "--force",
    ]
    return {
        "schema": HPRC_SPINE_SECTION_CUT_MATERIALIZER_WORK_ORDER_SCHEMA,
        "work_order_id": _section_cut_work_order_id(
            projection_manifest_path=projection,
            archive_sha256=archive_sha256,
            payload_kind=payload_kind,
            sections=sections,
        ),
        "family": acquisition_row.get("family"),
        "source_payload_kind": payload_kind,
        "projection_manifest_path": projection,
        "archive_zip_path": archive_path,
        "archive_zip_sha256": archive_sha256,
        "archive_zip_bytes": archive_bytes,
        "full_video_profile_path": profile_path or None,
        "cut_sections": sections,
        "unsupported_sections": unsupported_sections,
        "materializer_tool": materializer["tool"],
        "argv": argv,
        "shell_command": " ".join(_shell_quote_arg(arg) for arg in argv),
        "preferred_output_dir": output_dir,
        "storage_waterfall": list(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL),
        "cleanup_contract": {
            "schema": "ssd_first_section_cut_materializer_cleanup_contract.v1",
            "artifact_tier": "ssd_preferred",
            "receiver_raw_output_retained_by_default": False,
            "report_archive_hashes_and_receiver_proof_retained": True,
            "no_signal_loss": True,
        },
        "provenance_requirements": [
            "archive_zip_path_bytes_sha256",
            "full_video_profile_path_sha256",
            "tool_argv",
            "receiver_proof_report",
            "false_authority_flags",
        ],
        "status": status,
        "blockers": _dedupe(
            [
                *blockers,
                "receiver_proof_required_before_exact_gate",
                "contest_cpu_cuda_exact_eval_not_executed",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _section_cut_work_order_output_dir(
    *,
    acquisition_row: dict[str, Any],
    archive_sha256: Any,
    sections: list[str],
) -> str:
    family = str(acquisition_row.get("family") or "unknown")
    payload_kind = str(acquisition_row.get("representation_source_payload_kind") or "unknown")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    section_token = ",".join(sections)
    fingerprint_source = f"{projection}\n{archive_sha256 or ''}\n{payload_kind}\n{section_token}"
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    safe_name = _safe_path_token(f"{family}_{payload_kind}_{section_token}_{fingerprint}")
    return (
        Path(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL[0])
        / "hprc_section_cut_candidates"
        / safe_name
    ).as_posix()


def _section_cut_work_order_id(
    *,
    projection_manifest_path: str,
    archive_sha256: Any,
    payload_kind: str,
    sections: list[str],
) -> str:
    section_token = ",".join(sections)
    fingerprint_source = (
        f"{projection_manifest_path}\n{archive_sha256 or ''}\n{payload_kind}\n"
        f"{section_token}"
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    return f"section_cut:{_safe_path_token(payload_kind)}:{fingerprint}"


def _projection_gap_repair_work_orders(
    *,
    acquisition_rows: list[dict[str, Any]],
    section_value_rows: list[dict[str, Any]],
    mlx_profiles: list[dict[str, Any]],
    repo_root: Path,
    upstream_dir: Path,
) -> list[dict[str, Any]]:
    rows_by_projection: dict[str, list[dict[str, Any]]] = {}
    for row in section_value_rows:
        if not _section_row_indicates_projection_gap(row):
            continue
        key = str(row.get("projection_manifest_path") or "")
        rows_by_projection.setdefault(key, []).append(row)

    profile_analyses = _projection_gap_profile_analyses(mlx_profiles)
    work_orders: list[dict[str, Any]] = []
    for acquisition_row in acquisition_rows:
        projection = str(acquisition_row.get("projection_manifest_path") or "")
        gap_rows = rows_by_projection.get(projection, [])
        profile_analysis_rows = [
            row
            for row in profile_analyses
            if _profile_analysis_matches_acquisition(
                analysis=row,
                acquisition_row=acquisition_row,
            )
        ]
        suspected_by_profile = any(
            row.get("status") == "archive_projection_gap_suspected"
            for row in profile_analysis_rows
        )
        if not gap_rows and not suspected_by_profile:
            continue
        work_order = _projection_gap_repair_work_order(
            acquisition_row=acquisition_row,
            gap_rows=gap_rows,
            profile_analysis_rows=profile_analysis_rows,
            repo_root=repo_root,
            upstream_dir=upstream_dir,
        )
        if work_order is not None:
            work_orders.append(work_order)
    work_orders.sort(
        key=lambda row: (
            row["status"],
            str(row.get("source_payload_kind") or ""),
            str(row.get("projection_manifest_path") or ""),
        )
    )
    return work_orders


def _section_row_indicates_projection_gap(row: dict[str, Any]) -> bool:
    if str(row.get("section_name") or "") not in _PROJECTION_GAP_STRUCTURAL_SECTIONS:
        return False
    if row.get("evidence_status") != "measured_mlx_advisory":
        return False
    observed_total = row.get("measured_removal_delta_total_mlx_advisory")
    return observed_total is not None and float(observed_total) < 0.0


def _projection_gap_profile_analyses(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for profile in profiles:
        payload = profile["payload"]
        analysis = payload.get("projection_gap_analysis")
        if not isinstance(analysis, dict):
            continue
        analyses.append(
            {
                **analysis,
                "profile_path": profile["path"],
                "profile_sha256": profile["sha256"],
                "projection_manifest_path": payload.get("projection_manifest_path"),
                "candidate_archive": _profile_candidate_archive(payload),
                "family": payload.get("family"),
            }
        )
    return analyses


def _projection_gap_repair_work_order(
    *,
    acquisition_row: dict[str, Any],
    gap_rows: list[dict[str, Any]],
    profile_analysis_rows: list[dict[str, Any]],
    repo_root: Path,
    upstream_dir: Path,
) -> dict[str, Any] | None:
    payload_kind = str(acquisition_row.get("representation_source_payload_kind") or "")
    if payload_kind != "pact_nerv_vq_pvq":
        return None
    source = (
        acquisition_row.get("source_archive")
        if isinstance(acquisition_row.get("source_archive"), dict)
        else {}
    )
    archive_path = _source_archive_path(source)
    archive_sha256 = source.get("archive_zip_sha256") or source.get("sha256")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    evidence_profile_paths = sorted(
        {
            str(evidence.get("profile_path") or "")
            for row in gap_rows
            for evidence in _rows(row, "evidence_rows")
            if str(evidence.get("profile_path") or "")
        }
        | {
            str(row.get("profile_path") or "")
            for row in profile_analysis_rows
            if str(row.get("profile_path") or "")
        }
    )
    negative_sections = sorted(
        {
            str(row.get("section_name") or "")
            for row in gap_rows
            if str(row.get("section_name") or "")
        }
        | {
            str(section)
            for analysis in profile_analysis_rows
            for section in analysis.get("negative_structural_sections", [])
        }
    )
    output_dir = _projection_gap_work_order_output_dir(
        acquisition_row=acquisition_row,
        archive_sha256=archive_sha256,
    )
    blockers: list[str] = []
    if not archive_path:
        blockers.append("source_archive_zip_path_missing_for_projection_gap_repair")
    if not evidence_profile_paths:
        blockers.append("full_video_section_value_profile_missing_for_projection_gap")
    if not negative_sections:
        blockers.append("negative_structural_section_evidence_missing")
    status = (
        "blocked_projection_gap_repair_work_order"
        if blockers
        else "queued_for_pact_vq_projection_gap_repair"
    )
    repair_grid = _pact_vq_projection_gap_repair_grid()
    launch_rows = [
        _pact_vq_projection_gap_repair_launch_row(
            row=row,
            base_output_dir=Path(output_dir),
            repo_root=repo_root,
            upstream_dir=upstream_dir,
        )
        for row in repair_grid
    ]
    argv_rows = [row["argv"] for row in launch_rows]
    return {
        "schema": HPRC_SPINE_PROJECTION_GAP_REPAIR_WORK_ORDER_SCHEMA,
        "work_order_id": _projection_gap_work_order_id(
            projection_manifest_path=projection,
            archive_sha256=archive_sha256,
            payload_kind=payload_kind,
        ),
        "family": acquisition_row.get("family"),
        "source_payload_kind": payload_kind,
        "projection_manifest_path": projection,
        "archive_zip_path": archive_path,
        "archive_zip_sha256": archive_sha256,
        "negative_structural_sections": negative_sections,
        "evidence_profile_paths": evidence_profile_paths,
        "profile_projection_gap_analyses": profile_analysis_rows,
        "repair_objective": (
            "close direct-MLX-model vs exported-archive replay gap, then "
            "reprofile structural sections; exact spend stays blocked until "
            "decoder/codebook/latent bytes have nonnegative measured value"
        ),
        "repair_grid": repair_grid,
        "launch_rows": launch_rows,
        "argv_rows": argv_rows,
        "preferred_output_dir": output_dir,
        "storage_waterfall": list(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL),
        "cleanup_contract": {
            "schema": "ssd_first_projection_gap_repair_cleanup_contract.v1",
            "artifact_tier": "ssd_preferred",
            "large_artifacts_under_output_dir": True,
            "delete_policy": (
                "success_scratch_only; reports, archive hashes, argv, "
                "telemetry, receiver proofs, and replay profiles retained"
            ),
            "no_signal_loss": True,
        },
        "provenance_requirements": [
            "source_archive_bytes_sha256",
            "full_video_profile_path_sha256",
            "training_argv",
            "scorer_upstream_snapshot_hashes",
            "receiver_proof_report",
            "followup_full_video_section_value_profile",
        ],
        "status": status,
        "blockers": _dedupe(
            [
                *blockers,
                "archive_projection_gap_requires_training_or_export_repair",
                "full_video_replay_required_after_each_repair_candidate",
                "contest_cpu_cuda_exact_eval_not_executed",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _pact_vq_projection_gap_repair_grid() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "capacity_l8_e16_k32_ch32",
            "latent_dim": 8,
            "embed_dim": 16,
            "codebook_size": 32,
            "decoder_channel": 32,
        },
        {
            "run_id": "capacity_l16_e32_k64_ch48",
            "latent_dim": 16,
            "embed_dim": 32,
            "codebook_size": 64,
            "decoder_channel": 48,
        },
        {
            "run_id": "capacity_l16_e32_k128_ch48",
            "latent_dim": 16,
            "embed_dim": 32,
            "codebook_size": 128,
            "decoder_channel": 48,
        },
    ]


def _pact_vq_projection_gap_repair_launch_row(
    *,
    row: dict[str, Any],
    base_output_dir: Path,
    repo_root: Path,
    upstream_dir: Path,
) -> dict[str, Any]:
    run_root = base_output_dir / str(row["run_id"])
    runner_output_dir = run_root / "runner_output"
    launch_metadata_dir = run_root / "launch_metadata"
    return {
        "schema": "hprc_projection_gap_repair_launch_row.v1",
        "run_id": row["run_id"],
        "runner_output_dir": runner_output_dir.as_posix(),
        "launch_metadata_dir": launch_metadata_dir.as_posix(),
        "stdout_log": (launch_metadata_dir / "runner.stdout.log").as_posix(),
        "stderr_log": (launch_metadata_dir / "runner.stderr.log").as_posix(),
        "exit_code_path": (launch_metadata_dir / "runner.exit_code").as_posix(),
        "lifecycle_path": (launch_metadata_dir / "runner.lifecycle").as_posix(),
        "output_dir_must_be_empty_before_runner_start": True,
        "metadata_dir_may_exist_before_runner_start": True,
        "launcher_guard": (
            "write launch manifests/logs under launch_metadata_dir only; "
            "the runner owns runner_output_dir and rejects pre-populated output"
        ),
        "argv": _pact_vq_projection_gap_repair_argv(
            row=row,
            output_dir=runner_output_dir,
            repo_root=repo_root,
            upstream_dir=upstream_dir,
        ),
        "post_export_codec_sweep": _pact_vq_post_export_codec_sweep_row(
            run_root=run_root,
            runner_output_dir=runner_output_dir,
            repo_root=repo_root,
        ),
    }


def _pact_vq_post_export_codec_sweep_row(
    *,
    run_root: Path,
    runner_output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_archive = runner_output_dir / "pact_nerv_vq_mlx_training" / "archive.zip"
    output_dir = run_root / "decoder_codec_sweep"
    argv = [
        ".venv/bin/python",
        "tools/sweep_compact_decoder_codecs.py",
        "--source-archive-zip",
        source_archive.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--family",
        "pact_nerv_vq",
        "--repo-root",
        repo_root.as_posix(),
        "--receiver-proof-timeout-seconds",
        "1800",
    ]
    return {
        "schema": HPRC_SPINE_COMPACT_DECODER_CODEC_SWEEP_WORK_ORDER_SCHEMA,
        "status": "queued_after_projection_gap_runner_success",
        "tool": "tools/sweep_compact_decoder_codecs.py",
        "source_archive_zip": source_archive.as_posix(),
        "output_dir": output_dir.as_posix(),
        "decoder_codecs": list(_COMPACT_DECODER_CODEC_SWEEP_PORTFOLIO),
        "run_receiver_proof": True,
        "receiver_output_retained": False,
        "promotion_boundary": (
            "codec variants remain false-authority until full-video MLX "
            "section-value replay and contest CPU/CUDA exact gate"
        ),
        "argv": argv,
        "blockers": [
            "projection_gap_runner_must_complete_successfully_first",
            "full_video_mlx_scorer_replay_not_attached",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def _pact_vq_projection_gap_repair_argv(
    *,
    row: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    upstream_dir: Path,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "pact_nerv_vq",
        "--output-dir",
        output_dir.as_posix(),
        "--repo-root",
        repo_root.as_posix(),
        "--upstream-dir",
        upstream_dir.as_posix(),
        "--num-pairs",
        str(CONTEST_PAIR_COUNT),
        "--epochs",
        "2000",
        "--batch-pairs",
        "4",
        "--compact-latent-dim",
        str(row["latent_dim"]),
        "--compact-embed-dim",
        str(row["embed_dim"]),
        "--compact-codebook-size",
        str(row["codebook_size"]),
        "--compact-decoder-channel",
        str(row["decoder_channel"]),
        "--compact-decoder-codec",
        "int2_scale_bundled",
        "--segnet-distillation-weight",
        "0.05",
        "--pose-distillation-weight",
        "0.0005",
        "--segnet-distillation-objective",
        "boundary_argmax_hinge",
        "--segnet-hinge-margin",
        "1.0",
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        "4",
        "--hard-byte-ceiling",
        "178000",
        "--hard-byte-ceiling",
        "216000",
        "--hard-byte-ceiling",
        "285000",
        "--skip-local-cpu-replay",
    ]


def _projection_gap_work_order_output_dir(
    *,
    acquisition_row: dict[str, Any],
    archive_sha256: Any,
) -> str:
    family = str(acquisition_row.get("family") or "unknown")
    payload_kind = str(acquisition_row.get("representation_source_payload_kind") or "unknown")
    projection = str(acquisition_row.get("projection_manifest_path") or "")
    fingerprint_source = f"{projection}\n{archive_sha256 or ''}\n{payload_kind}\nprojection_gap"
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    safe_name = _safe_path_token(f"{family}_{payload_kind}_projection_gap_{fingerprint}")
    return (
        Path(_SECTION_VALUE_PROFILE_STORAGE_WATERFALL[0])
        / "hprc_projection_gap_repairs"
        / safe_name
    ).as_posix()


def _projection_gap_work_order_id(
    *,
    projection_manifest_path: str,
    archive_sha256: Any,
    payload_kind: str,
) -> str:
    fingerprint_source = (
        f"{projection_manifest_path}\n{archive_sha256 or ''}\n{payload_kind}\n"
        "projection_gap"
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16]
    return f"projection_gap:{_safe_path_token(payload_kind)}:{fingerprint}"


def _safe_path_token(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "unknown"


def _shell_quote_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%")
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _choose_runner_rows(*, compact_base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_demoted_rows = [
        row
        for row in compact_base_rows
        if row.get("route_status") != "demoted_by_hprc_queue_followup"
    ]
    candidate_rows = non_demoted_rows or compact_base_rows
    readyish = [
        row
        for row in candidate_rows
        if row["coverage_valid_for_base_comparison"] and row["fits_ceiling"]
    ]
    readyish.sort(key=lambda row: (int(row["ceiling_bytes"]), int(row["effective_archive_bytes"] or 0)))
    if readyish:
        return readyish[:3]
    shrink = [
        row for row in candidate_rows if row["coverage_valid_for_base_comparison"]
    ]
    shrink.sort(key=lambda row: (int(row["excess_bytes"] or 0), int(row["ceiling_bytes"])))
    if shrink:
        return shrink[:3]
    blocked = sorted(
        candidate_rows,
        key=lambda row: (
            int(row["ceiling_bytes"]),
            int(row["effective_archive_bytes"] or 0),
            row["family"],
        ),
    )
    return blocked[:3]


def _index_section_evidence(
    profiles: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        payload = profile["payload"]
        rows = payload.get("section_value_rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            section = str(row.get("neutralized_section") or row.get("section") or "")
            if not section or section == "none":
                continue
            evidence = _section_evidence_row(row=row, profile=profile)
            index.setdefault(section, []).append(evidence)
    for evidence_rows in index.values():
        evidence_rows.sort(
            key=lambda row: (
                row["presence_admission_delta"],
                row["profile_path"],
                row["variant_id"],
            )
        )
    return index


def _residual_candidate_value_rows(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        payload = profile["payload"]
        section_rows = payload.get("section_value_rows")
        if not isinstance(section_rows, list):
            continue
        for raw in section_rows:
            if not isinstance(raw, dict):
                continue
            section = str(raw.get("neutralized_section") or "")
            variant_id = str(raw.get("variant_id") or "")
            if section != "residual_rc" or variant_id == "baseline":
                continue
            rows.append(_residual_candidate_value_row(profile=profile, row=raw))
    rows.sort(
        key=lambda row: (
            0 if row["admission_status"].startswith("admit") else 1,
            float(row["objective_delta"]),
            row["profile_path"],
            row["variant_id"],
        )
    )
    return rows


def _residual_candidate_value_row(
    *,
    profile: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    variant_id = str(row.get("variant_id") or "")
    delta_nonrate = float(row.get("delta_nonrate_score") or 0.0)
    delta_rate = row.get("delta_rate_score")
    if delta_rate is None:
        delta_rate = -contest_rate_term(int(row.get("archive_bytes_removed_vs_baseline") or 0))
    objective_delta = row.get("delta_total_mlx_score_advisory")
    if objective_delta is None:
        objective_delta = delta_nonrate + float(delta_rate)
    objective_delta = float(objective_delta)
    if variant_id.startswith("neutralize_"):
        action = (
            "demote_existing_residual_section"
            if objective_delta < 0.0
            else "protect_existing_residual_section"
        )
    else:
        action = (
            "admit_residual_token_variant_for_receiver_proof"
            if objective_delta < 0.0
            else "demote_residual_token_variant"
        )
    return {
        "schema": "hprc_residual_token_candidate_admission_row.v1",
        "profile_path": profile["path"],
        "profile_sha256": profile["sha256"],
        "profile_max_pairs": profile["payload"].get("max_pairs"),
        "variant_id": variant_id,
        "section_name": "residual_rc",
        "archive_zip_bytes": row.get("archive_zip_bytes"),
        "archive_bytes_removed_vs_baseline": row.get("archive_bytes_removed_vs_baseline"),
        "delta_nonrate_score": delta_nonrate,
        "delta_rate_score": float(delta_rate),
        "objective_delta": objective_delta,
        "admission_rule": "candidate_delta_nonrate + candidate_delta_rate < 0",
        "admission_status": action,
        "marginal_status": row.get("marginal_status"),
        "requires_receiver_proof": True,
        "requires_exact_gate": True,
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        **FALSE_AUTHORITY,
    }


def _section_evidence_row(*, row: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    delta_nonrate_remove = float(row.get("delta_nonrate_score") or 0.0)
    bytes_removed = int(row.get("archive_bytes_removed_vs_baseline") or 0)
    presence_delta_nonrate = -delta_nonrate_remove
    rate_cost = contest_rate_term(max(bytes_removed, 0))
    presence_admission_delta = presence_delta_nonrate + rate_cost
    observed_total = row.get("delta_total_mlx_score_advisory")
    return {
        "schema": "hprc_spine_section_mlx_value_evidence.v1",
        "profile_path": profile["path"],
        "profile_sha256": profile["sha256"],
        "profile_max_pairs": profile["payload"].get("max_pairs"),
        "profile_scope_status": profile["payload"].get("scope_status"),
        "profile_family": row.get("family") or profile["payload"].get("family"),
        "profile_projection_manifest_path": (
            row.get("projection_manifest_path")
            or profile["payload"].get("projection_manifest_path")
        ),
        "profile_candidate_archive": _profile_candidate_archive(profile["payload"]),
        "variant_id": row.get("variant_id"),
        "neutralized_section": row.get("neutralized_section"),
        "archive_bytes_removed_vs_baseline": bytes_removed,
        "observed_removal_delta_nonrate": delta_nonrate_remove,
        "observed_removal_delta_rate": row.get("delta_rate_score"),
        "observed_removal_delta_total_mlx_advisory": observed_total,
        "presence_delta_nonrate": presence_delta_nonrate,
        "presence_rate_cost_from_removed_bytes": rate_cost,
        "presence_admission_delta": presence_admission_delta,
        "marginal_status": row.get("marginal_status"),
        **FALSE_AUTHORITY,
    }


def _section_evidence_matches_acquisition(
    *,
    evidence: dict[str, Any],
    acquisition_row: dict[str, Any],
) -> bool:
    evidence_family = evidence.get("profile_family")
    if evidence_family not in (None, "") and evidence_family != acquisition_row.get("family"):
        return False
    evidence_projection = evidence.get("profile_projection_manifest_path")
    if evidence_projection in (None, ""):
        return True
    if str(evidence_projection) == str(acquisition_row.get("projection_manifest_path") or ""):
        return True
    return _candidate_archive_matches_acquisition(
        candidate_archive=evidence.get("profile_candidate_archive"),
        acquisition_row=acquisition_row,
    )


def _profile_analysis_matches_acquisition(
    *,
    analysis: dict[str, Any],
    acquisition_row: dict[str, Any],
) -> bool:
    analysis_family = analysis.get("family")
    if analysis_family not in (None, "") and analysis_family != acquisition_row.get("family"):
        return False
    projection = analysis.get("projection_manifest_path")
    if projection in (None, ""):
        return True
    if str(projection) == str(acquisition_row.get("projection_manifest_path") or ""):
        return True
    return _candidate_archive_matches_acquisition(
        candidate_archive=analysis.get("candidate_archive"),
        acquisition_row=acquisition_row,
    )


def _candidate_archive_matches_acquisition(
    *,
    candidate_archive: Any,
    acquisition_row: dict[str, Any],
) -> bool:
    if not isinstance(candidate_archive, dict):
        return False
    source = (
        acquisition_row.get("source_archive")
        if isinstance(acquisition_row.get("source_archive"), dict)
        else {}
    )
    candidate_sha = str(
        candidate_archive.get("archive_zip_sha256")
        or candidate_archive.get("sha256")
        or ""
    )
    source_sha = str(source.get("archive_zip_sha256") or source.get("sha256") or "")
    if candidate_sha and source_sha and candidate_sha == source_sha:
        return True
    candidate_path = str(
        candidate_archive.get("archive_zip_path")
        or candidate_archive.get("path")
        or ""
    )
    source_path = str(_source_archive_path(source) or "")
    return bool(candidate_path and source_path and candidate_path == source_path)


def _profile_candidate_archive(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate = payload.get("candidate_archive")
    if not isinstance(candidate, dict):
        return None
    return {
        "path": candidate.get("path") or candidate.get("archive_zip_path"),
        "sha256": candidate.get("sha256") or candidate.get("archive_zip_sha256"),
        "bytes": candidate.get("bytes") or candidate.get("archive_zip_bytes"),
    }


def _best_section_evidence(evidence_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evidence_rows:
        return None
    full_video_rows = [
        row
        for row in evidence_rows
        if int(row.get("profile_max_pairs") or 0) >= CONTEST_PAIR_COUNT
    ]
    candidates = full_video_rows or evidence_rows
    return min(
        candidates,
        key=lambda row: (
            0 if int(row.get("profile_max_pairs") or 0) >= CONTEST_PAIR_COUNT else 1,
            float(row["presence_admission_delta"]),
        ),
    )


def _index_exact_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in reports:
        payload = item["payload"]
        packet = payload.get("exact_packet") if isinstance(payload.get("exact_packet"), dict) else {}
        for key in (
            payload.get("projection_manifest_path"),
            packet.get("projection_manifest_path"),
            payload.get("execution_report_path"),
        ):
            if isinstance(key, str) and key:
                index[key] = payload
    return index


def _index_hprc_queue_followups(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in reports:
        payload = item["payload"]
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        keys = (
            archive.get("archive_zip_path"),
            archive.get("archive_zip_sha256"),
            payload.get("training_result_path"),
        )
        for key in keys:
            if isinstance(key, str) and key:
                index[key] = payload
    return index


def _index_receiver_proofs(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in reports:
        payload = item["payload"]
        for key in (
            payload.get("archive_path"),
            payload.get("archive_zip_path"),
            payload.get("archive_sha256"),
            payload.get("archive_zip_sha256"),
            payload.get("proof_path"),
            payload.get("report_path"),
        ):
            if isinstance(key, str) and key:
                index[key] = payload
    return index


def _lookup_hprc_queue_followup(
    *,
    acquisition_row: dict[str, Any],
    source: dict[str, Any],
    hprc_queue_followup_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    keys = (
        source.get("archive_zip_path"),
        source.get("archive_path"),
        source.get("path"),
        source.get("archive_zip_sha256"),
        source.get("sha256"),
        acquisition_row.get("projection_manifest_path"),
    )
    for key in keys:
        if isinstance(key, str) and key in hprc_queue_followup_index:
            return hprc_queue_followup_index[key]
    return None


def _lookup_receiver_proof(
    *,
    acquisition_row: dict[str, Any],
    source: dict[str, Any],
    receiver_proof_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    keys = (
        source.get("archive_zip_path"),
        source.get("archive_path"),
        source.get("path"),
        source.get("archive_zip_sha256"),
        source.get("sha256"),
        acquisition_row.get("projection_manifest_path"),
    )
    for key in keys:
        if isinstance(key, str) and key in receiver_proof_index:
            return receiver_proof_index[key]
    return None


def _hprc_queue_followup_summary(followup: dict[str, Any] | None) -> dict[str, Any] | None:
    if followup is None:
        return None
    archive = followup.get("archive") if isinstance(followup.get("archive"), dict) else {}
    byte_intelligence = (
        archive.get("byte_intelligence")
        if isinstance(archive.get("byte_intelligence"), dict)
        else {}
    )
    replay_gate = (
        followup.get("local_replay_gate")
        if isinstance(followup.get("local_replay_gate"), dict)
        else {}
    )
    promotion_gate = (
        followup.get("promotion_gate")
        if isinstance(followup.get("promotion_gate"), dict)
        else {}
    )
    return {
        "schema": "hprc_queue_followup_compact_summary.v1",
        "training_result_path": followup.get("training_result_path"),
        "archive_zip_path": archive.get("archive_zip_path"),
        "archive_zip_sha256": archive.get("archive_zip_sha256"),
        "archive_zip_bytes": archive.get("archive_zip_bytes"),
        "resolution_rate_feasibility": byte_intelligence.get(
            "resolution_rate_feasibility"
        ),
        "local_replay_gate": {
            "required": replay_gate.get("required"),
            "evaluation_passed": replay_gate.get("evaluation_passed"),
            "blockers": replay_gate.get("blockers"),
        },
        "promotion_gate": {
            "ready_for_exact_eval_dispatch": promotion_gate.get(
                "ready_for_exact_eval_dispatch"
            ),
            "blockers": promotion_gate.get("blockers"),
        },
        "planner_learning_signals": [
            {
                "signal_id": signal.get("signal_id"),
                "status": signal.get("status"),
                "metric_name": signal.get("metric_name"),
                "metric_value": signal.get("metric_value"),
                "next_architecture_priorities": signal.get(
                    "next_architecture_priorities"
                ),
                "reactivation_criteria": signal.get("reactivation_criteria"),
            }
            for signal in _rows(followup, "planner_learning_signals")
        ],
        **FALSE_AUTHORITY,
    }


def _hprc_queue_followup_demoting_blockers(
    followup: dict[str, Any] | None,
) -> list[str]:
    if followup is None:
        return []
    blockers: list[str] = []
    local_gate = (
        followup.get("local_replay_gate")
        if isinstance(followup.get("local_replay_gate"), dict)
        else {}
    )
    for blocker in local_gate.get("blockers") or []:
        if isinstance(blocker, str) and blocker:
            blockers.append(blocker)
    for signal in _rows(followup, "planner_learning_signals"):
        signal_id = str(signal.get("signal_id") or "")
        if signal_id in {
            "defer_lowres_dense_residual_collapse_until_mlx_distortion_recovers",
            "hprc_rate_feasible_but_resolution_distortion_bound",
            "hprc_rate_bound_before_distortion_gate",
        }:
            blockers.append(signal_id)
    return _dedupe(blockers)


def _hprc_queue_followup_signal_rows(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in reports:
        payload = item["payload"]
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        for signal in _rows(payload, "planner_learning_signals"):
            rows.append(
                {
                    "schema": "hprc_queue_followup_signal_row.v1",
                    "report_path": item["path"],
                    "report_sha256": item["sha256"],
                    "archive_zip_path": archive.get("archive_zip_path"),
                    "archive_zip_sha256": archive.get("archive_zip_sha256"),
                    "archive_zip_bytes": archive.get("archive_zip_bytes"),
                    "signal_id": signal.get("signal_id"),
                    "status": signal.get("status"),
                    "metric_name": signal.get("metric_name"),
                    "metric_value": signal.get("metric_value"),
                    "blockers": signal.get("blockers"),
                    "next_architecture_priorities": signal.get(
                        "next_architecture_priorities"
                    ),
                    "reactivation_criteria": signal.get("reactivation_criteria"),
                    **FALSE_AUTHORITY,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row.get("archive_zip_sha256") or ""),
            str(row.get("signal_id") or ""),
            str(row.get("report_path") or ""),
        )
    )
    return rows


def _receiver_proof_summary(proof: dict[str, Any] | None) -> dict[str, Any] | None:
    if proof is None:
        return None
    return {
        "proof_path": proof.get("proof_path"),
        "archive_path": proof.get("archive_path"),
        "archive_sha256": proof.get("archive_sha256"),
        "runtime_consumption_proof_passed": proof.get(
            "runtime_consumption_proof_passed"
        ),
        "receiver_contract_satisfied": proof.get("receiver_contract_satisfied"),
        "receiver_output_kind": proof.get("receiver_output_kind"),
        "receiver_output_bytes": proof.get("receiver_output_bytes"),
        "blockers": proof.get("blockers"),
    }


def _posterior_update_hooks(
    *,
    compact_base_rows: list[dict[str, Any]],
    section_value_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hooks: list[dict[str, Any]] = []
    for row in compact_base_rows:
        hooks.append(
            {
                "schema": "hprc_spine_posterior_update_hook.v1",
                "family": row["family"],
                "stage": "compact_base_sweep",
                "scope": "full_video",
                "status": row["route_status"],
                "demote_on_blockers": row["blockers"],
                "record_positive_after_exact_axis_only": True,
            }
        )
        if row.get("route_status") == "demoted_by_hprc_queue_followup":
            summary = row.get("hprc_queue_followup_summary")
            signals = (
                summary.get("planner_learning_signals")
                if isinstance(summary, dict)
                else []
            )
            hooks.append(
                {
                    "schema": "hprc_spine_posterior_update_hook.v1",
                    "family": row["family"],
                    "stage": "hprc_queue_followup_demotion",
                    "scope": "compact_base_full_video_candidate",
                    "status": "demote_from_queue_followup_signal",
                    "signal_ids": [
                        signal.get("signal_id")
                        for signal in signals
                        if isinstance(signal, dict)
                    ],
                    "record_negative_now": True,
                }
            )
    for row in section_value_rows:
        if row["admission_status"].startswith("demote"):
            hooks.append(
                {
                    "schema": "hprc_spine_posterior_update_hook.v1",
                    "family": row["family"],
                    "stage": "section_value_admission",
                    "scope": row["section_name"],
                    "status": "demote_from_measured_value_per_byte",
                    "admission_objective_delta": row["admission_objective_delta"],
                    "record_negative_now": True,
                }
            )
    return hooks


def _plan_blockers(
    *,
    compact_base_rows: list[dict[str, Any]],
    section_value_rows: list[dict[str, Any]],
    mlx_profiles: list[dict[str, Any]],
    projection_gap_repair_work_orders: list[dict[str, Any]],
) -> list[str]:
    blockers = ["contest_cpu_cuda_exact_eval_not_executed"]
    if not mlx_profiles:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    elif not any(
        mlx_profile_has_full_video_coverage(profile["payload"])
        for profile in mlx_profiles
    ):
        blockers.append("full_video_mlx_scorer_replay_not_attached")
        if any(
            mlx_profile_full_video_scope(profile["payload"])
            == "sampled_prefix_requires_full_video_rerun"
            for profile in mlx_profiles
        ):
            blockers.append("sampled_mlx_prefilter_requires_full_video_rerun")
        else:
            blockers.append("mlx_prefilter_not_full_video")
    if not any(row["coverage_valid_for_base_comparison"] for row in compact_base_rows):
        blockers.append("no_full_coverage_compact_base_candidate")
    if not any(row["fits_ceiling"] and row["coverage_valid_for_base_comparison"] for row in compact_base_rows):
        blockers.append("no_full_coverage_candidate_under_any_hard_ceiling")
    if any(row["evidence_status"] == "missing" for row in section_value_rows):
        blockers.append("some_sections_missing_value_per_byte_measurement")
    if any(
        row.get("route_status") == "demoted_by_hprc_queue_followup"
        for row in compact_base_rows
    ):
        blockers.append("hprc_queue_followup_demoted_candidate_before_replay")
    if projection_gap_repair_work_orders:
        blockers.append("archive_projection_gap_requires_training_or_export_repair")
    return _dedupe(blockers)


def _load_profile(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    payload = _load_json_object(resolved)
    if payload.get("schema") != HPRC_MLX_COMPONENT_PROFILE_SCHEMA:
        raise ValueError(f"MLX profile has unexpected schema: {resolved}")
    return {"path": resolved.as_posix(), "sha256": _sha256_file(resolved), "payload": payload}


def _section_evidence_has_full_video_scope(row: dict[str, Any]) -> bool:
    return (
        mlx_profile_full_video_scope(
            {
                "scope_status": row.get("profile_scope_status"),
            }
        )
        == "executed"
        and int(row.get("profile_max_pairs") or 0) >= CONTEST_PAIR_COUNT
    )


def _load_receiver_proof(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    payload = _load_json_object(resolved)
    return {"path": resolved.as_posix(), "sha256": _sha256_file(resolved), "payload": payload}


def _load_exact_report(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    payload = _load_json_object(resolved)
    return {"path": resolved.as_posix(), "sha256": _sha256_file(resolved), "payload": payload}


def _load_hprc_queue_followup(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    payload = _load_json_object(resolved)
    if payload.get("schema") != HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA:
        raise ValueError(f"HPRC queue followup has unexpected schema: {resolved}")
    return {"path": resolved.as_posix(), "sha256": _sha256_file(resolved), "payload": payload}


def _rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(values: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA",
    "HPRC_SPINE_COMPACT_BASE_SWEEP_ROW_SCHEMA",
    "HPRC_SPINE_SECTION_VALUE_PROFILE_WORK_ORDER_SCHEMA",
    "HPRC_SPINE_SECTION_VALUE_ROW_SCHEMA",
    "build_spine_bounded_runner_plan",
    "write_spine_bounded_runner_plan",
]
