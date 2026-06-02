# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.analysis.compact_vq_pivot_audit import (
    COMPACT_VQ_MISMATCH_STATUS,
)
from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.campaign import HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA
from tac.substrates.hprc.representation_spine import (
    HprcRepresentationFamily,
    build_representation_spine_packet,
    write_representation_spine_projection,
)
from tac.substrates.hprc.spine_acquisition import build_spine_acquisition_report
from tac.substrates.hprc.spine_bounded_runner import (
    HPRC_SPINE_COMPACT_CARRIER_PIVOT_WORK_ORDER_SCHEMA,
    HPRC_SPINE_PROJECTION_GAP_REPAIR_WORK_ORDER_SCHEMA,
    HPRC_SPINE_SECTION_CUT_MATERIALIZER_WORK_ORDER_SCHEMA,
    HPRC_SPINE_SECTION_VALUE_PROFILE_WORK_ORDER_SCHEMA,
    build_spine_bounded_runner_plan,
)

REPO = Path(__file__).resolve().parents[5]


def test_spine_bounded_runner_forces_receiver_proof_and_coverage(tmp_path: Path) -> None:
    rnerv = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
    )
    short_vq = _projection(
        tmp_path / "short_vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 8,
        selectors=b"s" * 4,
        manifest_extra={"num_pairs": 32},
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[rnerv, short_vq],
        hard_byte_ceilings=[178_000, 216_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    rows = {row["runner_row_id"]: row for row in plan["compact_base_sweep_rows"]}
    assert rows["rnerv:178000"]["action"] == (
        "receiver_proof_then_full_video_mlx_replay_then_exact_gate"
    )
    assert rows["rnerv:178000"]["requires_receiver_proof"] is True
    assert rows["rnerv:178000"]["requires_full_video_mlx_replay"] is True
    assert rows["pact_nerv_vq:178000"]["action"] == (
        "train_or_scale_to_full_coverage_emit_spine_then_receiver_proof"
    )
    assert "declared_pair_coverage_below_full_video" in rows["pact_nerv_vq:178000"]["blockers"]
    assert plan["score_claim"] is False


def test_spine_bounded_runner_consumes_receiver_proof_report(tmp_path: Path) -> None:
    manifest = _projection(
        tmp_path / "vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 8,
        selectors=b"s" * 4,
        manifest_extra={"num_pairs": 600},
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = payload["manifest"]["representation_spine"]["source"]
    source.update(
        {
            "archive_zip_path": (tmp_path / "archive.zip").as_posix(),
            "archive_zip_sha256": "a" * 64,
            "archive_zip_bytes": 12345,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    proof_path = tmp_path / "receiver_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "generated_receiver_proof.v1",
                "proof_path": proof_path.as_posix(),
                "archive_path": (tmp_path / "archive.zip").as_posix(),
                "archive_sha256": "a" * 64,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
                "receiver_output_kind": "directory",
                "receiver_output_bytes": 2860,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        receiver_proof_report_paths=[proof_path],
    )

    row = plan["compact_base_sweep_rows"][0]
    assert row["receiver_proof_observed"] is True
    assert row["receiver_proof_passed"] is True
    assert row["receiver_proof_summary"]["receiver_output_kind"] == "directory"
    assert "receiver_proof_not_attached" not in row["blockers"]
    assert "receiver_proof_failed" not in row["blockers"]
    assert "exact_gate_not_yet_attached" in row["blockers"]


def test_section_value_admission_demotes_bad_residual_tokens(tmp_path: Path) -> None:
    residual = _projection(
        tmp_path / "residual",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 30,
        residual=b"r" * 17,
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[residual],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": True},
                "section_value_rows": [
                    {"variant_id": "baseline", "neutralized_section": "none"},
                    {
                        "variant_id": "neutralize_residual_rc",
                        "neutralized_section": "residual_rc",
                        "archive_bytes_removed_vs_baseline": 17,
                        "delta_nonrate_score": -0.25,
                        "delta_rate_score": -contest_rate_term(17),
                        "delta_total_mlx_score_advisory": -0.25 - contest_rate_term(17),
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "archive_bytes_removed_vs_baseline": 30,
                        "delta_nonrate_score": 1.0,
                        "delta_rate_score": -contest_rate_term(30),
                        "delta_total_mlx_score_advisory": 1.0 - contest_rate_term(30),
                        "marginal_status": "protect_candidate_value_exceeds_rate_price",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    residual_row = sections["residual_rc"]
    assert residual_row["evidence_status"] == "measured_mlx_advisory"
    assert residual_row["admission_status"] == "demote_or_block_residual_tokens"
    assert residual_row["admission_objective_delta"] > 0
    residual_candidates = plan["residual_token_admission_rows"]
    assert any(
        row["schema"] == "hprc_residual_token_candidate_admission_row.v1"
        and row["admission_status"] == "demote_existing_residual_section"
        for row in residual_candidates
    )
    assert sections["decoder_qw"]["admission_status"] == "admit_section_bytes_for_receiver_proof"
    assert any(hook["status"] == "demote_from_measured_value_per_byte" for hook in plan["posterior_update_hooks"])


def test_spine_bounded_runner_consumes_queue_followup_demotion(
    tmp_path: Path,
) -> None:
    manifest = _projection(
        tmp_path / "hprc",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        manifest_extra={"num_pairs": 600},
    )
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"archive")
    archive_sha = "d" * 64
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["manifest"]["representation_spine"]["source"].update(
        {
            "archive_zip_path": archive_path.as_posix(),
            "archive_zip_sha256": archive_sha,
            "archive_zip_bytes": archive_path.stat().st_size,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    followup_path = tmp_path / "hprc_queue_followup_report.json"
    followup_path.write_text(
        json.dumps(
            {
                "schema": HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA,
                "training_result_path": "hprc_rate_collapse_report.json",
                "archive": {
                    "archive_zip_path": archive_path.as_posix(),
                    "archive_zip_sha256": archive_sha,
                    "archive_zip_bytes": archive_path.stat().st_size,
                    "byte_intelligence": {
                        "resolution_rate_feasibility": {
                            "schema": "hprc_resolution_rate_feasibility.v1",
                            "status": (
                                "rate_feasible_but_distortion_bound_resolution_risk"
                            ),
                            "decoder_pixels_per_scorer_pixel": 0.0625,
                        }
                    },
                },
                "local_replay_gate": {
                    "required": True,
                    "evaluation_passed": False,
                    "blockers": [
                        "mlx_prefilter_rejected_candidate_before_cpu_replay"
                    ],
                },
                "promotion_gate": {
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                },
                "planner_learning_signals": [
                    {
                        "schema": "hprc_planner_learning_signal.v1",
                        "signal_id": (
                            "hprc_rate_feasible_but_resolution_distortion_bound"
                        ),
                        "status": "route_to_pose_geometry_or_predictive_redesign",
                        "metric_name": "decoder_pixels_per_scorer_pixel",
                        "metric_value": 0.0625,
                        "next_architecture_priorities": [
                            "native_scorer_aware_training_at_compress_time",
                            "sparse_procedural_pose_geometry_tokens",
                        ],
                        "reactivation_criteria": [
                            "decoder_or_protected_pose_pathway_preserves_full600_mlx_pose"
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        hprc_queue_followup_report_paths=[followup_path],
    )

    row = plan["compact_base_sweep_rows"][0]
    assert row["route_status"] == "demoted_by_hprc_queue_followup"
    assert row["requires_architecture_redesign_before_replay"] is True
    assert row["hprc_queue_followup_observed"] is True
    assert "hprc_rate_feasible_but_resolution_distortion_bound" in row["blockers"]
    assert "mlx_prefilter_rejected_candidate_before_cpu_replay" in row["blockers"]
    assert plan["hprc_queue_followup_signal_rows"][0]["signal_id"] == (
        "hprc_rate_feasible_but_resolution_distortion_bound"
    )
    assert "hprc_queue_followup_demoted_candidate_before_replay" in plan["blockers"]
    assert any(
        hook["status"] == "demote_from_queue_followup_signal"
        for hook in plan["posterior_update_hooks"]
    )


def test_projection_metadata_and_short_coverage_do_not_force_mlx_value_replay(
    tmp_path: Path,
) -> None:
    short_vq = _projection(
        tmp_path / "short_vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 8,
        codebooks=b"c" * 12,
        selectors=b"s" * 4,
        manifest_extra={"num_pairs": 32},
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[short_vq],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    assert sections["rdo_plan"]["evidence_status"] == (
        "metadata_contract_no_mlx_replay_required"
    )
    assert sections["rdo_plan"]["requires_full_video_mlx_replay"] is False
    assert sections["codebooks_q"]["evidence_status"] == (
        "not_required_until_full_video_coverage"
    )
    assert sections["codebooks_q"]["requires_full_video_mlx_replay"] is False
    assert "some_sections_missing_value_per_byte_measurement" not in plan["blockers"]


def test_spine_bounded_runner_emits_selector_profile_work_order_for_full_coverage(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "selector_v4_archive.zip"
    archive.write_bytes(b"archive")
    manifest = _projection(
        tmp_path / "selector_v4",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "9" * 64,
            "archive_zip_bytes": archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_selector_v4_psv4",
            "side_channel_kind": "rle_selector",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    work_orders = plan["section_value_profile_work_orders"]
    assert len(work_orders) == 1
    order = work_orders[0]
    assert order["schema"] == HPRC_SPINE_SECTION_VALUE_PROFILE_WORK_ORDER_SCHEMA
    assert order["status"] == "queued_for_full_video_mlx_section_value_profile"
    assert order["profile_tool"] == (
        "tools/profile_pact_nerv_selector_v4_mlx_section_value.py"
    )
    assert order["archive_zip_path"] == archive.as_posix()
    assert order["projection_manifest_path"] == manifest.as_posix()
    assert order["profile_sections"] == [
        "decoder_qw",
        "latents_rc",
        "selectors_rc",
    ]
    assert "--max-pairs" in order["argv"]
    assert "600" in order["argv"]
    assert order["preferred_output_dir"].startswith("/Volumes/VertigoDataTier/pact/")
    assert order["score_claim"] is False
    assert "macos_mlx_section_value_profile_is_advisory_not_score_authority" in order[
        "blockers"
    ]
    acquisition_row = acquisition["rows"][0]
    assert acquisition_row["representation_source_payload_kind"] == (
        "pact_nerv_selector_v4_psv4"
    )


def test_sampled_section_value_profile_opens_full_video_work_order(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "selector_v4_archive.zip"
    archive.write_bytes(b"archive")
    manifest = _projection(
        tmp_path / "selector_v4",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "a" * 64,
            "archive_zip_bytes": archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_selector_v4_psv4",
            "side_channel_kind": "rle_selector",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "sampled_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 32,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "sampled_prefix_requires_full_video_rerun"},
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "family": "pact_nerv",
                        "projection_manifest_path": manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 20,
                        "delta_nonrate_score": 1.0,
                        "delta_rate_score": -contest_rate_term(20),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    assert sections["decoder_qw"]["evidence_status"] == (
        "sampled_mlx_advisory_requires_full_video_replay"
    )
    assert sections["decoder_qw"]["requires_full_video_mlx_replay"] is True
    assert "sampled_mlx_section_value_replay_not_budget_authority" in sections[
        "decoder_qw"
    ]["blockers"]
    assert "full_video_mlx_scorer_replay_not_attached" in plan["blockers"]
    assert "sampled_mlx_prefilter_requires_full_video_rerun" in plan["blockers"]
    work_orders = plan["section_value_profile_work_orders"]
    assert len(work_orders) == 1
    assert work_orders[0]["profile_sections"] == [
        "decoder_qw",
        "latents_rc",
        "selectors_rc",
    ]


def test_pr95_section_value_profile_work_order_requires_runtime_custody(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pr95_archive.zip"
    archive.write_bytes(b"archive")
    runtime = tmp_path / "hnerv_muon"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    manifest = _projection(
        tmp_path / "pr95",
        family=HprcRepresentationFamily.PR95_HNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "9" * 64,
            "archive_zip_bytes": archive.stat().st_size,
            "submission_dir": runtime.as_posix(),
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pr95_u32_prefixed_hnerv",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
    )

    work_orders = plan["section_value_profile_work_orders"]
    assert len(work_orders) == 1
    order = work_orders[0]
    assert order["status"] == "queued_for_full_video_mlx_section_value_profile"
    assert order["profile_tool"] == "tools/profile_pr95_hnerv_mlx_section_value.py"
    assert order["profile_sections"] == ["decoder_qw", "latents_rc"]
    assert order["submission_runtime_dir"] == runtime.as_posix()
    assert "--submission-dir" in order["argv"]
    assert runtime.as_posix() in order["argv"]
    assert "submission_runtime_dir_missing_for_section_value_profile" not in order[
        "blockers"
    ]


def test_full_video_section_value_profile_satisfies_work_order(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "selector_v4_archive.zip"
    archive.write_bytes(b"archive")
    manifest = _projection(
        tmp_path / "selector_v4",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "b" * 64,
            "archive_zip_bytes": archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_selector_v4_psv4",
            "side_channel_kind": "rle_selector",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "full_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "section_value_rows": [
                    {
                        "variant_id": f"neutralize_{section}",
                        "neutralized_section": section,
                        "family": "pact_nerv",
                        "projection_manifest_path": manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 1,
                        "delta_nonrate_score": 1.0,
                        "delta_rate_score": -contest_rate_term(1),
                    }
                    for section in ("decoder_qw", "latents_rc", "selectors_rc")
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    assert sections["decoder_qw"]["evidence_status"] == "measured_mlx_advisory"
    assert sections["latents_rc"]["evidence_status"] == "measured_mlx_advisory"
    assert sections["selectors_rc"]["evidence_status"] == "measured_mlx_advisory"
    assert plan["section_value_profile_work_orders"] == []


def test_full_video_section_value_profile_marks_cut_candidates(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "selector_v4_archive.zip"
    archive.write_bytes(b"archive")
    manifest = _projection(
        tmp_path / "selector_v4_cut",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "c" * 64,
            "archive_zip_bytes": archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_selector_v4_psv4",
            "side_channel_kind": "rle_selector",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    profile_path = tmp_path / "full_profile_cut.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "family": "pact_nerv",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "family": "pact_nerv",
                        "projection_manifest_path": manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 20,
                        "delta_nonrate_score": 1.0,
                        "delta_total_mlx_score_advisory": 1.0,
                        "marginal_status": "protect_candidate_value_exceeds_rate_price",
                    },
                    {
                        "variant_id": "neutralize_latents_rc",
                        "neutralized_section": "latents_rc",
                        "family": "pact_nerv",
                        "projection_manifest_path": manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 8,
                        "delta_nonrate_score": 0.0,
                        "delta_total_mlx_score_advisory": -contest_rate_term(8),
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    sections = {row["section_name"]: row for row in plan["section_value_rows"]}
    assert sections["decoder_qw"]["admission_status"] == (
        "admit_section_bytes_for_receiver_proof"
    )
    assert sections["decoder_qw"]["section_spend_recommendation"] == (
        "protect_section_bytes_measured_value_exceeds_rate_price"
    )
    assert sections["latents_rc"]["admission_status"] == (
        "cut_section_bytes_for_receiver_proof"
    )
    assert sections["latents_rc"]["section_spend_recommendation"] == (
        "cut_section_bytes_measured_removal_improves_objective"
    )
    assert sections["latents_rc"]["measured_removal_delta_total_mlx_advisory"] < 0
    work_orders = plan["section_cut_materializer_work_orders"]
    assert len(work_orders) == 1
    order = work_orders[0]
    assert order["schema"] == HPRC_SPINE_SECTION_CUT_MATERIALIZER_WORK_ORDER_SCHEMA
    assert order["status"] == "queued_for_byte_closed_section_cut_materializer"
    assert order["materializer_tool"] == (
        "tools/materialize_pact_nerv_selector_v4_section_cut_candidate.py"
    )
    assert order["cut_sections"] == ["latents_rc"]
    assert order["archive_zip_path"] == archive.as_posix()
    assert order["full_video_profile_path"] == profile_path.as_posix()
    assert "--run-receiver-proof" in order["argv"]
    assert order["score_claim"] is False


def test_pact_vq_core_negative_sections_open_projection_gap_repair(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pact_vq_archive.zip"
    archive.write_bytes(b"archive")
    manifest = _projection(
        tmp_path / "pact_vq_gap",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 20,
        codebooks=b"c" * 12,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": archive.as_posix(),
            "archive_zip_sha256": "e" * 64,
            "archive_zip_bytes": archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_vq_pvq",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    copied_manifest = tmp_path / "copied_manifest" / manifest.name
    copied_manifest.parent.mkdir()
    copied_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    profile_path = tmp_path / "pact_vq_full_profile_gap.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "family": "pact_nerv_vq",
                "max_pairs": 600,
                "scorer_batch_pairs": 1,
                "projection_manifest_path": copied_manifest.as_posix(),
                "candidate_archive": {
                    "path": archive.as_posix(),
                    "sha256": "e" * 64,
                    "bytes": archive.stat().st_size,
                },
                "scope_status": {"full_video": "executed"},
                "projection_gap_analysis": {
                    "schema": "hprc_archive_projection_gap_analysis.v1",
                    "status": "archive_projection_gap_suspected",
                    "negative_structural_sections": ["decoder_qw", "codebooks_q"],
                    "requires_direct_model_vs_archive_replay": True,
                    "requires_capacity_or_export_projection_repair": True,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "section_value_rows": [
                    {
                        "variant_id": "neutralize_decoder_qw",
                        "neutralized_section": "decoder_qw",
                        "family": "pact_nerv_vq",
                        "projection_manifest_path": copied_manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 20,
                        "delta_nonrate_score": -0.5,
                        "delta_total_mlx_score_advisory": -0.5,
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                    {
                        "variant_id": "neutralize_codebooks_q",
                        "neutralized_section": "codebooks_q",
                        "family": "pact_nerv_vq",
                        "projection_manifest_path": copied_manifest.as_posix(),
                        "archive_bytes_removed_vs_baseline": 12,
                        "delta_nonrate_score": -0.25,
                        "delta_total_mlx_score_advisory": -0.25,
                        "marginal_status": "cut_candidate_distortion_nonworse",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        mlx_profile_paths=[profile_path],
    )

    orders = plan["projection_gap_repair_work_orders"]
    assert len(orders) == 1
    order = orders[0]
    assert order["schema"] == HPRC_SPINE_PROJECTION_GAP_REPAIR_WORK_ORDER_SCHEMA
    assert order["status"] == "queued_for_pact_vq_projection_gap_repair"
    assert order["negative_structural_sections"] == ["codebooks_q", "decoder_qw"]
    assert order["evidence_profile_paths"] == [profile_path.as_posix()]
    assert len(order["repair_grid"]) == 3
    assert len(order["launch_rows"]) == 3
    first_launch = order["launch_rows"][0]
    assert first_launch["run_id"] == "capacity_l8_e16_k32_ch32"
    assert first_launch["runner_output_dir"].endswith(
        "/capacity_l8_e16_k32_ch32/runner_output"
    )
    assert first_launch["launch_metadata_dir"].endswith(
        "/capacity_l8_e16_k32_ch32/launch_metadata"
    )
    assert first_launch["runner_output_dir"] in first_launch["argv"]
    assert first_launch["runner_output_dir"] not in first_launch["launch_metadata_dir"]
    assert first_launch["output_dir_must_be_empty_before_runner_start"] is True
    sweep = first_launch["post_export_codec_sweep"]
    assert sweep["schema"] == "hprc_spine_compact_decoder_codec_sweep_work_order.v1"
    assert sweep["status"] == "queued_after_projection_gap_runner_success"
    assert sweep["tool"] == "tools/sweep_compact_decoder_codecs.py"
    assert sweep["source_archive_zip"].endswith(
        "/capacity_l8_e16_k32_ch32/runner_output/pact_nerv_vq_mlx_training/archive.zip"
    )
    assert sweep["output_dir"].endswith(
        "/capacity_l8_e16_k32_ch32/decoder_codec_sweep"
    )
    assert "portfolio_auto" in sweep["decoder_codecs"]
    assert "int2_scale_bundled" in sweep["decoder_codecs"]
    assert "tools/sweep_compact_decoder_codecs.py" in sweep["argv"]
    assert sweep["run_receiver_proof"] is True
    assert sweep["score_claim"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in sweep["blockers"]
    assert all("--execute-family" in argv for argv in order["argv_rows"])
    assert any("pact_nerv_vq" in argv for argv in order["argv_rows"])
    assert "archive_projection_gap_requires_training_or_export_repair" in plan[
        "blockers"
    ]
    assert order["score_claim"] is False


def test_compact_vq_pivot_audit_demotes_pact_vq_runner_rows(
    tmp_path: Path,
) -> None:
    pact_archive = tmp_path / "pact_vq_archive.zip"
    pact_archive.write_bytes(b"pact-vq")
    pact_manifest = _projection(
        tmp_path / "pact_vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 20,
        codebooks=b"c" * 12,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": pact_archive.as_posix(),
            "archive_zip_sha256": "f" * 64,
            "archive_zip_bytes": pact_archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_vq_pvq",
        },
    )
    rnerv_archive = tmp_path / "rnerv_archive.zip"
    rnerv_archive.write_bytes(b"rnerv")
    rnerv_manifest = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
        source={
            "archive_zip_path": rnerv_archive.as_posix(),
            "archive_zip_sha256": "1" * 64,
            "archive_zip_bytes": rnerv_archive.stat().st_size,
        },
        manifest_extra={"num_pairs": 600},
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[pact_manifest, rnerv_manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    audit_path = tmp_path / "compact_vq_pivot.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "compact_vq_pivot_audit.v1",
                "family": "pact_nerv_vq",
                "verdict": COMPACT_VQ_MISMATCH_STATUS,
                "spend_recommendation": (
                    "route_compact_training_budget_to_pr95_hinerv_snerv_"
                    "stage8_or_rebuild_vq_as_rt_residual_token_bolton"
                ),
                "profile_signal": {
                    "best_full_video_mlx_score": 90.0,
                    "local_replay_threshold_passed": False,
                },
                "implementation_contract": {
                    "per_pair_single_vector_vq_present": True,
                    "residual_tokenization_present": False,
                    "shallow_interframe_feature_path_present": False,
                    "codebook_utilization_repair_present": False,
                },
                "blockers": [
                    "compact_vq_is_per_pair_latent_not_residual_tokenization",
                    "full_video_mlx_score_above_local_replay_threshold",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        compact_vq_pivot_audit_paths=[audit_path],
    )

    rows = {row["family"]: row for row in plan["compact_base_sweep_rows"]}
    assert rows["pact_nerv_vq"]["route_status"] == (
        "demoted_by_compact_vq_pivot_audit"
    )
    assert rows["pact_nerv_vq"]["compact_vq_pivot_audit_observed"] is True
    assert rows["pact_nerv_vq"]["requires_architecture_redesign_before_replay"] is True
    assert "compact_vq_pivot_audit_demoted_family" in rows["pact_nerv_vq"][
        "blockers"
    ]
    assert plan["compact_vq_pivot_signal_rows"][0]["verdict"] == (
        COMPACT_VQ_MISMATCH_STATUS
    )
    assert "compact_vq_pivot_audit_demoted_family" in plan["blockers"]
    assert all(row["family"] != "pact_nerv_vq" for row in plan["selected_runner_rows"])
    pivot_orders = plan["compact_carrier_pivot_work_orders"]
    assert len(pivot_orders) == 1
    pivot_order = pivot_orders[0]
    assert pivot_order["schema"] == HPRC_SPINE_COMPACT_CARRIER_PIVOT_WORK_ORDER_SCHEMA
    assert pivot_order["status"] == "queued_for_compact_carrier_pivot_execution"
    assert pivot_order["demoted_family"] == "pact_nerv_vq"
    assert pivot_order["best_full_video_mlx_score"] == 90.0
    assert pivot_order["pivot_families"] == ["pr95_hnerv", "hi_nerv", "snerv"]
    assert "pivot_launch_rows_not_executed" in pivot_order["blockers"]
    launch = {row["family"]: row for row in pivot_order["launch_rows"]}
    assert launch["pr95_hnerv"]["run_id"] == (
        "pr95_hnerv_stage8_faithful_scoreaware_600pair"
    )
    assert "--execute-family" in launch["pr95_hnerv"]["argv"]
    assert "pr95_hnerv" in launch["pr95_hnerv"]["argv"]
    assert "--run-receiver-proof" in launch["pr95_hnerv"]["argv"]
    assert "--coder-aware-qat" in launch["hi_nerv"]["argv"]
    assert "--run-local-cpu-replay" in launch["snerv"]["argv"]
    assert all("--overwrite" not in row["argv"] for row in pivot_order["launch_rows"])
    assert any(
        hook["status"] == "demote_from_upstream_eval_and_rt_vq_mismatch"
        for hook in plan["posterior_update_hooks"]
    )


def test_all_demoted_compact_vq_rows_are_not_selected_without_pivot(
    tmp_path: Path,
) -> None:
    pact_archive = tmp_path / "pact_vq_archive.zip"
    pact_archive.write_bytes(b"pact-vq")
    pact_manifest = _projection(
        tmp_path / "pact_vq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 20,
        codebooks=b"c" * 12,
        selectors=b"s" * 4,
        source={
            "archive_zip_path": pact_archive.as_posix(),
            "archive_zip_sha256": "f" * 64,
            "archive_zip_bytes": pact_archive.stat().st_size,
        },
        manifest_extra={
            "num_pairs": 600,
            "source_payload_kind": "pact_nerv_vq_pvq",
        },
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[pact_manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    audit_path = tmp_path / "compact_vq_pivot.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "compact_vq_pivot_audit.v1",
                "family": "pact_nerv_vq",
                "verdict": COMPACT_VQ_MISMATCH_STATUS,
                "spend_recommendation": "pivot",
                "profile_signal": {
                    "best_full_video_mlx_score": 90.0,
                    "local_replay_threshold_passed": False,
                },
                "implementation_contract": {
                    "per_pair_single_vector_vq_present": True,
                    "residual_tokenization_present": False,
                    "shallow_interframe_feature_path_present": False,
                    "codebook_utilization_repair_present": False,
                },
                "blockers": ["compact_vq_is_per_pair_latent_not_residual_tokenization"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=acquisition_path,
        repo_root=REPO,
        compact_vq_pivot_audit_paths=[audit_path],
    )

    assert plan["selected_runner_rows"] == []
    assert len(plan["compact_carrier_pivot_work_orders"]) == 1
    assert "compact_carrier_pivot_work_order_opened" in plan["blockers"]


def test_spine_bounded_runner_cli_writes_plan(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 20,
        latents=b"l" * 8,
    )
    acquisition = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[178_000],
    )
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    output = tmp_path / "runner.json"
    followup_path = tmp_path / "unused_followup.json"
    followup_path.write_text(
        json.dumps(
            {
                "schema": HPRC_QUEUE_FOLLOWUP_REPORT_SCHEMA,
                "training_result_path": "unused.json",
                "archive": {},
                "planner_learning_signals": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--acquisition-report",
            acquisition_path.as_posix(),
            "--output",
            output.as_posix(),
            "--repo-root",
            REPO.as_posix(),
            "--queue-followup-report",
            followup_path.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "hprc_spine_bounded_runner_plan.v1"
    assert payload["compact_base_sweep_rows"][0]["requires_receiver_proof"] is True


def _projection(
    out: Path,
    *,
    family: HprcRepresentationFamily,
    decoder: bytes,
    latents: bytes = b"",
    codebooks: bytes = b"",
    selectors: bytes = b"",
    residual: bytes = b"",
    source: dict[str, object] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    spine = build_representation_spine_packet(
        family=family,
        decoder_blob=decoder,
        latents_blob=latents,
        codebooks_blob=codebooks,
        selectors_blob=selectors,
        residual_blob=residual,
        source=source,
        manifest_extra=manifest_extra,
    )
    written = write_representation_spine_projection(output_dir=out, spine=spine)
    return Path(written["manifest_path"])


def _load_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "build_hprc_spine_bounded_runner_test",
        REPO / "tools/build_hprc_spine_bounded_runner.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
