# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
    REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA,
    REPAIR_BUDGET_MATERIALIZER_BINDING_ROW_SCHEMA,
    REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA,
    REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA,
    REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
    build_frontier_repair_budget_materialization_execution_report,
    build_frontier_repair_budget_materialization_plan,
    build_frontier_repair_budget_materializer_binding_report,
    build_frontier_repair_budget_waterfill_queue,
)
from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key, value in FALSE_AUTHORITY.items():
        assert payload.get(key) is value, key


def _materialization_plan() -> dict[str, object]:
    parent_id = "repair_rate_floor_parent_abc123"
    child_id = "repair_budget_spent_child_def456"
    return {
        "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
        "chain_id": "global_many_op_rate_distortion_receiver_campaign",
        "parent_candidate_chain_id": parent_id,
        "candidate_chain_row_count": 2,
        "rate_only_floor_preserved_before_repair_spend": True,
        "spent_budget_candidates_are_children_of_rate_only_floor": True,
        "rate_only_candidate_remains_valid_even_if_child_regresses": True,
        "candidate_archive_materialized": False,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
        "candidate_chain_rows": [
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "rate_only_floor_parent",
                "candidate_chain_id": parent_id,
                "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                "materialization_order": 1,
                "parent_candidate_chain_id": None,
                "saved_bytes_total": 160,
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": ["rate_only_candidate_archive_materialization_missing"],
                **FALSE_AUTHORITY,
            },
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": child_id,
                "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                "materialization_order": 2,
                "parent_candidate_chain_id": parent_id,
                "parent_must_be_preserved_before_child": True,
                "child_must_not_replace_parent_archive": True,
                "allocation_rank": 1,
                "allocation_candidate_id": "pairset_drop_many_fixture",
                "requested_repair_bytes": 32,
                "proposed_encoder_repair_bytes": 32,
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": ["parent_rate_only_archive_materialization_required"],
                **FALSE_AUTHORITY,
            },
        ],
    }


def _receiver_consumed_manifest(
    tmp_path: Path,
    *,
    candidate_chain_id: str,
    target_kind: str = "archive_section_entropy_recode_v1",
    palette_modes: list[str] | None = None,
    component_response_replayed: bool = False,
    component_response_replay_axis_tag: str = "[macOS-MLX research-signal]",
) -> dict[str, object]:
    archive_path = tmp_path / f"{candidate_chain_id}.zip"
    proof_path = tmp_path / f"{candidate_chain_id}.proof.json"
    archive_path.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    proof_path.write_text('{"receiver_consumed": true}\n', encoding="utf-8")
    manifest: dict[str, object] = {
        "schema": "frontier_rate_attack_materializer_manifest_fixture.v1",
        "materializer_id": "fixture_receiver_consumed_materializer",
        "target_kind": target_kind,
        "candidate_chain_id": candidate_chain_id,
        "byte_closed_candidate_emitted": True,
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": "a" * 64,
            "bytes": archive_path.stat().st_size,
        },
        "source_archive": {
            "path": str(tmp_path / "source.zip"),
            "sha256": "b" * 64,
            "bytes": 128,
        },
        "runtime_consumption_proof_path": str(proof_path),
        "receiver_contract_satisfied": True,
        "receiver_verification": {
            "proof_path": str(proof_path),
            "proof_present": True,
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
        },
        "archive_manifest": {
            "canonical_palette": list(palette_modes or []),
        },
        "readiness_blockers": [],
        **FALSE_AUTHORITY,
    }
    if component_response_replayed:
        replay_path = tmp_path / f"{candidate_chain_id}.component_response.json"
        replay_path.write_text(
            json.dumps(
                {
                    "schema": "component_response_replay_fixture.v1",
                    "candidate_chain_id": candidate_chain_id,
                    "axis_tag": component_response_replay_axis_tag,
                    "authority": "local_research_signal_only",
                    **FALSE_AUTHORITY,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["component_response_replayed"] = True
        manifest["component_response_replay"] = {
            "schema": "component_response_replay_fixture.v1",
            "replayed": True,
            "artifact_path": str(replay_path),
            "axis_tag": component_response_replay_axis_tag,
            "evidence_grade": "local_materialization_audit_only",
            "blockers": [
                "exact_axis_component_response_required_before_budget_spend"
            ],
        }
    return manifest


def test_repair_budget_materialization_execution_report_refuses_until_runtime_proof(
    tmp_path: Path,
) -> None:
    plan = _materialization_plan()

    report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path / "plan.json",
    )

    assert report["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    _assert_false_authority(report)
    assert report["rate_only_floor_parent_first"] is True
    assert report["candidate_archive_materialized"] is False
    assert report["runtime_consumption_proof_present"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["exact_readiness_refusal"]["ready"] is False
    assert "candidate_archives_not_materialized" in report["blockers"]
    rows = report["execution_rows"]
    assert [row["candidate_kind"] for row in rows] == [
        "rate_only_floor_parent",
        "spent_budget_repair_child",
    ]
    assert all(
        row["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA
        for row in rows
    )
    assert rows[1]["parent_candidate_chain_id"] == report["parent_candidate_chain_id"]
    assert "parent_rate_only_archive_materialization_required" in rows[1]["blockers"]

    plan_path = _write_json(tmp_path / "repair_budget_materialization_plan.json", plan)
    report_path = tmp_path / "repair_budget_materialization_execution_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_repair_budget_materialization_execution_report.py",
            "--materialization-plan",
            str(plan_path),
            "--execution-report-out",
            str(report_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["candidate_archive_materialized"] is False
    materialized = json.loads(report_path.read_text(encoding="utf-8"))
    assert materialized["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    assert materialized["exact_readiness_refusal"]["ready"] is False


def test_repair_budget_materializer_binding_report_preserves_parent_fail_closed(
    tmp_path: Path,
) -> None:
    plan = _materialization_plan()
    parent_id = str(plan["parent_candidate_chain_id"])
    pr110_palette = list(FEC6_FIXED_K16_MODE_IDS)
    parent_manifest = _receiver_consumed_manifest(
        tmp_path,
        candidate_chain_id=parent_id,
        palette_modes=pr110_palette,
    )

    binding_report = build_frontier_repair_budget_materializer_binding_report(
        repo_root=REPO_ROOT,
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_manifests=[parent_manifest],
    )

    assert binding_report["schema"] == REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
    _assert_false_authority(binding_report)
    assert binding_report["candidate_archive_materialized"] is False
    assert binding_report["candidate_archive_materialized_count"] == 1
    assert binding_report["repair_dynamics_prior_count"] == 1
    assert (
        "not_all_candidate_chain_rows_bound_to_receiver_consumed_manifests"
        in binding_report["blockers"]
    )
    parent_row, child_row = binding_report["binding_rows"]
    assert parent_row["schema"] == REPAIR_BUDGET_MATERIALIZER_BINDING_ROW_SCHEMA
    assert parent_row["candidate_chain_id"] == parent_id
    assert parent_row["candidate_archive_materialized"] is True
    assert parent_row["runtime_consumption_proof_present"] is True
    assert parent_row["receiver_consumed"] is True
    parent_prior = parent_row["repair_dynamics_prior"]
    assert parent_prior["schema"] == REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA
    assert parent_prior["mode_count"] == 16
    assert parent_prior["frame0_mode_count"] == 15
    assert parent_prior["frame1_mode_count"] == 0
    assert parent_prior["zero_frame1_modes"] is True
    assert parent_prior["dominant_dynamics_interpretation"] == (
        "frame0_global_color_geometry_calibration_prior"
    )
    assert (
        "empirical_non_identity_palette_is_all_frame0"
        in parent_prior["repair_waterfill_hints"]
    )
    assert child_row["candidate_archive_materialized"] is False
    assert "candidate_chain_materializer_manifest_missing" in child_row["blockers"]
    aggregate_prior = binding_report["repair_dynamics_palette_prior"]
    assert aggregate_prior["schema"] == REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA
    assert aggregate_prior["mode_count"] == 16
    assert aggregate_prior["identity_mode_count"] == 1
    assert aggregate_prior["frame0_mode_count"] == 15
    assert aggregate_prior["frame1_mode_count"] == 0
    assert aggregate_prior["frame0_non_identity_fraction"] == 1.0
    assert aggregate_prior["zero_frame1_modes"] is True

    execution_report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_binding_report=binding_report,
        materializer_binding_report_path=tmp_path
        / "repair_budget_materializer_binding_report.json",
    )

    assert execution_report["schema"] == (
        REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    )
    assert execution_report["materializer_binding_row_count"] == 2
    assert execution_report["candidate_archive_materialized_count"] == 1
    assert execution_report["runtime_consumption_proof_present_count"] == 1
    assert execution_report["execution_rows"][0]["candidate_archive_materialized"] is True
    assert execution_report["execution_rows"][0]["receiver_consumed"] is True
    assert execution_report["execution_rows"][1]["candidate_archive_materialized"] is False
    assert execution_report["ready_for_exact_eval_dispatch"] is False
    _assert_false_authority(execution_report)

    plan_path = _write_json(tmp_path / "repair_budget_materialization_plan.json", plan)
    manifest_path = _write_json(tmp_path / "parent_manifest.json", parent_manifest)
    binding_path = tmp_path / "repair_budget_materializer_binding_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_repair_budget_materializer_binding_report.py",
            "--materialization-plan",
            str(plan_path),
            "--materializer-manifest",
            str(manifest_path),
            "--repair-palette-mode",
            "frame0_blue_chroma_amp_1",
            "--binding-report-out",
            str(binding_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["candidate_archive_materialized"] is False
    assert stdout["repair_dynamics_palette_prior_present"] is True
    materialized = json.loads(binding_path.read_text(encoding="utf-8"))
    assert materialized["schema"] == REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
    assert materialized["candidate_archive_materialized_count"] == 1
    assert materialized["repair_dynamics_prior_count"] == 2
    assert materialized["repair_dynamics_palette_prior"]["frame1_mode_count"] == 0


def test_repair_budget_child_manifest_requires_component_replay_for_execution(
    tmp_path: Path,
) -> None:
    plan = _materialization_plan()
    parent_id = str(plan["parent_candidate_chain_id"])
    child_id = str(plan["candidate_chain_rows"][1]["candidate_chain_id"])  # type: ignore[index]
    parent_manifest = _receiver_consumed_manifest(
        tmp_path,
        candidate_chain_id=parent_id,
    )
    child_manifest = _receiver_consumed_manifest(
        tmp_path,
        candidate_chain_id=child_id,
        target_kind="segnet_posenet_repair_child_v1",
    )

    binding_report = build_frontier_repair_budget_materializer_binding_report(
        repo_root=REPO_ROOT,
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_manifests=[parent_manifest, child_manifest],
    )

    parent_row, child_row = binding_report["binding_rows"]
    assert parent_row["candidate_archive_materialized"] is True
    assert child_row["candidate_archive_materialized"] is True
    assert child_row["receiver_consumed"] is True
    assert child_row["component_response_replayed"] is False

    execution_report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_binding_report=binding_report,
        materializer_binding_report_path=tmp_path
        / "repair_budget_materializer_binding_report.json",
    )

    child_execution = execution_report["execution_rows"][1]
    assert child_execution["candidate_archive_materialized"] is True
    assert child_execution["receiver_consumed"] is True
    assert child_execution["component_response_replayed"] is False
    assert child_execution["ready_for_local_materialization"] is False
    assert "component_response_replayed_false" in child_execution["blockers"]
    assert execution_report["ready_for_local_materialization_count"] == 1
    assert execution_report["ready_for_exact_eval_dispatch"] is False
    _assert_false_authority(execution_report)


def test_repair_budget_child_component_replay_manifest_unblocks_local_execution(
    tmp_path: Path,
) -> None:
    plan = _materialization_plan()
    parent_id = str(plan["parent_candidate_chain_id"])
    child_id = str(plan["candidate_chain_rows"][1]["candidate_chain_id"])  # type: ignore[index]
    parent_manifest = _receiver_consumed_manifest(
        tmp_path,
        candidate_chain_id=parent_id,
    )
    child_manifest = _receiver_consumed_manifest(
        tmp_path,
        candidate_chain_id=child_id,
        target_kind="segnet_posenet_repair_child_v1",
        component_response_replayed=True,
    )

    binding_report = build_frontier_repair_budget_materializer_binding_report(
        repo_root=REPO_ROOT,
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_manifests=[parent_manifest, child_manifest],
    )

    child_row = binding_report["binding_rows"][1]
    assert child_row["candidate_archive_materialized"] is True
    assert child_row["receiver_consumed"] is True
    assert child_row["component_response_replayed"] is True
    assert child_row["component_response_replay_path"].endswith(
        f"{child_id}.component_response.json"
    )
    assert child_row["component_response_replay_axis_tag"] == (
        "[macOS-MLX research-signal]"
    )
    assert (
        "component_response_replayed_false" not in child_row["blockers"]
    )
    assert binding_report["candidate_archive_materialized_count"] == 2
    assert binding_report["receiver_consumed_count"] == 2
    assert binding_report["candidate_archive_materialized"] is True
    assert binding_report["ready_for_exact_eval_dispatch"] is False
    _assert_false_authority(binding_report)

    execution_report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_binding_report=binding_report,
        materializer_binding_report_path=tmp_path
        / "repair_budget_materializer_binding_report.json",
    )

    child_execution = execution_report["execution_rows"][1]
    assert child_execution["candidate_archive_materialized"] is True
    assert child_execution["receiver_consumed"] is True
    assert child_execution["component_response_replayed"] is True
    assert child_execution["component_response_replay_axis_tag"] == (
        "[macOS-MLX research-signal]"
    )
    assert child_execution["ready_for_local_materialization"] is True
    assert child_execution["ready_for_budget_spend"] is False
    assert child_execution["ready_for_exact_eval_dispatch"] is False
    assert "component_response_replayed_false" not in child_execution["blockers"]
    assert execution_report["ready_for_local_materialization_count"] == 2
    assert execution_report["component_response_replayed_count"] == 2
    assert execution_report["ready_for_exact_eval_dispatch"] is False
    _assert_false_authority(execution_report)


def test_receiver_closed_rate_credit_materializes_rate_only_parent_locally(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "parent_archive.zip"
    proof_path = tmp_path / "runtime_consumption_proof.json"
    archive_path.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    proof_path.write_text('{"receiver_consumed": true}\n', encoding="utf-8")
    work_order = {
        "schema": REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA,
        "chain_id": "global_many_op_rate_distortion_receiver_campaign",
        "rate_budget_preservation_plan": {
            "rows": [
                {
                    "schema": "frontier_rate_attack_rate_budget_preservation_row.v1",
                    "preservation_id": "preserve_packet_merge_fixture",
                    "candidate_id": "materializer_packet_member_merge_v1",
                    "target_kind": "packet_member_merge_v1",
                    "saved_bytes": 258,
                    "rate_credit_score_units": 0.0001717916099055202,
                    "distortion_debt_score_units": 0.0,
                    "net_score_delta_score_units": -0.0001717916099055202,
                    "preserve_as_rate_only_candidate": True,
                    **FALSE_AUTHORITY,
                }
            ],
            "operator_action_ledger": {
                "schema": "frontier_rate_attack_operator_action_ledger.v1",
                "term_count": 1,
                "terms": [],
                **FALSE_AUTHORITY,
            },
            **FALSE_AUTHORITY,
        },
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 258,
            "receiver_closed_rate_credit_rows": [
                {
                    "schema": "frontier_rate_attack_receiver_closed_rate_credit_row.v1",
                    "candidate_id": "packet_member_merge_fixture",
                    "target_kind": "packet_member_merge_v1",
                    "saved_bytes": 258,
                    "archive_path": str(archive_path),
                    "archive_sha256": "c" * 64,
                    "archive_bytes": archive_path.stat().st_size,
                    "runtime_consumption_proof_path": str(proof_path),
                    "receiver_closed": True,
                    **FALSE_AUTHORITY,
                }
            ],
            **FALSE_AUTHORITY,
        },
        "allocation_rows": [],
        "repair_cascade_opportunity_rows": [
            {
                "schema": REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA,
                "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
                "label": "Cascade C",
                "source_relation": "PR110-OPT-5+7+10+12_UNTOUCHED",
                "targeted_positions": [
                    {"position_id": "P19", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P18", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P11", "entropy_surface": "selector_codec_entropy"},
                ],
                "pipeline_position": "scorer_entropy_repair_before_selector_codec",
                "canonical_mechanisms": [
                    {"mechanism_id": "uniward_textured_region_undetectability"},
                    {"mechanism_id": "detector_informed_embedding"},
                ],
                "required_probe_measurements": [
                    "posenet_null_bottom_decile_pair_ids",
                    "segnet_class_region_mask_ids",
                ],
                "optimization_implication": "stack_scorer_repair_with_selector_codec",
                "estimate_status": (
                    "per_region_selector_codec_variant_empirically_falsified_"
                    "scorer_repair_hypothesis_survives"
                ),
                "empirical_feedback": {
                    "schema": "frontier_rate_attack_repair_cascade_empirical_feedback.v1",
                    "variant_verdict": "implementation_level_falsification",
                    "best_observed_delta_vs_fec6_wire_bytes": 83,
                    "surviving_hypotheses": [
                        "fold_posenet_null_signal_into_fec8_markov_transition_matrix",
                    ],
                    **FALSE_AUTHORITY,
                },
                "required_empirical_landing": "mlx_local_probe",
                "next_queue_action": "build_cascade_c_mlx_local_probe_queue",
                "blockers": [
                    "cascade_c_empirical_component_response_missing",
                    "per_region_selector_codec_materializer_missing",
                ],
                **FALSE_AUTHORITY,
            }
        ],
        **FALSE_AUTHORITY,
    }

    plan = build_frontier_repair_budget_materialization_plan(
        repair_budget_waterfill_work_order=work_order,
        repair_budget_waterfill_work_order_path=tmp_path / "work_order.json",
    )

    parent = plan["candidate_chain_rows"][0]
    assert parent["candidate_kind"] == "rate_only_floor_parent"
    assert parent["candidate_archive_materialized"] is True
    assert parent["candidate_archive_path"] == str(archive_path)
    assert parent["runtime_consumption_proof_present"] is True
    assert parent["receiver_consumed"] is True
    assert "rate_only_candidate_archive_materialization_missing" not in parent["blockers"]
    assert "receiver_runtime_consumption_proof_missing" not in parent["blockers"]
    entropy_positions = parent["entropy_pipeline_positions"]
    assert entropy_positions[0]["target_kind"] == "packet_member_merge_v1"
    assert entropy_positions[0]["entropy_pipeline_position"] == (
        "after_entropy_coder_container_or_zip_grammar"
    )
    cascade = plan["candidate_chain_rows"][1]
    assert cascade["candidate_kind"] == "structural_repair_cascade_probe"
    assert cascade["cascade_id"] == "cascade_c_posenet_null_segnet_region_selector_codec"
    assert cascade["source_relation"] == "PR110-OPT-5+7+10+12_UNTOUCHED"
    assert cascade["targeted_positions"][0]["position_id"] == "P19"
    assert cascade["pipeline_position"] == "scorer_entropy_repair_before_selector_codec"
    assert cascade["cascade_opportunity"]["canonical_mechanisms"][0]["mechanism_id"] == (
        "uniward_textured_region_undetectability"
    )
    assert "segnet_class_region_mask_ids" in cascade["cascade_opportunity"][
        "required_probe_measurements"
    ]
    assert cascade["estimate_status"] == (
        "per_region_selector_codec_variant_empirically_falsified_"
        "scorer_repair_hypothesis_survives"
    )
    assert cascade["empirical_feedback"]["variant_verdict"] == (
        "implementation_level_falsification"
    )
    assert cascade["empirical_feedback"]["best_observed_delta_vs_fec6_wire_bytes"] == 83
    assert cascade["parent_candidate_chain_id"] == parent["candidate_chain_id"]
    assert "cascade_c_empirical_component_response_missing" in cascade["blockers"]

    binding_report = build_frontier_repair_budget_materializer_binding_report(
        repo_root=REPO_ROOT,
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
    )

    binding_parent = binding_report["binding_rows"][0]
    assert binding_parent["plan_row_receiver_closed_binding"] is True
    assert binding_parent["candidate_archive_materialized"] is True
    assert binding_parent["candidate_archive_sha256"] == "c" * 64
    assert binding_parent["candidate_archive_bytes"] == archive_path.stat().st_size
    assert binding_parent["receiver_consumed"] is True
    assert binding_report["candidate_archive_materialized_count"] == 1
    assert binding_report["receiver_consumed_count"] == 1

    execution_report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_binding_report=binding_report,
        materializer_binding_report_path=tmp_path
        / "repair_budget_materializer_binding_report.json",
    )
    execution_parent = execution_report["execution_rows"][0]
    assert execution_parent["candidate_kind"] == "rate_only_floor_parent"
    assert execution_parent["candidate_archive_sha256"] == "c" * 64
    assert execution_parent["candidate_archive_bytes"] == archive_path.stat().st_size
    assert execution_parent["ready_for_local_materialization"] is True
    assert execution_parent["execution_status"] == (
        "ready_for_receiver_closed_candidate_replay"
    )
    assert execution_report["ready_for_local_materialization_count"] == 1


def test_repair_budget_waterfill_queue_emits_execution_audit_step(
    tmp_path: Path,
) -> None:
    autonomous_path = _write_json(
        tmp_path / "autonomous_chain.json",
        {
            "schema": "frontier_rate_attack_autonomous_chain_optimization.v1",
            "rows": [
                {
                    "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
                    "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                    "chain_family": "rate_distortion_receiver_closed_many_op_campaign",
                    "rate_budget_preservation_plan": {},
                    **FALSE_AUTHORITY,
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    harvest_path = _write_json(
        tmp_path / "harvest.json",
        {
            "schema": TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
            "rows": [],
            **FALSE_AUTHORITY,
        },
    )
    budget_path = _write_json(
        tmp_path / "receiver_closed_budget.json",
        {
            "schema": "frontier_rate_attack_receiver_closed_correction_budget.v1",
            "receiver_closed_saved_bytes_total": 0,
            **FALSE_AUTHORITY,
        },
    )

    queue = build_frontier_repair_budget_waterfill_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=json.loads(autonomous_path.read_text()),
        autonomous_chain_optimization_path=autonomous_path,
        targeted_component_correction_response_harvest=json.loads(
            harvest_path.read_text()
        ),
        targeted_component_correction_response_harvest_path=harvest_path,
        receiver_closed_correction_budget=json.loads(budget_path.read_text()),
        receiver_closed_correction_budget_path=budget_path,
        results_root=tmp_path / "results",
        queue_id="repair_waterfill_execution_audit_unit",
        chain_limit=1,
    )

    assert queue is not None
    experiment = queue["experiments"][0]
    step_ids = [step["id"] for step in experiment["steps"]]
    assert step_ids == [
        "emit_repair_budget_waterfill_work_order",
        "emit_repair_budget_materialization_plan",
        "bind_repair_budget_materializer_execution",
        "audit_repair_budget_materialization_execution",
    ]
    binding_step = experiment["steps"][2]
    assert binding_step["requires"] == ["emit_repair_budget_materialization_plan"]
    assert binding_step["command"][1] == (
        "tools/build_frontier_repair_budget_materializer_binding_report.py"
    )
    assert any(
        condition.get("equals") == REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
        for condition in binding_step["postconditions"]
    )
    audit_step = experiment["steps"][3]
    assert audit_step["requires"] == ["bind_repair_budget_materializer_execution"]
    assert audit_step["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_execution_report.py"
    )
    assert "--materializer-binding-report" in audit_step["command"]
    assert any(
        condition.get("equals") == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
        for condition in audit_step["postconditions"]
    )
    assert (
        experiment["metadata"]["candidate_chain_materializer_binding_report_path"].endswith(
            "repair_budget_materializer_binding_report.json"
        )
    )
    assert (
        experiment["metadata"]["candidate_chain_execution_report_path"].endswith(
            "repair_budget_materialization_execution_report.json"
        )
    )
    assert experiment["metadata"]["candidate_archive_materialized"] is False
    _assert_false_authority(experiment["metadata"])
