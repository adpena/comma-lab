from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (
    SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA,
    SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA,
    build_scorer_region_selector_cascade_campaign_queue,
    build_scorer_region_selector_cascade_campaign_report,
    enumerate_cascade_variants,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", b"payload")


def _source_submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    _write_zip(submission / "archive.zip")
    return submission


def _upstream_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    pose_null = tmp_path / "pose_null.json"
    soft16 = tmp_path / "soft16.npy"
    soft256 = tmp_path / "soft256.npy"
    _write_json(pose_null, {"analysis": {"pose_null_decile": []}, **FALSE_AUTHORITY})
    soft16.write_bytes(b"fake-softmax-16")
    soft256.write_bytes(b"fake-softmax-256")
    return pose_null, soft16, soft256


def test_enumerate_cascade_variants_crosses_grouped_operator_dimensions() -> None:
    variants = enumerate_cascade_variants(
        null_fractions=(0.05, 0.10),
        top_regions_per_pair_values=(2,),
        receiver_patch_max_pair_values=(12, 24),
        receiver_patch_regions_per_pair_values=(1,),
        receiver_patch_rgb_deltas=((-1, -1, -1),),
        receiver_patch_yuv_deltas=((1, 0, 0),),
        selector_codec_family_groups=(("fec10_adaptive_blend",),),
        scales=(64,),
        alphas=(1,),
        max_variants=None,
    )

    assert len(variants) == 8
    assert {variant.receiver_patch_delta_space for variant in variants} == {
        "rgb",
        "yuv601_proxy_as_rgb",
    }
    assert {variant.receiver_patch_max_pairs for variant in variants} == {12, 24}
    assert all(variant.selector_codec_families for variant in variants)


def test_campaign_queue_builds_independent_variants_and_harvest_dependency(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    pose_null, soft16, soft256 = _upstream_artifacts(tmp_path)

    queue = build_scorer_region_selector_cascade_campaign_queue(
        repo_root=tmp_path,
        queue_id="campaign_q",
        source_submission_dir=submission,
        output_root=tmp_path / "campaign_out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        pose_null_modes_artifact=pose_null,
        segnet_softmax_16=soft16,
        segnet_softmax_256=soft256,
        null_fractions=(0.05,),
        top_regions_per_pair_values=(2,),
        receiver_patch_max_pair_values=(12, 24),
        receiver_patch_regions_per_pair_values=(1,),
        receiver_patch_rgb_deltas=((-1, -1, -1),),
        selector_codec_family_groups=(("fec10_adaptive_blend",),),
        scales=(64,),
        alphas=(1,),
        prove_receiver_patch_output_change=True,
        receiver_patch_output_change_file_list_entries=("0.raw",),
        receiver_patch_output_change_expected_file_list_sha256="a" * 64,
        receiver_patch_output_change_expected_entry_count=1,
        receiver_patch_output_change_file_list_source="tests/file_list.txt",
        include_local_component_loop=True,
        include_mlx_component_response=True,
        include_scorer_response_dataset=True,
        include_local_component_retention_plan=True,
        scorer_response_baseline_score=0.1919853363,
        scorer_response_baseline_archive_bytes=178493,
        max_concurrency_local_cpu=2,
        max_concurrency_local_mlx=1,
    )

    assert queue["metadata"]["schema"] == SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA
    assert queue["metadata"]["variant_count"] == 2
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 1
    variant_experiments = [exp for exp in queue["experiments"] if exp["id"] != "campaign_harvest"]
    assert len(variant_experiments) == 2
    for experiment in variant_experiments:
        assert experiment["metadata"]["schema"] == "scorer_region_selector_cascade_variant.v1"
        assert "grouped-cascade-campaign" in experiment["tags"]
        step_ids = [step["id"] for step in experiment["steps"]]
        assert "materialize_p19_posenet_null_pairs" in step_ids
        assert "build_scorer_response_dataset" in step_ids
        patch_step = next(
            step
            for step in experiment["steps"]
            if step["id"] == "materialize_frame1_region_waterfill_runtime_patch"
        )
        assert "--selected-archive-chain-report" in patch_step["command"]
    harvest = queue["experiments"][-1]["steps"][0]
    assert harvest["id"] == "harvest_campaign_learning_surface"
    assert len(harvest["requires"]) == 2
    assert all(requirement.endswith(".plan_local_component_artifact_retention") for requirement in harvest["requires"])


def test_campaign_report_harvests_variant_learning_rows(tmp_path: Path) -> None:
    root_a = tmp_path / "campaign" / "a"
    root_b = tmp_path / "campaign" / "b"
    _write_json(
        root_a / "scorer_region_selector_chain_report.json",
        {
            "schema": "scorer_region_selector_chain_report.v1",
            "selector_saved_bytes": 7,
            "repack_saved_bytes_after_selector": 3,
            "cumulative_rate_saved_bytes_vs_source": 10,
            "selected_local_survivor_stage": "P15_archive_zip_repack",
            **FALSE_AUTHORITY,
        },
    )
    _write_json(
        root_a
        / "frame1_region_waterfill_runtime_patch"
        / "full_frame_output_change_proof"
        / "shell_inflate_output_change.json",
        {
            "schema": "shell_inflate_output_change_proof_v1",
            "output_change_observed": True,
            "raw_shape_preserving_output_change_observed": True,
            "differing_byte_count": 12,
            **FALSE_AUTHORITY,
        },
    )
    _write_json(
        root_a
        / "frame1_region_waterfill_runtime_patch"
        / "local_component_spot_check"
        / "local_cpu_advisory.json",
        {
            "score_axis": "cpu_advisory",
            "canonical_score": 0.1920003362662307,
            "avg_posenet_dist": 0.00002943,
            "avg_segnet_dist": 0.00055994,
            **FALSE_AUTHORITY,
        },
    )
    _write_json(
        root_a
        / "frame1_region_waterfill_runtime_patch"
        / "local_component_spot_check"
        / "local_cpu_contest_drift_eureka.json",
        {
            "local_score": 0.1920003362662307,
            "auth_frontier_score": 0.19198533626623068,
            "eureka_trigger": False,
            "recommended_action": "observe_only",
            **FALSE_AUTHORITY,
        },
    )
    _write_json(
        root_a
        / "scorer_region_exact_ready_bridge_report.json",
        {
            "schema": "scorer_region_exact_ready_bridge_report.v1",
            "blockers": [
                "local_cpu_eureka_trigger_false",
                "local_cpu_score_not_below_auth_frontier",
            ],
            "rows": [
                {
                    "local_cpu_gate": {
                        "blockers": [
                            "local_cpu_eureka_trigger_false",
                            "local_cpu_score_not_below_auth_frontier",
                        ],
                        **FALSE_AUTHORITY,
                    },
                    **FALSE_AUTHORITY,
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    _write_json(
        root_a
        / "frame1_region_waterfill_runtime_patch"
        / "local_component_spot_check"
        / "scorer_response_dataset.json",
        {
            "schema": "scorer_response_dataset.v1",
            "rows": [
                {
                    "delta_vs_baseline_score": -0.001,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    root_b.mkdir(parents=True)

    report = build_scorer_region_selector_cascade_campaign_report(
        repo_root=tmp_path,
        variant_roots={"a": root_a, "b": root_b},
    )

    assert report["schema"] == SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA
    assert report["variant_count"] == 2
    assert report["completed_learning_variant_count"] == 1
    assert report["best_variant_id"] == "a"
    assert report["best_variant_selection_basis"] == "local_cpu_gate_failed"
    assert report["rows"][0]["best_dataset_delta_vs_baseline_score"] == -0.001
    assert report["rows"][0]["candidate_passed_local_cpu_gate"] is False
    assert "local_cpu_score_not_below_auth_frontier" in report["blockers"]
    assert report["rows"][0]["output_change_observed"] is True
    assert report["mlx_positive_full_cpu_negative_split_count"] == 1
    update = report["posterior_acquisition_updates"][0]
    assert update["operator_position_group"] == ["P19", "P18", "P11", "P15"]
    assert update["mlx_acquisition_positive"] is True
    assert update["full_cpu_negative"] is True
    assert update["mlx_positive_full_cpu_negative_split"] is True
    assert update["cpu_pre_gate_status"] == "failed"
    assert update["byte_pressure"]["saved_bytes"] == 10
    assert update["posterior_budget_routing_decision"] == (
        "demote_grouped_stack_and_remeasure_cpu_before_budget"
    )
    assert update["budget_spend_allowed"] is False
    assert update["ready_for_exact_eval_dispatch"] is False
    assert report["score_claim"] is False
