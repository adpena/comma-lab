from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (
    SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA,
    SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA,
    SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA,
    SCORER_REGION_SELECTOR_CASCADE_SELECTION_MANIFEST_SCHEMA,
    build_scorer_region_selector_cascade_acquisition_policy,
    build_scorer_region_selector_cascade_campaign_queue,
    build_scorer_region_selector_cascade_campaign_report,
    discover_scorer_region_selector_cascade_variant_roots,
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
        mlx_first_acquisition=True,
        mlx_cpu_gate_max_score_delta=0.01,
        include_scorer_response_dataset=True,
        include_local_component_retention_plan=True,
        scorer_response_baseline_score=0.1919853363,
        scorer_response_baseline_archive_bytes=178493,
        max_concurrency_local_cpu=2,
        max_concurrency_local_mlx=1,
    )

    assert queue["metadata"]["schema"] == SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_QUEUE_METADATA_SCHEMA
    assert queue["metadata"]["variant_count"] == 2
    assert queue["metadata"]["exact_auth_policy"]["mlx_first_acquisition"] is True
    assert queue["metadata"]["exact_auth_policy"]["cpu_gate_only_after_mlx_spend_gate"] is True
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 1
    variant_experiments = [
        exp
        for exp in queue["experiments"]
        if exp["id"]
        not in {
            "archive_master_gradient_hydration",
            "campaign_harvest",
            "campaign_acquisition_policy",
            "campaign_dynamic_followup_queue",
        }
    ]
    assert len(variant_experiments) == 2
    hydration_experiment = next(
        exp for exp in queue["experiments"] if exp["id"] == "archive_master_gradient_hydration"
    )
    assert hydration_experiment["steps"][0]["id"] == "hydrate_archive_specific_master_gradient"
    for experiment in variant_experiments:
        assert experiment["metadata"]["schema"] == "scorer_region_selector_cascade_variant.v1"
        assert "grouped-cascade-campaign" in experiment["tags"]
        step_ids = [step["id"] for step in experiment["steps"]]
        assert "materialize_p19_posenet_null_pairs" in step_ids
        by_id = {step["id"]: step for step in experiment["steps"]}
        assert by_id["materialize_p19_posenet_null_pairs"]["requires"] == [
            "archive_master_gradient_hydration.hydrate_archive_specific_master_gradient"
        ]
        assert by_id["local_cpu_component_spot_check"]["requires"] == ["mlx_cpu_spend_gate"]
        assert "build_scorer_response_dataset" in step_ids
        patch_step = next(
            step
            for step in experiment["steps"]
            if step["id"] == "materialize_frame1_region_waterfill_runtime_patch"
        )
        assert "--selected-archive-chain-report" in patch_step["command"]
    harvest_experiment = next(exp for exp in queue["experiments"] if exp["id"] == "campaign_harvest")
    harvest = harvest_experiment["steps"][0]
    assert harvest["id"] == "harvest_campaign_learning_surface"
    assert len(harvest["requires"]) == 2
    assert all(requirement.endswith(".plan_local_component_artifact_retention") for requirement in harvest["requires"])
    policy_experiment = next(
        exp for exp in queue["experiments"] if exp["id"] == "campaign_acquisition_policy"
    )
    policy_step = policy_experiment["steps"][0]
    assert policy_step["id"] == "build_campaign_acquisition_policy"
    assert policy_step["requires"] == ["campaign_harvest.harvest_campaign_learning_surface"]
    assert "--master-gradient-tensor" in policy_step["command"]
    assert "--archive-master-gradient-hydration" in policy_step["command"]
    assert "--pixel-gradient-cache" in policy_step["command"]
    followup_experiment = next(
        exp for exp in queue["experiments"] if exp["id"] == "campaign_dynamic_followup_queue"
    )
    followup_step = followup_experiment["steps"][0]
    assert followup_step["id"] == "compile_dynamic_followup_campaign_queue"
    assert followup_step["requires"] == [
        "campaign_acquisition_policy.build_campaign_acquisition_policy"
    ]
    assert "tools/build_scorer_region_selector_cascade_queue_from_policy.py" in followup_step["command"]
    assert "--mlx-first-acquisition" in followup_step["command"]
    assert queue["metadata"]["full_video_mlx_first_acquisition"] is True
    assert queue["metadata"]["dynamic_followup_queue_path"].endswith(
        "dynamic_followup_campaign_queue.json"
    )


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
    assert report["aggregate_learning"]["local_cpu_observed_count"] == 1
    assert report["aggregate_learning"]["local_cpu_passed_gate_count"] == 0
    assert report["aggregate_learning"]["local_cpu_all_observed_failed_gate"] is True
    assert report["aggregate_learning"]["recommended_next_queue_policy"] == (
        "acquisition_first_or_cpu_gate_only_no_post_cpu_mlx"
    )
    assert report["aggregate_learning"]["posterior_routing_decision"] == (
        "demote_post_cpu_mlx_for_current_operator_family_until_acquisition_model_changes"
    )
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


def test_campaign_variant_root_discovery_uses_immediate_child_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    (root / "variant_b").mkdir(parents=True)
    (root / "variant_a").mkdir(parents=True)
    (root / ".scratch").mkdir(parents=True)
    (root / "campaign_report.json").write_text("{}", encoding="utf-8")

    discovered = discover_scorer_region_selector_cascade_variant_roots(
        repo_root=tmp_path,
        variant_root_dir=root,
    )

    assert list(discovered) == ["variant_a", "variant_b"]
    assert discovered == {
        "variant_a": "campaign/variant_a",
        "variant_b": "campaign/variant_b",
    }


def test_acquisition_policy_consumes_cpu_negative_and_gradient_priors(tmp_path: Path) -> None:
    import numpy as np

    tensor = tmp_path / "mg.npy"
    arr = np.zeros((4, 3, 3), dtype=np.float64)
    arr[:, 0, 1] = 0.01
    arr[:, 1, 1] = 0.001
    arr[:, 2, 0] = 0.02
    np.save(tensor, arr)

    pixel_cache = tmp_path / "pixel.npz"
    seg = np.ones((2, 4, 4), dtype=np.float32)
    pose = np.ones((2, 4, 4), dtype=np.float32)
    seg[0, :2, :2] = 0.0
    pose[0, :2, :2] = 0.0
    np.savez(pixel_cache, seg_grads=seg, pose_grads=pose)

    report = {
        "schema": SCORER_REGION_SELECTOR_CASCADE_CAMPAIGN_REPORT_SCHEMA,
        "variant_count": 1,
        "completed_learning_variant_count": 1,
        "best_variant_id": "nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch",
        "best_variant_selection_basis": "local_cpu_gate_failed",
        "aggregate_learning": {
            "mlx_positive_full_cpu_negative_split_count": 1,
            "local_cpu_observed_count": 1,
            "local_cpu_passed_gate_count": 0,
            **FALSE_AUTHORITY,
        },
        "rows": [
            {
                "variant_id": "nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch",
                "local_cpu_present": True,
                "local_cpu_delta_vs_auth_frontier": 0.000015,
                "candidate_passed_local_cpu_gate": False,
                "cumulative_rate_saved_bytes_vs_source": 10,
                "best_dataset_delta_vs_baseline_score": -0.00002,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        **FALSE_AUTHORITY,
    }

    policy = build_scorer_region_selector_cascade_acquisition_policy(
        repo_root=tmp_path,
        campaign_report=report,
        master_gradient_tensor_path=tensor,
        pixel_gradient_cache_path=pixel_cache,
    )

    assert policy["schema"] == SCORER_REGION_SELECTOR_CASCADE_ACQUISITION_POLICY_SCHEMA
    assert policy["next_queue_policy"]["mode"] == "vectorized_mlx_acquisition_then_cpu_gate_only"
    assert policy["master_gradient_prior"]["available"] is True
    assert policy["master_gradient_prior"]["shape"] == [4, 3, 3]
    assert policy["pixel_gradient_prior"]["available"] is True
    assert policy["pixel_gradient_prior"]["shape"] == [2, 4, 4]
    selection = policy["selection_manifest"]
    assert selection["schema"] == SCORER_REGION_SELECTOR_CASCADE_SELECTION_MANIFEST_SCHEMA
    assert selection["selection_ready"] is False
    assert "archive_specific_master_gradient_anchor_missing_for_current_campaign" in selection[
        "selection_ready_blockers"
    ]
    assert "pixel_gradient_cache_is_partial_sample_not_full_contest_video" in selection[
        "selection_ready_blockers"
    ]
    assert "master_gradient_sha256_missing_or_skipped" in selection["selection_ready_blockers"]
    assert selection["pose_pair_source"] == (
        "blocked_until_archive_specific_master_gradient_manifest"
    )
    assert selection["region_source"] == "blocked_until_full_video_pixel_gradient_manifest"
    assert policy["next_queue_policy"]["selection_manifest_ready"] is False
    assert policy["next_queue_policy"]["preferred_next_grid"]["pose_pair_source"] == (
        "blocked_until_archive_specific_master_gradient_manifest"
    )
    assert policy["rate_credit_rows"][0]["estimated_distortion_spend_score_units"] > 0.0
    assert policy["contest_space_action_functional"]["row_count"] == 1
    assert policy["contest_space_action_functional"]["local_gate_passed_count"] == 0
    assert policy["score_claim"] is False
    assert "current_operator_family_all_observed_local_cpu_rows_failed" in policy["blockers"]
