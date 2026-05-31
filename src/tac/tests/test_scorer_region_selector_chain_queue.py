from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.scheduler.scorer_region_exact_ready_bridge import (
    SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA,
    build_scorer_region_exact_ready_bridge,
)
from comma_lab.scheduler.scorer_region_selector_chain_queue import (
    SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
    SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
    build_scorer_region_selector_chain_context,
    build_scorer_region_selector_chain_queue,
    build_scorer_region_selector_chain_report,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import ARCHIVE_ZIP_REPACK_SCHEMA
from tac.optimization.scorer_region_operator_contract import (
    SCORER_REGION_OPERATOR_CONTRACT_SCHEMA,
)
from tac.optimization.scorer_region_waterfill import (
    DISTORTION_BUDGET_ATTACK_PLAN_SCHEMA,
    FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA,
    P18_SEGNET_REGION_WATERFILL_SCHEMA,
    P19_POSENET_NULL_PAIRS_SCHEMA,
    build_frame1_region_waterfill_runtime_patch,
    build_p18_segnet_region_waterfill,
    build_p19_posenet_null_pairs,
    build_receiver_closed_distortion_budget_attack_plan,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
)


def _write_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", b"selector-stream")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _source_submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    _write_zip(submission / "archive.zip")
    return submission


def _waterfill_work_order(tmp_path: Path) -> Path:
    path = tmp_path / "repair_budget_waterfill_work_order.json"
    _write_json(
        path,
        {
            "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
            "repair_cascade_opportunity_rows": [
                {
                    "schema": "frontier_rate_attack_repair_cascade_opportunity_row.v1",
                    "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
                    "label": "Cascade C",
                    "pipeline_position": "P19+P18+P11",
                    "targeted_positions": [
                        {"position_id": "P19", "entropy_surface": "PoseNet"},
                        {"position_id": "P18", "entropy_surface": "SegNet"},
                        {"position_id": "P11", "entropy_surface": "selector"},
                    ],
                    "blockers": [],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    return path


def test_chain_context_preserves_upstream_blockers_without_score_authority(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    work_order = _waterfill_work_order(tmp_path)
    parity = tmp_path / "parity.json"
    _write_json(parity, {"passed": True, **FALSE_AUTHORITY})

    context = build_scorer_region_selector_chain_context(
        repo_root=tmp_path,
        source_submission_dir=submission,
        source_waterfill_work_order=work_order,
        full_frame_inflate_parity_proof=parity,
    )

    assert context["schema"] == SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA
    assert context["operator_contract"]["schema"] == SCORER_REGION_OPERATOR_CONTRACT_SCHEMA
    assert context["operator_contract"]["chain_position_order"] == ["P19", "P18", "P11", "P15"]
    assert context["p11_rate_anchor_can_run"] is True
    assert context["p18_p19_upstream_ready"] is False
    assert "p19_posenet_null_pairs_missing" in context["blockers"]
    assert "p18_segnet_region_masks_missing" in context["blockers"]
    assert context["score_claim"] is False
    assert context["ready_for_exact_eval_dispatch"] is False


def test_chain_queue_orders_context_selector_repack_report(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    work_order = _waterfill_work_order(tmp_path)
    parity = tmp_path / "parity.json"
    _write_json(parity, {"passed": True, **FALSE_AUTHORITY})

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        source_waterfill_work_order=work_order,
        full_frame_inflate_parity_proof=parity,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps] == [
        "build_p18_p19_chain_context",
        "materialize_p11_selector_context_recode",
        "materialize_p15_archive_zip_repack",
        "emit_composed_chain_report",
    ]
    assert steps[1]["requires"] == ["build_p18_p19_chain_context"]
    assert steps[2]["requires"] == ["materialize_p11_selector_context_recode"]
    assert steps[3]["requires"] == ["materialize_p15_archive_zip_repack"]
    assert "--chain-parent-artifact" in steps[1]["command"]
    assert "archive_zip_repack_v1" in steps[2]["command"]
    assert queue["metadata"]["operator_contract"]["schema"] == SCORER_REGION_OPERATOR_CONTRACT_SCHEMA
    assert (
        queue["metadata"]["operator_contract"]["composition_law"]["selected_survivor_rule"]
        .startswith("use_P15_archive_zip_repack_only_when_rate_positive")
    )
    assert queue["metadata"]["ready_for_exact_eval_dispatch"] is False


def test_chain_queue_can_materialize_upstream_p18_p19_artifacts(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    pose_null = tmp_path / "pose_null.json"
    soft16 = tmp_path / "soft16.npy"
    soft256 = tmp_path / "soft256.npy"
    _write_json(pose_null, {"analysis": {"pose_null_decile": []}, **FALSE_AUTHORITY})
    soft16.write_bytes(b"fake-npy")
    soft256.write_bytes(b"fake-npy")

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        pose_null_modes_artifact=pose_null,
        segnet_softmax_16=soft16,
        segnet_softmax_256=soft256,
        materialize_upstream_artifacts=True,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps] == [
        "materialize_p19_posenet_null_pairs",
        "materialize_p18_segnet_region_waterfill",
        "build_p18_p19_chain_context",
        "materialize_p11_selector_context_recode",
        "materialize_p15_archive_zip_repack",
        "emit_composed_chain_report",
        "emit_receiver_closed_distortion_budget_attack_plan",
    ]
    assert steps[2]["requires"] == ["materialize_p18_segnet_region_waterfill"]
    assert queue["metadata"]["materialize_upstream_artifacts"] is True


def test_chain_queue_can_materialize_receiver_patch(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    p18 = tmp_path / "p18.json"
    _write_json(
        p18,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "selected_pair_count": 1,
            "rows": [
                {
                    "pair_id": 0,
                    "regions256": [
                        {
                            "box": {"x0": 0.0, "y0": 0.0, "x1": 0.25, "y1": 0.25},
                            "class_id": 0,
                        }
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        segnet_region_masks=p18,
        materialize_receiver_patch=True,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = [step["id"] for step in queue["experiments"][0]["steps"]]
    patch_command = queue["experiments"][0]["steps"][-2]["command"]
    assert steps[-2:] == [
        "materialize_frame1_region_waterfill_runtime_patch",
        "emit_scorer_region_exact_ready_bridge_inputs",
    ]
    assert "--selected-archive-chain-report" in patch_command
    assert "--candidate-archive" not in patch_command
    assert queue["metadata"]["materialize_receiver_patch"] is True
    assert (
        queue["experiments"][0]["steps"][-2]["postconditions"][0]["equals"]
        == FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA
    )
    assert (
        queue["experiments"][0]["steps"][-1]["postconditions"][0]["equals"]
        == SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA
    )


def test_chain_queue_can_prove_receiver_patch_output_change_before_bridge(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    p18 = tmp_path / "p18.json"
    _write_json(
        p18,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "rows": [
                {
                    "pair_id": 0,
                    "regions256": [
                        {
                            "box": {"x0": 0.0, "y0": 0.0, "x1": 0.25, "y1": 0.25},
                            "class_id": 0,
                        }
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        segnet_region_masks=p18,
        materialize_receiver_patch=True,
        prove_receiver_patch_output_change=True,
        receiver_patch_output_change_file_list_entries=("0.raw", "1.raw"),
        receiver_patch_output_change_expected_file_list_sha256="a" * 64,
        receiver_patch_output_change_expected_entry_count=2,
        receiver_patch_output_change_file_list_source="tests/full_frame_file_list.txt",
        receiver_patch_output_change_contest_full_sample_claim=True,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps][-3:] == [
        "materialize_frame1_region_waterfill_runtime_patch",
        "prove_receiver_patch_full_frame_output_change",
        "emit_scorer_region_exact_ready_bridge_inputs",
    ]
    proof_step = steps[-2]
    bridge_step = steps[-1]
    assert proof_step["requires"] == ["materialize_frame1_region_waterfill_runtime_patch"]
    assert proof_step["postconditions"][0]["equals"] == "shell_inflate_output_change_proof_v1"
    assert "--left-selected-archive-chain-report" in proof_step["command"]
    assert "--left-archive" not in proof_step["command"]
    assert proof_step["command"].count("--file-list-entry") == 2
    assert "--require-output-change" in proof_step["command"]
    assert "--contest-full-sample-claim" in proof_step["command"]
    assert bridge_step["requires"] == ["prove_receiver_patch_full_frame_output_change"]
    assert "--shell-inflate-output-change-proof" in bridge_step["command"]
    assert (
        queue["metadata"]["receiver_patch_output_change_proof_schema"]
        == "shell_inflate_output_change_proof_v1"
    )
    assert queue["metadata"]["operator_contract"]["receiver_contract"]["enabled"] is True


def test_chain_queue_can_close_local_component_learning_loop(
    tmp_path: Path,
) -> None:
    submission = _source_submission(tmp_path)
    p18 = tmp_path / "p18.json"
    _write_json(
        p18,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "rows": [
                {
                    "pair_id": 0,
                    "regions256": [
                        {
                            "box": {"x0": 0.0, "y0": 0.0, "x1": 0.25, "y1": 0.25},
                            "class_id": 0,
                        }
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )

    queue = build_scorer_region_selector_chain_queue(
        repo_root=tmp_path,
        queue_id="chain_q",
        source_submission_dir=submission,
        output_root=tmp_path / "chain_out",
        full_frame_inflate_parity_proof=tmp_path / "parity.json",
        segnet_region_masks=p18,
        materialize_receiver_patch=True,
        prove_receiver_patch_output_change=True,
        receiver_patch_output_change_file_list_entries=("0.raw",),
        receiver_patch_output_change_expected_file_list_sha256="a" * 64,
        receiver_patch_output_change_expected_entry_count=1,
        receiver_patch_output_change_file_list_source="tests/full_frame_file_list.txt",
        include_local_component_loop=True,
        include_mlx_component_response=True,
        include_scorer_response_dataset=True,
        include_local_component_retention_plan=True,
        scorer_response_baseline_score=0.1919853363,
        scales=(64,),
        alphas=(1,),
        codec_families=("fec10_adaptive_blend",),
    )

    steps = queue["experiments"][0]["steps"]
    step_ids = [step["id"] for step in steps]
    assert step_ids[-7:] == [
        "local_cpu_component_spot_check",
        "local_cpu_contest_drift_eureka",
        "build_mlx_component_cache",
        "local_mlx_component_response",
        "build_scorer_response_dataset",
        "plan_local_component_artifact_retention",
        "emit_scorer_region_exact_ready_bridge_inputs",
    ]
    by_id = {step["id"]: step for step in steps}
    assert by_id["local_cpu_component_spot_check"]["requires"] == [
        "prove_receiver_patch_full_frame_output_change"
    ]
    assert by_id["build_mlx_component_cache"]["requires"] == [
        "local_cpu_component_spot_check",
        "local_cpu_contest_drift_eureka",
    ]
    assert by_id["build_scorer_response_dataset"]["requires"] == [
        "local_mlx_component_response"
    ]
    assert by_id["plan_local_component_artifact_retention"]["requires"] == [
        "build_scorer_response_dataset"
    ]
    assert by_id["emit_scorer_region_exact_ready_bridge_inputs"]["requires"] == [
        "plan_local_component_artifact_retention"
    ]
    bridge_command = by_id["emit_scorer_region_exact_ready_bridge_inputs"]["command"]
    assert "--local-cpu-advisory" in bridge_command
    assert "--local-cpu-eureka" in bridge_command
    assert "--local-mlx-response" in bridge_command
    assert "--scorer-response-dataset" in bridge_command
    assert "--reuse-valid-json-out" in by_id["local_cpu_component_spot_check"]["command"]
    assert "--reuse-valid-cache" in by_id["build_mlx_component_cache"]["command"]
    assert "--consumer-routing-json-out" in by_id["build_scorer_response_dataset"]["command"]
    assert "tools/compact_experiment_artifacts.py" in by_id[
        "plan_local_component_artifact_retention"
    ]["command"]
    retention_command = by_id["plan_local_component_artifact_retention"]["command"]
    assert "mlx_scorer_input_cache" in retention_command
    assert queue["metadata"]["include_local_component_loop"] is True
    assert queue["metadata"]["include_mlx_component_response"] is True
    assert queue["metadata"]["include_scorer_response_dataset"] is True
    assert queue["metadata"]["include_local_component_retention_plan"] is True


def test_chain_report_selects_repack_only_when_positive_and_receiver_closed(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "chain_context.json"
    selector_path = tmp_path / "selector.json"
    repack_path = tmp_path / "repack.json"
    context = {
        "schema": SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
        "chain_label": "cascade-c",
        "p18_p19_upstream_ready": True,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    selector = {
        "schema": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
        "candidate_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "source_archive": {"path": "source/archive.zip", "bytes": 100, "sha256": "b" * 64},
        "selected_recode": {"saved_bytes": 10, "codec_family": "fec10_adaptive_blend"},
        "receiver_contract_satisfied": True,
        "readiness_blockers": [],
        **FALSE_AUTHORITY,
    }
    repack = {
        "schema": ARCHIVE_ZIP_REPACK_SCHEMA,
        "candidate_archive": {"path": "repack/archive.zip", "bytes": 86, "sha256": "c" * 64},
        "source_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "selected_repack": {"saved_bytes": 4, "strategy": "greedy", "plan_key": "deflated:9"},
        "receiver_contract_satisfied": True,
        "readiness_blockers": [],
        **FALSE_AUTHORITY,
    }
    _write_json(context_path, context)
    _write_json(selector_path, selector)
    _write_json(repack_path, repack)

    report = build_scorer_region_selector_chain_report(
        repo_root=tmp_path,
        chain_context=context,
        chain_context_path=context_path,
        selector_manifest=selector,
        selector_manifest_path=selector_path,
        repack_manifest=repack,
        repack_manifest_path=repack_path,
    )

    assert report["schema"] == SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA
    assert report["selected_local_survivor_stage"] == "P15_archive_zip_repack"
    assert report["operator_contract"]["schema"] == SCORER_REGION_OPERATOR_CONTRACT_SCHEMA
    assert report["cumulative_rate_saved_bytes_vs_source"] == 14
    assert report["selected_local_survivor_archive"]["sha256"] == "c" * 64
    assert report["blockers"] == report["readiness_blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_chain_report_ignores_nonpositive_repack_blocker_when_selector_saved_bytes(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "chain_context.json"
    selector_path = tmp_path / "selector.json"
    repack_path = tmp_path / "repack.json"
    context = {
        "schema": SCORER_REGION_SELECTOR_CHAIN_CONTEXT_SCHEMA,
        "chain_label": "cascade-c",
        "p18_p19_upstream_ready": True,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    selector = {
        "schema": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
        "candidate_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "source_archive": {"path": "source/archive.zip", "bytes": 100, "sha256": "b" * 64},
        "receiver_contract_satisfied": True,
        "readiness_blockers": ["candidate_requires_exact_auth_eval_before_promotion"],
        **FALSE_AUTHORITY,
    }
    repack = {
        "schema": ARCHIVE_ZIP_REPACK_SCHEMA,
        "candidate_archive": {"path": "repack/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "source_archive": {"path": "selector/archive.zip", "bytes": 90, "sha256": "a" * 64},
        "selected_repack": {"saved_bytes": 0, "strategy": "uniform"},
        "receiver_contract_satisfied": True,
        "readiness_blockers": ["candidate_not_rate_positive"],
        **FALSE_AUTHORITY,
    }
    _write_json(context_path, context)
    _write_json(selector_path, selector)
    _write_json(repack_path, repack)

    report = build_scorer_region_selector_chain_report(
        repo_root=tmp_path,
        chain_context=context,
        chain_context_path=context_path,
        selector_manifest=selector,
        selector_manifest_path=selector_path,
        repack_manifest=repack,
        repack_manifest_path=repack_path,
    )

    assert report["selected_local_survivor_stage"] == "P11_selector_context_recode"
    assert report["cumulative_rate_saved_bytes_vs_source"] == 10
    assert "candidate_not_rate_positive" not in report["readiness_blockers"]


def test_p18_p19_artifacts_promote_rate_credit_without_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    submission = _source_submission(tmp_path)
    (submission / "inflate.py").write_text(
        'FEC6_FIXED_K16_MODE_IDS = ("none", "frame0_blue_chroma_amp_1", "other")\n',
        encoding="utf-8",
    )
    pose_null = tmp_path / "pose_null.json"
    _write_json(
        pose_null,
        {
            "analysis": {
                "ranked_top_n_by_abs_pose": [
                    {"mode_id": "frame0_blue_chroma_amp_1", "abs_pose_delta": 0.2}
                ]
            },
            **FALSE_AUTHORITY,
        },
    )

    import tac.optimization.scorer_region_waterfill as waterfill

    monkeypatch.setattr(waterfill, "decode_selector_codes", lambda _source: [2, 0, 1, 2])
    p19 = build_p19_posenet_null_pairs(
        repo_root=tmp_path,
        source_submission_dir=submission,
        pose_null_modes_artifact=pose_null,
        null_fraction=0.5,
    )
    assert p19["schema"] == P19_POSENET_NULL_PAIRS_SCHEMA
    assert p19["selected_pair_ids"] == [1, 2]
    assert p19["score_claim"] is False

    p19_path = tmp_path / "p19.json"
    _write_json(p19_path, p19)
    soft16 = tmp_path / "soft16.npy"
    soft256 = tmp_path / "soft256.npy"
    arr16 = [[[[0.2, 0.2, 0.2, 0.2, 0.2]] * 16][0]] * 4
    arr256 = [[[[0.2, 0.2, 0.2, 0.2, 0.2]] * 256][0]] * 4
    import numpy as np

    np.save(soft16, np.asarray(arr16, dtype=np.float32))
    np.save(soft256, np.asarray(arr256, dtype=np.float32))
    p18 = build_p18_segnet_region_waterfill(
        repo_root=tmp_path,
        posenet_null_pairs=p19_path,
        segnet_softmax_16=soft16,
        segnet_softmax_256=soft256,
        top_regions_per_pair=2,
    )
    assert p18["schema"] == P18_SEGNET_REGION_WATERFILL_SCHEMA
    assert p18["selected_pair_count"] == 2
    assert len(p18["rows"][0]["regions256"]) == 2
    assert p18["ready_for_exact_eval_dispatch"] is False

    report_path = tmp_path / "report.json"
    p18_path = tmp_path / "p18.json"
    _write_json(p18_path, p18)
    _write_json(
        report_path,
        {
            "schema": SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
            "cumulative_rate_saved_bytes_vs_source": 16,
            "readiness_blockers": ["contest_auth_eval_required_before_score_or_promotion_claim"],
            **FALSE_AUTHORITY,
        },
    )
    plan = build_receiver_closed_distortion_budget_attack_plan(
        repo_root=tmp_path,
        chain_report=report_path,
        posenet_null_pairs=p19_path,
        segnet_region_waterfill=p18_path,
    )
    assert plan["schema"] == DISTORTION_BUDGET_ATTACK_PLAN_SCHEMA
    assert plan["rate_saved_bytes"] == 16
    assert plan["budget_pair_count"] == 2
    assert plan["budget_spend_allowed"] is False


def test_frame1_region_waterfill_runtime_patch_materializes_submission(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    (submission / "src").mkdir(parents=True)
    (submission / "archive.zip").write_bytes(b"fake")
    (submission / "runtime_consumption_proof.json").write_text(
        '{"schema":"stale_source_runtime_consumption_proof"}\n',
        encoding="utf-8",
    )
    (submission / "inflate.py").write_text(
        "from model import HNeRVDecoder  # type: ignore[import-not-found]\n"
        "def f():\n"
        "            rounded = apply_pr101_selector_to_frames(\n"
        "                rounded,\n"
        "                selector_kind,\n"
        "                selector_codes,\n"
        "                selector_specs,\n"
        "                pair_start=i,\n"
        "            )\n"
        "            frames = rounded.to(torch.uint8)\n",
        encoding="utf-8",
    )
    p18 = tmp_path / "p18.json"
    _write_json(
        p18,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "rows": [
                {
                    "pair_id": 7,
                    "regions256": [
                        {
                            "box": {"x0": 0.5, "y0": 0.25, "x1": 0.75, "y1": 0.5}
                        }
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )

    payload = build_frame1_region_waterfill_runtime_patch(
        repo_root=tmp_path,
        source_submission_dir=submission,
        segnet_region_waterfill=p18,
        output_submission_dir=tmp_path / "patched",
        overwrite=True,
    )

    assert payload["schema"] == FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA
    assert payload["patched_pair_count"] == 1
    assert (tmp_path / "patched" / "src" / "region_waterfill_patch.py").is_file()
    assert not (tmp_path / "patched" / "runtime_consumption_proof.json").exists()
    assert payload["runtime_consumption_proof_present"] is False
    inflate = (tmp_path / "patched" / "inflate.py").read_text(encoding="utf-8")
    assert "from region_waterfill_patch import apply_region_waterfill" in inflate
    assert "apply_region_waterfill(rounded, pair_start=i)" in inflate
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_scorer_region_exact_ready_bridge_blocks_without_runtime_content_tree(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    (submission / "src").mkdir(parents=True)
    (submission / "archive.zip").write_bytes(b"fake")
    (submission / "inflate.py").write_text(
        "from model import HNeRVDecoder  # type: ignore[import-not-found]\n"
        "def f():\n"
        "            rounded = apply_pr101_selector_to_frames(\n"
        "                rounded,\n"
        "                selector_kind,\n"
        "                selector_codes,\n"
        "                selector_specs,\n"
        "                pair_start=i,\n"
        "            )\n",
        encoding="utf-8",
    )
    p18 = tmp_path / "p18.json"
    _write_json(
        p18,
        {
            "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
            "rows": [
                {
                    "pair_id": 0,
                    "regions256": [
                        {"box": {"x0": 0.0, "y0": 0.0, "x1": 0.25, "y1": 0.25}}
                    ],
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    patch_manifest = build_frame1_region_waterfill_runtime_patch(
        repo_root=tmp_path,
        source_submission_dir=submission,
        segnet_region_waterfill=p18,
        output_submission_dir=tmp_path / "patched",
        overwrite=True,
    )
    patch_manifest_path = tmp_path / "patch_manifest.json"
    _write_json(patch_manifest_path, patch_manifest)
    chain_report_path = tmp_path / "chain_report.json"
    _write_json(
        chain_report_path,
        {
            "schema": SCORER_REGION_SELECTOR_CHAIN_REPORT_SCHEMA,
            "chain_label": "cascade-c",
            "selected_local_survivor_archive": patch_manifest["candidate_archive"],
            "cumulative_rate_saved_bytes_vs_source": 0,
            "readiness_blockers": [
                "contest_auth_eval_required_before_score_or_promotion_claim"
            ],
            **FALSE_AUTHORITY,
        },
    )

    bridge = build_scorer_region_exact_ready_bridge(
        chain_report_path=chain_report_path,
        receiver_patch_manifest_path=patch_manifest_path,
        repo_root=tmp_path,
    )

    report = bridge["bridge_report"]
    source_queue = bridge["source_optimizer_queue"]
    blocked_queue = bridge["blocked_exact_ready_queue"]
    assert report["schema"] == SCORER_REGION_EXACT_READY_BRIDGE_REPORT_SCHEMA
    assert report["archive_custody_proven_count"] == 1
    assert report["runtime_patch_custody_proven_count"] == 1
    assert report["runtime_content_tree_custody_proven_count"] == 0
    assert source_queue["top_k"][0]["score_affecting_runtime_changed"] is True
    assert source_queue["dispatch_ready_count"] == 0
    assert blocked_queue["dispatch_ready_count"] == 0
    assert "receiver_patch_inflate_sh_missing" in report["blockers"]
    assert "shell_inflate_output_change_proof_missing" in report["blockers"]

    output_change_proof = tmp_path / "shell_inflate_output_change.json"
    _write_json(
        output_change_proof,
        {
            "schema": "shell_inflate_output_change_proof_v1",
            "output_change_observed": True,
            "raw_shape_preserving_output_change_observed": True,
            "full_frame_output_change_claim": True,
            "contest_full_sample_change_claim": True,
            "differing_byte_count": 8,
            "differing_output_count": 1,
            "blockers": [],
            **FALSE_AUTHORITY,
        },
    )
    bridge_with_change_proof = build_scorer_region_exact_ready_bridge(
        chain_report_path=chain_report_path,
        receiver_patch_manifest_path=patch_manifest_path,
        shell_inflate_output_change_proof_path=output_change_proof,
        repo_root=tmp_path,
    )
    report_with_change_proof = bridge_with_change_proof["bridge_report"]
    row = report_with_change_proof["rows"][0]["bridge_source_queue_row"]
    assert report_with_change_proof["output_change_proof_proven_count"] == 1
    assert (
        report_with_change_proof["operator_contract"]["receiver_contract"]["proof_required"]
        == "shape_preserving_full_frame_shell_inflate_output_change"
    )
    assert row["operator_contract"]["schema"] == SCORER_REGION_OPERATOR_CONTRACT_SCHEMA
    assert "shell_inflate_output_change_proof_missing" not in report_with_change_proof[
        "blockers"
    ]
    assert "inflated_output_change_proof_required_before_budget_spend_claim" not in (
        report_with_change_proof["blockers"]
    )
    assert row["full_frame_output_change_proven"] is True
    assert row["contest_full_sample_change_proven"] is True

    local_cpu_advisory = tmp_path / "local_cpu_advisory.json"
    _write_json(
        local_cpu_advisory,
        {
            "score_axis": "cpu_advisory",
            "canonical_score": 0.1920003362662307,
            "score_recomputed_from_components": 0.1920003362662307,
            **FALSE_AUTHORITY,
        },
    )
    local_cpu_eureka = tmp_path / "local_cpu_eureka.json"
    _write_json(
        local_cpu_eureka,
        {
            "schema": "local_cpu_contest_drift_eureka.v1",
            "local_score": 0.1920003362662307,
            "auth_frontier_score": 0.19198533626623068,
            "eureka_trigger": False,
            "recommended_action": "observe_only",
            **FALSE_AUTHORITY,
        },
    )
    bridge_with_negative_cpu = build_scorer_region_exact_ready_bridge(
        chain_report_path=chain_report_path,
        receiver_patch_manifest_path=patch_manifest_path,
        shell_inflate_output_change_proof_path=output_change_proof,
        local_cpu_advisory_path=local_cpu_advisory,
        local_cpu_eureka_path=local_cpu_eureka,
        repo_root=tmp_path,
    )
    negative_report = bridge_with_negative_cpu["bridge_report"]
    assert "local_cpu_eureka_trigger_false" in negative_report["blockers"]
    assert "local_cpu_score_not_below_auth_frontier" in negative_report["blockers"]
    assert (
        negative_report["rows"][0]["local_cpu_gate"]["local_score"]
        == 0.1920003362662307
    )
