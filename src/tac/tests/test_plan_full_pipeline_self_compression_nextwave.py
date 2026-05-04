from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_full_pipeline_self_compression_nextwave.py"


def _load_planner(name: str = "_plan_full_pipeline_self_compression_nextwave_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, b"P6synthetic-payload")
    return path


def _eval_json() -> dict[str, Any]:
    return {
        "archive_size_bytes": 276_342,
        "avg_posenet_dist": 0.00049601,
        "avg_segnet_dist": 0.00061038,
        "n_samples": 600,
        "score_recomputed_from_components": 0.3154707273953505,
    }


def _profile_json(archive: Path) -> dict[str, Any]:
    segments = [
        ("masks.mkv", 219_472, "brotli_av1_obu", 223_385),
        ("renderer.bin", 55_965, "brotli_qzs3", 59_288),
        ("seg_tile_actions.bin", 116, "seg_tile_actions_delta_varint_v1", 160),
        ("optimized_poses.qp1", 677, "public_qp1_brotli", 1_140),
    ]
    return {
        "archives": [
            {
                "path": str(archive),
                "bytes": 276_342,
                "sha256": "0" * 64,
                "zip_global_overhead_bytes": 22,
                "zip_member_header_bytes": 78,
                "members": [
                    {
                        "name": "p",
                        "fixed_slice_or_payload_anatomy": {
                            "payload_format": (
                                "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
                            ),
                            "segments": [
                                {
                                    "name": name,
                                    "encoded_bytes": encoded,
                                    "codec": codec,
                                    "decoded_bytes_estimate": decoded,
                                    "compression_probe": {
                                        "best_probe": {
                                            "codec": "input_bytes",
                                            "bytes": encoded,
                                        }
                                    },
                                }
                                for name, encoded, codec, decoded in segments
                            ],
                        },
                    }
                ],
            }
        ],
        "promotion_eligible": False,
        "score_claim": False,
        "schema": "archive_bit_budget_profile_v1",
    }


def _artifact_paths(tmp_path: Path) -> dict[str, Path]:
    renderer_greenup = _write_json(
        tmp_path / "renderer_greenup.json",
        {
            "candidates": [
                {
                    "archive_bytes": 272_986,
                    "archive_path": "/workspace/pact/qzs3_rp2_qp1/archive.zip",
                    "archive_sha256": "a" * 64,
                    "promotion_eligible": False,
                    "score_claim": False,
                }
            ],
            "promotion_eligible": False,
            "score_claim": False,
            "schema": "trained_renderer_export_unlock_plan_v1",
        },
    )
    renderer_parity_summary = _write_json(
        tmp_path / "renderer_parity_summary.json",
        {
            "candidates": [
                {
                    "archive": "zero_fp4_all_fp4_0.075/archive.zip",
                    "archive_bytes": 273_722,
                    "byte_only_crosses_target": True,
                    "candidate_id": "zero_fp4_all_fp4_0.075",
                    "pose_safety": {
                        "output_parity": {
                            "aggregate": {
                                "max_abs_delta": 197.95,
                                "mean_abs_delta": 7.16,
                                "ok": False,
                                "rms_delta": 11.26,
                            }
                        }
                    },
                    "score_claim": False,
                }
            ],
            "promotion_eligible": False,
            "score_claim": False,
        },
    )
    renderer_parity_dispatch = _write_json(
        tmp_path / "renderer_parity_dispatch.json",
        {
            "candidate": {
                "archive": "zero_fp4_frame1_head_0.1/archive.zip",
                "archive_bytes": 275_900,
            },
            "recommendation": "do_not_dispatch_yet_safe_but_too_small",
        },
    )
    action_recommendations = _write_json(
        tmp_path / "action_recommendations.json",
        {
            "ranked_candidates": [
                {
                    "archive": "c067_pr75_actions_top40_drop40_add41_p6/archive.zip",
                    "archive_size_bytes": 276_341,
                    "decoded_stream_closure_ok": True,
                    "dispatch_recommendation": "local_cuda_exact_eval_optional_no_remote_dispatch",
                    "estimated_score_recomputed_from_c067_trace": 0.31546827267490696,
                    "name": "c067_pr75_actions_top40_drop40_add41_p6",
                }
            ]
        },
    )
    lossless = _write_json(
        tmp_path / "lossless.json",
        {
            "candidates": [
                {
                    "archive_bytes": 276_333,
                    "archive_path": "c082_p6_delta_varint_actions_stream_resweep/archive.zip",
                    "candidate_id": "c082_p6_delta_varint_actions_stream_resweep",
                    "noop": False,
                }
            ],
            "score_claim": False,
        },
    )
    return {
        "c088_lossless_repack": lossless,
        "pr75_action_recommendations": action_recommendations,
        "renderer_greenup": renderer_greenup,
        "renderer_parity_dispatch": renderer_parity_dispatch,
        "renderer_parity_summary": renderer_parity_summary,
    }


def test_c089_break_even_math_matches_frontier_gap() -> None:
    planner = _load_planner()

    score = planner._score_math(_eval_json(), 0.314)

    assert score["bytes_to_save_for_strict_target_at_unchanged_distortion"] == 2209
    assert score["target_archive_bytes_strict"] == 274_133
    assert score["score_after_byte_save"] < 0.314
    assert score["rate_score_per_byte"] == 25.0 / 37_545_489


def test_stream_screen_marks_generic_recompression_exhausted(tmp_path: Path) -> None:
    planner = _load_planner("_plan_full_pipeline_stream_screen_test")
    archive = _write_archive(tmp_path / "archive.zip")
    profile = _profile_json(archive)
    score = planner._score_math(_eval_json(), 0.314)

    rows = {row["name"]: row for row in planner._stream_screen(profile, score)}

    assert rows["masks.mkv"]["generic_nested_recompression_savings_bytes"] == 0
    assert rows["renderer.bin"]["max_stream_bytes_if_this_stream_alone_crosses_target"] == 53_756
    assert rows["seg_tile_actions.bin"]["stream_can_close_target_by_byte_count"] is False
    assert rows["payload_internal_header"]["current_encoded_bytes"] == 12
    assert rows["archive_overhead"]["current_encoded_bytes"] == 100


def test_build_plan_ranks_not_noop_byte_crossing_work_without_score_claim(
    tmp_path: Path,
) -> None:
    planner = _load_planner("_plan_full_pipeline_build_plan_test")
    archive = _write_archive(tmp_path / "archive.zip")
    eval_json = _write_json(tmp_path / "eval.json", _eval_json())
    profile_json = _write_json(tmp_path / "profile.json", _profile_json(archive))

    plan = planner.build_plan(
        archive=archive,
        eval_json=eval_json,
        profile_json=profile_json,
        artifact_paths=_artifact_paths(tmp_path),
    )

    assert plan["schema"] == "full_pipeline_self_compression_nextwave_plan_v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["remote_gpu_dispatch_performed"] is False
    assert plan["planning_constraints"]["candidate_archives_written"] is False
    assert plan["top5_recommendations"][0]["opportunity_id"] == (
        "renderer_trained_self_compression_c089_transplant"
    )
    assert plan["top5_recommendations"][0]["bytes_saved_vs_c089"] == 3356
    assert plan["top5_recommendations"][0]["projected_score_if_only_bytes_change"] < 0.314
    opportunities = {item["opportunity_id"]: item for item in plan["opportunities"]}
    assert opportunities["renderer_zero_fp4_recovery_training"]["status"] == (
        "do_not_dispatch_current_candidate"
    )
    assert opportunities["pr75_p6_action_dictionary_v2_micro_stack"][
        "component_score_gain_needed_after_rate"
    ] > 0.001
    assert opportunities["mask_exact_lossless_transcoder_target_217263"]["not_noop"] is False


def test_markdown_and_json_outputs_are_deterministic(tmp_path: Path) -> None:
    planner = _load_planner("_plan_full_pipeline_write_test")
    archive = _write_archive(tmp_path / "archive.zip")
    eval_json = _write_json(tmp_path / "eval.json", _eval_json())
    profile_json = _write_json(tmp_path / "profile.json", _profile_json(archive))
    kwargs = {
        "archive": archive,
        "eval_json": eval_json,
        "profile_json": profile_json,
        "artifact_paths": _artifact_paths(tmp_path),
    }

    first = planner.build_plan(**kwargs)
    second = planner.build_plan(**kwargs)
    first_json = planner._json_bytes(first)
    second_json = planner._json_bytes(second)
    first_md = tmp_path / "first.md"
    second_md = tmp_path / "second.md"

    planner.write_markdown(first_md, first)
    planner.write_markdown(second_md, second)

    assert first_json == second_json
    assert first_md.read_text() == second_md.read_text()
    assert "Bytes needed at unchanged distortion: `2209`" in first_md.read_text()
