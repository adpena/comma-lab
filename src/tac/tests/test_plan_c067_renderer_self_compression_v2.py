# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_c067_renderer_self_compression_v2.py"


def _load_planner(name: str = "_plan_c067_renderer_self_compression_v2_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_direct_qzs3_source(path: Path) -> tuple[Path, bytes]:
    torch.manual_seed(0)
    model = build_quantizr_faithful_renderer().eval()
    renderer = encode_qzs3_state_dict(model.state_dict(), block_size=32)
    pose_values: list[float] = []
    for row in range(4):
        pose_values.extend([20.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    members = {
        "renderer.bin": renderer,
        "masks.mkv": b"\x12\x00synthetic-mask-obu" * 16,
        "optimized_poses.bin": struct.pack("<" + "e" * len(pose_values), *pose_values),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.extra = b""
            info.comment = b""
            zf.writestr(info, data)
    return path, renderer


def test_plan_decodes_synthetic_qzs3_and_keeps_no_score_contract(tmp_path: Path) -> None:
    planner = _load_planner()
    source, renderer = _write_direct_qzs3_source(tmp_path / "source.zip")

    plan = planner.build_plan(
        source_archive=source,
        qzs3_block_sizes=(32,),
        mixed_policy_specs=(),
        qbf1_block_sizes=(32,),
        imp_cycle_counts=(1, 2),
    )

    assert plan["schema"] == "c067_renderer_self_compression_v2_plan_v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["remote_or_gpu_dispatch"] is False
    assert plan["planning_constraints"]["candidate_archives_written"] is False
    assert plan["planning_constraints"]["scorers_loaded"] is False
    assert plan["planning_constraints"]["imp_bridge_builder_invoked"] is False
    assert plan["source_renderer"]["wire_format"] == "QZS3"
    assert plan["source_renderer"]["raw_bytes"] == len(renderer)
    assert plan["source_renderer"]["encoded_stream_bytes_used_for_gate"] == len(renderer)
    assert plan["source_renderer"]["qzs3_block_size"] == 32
    assert plan["qzs3_block_search"]["candidate_count"] == 1
    qzs3_candidate = plan["qzs3_block_search"]["candidates"][0]
    assert qzs3_candidate["wire_format"] == "QZS3"
    assert qzs3_candidate["exact_evaluable_archive"] is True
    assert qzs3_candidate["score_claim"] is False
    assert qzs3_candidate["promotion_eligible"] is False
    assert qzs3_candidate["delta_vs_current_stream_bytes"] >= 0
    assert plan["qbf1_negative_evidence"]["decision"]["qbf1_dispatch_warranted"] is False
    assert plan["imp_sparsity_prior"]["dispatchable_candidate"] is False
    assert plan["imp_sparsity_prior"]["bridge_builder_invoked"] is False
    assert plan["dispatch_recommendation"]["exact_cuda_dispatch_warranted"] is False


def test_mixed_local_candidate_can_warrant_dispatch_only_as_planning_gate(
    tmp_path: Path,
) -> None:
    planner = _load_planner("_plan_c067_renderer_self_compression_v2_mixed_test")
    source, _renderer = _write_direct_qzs3_source(tmp_path / "source.zip")

    plan = planner.build_plan(
        source_archive=source,
        qzs3_block_sizes=(32,),
        mixed_policy_specs=("component-aware-v1:frame2_pre64",),
        qbf1_block_sizes=(32,),
        imp_cycle_counts=(1,),
        reference_renderer_stream_bytes=10**9,
    )

    mixed = plan["mixed_local_block_candidates"]
    assert mixed["available"] is True
    assert mixed["candidate_count"] == 1
    candidate = mixed["candidates"][0]
    assert candidate["wire_format"] == "MQZ1"
    assert candidate["exact_evaluable_archive"] is True
    assert candidate["runtime_loader_ready"] is True
    assert candidate["state_change_vs_source_renderer"] is True
    assert candidate["score_claim"] is False

    recommendation = plan["dispatch_recommendation"]
    assert recommendation["exact_cuda_dispatch_warranted"] is True
    assert recommendation["score_claim"] is False
    assert recommendation["best_concrete_candidate"]["candidate_id"]
    assert any("claim lane" in item for item in recommendation["dispatch_prerequisites"])


def test_micro_renderer_byte_win_is_polish_only_by_default(
    tmp_path: Path,
) -> None:
    planner = _load_planner("_plan_c067_renderer_self_compression_v2_micro_gate_test")
    decision = planner._select_dispatch_decision(
        qzs3={"candidates": []},
        mixed_local={
            "candidates": [
                {
                    "candidate_id": "mixed_local_micro",
                    "family": "mixed_local_qzs_blocks",
                    "wire_format": "MQZ1",
                    "projected_renderer_stream_bytes": 55_878,
                    "delta_vs_current_stream_bytes": -87,
                    "raw_bytes": 59_321,
                    "sha256": "a" * 64,
                    "archive_builder_hint": "experiments/build_mixed_qzs_block_candidate.py",
                    "exact_evaluable_archive": True,
                    "runtime_loader_ready": True,
                    "state_change_vs_source_renderer": True,
                    "fail_closed_by_contract": False,
                }
            ]
        },
        qbf1={"decision": {"qbf1_dispatch_warranted": False}},
        source_renderer_stream_bytes=55_965,
    )

    assert decision["exact_cuda_dispatch_warranted"] is False
    assert decision["best_renderer_byte_savings"] < decision["min_dispatch_renderer_byte_savings"]
    assert "polish-only" in decision["reason"]


def test_write_plan_is_deterministic_json(tmp_path: Path) -> None:
    planner = _load_planner("_plan_c067_renderer_self_compression_v2_write_test")
    source, _renderer = _write_direct_qzs3_source(tmp_path / "source.zip")

    kwargs = {
        "source_archive": source,
        "qzs3_block_sizes": (32,),
        "mixed_policy_specs": (),
        "qbf1_block_sizes": (32,),
        "imp_cycle_counts": (1,),
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    plan_a = planner.write_plan(first, **kwargs)
    plan_b = planner.write_plan(second, **kwargs)

    assert first.read_bytes() == second.read_bytes()
    assert plan_a == plan_b
    loaded = json.loads(first.read_text())
    assert loaded["planning_constraints"]["deterministic_json"] is True
    assert loaded["dispatch_recommendation"]["promotion_eligible"] is False


def test_exact_negative_and_archive_screens_fail_close_global_reblocks(
    tmp_path: Path,
) -> None:
    planner = _load_planner("_plan_c067_renderer_self_compression_v2_fail_closed_test")
    source, _renderer = _write_direct_qzs3_source(tmp_path / "source.zip")
    evidence_dir = tmp_path / "exact_eval_c067_qzs3_b512_l40sdiag"
    evidence_dir.mkdir()
    negative = evidence_dir / "contest_auth_eval.json"
    negative.write_text(
        json.dumps(
            {
                "archive_size_bytes": 271886,
                "avg_posenet_dist": 0.3804819,
                "avg_segnet_dist": 0.00108114,
                "score_recomputed_from_components": 2.2397462747539274,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": "4271b2c855fddc089ae590392caec1d92cf408228664c4a9a0249d3b375e9d43",
                    "device": "cuda",
                    "gpu_model": "NVIDIA L40S",
                },
            }
        )
        + "\n"
    )
    (evidence_dir / "lightning_queue_metadata.json").write_text(
        json.dumps(
            {
                "queue_metadata": {
                    "candidate": "global_b512",
                    "lane": "c067_qzs3_b512",
                    "purpose": "c067_renderer_qzs3_reblock_distortion_curve",
                }
            }
        )
        + "\n"
    )
    archive_screen = tmp_path / "summary.json"
    archive_screen.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "policy": {
                            "name": "global_b512",
                            "spec": "global:512",
                        },
                        "output_archive": "global_b512/archive.zip",
                        "output_archive_bytes": 271886,
                        "output_archive_sha256": "4271b2c855fddc089ae590392caec1d92cf408228664c4a9a0249d3b375e9d43",
                        "archive_byte_delta": -4328,
                        "exact_evaluable_archive": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                    },
                    {
                        "policy": {
                            "name": "component_aware_v1_frame2_all64",
                            "spec": "component-aware-v1:frame2_all64",
                        },
                        "output_archive": "component_aware_v1_frame2_all64/archive.zip",
                        "output_archive_bytes": 276282,
                        "output_archive_sha256": "86684e1eff2c111c2f266f43a762952ae64976c8d25816b8935fdfbe42d36d52",
                        "archive_byte_delta": 68,
                        "exact_evaluable_archive": True,
                        "score_claim": False,
                        "promotion_eligible": False,
                    },
                ]
            }
        )
        + "\n"
    )

    plan = planner.build_plan(
        source_archive=source,
        qzs3_block_sizes=(32, 512),
        mixed_policy_specs=("component-aware-v1:frame2_all64",),
        qbf1_block_sizes=(32,),
        imp_cycle_counts=(1,),
        exact_negative_evidence_paths=(negative,),
        archive_screen_summary_paths=(archive_screen,),
    )

    assert plan["exact_negative_contract"]["closed_families"][
        "global_qzs3_reblock_above_source_block_size"
    ] is True
    qzs3_b512 = {
        item["candidate_id"]: item for item in plan["qzs3_block_search"]["candidates"]
    }["qzs3_b0512"]
    assert qzs3_b512["fail_closed_by_contract"] is True
    assert "PoseNet-collapse" in qzs3_b512["fail_closed_reason"]
    mixed = plan["mixed_local_block_candidates"]["candidates"][0]
    assert mixed["archive_screen"]["archive_byte_delta"] == 68
    assert mixed["fail_closed_by_contract"] is True
    assert "not a local archive byte win" in mixed["fail_closed_reason"]
    assert plan["dispatch_recommendation"]["exact_cuda_dispatch_warranted"] is False
    assert plan["archive_screen_contract"]["dispatchable_archive_byte_win_count"] == 0
