# SPDX-License-Identifier: MIT
"""Tests for HiNeRV target-region sidecar/backend comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tac.analysis.hinerv_target_region_action_comparison import (
    build_hinerv_target_region_action_comparison_from_archive,
    write_hinerv_target_region_action_comparison,
)
from tac.submission_archive import build_minimal_single_member_archive_bytes
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hi_nerv.target_region_actions import (
    TargetRegionPixelAction,
    encode_target_region_actions_meta,
    encode_target_region_actions_payload,
    target_region_action_support_sha256,
)


def _tiny_archive_with_action(tmp_path: Path) -> tuple[Path, TargetRegionPixelAction]:
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=2,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(2, 2, 2),
        sin_frequency=3.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(11)
    model = HinervSubstrate(cfg).eval()
    action = TargetRegionPixelAction(
        pair_index=0,
        frame_index=1,
        height=8,
        width=8,
        yx=np.asarray([[1, 1], [1, 2], [2, 1], [2, 2], [4, 5]], dtype=np.uint16),
        rgb_u8=np.asarray(
            [[255, 0, 0], [254, 1, 0], [0, 255, 0], [0, 254, 1], [20, 30, 40]],
            dtype=np.uint8,
        ),
    )
    decoder_state = {
        key: value
        for key, value in dict(model.state_dict()).items()
        if key not in {"latents_coarse", "latents_mid", "latents_fine"}
    }
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "_target_region_actions_v1_b64": encode_target_region_actions_meta([action]),
    }
    packet = pack_archive(
        decoder_state,
        model.latents_coarse.detach(),
        model.latents_mid.detach(),
        model.latents_fine.detach(),
        meta,
    )
    archive_bytes, _method = build_minimal_single_member_archive_bytes(packet)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(archive_bytes)
    return archive, action


def _receipts(tmp_path: Path, archive: Path, action: TargetRegionPixelAction) -> tuple[Path, Path]:
    action_id = "hinerv-test-action"
    support_sha = target_region_action_support_sha256([action])
    payload = encode_target_region_actions_payload([action])
    survival = {
        "schema": "hi_nerv_target_region_action_parseback_survival.v1",
        "action_id": action_id,
        "archive_path": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": "a" * 64,
        "fakequant_survived": True,
        "parseback_survived": True,
        "inflate_survived": True,
        "pair_indices": [0],
        "total_action_pixels": action.pixel_count,
        "receiver_changed_action_pixels": action.pixel_count,
        "inflated_raw_action_changed_pixels": action.pixel_count,
        "target_region_actions": {
            "payload_bytes": len(payload),
            "support_sha256": support_sha,
            "support_cardinality": action.pixel_count,
        },
        "blockers": [],
    }
    runner = {
        "target_region_wall_normal_lift": {
            "schema": "tac.target_region_wall_normal_lift.v1",
            "action_id": action_id,
            "target_class": 4,
            "region_id": "b0/c4/r1",
            "direct_teacher": {
                "support_sha256": "b" * 64,
                "wrong_to_target_count": 2,
            },
            "backend_fit": {
                "attempted": True,
                "accepted_step_count": 0,
                "trained_groups": [],
                "realized_target_wall": False,
                "wrong_to_target_count": 0,
                "target_to_wrong_count": 0,
                "blockers": ["target_region_wall_normal_backend_not_realized"],
            },
            "sidecar_fallback": {
                "exact_delta_score_nonrate": -2.5,
                "payload_bytes": len(payload),
                "support_sha256": support_sha,
            },
        },
        "candidate_frontier_telemetry": {
            "masked_residual_oracle": {
                "best_candidate": {
                    "schema": "hi_nerv_target_region_masked_residual_oracle_candidate.v1",
                    "target_class": 4,
                    "region_id": "b0/c4/r1",
                    "target_region_action_pixel_count": action.pixel_count,
                    "target_region_action_payload_bytes": len(payload),
                    "region_argmax_transitions": {
                        "argmax_changed_count_region": 4,
                        "net_target_support_delta": 3,
                        "target_to_wrong_count": 0,
                        "wrong_to_target_count": 3,
                        "wrong_to_wrong_count": 1,
                    },
                    "admission_decision": {
                        "old_d_seg": 0.5,
                        "new_d_seg": 0.47,
                        "old_d_pose": 2.0,
                        "new_d_pose": 1.9,
                        "seg_score_delta": -3.0,
                        "pose_score_delta": -0.113246,
                        "exact_delta_score_nonrate": -3.113246,
                    },
                }
            }
        },
    }
    survival_path = tmp_path / "survival.json"
    runner_path = tmp_path / "runner.json"
    survival_path.write_text(json.dumps(survival), encoding="utf-8")
    runner_path.write_text(json.dumps(runner), encoding="utf-8")
    return survival_path, runner_path


def test_hinerv_action_comparison_decomposes_receiver_survived_sidecar(tmp_path: Path) -> None:
    archive, action = _tiny_archive_with_action(tmp_path)
    survival_path, runner_path = _receipts(tmp_path, archive, action)

    report = build_hinerv_target_region_action_comparison_from_archive(
        archive,
        survival_receipt=survival_path,
        runner_report=runner_path,
    )

    assert report["action_id"] == "hinerv-test-action"
    assert report["support_cardinality"] == action.pixel_count
    assert report["byte_decomposition"]["support_coord_u16_bytes"] == action.yx.nbytes
    assert report["byte_decomposition"]["rgb_u8_bytes"] == action.rgb_u8.nbytes
    assert report["comparison"]["sidecar_current_inflate_survived"] is True
    assert report["comparison"]["backend_realized"] is False
    assert report["comparison"]["next_blocker"] == "direct_teacher_and_survived_sidecar_support_hashes_diverge"
    assert report["comparison"]["best_lowering"] == "none"
    assert report["comparison"]["first_failing_surface"] == "support_identity_mismatch"
    assert report["support_identity"]["same_as_direct_teacher"] is False
    assert report["lowering_race"]["verdict"]["sidecar_status"] == "support_identity_mismatch"
    assert report["same_action_support"]["all_rows_same_action_support"] is True
    assert report["sidecar_economics"]["support_cardinality"] == action.pixel_count
    assert report["sidecar_economics"]["decision_axis"] == "exact_score_saved_per_charged_byte"
    assert report["sidecar_economics"]["sections"][0]["name"] == "support"
    assert report["sidecar_economics"]["sections"][1]["name"] == "action"
    assert report["sidecar_economics"]["sections"][2]["name"] == "metadata"
    assert report["byte_decomposition"]["entropy_sections"]["support_coord_u16"]["bytes"] == action.yx.nbytes
    assert report["byte_decomposition"]["entropy_sections"]["action_rgb_u8"]["bytes"] == action.rgb_u8.nbytes
    assert report["comparison"]["best_sidecar_value_per_byte"] is not None

    current = report["sidecar_encoding_candidates"][0]
    assert current["candidate_id"] == "current_hiv1_target_region_action_brotli"
    assert current["survival"]["parseback_survived"] is True
    assert current["survival"]["inflate_survived"] is True
    assert current["first_failed_surface"] == "support_identity_mismatch"
    assert "direct_teacher_and_survived_sidecar_support_hashes_diverge" in current["blockers"]
    assert current["action_effect"]["wrong_to_target"] == 3
    assert current["action_effect"]["target_to_wrong"] == 0
    assert current["action_effect"]["value_per_byte"] is not None
    assert "lowering_target=byte_priced_sidecar" in current["action_effect"]["payload_sections"]
    assert report["lowering_race"]["lowering_candidates"][0]["lowering_target_source"] == "explicit"
    assert report["lowering_race"]["lowering_candidates"][0]["lowering_target"] == "byte_priced_sidecar"
    assert (
        "direct_teacher_and_survived_sidecar_support_hashes_diverge"
        in current["action_effect"]["blockers"]
    )
    assert f"action_payload_bytes={action.rgb_u8.nbytes}" in current["action_effect"]["payload_sections"]
    assert current["action_effect"]["support_encoded_bytes"] == action.yx.nbytes

    blocked = [
        row
        for row in report["sidecar_encoding_candidates"]
        if row["candidate_id"] != "current_hiv1_target_region_action_brotli"
    ]
    assert blocked
    assert any("target_region_action_runtime_decoder_not_bound" in row["blockers"] for row in blocked)
    assert any(row["support_encoding"] == "path_tube_zlib_rdp2" for row in blocked)
    assert any(row["action_encoding"] == "constant_class_attractor_rgb_u8" for row in blocked)
    assert all(row["promotion_eligible"] is False for row in report["sidecar_encoding_candidates"])
    assert any(row["status"] == "measured" for row in report["backend_ladder"])


def test_hinerv_action_comparison_uses_archive_executable_direct_support(
    tmp_path: Path,
) -> None:
    archive, action = _tiny_archive_with_action(tmp_path)
    survival_path, runner_path = _receipts(tmp_path, archive, action)
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    support_sha = target_region_action_support_sha256([action])
    direct = runner["target_region_wall_normal_lift"]["direct_teacher"]
    direct["support_sha256"] = "b" * 64
    direct["support_hash_domain"] = "bool_mask_bhw"
    direct["archive_executable_support_sha256"] = support_sha
    direct["archive_executable_support_hash_domain"] = (
        "target_region_action_coordinates_v1"
    )
    direct["archive_executable_support_encoding"] = (
        "target_region_action_coordinates_v1"
    )
    direct["archive_executable_support_cardinality"] = action.pixel_count
    direct["archive_executable_support_encoded_bytes"] = action.yx.nbytes
    runner_path.write_text(json.dumps(runner), encoding="utf-8")

    report = build_hinerv_target_region_action_comparison_from_archive(
        archive,
        survival_receipt=survival_path,
        runner_report=runner_path,
    )

    assert report["support_identity"]["same_as_direct_teacher"] is True
    assert report["support_identity"]["direct_teacher_support_sha256"] == support_sha
    assert report["support_identity"]["direct_teacher_mask_support_sha256"] == "b" * 64
    assert report["support_identity"]["direct_teacher_comparison_hash_domain"] == (
        "target_region_action_coordinates_v1"
    )
    current = report["sidecar_encoding_candidates"][0]
    assert current["first_failed_surface"] is None
    assert "direct_teacher_and_survived_sidecar_support_hashes_diverge" not in current[
        "blockers"
    ]
    assert report["comparison"]["next_blocker"] == (
        "optimize_sidecar_grammar_current_receiver_survives_backend_does_not"
    )


def test_hinerv_action_comparison_writes_report_and_action_effect_rows(tmp_path: Path) -> None:
    archive, action = _tiny_archive_with_action(tmp_path)
    survival_path, runner_path = _receipts(tmp_path, archive, action)
    report = build_hinerv_target_region_action_comparison_from_archive(
        archive,
        survival_receipt=survival_path,
        runner_report=runner_path,
    )

    written = write_hinerv_target_region_action_comparison(report, tmp_path / "out")

    report_path = Path(written["report_path"])
    rows_path = Path(written["action_effect_rows_path"])
    assert report_path.is_file()
    assert rows_path.is_file()
    assert written["row_count"] == len(report["sidecar_encoding_candidates"])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert rows[0]["schema"] == "tac.action_effect.v1"
    assert rows[0]["action_id"] == "hinerv-test-action"
