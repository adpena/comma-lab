# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from tools import export_snerv_checkpoint_archive as export_tool
from tools import xray_snerv_receiver_value_domain as tool


def test_snerv_receiver_value_domain_xray_detects_unclipped_saturation(
    monkeypatch,
) -> None:
    packet = b"fake-snar"
    unclipped = np.array(
        [
            [
                [[[[-12.0, 100.0], [300.0, 260.0]]]],
                [[[[255.0, 255.0], [255.0, 255.0]]]],
            ]
        ],
        dtype=np.float32,
    )
    clipped = np.clip(unclipped, 0.0, 255.0)

    def fake_unpack(packet_bytes: bytes):
        assert packet_bytes == packet
        return SimpleNamespace(
            metadata={
                "schema": "snerv_inverse_steg_archive.v1",
                "n_pairs": 1,
                "frames_per_pair": 2,
                "channels": 1,
                "height": 2,
                "width": 2,
            },
            sections={
                "metadata_payload": b"meta",
                "lf_payload": b"lf" * 8,
                "decoder_payload": b"decoder",
                "step_map_packet": b"step",
            },
        )

    def fake_decode(packet_bytes, pair_indices, *, clip_to_uint8_range):
        assert packet_bytes == packet
        assert tuple(pair_indices) == (0,)
        return clipped if clip_to_uint8_range else unclipped

    monkeypatch.setattr(tool, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(tool, "decode_snerv_archive_pair_frames", fake_decode)
    monkeypatch.setattr(
        tool,
        "inspect_decoder_payload_header",
        lambda payload: {"schema": "unit_decoder_payload.v1"},
    )

    report = tool.build_snerv_receiver_value_domain_xray(
        packet=packet,
        pair_indices=(0,),
        packet_path="/tmp/candidate.snar",
    )

    assert report["schema"] == "snerv_receiver_value_domain_xray.v1"
    assert report["verdict"] == "RECEIVER_VALUE_DOMAIN_OUT_OF_RANGE"
    assert report["receiver_payload_decode_sample_proven"] is True
    assert report["value_domain_noncollapse_proof_passed"] is False
    assert report["closed_campaign_blockers"] == []
    assert report["packet_section_bytes"]["lf_payload"] == 16
    assert report["unclipped_receiver_stats"]["outside_0_255_fraction"] > 0.0
    assert report["clip_delta_abs_stats"]["mean_abs"] > 0.0
    assert "snerv_receiver_decode_unclipped_outside_uint8_domain" in report[
        "blockers"
    ]
    assert "snerv_receiver_decode_last_frame_saturated_for_segnet" in report[
        "blockers"
    ]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_snerv_receiver_value_domain_xray_cli_writes_json(
    tmp_path,
    monkeypatch,
) -> None:
    packet = tmp_path / "candidate.snar"
    packet.write_bytes(b"fake")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "scorer_input_diagnosis": {
                    "schema": "mlx_renderer_prefilter_scorer_input_diagnosis.v1",
                    "verdict": "SCORER_INPUT_OUT_OF_DISTRIBUTION",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        tool,
        "build_snerv_receiver_value_domain_xray",
        lambda **kwargs: {
            "schema": "snerv_receiver_value_domain_xray.v1",
            "packet_bytes": len(kwargs["packet"]),
            "pair_indices": list(kwargs["pair_indices"]),
            "profile_scorer_input_diagnosis": kwargs["profile"][
                "scorer_input_diagnosis"
            ],
            "blockers": ["snerv_receiver_value_domain_xray_false_authority"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    out = tmp_path / "xray.json"

    rc = tool.main(
        [
            "--packet",
            packet.as_posix(),
            "--profile-json",
            profile.as_posix(),
            "--pair-indices",
            "0,2",
            "--output-json",
            out.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["packet_bytes"] == 4
    assert payload["pair_indices"] == [0, 2]
    assert payload["profile_scorer_input_diagnosis"]["verdict"] == (
        "SCORER_INPUT_OUT_OF_DISTRIBUTION"
    )


def test_snerv_receiver_value_domain_xray_falls_back_to_official_header(
    monkeypatch,
) -> None:
    packet = b"official-snar"

    def fake_unpack(packet_bytes: bytes):
        assert packet_bytes == packet
        return SimpleNamespace(
            metadata={
                "n_pairs": 600,
                "frames_per_pair": 2,
                "channels": 3,
                "height": 384,
                "width": 512,
            },
            sections={
                "metadata_payload": b"meta",
                "lf_payload": b"lf",
                "decoder_payload": b"official",
                "step_map_packet": b"step",
            },
        )

    def fake_decode(packet_bytes, pair_indices, *, clip_to_uint8_range):
        raise tool.SnervArchiveError(
            "selected-frame decode is not supported for official MFU/HFR/TUB "
            "proof payloads"
        )

    monkeypatch.setattr(tool, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(tool, "decode_snerv_archive_pair_frames", fake_decode)
    monkeypatch.setattr(
        tool,
        "inspect_decoder_payload_header",
        lambda payload: {
            "schema": "snerv_decoder_payload.official_mfu_hfr_tub.v1",
            "skip_high_storage": {
                "codec": "scalar_mean_float64",
                "stored_shape": [1, 1, 1, 1],
                "source_shape": [1200, 3, 192, 256],
                "receiver_expands_skip_high": True,
                "lossless_relative_to_source_skip_high": False,
            },
        },
    )

    profile = {
        "scorer_input_distribution": {
            "candidate_segnet_last_rgb": {"saturation_fraction": 0.99},
            "candidate_posenet_yuv6_pair": {"saturation_fraction": 0.66},
            "segnet_last_rgb_absdiff": {"mean_abs": 74.0},
            "posenet_yuv6_pair_absdiff": {"mean_abs": 50.5},
        }
    }
    report = tool.build_snerv_receiver_value_domain_xray(
        packet=packet,
        pair_indices=(0, 599),
        packet_path="/tmp/official.snar",
        profile=profile,
    )

    assert report["verdict"] == "OFFICIAL_SKIP_HIGH_SCALAR_MEAN_COLLAPSE_RISK"
    assert report["sample_shape_b2chw"] is None
    assert report["receiver_payload_decode_sample_proven"] is False
    assert report["value_domain_noncollapse_proof_passed"] is False
    assert report["closed_campaign_blockers"] == []
    assert report["value_domain_sample_status"] == (
        "selected_pair_decode_unavailable_for_official_payload"
    )
    assert "snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk" in (
        report["blockers"]
    )
    assert "snerv_profile_segnet_last_frame_saturated" in report["blockers"]
    assert "rerun_snerv_local_export_with_full_or_shared_skip_high_before_exact_eval" in (
        report["recommended_next_actions"]
    )
    assert report["ready_for_exact_eval_dispatch"] is False


def test_snerv_receiver_value_domain_xray_emits_noncollapse_closures(
    monkeypatch,
) -> None:
    packet = b"clean-snar"
    clean = np.asarray(
        [
            [
                [[[[124.0, 126.0], [128.0, 130.0]]]],
                [[[[132.0, 134.0], [136.0, 138.0]]]],
            ]
        ],
        dtype=np.float32,
    )

    def fake_unpack(packet_bytes: bytes):
        assert packet_bytes == packet
        return SimpleNamespace(
            metadata={"n_pairs": 1, "frames_per_pair": 2},
            sections={
                "metadata_payload": b"meta",
                "lf_payload": b"lf",
                "decoder_payload": b"decoder",
                "step_map_packet": b"step",
            },
        )

    monkeypatch.setattr(tool, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(
        tool,
        "decode_snerv_archive_pair_frames",
        lambda packet_bytes, pair_indices, *, clip_to_uint8_range: clean,
    )
    monkeypatch.setattr(
        tool,
        "inspect_decoder_payload_header",
        lambda payload: {"schema": "unit_decoder_payload.v1"},
    )

    report = tool.build_snerv_receiver_value_domain_xray(
        packet=packet,
        pair_indices=(0,),
        packet_path="/tmp/clean.snar",
    )

    assert report["verdict"] == "receiver_value_domain_sample_within_limits"
    assert report["receiver_payload_decode_sample_proven"] is True
    assert report["value_domain_noncollapse_proof_passed"] is True
    assert report["closed_campaign_blockers"] == [
        "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
        "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
        "snerv_renderer_nondegenerate_target_value_domain_not_passed",
    ]
    assert report["blockers"] == ["snerv_receiver_value_domain_xray_false_authority"]


def test_snerv_receiver_value_domain_xray_rejects_constant_receiver_decode(
    monkeypatch,
) -> None:
    packet = b"constant-snar"
    constant = np.full((1, 2, 1, 1, 2, 2), 128.0, dtype=np.float32)

    def fake_unpack(packet_bytes: bytes):
        assert packet_bytes == packet
        return SimpleNamespace(
            metadata={"n_pairs": 1, "frames_per_pair": 2},
            sections={
                "metadata_payload": b"meta",
                "lf_payload": b"lf",
                "decoder_payload": b"decoder",
                "step_map_packet": b"step",
            },
        )

    monkeypatch.setattr(tool, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(
        tool,
        "decode_snerv_archive_pair_frames",
        lambda packet_bytes, pair_indices, *, clip_to_uint8_range: constant,
    )
    monkeypatch.setattr(
        tool,
        "inspect_decoder_payload_header",
        lambda payload: {"schema": "unit_decoder_payload.v1"},
    )

    report = tool.build_snerv_receiver_value_domain_xray(
        packet=packet,
        pair_indices=(0,),
        packet_path="/tmp/constant.snar",
    )

    assert report["verdict"] == "RECEIVER_VALUE_DOMAIN_OUT_OF_RANGE"
    assert report["receiver_payload_decode_sample_proven"] is True
    assert report["value_domain_noncollapse_proof_passed"] is False
    assert report["closed_campaign_blockers"] == []
    assert "snerv_receiver_decode_clipped_output_near_constant" in report["blockers"]
    assert "snerv_receiver_decode_last_frame_near_constant_for_segnet" in (
        report["blockers"]
    )


def test_snerv_receiver_value_domain_xray_decodes_official_payload_selected_pairs() -> None:
    model_size = export_tool.SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1",
        official_skip_high_mode="scalar_mean",
        official_tub_output2_store_for_receiver_proof=True,
    )
    packet = export_tool.build_snerv_official_checkpoint_packet(
        _official_checkpoint_state_with_tub_output2_payload(),
        model_size=model_size,
        metadata_extra={"unit_test": True},
    )

    report = tool.build_snerv_receiver_value_domain_xray(
        packet=packet.packet,
        pair_indices=(0,),
        packet_path="/tmp/official_output2.snar",
    )

    assert report["value_domain_sample_status"] == "selected_pair_decode_completed"
    assert report["receiver_payload_decode_sample_proven"] is True
    assert report["sample_shape_b2chw"] == [1, 2, 3, 16, 16]
    assert report["value_domain_noncollapse_proof_passed"] is True
    assert report["closed_campaign_blockers"] == [
        "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
        "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
        "snerv_renderer_nondegenerate_target_value_domain_not_passed",
    ]
    assert "snerv_official_payload_selected_pair_value_xray_unavailable" not in (
        report["blockers"]
    )
    assert report["decoder_payload_header"]["schema"] == (
        "snerv_decoder_payload.official_mfu_hfr_tub.v1"
    )


def _official_checkpoint_state_with_tub_output2_payload() -> dict[str, np.ndarray]:
    state = _official_checkpoint_state()
    state["tub.temporal_encoder_concat"] = np.linspace(
        0.0,
        1.0,
        1 * 6 * 8 * 8,
        dtype=np.float32,
    ).reshape(1, 6, 8, 8)
    state["tub.output2_raw"] = (
        np.arange(2 * 12 * 8 * 8, dtype=np.float32).reshape(2, 12, 8, 8)
        / 251.0
    )
    return state


def _official_checkpoint_state() -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "low": np.zeros((2, 3, 2, 2), dtype=np.float32),
        "skip_mid": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "skip_high": np.asarray([[[[32.0]]]], dtype=np.float32),
    }
    for name in ("lh", "hl", "hh"):
        state[f"hfr_{name}_conv1_weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
        state[f"hfr_{name}_conv1_bias"] = np.zeros((3,), dtype=np.float32)
        state[f"hfr_{name}_conv2_weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"hfr_{name}_conv2_bias"] = np.zeros((3,), dtype=np.float32)
    return state
