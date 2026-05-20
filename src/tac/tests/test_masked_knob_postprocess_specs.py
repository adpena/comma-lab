from __future__ import annotations

from tools.build_masked_knob_postprocess_specs import build_specs


def test_build_specs_emits_exact_frame_channel_and_temporal_specs() -> None:
    curriculum = {
        "schema": "ll_frame_pair_curriculum.v1",
        "adjustment_layers": [
            {
                "layer_id": "pair_0007_seg_boundary_last_frame",
                "primary_axis": "seg",
                "target": {"frames": [15]},
            },
            {
                "layer_id": "pair_0007_pose_global_pair",
                "primary_axis": "pose",
                "target": {"frames": [14, 15]},
            },
        ],
    }

    payload = build_specs(curriculum, max_specs=8)

    assert payload["score_claim"] is False
    assert payload["spec_count"] == 3
    assert payload["specs"][0]["kind"] == "channel_bias"
    assert payload["specs"][0]["frame_indices"] == [15]
    assert payload["specs"][2]["kind"] == "temporal_blend"
    assert payload["specs"][2]["frame_indices"] == [14, 15]


def test_build_specs_respects_max_specs() -> None:
    curriculum = {
        "schema": "ll_frame_pair_curriculum.v1",
        "adjustment_layers": [
            {
                "layer_id": f"pair_{idx:04d}_seg_boundary_last_frame",
                "primary_axis": "seg",
                "target": {"frames": [idx]},
            }
            for idx in range(4)
        ],
    }

    payload = build_specs(curriculum, max_specs=5)

    assert payload["spec_count"] == 5
    assert len(payload["specs"]) == 5
