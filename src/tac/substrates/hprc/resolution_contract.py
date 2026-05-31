# SPDX-License-Identifier: MIT
"""Resolution contract for HPRC archive-runnable candidates."""

from __future__ import annotations

from typing import Any

from tac.camera import CAMERA_H, CAMERA_W, SEGNET_INPUT_H, SEGNET_INPUT_W

HPRC_RESOLUTION_CONTRACT_SCHEMA = "hprc_resolution_contract.v1"
CONTEST_FRAME_COUNT = 1200
CONTEST_PAIR_COUNT = 600
RGB_CHANNEL_COUNT = 3
POSENET_PAIR_FRAME_COUNT = 2
POSENET_YUV6_CHANNEL_COUNT = 6


def hprc_resolution_contract() -> dict[str, Any]:
    """Return the canonical frame/scorer shape contract HPRC must satisfy."""

    return {
        "schema": HPRC_RESOLUTION_CONTRACT_SCHEMA,
        "contest_output": {
            "frame_count": CONTEST_FRAME_COUNT,
            "pair_count": CONTEST_PAIR_COUNT,
            "frame_order": "non_overlapping_pairs_(2*k,2*k+1)",
            "rgb_channels": RGB_CHANNEL_COUNT,
            "width": CAMERA_W,
            "height": CAMERA_H,
            "layout": "uint8_rgb_hwc_at_inflate_output",
        },
        "scorer_preprocess": {
            "width": SEGNET_INPUT_W,
            "height": SEGNET_INPUT_H,
            "resize": "official_scorer_bilinear_to_512x384_before_model_preprocess",
        },
        "posenet": {
            "sample": "frame_pair",
            "frames_per_sample": POSENET_PAIR_FRAME_COUNT,
            "preprocess": "rgb_pair_to_yuv6_after_scorer_resize",
            "channels_after_preprocess": POSENET_YUV6_CHANNEL_COUNT,
            "shape_authority": "tac.scorer.extract_gt_pose_targets",
        },
        "segnet": {
            "sample": "rgb_frame",
            "preprocess": "rgb_frame_after_scorer_resize",
            "classes": "tac.semantic_label_contract.CONTEST_SEGNET_CLASS_NAME_TUPLE",
            "shape_authority": "tac.scorer.extract_gt_masks",
        },
        "hprc_receiver_requirements": [
            "inflate_sh_must_emit_1200_native_1164x874_uint8_rgb_frames",
            "any_internal_384x512_renderer_state_must_roundtrip_through_native_output",
            "p18_p19_gradients_must_record_native_and_scorer_coordinate_spaces",
            "z8_or_residual_sidecar_subbands_must_declare_native_to_scorer_projection",
        ],
    }


__all__ = [
    "CONTEST_FRAME_COUNT",
    "CONTEST_PAIR_COUNT",
    "HPRC_RESOLUTION_CONTRACT_SCHEMA",
    "POSENET_PAIR_FRAME_COUNT",
    "POSENET_YUV6_CHANNEL_COUNT",
    "RGB_CHANNEL_COUNT",
    "hprc_resolution_contract",
]
