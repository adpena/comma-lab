# SPDX-License-Identifier: MIT
"""Shared PR95/HNeRV MLX authority and readiness contract labels."""

from __future__ import annotations

PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER = "pr95_source_video_loader_not_ported_to_mlx"
PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER = (
    "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx"
)
PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER = (
    "pr95_source_video_rgb_targets_are_not_full_scorer_loss"
)
PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER = (
    "pr95_source_video_rgb_yuv6_preprocess_loss_is_not_full_scorer_loss"
)
PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER = (
    "pr95_preprocess_smoke_is_not_source_video_training_loop"
)
PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER = (
    "pr95_source_video_targets_ready_but_scorer_loss_not_wired_to_mlx"
)
PR95_LEGACY_TRAINING_LOOP_NOT_SOURCE_FAITHFUL_BLOCKER = (
    "pr95_training_loop_not_yet_source_faithful"
)
PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER = (
    "pr95_segnet_posenet_network_loss_not_wired_to_mlx"
)
PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER = (
    "pr95_stage_hparams_and_cosine_schedules_not_all_source_matched"
)
PR95_QAT_RESUME_UNPORTED_BLOCKER = (
    "pr95_qat_c1a_and_resume_semantics_not_ported_to_mlx"
)
PR95_EXPORT_FORWARD_PARITY_BLOCKER = "pr95_export_forward_parity_not_established"
PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER = (
    "full_frame_inflate_parity_against_source_runtime_not_run"
)


__all__ = [
    "PR95_EXPORT_FORWARD_PARITY_BLOCKER",
    "PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER",
    "PR95_LEGACY_TRAINING_LOOP_NOT_SOURCE_FAITHFUL_BLOCKER",
    "PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER",
    "PR95_QAT_RESUME_UNPORTED_BLOCKER",
    "PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER",
    "PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER",
    "PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER",
    "PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER",
    "PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER",
    "PR95_STAGE_SCHEDULE_SOURCE_MISMATCH_BLOCKER",
    "PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER",
]
