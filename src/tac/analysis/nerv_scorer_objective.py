# SPDX-License-Identifier: MIT
"""Shared contest-scorer objective authority for NeRV analysis surfaces."""

from __future__ import annotations

from typing import Any

SCORER_ONLY_OBJECTIVE_AUTHORITY: dict[str, Any] = {
    "objective": "contest_auth_eval_scorer_only",
    "allowed_selection_terms": [
        "SegNet_last_frame_distortion",
        "PoseNet_pair_distortion",
        "archive_zip_bytes_rate_term",
    ],
    "forbidden_selection_terms": [
        "human_visual_fidelity",
        "PSNR",
        "SSIM",
        "LPIPS",
        "perceptual_quality_unless_proven_scorer_causal",
    ],
    "rule": (
        "human visual fidelity is not an authority surface; optimize only the "
        "contest auth eval scorer and byte price"
    ),
}

__all__ = ["SCORER_ONLY_OBJECTIVE_AUTHORITY"]
