# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tools.run_compact_renderer_mlx_spine_runner import (
    _segnet_boundary_recon_pixel_weight,
)


class _Teacher:
    def __init__(self, logits: np.ndarray) -> None:
        self.teacher_logits_thwk = logits


def test_segnet_boundary_recon_weight_is_per_pair_last_frame_only() -> None:
    logits = np.zeros((2, 384, 512, 3), dtype=np.float32)
    logits[0, ..., 0] = 4.0
    logits[0, 10:20, 30:40, 1] = 3.9
    logits[1, ..., 0] = 4.0
    logits[1, 100:110, 130:140, 1] = 3.9

    weight, metadata = _segnet_boundary_recon_pixel_weight(
        _Teacher(logits),
        tau=1.0,
        normalize="mean",
    )

    assert weight.shape == (2, 2, 384, 512, 1)
    np.testing.assert_allclose(weight[:, 0, :, :, 0], 1.0)
    assert weight[0, 1, 15, 35, 0] > weight[0, 1, 105, 135, 0]
    assert weight[1, 1, 105, 135, 0] > weight[1, 1, 15, 35, 0]
    assert metadata["frame_policy"]["frame1"] == "per_pair_segnet_top2_boundary_saliency"
    assert "per_pair_last_frame" in metadata["scorer_terms"]["p18_segnet"]
