# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tools.run_compact_renderer_mlx_spine_runner import (
    _lazy_segnet_boundary_recon_pixel_weight,
    _segnet_boundary_recon_pixel_weight,
)


class _Teacher:
    def __init__(self, logits: np.ndarray) -> None:
        self.teacher_logits_thwk = logits

    def teacher_logits_for_indices(self, idx):
        import mlx.core as mx

        return mx.array(self.teacher_logits_thwk)[idx]


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


def test_lazy_segnet_boundary_recon_weight_matches_pair_frame_policy() -> None:
    import mlx.core as mx

    logits = np.zeros((2, 384, 512, 3), dtype=np.float32)
    logits[0, ..., 0] = 4.0
    logits[0, 10:20, 30:40, 1] = 3.9
    logits[1, ..., 0] = 4.0
    logits[1, 100:110, 130:140, 1] = 3.9

    provider, metadata = _lazy_segnet_boundary_recon_pixel_weight(
        _Teacher(logits),
        tau=1.0,
        normalize="mean",
    )

    idx = mx.array([1, 0], dtype=mx.int32)
    frame0 = provider.recon_pixel_weight_for_batch(
        idx=idx,
        frame_shape=(2, 384, 512, 3),
        frame_index=0,
    )
    frame1 = provider.recon_pixel_weight_for_batch(
        idx=idx,
        frame_shape=(2, 384, 512, 3),
        frame_index=1,
    )
    mx.eval(frame0, frame1)

    assert frame0.shape == (2, 384, 512, 1)
    assert frame1.shape == (2, 384, 512, 1)
    assert float(mx.min(frame0).item()) == 1.0
    assert float(mx.max(frame0).item()) == 1.0
    assert float(frame1[0, 105, 135, 0].item()) > float(frame1[0, 15, 35, 0].item())
    assert float(frame1[1, 15, 35, 0].item()) > float(frame1[1, 105, 135, 0].item())
    assert metadata["provider_kind"] == "lazy_segnet_top2_boundary_margin"
    assert metadata["materialization"] == "batch_slices_only_no_full_video_dense_weight"
