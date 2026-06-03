# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV LF allocation saliency.

These guard the G3 wire-in at the allocation boundary: pixel saliency must be
pushed through the native cropped-synthesis adjoint before taking the stored LF
block. Tests verify behaviour, not marker constants.
"""

from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.allocation import (
    allocate_lf_linf,
    push_pixel_saliency_to_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    dwt2_multilevel,
    dwt2_native_synthesis_adjoint,
)


def test_push_pixel_saliency_uses_native_crop_adjoint_lf_block_on_odd_dims():
    """NO-FAKE: LF saliency equals the real crop-adjoint LF block on odd dims."""
    rng = np.random.default_rng(20)
    hw = (65, 97)
    seg = np.abs(rng.standard_normal(hw)) + 1e-6
    pose = np.abs(rng.standard_normal(hw)) + 1e-6
    pose_weight = 0.75
    result = push_pixel_saliency_to_lf(
        seg,
        pose,
        carrier_hw=hw,
        levels=3,
        pose_weight=pose_weight,
    )

    seg_n = seg / seg.sum()
    pose_n = pose / pose.sum()
    combined = seg_n + pose_weight * pose_n
    expected = np.abs(dwt2_native_synthesis_adjoint(combined, levels=3).lf)
    old_reflect = np.abs(dwt2_multilevel(combined, levels=3).lf)

    assert result.lf_shape == expected.shape
    assert np.allclose(result.lf_saliency, expected)
    assert not np.allclose(result.lf_saliency, old_reflect)
    assert 0.0 < result.pixel_seg_mass <= 1.0
    assert 0.0 < result.pixel_pose_mass <= 1.0


def test_allocate_lf_linf_produces_real_nonuniform_steps_from_saliency():
    """NO-FAKE: allocation consumes saliency and emits a nonconstant step map."""
    rng = np.random.default_rng(21)
    hw = (65, 97)
    seg = np.zeros(hw, dtype=np.float64)
    pose = np.zeros(hw, dtype=np.float64)
    seg[10:20, 30:45] = 10.0
    pose += np.abs(rng.standard_normal(hw)) * 0.01
    lfs = push_pixel_saliency_to_lf(seg, pose, carrier_hw=hw, levels=3)
    target_bits = float(lfs.lf_saliency.size) * 4.0
    alloc = allocate_lf_linf(
        lfs,
        target_bits=target_bits,
        dynamic_range=37.5,
        min_step=37.5 / 256.0,
        max_step=37.5,
    )
    steps = alloc.steps.reshape(lfs.lf_shape)

    assert steps.shape == lfs.lf_shape
    assert np.all(np.isfinite(steps))
    assert np.all(steps > 0)
    assert float(steps.max()) > float(steps.min())
    assert alloc.total_bits >= target_bits
