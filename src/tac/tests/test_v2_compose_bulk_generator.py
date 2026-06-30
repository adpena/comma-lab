# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.bulk_generator — keyframe selection + render (no SegNet/cache needed)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.v2_compose.bulk_generator import (
    BulkConfig,
    compute_class_mean_palette,
    load_calibration_from_reach,
    nearest_keyframe,
    render_partition,
    select_keyframes,
)


def test_select_keyframes_spacing():
    kf = select_keyframes(600, 47)
    assert kf[0] == 0
    assert kf == list(range(0, 600, 47))
    assert len(kf) == 13  # 0,47,...,564
    assert all(kf[i + 1] - kf[i] == 47 for i in range(len(kf) - 1))


def test_select_keyframes_validates():
    with pytest.raises(ValueError):
        select_keyframes(600, 0)
    with pytest.raises(ValueError):
        select_keyframes(0, 47)


def test_nearest_keyframe():
    kf = [0, 47, 94]
    assert nearest_keyframe(0, kf) == (0, 0)
    assert nearest_keyframe(46, kf) == (0, 46)
    assert nearest_keyframe(47, kf) == (47, 0)
    assert nearest_keyframe(100, kf) == (94, 6)


def test_render_partition_uses_palette_and_blurs():
    palette = np.array([[10, 20, 30], [200, 200, 200], [0, 0, 0],
                        [50, 60, 70], [255, 255, 255]], np.float64)
    label = np.zeros((8, 8), np.int64)
    label[4:, :] = 1  # bottom half class 1
    out = render_partition(label, palette)
    assert out.shape == (8, 8, 3)
    # interior of class-0 region ~ palette[0]; class-1 ~ palette[1] (gauss blur softens edges)
    assert out[0, 0, 0] == pytest.approx(10, abs=1.0)
    assert out[7, 7, 0] == pytest.approx(200, abs=1.0)
    # edge row is blended (between 10 and 200) -> not equal to either pure value
    assert 10 < out[4, 0, 0] < 200


def test_compute_class_mean_palette():
    n, H, W = 2, 6, 8
    # camera-res gt (the palette resizes to SEG res internally; small synthetic ok)
    gt_f1 = np.zeros((n, H, W, 3), np.uint8)
    gt_f1[:, :, :, 0] = 100  # all red=100
    lstars = np.zeros((n, 384, 512), np.int64)  # all class 0 (palette resize uses SEG res target)
    # NB: compute_class_mean_palette resizes gt to (384,512); class-0 mean R ~ 100
    pal = compute_class_mean_palette(gt_f1, lstars)
    assert pal.shape == (5, 3)
    assert pal[0, 0] == pytest.approx(100.0, abs=2.0)  # class 0 red mean
    # classes absent from lstars fall back to 127
    assert pal[1, 0] == pytest.approx(127.0, abs=1e-6)


def test_bulk_config_params():
    cfg = BulkConfig(s_t=-0.003, s_r=0.0, pitch=-0.01, reach_kstar=47, n_pairs=600)
    assert cfg.params == (-0.003, 0.0, -0.01)


def test_load_calibration_from_reach(tmp_path):
    import json
    j = {"calibration_fit": {"s_t": -0.0032, "s_r": 0.0, "pitch": -0.01}}
    p = tmp_path / "reach.json"
    p.write_text(json.dumps(j))
    cfg = load_calibration_from_reach(p, reach_kstar=47, n_pairs=600)
    assert cfg.s_t == pytest.approx(-0.0032)
    assert cfg.pitch == pytest.approx(-0.01)
    assert cfg.reach_kstar == 47
    assert cfg.n_pairs == 600
