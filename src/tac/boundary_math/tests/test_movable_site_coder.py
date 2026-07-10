# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.movable_site_coder (#394 UNIT A Movable sparse-site carrier)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.movable_site_coder import (
    MOVABLE_CLASS,
    MovableSiteCoderError,
    byte_account_sites,
    extract_movable_sites,
    render_sites_to_mask,
    track_sites,
)
from tac.through_r.resolution_chain import SEG_H, SEG_W


def _lab_with_boxes(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Label map (all Road) with Movable rectangles at (y0,y1,x0,x1)."""
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)
    for y0, y1, x0, x1 in boxes:
        lab[y0:y1, x0:x1] = MOVABLE_CLASS
    return lab


def test_extract_no_movable_returns_empty():
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)
    sites = extract_movable_sites(lab)
    assert sites.shape == (0, 5)


def test_extract_single_box_center_and_extent():
    lab = _lab_with_boxes([(100, 140, 200, 260)])  # 40 tall, 60 wide
    sites = extract_movable_sites(lab, min_area_px=1)
    assert sites.shape == (1, 5)
    cx, cy, bw, bh, area = sites[0]
    assert bw == pytest.approx(60.0)
    assert bh == pytest.approx(40.0)
    assert cx == pytest.approx((200 + 259) / 2.0)
    assert cy == pytest.approx((100 + 139) / 2.0)
    assert area == pytest.approx(40 * 60)


def test_extract_two_boxes_sorted_by_area_desc():
    lab = _lab_with_boxes([(10, 20, 10, 20), (100, 160, 100, 200)])  # small then large
    sites = extract_movable_sites(lab, min_area_px=1)
    assert sites.shape[0] == 2
    assert sites[0, 4] >= sites[1, 4]  # descending area


def test_extract_min_area_filters_noise():
    lab = _lab_with_boxes([(10, 12, 10, 12)])  # 2x2 = 4 px
    assert extract_movable_sites(lab, min_area_px=5).shape[0] == 0
    assert extract_movable_sites(lab, min_area_px=4).shape[0] == 1


def test_extract_wrong_shape_raises():
    with pytest.raises(MovableSiteCoderError):
        extract_movable_sites(np.zeros((10, 10), dtype=np.int64))


def test_track_no_sites():
    per_frame = [np.zeros((0, 5)) for _ in range(5)]
    tr = track_sites(per_frame)
    assert tr.K == 0
    assert tr.M.shape == (5, 0)


def test_track_persistent_site_keeps_slot():
    # a car drifting slowly across 4 frames -> one persistent slot, low delta.
    per_frame = []
    for t in range(4):
        per_frame.append(np.array([[200.0 + t, 150.0, 40.0, 30.0, 1200.0]]))
    tr = track_sites(per_frame)
    assert tr.K == 1
    assert tr.presence.all()
    # slot 0 cx increments by ~1/frame.
    assert tr.M[3, 0] - tr.M[0, 0] == pytest.approx(3.0)


def test_track_matched_count_positive_for_persistent():
    per_frame = [np.array([[200.0 + t, 150.0, 40.0, 30.0, 1200.0]]) for t in range(4)]
    tr = track_sites(per_frame)
    assert tr.n_matched >= 3


def test_byte_account_no_sites():
    tr = track_sites([np.zeros((0, 5)) for _ in range(3)])
    b = byte_account_sites(tr, [np.zeros((0, 5)) for _ in range(3)])
    assert b.n_sites_total == 0
    assert b.K == 0


def test_byte_account_tracked_beats_or_matches_raw_for_coherent():
    # a persistent slowly-moving car -> tracked temporal-delta should be <= raw per-frame.
    per_frame = [np.array([[200.0 + t, 150.0, 40.0, 30.0, 1200.0]]) for t in range(30)]
    tr = track_sites(per_frame)
    b = byte_account_sites(tr, per_frame)
    assert b.tracked_bytes > 0
    assert b.raw_perframe_bytes > 0
    # correspondence-first: a coherent track codes no worse than independent frames.
    assert b.tracked_bytes <= b.raw_perframe_bytes + 8


def test_byte_account_reports_counts():
    per_frame = [np.array([[200.0, 150.0, 40.0, 30.0, 1200.0]]) for _ in range(5)]
    tr = track_sites(per_frame)
    b = byte_account_sites(tr, per_frame)
    assert b.n_sites_total == 5
    assert b.P == 5
    assert b.presence_bytes > 0


def test_render_sites_roundtrip_covers_box():
    sites = np.array([[200.0, 150.0, 60.0, 40.0, 2400.0]])
    mask = render_sites_to_mask(sites)
    assert mask.shape == (SEG_H, SEG_W)
    assert mask.sum() > 0
    # centre of the box is set.
    assert mask[150, 200]


def test_render_empty_sites_empty_mask():
    mask = render_sites_to_mask(np.zeros((0, 5)))
    assert not mask.any()


def test_render_recovers_extracted_box_area():
    lab = _lab_with_boxes([(100, 140, 200, 260)])
    sites = extract_movable_sites(lab, min_area_px=1)
    mask = render_sites_to_mask(sites)
    gt = lab == MOVABLE_CLASS
    # a single axis-aligned box -> exact recovery (IoU 1).
    inter = (mask & gt).sum()
    union = (mask | gt).sum()
    assert inter / union == pytest.approx(1.0)


def test_render_ellipse_shape_has_coverage_loss():
    # a non-rectangular blob -> box over-covers (IoU < 1) but recall high (the honest lossy tell).
    lab = np.zeros((SEG_H, SEG_W), dtype=np.int64)
    yy, xx = np.mgrid[0:SEG_H, 0:SEG_W]
    disk = ((yy - 150) ** 2 + (xx - 200) ** 2) < 30 ** 2
    lab[disk] = MOVABLE_CLASS
    sites = extract_movable_sites(lab, min_area_px=1)
    mask = render_sites_to_mask(sites)
    gt = lab == MOVABLE_CLASS
    recall = (mask & gt).sum() / gt.sum()
    iou = (mask & gt).sum() / (mask | gt).sum()
    assert recall == pytest.approx(1.0)  # box covers the whole disk
    assert iou < 1.0                     # but over-covers the corners
