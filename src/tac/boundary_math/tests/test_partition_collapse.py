# SPDX-License-Identifier: MIT
"""Tests for the per-class partition-collapse feasibility primitives (v8 probe).

Covers: power assignment correctness (weight shifts the boundary as Laguerre
theory says), the synthetic KNOWN-power-diagram recovery control, greedy-fit
monotonicity, lane curve fit/render on a synthetic dashed curve, movable
ellipses, static MyCar mask, contour stats, and the byte models.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.partition_collapse import (
    BITS_PER_GENERATOR,
    LANE,
    MOVABLE,
    MYCAR,
    NUM_CLASSES,
    ROAD,
    UNDRIV,
    PartitionCollapseError,
    PowerDiagram,
    boundary_band_mask,
    contour_stats,
    dash_occupancy_models,
    evaluate_partition,
    fit_lane_curves,
    fit_movable_ellipses,
    fit_power_diagram_greedy,
    generator_bytes,
    lane_curve_bytes,
    movable_bytes,
    power_assign,
    refine_weights_ce,
    render_ellipses,
    render_lane_band,
    road_undriv_base,
    static_mask_bytes,
    static_mycar_mask,
    synthetic_power_diagram,
    union_hybrid_reconstruction,
    zlib_label_bytes,
)


# ---------------------------------------------------------------------------
# power diagram assignment
# ---------------------------------------------------------------------------
def test_power_assign_two_sites_equal_weight_is_voronoi():
    d = PowerDiagram(
        np.array([[10.0, 10.0], [10.0, 30.0]]),
        np.zeros(2),
        np.array([ROAD, UNDRIV]),
    )
    _, pred = power_assign(d, (21, 41))
    # midline at col 20: cols <20 -> site 0, cols >20 -> site 1
    assert (pred[:, :20] == ROAD).all()
    assert (pred[:, 21:] == UNDRIV).all()


def test_power_assign_weight_shifts_boundary_toward_lighter_site():
    # Laguerre: raising w_0 grows cell 0 (boundary moves toward site 1).
    d0 = PowerDiagram(
        np.array([[10.0, 10.0], [10.0, 30.0]]),
        np.array([0.0, 0.0]),
        np.array([ROAD, UNDRIV]),
    )
    d1 = PowerDiagram(d0.sites.copy(), np.array([120.0, 0.0]), d0.classes.copy())
    _, p0 = power_assign(d0, (21, 41))
    _, p1 = power_assign(d1, (21, 41))
    assert (p1 == ROAD).sum() > (p0 == ROAD).sum()
    # analytic boundary: ||p-x0||^2 - w0 = ||p-x1||^2 -> col = 20 + w0/(2*20) = 23
    row = p1[10]
    assert row[22] == ROAD and row[24] == UNDRIV


def test_power_assign_rejects_empty_diagram():
    d = PowerDiagram(np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int32))
    with pytest.raises(PartitionCollapseError):
        power_assign(d, (8, 8))


def test_synthetic_control_known_power_diagram_recovered():
    """A KNOWN power diagram must be recovered near-exactly by greedy + CE refine."""
    _, labels = synthetic_power_diagram((96, 128), k=12, seed=3)
    diagram, curve = fit_power_diagram_greedy(labels, k_max=24, checkpoints=(24,))
    refined = refine_weights_ce(
        diagram, labels, steps=300, n_sample=8000, tau=10.0, optimize_positions=True
    )
    _, pred = power_assign(refined, labels.shape)
    agreement = float((pred == labels).mean())
    assert agreement >= 0.98, (agreement, curve[-1])
    # and the weight refinement itself must beat the weight-blind Voronoi fit
    _, pred_voronoi = power_assign(diagram, labels.shape)
    assert agreement > float((pred_voronoi == labels).mean())


def test_greedy_fit_curve_is_monotone_improving():
    _, labels = synthetic_power_diagram((96, 128), k=20, seed=7)
    _, curve = fit_power_diagram_greedy(labels, k_max=64, checkpoints=(8, 16, 32, 64))
    agr = [row["bulk_agreement"] for row in curve]
    assert all(b >= a - 5e-3 for a, b in zip(agr, agr[1:])), agr
    ks = [row["k"] for row in curve]
    assert ks == sorted(ks)


def test_refine_weights_ce_does_not_break_and_can_only_be_judged_by_metric():
    _, labels = synthetic_power_diagram((64, 96), k=10, seed=1)
    diagram, _ = fit_power_diagram_greedy(labels, k_max=16, checkpoints=(16,))
    _, pred_before = power_assign(diagram, labels.shape)
    before = float((pred_before == labels).mean())
    refined = refine_weights_ce(diagram, labels, steps=60, n_sample=4000, seed=0)
    _, pred_after = power_assign(refined, labels.shape)
    after = float((pred_after == labels).mean())
    assert refined.k == diagram.k
    assert np.isfinite(refined.weights).all()
    # refinement is measured, not assumed: only require it does not collapse
    assert after >= before - 0.05


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_evaluate_partition_perfect_and_band():
    labels = np.zeros((32, 32), np.int32)
    labels[:, 16:] = UNDRIV
    m = evaluate_partition(labels, labels)
    assert m["bulk_agreement"] == 1.0
    assert m["band_agreement"] == 1.0
    band = boundary_band_mask(labels, radius=3)
    assert band[:, 14:18].all()
    assert not band[:, :10].any()


def test_evaluate_partition_shape_mismatch_raises():
    with pytest.raises(PartitionCollapseError):
        evaluate_partition(np.zeros((4, 4), np.int32), np.zeros((4, 5), np.int32))


def test_labels_validation_rejects_out_of_range():
    bad = np.full((8, 8), NUM_CLASSES, np.int32)
    with pytest.raises(PartitionCollapseError):
        boundary_band_mask(bad)


# ---------------------------------------------------------------------------
# lane curve + dashes
# ---------------------------------------------------------------------------
def _dashed_lane_frame(h=200, w=160):
    labels = np.zeros((h, w), np.int32)
    rows = np.arange(40, 190)
    centers = 80 + 0.002 * (rows - 40) ** 2 * 0.15  # gentle quadratic
    for i, r in enumerate(rows):
        if (i // 12) % 2 == 0:  # 12-row dashes with 12-row gaps
            c = int(centers[i])
            labels[r, c - 1 : c + 2] = LANE
    return labels


def test_fit_lane_curves_recovers_dashed_quadratic():
    labels = _dashed_lane_frame()
    curves, uncovered = fit_lane_curves(labels)
    assert len(curves) == 1, (len(curves), uncovered)
    band = render_lane_band(curves, labels.shape)
    lane = labels == LANE
    recall = (band & lane).sum() / lane.sum()
    assert recall >= 0.9, recall
    # band should not massively overpaint (precision sanity)
    precision = (band & lane).sum() / max(1, band.sum())
    assert precision >= 0.5, precision


def test_dash_occupancy_models_on_regular_dashes():
    labels = _dashed_lane_frame()
    curves, _ = fit_lane_curves(labels)
    out = dash_occupancy_models(curves[0])
    assert not out["skipped"]
    assert not out["solid"]
    assert out["n_runs"] >= 4
    # image-space regular dashes: the periodic-in-image model must fit well
    assert out["periodic_image_agreement"] >= 0.9, out


def test_fit_lane_curves_empty_frame():
    labels = np.zeros((32, 32), np.int32)
    curves, uncovered = fit_lane_curves(labels)
    assert curves == [] and uncovered == 0


# ---------------------------------------------------------------------------
# movable / mycar / base
# ---------------------------------------------------------------------------
def test_movable_ellipses_cover_blob():
    labels = np.zeros((64, 64), np.int32)
    rr, cc = np.mgrid[0:64, 0:64]
    blob = ((rr - 30) ** 2 / 100 + (cc - 30) ** 2 / 36) <= 1.0
    labels[blob] = MOVABLE
    ellipses, uncovered = fit_movable_ellipses(labels)
    assert len(ellipses) == 1 and uncovered == 0
    mask = render_ellipses(ellipses, labels.shape)
    inter = (mask & blob).sum()
    assert inter / blob.sum() >= 0.9
    assert movable_bytes(len(ellipses)) > 0


def test_static_mycar_mask_majority_vote():
    stack = np.zeros((5, 16, 16), np.int32)
    stack[:, 12:, :] = MYCAR
    stack[0, 12, :] = ROAD  # one-frame flicker must not break the static mask
    mask = static_mycar_mask(stack)
    assert mask[12:, :].all()
    assert not mask[:12, :].any()
    assert static_mask_bytes(mask) > 0


def test_road_undriv_base_inpaints_other_classes():
    labels = np.zeros((32, 32), np.int32)
    labels[:16] = UNDRIV
    labels[20:24, 10:20] = MOVABLE
    base = road_undriv_base(labels)
    assert set(np.unique(base)) <= {ROAD, UNDRIV}
    assert (base[20:24, 10:20] == ROAD).all()


# ---------------------------------------------------------------------------
# contour + byte models
# ---------------------------------------------------------------------------
def test_contour_stats_counts_crack_edges():
    labels = np.zeros((10, 10), np.int32)
    labels[:, 5:] = UNDRIV
    st = contour_stats(labels)
    assert st["edge_px"] == 10
    assert st["per_pair"] == {"Road-Undrivable": 10}
    assert st["bytes_floor"] == pytest.approx(10 * 1.25 / 8.0)


def test_byte_models():
    assert generator_bytes(8) == pytest.approx(8 * BITS_PER_GENERATOR / 8.0)
    assert generator_bytes(8, with_weights=False) < generator_bytes(8)
    labels = np.zeros((64, 64), np.int32)
    assert zlib_label_bytes(labels) < 64 * 64
    assert lane_curve_bytes([], model="runs") == 0.0


# ---------------------------------------------------------------------------
# union hybrid
# ---------------------------------------------------------------------------
def test_union_hybrid_on_synthetic_scene():
    labels = np.zeros((96, 128), np.int32)
    labels[:40] = UNDRIV  # sky/top
    labels[80:] = MYCAR  # hood
    labels[50:60, 60:70] = MOVABLE
    rows = np.arange(42, 78)
    for r in rows:
        labels[r, 30:33] = LANE
    out = union_hybrid_reconstruction(labels, ru_k=8)
    m = out["metrics"]
    assert m["bulk_agreement"] >= 0.97, m
    assert m["per_class"]["MyCar"]["recall"] >= 0.999
    assert m["per_class"]["Movable"]["recall"] >= 0.8
    assert m["per_class"]["Lane"]["recall"] >= 0.8
    assert out["n_lane_curves"] == 1
    assert out["n_movable_islands"] == 1
