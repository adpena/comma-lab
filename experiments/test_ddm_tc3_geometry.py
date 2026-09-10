"""Algorithm controls only: synthetic tests do not establish a science verdict."""

import numpy as np
import pytest

from experiments import ddm_tc3_geometry as geom


@pytest.mark.parametrize("kind", [geom.LaneGeometry, geom.RunTrackingGeometry])
def test_group_prefix_cannot_be_skipped_or_partially_observed(kind):
    tracker = kind(None)
    with pytest.raises(ValueError, match="next HPAC group"):
        tracker.contexts(geom.POSITIONS[1])
    positions = geom.POSITIONS[0]
    tracker.contexts(positions)
    with pytest.raises(ValueError, match="observe must follow"):
        tracker.contexts(positions)
    with pytest.raises(ValueError, match="next HPAC group"):
        tracker.observe(positions[:-1], np.zeros(len(positions) - 1, dtype=np.uint8))
    tracker.observe(positions, np.zeros(len(positions), dtype=np.uint8))
    np.testing.assert_array_equal(tracker.plane.reshape(-1)[positions], 0)
    assert tracker.group == 1


@pytest.mark.parametrize("kind", [geom.LaneGeometry, geom.RunTrackingGeometry])
def test_previous_is_copied_and_rejects_invalid_symbols(kind):
    previous = np.zeros((geom.H, geom.W), dtype=np.uint8)
    tracker = kind(previous)
    previous[:] = 1
    assert not tracker.previous.any()
    with pytest.raises(ValueError, match="complete decoded"):
        kind(np.full((geom.H, geom.W), 5, dtype=np.uint8))
    with pytest.raises(ValueError, match="negative"):
        kind(np.full((geom.H, geom.W), -1, dtype=np.int8))


def test_two_sloped_runs_cross_slots_without_being_averaged():
    mask = np.zeros((geom.H, geom.W), dtype=bool)
    for y in range(100, 160):
        left = 30 + y - 100
        mask[y, left : left + 3] = True
        mask[y, left + 12 : left + 15] = True
    forecast = geom._run_forecasts(mask, np.array([135, 145]))
    for y in (135, 145):
        assert len(forecast[y]) == 2
        for index, (left, right) in enumerate(sorted(forecast[y])):
            expected = 4 * (30 + y - 100 + 12 * index)
            assert abs(left - expected) <= 2
            assert abs(right - (expected + 8)) <= 2


def test_dash_gap_expires_and_forecast_does_not_read_current_row():
    mask = np.zeros((geom.H, geom.W), dtype=bool)
    mask[100:120, 61:65] = True
    rows = np.array([120, 124, 125, 130])
    forecast = geom._run_forecasts(mask, rows)
    assert forecast[120] == [(244, 256)]
    assert forecast[124] == [(244, 256)]
    assert forecast[125] == []
    changed = mask.copy()
    changed[124:] = True
    assert geom._run_forecasts(changed, rows)[124] == forecast[124]


def test_current_row_labels_cannot_change_its_prediction():
    previous = np.zeros((geom.H, geom.W), dtype=np.uint8)
    previous[:, 61:65] = 1
    baseline, changed = geom.RunTrackingGeometry(previous), geom.RunTrackingGeometry(previous)
    for group in range(40):
        positions = geom.POSITIONS[group]
        b = baseline.contexts(positions)
        c = changed.contexts(positions)
        np.testing.assert_array_equal(b[positions // geom.W == 128], c[positions // geom.W == 128])
        symbols = previous.reshape(-1)[positions].copy()
        baseline.observe(positions, symbols)
        symbols[positions // geom.W >= 128] = 0
        changed.observe(positions, symbols)


def test_run_geometry_empty_and_prior_interface_behavior():
    empty = geom.RunTrackingGeometry(None)
    positions = geom.POSITIONS[0]
    bins = empty.contexts(positions)
    band = (positions // geom.W >= 128) & (positions // geom.W < 320)
    np.testing.assert_array_equal(bins[band], 7)
    np.testing.assert_array_equal(bins[~band], 8)
    previous = np.zeros((geom.H, geom.W), dtype=np.uint8)
    previous[:, 63:66] = 1
    tracker = geom.RunTrackingGeometry(previous)
    bins = tracker.contexts(positions)
    # The x=64 positions in group zero sit inside the three-pixel run: d=1.
    chosen = band & (positions % geom.W == 64)
    assert chosen.any()
    np.testing.assert_array_equal(bins[chosen], 1)
