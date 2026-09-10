from __future__ import annotations

import numpy as np

from experiments import ddm_gdc3_anisotropic_keyrow_falsifier as keyrow


def test_fixed_anchor_schedules_are_exact_and_canonical() -> None:
    expected_counts = {
        "uniform_2": 192,
        "uniform_4": 96,
        "bands_8_4_2": 112,
        "bands_16_8_4_2": 90,
    }
    for variant, expected_count in expected_counts.items():
        anchors = keyrow.anchor_rows(variant)
        assert len(anchors) == expected_count
        assert anchors[0] == 0
        assert np.all(np.diff(anchors.astype(np.int32)) > 0)
        assert anchors[-1] < keyrow.geometry.HEIGHT
    assert set(keyrow.anchor_rows("bands_8_4_2")) >= {0, 128, 256}
    assert set(keyrow.anchor_rows("bands_16_8_4_2")) >= {0, 96, 192, 288}


def test_expand_key_rows_covers_height_without_cross_pair_state() -> None:
    anchors = np.array([0, 2, 5], dtype=np.uint16)
    key_rows = np.array(
        [
            [[10, 11], [20, 21], [30, 31]],
            [[40, 41], [50, 51], [60, 61]],
        ],
        dtype=np.uint8,
    )
    expanded = keyrow.expand_key_rows(key_rows, anchors, height=7)
    assert expanded.shape == (2, 7, 2)
    assert np.array_equal(expanded[0, :, 0], [10, 10, 20, 20, 20, 30, 30])
    assert np.array_equal(expanded[1, :, 0], [40, 40, 50, 50, 50, 60, 60])


def test_unknown_schedule_fails_closed() -> None:
    try:
        keyrow.anchor_rows("data_selected")
    except keyrow.KeyRowError as error:
        assert "unknown fixed schedule" in str(error)
    else:
        raise AssertionError("unknown target-dependent schedule was accepted")
