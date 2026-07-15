from __future__ import annotations

import pytest

from tac.confound_observability import (
    ema_warmup_updates,
    is_known_dseg_descent,
    is_partial_freeze,
    verdict_live_gap_due,
)


def test_ema_warmup_is_two_time_constants():
    assert ema_warmup_updates(0.997) == 667


def test_live_gap_auto_is_only_due_inside_warmup():
    assert verdict_live_gap_due(-1, epoch=25, ema_updates=666, ema_decay=0.997)
    assert not verdict_live_gap_due(-1, epoch=25, ema_updates=667, ema_decay=0.997)


def test_live_gap_explicit_modes_preserve_all_run_cadence():
    assert not verdict_live_gap_due(0, epoch=10, ema_updates=0, ema_decay=0.997)
    assert verdict_live_gap_due(5, epoch=10, ema_updates=100_000, ema_decay=0.997)
    assert not verdict_live_gap_due(5, epoch=11, ema_updates=0, ema_decay=0.997)
    with pytest.raises(ValueError):
        verdict_live_gap_due(-2, epoch=1, ema_updates=0, ema_decay=0.997)


def test_partial_freeze_band_is_open():
    assert not is_partial_freeze(0.02)
    assert is_partial_freeze(0.020001)
    assert is_partial_freeze(0.499999)
    assert not is_partial_freeze(0.5)


def test_known_dseg_descent_rejects_flat_or_rising_controls():
    assert is_known_dseg_descent((0.03, 0.02, 0.01))
    assert not is_known_dseg_descent((0.03, 0.03, 0.01))
    assert not is_known_dseg_descent((0.01, 0.02))
