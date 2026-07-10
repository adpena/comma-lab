"""P4 canary tests for the verdict-trend alarm (design_philosophies_eightfold P4 "no meter without a
canary"). The positive control (synthetic TRAIN<->VERDICT decoupling) MUST fire; the negative control
(a clean co-descending run) MUST NOT — the known-effect + not-fire proof this meter can gate."""
from __future__ import annotations

from tac.witness_control.verdict_trend_alarm import (
    NO_ALARM,
    TRAIN_VERDICT_DECOUPLING,
    canary_suite,
    synthetic_codescending_verdicts,
    synthetic_decoupling_verdicts,
    verdict_trend_alarm,
)


def test_positive_control_fires_decoupling():
    alarm = verdict_trend_alarm(synthetic_decoupling_verdicts())
    assert alarm.classification == TRAIN_VERDICT_DECOUPLING, alarm.reason
    assert alarm.fired() is True


def test_negative_control_does_not_fire():
    alarm = verdict_trend_alarm(synthetic_codescending_verdicts())
    assert alarm.fired() is False
    assert alarm.classification == NO_ALARM, alarm.reason


def test_canary_suite_passes():
    res = canary_suite()
    assert res.passed is True, res.reason
    assert res.positive_fired_decoupling is True
    assert res.negative_fired is False
