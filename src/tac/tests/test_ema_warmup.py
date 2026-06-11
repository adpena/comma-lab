# SPDX-License-Identifier: MIT
"""Tests for the canonical EMA warmup-decay schedule (tac.ema_warmup).

NO-FAKE: these assert the warmup BEHAVIOR (the shadow tracks live early), not
constants. They FAIL if warmup_ema_decay is reverted to a constant decay.
"""
from __future__ import annotations

import pytest

from tac.ema_warmup import DEFAULT_EMA_WARMUP_CONST, warmup_ema_decay


def test_warmup_starts_far_below_cap():
    # update 1: (1+1)/(10+1) = 0.1818 << 0.997 cap (a constant-decay revert -> 0.997).
    assert warmup_ema_decay(1, 0.997) == pytest.approx(2.0 / 11.0, rel=1e-9)
    assert warmup_ema_decay(1, 0.997) < 0.5


def test_warmup_is_monotone_nondecreasing():
    prev = -1.0
    for t in range(0, 5000, 7):
        d = warmup_ema_decay(t, 0.997)
        assert d >= prev - 1e-12, f"warmup must be monotone; t={t}"
        prev = d


def test_warmup_saturates_at_cap_and_never_exceeds():
    # far future: exactly the cap.
    assert warmup_ema_decay(10_000_000, 0.997) == pytest.approx(0.997, abs=1e-5)
    # never exceeds the cap at any t.
    for t in range(0, 100_000, 137):
        assert warmup_ema_decay(t, 0.997) <= 0.997 + 1e-12


def test_warmup_respects_a_low_cap():
    # with cap 0.5 the ramp is capped from t where (1+t)/(10+t) >= 0.5, i.e. t>=8.
    assert warmup_ema_decay(0, 0.5) == pytest.approx(0.1, rel=1e-9)
    assert warmup_ema_decay(100, 0.5) == pytest.approx(0.5, abs=1e-9)


def test_time_constant_is_short_early_long_late():
    # the WHOLE point: time constant tau = 1/(1-decay) is SHORT early (tracks live)
    # and approaches the cap's tau late (averages). At t=1 tau ~= 1.2 steps; at the
    # cap 0.997 tau = 333 steps.
    tau_early = 1.0 / (1.0 - warmup_ema_decay(1, 0.997))
    tau_late = 1.0 / (1.0 - warmup_ema_decay(10_000, 0.997))
    assert tau_early < 2.0, f"early time const must be short (got {tau_early})"
    assert tau_late > 100.0, f"late time const must be long (got {tau_late})"


def test_negative_updates_clamped():
    assert warmup_ema_decay(-5, 0.997) == warmup_ema_decay(0, 0.997)


def test_warmup_const_default():
    assert DEFAULT_EMA_WARMUP_CONST == 10.0
    # a larger warmup_const ramps slower (lower decay at the same t).
    assert warmup_ema_decay(20, 0.997, warmup_const=50.0) < warmup_ema_decay(20, 0.997)


# --- integration guards: the canonical schedule is actually WIRED into the EMAs ---


def test_torch_canonical_ema_warms_up_shadow_tracks_live_on_short_run():
    """``tac.training.EMA`` (default decay 0.997) tracks live early via warmup.

    The constant-0.997 bug would leave the shadow ~init after 5 updates
    (0.997^5 ~= 0.985 weight on init); warmup makes it track. FAILS if the wiring
    is reverted. ``.decay`` still reports the cap (0.997) for back-compat.
    """
    import torch
    import torch.nn as nn

    from tac.training import EMA

    m = nn.Linear(8, 8)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(2.0)
    ema = EMA(m, decay=0.997)
    assert ema.decay == 0.997  # cap reported for back-compat
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)  # live moves to 1.0
    for _ in range(5):
        ema.update(m)
    shadow0 = next(iter(ema.shadow.values())).flatten()[0].item()
    # warmup: after 5 updates the shadow is well past init(2.0) toward live(1.0);
    # constant 0.997 would still be ~1.985.
    assert shadow0 < 1.5, f"warmup must track live; shadow={shadow0} (bug -> ~1.985)"


def test_polyak_shadow_warmup_is_opt_in():
    """PolyakEMAShadow preserves EXACT Polyak by default; warmup is opt-in.

    Guards the Catalog #2 exact-averaging contract (default off) AND the opt-in
    capability (short-run callers pass enable_warmup=True).
    """
    import torch
    import torch.nn as nn

    from tac.training.long_training_canonical import PolyakEMAShadow

    def _one_update(enable_warmup):
        m = nn.Linear(4, 4)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(2.0)
        e = PolyakEMAShadow(m, decay=0.5, enable_warmup=enable_warmup)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1.0)
        e.update(m)
        return float(next(iter(e._shadow.values())).flatten()[0]), e.decay

    off_val, off_cap = _one_update(False)
    on_val, on_cap = _one_update(True)
    # default OFF: exact Polyak 0.5*2 + 0.5*1 = 1.5
    assert off_val == pytest.approx(1.5, abs=1e-6)
    # opt-in ON: first-step decay min(0.5, 2/11)=0.1818 -> shadow closer to live(1.0)
    assert on_val < off_val
    # cap restored after update in BOTH cases (external readers see 0.5)
    assert off_cap == pytest.approx(0.5) and on_cap == pytest.approx(0.5)
