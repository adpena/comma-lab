"""Tests for tools/witness_control_monitor.py — the self-converging control monitor (facet-5).

Proves the two certificates on synthetic + real-#205-shaped verdict trajectories: the tau-CREEP
detector fires on d_seg-UP-while-loss-DOWN (the #205 erosion), the Lyapunov descent certificate
distinguishes converging / plateau, and the stage filter only fits the current stage.

Run: ``.venv/bin/python -m pytest tools/test_witness_control_monitor.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.witness_control_monitor import (  # noqa: E402
    CONVERGING,
    DIVERGING_ERASING,
    PLATEAU,
    VOLATILE,
    _lstsq_slope,
    classify_trajectory,
)


def _v(epoch: int, d_seg: float, ep_loss: float, seg_form: str = "tau_softplus") -> dict:
    return {"stage": "verdict", "epoch": epoch, "seg_form": seg_form,
            "d_seg": d_seg, "ep_loss": ep_loss}


def test_lstsq_slope_exact():
    assert abs(_lstsq_slope([0, 1, 2, 3], [0, 2, 4, 6]) - 2.0) < 1e-12
    assert abs(_lstsq_slope([0, 1, 2], [5, 5, 5])) < 1e-12   # flat
    assert _lstsq_slope([1], [1]) == 0.0                      # <2 points


def test_creep_is_diverging_erasing():
    """d_seg RISING while ep_loss FALLS => the #205 tau-creep signature."""
    vs = [_v(300 + 25 * i, 0.0047 + 0.0004 * i, 150.0 - 4.0 * i) for i in range(5)]
    out = classify_trajectory(vs, window=5)
    assert out.classification == DIVERGING_ERASING, out
    assert out.d_seg_slope_per_ep > 0.0 and out.ep_loss_slope_per_ep < 0.0
    assert any("paint" in d for d in out.config_diffs)


def test_healthy_descent_is_converging():
    """d_seg FALLING => Lyapunov dV/dt < 0 => converging."""
    vs = [_v(300 + 25 * i, 0.0060 - 0.0005 * i, 150.0 - 4.0 * i) for i in range(5)]
    out = classify_trajectory(vs, window=5)
    assert out.classification == CONVERGING, out
    assert out.d_seg_slope_per_ep < 0.0


def test_flat_is_plateau():
    """|dV/dt| ~ 0 => plateau (candidate early-stop / stage-advance)."""
    vs = [_v(300 + 25 * i, 0.0050 + 1e-9 * i, 150.0 - 4.0 * i) for i in range(5)]
    out = classify_trajectory(vs, window=5)
    assert out.classification == PLATEAU, out


def test_volatile_when_high_cv():
    """Large within-window swings => volatile (no clean slope)."""
    dsegs = [0.004, 0.012, 0.003, 0.014, 0.002]
    vs = [_v(300 + 25 * i, dsegs[i], 150.0 - 4.0 * i) for i in range(5)]
    out = classify_trajectory(vs, window=5)
    assert out.classification == VOLATILE, out


def test_stage_filter_only_fits_current_stage():
    """Only the latest seg_form's verdicts are fit (a CE->tau transition must not leak)."""
    ce = [_v(0 + 25 * i, 0.02 - 0.001 * i, 200.0, seg_form="ce") for i in range(4)]
    tau = [_v(300 + 25 * i, 0.0047 + 0.0004 * i, 150.0 - 4.0 * i) for i in range(5)]
    out = classify_trajectory(ce + tau, window=5)
    assert out.stage == "tau_softplus"
    assert out.classification == DIVERGING_ERASING   # the tau creep, not the CE descent


def test_real_205_shape_is_flagged():
    """The actual #205 tau trace (ep 350-450, d_seg up, loss down) is DIVERGING_ERASING."""
    real = [
        _v(350, 0.006267, 141.335), _v(375, 0.006424, 137.598),
        _v(400, 0.006568, 134.122), _v(425, 0.006652, 133.580),
        _v(450, 0.006674, 129.827),
    ]
    out = classify_trajectory(real, window=5)
    assert out.classification == DIVERGING_ERASING
    assert out.d_seg_slope_per_ep > 0.0
    assert out.ep_loss_slope_per_ep < 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
