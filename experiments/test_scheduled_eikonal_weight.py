"""Regression guard for the eikonal STEP-ramp control lever (#292, build 1 of the control system).

Verifies ``_scheduled_eikonal_weight``: BYTE-IDENTICAL constant when ``--eikonal-weight-end`` is
unset or == base (the #205 path), and a cosine-eased STEP base->end at the tau/MCF onset otherwise
(the fresh-run survival lever, MEASURED knee 0.05->0.10). Imports the function by file path so the
test is self-contained (no package requirement).

Run: ``.venv/bin/python -m pytest experiments/test_scheduled_eikonal_weight.py``
"""
from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"

_spec = importlib.util.spec_from_file_location("_lv_eik", _TRAINER)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_scheduled_eikonal_weight = _m._scheduled_eikonal_weight


def _args(**kw) -> Namespace:
    d = dict(eikonal_weight=0.05, eikonal_weight_end=None, curriculum=True,
             tau_softplus_start_epoch=300, stage_transition_rewarmup_epochs=20)
    d.update(kw)
    return Namespace(**d)


def test_unset_is_byte_identical_constant():
    """--eikonal-weight-end unset => the base value at EVERY epoch (the #205 byte-identical path)."""
    a = _args()
    assert all(_scheduled_eikonal_weight(e, a) == 0.05 for e in (0, 150, 299, 300, 400, 900))


def test_end_equals_base_is_constant():
    a = _args(eikonal_weight_end=0.05)
    assert all(_scheduled_eikonal_weight(e, a) == 0.05 for e in (0, 300, 400))


def test_non_curriculum_is_base():
    assert _scheduled_eikonal_weight(400, _args(eikonal_weight_end=0.10, curriculum=False)) == 0.05


def test_step_ramp_ce_then_ease_then_end():
    a = _args(eikonal_weight_end=0.10)
    assert _scheduled_eikonal_weight(299, a) == 0.05, "CE stage holds base"
    assert _scheduled_eikonal_weight(300, a) == 0.05, "at tau onset frac=0 -> base"
    assert _scheduled_eikonal_weight(320, a) == 0.10, "past the ease window -> end"
    assert _scheduled_eikonal_weight(400, a) == 0.10, "steady tau -> end"
    mid = _scheduled_eikonal_weight(310, a)
    assert 0.05 < mid < 0.10, f"eased midpoint must be strictly between base and end: {mid}"


def test_ease_is_monotone_nondecreasing():
    a = _args(eikonal_weight_end=0.10)
    vals = [_scheduled_eikonal_weight(e, a) for e in range(300, 321)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)), "ease must not decrease"
    assert vals[0] == 0.05 and abs(vals[-1] - 0.10) < 1e-12


def test_step_epoch_override_tracks_event_fired_boundary():
    """(SEAL fix) With event-triggering, the step must track the RESOLVED boundary: fired earlier
    -> step earlier; unfired (large sentinel) -> base even past the hardcoded cap; None -> identical
    to the original hardcoded behavior (byte-identity of the OFF path)."""
    a = _args(eikonal_weight_end=0.10)
    # None == original behavior at every epoch (the OFF/#205 path)
    for e in (0, 299, 300, 310, 320, 400):
        assert _scheduled_eikonal_weight(e, a, step_epoch=None) == _scheduled_eikonal_weight(e, a)
    # event fired EARLY at ep180: step tracks 180, fully stepped by 200, NOT waiting for 300
    assert _scheduled_eikonal_weight(179, a, step_epoch=180) == 0.05
    assert _scheduled_eikonal_weight(200, a, step_epoch=180) == 0.10
    assert _scheduled_eikonal_weight(250, a, step_epoch=180) == 0.10
    # unfired sentinel: base everywhere (still CE), even past the hardcoded cap epoch
    big = 1 << 30
    for e in (100, 300, 500):
        assert _scheduled_eikonal_weight(e, a, step_epoch=big) == 0.05


def test_zero_ease_is_a_hard_step():
    a = _args(eikonal_weight_end=0.10, stage_transition_rewarmup_epochs=0)
    assert _scheduled_eikonal_weight(299, a) == 0.05
    assert _scheduled_eikonal_weight(300, a) == 0.10, "no ease window => hard step at tau onset"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
