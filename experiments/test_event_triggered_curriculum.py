"""Regression guard for the EVENT-TRIGGERED CURRICULUM control lever (#292, build 2 of 3).

Operator 2026-07-04: "epochs should not be hardcoded but dynamical according to convergence
trajectory." Verifies the pure convergence controller ``_stage_converged`` + the stateful resolver
``_evt_resolve_seg_form`` that fires the CE -> tau_softplus -> l7_softplus curriculum transitions on
loss-plateau CONVERGENCE (deterministic, from the SYNCHRONOUS ep_loss) instead of the hardcoded
epochs, which then act as HARD CAPS (the trigger only fires EARLIER, never later).

The two binding non-negotiables proved here:
  * BYTE-IDENTITY: a run whose loss NEVER plateaus reproduces the hardcoded ``_seg_form_for_epoch``
    schedule EXACTLY (transitions fire at the caps) -- so the OFF path (which calls
    ``_seg_form_for_epoch`` unchanged) is byte-identical, and even the ON path degenerates to it.
  * DETERMINISM: the same seeded loss history drives the SAME fired transition epochs (no async
    verdict, no wall-clock).

Imports the trainer by file path (self-contained, no package requirement), mirroring
``test_scheduled_eikonal_weight.py``. Run:
``.venv/bin/python -m pytest experiments/test_event_triggered_curriculum.py``
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

_spec = importlib.util.spec_from_file_location("_lv_evtcur", _TRAINER)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_stage_converged = _m._stage_converged
_evt_resolve_seg_form = _m._evt_resolve_seg_form
_evt_current_stage_form = _m._evt_current_stage_form
_seg_form_for_epoch = _m._seg_form_for_epoch


def _args(*, tau=300, l7=800, epochs=1500, min_stage=150, rel_eps=1e-3, windows=4,
          curriculum=True, seg_loss="ce") -> Namespace:
    return Namespace(
        curriculum=curriculum, seg_loss=seg_loss,
        tau_softplus_start_epoch=tau, l7_start_epoch=l7, epochs=epochs,
        curriculum_min_stage_epochs=min_stage,
        curriculum_plateau_rel_eps=rel_eps, curriculum_plateau_windows=windows,
    )


def _drive(args, loss_fn, upto):
    """Step ``_evt_resolve_seg_form`` epoch-by-epoch EXACTLY like the training loop: resolve at the
    START of the epoch, then append ``loss_fn(ep)`` at the END. Returns (per-epoch seg_form list,
    fired-event list, final state) so a test can assert the whole trajectory + the fired epochs."""
    state = {"tau": None, "l7": None, "stage_start": 1, "losses": []}
    forms, events = [], []
    for ep in range(1, upto + 1):
        sf, ev = _evt_resolve_seg_form(ep, state, args)
        forms.append(sf)
        if ev is not None:
            events.append(ev)
        state["losses"].append(float(loss_fn(ep)))
    return forms, events, state


# ───────────────────────── pure controller: _stage_converged ─────────────────────────

def test_stage_converged_plateau_fires():
    """A flat loss window past the min floor => plateau => True."""
    ep = list(range(1, 21))
    ls = [0.5] * 20
    assert _stage_converged(ep, ls, min_stage_epochs=10, plateau_rel_eps=1e-3, plateau_windows=4) is True


def test_stage_converged_min_floor_blocks():
    """Flat but FEWER than min_stage_epochs completed => False (the min floor blocks an early fire)."""
    ep = list(range(1, 6))
    ls = [0.5] * 5
    assert _stage_converged(ep, ls, min_stage_epochs=10, plateau_rel_eps=1e-3, plateau_windows=4) is False


def test_stage_converged_falling_does_not_fire():
    """A steeply-DECREASING loss window is not a plateau => False (the stage keeps training)."""
    ep = list(range(1, 21))
    ls = [1.0 - 0.03 * i for i in range(20)]   # slope -0.03/ep, |rel| ~ 0.06 >> 1e-3
    assert _stage_converged(ep, ls, min_stage_epochs=10, plateau_rel_eps=1e-3, plateau_windows=4) is False


def test_stage_converged_gentle_slope_under_eps_fires():
    """A very gentle slope BELOW plateau_rel_eps counts as converged (the CE ~-4.4e-7/ep regime)."""
    ep = list(range(1, 21))
    ls = [1.0 - 1e-5 * i for i in range(20)]   # |rel| ~ 1e-5 << 1e-3
    assert _stage_converged(ep, ls, min_stage_epochs=10, plateau_rel_eps=1e-3, plateau_windows=4) is True


def test_stage_converged_too_few_losses_blocks():
    """Fewer losses than the window => cannot compute the slope => False."""
    assert _stage_converged(list(range(1, 21)), [0.5, 0.5],
                            min_stage_epochs=1, plateau_rel_eps=1e-3, plateau_windows=4) is False


def test_stage_converged_nonfinite_blocks():
    """A non-finite in the window => False (never fire on a NaN/inf spike)."""
    ep = list(range(1, 21))
    ls = [0.5] * 18 + [float("nan"), 0.5]
    assert _stage_converged(ep, ls, min_stage_epochs=10, plateau_rel_eps=1e-3, plateau_windows=4) is False


# ───────────────────── current-stage read (no fire / no mutation) ─────────────────────

def test_current_stage_form_reads_without_firing():
    assert _evt_current_stage_form({"tau": None, "l7": None}) == "ce"
    assert _evt_current_stage_form({"tau": 100, "l7": None}) == "tau_softplus"
    assert _evt_current_stage_form({"tau": 100, "l7": 300}) == "l7_softplus"


# ───────────────────────── BYTE-IDENTITY: never-plateau == hardcoded ─────────────────────────

def test_hardcoded_schedule_pinned():
    """Pin ``_seg_form_for_epoch`` (the OFF path) so an accidental change to the hardcoded schedule
    is caught. The event-triggered OFF branch calls THIS function verbatim."""
    a = _args(tau=300, l7=800, epochs=1500)
    assert _seg_form_for_epoch(1, a) == "ce"
    assert _seg_form_for_epoch(299, a) == "ce"
    assert _seg_form_for_epoch(300, a) == "tau_softplus"
    assert _seg_form_for_epoch(799, a) == "tau_softplus"
    assert _seg_form_for_epoch(800, a) == "l7_softplus"
    assert _seg_form_for_epoch(1500, a) == "l7_softplus"
    assert _seg_form_for_epoch(400, _args(curriculum=False, seg_loss="ce")) == "ce"


def test_never_plateau_reproduces_hardcoded_schedule_and_caps_force():
    """THE byte-identity proof at the controller layer: a loss that NEVER plateaus (exponential
    decay, constant large relative slope) makes the resolver reproduce ``_seg_form_for_epoch`` at
    EVERY epoch, with both transitions firing EXACTLY at the hardcoded caps (cap = hard ceiling)."""
    a = _args(tau=30, l7=80, epochs=120, min_stage=10, windows=4, rel_eps=1e-3)
    forms, events, state = _drive(a, lambda ep: 100.0 * (0.9 ** ep), 120)
    for ep in range(1, 121):
        assert forms[ep - 1] == _seg_form_for_epoch(ep, a), f"seg_form drift at ep {ep}"
    assert [(e["epoch"], e["trigger"]) for e in events] == [(30, "cap"), (80, "cap")]
    assert state["tau"] == 30 and state["l7"] == 80


def test_off_branch_calls_hardcoded_verbatim():
    """Source guard: the event-triggered OFF branch in the epoch loop must call the ORIGINAL
    ``_seg_form_for_epoch(ep, args)`` verbatim (the literal byte-identical #205 path)."""
    src = _TRAINER.read_text()
    assert "seg_form = _seg_form_for_epoch(ep, args)" in src, \
        "the OFF branch must call the hardcoded curriculum fn verbatim (byte-identical)"


def test_cap_forces_even_before_min_stage():
    """The cap is a HARD CEILING: it fires even when min_stage_epochs has NOT elapsed."""
    a = _args(tau=5, l7=9, epochs=12, min_stage=10, windows=4)
    forms, events, _ = _drive(a, lambda ep: 100.0 * (0.9 ** ep), 12)
    assert [(e["epoch"], e["trigger"]) for e in events] == [(5, "cap"), (9, "cap")]
    for ep in range(1, 13):
        assert forms[ep - 1] == _seg_form_for_epoch(ep, a)


# ───────────────────────── plateau fires EARLY + immutability ─────────────────────────

def test_plateau_fires_earlier_than_cap():
    """A loss that goes flat well before the cap => the transition fires EARLY on the plateau,
    not at the cap (the whole point of the feature)."""
    a = _args(tau=200, l7=500, epochs=600, min_stage=10, windows=4, rel_eps=1e-3)
    # decreasing to 0.10 by ep 18, then flat 0.10 forever
    forms, events, _ = _drive(a, lambda ep: max(0.10, 1.0 - 0.05 * ep), 600)
    ce_to_tau = next(e for e in events if e["from"] == "ce")
    assert ce_to_tau["trigger"] == "loss_plateau"
    assert ce_to_tau["epoch"] < 200, "must fire EARLIER than the tau cap"
    assert ce_to_tau["epoch"] == 22


def test_boundaries_immutable_once_resolved():
    """Each boundary fires exactly once (ce->tau then tau->l7); the resolved state is terminal."""
    a = _args(tau=5, l7=9, epochs=20, min_stage=1, windows=2)
    _, events, state = _drive(a, lambda ep: 0.5, 20)   # flat => plateau fires as soon as legal
    assert [e["from"] for e in events] == ["ce", "tau_softplus"]
    assert state["tau"] is not None and state["l7"] is not None
    assert _evt_current_stage_form(state) == "l7_softplus"


# ───────────────────────── DETERMINISM: same history => same fired epochs ─────────────────────────

def test_same_loss_history_same_fired_boundaries():
    """The fired transition epochs are a deterministic function of the (seeded) loss history: two
    runs over the IDENTICAL history yield IDENTICAL fired epochs + resolved boundaries."""
    a = _args(tau=200, l7=500, epochs=600, min_stage=10, windows=4, rel_eps=1e-3)
    loss_fn = lambda ep: max(0.10, 1.0 - 0.05 * ep)   # early plateau at 0.10

    def run():
        state = {"tau": None, "l7": None, "stage_start": 1, "losses": []}
        fired = []
        for ep in range(1, 601):
            _sf, ev = _evt_resolve_seg_form(ep, state, a)
            if ev is not None:
                fired.append((ev["from"], ev["epoch"], ev["trigger"]))
            state["losses"].append(float(loss_fn(ep)))
        return fired, state["tau"], state["l7"]

    r1 = run()
    r2 = run()
    assert r1 == r2, f"nondeterministic fired boundaries: {r1} != {r2}"
    fired, tau, l7 = r1
    # both fired EARLY via plateau (history-determined, not the cap): ce->tau @22, tau->l7 @32
    assert fired == [("ce", 22, "loss_plateau"), ("tau_softplus", 32, "loss_plateau")]
    assert tau == 22 and l7 == 32


def test_l7_never_fires_when_demoted():
    """(C2 SEAL-review guard) l7 is the measured-defect stage, demoted via l7-start >= epochs
    ("never"). The convergence trigger must honor that: a hard tau plateau must NOT fire l7."""
    import importlib.util as _ilu
    from argparse import Namespace
    from pathlib import Path
    _t = Path(__file__).resolve().parent / "train_levelset_witness_realized_through_R_mlx.py"
    _sp = _ilu.spec_from_file_location("_lv_c2", _t)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    args = Namespace(curriculum=True, seg_loss="ce", tau_softplus_start_epoch=300,
                     l7_start_epoch=1000, epochs=1000,
                     curriculum_min_stage_epochs=5, curriculum_plateau_rel_eps=1e-3,
                     curriculum_plateau_windows=3)
    state = {"tau": 300, "l7": None, "stage_start": 300, "losses": []}
    # feed a HARD tau plateau (identical losses) far past min-stage: must stay tau, never l7
    for ep in range(301, 400):
        state["losses"].append(100.0)
        form, event = _m._evt_resolve_seg_form(ep, state, args)
        assert form == "tau_softplus", f"l7 fired at ep {ep} despite l7_start>=epochs"
        assert event is None or event.get("to") != "l7_softplus"
    assert state["l7"] is None
    # control: with l7 GENUINELY scheduled (l7_start < epochs) the same plateau MAY fire it
    args2 = Namespace(**{**vars(args), "l7_start_epoch": 500, "epochs": 1000})
    state2 = {"tau": 300, "l7": None, "stage_start": 300, "losses": []}
    fired = False
    for ep in range(301, 520):
        state2["losses"].append(100.0)
        form, event = _m._evt_resolve_seg_form(ep, state2, args2)
        if event is not None and event.get("to") == "l7_softplus":
            fired = True
            break
    assert fired, "guard must NOT block a genuinely-scheduled l7"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
