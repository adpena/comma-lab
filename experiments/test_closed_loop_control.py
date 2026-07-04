"""Regression guard for the CLOSED-LOOP LEVER CONTROL (#292, build 3 of 3).

Operator 2026-07-04 Tier-3: "closed-loop monitor->lever control, ramp eikonal/lane-prior on creep"
+ "self convergence + early termination using a mathematical system". Verifies the pure classifier
``_cl_classify`` (MUST match tools/witness_control_monitor.classify_trajectory — cross-checked here
by importlib-loading BOTH), the bounded controller ``_cl_step`` (eikonal bump budget -> early-stop
countdown), the composition helper ``_cl_effective_eikonal`` (build-1 schedule + bounded bump), and
the resume-sidecar round-trip ``_cl_state_arrays`` / ``_cl_restore_from_cfg``.

The binding non-negotiables proved here:
  * BYTE-IDENTITY OFF: effective eikonal == the build-1 ``_scheduled_eikonal_weight`` EXACTLY when
    no bump is active; the loop-side composition/capture/decision/sidecar are all gated on
    ``_cl_on`` (source guards); OFF writes ZERO ``__cl_*`` sidecar keys.
  * DETERMINISM: the same d_seg verdict history drives the SAME classifications, the SAME bump
    epochs, and the SAME stop epoch (pure functions; no wall-clock, no thread state).
  * BOUNDED: never exceeds --closed-loop-eikonal-max nor --closed-loop-max-bumps; the cap can never
    pull the weight BELOW the schedule.

Imports the trainer + the monitor by file path (self-contained), mirroring
``test_event_triggered_curriculum.py``. Run:
``.venv/bin/python -m pytest experiments/test_closed_loop_control.py``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_MONITOR = _REPO / "tools" / "witness_control_monitor.py"

_spec = importlib.util.spec_from_file_location("_lv_clc", _TRAINER)
_m = importlib.util.module_from_spec(_spec)
sys.modules["_lv_clc"] = _m          # register BEFORE exec (dataclass/typing machinery looks it up)
_spec.loader.exec_module(_m)

_mspec = importlib.util.spec_from_file_location("_wcm_clc", _MONITOR)
_mon = importlib.util.module_from_spec(_mspec)
sys.modules["_wcm_clc"] = _mon
_mspec.loader.exec_module(_mon)

_cl_classify = _m._cl_classify
_cl_step = _m._cl_step
_cl_effective_eikonal = _m._cl_effective_eikonal
_cl_state_arrays = _m._cl_state_arrays
_cl_restore_from_cfg = _m._cl_restore_from_cfg
_scheduled = _m._scheduled_eikonal_weight

DIVERGING = _m._CL_DIVERGING_ERASING
TRANSIENT = _m._CL_TRANSITION_TRANSIENT
PLATEAU = _m._CL_PLATEAU
CONVERGING = _m._CL_CONVERGING


def _v(ep: int, d: float, loss: float, form: str = "tau_softplus") -> dict:
    return {"epoch": ep, "seg_form": form, "d_seg": d, "ep_loss": loss}


def _erosion(n: int, start_ep: int = 100, step: int = 25) -> list[dict]:
    """d_seg strictly RISING while ep_loss strictly FALLS (the #205 tau-creep signature)."""
    return [_v(start_ep + i * step, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(n)]


def _fresh_state() -> dict:
    return {"bumps": 0, "bump_add": 0.0, "post_budget_windows": 0, "stop_epoch": None}


def _drive(dsegs_losses: list[tuple[int, float, float]], *, bump=0.05, max_bumps=2,
           stop_after=3, min_sustained=3):
    """Mimic the loop's eval-point decision EXACTLY: capture verdict -> classify -> step -> maybe
    stop. Returns (actions, state) so a test can assert the whole deterministic trajectory."""
    state = _fresh_state()
    verdicts: list[dict] = []
    actions: list[tuple[int, str, str]] = []
    for ep, d, loss in dsegs_losses:
        verdicts.append(_v(ep, d, loss))
        c = _cl_classify(verdicts, min_sustained_windows=min_sustained)
        a = _cl_step(c["classification"], state, ep, bump=bump, max_bumps=max_bumps,
                     stop_after=stop_after)
        actions.append((ep, c["classification"], (a or {}).get("action", "none")))
        if state["stop_epoch"] is not None:
            break
    return actions, state


# ───────────────────── parity: _cl_classify MUST MATCH the monitor ─────────────────────

def test_classify_parity_with_monitor():
    """The inline replica and tools/witness_control_monitor.classify_trajectory agree EXACTLY
    (classification + slopes) on erosion / transient / converging / plateau / mixed-stage /
    volatile-ish histories — the MUST-MATCH contract from the build-3 spec."""
    histories = [
        _erosion(6),                                              # sustained erosion
        _erosion(2),                                              # too recent -> transient
        [_v(100 + 25 * i, 0.008 - 3e-4 * i, 150.0 - i) for i in range(6)],   # converging
        [_v(100 + 25 * i, 0.005, 150.0 - i) for i in range(6)],              # plateau (loss falls)
        [_v(100 + 25 * i, 0.005, 130.0) for i in range(5)],                  # plateau (loss flat)
        # mixed stages: within-stage filter must key on the LATEST seg_form only
        [_v(50, 0.009, 200.0, "ce"), _v(75, 0.006, 180.0, "ce")] + _erosion(4, start_ep=100),
        # big swings, near-zero net slope (exercises the volatile CV branch)
        [_v(100 + 25 * i, [0.001, 0.02, 0.001, 0.02, 0.001, 0.02][i], 150.0 - i) for i in range(6)],
        [_v(100, 0.005, 150.0)],                                  # single verdict
    ]
    for h in histories:
        ours = _cl_classify(h)
        theirs = _mon.classify_trajectory(h)
        assert ours["classification"] == theirs.classification, f"class drift on {h}"
        assert ours["d_seg_slope"] == theirs.d_seg_slope_per_ep, f"d_seg slope drift on {h}"
        assert ours["ep_loss_slope"] == theirs.ep_loss_slope_per_ep, f"loss slope drift on {h}"
        assert ours["d_seg_cv"] == theirs.d_seg_cv, f"cv drift on {h}"


def test_classify_empty_raises_like_monitor():
    for fn in (_cl_classify, lambda v: _mon.classify_trajectory(v)):
        try:
            fn([])
            raise AssertionError("empty verdicts must raise")
        except ValueError:
            pass


def test_sustained_vs_transient_distinction():
    """The persistence gate: the SAME rising signal is TRANSIENT below min_sustained_windows
    within-stage verdicts and DIVERGING_ERASING at/after (net-stage slope > 0)."""
    assert _cl_classify(_erosion(2))["classification"] == TRANSIENT
    assert _cl_classify(_erosion(3))["classification"] == DIVERGING
    # a rise that RECOVERS (net-stage slope <= 0) stays transient even with a long history
    rise_recover = ([_v(100 + 25 * i, 0.008 - 4e-4 * i, 160.0 - i) for i in range(6)]
                    + [_v(250 + 25 * i, 0.0062 + 1e-4 * i, 152.0 - i) for i in range(3)])
    c = _cl_classify(rise_recover)
    assert c["classification"] == TRANSIENT and c["net_stage_slope"] <= 0.0


# ───────────────────── controller: bump budget then early-stop ─────────────────────

def test_sustained_erosion_bumps_then_stops():
    """THE closed-loop trajectory: erosion -> bump x max_bumps -> countdown x stop_after ->
    early_stop, with the stop epoch deterministic."""
    stream = [(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(12)]
    actions, state = _drive(stream, bump=0.05, max_bumps=2, stop_after=3)
    kinds = [a[2] for a in actions]
    # ep100: 1 point -> plateau/none; ep125: transient/none; ep150+: sustained erosion.
    assert kinds[:2] == ["none", "none"]
    assert kinds[2:4] == ["eikonal_bump", "eikonal_bump"]
    assert kinds[4:7] == ["stop_countdown", "stop_countdown", "early_stop"]
    assert state["bumps"] == 2 and abs(state["bump_add"] - 0.10) < 1e-12
    assert state["stop_epoch"] == actions[6][0] == 250
    # terminal: once armed, further steps are no-ops
    assert _cl_step(DIVERGING, state, 999, bump=0.05, max_bumps=2, stop_after=3) is None
    assert state["stop_epoch"] == 250


def test_recovery_resets_stop_countdown():
    """Erosion must PERSIST consecutively post-budget: a recovered window resets the countdown."""
    state = _fresh_state()
    state["bumps"] = 2  # budget already spent
    _cl_step(DIVERGING, state, 100, bump=0.05, max_bumps=2, stop_after=3)
    _cl_step(DIVERGING, state, 125, bump=0.05, max_bumps=2, stop_after=3)
    assert state["post_budget_windows"] == 2 and state["stop_epoch"] is None
    _cl_step(CONVERGING, state, 150, bump=0.05, max_bumps=2, stop_after=3)   # recovery
    assert state["post_budget_windows"] == 0
    _cl_step(DIVERGING, state, 175, bump=0.05, max_bumps=2, stop_after=3)
    assert state["post_budget_windows"] == 1 and state["stop_epoch"] is None


def test_transient_plateau_converging_take_no_action():
    for cls in (TRANSIENT, PLATEAU, CONVERGING, "volatile"):
        state = _fresh_state()
        assert _cl_step(cls, state, 100, bump=0.05, max_bumps=2, stop_after=3) is None
        assert state == _fresh_state()


def test_zero_bump_flag_skips_straight_to_countdown():
    """--closed-loop-eikonal-bump 0 => no bump is ever added; erosion goes straight to countdown."""
    actions, state = _drive([(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - i) for i in range(8)],
                            bump=0.0, max_bumps=2, stop_after=2)
    assert "eikonal_bump" not in [a[2] for a in actions]
    assert state["bumps"] == 0 and state["bump_add"] == 0.0 and state["stop_epoch"] is not None


# ───────────────────── BOUNDED: eikonal cap + bump budget ─────────────────────

def test_bounded_never_exceeds_max_or_budget():
    # cap binds
    assert _cl_effective_eikonal(0.05, 10.0, 0.20) == 0.20
    assert _cl_effective_eikonal(0.15, 0.10, 0.20) == 0.20
    # under the cap: exact composition
    assert abs(_cl_effective_eikonal(0.05, 0.10, 0.20) - 0.15) < 1e-15
    # the cap is floored at the schedule: a mis-set max NEVER pulls below the schedule
    assert _cl_effective_eikonal(0.30, 0.05, 0.20) == 0.30
    # bump budget: 20 erosion windows still yield exactly max_bumps bumps
    state = _fresh_state()
    for i in range(20):
        _cl_step(DIVERGING, state, 100 + 25 * i, bump=0.05, max_bumps=2, stop_after=999)
    assert state["bumps"] == 2 and abs(state["bump_add"] - 0.10) < 1e-12


# ───────────────────── BYTE-IDENTITY OFF ─────────────────────

def test_effective_eikonal_off_is_exactly_scheduled():
    """bump_add == 0 (OFF, or ON with no bump fired) => the effective weight IS the build-1
    schedule, bit-for-bit (same float object value; no arithmetic applied)."""
    for s in (0.0, 0.01, 0.05, 0.1, 0.123456789, 1e-9):
        assert _cl_effective_eikonal(s, 0.0, 0.20) == s
        assert _cl_effective_eikonal(s, -1.0, 0.20) == s


def test_off_writes_zero_sidecar_keys_and_loop_is_gated():
    """Source guards (build-2 style): every loop-side surface is gated on _cl_on; the OFF path's
    eik_w_ep assignment is the UNCHANGED build-1 call; the flag defaults to False; the sidecar
    passes None when OFF (zero __cl_* keys)."""
    src = _TRAINER.read_text()
    assert "eik_w_ep = _scheduled_eikonal_weight(ep, args)" in src
    assert 'if _cl_on and _cl_state["bump_add"] > 0.0:' in src
    assert '"--closed-loop-control", action=argparse.BooleanOptionalAction, default=False' in src
    assert "closed_loop_state=({**_cl_state" in src and "if _cl_on else None" in src
    assert "if _cl_on and (ep % args.eval_every == 0 or ep == args.epochs):" in src
    # the deterministic-read join precedes the decision
    assert src.index("if _cl_on and (ep % args.eval_every == 0 or ep == args.epochs):") \
        < src.index('_clc = _cl_classify(')
    # capture appends are gated (async _emit_verdict_row + sync path)
    assert src.count("if _cl_on:\n                _cl_verdicts.append(") >= 1


def test_sidecar_none_means_no_cl_keys():
    """_build_resume_state_arrays contract at the helper layer: OFF passes closed_loop_state=None
    and the writer emits __cl_* ONLY from _cl_state_arrays — so None => zero keys by construction.
    Prove the arrays helper emits exactly the 8 keys, and nothing else references __cl_ writes."""
    arrs = _cl_state_arrays(_fresh_state(), [])
    assert sorted(arrs) == ["__cl_bump_add", "__cl_bumps", "__cl_post_budget_windows",
                            "__cl_stop_epoch", "__cl_v_dseg", "__cl_v_eploss", "__cl_v_epochs",
                            "__cl_v_segform"]
    src = _TRAINER.read_text()
    assert src.count('out["__cl_') == 0 or "closed_loop_state is not None" in src
    assert "if closed_loop_state is not None:" in src


# ───────────────────── resume sidecar round-trip ─────────────────────

def _sim_loader(arrays: dict) -> dict:
    """EXACTLY the _load_resume_state cfg parse: a.item() if size==1 else a.tolist()."""
    out = {}
    for k, a in arrays.items():
        a = np.asarray(a)
        out[k] = a.item() if a.size == 1 else a.tolist()
    return out


def test_sidecar_roundtrip_bit_faithful():
    state = {"bumps": 2, "bump_add": 0.1, "post_budget_windows": 1, "stop_epoch": None}
    verdicts = _erosion(3)
    cfg = _sim_loader(_cl_state_arrays(state, verdicts))
    restored = _cl_restore_from_cfg(cfg)
    assert restored is not None
    rst, rv = restored
    assert rst == state
    assert rv == [{"epoch": v["epoch"], "d_seg": v["d_seg"], "ep_loss": v["ep_loss"],
                   "seg_form": v["seg_form"]} for v in verdicts]
    # single-verdict edge (size==1 arrays -> .item() scalars, not lists)
    cfg1 = _sim_loader(_cl_state_arrays(state, verdicts[:1]))
    _, rv1 = _cl_restore_from_cfg(cfg1)
    assert len(rv1) == 1 and rv1[0]["seg_form"] == "tau_softplus"
    # stop_epoch persists through the -1 sentinel
    cfg2 = _sim_loader(_cl_state_arrays({**state, "stop_epoch": 425}, []))
    rst2, rv2 = _cl_restore_from_cfg(cfg2)
    assert rst2["stop_epoch"] == 425 and rv2 == []


def test_sidecar_prefeature_returns_none():
    assert _cl_restore_from_cfg({"__resume_epoch": 100}) is None


def test_resume_equals_continuous_decisions():
    """Bit-faithful ON-resume: splitting the verdict stream at any point, persisting through the
    sidecar round-trip, and continuing yields the SAME actions + stop epoch as the continuous run."""
    stream = [(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(12)]
    cont_actions, cont_state = _drive(stream)
    for cut in (1, 3, 5):
        state = _fresh_state()
        verdicts: list[dict] = []
        actions = []
        for ep, d, loss in stream[:cut]:
            verdicts.append(_v(ep, d, loss))
            c = _cl_classify(verdicts, min_sustained_windows=3)
            a = _cl_step(c["classification"], state, ep, bump=0.05, max_bumps=2, stop_after=3)
            actions.append((ep, c["classification"], (a or {}).get("action", "none")))
        # checkpoint -> loader parse -> restore (the resume boundary)
        state2, verdicts2 = _cl_restore_from_cfg(
            _sim_loader(_cl_state_arrays(state, verdicts)))
        for ep, d, loss in stream[cut:]:
            verdicts2.append(_v(ep, d, loss))
            c = _cl_classify(verdicts2, min_sustained_windows=3)
            a = _cl_step(c["classification"], state2, ep, bump=0.05, max_bumps=2, stop_after=3)
            actions.append((ep, c["classification"], (a or {}).get("action", "none")))
            if state2["stop_epoch"] is not None:
                break
        assert actions == cont_actions, f"resume drift at cut={cut}"
        assert state2 == cont_state, f"state drift at cut={cut}"


# ───────────────────── DETERMINISM ─────────────────────

def test_same_history_same_actions():
    """The same seeded d_seg trajectory => the same classifications, bump epochs, stop epoch."""
    stream = [(100 + 25 * i, 0.005 + (1e-4 * i if i < 9 else -5e-5), 150.0 - 1.5 * i)
              for i in range(14)]
    r1 = _drive(stream)
    r2 = _drive(stream)
    assert r1 == r2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
