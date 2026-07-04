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
  * (M2 fix) DECIDE-ON-PREVIOUS: in async mode the decision at eval N consumes rows ending at eval
    N-1 (join previous -> decide -> schedule; NO join of the just-scheduled verdict => the GPU
    never waits the full 2062-2439s verdict wall), and the PENDING-VERDICT sidecar record makes a
    resume cut AT an in-flight verdict reproduce the continuous run's decisions EXACTLY (the
    persisted snapshot inputs round-trip bit-exactly => the recompute is bit-identical).

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
_cl_pending_from_cfg = _m._cl_pending_from_cfg
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
    passes None when OFF (zero __cl_* keys). (M2 fix) the sidecar snapshot moved into the locked
    _cl_sidecar_snapshot helper and the decision into _cl_decide — SAME OFF contract, tracked here
    at the refactored call sites."""
    src = _TRAINER.read_text()
    assert "eik_w_ep = _scheduled_eikonal_weight(ep, args)" in src
    assert 'if _cl_on and _cl_state["bump_add"] > 0.0:' in src
    assert '"--closed-loop-control", action=argparse.BooleanOptionalAction, default=False' in src
    # sidecar: OFF passes None (zero __cl_* keys); ON goes through the locked snapshot helper.
    assert "closed_loop_state=(_cl_sidecar_snapshot() if _cl_on else None)" in src
    # BOTH decision call sites are gated on _cl_on (async decide-on-previous + sync-only block).
    assert src.count("_cl_stop_now = _cl_decide(ep)") == 2
    assert ("if _cl_on and not args.async_verdict and "
            "(ep % args.eval_every == 0 or ep == args.epochs):") in src
    # the OFF flag also gates the flag's per-epoch re-arm default (never fires when OFF).
    assert "_cl_stop_now = False" in src
    # capture appends are gated (async _emit_verdict_row + sync path)
    assert src.count("if _cl_on:\n                _cl_verdicts.append(") >= 1
    # the pending record is only ever SET behind _cl_on (OFF => never set => never persisted).
    assert 'if _cl_on:\n            # (M2 fix) PENDING-VERDICT record' in src


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


# ───────────────────── (M2 fix) DECIDE-ON-PREVIOUS (async reorder) ─────────────────────

def _drive_async(dsegs_losses: list[tuple[int, float, float]], *, bump=0.05, max_bumps=2,
                 stop_after=3, min_sustained=3):
    """Mimic the M2-reordered ASYNC eval point EXACTLY: (1) JOIN the previous eval's verdict —
    its row lands in the history; (2) DECIDE on the rows so far (they end at the PREVIOUS eval);
    (3) THEN schedule this eval's verdict (held in flight; NEVER visible to this decision).
    Returns (actions, state, inflight, verdicts) so a test can also assert the resume boundary
    (the in-flight row is exactly what the PENDING-VERDICT record persists)."""
    state = _fresh_state()
    verdicts: list[dict] = []
    inflight: dict | None = None
    actions: list[tuple[int, str | None, str]] = []
    for ep, d, loss in dsegs_losses:
        if inflight is not None:                      # 1. join the PREVIOUS eval's verdict
            verdicts.append(inflight)
            inflight = None
        if verdicts:                                  # 2. decide on rows < ep (_cl_decide contract)
            c = _cl_classify(verdicts, min_sustained_windows=min_sustained)
            a = _cl_step(c["classification"], state, ep, bump=bump, max_bumps=max_bumps,
                         stop_after=stop_after)
            actions.append((ep, c["classification"], (a or {}).get("action", "none")))
        else:                                         # first eval: no previous row => no decision
            actions.append((ep, None, "none"))
        inflight = _v(ep, d, loss)                    # 3. schedule THIS eval's verdict
        if state["stop_epoch"] is not None:
            break
    return actions, state, inflight, verdicts


def test_decide_on_previous_semantics_async():
    """At eval N the async decision consumes rows ending at eval N-1: every classification/action
    lands EXACTLY one eval window later than the sync (current-row) drive, and the stop epoch
    shifts by exactly one window (25 ep ≪ the ~100-ep erosion timescale)."""
    stream = [(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(12)]
    sync_actions, sync_state = _drive(stream)
    async_actions, async_state, inflight, _ = _drive_async(stream)
    # first async eval decides on NOTHING (no previous verdict yet)
    assert async_actions[0] == (100, None, "none")
    # lag-1: async decision at eval k == sync decision at eval k-1 (classification + action)
    for k in range(1, len(async_actions)):
        assert async_actions[k][1:] == sync_actions[k - 1][1:], f"lag-1 drift at k={k}"
    # the whole trajectory (bumps -> countdown -> stop) fires one window later
    assert sync_state["stop_epoch"] == 250
    assert async_state["stop_epoch"] == 275
    assert async_state["bumps"] == sync_state["bumps"] == 2
    # the just-scheduled verdict is IN FLIGHT at the stop (scheduled even when stopping —
    # the post-loop join lands it as telemetry; the decision never saw it)
    assert inflight is not None and inflight["epoch"] == 275


def test_async_determinism_same_history_same_actions():
    stream = [(100 + 25 * i, 0.005 + (1e-4 * i if i < 9 else -5e-5), 150.0 - 1.5 * i)
              for i in range(14)]
    assert _drive_async(stream) == _drive_async(stream)


# ───────────────── (M2 fix) PENDING-VERDICT record: roundtrip + resume-reconcile ─────────────────

def _pend_rec(row: dict, *, temp=0.87654321, beta=3.1400001):
    """A pending record exactly as _schedule_async_verdict captures it, with a shape-diverse
    fp32 shadow (incl. a size-1 array — the .item() flatten case the reconcile reshapes)."""
    return {"epoch": row["epoch"], "seg_form": row["seg_form"], "ep_loss": row["ep_loss"],
            "softmax_temp": temp, "hosc_beta": beta,
            "ema_np": {"film.weight": np.arange(6, dtype=np.float32).reshape(2, 3) / 7.0,
                       "film.bias": np.asarray([0.333333343], np.float32),   # size-1 -> .item()
                       "code": np.linspace(-1, 1, 8, dtype=np.float32).reshape(4, 2)}}


def test_pending_record_roundtrip_bitexact_and_absent_when_none():
    """The pending record's inputs round-trip the sidecar BIT-EXACTLY (fp32 -> tolist float64 ->
    fp32 is lossless; scalars exact) — this is what buys bit-identity of the resume recompute.
    No pending => ZERO __cl_pend_* keys (the pre-M2 sidecar shape)."""
    # absent when None/missing (OFF and no-inflight paths)
    assert not any(k.startswith("__cl_pend") for k in _cl_state_arrays(_fresh_state(), []))
    assert not any(k.startswith("__cl_pend")
                   for k in _cl_state_arrays({**_fresh_state(), "pending": None}, []))
    assert _cl_pending_from_cfg({"__resume_epoch": 100, "__cl_bumps": 0}) is None
    # present + bit-exact roundtrip
    row = _v(425, 0.004321, 111.5)
    pend = _pend_rec(row)
    arrs = _cl_state_arrays({**_fresh_state(), "pending": pend}, [row])
    assert {"__cl_pend_epoch", "__cl_pend_segform", "__cl_pend_eploss", "__cl_pend_temp",
            "__cl_pend_beta"}.issubset(arrs)
    restored = _cl_pending_from_cfg(_sim_loader(arrs))
    assert restored is not None
    assert restored["epoch"] == 425 and restored["seg_form"] == "tau_softplus"
    assert restored["ep_loss"] == row["ep_loss"]
    assert restored["softmax_temp"] == pend["softmax_temp"]
    assert restored["hosc_beta"] == pend["hosc_beta"]
    for k, orig in pend["ema_np"].items():
        got = np.asarray(restored["ema_np"][k], np.float32).reshape(orig.shape)  # reconcile reshape
        assert got.tobytes() == orig.tobytes(), f"shadow {k} not bit-exact through the sidecar"


def test_resume_reconcile_pending_equals_continuous():
    """THE M2 resume crux: cut the async drive AT an eval whose verdict is still in flight,
    persist {state, joined rows, PENDING record} through the sidecar round-trip, restore,
    recompute the pending row from its persisted inputs (deterministic given them — proven
    bit-exact above), append, continue => actions + stop epoch EQUAL the continuous run's."""
    stream = [(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(12)]
    cont_actions, cont_state, _, _ = _drive_async(stream)
    for cut in (1, 3, 5, 7):
        # run the reordered loop up to the cut; the cut eval's verdict is IN FLIGHT (pending)
        actions, state, inflight, verdicts = _drive_async(stream[:cut])
        assert inflight is not None
        # checkpoint: state + joined rows + pending record -> npz-array -> loader-parse boundary
        cfg = _sim_loader(_cl_state_arrays({**state, "pending": _pend_rec(inflight)}, verdicts))
        state2, verdicts2 = _cl_restore_from_cfg(cfg)
        pend = _cl_pending_from_cfg(cfg)
        assert pend is not None and all(v["epoch"] != pend["epoch"] for v in verdicts2)
        # reconcile: recompute the verdict from the persisted inputs (the harness's stand-in for
        # _verdict_from_snapshot: same inputs => same d_seg, proven bit-exact in the test above)
        # and append — EXACTLY the trainer's resume_pending_verdict path.
        verdicts2.append({"epoch": pend["epoch"], "seg_form": pend["seg_form"],
                          "d_seg": inflight["d_seg"], "ep_loss": pend["ep_loss"]})
        # continue the reordered loop over the remaining stream; the reconciled row is the
        # "previous eval's row" the next decision joins-and-consumes.
        inflight2: dict | None = None
        for ep, d, loss in stream[cut:]:
            if inflight2 is not None:
                verdicts2.append(inflight2)
                inflight2 = None
            if verdicts2:
                c = _cl_classify(verdicts2, min_sustained_windows=3)
                a = _cl_step(c["classification"], state2, ep, bump=0.05, max_bumps=2, stop_after=3)
                actions.append((ep, c["classification"], (a or {}).get("action", "none")))
            else:
                actions.append((ep, None, "none"))
            inflight2 = _v(ep, d, loss)
            if state2["stop_epoch"] is not None:
                break
        assert actions == cont_actions, f"resume-reconcile drift at cut={cut}"
        assert state2 == cont_state, f"state drift at cut={cut}"


def test_reconcile_skipped_when_row_already_landed():
    """The locked sidecar invariant's other half: if the worker finished before the checkpoint the
    row is IN verdicts and pending is cleared — but even a (theoretical) sidecar carrying BOTH must
    not double-append: the reconcile is guarded on the epoch being absent."""
    row = _v(300, 0.0042, 120.0)
    cfg = _sim_loader(_cl_state_arrays({**_fresh_state(), "pending": _pend_rec(row)}, [row]))
    _, verdicts = _cl_restore_from_cfg(cfg)
    pend = _cl_pending_from_cfg(cfg)
    # the trainer's guard: all(epoch != pending) is False here => NO recompute, NO append
    assert not all(int(v["epoch"]) != int(pend["epoch"]) for v in verdicts)


# ──────────── (M2-F2 fix) failed pending recompute: DROPPED explicit + deterministic ────────────

def _drive_async_with_failed_verdict(dsegs_losses: list[tuple[int, float, float]], failed_ep: int,
                                     *, bump=0.05, max_bumps=2, stop_after=3, min_sustained=3):
    """``_drive_async`` where the verdict scheduled at ``failed_ep`` FAILS in flight: its row is
    never joined (the worker's ``verdict_async_failed`` path — a failed verdict produces NO row)
    and training continues. This is the CONTINUOUS reference for the M2-F2 fail-safe."""
    state = _fresh_state()
    verdicts: list[dict] = []
    inflight: dict | None = None
    actions: list[tuple[int, str | None, str]] = []
    for ep, d, loss in dsegs_losses:
        if inflight is not None:
            if int(inflight["epoch"]) != int(failed_ep):
                verdicts.append(inflight)
            inflight = None
        if verdicts:
            c = _cl_classify(verdicts, min_sustained_windows=min_sustained)
            a = _cl_step(c["classification"], state, ep, bump=bump, max_bumps=max_bumps,
                         stop_after=stop_after)
            actions.append((ep, c["classification"], (a or {}).get("action", "none")))
        else:
            actions.append((ep, None, "none"))
        inflight = _v(ep, d, loss)
        if state["stop_epoch"] is not None:
            break
    return actions, state


def test_resume_reconcile_failed_recompute_drops_row_matches_continuous_failure():
    """(M2-F2 fix, throughput review 2026-07-04) A pending record whose SYNCHRONOUS recompute
    FAILS on resume is DROPPED — explicitly logged, deterministic, never appended, and never
    crashing a resume the continuous run would have survived. The resumed decision history then
    equals the CONTINUOUS run in which that verdict FAILED in flight (worker produced no row)."""
    stream = [(100 + 25 * i, 0.005 + 1e-4 * i, 150.0 - 2.0 * i) for i in range(12)]
    for cut in (1, 3, 5, 7):
        actions, state2, inflight, verdicts = _drive_async(stream[:cut])
        assert inflight is not None
        cfg = _sim_loader(_cl_state_arrays({**state2, "pending": _pend_rec(inflight)}, verdicts))
        state2, verdicts2 = _cl_restore_from_cfg(cfg)
        pend = _cl_pending_from_cfg(cfg)
        assert pend is not None
        # trainer failure path: the recompute raises => the row is DROPPED (NOTHING appended);
        # the resumed history is exactly the restored rows.
        assert [v["epoch"] for v in verdicts2] == [v["epoch"] for v in verdicts]
        # continuous reference: the SAME run where that eval's worker FAILED (row never joined).
        cont_actions, cont_state = _drive_async_with_failed_verdict(stream, int(pend["epoch"]))
        # continue the reordered loop over the remaining stream with NO reconciled row.
        inflight2: dict | None = None
        for ep, d, loss in stream[cut:]:
            if inflight2 is not None:
                verdicts2.append(inflight2)
                inflight2 = None
            if verdicts2:
                c = _cl_classify(verdicts2, min_sustained_windows=3)
                a = _cl_step(c["classification"], state2, ep, bump=0.05, max_bumps=2, stop_after=3)
                actions.append((ep, c["classification"], (a or {}).get("action", "none")))
            else:
                actions.append((ep, None, "none"))
            inflight2 = _v(ep, d, loss)
            if state2["stop_epoch"] is not None:
                break
        assert actions == cont_actions, f"failed-recompute drop drift at cut={cut}"
        assert state2 == cont_state, f"state drift at cut={cut}"


def test_trainer_reconcile_failed_recompute_is_explicit_logged_gated():
    """(M2-F2 fix) Source guard on the trainer's pending reconcile: the recompute is wrapped in
    try/except, the failure path emits an explicit ``resume_pending_verdict_failed`` row with
    ``action=dropped``, and the append is GATED on success (no silent resurrection, no silent
    drop). The sidecar reshape stays OUTSIDE the guard (a corrupted resume must still raise,
    per NO-FAKE fail-loud)."""
    src = _TRAINER.read_text()
    start = src.index("PENDING-VERDICT reconcile")
    block = src[start:src.index("CURRICULUM stage-transition spike-guard", start)]
    r = block.index("_pshadow = {")
    t = block.index("try:")
    v = block.index("_pv = _verdict_from_snapshot")
    e = block.index("except Exception")
    f = block.index('"stage": "resume_pending_verdict_failed"')
    g = block.index("if _pv is not None:")
    ap = block.index("_cl_verdicts.append")
    assert r < t < v < e < f < g < ap, \
        "reconcile order must be reshape(fail-loud) -> try recompute -> except(log DROPPED) -> gated append"
    assert '"action": "dropped"' in block, "the drop must be machine-readable in the log row"


# ───────────────── (M2 fix) structural wall guard: NO join of the just-scheduled verdict ─────────

def test_async_decision_path_has_no_join_after_schedule():
    """The M2 wall bug was joining the verdict scheduled seconds earlier (full 2062-2439s stall at
    every eval). Structural source guard: in the async+ON eval block the order is JOIN(previous) ->
    DECIDE -> SCHEDULE, with NO join anywhere after the schedule; and _cl_decide itself never
    joins/schedules. The old pattern (decision block joining after the schedule) is GONE."""
    src = _TRAINER.read_text()
    start = src.index("DECIDE-ON-PREVIOUS-VERDICT reorder")
    end = src.index("v = realized_verdict()")          # first line of the sync else-branch
    block = src[start:end]
    j = block.index("_join_async_verdict()")
    d = block.index("_cl_stop_now = _cl_decide(ep)")
    s = block.index("_schedule_async_verdict(ep, seg_form, ep_loss)")
    assert j < d < s, "async+ON order must be join(previous) -> decide -> schedule"
    assert "_join_async_verdict" not in block[s:], "nothing may join the just-scheduled verdict"
    # _cl_decide is pure decision: no join, no schedule (no hidden wall inside the decision)
    dstart = src.index("def _cl_decide(")
    dbody = src[dstart:src.index("\n    def ", dstart)]
    assert "_join_async_verdict" not in dbody and "_schedule_async_verdict" not in dbody
    # exactly the two gated decision call sites remain (async reorder + sync-only block);
    # the old always-join decision block is extinct.
    assert src.count("_cl_stop_now = _cl_decide(ep)") == 2
    assert "if args.async_verdict:\n                    _join_async_verdict()" not in src


def test_off_async_path_keeps_original_final_epoch_join_order():
    """Closed-loop OFF async path (the #205/FEED-em contract): final-epoch join THEN schedule,
    unchanged — the reorder is entirely inside the _cl_on branch."""
    src = _TRAINER.read_text()
    off = src.index("closed-loop OFF async path")
    blk = src[off:src.index("v = realized_verdict()")]
    assert blk.index("_join_async_verdict()") < blk.index(
        "_schedule_async_verdict(ep, seg_form, ep_loss)")
    assert "elif ep == args.epochs:" in src[off - 200:off]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
