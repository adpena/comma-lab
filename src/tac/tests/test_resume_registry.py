"""Tests for the canonical RESUMABILITY REGISTRY (tac.witness_control.resume_registry).

The CLASS-fix for the "forgot to route a stateful gate into resume" bug (operator 2026-07-08). Proves:
  (A) the Resumable protocol + registry register/dedupe/prefix discipline;
  (B) BYTE-IDENTITY: an all-cap-only (event-mode OFF) registry emits {} with NO manifest;
  (C) the manifest round-trips + completeness self-protection (vanished EVENT state fails closed;
      benign cases warn, not raise);
  (D) the DECISIVE crash-resume equivalence: fresh gates restored via the registry are bit-identical
      to the uninterrupted continuation (fired state, fired epoch, fired_by, detector history);
  (E) TrailingSeriesResumable persists the chroma detector window so an unfired plateau re-fires
      bit-faithfully;
  (F) STATIC COVERAGE: every EventBackstopGate constructed in the live trainer is registered (a new
      gate cannot ship un-persisted) — the class gate.
  (G) NON-GATE FOLD (2026-07-08): the FunctionResumable adapter round-trips byte-identically to a
      legacy direct call; a non-event controller writing does NOT stamp a manifest (byte-identity for
      the always-on rng); when an event gate DOES stamp one, the non-event controllers are listed for
      vanish coverage (a warning, never the event fail-closed).

These are CODE-CORRECTNESS tests (bit-equality of control-flow state), NOT a score claim — no n600.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.event_wirings import (
    EventBackstopGate,
    annulus_plateau_event,
    muon_gate_restore_from_cfg,
    muon_gate_state_arrays,
)
from tac.witness_control.resume_registry import (
    CHROMA_HISTORY_PREFIX,
    GATE_KEY_PREFIXES,
    RESUME_REGISTRY_MANIFEST_KEY,
    FunctionResumable,
    Resumable,
    ResumeIntegrityError,
    ResumeRegistry,
    TrailingSeriesResumable,
    build_gate_resume_registry,
)

_TRAINER = Path(__file__).resolve().parents[3] / "experiments" / \
    "train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer():
    """Import the levelset trainer module by path to reach its resume free functions (module-level,
    MLX-free -- the heavy mx imports live inside run_train)."""
    spec = importlib.util.spec_from_file_location("_levelset_trainer_for_test", _TRAINER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────
def _gate(name: str, prefix_cap: int, *, event: bool) -> EventBackstopGate:
    return EventBackstopGate(
        name=name, start_epoch_flag=f"--{name}-start-epoch",
        start_event_flag=f"--{name}-start-event",
        sensor=("some_sensor" if event else None), cap=prefix_cap)


def _sidecar_parse(arrays: dict) -> dict:
    """Emulate the trainer's ``_load_resume_state`` cfg parse for ``__``-prefixed arrays."""
    return {k: (v.item() if np.asarray(v).size == 1 else np.asarray(v).tolist())
            for k, v in arrays.items()}


# ── (A) protocol + register discipline ────────────────────────────────────────────────────────
def test_event_backstop_gate_satisfies_resumable_protocol():
    assert isinstance(_gate("muon", 726, event=True), Resumable)
    assert isinstance(TrailingSeriesResumable({}, "k"), Resumable)


def test_register_rejects_duplicate_name_prefix_and_non_resumable():
    reg = ResumeRegistry()
    reg.register("muon", "__mg_", _gate("muon", 726, event=True))
    with pytest.raises(ValueError, match="duplicate controller name"):
        reg.register("muon", "__x_", _gate("muon", 1, event=True))
    with pytest.raises(ValueError, match="duplicate key prefix"):
        reg.register("other", "__mg_", _gate("other", 1, event=True))
    with pytest.raises(ValueError, match="must start with"):
        reg.register("bad", "nope_", _gate("bad", 1, event=True))
    with pytest.raises(TypeError, match="Resumable protocol"):
        reg.register("obj", "__o_", object())  # type: ignore[arg-type]


def test_register_rejects_key_outside_prefix():
    class _Bad:
        def state_arrays(self, prefix):
            return {"__WRONG_key": np.asarray(1)}

        def restore_from_cfg(self, prefix, cfg):
            return False

    reg = ResumeRegistry()
    reg.register("bad", "__b_", _Bad())
    with pytest.raises(ValueError, match="outside its.*prefix"):
        reg.state_arrays()


# ── (B) byte-identity: all cap-only => {} + no manifest ────────────────────────────────────────
def test_all_cap_only_registry_emits_nothing_byte_identical():
    reg = build_gate_resume_registry([
        _gate("muon", 726, event=False),
        _gate("lane_band", 300, event=False),
        _gate("seg_chroma_boundary", 0, event=False),
    ])
    assert reg.state_arrays() == {}          # ZERO keys, NO manifest => sidecar byte-identical
    # a legacy/empty cfg restore => every gate False, no warnings, no raise.
    report = reg.restore({})
    assert report.legacy is True and report.manifest_present is False
    assert report.restored == {"muon": False, "lane_band": False, "seg_chroma_boundary": False}
    assert report.warnings == []


def test_chroma_history_not_registered_when_chroma_cap_only():
    reg = build_gate_resume_registry(
        [_gate("muon", 726, event=False), _gate("lane_band", 300, event=False),
         _gate("seg_chroma_boundary", 0, event=False)],
        wire_sense={"annulus_series": [(1, 0.5)]})
    assert "seg_chroma_boundary_history" not in reg.names   # only registered when chroma event-mode


# ── (C) manifest round-trip + completeness self-protection ─────────────────────────────────────
def test_manifest_stamped_only_when_state_written():
    g = _gate("lane_band", 300, event=True)
    g.update(120, event_fired=True)          # fire it => it writes keys
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", g)
    arrays = reg.state_arrays()
    assert RESUME_REGISTRY_MANIFEST_KEY in arrays
    assert "__lbg_fired_epoch" in arrays and "__lbg_fired_by" in arrays


def test_event_mode_unfired_gate_stamps_manifest_no_window():
    """B5 REVISE-2 (the pre-first-event-fire window, EMPIRICALLY CLOSED): an event-mode run with ALL
    gates UNFIRED (the ep0..muon-fire window) STILL stamps the manifest at checkpoint 1 — because an
    event-mode-ON but unfired gate writes its ``fired_epoch=-1`` sentinel keys (non-empty => counted as
    an event write). So every co-writing non-event controller IS vanish-protected from the first save;
    there is NO manifest-free window in event mode. (The CLOCK/cap-only registry stays manifest-free by
    design — its byte-identity contract — see the sister ``test_all_cap_only...``.)"""
    reg = build_gate_resume_registry([
        _gate("muon", 726, event=True),
        _gate("lane_band", 300, event=True),
        _gate("seg_chroma_boundary", 400, event=True),
    ])
    # simulate the pre-fire window: many epochs, no gate ever fed an event_fired.
    # (build_gate_resume_registry holds the gates; drive them directly for the unfired state.)
    arrays = reg.state_arrays()
    assert RESUME_REGISTRY_MANIFEST_KEY in arrays, (
        "an all-unfired EVENT-mode registry MUST stamp the manifest at ckpt 1 (sentinel keys are a "
        "write) — else a pre-fire crash-resume silently legacy-restores the non-event controllers")
    # the unfired sentinels are present (the write that stamps the manifest).
    assert int(_sidecar_parse(arrays)["__mg_fired_epoch"]) == -1
    # sister invariant: a cap-only (clock) registry has NO manifest (byte-identity contract preserved).
    clock = build_gate_resume_registry([
        _gate("muon", 726, event=False),
        _gate("lane_band", 300, event=False),
        _gate("seg_chroma_boundary", 0, event=False),
    ])
    assert RESUME_REGISTRY_MANIFEST_KEY not in clock.state_arrays()


def test_vanished_event_state_fails_closed():
    """Manifest says an EVENT gate persisted, but its keys are gone => ResumeIntegrityError (a
    corrupt sidecar; silently re-arming would break resume-determinism)."""
    g = _gate("lane_band", 300, event=True)
    g.update(120, event_fired=True)
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", g)
    cfg = _sidecar_parse(reg.state_arrays())
    # simulate corruption: drop the persisted keys, keep the manifest.
    cfg.pop("__lbg_fired_epoch")
    cfg.pop("__lbg_fired_by")
    reg2 = ResumeRegistry()
    reg2.register("lane_band", "__lbg_", _gate("lane_band", 300, event=True))
    with pytest.raises(ResumeIntegrityError, match="persisted_state_vanished"):
        reg2.restore(cfg)


def test_event_controller_added_since_checkpoint_warns_not_raises():
    """A checkpoint written with only muon in event mode; resume adds an event-mode lane_band gate =>
    a WARNING (re-arms), never a raise."""
    gm = _gate("muon", 726, event=True)
    gm.update(600, event_fired=True)
    reg = ResumeRegistry()
    reg.register("muon", "__mg_", gm)
    cfg = _sidecar_parse(reg.state_arrays())      # manifest lists only muon
    reg2 = ResumeRegistry()
    reg2.register("muon", "__mg_", _gate("muon", 726, event=True))
    reg2.register("lane_band", "__lbg_", _gate("lane_band", 300, event=True))  # event, not in manifest
    report = reg2.restore(cfg)
    assert report.restored["muon"] is True
    stages = {w["stage"] for w in report.warnings}
    assert "resume_registry_event_controller_added_since_checkpoint" in stages


def test_legacy_sidecar_no_manifest_event_gate_warns():
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", _gate("lane_band", 300, event=True))
    report = reg.restore({})                       # no manifest => legacy path
    assert report.legacy is True
    assert any(w["stage"] == "resume_registry_legacy_no_manifest" for w in report.warnings)


def test_legacy_pre_registry_muon_sidecar_still_restores():
    """A pre-registry MAJOR-1 sidecar carries bare ``__mg_fired_epoch``/``__mg_fired_by`` and NO
    manifest. The registry must STILL restore the muon fire state (backward compat with every sealed
    event-muon sidecar), emitting only a benign legacy warning."""
    legacy_cfg = {"__mg_fired_epoch": 611, "__mg_fired_by": "event"}  # exactly what old code wrote
    reg = build_gate_resume_registry([
        _gate("muon", 726, event=True),          # event-muon resume
        _gate("lane_band", 300, event=False),
        _gate("seg_chroma_boundary", 0, event=False),
    ])
    report = reg.restore(legacy_cfg)
    assert report.restored["muon"] is True
    assert report.legacy is True                 # no manifest
    assert any(w["stage"] == "resume_registry_legacy_no_manifest" for w in report.warnings)


# ── (D) DECISIVE crash-resume equivalence ──────────────────────────────────────────────────────
def test_crash_resume_all_gates_bit_identical_to_uninterrupted():
    """Run K epochs with all three gates in EVENT mode firing at distinct epochs; checkpoint via the
    registry; restore into FRESH gates via a fresh registry; assert every fired-state field is
    bit-identical to the uninterrupted continuation. THE decisive proof of the class-fix."""
    # uninterrupted reference gates.
    ref = {
        "muon": _gate("muon", 726, event=True),
        "lane_band": _gate("lane_band", 300, event=True),
        "seg_chroma_boundary": _gate("seg_chroma_boundary", 400, event=True),
    }
    fire_at = {"muon": 610, "lane_band": 118, "seg_chroma_boundary": 355}
    reg_ref = build_gate_resume_registry(list(ref.values()))
    CRASH = 620
    for ep in range(1, CRASH + 1):
        for nm, g in ref.items():
            g.update(ep, event_fired=(ep == fire_at[nm]))

    # checkpoint at CRASH -> sidecar -> parse (mimic _load_resume_state).
    cfg = _sidecar_parse(reg_ref.state_arrays())

    # fresh gates + fresh registry restore.
    fresh = {
        "muon": _gate("muon", 726, event=True),
        "lane_band": _gate("lane_band", 300, event=True),
        "seg_chroma_boundary": _gate("seg_chroma_boundary", 400, event=True),
    }
    reg_fresh = build_gate_resume_registry(list(fresh.values()))
    report = reg_fresh.restore(cfg)

    for nm in ref:
        assert fresh[nm].fired == ref[nm].fired, nm
        assert fresh[nm].fired_epoch == ref[nm].fired_epoch == fire_at[nm], nm
        assert fresh[nm]._fired_by == ref[nm]._fired_by == "event", nm
        assert report.restored[nm] is True, nm

    # and the restored gates keep latching identically through the continuation.
    for ep in range(CRASH + 1, CRASH + 20):
        for nm in ref:
            s_ref = ref[nm].update(ep, event_fired=False)
            s_new = fresh[nm].update(ep, event_fired=False)
            assert (s_ref.start_reached, s_ref.just_fired) == (s_new.start_reached, s_new.just_fired), nm


def test_crash_resume_unfired_gate_stays_fresh_and_rearms():
    """An event gate NOT yet fired at the crash persists the -1 sentinel; restore leaves it fresh so it
    re-arms from post-resume history (matches the sealed muon-unfired contract)."""
    g = _gate("lane_band", 300, event=True)
    for ep in range(1, 100):
        g.update(ep, event_fired=False)          # never fires
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", g)
    cfg = _sidecar_parse(reg.state_arrays())
    assert int(cfg["__lbg_fired_epoch"]) == -1   # sentinel
    g2 = _gate("lane_band", 300, event=True)
    reg2 = ResumeRegistry()
    reg2.register("lane_band", "__lbg_", g2)
    report = reg2.restore(cfg)
    assert report.restored["lane_band"] is False
    assert g2.fired is False and g2.fired_epoch is None


# ── (E) chroma detector history round-trip => bit-faithful plateau re-fire ──────────────────────
def test_chroma_detector_history_roundtrip_refires_identically():
    """The annulus-plateau detector inspects only the trailing dwell window; persisting that window
    makes an UNFIRED-mid-dwell plateau re-fire at the identical decision after a crash."""
    # a flat, dwelled series that the plateau detector fires on.
    series = [(100, 0.010), (200, 0.0100), (300, 0.0100), (400, 0.0100)]
    ref_fires = annulus_plateau_event(series)["fired"]
    assert ref_fires is True                     # sanity: this window fires

    wire = {"annulus_series": list(series)}
    chroma = _gate("seg_chroma_boundary", 999, event=True)
    reg = build_gate_resume_registry(
        [_gate("muon", 726, event=False), _gate("lane_band", 300, event=False), chroma],
        wire_sense=wire)
    assert "seg_chroma_boundary_history" in reg.names
    cfg = _sidecar_parse(reg.state_arrays())
    assert CHROMA_HISTORY_PREFIX + "ep" in cfg

    # fresh wire_sense (empty) + restore -> the detector sees the SAME window -> SAME fire decision.
    wire2 = {"annulus_series": []}
    chroma2 = _gate("seg_chroma_boundary", 999, event=True)
    reg2 = build_gate_resume_registry(
        [_gate("muon", 726, event=False), _gate("lane_band", 300, event=False), chroma2],
        wire_sense=wire2)
    reg2.restore(cfg)
    assert annulus_plateau_event(wire2["annulus_series"])["fired"] == ref_fires
    assert wire2["annulus_series"] == list(series)


def test_trailing_series_bounds_to_cap():
    wire = {"s": [(i, float(i)) for i in range(100)]}
    tsr = TrailingSeriesResumable(wire, "s", cap=8)
    arrays = tsr.state_arrays("__t_")
    assert len(arrays["__t_ep"]) == 8
    assert arrays["__t_ep"].tolist() == list(range(92, 100))


# ── backward compat: muon wrappers still produce __mg_ keys through the generalized methods ──────
def test_muon_wrappers_backward_compatible_keys():
    g = _gate("muon", 726, event=True)
    g.update(600, event_fired=True)
    arrays = muon_gate_state_arrays(g)
    assert set(arrays) == {"__mg_fired_epoch", "__mg_fired_by"}
    cfg = _sidecar_parse(arrays)
    g2 = _gate("muon", 726, event=True)
    assert muon_gate_restore_from_cfg(g2, cfg) is True
    assert g2.fired_epoch == 600
    # off-mode + None still empty (the byte-identical contract).
    assert muon_gate_state_arrays(_gate("muon", 726, event=False)) == {}
    assert muon_gate_state_arrays(None) == {}


# ── (G) NON-GATE FOLD: FunctionResumable + non-event manifest discipline ────────────────────────
def _legacy_pair(prefix: str):
    """A tiny legacy ``(write, restore)`` pair keyed under ``prefix`` — stands in for a trainer free
    function (e.g. ``_rng_state_arrays`` / ``_cl_state_arrays``). ``store`` is the mutable 'controller
    state' the round-trip restores into."""
    store = {"v": 7}

    def write(_p: str) -> dict:
        return {} if store.get("v") is None else {prefix + "v": np.asarray(int(store["v"]), np.int64)}

    def restore(_p: str, cfg: dict) -> bool:
        if (prefix + "v") not in cfg:
            return False
        store["v"] = int(cfg[prefix + "v"])
        return True

    return write, restore, store


def test_function_resumable_satisfies_protocol_and_is_non_event():
    w, r, _ = _legacy_pair("__x_")
    fr = FunctionResumable(write=w, restore=r)
    assert isinstance(fr, Resumable)
    reg = ResumeRegistry()
    reg.register("x", "__x_", fr)
    # non-event: no event_mode attr => event_active False (never triggers the fail-closed / manifest).
    assert reg._entries[0].event_active is False  # noqa: SLF001


def test_function_resumable_roundtrip_byte_identical_to_legacy_call():
    """The registry-emitted keys+arrays for the adapter MUST equal the legacy direct write call, and a
    restore reconstructs the source state (the binding byte-identity + real-restore contract)."""
    w, r, store = _legacy_pair("__x_")
    fr = FunctionResumable(write=w, restore=r)
    reg = ResumeRegistry()
    reg.register("x", "__x_", fr)
    legacy = w("__x_")                       # what the trainer would have written directly
    emitted = reg.state_arrays()             # what the registry writes (no event gate => no manifest)
    assert RESUME_REGISTRY_MANIFEST_KEY not in emitted
    assert set(emitted) == set(legacy)
    for k in legacy:
        assert np.asarray(emitted[k]).tolist() == np.asarray(legacy[k]).tolist()
    # real restore into a FRESH controller reconstructs the value.
    w2, r2, store2 = _legacy_pair("__x_")
    store2["v"] = 0                          # perturb so the restore is observable
    fr2 = FunctionResumable(write=w2, restore=r2)
    reg2 = ResumeRegistry()
    reg2.register("x", "__x_", fr2)
    rep = reg2.restore(_sidecar_parse(emitted))
    assert rep.restored["x"] is True and store2["v"] == store["v"] == 7


def test_non_event_controller_writing_does_not_stamp_manifest():
    """Folding an ALWAYS-ON non-event controller (rng) alongside cap-only gates must NOT add a manifest
    to a config that had none — the byte-identity contract for the live #205/v7 run."""
    w, r, _ = _legacy_pair("__rng_")
    reg = build_gate_resume_registry([
        _gate("muon", 726, event=False), _gate("lane_band", 300, event=False),
        _gate("seg_chroma_boundary", 0, event=False)])
    reg.register("rng_streams", "__rng_", FunctionResumable(write=w, restore=r))
    arrays = reg.state_arrays()
    assert "__rng_v" in arrays                          # the controller DID write
    assert RESUME_REGISTRY_MANIFEST_KEY not in arrays   # but NO manifest => byte-identical sidecar
    # restore of a manifest-free sidecar is the legacy path (no raise, real restore of the non-event ctl).
    report = reg.restore(_sidecar_parse(arrays))
    assert report.legacy is True and report.manifest_present is False
    assert report.restored["rng_streams"] is True


def test_manifest_lists_non_event_controller_when_event_gate_also_wrote():
    """When an EVENT gate writes (fired), the manifest IS stamped and lists the non-event controllers
    too, so their vanish is covered."""
    g = _gate("lane_band", 300, event=True)
    g.update(120, event_fired=True)
    w, r, _ = _legacy_pair("__cl_")
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", g)
    reg.register("closed_loop", "__cl_", FunctionResumable(write=w, restore=r))
    arrays = reg.state_arrays()
    assert RESUME_REGISTRY_MANIFEST_KEY in arrays
    import json as _json
    names = {m["name"] for m in _json.loads(str(arrays[RESUME_REGISTRY_MANIFEST_KEY]))}
    assert names == {"lane_band", "closed_loop"}


def test_vanished_non_event_state_warns_not_raises():
    """A manifest (forced by an event gate) lists a non-event controller; its keys then vanish =>
    a LOUD warning row, NOT a ResumeIntegrityError (only EVENT-gate vanish fails closed)."""
    g = _gate("lane_band", 300, event=True)
    g.update(120, event_fired=True)
    w, r, _ = _legacy_pair("__cl_")
    reg = ResumeRegistry()
    reg.register("lane_band", "__lbg_", g)
    reg.register("closed_loop", "__cl_", FunctionResumable(write=w, restore=r))
    cfg = _sidecar_parse(reg.state_arrays())
    cfg.pop("__cl_v")                                   # simulate the non-event state vanishing
    reg2 = ResumeRegistry()
    reg2.register("lane_band", "__lbg_", _gate("lane_band", 300, event=True))
    w2, r2, _ = _legacy_pair("__cl_")
    reg2.register("closed_loop", "__cl_", FunctionResumable(write=w2, restore=r2))
    report = reg2.restore(cfg)                          # must NOT raise
    stages = {x["stage"] for x in report.warnings}
    assert "resume_registry_persisted_state_vanished" in stages
    assert any(x.get("controller") == "closed_loop" and not x.get("event") for x in report.warnings)


# ── (F) STATIC COVERAGE — the class gate: every trainer gate is registered ──────────────────────
def test_every_trainer_ebgate_has_canonical_prefix_and_is_registered():
    """Scan the live trainer for EventBackstopGate constructions. Assert (1) every gate name has a
    canonical key prefix in GATE_KEY_PREFIXES, and (2) every gate closure-var is passed into the
    build_gate_resume_registry([...]) call — so a NEW gate cannot ship without a resume prefix AND
    without being routed through the registry. This is the structural self-protection."""
    src = _TRAINER.read_text()
    gate_names = set(re.findall(r'_EBGate\(\s*name="([^"]+)"', src))
    assert gate_names, "no _EBGate(name=...) constructions found — did the alias change?"
    assert gate_names == set(GATE_KEY_PREFIXES), (
        f"trainer gate names {sorted(gate_names)} != GATE_KEY_PREFIXES {sorted(GATE_KEY_PREFIXES)} — "
        "add the new gate's canonical key prefix to resume_registry.GATE_KEY_PREFIXES")

    gate_vars = set(re.findall(r'(\w+)\s*=\s*_EBGate\(', src))
    assert len(gate_vars) == len(gate_names), "each gate should bind a distinct closure var"
    m = re.search(r'_build_resume_registry\(\s*\[([^\]]*)\]', src)
    assert m is not None, "the trainer must build the registry via _build_resume_registry([...])"
    reg_list = m.group(1)
    for gv in gate_vars:
        assert gv in reg_list, (
            f"gate var {gv!r} is constructed but NOT passed into build_gate_resume_registry([...]) — "
            "it would be silently unpersisted on resume")


# ── (H) NON-GATE FOLD — trainer byte-identity + widened static coverage (#358) ───────────────────
def test_evt_state_arrays_canonical_keys_and_roundtrip():
    """The extracted _evt_state_arrays produces EXACTLY the former inline block's keys and round-trips
    through _evt_restore_from_cfg (byte-identical keys + real restore)."""
    m = _load_trainer()
    evt = {"tau": 300, "l7": None, "stage_start": 305, "losses": [0.1, 0.2], "nucleus_ready": False}
    arrays = m._evt_state_arrays(evt)
    assert set(arrays) == {"__evt_boundary_tau", "__evt_boundary_l7", "__evt_stage_start",
                           "__evt_stage_losses", "__evt_nucleus_ready"}
    assert m._evt_state_arrays(None) == {}                       # OFF => {} (byte-identical)
    cfg = _sidecar_parse(arrays)
    fresh = {"tau": None, "l7": None, "stage_start": 1, "losses": [], "nucleus_ready": True}
    assert m._evt_restore_from_cfg(fresh, cfg) is True
    assert fresh["tau"] == 300 and fresh["l7"] is None and fresh["stage_start"] == 305
    assert fresh["losses"] == [0.1, 0.2] and fresh["nucleus_ready"] is False
    assert m._evt_restore_from_cfg(fresh, {}) is False          # no keys => no restore
    assert m._evt_restore_from_cfg(None, cfg) is False


def test_nongate_registry_byte_identical_to_legacy_direct_calls():
    """The registry-emitted keys+arrays for the folded non-gate controllers EQUAL the legacy direct
    free-function calls, and a cap-only gate set stamps NO manifest (the byte-identity contract)."""
    m = _load_trainer()
    evt = {"tau": 300, "l7": 450, "stage_start": 305, "losses": [0.1, 0.2], "nucleus_ready": True}
    cl = {"bumps": 1, "bump_add": 0.5, "post_budget_windows": 0, "stop_epoch": None,
          "verdicts": [{"epoch": 10, "d_seg": 0.01, "ep_loss": 0.5, "seg_form": "ce"}], "pending": None}
    rng = np.random.default_rng(123)
    legacy = {**m._evt_state_arrays(evt), **m._cl_state_arrays(cl, cl["verdicts"]),
              **m._rng_state_arrays(rng)}                        # tau None (clock) => {}
    reg = build_gate_resume_registry([
        _gate("muon", 726, event=False), _gate("lane_band", 300, event=False),
        _gate("seg_chroma_boundary", 0, event=False)])
    reg.register("evt_curriculum", "__evt_", FunctionResumable(
        write=lambda _p: m._evt_state_arrays(evt), restore=lambda _p, c: False))
    reg.register("closed_loop", "__cl_", FunctionResumable(
        write=lambda _p: m._cl_state_arrays(cl, cl["verdicts"]), restore=lambda _p, c: False))
    reg.register("rng_streams", "__rng_", FunctionResumable(
        write=lambda _p: m._rng_state_arrays(rng), restore=lambda _p, c: False))
    from tac.witness_control.tau_advance import tau_advance_state_arrays as _tw
    reg.register("tau_advance", "__ta_", FunctionResumable(
        write=lambda _p: _tw(None), restore=lambda _p, c: False))   # clock mode => {} (byte-identical)
    emitted = reg.state_arrays()
    assert RESUME_REGISTRY_MANIFEST_KEY not in emitted           # cap-only + non-event => NO manifest
    assert set(emitted) == set(legacy), (set(emitted) ^ set(legacy))
    for k in legacy:
        assert np.asarray(emitted[k]).tolist() == np.asarray(legacy[k]).tolist(), k


def test_every_nongate_state_arrays_producer_is_registered():
    """WIDENED static class-gate: the four canonical non-gate controllers are registered under their
    canonical prefixes, and EVERY *_state_arrays producer in the trainer (except the model-tensor
    builder) is wired into the #358 registry fold region — a NEW non-gate producer cannot ship
    unpersisted."""
    src = _TRAINER.read_text()
    regs = dict(re.findall(r'_resume_registry\.register\(\s*"([^"]+)"\s*,\s*"([^"]+)"', src))
    for name, prefix in {"rng_streams": "__rng_", "closed_loop": "__cl_",
                         "tau_advance": "__ta_", "evt_curriculum": "__evt_"}.items():
        assert regs.get(name) == prefix, f"{name} not registered under {prefix} (found {regs})"
    marker = "NON-GATE RESUME FOLD (#358)"
    assert marker in src, "the #358 fold region marker is missing"
    fold_region = src.split(marker, 1)[1].split("CURRICULUM stage-transition", 1)[0]
    producers = set(re.findall(r'def (_\w+_state_arrays)\(', src)) - {"_build_resume_state_arrays"}
    producers.add("tau_advance_state_arrays")                    # imported (not def'd) producer
    for p in sorted(producers):
        assert p in fold_region, (
            f"{p} produces resume-sidecar keys but is not wired into the #358 registry fold region — "
            "it would be silently unpersisted (register it as a FunctionResumable there)")
