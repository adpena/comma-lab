"""Canonical RESUMABILITY REGISTRY for the level-set witness trainer.

Operator directive 2026-07-08 (verbatim, the CHARTER): *"There is a way to engineer resumability
optimally and you should do it."* This is the CLASS-fix for a bug class that recurred: the trainer's
resume sidecar is an ACCRETION of hand-written per-controller ``(state_arrays, restore_from_cfg)``
pairs (``_cl_*``, ``tau_advance_*``, ``muon_gate_*``, ``_rng_*``), and forgetting to route a NEW
stateful controller into ``_build_resume_state_arrays`` silently breaks resume-determinism — a lever
that was ON before a crash comes back OFF (a config that never existed).

Empirical anchors (all real):
  * SEAL-v7-r1 MAJOR-1: the muon event-gate's sensor-fire epoch was not persisted -> a crash between
    the fire and the backstop cap would restore a fresh AdamW against a Muon checkpoint. Point-fixed
    with ``__mg_*`` keys (commit 2b17e55f6). ONE gate fixed; the other two left unpersisted.
  * Orchestrator verification 2026-07-08: the trainer instantiates THREE latching
    :class:`tac.witness_control.event_wirings.EventBackstopGate` gates (muon / lane_band /
    seg_chroma_boundary) but ONLY the muon gate flowed into resume. A FIRED lane-band/chroma lever
    silently turned OFF on resume until its sensor re-fired; the chroma annulus-plateau detector needs
    4 dwell windows of fresh history, so the chroma lever could stay dark for hundreds of epochs.

The optimal form (NOT another point-fix):
  1. A small :class:`Resumable` protocol (``state_arrays(prefix)`` + ``restore_from_cfg(prefix, cfg)``)
     with a unique per-instance key PREFIX.
  2. A run-scoped :class:`ResumeRegistry`: every stateful controller REGISTERS once, on construction.
     ``_build_resume_state_arrays`` iterates the registry (SYMMETRIC iteration at resume) -> a
     registered controller CANNOT be forgotten at checkpoint time.
  3. Completeness self-protection: a ``__resume_registry_manifest`` key stamped at write time; at
     resume, a manifest entry whose persisted keys have VANISHED is an integrity violation, and for an
     EVENT-mode gate that is a fail-closed refusal to silently re-arm.

BYTE-IDENTITY CONTRACT (binding — the live run resumes with this code): when every registered
controller emits ZERO keys (all gates event-mode OFF — the #205/v6/v7 cap-only path),
:meth:`ResumeRegistry.state_arrays` returns ``{}`` with NO manifest -> the sidecar is byte-identical to
the pre-registry path. The manifest appears ONLY when there is real event-gate state to protect. A
legacy sidecar (no manifest) restores via each controller's own back-compat path (each returns False
for missing keys) -> byte-identical legacy behavior.

means != ends: this is APPARATUS. Only a byte-closed n600 exact row < 0.19110 from
``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "RESUME_REGISTRY_MANIFEST_KEY",
    "GATE_KEY_PREFIXES",
    "DIRECT_CONTROLLER_NAMES",
    "FunctionResumable",
    "Resumable",
    "ResumeIntegrityError",
    "ResumeRegistry",
    "RestoreReport",
    "TrailingSeriesResumable",
    "build_gate_resume_registry",
]

# The manifest key. ``__``-prefixed so ``_load_resume_state`` routes it to ``cfg`` (not a param tensor);
# its value is a JSON string (a size-1 str array -> ``.item()`` -> the python str at load time).
RESUME_REGISTRY_MANIFEST_KEY = "__resume_registry_manifest"

# Canonical per-gate sidecar key PREFIXES. muon MUST stay ``__mg_`` (backward compat with every sidecar
# already written by muon_gate_state_arrays / the live code). New gates get a distinct prefix here; a
# gate whose ``name`` is absent from this map cannot be registered (a fail-closed at construction, so
# the NEXT gate must be added here + covered by the static registry-coverage test).
GATE_KEY_PREFIXES: dict[str, str] = {
    "muon": "__mg_",
    "lane_band": "__lbg_",
    "seg_chroma_boundary": "__cbg_",
}

# The chroma annulus-plateau detector-history prefix (a SECOND resumable registered alongside the
# chroma gate when it is event-mode; persists the trailing detector window so an UNFIRED-mid-dwell
# plateau re-fires at the IDENTICAL epoch after a crash — see :func:`build_gate_resume_registry`).
CHROMA_HISTORY_PREFIX = "__cbh_"

# Canonical set of the trainer-side DIRECT-registered controllers — the FunctionResumable (and
# scalar) controllers wired via ``_resume_registry.register(name, prefix, ...)`` OUTSIDE
# :func:`build_gate_resume_registry` (which is only for the EventBackstopGate class in
# GATE_KEY_PREFIXES). #358 wired these as resumables (so their state persists + the runtime
# ``__resume_registry_manifest`` self-protection covers them), but until now they had NO single
# STATIC "all present" coverage assertion — the sibling of the gate coverage test. birth_completion
# (the v7.5 ramp's latched fire-epoch state) is the newest member. A NEW direct controller cannot
# ship without being added here (the static coverage test ``test_every_trainer_direct_registered_
# controller_is_canonical`` fails), so it is consciously acknowledged as resume-relevant — the same
# no-silent-orphan discipline GATE_KEY_PREFIXES enforces for gates.
DIRECT_CONTROLLER_NAMES: frozenset[str] = frozenset({
    "polyak_finisher",
    "rng_streams",
    "closed_loop",
    "tau_advance",
    "evt_curriculum",
    "birth_completion",
})


class ResumeIntegrityError(RuntimeError):
    """Raised at resume when an EVENT-mode controller's persisted state has VANISHED from the sidecar
    (the manifest says it was written, but no key with its prefix is present) — a corrupted sidecar
    where silently re-arming the gate would break resume-determinism. NOT raised for the benign cases
    (legacy sidecar with no manifest, an event gate added since the checkpoint, a persisted-unfired
    sentinel, or an event-mode-OFF gate that legitimately wrote nothing)."""


@runtime_checkable
class Resumable(Protocol):
    """A stateful controller that round-trips through the resume sidecar under a unique key ``prefix``.

    ``state_arrays(prefix)`` returns ``{prefix+<name>: np.ndarray}`` (``{}`` when there is nothing to
    persist -> byte-identical). ``restore_from_cfg(prefix, cfg)`` restores from the parsed sidecar cfg
    and returns True iff state was actually restored (False for missing keys / a not-to-be-restored
    sentinel -> the controller stays fresh, deterministic going forward)."""

    def state_arrays(self, prefix: str) -> dict[str, Any]: ...
    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool: ...


@dataclass
class RestoreReport:
    """Outcome of :meth:`ResumeRegistry.restore`. ``restored`` maps each registered controller name to
    whether it restored real state; ``warnings`` are JSON-ready rows to print (LOUD but non-fatal);
    ``manifest_present`` records whether the sidecar carried a registry manifest (a legacy sidecar does
    not); ``legacy`` is True for the no-manifest fall-back path."""

    restored: dict[str, bool]
    warnings: list[dict]
    manifest_present: bool
    legacy: bool


@dataclass
class _Entry:
    name: str
    prefix: str
    controller: Resumable

    @property
    def event_active(self) -> bool:
        # EventBackstopGate exposes ``event_mode``; non-gate resumables (history) default to False so
        # they never trip the event fail-closed. Used for the manifest ``event`` flag.
        return bool(getattr(self.controller, "event_mode", False))


@dataclass
class ResumeRegistry:
    """Run-scoped registry of :class:`Resumable` controllers. Built ONCE per run (right after the
    controllers are constructed); every stateful controller registers here so both checkpoint-write and
    resume-restore iterate the SAME set — a registered controller can never be silently forgotten."""

    _entries: list[_Entry] = field(default_factory=list)
    _names: set[str] = field(default_factory=set)
    _prefixes: set[str] = field(default_factory=set)

    def register(self, name: str, prefix: str, controller: Resumable) -> Resumable:
        """Register ``controller`` under a unique ``name`` + key ``prefix``. Raises on a duplicate name
        OR prefix (a collision would silently overwrite another controller's persisted state), or when
        ``controller`` does not implement the :class:`Resumable` protocol (fail-closed at construction —
        a mis-shaped controller can never ship un-persisted)."""
        if not (hasattr(controller, "state_arrays") and hasattr(controller, "restore_from_cfg")):
            raise TypeError(
                f"resume registry: controller {name!r} does not implement the Resumable protocol "
                "(needs state_arrays(prefix) + restore_from_cfg(prefix, cfg))")
        if name in self._names:
            raise ValueError(f"resume registry: duplicate controller name {name!r}")
        if prefix in self._prefixes:
            raise ValueError(f"resume registry: duplicate key prefix {prefix!r} (name {name!r})")
        if not prefix.startswith("__"):
            raise ValueError(f"resume registry: prefix {prefix!r} must start with '__' so "
                             "_load_resume_state routes it to cfg, not a param tensor")
        self._entries.append(_Entry(name, prefix, controller))
        self._names.add(name)
        self._prefixes.add(prefix)
        return controller

    @property
    def names(self) -> list[str]:
        return [e.name for e in self._entries]

    def state_arrays(self) -> dict[str, Any]:
        """Merge every registered controller's ``state_arrays(prefix)``. When the merge is EMPTY
        (every controller had nothing to persist — the cap-only path) return ``{}`` with NO manifest,
        so the sidecar is byte-identical. Otherwise stamp :data:`RESUME_REGISTRY_MANIFEST_KEY`
        listing ONLY the controllers that actually wrote keys (with their event flag), so resume can
        detect a persisted-but-vanished controller.

        (B5 REVISE-2 — the pre-first-event-fire window, EVALUATED + documented) There is NO
        manifest-free window in EVENT mode. A concern was raised that in event mode a crash BEFORE the
        first event fires (~ep0..muon-fire) could persist the non-event controllers (rng/closed-loop/
        tau/evt-curriculum) manifest-free and silently legacy-restore them. That window does NOT exist:
        an event-mode-ON but UNFIRED :class:`EventBackstopGate` writes its ``fired_epoch=-1`` SENTINEL
        keys (non-empty) => it is counted in ``wrote`` with ``event: True`` => the manifest is stamped
        from checkpoint 1, so every co-writing non-event controller is vanish-protected from the first
        save. MEASURED: :func:`test_event_mode_unfired_gate_stamps_manifest_no_window` (an all-unfired
        event registry stamps the manifest; the sister cap-only registry stays manifest-free).

        The ONLY manifest-free case is the CLOCK / cap-only path (no event gate at all): there the
        non-event controllers persist manifest-free by DESIGN — the byte-identity contract for the live
        cap-only run (§190 below) — and non-event vanish is fail-open BY DESIGN (a truncated-sidecar
        resume re-inits them cleanly; pre-registry these had no protection at all). ``stamp-always`` was
        REJECTED: in event mode it is redundant (sentinel already stamps), and in cap-only mode it would
        add a manifest key => break the byte-identity contract the live run depends on (and would perturb
        a clock-mode run's resume — forbidden)."""
        import numpy as np
        out: dict[str, Any] = {}
        wrote: list[dict] = []
        for e in self._entries:
            arrays = e.controller.state_arrays(e.prefix)
            if not arrays:
                continue
            # guard: a controller must only emit keys under its own prefix (else the manifest key-scan
            # would miss them + a sister controller could collide).
            for k in arrays:
                if not str(k).startswith(e.prefix):
                    raise ValueError(
                        f"resume registry: controller {e.name!r} emitted key {k!r} outside its "
                        f"prefix {e.prefix!r}")
            out.update(arrays)
            wrote.append({"name": e.name, "prefix": e.prefix, "event": e.event_active})
        if not out:
            return {}
        # Stamp the manifest ONLY when a VANISH-PROTECTED (event-active) controller actually wrote — so
        # folding an ALWAYS-ON non-event controller (the RNG streams) or an opt-in non-event controller
        # (closed-loop / tau-advance / evt-curriculum) does NOT add a manifest to a config that had none.
        # This is the byte-identity contract: the live #205/v6/v7 cap-only run has no event gate, so it
        # stays manifest-free even though it writes __rng_* every checkpoint. When a manifest IS stamped
        # (an event gate wrote), it lists ALL controllers that wrote — so a non-event controller's
        # vanished state is still DETECTED (a LOUD warning row, never the event fail-closed).
        if any(w["event"] for w in wrote):
            out[RESUME_REGISTRY_MANIFEST_KEY] = np.asarray(json.dumps(wrote))
        return out

    def restore(self, cfg: dict) -> RestoreReport:
        """Restore every registered controller from the parsed sidecar ``cfg`` (symmetric to
        :meth:`state_arrays`). Completeness self-protection:

          * a manifest entry whose persisted keys have VANISHED (no ``prefix``-key present) is an
            integrity violation; for an EVENT gate it FAILS CLOSED (``ResumeIntegrityError``) rather
            than silently re-arm; for a non-event controller it is a LOUD warning row.
          * an event-mode controller registered NOW but absent from the manifest (added since the
            checkpoint, or event-mode-OFF at write) is a warning — it re-arms deterministically.
          * a legacy sidecar (no manifest) falls back to each controller's own back-compat restore
            (byte-identical) with a single info row if any registered controller is event-mode.
        """
        restored: dict[str, bool] = {}
        for e in self._entries:
            restored[e.name] = bool(e.controller.restore_from_cfg(e.prefix, cfg))

        warnings: list[dict] = []
        manifest_raw = cfg.get(RESUME_REGISTRY_MANIFEST_KEY)
        manifest_present = manifest_raw is not None

        def _prefix_present(prefix: str) -> bool:
            return any(str(k).startswith(prefix) for k in cfg)

        if not manifest_present:
            if any(e.event_active for e in self._entries):
                warnings.append({
                    "stage": "resume_registry_legacy_no_manifest",
                    "note": "sidecar predates the resume registry (no manifest); event-mode "
                            "controllers restore via their own back-compat path and re-arm from "
                            "post-resume history where unfired. Byte-identical legacy behavior.",
                    "event_controllers": [e.name for e in self._entries if e.event_active],
                })
            return RestoreReport(restored, warnings, manifest_present=False, legacy=True)

        try:
            manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else list(manifest_raw)
        except (ValueError, TypeError):
            warnings.append({"stage": "resume_registry_manifest_unparseable",
                             "manifest_raw": str(manifest_raw)[:200]})
            return RestoreReport(restored, warnings, manifest_present=True, legacy=False)

        manifest_names = {str(m.get("name")) for m in manifest}
        for m in manifest:
            mname, mprefix = str(m.get("name")), str(m.get("prefix"))
            mevent = bool(m.get("event", False))
            if _prefix_present(mprefix):
                continue
            row = {
                "stage": "resume_registry_persisted_state_vanished",
                "controller": mname, "prefix": mprefix, "event": mevent,
                "note": ("the manifest recorded this controller as PERSISTED, but no key with its "
                         "prefix is present in the sidecar — the sidecar is corrupt/truncated."),
            }
            if mevent:
                row["fatal"] = True
                raise ResumeIntegrityError(json.dumps(row))
            warnings.append(row)

        for e in self._entries:
            if e.event_active and e.name not in manifest_names:
                warnings.append({
                    "stage": "resume_registry_event_controller_added_since_checkpoint",
                    "controller": e.name, "prefix": e.prefix,
                    "note": "an event-mode controller not present in the checkpoint manifest; it was "
                            "event-mode-OFF at write or added since. Re-arms from post-resume history.",
                })
        return RestoreReport(restored, warnings, manifest_present=True, legacy=False)


@dataclass
class TrailingSeriesResumable:
    """A :class:`Resumable` for a controller's bounded DETECTOR-HISTORY series held in a plain dict
    (e.g. the trainer's ``_wire_sense['annulus_series']`` — a list of ``(epoch, value)`` points feeding
    the chroma annulus-plateau detector). Persists the TRAILING ``cap`` points so an UNFIRED-mid-dwell
    detector re-fires at the identical epoch after a crash (the annulus-plateau detector only ever
    inspects ``series[-dwell_windows:]``, so a bounded trailing window is exactly sufficient — the same
    bounded-window discipline as the spike-guard ``__recent_losses``). Persist (not re-derive from
    telemetry) so restore is SELF-CONTAINED and independent of the JSONL schema / verdict cadence."""

    container: dict
    key: str
    cap: int = 16

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        import numpy as np
        series = list(self.container.get(self.key) or [])
        if not series:
            return {}
        tail = series[-int(self.cap):]
        eps = [int(e) for e, _ in tail]
        vals = [float(v) for _, v in tail]
        return {
            prefix + "ep": np.asarray(eps, np.int64),
            prefix + "val": np.asarray(vals, np.float64),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        if (prefix + "ep") not in cfg:
            return False

        def _as_list(x) -> list:
            return x if isinstance(x, list) else [] if x is None else [x]

        eps = _as_list(cfg.get(prefix + "ep", []))
        vals = _as_list(cfg.get(prefix + "val", []))
        self.container[self.key] = [(int(e), float(v)) for e, v in zip(eps, vals)]
        return True


@dataclass
class FunctionResumable:
    """A generic :class:`Resumable` adapter wrapping a controller whose ``(state_arrays,
    restore_from_cfg)`` logic already lives as free functions / closures in the trainer — the four
    NON-GATE controllers folded under the registry 2026-07-08: closed-loop lever control, the τ-advance
    octave ladder, the RNG streams, and the event-triggered curriculum.

    THIN by construction: it delegates verbatim to the injected ``write(prefix) -> dict`` and
    ``restore(prefix, cfg) -> bool`` callables, which OWN the canonical hard-coded key names
    (``__cl_`` / ``__ta_`` / ``__rng_`` / ``__evt_``). The registry-emitted keys+arrays are therefore
    BYTE-IDENTICAL to a direct legacy call — the binding contract, asserted by the round-trip tests.
    ``write`` returns ``{}`` when its controller is OFF/None (byte-identical: no keys emitted).

    NON-EVENT: it exposes no ``event_mode`` attribute, so :attr:`_Entry.event_active` is False. It
    never trips the manifest event fail-closed, and (per :meth:`ResumeRegistry.state_arrays`) it never
    by itself stamps a manifest — which is exactly what keeps folding an ALWAYS-ON controller (rng)
    byte-identical for a manifest-free legacy sidecar. A vanished non-event state is a LOUD warning
    (only when a manifest already exists because an event gate wrote), never a raise."""

    write: "Callable[[str], dict[str, Any]]"
    restore: "Callable[[str, dict], bool]"

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        return self.write(prefix)

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        return bool(self.restore(prefix, cfg))


def build_gate_resume_registry(gates, *, wire_sense: "dict | None" = None) -> ResumeRegistry:
    """Build the run-scoped :class:`ResumeRegistry` for the trainer's :class:`EventBackstopGate` set.

    Every gate is registered under its canonical :data:`GATE_KEY_PREFIXES` prefix (a gate whose
    ``name`` is not in that map raises — the fail-closed that forces the NEXT gate to be added there +
    covered by the static coverage test). Each gate self-gates to ``{}`` when event-mode OFF, so a
    cap-only run persists NOTHING (byte-identical). When the seg_chroma_boundary gate is event-mode and
    ``wire_sense`` is provided, its annulus-plateau detector history is registered as a second resumable
    (:class:`TrailingSeriesResumable`) so an unfired-mid-dwell plateau re-fires bit-faithfully."""
    reg = ResumeRegistry()
    chroma_gate = None
    for g in gates:
        name = getattr(g, "name", None)
        if name not in GATE_KEY_PREFIXES:
            raise KeyError(
                f"resume registry: gate name {name!r} has no canonical key prefix — add it to "
                "GATE_KEY_PREFIXES (and the static coverage test) before shipping the gate")
        reg.register(name, GATE_KEY_PREFIXES[name], g)
        if name == "seg_chroma_boundary":
            chroma_gate = g
    if (chroma_gate is not None and getattr(chroma_gate, "event_mode", False)
            and wire_sense is not None):
        reg.register("seg_chroma_boundary_history", CHROMA_HISTORY_PREFIX,
                     TrailingSeriesResumable(wire_sense, "annulus_series", cap=16))
    return reg
