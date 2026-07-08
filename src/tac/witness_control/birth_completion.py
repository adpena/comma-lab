"""BIRTH COMPLETION — the Morse-Smale persistence hand-off for the level-set witness rare-class birth.

v7.5 birth-counter-force Lever-2 (operator directive 2026-07-08, verbatim): *"Lever 2 (completion
event) = a MORSE-SMALE PERSISTENCE event, not a bare part_frac threshold: birth-complete when the
class's island persistence (prominence above the argmax margin) exceeds tau_persist AND part_frac in
[(1-delta),(1+delta)]*GT. On fire: ramp the birth stack (logit-adjust offset, persistence-recall,
island-amplify boost) to its post-birth level and hand off to the boundary/annulus regime."*

## Why this is the missing "stop growing after birth" law

The rare-class-birth stack (seed/amplify/persistence-recall/logit-adjust) applies RECALL/GROWTH
pressure to nucleate the born-empty classes {Lane,Movable}, but has no COMPLETION event: once a class
has nucleated + formed, the growth machinery keeps pushing (measured over-paint 13.8x/4.6x). The
Chan-Vese area constraint (Lever-1, ``chan_vese_area_constraint_birth_balance_20260708``) caps the
area CONTINUOUSLY; this event is the DISCRETE regime hand-off that RE-ALLOCATES the freed capacity
from birth (recall) to boundary (precision) once the class is genuinely done. With Lever-1 active this
is DEFENSE-IN-DEPTH, not the sole safety mechanism (#302 discipline: derive the annealing from the
level-set energy — the multiplier self-limits, the event re-allocates).

## The Morse-Smale persistence completion predicate

A class ``c`` is BIRTH-COMPLETE at a verdict epoch iff BOTH:

  1. **persistence >= tau_persist** — its island PROMINENCE above the argmax margin. The realized
     per-class prominence proxy is ``persistence_c = 1 - within_flip_c`` (the fraction of the class's
     GT pixels the witness holds ABOVE the top1-top2 argmax margin — a Morse-Smale prominence reading
     of the class's basin: a well-formed class wins most of its support, a nascent one does not). This
     is the SAME ``within_flip`` the ``_evt_nucleus_stats`` handoff-readiness telemetry already
     computes (consume it, do not rebuild it).
  2. **part_frac in [(1-area_tol), (1+area_tol)] * gt_area** — the class area has settled INTO the
     Chan-Vese equilibrium band (not still under-born, not still over-painted).

Both gates are necessary: prominence alone can be high on a tiny over-confident nucleus; area alone
can be right while the partition is still churning. Together they read "the class has BOTH nucleated
to its GT extent AND formed a persistent basin" — the Morse-Smale birth-complete signature.

## The hand-off ramp (on fire)

Once fired for class ``c`` (LATCHED — birth completion is monotone; a class does not un-complete), the
birth-stack multiplier ramps 1.0 -> ``post_level`` over ``ramp_epochs`` (a smooth event-gated ramp,
NOT a hard cut — mirrors the other v7 event gates). The trainer multiplies class ``c``'s birth
contributions (island-amplify weight, persistence-recall, logit-adjust offset) by this multiplier,
handing the freed capacity to the boundary/annulus regime.

## Resume determinism (launch-critical)

The controller OWNS its per-class latched fire-epoch, so a crash-resume reconstructs the IDENTICAL
subsequent multiplier trajectory. State persists via :func:`birth_completion_state_arrays` /
:func:`birth_completion_restore_from_cfg` (``__bc_*`` keys), mirroring the tau-advance / muon-gate
sidecar pattern. Default-OFF (no controller instantiated) => byte-identical.

PURE + unit-testable (NO MLX / model / async / wall-clock): the trainer feeds the per-class
(part_frac, gt_area, within_flip) it already computes and reads back the per-class multiplier.

means != ends: advisory control logic. Only a byte-closed n600 exact row < 0.19110 from
``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_TAU_PERSIST",
    "DEFAULT_AREA_BAND",
    "DEFAULT_RAMP_EPOCHS",
    "DEFAULT_POST_LEVEL",
    "birth_persistence",
    "birth_complete",
    "birth_ramp_multiplier",
    "BirthCompletionController",
    "birth_completion_state_arrays",
    "birth_completion_restore_from_cfg",
    "birth_completion_apply_restore",
    "birth_ramp_multiplier_vector",
    "derive_post_level_from_persistence",
]

#: persistence (=1-within_flip) a class must exceed to be birth-complete. DERIVED from the nucleus
#: guard's within_flip threshold family (a formed class holds >= this fraction of its GT support).
DEFAULT_TAU_PERSIST = 0.8
#: area band around GT the class must settle into (matches the Chan-Vese equilibrium tolerance 0.25).
DEFAULT_AREA_BAND = 0.25
#: epochs over which the birth stack ramps down after completion (smooth hand-off, not a hard cut).
DEFAULT_RAMP_EPOCHS = 50
#: post-birth birth-stack level the ramp settles to (0.0 = full hand-off to boundary regime).
DEFAULT_POST_LEVEL = 0.0


def birth_persistence(within_flip: float) -> float:
    """The realized per-class Morse-Smale prominence proxy ``1 - within_flip`` (the fraction of the
    class's GT support held above the argmax margin). Clamped to [0, 1]. Pure."""
    return max(0.0, min(1.0, 1.0 - float(within_flip)))


def birth_complete(
    part_frac: float, gt_area: float, within_flip: float, *,
    tau_persist: float = DEFAULT_TAU_PERSIST, area_band: float = DEFAULT_AREA_BAND,
) -> bool:
    """The Morse-Smale persistence birth-completion predicate for one class (see the module docstring).

    True iff ``birth_persistence(within_flip) >= tau_persist`` AND ``part_frac`` lies in the band
    ``[(1-area_band), (1+area_band)] * gt_area``. A class with ``gt_area <= 0`` (unscored) is NEVER
    complete (vacuously false — nothing to birth). Pure / fail-closed on malformed inputs."""
    g = float(gt_area)
    if g <= 0.0:
        return False
    if not (0.0 <= float(tau_persist) <= 1.0):
        raise ValueError(f"tau_persist must be in [0,1], got {tau_persist!r}")
    if not (float(area_band) >= 0.0):
        raise ValueError(f"area_band must be >= 0, got {area_band!r}")
    persist_ok = birth_persistence(within_flip) >= float(tau_persist)
    lo = (1.0 - float(area_band)) * g
    hi = (1.0 + float(area_band)) * g
    area_ok = lo <= float(part_frac) <= hi
    return bool(persist_ok and area_ok)


def birth_ramp_multiplier(
    fired_epoch: "int | None", epoch: int, *,
    ramp_epochs: int = DEFAULT_RAMP_EPOCHS, post_level: float = DEFAULT_POST_LEVEL,
) -> float:
    """The birth-stack multiplier for a class at ``epoch`` given its latched ``fired_epoch``.

    ``fired_epoch is None`` (not yet complete) => 1.0 (full birth pressure). Once fired, linearly ramp
    ``1.0 -> post_level`` over ``[fired_epoch, fired_epoch + ramp_epochs]`` then hold at ``post_level``.
    Monotone non-increasing after fire (the hand-off is one-way). Pure."""
    if fired_epoch is None:
        return 1.0
    fe = int(fired_epoch)
    re = max(1, int(ramp_epochs))
    pl = float(post_level)
    if int(epoch) <= fe:
        return 1.0
    frac = min(1.0, (int(epoch) - fe) / re)
    return float(1.0 + (pl - 1.0) * frac)


@dataclass
class BirthCompletionController:
    """Per-class latched birth-completion state + hand-off ramp (v7.5 Lever-2).

    ``classes``: the birthed classes to watch (e.g. (1, 3) = Lane, Movable). ``tau_persist`` /
    ``area_band`` parametrize :func:`birth_complete`; ``ramp_epochs`` / ``post_level`` the hand-off
    ramp. The controller is fed the per-class (part_frac, gt_area, within_flip) at each verdict epoch
    via :meth:`observe` (decide-on-previous, single-threaded in the main loop) and exposes the current
    per-class multiplier via :meth:`multiplier`. Fire epochs are LATCHED (monotone completion)."""

    classes: tuple[int, ...]
    tau_persist: float = DEFAULT_TAU_PERSIST
    area_band: float = DEFAULT_AREA_BAND
    ramp_epochs: int = DEFAULT_RAMP_EPOCHS
    post_level: float = DEFAULT_POST_LEVEL
    fired: dict[int, int] = field(default_factory=dict)   # class -> latched fire epoch
    _fires: list[tuple[int, int]] = field(default_factory=list)  # (class, epoch) fire log

    def observe(self, epoch: int, stats: "dict[int, dict]") -> list[dict]:
        """Ingest a verdict-epoch per-class ``stats`` dict ({class: {part_frac, within_flip, gt_px, ...}}
        or {..., "gt_area"}). Latches any newly birth-complete class. Returns the LOUD fire rows emitted
        this epoch (empty if none). Idempotent per class (a re-observe never un-latches)."""
        rows: list[dict] = []
        for c in self.classes:
            ci = int(c)
            if ci in self.fired:
                continue
            s = stats.get(ci) or stats.get(str(ci))
            if not s:
                continue
            gt_area = s.get("gt_area")
            if gt_area is None:
                gt_px = float(s.get("gt_px", 0.0))
                # derive gt_area from gt_px/total_px if the caller passed counts; else 0 (unscored).
                total = float(s.get("total_px", 0.0))
                gt_area = (gt_px / total) if total > 0.0 else 0.0
            if birth_complete(float(s.get("part_frac", 0.0)), float(gt_area),
                              float(s.get("within_flip", 1.0)),
                              tau_persist=self.tau_persist, area_band=self.area_band):
                self.fired[ci] = int(epoch)
                self._fires.append((ci, int(epoch)))
                rows.append({"stage": "birth_completion", "event": "fired", "class": ci,
                             "epoch": int(epoch), "part_frac": round(float(s.get("part_frac", 0.0)), 6),
                             "gt_area": round(float(gt_area), 6),
                             "persistence": round(birth_persistence(float(s.get("within_flip", 1.0))), 4),
                             "note": "Morse-Smale birth-complete: hand off birth->boundary regime"})
        return rows

    def multiplier(self, cls: int, epoch: int) -> float:
        """Current birth-stack multiplier for ``cls`` at ``epoch`` (1.0 if not yet complete)."""
        return birth_ramp_multiplier(
            self.fired.get(int(cls)), int(epoch),
            ramp_epochs=self.ramp_epochs, post_level=self.post_level)

    def all_multipliers(self, epoch: int) -> dict[int, float]:
        """Per-class current multipliers (for the watched classes)."""
        return {int(c): self.multiplier(int(c), int(epoch)) for c in self.classes}


def birth_completion_state_arrays(ctrl: "BirthCompletionController | None"):
    """Serialize the controller to resume-sidecar arrays (``__bc_*`` keys). ``{}`` when ``ctrl`` is
    None (default-OFF => nothing persisted => byte-identical). Mirrors ``tau_advance_state_arrays``."""
    import numpy as np

    if ctrl is None:
        return {}
    fired_items = sorted(ctrl.fired.items())
    return {
        "__bc_classes": np.asarray([int(c) for c in ctrl.classes], np.int64),
        "__bc_tau_persist": np.asarray(float(ctrl.tau_persist)),
        "__bc_area_band": np.asarray(float(ctrl.area_band)),
        "__bc_ramp_epochs": np.asarray(int(ctrl.ramp_epochs)),
        "__bc_post_level": np.asarray(float(ctrl.post_level)),
        "__bc_fired_class": np.asarray([int(c) for c, _ in fired_items], np.int64),
        "__bc_fired_epoch": np.asarray([int(e) for _, e in fired_items], np.int64),
    }


def birth_completion_restore_from_cfg(cfg) -> "BirthCompletionController | None":
    """Reconstruct a controller from a resume sidecar/cfg dict written by
    :func:`birth_completion_state_arrays`. Returns None when the ``__bc_*`` keys are absent (the run
    was not using the lever). Bit-faithful (the latched fire epochs are restored exactly)."""
    import numpy as np

    if cfg is None or "__bc_classes" not in cfg:
        return None
    classes = tuple(int(x) for x in np.asarray(cfg["__bc_classes"]).reshape(-1).tolist())
    ctrl = BirthCompletionController(
        classes=classes,
        tau_persist=float(np.asarray(cfg.get("__bc_tau_persist", DEFAULT_TAU_PERSIST))),
        area_band=float(np.asarray(cfg.get("__bc_area_band", DEFAULT_AREA_BAND))),
        ramp_epochs=int(np.asarray(cfg.get("__bc_ramp_epochs", DEFAULT_RAMP_EPOCHS))),
        post_level=float(np.asarray(cfg.get("__bc_post_level", DEFAULT_POST_LEVEL))),
    )
    fc = np.asarray(cfg.get("__bc_fired_class", np.asarray([], np.int64))).reshape(-1).tolist()
    fe = np.asarray(cfg.get("__bc_fired_epoch", np.asarray([], np.int64))).reshape(-1).tolist()
    for c, e in zip(fc, fe):
        ctrl.fired[int(c)] = int(e)
        ctrl._fires.append((int(c), int(e)))
    return ctrl


def birth_completion_apply_restore(ctrl: "BirthCompletionController | None", cfg) -> bool:
    """Restore the LATCHED per-class fire epochs from a resume sidecar ``cfg`` INTO an EXISTING
    controller.

    Unlike :func:`birth_completion_restore_from_cfg` (which builds a FRESH controller from the
    persisted params), this keeps ``ctrl``'s argv-derived ``classes`` / ``tau_persist`` / ramp
    params (the resume command's flags are authoritative; the F2 divergence guard fails closed if
    they diverged) and restores ONLY the monotone fired-epoch state, so a crash-resume reconstructs
    the IDENTICAL subsequent multiplier trajectory. Returns True iff any fire epoch was restored.

    A legacy/un-fired sidecar (``__bc_fired_class`` absent OR empty) restores NOTHING => the
    controller stays un-fired => byte-identical PRE-FIRE behavior (the additive ``__bc_*`` contract).
    Only classes the live controller WATCHES are restored (a stale key for a no-longer-watched class
    is ignored, fail-closed)."""
    import numpy as np

    if ctrl is None or cfg is None or "__bc_fired_class" not in cfg:
        return False
    fc = np.asarray(cfg.get("__bc_fired_class")).reshape(-1).tolist()
    fe = np.asarray(cfg.get("__bc_fired_epoch", np.asarray([], np.int64))).reshape(-1).tolist()
    watched = set(int(c) for c in ctrl.classes)
    restored = False
    for c, e in zip(fc, fe):
        ci = int(c)
        if ci in watched and ci not in ctrl.fired:
            ctrl.fired[ci] = int(e)
            ctrl._fires.append((ci, int(e)))
            restored = True
    return restored


def derive_post_level_from_persistence(tau_persist: float = DEFAULT_TAU_PERSIST) -> float:
    """The DERIVED post-birth birth-stack level (value-provenance: DERIVED-AT-CONFIG).

    Balance arithmetic (Lever-1 Chan-Vese equilibrium): with the birth force ramped to
    ``post_level * F_birth`` the equilibrium area is ``A* = A_GT * (1 + post_level * delta)``
    (``delta`` = the area tolerance). At birth-completion, ``persistence >= tau_persist`` means the
    class holds a fraction ``tau_persist`` of its GT support ABOVE the argmax margin (formed); the
    remaining fraction ``1 - tau_persist`` is still below margin and is the ONLY part that still
    needs birth pressure. Retaining exactly that unformed fraction of the force =>
    ``post_level = 1 - tau_persist`` (a class 80%-formed at fire keeps 20% birth force for its
    unformed tail; a fully-formed class would keep ~0). With the ``tau_persist = 0.8`` default the
    residual equilibrium overshoot is ``0.2 * delta`` (e.g. 0.05 at delta=0.25 => A* = 1.05*A_GT),
    a tight PRECISION band inside the completion band. Clamped to [0, 1]. Pure.

    Epistemic (NO-FAKE): the FORM (residual force ∝ unformed fraction) is DERIVED; the absolute
    best post_level is ASSUMED_AWAITING_VERIFICATION owed to the v7.5 A/B (sister of the Lever-1
    lambda scale + the ramp-length fraction)."""
    return max(0.0, min(1.0, 1.0 - float(tau_persist)))


def birth_ramp_multiplier_vector(
    ctrl: "BirthCompletionController | None", epoch: int, n_classes: int = 5,
) -> list[float]:
    """The per-class ramp-multiplier vector (length ``n_classes``) at ``epoch``.

    ``mult[c] = ctrl.multiplier(c, epoch)`` for a watched class ``c`` (ramping 1.0 -> post_level
    after its latched fire), else 1.0 (unwatched classes are never ramped). Used to scale the
    per-class logit-adjust offset vector EXACTLY (element-wise, byte-identical when every entry is
    1.0). ``ctrl is None`` => all-ones (identity). Pure."""
    out = [1.0] * int(n_classes)
    if ctrl is None:
        return out
    for c in ctrl.classes:
        ci = int(c)
        if 0 <= ci < int(n_classes):
            out[ci] = float(ctrl.multiplier(ci, int(epoch)))
    return out
