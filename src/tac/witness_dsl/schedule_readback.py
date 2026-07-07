"""Canonical curriculum-schedule READ-BACK — the inverse direction of the DSL compile path.

Operator directive 2026-07-07 (verbatim class-fix): the dashboard "should not have
hardcoded parameters like that but should use conditional rendering integrated with
DSL." The measured incident: ``tools/dashboard_server.py`` took stage boundaries as
hand-fed CLI constants (``--tau 300 --l7 600``) while the live run had l7 DISABLED
(``--l7-start-epoch 1001``) — epochs >= 600 were mislabeled "l7" during tau. The
CLASS bug: observability consumers must DERIVE the stage map from the run's OWN
config via the DSL (single source of truth), never from hand-fed constants.

This module is the read-back consumers use: given a RUN DIRECTORY it extracts the
stage map the run ACTUALLY has, in TWO LAYERS (operator amendment 2026-07-07):

1. PLANNED — static boundaries parsed from the run's ``launch.sh`` argv through the
   trainer's REAL argparse (:func:`curriculum_dsl.build_real_trainer_parser`, the
   never-invent-flags pattern; defaults/types are the trainer's own). Valid for
   fixed-epoch runs (e.g. the live mod32cap). Disabled stages (start >= epochs, or
   shadowed by stage precedence) are OMITTED.
2. ACTUAL — transitions derived from the run's EMITTED evidence: the trainer's
   ``{"stage": "curriculum_transition_fired", ...}`` / ``{"stage":
   "muon_finisher_switch", ...}`` log rows and the per-stage checkpoint filenames
   (``levelset_ckpt_stage<TAG>_ep<N>.npz``). ACTUAL overrides PLANNED wherever both
   exist.

Event-gated stages (the #315 nucleus-guarded CE->tau hand-off / #334
``Curriculum(handoff="event")``) yield a UNION entry: fixed stages are
``{name, start}``; event-gated stages are ``{name, trigger, status: pending|fired,
fired_epoch, cap}`` where ``trigger`` is derived from the DSL ``Curriculum`` object
(its own emitted hand-off flags — never re-described by hand). The trainer's cap is
a HARD CEILING (``_evt_resolve_seg_form`` fires at the cap unconditionally), so an
event stage with no fired evidence is still provably active at ``epoch >= cap``.

Dependency-light BY CONTRACT: stdlib + :mod:`tac.witness_dsl.curriculum_dsl` only
(no numpy/mlx/torch) — the live dashboard daemon imports this every refresh tick.
Every entry point is fail-open (returns ``ok=False`` with a reason; never raises)
because the consumer is a load-bearing multi-day daemon.

Authority: this is OBSERVABILITY (score-neutral, read-only). The frontier pointer
(contest-CPU 0.19110) is UNMOVED by anything here.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .curriculum_dsl import Anneal, Curriculum, Stage, build_real_trainer_parser

__all__ = [
    "STAGE_ORDER",
    "ScheduleReadback",
    "StageEntry",
    "describe_primitive",
    "display_entry",
    "event_trigger_description",
    "read_schedule",
    "resolve_run_dir_for_log",
    "stage_map_from_curriculum",
    "trainer_argv_from_launch_sh",
]

#: Canonical stage precedence (later wins at an epoch), mirroring the trainer's
#: ``_seg_form_for_epoch`` / ``dashboard_trajectory_model.stage_at_epoch`` order.
STAGE_ORDER: tuple[str, ...] = ("CE", "tau", "l7", "Muon")
_ORDER_IDX = {n: i for i, n in enumerate(STAGE_ORDER)}

#: trainer ``_STAGE_TAGS`` filename tags -> canonical stage names (checkpoint evidence).
_CKPT_TAG_TO_STAGE = {"stageCE": "CE", "stageTau": "tau", "stageL7": "l7",
                      "stageMuonStart": "Muon"}
#: trainer event-controller seg_form tokens -> canonical stage names (log evidence).
_FORM_TO_STAGE = {"ce": "CE", "tau_softplus": "tau", "l7_softplus": "l7"}

_CKPT_RE = re.compile(
    r"^levelset_(?:ckpt|resume)_(?P<tag>stage[A-Za-z0-9]+?)(?:_muon)?_ep(?P<ep>\d+)\.npz$")


# ---------------------------------------------------------------------------
# typed structure (the union entry from the operator amendment)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageEntry:
    """One stage of the run's derived map.

    Fixed stages: ``{name, start}`` (``mode="fixed"``, ``status="scheduled"``).
    Event-gated stages: ``{name, trigger, status: pending|fired, fired_epoch, cap}``;
    ``start`` is the resolved band start (the fired epoch) or ``None`` while pending.
    """

    name: str
    start: int | None
    mode: str = "fixed"          # "fixed" | "event"
    status: str = "scheduled"    # fixed: "scheduled"; event: "pending" | "fired"
    fired_epoch: int | None = None
    cap: int | None = None       # event mode: the hard-ceiling boundary flag value
    trigger: str | None = None   # event mode: condition description from the DSL
    source: str = "launch.sh"    # provenance: launch.sh | log | checkpoint

    def to_dict(self) -> dict:
        d = {"name": self.name, "start": self.start, "mode": self.mode,
             "status": self.status, "source": self.source}
        if self.mode == "event":
            d.update({"fired_epoch": self.fired_epoch, "cap": self.cap,
                      "trigger": self.trigger})
        return d


@dataclass(frozen=True)
class ScheduleReadback:
    """The derived stage map for one run directory (see module docstring)."""

    ok: bool
    source: str                          # "launch.sh" | "none"
    epochs: int | None = None
    eval_every: int | None = None
    event_triggered: bool = False
    stages: tuple[StageEntry, ...] = ()
    reason: str = ""                     # why ok=False (fail-open diagnostics)
    run_dir: str = ""
    unknown_args: tuple[str, ...] = field(default=())  # argv the real parser skipped

    def stage_at(self, epoch: float | int | None) -> str:
        """Stage name active at ``epoch`` per the derived map (precedence: the
        highest-precedence stage whose resolved start — or, for a still-pending
        event stage, its hard-ceiling cap — is <= epoch). Epoch None/pre-map -> the
        first mapped stage (CE for curriculum runs)."""
        if epoch is None:
            return self.stages[0].name if self.stages else "CE"
        best: StageEntry | None = None
        for st in self.stages:
            bound = st.start
            if bound is None and st.mode == "event" and st.status == "pending":
                bound = st.cap  # trainer cap fires unconditionally -> provable at >= cap
            if bound is None or float(epoch) < float(bound):
                continue
            if best is None or _ORDER_IDX.get(st.name, 0) >= _ORDER_IDX.get(best.name, 0):
                best = st
        if best is not None:
            return best.name
        return self.stages[0].name if self.stages else "CE"

    def as_schedule_dict(self) -> dict:
        """The ``dashboard_trajectory_model``-compatible schedule dict (tau_start /
        l7_start / muon_start / epochs / eval_every); omitted stages map to None so
        the label step-function can never emit a stage this run does not have."""
        starts: dict[str, int | None] = {}
        for st in self.stages:
            starts[st.name] = st.start if st.start is not None else st.cap
        return {"tau_start": starts.get("tau"), "l7_start": starts.get("l7"),
                "muon_start": starts.get("Muon"), "epochs": self.epochs,
                "eval_every": self.eval_every}

    def to_dict(self) -> dict:
        return {"ok": self.ok, "source": self.source, "epochs": self.epochs,
                "eval_every": self.eval_every, "event_triggered": self.event_triggered,
                "stages": [s.to_dict() for s in self.stages], "reason": self.reason,
                "run_dir": self.run_dir}


# ---------------------------------------------------------------------------
# GENERIC-over-the-DSL rendering (operator amendment 2026-07-07: "As the DSL
# evolves, update the costate controller and dashboard accordingly"). Consumers
# enumerate WHATEVER the DSL declares: a uniform ``to_display_dict()`` /
# ``describe()`` surface is SOFT-DETECTED and preferred when a primitive
# declares it (a sibling is adding it uniformly); otherwise the flag-parsing
# fallback below stands. Unknown kinds render as their describe data — a
# declared schedule element is NEVER silently omitted, so a NEW DSL stage kind /
# level-path / operational-cadence primitive renders with ZERO dashboard change.
# ---------------------------------------------------------------------------
def describe_primitive(obj) -> dict | None:
    """The uniform display surface of a DSL primitive, soft-detected:
    ``to_display_dict()`` preferred, then ``describe()``. A dict result passes
    through; a non-dict result becomes ``{"text": str(result)}``. None when the
    object declares neither (callers then use a generic fallback). Never raises."""
    for attr in ("to_display_dict", "describe"):
        fn = getattr(obj, attr, None)
        if not callable(fn):
            continue
        try:
            d = fn()
        except Exception:  # noqa: BLE001 — a broken surface must not drop the element
            continue
        if isinstance(d, dict):
            return dict(d)
        if d is not None:
            return {"text": str(d)}
    return None


def display_entry(obj) -> dict:
    """A JSON-able display entry for ANY DSL schedule primitive: its uniform
    describe surface when declared, else a minimal ``{kind, text}`` fallback.
    Always returns a dict (never None) — a declared element cannot be dropped."""
    d = describe_primitive(obj)
    if d is None:
        d = {"text": str(getattr(obj, "name", None) or repr(obj))}
    out = {"kind": type(obj).__name__}
    out.update(d)
    return out


def stage_map_from_curriculum(curriculum, epochs: int | None = None) -> list[dict]:
    """GENERIC stage-map entries from a DSL Curriculum-like object.

    Enumerates whatever the object DECLARES — its dataclass fields (including
    future ones the DSL gains) and their tuple/list elements; duck-typed
    fallback = ``.stages``. Epoch-boundary ``Stage`` primitives (they carry
    ``start_epoch_flag``) become ``{name, start}`` entries — a boundary with
    ``start >= epochs`` is omitted, matching the flag-derived path's disabled
    semantics. EVERY other declared primitive — temp anneal, hosc schedule,
    transition, regularizers, and ANY unknown future kind — becomes a
    ``mode="declared"`` entry carrying its describe data (never dropped)."""
    import dataclasses

    entries: list[dict] = []

    def _one(obj) -> None:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return  # scalar config fields (handoff enum, tau float) are not elements
        e = display_entry(obj)
        name = getattr(obj, "name", None)
        if name is not None and hasattr(obj, "start_epoch_flag"):
            start = getattr(obj, "start_epoch", None)
            e.setdefault("name", str(name))
            e["start"] = 0 if start is None else int(start)
            e.setdefault("mode", "fixed")
            e.setdefault("status", "scheduled")
            if epochs is not None and e["start"] >= int(epochs):
                return  # disabled boundary ("never" form) — intentional omission
        else:
            e.setdefault("name", str(name) if name is not None else e["kind"])
            e.setdefault("start", None)
            e.setdefault("mode", "declared")
            e.setdefault("status", "declared")
        entries.append(e)

    if hasattr(type(curriculum), "__dataclass_fields__"):
        for f in dataclasses.fields(curriculum):
            v = getattr(curriculum, f.name, None)
            if isinstance(v, (tuple, list)):
                for item in v:
                    _one(item)
            else:
                _one(v)
    else:
        for item in getattr(curriculum, "stages", ()) or ():
            _one(item)
    return entries


# ---------------------------------------------------------------------------
# trigger description — read from the DSL Curriculum object, never hand-prose
# ---------------------------------------------------------------------------
def event_trigger_description(nucleus_guard: bool = True) -> str:
    """The event hand-off condition description, DERIVED from the DSL
    :class:`~tac.witness_dsl.curriculum_dsl.Curriculum` object: prefer its
    uniform describe surface when it declares one (soft-detected; the sibling
    uniform-display landing), else read the hand-off flags it EMITS (the #334
    single emitter) — either way the description tracks the DSL, never a
    hand-typed copy."""
    cur = Curriculum(stages=(Stage("CE", None),), temp=Anneal(1.0, 0.05), handoff="event")
    d = describe_primitive(cur)
    if d is not None:
        # only EXPLICIT description fields qualify (the uniform surface's generic
        # field dump — e.g. handoff='event' — is data, not a trigger description).
        for key in ("trigger", "trigger_description", "handoff_description"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
    evt_flags = sorted(k for k in cur.flags() if k.startswith("--curriculum-"))
    if not nucleus_guard:
        evt_flags = [f for f in evt_flags if "nucleus" not in f]
    return ("event-gated (" + ", ".join(evt_flags)
            + "): loss-plateau convergence OR hard-ceiling cap"
            + ("; nucleus-guarded CE->tau hand-off" if nucleus_guard else ""))


# ---------------------------------------------------------------------------
# run-dir resolution: a training log often lives OUTSIDE the run dir
# ---------------------------------------------------------------------------
_LAUNCH_PATH_RE = re.compile(r"(/[^\s'\"]+/launch\.sh)\b")


def resolve_run_dir_for_log(log_path) -> Path | None:
    """The RUN DIR (where launch.sh + artifacts live) for a training log.

    The durable-daemon launcher tees the trainer into ``.omx/tmp/levelset_*.log``
    while the run's artifacts live under ``--out-dir`` — so the log's parent is NOT
    the run dir (the exact gap that forced hand-fed dashboard constants). Resolve:
    the log's own parent when it holds launch.sh (in-run-dir logs), else the
    launch.sh path the launcher echoes in the log head (``launching: bash
    <run_dir>/launch.sh``). None when unresolvable. Never raises."""
    try:
        lp = Path(log_path)
        if (lp.parent / "launch.sh").is_file():
            return lp.parent
        with open(lp, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 80:  # launcher echo is a head line; bound the scan
                    break
                m = _LAUNCH_PATH_RE.search(line)
                if m:
                    cand = Path(m.group(1)).parent
                    if (cand / "launch.sh").is_file():
                        return cand
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# PLANNED layer: launch.sh argv -> the REAL trainer argparse
# ---------------------------------------------------------------------------
def trainer_argv_from_launch_sh(text: str) -> list[str] | None:
    """Extract the trainer argv from a launch.sh: join backslash line-continuations
    into logical lines, find the one invoking the trainer ``.py``, shlex it, and
    return the tokens AFTER the script path (env-assignment prefixes, ``cd`` /
    ``export`` / ``set`` lines are naturally excluded). None when no trainer
    invocation is found. Never raises on malformed shell (returns None)."""
    logical: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            logical.append(buf)
        buf = ""
    if buf.strip():
        logical.append(buf)
    for cmd in logical:
        if ".py" not in cmd or "train" not in cmd:
            continue
        try:
            toks = shlex.split(cmd, comments=True)
        except ValueError:
            continue
        for i, t in enumerate(toks):
            if t.endswith(".py") and "train" in Path(t).name:
                return toks[i + 1:]
    return None


def _planned_from_argv(argv: list[str], trainer_path: Path | None):
    """Parse ``argv`` through the trainer's REAL argparse -> (namespace, unknown).
    ``parse_known_args`` so an additive future flag never breaks the read-back; a
    SystemExit (bad value on a known flag) propagates to the caller's fail-open."""
    ap = build_real_trainer_parser(trainer_path)
    ns, unknown = ap.parse_known_args(argv)
    return ns, list(unknown)


def _opt_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _planned_stages(ns) -> tuple[list[StageEntry], bool, int | None, int | None]:
    """The PLANNED stage list from the parsed trainer namespace. Mirrors the
    trainer's precedence semantics (Muon > l7 > tau > CE step function) so a
    disabled (start >= epochs) or shadowed (window collapsed under precedence)
    stage is OMITTED — the exact class the '--l7 600 while l7 disabled at 1001'
    incident belongs to."""
    epochs = _opt_int(getattr(ns, "epochs", None))
    eval_every = _opt_int(getattr(ns, "eval_every", None))
    curriculum_on = bool(getattr(ns, "curriculum", False))
    evt = curriculum_on and bool(getattr(ns, "curriculum_event_triggered", False))
    nucleus = bool(getattr(ns, "curriculum_nucleus_guard", False))
    tau_s = _opt_int(getattr(ns, "tau_softplus_start_epoch", None)) if curriculum_on else None
    l7_s = _opt_int(getattr(ns, "l7_start_epoch", None)) if curriculum_on else None
    muon_s = _opt_int(getattr(ns, "muon_start_epoch", None))

    def _enabled(start: int | None) -> bool:
        if start is None:
            return False
        if epochs is not None and start >= epochs:
            return False  # the "never" form (e.g. --l7-start-epoch 1001 @ epochs 1000)
        return start > 0

    bounds: dict[str, int] = {}
    if _enabled(tau_s):
        bounds["tau"] = int(tau_s)
    if _enabled(l7_s):
        bounds["l7"] = int(l7_s)
    if _enabled(muon_s):
        bounds["Muon"] = int(muon_s)

    def _at(ep: int) -> str:  # precedence step function (Muon > l7 > tau > CE)
        for name in reversed(STAGE_ORDER):
            b = bounds.get(name)
            if b is not None and ep >= b:
                return name
        return "CE"

    # keep only stages that OWN >= 1 epoch under precedence (shadowed windows collapse)
    horizon = epochs if epochs is not None else (max(bounds.values(), default=0) + 1)
    points = sorted({0, *[b for b in bounds.values() if b < horizon], horizon})
    owned: dict[str, int] = {}
    for a in points[:-1]:
        owned.setdefault(_at(a), a)

    stages: list[StageEntry] = []
    for name in STAGE_ORDER:
        if name == "CE":
            if curriculum_on or not bounds:
                stages.append(StageEntry("CE", 0))
            elif "CE" in owned:
                stages.append(StageEntry("CE", 0))
            continue
        if name not in owned:
            continue
        start = bounds[name]
        # event mode: tau/l7 boundaries become HARD-CEILING caps (the trainer fires
        # at plateau-convergence OR the cap, whichever is earlier); Muon stays fixed
        # (the Muon event-trigger is not built — trainer help, symposium C.ii item 5).
        if evt and name in ("tau", "l7"):
            stages.append(StageEntry(
                name, None, mode="event", status="pending", cap=start,
                trigger=event_trigger_description(nucleus)))
        else:
            stages.append(StageEntry(name, start))
    return stages, evt, epochs, eval_every


# ---------------------------------------------------------------------------
# ACTUAL layer: fired-transition evidence from the run's own emissions
# ---------------------------------------------------------------------------
def _actual_from_logs(log_paths) -> dict[str, dict]:
    """Fired transitions from trainer log rows: ``curriculum_transition_fired``
    (event mode; the fired epoch IS the first epoch of the ``to`` stage) and
    ``muon_finisher_switch`` (the actual AdamW->Muon epoch). Latest row wins."""
    out: dict[str, dict] = {}
    for lp in log_paths or ():
        try:
            with open(lp, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if ('"curriculum_transition_fired"' not in line
                            and '"muon_finisher_switch"' not in line):
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("stage") == "curriculum_transition_fired":
                        name = _FORM_TO_STAGE.get(str(row.get("to")))
                        ep = _opt_int(row.get("epoch"))
                        if name and ep is not None:
                            out[name] = {"fired_epoch": ep, "source": "log",
                                         "trigger_fired": row.get("trigger")}
                    elif row.get("stage") == "muon_finisher_switch":
                        ep = _opt_int(row.get("epoch"))
                        if ep is not None:
                            out["Muon"] = {"fired_epoch": ep, "source": "log"}
        except OSError:
            continue
    return out


def _actual_from_checkpoints(run_dir: Path, event_triggered: bool) -> dict[str, dict]:
    """Fired transitions from per-stage checkpoint FILENAMES (no npz read needed).

    Naming semantics differ by mode (verified against the trainer source):
      * ``stageMuonStart_ep<N>`` — Muon switched AT N (both modes).
      * fixed mode: the transition ckpt is saved at the LAST epoch of the COMPLETED
        stage with that stage's tag (``stageCE_ep299`` -> the NEXT stage started at
        300) — corroboration for the planned boundary, so the next-stage start is
        derived as N+1.
      * event mode: the ckpt is saved AT the fired epoch with the NEW stage's tag
        (``stageTau_ep213`` -> tau fired at 213).
    """
    out: dict[str, dict] = {}
    try:
        names = sorted(p.name for p in Path(run_dir).iterdir())
    except OSError:
        return out
    for fn in names:
        m = _CKPT_RE.match(fn)
        if not m:
            continue
        tag, ep = m.group("tag"), int(m.group("ep"))
        stage = _CKPT_TAG_TO_STAGE.get(tag)
        if stage is None:
            continue
        if stage == "Muon":
            out["Muon"] = {"fired_epoch": ep, "source": "checkpoint"}
        elif event_triggered:
            out[stage] = {"fired_epoch": ep, "source": "checkpoint"}
        else:
            nxt_i = _ORDER_IDX[stage] + 1
            if nxt_i < len(STAGE_ORDER) - 1:  # next non-Muon stage boundary = N+1
                out.setdefault(STAGE_ORDER[nxt_i],
                               {"fired_epoch": ep + 1, "source": "checkpoint"})
    return out


# ---------------------------------------------------------------------------
# the read-back entry point
# ---------------------------------------------------------------------------
def read_schedule(run_dir, log_paths=None, trainer_path: Path | None = None
                  ) -> ScheduleReadback:
    """Derive the stage map a run ACTUALLY has: PLANNED (launch.sh through the real
    trainer argparse) merged with ACTUAL (emitted transition evidence; overrides
    PLANNED wherever both exist). Fail-open: any failure returns ``ok=False`` with
    a reason — the consumer (a live dashboard) then falls back visibly.

    ``log_paths``: optional extra log files to scan for transition rows (the
    training log often lives OUTSIDE the run dir, e.g. ``.omx/tmp/levelset_*.log``;
    the dashboard passes its resolved chain). The run dir's own ``*.log`` files are
    always scanned too.
    """
    if run_dir is None:
        return ScheduleReadback(ok=False, source="none", reason="no run_dir")
    rd = Path(run_dir)
    launch = rd / "launch.sh"
    if not launch.is_file():
        return ScheduleReadback(ok=False, source="none", run_dir=str(rd),
                                reason=f"no launch.sh in {rd}")
    try:
        argv = trainer_argv_from_launch_sh(launch.read_text(errors="replace"))
        if not argv:
            return ScheduleReadback(ok=False, source="none", run_dir=str(rd),
                                    reason="no trainer invocation found in launch.sh")
        ns, unknown = _planned_from_argv(argv, trainer_path)
        stages, evt, epochs, eval_every = _planned_stages(ns)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 — fail-open by contract
        return ScheduleReadback(ok=False, source="none", run_dir=str(rd),
                                reason=f"launch.sh parse failed: {exc}")

    # ACTUAL evidence: log rows (authoritative, carry the fired trigger) then
    # checkpoint filenames (corroboration / log-less fallback).
    paths = [str(p) for p in (log_paths or [])]
    try:
        paths += [str(p) for p in sorted(rd.glob("*.log"))]
    except OSError:
        pass
    actual = _actual_from_checkpoints(rd, evt)
    actual.update(_actual_from_logs(paths))  # log rows override filename evidence

    merged: list[StageEntry] = []
    for st in stages:
        ev = actual.get(st.name)
        if ev is None:
            merged.append(st)
            continue
        fired = int(ev["fired_epoch"])
        if st.mode == "event":
            merged.append(StageEntry(st.name, fired, mode="event", status="fired",
                                     fired_epoch=fired, cap=st.cap,
                                     trigger=st.trigger, source=str(ev["source"])))
        else:
            # ACTUAL overrides PLANNED (e.g. a resumed run whose real Muon switch
            # differs from the flag); keep mode=fixed, record provenance.
            merged.append(StageEntry(st.name, fired, source=str(ev["source"])))
    return ScheduleReadback(ok=True, source="launch.sh", epochs=epochs,
                            eval_every=eval_every, event_triggered=evt,
                            stages=tuple(merged), run_dir=str(rd),
                            unknown_args=tuple(unknown))
