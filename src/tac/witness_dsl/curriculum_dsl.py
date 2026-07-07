"""Witness curriculum/behavior DSL — compiles to the proven trainer CLI.

See package docstring. This module is Layer-0 of the here->theta* bridge: a
declarative front-end whose programs compile to validated launch commands for
``experiments/train_levelset_witness_realized_through_R_mlx.py``.

Design (recursion+math+enforced-behaviors, operator riff 2026-06-28):
  * The contest energy S = 100*INT d_seg + sqrt(10*INT d_pose) + 25*bytes is the
    ROOT; every lever is a term/relaxation of S, so composition is principled.
  * The curriculum is a homotopy of relaxations (CE -> tau -> l7), expressed as
    ``Stage`` tuples; the temperature anneal is a ``Schedule``.
  * Desired behaviors (preserve / contain / authority) are ENFORCED clauses, not
    advisory prose — ``validate()`` refuses a program that violates them, and
    ``compile_*`` bakes them into the emitted commands.
  * never-invent-flags is STRUCTURAL: ``validate()`` checks every emitted flag
    against the trainer's real argparse flag set (``real_trainer_flags``).

§14 schedule-design layer (task #339, operator directive 2026-07-07 "design the
SCHEDULE, not just the lever set" + "Schedule should be DSL too" + "consumers
must track DSL evolution"): the full GAP ANALYSIS of §14's six axes vs the #334
objects lives in the "§14 SCHEDULE-DESIGN PRIMITIVES" section below (LevelPath /
StageSpec{repeat_until, priming, exit_event} / OperationalSchedule /
TrainerSupportGap). Un-compilable schedule intent NEVER becomes an invented flag
— it compiles to the nearest REAL flags and surfaces as a typed
``TrainerSupportGap`` (``Curriculum.support_gaps()`` /
``validate(surface_gaps=True)``). Every schedule primitive exposes the uniform
``to_display_dict()``/``describe()`` consumer surface and auto-registers in
``schedule_primitive_kinds()`` (no hand-typed registry) so the dashboard
read-back + costate digest read the DSL live instead of drifting behind it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_REL = "experiments/train_levelset_witness_realized_through_R_mlx.py"
TRAINER_PATH = _REPO_ROOT / TRAINER_REL


def real_trainer_flags(trainer_path: Path | None = None) -> frozenset[str]:
    """Parse the trainer's argparse and return the SET of real ``--flag`` names.

    This is the structural never-invent-flags guard: a program that emits a flag
    not in this set fails ``validate()`` before any launch.
    """
    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    text = path.read_text()
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


def real_store_true_flags(trainer_path: Path | None = None) -> frozenset[str]:
    """Flags whose action is ``store_true`` — these have NO ``--no-<flag>`` form,
    so emitting them as False (which ``compile`` renders as ``--no-X``) would crash
    argparse at launch. (DSL adversarial-review C2, 2026-06-28.)"""
    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    text = path.read_text()
    return frozenset(re.findall(
        r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*action\s*=\s*["\']store_true["\']', text))


def real_boolean_flags(trainer_path: Path | None = None) -> frozenset[str]:
    """Flags whose trainer argparse action is boolean-valued — ``store_true`` OR
    ``argparse.BooleanOptionalAction``. These are the ONLY flags on which a True/False
    override is legal (they take NO value token); conversely a non-boolean flag's override
    must be a non-bool value (compiling ``True`` to a bare token on a ``type=float`` flag
    is the EikonalViscosity-class crash: argparse "expected one argument" AFTER every
    launcher gate passed). Static half of the type-compat CLASS-fix (review 2026-07-06);
    the dynamic half is the real-argparse parse test in ``test_lever_registry``."""
    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    text = path.read_text()
    return real_store_true_flags(path) | frozenset(re.findall(
        r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*action\s*=\s*argparse\.BooleanOptionalAction',
        text))


def build_real_trainer_parser(trainer_path: Path | None = None):
    """Build the trainer's REAL ``argparse.ArgumentParser`` — the dynamic never-invent-flags
    / type-compat authority (``parser.parse_args(argv)`` on an emitted argv catches wrong
    flag names AND wrong-typed values, the whole EikonalViscosity/Muon crash family, at CI
    time instead of after daemon spawn).

    The trainer has no separate parser-builder function (the parser is built inline in
    ``main()``) and importing the module pulls mlx/heavy deps — so this extracts, by AST,
    the trainer's OWN ``ap = argparse.ArgumentParser(...)`` assignment plus every
    ``ap.add_argument(...)`` statement from ``main()`` and executes exactly those statements
    (they reference only ``argparse``/``Path``/builtins — asserted below). The executed code
    IS the trainer's source, so defaults/types/actions/choices are the trainer's own, not a
    regex approximation. Deterministic; cross-checked against :func:`real_trainer_flags`
    (fail-LOUD on extraction drift, never a silently stale parser)."""
    import argparse as _argparse
    import ast as _ast

    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    tree = _ast.parse(path.read_text())
    main_fn = next(
        (n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "main"), None)
    if main_fn is None:
        raise LookupError(f"build_real_trainer_parser: no main() found in {path}")
    stmts: list[_ast.stmt] = []
    for node in _ast.walk(main_fn):
        if isinstance(node, _ast.Assign) and any(
                isinstance(t, _ast.Name) and t.id == "ap" for t in node.targets):
            stmts.append(node)
        elif (isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call)
              and isinstance(node.value.func, _ast.Attribute)
              and node.value.func.attr == "add_argument"
              and isinstance(node.value.func.value, _ast.Name)
              and node.value.func.value.id == "ap"):
            stmts.append(node)
    stmts.sort(key=lambda n: n.lineno)  # ast.walk order is not source order
    ns: dict = {"argparse": _argparse, "Path": Path}
    exec(compile(_ast.Module(body=stmts, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    ap = ns.get("ap")
    if not isinstance(ap, _argparse.ArgumentParser):
        raise LookupError(
            f"build_real_trainer_parser: extraction did not yield an ArgumentParser from {path} "
            "(trainer main()/parser shape changed — update the extractor)")
    # fail-LOUD cross-check: every regex-visible flag must exist on the built parser.
    built = {opt for a in ap._actions for opt in a.option_strings}
    missing = real_trainer_flags(path) - built
    if missing:
        raise LookupError(
            "build_real_trainer_parser: extracted parser is missing flags the trainer source "
            f"declares (extraction drift): {sorted(missing)[:10]}")
    return ap


# sentinel so with_lever() can explicitly CLEAR resume_from (fresh run) vs inherit it
_INHERIT = object()


# ---------------------------------------------------------------------------
# Consumer-introspection surface (operator amendment 2026-07-07: "As the DSL
# evolves, update the costate controller and dashboard accordingly") — every
# schedule/curriculum primitive exposes ONE uniform display contract so generic
# consumers (dashboard schedule read-back, costate digest) enumerate primitives
# as plain data instead of per-primitive code. New primitives inherit this and
# are AUTO-registered by :func:`schedule_primitive_kinds` (no hand-typed list).
# ---------------------------------------------------------------------------
def _display_plain(v):
    """Recursively convert a primitive's field value to plain JSON-able data."""
    from dataclasses import fields as _fields, is_dataclass as _is_dc
    if _is_dc(v) and not isinstance(v, type):
        td = getattr(v, "to_display_dict", None)
        if callable(td):
            return td()
        return {f.name: _display_plain(getattr(v, f.name)) for f in _fields(v)}
    if isinstance(v, (tuple, list)):
        return [_display_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _display_plain(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    return v


class ScheduleDisplay:
    """Uniform describe()/to_display_dict() surface for schedule primitives.

    ``to_display_dict()`` returns ``{"kind": <ClassName>, <field>: <plain data>...}``
    plus, when the primitive compiles with a no-arg ``flags()``, the compiled
    ``"flags"``, and when it carries a no-arg ``support_gaps()``, the ``"gaps"``
    (each gap as plain data). A generic renderer needs NO type knowledge.
    """

    def to_display_dict(self) -> dict:
        import inspect as _inspect
        from dataclasses import fields as _fields
        d: dict = {"kind": type(self).__name__}
        for fld in _fields(self):  # type: ignore[arg-type]
            d[fld.name] = _display_plain(getattr(self, fld.name))
        for meth, key in (("flags", "flags"), ("support_gaps", "gaps")):
            fn = getattr(self, meth, None)
            if not callable(fn):
                continue
            try:
                sig = _inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            required = [p for p in sig.parameters.values()
                        if p.default is p.empty
                        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if required:
                continue  # e.g. Anneal.flags(start_flag, end_flag) — fields already shown
            try:
                d[key] = _display_plain(fn())
            except Exception as exc:  # noqa: BLE001 — display is FAIL-OPEN by contract:
                # the consumer is a load-bearing multi-day dashboard/costate daemon
                # (mirrors schedule_readback); an invalid in-flight object must render
                # its error, never crash the tick. validate() is the fail-CLOSED gate.
                d[key] = {"error": f"{type(exc).__name__}: {exc}"}
        return d

    def describe(self) -> str:
        d = self.to_display_dict()
        kind = d.pop("kind")
        parts = ", ".join(
            f"{k}={v!r}" for k, v in d.items()
            if v is not None and not isinstance(v, (dict, list)))
        return f"{kind}({parts})"


@dataclass(frozen=True)
class TrainerSupportGap(ScheduleDisplay):
    """A NAMED, TYPED expressiveness gap: schedule intent the council can EXPRESS
    in the DSL but the trainer cannot yet CONSUME. This is a FEATURE, not an error
    channel — it tells the council exactly which schedule ideas need trainer builds.

    never-invent-flags is preserved STRUCTURALLY: ``flag_proposal`` is a PROPOSAL
    for a trainer build and is NEVER emitted into argv; ``nearest_real_compilation``
    documents the conservative compile (real flags only) actually emitted."""

    axis: str                      # §14 axis, e.g. "stage_repetition" | "levels_as_paths"
    requirement: str               # what the council asked for
    nearest_real_compilation: str  # what compile_trainer_argv() emits instead (REAL flags)
    flag_proposal: str             # the trainer build this needs (NOT emitted, ever)
    notes: str = ""

    def describe(self) -> str:
        return (f"TRAINER-SUPPORT GAP [{self.axis}]: {self.requirement}"
                f" | compiled-nearest: {self.nearest_real_compilation}"
                f" | requires trainer support: {self.flag_proposal}"
                + (f" | {self.notes}" if self.notes else ""))


# ---------------------------------------------------------------------------
# Schedule primitives (the homotopy / anneal math)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anneal(ScheduleDisplay):
    """A cosine-annealed schedule start->end (e.g. softmax temperature tau)."""

    start: float
    end: float

    def flags(self, start_flag: str, end_flag: str) -> dict:
        return {start_flag: self.start, end_flag: self.end}


def Freeze(value: float) -> Anneal:  # noqa: N802 (DSL keyword)
    """Freeze a schedule at a constant value (Anneal with start==end)."""
    return Anneal(value, value)


# ---------------------------------------------------------------------------
# Curriculum stage (a relaxation of S) + regularizers (live PDE constraints)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage(ScheduleDisplay):
    """A curriculum relaxation. ``start_epoch`` maps to the trainer's stage gate."""

    name: str
    start_epoch_flag: str | None  # e.g. "--tau-softplus-start-epoch"; None for the CE base
    start_epoch: int | None = None

    def flags(self) -> dict:
        if self.start_epoch_flag is None or self.start_epoch is None:
            return {}
        return {self.start_epoch_flag: self.start_epoch}


@dataclass(frozen=True)
class Regularizer(ScheduleDisplay):
    """A live derivative/integral regularizer (eikonal |grad phi|=1, length INT ds)."""

    flag: str
    weight: float

    def flags(self) -> dict:
        return {self.flag: self.weight}


@dataclass(frozen=True)
class HoscSchedule(ScheduleDisplay):
    """The hosc activation-slope β anneal (start→end over the run, shape) + ω — the
    REPRESENTATION-sharpening schedule half of the curriculum (distinct from the render-partition
    temp Anneal and the seg-surrogate tau). MEASURED (DAG FEED-ly): FIXED β diverges
    (tanh(β·sin) saturation → vanishing grad → AdamW random-walk → d_seg RISES); the ANNEALED
    β 1→4 is the stable survivor. Elevated from a raw ``base`` flag to a first-class schedule
    object so the whole sharpening path is DSL-held + costate-controllable."""

    beta_start: float = 1.0
    beta_end: float = 4.0
    shape: str = "linear"
    omega: float = 1.0

    def flags(self) -> dict:
        return {
            "--hosc-beta": self.beta_start,
            "--hosc-beta-end": self.beta_end,
            "--hosc-beta-anneal": self.shape,
            "--hosc-omega": self.omega,
        }


@dataclass(frozen=True)
class Transition(ScheduleDisplay):
    """The stage-transition REHEAT treatment (FEED-fz BUILD 1 / FEED-bu, "different stages need
    different treatment"): LR floor→1× rewarmup over a window + optional AdamW 2nd-moment reset,
    so a curriculum stage boundary is stable BY CONSTRUCTION (not a d_seg spike). Elevated from
    raw ``--stage-transition-*`` flags to a first-class object the Curriculum owns."""

    rewarmup_epochs: int = 8
    rewarmup_floor: float = 0.1
    rewarmup_shape: str = "linear"
    reset_moments: bool = True

    def flags(self) -> dict:
        f: dict = {
            "--stage-transition-rewarmup-epochs": self.rewarmup_epochs,
            "--stage-transition-rewarmup-floor": self.rewarmup_floor,
            "--stage-transition-rewarmup-shape": self.rewarmup_shape,
        }
        # --stage-transition-reset-moments is store_true → emit ONLY when True (review C2:
        # a False on a store_true flag compiles to --no-X and crashes argparse at launch).
        if self.reset_moments:
            f["--stage-transition-reset-moments"] = True
        return f


# ---------------------------------------------------------------------------
# §14 SCHEDULE-DESIGN PRIMITIVES (task #339; operator directive 2026-07-07,
# DRAFT_derived_optimal_next_run_for_council_20260707.md §14 — "design the
# SCHEDULE, not just the lever set" + amendment "Schedule should be DSL too").
#
# GAP ANALYSIS — §14's six axes mapped onto the #334 objects (what could already
# be expressed / what THIS block adds / what needs a trainer build):
#
#  axis 1 ACTIVATION TIMING: ALREADY — fixed entry via Stage.start_epoch; the
#    CE→tau EVENT hand-off via Curriculum(handoff="event") (#315 machinery,
#    EventTriggeredCurriculum lever). ADDED — ExitEvent carries the plateau/
#    nucleus-guard PARAMETERS (--curriculum-min-stage-epochs / -plateau-rel-eps /
#    -plateau-windows / -nucleus-within-flip / -nucleus-min-part-frac) as typed
#    fields. GAP — a Muon ENTRY event ("enter when tau's conditioning stalls",
#    #302): the trainer's event controller governs the CE→tau(→l7) seg-form
#    hand-offs only; --muon-start-epoch is fixed-epoch. TrainerSupportGap.
#  axis 2 LEVELS AS PATHS λ(t): ALREADY — single cosine temp Anneal, hosc β
#    linear/cosine anneal, Transition reheat. ADDED — LevelPath: per-quantity
#    typed paths compiling to the trainer's REAL path mechanisms: softmax_temp
#    {constant|cosine|geometric|cosine_hold (+--tau-hold-frac), span
#    --anneal-epochs}; hosc_beta {constant|linear|cosine}; lr {constant|cosine
#    (--lr/--lr-end/--lr-schedule/--warmup-epochs)}; eikonal_weight {constant|
#    step-at-tau-onset (--eikonal-weight-end, cosine-eased over the rewarmup
#    window)}; ema_decay {2-segment piecewise-constant (--ema-decay-finisher[-
#    start-epoch]) — the §14 "EMA per stage" π-group}; muon_lr_frac {cosine
#    (--muon-lr-final-frac)}. GAP — shapes/segment-counts beyond those (e.g.
#    geometric LR, linear temp, 3+ segments, per-class λ homotopy paths):
#    conservative compile = nearest supported shape + TrainerSupportGap.
#  axis 3 STAGE SET/ORDER/REPETITION: ALREADY — one-shot ordered stages.
#    ADDED — StageSpec.repeat_until (RepeatUntil): repeat-a-block-until-EVENT
#    cycles (tau→Muon→(Muon+leap)×until-dry). GAP — the trainer's stage ladder
#    is single-pass monotone (seg_form step function); NO repeat support:
#    conservative compile = the DETERMINISTIC bound block_epochs×max_repeats
#    (council sets --epochs to cover it) + TrainerSupportGap w/ flag proposal.
#  axis 4 PRIMING: ALREADY — loose flags only (--muon-warm-start-momentum via
#    the MuonWarmStart lever, --structured-init/--siren-init/--finer-bias-init
#    in base dicts, --stage-transition-reset-moments via Transition). ADDED —
#    StageSpec.priming (Priming): per-stage entry actions as first-class typed
#    fields. GAP — per-stage SCOPING of reset_moments (trainer flag is global,
#    fires at ALL transitions) and mid-run re-init (init primings apply at run
#    entry only): compiled globally + TrainerSupportGap notes the approximation.
#  axis 5 MAX-CONVERGENCE TERMINATION: ADDED — ExitEvent criteria
#    "marginal_dseg_floor"/"lever_exhaustion" (per-lever marginal-Δd_seg floors)
#    are EXPRESSIBLE but the trainer has NO such run-end criterion (fixed
#    --epochs): conservative compile = the fixed stage/run bound +
#    TrainerSupportGap (this is the "no meat left" trainer build the council
#    should commission).
#  axis 6 OPERATIONAL SCHEDULE / WALL-CLOCK (amendment "Schedule should be DSL
#    too"): ALREADY — Preserve holds the checkpoint cadence (--ckpt-every /
#    --stage-checkpoints; COMPOSE, don't duplicate — OperationalSchedule does
#    NOT re-own checkpoint flags). ADDED — OperationalSchedule: verdict cadence
#    (--eval-every; +16 s/ep every other 25-ep block MEASURED, a real wall-clock
#    knob) + verdict scope/chunk (--verdict-pairs/--verdict-batch) + async
#    verdict (--async-verdict) + telemetry cadences (--annulus-telemetry/-band/
#    -bottom-k, --loss-term-log-every, --handoff-readiness-telemetry,
#    --dm1-telemetry) + self-orient re-orientation cadence (--reorient-every).
#    GAP — PER-STAGE verdict-cadence overrides (dense in the Muon finisher,
#    sparse in bulk descent): every trainer cadence flag is GLOBAL; conservative
#    compile = the DENSEST requested cadence globally (no verdict evidence lost,
#    costs wall-clock) + TrainerSupportGap.
#
# Un-compilable expressiveness ALWAYS surfaces as a typed TrainerSupportGap
# (never an invented flag); collect via Curriculum.support_gaps() or
# Curriculum.validate(surface_gaps=True).
# ---------------------------------------------------------------------------
_PATH_SHAPES = ("constant", "linear", "geometric", "cosine", "cosine_hold", "step")

#: LevelPath quantities → the trainer-supported shape set for that quantity
#: (consumed by LevelPath._compile; adding a quantity = one entry + one branch).
_LEVEL_PATH_QUANTITIES: dict[str, frozenset[str]] = {
    "softmax_temp": frozenset({"constant", "cosine", "geometric", "cosine_hold"}),
    "hosc_beta": frozenset({"constant", "linear", "cosine"}),
    "lr": frozenset({"constant", "cosine"}),
    "eikonal_weight": frozenset({"constant", "step"}),
    "ema_decay": frozenset({"constant"}),   # 2-segment piecewise-constant supported
    "muon_lr_frac": frozenset({"cosine"}),
}

#: quantities whose trainer mechanism is inherently STAGE-ANCHORED (a per-stage
#: LevelPath on one of these is genuinely per-stage; any other quantity's
#: ``stage=`` scoping is a TrainerSupportGap).
_STAGE_ANCHORED_QUANTITIES = frozenset({"ema_decay", "eikonal_weight", "muon_lr_frac"})


@dataclass(frozen=True)
class PathSegment(ScheduleDisplay):
    """One segment of a λ(t) LevelPath. ``end=None`` == constant at ``start``.

    ``epochs`` is the segment span; for ``ema_decay`` (the 2-segment piecewise-
    constant path) segment-0's ``epochs`` is the ABSOLUTE epoch at which segment
    1 engages (--ema-decay-finisher-start-epoch). ``hold_frac`` is cosine_hold
    only (the (0,1] fraction of the anneal window at which the floor is reached)."""

    shape: str
    start: float
    end: float | None = None
    epochs: int | None = None
    hold_frac: float | None = None

    def resolved_end(self) -> float:
        return self.start if self.end is None else self.end


@dataclass(frozen=True)
class LevelPath(ScheduleDisplay):
    """§14 axis 2 — a LEVEL AS A PATH λ(t) for one schedule quantity.

    COMPOSE, DON'T DUPLICATE (review attack surface): for ``softmax_temp`` /
    ``hosc_beta`` the endpoint flags are ALSO owned by ``Curriculum.temp`` /
    ``Curriculum.hosc`` — a LevelPath on those quantities must AGREE with the
    endpoint object (Curriculum.validate()'s duplicate-emitter check refuses
    unequal values); the path adds ONLY the shape flags (--tau-anneal-shape /
    --hosc-beta-anneal etc.) the endpoint objects do not hold. Constancy is a
    DECISION, not a default (§14 axis 2): a constant path emits start==end
    explicitly.

    ``stage``: optional per-stage scoping declaration. Genuinely per-stage only
    for the stage-anchored quantities (ema_decay / eikonal_weight /
    muon_lr_frac); otherwise a TrainerSupportGap (the trainer holds ONE global
    anneal per quantity)."""

    quantity: str
    segments: tuple[PathSegment, ...]
    stage: str | None = None
    anneal_epochs: int | None = None   # --anneal-epochs (softmax_temp schedule denominator)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.quantity not in _LEVEL_PATH_QUANTITIES:
            problems.append(
                f"LevelPath: unknown quantity {self.quantity!r} "
                f"(known: {sorted(_LEVEL_PATH_QUANTITIES)})")
        if not self.segments:
            problems.append(f"LevelPath[{self.quantity}]: needs >=1 segment")
        for i, seg in enumerate(self.segments):
            if seg.shape not in _PATH_SHAPES:
                problems.append(
                    f"LevelPath[{self.quantity}] segment {i}: unknown shape {seg.shape!r} "
                    f"(known: {_PATH_SHAPES})")
            if seg.shape == "geometric" and (seg.start <= 0 or seg.resolved_end() <= 0):
                problems.append(
                    f"LevelPath[{self.quantity}] segment {i}: geometric needs positive "
                    f"endpoints, got start={seg.start} end={seg.resolved_end()}")
            if seg.shape == "cosine_hold" and not (
                    seg.hold_frac is not None and 0.0 < seg.hold_frac <= 1.0):
                problems.append(
                    f"LevelPath[{self.quantity}] segment {i}: cosine_hold needs "
                    f"hold_frac in (0,1], got {seg.hold_frac!r}")
        return problems

    # --- compile: nearest REAL flags + typed gaps for the rest ---------------
    def _compile(self) -> tuple[dict, tuple[TrainerSupportGap, ...]]:
        if self.quantity not in _LEVEL_PATH_QUANTITIES or not self.segments:
            raise ValueError(
                f"LevelPath[{self.quantity!r}]: not compilable — run .validate() "
                "(unknown quantity or empty segments)")
        gaps: list[TrainerSupportGap] = []
        f: dict = {}
        q = self.quantity
        segs = self.segments
        first = segs[0]
        supported = _LEVEL_PATH_QUANTITIES[q]
        shape = first.shape

        # multi-segment: only ema_decay's 2-constant-segment form has trainer support.
        multi_ok = (q == "ema_decay" and len(segs) == 2
                    and all(s.shape == "constant" for s in segs))
        if len(segs) > 1 and not multi_ok:
            gaps.append(TrainerSupportGap(
                axis="levels_as_paths",
                requirement=(f"{q}: {len(segs)}-segment path "
                             f"[{', '.join(s.shape for s in segs)}]"),
                nearest_real_compilation=(
                    f"first segment only ({shape} {first.start}->{first.resolved_end()})"),
                flag_proposal=f"--{q.replace('_', '-')}-schedule <piecewise spec> (trainer build)",
                notes="trainer holds ONE anneal per quantity (ema_decay: 2 constants)"))
            segs = (first,)

        if shape not in supported and not multi_ok:
            nearest = ("cosine" if "cosine" in supported
                       else "step" if "step" in supported
                       else "constant")
            gaps.append(TrainerSupportGap(
                axis="levels_as_paths",
                requirement=f"{q}: shape {shape!r} {first.start}->{first.resolved_end()}",
                nearest_real_compilation=(
                    f"{nearest} {first.start}->{first.resolved_end()} (same endpoints)"),
                flag_proposal=f"extend the trainer's {q} anneal choices with {shape!r}",
                notes=f"trainer-supported shapes for {q}: {sorted(supported)}"))
            shape = nearest

        if q == "softmax_temp":
            f["--softmax-temp-start"] = float(first.start)
            f["--softmax-temp-end"] = float(first.resolved_end())
            if shape in ("cosine", "geometric", "cosine_hold"):
                f["--tau-anneal-shape"] = shape
            if shape == "cosine_hold" and first.hold_frac is not None:
                f["--tau-hold-frac"] = float(first.hold_frac)
            if self.anneal_epochs is not None:
                f["--anneal-epochs"] = int(self.anneal_epochs)
        elif q == "hosc_beta":
            f["--hosc-beta"] = float(first.start)
            f["--hosc-beta-end"] = float(first.resolved_end())
            if shape in ("linear", "cosine"):
                f["--hosc-beta-anneal"] = shape
        elif q == "lr":
            f["--lr"] = float(first.start)
            if shape == "constant":
                f["--lr-schedule"] = False
            else:
                f["--lr-end"] = float(first.resolved_end())
                f["--lr-schedule"] = True
        elif q == "eikonal_weight":
            f["--eikonal-weight"] = float(first.start)
            if shape == "step":
                f["--eikonal-weight-end"] = float(first.resolved_end())
        elif q == "ema_decay":
            f["--ema-decay"] = float(first.start)
            if multi_ok:
                f["--ema-decay-finisher"] = float(self.segments[1].start)
                if first.epochs is not None:
                    f["--ema-decay-finisher-start-epoch"] = int(first.epochs)
        elif q == "muon_lr_frac":
            f["--muon-lr-final-frac"] = float(first.resolved_end())

        if self.stage is not None and q not in _STAGE_ANCHORED_QUANTITIES:
            gaps.append(TrainerSupportGap(
                axis="levels_as_paths",
                requirement=f"{q}: path scoped to stage {self.stage!r}",
                nearest_real_compilation="the path applies GLOBALLY (whole-run anneal)",
                flag_proposal=f"per-stage {q} path table (trainer build)",
                notes=f"stage-anchored quantities: {sorted(_STAGE_ANCHORED_QUANTITIES)}"))
        return f, tuple(gaps)

    def flags(self) -> dict:
        return self._compile()[0]

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
        return self._compile()[1]


@dataclass(frozen=True)
class Priming(ScheduleDisplay):
    """§14 axis 4 — per-stage ENTRY actions as first-class fields (not loose flags).

    * ``warm_start_momentum`` → ``--muon-warm-start-momentum`` (#269/#270: seeds
      fresh Muon momentum from the outgoing AdamW first moment; valid on the
      muon stage — the trainer fires it at the AdamW→Muon switch).
    * ``reset_moments`` → ``--stage-transition-reset-moments`` (store_true;
      GLOBAL — fires at ALL stage transitions; per-stage scoping is a gap).
    * ``structured_init`` / ``lane_prior_phi1`` / ``siren_init`` /
      ``finer_bias_k`` → the run-ENTRY seeded/bias init primings
      (``--structured-init`` / ``--lane-prior-phi1`` / ``--siren-init`` /
      ``--finer-bias-init``+``--finer-bias-k``); on a non-entry stage they are
      a gap (the trainer applies init once, at run entry).

    Warm-start-from-checkpoint priming stays PROGRAM-level (``resume_from`` +
    ``--anneal-epochs``), not per-stage — see WitnessProgram."""

    warm_start_momentum: bool = False
    reset_moments: bool = False
    structured_init: bool = False
    lane_prior_phi1: bool = False
    siren_init: bool = False
    finer_bias_k: float | None = None

    def flags(self) -> dict:
        f: dict = {}
        if self.warm_start_momentum:
            f["--muon-warm-start-momentum"] = True
        if self.reset_moments:
            f["--stage-transition-reset-moments"] = True  # store_true: True only (C2)
        if self.structured_init:
            f["--structured-init"] = True
        if self.lane_prior_phi1:
            f["--lane-prior-phi1"] = True
        if self.siren_init:
            f["--siren-init"] = True
        if self.finer_bias_k is not None:
            f["--finer-bias-init"] = True
            f["--finer-bias-k"] = float(self.finer_bias_k)
        return f

    def _has_entry_init(self) -> bool:
        return bool(self.structured_init or self.lane_prior_phi1
                    or self.siren_init or self.finer_bias_k is not None)


_EXIT_EVENT_CRITERIA = ("plateau", "nucleus_guarded_plateau",
                     "marginal_dseg_floor", "lever_exhaustion")


@dataclass(frozen=True)
class ExitEvent(ScheduleDisplay):
    """§14 axes 1+5 — a per-stage EXIT criterion (distinct from entry triggers).

    * ``plateau`` / ``nucleus_guarded_plateau``: COMPILABLE — the trainer's #315
      event controller (--curriculum-event-triggered + plateau params; nucleus
      kind adds --curriculum-nucleus-guard + its params). Governs the CE→tau(→l7)
      seg-form hand-offs; on a Muon stage it is a gap (Muon event not built).
    * ``marginal_dseg_floor`` / ``lever_exhaustion``: the §14 axis-5 "no meat
      left" exits (per active lever, marginal Δd_seg/epoch below ``floor``) —
      NO trainer support; conservative compile = the fixed stage/run boundary
      (``cap_epoch`` documents the deterministic ceiling) + TrainerSupportGap.
    """

    criterion: str
    min_stage_epochs: int = 150       # --curriculum-min-stage-epochs
    rel_eps: float = 1e-4             # --curriculum-plateau-rel-eps
    windows: int = 4                  # --curriculum-plateau-windows
    within_flip: float = 0.5          # --curriculum-nucleus-within-flip (nucleus kind)
    min_part_frac: float = 0.0        # --curriculum-nucleus-min-part-frac (nucleus kind)
    floor: float | None = None        # marginal-Δd_seg/epoch floor (gap kinds)
    cap_epoch: int | None = None      # deterministic hard ceiling (gap kinds' compile)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.criterion not in _EXIT_EVENT_CRITERIA:
            problems.append(
                f"ExitEvent: unknown criterion {self.criterion!r} (known: {_EXIT_EVENT_CRITERIA})")
        if self.criterion in ("marginal_dseg_floor", "lever_exhaustion") and self.floor is None:
            problems.append(
                f"ExitEvent[{self.criterion}]: needs an explicit ``floor`` (the pre-registered "
                "marginal-Δd_seg/epoch floor — §14 axis 5)")
        if self.criterion in ("plateau", "nucleus_guarded_plateau"):
            if self.min_stage_epochs <= 0 or self.windows <= 0 or self.rel_eps <= 0:
                problems.append(
                    f"ExitEvent[{self.criterion}]: plateau params must be positive "
                    f"(min_stage_epochs={self.min_stage_epochs}, windows={self.windows}, "
                    f"rel_eps={self.rel_eps})")
        return problems

    def flags(self) -> dict:
        if self.criterion not in ("plateau", "nucleus_guarded_plateau"):
            return {}  # conservative compile: the fixed stage boundary IS the exit
        f: dict = {
            "--curriculum-event-triggered": True,
            "--curriculum-min-stage-epochs": int(self.min_stage_epochs),
            "--curriculum-plateau-rel-eps": float(self.rel_eps),
            "--curriculum-plateau-windows": int(self.windows),
        }
        if self.criterion == "nucleus_guarded_plateau":
            f["--curriculum-nucleus-guard"] = True
            f["--curriculum-nucleus-within-flip"] = float(self.within_flip)
            f["--curriculum-nucleus-min-part-frac"] = float(self.min_part_frac)
        return f

    def support_gaps(self, stage: str | None = None) -> tuple[TrainerSupportGap, ...]:
        gaps: list[TrainerSupportGap] = []
        if self.criterion in ("marginal_dseg_floor", "lever_exhaustion"):
            gaps.append(TrainerSupportGap(
                axis="exit_events",
                requirement=(f"{self.criterion} exit"
                             + (f" on stage {stage!r}" if stage else "")
                             + f" (floor={self.floor})"),
                nearest_real_compilation=(
                    f"fixed boundary (deterministic cap_epoch={self.cap_epoch})"),
                flag_proposal=("--stage-exit-marginal-dseg-floor <float> + per-lever "
                               "marginal-Δd_seg telemetry exit (trainer build)"),
                notes="§14 axis 5: 'no meat left' = every stage exits on evidence"))
        elif stage is not None and "muon" in stage.lower():
            gaps.append(TrainerSupportGap(
                axis="activation_timing",
                requirement=f"event exit on the Muon stage ({self.criterion})",
                nearest_real_compilation="fixed --muon-start-epoch boundary",
                flag_proposal="Muon entry/exit event in the trainer's event controller "
                              "(#302 Muon-from-conditioning; trainer build)",
                notes="the #315 controller governs the CE→tau(→l7) seg-form hand-offs only"))
        return tuple(gaps)


_REPEAT_CRITERIA = ("plateau", "exhaustion", "marginal_dseg_floor")


@dataclass(frozen=True)
class RepeatUntil(ScheduleDisplay):
    """§14 axis 3 — repeat a stage/block until a MEASURED criterion (cycles).

    NO trainer support (the stage ladder is single-pass monotone): compiles to a
    conservative DETERMINISTIC bound ``block_epochs * max_repeats`` (the council
    sets ``--epochs`` to cover it) + a TrainerSupportGap naming the build.

    DETERMINISTIC-REPRODUCIBILITY SPINE (review attack, documented): the bounds
    are explicit ints; when trainer support lands, the repeat decision MUST be a
    deterministic function of the recorded telemetry stream (same seed + config
    + inputs → same telemetry → same repeat count), never wall-clock/host state."""

    criterion: str                 # plateau | exhaustion | marginal_dseg_floor
    block: tuple[str, ...]         # stage names cycled, e.g. ("muon", "muon_leap")
    block_epochs: int              # one repetition window (explicit, deterministic)
    max_repeats: int               # REQUIRED bound (never unbounded)
    floor: float | None = None     # the "dry" floor for marginal_dseg_floor/exhaustion

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.criterion not in _REPEAT_CRITERIA:
            problems.append(
                f"RepeatUntil: unknown criterion {self.criterion!r} (known: {_REPEAT_CRITERIA})")
        if not self.block:
            problems.append("RepeatUntil: block must name >=1 stage")
        if self.block_epochs <= 0:
            problems.append(f"RepeatUntil: block_epochs must be > 0, got {self.block_epochs}")
        if self.max_repeats < 1:
            problems.append(
                f"RepeatUntil: max_repeats must be >= 1 (the DETERMINISTIC bound; "
                f"unbounded repetition breaks the reproducibility spine), got {self.max_repeats}")
        return problems

    def conservative_epoch_bound(self) -> int:
        return int(self.block_epochs) * int(self.max_repeats)


@dataclass(frozen=True)
class StageSpec(Stage):
    """A Stage with §14 schedule semantics: ``repeat_until`` (axis 3) +
    ``priming`` (axis 4) + ``exit_event`` (axes 1+5). Drop-in wherever a Stage
    goes (``Curriculum.stages``): ``flags()`` emits the entry flag plus the
    priming/exit-event flags; un-compilable intent surfaces via
    ``support_gaps()`` (collected by ``Curriculum.support_gaps()``)."""

    repeat_until: RepeatUntil | None = None
    priming: Priming | None = None
    exit_event: ExitEvent | None = None

    def flags(self) -> dict:
        f = super().flags()
        if self.priming is not None:
            f.update(self.priming.flags())
        if self.exit_event is not None:
            f.update(self.exit_event.flags())
        return f

    def validate(self) -> list[str]:
        problems: list[str] = []
        if (self.priming is not None and self.priming.warm_start_momentum
                and "muon" not in self.name.lower()):
            problems.append(
                f"StageSpec[{self.name}]: warm_start_momentum priming fires only at the "
                "AdamW->Muon switch — put it on the muon stage")
        if self.repeat_until is not None:
            problems.extend(self.repeat_until.validate())
        if self.exit_event is not None:
            problems.extend(self.exit_event.validate())
        return problems

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
        gaps: list[TrainerSupportGap] = []
        if self.repeat_until is not None:
            ru = self.repeat_until
            gaps.append(TrainerSupportGap(
                axis="stage_repetition",
                requirement=(f"repeat block {list(ru.block)} until {ru.criterion}"
                             + (f" (floor={ru.floor})" if ru.floor is not None else "")),
                nearest_real_compilation=(
                    f"fixed bound {ru.conservative_epoch_bound()} epochs "
                    f"({ru.block_epochs}x{ru.max_repeats}) — set --epochs to cover it"),
                flag_proposal=("--stage-repeat-block <names> --stage-repeat-until "
                               f"<{ru.criterion}> --stage-repeat-max <int> (trainer build)"),
                notes="repeat decision must be deterministic given the telemetry stream"))
        if self.exit_event is not None:
            gaps.extend(self.exit_event.support_gaps(stage=self.name))
        if self.priming is not None:
            is_entry = self.start_epoch_flag is None  # the CE/base stage
            if self.priming.reset_moments:
                gaps.append(TrainerSupportGap(
                    axis="priming",
                    requirement=f"reset_moments scoped to stage {self.name!r}",
                    nearest_real_compilation=("--stage-transition-reset-moments "
                                              "(GLOBAL: fires at ALL stage transitions)"),
                    flag_proposal="per-transition moment-reset selector (trainer build)"))
            if self.priming._has_entry_init() and not is_entry:
                gaps.append(TrainerSupportGap(
                    axis="priming",
                    requirement=f"seeded/bias init priming at MID-RUN stage {self.name!r}",
                    nearest_real_compilation="init flags apply at RUN ENTRY only",
                    flag_proposal="per-stage re-init hook (trainer build)",
                    notes="structured/siren/FINER-bias init are run-entry primings"))
        return tuple(gaps)


# ---------------------------------------------------------------------------
# The OPERATIONAL schedule (operator amendment 2026-07-07 "Schedule should be
# DSL too"): verdict / telemetry / re-orientation cadences as first-class,
# round-trip-validated objects. Checkpoint cadence stays with Preserve (the
# existing first-class owner — COMPOSE, don't duplicate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerdictCadence(ScheduleDisplay):
    """Verdict cadence + scope + chunking (--eval-every / --verdict-pairs /
    --verdict-batch / --async-verdict). MEASURED wall-clock knob: +16 s/ep every
    other 25-ep block (§14 axis 6). ``verdict_pairs=0`` = ALL 600 (the n600
    non-negotiable default; subsetting stays OPT-IN)."""

    eval_every: int = 25
    verdict_pairs: int = 0
    verdict_batch: int = 32
    async_verdict: bool = False

    def flags(self) -> dict:
        return {
            "--eval-every": int(self.eval_every),
            "--verdict-pairs": int(self.verdict_pairs),
            "--verdict-batch": int(self.verdict_batch),
            "--async-verdict": bool(self.async_verdict),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.eval_every <= 0:
            problems.append(f"VerdictCadence: eval_every must be > 0, got {self.eval_every}")
        if self.verdict_pairs < 0 or self.verdict_batch < 0:
            problems.append(
                f"VerdictCadence: verdict_pairs/verdict_batch must be >= 0, got "
                f"{self.verdict_pairs}/{self.verdict_batch}")
        return problems


@dataclass(frozen=True)
class TelemetryCadence(ScheduleDisplay):
    """Telemetry/observer cadences. Score-neutral read-only observability
    DEFAULTS ON per the 'off is a tracked queue' law (annulus rides the verdict
    cadence; --loss-term-log-every 0 = per-epoch summary, -1 = fully off)."""

    annulus: bool = True
    annulus_band: float = 2.0
    annulus_bottom_k: float = 0.05
    loss_term_log_every: int = 0
    handoff_readiness: bool = False
    dm1: bool = False

    def flags(self) -> dict:
        f: dict = {
            "--annulus-telemetry": bool(self.annulus),
            "--loss-term-log-every": int(self.loss_term_log_every),
            "--handoff-readiness-telemetry": bool(self.handoff_readiness),
        }
        if self.annulus:
            f["--annulus-band"] = float(self.annulus_band)
            f["--annulus-bottom-k"] = float(self.annulus_bottom_k)
        if self.dm1:
            f["--dm1-telemetry"] = True  # store_true: True only (C2)
        return f


@dataclass(frozen=True)
class OperationalSchedule(ScheduleDisplay):
    """§14 axis 6 — the run's OPERATIONAL schedule as one first-class object.

    ``per_stage_verdict`` ({stage_name: eval_every}) expresses the council's
    "verdict sparse in CE/tau, dense in the Muon finisher" — NO trainer support
    (every cadence flag is global): the conservative compile is the DENSEST
    requested cadence GLOBALLY (no verdict evidence lost; costs wall-clock) +
    a TrainerSupportGap. ``reorient_every`` = the self-orient re-orientation
    cadence (None = leave the trainer default)."""

    verdict: VerdictCadence = VerdictCadence()
    telemetry: TelemetryCadence = TelemetryCadence()
    reorient_every: int | None = None
    per_stage_verdict: dict = field(default_factory=dict)

    def _effective_eval_every(self) -> int:
        cands = [int(self.verdict.eval_every)]
        cands += [int(v) for v in self.per_stage_verdict.values()]
        return min(cands)

    def flags(self) -> dict:
        f = self.verdict.flags()
        if self.per_stage_verdict:
            f["--eval-every"] = self._effective_eval_every()  # densest = conservative
        f.update(self.telemetry.flags())
        if self.reorient_every is not None:
            f["--reorient-every"] = int(self.reorient_every)
        return f

    def validate(self) -> list[str]:
        problems = list(self.verdict.validate())
        for k, v in self.per_stage_verdict.items():
            if int(v) <= 0:
                problems.append(
                    f"OperationalSchedule: per_stage_verdict[{k!r}] must be > 0, got {v}")
        if self.reorient_every is not None and self.reorient_every <= 0:
            problems.append(
                f"OperationalSchedule: reorient_every must be > 0, got {self.reorient_every}")
        return problems

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
        if not self.per_stage_verdict:
            return ()
        return (TrainerSupportGap(
            axis="operational_schedule",
            requirement=f"per-stage verdict cadence {dict(self.per_stage_verdict)!r}",
            nearest_real_compilation=(
                f"--eval-every {self._effective_eval_every()} (densest requested cadence, "
                "globally — no verdict evidence lost; costs wall-clock)"),
            flag_proposal="--eval-every-per-stage <stage>:<N> (trainer build)",
            notes="verdict cost MEASURED +16 s/ep every other 25-ep block (§14 axis 6)"),)


@dataclass(frozen=True)
class TerminalSolve(ScheduleDisplay):
    """§16.1 (council draft 2026-07-07): the QUADRATIC BASIN FINISHER as a schedule primitive —
    once training enters the terminal basin, STOP iterating and SOLVE the local Gauss-Newton/
    Fisher quadratic of the through-R scorer loss (CG with HVPs; Morse lemma: exact in-chart).
    Replaces the measured quadratic-regime crawl (tau ~0.2%/25ep over ~400ep; slow cold-Muon
    recovery) with an explicit solve, verified at full n600 through the real verdict.

    STATUS: DESIGNED, NOT BUILT — a PREDICTION until the $0 probe measures (GN/CG from the
    mod32cap ep650-best vs the live run's own remaining crawl; owed per DAG FEED-07t/07u).
    Validity conditions (all three, checked at fire time): (a) local quadratic regime entered
    (power-law tail / gradient-norm plateau per §15.4 detectors); (b) partition topology STABLE
    (no island births pending — persistence diagram unchanged over the detector window);
    (c) frozen scorer + fixed levers (no schedule transitions remaining). Never fires mid-homotopy.

    Compiles to NO argv (never-invent-flags): the trainer has no solve stage; the nearest real
    compilation is "run --epochs to the end" and the solve runs as a post-run tool against the
    final checkpoint. The typed gap below is the council-visible trainer-build request."""

    method: str = "gn_cg"          # gauss-newton via CG on HVPs (the only designed method)
    verify_pairs: int = 600        # n600 verdict verification is MANDATORY (allergic-to-toys)

    def flags(self) -> dict:
        return {}  # no trainer support yet; real flags only

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.method != "gn_cg":
            problems.append(f"TerminalSolve: unknown method {self.method!r} (designed: 'gn_cg')")
        if self.verify_pairs != 600:
            problems.append(
                f"TerminalSolve: verify_pairs must be 600 (n600-or-not-evidence), got {self.verify_pairs}")
        return problems

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
        return (TrainerSupportGap(
            axis="terminal_solve",
            requirement=(f"quadratic basin finisher: GN/CG solve ({self.method}) at run end, "
                         f"verified at n{self.verify_pairs} through the real verdict"),
            nearest_real_compilation=("run --epochs to the scheduled end; the solve runs as a "
                                      "post-run tool against the final checkpoint (no argv)"),
            flag_proposal="--terminal-solve gn_cg (trainer build; gate on the $0 ep650 probe)",
            notes=("§16.1 PREDICTION until measured; Morse-lemma chart; conditions: in-basin + "
                   "topology-stable + no transitions remaining")),)


@dataclass(frozen=True)
class Curriculum(ScheduleDisplay):
    """The witness curriculum as a FIRST-CLASS DSL object (operator 2026-07-06 "we need schedule
    and curriculum in DSL as well"). Bundles the ordered homotopy of relaxations (``stages``) +
    the per-stage anneal schedules (render-partition ``temp``, ``hosc`` β, ``tau``) + the live PDE
    ``regularizers`` + the stage-transition ``reheat`` treatment + the stage HAND-OFF mode — what
    was previously scattered across raw ``base`` flags — into ONE controllable object.

    This is the ACT actuator the #247 costate controller reads/writes: with ``handoff="event"`` the
    stage boundary is a DECISION VARIABLE (the #315 nucleus-guarded CE→tau hand-off — hold tau until
    every scored class consolidates), not the hardcoded epoch the CE-didn't-plateau finding indicts.

    * ``handoff="fixed"`` (default): stages fire at their ``Stage.start_epoch`` (the current
      PR95-echo schedule).
    * ``handoff="event"``: emit ``--curriculum-event-triggered`` + ``--curriculum-nucleus-guard`` —
      the costate/plateau-driven hand-off.

    ``.flags()`` is the SINGLE emitter for the whole schedule; ``.validate()`` enforces stage
    ordering + the hand-off enum (structural, before any launch)."""

    stages: tuple[Stage, ...]
    temp: Anneal                                       # --softmax-temp-* render-partition anneal
    regularizers: tuple[Regularizer, ...] = ()
    hosc: HoscSchedule | None = None
    tau: float | None = None                           # --tau-softplus-tau (tau_softplus stage T)
    transition: Transition | None = None
    handoff: str = "fixed"                             # "fixed" | "event"
    curriculum_on: bool = True                         # the --curriculum master flag
    # §14 additions (task #339): levels-as-paths λ(t) + the operational schedule.
    level_paths: tuple[LevelPath, ...] = ()
    operational: OperationalSchedule | None = None
    # §16.1 addition: the quadratic basin finisher (DESIGNED-not-built; compiles to no argv;
    # always surfaces as a TrainerSupportGap until the $0 probe measures + the trainer build lands).
    terminal_solve: TerminalSolve | None = None

    # --- the per-owner flag sets (shared by flags() + the duplicate-emitter check) ---
    def _flag_owners(self) -> list[tuple[str, dict]]:
        owners: list[tuple[str, dict]] = []
        if self.curriculum_on:
            owners.append(("curriculum_on", {"--curriculum": True}))
        owners.append(("temp", self.temp.flags("--softmax-temp-start", "--softmax-temp-end")))
        for st in self.stages:
            owners.append((f"stage:{st.name}", st.flags()))
        for rg in self.regularizers:
            owners.append((f"regularizer:{rg.flag}", rg.flags()))
        if self.hosc is not None:
            owners.append(("hosc", self.hosc.flags()))
        if self.tau is not None:
            owners.append(("tau", {"--tau-softplus-tau": self.tau}))
        if self.transition is not None:
            owners.append(("transition", self.transition.flags()))
        for lp in self.level_paths:
            owners.append((f"level_path:{lp.quantity}", lp.flags()))
        if self.operational is not None:
            owners.append(("operational", self.operational.flags()))
        if self.handoff == "event":
            # #315 nucleus-guarded CE→tau hand-off (byte-identical to fixed when the guard never
            # fires; the schedule boundary becomes costate/plateau-driven, not a fixed epoch).
            owners.append(("handoff", {"--curriculum-event-triggered": True,
                                       "--curriculum-nucleus-guard": True}))
        return owners

    def flags(self) -> dict:
        f: dict = {}
        for _owner, fl in self._flag_owners():
            f.update(fl)
        return f

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
        """Every typed TrainerSupportGap this schedule's un-compilable intent produced
        (§14: the council reads THIS to see which schedule ideas need trainer builds)."""
        gaps: list[TrainerSupportGap] = []
        for st in self.stages:
            sg = getattr(st, "support_gaps", None)
            if callable(sg):
                gaps.extend(sg())
        for lp in self.level_paths:
            gaps.extend(lp.support_gaps())
        if self.operational is not None:
            gaps.extend(self.operational.support_gaps())
        if self.terminal_solve is not None:
            gaps.extend(self.terminal_solve.support_gaps())
        return tuple(gaps)

    def validate(self, surface_gaps: bool = False) -> list[str]:
        """Violations (empty == valid/launchable). ``surface_gaps=True`` ADDITIONALLY
        appends each TrainerSupportGap's describe() line — for council review surfaces;
        the default False keeps gaps NON-BLOCKING (the conservative compile emits real
        flags and is legal to launch)."""
        problems: list[str] = []
        if self.handoff not in ("fixed", "event"):
            problems.append(
                f"Curriculum.handoff must be 'fixed' or 'event', got {self.handoff!r}")
        # The BINDING order constraint is the trainer's (mirrors WitnessProgram.validate): the
        # tau_softplus stage must FORM the partition before the l7 sharpening stage → tau_start
        # < l7_start. We do NOT require all stage epochs monotonic: the Muon finisher may be
        # placed BEFORE l7 (the trainer allows `muon < l7` with a WARN — "operator freedom"),
        # and l7 is often PARKED above Muon/epochs (the sealed #205 demotes l7 to a no-op tail).
        by_flag = {st.start_epoch_flag: st.start_epoch
                   for st in self.stages if st.start_epoch_flag is not None}
        tau_s = by_flag.get("--tau-softplus-start-epoch")
        l7_s = by_flag.get("--l7-start-epoch")
        if tau_s is not None and l7_s is not None and not (0 < tau_s < l7_s):
            problems.append(
                f"Curriculum ordering: need 0 < tau_start ({tau_s}) < l7_start ({l7_s}) "
                "(the tau stage forms the partition before l7 sharpens it)")
        # §14 sub-object validation (fail-closed BEFORE compile).
        for st in self.stages:
            v = getattr(st, "validate", None)
            if callable(v):
                problems.extend(v())
        for lp in self.level_paths:
            problems.extend(lp.validate())
        if self.operational is not None:
            problems.extend(self.operational.validate())
        # DUPLICATE-EMITTER check (compose, don't duplicate — review attack surface): a flag
        # emitted by 2+ owners with UNEQUAL values is ambiguous (dict merge order would silently
        # pick a winner). Equal values are legal (e.g. a LevelPath agreeing with the temp Anneal
        # endpoints, or two stages both requesting the event controller).
        seen: dict[str, tuple[str, object]] = {}
        try:
            owners = self._flag_owners()
        except ValueError as exc:  # an uncompilable LevelPath — already reported above
            owners = []
            if not any("LevelPath" in p for p in problems):
                problems.append(str(exc))
        for owner, fl in owners:
            for k, v in fl.items():
                if k in seen and seen[k][1] != v:
                    problems.append(
                        f"DUPLICATE EMITTER: {k} from {seen[k][0]}={seen[k][1]!r} AND "
                        f"{owner}={v!r} — compose, don't duplicate (one owner per flag, "
                        "or equal values)")
                else:
                    seen.setdefault(k, (owner, v))
        if surface_gaps:
            problems.extend(g.describe() for g in self.support_gaps())
        return problems


# ---------------------------------------------------------------------------
# AUTO-DERIVED schedule-primitive registry (operator amendment 2026-07-07:
# consumers must track DSL evolution automatically). Mirrors lever_registry's
# auto-derivation — NO hand-typed list: membership = "a public dataclass in
# THIS module carrying the schedule surface" (ScheduleDisplay mixin, and/or a
# flags()/support_gaps() compiler). A new primitive added anywhere in this
# module auto-registers; the dashboard/costate consumers enumerate THIS.
# ---------------------------------------------------------------------------
def schedule_primitive_kinds() -> dict[str, type]:
    """{class_name: class} for every schedule/curriculum primitive in this module."""
    import dataclasses as _dc
    import inspect as _inspect
    import sys as _sys
    mod = _sys.modules[__name__]
    out: dict[str, type] = {}
    for name, obj in vars(mod).items():
        if name.startswith("_") or not _inspect.isclass(obj):
            continue
        if not _dc.is_dataclass(obj) or obj.__module__ != __name__:
            continue
        if (issubclass(obj, ScheduleDisplay)
                or callable(getattr(obj, "flags", None))
                or callable(getattr(obj, "support_gaps", None))):
            out[name] = obj
    return out


# ---------------------------------------------------------------------------
# Lever (an A/B toggle = a flag override set + optional epoch extension)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lever:
    """A named A/B lever: a set of flag overrides + optional extra epochs.

    Levers COMPOSE by merging their override dicts (later levers win on conflict),
    which is exactly theta* composition (binding the winning fragments).
    """

    name: str
    overrides: dict = field(default_factory=dict)
    epochs_delta: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Enforced-behavior clauses (preserve / contain / authority)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preserve(ScheduleDisplay):
    """PRESERVE: per-stage boundary ckpts + intra-stage cadence (<=25, binding)."""

    stage_boundaries: bool = True
    ckpt_every: int = 25

    def flags(self) -> dict:
        f = {"--ckpt-every": self.ckpt_every}
        # --stage-checkpoints is BooleanOptionalAction default True; emit explicitly.
        f["--stage-checkpoints"] = bool(self.stage_boundaries)
        return f


@dataclass(frozen=True)
class Contain:
    """CONTAIN: daemon-level blast-radius bounds (>=10GB floor, RSS cap)."""

    min_free_gb: float = 10.0
    projected_gb: float = 40.0
    rss_cap_mb: int = 90000
    walltime_cap_s: int = 288000


@dataclass(frozen=True)
class Authority:
    """AUTHORITY: the verdict contract. macOS-MLX/CPU is ADVISORY; only a
    byte-closed contest-CPU/CUDA exact row is a score. Recorded, asserted."""

    realized_through_R: bool = True
    numpy_fp32_reference: bool = True
    advisory_until_byte_closed: bool = True


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WitnessProgram(ScheduleDisplay):
    out_dir: str
    gt_cache: str
    epochs: int
    num_pairs: int
    temp: Anneal
    stages: tuple[Stage, ...]
    regularizers: tuple[Regularizer, ...]
    preserve: Preserve
    contain: Contain
    authority: Authority
    base: dict = field(default_factory=dict)  # substrate flags (arch, basis, chroma, ...)
    levers: tuple[Lever, ...] = ()
    resume_from: str | None = None
    mlx_device: str = "gpu"
    # the fixed gauge (a tac.witness_dsl.gauge.GaugeChoice); annotated lazily (object) so
    # curriculum_dsl never imports the gauge module at load time (no import cycle). Does NOT
    # affect flag_dict() — it is the chart-selection meta-layer ABOVE the trainer flags.
    gauge: object | None = None
    # FIRST-CLASS schedule/curriculum object (operator 2026-07-06). When set, it is the SINGLE
    # SoT for the schedule flags (temp anneal + stages + regularizers + hosc + tau + reheat +
    # hand-off); flag_dict() sources them from it and SKIPS the legacy temp/stages/regularizers
    # emission (no double-emit). None (default) => the legacy path => byte-identical for every
    # existing program (BASELINE / sealed_205 / openpilot_seeded_opening are unchanged).
    curriculum: "Curriculum | None" = None

    # --- composition ---------------------------------------------------------
    def with_lever(self, *levers: Lever, resume_from=_INHERIT,
                   out_dir: str | None = None) -> "WitnessProgram":
        """Return a new program with levers appended (theta* composition step).

        Epochs are extended by the sum of the levers' ``epochs_delta`` (e.g. a
        Muon finisher adds its window on top of the warm-start epoch).

        ``resume_from`` defaults to INHERIT (keep the base's); pass ``None`` to
        explicitly CLEAR it for a fresh run, or a path to override (DSL review M2)."""
        new_epochs = self.epochs + sum(lv.epochs_delta for lv in levers)
        new_resume = self.resume_from if resume_from is _INHERIT else resume_from
        return replace(
            self,
            levers=self.levers + tuple(levers),
            epochs=new_epochs,
            resume_from=new_resume,
            out_dir=out_dir if out_dir is not None else self.out_dir,
        )

    def with_gauge(self, gauge_choice=None, *, table=None,
                   warp=None, carrier=None, residual=None, pose=None,
                   movables=None, generation=None,
                   render_aa=None, lane_band=None, head_geometry=None) -> "WitnessProgram":
        """Fix the gauge for this program (the gauge-FIXING step of the 4-layer stack,
        FEED-ji). Returns a NEW program with a validated ``GaugeChoice`` attached; this one
        is UNMUTATED (pure composition, parallel to ``with_lever``). Composes:
        ``BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF, ...).with_lever(...)``.

        Pass a full ``GaugeChoice`` as the positional, OR per-component keyword overrides
        (unspecified components inherit this program's current gauge, else the canonical
        gauge). The #224 Option-B render/head components (``render_aa`` / ``lane_band`` /
        ``head_geometry``) are also selectable here; unspecified ones inherit the OFF /
        byte-identical default. ``validate`` raises ``GaugeViolation`` if any selected chart is
        non-compliant / non-deterministic per the cost ``table`` (BY CONSTRUCTION).

        Imported lazily so ``curriculum_dsl`` never imports ``gauge`` at module-load time
        (the gauge module imports THIS module — lazy import breaks the cycle).
        """
        from tac.witness_dsl.gauge import GaugeChoice, CANONICAL_GAUGE
        if gauge_choice is None:
            base_gauge = self.gauge if isinstance(self.gauge, GaugeChoice) else CANONICAL_GAUGE
            overrides = {k: v for k, v in dict(
                warp=warp, carrier=carrier, residual=residual,
                pose=pose, movables=movables, generation=generation,
                render_aa=render_aa, lane_band=lane_band, head_geometry=head_geometry).items()
                if v is not None}
            gauge_choice = replace(base_gauge, **overrides)
        elif not isinstance(gauge_choice, GaugeChoice):
            raise TypeError("with_gauge expects a GaugeChoice (or per-component keyword charts)")
        gauge_choice.validate(table)
        return replace(self, gauge=gauge_choice)

    # --- flag assembly -------------------------------------------------------
    def flag_dict(self) -> dict:
        f: dict = {}
        f.update(self.base)
        f["--num-pairs"] = self.num_pairs
        f["--epochs"] = self.epochs
        f["--gt-cache"] = self.gt_cache
        f["--out-dir"] = self.out_dir
        f["--mlx-device"] = self.mlx_device
        if self.curriculum is not None:
            # FIRST-CLASS curriculum object is the SoT for the whole schedule (temp + stages +
            # regularizers + hosc + tau + reheat + hand-off) — ONE emitter, no legacy double-emit.
            f.update(self.curriculum.flags())
        else:
            # legacy path (unchanged): temp anneal + stages + regularizers emitted directly.
            f.update(self.temp.flags("--softmax-temp-start", "--softmax-temp-end"))
            for st in self.stages:
                f.update(st.flags())
            for rg in self.regularizers:
                f.update(rg.flags())
        f.update(self.preserve.flags())
        if self.resume_from is not None:
            f["--resume-from"] = self.resume_from
        # levers LAST so they override (the A/B toggle wins)
        for lv in self.levers:
            f.update(lv.overrides)
        return f

    # --- validation (structural never-invent-flags + behavior clauses) -------
    def validate(self, trainer_path: Path | None = None) -> list[str]:
        """Return a list of violations (empty == valid)."""
        problems: list[str] = []
        real = real_trainer_flags(trainer_path)
        fd = self.flag_dict()
        for flag in fd:
            if flag not in real:
                problems.append(f"INVENTED FLAG (not in trainer argparse): {flag}")
        # C2 (review): a False on a store_true flag compiles to --no-X → argparse crash
        store_true = real_store_true_flags(trainer_path)
        for flag, val in fd.items():
            if val is False and flag in store_true:
                problems.append(
                    f"INVALID --no-{flag[2:]}: {flag} is store_true (no --no- form); "
                    "False would crash argparse at launch")
        # TYPE-COMPAT (review 2026-07-06, the EikonalViscosity-class static guard): a True/False
        # override is only legal on a boolean-action flag (store_true / BooleanOptionalAction —
        # these take NO value token); a non-boolean flag's override must be a non-bool value.
        # Either mismatch compiles to an argv the trainer argparse rejects AT LAUNCH (bare token
        # on a type=float flag → "expected one argument"; a value token after a boolean flag →
        # "unrecognized arguments"). Static half of the class-fix; the dynamic half is the
        # real-argparse parse test over every composable lever (test_lever_registry).
        boolean_flags = real_boolean_flags(trainer_path)
        for flag, val in fd.items():
            if flag not in real:
                continue  # already reported as INVENTED above
            if isinstance(val, bool) and flag not in boolean_flags:
                problems.append(
                    f"TYPE-INCOMPATIBLE OVERRIDE: {flag} is a value flag (not store_true/"
                    f"BooleanOptionalAction) but the override is {val!r}; compile would emit a "
                    "bare/negated token the trainer argparse rejects (expected one argument)")
            elif not isinstance(val, bool) and flag in boolean_flags:
                problems.append(
                    f"TYPE-INCOMPATIBLE OVERRIDE: {flag} is boolean-action (takes no value) but "
                    f"the override is {val!r}; compile would emit '{flag} {val}' which the "
                    "trainer argparse rejects (unrecognized arguments)")
        # C1 (review): DEAD ARM — resuming from a ckpt at/after epochs == zero gradient steps
        if self.resume_from is not None:
            try:
                import numpy as _np
                _p = Path(self.resume_from)
                if _p.exists():
                    _z = _np.load(_p, allow_pickle=True)
                    _ep = None
                    for _k in ("epoch", "__epoch", "__resume_epoch"):
                        if _k in _z.files:
                            _ep = int(_z[_k]); break
                    if _ep is not None and self.epochs <= _ep:
                        problems.append(
                            f"DEAD ARM: epochs={self.epochs} <= resume epoch {_ep} → "
                            "range(start,epochs) empty → ZERO gradient steps (give the "
                            "lever an epochs_delta window)")
            except Exception:
                pass  # validation must not hard-fail on a missing/odd ckpt
        # CURRICULUM ORDERING — surface the trainer's runtime assert at DSL-validate time so a
        # doomed config (a tau stage that silently never runs) is refused BEFORE any launch.
        # Aligned to the trainer's ACTUAL rule (L1 SEAL-review relax, 4bf533cab) and to
        # Curriculum.validate(): only 0 < tau_start < l7_start is required; l7_start > epochs is
        # the LEGITIMATE "l7 NEVER runs" form (l7 is a measured defect demoted from the default
        # curriculum — e.g. fresh_seeded parks l7 at epochs+1). The prior "<= epochs" clause here
        # was stale and refused that legitimate form.
        _curr = fd.get("--curriculum")
        if _curr is True or _curr == 1:
            _tau_s = fd.get("--tau-softplus-start-epoch")
            _l7_s = fd.get("--l7-start-epoch")
            if None not in (_tau_s, _l7_s) and not (0 < _tau_s < _l7_s):
                problems.append(
                    f"CURRICULUM ORDERING: need 0 < tau_start ({_tau_s}) < l7_start ({_l7_s}) "
                    "(the tau stage forms the partition before l7 sharpens it; trainer asserts "
                    "this — l7_start > epochs is allowed and means l7 NEVER runs)")
        # PRESERVE: ckpt cadence binding (<=25)
        if self.preserve.ckpt_every <= 0 or self.preserve.ckpt_every > 25:
            problems.append(
                f"PRESERVE violation: --ckpt-every={self.preserve.ckpt_every} (must be 1..25)")
        if not self.preserve.stage_boundaries:
            problems.append("PRESERVE violation: stage-boundary ckpts disabled")
        # CONTAIN: >=10GB floor binding
        if self.contain.min_free_gb < 10.0:
            problems.append(
                f"CONTAIN violation: min_free_gb={self.contain.min_free_gb} (<10GB floor)")
        # AUTHORITY: realized-through-R required for a trustworthy verdict
        if not self.authority.realized_through_R:
            problems.append("AUTHORITY violation: realized_through_R must be True")
        # FIRST-CLASS curriculum object (when set): stage-ordering + hand-off-enum structural check.
        if self.curriculum is not None:
            problems.extend(self.curriculum.validate())
            # §14 double-emitter check across the program: a schedule flag set BOTH in ``base``
            # AND by the curriculum object with different values is ambiguous (flag_dict emits
            # curriculum AFTER base, silently winning) — the curriculum object is the schedule
            # SoT; remove the flag from base or make the values agree.
            cur_flags = self.curriculum.flags()
            for k, v in cur_flags.items():
                if k in self.base and self.base[k] != v:
                    problems.append(
                        f"DOUBLE EMITTER: {k} set in base={self.base[k]!r} AND by the "
                        f"curriculum object={v!r} — the curriculum object is the schedule "
                        "SoT; remove it from base (or make the values agree)")
        return problems

    def support_gaps(self) -> tuple["TrainerSupportGap", ...]:
        """The program's typed TrainerSupportGaps (§14): delegated to the first-class
        curriculum object; () when the legacy (curriculum=None) path is in use."""
        if self.curriculum is None:
            return ()
        return self.curriculum.support_gaps()

    # --- compilation ---------------------------------------------------------
    def compile_trainer_argv(self, python: str = ".venv/bin/python") -> list[str]:
        argv = [python, TRAINER_REL]
        for flag, val in self.flag_dict().items():
            if val is True:
                argv.append(flag)
            elif val is False:
                # BooleanOptionalAction: emit --no-<name>
                argv.append(flag.replace("--", "--no-", 1))
            else:
                argv.extend([flag, str(val)])
        return argv

    def compile_daemon_argv(self, label: str, log: str,
                            python: str = ".venv/bin/python") -> list[str]:
        """Wrap the trainer in the canonical durable daemon + containment caps."""
        argv = [
            python, "tools/spawn_durable_daemon.py",
            "--label", label, "--log", log,
            "--projected-gb", str(self.contain.projected_gb),
            "--min-free-gb", str(self.contain.min_free_gb),
            "--rss-cap-mb", str(self.contain.rss_cap_mb),
            "--walltime-cap-s", str(self.contain.walltime_cap_s),
            "--",
        ]
        argv.extend(self.compile_trainer_argv(python=python))
        return argv


# ---------------------------------------------------------------------------
# BASELINE — the exact completed CE->tau->l7 run, expressed as a program.
# (round-trip target: BASELINE.flag_dict() reproduces the launched config.)
# ---------------------------------------------------------------------------
_CE_CKPT = ("experiments/results/levelset_amort_decoder_n200_20260627T143830Z/"
            "levelset_resume_stageCE_ep299.npz")
_L7_CKPT = ("experiments/results/levelset_l7_preserved_snapshots/"
            "levelset_resume_stageL7_ep1500.npz")

BASELINE = WitnessProgram(
    out_dir="experiments/results/levelset_amort_deconf_n200_taualone_20260627T194432Z",
    gt_cache="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz",
    epochs=1500,
    num_pairs=200,
    temp=Anneal(1.0, 0.05),
    stages=(
        Stage("CE", None, None),
        Stage("tau_softplus", "--tau-softplus-start-epoch", 300),
        Stage("l7_softplus", "--l7-start-epoch", 900),
    ),
    regularizers=(
        Regularizer("--eikonal-weight", 0.01),
        Regularizer("--length-weight", 0.001),
    ),
    preserve=Preserve(stage_boundaries=True, ckpt_every=25),
    contain=Contain(min_free_gb=10.0, projected_gb=40.0, rss_cap_mb=90000),
    authority=Authority(),
    resume_from=_CE_CKPT,
    base={
        "--render-h": 384, "--render-w": 512,
        "--hidden-dim": 96, "--mod-dim": 32,
        "--activation": "hosc", "--siren-init": True,
        "--curriculum": True,
        "--palette-anchor": True, "--self-orient": True, "--reorient-every": 50,
        "--freq-across": 32, "--n-dir-freqs": 2, "--freq-along": 4, "--max-bank-freq": 64,
        "--chroma": True,
        "--lane-edge-weight": 0, "--lane-edge-class": 1, "--lane-margin-target": 0.5,
        "--lane-edge-start-epoch": 300,
        "--w-seg": 100, "--w-pose": 1.0,
        "--ema-decay": 0.997, "--accum-pairs": 8, "--grad-clip": 1.0,
        "--verdict-pairs": 96, "--eval-every": 25,
        "--async-verdict": True,
    },
)


# ---------------------------------------------------------------------------
# Lever library (the A/B campaign, as composable DSL fragments)
# ---------------------------------------------------------------------------
def PoseDecouple(window: int = 100) -> Lever:  # noqa: N802 (DSL keyword) — A5
    """A5: drop pose from the loss (w-pose=0) to free decoder capacity for d_seg —
    a TRADE (d_pose worsens; pose is carried in-frame, NOT sidecar-able, per the
    byte-close finding). Carries a warm-start window (else dead-arm, review C1)."""
    return Lever("A5_pose_decouple", overrides={"--w-pose": 0.0}, epochs_delta=window,
                 notes="drop pose-loss to free d_seg capacity (trades d_pose up)")


def Muon(start_epoch: int, window: int = 100) -> Lever:  # noqa: N802 — A4
    """A4: Muon finisher from ``start_epoch`` for ``window`` epochs, with moments
    reset at the optimizer-stage transition and tau FROZEN at 0.05 (the run's
    final hard temperature) for apples-to-apples. muon-lr auto-derives 0.1*lr."""
    return Lever(
        "A4_muon",
        overrides={
            "--muon-start-epoch": start_epoch,
            "--stage-transition-reset-moments": True,
            "--softmax-temp-start": 0.05,  # freeze tau at the l7-end value
            "--softmax-temp-end": 0.05,
        },
        epochs_delta=window,
        notes="Muon finisher; is it the d_seg finisher (conditioning) or no?",
    )


def DirectionalBasis(weight: float = 0.5, start_epoch: int = 300,  # noqa: N802
                     window: int = 100) -> Lever:
    """Turn the lane-edge directional term ON (the completed run had weight 0).
    The all-class directional/tangent basis measured -48% d_seg earlier.
    Carries a warm-start window (else dead-arm, review C1).

    SISTER (FEED-07a): this is the directional LOSS-term lever (``--lane-edge-*``); the
    directional BASIS-frequency allocation lever (``--freq-across``/``--freq-along``/
    ``--n-dir-freqs``/``--self-orient``, the derived two-regime along-tangent law) is
    :func:`DirectionalBasisRebalance` — different flags, different mechanism, composable."""
    return Lever("directional_basis",
                 overrides={"--lane-edge-weight": weight,
                            "--lane-edge-start-epoch": start_epoch},
                 epochs_delta=window,
                 notes="all-class directional tangent basis (was OFF: weight 0)")


def TauFrozen(value: float = 0.05, window: int = 100) -> Lever:  # noqa: N802 — A1b isolation
    """A1b: freeze tau (start==end) to isolate an l7 effect from the tau anneal.

    MUST carry an ``epochs_delta`` (the warm-start window) or the arm runs ZERO
    gradient steps when resumed from an end-of-run ckpt (DSL review C1, 2026-06-28:
    epochs==resume_epoch → empty range → scientifically-dead arm)."""
    return Lever("A1b_tau_frozen",
                 overrides={"--softmax-temp-start": value, "--softmax-temp-end": value},
                 epochs_delta=window,
                 notes="freeze tau to isolate l7-loss vs tau-anneal (diff refutation)")


def SoftBoundary(beta: float = 2.0, window: int = 100) -> Lever:  # noqa: N802
    """Anti-aliased SOFT boundary (lower HOSC beta) — tests Signal's hypothesis that
    a soft edge carries sub-pixel boundary position through R better than a hard
    step (β→∞). Replaces the confounded constant-β≈16 'beta_steplim' arm (review H2)."""
    return Lever("soft_boundary",
                 overrides={"--hosc-beta": beta},
                 epochs_delta=window,
                 notes="soft anti-aliased edge (low beta) for sub-pixel R-survival")


def FiLMFix(per_layer: bool = True, concat_code: bool = True,  # noqa: N802 — LEVER-A
            rank_floor_weight: float = 0.0, rank_floor_target: float = 4.0,
            window: int = 100) -> Lever:
    """LEVER-A (FiLM-rank-fix): attack the MEASURED per-pair FiLM modulation participation-ratio
    collapse (3.34@CE -> 1.27@tau -> 1.19@l7; 91.8% of per-pair variation in ONE axis) that caps
    d_seg AND held-out amortization. Composes three default-OFF trainer routes:

      * ``per_layer`` -> ``--film-per-layer``: SEPARATE per-layer RESIDUAL FiLM (identity at init) =
        more INDEPENDENT multiplicative modulation routes.
      * ``concat_code`` -> ``--film-concat-code``: an ADDITIVE per-pair code-injection (folded concat;
        identity at init) = a NON-collapsing per-pair TRANSLATION route (what a moving lane needs).
      * ``rank_floor_weight`` > 0 -> ``--film-rank-floor-weight``/``--film-rank-floor-target``: a soft
        participation-ratio FLOOR penalty so the curriculum cannot funnel the modulation to rank-1.

    Emits ONLY flags that are turned on (store_true flags are never emitted False, per DSL review C2).
    Carries a warm-start ``window`` (else dead-arm when resumed at end-of-run, review C1)."""
    ov: dict = {}
    if per_layer:
        ov["--film-per-layer"] = True
    if concat_code:
        ov["--film-concat-code"] = True
    if rank_floor_weight > 0.0:
        ov["--film-rank-floor-weight"] = rank_floor_weight
        ov["--film-rank-floor-target"] = rank_floor_target
    return Lever("A_film_rank_fix", overrides=ov, epochs_delta=window,
                 notes="FiLM rank-fix: per-layer + concat-code + rank-floor (attacks PR collapse)")


def LanePrior(weight: float = 1.0, start_epoch: int = 300,  # noqa: N802 — LEVER-B
              lane_class: int = 1, radius: int = 4, target: float = 0.5,
              window: int = 100) -> Lever:
    """LEVER-B (thin-lane dropped-dash prior): up-weight the realized through-R seg margin hinge on
    THIN GT-lane structures the unweighted mean loss drops (MEASURED: 57% Road<->Lane confusion, PC0
    = Lane->Road DROP, 52.7% of GT-lane components wholesale-missed, miss-fraction monotone in dash
    size). A precomputed thin-lane weight map (local lane density in a (2r+1)^2 window) concentrates
    pressure on the thin dashes. Carries a warm-start ``window`` (else dead-arm, review C1).

    NOTE: this is the ``--lane-thin-*`` realized-margin prior; it is DISTINCT from the
    ``--lane-prior-phi1`` structured-init lane-SDF flag (a different mechanism)."""
    return Lever("B_lane_thin_prior",
                 overrides={"--lane-thin-weight": weight,
                            "--lane-thin-start-epoch": start_epoch,
                            "--lane-thin-class": lane_class,
                            "--lane-thin-radius": radius,
                            "--lane-thin-target": target},
                 epochs_delta=window,
                 notes="thin-lane dropped-dash prior (realized margin hinge weighted by thinness)")


def AnalyticLaneRenderBand(  # noqa: N802 — FEED-dv render-band lever
    softness: float = 1.0, dash_forward_max_m: float = 55.0,
    uncertainty_source: str = "witness", tau: float = 0.85, eps: float = 0.35,
    weight: float = 1.0, start_epoch: int = 300, window: int = 200,
) -> Lever:
    """FEED-dv (#203/#213/#215) analytic-lane RENDER-BAND: composite the analytic
    openpilot lane band OVER the witness render BEFORE R (the ``compose_fn`` hook), so
    the frozen SegNet reads the composited frame and the d_seg loss backprops into the
    witness. NON-NAIVE form (the naive band HURT +0.00082, sizing c3): AA-SDF coverage x
    RANGE-DEPENDENT dash gate (#215) x WITNESS-UNCERTAINTY mask (rides #141 margin) so the
    band paints ONLY where the witness ERASES the lane, killing the dash-gap FALSE-POSITIVE.

    Impl: ``tac.boundary_math.analytic_lane_render_band.make_lane_band_compose_fn`` passed
    to the trainer's ``render_fn`` (compose hook). MEASURED post-hoc n600: the levers take
    naive +0.00082 -> ~break-even; the NET-NEGATIVE win is realized by TRAINING WITH the
    band active (the witness re-adapts its boundaries; sizing VERDICT).

    PAIRED WITH the trainer wire-in (docs/analytic_lane_render_band_wire_in_spec.md): the
    8 ``--lane-band-*`` / ``--lane-render-band`` flags are LANDED in the trainer argparse
    and consumed by the compose wire-in (the #224 Option-B lane-band path)."""
    return Lever("FEED_dv_analytic_lane_render_band",
                 overrides={"--lane-render-band": True,
                            "--lane-band-softness": softness,
                            "--lane-band-dash-forward-max-m": dash_forward_max_m,
                            "--lane-band-uncertainty-source": uncertainty_source,
                            "--lane-band-tau": tau,
                            "--lane-band-eps": eps,
                            "--lane-band-weight": weight,
                            "--lane-band-start-epoch": start_epoch},
                 epochs_delta=window,
                 notes="analytic-lane render-band compose (AA-SDF x range-dash-gate x "
                       "witness-uncertainty); FP-killed non-naive form; realized THROUGH R")


def DashComb(comb_softness_m: float = 0.3, window: int = 0) -> Lever:  # noqa: N802 — #287
    """#287 EGO-PHASE DASH COMB — the cell-problem corrector of the dash-erasure
    homogenization law (``tac.canonical_equations.dash_erasure_homogenization_20260707``):
    replace the band's per-pair FITTED dash phase with a world-static max-plus comb —
    global (period T, duty, ego-scale) + per-slot world phase, transported by cumulative
    ego forward distance (phase-from-ξ, #215) — so sub-δ dash structure the coarse flow
    provably erases is supplied analytically, rule-118 FREE at decode (counted payload
    ~2-6 floats vs 1 fitted phase float per line per pair).

    §14 STAGE PLACEMENT: render-time corrector with NO curriculum stage of its own — when
    ON it is active from epoch 0 of the band's engagement window (it rewrites the
    precomputed band coverage priors at build time and takes effect the moment
    ``--lane-band-start-epoch`` opens the band gate). It also DISCHARGES the τ_end
    coupling rule ("do not anneal τ below the dash period unless a corrector is active").
    COMPOSES WITH (does not replace) ``AnalyticLaneRenderBand`` — it only modulates that
    lever's coverage; alone it is inert (``--lane-render-band`` off => no band to comb).

    Impl: ``tac.boundary_math.dash_comb.build_combed_lane_band_priors`` consumed by the
    levelset trainer's ``--lane-band-dash-comb`` wire-in (the #224 lane-band path)."""
    return Lever("n287_dash_comb",
                 overrides={"--lane-band-dash-comb": True,
                            "--lane-band-comb-softness-m": comb_softness_m},
                 epochs_delta=window,
                 notes="#287 ego-phase dash comb (homogenization corrector; world-static "
                       "period/duty/phase-from-xi replaces per-pair fitted dash phase; "
                       "render-time, active-from-ep0 of the band window, composes with "
                       "AnalyticLaneRenderBand)")


def StiefelW(window: int = 100) -> Lever:  # noqa: N802 — DM1a
    """DM1a (Stiefel-W): per-step project film.weight onto orthonormal columns (WᵀW=I) so W is an
    ISOMETRY => PR(M)=PR(cov(code)) to the projection's ~1e-2 residual (the byte-free root half-1 of the
    FiLM rank-collapse cure; design memo per_stage_fractal_optimizer §0/§4). store_true => emitted ONLY
    when on (never False, review C2). Carries a warm-start ``window`` (else dead-arm when resumed at
    end-of-run, C1)."""
    return Lever("DM1a_stiefel_w",
                 overrides={"--film-stiefel": True},
                 epochs_delta=window,
                 notes="Stiefel-orthonormal film.weight (WᵀW=I => PR(M)~PR(cov code) to ~1e-2 residual; "
                       "global-magnitude WD on W neutralized)")


def CodeSpectralEntropy(beta: float = 0.01, window: int = 100) -> Lever:  # noqa: N802 — DM1b
    """DM1b (code spectral-entropy): add the CAPACITY penalty -beta*log(PR(cov(code))) keeping all
    code directions live (the byte-free root half-2; design memo §0/§4). A value flag (not store_true);
    omitted when beta<=0 (off). Carries a warm-start ``window`` (else dead-arm when resumed, C1)."""
    ov: dict = {}
    if beta > 0.0:
        ov["--code-spectral-entropy-weight"] = beta
    return Lever("DM1b_code_spectral_entropy",
                 overrides=ov,
                 epochs_delta=window,
                 notes="spectral-entropy CAPACITY penalty on cov(code) (raises PR(cov code) => PR(M))")


def DM1Minimal(beta: float = 0.01, window: int = 100) -> tuple[Lever, Lever]:  # noqa: N802 — A3
    """DM1 minimal cure = Stiefel-W + code-spectral-entropy (design memo §4 the 80/20). Returns the
    two composable levers (use ``BASELINE.with_lever(*DM1Minimal())``); both halves target DIFFERENT
    params (W via projection, code via penalty) so they compose without double-counting (§3 routing).
    The per-stage moment-reset (the third minimal item) is the existing ``--stage-transition-reset-moments``
    (already wired); add ``Muon(...)`` or ``StiefelW(window=...)`` arms to engage it at a boundary.

    The warm-start ``window`` is carried ONCE (on the Stiefel lever); the entropy lever uses
    ``window=0`` so composing both extends epochs by ``window`` (not ``2*window``)."""
    return StiefelW(window=window), CodeSpectralEntropy(beta=beta, window=0)


def MarginSaliency(weight: float = 1.0, start_epoch: int = 900,  # noqa: N802 — LEVER (KKT waterfill)
                   tau: float = 0.5, target: float = 0.5, window: int = 100) -> Lever:
    """Margin-saliency hinge (KKT waterfill on margin-saliency, `boundary_routing.py`) engaged
    LATE (l7/Muon finetune; from-scratch margin starves the interior). Composes with ``UniWARD``
    (the texture mask) + ``DirectionalBasis`` (the curvelet basis) in the synergy map. Carries a
    warm-start ``window`` (else dead-arm when resumed at end-of-run, review C1)."""
    return Lever("margin_saliency",
                 overrides={"--margin-saliency-weight": weight,
                            "--margin-saliency-start-epoch": start_epoch,
                            "--margin-saliency-tau": tau,
                            "--margin-saliency-target": target},
                 epochs_delta=window,
                 notes="KKT-waterfill margin-saliency hinge (late finetune)")


def UniWARD(weight: float = 1.0, start_epoch: int = 900, beta: float = 4.0,  # noqa: N802 — LEVER-4
            tau: float = 0.5, target: float = 0.5, window: int = 100) -> Lever:
    """LEVER-4 (UniWARD inverse-steganalysis, Fridrich; BUILT + smoke-verified, `uniward_texture.py`
    `compute_texture_probability` + `uniward_delta.py`): margin-saliency with the UNIWARD texture
    DOWN-weight ``sal /= (1+beta*tex)`` (tex = stop-grad spatial-gradient energy of the realized
    frame) -> let error HIDE in textured (SegNet-undetectable) regions, CONCENTRATE correctness on
    the smooth boundary. On-theme: the contest IS inverse steganalysis. A LATE-STAGE (l7/Muon) A/B
    arm; composes with ``MarginSaliency`` + ``DirectionalBasis`` (the curvelet directional basis) in
    the synergy map. ``--margin-saliency-uniward`` is store_true -> emitted True ONLY (never False,
    review C2). Carries a warm-start ``window`` (else dead-arm, review C1)."""
    return Lever("LEVER4_uniward",
                 overrides={"--margin-saliency-weight": weight,
                            "--margin-saliency-start-epoch": start_epoch,
                            "--margin-saliency-uniward": True,
                            "--margin-saliency-uniward-beta": beta,
                            "--margin-saliency-tau": tau,
                            "--margin-saliency-target": target},
                 epochs_delta=window,
                 notes="UniWARD texture-masked margin-saliency (Fridrich; concentrate on smooth boundary)")


def WarpRealLumaFrame0(  # noqa: N802 — DSL constructor
    w_pose: float = 1.0, start_epoch: int = 0, window: int = 0,
) -> Lever:
    """POSE CARRIER B — warp-real-luma FRAME0 (``tac.boundary_math.warp_real_luma_frame0``).

    Engages the pose term (``--w-pose``, default 0.0 = pose-blind) so the SE(3)-twist
    residual trains to close d_pose from the deterministic warp floor (~2.6-10.5 at n600
    advisory) toward ~3.4e-5. The CARRIER ITSELF is a CODE wire-in (NOT a trainer flag):
    the parent routes the f0 render slot through
    ``WarpRealLumaFrame0Carrier.make_pair_render_dispatch(...)`` and passes it to
    ``make_loss_fn(render_fn=...)`` (even code_idx=f0 -> warp(gt_f0, xi) through R; odd=f1
    -> witness). frame0 is seg-free (upstream/modules.py:108) so this lever CANNOT disturb
    d_seg — it side-steps the W8 d_seg-vs-d_pose warp-scale crux. Byte cost = the per-pair
    6-DOF twist (~875 B/600 fp16, ~325 B low-rank r2; dual-use with the stored pose).

    Gauge: ``WarpGauge.SCREW_TWIST`` x ``PoseGauge.LOW_RANK`` (the twist IS the pose).
    Composes with every d_seg lever (orthogonal frame). MEASURED reference: FEED-lj/W7.
    ``w_pose`` is the pose-term weight; ``--pose-eps`` is left at the trainer default.
    """
    return Lever("warp_real_luma_frame0_pose_carrier",
                 overrides={"--w-pose": w_pose},
                 epochs_delta=window,
                 notes=("pose carrier B: warp-real-luma frame0 (SE(3)-twist ground-homography, seg-free); "
                        "render_fn code wire-in via make_pair_render_dispatch; --w-pose>0 trains the "
                        "rank-6 twist residual to d_pose~3.4e-5 (FEED-lj/W7; advisory pointer 0.19110)"))


def GroundFrameChart(  # noqa: N802 — DSL constructor
    ref_pair: int = 0,
    s_t: float = -0.003224707899359239,
    s_r: float = 0.0,
    pitch: float = -0.01,
) -> Lever:
    """GROUND-FRAME CHART (#194 / council draft §17.1) — evaluate the witness in ONE canonical frame.

    Pre-composes the per-pair witness INPUT coords with the cumulative ξ-homography
    (``tac.boundary_math.ground_frame_chart``; the FEED-ll stratified-warp math, bit-parity-pinned
    to the measured reach tool). A CHART CHANGE on the input coords, NOT a pixel warp: the field is
    still trained through R + the frozen scorer, so it does NOT inherit the #190 deterministic-render
    d_seg floor. One ξ (dual-use with the stored pose sidecar — rule-118 FREE, 0 new archive bytes)
    targets three measured residuals: temporal flicker (44 % lane spikes), the dash-comb phase home,
    and §15 pinning/zero-mobility.

    SCHEDULE (§14): STRUCTURAL — active from ep0 BY CONSTRUCTION when on (it is the input coordinate
    SYSTEM, not a weighted loss term; there is no level path λ(t) to anneal — constancy is the
    decision, declared here, not a silent default). INTERACTIONS (treatment-arm design notes, NOT
    silently inherited): (a) the chart changes the input coordinate DISTRIBUTION, so Fourier/
    Nyquist-derived bank constants (``--bank-*`` / ``--max-bank-freq``) may need re-derivation under
    the chart; (b) v0 is GROUND-plane-only and the trainer FAIL-CLOSES with ``--self-orient`` and
    ``--render-aa != none`` (coordinate-system consistency; the self-orient-in-chart-coords
    composition and per-class stratified routing via ``screw_blend`` are designed follow-ups);
    (c) byte-close/inflate must apply the same chart at decode (owed WITH the GO-gated A/B).
    Defaults = the MEASURED FEED-ll reach calibration (reach_n96.json fit). NEVER-FIRED until the
    operator-GO-gated n600 training A/B (§17.1 completion criterion for #194)."""
    return Lever("ground_frame_chart",
                 overrides={"--ground-frame-chart": True,
                            "--gfc-ref-pair": int(ref_pair),
                            "--gfc-s-t": float(s_t),
                            "--gfc-s-r": float(s_r),
                            "--gfc-pitch": float(pitch)},
                 notes=("#194/§17.1 ground-frame chart: witness input coords pre-composed with the "
                        "cumulative ξ-homography (FEED-ll math, chart change; trained through R => "
                        "no #190 floor); structural-from-ep0; rule-118 FREE from the stored pose "
                        "table; fail-closed with self-orient/render-aa in v0"))


# ---------------------------------------------------------------------------
# COMPUTE/SPEED levers (the gauge's non-curriculum config that compiles to trainer argv).
# These move WALL-CLOCK, not the witness math/bytes/verdict: CacheGtSkeleton is BIT-IDENTICAL (a
# constant-recompute elision, PROVEN by the #260 n=8 CPU A/B: EMA-shadow max_abs=0); MicroBatch is
# trajectory-affecting (batched fp reduction) but its accum-step grad == the serial mean-over-chunk
# within fp tol. Both compose with any curriculum lever; neither carries an epochs_delta (they are
# GLOBAL config, not a warm-start A/B stage). means != ends: SPEED buys nothing on S — only a
# byte-closed n600 exact row moves the 0.19110 pointer.
# ---------------------------------------------------------------------------
def CacheGtSkeleton() -> Lever:  # noqa: N802 — SPEED lever (#260, bit-identical)
    """SPEED (BIT-IDENTICAL): cache the CONSTANT per-pair GT soft-skeleton for the persistence loss
    (``--cache-gt-skeleton``). ``sg = soft_skeleton(gt)`` is epoch-invariant + gradient-free (it
    multiplies ``pred`` in the clDice ``tsens`` term), so precomputing it once per pair + reusing it
    every step is bit-identical (a materialized concrete constant == the inline recompute) while
    skipping ~half the clDice cost. No-op unless ``--persistence-loss-weight>0`` (the only consumer);
    skipped under ``--micro-batch-pairs>1`` (the serial ``total_loss_fn`` is the sole consumer).
    ``--cache-gt-skeleton`` is store_true -> emitted True ONLY (never False, review C2)."""
    return Lever("cache_gt_skeleton",
                 overrides={"--cache-gt-skeleton": True},
                 notes="speed (bit-identical): cache the constant GT soft-skeleton for the persistence loss")


def MicroBatch(pairs: int = 4) -> Lever:  # noqa: N802 — SPEED lever
    """SPEED (trajectory-affecting): batch ``pairs`` per frozen-scorer forward (``--micro-batch-pairs``).

    The single-pair EfficientNet-B2 SegNet / FastViT PoseNet forward under-utilizes the GPU; B>1
    renders + scores B pairs in ONE batched forward (~2-4x). Grads are weighted by pair count so the
    accum-step grad == the serial mean-over-chunk (NOT bit-identical: batched fp reduction order -> a
    short trajectory A/B validates it). Incompatible with ``--seed-islands`` (fail-closed at the
    trainer build). B=1 (the default) is the byte-identical serial path -> ``MicroBatch(1)`` emits it
    explicitly for an apples-to-apples A/B baseline. A value flag (never store_true)."""
    return Lever("micro_batch_pairs",
                 overrides={"--micro-batch-pairs": int(pairs)},
                 notes="speed lever: B pairs per batched scorer forward (trajectory-affecting, ~2-4x)")


# ---------------------------------------------------------------------------
# The FIXED, KNOWN OPENING of the from-scratch openpilot-seeded d_seg curriculum.
# (S0 seed -> S1 short-CE -> S2 tau_softplus). l7 + Muon are STACKED ADAPTIVELY by
# ``campaign.plan_adaptive_step`` off this opening's measured per-stage checkpoints.
# Deep-math anchors: FEED-bv (measured per-stage d_seg dirs), FEED-fs (separatrix
# seed), FEED-fz/-bu (reheat), anneal-memo (tau=0.3 == reachability floor), FEED-fi
# (Muon = spectral conditioner -> stacked, not fixed). DAG FEED-ln.
# ---------------------------------------------------------------------------
def openpilot_seeded_opening(  # noqa: N802 — DSL constructor
    out_dir: str,
    gt_cache: str,
    num_pairs: int = 200,
    *,
    ce_to: int = 300,
    tau_window: int = 300,
    tau: float = 0.3,
    w_pose: float = 0.0,
    rewarmup_epochs: int = 8,
    rewarmup_floor: float = 0.1,
    seed: int = 0,
    mlx_device: str = "gpu",
) -> WitnessProgram:
    """The FIXED, KNOWN OPENING (S0 seed -> S1 short-CE -> S2 tau_softplus) as ONE program.

    The curriculum is NOT fully fixed up front (operator riff 2026-06-29): we KNOW the
    opening; l7 + Muon are STACKED ADAPTIVELY from the MEASURED tau-stage d_seg trajectory
    off the per-stage checkpoints (see ``campaign.plan_adaptive_step`` / ``decide_next_stage``).

    S0 (pre-train seed, NOT an epoch stage): ``--structured-init`` + ``--lane-prior-phi1``
       inject the openpilot deg-3 centerline SIGNED-DISTANCE field into the phi1 (lane)
       channel of the structured-init pretrain target (FEED-fs separatrix residual 1.9e-5)
       -> the level-set homotopy STARTS in-basin AT the Road<->Lane separatrix. NTK view:
       the seed supplies the LOW-FREQUENCY lane structure free, so CE need not learn it.
    S1 CE [1, ce_to): SHORT confidence-calibration over all pixels. The seed gives the
       geometry (zero-level-set placement); CE only calibrates per-pixel argmax confidence
       -> SHORTENED (not eliminated). Measured CE descent 0.01045 -> 0.00643 (FEED-bv).
    S2 tau_softplus [ce_to, ce_to+tau_window): the PRIMARY measured d_seg drop
       (0.00643 -> 0.00396, FEED-bv). ``tau=0.3`` == the anneal-memo reachability floor
       Delta_min ~= 0.3 (the margin-RESONANCE T*=Delta for the *fixable* boundary flips;
       grad ∝ (1/T)e^{-Delta/T} peaks at T=Delta).

    l7 + Muon parked: ``--l7-start-epoch`` is set to ``epochs`` (no-op tail) so the opening
    is EXACTLY ce->tau and the trainer validator ``tau_start < l7_start <= epochs`` holds;
    the adaptive engine engages l7 (and then the Muon finisher) by warm-starting from the
    preserved tau checkpoint.

    REHEAT (FEED-fz BUILD 1 / FEED-bu, "different stages need different treatment") is ON at
    every transition: ``--stage-transition-rewarmup-epochs`` (LR floor->1x over the window,
    measured 0.1x/~8ep) + ``--stage-transition-reset-moments`` (zero stale AdamW 2nd-moments)
    -> the ce->tau boundary is stable BY CONSTRUCTION. smooth + lambda/sigma stages are
    SKIPPED (smooth measured to RAISE d_seg +6.8%; the trainer has no such curriculum stages
    so the skip is STRUCTURAL, not a flag).

    Pose rides the stored Quantizr-style sidecar -> ``w_pose=0`` (the witness's sole
    controllable job is d_seg). DETERMINISTIC-REPRODUCIBLE: single recorded ``--seed``;
    per-stage + periodic checkpoints ON (PRESERVE clause); EMA-shadow saved; ``--resume-from``
    compatible. FROM-SCRATCH: ``resume_from=None`` (the structured-init IS the seed, not a ckpt).
    """
    epochs = ce_to + tau_window
    base = dict(BASELINE.base)
    base.update({
        "--w-pose": w_pose,
        "--tau-softplus-tau": tau,
        "--structured-init": True,
        "--structured-init-include-lane": True,
        "--lane-prior-phi1": True,
        "--lane-prior-phi1-mode": "replace",
        "--lane-prior-phi1-dash-gate": True,
        "--stage-transition-rewarmup-epochs": rewarmup_epochs,
        "--stage-transition-rewarmup-floor": rewarmup_floor,
        "--stage-transition-rewarmup-shape": "linear",
        "--stage-transition-reset-moments": True,
        "--seed": seed,
    })
    return WitnessProgram(
        out_dir=out_dir,
        gt_cache=gt_cache,
        epochs=epochs,
        num_pairs=num_pairs,
        temp=Anneal(1.0, 0.05),  # RENDER-partition sharpness anneal (NOT the seg-surrogate tau);
        stages=(                  # frozen at 0.05 by the Muon finisher (FEED-fm FIX-2).
            Stage("CE", None, None),
            Stage("tau_softplus", "--tau-softplus-start-epoch", ce_to),
            # l7 PARKED at epochs (no-op tail); engaged adaptively via warm-start continuation.
            Stage("l7_softplus", "--l7-start-epoch", epochs),
        ),
        regularizers=(
            Regularizer("--eikonal-weight", 0.01),
            Regularizer("--length-weight", 0.001),
        ),
        preserve=Preserve(stage_boundaries=True, ckpt_every=25),
        contain=Contain(),
        authority=Authority(),
        base=base,
        resume_from=None,  # FROM SCRATCH (structured-init seed, not a checkpoint)
        mlx_device=mlx_device,
    )


# ---------------------------------------------------------------------------
# The #205 Phase-3 SEALED capstone program — the DSL leg of the triality.
# ---------------------------------------------------------------------------
# flag_dict()-managed flags (emitted via the WitnessProgram fields below, NOT via ``base``,
# so there is exactly one emitter per flag and no dict-overwrite mismatch).
# ---------------------------------------------------------------------------------------------
# #332 DE-ORPHANING WAVE: designed-but-orphaned levers folded into the DSL so the config can
# TOGGLE them by name instead of hand-adding flags at finalize time (the config-orphan confound,
# [[config_orphan_confound_permanent_fix_lever_registry_20260706]]). Each carries its DESIGNED
# intent (the "on" value argparse cannot supply). Flag names are REAL (validate() fail-closes on
# any invented flag). Generic tuning knobs (--adam-beta2 etc.) are levers WITH a swept param.
# ---------------------------------------------------------------------------------------------
def SeedIslandEased(window: int = 100) -> Lever:  # noqa: N802 — #323 LADDER island-birth
    """#323 LADDER island-birth: use the EASED per-class island targets at the seed/amplify
    sites — movable via SDF forward-Euler DILATION (proven 1-Lipschitz transfer) + lane via
    openpilot VP-TANGENT along-tangent widening (manifold-preserving; isotropic-of-a-curve is
    the NO-GO). Consumes ``tac.boundary_math.island_protection.eased_island_masks`` (wired
    705afea84). COMPOSES WITH the island-birth path (seed/amplify) — a MODIFIER, not standalone;
    default-OFF ⇒ byte-identical when unfired. DAG FEED + equation owed on first measured row."""
    return Lever("island_seed_eased",
                 overrides={"--seed-island-eased": True},
                 epochs_delta=window,
                 notes="#323 eased island targets: movable SDF-dilation + lane VP-tangent")


def SeedIslandBirth(window: int = 100) -> Lever:  # noqa: N802 — #224/#300 island-birth seed PAIR
    """#224 island seed + the #300 seed-absorption FIX, emitted as a PAIR (the trainer fail-closes
    ``--witness-alone-island-loss`` without ``--seed-islands``, so the DSL holds them together):
    early-seed the self-detected lane+movable islands into the SegNet-scored frame1 (accelerant,
    0 archive bytes, absent from EMA/blob/deploy) AND score the island-FORMATION levers
    (``--amplify-weight`` birth + persistence recall) on the WITNESS-ALONE render so the seed
    cannot starve the witness's own island gradient (the MEASURED #300 starvation: 71% of the
    paint-seed plateau = the 2 seeded classes at ~100% within-flip). SERIAL-accum only (fails
    closed under ``--micro-batch-pairs``); requires ``--w-seg > 0``. Compose modifiers over it:
    :func:`SeedIslandEased` (#323 eased per-class targets) + :func:`AmplifyIsland` (the birth
    term itself). DEFAULT-OFF trainer flags ⇒ byte-identical when unfired."""
    return Lever("island_seed_birth",
                 overrides={"--seed-islands": True,
                            "--witness-alone-island-loss": True},
                 epochs_delta=window,
                 notes="#224 island seed + #300 witness-alone island loss (the anti-starvation PAIR)")


def Mod32SegOnlyControlBase(window: int = 0) -> Lever:  # noqa: N802 — mod32cap control-base reproduction
    """The mod32cap CLEAN-BASELINE control config (run
    ``levelset_n600_witness_mod32cap_20260706T115554Z``, FEED-06u: the confound-fixed seg-only
    mod-32 council baseline) expressed as the DELTA over the launcher's ``proven_base`` — so a
    TREATMENT arm launched as ``--config proven_base --dsl-lever Mod32SegOnlyControlBase
    --dsl-lever <treatments...>`` matches the live control argv EXACTLY except the treatment
    levers (clean matched-epoch A/B attribution; the config generator stays the SoT instead of a
    hand-edited launch.sh).

    The 8 measured deltas (diff of the control launch.sh vs the proven_base emission, 2026-07-07):
    eikonal OFF (seg-only control) · freq-along 8 · annealed hosc 1→4 (fixed-high-β diverges,
    MEASURED) · l7 parked at 1001 (TRUE never) · lane paint-prior OFF (treatment arm, lever_ledger)
    · mod-dim 32 (the capacity GO-A) · n-dir-freqs 4 · verdict-pairs 0 (ALL 600, n600-scale rule).
    ``--lane-prior-phi1`` is BooleanOptionalAction ⇒ the ``False`` override renders
    ``--no-lane-prior-phi1`` (argparse-equal to the control's omission; the inert
    ``--lane-prior-phi1-mode/-dash-gate`` companions keep their defaults). window=0 = base-config
    reproduction, no epoch budget of its own."""
    return Lever(
        "mod32_seg_only_control_base",
        overrides={"--eikonal-weight": 0.0,
                   "--freq-along": 8.0,
                   "--hosc-beta": 1.0,
                   "--hosc-beta-end": 4.0,
                   "--hosc-beta-anneal": "linear",
                   "--l7-start-epoch": 1001,
                   "--lane-prior-phi1": False,
                   "--mod-dim": 32,
                   "--n-dir-freqs": 4,
                   "--verdict-pairs": 0},
        epochs_delta=window,
        notes=("mod32cap control base (FEED-06u) as proven_base deltas — treatment arms compose "
               "over it for clean A/B vs the live control"),
    )


def EventTriggeredCurriculum(window: int = 0) -> Lever:  # noqa: N802 — #315 derived-schedule flagship
    """#315 event-triggered CE→tau hand-off + per-class critical-nucleus guard: hold tau until
    every scored class consolidates (boundary re-anchor C1/C2 gate + nucleus predicate π₁≳5 +
    plateau), so a born-but-nascent island is not taxed by a fixed-epoch tau onset. Byte-identical
    to the fixed schedule when unfired (cap-ceiling ⇒ never hangs). The PAIR with island-birth is
    the only config where birth pressure pays (T3 symposium 20260706). Numeric guard params keep
    their trainer defaults; window=0 = schedule change, no epoch budget of its own."""
    return Lever("curriculum_event_triggered_nucleus_guard",
                 overrides={"--curriculum-event-triggered": True,
                            "--curriculum-nucleus-guard": True},
                 epochs_delta=window,
                 notes="#315 nucleus-guarded CE→tau hand-off; protects born islands at 0 d_seg cost")


def EikonalViscosity(eps: float = 0.05, adaptive: bool = True, window: int = 0) -> Lever:  # noqa: N802 — #316/#320 DE cure
    """#316/#320 viscous eikonal stabilization: the DE-derived adaptive-ε cure ε(t)=clamp(
    |c_a(t)|·√(ηλ_eik/8)(1+margin), floor, upper) that floors ε above the rising lower-CFL edge
    (the measured v5 ep110 re-entry cause). ``adaptive`` toggles the ε(t) law vs a fixed viscous
    ε; ``eps`` is the base/floor viscous ε (the adaptive law scales from it). Requires
    --eikonal-weight>0 to have effect.

    (review-fix CRITICAL) ``--eikonal-viscosity`` is ``type=float`` in the trainer argparse — a
    numeric ε, NOT a boolean. The prior ``{"--eikonal-viscosity": True}`` compiled to a BARE flag
    with no value, so ``--dsl-lever EikonalViscosity`` crashed the trainer at argparse
    (``expected one argument``) AFTER passing every launcher gate + spawning the daemon. Emit the
    float. (``--eikonal-viscosity-adaptive`` IS a store_true → True is correct there.)"""
    ov = {"--eikonal-viscosity": float(eps)}
    if adaptive:
        ov["--eikonal-viscosity-adaptive"] = True
    return Lever("eikonal_viscosity",
                 overrides=ov, epochs_delta=window,
                 notes="#316/#320 DE-derived viscous/adaptive-ε eikonal stabilization")


def AmplifyIsland(weight: float = 1.0, window: int = 100) -> Lever:  # noqa: N802 — island-amplify loss
    """Island-amplify loss: raise the rare-class island logit on the COSTATE/margin-GATED support
    (amplify only where the big-3 margin is preserved ⇒ n_big3→0 ⇒ net-positive by construction;
    UNIFORM amplification is the measured net-negative). ``--amplify-form``/``--amplify-margin-
    target`` keep trainer defaults; a nonzero weight turns the term on."""
    return Lever("island_amplify",
                 overrides={"--amplify-weight": weight},
                 epochs_delta=window,
                 notes="rare-class island-amplify on margin-gated support (net-positive by construction)")


def BoundaryDistance(weight: float = 1.0, window: int = 100) -> Lever:  # noqa: N802 — #301 loss-geometry
    """#301 loss-geometry: boundary-distance-weighted seg loss (concentrate pressure on the
    codim-1 separatrix annulus). Default-OFF (weight 0) in the baseline; a nonzero weight engages."""
    return Lever("boundary_distance",
                 overrides={"--boundary-distance-weight": weight},
                 epochs_delta=window,
                 notes="#301 boundary-distance-weighted seg loss (separatrix-annulus concentration)")


def SegFocalGamma(gamma: float = 2.0, window: int = 100) -> Lever:  # noqa: N802 — #301 focal calibration
    """#301 focal-γ seg loss: down-weight easy (high-margin) pixels, focus on the flip-prone
    boundary. γ calibrated $0-measured (#301); default 2.0 is the canonical focal exponent."""
    return Lever("seg_focal_gamma",
                 overrides={"--seg-focal-gamma": gamma},
                 epochs_delta=window,
                 notes="#301 focal-γ seg loss (down-weight easy pixels toward the flip boundary)")


def AdamBeta2(beta2: float = 0.99, window: int = 0) -> Lever:  # noqa: N802 — #222 β₂ optimizer lever
    """#222 Adam β₂ lever (arXiv 2603.02092): sweep β₂ (second-moment decay). The launch-gate
    guard requires β₁<√β₂; window=0 = optimizer-config change, no epoch budget of its own."""
    return Lever("adam_beta2",
                 overrides={"--adam-beta2": beta2},
                 epochs_delta=window,
                 notes="#222 Adam β₂ second-moment decay sweep (β₁<√β₂ guard)")


def DirectionalBasisRebalance(freq_across: int = 32, regime: str = "lane_offloaded",  # noqa: N802
                              window: int = 0) -> Lever:
    """FEED-07a arm-(A): DERIVED two-regime along-tangent frequency rebalance of the directional
    (anisotropic/curvelet) self-orient basis — a derivation, not a sweep (equations leg:
    ``tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707``, the imported
    ``freq_along_for_regime`` law).

    SISTER of :func:`DirectionalBasis` (the ``--lane-edge-*`` directional LOSS-term lever):
    different flags, different mechanism — this lever reallocates the BASIS frequencies the
    self-orient front-end spends; the two compose.

    * ``regime="lane_offloaded"`` (lane → the FREE rule-118 analytic band, MEASURED lane d_seg
      0.00087): the boundaries the witness still carries are C²-cartoon edges → Candès–Donoho
      parabolic scaling → ``freq_along = max(4, round(sqrt(freq_across)))`` (across=32 → along≈6).
    * ``regime="lane_carried"`` (the witness carries the dash comb): the dashes are ~25 cyc/unit
      along-edge TEXTURE violating the cartoon model; MEASURED 3.2× deficit at along=8 →
      ``freq_along = min(freq_across, round(8*3.2)) = 26``.

    Anchors: all-class directional basis −48% d_seg (~0 byte; DAG FEED 2026-06-25t) + the 3.2×
    along-tangent deficit (4-lens, FEED 2026-07-03); the regime-1 √-optimum is OWED to the
    next-run A/B (ASSUMED_AWAITING_VERIFICATION in the registered equation). The live mod32cap
    spends across=32/along=8 — BACKWARDS vs the measured deficit in the lane-carried regime.

    Emits REAL trainer flags only: ``--freq-across``/``--freq-along`` are ``type=float`` in the
    trainer argparse, ``--n-dir-freqs`` is ``type=int``, ``--self-orient`` is
    BooleanOptionalAction (default False; emitting True is a no-op override if a base already
    enables it). window=0 = basis-config change, no epoch budget of its own."""
    from tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707 import (
        freq_along_for_regime,
    )
    fa = int(freq_across)
    if fa <= 0:
        raise ValueError(f"DirectionalBasisRebalance: freq_across must be > 0, got {freq_across!r}")
    freq_along = freq_along_for_regime(fa, regime)  # fail-closed on an unknown regime
    return Lever(
        "FEED_07a_directional_basis_rebalance",
        overrides={"--self-orient": True,
                   "--n-dir-freqs": 4,
                   "--freq-across": float(fa),
                   "--freq-along": float(freq_along)},
        epochs_delta=window,
        notes=(f"FEED-07a derived two-regime along-tangent rebalance ({regime}: "
               f"across={fa} -> along={freq_along}; -48%/3.2x anchors; sqrt-optimum owed to A/B)"),
    )


def AACoverageRender(ss: int = 2, grid_h: int = 384, grid_w: int = 512,  # noqa: N802
                     window: int = 100) -> Lever:
    """FEED-07b lever #1 (#220): AA coverage render + grid≥384 — the gate's "#1 MEASURED islands
    lever" (own gate verdict). Supersampled anti-aliased rendering resolves the sub-pixel island
    coverage the base-grid point-sample misses.

    COMPOSE GUARD RESOLVED (#220 unblock, 2026-07-07): ``aa_sdf_observation_render`` now invokes
    ``compose_fn`` AFTER the box-downsample (at the BASE grid), so ``--render-aa supersample``
    COMPOSES with the base-grid compose levers ``--residual-mode`` / ``--seed-islands`` /
    ``--lane-render-band`` by construction (the trainer's ``_validate_aa_compose_compat`` guard is
    retained but currently accepts every tracked combination; ss=1 stays byte-identical). This
    lever therefore composes with SeedIslandEased(+seed path) / AnalyticLaneRenderBand as well as
    DirectionalBasisRebalance / MuonWarmStart. CAVEAT unchanged: the ORTHOGONAL --self-orient x
    supersample fine-dir-feats memory/wall-clock gate (``--aa-self-orient-fine-mode`` refuse
    default) still applies when composing over a --self-orient base."""
    if int(ss) < 1:
        raise ValueError(f"AACoverageRender: ss must be >= 1, got {ss!r}")
    return Lever(
        "FEED_07b_aa_coverage_render",
        overrides={"--render-aa": "supersample",
                   "--aa-supersample": int(ss),
                   "--render-h": int(grid_h),
                   "--render-w": int(grid_w)},
        epochs_delta=window,
        notes=("#220 AA coverage render (supersample) + grid>=384 — #1 measured islands lever; "
               "composes with residual-mode/seed-islands/lane-render-band (compose-after-"
               "downsample landed 2026-07-07); self-orient fine-mode gate still applies"),
    )


def StepNativeActivation(beta_start: float = 4.0, beta_end: float = 8.0,  # noqa: N802
                         anneal: str = "linear", window: int = 100) -> Lever:
    """FEED-07b lever #2 (#310, capstone lever #5 — in-code UNSWEPT): the step-native activation
    chart, deep-math L∞-at-edge optimal for the piecewise-constant argmax target (error confined
    to the flip band, O(1) params/edge, no Gibbs).

    REAL-FLAG ROUTE (never-invent-flags): this trainer's ``--activation`` choices are
    ``wire|hosc|relu`` — there is NO ``step_basis`` token — so the trainer's own named step-native
    lever is the hosc BETA-ANNEAL (FEED-fb): hosc = ``tanh(beta*sin(omega*u))`` and beta→∞ IS the
    step limit; ``--hosc-beta-end > --hosc-beta`` step-sharpens as the SDF partition forms.
    CAVEAT (MEASURED, DAG FEED 2026-06-25a + FEED-ly): FIXED high beta from scratch DIVERGES
    (tanh saturation → vanishing grad) — approach the step limit by ANNEAL, never start at it.
    A discrete ``step_basis`` activation choice remains a BUILD-NEEDED sister (no flag; do NOT
    emit one). The FINER++ bias-init sister is now BUILT (2026-07-07, #310 build half): compose
    :func:`FinerBiasInit` with this lever — the wide first-layer bias is the published fix for
    the measured fixed-β saturation-death this anneal works around.
    Equations leg: ``step_native_activation_edge_optimality_v1`` (#310 sweep = the owed anchor)."""
    if not (0.0 < float(beta_start) <= float(beta_end)):
        raise ValueError(
            f"StepNativeActivation: need 0 < beta_start <= beta_end (anneal TOWARD the step "
            f"limit), got start={beta_start!r} end={beta_end!r}")
    if anneal not in ("linear", "cosine"):
        raise ValueError(f"StepNativeActivation: anneal must be linear|cosine, got {anneal!r}")
    return Lever(
        "FEED_07b_step_native_activation",
        overrides={"--activation": "hosc",
                   "--hosc-beta": float(beta_start),
                   "--hosc-beta-end": float(beta_end),
                   "--hosc-beta-anneal": anneal},
        epochs_delta=window,
        notes=("#310 step-native chart via hosc beta-anneal (beta->inf = step limit; fixed high "
               "beta diverges — anneal only); L-inf-at-edge optimality; #310 sweep owed"),
    )


def MuonWarmStart(lr_final_frac: float = 0.1, window: int = 0) -> Lever:  # noqa: N802
    """FEED-07b lever #7 (#270/#272): Muon finisher warm-start + LR anneal — kills the MEASURED
    +8% cold-Muon transition spike. ``--muon-warm-start-momentum`` seeds the fresh Muon momentum
    from the outgoing AdamW first moment (direction transfer; Newton–Schulz re-normalizes);
    ``--muon-lr-final-frac`` cosine-decays the Muon-group LR to this fraction across the finisher
    span (a flat LR cannot self-reduce near the minimum — river-valley Muon 2606.21514). Both
    default-off trainer flags (byte-identical when unfired); fires at the l7→Muon switch (ep726
    of the live #205). window=0 = optimizer-config change, no epoch budget of its own."""
    f = float(lr_final_frac)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"MuonWarmStart: lr_final_frac must be in (0, 1], got {lr_final_frac!r}")
    return Lever(
        "FEED_07b_muon_warm_start",
        overrides={"--muon-warm-start-momentum": True,
                   "--muon-lr-final-frac": f},
        epochs_delta=window,
        notes=("#270/#272 Muon warm-start momentum + lr-final-frac anneal (measured +8% "
               "cold-Muon transient; fires at the l7->Muon switch)"),
    )


def PersistenceTopology(weight: float = 1.0, warmup_epochs: int = 0,  # noqa: N802
                        window: int = 100) -> Lever:
    """FEED-07b lever #3 (partial, #218/#224 byte-free head): soft-clDice + persistence-weighted
    island-RECALL term on the SHARED realized-through-R seg forward — births the finest-scale
    erasure tail (lane dashes / movable specks) the plain CE drops (error ∝ 1/persistence).
    ``--persistence-loss-weight`` 0 (trainer default) = branch skipped = byte-identical; a nonzero
    weight engages. ``--persistence-classes auto`` (trainer default kept) self-detects the
    thin/small erasure-tail classes from the cached GT argmax. The #218 LOSS-time logit-adjustment
    sister is now BUILT (2026-07-07): compose :func:`LogitAdjust` (``--logit-adjust-loss-tau``)."""
    ov: dict = {"--persistence-loss-weight": float(weight)}
    if int(warmup_epochs) > 0:
        ov["--persistence-warmup-epochs"] = int(warmup_epochs)
    return Lever(
        "FEED_07b_persistence_topology",
        overrides=ov, epochs_delta=window,
        notes=("#218/#224 soft-clDice + persistence island-recall (births the finest-scale "
               "erasure tail CE drops); loss-time logit-adjust sister = LogitAdjust (built)"),
    )


def MarginFieldHead(weight: float = 1.0, window: int = 100,  # noqa: N802
                    logit_adjust_per_class: bool = False,
                    logit_adjust_tau: float = 1.0) -> Lever:
    """FEED-07b lever #3 (partial, #218 facets 1b/3): realized through-R per-class margin-hinge
    head weight (``--margin-field-head-weight``; trainer default 0.0 = off = byte-identical).
    Composes with LEVER-3/4/B on the shared ``_signed`` margin field.

    ``logit_adjust_per_class=True`` additionally emits the #218 facet-3 pair
    ``--logit-adjust-per-class`` (+ ``--logit-adjust-tau``, its scale): the Menon DECODE-form
    offsets boost this head's per-class margin TARGET for the rare classes (fires only with this
    head's weight > 0 — the trainer reads the pair inside the mfh-target build). SISTER of the
    LOSS-time :func:`LogitAdjust` (``--logit-adjust-loss-tau``, a different surface); the two
    compose. store_true flag emitted True ONLY (review C2)."""
    ov: dict = {"--margin-field-head-weight": float(weight)}
    if logit_adjust_per_class:
        ov["--logit-adjust-per-class"] = True
        ov["--logit-adjust-tau"] = float(logit_adjust_tau)
    return Lever(
        "FEED_07b_margin_field_head",
        overrides=ov,
        epochs_delta=window,
        notes="#218 per-class realized-margin hinge head (shared _signed; byte-free head facet; "
              "optional facet-3 Menon margin-target boost)",
    )


def FinerBiasInit(k: float = 10.0, window: int = 0) -> Lever:  # noqa: N802 — #310 build half
    """FEED-07b lever #2's BUILD half (#310, 2026-07-07): FINER++ variable-periodic FIRST-LAYER
    bias init (FINER arXiv 2312.02434 / FINER++ arXiv 2407.19434) — ``in_proj.bias ~ U(-k, k)``
    over a WIDE range so each first-layer neuron selects its OWN frequency/phase of the periodic
    (hosc/wire) activation: the published fix for the MEASURED fixed-β hosc saturation-death
    (DAG FEED 2026-06-25a + FEED-ly — with all biases ~0 every neuron sits at the SAME point of
    ``tanh(β·sin)`` and saturates together as β rises).

    Trainer leg: ``--finer-bias-init`` (BooleanOptionalAction, default OFF = byte-identical:
    the ON path draws from a DEDICATED ``np.random.default_rng(seed+salt)`` stream, never the
    shared ``np.random``/``mx.random`` streams, so OFF makes ZERO draws and ON shifts no other
    seeded draw) + ``--finer-bias-k`` (the range; paper-range default 10.0). FROM-SCRATCH init
    lever (applied after siren-init, before structured-init; a ``--resume-from`` overwrites it —
    the trainer stamps ``applied:false``); fails closed on ``--activation relu`` (no period).
    COMPOSES with :func:`StepNativeActivation` (the β-anneal this init de-fragilizes) — the
    natural pair for the #310 sweep. ``window=0`` = init-config change, no epoch budget of its
    own. Equations leg: ``step_native_activation_edge_optimality_v1`` (the #310 sweep is the
    owed anchor for BOTH halves)."""
    if not (float(k) > 0.0):
        raise ValueError(f"FinerBiasInit: k must be > 0, got {k!r}")
    return Lever(
        "FEED_07b_finer_bias_init",
        overrides={"--finer-bias-init": True,
                   "--finer-bias-k": float(k)},
        epochs_delta=window,
        notes=("#310 FINER++ wide first-layer bias init (dedicated rng; fix for the measured "
               "fixed-beta hosc saturation-death); pairs with StepNativeActivation; #310 sweep owed"),
    )


def LogitAdjust(tau: float = 1.0, window: int = 100) -> Lever:  # noqa: N802 — #218 build half
    """FEED-07b lever #3's BUILD half (#218, 2026-07-07): class-prior LOGIT ADJUSTMENT on the
    TRAINING seg loss (Menon et al. 2021, arXiv 2007.07314 — the textbook ZERO-BYTE rare-class
    cure): the frozen-SegNet logits ``base_loss`` reads get ``logits_c += tau*log(prior_c)`` with
    priors = the GT class-area fractions from the cached L* (measured n600
    ~[0.232, 0.0059, 0.495, 0.0124, 0.254] — Lane/Movable log-priors −5.13/−4.39 vs Road −1.46,
    so under-predicting the two MEASURED un-born island classes costs more gradient; FEED-07c:
    lane 83.9% / movable 93.1% un-born). ``tau=1.0`` is the canonical Menon setting.

    BYTE-IDENTITY BOUNDARY (binding, documented at the trainer's ``_LogitAdjustSegAdapter``): the
    adjustment lives ONLY inside the training-loss adapter — the deployed/rendered argmax path
    (verdict CPU-torch SegNet, byte-close decode, inflate) reads RAW logits and is UNCHANGED; the
    witness WEIGHTS absorb the pressure. Trainer leg: ``--logit-adjust-loss-tau`` (default 0.0 =
    the loss adapter is the SAME object = byte-identical); fails closed with
    ``--micro-batch-pairs>1`` (not routed into the batched twin). SISTER (do not confuse): the
    #218 facet-3 pair ``--logit-adjust-per-class`` + ``--logit-adjust-tau`` boost the
    MARGIN-FIELD-HEAD per-class TARGET (see :func:`MarginFieldHead`); the two compose. Composes
    with :func:`SegFocalGamma` (logit-adjusted-focal) + :func:`PersistenceTopology`. Carries a
    warm-start ``window`` (else dead-arm on resume, DSL review C1). Equations leg:
    ``logit_adjustment_class_prior_law_v1`` (#218 A/B = the owed anchor)."""
    _t = float(tau)
    if _t == 0.0 or _t != _t or _t in (float("inf"), float("-inf")):
        raise ValueError(f"LogitAdjust: tau must be a nonzero finite float (0.0 = OFF), got {tau!r}")
    return Lever(
        "FEED_07b_logit_adjust_loss",
        overrides={"--logit-adjust-loss-tau": float(tau)},
        epochs_delta=window,
        notes=("#218 Menon loss-time logit adjustment (zero-byte rare-class cure; training-loss "
               "surface only, deployed argmax unchanged); #218 A/B owed"),
    )


_SEALED_205_MANAGED_FLAGS = frozenset({
    "--num-pairs", "--epochs", "--gt-cache", "--out-dir", "--mlx-device",
    "--softmax-temp-start", "--softmax-temp-end",
    "--tau-softplus-start-epoch", "--l7-start-epoch", "--muon-start-epoch",
    "--eikonal-weight", "--length-weight", "--ckpt-every", "--stage-checkpoints",
})

# The schedule/curriculum flags a first-class Curriculum object OWNS (operator 2026-07-06 "we
# need schedule and curriculum in DSL as well"). Pulled OUT of ``base`` in
# ``sealed_205_program`` when ``as_curriculum=True`` so there is exactly one schedule emitter.
_SEALED_205_CURRICULUM_FLAGS = frozenset({
    "--curriculum",
    "--hosc-beta", "--hosc-beta-end", "--hosc-beta-anneal", "--hosc-omega",
    "--tau-softplus-tau",
    "--stage-transition-rewarmup-epochs", "--stage-transition-rewarmup-floor",
    "--stage-transition-rewarmup-shape", "--stage-transition-reset-moments",
})


def sealed_205_curriculum(cfg, *, handoff: str = "fixed") -> Curriculum:
    """The #205 SEALED schedule as a FIRST-CLASS :class:`Curriculum` object (operator 2026-07-06).

    Values are pulled from ``tac.witness_autoconfig.derive_sealed_205_config`` (``cfg``, the SAME
    SoT the byte-exact gate leg uses) so the curriculum leg provably AGREES with the sealed program:
    the ordered stages (CE→tau→l7-parked→Muon), the render-partition ``temp`` anneal, the ``hosc`` β
    anneal, the ``tau_softplus`` temperature, the live PDE ``regularizers``, and the stage-transition
    ``reheat``. ``handoff="event"`` swaps the fixed CE→tau epoch for the #315 nucleus-guarded hand-off
    (the CE-didn't-plateau fix; byte-identical to fixed until the guard fires).

    Use it via the ``WitnessProgram.curriculum`` field:
        ``replace(sealed_205_program(out_dir), curriculum=sealed_205_curriculum(cfg))``
    — ``flag_dict()`` then sources the whole schedule from this ONE object.
    """
    # Read the schedule VALUES from the emitted trainer-flag dict — the SAME SoT the sealed program
    # pulls from (``cfg.to_trainer_flags``), NOT the ``proven_base`` dict (which does not carry
    # hosc-beta-end / hosc-beta-anneal / the stage-transition flags). out_dir does not affect any
    # schedule flag, so a placeholder is fine. A store_true flag emits None (bare) → reset=True.
    tf = dict(cfg.to_trainer_flags("_"))
    return Curriculum(
        stages=(
            Stage("CE", None, None),
            Stage("tau_softplus", "--tau-softplus-start-epoch", cfg.tau_softplus_start_epoch),
            Stage("l7_softplus", "--l7-start-epoch", cfg.l7_start_epoch),
            Stage("muon", "--muon-start-epoch", cfg.muon_start_epoch),
        ),
        temp=Anneal(tf["--softmax-temp-start"], tf["--softmax-temp-end"]),
        regularizers=(
            Regularizer("--eikonal-weight", tf["--eikonal-weight"]),
            Regularizer("--length-weight", tf["--length-weight"]),
        ),
        hosc=HoscSchedule(
            beta_start=tf["--hosc-beta"], beta_end=tf["--hosc-beta-end"],
            shape=tf["--hosc-beta-anneal"], omega=tf["--hosc-omega"],
        ),
        tau=tf["--tau-softplus-tau"],
        transition=Transition(
            rewarmup_epochs=tf["--stage-transition-rewarmup-epochs"],
            rewarmup_floor=tf["--stage-transition-rewarmup-floor"],
            rewarmup_shape=tf["--stage-transition-rewarmup-shape"],
            # store_true flag: present (value None) ⇒ True; absent ⇒ False.
            reset_moments=("--stage-transition-reset-moments" in tf),
        ),
        handoff=handoff,
    )


#: the schedule/curriculum flags the DSL Curriculum object OWNS — the subset of the trainer argv that
#: MUST be sourced from the first-class :class:`Curriculum` object, not hand-set independently (#334).
CURRICULUM_OWNED_FLAGS = frozenset({
    "--curriculum", "--tau-softplus-start-epoch", "--tau-softplus-tau",
    "--l7-start-epoch", "--muon-start-epoch",
    "--softmax-temp-start", "--softmax-temp-end",
    "--hosc-beta", "--hosc-beta-end", "--hosc-beta-anneal", "--hosc-omega",
    "--eikonal-weight", "--length-weight",
    "--stage-transition-rewarmup-epochs", "--stage-transition-rewarmup-floor",
    "--stage-transition-rewarmup-shape", "--stage-transition-reset-moments",
})


def verify_schedule_consistency(cfg, *, handoff: str = "fixed") -> list[str]:
    """#334 consistency gate: the first-class :class:`Curriculum` object's compiled schedule flags
    are compared against what the autoconfig emits for the curriculum-owned flags. Returns a list of
    disagreements (empty == consistent).

    HONEST COVERAGE (do NOT over-read this gate). The genuine second-source binding is exactly the
    THREE stage-epoch fields — ``--tau-softplus-start-epoch`` / ``--l7-start-epoch`` /
    ``--muon-start-epoch``. For those, :func:`sealed_205_curriculum` reads ``cfg.<attr>`` directly
    (``cfg.tau_softplus_start_epoch`` etc.) while the emission leg reads ``cfg.to_trainer_flags`` — two
    independent reads of ``cfg``, so if they drift this gate catches it. That is a REAL config-orphan
    check for the stage boundaries.

    The OTHER ~12 curriculum-owned flags (``--softmax-temp-*``, ``--hosc-*``, ``--eikonal-weight``,
    ``--length-weight``, ``--tau-softplus-tau``, ``--stage-transition-*``, ``--curriculum``) are a
    STRUCTURAL ECHO, not a second source: :func:`sealed_205_curriculum` builds the object FROM the
    same ``cfg.to_trainer_flags("_")`` dict that the emission leg reads, so both legs share one source
    and this gate CANNOT detect drift in them (it is tautological for those flags — they compare a
    value against itself). Do not treat a green result as proof those flags are set in only one place.

    The deeper refactor — the launcher EMITTING the whole schedule by calling the object, which would
    make ALL curriculum-owned flags a genuine single-source — is deferred to protect the sealed
    byte-identity gates. Until then, only the 3 stage-epoch fields are truly bound here.

    ``cfg`` is a ``tac.witness_autoconfig`` config with ``.to_trainer_flags`` +
    ``.tau_softplus_start_epoch`` etc. Bare-boolean flags (emitted as value ``None`` in the argv tuple
    convention) match the object's ``True``. Event-mode-only flags are exempt under ``handoff="fixed"``.
    """
    curr = sealed_205_curriculum(cfg, handoff=handoff)
    obj = curr.flags()
    emitted = dict(cfg.to_trainer_flags("_"))
    problems: list[str] = []
    _event_only = {"--curriculum-event-triggered", "--curriculum-nucleus-guard"}
    for flag, oval in obj.items():
        if flag in _event_only:
            continue  # handoff-mode flags, not part of the flat sealed schedule
        if flag not in CURRICULUM_OWNED_FLAGS:
            continue  # only enforce the curriculum-owned subset
        if flag not in emitted:
            problems.append(f"{flag}: Curriculum object emits {oval!r} but autoconfig does not")
            continue
        eval_ = emitted[flag]
        # bare-boolean: object True == argv-tuple None (present, valueless)
        if oval is True and eval_ in (None, True):
            continue
        if str(oval) != str(eval_):
            problems.append(f"{flag}: Curriculum object={oval!r} != autoconfig={eval_!r}")
    return problems


def sealed_205_program(  # noqa: N802 — DSL constructor
    out_dir: str,
    gt_cache: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    num_pairs: int = 600,
    epochs: int = 1000,
) -> WitnessProgram:
    """The **#205 Phase-3 SEALED capstone config** as a :class:`WitnessProgram` — the DSL leg of
    the DAG↔DSL↔equations triality (nexus §5c; the calibrated OPTIMAL-CONTROL schedule expressed
    as a program the campaign engine can EXTEND/ADVANCE/ROLLBACK via ``decide_next_stage`` #188).

    SINGLE SOURCE OF TRUTH: the flag VALUES are pulled from
    ``tac.witness_autoconfig.derive_sealed_205_config`` (the byte-exact gate leg) so the two legs
    provably AGREE — the curriculum schedule (CE→tau→l7-parked→Muon @726) is expressed via
    ``Stage`` objects, the render-partition anneal via ``Anneal``, the live PDE regularizers via
    ``Regularizer``, the PRESERVE cadence via ``Preserve``, and every remaining lever/substrate flag
    flows through ``base``. ``validate()`` re-checks every flag against the trainer argparse
    (never-invent-flags) + the curriculum-ordering / preserve / contain / authority clauses.

    NOTE: the DSL compiles flags in ``flag_dict`` INSERTION order (its own canonical order), which is
    NOT the hand-authored §7 token order — the BYTE-IDENTICAL-to-§7 gate is the ``witness_autoconfig``
    ``sealed_205`` launcher path. This program is the SYMBOLIC leg (same flag SET + values, campaign-
    engine-operable); the ``test_sealed_205_canonical_config`` cross-check asserts the two agree.

    means != ends: a MEANS (a launch program). Only a byte-closed n600 exact row < 0.19110 moves
    the pointer.
    """
    from tac import witness_autoconfig as _wac  # lazy: no module-load cycle (wac imports numpy only)

    cfg = _wac.derive_sealed_205_config(gt_cache, num_pairs=num_pairs, epochs=epochs)
    pb = cfg.proven_base
    base: dict = {}
    for flag, val in cfg.to_trainer_flags(out_dir):
        if flag in _SEALED_205_MANAGED_FLAGS:
            continue
        base[flag] = True if val is None else val  # DSL convention: bare flag == True
    return WitnessProgram(
        out_dir=out_dir,
        gt_cache=cfg.gt_cache,
        epochs=cfg.epochs,
        num_pairs=cfg.num_pairs,
        temp=Anneal(pb["softmax_temp_start"], pb["softmax_temp_end"]),
        stages=(
            Stage("CE", None, None),
            Stage("tau_softplus", "--tau-softplus-start-epoch", cfg.tau_softplus_start_epoch),
            # l7 DEMOTED to epochs (measured L∞-sharpening defect; parks as a <=1-ep no-op tail).
            Stage("l7_softplus", "--l7-start-epoch", cfg.l7_start_epoch),
            # Muon finisher (spectral conditioner) — the LAST stage of the annealing flow.
            Stage("muon", "--muon-start-epoch", cfg.muon_start_epoch),
        ),
        regularizers=(
            Regularizer("--eikonal-weight", pb["eikonal_weight"]),
            Regularizer("--length-weight", pb["length_weight"]),
        ),
        preserve=Preserve(stage_boundaries=True, ckpt_every=pb["ckpt_every"]),
        contain=Contain(),
        authority=Authority(),
        base=base,
        resume_from=None,  # FROM SCRATCH (structured-init + lane-prior seed, not a checkpoint)
        mlx_device=cfg.proven_base["mlx_device"],
    )
