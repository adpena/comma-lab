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

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from tac.witness_dsl.basis_control import normalize_basis_family

if TYPE_CHECKING:
    from tac.witness_dsl.integer_plane_emitter_policy import IntegerPlaneEmitterPolicy
    from tac.witness_dsl.yhat_native_generator_policy import YhatNativeGeneratorPolicy

_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_REL = "experiments/train_levelset_witness_realized_through_R_mlx.py"
TRAINER_PATH = _REPO_ROOT / TRAINER_REL
# EXPLICIT two-trainer binding for the package-wide registry scan (ddm_lr2, 2026-08-03).
# This module's levers legitimately span the levelset entry point AND the base it imports its
# primitives from (MEASURED: 35 of its flags exist ONLY on the base), so it is one of the rare
# genuine multi-trainer modules. It previously relied on the registry's silent default, which
# happens to resolve to exactly this pair — but "correct by default" and "correct by intent"
# were indistinguishable to every reader, and that indistinguishability is what let three
# TR1-targeted modules be graded against the retired vehicle unnoticed. Stating it changes NO
# behaviour (same two paths, verified by test); it makes the binding auditable.
TRAINER_RELPATHS = (
    "experiments/train_levelset_witness_realized_through_R_mlx.py",
    "experiments/train_witness_realized_through_R_mlx.py",
)


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
        if (isinstance(node, _ast.Assign) and any(
                isinstance(t, _ast.Name) and t.id == "ap" for t in node.targets)) or (isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call)
              and isinstance(node.value.func, _ast.Attribute)
              and node.value.func.attr == "add_argument"
              and isinstance(node.value.func.value, _ast.Name)
              and node.value.func.value.id == "ap"):
            stmts.append(node)
    stmts.sort(key=lambda n: n.lineno)  # ast.walk order is not source order
    ns: dict = {
        "argparse": _argparse,
        "Path": Path,
        "normalize_basis_family": normalize_basis_family,
    }
    exec(compile(_ast.Module(body=stmts, type_ignores=[]), str(path), "exec"), ns)
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


def schedule_epoch_budget_violations(
    flags: Mapping[str, object] | Iterable[tuple[str, object]],
    trainer_path: Path | None = None,
) -> list[str]:
    """Return enabled-curriculum stage caps that exceed the configured epoch budget.

    The trainer's real argparse parser supplies defaults, so an omitted schedule
    flag cannot hide an out-of-budget default.  Emitted ``--no-*`` BooleanOptional
    tokens are normalized back to their canonical flag.  ``--warm-start-epoch`` is
    resume metadata, not a curriculum stage.  With curriculum explicitly disabled
    the curriculum-family law is vacuous; independent controllers still own their
    trainer guards and a single-stage config must disable/strip those separately.

    This is a config/boot-runnability invariant only.  It does not claim training
    quality, score movement, archive closure, or promotion authority.
    """

    parser = build_real_trainer_parser(trainer_path)
    action_by_option = {
        option: action
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    effective: dict[str, object] = {}
    canonical_for_action: dict[int, str] = {}
    for action in parser._actions:
        canonical = next(
            (opt for opt in action.option_strings
             if opt.startswith("--") and not opt.startswith("--no-")),
            None,
        )
        if canonical is None:
            continue
        canonical_for_action[id(action)] = canonical
        effective[canonical] = action.default

    mapping_input = isinstance(flags, Mapping)
    items = flags.items() if mapping_input else flags
    for option, raw_value in items:
        option = str(option)
        action = action_by_option.get(option)
        if action is None:
            # Unknown flags are reported by the existing never-invent-flags gate.
            continue
        canonical = canonical_for_action[id(action)]
        if option.startswith("--no-"):
            value: object = False
        elif raw_value is None and not mapping_input:
            # Bare emitted options are boolean actions in this CLI.
            value = True
        else:
            value = raw_value
        effective[canonical] = value

    curriculum = effective.get("--curriculum", False)
    if isinstance(curriculum, str):
        curriculum = curriculum.strip().lower() not in {"", "0", "false", "no", "off"}
    if not bool(curriculum):
        return []

    try:
        epochs = int(effective["--epochs"])
    except (KeyError, TypeError, ValueError):
        return [
            "CURRICULUM EPOCH-BUDGET FEASIBILITY: enabled curriculum has no valid "
            "--epochs value; disable curriculum for a true single-stage program or "
            "compile a feasible schedule"
        ]

    offenders: list[tuple[str, int]] = []
    for flag, value in effective.items():
        if flag == "--warm-start-epoch" or not flag.endswith("-start-epoch"):
            continue
        if value is None:
            continue
        try:
            start_epoch = int(value)
        except (TypeError, ValueError):
            continue  # real argparse/type validation owns malformed values
        if flag == "--l7-start-epoch" and start_epoch == epochs + 1:
            # (c2 adversarial review 2026-07-16) l7_start == epochs + 1 is the CANONICAL "l7
            # NEVER runs" parking form (L1 SEAL-review relax 4bf533cab; l7 is a MEASURED DEFECT
            # demoted from the default curriculum — the mod32cap config of record AND
            # fresh_seeded both park it at 1001 with epochs=1000, "TRUE never"). The trainer's
            # epoch loop is range(start, epochs+1) INCLUSIVE, so l7_start == epochs RUNS l7 on
            # the final epoch (the trainer's own documented off-by-one at ~L15646) — this gate
            # refusing the epochs+1 form is what squeezed c2_surgical_warm into that off-by-one.
            # NARROW exemption: exactly epochs+1 (the deliberate parking convention); any OTHER
            # value past epochs — and every other stage flag parked past epochs — is still a
            # dead-stage config bug and stays refused below.
            continue
        if start_epoch > epochs:
            offenders.append((flag, start_epoch))
    if not offenders:
        return []
    offenders.sort(key=lambda row: row[0])
    detail = ", ".join(f"{flag}={value}" for flag, value in offenders)
    return [
        "CURRICULUM EPOCH-BUDGET FEASIBILITY: enabled curriculum with "
        f"epochs={epochs} cannot engage every configured stage/cap: {detail}. "
        "Disable curriculum for a true single-stage program or use a feasible "
        "schedule whose start epochs are <= epochs. verdict_scope=config/boot-runnability-only"
    ]


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
    from dataclasses import fields as _fields
    from dataclasses import is_dataclass as _is_dc
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
        for fld in _fields(self):  # ty: ignore[invalid-argument-type]
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
            except Exception as exc:
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


def Freeze(value: float) -> Anneal:
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
class MorseContinuationSchedule(ScheduleDisplay):
    """#302 action-native replacement for the three inherited PR95 scalars.

    Construct this object through :func:`WitnessNativeMorseContinuationSchedule`.
    The schedule stores the resolved equation inputs and emits only real trainer
    flags.  In particular Muon's step is a checkpoint-local Fisher trust radius,
    not ``0.1 * Adam lr``; the discontinuous L7 loss is retired under unified-tau
    while its R-safe boundary remains explicit for telemetry/resume compatibility.
    """

    muon_lr: float
    l7_mult: float
    l7_threshold: float
    trust_region_kl: float
    fisher_curvature_upper: float
    m_safe: float
    equation_id: str = "witness_native_morse_continuation_v1"
    margin_law_manifest: dict = field(default_factory=dict, compare=False, repr=False)

    def flags(self) -> dict:
        return {
            "--muon-lr": self.muon_lr,
            "--l7-mult": self.l7_mult,
            "--l7-threshold": self.l7_threshold,
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not math.isfinite(self.muon_lr) or self.muon_lr <= 0.0:
            problems.append(f"MorseContinuationSchedule.muon_lr must be > 0, got {self.muon_lr!r}")
        if self.l7_mult != 0.0:
            problems.append(
                "MorseContinuationSchedule.l7_mult must be the structural zero under unified-tau"
            )
        if not math.isclose(
            self.l7_threshold, self.m_safe, rel_tol=1e-12, abs_tol=1e-12
        ):
            problems.append(
                "MorseContinuationSchedule.l7_threshold must equal canonical m_safe"
            )
        expected = math.sqrt(
            2.0 * self.trust_region_kl / self.fisher_curvature_upper
        )
        if not math.isclose(self.muon_lr, expected, rel_tol=1e-12, abs_tol=1e-12):
            problems.append(
                "MorseContinuationSchedule.muon_lr diverges from sqrt(2*delta_KL/lambda_max)"
            )
        return problems

    def canonical_manifest(self) -> dict:
        """Return equation custody for all emitted values."""

        return {
            "equation_id": self.equation_id,
            "ladder_class": "derived_at_config",
            "values": self.flags(),
            "inputs": {
                "trust_region_kl": self.trust_region_kl,
                "fisher_curvature_upper": self.fisher_curvature_upper,
                "m_safe": self.m_safe,
                "m_safe_law": dict(self.margin_law_manifest),
            },
            "verdict_scope": (
                "FORMULATION x config compilation only; checkpoint curvature and byte-closed "
                "A/B remain owed"
            ),
        }


def WitnessNativeMorseContinuationSchedule(
    *,
    trust_region_kl: float,
    fisher_curvature_upper: float,
    margin_headroom: float | None = None,
    delta_r_artifact: str | Path = "reports/delta_R_noise_floor.json",
) -> MorseContinuationSchedule:
    """Derive the #302 schedule from local action geometry and the R-safe margin law.

    No defaults are provided for the trust-region inputs: a launch compiler must
    bind them to checkpoint-local curvature custody rather than smuggling the old
    ``0.1 * lr`` convention back through another literal.
    """

    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        resolve_margin_band_threshold,
    )
    from tac.canonical_equations.witness_native_morse_continuation_20260715 import (
        derive_morse_continuation_controls,
    )

    margin = resolve_margin_band_threshold(
        headroom=margin_headroom,
        artifact_path=delta_r_artifact,
        repo_root=_REPO_ROOT,
    )
    controls = derive_morse_continuation_controls(
        trust_region_kl=trust_region_kl,
        fisher_curvature_upper=fisher_curvature_upper,
        m_safe=margin.m_safe,
    )
    schedule = MorseContinuationSchedule(
        muon_lr=controls.muon_lr,
        l7_mult=controls.l7_mult,
        l7_threshold=controls.l7_threshold,
        trust_region_kl=controls.trust_region_kl,
        fisher_curvature_upper=controls.fisher_curvature_upper,
        m_safe=controls.m_safe,
        margin_law_manifest=margin.lawref_manifest,
    )
    problems = schedule.validate()
    if problems:
        raise ValueError("WitnessNativeMorseContinuationSchedule: " + "; ".join(problems))
    unknown = set(schedule.flags()) - real_trainer_flags()
    if unknown:
        raise ValueError(
            "WitnessNativeMorseContinuationSchedule invented trainer flags: "
            f"{sorted(unknown)}"
        )
    return schedule


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
                     "marginal_dseg_floor", "lever_exhaustion", "powerlaw_meat")


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
    * ``powerlaw_meat``: the weak-KAM tail exit (solver-pack memo 2026-07-07) —
      exit on the AIC-preferred power-law/exponential fit's EXTRAPOLATED
      remaining meat below ``floor`` (``tac.witness_control.powerlaw_exit:
      powerlaw_meat_exit``; fail-safe: unfittable ⇒ NOT exhausted). An
      exponential-window plateau detector fires EARLY on the binding lane class
      (power-law O(1/t) descent — "meat left on the bone"); this criterion is
      the fix. GAP kind: the trainer's #315 event controller has NO pluggable
      exit-criterion registry (plateau params only, VERIFIED in the trainer
      argparse), so the compile is NO argv (the fixed stage/run boundary via
      ``cap_epoch``) + a TrainerSupportGap naming the build. ``min_points`` /
      ``per_class`` mirror the callable's contract (per-class fits need the
      per-class d_seg verdict-row telemetry, itself a recorded gap).

    FRAME CONTRACT (seal v7.4 r3 F-1): gate fire telemetry records ``sensor_data_epoch``
    in the SAME epoch frame as the persisted ``_fired_epoch`` (the lever frame under
    re-anchor); since the re-anchor shift is additive it cancels in
    ``sensor_lag_epochs = fire − sde``, which therefore always reads as the true real-epoch
    verdict-cadence lag. Consumers must never mix frames when extending fire rows.
    """

    criterion: str
    min_stage_epochs: int = 150       # --curriculum-min-stage-epochs
    rel_eps: float = 1e-4             # --curriculum-plateau-rel-eps
    windows: int = 4                  # --curriculum-plateau-windows
    within_flip: float = 0.5          # --curriculum-nucleus-within-flip (nucleus kind)
    min_part_frac: float = 0.0        # --curriculum-nucleus-min-part-frac (nucleus kind)
    floor: float | None = None        # marginal-Δd_seg/epoch OR remaining-meat floor (gap kinds)
    cap_epoch: int | None = None      # deterministic hard ceiling (gap kinds' compile)
    min_points: int = 8               # powerlaw_meat: min verdict points per fittable class
    per_class: bool = False           # powerlaw_meat: per-class fits (needs per-class telemetry)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.criterion not in _EXIT_EVENT_CRITERIA:
            problems.append(
                f"ExitEvent: unknown criterion {self.criterion!r} (known: {_EXIT_EVENT_CRITERIA})")
        if self.criterion in ("marginal_dseg_floor", "lever_exhaustion") and self.floor is None:
            problems.append(
                f"ExitEvent[{self.criterion}]: needs an explicit ``floor`` (the pre-registered "
                "marginal-Δd_seg/epoch floor — §14 axis 5)")
        if self.criterion == "powerlaw_meat":
            if self.floor is None:
                problems.append(
                    "ExitEvent[powerlaw_meat]: needs an explicit ``floor`` (the pre-registered "
                    "remaining-meat floor the extrapolated tail must fall below — weak-KAM exit)")
            if self.min_points < 4:
                problems.append(
                    f"ExitEvent[powerlaw_meat]: min_points must be >= 4 (two 3-parameter tail "
                    f"models cannot fit fewer points), got {self.min_points}")
        if (
            self.criterion in ("plateau", "nucleus_guarded_plateau")
            and (self.min_stage_epochs <= 0 or self.windows <= 0 or self.rel_eps <= 0)
        ):
            problems.append(
                f"ExitEvent[{self.criterion}]: plateau params must be positive "
                f"(min_stage_epochs={self.min_stage_epochs}, windows={self.windows}, "
                f"rel_eps={self.rel_eps})"
            )
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
        elif self.criterion == "powerlaw_meat":
            gaps.append(TrainerSupportGap(
                axis="exit_events",
                requirement=("powerlaw_meat exit"
                             + (f" on stage {stage!r}" if stage else "")
                             + f" (floor={self.floor}, min_points={self.min_points}, "
                             + f"per_class={self.per_class})"),
                nearest_real_compilation=(
                    f"fixed boundary (deterministic cap_epoch={self.cap_epoch})"),
                flag_proposal=("--stage-exit-powerlaw-meat-floor <float> + "
                               "--stage-exit-powerlaw-horizon <int> (trainer build; calls "
                               "tac.witness_control.powerlaw_exit:powerlaw_meat_exit on the "
                               "stage's verdict window)"),
                notes=("weak-KAM tail exit: exponential-window plateau detectors fire EARLY on "
                       "the power-law binding class ('meat left on the bone'); exit on the "
                       "AIC-preferred fit's extrapolated remaining meat, fail-safe on "
                       "unfittable data; per-class fits gated on per-class d_seg verdict rows")))
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
    # (operator 2026-07-08 GPU-verdict HYBRID) device for the ADVISORY d_seg/d_pose scalars +
    # the CPU-torch positive-control ANCHOR cadence. "cpu" (default) => byte-identical to the
    # sealed #205 verdict. "gpu" => the MLX scorer ports (fast trajectory monitor); pair with
    # verdict_anchor_every>0 to keep the CPU-torch sentinel. NON-PROMOTABLE either way (CLAUDE.md:
    # MLX/MPS is NEVER a score; only a byte-closed exact eval moves the pointer).
    verdict_device: str = "cpu"
    verdict_anchor_every: int = 0

    def flags(self) -> dict:
        return {
            "--eval-every": int(self.eval_every),
            "--verdict-pairs": int(self.verdict_pairs),
            "--verdict-batch": int(self.verdict_batch),
            "--async-verdict": bool(self.async_verdict),
            "--verdict-device": str(self.verdict_device),
            "--verdict-anchor-every": int(self.verdict_anchor_every),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.eval_every <= 0:
            problems.append(f"VerdictCadence: eval_every must be > 0, got {self.eval_every}")
        if self.verdict_pairs < 0 or self.verdict_batch < 0:
            problems.append(
                f"VerdictCadence: verdict_pairs/verdict_batch must be >= 0, got "
                f"{self.verdict_pairs}/{self.verdict_batch}")
        if self.verdict_device not in ("cpu", "gpu"):
            problems.append(
                f"VerdictCadence: verdict_device must be 'cpu' or 'gpu', got {self.verdict_device!r}")
        if self.verdict_anchor_every < 0:
            problems.append(
                f"VerdictCadence: verdict_anchor_every must be >= 0, got {self.verdict_anchor_every}")
        if self.verdict_device == "gpu" and self.async_verdict:
            problems.append(
                "VerdictCadence: verdict_device='gpu' cannot combine with async_verdict "
                "(MLX off the main thread races the training GPU stream)")
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
    # (#312 Phase A/B) INTERACTION-layer telemetry. Score-neutral read-only observability, BUT
    # HEAVY compute (per-term backward passes / HVP-Lanczos through R), so per the 'off is a tracked
    # queue' reconciliation these default OFF on the COMPUTE-COST exception and are HELD here as a
    # tracked cadence knob (reason recorded), never a forgotten hidden switch. grad_interaction fires
    # at stage boundaries (+ every grad_interaction_every epochs); curvature at checkpoint cadence,
    # governor-gated. When the compute envelope permits, the operator flips these on.
    grad_interaction: bool = False
    grad_interaction_k_pairs: int = 32
    grad_interaction_every: int = 0
    curvature: bool = False
    curvature_k: int = 8
    curvature_k_pairs: int = 8
    # D-A/D-B launch observers (2026-07-13).  Both are score-neutral and
    # default ON.  The wall-clock producer performs one same-function
    # decomposition probe per epoch; the SPS observer is inert unless phase or
    # screw is configured and samples four deterministic strata at boundary
    # +/- window plus the actual engagement transition.
    component_wallclock: bool = True
    component_wallclock_probe_every: int = 1
    sps_engagement: bool = True
    sps_engagement_k_pairs: int = 4
    sps_engagement_window: int = 2
    # (JACOBIAN BASIN, 2026-07-09) the ξ→PoseNet Jacobian conditioning basin sensor — an OBSERVER of
    # render coherence (σ_min of J_ξ=∂PoseNet(R(θ,ξ))[:6]/∂ξ). Score-neutral read-only, so BOTH tiers
    # DEFAULT ON per the 'off is a tracked queue' law: T0 (near-free ∇f0 proxy, every verdict) + T1 (the
    # σ_min authority via mx.vjp, SUBSAMPLED to k_pairs stratified by |ego-t| + CADENCED every N-th
    # verdict — the compute-cost tier holds its cadence knob, reason recorded). B1 byte-identical, B2
    # fail-open, B5 OBSERVER-ONLY (the basin TRIGGER that would actuate the engage-point is a run-2 lever,
    # NOT this sensor — see TerminalPoseFinish(start_event=...) + jacobian_basin.JACOBIAN_BASIN_ENTRY_TRIGGER).
    jacobian_basin: bool = True
    jacobian_basin_t0: bool = True
    jacobian_basin_k_pairs: int = 32
    jacobian_basin_every: int = 4
    jacobian_basin_stratify_t: bool = True
    jacobian_basin_sigma_floor: float = 1e-4
    jacobian_basin_f_basin: float = 1.0
    jacobian_basin_quorum_q: float = 0.8

    def flags(self) -> dict:
        f: dict = {
            "--annulus-telemetry": bool(self.annulus),
            "--loss-term-log-every": int(self.loss_term_log_every),
            "--handoff-readiness-telemetry": bool(self.handoff_readiness),
            "--component-wallclock-telemetry": bool(self.component_wallclock),
            "--component-wallclock-probe-every": int(self.component_wallclock_probe_every),
            "--sps-engagement-telemetry": bool(self.sps_engagement),
            "--sps-engagement-k-pairs": int(self.sps_engagement_k_pairs),
            "--sps-engagement-window": int(self.sps_engagement_window),
        }
        if self.annulus:
            f["--annulus-band"] = float(self.annulus_band)
            f["--annulus-bottom-k"] = float(self.annulus_bottom_k)
        if self.dm1:
            f["--dm1-telemetry"] = True  # store_true: True only (C2)
        if self.grad_interaction:  # #312 Phase A store_true + its cadence/sample knobs
            f["--grad-interaction-telemetry"] = True
            f["--grad-interaction-k-pairs"] = int(self.grad_interaction_k_pairs)
            f["--grad-interaction-every"] = int(self.grad_interaction_every)
        if self.curvature:  # #312 Phase B store_true + its top-k / sample knobs
            f["--curvature-telemetry"] = True
            f["--curvature-k"] = int(self.curvature_k)
            f["--curvature-k-pairs"] = int(self.curvature_k_pairs)
        # (JACOBIAN BASIN, 2026-07-09) OBSERVER flags. BooleanOptionalAction on the trainer, so the DSL
        # emits the affirmative flag (bool value); the registry drops the '--no-' negation on both sides.
        f["--jacobian-basin-telemetry"] = bool(self.jacobian_basin)
        if self.jacobian_basin:
            f["--jacobian-basin-t0"] = bool(self.jacobian_basin_t0)
            f["--jacobian-basin-k-pairs"] = int(self.jacobian_basin_k_pairs)
            f["--jacobian-basin-every"] = int(self.jacobian_basin_every)
            f["--jacobian-basin-stratify-t"] = bool(self.jacobian_basin_stratify_t)
            f["--jacobian-basin-sigma-floor"] = float(self.jacobian_basin_sigma_floor)
            f["--jacobian-basin-f-basin"] = float(self.jacobian_basin_f_basin)
            f["--jacobian-basin-quorum-q"] = float(self.jacobian_basin_quorum_q)
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
    morse_continuation: MorseContinuationSchedule | None = None
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
        if self.morse_continuation is not None:
            owners.append(("morse_continuation", self.morse_continuation.flags()))
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
        if self.morse_continuation is not None:
            problems.extend(self.morse_continuation.validate())
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
    lawrefs: dict = field(default_factory=dict, compare=False, repr=False)
    constant_manifest: dict = field(default_factory=dict, compare=False, repr=False)
    runtime_receipt_schemas: dict = field(default_factory=dict, compare=False, repr=False)
    policy_contracts: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def constant_refs(self) -> dict:
        """Compatibility name consumed by the V9 provenance-bijection gate.

        ``lawrefs`` is the historical DSL field used by the Ladder homotopy.  The
        provenance gate names the same ownership edge ``constant_refs``.  Keeping
        one stored mapping and exposing this read-only alias prevents the two
        vocabularies from drifting into parallel sources of truth.
        """
        return self.lawrefs


def _v9_scientific_constant_custody(
    equation_id: str,
    declarations: dict[str, float | int | bool],
    *,
    provenance: str,
    ladder_by_flag: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Resolve scalar V9 declarations into LawRefs + compiler manifest rows.

    This helper is intentionally private to the DSL factories: callers author a
    scientific Lever, never a raw flag dictionary.  Boolean declarations resolve
    through integer 0/1 because LawRef inputs are numeric; the emitted override
    retains its real bool type and the trainer parser normalizes it as a bool.
    """
    from tac.witness_dsl.lawref import (
        LADDER_DERIVED_AT_CONFIG,
        InputRef,
        LawRef,
        resolve,
    )

    ladders = dict(ladder_by_flag or {})
    refs: dict = {}
    manifests: dict = {}
    for flag, declared in declarations.items():
        scalar = int(declared) if isinstance(declared, bool) else declared
        ref = LawRef(
            equation_id=equation_id,
            inputs={
                "value": InputRef.literal(
                    scalar,
                    f"{provenance}; declaration={flag}",
                    config_tags={"vehicle": "v9_cgauge_ideal_mod19"},
                ),
            },
            ladder_class=ladders.get(flag, LADDER_DERIVED_AT_CONFIG),
        )
        resolved = resolve(
            ref,
            target_config_tags={"vehicle": "v9_cgauge_ideal_mod19"},
            repo_root=_REPO_ROOT,
        )
        refs[flag] = ref
        manifests[flag] = resolved.to_dict()
    return refs, manifests


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
    curriculum: Curriculum | None = None
    # DECLARED RUN INTENT (operator 2026-07-07: "clean baseline or frontier score lowering?
    # a/b probe?"). One human line stating WHY this program's run exists. Metadata ONLY —
    # never emitted into flag_dict()/trainer argv (argv-inert by construction); the launch
    # path threads it to WitnessConfig.purpose / launch_witness_run --purpose, which stamps
    # the run dir's config record so the dashboard renders it VERBATIM as "declared" intent
    # (else it falls back to the labelled derived heuristic). A DSL-generated launch without
    # this is orphaned-intent — declare it at program construction.
    purpose: str | None = None

    # --- composition ---------------------------------------------------------
    def with_lever(self, *levers: Lever, resume_from=_INHERIT,
                   out_dir: str | None = None) -> WitnessProgram:
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
                   render_aa=None, lane_band=None, head_geometry=None) -> WitnessProgram:
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
        from tac.witness_dsl.gauge import CANONICAL_GAUGE, GaugeChoice
        if gauge_choice is None:
            base_gauge = self.gauge if isinstance(self.gauge, GaugeChoice) else CANONICAL_GAUGE
            overrides = {
                k: v
                for k, v in {
                    "warp": warp,
                    "carrier": carrier,
                    "residual": residual,
                    "pose": pose,
                    "movables": movables,
                    "generation": generation,
                    "render_aa": render_aa,
                    "lane_band": lane_band,
                    "head_geometry": head_geometry,
                }.items()
                if v is not None
            }
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
        # #332 CUSTODY COMPOSITION LAW (2026-07-17): a flag-custody rollup Lever
        # (spec_v9_cgauge.attach_flag_custody) is VALUE-NEUTRAL by contract — it
        # re-asserts already-composed values purely to carry ownership/LawRef/
        # receipt custody.  If a derived config mutates ``base``/``out_dir``
        # AFTER custody was attached, the rollup's overrides would silently
        # shadow the mutation (levers compose last).  Refuse LOUDLY instead:
        # composing WITH vs WITHOUT the custody lever(s) must yield an
        # identical flag_dict, else the caller must strip_flag_custody() first
        # and re-attach after its mutations.
        custody_levers = [lv for lv in self.levers
                          if str(lv.name).startswith("v9_flag_custody")]
        if custody_levers:
            stripped = replace(
                self,
                levers=tuple(lv for lv in self.levers
                             if not str(lv.name).startswith("v9_flag_custody")))
            with_custody = fd
            without_custody = stripped.flag_dict()
            if with_custody != without_custody:
                _sentinel = object()
                drifted = sorted(
                    k for k in set(with_custody) | set(without_custody)
                    if with_custody.get(k, _sentinel) != without_custody.get(k, _sentinel))
                problems.append(
                    "CUSTODY NON-NEUTRAL: the v9_flag_custody rollup shadows post-custody "
                    f"mutations on {drifted[:6]} — strip_flag_custody() before mutating "
                    "base/out_dir and re-attach custody at the end (spec_v9_cgauge)")
        # CONFIG-BUILD-TIME RUNNABILITY: an enabled curriculum whose effective
        # stage caps lie beyond this run's epoch budget would produce a false
        # verdict (the stage can never engage).  Use the real parser defaults so
        # omission cannot conceal a latent tau/l7 window.  This deliberately
        # precedes the narrower ordering/stagger checks and reports every
        # out-of-budget cap in one violation.
        problems.extend(schedule_epoch_budget_violations(fd, trainer_path))
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
                            _ep = int(_z[_k])
                            break
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
            if (
                _tau_s is not None
                and _l7_s is not None
                and not (0 < _tau_s < _l7_s)
            ):
                problems.append(
                    f"CURRICULUM ORDERING: need 0 < tau_start ({_tau_s}) < l7_start ({_l7_s}) "
                    "(the tau stage forms the partition before l7 sharpens it; trainer asserts "
                    "this — l7_start > epochs is allowed and means l7 NEVER runs)")
        # S2-REV-A LADDER↔Muon STAGGER INVARIANT (T3 v7 council): a future config cannot silently
        # lengthen a LADDER arm's anneal past the Muon finisher (→ live-LADDER-during-Muon/TAIL). Read
        # the composed lever's window flags + the Muon cap from the flag dict; the shared pure helper
        # is the single source of truth (also called by the trainer's config-validation path).
        if fd.get("--ladder-island-homotopy") in (True, 1):
            from tac.witness_curriculum.ladder_homotopy import (
                ladder_arm_window,
                ladder_muon_stagger_violation,
            )
            _lane_win = ladder_arm_window(
                fd.get("--ladder-lane-birth-epochs", 0), fd.get("--ladder-lane-hold-epochs", 0),
                fd.get("--ladder-lane-anneal-epochs", 0))
            _mov_win = ladder_arm_window(
                fd.get("--ladder-movable-birth-epochs", 0), fd.get("--ladder-movable-hold-epochs", 0),
                fd.get("--ladder-movable-anneal-epochs", 0))
            _muon_cap = fd.get("--muon-start-epoch")
            _stagger = ladder_muon_stagger_violation(
                ladder_on=True, lane_window=_lane_win, movable_window=_mov_win,
                muon_start_epoch=(None if _muon_cap is None else int(_muon_cap)))
            if _stagger is not None:
                problems.append(_stagger)
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
        # #218 ADDITIVE-MARGIN INERT COMPOSITION (fail-closed; #404 binding-vs-inert, operator
        # elevation 2026-07-10). The HeadGeometry additive-margin arm (--head additive-margin, or a
        # non-zero --additive-margin) is a SILENT NO-OP unless a sibling lever (MarginFieldHead) sets
        # --margin-field-head-weight>0 — the trainer builds the margin-field target ONLY then. An inert
        # arm reads as ON but shapes no loss => any verdict from that run is corrupted (surrogate !=
        # authority). REFUSE the composition here (never-invent-flags; do NOT auto-arm it). ONE classifier
        # SoT (tac.confound_gates.additive_margin_engagement) shared with the trainer L1 alarm + the L2
        # preflight gate. Lazy import (parallel to the gauge/ladder_homotopy lazy imports) so curriculum_dsl
        # keeps no confound_gates import at module load.
        _head = fd.get("--head")
        _am = fd.get("--additive-margin")
        if _head is not None or _am is not None:
            from tac.confound_gates import additive_margin_engagement
            try:
                _eng = additive_margin_engagement(
                    str(_head or "softmax"), float(_am or 0.0),
                    float(fd.get("--margin-field-head-weight", 0.0) or 0.0))
            except (TypeError, ValueError):
                _eng = None
            if _eng is not None and _eng["inert"]:
                problems.append(
                    "INERT ADDITIVE-MARGIN COMPOSITION (#404 binding-vs-inert): "
                    f"{_eng['reason']}. The #218 additive-margin arm reads as ON but shapes no loss "
                    "(the trainer builds the margin-field target only when --margin-field-head-weight>0). "
                    "Compose MarginFieldHead(weight>0) alongside HeadGeometry (and a non-zero "
                    "additive_margin), or drop the additive-margin arm — never ship an inert lever.")
        return problems

    def support_gaps(self) -> tuple[TrainerSupportGap, ...]:
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

    # --- LawRef constant compilation (task #351) -----------------------------
    def compile_trainer_argv_with_constants(
        self, target_config_tags: dict | None = None,
        python: str = ".venv/bin/python", repo_root=None,
    ) -> tuple[list[str], dict]:
        """Resolve any LawRef-valued flags into values, emit argv, return (argv, manifest).

        A flag whose value is a ``tac.witness_dsl.lawref.LawRef`` is resolved at
        compile time (the mx.compile analogy) into its actual value + a provenance
        record; the ``manifest`` (dict {flag: {value, equation_id, inputs+shas,
        ladder_class, resolved_at, fallback_used}}) is the ``constants_manifest.json``
        content a launcher writes beside ``launch.sh``. Programs with NO LawRef flags
        yield argv byte-identical to :meth:`compile_trainer_argv` + an empty manifest.
        Fail-closed: a config-conditionality conflict / unresolved input raises
        (see ``tac.witness_dsl.lawref.resolve``). This method does NOT write files.
        """
        from tac.witness_dsl.lawref import resolve_flag_dict_constants

        resolved_fd, manifest = resolve_flag_dict_constants(
            self.flag_dict(), target_config_tags, repo_root=repo_root)
        # Some factories resolve their LawRefs eagerly so the typed scalar
        # layer never has to stringify a LawRef object.  Merge those canonical
        # compiler rows here and prove their resolved value still matches the
        # final composed flag value (later-lever drift is a hard refusal).
        for lever in self.levers:
            for flag, record in lever.constant_manifest.items():
                if flag not in resolved_fd:
                    raise ValueError(
                        f"{self.purpose or self.out_dir}: constant manifest owns absent flag {flag}")
                expected = record.get("value")
                actual = resolved_fd[flag]
                if isinstance(actual, bool) and isinstance(expected, int):
                    matched = int(actual) == expected
                else:
                    matched = actual == expected
                if not matched:
                    raise ValueError(
                        f"{lever.name}: LawRef compiler value {flag}={expected!r} "
                        f"does not match composed DSL value {actual!r}")
                prior = manifest.get(flag)
                if prior is not None and prior != record:
                    raise ValueError(f"{lever.name}: duplicate conflicting LawRef record for {flag}")
                manifest[flag] = dict(record)
        argv = [python, TRAINER_REL]
        for flag, val in resolved_fd.items():
            if val is True:
                argv.append(flag)
            elif val is False:
                argv.append(flag.replace("--", "--no-", 1))
            else:
                argv.extend([flag, str(val)])
        return argv, manifest


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
def YhatNativeGenerator(*, policy: YhatNativeGeneratorPolicy) -> Lever:
    """Default-OFF yhat-native measurement leg; deliberately not bare-name composable."""

    from tac.witness_dsl.yhat_native_generator_policy import YhatNativeGeneratorPolicy

    if not isinstance(policy, YhatNativeGeneratorPolicy):
        raise TypeError("YhatNativeGenerator requires a YhatNativeGeneratorPolicy")
    policy.compile_contract()
    return Lever(
        "YhatNativeGenerator",
        overrides={},
        epochs_delta=0,
        notes="argv-inert default-OFF policy; n24 receipt closed; receiver/archive gates owed",
        lawrefs={},
        constant_manifest={},
    )


def IntegerPlaneEmitter(*, policy: IntegerPlaneEmitterPolicy) -> Lever:
    """Typed C2 policy lever for the dedicated band trainer.

    The compatibility mode is deliberately argv-inert. ``BANDED_TRAINING``
    emits only flags owned by the dedicated C2 parser; it is therefore not
    composable into the level-set ``BASELINE`` program.
    """

    from tac.witness_dsl.integer_plane_emitter_policy import (
        BANDED_TRAINING_RECEIPT_SCHEMA,
        POLICY_CONTRACT_RECEIPT_KEY,
        IntegerPlaneEmitterPolicy,
        PolicyMode,
    )

    if not isinstance(policy, IntegerPlaneEmitterPolicy):
        raise TypeError("IntegerPlaneEmitter requires an IntegerPlaneEmitterPolicy")
    contract = policy.compile_contract()
    active = policy.mode is PolicyMode.BANDED_TRAINING
    overrides = (
        {
            "--integer-plane-emitter-mode": policy.mode.value,
            "--integer-plane-emitter-basis": policy.basis.value,
            "--integer-plane-emitter-policy-sha256": contract["policy_sha256"],
        }
        if active
        else {}
    )
    receipts = (
        {"--integer-plane-emitter-policy-sha256": BANDED_TRAINING_RECEIPT_SCHEMA}
        if active
        else {}
    )
    return Lever(
        "IntegerPlaneEmitter",
        overrides=overrides,
        epochs_delta=0,
        notes=(
            ("argv-effective dedicated C2 band trainer; " if active else "argv-inert default-OFF C2 emitter; ")
            + "basis="
            f"{contract['basis']}; policy_sha256={contract['policy_sha256']}; "
            "launch/score/promotion/pointer authority sealed false"
        ),
        lawrefs={},
        constant_manifest={},
        runtime_receipt_schemas=receipts,
        policy_contracts={POLICY_CONTRACT_RECEIPT_KEY: contract},
    )


def PoseDecouple(window: int = 100) -> Lever:
    """A5: drop pose from the loss (w-pose=0) to free decoder capacity for d_seg —
    a TRADE (d_pose worsens; pose is carried in-frame, NOT sidecar-able, per the
    byte-close finding). Carries a warm-start window (else dead-arm, review C1)."""
    return Lever("A5_pose_decouple", overrides={"--w-pose": 0.0}, epochs_delta=window,
                 notes="drop pose-loss to free d_seg capacity (trades d_pose up)")


def PoseFinishConditioningGate(
    backstop_epoch: int | None = None, w_pose: float | None = None,
    engage_mode: str = "sigma_min_plateau",
) -> Lever:
    """owed-1 (SYNTHESIS_v3_v752 §A.4, A-1 FIX): engage the TERMINAL pose-finish on the SEALED d_seg-
    CONDITIONING event — a scale-free ROLLING-SLOPE plateau of the DE-NOISED σ_min(J_ξ) conditioning
    series — INSTEAD of the muon proxy. Realizes the operator binding "pose must not be fired for joint
    descent until dseg is sufficiently conditioned first."

    Sets ``--pose-finish-engage-on sigma_min_plateau`` (the score-affecting engage lever; DEFAULT-OFF =
    'muon'). Optionally arms the two-phase ``--pose-finish-start-epoch`` (the fail-safe BACKSTOP cap) +
    the finish weight ``--w-pose`` (else the program's pose config supplies them). The σ_min sensor
    ``--jacobian-basin-telemetry`` (default ON) is the gate's ONLY σ_min source and MUST stay ON. A
    degenerate/canary-fail/never-fired gate ships the banked R1 dxi (DISENGAGED, LOUD alarm) — NEVER
    blocks (SYNTHESIS §A.4 Repair 2b/4). Detector: ``tac.witness_control.sigma_min_plateau``.

    ``engage_mode`` (SPEC_v10 §13.2): ``'sigma_min_plateau'`` (DEFAULT — the SEALED rolling-slope
    plateau) or ``'sigma_min_crest'`` (fire-on-crest: the smoothed σ_min slope SIGN-CHANGE = the
    conditioning PEAK, hysteresis-held; same sensor/de-noise/guard machinery + banked-R1 fallback
    contract. Live c2 anchor: σ_min 0.0010→0.0068 still climbing +15%/ep at ep798 — a plateau gate
    keeps waiting while a crest gate arms the moment conditioning stops improving)."""
    if str(engage_mode) not in ("sigma_min_plateau", "sigma_min_crest"):
        raise ValueError(
            "PoseFinishConditioningGate: engage_mode must be 'sigma_min_plateau' or "
            f"'sigma_min_crest', got {engage_mode!r} ('muon' is the incumbent default, not a "
            "conditioning gate — omit this lever for it)")
    ov: dict = {"--pose-finish-engage-on": str(engage_mode)}
    if backstop_epoch is not None:
        if int(backstop_epoch) <= 0:
            raise ValueError(
                "PoseFinishConditioningGate: backstop_epoch must be > 0 (the two-phase arm / fail-safe "
                "cap); pass None to let the program's pose config set --pose-finish-start-epoch")
        ov["--pose-finish-start-epoch"] = int(backstop_epoch)
    if w_pose is not None:
        if not (float(w_pose) > 0.0):
            raise ValueError("PoseFinishConditioningGate: w_pose must be > 0 (the finish-phase weight)")
        ov["--w-pose"] = float(w_pose)
    return Lever("pose_finish_conditioning_gate", overrides=ov,
                 notes="owed-1: pose-finish engages on the de-noised σ_min conditioning event "
                 f"({engage_mode}; plateau=A-1 sealed rolling-slope, crest=SPEC_v10 §13.2 slope "
                 "sign-change peak); ships banked R1 if never/degenerate/untrusted (never blocks)")


def PoseFinishBetaAnnealCoupling() -> Lever:
    """SPEC_v10 §13.2 "coupling not coincidence" (arm B 2026-07-17): the β-ANNEAL-COMPLETE →
    POSE-FINISH-ELIGIBLE event coupling. The c2 config's ``anneal-epochs(1000) ==
    pose-finish-start(1000)`` is two constants AGREEING; this lever expresses the encoded intent
    as the EVENT — pose-finish may not engage (by ANY signal: muon / σ_min gate / backstop) before
    the β/τ anneal SCHEDULE completes (ep >= --anneal-epochs, fallback --epochs).

    MEASURED context [live c2 run 20260717T113932Z, advisory]: the σ_min conditioning CREST landed
    at ~ep802 — BEFORE the constant eligibility epoch (1000) — so the constant is measured-
    SUBOPTIMAL on that run (an event-derived gate could have engaged at the measured optimum).
    Composes with ``PoseFinishConditioningGate(engage_mode=...)``: the gate supplies the ENGAGE
    signal, this coupling supplies the ELIGIBILITY floor. Requires the two-phase arm
    (``--pose-finish-start-epoch > 0``; the trainer fails loud on an inert arm). DEFAULT-OFF;
    absent ⇒ byte-identical."""
    return Lever(
        "pose_finish_beta_anneal_coupling",
        overrides={"--pose-finish-eligible-on-beta-anneal-complete": True},
        notes="SPEC_v10 §13.2: beta-anneal-complete -> pose-finish-eligible event coupling "
        "(replaces the anneal-epochs==pose-finish-start constants coincidence); measured live-c2 "
        "context: crest @~ep802 preceded the ep1000 constant (constant measured-suboptimal)")


def PoseMarginalWeightLaw(clamp: float | None = None) -> Lever:
    """SPEC_v10 §13.3 (arm B 2026-07-17): the w_pose(t) DERIVED-WEIGHT LAW —
    ``w_pose(t) = min(clamp, 5/sqrt(10*d_pose(t)))``, the score's OWN pose marginal
    (``dS/dd_pose``) as the pose-finish weight, replacing the static ``--w-pose`` constant.

    THE CLAMP IS DERIVED, not tuned: the marginal diverges as d_pose→0; the seg marginal is the
    constant ``dS/dd_seg = 100``; the two cross at ``d_pose = 2.5e-4`` where the pose marginal
    equals 100 — so ``clamp = 100.0`` caps the pose weight at the score's own seg exchange rate
    (law module ``tac.canonical_equations.w_pose_marginal_weight_law_20260717``, eq
    ``w_pose_marginal_weight_law_v1``; sister of the ``--pose-grad-coeff-max`` divergence guard).

    Consumption point: the POSE-FINISH stage only (the trainer fails loud if the two-phase arm is
    absent — inert-flag NO-FAKE guard). Updated at VERDICT cadence when a measured d_pose lands
    (piecewise-constant, never per-step — SPEC_v75 §8 loss-weights-at-boundaries). DEFAULT-OFF;
    absent ⇒ byte-identical.

    ⚠ INCOMPATIBLE with score-domain loss (SOL v10 review A2-C1; SPEC_v10 §13.13). The marginal
    ``w = 5/sqrt(10*d_pose)`` is the exact ``dS/dd_pose`` — the weight for a RAW-``d_pose`` (weight-
    domain) loss term, where ``dL/dd_pose = w*1`` = the contest marginal. Under ``--score-domain-loss``
    (trainer default ON) the pose term is ALREADY ``sqrt(10*d_pose)`` (the exact score contribution),
    so applying this weight yields ``dL/dd_pose = (5/sqrt(10*d_pose))^2`` = the marginal SQUARED. The
    trainer (the fail-closed launch authority) REFUSES ``--w-pose-marginal-law`` together with
    ``--score-domain-loss``; this law is admissible ONLY with ``--no-score-domain-loss``. Under
    score-domain loss the exact objective is a static ``--w-pose 1`` (the sqrt term IS the score)."""
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG, InputRef, LawRef

    from tac.canonical_equations.w_pose_marginal_weight_law_20260717 import (
        clamp_from_crossover,
    )

    c = clamp_from_crossover() if clamp is None else float(clamp)
    if not (c > 0.0):
        raise ValueError(f"PoseMarginalWeightLaw: clamp must be > 0, got {clamp!r}")
    refs = {
        "--w-pose-marginal-clamp": LawRef(
            equation_id="w_pose_marginal_weight_law_v1",
            inputs={"value": InputRef.literal(
                c,
                "DERIVED clamp_from_crossover(): pose marginal 5/sqrt(10*d_pose) equals the seg "
                "marginal dS/dd_seg=100 at d_pose=2.5e-4; cap = the score's own seg exchange rate "
                "(tac.canonical_equations.w_pose_marginal_weight_law_20260717)")},
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        ),
    }
    return Lever(
        "pose_marginal_weight_law",
        overrides={"--w-pose-marginal-law": True, "--w-pose-marginal-clamp": c},
        lawrefs=refs,
        notes="SPEC_v10 §13.3: w_pose(t)=min(clamp,5/sqrt(10*d_pose(t))) — the score's own pose "
        "marginal as the pose-finish weight; clamp DERIVED at the seg-marginal crossover (100.0); "
        "verdict-cadence piecewise-constant; requires the two-phase pose finish (fail-loud)")


def Muon(start_epoch: int, window: int = 100) -> Lever:
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


def MuonAtCheckpointBoundary(start_epoch: int, window: int = 0) -> Lever:
    """Enter the incumbent Muon stage at a warm-start checkpoint boundary.

    Unlike the historical :func:`Muon` experiment factory, this does not
    overwrite temperature with a hard-coded 0.05.  The trainer freezes the
    compiled schedule at ``start_epoch`` itself, which makes this the matched
    control for a checkpoint-derived finisher probe.
    """

    if int(start_epoch) < 1:
        raise ValueError("MuonAtCheckpointBoundary: start_epoch must be >= 1")
    if int(window) < 0:
        raise ValueError("MuonAtCheckpointBoundary: window must be >= 0")
    return Lever(
        "muon_at_checkpoint_boundary",
        overrides={"--muon-start-epoch": int(start_epoch)},
        epochs_delta=int(window),
        notes=("matched warm-start Muon control; freezes the existing typed tau schedule at the "
               "checkpoint-derived boundary; no hand trainer flags"),
    )


def DirectionalBasis(weight: float = 0.5, start_epoch: int = 300,
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


def RangeAProjection(cadence: str = "post_render") -> Lever:
    """P3/#520 (SPEC_v10): arm the render-target range(A) projection — restrict the render
    to the frozen scorers' sigma-algebra (drop ker(A), the MEASURED ~52% scorer-invisible
    render energy, #519 ``null_subspace_rate_measure``).  DEFAULT-OFF duty-to-measure lever:
    the trainer flag ``--range-a-projection`` defaults off (byte-identity); composing this
    lever turns it on for the A/B.

    ``cadence`` ∈ {"post_render","every_step"} — project once after the render, or the loss
    render each step.  The exact projector is ``tac.boundary_math.range_a_projection``
    (self-test residual max|A(X-PX)| = 1.65e-15, reproduced from #519).  A byte-closed n600
    realized-through-R row is OWED (operator-GO, PREPARED_NOT_FIRED); the score EFFECT is
    measurement-gated (SPEC_v10 launch-gate chain).  The factory is here (not only in a side
    module) so ``lever_registry.completeness`` + the activation-ledger duty queue discover it
    generically.
    """
    if cadence not in ("post_render", "every_step"):
        raise ValueError(
            f"RangeAProjection: cadence must be post_render|every_step; got {cadence!r}")
    return Lever(
        "range_a_projection",
        overrides={"--range-a-projection": True,
                   "--range-a-projection-cadence": cadence},
        notes=("P3/#520 render-target range(A) restriction; default-off duty-to-measure; "
               "projector residual 1.65e-15 (null_subspace_rate_measure #519); "
               "n600 realized-through-R row OWED (operator-GO, PREPARED_NOT_FIRED)"),
    )


def WindowedCurveletBasis(window: int = 0) -> Lever:
    """Select the generated-receiver-sealed windowed-directional coordinate frame.

    Default-off means this factory is absent from ``BASELINE``/``proven_base``; composing the named
    lever is the explicit treatment.  The factory exists here (rather than only in a side module) so
    ``lever_registry.completeness`` and the activation-ledger duty queue discover it generically.
    A byte-closed n600 realized-through-R row is still owed and requires operator GO.
    """
    if int(window) < 0:
        raise ValueError("WindowedCurveletBasis: window must be >= 0")
    return Lever(
        "basis_family::windowed_curvelet",
        overrides={
            "--bank-n-scales": 4,
            "--bank-n-orient0": 6,
            "--bank-f0": 2.0,
            "--bank-base": 2.0,
            "--bank-n-iso": 4,
            "--max-bank-freq": 64.0,
            "--self-orient": False,
            "--basis": "windowed_curvelet",
        },
        epochs_delta=int(window),
        notes=("FEED-cvl-throughR: train+generated-inflate op parity wired; default-off treatment; "
               "byte-closed n600 realized d_seg A/B OWED (operator-GO, PREPARED_NOT_FIRED)"),
    )


def CompactShearletBasis(window: int = 0) -> Lever:
    """Select the generated-receiver-sealed compact cone-adapted shearlet frame.

    This remains a default-off, PREPARED_NOT_FIRED treatment.  Its structural frame proof and
    advisory receiver row are not a byte-equal n600 score verdict.
    """
    if int(window) < 0:
        raise ValueError("CompactShearletBasis: window must be >= 0")
    return Lever(
        "basis_family::compact_shearlet",
        overrides={
            "--bank-n-scales": 4,
            "--bank-n-orient0": 6,
            "--bank-f0": 2.0,
            "--bank-base": 2.0,
            "--bank-n-iso": 4,
            "--max-bank-freq": 64.0,
            "--self-orient": False,
            "--basis": "compact_shearlet",
        },
        epochs_delta=int(window),
        notes=("FEED-shr-throughR: train+generated-inflate op parity wired; default-off treatment; "
               "byte-closed n600 realized d_seg A/B OWED (operator-GO, PREPARED_NOT_FIRED)"),
    )


def LiteralPolarCurveletBasis(
    window: int = 0,
    native_orient: bool = True,
    kappa: float = 2.0,
    fixed_point_iters: int = 6,
) -> Lever:
    """Select the literal finite polar-frequency-wedge curvelet frame (p0_497 treatment).

    The genuine Candes-Donoho-style dictionary recovered by the curvelet crux
    (``tac.boundary_math.localized_basis_frames``, family ``literal_polar_curvelet``,
    80 columns: 4 scaling + 76 directional; atom-spec-hash sealed by
    ``BasisProgramConfig``). Same-width decoder-native orientation gating is the
    selected mechanism (SPEC ``curvelet_optimal_form_crux_20260715_SPEC.md``): it
    gates the existing 80 columns instead of appending a directional Fourier bank,
    so learned tensor shapes match the ``legacy_fourier_ab_control`` arm exactly
    (equal-value precondition of the matched-COUNTED-BYTES A/B).

    Composition seals honored by trainer + byte-close (fail-closed, NOT this
    factory's job to bypass): scalar Fourier IPE is invalid for the literal family
    (``--render-aa`` is forced to ``none`` here — the sealed base config carries
    ``ipe``, which the literal trainer refuses); native orientation currently
    refuses ``--render-aa supersample`` until the fine-grid native gates are
    receiver-sealed; the ground chart refuses until its counted receiver program
    lands. Default-off treatment; the byte-closed n600 realized-through-R
    matched-bytes A/B row is OWED (operator-GO, PREPARED_NOT_FIRED).
    """
    if int(window) < 0:
        raise ValueError("LiteralPolarCurveletBasis: window must be >= 0")
    if not (float(kappa) > 0.0):
        raise ValueError("LiteralPolarCurveletBasis: kappa must be > 0")
    if int(fixed_point_iters) <= 0:
        raise ValueError("LiteralPolarCurveletBasis: fixed_point_iters must be > 0")
    return Lever(
        "basis_family::literal_polar_curvelet",
        overrides={
            "--basis": "literal_polar_curvelet",
            "--self-orient": False,  # appended dir-bank would break the equal-value A/B (trainer refuses)
            "--render-aa": "none",   # scalar IPE fail-closed for the literal family (SPEC); base cfg has ipe
            "--literal-curvelet-native-orient": bool(native_orient),
            "--literal-curvelet-kappa": float(kappa),
            "--literal-curvelet-fixed-point-iters": int(fixed_point_iters),
        },
        epochs_delta=int(window),
        notes=("p0_497 basis-cure treatment: literal polar-curvelet + same-width native "
               "orientation; equal learned shapes vs control; matched-COUNTED-BYTES n600 "
               "realized d_seg A/B OWED (operator-GO, PREPARED_NOT_FIRED; equal-byte law "
               "tools/curvelet_equal_byte_ab_receipt.py)"),
    )


def LegacyFourierABControl(window: int = 0) -> Lever:
    """Select the byte-identical historical Fourier computation as A/B control only.

    This is not a ship-default claim. It lets the DSL author the owed
    curvelet-vs-control n600 through-R A/B without relying on an implicit parser
    default or the deprecated ``polar_fourier`` token.
    """
    if int(window) < 0:
        raise ValueError("LegacyFourierABControl: window must be >= 0")
    return Lever(
        "basis_family::legacy_fourier_ab_control",
        overrides={
            "--bank-n-scales": 4,
            "--bank-n-orient0": 6,
            "--bank-f0": 2.0,
            "--bank-base": 2.0,
            "--bank-n-iso": 4,
            "--max-bank-freq": 64.0,
            "--self-orient": False,
            "--basis": "legacy_fourier_ab_control",
        },
        epochs_delta=int(window),
        notes=("historical global Fourier plane-wave computation retained only as explicit "
               "legacy A/B control; no curvelet win or default-ship claim"),
    )


def TextureTrunk(band_hi: float = 8.0, annulus_power: float = 0.0,
                 coeff_scale: float = 0.02, window: int = 0) -> Lever:
    """#395 P0 — the band-designed per-class STATIONARY texture trunk (T of W=(G, ξ, T)).

    Turns ON a SEPARATE tiny texture trunk whose BASIS is texture-native: a fixed rule-118 Gabor bank
    pinned to the MEASURED SegNet stem pass-band ([period-4 Nyquist .. ``band_hi`` render-px];
    ``tac.through_r.stem_perception`` / ``.omx/research/segnet_texture_perception_20260710.md``), per-
    class fitted coefficients (F·K·3+K·3 = 375 counted at the default band; rate ~2.5e-4 S uncoded),
    PLACED through the partition softmax masks, trained JOINTLY through the seg loss (the #300 gradient-
    through invariant). DECOUPLES texture from the partition trunk G (P12 antagonism: G wants smooth
    off-boundary, T wants in-region oscillation) — G's ``out_sdf``/``out_tex`` are untouched.

    ``annulus_power`` >0 attenuates texture near the decision boundary (a prior; the joint loss is the
    primary guard). ``window`` = optional warm-start epochs when this rides a ``--resume-from`` of a
    converged partition trunk (the grokking-timing deployment: a fresh trunk beside a converged G risks
    delayed generalization — see ``.omx/research/texture_trunk_p0_design_20260710.md`` §training-
    dynamics). Default ``window=0`` = a full-run-from-scratch arch config (NOT a warm-start isolation
    arm). Module: ``tac.boundary_math.texture_trunk``. The BYTE-designed sister of ``out_tex`` for the
    matched-bytes A/B (linear head vs widened MLP head vs texture trunk)."""
    return Lever("texture_trunk",
                 overrides={"--texture-trunk": True,
                            "--texture-trunk-band-hi": float(band_hi),
                            "--texture-trunk-annulus-power": float(annulus_power),
                            "--texture-trunk-coeff-scale": float(coeff_scale)},
                 epochs_delta=int(window),
                 notes="band-designed per-class stationary texture trunk (stem-Nyquist band); "
                       "decouples T from the partition trunk; counted coeffs only (bank rule-118 free)")


def OutTexHidden(hidden: int = 16, window: int = 0) -> Lever:
    """#395 A2 arm — widen the ``out_tex`` texture head from a linear map (hidden→3) to a
    1-hidden-layer ReLU MLP (hidden→``hidden``→3), giving the texture/pose channel NONLINEAR
    capacity WITHOUT the fixed Gabor bank. The matched-bytes MIDDLE rung of the #395 3-arm A/B
    (A1 linear head / **A2 +out-tex-hidden** / A3 texture trunk): it pins whether ``d_seg*(T)``
    is capacity-bound (A2≈A3 ⇒ the widened head is enough) or basis-bound (A3<A2 ⇒ the stem-band
    Gabor basis is doing the work) — the P7 falsifier the trunk-composes-ON decision reads.

    Emits the REAL trainer flag ``--out-tex-hidden`` (``type=int``; DEFAULT 0 = OFF =
    byte-identical linear head; the trainer builds ``out_tex_h`` + reshapes ``out_tex`` only when
    >0 — a PARAM-SHAPE lever, resume-guarded via ``__cfg_out_tex_hidden``). Adds
    ``hidden*(H+3)+hidden+3`` counted params. Module wire-in:
    ``build_levelset_rgb_witness(out_tex_hidden=...)`` / ``LevelSetRGBWitness._tex``.

    DEFAULT-OFF: this factory is what ARMS the A2 rung; un-composed the trainer never widens the
    head (byte-identical). ``window`` = optional warm-start epochs for a ``--resume-from`` arm
    (0 = a full-run-from-scratch arch config). means != ends: builds the mechanism, makes NO score
    claim — the head width's d_seg effect is ASSUMED_AWAITING_VERIFICATION until the 3-arm A/B
    measures it byte-closed at n600."""
    if int(hidden) <= 0:
        raise ValueError(f"OutTexHidden: hidden must be > 0 (0 is the OFF/linear-head default the "
                         f"lever is not needed for), got {hidden!r}")
    return Lever("out_tex_hidden",
                 overrides={"--out-tex-hidden": int(hidden)},
                 epochs_delta=int(window),
                 notes="#395 A2 arm: widened ReLU-MLP texture head (hidden->N->3); nonlinear texture "
                       "capacity without the Gabor bank; the matched-bytes middle rung of the 3-arm A/B")


def DecoupledField(field_hidden: int = 32, field_layers: int = 2,
                   window: int = 0) -> Lever:
    """v8 B1 (#398) — the per-class DECOUPLED-FIELD partition head (G of W=(G, ξ, T)).

    Replaces the shared ``out_sdf`` linear readout with K INDEPENDENT per-class coordinate-INR fields:
    ``P(x) = argmax_c ( phi_c(x) + b_c )`` with ``∂phi_c/∂θ_{c'} = 0`` by construction (block-diagonal
    parameters over the class axis). This makes the measured cross-class THEFT (run-1: Lane 13.8× /
    Movable 4.6× stealing Road) IMPOSSIBLE for its gradient mechanism — the v8 architecture
    (``SPEC_v8_perclass_decomposition_20260708`` §1). The paint-free MASK partition ``argmax_c phi_c``
    (what the 1a decoupling screen ``tac.inc1a_harness.decoupling_screen`` measures against a
    MATCHED-COMPUTE shared-head control) is fully decoupled; ``out_tex`` stays on the shared trunk
    (texture/pose channel, SPEC §3 luma-reserved-for-pose).

    Increment-1 scope (``v8_unlock_398a_20260710`` B1): the TRAINING MODE + composition forward ONLY —
    NOT the residual coder, NOT the paint/reconciliation-vs-frozen-scorer stage (SPEC §3), NOT a
    byte-close carrier (that is E4/1b). Module: ``tac.boundary_math.decoupled_field``. ``field_hidden``
    (H) / ``field_layers`` (L) are the clause-B minimal per-class field dims (each field carries ONE
    class's separatrix, a fraction of the shared trunk's K-way job). ``window`` = optional warm-start
    epochs when this rides a ``--resume-from`` of a converged shared trunk (else default 0 = a
    full-run-from-scratch arch config). ALL params COUNTED (no rule-118-free table); the param count is
    the number E1 measures to build ``matched_control_spec(P_dec)`` for the fair 1a A/B."""
    return Lever("decoupled_field",
                 overrides={"--decoupled-field": True,
                            "--decoupled-field-hidden": int(field_hidden),
                            "--decoupled-field-layers": int(field_layers)},
                 epochs_delta=int(window),
                 notes="v8 per-class decoupled partition fields (∂phi_c/∂θ_{c'}=0); the mask partition "
                       "argmax_c phi_c is decoupled; increment-1 = training mode + composition forward")


def TauFrozen(value: float = 0.05, window: int = 100) -> Lever:
    """A1b: freeze tau (start==end) to isolate an l7 effect from the tau anneal.

    MUST carry an ``epochs_delta`` (the warm-start window) or the arm runs ZERO
    gradient steps when resumed from an end-of-run ckpt (DSL review C1, 2026-06-28:
    epochs==resume_epoch → empty range → scientifically-dead arm)."""
    return Lever("A1b_tau_frozen",
                 overrides={"--softmax-temp-start": value, "--softmax-temp-end": value},
                 epochs_delta=window,
                 notes="freeze tau to isolate l7-loss vs tau-anneal (diff refutation)")


def LrAnnealPin(anneal_epochs: int = 1000, hold_frac: float = 1.0) -> Lever:
    """Pin the LR cosine to its OWN denominator (+ optional hold), decoupling it from the SHARED
    ``--anneal-epochs`` that also drives τ + hosc-β. The trainer couples all three schedules on one
    denominator (no per-schedule den); a shallow shared-den cosine CANNOT reproduce a DEEPER LR
    descent by endpoint choice — the curvature differs (unlike the LINEAR β, which endpoint-rephases
    exactly; that is why β could be pinned with ``--hosc-beta-end`` but LR needs its own denominator).

    The crucible pins ``anneal_epochs=1000`` (the mod32cap CONTROL's den) so the AdamW LR(ep) on
    [1,726] is BIT-IDENTICAL to the control the window laws (ν, settle 237, s*, fire band ep675) were
    measured on — vs the shared den 3000, which runs the AdamW phase at 2.83× (ep675) → 3.41× (ep726)
    the control LR. ``hold_frac=1.0`` (the control had no LR hold before its Muon freeze at 726 < its
    den 1000) = no rescale = bit-identical cosine; ``< 1.0`` reaches ``--lr-end`` early and HOLDS,
    clamping the past-denominator cosine rebound. No ``epochs_delta``: this is a full-run config pin,
    NOT a warm-start isolation arm."""
    return Lever("C2_lr_anneal_pin",
                 overrides={"--lr-anneal-epochs": int(anneal_epochs),
                            "--lr-hold-frac": float(hold_frac)},
                 notes="LR-specific cosine denominator + hold (decouple from shared --anneal-epochs)")


def TailCycles(cycles_max: int = 5, start_epoch: int = 0,
               cycle_floor_epochs: float = 387.09, dwell_min: int = 237,
               tau_halving: float = 0.5, lr_prop_tau: float = 1.0,
               stop_marginal_s: float = 1e-4) -> Lever:
    """TAIL_k warm-restart refinement stage (crucible req L; DRAFT_OPTIMAL_STACK_v6 §2.2e; §row-9 τ_k).

    The post-Muon stage: K warm-restart cycles that each re-sharpen τ (``τ_k = max(τ_{k-1}·halving,
    τ*_k=m_q/ln5)`` clamped ≥ ``--softmax-temp-end``) and re-warm LR ∝ τ_k (moments NEVER reset),
    running until a per-cycle ``powerlaw_meat`` exit (or the ``cycle_floor`` fail-safe cap), then
    PowerPlay-stopping when a cycle's marginal ΔS/epoch falls below ``stop_marginal_s`` and hard-capping
    at ``cycles_max`` (req-B). The engine is :class:`tac.witness_control.tail_cycles.TailController`.

    DEFAULTS CONSUME the sealed laws (req T; full rows in ``tac.witness_control.tail_cycles.
    TAIL_CONSTANT_PROVENANCE``): ``cycle_floor_epochs=387.09`` = ``tail_cycle_floor_v1`` (settle 237 +
    150 dwell floor); ``dwell_min=237`` = ``settle_window_v1`` (3/ν @ ν(tau)=0.012653); ``cycles_max=5``
    = the sealed net cap floor((budget−FIN)/cycle_floor). ``tau_halving=0.5`` (SGDR geometric base, one
    octave/cycle) + ``stop_marginal_s=1e-4`` (the PowerPlay attribution floor; S1-R1: NET-ΔS/ep, the
    d_seg leg MINUS the coded-bytes rate cost, not d_seg-marginal-alone) are HARDCODED-WITH-WAIVER (no
    closed-form derivation; the req-T class-4, tagged not bare — S4-R2). The trainer flags are
    default-OFF (``--tail-cycles-max 0`` = byte-identical) so this factory is what ARMS the stage; it
    requires a Muon finisher (``--muon-start-epoch``) in the same program. Live-m_q (SC-3) is the owed
    render build — the factory arms the τ-halving fallback (``--tail-live-mq`` stays unset)."""
    from tac.witness_control.tail_cycles import validated_stop_marginal_s

    stop_value = validated_stop_marginal_s(stop_marginal_s)
    return Lever("tail_k_warm_restart",
                 overrides={"--tail-cycles-max": int(cycles_max),
                            "--tail-start-epoch": int(start_epoch),
                            "--tail-cycle-floor-epochs": float(cycle_floor_epochs),
                            "--tail-dwell-min": int(dwell_min),
                            "--tail-tau-halving": float(tau_halving),
                            "--tail-lr-prop-tau": float(lr_prop_tau),
                            "--tail-stop-marginal-s": stop_value},
                 notes="post-Muon warm-restart cycles (τ_k halving/live-m_q; LR ∝ τ_k, moments kept; "
                       "per-cycle powerlaw_meat exit; PowerPlay stop; k_max fail-safe)")


def SoftBoundary(beta: float = 2.0, window: int = 100) -> Lever:
    """Anti-aliased SOFT boundary (lower HOSC beta) — tests Signal's hypothesis that
    a soft edge carries sub-pixel boundary position through R better than a hard
    step (β→∞). Replaces the confounded constant-β≈16 'beta_steplim' arm (review H2)."""
    return Lever("soft_boundary",
                 overrides={"--hosc-beta": beta},
                 epochs_delta=window,
                 notes="soft anti-aliased edge (low beta) for sub-pixel R-survival")


def Beta2WindowRewarmup(beta2: float = 0.999, steps_per_epoch: int = 75,
                        floor: float = 0.1, shape: str = "cosine") -> Lever:
    """R-7 finisher 1 (T5 crucible seal-round-1 structure lens): DERIVE the stage-transition LR
    re-warmup window from the Adam β2 second-moment MEMORY horizon 1/(1-β2) steps, so the LR ramp
    spans exactly long enough for the (reset) 2nd-moment estimate to re-converge to the post-transition
    gradient scale instead of dividing full-LR steps by stale/unconverged moments.

    ARCHAEOLOGY (what already existed — this lever does NOT re-build the mechanism): the trainer already
    has the re-warmup RAMP (``_stage_rewarmup_factor`` + ``--stage-transition-rewarmup-{epochs,floor,
    shape}``), an EVENT-fired boundary (``last_boundary_epoch`` set in the main loop at every stage
    transition), and the SIZING LAW callable (``rewarmup_beta2_memory_window_v1`` →
    ``min_rewarmup_epochs(beta2, steps_per_epoch)``). What was unbuilt is a COMPOSABLE lever that
    turns the previously-HARDCODED ``rewarmup_epochs`` literal into a DERIVED-AT-CONFIG value.

    DERIVED-AT-CONFIG (no bare literal): the window = ``ceil((1/(1-β2))/steps_per_epoch)`` via the law
    callable. DEFAULT-OFF: this lever is opt-in (not in any base program); un-composed => the base's
    rewarmup flags are untouched. ``floor=0.1`` + ``shape=cosine`` are the LAW's declared PROVISIONAL
    profile (``rewarmup_beta2_memory_window_v1`` domain: "cosine shape + floor 0.1 underived"), NOT a
    new invention. Pairs with ``--stage-transition-reset-moments`` (the law's premise: with moments
    reset, v re-accumulates over the 1/(1-β2) window)."""
    from tac.canonical_equations.curriculum_derivation_laws_20260705 import min_rewarmup_epochs
    win = int(min_rewarmup_epochs(float(beta2), int(steps_per_epoch)))
    return Lever(
        "R7_beta2_window_rewarmup",
        overrides={
            "--stage-transition-rewarmup-epochs": win,
            "--stage-transition-rewarmup-floor": float(floor),
            "--stage-transition-rewarmup-shape": str(shape),
            "--stage-transition-reset-moments": True,
        },
        notes=(f"β2-window LR rewarmup: win={win}ep = ceil((1/(1-{beta2}))/{steps_per_epoch}) "
               "DERIVED-AT-CONFIG per rewarmup_beta2_memory_window_v1 (law min_rewarmup_epochs); "
               "floor/shape = the law's PROVISIONAL profile; pairs with reset-moments"))


def PolyakFinisher(start_epoch: int = 0) -> Lever:
    """R-7 finisher 2: uniform Polyak/Ruppert TAIL average of the live iterates over the finishing
    window, exported as an ADDITIONAL checkpoint candidate ALONGSIDE the EMA shadow (NEVER replaces it
    — the EMA non-negotiable stands). At the SEALED constant-τ* TURNPIKE the iterates orbit a basin
    center; the uniform tail mean averages the orbit out to O(1/√n) — a strictly better basin-center
    than a fixed-horizon EMA (which still carries orbit phase). The byte-close/eval stop-time checklist
    picks whichever candidate MEASURES better (that measurement is NOT the lever's job).

    DEFAULT-OFF (``--polyak-finisher-arm`` store_true off): un-composed => the averager is never
    constructed => ZERO new checkpoint keys => byte-identical. ``start_epoch`` is a config choice (size
    it from the stage window via ``tac.witness_control.polyak_finisher.polyak_finisher_window_provenance``
    ~0.1-0.3× stage per ``muon_finisher_schedule_warmstart_and_lr_anneal_v1``; default 0 arms from run
    start). No ``epochs_delta``: this rides the existing finishing epochs, it does not extend them."""
    return Lever(
        "R7_polyak_finisher",
        overrides={
            "--polyak-finisher-arm": True,
            "--polyak-finisher-start-epoch": int(start_epoch),
        },
        notes=("uniform tail (Polyak/Ruppert) average → extra ckpt candidate alongside EMA (never "
               "replaces it); byte-close picks best per "
               "muon_finisher_schedule_warmstart_and_lr_anneal_v1 (finisher-EMA=Polyak)"))


def FiLMFix(per_layer: bool = True, concat_code: bool = True,
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


def LanePrior(weight: float = 1.0, start_epoch: int = 300,
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


def AnalyticLaneRenderBand(
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
    and consumed by the compose wire-in (the #224 Option-B lane-band path). Because composition
    happens inside the shared ``render_fn`` before frames are stacked, ``MicroBatch(B>1)`` consumes
    the same analytic band without a duplicate loss term; the B=2/B=4 render-composition parity gate
    is the authority for that shared-path claim."""
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
                       "witness-uncertainty); shared render_fn composes before MicroBatch stacking; "
                       "FP-killed non-naive form; realized THROUGH R")


def AnalyticLaneBandTraining(
    softness: float = 1.0, dash_forward_max_m: float = 55.0,
    uncertainty_source: str = "witness", tau: float = 0.85, eps: float = 0.35,
    weight: float = 1.0, window: int = 0,
) -> Lever:
    """v7.5.3 Δ3 — the analytic openpilot lane band as a **TRAINING lever** (RANK-1 negcure join):
    the band participates in the TRAINED render FROM EPOCH 0 (``start_epoch=0``) so the witness
    RE-ADAPTS its boundaries with the band active — NOT a post-hoc carrier composited only at
    byte-close. This is the distinction the ``render-post-hoc-dead`` law forces: the post-hoc
    (start-late / carrier-only) verdict was NET-NEUTRAL because the frozen witness never trained
    WITH the band; trained-in predicts ``fn_recovered ≳ 2e-4`` (the pre-registered P7 falsifier —
    < 2e-4 at n600 ⇒ refuted at FORMULATION scope, band stays carrier-only).

    NO NEW TRAINER FLAG (never-invent-flags): the trained-in mode is EXACTLY the existing
    :func:`AnalyticLaneRenderBand` machinery (``--lane-render-band`` + the 6 ``--lane-band-*``
    flags, LANDED and consumed by the #224 Option-B compose wire-in) pinned to
    ``--lane-band-start-epoch 0`` — the epoch that makes the band active in the TRAINING loss from
    the start rather than a late-engaged carrier. A DISTINCT NAMED lever (its own activation-ledger
    row) expressing the trained-in intent, reusing the render-band flags rather than duplicating them.

    DEFAULT-OFF: composing this lever is what turns the band trained-in; un-composed the trainer
    renders the byte-identical default (no band). ``window`` = optional warm-start epochs (0 = a
    full-run-from-scratch arm). means != ends: builds the mechanism, makes NO score claim — the
    fn-recovery is ASSUMED_AWAITING_VERIFICATION until the RANK-1 trainer A/B measures it at n600."""
    return Lever("analytic_lane_band_training",
                 overrides={"--lane-render-band": True,
                            "--lane-band-softness": softness,
                            "--lane-band-dash-forward-max-m": dash_forward_max_m,
                            "--lane-band-uncertainty-source": uncertainty_source,
                            "--lane-band-tau": tau,
                            "--lane-band-eps": eps,
                            "--lane-band-weight": weight,
                            "--lane-band-start-epoch": 0},
                 epochs_delta=window,
                 notes="v7.5.3 Δ3: analytic lane band TRAINED-IN from ep0 (RANK-1 join; the witness "
                       "re-adapts its boundaries with the band active; render-post-hoc-dead law)")


def DsegAwareTaper(
    strength: float = 1.0, scale: float = 0.0, floor: float = 0.05,
    *, scientific_declaration: bool = False,
) -> Lever:
    """#121 d_seg-aware Fourier-feature amplitude taper: reweight each FIXED Fourier/curvelet basis
    column's amplitude by the GT d_seg saliency (top1-top2 SegNet argmax MARGIN) field, moving the
    coord-INR's spectral prior toward the boundary annulus where d_seg is decided — the spectral
    analogue of the vendored ``configurable_taper_decoder`` capacity reallocation, on the witness's
    OWN Fourier basis instead of the HNeRV channel schedule. BYTE-NEUTRAL (adds ZERO trainable
    params → archive unchanged) + rule-118 FREE (a deterministic prior from the GT margin geometry,
    recomputable at decode like the self-orient basis / lane band). DEFAULT-OFF: this factory is what
    ARMS it; un-composed the trainer never applies the taper (byte-identical).

    ``strength`` scales the reallocation (0 ⇒ flat taper = no-op); ``scale`` = the saliency exp-kernel
    width in margin units (0 ⇒ AUTO = median ``|margin|``, the robust default); ``floor`` = the
    positivity clamp on each per-column weight. Mechanism:
    ``tac.boundary_math.dseg_aware_fourier_taper``; law:
    ``tac.canonical_equations.dseg_aware_fourier_taper_20260709``.

    NO warm-start window: the taper is STRUCTURAL (active from ep0 by construction — it changes the
    input feats the ``in_proj`` is trained on), so an end-of-run resume that ADDS/CHANGES it is
    REFUSED by the trainer's F2 resume-divergence guard (a basis change), not silently applied; an
    A/B fires it from a fresh run (or a matching resume).

    VERDICT SCOPE (ledger #121): "+18% NO-GO RETRACTED (under-converged ge300/3000); converged anchors
    flip sign to -8% ~0.03; RE-VALIDATE at convergence (cheap disk A/B)" — the d_seg effect is
    ASSUMED_AWAITING_VERIFICATION until a CONVERGED byte-close A/B measures it. means != ends: this
    factory BUILDS the mechanism; it makes NO score claim; pointer UNMOVED."""
    overrides = {"--dseg-aware-taper": True,
                 "--dseg-aware-taper-strength": float(strength),
                 "--dseg-aware-taper-scale": float(scale),
                 "--dseg-aware-taper-floor": float(floor)}
    lawrefs: dict = {}
    constant_manifest: dict = {}
    receipt_schemas: dict = {}
    if scientific_declaration:
        lawrefs, constant_manifest = _v9_scientific_constant_custody(
            "dseg_aware_fourier_taper_reweight_v1",
            overrides,
            provenance=(
                "V9 TAPER control declaration: enabled 1, strength 1, AUTO scale 0, "
                "positive floor 0.05; rank-1 78.9% treatment removes this whole Lever"
            ),
        )
        receipt_schemas = dict.fromkeys(overrides, "v9_config_compile.v1")
    return Lever(
        "dseg_aware_taper",
        overrides=overrides,
        notes="#121 d_seg-aware Fourier-feature amplitude taper (byte-neutral spectral "
              "reallocation by GT margin saliency; RE-VALIDATE at convergence)",
        lawrefs=lawrefs,
        constant_manifest=constant_manifest,
        runtime_receipt_schemas=receipt_schemas,
    )


def AdaptiveGradClip(
    percentile: float = 10.0, window: int = 1000, warmup_steps: int = 10,
    *, scientific_declaration: bool = False,
) -> Lever:
    """#B-4 grad-clip cure: AutoClip percentile clip law (arXiv:2007.14469) replacing the
    MEASURED-SATURATED fixed ``--grad-clip 0.5``.

    THE TELEMETRY (audit ``.omx/research/v9_missing_signal_constants_audit_20260715.md`` §A-1,
    MEASURED on C0 ``levelset_n600_witness_20260715T095030Z``): ``grad_clip_activation`` rows
    ep1-39 show global ``frac_clipped=1.0`` at EVERY accum step with ``norm_mean≈5.9-6.2``
    (max 17.5) vs threshold 0.5 — the clip is saturated 100% of CE-stage steps.

    HONEST MECHANISM STATE (fresh-eyes F5 correction, 2026-07-15): the "effective step
    ``lr·0.5/‖g‖ ≈ lr/12``, LR cosine dethroned" mechanism is REFUTED on the C0 per-param-
    normalize lineage — normalize runs on the already-clipped tree and divides out any uniform
    norm scaling, so the saturation was INERT there (telemetry != mechanism;
    ``perparam_normalize_masks_all_norm_clipping_c0_confound_20260715``). The magnitude
    mechanism can bind ONLY on normalize-none arms (formulation-scoped; the n24 maglaw A/B is
    the lineage test), and the trainer REFUSES autoclip x per-param normalize outright
    (armed-but-inert lever = orphaned signal).

    THE LAW: ``clip_t = percentile_p(‖g‖ history, window w)`` — observe-then-threshold per
    accum step; the fixed ``--grad-clip`` is the warmup fallback. With ``--per-group-grad-clip``
    each top-level parameter group gets its OWN percentile threshold (the C4 anti-starvation
    sibling preserved). Constants: ``percentile=10`` is the AutoClip paper default; window/warmup
    are stage-tracking pragmatics — all three custodied as scientific declarations below.

    Training-path lever under the 2026-07-15 relaxed-identity directive (drift OK if gradient
    quality + no flicker); decode/verdict/byte-close paths untouched. DEFAULT-OFF in the trainer
    (``--grad-clip-mode fixed`` = byte-identical incumbent); this factory is what ARMS it.
    ``scientific_declaration`` defaults FALSE (the DsegAwareTaper precedent): the launcher's
    ``--dsl-lever`` compose path currently fails the #406 self-recompile for lawref-CARRYING
    internal levers ('inputs must be a non-empty mapping' round-trip bug, 2026-07-15 —
    memo wallclock_burndown_build_20260715.md §2c sister); spec-authored configs pass True
    through the symmetric ``_typed_ideal_lever`` codec for full LawRef custody.
    Mechanism: ``tac.witness_control.adaptive_grad_clip`` (resume-safe under ``__acl_``). Law:
    ``tac.canonical_equations.autoclip_percentile_grad_clip_20260715``
    (``autoclip_percentile_threshold_v1``).

    means != ends: ARMS the mechanism; NO score claim. The descent-speed effect is
    ASSUMED_AWAITING_VERIFICATION until the bounded n24 A/B (d_seg descent per WALL-CLOCK,
    gradient-quality + no-flicker admission bar) and any promotion needs a byte-closed exact
    n600 row. Pointer UNMOVED."""
    numeric = {"--grad-clip-percentile": float(percentile),
               "--grad-clip-window": int(window),
               "--grad-clip-warmup-steps": int(warmup_steps)}
    overrides = {"--grad-clip-mode": "autoclip", **numeric}
    lawrefs: dict = {}
    constant_manifest: dict = {}
    receipt_schemas: dict = {}
    if scientific_declaration:
        lawrefs, constant_manifest = _v9_scientific_constant_custody(
            "autoclip_percentile_threshold_v1",
            numeric,
            provenance=(
                "AutoClip percentile clip law (arXiv:2007.14469): percentile 10 = paper default; "
                "window 1000 / warmup 10 = stage-tracking pragmatics. Responds to the C0-measured "
                "frac_clipped=1.0 saturation telemetry; the lr/12 magnitude mechanism is REFUTED "
                "on per-param-normalize lineage (inert there; trainer refuses that composition) "
                "and can bind only on normalize-none arms (maglaw A/B attribution owed)"
            ),
        )
        receipt_schemas = dict.fromkeys(numeric, "v9_config_compile.v1")
    return Lever(
        "adaptive_grad_clip_autoclip",
        overrides=overrides,
        notes="#B-4 AutoClip percentile grad-clip (cures measured saturation; epochs-to-target "
              "lever under the joint wall-clock objective; n24 A/B owed)",
        lawrefs=lawrefs,
        constant_manifest=constant_manifest,
        runtime_receipt_schemas=receipt_schemas,
    )


def GradNormalizeNone() -> Lever:
    """#B-4 A/B arm-B companion: DISARM per-parameter gradient normalization (--grad-normalize
    none) so a norm-clip law actually controls the update magnitude.

    THE CONFOUND THIS EXPRESSES (memo wallclock_burndown_build_20260715.md §2b, FORMULATION,
    source-verified): the live v9 configs run ``--grad-normalize per-param``
    (``tac.witness_stability.per_param_normalize_grads``: g_p <- g_p/(||g_p||+eps) per tensor,
    applied AFTER the clip) — a uniform per-tensor scale is divided out exactly, so ANY norm
    clip (fixed 0.5 / per-group / AutoClip) is a NO-OP on the applied update. The C0 clip
    saturation telemetry was real but INERT. The honest epochs-to-target A/B is magnitude-LAW
    vs magnitude-LAW: incumbent (per-param) vs [this lever + AdaptiveGradClip] vs
    [this lever + fixed clip]. per-param-normalize's own docstring: "ALTERS the seg-vs-pose
    gradient SCALE ratio ... NOT proven for our objective" (an owed A/B since #146).

    means != ends: an A/B arm definition; NO score claim; pointer UNMOVED."""
    return Lever("grad_normalize_none",
                 overrides={"--grad-normalize": "none"},
                 notes="#B-4 arm-B companion: disarm per-param normalize so the clip law "
                       "controls magnitude (per-param masks ALL norm clipping; memo §2b)")


def CorrectedWeightDecay() -> Lever:
    """AdamC schedule-corrected decoupled weight decay on the AdamW TRUNK (arXiv:2506.02285).

    THE LAW (Defazio 2025, read-in-full 2026-07-15; memo
    ``.omx/research/adamc_muonc_optimizer_research_20260715.md``): for weights whose update
    magnitude is weight-independent, decoupled wd drives ``||g||/||x||`` to the Van Laarhoven
    steady state ``sqrt(2*lambda/gamma_t)`` — a DECAYING lr schedule therefore RAISES the target
    like ``1/sqrt(gamma_t)`` (the tail gradient-norm blow-up + weight-norm collapse the paper
    measures at LLM scale). The correction ``lambda_hat_t = lambda * gamma_t/gamma_max`` pins the
    steady state at ``sqrt(2*lambda/gamma_max)`` — schedule-independent. Trainer mechanism:
    per-epoch ``opt.weight_decay = _corrected_weight_decay(...)`` after every trunk lr driver.

    ASSUMPTION FORK (why this is DEFAULT-OFF with a PREDICTED-NULL): their setting is
    large-scale pretraining with lambda=0.05 and a full cosine tail (mechanism strength
    ``lambda*sum(gamma_t) >~ O(1)``, weight norms moved ~70%); OUR premises are n=1 overfit,
    lambda=1e-4, no normalization layers (Chou arXiv:2512.08217 independence form carries the
    transfer; near-exact under per-param grad normalize) => mechanism strength ~2e-4 (n24) ..
    ~1e-2 (n600 full) => PREDICTED-NULL (memo P1). The bounded n24 A/B is the CARGO-CULT GUARD:
    a null keeps the correction out of the sealed config on evidence; a NON-null falsifies our
    effective-decay-channel accounting (a bigger finding than the lever). TRUNK-ONLY: the MLX
    Muon finisher's wd is coupled-through-NS ~= INERT (arming a scaled version would be the
    #417 counted-but-inert fake) — the decoupled+corrected Muon path is an
    ``adaptivization_tickets_20260715`` ticket, never a hand flag.

    means != ends: ARMS the mechanism; NO score claim; verdict_scope FORMULATION;
    [macOS-MLX research-signal] NON-PROMOTABLE. Pointer UNMOVED."""
    return Lever(
        "corrected_weight_decay_adamc",
        overrides={"--weight-decay-corrected": True},
        notes="AdamC lambda_hat_t = lambda*lr_t/lr_max on the AdamW trunk (arXiv:2506.02285); "
              "PREDICTED-NULL at live lambda=1e-4 — the n24 A/B is the cargo-cult guard "
              "(memo adamc_muonc_optimizer_research_20260715 P1); Muon side ticketed",
    )


def LaneBandStaticCache(enabled: bool = True) -> Lever:
    """#509 burn-down 3: cache the PAIR-STATIC lane-band constants (weighted stop-grad coverage
    per unique prior + gt-source u_mask per code) across compose calls, killing the per-call
    numpy->mx conversion once the band gate opens (measured +75 s/ep from ep33 on C0; the cache
    recovers the CONVERSION share — the theta-dependent witness margin/appearance forwards are
    intrinsic to the lever and stay per-call).

    Values are BIT-IDENTICAL by construction (same source array -> same mx constant, computed
    once and reused — the ``_cf_mx`` cache precedent), so this is a SCORE-NEUTRAL speed lever:
    trainer default ON per the off-is-orphan rule; this factory exists to compose the OFF arm
    (``enabled=False``) for the paired A/B and to keep the flag DSL-held (never a hand flag).
    Mechanism: ``tac.boundary_math.analytic_lane_render_band.make_lane_band_compose_fn``
    (``cache_static``). Memory: one (H,W) fp32 per unique prior (~0.5 GiB @ n600), inside the
    launcher memory-preflight envelope. means != ends: sec/ep lever; NO score claim; pointer
    UNMOVED."""
    return Lever("lane_band_static_cache",
                 overrides={"--lane-band-cache-static": bool(enabled)},
                 notes="#509 pair-static lane-band constant cache (bit-identical values; "
                       "sec/ep lever; OFF arm = the A/B control)")


def VerdictParallelWorkers(workers: "int | None" = None) -> Lever:
    """#509 burn-down 2 / m5max unconstrained-leverage constraint 4 (2026-07-15): fan the
    ADVISORY CPU-torch verdict's ``--verdict-batch`` chunks across ``workers``
    ThreadPoolExecutor threads (idle CPU cores).

    The 1-thread law (operator_1thread_training_standard_20260713) binds the TRAINING
    determinism path; the verdict is ADVISORY (never read back into training), so
    idle-core parallelism is legal. BIT-IDENTICAL values by construction: same chunk
    spans, same per-chunk ``cpu_verdict_*`` calls with unchanged torch intra-op thread
    count, ``Executor.map`` chunk-index-order aggregation => the same float sequence and
    mean as the sequential loop. HONEST SCOPE (bench receipt 20260715T184252Z + fresh-eyes
    F4): the lever divides the SCORER-FORWARD share of the verdict — MEASURED 370.6s -> 65.2s
    at w=8 (5.686x, eta_8=0.711, values float-identical) — NOT the whole ~2555.7 s C0 in-run
    verdict wall (audit §D.2), which is dominated by the un-parallelized render/realized
    stages (a SEPARATE owed lever). The per-dim mdd-ablation sweep (~443 s of the measured
    630 s n24 verdict-epoch tail) inherits this same pool via per_dim_dseg_ablation(workers=).

    Memory: the #205 per-chunk transient is multiplied by chunks in flight (<= workers);
    size against free-RSS headroom (safe-run guard + launcher memory preflight stay).
    Default-OFF at the trainer (0 = sequential, byte-identical); this factory is the
    DSL custody so the flag is never hand-typed. ``workers=None`` (the default) DERIVES
    the count from measured headroom via the registered law
    (``verdict_parallel_workers_speedup_v1``:
    ``derived_verdict_workers`` — base + w*2.9 GiB <= 0.70*available, capped at the
    measured ladder top w=8; value-provenance DERIVED, never a hand count). The wired
    trainer path self-checks bit-identity on its FIRST parallel verdict (chunk-0
    sequential recompute; mismatch => confound_alarm + permanent sequential fallback).
    means != ends: sec lever on the advisory verdict path; NO score claim; pointer
    UNMOVED."""
    if workers is None:
        from tac.canonical_equations.verdict_parallel_workers_speedup_20260715 import (  # noqa: PLC0415
            derived_verdict_workers,
        )
        workers = derived_verdict_workers()
    w = int(workers)
    if w < 2:
        raise ValueError(
            f"VerdictParallelWorkers requires workers >= 2 (got {w}): 0/1 is the incumbent "
            "sequential path — compose nothing instead of a no-op lever (off-is-orphan rule: "
            "an inert composed lever is orphaned signal).")
    return Lever("verdict_parallel_workers",
                 overrides={"--verdict-parallel-workers": w},
                 notes="#509 chunk-parallel ADVISORY CPU verdict (bit-identical values; "
                       "verdict-wall sec lever; OFF arm = sequential control)")


def ComputeDtype(dtype: str = "bf16", quality_check: int = 0) -> Lever:
    """#509 burn-down batch 3 / m5max unconstrained-leverage constraint 1 (2026-07-15): the
    bf16/fp16 mixed-precision COMPUTE SEAM — fp32 master weights + low-precision witness
    forward/backward ONLY.

    THE BUILD-OWED THIS ARMS (memo ``wallclock_burndown_build_20260715.md`` §3): no dtype seam
    existed in either trainer; M-series GPUs run bf16/fp16 at ~2x the fp32 rate (ESTIMATE-flagged
    — the ceiling row becomes measured only via the paired bench this lever enables). Mechanism:
    ``tac.witness_control.compute_dtype_seam.ComputeDtypeSeam`` — masters cast to the compute
    dtype INSIDE the traced loss (gradients return fp32 through the astype VJP); module entry
    shims cast inputs down / outputs back UP to fp32 so the render/R (incl. the fused-R Metal
    kernel), FROZEN-SCORER forwards, verdict, EMA, checkpoints, and decode stay fp32. Masters
    restored after every call => resume-safe by construction (nothing new persisted). Law leg:
    ``tac.canonical_equations.mixed_precision_compute_seam_20260715``
    (``bf16_compute_seam_gradient_quality_v1``).

    ``quality_check=N`` arms the ADMISSION GATE (the C0 lesson,
    ``perparam_normalize_masks_all_norm_clipping_c0_confound_20260715``): the first N optimizer
    steps compute BOTH dtype arms from the same masters, compare the POST-normalize update
    direction (cosine + rel-norm — per-param normalize divides out uniform per-tensor scales,
    so a pre-normalize comparison grades a direction the optimizer never sees), and step with
    the fp32 REFERENCE (the seam is measured along the reference trajectory). Receipts:
    ``compute_dtype_quality.jsonl``. Trainer refuses QC with autoclip / seed-islands / the
    per-group+normalize-none pairing (un-replicated pipeline stages would contaminate the
    comparison).

    bf16 is the recommended arm (fp32-range exponent; NO loss scaling is implemented — fp16 is
    exposed for the measurement matrix only). Trainer default ``--compute-dtype fp32`` =>
    byte-identical (the seam is never constructed); this factory is what ARMS it. Training-only
    drift under the 2026-07-15 relaxed-identity directive (drift OK if gradient quality + no
    flicker); decode/verdict/byte-close untouched.

    means != ends: a sec/ep lever candidate; the speed AND gradient-quality effects are BOTH
    ASSUMED_AWAITING_VERIFICATION until the bounded n24 QC run + paired sec/ep bench land;
    NO score claim; pointer UNMOVED."""
    d = str(dtype)
    if d not in ("bf16", "fp16"):
        raise ValueError(
            f"ComputeDtype requires dtype in ('bf16','fp16') (got {d!r}): fp32 is the incumbent "
            "path — compose nothing instead of a no-op lever (off-is-orphan rule).")
    qc = int(quality_check)
    if qc < 0:
        raise ValueError(f"ComputeDtype quality_check must be >= 0 (got {qc})")
    overrides: dict = {"--compute-dtype": d}
    if qc > 0:
        overrides["--compute-dtype-quality-check"] = qc
    return Lever("compute_dtype_seam",
                 overrides=overrides,
                 notes="#509 bf16/fp16 compute seam (fp32 masters; low-precision witness "
                       "fwd/bwd only; scorer/verdict/decode fp32; QC gate = post-normalize "
                       "update-direction cosine vs fp32 reference)")


# QC measurement window: enough per-step rows for a stable median + tail read of the
# cosine/rel-norm distribution (>= 50 rows; 60 = 20 bounded n24 epochs at accum-pairs 8
# -> 3 optimizer steps/epoch). A window length (CONFIG-class), not a scientific constant;
# the ADMISSION thresholds live in the law module (bf16_compute_seam_gradient_quality_v1).
COMPUTE_DTYPE_QC_WINDOW_STEPS = 60


def ComputeDtypeBf16QCGate() -> Lever:
    """#509 batch 3: the ZERO-ARG composable arm for the OWED bounded-n24 bf16 QC anchor —
    ``--dsl-lever ComputeDtypeBf16QCGate`` through the governed launcher (the launcher's
    ``--dsl-lever`` accepts only zero-arg factories, and the parameterized ``ComputeDtype``
    cannot express the QC gate's config COMPATIBILITY set by itself).

    Composes, in ONE lever, everything the trainer's QC admission requires
    (train_levelset ``_cdt_qc_n`` refusal cascade):
    - ``--compute-dtype bf16`` + ``--compute-dtype-quality-check 60`` (the seam + the gate;
      window = ``COMPUTE_DTYPE_QC_WINDOW_STEPS``);
    - ``--grad-clip-mode fixed`` (QC replicates only the fixed-clip pipeline; autoclip's
      history state would be double-fed by the dual grad computes);
    - seed-islands OFF (``--seed-islands/--seed-island-eased/--witness-alone-island-loss``
      False -> ``--no-*`` negation tokens; the reference recompute has no dual seed leg).
      The FreSh matched-control lever set the precedent for this exact off-triple.

    The QC mode STEPS WITH THE FP32 REFERENCE, so the bounded run's trajectory is the
    fp32 one — this arm produces the ``compute_dtype_quality.jsonl`` distribution that
    turns the law's PROPOSED cos_min 0.99 / rel-band [0.9, 1.1] into measured admission
    thresholds. means != ends: an apparatus/measurement arm; NO score claim; pointer
    UNMOVED."""
    return Lever(
        "compute_dtype_bf16_qc_gate",
        overrides={
            "--compute-dtype": "bf16",
            "--compute-dtype-quality-check": int(COMPUTE_DTYPE_QC_WINDOW_STEPS),
            "--grad-clip-mode": "fixed",
            "--seed-islands": False,
            "--seed-island-eased": False,
            "--witness-alone-island-loss": False,
        },
        notes=("#509 bounded-n24 bf16 QC anchor arm: seam + 60-step QC window + the trainer's "
               "full QC compatibility set (fixed clip, seed-islands off); steps with the fp32 "
               "reference"),
    )


def HardnessOversample(
    oversample: float = 0.5, weighted: bool = True, source: str = "realized",
    power: float = 1.0, band: float = 0.5,
) -> Lever:
    """LEVER-5 hard-pair emphasis DATA curriculum — the built-in-the-trainer, DSL-orphaned-until-this-
    landing candidate (curriculum-candidate pool §2.1, task #403). Each epoch keeps the FULL
    ``permutation(P)`` (every pair ≥1 step, never starved) PLUS ``round(P*oversample)`` EXTRA per-pair
    code-fit steps drawn ~ per-pair hardness^power — giving a hard pair MORE update STEPS (not a bigger
    loss scale, which Adam normalizes to ~no-op). The fair A/B at fixed ``oversample`` is
    ``weighted`` on (extras ~ hardness) vs off (extras uniform): SAME total steps, different allocation.

    ``source`` ∈ {margin, realized}: ``margin`` = $0 cached-GT small-margin pixel fraction (MEASURED
    weak 1.31× spread, trainer L11306); ``realized`` = one-time per-pair baseline realized d_seg (the
    SHARPER signal the trainer recommends for the code-fit). DEFAULT-OFF in the trainer
    (``--hardness-oversample 0.0``) ⇒ byte-identical when this lever is not composed; this factory is
    what ARMS it (oversample>0 + weighted + realized source).

    MEASURED anchors: 44%-of-CE-residual-spikes-are-LANE (#205 CE-floor, L67) + margin-saliency #141.
    means != ends: this factory ARMS the mechanism; it makes NO score claim; the pointer is UNMOVED
    until a byte-closed n600 A/B measures it (curriculum-pool duty-to-measure row)."""
    if source not in ("margin", "realized"):
        raise ValueError(f"--hardness-source must be 'margin' or 'realized', got {source!r}")
    return Lever("hardness_oversample_lever5",
                 overrides={"--hardness-oversample": float(oversample),
                            "--hardness-weighted": bool(weighted),
                            "--hardness-source": str(source),
                            "--hardness-power": float(power),
                            "--hardness-band": float(band)},
                 notes="LEVER-5 hard-pair emphasis data curriculum (extra code-fit steps ~ per-pair "
                       "hardness; fair A/B = weighted on/off at fixed oversample; advisory until byte-closed)")


def HeadGeometry(
    head: str = "etf", additive_margin: float = 0.0,
) -> Lever:
    """#218 facet-1 out_sdf HEAD GEOMETRY — the BYTE-FREE rare-class lane-margin fix (curriculum-
    candidate pool §2.4, task #403). Selects the classifier-head geometry:

      * ``etf`` (the composable default) — fixed simplex-ETF weight (frozen): byte-free + rate-win +
        the neural-collapse minority-norm fix (raises the Lane/Movable minority-class norm the softmax
        head under-allocates). FIRE FIRST (byte-free).
      * ``additive-margin`` — use the AM realized-margin hinge target from ``additive_margin`` (the
        target realized SegNet decision margin). NOTE the AM-hinge additionally needs
        ``--margin-field-head-weight>0`` — compose ``MarginFieldHead`` alongside this lever to arm it.
      * ``softmax`` — the default head (byte-identical; a no-op arm for the A/B).

    DEFAULT (in the trainer) is ``--head softmax`` + weight 0 ⇒ byte-identical; this factory ARMS the
    ETF geometry. The already-held sisters ``MarginFieldHead`` (facets 1b/3 loss weight) +
    ``PersistenceTopology`` (soft-clDice) compose with it on the shared ``_signed`` field. Mechanism +
    probe: ``src/tac/boundary_math/laguerre_logit_offset.py`` +
    ``experiments/probe_laguerre_logit_offset_sweep.py``.

    means != ends: byte-free head geometry; NO score claim; the ETF rate-win + minority-norm fix are
    the DERIVED #218 rationale, advisory until a byte-closed n600 A/B measures d_seg (pool duty row)."""
    if head not in ("softmax", "etf", "additive-margin"):
        raise ValueError(f"--head must be one of softmax/etf/additive-margin, got {head!r}")
    return Lever("head_geometry_218",
                 overrides={"--head": str(head),
                            "--additive-margin": float(additive_margin)},
                 notes="#218 facet-1 out_sdf head geometry (ETF frozen simplex-weight = byte-free "
                       "rate-win + neural-collapse minority-norm fix; AM needs MarginFieldHead composed)")


def LadderIslandHomotopy(
    amplify_weight: float = 1.0,
    absolute_scale: float = 2.0,
    movable_birth_epochs: int = 60, movable_hold_epochs: int = 0,
    movable_anneal_epochs: int = 200, movable_lambda_gate: float = 0.0,
    lane_birth_epochs: int = 80, lane_hold_epochs: int = 0,
    lane_anneal_epochs: int = 260, lane_lambda_gate: float = 0.0,
    gate_softness: float = 0.5, release_coeff: float = 0.95, sigma_eff: float = 1.5,
    lane_dash_gate: bool = True, max_step_px: float = 1.0, refresh_every: int = 25,
    window: int = 0,
) -> Lever:
    """#323 FULL LADDER island-birth lever — the per-class-λ-GATED homotopy the amplify/nucleus
    machinery only PARTIALLY realized. Drives the AMPLIFY island-birth support RADIUS by a per-epoch,
    per-class continuation (``tac.witness_curriculum.ladder_homotopy``) instead of the fixed
    ``--island-dilate-px``:

      * **movable arm — dilation-GO**: SDF forward-Euler dilation (proven transfer, 1-Lipschitz),
        ceiling'd by the critical-nucleus RELEASE law r*(t)=coeff·σ_eff (LawRef
        ``critical_nucleus_release_v1``; MEASURED dilation knee native 44.6% → +1px 90.0% → +2px 98.3%).
        Its λ-gate defaults OPEN (dilation-GO is sound independent of lane-share).
      * **lane arm — curve-prior**: grows support ALONG the openpilot VP-tangent (stays on the ~8-dim
        lane manifold; isotropic dilation of a curve is the measured NO-GO) with a dash-phase window.
      * **per-class-λ gate**: support flows to a class ONLY while its MEASURED costate
        λ_c = flip_share_c·d_seg_by_class_c (the #315 per-class verdict sensor) exceeds ``lambda_gate``;
        the soft-gate band (``gate_softness``) fades support out continuously as the class's residual is
        won. UNIFORM always-on amplification is the MEASURED net-negative anti-pattern (T3 islands-
        treatment symposium ``council_t3_symposium_islands_treatment_arm_20260706``: Δd_seg ∝
        n_big3 − n_isl) — this lever NEVER emits it.

    LADDER continuation form (CT-2 §6 bifurcation-control): starts EASED (support r0 = winnable variant),
    holds, then anneals r0→0 (transfer to the true argmax); the 1-Lipschitz stepper (``max_step_px``,
    pseudo-arclength Δr ≤ c/‖dθ/dλ‖) guarantees no hard switch. Modulates the AMPLIFY masks only
    (``--amplify-weight`` > 0; auto-forces class-aware eased masks) — the SEED keeps its own
    ``--seed-anneal-epochs`` transfer schedule (no double-application). DEFAULT OFF in the trainer
    (``--ladder-island-homotopy``) ⇒ byte-identical when this lever is not composed.

    ``release_coeff`` / ``sigma_eff`` seed the movable release ceiling (LawRef; req-T DERIVED-AT-CONFIG,
    re-derive on a σ_eff-probe change). The two ``*_lambda_gate`` defaults are 0.0 = OPEN (req-T
    DERIVED-AT-CONFIG, NOT bare literals — rows in ``tac.witness_curriculum.ladder_homotopy.
    LADDER_LAMBDA_GATE_PROVENANCE``: movable dilation-GO is sound independent of lane-share, and the
    lane arm's release is schedule+dash-phase, so both costate floors are left OPEN at launch — S4-R2).
    STAGGER (S2-REV-A): ``max(birth+hold+anneal) < muon_start`` is asserted by ``WitnessProgram.
    validate`` so an anneal window cannot silently run past the Muon finisher. ``window`` is a
    warm-start epochs_delta (0 = full-run config lever, the default).

    ``absolute_scale`` is the one positive finite radius-scale authoring knob. Lane is the reference
    class, so ``lane_r0 = absolute_scale``; Movable is resolved from the sealed n96 isoperimetric
    receipt by ``isoperimetric_birth_weight_scaling_v1``, giving
    ``movable_r0 = absolute_scale / 8.881199197033954``. Both argv scalars are produced by actual
    receipt-backed LawRef resolution; the LawRefs and resolution records remain machine-inspectable on
    the returned Lever and are never stringified into argv. The ratio is DERIVED geometry only: the
    absolute scale and efficacy remain unmeasured."""
    scale = float(absolute_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"LadderIslandHomotopy: absolute_scale must be finite and > 0, got {absolute_scale!r}"
        )

    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        ISLAND_BIRTH_RATIO_RECEIPT_PATH,
        ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
        ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID,
    )
    from tac.witness_dsl.lawref import (
        LADDER_DERIVED_AT_CONFIG,
        InputRef,
        LawRef,
        resolve,
    )

    def _radius_lawref(class_extract: str, class_name: str) -> LawRef:
        anchor_provenance = (
            "MEASURED frozen-n96 class perimeter/area geometry; content-addressed receipt chains "
            "to gt_n96.npz source custody"
        )
        return LawRef(
            equation_id=ISOPERIMETRIC_BIRTH_WEIGHT_EQUATION_ID,
            inputs={
                "absolute_scale": InputRef.literal(
                    scale,
                    "UNMEASURED positive DSL treatment scale; Lane is the reference class",
                ),
                "class_p_over_a": InputRef.anchor(
                    ISLAND_BIRTH_RATIO_RECEIPT_PATH,
                    class_extract,
                    f"{anchor_provenance}; class={class_name}",
                    expected_sha256=ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
                ),
                "reference_p_over_a": InputRef.anchor(
                    ISLAND_BIRTH_RATIO_RECEIPT_PATH,
                    "classes/lane/p_over_a",
                    f"{anchor_provenance}; reference=Lane",
                    expected_sha256=ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
                ),
            },
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        )

    radius_lawrefs = {
        "--ladder-lane-r0": _radius_lawref("classes/lane/p_over_a", "Lane"),
        "--ladder-movable-r0": _radius_lawref("classes/movable/p_over_a", "Movable"),
    }
    radius_resolutions = {
        flag: resolve(lawref, repo_root=_REPO_ROOT)
        for flag, lawref in radius_lawrefs.items()
    }
    lane_r0 = float(radius_resolutions["--ladder-lane-r0"].value)
    movable_r0 = float(radius_resolutions["--ladder-movable-r0"].value)
    if not math.isclose(lane_r0, scale, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "LadderIslandHomotopy: REFUSE inconsistent Lane reference resolution: "
            f"lane_r0={lane_r0!r}, absolute_scale={scale!r}"
        )

    return Lever("n323_ladder_island_homotopy",
                 overrides={"--ladder-island-homotopy": True,
                            "--amplify-weight": amplify_weight,
                            "--ladder-movable-r0": movable_r0,
                            "--ladder-movable-birth-epochs": movable_birth_epochs,
                            "--ladder-movable-hold-epochs": movable_hold_epochs,
                            "--ladder-movable-anneal-epochs": movable_anneal_epochs,
                            "--ladder-movable-lambda-gate": movable_lambda_gate,
                            "--ladder-lane-r0": lane_r0,
                            "--ladder-lane-birth-epochs": lane_birth_epochs,
                            "--ladder-lane-hold-epochs": lane_hold_epochs,
                            "--ladder-lane-anneal-epochs": lane_anneal_epochs,
                            "--ladder-lane-lambda-gate": lane_lambda_gate,
                            "--ladder-gate-softness": gate_softness,
                            "--ladder-release-coeff": release_coeff,
                            "--ladder-sigma-eff": sigma_eff,
                            "--ladder-lane-dash-gate": lane_dash_gate,
                            "--ladder-max-step-px": max_step_px,
                            "--ladder-refresh-every": refresh_every},
                 epochs_delta=window,
                 notes="#323 FULL LADDER island-birth: per-class-λ-gated continuation over the AMPLIFY "
                       "island radius (movable dilation-GO r*(t) release + lane VP-tangent curve-prior); "
                       "Lane/Movable r0 ratio DERIVED from sealed n96 P/A receipt; absolute scale and "
                       "efficacy UNMEASURED; never uniform amplification; realized THROUGH R; advisory "
                       "until byte-closed",
                 lawrefs=radius_lawrefs,
                 constant_manifest={
                     flag: resolution.to_dict()
                     for flag, resolution in radius_resolutions.items()
                 })


def SegFormUnifyTau() -> Lever:
    """DISSOLVE the discrete CE→tau_softplus loss-form switch into the ONE continuous τ-homotopy
    (witness-native schedule derivation ``.omx/research/witness_native_schedule_derivation_20260709.md``
    §1.2): the seg loss becomes the ONE family ``L_τ = τ·logsumexp(φ/τ) − φ_y`` with loss-τ COUPLED to
    the render softmax-temp anneal (``--softmax-temp-*``, geometric shape), floored at the resolution
    knee ``--softmax-temp-end`` (τ*).

    ``τ=1`` recovers cross-entropy EXACTLY (the "CE stage" is L_τ's τ≈1 arc); ``τ→0`` is the max-margin
    form (the "tau_softplus stage" is its τ→τ* arc). There is NO ep300 CE→tau switch — this removes the
    LAST PR95 stage bone (the ``_seg_form_for_epoch`` "PR95 d_seg sequence" hard dispatch) instead of
    easing it. The trainer flag ``--seg-form-unify-tau`` is a store_true default-OFF => byte-identical
    when this lever is not composed. COMPOSE with a geometric τ anneal (``--tau-anneal-shape geometric``,
    the derivation's element-2 flip) for the full derived schedule.

    MUTUALLY EXCLUSIVE with an explicit ``--tau-softplus-start-epoch`` (the trainer's
    ``validate_seg_form_unify_tau_config`` refuses both) — the discrete-switch epoch is meaningless once
    the switch is dissolved; the v7 config DROPS that flag entirely. No ``epochs_delta``: this is a
    full-run schedule reshape, not a warm-start isolation arm."""
    return Lever("seg_form_unify_tau",
                 overrides={"--seg-form-unify-tau": True},
                 notes="dissolve CE→tau_softplus into ONE continuous L_τ=τ·logsumexp(φ/τ)−φ_y "
                       "(loss-τ = render softmax-temp, floored at τ*); removes the last PR95 stage bone")


def TauAdvanceEvent(octaves: int | None = None,
                    min_dwell: int | None = None, max_dwell: int | None = None) -> Lever:
    """S6-R4 SELF-PACED τ-ADVANCE: convert the τ-anneal from an epoch-clock to an EVENT-driven
    GEOMETRIC OCTAVE LADDER (operator 2026-07-08: *"Why is there a fixed number of epochs if our
    schedule and curriculum are no longer supposed to be hardcoded like pr95"* — the anneal-epochs
    denominator that clocks τ(t) is the LAST clock-hardcoding in the v7 schedule).

    ``--tau-advance-mode event``: τ holds at each rung of the geometric ladder
    ``τ_k = start·(end/start)^(k/N)`` (the SAME values the geometric clock passes through at prog=k/N
    — event mode reuses the clock ladder VALUES; only the per-rung DWELL is event-driven) and advances
    to k+1 when the per-band RELAXATION sensor fires: a ``powerlaw_meat`` detector on the through-R seg
    loss WITHIN the current octave (dwell-gated, thin-data fail-safe). A per-octave MAX-DWELL is the
    LOUD fail-safe backstop (``cap_fired_before_event``, S5 falsification-relevant). Engine:
    ``tac.witness_control.tau_advance`` (pure + unit-tested).

    REQUIRES ``--tau-anneal-shape geometric`` (the derived shape; validated by the trainer's
    ``validate_tau_advance_config``). COMPOSE with :func:`SegFormUnifyTau` for the full derived
    schedule. Defaults (octaves/min/max-dwell = None) => DERIVED in the trainer from --anneal-epochs +
    --curriculum-min-stage-epochs (no bare literals); pass explicit values to override.

    COUPLINGS (handled by the trainer, not extra flags): β co-anneals on the octave fraction; the LR
    pin rides the octave fraction (LR ∝ τ-control progress, S6/DE); unify-τ loss-τ follows the render
    τ automatically; the ladder FREEZES at the Muon switch (no double-driver of τ vs the finisher/TAIL).

    The trainer flag ``--tau-advance-mode`` DEFAULTS to ``clock`` (byte-identical) => this lever is
    what OPTS IN to event mode; a program without it is the incumbent per-epoch anneal."""
    ov: dict = {"--tau-advance-mode": "event"}
    if octaves is not None:
        ov["--tau-octaves"] = int(octaves)
    if min_dwell is not None:
        ov["--tau-octave-min-dwell"] = int(min_dwell)
    if max_dwell is not None:
        ov["--tau-octave-max-dwell"] = int(max_dwell)
    return Lever("tau_advance_event",
                 overrides=ov,
                 notes="S6-R4 self-paced τ-advance: geometric octave ladder, per-band-relaxation "
                       "event dwell (powerlaw_meat within octave) + loud max-dwell backstop; requires "
                       "--tau-anneal-shape geometric; β/LR co-anneal on the octave fraction; freezes at Muon")


def SafeCompileRegions(regions: str = "all-certified",
                       manifest: str | None = None) -> Lever:
    """SAFE-COMPILE (#252): activate the determinism-first ``mx.compile`` layer for the
    manifest-CERTIFIED elementwise hot-loop regions (``tac.mlx_safe_compile``).

    ``mx.compile`` was measured-EXCLUDED from the R operator (fp-contraction flips the
    uint8-STE d_seg knife-edge; v7 audit lever #3). This lever compiles ONLY regions
    whose per-region certificate proves bit-equality (max|Δ|=0) AND cross-process
    determinism — score-neutral by construction (a certified region is bit-identical to
    the uncompiled path). ``regions``: "all-certified" (every CERTIFIED manifest row),
    "none"/"off" (the default trainer behaviour), or a comma-separated id list (each
    intersected with the CERTIFIED rows; uncertified ids fail-closed to OFF).

    v7.1 / next-arm (NOT launch-gating for v7): needs the v7 baseline trajectory as the
    A/B comparator (sister of D15 micro-batch). The certified-region manifest IS the
    evidence gate; default OFF => byte-identical.

    ``manifest`` (#252 v2): optional non-default path to the certification manifest
    (``--safe-compile-manifest``); default None keeps the trainer's canonical
    ``.omx/state/mlx_safe_compile_manifest.json``. Holding it here maps the flag in the
    lever_registry (the manifest is the fingerprint/device-scoped evidence surface the
    v2 per-chip trust closure reads — see tac.mlx_safe_compile)."""
    overrides = {"--safe-compile-regions": regions}
    if manifest is not None:
        overrides["--safe-compile-manifest"] = manifest
    return Lever("n252_safe_compile_regions",
                 overrides=overrides,
                 notes="#252 determinism-first mx.compile of manifest-CERTIFIED elementwise "
                       "regions (bit-equal + cross-process deterministic); score-neutral; "
                       "v7.1 arm, was OFF (--safe-compile-regions none); v2 holds "
                       "--safe-compile-manifest (per-chip fingerprint/device-scoped evidence)")


def FusedRKernel(window: int = 0) -> Lever:
    """FUSED-R KERNEL (#252/#348, memory L70): swap the pure-MLX R roundtrip for the fused Metal
    kernel (``metal_fused_r_operator``) — a SPEED-only, score-NEUTRAL compute lever, sister of
    ``GROUPED_BACKWARD`` (~17x) and ``SafeCompileRegions``. This is the ONE completeness gap P7
    flagged (SPEC_v75 open-items / L70): a score-neutral always-on compute lever that was built in
    the trainer but never held by the DSL.

    MEASURED VERDICT (memory L70 / #348, 2026-07-07): bit-IDENTICAL fwd to the numpy-fp32 authority
    (~1 ULP VJP) AND ~8% faster; the fixed-order VJP LOCALIZES the one MLX-GPU non-determinism op
    class (dup-index atomic scatter in reference-R UP-backward), so the FULL witness goes 0/28
    cross-process wall (N=10) with fused-R ON => bit-exact proofs are reproducible on
    GPU-with-fused-R (per-config parity check owed at n600). A startup per-chip parity gate
    (``assert_metal_matches_cpu_oracle``) fails CLOSED if the kernel is not bit-identical on this
    GPU. NO-FAKE: buys SPEED, never a score.

    ``--fused-r-kernel`` (BooleanOptionalAction, default OFF => byte-identical when unfired) is the
    swept intent; it REQUIRES ``--mlx-device gpu`` (a Metal kernel — compose only onto a gpu
    program). ``window=0`` = a compute-config change with no epoch budget of its own. Score-neutral,
    so held OFF here only because it is device-gated, not because it perturbs the score."""
    return Lever(
        "n252_fused_r_kernel",
        overrides={"--fused-r-kernel": True},
        epochs_delta=int(window),
        notes=("#252/#348 fused Metal R roundtrip (metal_fused_r_operator); bit-identical fwd + "
               "~1 ULP VJP + ~8% faster + localizes GPU non-determinism (0/28 cross-proc N=10); "
               "score-NEUTRAL SPEED lever (sister of GROUPED_BACKWARD); requires --mlx-device gpu"),
    )


def VerdictLiveGap(every: int = 1, window: int = 0) -> Lever:
    """Advisory EMA-vs-live verdict-gap observer (task #408 Q3).

    The trainer default is ``-1`` (automatic during the two-time-constant EMA
    warmup only). Constructing this named Lever selects explicit all-run every-Kth
    verdict observation. Its values are appended to verdict telemetry and never
    consumed by a controller, optimizer, checkpoint selector, or score pointer.
    It remains a Lever because the extra inference changes execution timing; the
    activation ledger must keep explicit all-run use in duty-to-measure.
    """

    cadence = int(every)
    if cadence <= 0:
        raise ValueError(f"VerdictLiveGap: every must be > 0, got {every!r}")
    return Lever(
        "verdict_live_gap",
        overrides={"--verdict-live-gap-every": cadence},
        epochs_delta=int(window),
        notes=("task-408 Q3 explicit all-run advisory live-vs-EMA verdict cadence; trainer "
               "auto-observes EMA warmup; extra inference only; never feeds training/controller "
               "decisions; duty-to-measure"),
    )


def PoseVerdictGate(
    canary_every: int = 8,
    banked_r1_dpose: float = 0.001610,
    window: int = 0,
) -> Lever:
    """Refuse the retired unbound banked-pose substitution lever."""

    del canary_every, banked_r1_dpose, window
    raise ValueError(
        "PoseVerdictGate is retired: no payload-bound pose cache exists; "
        "live PoseNet is required"
    )


def PoseVerdictGateDryStart(window: int = 0) -> Lever:
    """Refuse the retired dry-start for the unbound pose substitution."""

    del window
    raise ValueError(
        "PoseVerdictGateDryStart is retired: live PoseNet is required"
    )


def PoseBlindComputeGate(window: int = 0) -> Lever:
    """Task #495 compute-only gate for the existing two-phase pose finish.

    This lever does not substitute or cache a pose value.  While effective
    ``w_pose`` is zero it skips the training PoseNet graph and emits d_seg-only,
    score-ineligible progress verdicts.  The pre-loop anchor and every verdict
    after the existing pose-finish event use live PoseNet.  DEFAULT OFF leaves
    both training and verdict paths byte-identical.

    # NO_EQUATION_NEEDED: removes a zero-weight scorer graph and an ineligible
    # observer forward; it adds no loss term, controller law, or score value.
    """

    return Lever(
        "pose_blind_compute_gate",
        overrides={
            "--pose-training-compute-gate": True,
            "--verdict-pose-gate": True,
        },
        epochs_delta=int(window),
        notes=(
            "task-495 compute-only gate: skip PoseNet while pose-finish is blind; "
            "d_seg-only progress rows are score/selection-ineligible; live PoseNet "
            "restored at engagement; no banked d_pose"
        ),
    )


def ClosedLoopEikonalControl(
    eikonal_bump: float = 0.05, eikonal_max: float = 0.20, max_bumps: int = 2,
    stop_after_windows: int = 3, min_sustained_windows: int = 3, window: int = 0,
) -> Lever:
    """CLOSED-LOOP EIKONAL CONTROL (#292 build-3, memory L15/L56): activate the in-run costate
    controller that JOINs the async d_seg verdict at each eval point, classifies the within-stage
    trend with the sustained-erosion-vs-transient math (``tools/witness_control_monitor``), and on
    SUSTAINED DIVERGING_ERASING takes BOUNDED action — step the effective eikonal weight up (capped
    at ``eikonal_max``, at most ``max_bumps`` times) then EARLY-STOP cleanly after
    ``stop_after_windows`` post-budget windows (best EMA-shadow ckpt already preserved).

    The activation flag ``--closed-loop-control`` (BooleanOptionalAction, default OFF =>
    byte-identical) IS the swept intent; the schedule params default to the trainer's OWN designed
    defaults (bump 0.05 / max 0.20 / 2 bumps / 3+3 windows) so composing the lever activates the
    mechanism AT its designed operating point (values are the trainer defaults, not invented) while
    staying tunable. CONTAINMENT: the loop only mutates the in-run eikonal + arms a clean stop — it
    never launches anything. ``window=0`` = a control-config change, no epoch budget of its own.
    This is the flagship #332-owed designed lever (SPEC_v75 §10 --closed-loop-* cluster); folding it
    maps all 6 --closed-loop-* flags into the DSL so the #247 duty-to-measure queue surfaces it."""
    return Lever(
        "n292_closed_loop_eikonal_control",
        overrides={
            "--closed-loop-control": True,
            "--closed-loop-eikonal-bump": float(eikonal_bump),
            "--closed-loop-eikonal-max": float(eikonal_max),
            "--closed-loop-max-bumps": int(max_bumps),
            "--closed-loop-stop-after-windows": int(stop_after_windows),
            "--closed-loop-min-sustained-windows": int(min_sustained_windows),
        },
        epochs_delta=int(window),
        notes=("#292 build-3 costate closed-loop d_seg-trend controller: bounded eikonal bump on "
               "sustained erosion + clean early-stop after budget; default-off byte-identical; "
               "schedule params = trainer designed defaults (not invented)"),
    )


def CurriculumReanchorLevers(window: int = 0) -> Lever:
    """#302 (M1): under event-triggering, RE-ANCHOR the TAU-RELATIVE wall-clock levers
    (persistence-warmup completion, seed-anneal withdrawal, analytic-band engage) to the FIRED tau
    boundary instead of their calibrated ep300-relative epochs (a shift, not a rescale). Requires
    ``--curriculum-event-triggered``. Activation flag ``--curriculum-reanchor-levers``
    (BooleanOptionalAction, default OFF); unfired / fired-at-cap / OFF => epochs unchanged =>
    byte-identical. hosc-beta is NOT re-anchored (its beta=4 freeze is Muon-anchored). ``window=0``
    = a schedule-anchoring change, no epoch budget of its own. SPEC_v75 §10 designed lever."""
    return Lever(
        "n302_curriculum_reanchor_levers",
        overrides={"--curriculum-reanchor-levers": True},
        epochs_delta=int(window),
        notes=("#302 M1 re-anchor tau-relative curriculum levers to the fired tau boundary "
               "(requires --curriculum-event-triggered); default-off byte-identical"),
    )


def MarginSaliencyReachability(window: int = 0) -> Lever:
    """LEVER-4 REACHABILITY (memory L76, #268): REPLACE the UNIWARD texture saliency path with the
    cached THROUGH-R fragility-weighted margin-Jacobian S_R (reachability of the CORRECT answer at
    the GT target frame). The texture proxy was MEASURED inert (Pearson -0.033 vs S_R, top-5%
    Jaccard 0.024 = statistical chance, mildly misdirects); S_R lives on the fragile margin band
    where the d_seg debt is. Activation flag ``--margin-saliency-reachability`` (store_true, default
    OFF => byte-identical). Requires an 'sR' key in --gt-cache (tools/precompute_sR_reachability.py).
    ``window=0`` = a saliency-source change, no epoch budget of its own. SPEC_v75 §10 designed
    lever (``--margin-saliency-reachability``)."""
    return Lever(
        "lever4_margin_saliency_reachability",
        overrides={"--margin-saliency-reachability": True},
        epochs_delta=int(window),
        notes=("LEVER-4 through-R reachability saliency (S_R) replacing the measured-inert UNIWARD "
               "texture proxy; requires sR in --gt-cache; default-off byte-identical"),
    )


def DashComb(comb_softness_m: float = 0.3, window: int = 0) -> Lever:
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
    levelset trainer's ``--lane-band-dash-comb`` wire-in (the #224 lane-band path).

    ACTIVATION STATUS (v7.5 actuation, 2026-07-08, v75_actuation_build_20260708.md item A.2):
    COMPOSED ACTIVE in ``crucible_v7`` (``tac.witness_autoconfig._build_crucible_v7`` levers tuple +
    ``_CRUCIBLE_V7_DSL_LEVERS``). The operator OVERRODE the spec's defensible-defer ("lane offloaded to
    the analytic band"): under the committed ``lane_offloaded`` basis regime the band's freq_along≈6
    cartoon scale provably CANNOT represent the ~25-cyc dash comb, so this analytic corrector is exactly
    what supplies that sub-δ structure — its value is HIGHEST precisely under lane_offloaded, not lowest.
    First activation-ledger fire on the v7.5 run. Advisory until byte-closed (pointer 0.19110 UNMOVED)."""
    return Lever("n287_dash_comb",
                 overrides={"--lane-band-dash-comb": True,
                            "--lane-band-comb-softness-m": comb_softness_m},
                 epochs_delta=window,
                 notes="#287 ego-phase dash comb (homogenization corrector; world-static "
                       "period/duty/phase-from-xi replaces per-pair fitted dash phase; "
                       "render-time, active-from-ep0 of the band window, composes with "
                       "AnalyticLaneRenderBand)")


def VerdictDevice(anchor_every: int = 25) -> Lever:
    """GPU (MLX) verdict device + CPU-torch positive-control ANCHOR (the HYBRID A/B activation
    surface for the duty-to-measure queue). Runs the ADVISORY d_seg/d_pose scalars through the
    MLX scorer ports (deterministic w/ fused-R, ~faster) as a FAST trajectory monitor, and runs
    the CPU-torch ANCHOR every ``anchor_every`` gpu verdicts (the positive-control sentinel that
    keeps the instrument out of what it measures + preserves comparability to the CPU-verdict
    baselines). NON-PROMOTABLE either way (CLAUDE.md: MLX/MPS is NEVER a score; only a byte-closed
    exact eval moves the pointer). REFUSED with --async-verdict / nucleus-guard / ladder-homotopy
    (those feed training -> CPU authority only)."""
    return Lever("verdict_device_gpu",
                 overrides={"--verdict-device": "gpu",
                            "--verdict-anchor-every": int(anchor_every)},
                 notes="gpu ADVISORY verdict scalars (MLX ports) + CPU-torch positive-control "
                       "anchor every N (was OFF: --verdict-device cpu)")


def StiefelW(window: int = 100) -> Lever:
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


def CodeSpectralEntropy(beta: float = 0.01, window: int = 100) -> Lever:
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


def DM1Minimal(beta: float = 0.01, window: int = 100) -> tuple[Lever, Lever]:
    """DM1 minimal cure = Stiefel-W + code-spectral-entropy (design memo §4 the 80/20). Returns the
    two composable levers (use ``BASELINE.with_lever(*DM1Minimal())``); both halves target DIFFERENT
    params (W via projection, code via penalty) so they compose without double-counting (§3 routing).
    The per-stage moment-reset (the third minimal item) is the existing ``--stage-transition-reset-moments``
    (already wired); add ``Muon(...)`` or ``StiefelW(window=...)`` arms to engage it at a boundary.

    The warm-start ``window`` is carried ONCE (on the Stiefel lever); the entropy lever uses
    ``window=0`` so composing both extends epochs by ``window`` (not ``2*window``)."""
    return StiefelW(window=window), CodeSpectralEntropy(beta=beta, window=0)


def MarginSaliency(weight: float = 1.0, start_epoch: int = 900,
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


def UniWARD(weight: float = 1.0, start_epoch: int = 900, beta: float = 4.0,
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


def WarpRealLumaFrame0(
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


def StoreNothingPoseCarrier(
    w_pose: float = 1.0, residual_mode: str = "table", start_epoch: int = 0, window: int = 0,
) -> Lever:
    """POSE CARRIER A — STORE-NOTHING joint pose-descent (R1's PROVEN recipe, #245/#238).

    The P0 joint-descent lever. Unlike ``WarpRealLumaFrame0`` (carrier B, which warps the
    STORED real keyframe luma and COUNTS it in archive.zip), this warps the witness's OWN
    plain frame0 INR render by the twist (``--pose-carrier-source generated``) -> stores ONLY
    xi/H (~0 marginal bytes; the render is FREE per rule-118). The per-pair ``dxi`` residual
    (``--pose-carrier-residual-mode table`` => (P,6)) co-adapts to the witness-render warp via
    the ONE ``nn.value_and_grad`` child-attach. ``--w-pose>0`` (default 0.0 = pose-blind)
    engages the pose term so the render CO-ADAPTS to make the cheap warp pose-legible.

    MEASURED reference (R1 = #245, warm-started from the CONVERGED mod-26 v2_attrclean witness):
    joint descent took d_pose 97->0.0011 (plateau ep1074/1093) holding d_seg ~0.0046 at n600
    (``[macOS-CPU advisory] NON-PROMOTABLE``). frame0 is seg-free (upstream/modules.py:108) so
    this carrier CANNOT disturb d_seg. Custody re-validated 2026-07-08
    (``r1_0011_custody_revalidation_20260708.md``): the 0.0011 is a VALID frozen-CPU-torch,
    contest-definition, through-R, EMA-conservative measurement -- BUT NOT byte-closed (the
    trained dxi is un-shipped; #238 = serialize xi_eff = xi_stored + dxi, fp16 ~7.2 KB,
    ~0.0005 rate, then re-measure through inflate). ``--pose-carrier-source`` defaults to fit
    (self-calibrating s_t via the frozen PoseNet d_pose grid at startup); s_t/s_r/pitch left at
    trainer defaults. Composes with every d_seg lever (orthogonal frame; pointer 0.19110 UNMOVED).
    """
    if residual_mode not in ("table", "film"):
        raise ValueError(
            f"StoreNothingPoseCarrier: residual_mode must be 'table' or 'film', got {residual_mode!r} "
            "(never-invent-flags: --pose-carrier-residual-mode choices are exactly table|film)"
        )
    return Lever("store_nothing_pose_carrier",
                 overrides={"--pose-carrier": True,
                            "--pose-carrier-source": "generated",
                            "--pose-carrier-residual-mode": residual_mode,
                            "--w-pose": w_pose},
                 epochs_delta=window,
                 notes=("pose carrier A: STORE-NOTHING joint pose-descent (R1 #245/#238; generated f0 "
                        "warp, ~0 marginal bytes rule-118); --w-pose>0 co-adapts the render + per-pair "
                        "dxi residual (d_pose 97->0.0011 @n600 advisory; byte-close #238; pointer 0.19110)"))


def TerminalPoseFinish(
    start_epoch: int, w_pose: float = 1.0, window: int = 0, *,
    start_event: str | None = None, f_basin: float = 1.0,
) -> Lever:
    """v7.5 D.9 TERMINAL POSE-FINISH — the R1 TWO-PHASE sequence (SPEC §D.9; FEED-238resolved).

    Gates the pose term to engage ONLY AFTER d_seg converges: pose-BLIND (effective w_pose 0) until the
    MUON switch fires (``--muon-start-event``, powerlaw_meat = the d_seg-converged coherent-render regime)
    OR ``start_epoch`` (the fail-safe BACKSTOP epoch), then ``--w-pose`` (the finish weight) engages for
    the terminal joint pose-descent. R1 (warm-started from a CONVERGED witness): d_pose 97->0.0011 while
    d_seg HELD; #238-serialize the dxi at export (SHIPPABLE, pose contribution 0.106, 7.2 KB). Pose is
    ORTHOGONAL to d_seg (frame0 is seg-free, upstream/modules.py:108) so this NEVER disturbs d_seg — it
    SUPERSEDES co-train-pose-from-ep0 (the incoherent-render pose the ~1.79 came from: pose belongs AFTER
    d_seg converges on a COHERENT render, not co-trained from ep0 on an incoherent one).

    The trainer flag ``--pose-finish-start-epoch`` is default 0 ⇒ DISABLED ⇒ byte-identical incumbent
    (pose from ep0). Requires a pose carrier + ``--w-pose > 0`` (the finish-phase weight; the carrier
    trains its dxi only on the realized d_pose term). ``start_epoch`` is a schedule WHEN-trigger — it MUST
    be governed as a fail-safe CAP backstopping ``--muon-start-event`` (the pose-finish co-fires with the
    muon switch), NEVER a naked positive epoch (the schedule-provenance gate refuses it otherwise).
    ``window=0`` = the finish rides the existing finisher span (no extra epoch budget). Advisory until
    byte-closed (pointer 0.19110 UNMOVED). NOTE: crucible_v7 emits this as a TypedStage (sister of the
    muon TypedStage); this factory HOLDS the ``--pose-finish-start-epoch`` flag in the DSL (config-orphan
    discipline — 'the DSL HOLDS every designed lever', same as the ``Muon`` factory holds
    ``--muon-start-epoch``)."""
    # (JACOBIAN BASIN actuator, 2026-07-09 — RUN-2, NOT built on run-1) ``start_event="jacobian_basin"``
    # would ENGAGE the pose-finish when the ξ→PoseNet Jacobian basin forms (median σ_min ≥ f_basin·
    # σ_min^plateau AND basin_frac ≥ q; see tac.witness_control.jacobian_basin.JACOBIAN_BASIN_ENTRY_TRIGGER)
    # instead of TERMINALLY at the muon switch. This is a SCORE-AFFECTING actuation (it changes WHEN pose
    # engages ⇒ changes the trained artifact), so per B5 it is a LEVER — default-off / duty-to-measure —
    # that does NOT fire on run-1 (the basin sensor must be TRUSTED by measurement first; run-1 measures
    # the σ_min(epoch) curve + the would-have-fired epoch as an OBSERVER). The trainer does NOT yet consume
    # an event-triggered pose-finish start (TrainerSupportGap), so referencing it fail-loud here (never an
    # invented flag): the DSL HOLDS the designed lever, and the run-2 resume-A/B on f_basin<1 builds it.
    from tac.witness_control.jacobian_basin import JACOBIAN_BASIN_ENTRY_TRIGGER
    if start_event is not None:
        if str(start_event) != JACOBIAN_BASIN_ENTRY_TRIGGER:
            raise ValueError(
                f"TerminalPoseFinish: start_event must be None or {JACOBIAN_BASIN_ENTRY_TRIGGER!r} "
                f"(the basin ENTRY trigger), got {start_event!r}")
        raise NotImplementedError(
            "TerminalPoseFinish(start_event='jacobian_basin') is the RUN-2 basin-TRIGGERED actuator "
            "(f_basin<1 = earlier engage). It is default-off / duty-to-measure and is NOT built on the "
            "trainer for run-1 (B5: the basin sensor is an OBSERVER first; the pose-finish stays TERMINAL). "
            "Build it in the run-2 resume-A/B against the run-1 σ_min(epoch) curve. f_basin=1.0 (the "
            "run-1 default) IS the current terminal policy — use TerminalPoseFinish(start_epoch=...) "
            "with no start_event.")
    if not (0.0 < float(f_basin) <= 1.0):
        raise ValueError(f"TerminalPoseFinish: f_basin must be in (0,1], got {f_basin!r}")
    if int(start_epoch) < 0:
        raise ValueError(f"TerminalPoseFinish: start_epoch must be >= 0, got {start_epoch!r}")
    if not (float(w_pose) > 0.0):
        raise ValueError(
            f"TerminalPoseFinish: w_pose must be > 0 (the finish-phase weight; the pose carrier trains "
            f"its dxi only on the realized d_pose term), got {w_pose!r}")
    return Lever(
        "terminal_pose_finish",
        overrides={"--pose-finish-start-epoch": int(start_epoch), "--w-pose": float(w_pose)},
        epochs_delta=window,
        notes=("v7.5 D.9 terminal pose-finish (R1 two-phase): pose-BLIND until d_seg converges (the muon "
               "switch), then terminal joint pose-descent at --w-pose; SUPERSEDES co-train-pose-from-ep0; "
               "start_epoch = fail-safe CAP backstopping --muon-start-event; #238 ship-dxi; advisory"))


# ── POSE-FINISHER LADDER, finisher-phase prep (#248/#366, 2026-07-15) ───────────────────────────
# Operator reframe 2026-07-15: pose is the FINISHER, not a parallel run — the R1 two-phase
# architecture already in the sealed config (pose-blind trunk → pose_finish engages on the #383
# sigma_min_plateau gate / muon backstop → terminal joint descent) IS the vehicle. These two
# zero-required-arg composable levers are the finisher-window DELTA that rides the NEXT converged
# trunk via ``--dsl-lever`` (they modify ONLY the pose_finish window's carrier shape + its
# observability; the engage criteria stay #383's). Plan + pre-registration:
# .omx/research/pose_finisher_ladder_prep_20260715.md.

def pose_finisher_live_gap_cadence(
    ema_decay: float = 0.997, num_pairs: int = 600, accum_pairs: int = 8,
) -> int:
    """DERIVED (value-provenance ladder, no bare constant): the live-vs-EMA verdict cadence that
    samples the EMA-lag window >= 2x during the terminal pose-finish descent.

    The trainer's ``--verdict-live-gap-every -1`` auto mode fires ONLY during the run-start
    two-time-constant EMA warmup (``tac.confound_observability.verdict_live_gap_due``) — it is
    structurally SILENT at the pose-finish engage (ep726-class, EMA long warm), exactly where the
    fast d_pose descent re-opens the shadow-vs-live gap (confound C-H2-1; DAG anchor "early-run
    verdict-d_pose RISE = EMA-shadow lag, CONFIRMED run-1"). Derivation:
    ``warmup_epochs = ceil(ema_warmup_updates(decay) / steps_per_epoch)`` with
    ``steps_per_epoch = ceil(num_pairs / accum_pairs)``; cadence = ``max(1, warmup_epochs // 2)``
    => >= 2 live-gap samples inside any EMA-lag window. Defaults (0.997 EMA non-negotiable /
    n600 / --accum-pairs 8 proven_base) give ceil(667/75)=9 -> cadence 4."""
    import math

    from tac.confound_observability import ema_warmup_updates
    if int(num_pairs) <= 0 or int(accum_pairs) <= 0:
        raise ValueError(
            f"pose_finisher_live_gap_cadence: num_pairs/accum_pairs must be > 0, got "
            f"{num_pairs!r}/{accum_pairs!r}")
    steps_per_epoch = math.ceil(int(num_pairs) / int(accum_pairs))
    warmup_epochs = math.ceil(ema_warmup_updates(ema_decay) / steps_per_epoch)
    return max(1, warmup_epochs // 2)


def PoseFinisherLiveGap() -> Lever:
    """Finisher-window OBSERVABILITY (score-neutral; composable via ``--dsl-lever``): the
    live-vs-EMA verdict sentinel at the DERIVED cadence (``pose_finisher_live_gap_cadence`` -> 4),
    all-run — so the pose-finish window's d_pose trajectory is READABLE (no EMA-lag misread) and
    the pre-registered stop criterion (plateau + EMA-settle,
    pose_finisher_ladder_prep_20260715.md) is measurable. Delegates to :func:`VerdictLiveGap`
    (single emitter for ``--verdict-live-gap-every``; never a duplicate flag home). Read-only
    telemetry — never consumed by training/controller; default-off-is-orphan rationale: the
    trainer's auto mode cannot cover the finisher window (see the cadence derivation)."""
    return VerdictLiveGap(every=pose_finisher_live_gap_cadence())


def PoseFinisherFilmReadbackArm() -> Lever:
    """#248 pose-ladder P-B rung in its FINISHER-WINDOW (joint-descent) form: the FiLM READ-BACK
    residual arm — flips ``--pose-carrier-residual-mode`` table -> film over a pose-carrier-active
    base (v9 lineage / store_nothing_205; composing on a carrier-less base is INERT — the flag is
    unread when the carrier is off).

    MECHANISM (trainer ``_PCCarrier`` film path): dxi is READ BACK from the ALREADY-SHIPPED
    per-pair latent code through a tiny shared FiLM MLP (code[mod_dim] -> 32 -> 6, gelu), trained
    JOINTLY in the pose-finish window (never post-hoc — post-hoc/stored is MEASURED DEAD on the
    witness, CLAUDE.md 2026-07-10 CLARIFICATION). vs the R1-proven per-pair TABLE ((P,6) fp16 =
    7,195 B counted, rate 0.004791, d_pose 0.001610 n600 byte-close, FEED-238resolved): film ships
    ~0.8-1.1k params ~1.7-2.2 KB total => rate ~0.0011-0.0015, a -~0.0035 RATE arm. HONEST
    prior: film is a CONSTRAINED reparameterization of the same 6-DOF twist residual — it cannot
    beat table on d_pose; its win is RATE at held d_pose. Pre-registered kill: film d_pose >
    1.5x table at matched finisher epochs => table ships, film = formulation-negative
    (pose_finisher_ladder_prep_20260715.md pre-registration)."""
    return Lever(
        "pose_finisher_film_readback_arm",
        overrides={"--pose-carrier-residual-mode": "film"},
        notes=("#248 P-B finisher-window FiLM read-back: dxi = FiLM(code) shared MLP (~1.7-2.2 KB) "
               "replacing the (P,6) dxi table (7,195 B, rate 0.004791) — a RATE arm at held d_pose; "
               "joint-descent only (post-hoc measured dead); kill if film d_pose > 1.5x table; "
               "advisory until byte-closed (pointer moves only via upstream/evaluate.py)"))


def GroundFrameChart(
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


def MarginCompandedGroundChart() -> Lever:
    """S1 inverse-depth Riemannian compander composed after ``GroundFrameChart``.

    The exact softened profile is MEASURED in
    ``.omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json``.  This is a
    structural chart change from epoch zero: it reallocates a fixed coordinate field's
    sampling density toward the dash-erasure band without changing feature width,
    parameter count, optimizer steps, or the matched-byte receiver budget.  The fitted
    video-derived profile is COUNTED at receiver close.  Default OFF because no canonical
    Program composes this factory; promotion remains owed on the ground-class-pair n600
    ledger plus a receiver-closed matched-bytes/matched-steps A/B.
    """
    from tac.boundary_math.inverse_depth_compander import (
        DEFAULT_COMPANDER_SEED,
        MEASURED_HORIZON_ROW,
        MEASURED_SOFTENING_OFFSET_ROWS,
    )

    return Lever(
        "margin_companded_ground_chart",
        overrides={
            "--ground-frame-chart": True,
            "--margin-companded-ground-chart": True,
            "--margin-compander-horizon-row": float(MEASURED_HORIZON_ROW),
            "--margin-compander-softening-offset-rows": float(
                MEASURED_SOFTENING_OFFSET_ROWS
            ),
            "--margin-compander-seed": int(DEFAULT_COMPANDER_SEED),
        },
        notes=(
            "S1 softened-inverse-depth Riemannian row compander composed AFTER #194 "
            "GroundFrameChart; structural-from-ep0; exact measured delta; fixed capacity "
            "moves toward dash-erasure rows; video-derived chart payload COUNTED; "
            "built-never-fired pending receiver-close n600 matched-byte/step A/B"
        ),
    )


# ---------------------------------------------------------------------------
# COMPUTE/SPEED levers (the gauge's non-curriculum config that compiles to trainer argv).
# SPEED-LEVER POLICY (#356 measurement + operator override 2026-07-12): fp-reorder transforms are
# deterministic-but-DIFFERENT (grad delta 1e-7..1e-3, 3 anchors), but bit identity is explicitly
# WAIVED for the TRAINING trajectory when functional parity holds and wall-clock improves. Exact
# byte-closed scoring authority is unchanged. The drift measurements remain telemetry, never a
# reason to silently omit an active loss term.
# These move WALL-CLOCK, not the witness math/bytes/verdict: CacheGtSkeleton is BIT-IDENTICAL (a
# constant-recompute elision, PROVEN by the #260 n=8 CPU A/B: EMA-shadow max_abs=0); MicroBatch is
# trajectory-affecting (batched fp reduction); its accum-step loss/grad must remain within the
# registered functional tolerances of the serial mean-over-chunk. Both compose with any curriculum
# lever; neither carries an epochs_delta (they are
# GLOBAL config, not a warm-start A/B stage). means != ends: SPEED buys nothing on S — only a
# byte-closed n600 exact row can move the canonical reports/latest.md contest-CPU pointer.
# ---------------------------------------------------------------------------
def CacheGtSkeleton() -> Lever:
    """SPEED (BIT-IDENTICAL): cache the CONSTANT per-pair GT soft-skeleton for the persistence loss
    (``--cache-gt-skeleton``). ``sg = soft_skeleton(gt)`` is epoch-invariant + gradient-free (it
    multiplies ``pred`` in the clDice ``tsens`` term), so precomputing it once per pair + reusing it
    every step is bit-identical (a materialized concrete constant == the inline recompute) while
    skipping ~half the clDice cost. No-op unless ``--persistence-loss-weight>0`` (the only consumer);
    both serial and micro-batched twins consume the same cached constant.
    ``--cache-gt-skeleton`` is store_true -> emitted True ONLY (never False, review C2)."""
    return Lever("cache_gt_skeleton",
                 overrides={"--cache-gt-skeleton": True},
                 notes="speed (bit-identical): cache the constant GT soft-skeleton for the persistence loss")


def MicroBatch(pairs: int = 4) -> Lever:
    """SPEED (training-only, trajectory-affecting): batch ``pairs`` per scorer forward.

    The single-pair EfficientNet-B2 SegNet / FastViT PoseNet forward under-utilizes the GPU; B>1
    renders + scores B pairs in one batched forward. The operator targets the historical ~2-4x
    speed class, but the faithful full-V9 end-to-end receipt remains required. Grads are weighted by pair count so the
    accum-step loss/gradient is functionally equivalent to the serial mean-over-chunk. It is NOT bit
    identical because scorer kernels and reductions change order; the operator explicitly waived that
    for training on 2026-07-12. It composes with ``--seed-islands`` through the trainer's dual batched
    co-gradient path and with every canonical V9 loss leg through their batched twins. B=1 is the
    serial baseline. This lever has no score authority; exact rows still require byte-closed evaluator
    replay. A value flag (never store_true). ``pairs`` must be positive so an invalid batch cannot
    survive typed compile and fail later inside the trainer loop."""
    pairs = int(pairs)
    if pairs < 1:
        raise ValueError(f"MicroBatch pairs must be >= 1, got {pairs}")
    return Lever("micro_batch_pairs",
                 overrides={"--micro-batch-pairs": pairs},
                 notes=("training-only speed lever: B pairs per batched scorer forward; functional "
                        "parity required, bit drift operator-waived, no score authority"))


# ---------------------------------------------------------------------------
# The FIXED, KNOWN OPENING of the from-scratch openpilot-seeded d_seg curriculum.
# (S0 seed -> S1 short-CE -> S2 tau_softplus). l7 + Muon are STACKED ADAPTIVELY by
# ``campaign.plan_adaptive_step`` off this opening's measured per-stage checkpoints.
# Deep-math anchors: FEED-bv (measured per-stage d_seg dirs), FEED-fs (separatrix
# seed), FEED-fz/-bu (reheat), anneal-memo (tau=0.3 == reachability floor), FEED-fi
# (Muon = spectral conditioner -> stacked, not fixed). DAG FEED-ln.
# ---------------------------------------------------------------------------
def openpilot_seeded_opening(
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
def SeedIslandEased(window: int = 100) -> Lever:
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


def SeedIslandBirth(window: int = 100) -> Lever:
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


def Mod32SegOnlyControlBase(window: int = 0) -> Lever:
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


def EventTriggeredCurriculum(window: int = 0) -> Lever:
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


def EikonalViscosity(eps: float = 0.05, adaptive: bool = True, window: int = 0) -> Lever:
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
        # #320 DERIVED adaptive-ε bounds made DSL-EXPLICIT (equal to the trainer argparse defaults →
        # byte-identical): floor 0.3 MEASURED (FEED-05v, above the rising lower-CFL edge), upper 0.7
        # (below the ε=1.0 biharmonic instability), margin 0.5 (CFL safety above the lower edge).
        ov["--eikonal-visco-eps-floor"] = 0.3
        ov["--eikonal-visco-eps-upper"] = 0.7
        ov["--eikonal-visco-margin-factor"] = 0.5
    return Lever("eikonal_viscosity",
                 overrides=ov, epochs_delta=window,
                 notes="#316/#320 DE-derived viscous/adaptive-ε eikonal stabilization "
                       "(adaptive-ε bounds DERIVED-explicit: floor 0.3 MEASURED / upper 0.7 / margin 0.5)")


def AmplifyIsland(weight: float = 1.0, window: int = 100) -> Lever:
    """Island-amplify loss: raise the rare-class island logit on the COSTATE/margin-GATED support
    (amplify only where the big-3 margin is preserved ⇒ n_big3→0 ⇒ net-positive by construction;
    UNIFORM amplification is the measured net-negative). ``--amplify-form``/``--amplify-margin-
    target`` keep trainer defaults; a nonzero weight turns the term on."""
    return Lever("island_amplify",
                 overrides={"--amplify-weight": weight},
                 epochs_delta=window,
                 notes="rare-class island-amplify on margin-gated support (net-positive by construction)")


def BoundaryDistance(weight: float = 1.0, window: int = 100) -> Lever:
    """#301 loss-geometry: boundary-distance-weighted seg loss (concentrate pressure on the
    codim-1 separatrix annulus). Default-OFF (weight 0) in the baseline; a nonzero weight engages."""
    return Lever("boundary_distance",
                 overrides={"--boundary-distance-weight": weight},
                 epochs_delta=window,
                 notes="#301 boundary-distance-weighted seg loss (separatrix-annulus concentration)")


def SegFocalGamma(gamma: float = 2.0, window: int = 100) -> Lever:
    """#301 focal-γ seg loss: down-weight easy (high-margin) pixels, focus on the flip-prone
    boundary. γ calibrated $0-measured (#301); default 2.0 is the canonical focal exponent."""
    return Lever("seg_focal_gamma",
                 overrides={"--seg-focal-gamma": gamma},
                 epochs_delta=window,
                 notes="#301 focal-γ seg loss (down-weight easy pixels toward the flip boundary)")


def FisherDensityWeight(blend: float = 1.0, source: str = "model", window: int = 100) -> Lever:  # noqa: N802
    """SPEC_v10 §13.1 row 2 / §13.4(1) (build-wave arm A): Fisher-density per-pixel seg-loss
    weight from the EXACT registered law ``tr g = ½·sech²(m/2)``
    (``fisher_curvature_equals_categorical_fisher_trace_caustic_v1``, ρ=0.978) — the seg descent
    force made metric-aware (capacity flows where decisions bend, the separatrix). ``blend`` λ ∈
    (0,1] mixes uniform→pure-Fisher; mean-1 stop-grad renorm conserves the gradient budget.
    ``source``: 'model' = the metric at the CURRENT live logits (DERIVED Fisher-natural for a
    training force) · 'gt' = the cached GT-margin stationary prior (the A/B arm). DEFAULT-OFF in
    the trainer (0.0 = byte-identical); ``blend`` is the SWEPT intent — RUN-GATED optimum (owed
    $0 cached-ckpt A/B), 1.0 is the pure-law form, not a measured optimum. COMPOSITION: same
    multiplicative seg_pixel_w surface as SegFocalGamma — they overlap on the boundary but
    DISAGREE on confidently-wrong pixels (focal up-weights, Fisher down-weights); prefer one.
    Fail-closed vs --micro-batch-pairs>1 (not routed into the batched twin)."""
    if not (0.0 < float(blend) <= 1.0):
        raise ValueError(f"FisherDensityWeight blend must be in (0, 1], got {blend!r}")
    if source not in ("model", "gt"):
        raise ValueError(f"FisherDensityWeight source must be 'model' or 'gt', got {source!r}")
    return Lever("fisher_density_weight",
                 overrides={"--fisher-density-weight": float(blend),
                            "--fisher-density-source": str(source)},
                 epochs_delta=window,
                 notes="arm A §13.4(1): exact-law sech² Fisher-trace seg reweight (blend RUN-GATED, "
                       "not a measured optimum; source model=Fisher-natural / gt=stationary-prior A/B)")


def HeadNaturalGradient(eps: float = 1e-3, window: int = 100) -> Lever:  # noqa: N802
    """SPEC_v10 §13.4(2) (build-wave arm A): logit-space NATURAL-GRADIENT preconditioning of the
    seg force. The frozen SegNet head is EXACT rank-4 linear
    (``segnet_head_rank4_linear_flipdist_v1``) ⇒ the categorical Fisher g = diag(p)−ppᵀ has a
    CLOSED-FORM damped pseudo-inverse g⁺v = v/(p+eps) − mean_k (O(K)/px, no solve); the trainer
    applies it as a forward-identity/backward-g⁺ transform on the seg logits, so the descent
    direction of EVERY witness param becomes the Fisher natural gradient of the active seg form.
    COMPOSES with (does not duplicate) --head-offset-solver / #423 Hessian head-offset (periodic
    closed-form SOLVE of out_sdf.bias at checkpoints) and the #518 fork head-SOLVE (head weights
    solved per stage): those act on HEAD PARAMS at discrete events; this preconditions the
    PER-STEP direction through the logit bottleneck. DEFAULT-OFF (flag absent = byte-identical);
    ``eps`` is the damping at simplex corners — RUN-GATED optimum (owed $0 cached-ckpt A/B).
    Fail-closed vs --micro-batch-pairs>1 (not routed into the batched twin)."""
    if not (float(eps) > 0.0):
        raise ValueError(f"HeadNaturalGradient eps must be > 0, got {eps!r}")
    return Lever("head_natural_grad",
                 overrides={"--head-natural-grad": True,
                            "--head-natural-grad-eps": float(eps)},
                 epochs_delta=window,
                 notes="arm A §13.4(2): closed-form g⁺ logit natural gradient (eps RUN-GATED, not a "
                       "measured optimum; composes with #423 head-offset + #518 head-SOLVE — "
                       "per-step direction vs discrete-event head-param solves)")


def EikonalStEik(weight: float = 0.05, normalized: bool = True, norm_eps: float = 1e-2,
                 window: int = 100) -> Lever:  # noqa: N802
    """#317 StEik directional-divergence eikonal stabilizer (Yang et al. StEik): penalise the
    normalised unit-normal curvature ``n^T H n = dir_div/(|∇m|^2+eps)`` to suppress the biharmonic
    re-entry instability the plain viscous ε cannot fully damp. DEFAULT-OFF in the sealed baseline
    (trainer ``--eikonal-steik-weight`` default 0.0); this Lever engages it (``normalized=True`` =
    the #317-N build). ``weight`` is the SWEPT intent — its optimum is RUN-GATED (owed: per-lever
    A/B vs plain EikonalViscosity through the real n600 verdict); the 0.05 default is a starting
    value, NOT a measured optimum. Requires --eikonal-weight>0 to have effect."""
    ov: dict = {"--eikonal-steik-weight": float(weight), "--eikonal-steik-norm-eps": float(norm_eps)}
    if normalized:
        ov["--eikonal-steik-normalized"] = True
    return Lever("eikonal_steik",
                 overrides=ov, epochs_delta=window,
                 notes="#317 StEik normalized directional-divergence stabilizer (weight RUN-GATED, "
                       "not a measured optimum)")


def EikonalJunctionRelax(relax: float = 0.5, tau: float = 0.5, window: int = 100) -> Lever:  # noqa: N802
    """θ* STRETCH-1 eikonal junction relaxation: down-weight the |∇m|→1 residual near triple
    junctions (top2/top3 SDF-gap < ``tau``) where the eikonal constraint is genuinely violated by
    the multi-class chart. DEFAULT-OFF (trainer ``--eikonal-junction-relax`` default 0.0); a nonzero
    ``relax`` engages it. ``relax`` is the SWEPT intent — RUN-GATED optimum (owed A/B); 0.5 is a
    starting value, not a measured optimum. ``tau`` keeps the trainer default 0.5."""
    return Lever("eikonal_junction_relax",
                 overrides={"--eikonal-junction-relax": float(relax), "--eikonal-junction-tau": float(tau)},
                 epochs_delta=window,
                 notes="θ* STRETCH-1 eikonal junction relaxation (relax RUN-GATED, not a measured optimum)")


def CodeNuclearNorm(weight: float = 1e-3, eps: float = 1e-3, ns_iters: int = 25,
                    window: int = 100) -> Lever:  # noqa: N802
    """θ* MUST-2 low-rank code regularizer: smoothed nuclear norm ‖C‖_* of the per-pair code matrix
    (Newton-Schulz matrix-sqrt, ``ns_iters`` iterations, relative smoothing floor ``eps``) to push
    the per-pair codes onto a low-rank manifold (the intrinsic-dim≈8 prior → cheaper stored code).
    Sibling of the CodeSpectralEntropy code-regularizer family. DEFAULT-OFF (trainer
    ``--code-nuclear-weight`` default 0.0); ``weight`` is the SWEPT intent — RUN-GATED optimum
    (owed A/B); 1e-3 is a starting value, not a measured optimum."""
    return Lever("code_nuclear_norm",
                 overrides={"--code-nuclear-weight": float(weight), "--code-nuclear-eps": float(eps),
                            "--code-nuclear-ns-iters": int(ns_iters)},
                 epochs_delta=window,
                 notes="θ* MUST-2 smoothed nuclear-norm low-rank code regularizer (weight RUN-GATED, "
                       "not a measured optimum)")


def SegSpikeReweight(downweight: float = 0.5, coherent_upweight: float = 1.0,
                     window: int = 100) -> Lever:  # noqa: N802
    """Flicker-aware per-pixel seg-CE reweight (L85 flicker-floor lever): down-weight SPIKE
    (single-frame flicker) pixels by ``downweight`` and up-weight COHERENT (temporally-consistent,
    unstable) boundary pixels by ``coherent_upweight`` — steer capacity off the GT-side sub-pixel
    advection phase-flicker (the measured d_seg floor cause) and onto the recoverable coherent
    boundary. DEFAULT-OFF (byte-identical unless ``--seg-spike-reweight`` set); ``downweight`` is the
    SWEPT intent — RUN-GATED optimum (owed A/B); 0.5 is a starting value, not a measured optimum."""
    return Lever("seg_spike_reweight",
                 overrides={"--seg-spike-reweight": True, "--seg-spike-downweight": float(downweight),
                            "--seg-coherent-upweight": float(coherent_upweight)},
                 epochs_delta=window,
                 notes="L85 flicker-aware per-pixel seg-CE reweight (downweight RUN-GATED, not a "
                       "measured optimum)")


def EmaDecayCalibrated(
    updates_per_run: int,
    target_seed_fraction: float | None = 0.01,
    warmup_fraction: float | None = None,
    window: int = 0,
    *,
    execution_mode: str = "constant_decay",
) -> Lever:
    """ARM-C p0_ema_calibration (SPEC_v10 §13.3): ``--ema-decay`` DERIVED from RUN GEOMETRY via
    the registered law ``ema_decay_run_geometry_v1`` (LawRef-resolved at DSL-compile time —
    value-provenance rung ``derived_at_config``). Exactly ONE of ``target_seed_fraction``
    (d = eps**(1/U): pin the weight the initial shadow seed retains at run end) or
    ``warmup_fraction`` (d = 1 - 2/(phi*U): pin the run fraction where the 2/(1-d) warmup
    completes) selects the inversion. The incumbent 0.997 is a Quantizr per-step-minibatch
    provenance that does NOT transfer to the 1-update/epoch full-batch regime (MEASURED on the
    live c2 run: warmup 667 updates ~ ep1318/1400; ~64% warm-start seed @ep800). NOT composing
    this Lever leaves the trainer default 0.997 untouched (byte-identical default path). The
    d_seg effect of the calibrated decay is RUN-GATED (shadow-vs-live byte-close A/B decides,
    SPEC_v10 §13.5).

    ``execution_mode`` is part of this typed Lever's policy contract.  The registered
    equation is a constant-decay law, so ``constant_decay`` is the canonical mode and
    compiles to ``warmup=False`` at the consumer.  ``warmup_ablation`` is the only
    supported warmup mode and remains explicit in the compiled config; silently applying
    the generic EMA warmup to a constant-decay LawRef changes the sealed intervention.
    """
    from tac.witness_dsl.lawref import (
        LADDER_DERIVED_AT_CONFIG,
        InputRef,
        LawRef,
        resolve,
    )
    if (target_seed_fraction is None) == (warmup_fraction is None):
        raise ValueError(
            "EmaDecayCalibrated: exactly ONE of target_seed_fraction / warmup_fraction must be "
            "set (the law inverts for d from one pinned quantity)")
    if execution_mode not in {"constant_decay", "warmup_ablation"}:
        raise ValueError(
            "EmaDecayCalibrated: execution_mode must be constant_decay or warmup_ablation"
        )
    u = int(updates_per_run)
    if target_seed_fraction is not None:
        mode_code, mode_name = 1, "decay_from_seed_fraction"
        quantity = {"target_seed_fraction": InputRef.literal(
            float(target_seed_fraction),
            "ARM-C ema calibration: pinned terminal seed fraction eps (d = eps**(1/U))")}
    else:
        mode_code, mode_name = 2, "decay_from_warmup_fraction"
        quantity = {"warmup_fraction": InputRef.literal(
            float(warmup_fraction),
            "ARM-C ema calibration: pinned warmup completion fraction phi (d = 1 - 2/(phi*U))")}
    ref = LawRef(
        equation_id="ema_decay_run_geometry_v1",
        inputs={
            "mode": InputRef.literal(
                mode_code, f"mode code {mode_code} == {mode_name} (numeric-literal encoding)"),
            "updates_per_run": InputRef.literal(
                u, "run geometry: optimizer updates in the averaging window "
                   "(full-batch accum => 1/epoch)"),
            **quantity,
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    rc = resolve(ref, repo_root=_REPO_ROOT)
    d = float(rc.value)
    if not 0.0 < d < 1.0:
        raise ValueError(f"EmaDecayCalibrated: resolved decay {d} outside (0,1) — check inputs")
    execution = {
        "mode": execution_mode,
        "warmup": execution_mode == "warmup_ablation",
        "ablation_declared": execution_mode == "warmup_ablation",
        "sealed_law": "constant_decay" if execution_mode == "constant_decay" else "warmup",
        "source": "tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated",
    }
    return Lever(
        "ema_decay_calibrated",
        overrides={"--ema-decay": d},
        epochs_delta=window,
        lawrefs={"--ema-decay": ref},
        constant_manifest={"--ema-decay": rc.to_dict()},
        policy_contracts={"ema_execution": execution},
        notes=(
            f"ema_decay DERIVED from run geometry ({mode_name}, U={u}) via "
            f"ema_decay_run_geometry_v1 -> {d:.6f}; execution={execution_mode} is "
            "typed in the Lever policy contract; incumbent 0.997 provenance does not "
            "transfer to full-batch (SPEC_v10 §13.3); effect RUN-GATED"
        ),
    )


def EmaDecayFinisher(decay: float = 0.999, start_epoch: int | None = None,
                     window: int = 0) -> Lever:  # noqa: N802
    """THETA* TIER-2 MUST-3 / SPEC_v10 §13.3: the BUILT wider-finisher EMA (SWA-style late-
    oscillation averaging). From the resolved finisher-start epoch onward the EMA update uses
    this WIDER decay (averages the late oscillation into a flat-basin center). DEFAULT-OFF in
    the trainer (``--ema-decay-finisher`` default None = bit-identical to --ema-decay
    everywhere); this Lever engages it. ``start_epoch`` None falls back to the trainer's
    ``--muon-start-epoch`` (the natural finisher boundary; the trainer fail-closes when
    neither is set). ``decay`` is the SWEPT intent — RUN-GATED optimum (owed A/B); 0.999 is a
    starting value, NOT a measured optimum. NEVER-FIRED as of 2026-07-17 (registered in the
    activation/duty ledger via tools/register_ema_finisher_duty.py so the costate SENSE layer
    surfaces it — the 'off is a tracked queue' discipline)."""
    ov: dict = {"--ema-decay-finisher": float(decay)}
    if start_epoch is not None:
        ov["--ema-decay-finisher-start-epoch"] = int(start_epoch)
    return Lever("ema_decay_finisher",
                 overrides=ov,
                 epochs_delta=window,
                 notes="THETA* MUST-3 SWA-style wider-finisher EMA (decay RUN-GATED, not a "
                       "measured optimum; start falls back to --muon-start-epoch)")


def LaneSkipBand(weight: float = 0.05, dilate: int = 2, start_epoch: int = 0,
                 window: int = 100) -> Lever:  # noqa: N802
    """ARM-C #524 (SPEC_v10 §13.1 row 4): Lane stride-2 SKIP-BAND supervision. DERIVED from the
    frozen-SegNet recursive-fractal factorization §5 (MEASURED: the final decoder block is
    skipless, so ALL sub-stride-4 boundary localization flows through the ONE 16-ch stride-2 skip
    at (192,256); ablating its sub-stride-4 detail via down-up 2x induces flips that are 77%
    Road-Lane — Lane is THE skip-limited pair). The lever supervises the witness render's skip
    DETAIL band ``SB = D2 - U2(D2(D2))`` (BT.601 luma/255, on the SHARED realized through-R frame)
    toward the GT frame's SB on the dilated GT-Lane band — shaping the Lane-band output to be
    LEGIBLE to the only channel through which the frozen scorer localizes Lane boundaries.
    Numpy reference authority: ``tac.boundary_math.lane_skipband`` (MLX twin parity-tested).
    DEFAULT-OFF in the trainer (``--lane-skipband-weight`` default 0.0; byte-identical); this
    Lever engages it. ``weight`` is the SWEPT intent — RUN-GATED optimum (duty-to-measure A/B);
    0.05 is a starting value, NOT a measured optimum. Requires --micro-batch-pairs 1 (the trainer
    fail-closes otherwise; the batched twin does not yet carry the term)."""
    return Lever("lane_skipband",
                 overrides={"--lane-skipband-weight": float(weight),
                            "--lane-skipband-dilate": int(dilate),
                            "--lane-skipband-start-epoch": int(start_epoch)},
                 epochs_delta=window,
                 notes="ARM-C #524 stride-2 skip-band Lane supervision (DERIVED from the fractal "
                       "factorization §5 skip-ablation measurement; weight RUN-GATED, not a "
                       "measured optimum)")


def SpikeGuardRollback(frac: float = 0.5, lr_cut: float = 0.5, max_rollbacks: int = 8,
                       window: int = 20, spike_factor: float = 5.0) -> Lever:  # noqa: N802
    """Spike-guard stability actuator in ROLLBACK mode (the CLAUDE.md confound-fix). The prior
    ``legacy`` median-freeze — a reference window that updates only on ACCEPTED batches — silently
    FROZE runs at ep103-114 (`ep_loss==0.0`) and poisoned a whole session's verdicts; ``rollback``
    (the trainer default since the fix) is the DERIVED-correct actuator. This Lever makes the mode
    choice DSL-EXPLICIT (byte-identical to the sealed default) and HOLDS the rollback tuning knobs
    (``frac``/``lr_cut``/``max_rollbacks``/``window``/``spike_factor``) whose optima are RUN-GATED
    (owed A/B) — the emitted values equal the trainer defaults, so composing this Lever alone is
    byte-identical; a swept override is the intent. Sister: Catalog #397/#398 confound gates."""
    return Lever("spike_guard_rollback",
                 overrides={"--spike-guard-mode": "rollback", "--spike-rollback-frac": float(frac),
                            "--spike-rollback-lr-cut": float(lr_cut), "--spike-rollback-max": int(max_rollbacks),
                            "--spike-rollback-window": int(window), "--spike-factor": float(spike_factor)},
                 epochs_delta=0,
                 notes="confound-fix: spike-guard ROLLBACK actuator DSL-explicit (mode DERIVED-correct; "
                       "rollback tuning RUN-GATED, byte-identical to sealed defaults)")


def LambdaPreProbe(iters: int = 4, fd_eps: float = 1e-3, window: int = 0) -> Lever:  # noqa: N802
    """EIK-STAB build-4 preconditioned power-iteration probe: run ``iters`` finite-difference HVP
    power iterations (relative FD step ``fd_eps``) to estimate the top Hessian eigen-direction of
    the loss and precondition the eikonal step — a stability diagnostic + preconditioner, not a
    score term. DEFAULT-OFF (trainer ``--lambda-pre-probe-iters`` default 0); ``iters``>0 engages it.
    ``iters`` is the SWEPT intent — RUN-GATED (owed A/B); 4 is a starting value, not a measured
    optimum."""
    return Lever("lambda_pre_probe",
                 overrides={"--lambda-pre-probe-iters": int(iters), "--lambda-pre-probe-fd-eps": float(fd_eps)},
                 epochs_delta=window,
                 notes="EIK-STAB build-4 FD-HVP power-iteration preconditioner probe (iters RUN-GATED)")


def AdamBeta2(beta2: float = 0.99, window: int = 0) -> Lever:
    """#222 Adam β₂ lever (arXiv 2603.02092): sweep β₂ (second-moment decay). The launch-gate
    guard requires β₁<√β₂; window=0 = optimizer-config change, no epoch budget of its own."""
    return Lever("adam_beta2",
                 overrides={"--adam-beta2": beta2},
                 epochs_delta=window,
                 notes="#222 Adam β₂ second-moment decay sweep (β₁<√β₂ guard)")


def AdamWReferenceSemantics(window: int = 0) -> Lever:
    """Default-OFF reference AdamW update for a matched optimizer A/B.

    MLX AdamW defaults ``bias_correction=False`` whereas reference AdamW and
    PyTorch use bias-corrected moments.  This typed lever enables correction at
    the incumbent beta2=0.999 without silently changing legacy trajectories.
    """

    if int(window) < 0:
        raise ValueError("AdamWReferenceSemantics: window must be >= 0")
    return Lever(
        "adamw_reference_semantics",
        overrides={"--adamw-reference-semantics": True},
        epochs_delta=int(window),
        notes=("round-2 AdamW audit treatment: standard bias-corrected moments at beta2=0.999; "
               "matched local A/B owed; default OFF preserves incumbent resumes"),
    )


def DirectionalBasisRebalance(freq_across: int = 32, regime: str = "lane_offloaded",
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
    enables it). window=0 = basis-config change, no epoch budget of its own.

    ## Regime coherence coupling (SEAL v7.3 round-2 M1 fix)
    The regime does NOT only set the basis frequencies — it must also gate the LEARNED lane-recall
    losses so the basis and the loss target AGREE (else a lane-CARRYING recall under a lane-OFFLOADED
    basis is unsatisfiable and jitters the binding Road↔Lane separatrix). This lever emits ONLY the
    basis flags (it is a SISTER of the persistence-recall lever, different mechanism); the coherent
    persistence-recall class targeting is DERIVED from the same regime via the companion law
    :func:`persistence_classes_for_basis_regime` (composed at the config layer — e.g.
    ``_build_crucible_v7`` sets ``--persistence-classes`` from it). lane_offloaded → recall targets
    movable ONLY (lane rides the analytic band); lane_carried → 'auto' (keep the learned lane recall
    at freq_along≈26). The LADDER island-amplify is ALREADY per-class-λ self-gated (it de-emphasizes a
    class whose measured cost is low, i.e. lane once the band handles it) so it needs no regime gate;
    the FIXED-weight persistence recall was the one regime-BLIND term the coupling fixes.
    STRUCTURAL ENFORCEMENT (seal v7.4 r3 F-3): the lane_offloaded regime is COMPILE-TIME
    coupled to the analytic band — ``_build_crucible_v7`` fail-louds if ``--lane-render-band``
    is absent from the emitted base when this regime is selected, so the recall drop can
    never ship without the band that justifies it (the coupling is asserted, not assumed)."""
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


def persistence_classes_for_basis_regime(regime: str, *, lane_class: int = 1,
                                         movable_class: int = 3) -> str:
    """FEED-07a two-regime COHERENCE coupling (SEAL v7.3 round-2 M1 fix): the persistence-RECALL class
    targeting DERIVED from the active :func:`DirectionalBasisRebalance` regime, so the learned recall
    loss and the basis frequency budget AGREE (returns a ``--persistence-classes`` value).

    * ``lane_offloaded``: the FREE rule-118 analytic band carries lane (MEASURED lane d_seg 0.00087)
      and the basis is set to freq_along≈6 (Candès–Donoho cartoon scale) which CANNOT represent the
      ~25-cyc dash comb. Demanding lane-skeleton RECALL from that frequency-starved learned render is
      an unsatisfiable target → at best wasted gradient, at worst boundary JITTER on the binding
      Road↔Lane separatrix (part of the 68%-of-flips Road residual). Persistence recall therefore
      targets ONLY the non-offloaded tail class (movable) → returns ``str(movable_class)`` = "3".
    * ``lane_carried``: the witness itself carries the dash comb (freq_along≈26) → KEEP lane in the
      recall target → returns "auto" (the trainer self-detects lane+movable erasure-tail classes).

    lane_class/movable_class default to the canonical comma10k order (1=Lane, 3=Movable). Pure /
    unit-testable; fail-closed on an unknown regime (mirrors :func:`freq_along_for_regime`)."""
    r = str(regime)
    if r == "lane_offloaded":
        return str(int(movable_class))          # movable only; lane rides the analytic band
    if r == "lane_carried":
        return "auto"                           # keep lane in the learned recall (freq_along≈26)
    raise ValueError(
        f"persistence_classes_for_basis_regime: unknown regime {regime!r} "
        "(expected 'lane_offloaded' or 'lane_carried')")


def logit_adjust_classes_for_basis_regime(regime: str, *, lane_class: int = 1,
                                          movable_class: int = 3) -> str:
    """v7.5 Lever-3 REGIME COHERENCE (sister of :func:`persistence_classes_for_basis_regime`): the
    --logit-adjust-classes value DERIVED from the active :func:`DirectionalBasisRebalance` regime, so
    the loss-time class-prior boost and the basis frequency budget AGREE.

    The Menon logit-adjustment boosts rare-class RECALL by shifting the seg-loss logits
    (offset_c = tau*log(prior_c); measured lane -5.14, movable -4.39). Under ``lane_offloaded`` the
    basis is freq_along≈6 (Candès–Donoho cartoon scale) which CANNOT represent the ~25-cyc dash comb —
    so a -5.14 lane RECALL boost demands lane skeleton from a frequency-starved render that physically
    cannot produce it. That over-boost is exactly the recall-without-precision driver the
    road_anomaly_probe measured over-painting lane 13.8x INTO Road. Coherence => drop lane from the
    boost when it is offloaded (returns ``str(movable_class)`` = "3", movable-only); ``lane_carried``
    keeps lane (returns "all"). Pure / fail-closed on an unknown regime (mirrors the persistence sister)."""
    r = str(regime)
    if r == "lane_offloaded":
        return str(int(movable_class))          # movable only; lane offloaded => no learned-recall boost
    if r == "lane_carried":
        return "all"                            # keep lane in the boost (freq_along≈26 carries it)
    raise ValueError(
        f"logit_adjust_classes_for_basis_regime: unknown regime {regime!r} "
        "(expected 'lane_offloaded' or 'lane_carried')")


def AreaConstraintBirth(birth_force: float = 1.0, tolerance: float = 0.25,
                        classes: str = "1,3", window: int = 0) -> Lever:
    """v7.5 birth-counter-force Lever-1 — the CHAN-VESE AREA-CONSTRAINT precision counter-force
    (equations leg ``tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708``, the
    imported balance law).

    The rare-class-birth stack (seed-islands + island-amplify + persistence-recall + logit-adjust) is
    RECALL-only: it over-grows the born classes {Lane,Movable} INTO GT-Road with no precision cap
    (MEASURED road_anomaly_probe_20260708.md: lane 13.8x / movable 4.6x over-paint at ep125,
    mass-conserved from Road => Road d_seg pinned ~0.40). This lever adds the ONE-SIDED Chan-Vese
    area-constraint Lagrange term of the level-set region energy

        E_area,c(phi) = (lambda_c / 2) * max(0, A_c(phi) - A_c^GT)^2

    whose gradient is an INWARD retraction on the boundary (delta(phi_c) via the softmax Jacobian)
    PROPORTIONAL to the area overshoot. It balances the birth force at ``A_c^* = (1+tolerance)*A_c^GT``
    — the equilibrium IS the spec, no ramp schedule needed (operator 2026-07-08: "the level set and
    Morse-Smale are perfect for engineering the precisely desired annealing behavior"). With this term
    active the completion event (:func:`BirthCompletionEvent`) becomes a regime hand-off, NOT the sole
    safety mechanism (defense-in-depth; the multiplier self-limits continuously).

    The trainer derives ``lambda_c = birth_force / (tolerance * A_c^GT)`` LIVE from the loaded GT
    areas (value-provenance gold standard — no frozen literal). ``birth_force`` (F_birth proxy = the
    birth loss weight, 1.0 in v7 argv), ``tolerance`` (equilibrium overshoot fraction; 0.25 =>
    equilibrium 1.25x GT), ``classes`` (comma list of birthed classes, 'auto' = the amplify/persistence
    island classes). Consumes the SHARED realized through-R seg forward (NO extra SegNet forward). The
    trainer flag ``--area-constraint-birth`` is a store_true default-OFF => byte-identical when this
    lever is not composed. ``MicroBatch(B>1)`` evaluates the same soft-mass hinge independently for
    each pair and only then averages across B; a global batch-area hinge is forbidden. The lambda
    SCALE is ASSUMED_AWAITING_VERIFICATION (owed to the v7.5 A/B); the FORM + balance are DERIVED.
    window=0 = loss-config lever, no epoch budget of its own."""
    if not (float(tolerance) > 0.0):
        raise ValueError(f"AreaConstraintBirth: tolerance must be > 0, got {tolerance!r}")
    if not (float(birth_force) >= 0.0):
        raise ValueError(f"AreaConstraintBirth: birth_force must be >= 0, got {birth_force!r}")
    return Lever(
        "v75_area_constraint_birth",
        overrides={"--area-constraint-birth": True,
                   "--area-constraint-birth-force": float(birth_force),
                   "--area-constraint-tolerance": float(tolerance),
                   "--area-constraint-classes": str(classes)},
        epochs_delta=window,
        notes=("v7.5 Lever-1 Chan-Vese area constraint (precision counter-force vs recall-only birth "
               "over-paint); lambda_c=birth_force/(tolerance*A_GT_c) DERIVED LIVE; equilibrium "
               f"(1+{tolerance})*A_GT; micro-batch twin preserves per-pair area before batch mean; "
               "reuses the realized seg forward; advisory until byte-closed"))


def BirthCompletionEvent(tau_persist: float = 0.8, area_band: float = 0.25,
                         ramp_epochs: int = 50, post_level: float = 0.0,
                         classes: str = "1,3", window: int = 0,
                         ramp_apply: bool = False) -> Lever:
    """v7.5 birth-counter-force Lever-2 — the MORSE-SMALE PERSISTENCE birth-completion event (engine
    ``tac.witness_control.birth_completion``; operator 2026-07-08: "a MORSE-SMALE PERSISTENCE event,
    not a bare part_frac threshold").

    The missing "stop growing after birth" law: the recall-only birth stack has no completion event, so
    it keeps pushing a class after it has nucleated + formed. This event fires per birthed class when
    BOTH (a) island PERSISTENCE (prominence = 1-within_flip, the Morse-Smale basin reading from the
    #333/nucleus telemetry) >= ``tau_persist`` AND (b) part_frac has settled into
    ``[(1-area_band),(1+area_band)]*GT`` (the Chan-Vese equilibrium band). On fire (LATCHED), the birth
    stack for that class ramps 1.0 -> ``post_level`` over ``ramp_epochs`` (smooth event-gated hand-off,
    resume-safe via the ``__bc_*`` sidecar), handing the freed capacity from birth (recall) to boundary
    (precision). With :func:`AreaConstraintBirth` active this is DEFENSE-IN-DEPTH — the Lagrange
    multiplier self-limits area CONTINUOUSLY, the event RE-ALLOCATES the freed capacity (#302 discipline;
    unified level-set flow). The trainer flag ``--birth-completion-event`` is a store_true default-OFF
    => byte-identical when this lever is not composed. window=0 = control-config lever, no epoch budget.

    ``ramp_apply`` (v7.5 RAMP-LANDED): when True, emits ``--birth-completion-ramp`` so the trainer
    APPLIES the per-class ramp multiplier to the three birth-loss surfaces (island-amplify /
    persistence-recall / logit-adjust offset), each PER-CLASS INDEPENDENTLY. Default False =
    DETECTOR-ONLY (byte-neutral observability; the LOUD hand-off telemetry + resume state still fire)
    so the ramp landing preserves the calibration-only mode. ``--birth-completion-ramp`` requires
    ``--birth-completion-event`` (the trainer fails closed otherwise). Serial and micro-batched loss
    paths both re-read the live per-class offset/amplify/persistence cells at every loss call. With
    ``ramp_apply=True`` the
    loss surfaces are byte-identical UNTIL a class actually fires (multiplier == 1.0 pre-fire); the
    OFF path (no ``--birth-completion-ramp``) is byte-identical always."""
    if not (0.0 <= float(tau_persist) <= 1.0):
        raise ValueError(f"BirthCompletionEvent: tau_persist must be in [0,1], got {tau_persist!r}")
    if not (float(area_band) >= 0.0):
        raise ValueError(f"BirthCompletionEvent: area_band must be >= 0, got {area_band!r}")
    if int(ramp_epochs) < 1:
        raise ValueError(f"BirthCompletionEvent: ramp_epochs must be >= 1, got {ramp_epochs!r}")
    if not (0.0 <= float(post_level) <= 1.0):
        raise ValueError(f"BirthCompletionEvent: post_level must be in [0,1], got {post_level!r}")
    overrides = {"--birth-completion-event": True,
                 "--birth-completion-tau-persist": float(tau_persist),
                 "--birth-completion-area-band": float(area_band),
                 "--birth-completion-ramp-epochs": int(ramp_epochs),
                 "--birth-completion-post-level": float(post_level),
                 "--birth-completion-classes": str(classes)}
    if bool(ramp_apply):
        overrides["--birth-completion-ramp"] = True
    return Lever(
        "v75_birth_completion_event",
        overrides=overrides,
        epochs_delta=window,
        notes=("v7.5 Lever-2 Morse-Smale birth-completion event (persistence>=tau AND area in band => "
               "ramp birth stack -> post_level; hand off birth->boundary regime); defense-in-depth with "
               "AreaConstraintBirth; resume-safe; live cells routed into serial and micro-batch twins; "
               + ("RAMP APPLIED to loss surfaces (per-class)" if bool(ramp_apply)
                  else "DETECTOR-ONLY (byte-neutral; ramp not applied)")
               + "; advisory until byte-closed"))


def TemporalScrewConsistency(
    weight: float = 0.1, start_epoch: int = 0, xi_source: str = "ground_gt",
    classes: str = "0,1,2", band: float = 2.0, window: int = 0,
    start_event: str | None = None,
    sky_rotation_only: bool = False, sky_row_hi: int = 96,
) -> Lever:
    """P0 FORCE 1 — TEMPORAL SCREW-CONSISTENCY (derivation
    ``.omx/research/p0_forces_derivation_20260708.md`` §FORCE 1; task #360). DEFAULT-OFF.

    The witness renders BOTH frames of each scored pair but SegNet scores ONLY f1. Under ego-motion
    the GROUND-plane classes (Road 0, Lane 1, Undrivable 2) transform by the plane-induced homography
    ``H(ξ) = K(R - t·nᵀ/d)K⁻¹``. This term enforces the temporally-consistent constraint
    ``φ_c(f1) ≈ Warp_ξ(φ_c(·, f0))`` on the annulus over the GROUND classes:
    ``L_temp = w_t · Σ_{annulus, c∈GROUND} ‖φ_c(f1) − Warp_ξ(φ_c(f0))‖² / |annulus|``. It kills the
    measured 44%-flicker residual (L67, lane-dominated) which is exactly the failure of this constraint.

    Warp = the differentiable MLX homography path in ``tac.boundary_math.warp_real_luma_frame0``
    (``warp_frame0_native_mlx``, bit-checked vs the numpy oracle) built at SEG resolution — the 3 GROUND
    softmax-prob channels warped as an ``(H,W,3)`` field. ``φ_c`` are the through-R realized softmax
    probs (consistent with what SegNet scores). Movable(3)/MyCar(4) are NON-ground → the homography is
    wrong for them → they are NEVER warped (classes⊆{0,1,2}).

    ``xi_source`` (default ``ground_gt`` — the confound-SAFE cold start): ξ = the per-pair GT screw from
    the cached ``gt_poses`` via ``xi_from_pose_calibration`` (the SAME calibration the pose carrier uses),
    a FIXED correct warp with grad flowing ONLY to the field φ ⇒ a PURE seg-consistency regularizer with
    ZERO coupling to the (open) pose facet. ``carrier_live`` = the DUAL-USE arm (ξ = the pose carrier's
    LIVE co-adapted twist ``xi_effective(pi)``; the seg face of the unified screw), gated on a d_pose
    tripwire (telemetry ``d_pose_guard``; revert to ground_gt at a stage boundary if d_pose rises — L68:
    pose is OPEN on this vehicle, so the dual-use arm bets the fragile pose optimum on the unification
    holding — do NOT default to it).

    ``weight`` cold-start 0.1 (order-of-magnitude ≈0.1% of total loss 6.7 ⇒ far under the 40%
    ``term_domination`` alarm); ramp at STAGE BOUNDARIES ONLY toward the gradient-share≈0.44 target using
    the per-term gnorm telemetry (NEVER per-step — the GradNorm-would-mute-the-canary warning). The
    trainer flag ``--seg-temporal-screw-weight`` is default 0.0 ⇒ byte-identical when this lever is not
    composed. The micro-batch twin performs one raw-witness frame-0 SegNet call for B pairs, a
    batch-native homography warp, and one fused Metal residual-map dispatch; it never borrows the
    pose carrier's frame-0 render. ``window=0`` = loss-config lever, no epoch budget. Advisory until
    byte-closed; this training lever cannot move the canonical ``reports/latest.md`` pointer.

    ``start_event`` (v7.5 B.4 SENSOR->START WIRING, operator 2026-07-08): when set to ``'annulus_plateau'``
    the lever co-emits ``--seg-temporal-screw-start-event`` so engagement FIRES on the #333 annulus_frac
    plateau (the SAME formed-margin-boundary sensor chroma-boundary uses) and ``--seg-temporal-screw-start-
    epoch`` is demoted to the fail-safe BACKSTOP CAP. This is the correct governance of the derivation's
    ``start >= l7`` ("needs a formed partition to warp"): under ``--seg-form-unify-tau`` the discrete l7
    boundary is DISSOLVED, so the annulus_plateau event is the unify-τ-native "partition formed" signal
    that replaces it (the term acts on the annulus, so a formed margin boundary is exactly when the
    warp-consistency constraint becomes meaningful). ``None`` (default) ⇒ EVENT MODE OFF ⇒ the plain
    ``start_epoch`` fixed-epoch gate (byte-identical incumbent). When ``start_event`` is set, pass a
    POSITIVE ``start_epoch`` (the backstop cap); a naked positive ``start_epoch`` WITHOUT an event
    declaration trips the schedule-provenance gate."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"TemporalScrewConsistency: weight must be >= 0, got {weight!r}")
    if str(xi_source) not in ("ground_gt", "carrier_live"):
        raise ValueError(
            f"TemporalScrewConsistency: xi_source must be 'ground_gt' or 'carrier_live', got {xi_source!r}")
    if not (float(band) > 0.0):
        raise ValueError(f"TemporalScrewConsistency: band must be > 0, got {band!r}")
    _cls = [c.strip() for c in str(classes).split(",") if c.strip() != ""]
    if not _cls or any(c not in ("0", "1", "2") for c in _cls):
        raise ValueError(
            f"TemporalScrewConsistency: classes must be a non-empty subset of GROUND {{0,1,2}}, "
            f"got {classes!r} (Movable/MyCar are non-ground -> the homography is wrong for them)")
    if start_event is not None and str(start_event) != "annulus_plateau":
        raise ValueError(
            "TemporalScrewConsistency: start_event must be None or 'annulus_plateau' (the only wired "
            f"formed-boundary sensor for temporal-screw), got {start_event!r}")
    if bool(sky_rotation_only) and not (int(sky_row_hi) >= 1):
        raise ValueError(
            f"TemporalScrewConsistency: sky_row_hi must be >= 1 when sky_rotation_only, got {sky_row_hi!r} "
            "(the sky/ground split row; rows < sky_row_hi warp rotation-only).")
    overrides = {"--seg-temporal-screw-weight": float(weight),
                 "--seg-temporal-screw-start-epoch": int(start_epoch),
                 "--seg-temporal-screw-xi-source": str(xi_source),
                 "--seg-temporal-screw-classes": str(classes),
                 "--seg-temporal-screw-band": float(band)}
    if start_event is not None:
        overrides["--seg-temporal-screw-start-event"] = str(start_event)
    # (v7.5 B.5) SKY=ROTATION-ONLY warp stratification: the sky is at infinity (no parallax) so its
    # correct warp is H_rot=K·R·K⁻¹ (ξ translation ρ zeroed), NOT the full ground homography. Emit the
    # store_true flag ONLY when enabled (default OFF => the single full-homography warp => byte-identical).
    if bool(sky_rotation_only):
        overrides["--seg-temporal-screw-sky-rotation-only"] = True
        overrides["--seg-temporal-screw-sky-row-hi"] = int(sky_row_hi)
    return Lever(
        "temporal_screw_consistency",
        overrides=overrides,
        epochs_delta=window,
        notes=("P0 FORCE 1 temporal screw-consistency (GROUND-class annulus prob-warp MSE; ego "
               "homography H(xi); kills the 44% lane-dominated flicker residual); xi_source="
               + str(xi_source) + " (ground_gt=pure seg regularizer, ZERO pose coupling); default-OFF; "
               + ("start_event=" + str(start_event) + " (fires on annulus_plateau formed boundary; "
                  "start_epoch is the backstop cap)" if start_event is not None
                  else "start=start_epoch (fixed gate; pass start_event=annulus_plateau to event-govern)")
               + "; micro-batch routed via raw-f0 batched scorer + batch-native warp + fused Metal map"
               + "; ramp w_t at stage boundaries only; advisory until byte-closed"))


def MarginBandSatisficing(
    weight: float = 0.2, msafe: float | None = None, delta_r: float | None = None,
    headroom: float | None = None, start_epoch: int = 0, band: float = 2.0, window: int = 0,
    delta_r_artifact: str | Path = "reports/delta_R_noise_floor.json",
) -> Lever:
    """P0 FORCE 2 — MARGIN-BAND SATISFICING (derivation
    ``.omx/research/p0_forces_derivation_20260708.md`` §FORCE 2; task #360). DEFAULT-OFF.

    A one-sided hinge that stops pushing a boundary pixel once it is R-robustly SAFE:
    ``L_sat = w_s · mean_annulus relu(m_safe − m_wit)``, ``m_wit`` = the witness GT-class signed margin
    (the ``_signed`` field, #141 top1−top2). Zero gradient where ``m_wit ≥ m_safe`` ⇒ the seg-gradient
    budget reallocates BY CONSTRUCTION off the ~95%-of-pixels stable interior (GT margin p50≈0.897) onto
    the ~2.6%-area boundary band. This is the UNIWARD/Fisher satisficing reading: spend the code where
    the detector's margin is fragile.

    ``m_safe = headroom·δ_R``. Both values are resolved at config compile time by canonical law
    ``margin_band_satisficing_threshold_v1``: ``δ_R`` is read from the MEASURED artifact
    ``reports/delta_R_noise_floor.json`` and the default headroom is the smallest integer factor whose
    threshold covers that artifact's full-R annulus p95. The current artifact gives DERIVED headroom 2
    and DERIVED ``m_safe = 0.039180326461791926`` from MEASURED
    ``δ_R = 0.019590163230895963``. Headroom 3 remains an OPEN, UNMEASURED treatment rather than a
    default. A documented WAIVER fallback uses the exact artifact values if the report is unavailable.

    ``msafe`` and ``delta_r`` are compatibility overrides only: when supplied, each MUST match the
    canonical resolution within floating-point tolerance or the factory REFUSES the config. This is the
    invariant that prevents an independently hardcoded threshold from drifting back in.

    Does NOT REPLACE CE — the incumbent ``tau_softplus`` seg loss is ALREADY a temperature-τ margin loss
    (its τ→0 hard limit IS this hinge). REPLACING it early would starve region formation (the area-
    Lagrange / island-birth stack needs interior-forming pressure), so this term MASKS-BY-STAGE: it
    anneals IN at the l7 sharpening stage boundary (partition formed → now satisfice), preserving the
    τ-anneal. ``start_epoch >= l7`` (default gate matches seg_chroma_boundary/lane_band starts).

    The trainer flag ``--seg-margin-satisfice-weight`` is default 0.0 ⇒ byte-identical when not composed.
    ``window=0`` = loss-config lever. Advisory until byte-closed (pointer 0.19110 UNMOVED)."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"MarginBandSatisficing: weight must be >= 0, got {weight!r}")
    if not (float(band) > 0.0):
        raise ValueError(f"MarginBandSatisficing: band must be > 0, got {band!r}")

    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        INVARIANT_FP_TOL,
        margin_safe_lawref,
        resolve_margin_band_threshold,
    )

    resolved = resolve_margin_band_threshold(
        headroom=headroom,
        artifact_path=delta_r_artifact,
        repo_root=_REPO_ROOT,
    )
    # Numerical gate tolerance only (not a scientific tolerance): both values
    # are config scalars produced from the same deterministic fp64 law.
    _rel_tol = INVARIANT_FP_TOL
    _abs_tol = INVARIANT_FP_TOL
    if delta_r is not None and not math.isclose(
        float(delta_r), resolved.delta_r, rel_tol=_rel_tol, abs_tol=_abs_tol
    ):
        raise ValueError(
            "MarginBandSatisficing: delta_r override does not match the MEASURED artifact: "
            f"override={delta_r!r}, resolved={resolved.delta_r!r}, artifact={delta_r_artifact!s}"
        )
    if msafe is not None and not math.isclose(
        float(msafe), resolved.m_safe, rel_tol=_rel_tol, abs_tol=_abs_tol
    ):
        raise ValueError(
            "MarginBandSatisficing: msafe override violates the canonical invariant "
            "m_safe = headroom * delta_R: "
            f"override={msafe!r}, derived={resolved.m_safe!r}, "
            f"headroom={resolved.headroom!r}, delta_r={resolved.delta_r!r}"
        )
    # Self-protect even if the resolver implementation changes: the emitted
    # values may never disagree with their defining law.
    expected_msafe = resolved.headroom * resolved.delta_r
    if not math.isclose(
        resolved.m_safe, expected_msafe, rel_tol=_rel_tol, abs_tol=_abs_tol
    ):
        raise ValueError(
            "MarginBandSatisficing: REFUSE inconsistent canonical resolution: "
            f"m_safe={resolved.m_safe!r}, headroom*delta_R={expected_msafe!r}"
        )
    return Lever(
        "margin_band_satisficing",
        overrides={"--seg-margin-satisfice-weight": float(weight),
                   "--seg-margin-satisfice-msafe": resolved.m_safe,
                   "--seg-margin-satisfice-delta-r": resolved.delta_r,
                   "--seg-margin-satisfice-headroom": resolved.headroom,
                   "--seg-margin-satisfice-start-epoch": int(start_epoch),
                   "--seg-margin-satisfice-band": float(band)},
        epochs_delta=window,
        lawrefs={
            "--seg-margin-satisfice-msafe": margin_safe_lawref(
                headroom=resolved.headroom, artifact_path=delta_r_artifact
            )
        },
        constant_manifest={
            # Keyed by the emitted --flag (TypedLever validator requires constant_manifest keys to
            # be override flags); _merge_lever_constant_manifests normalizes to the underscore key
            # 'seg_margin_satisfice_msafe' at the cfg.constants_manifest surface.
            "--seg-margin-satisfice-msafe": {
                **resolved.lawref_manifest,
                "single_value_owner": "margin_band_satisficing_threshold_v1",
            }
        },
        notes=("P0 FORCE 2 margin-band satisficing (one-sided relu(m_safe - m_wit) on the annulus; "
               f"m_safe={resolved.m_safe} DERIVED-LIVE = headroom({resolved.headroom}) * "
               f"delta_R({resolved.delta_r}) MEASURED R-noise "
               "floor); frees the interior gradient budget onto the band (UNIWARD satisficing); "
               "MASK-BY-STAGE at l7 preserves the tau-anneal; default-OFF; advisory until byte-closed"))


def HorizonWeightedMargin(
    weight: float = 0.0, target: float = 0.5, margin_lo: float = 0.3, margin_hi: float = 0.5,
    row_lo: int = 96, row_hi: int = 288, start_epoch: int = 0, window: int = 0,
    *, stage_share_derived_live: bool = False, scientific_declaration: bool = False,
) -> Lever:
    """v7.5 B.5 — HORIZON-WEIGHTED MARGIN (#169; derivation
    ``.omx/research/dseg_reducibility_gt_margin_verdict_20260623.md``). DEFAULT-OFF.

    The 0-byte SHARED-structure d_seg lever. That verdict MEASURED (exact frozen-SegNet argmax, real GT,
    through-R) that the residual d_seg flips split by GT top-2 margin: the ``<0.05``-margin flips are
    IRREDUCIBLE frozen-SegNet label-noise (a near-coin-flip, ~193× concentrated — chasing them is FITTING
    NOISE), while the flips at GT margin ∈ ``[0.3, 0.5]`` are the ONLY ones both REDUCIBLE and
    STABLY-DECIDED (oracle ceiling ΔS≈0.024 at margin≥0.3 / 0.012 at margin≥0.5). 97.8% of the frontier
    d_seg lives in the horizon band (SEG rows ~96-288, where sky/far meets the ground classes).

    So this is a one-sided hinge ``L_hz = w_h · mean_{mask} relu(m_target − m_wit)`` on the SHARED realized
    through-R witness GT-class margin ``m_wit`` (``_signed``, #141; NO 2nd SegNet forward, 0 archive
    bytes), STRATIFIED to the θ-independent mask ``(row ∈ [row_lo, row_hi)) AND (GT margin ∈ [lo, hi])`` —
    pushing ONLY the reducible confident-GT band toward the ``target`` ceiling and EXCLUDING the ``<lo``
    label-noise by construction. Zero gradient where ``m_wit ≥ target`` (satisficing — do not over-push
    into the noise regime). Sister of :func:`MarginBandSatisficing` (same one-sided-hinge shape on
    ``_signed``; DIFFERENT stratification — horizon×margin-band vs the full annulus).

    The trainer flag ``--seg-horizon-margin-weight`` is default 0.0 ⇒ byte-identical when not composed.
    ``window=0`` = loss-config lever. **A/B arm, NOT a claim** — the exit criterion (owed n600 A/B) must
    distinguish REAL d_seg recovery on the ``[0.3,0.5]`` band from chasing the ``<lo`` label-noise
    (re-run ``tools/measure_dseg_reducibility_gt_margin.py --n-pairs 600`` on the ON vs OFF ckpts;
    require the surviving flips shift to HIGHER GT margin, else terminal-finding). Advisory until
    byte-closed (pointer 0.19110 UNMOVED)."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"HorizonWeightedMargin: weight must be >= 0, got {weight!r}")
    if not (float(margin_lo) < float(margin_hi)):
        raise ValueError(
            f"HorizonWeightedMargin: margin_lo ({margin_lo!r}) must be < margin_hi ({margin_hi!r}) — the "
            "reducible GT-margin band [lo,hi] must be non-empty (#169 measured band [0.3,0.5]).")
    if not (int(row_lo) < int(row_hi)):
        raise ValueError(
            f"HorizonWeightedMargin: row_lo ({row_lo!r}) must be < row_hi ({row_hi!r}) — the horizon band "
            "must be a non-empty SEG-row range (#169 measured band rows ~96-288).")
    if not (float(target) > 0.0):
        raise ValueError(f"HorizonWeightedMargin: target must be > 0, got {target!r}")
    if stage_share_derived_live and not (0.0 < float(weight) < 1.0):
        raise ValueError(
            "HorizonWeightedMargin: derived-live mode interprets weight as the requested "
            f"single-force loss share and requires 0 < weight < 1, got {weight!r}")
    if stage_share_derived_live and int(start_epoch) <= 0:
        raise ValueError(
            "HorizonWeightedMargin: derived-live mode requires a positive typed stage boundary")
    overrides = {"--seg-horizon-margin-weight": float(weight),
                 "--seg-horizon-margin-target": float(target),
                 "--seg-horizon-margin-lo": float(margin_lo),
                 "--seg-horizon-margin-hi": float(margin_hi),
                 "--seg-horizon-row-lo": int(row_lo),
                 "--seg-horizon-row-hi": int(row_hi),
                 "--seg-horizon-margin-start-epoch": int(start_epoch)}
    if stage_share_derived_live:
        # Select the real boundary-measurement consumer.  The seven HWM
        # scientific scalars above retain exactly seven LawRefs; this Boolean is
        # a mode declaration, not an eighth fitted scientific constant.
        overrides["--seg-horizon-margin-derived-live"] = True
    lawrefs: dict = {}
    constant_manifest: dict = {}
    receipt_schemas: dict = {}
    if scientific_declaration:
        scientific = {
            flag: value for flag, value in overrides.items()
            if flag != "--seg-horizon-margin-derived-live"
        }
        from tac.witness_dsl.lawref import LADDER_DERIVED_LIVE
        lawrefs, constant_manifest = _v9_scientific_constant_custody(
            "horizon_weighted_margin_hinge_v1",
            scientific,
            provenance=(
                "V9 HORIZON isolation declaration: rank-2 47.3%; requested 15% "
                "single-force share resolves at the frozen n600 boundary via "
                "w_h=(0.15/0.85)*L_o/max(L_h,eps)"
            ),
            ladder_by_flag={"--seg-horizon-margin-weight": LADDER_DERIVED_LIVE},
        )
        receipt_schemas = dict.fromkeys(overrides, "hwm_v9_stage_share_boundary.v1")
    return Lever(
        "horizon_weighted_margin",
        overrides=overrides,
        epochs_delta=window,
        notes=(
            "v7.5 B.5 #169 horizon-weighted margin (one-sided relu(m_target - m_wit) on the "
            f"SHARED _signed, STRATIFIED to horizon rows [{int(row_lo)},{int(row_hi)}) AND "
            f"GT-margin [{float(margin_lo):.2f},{float(margin_hi):.2f}]); pushes ONLY the "
            "reducible confident-GT band, EXCLUDES the <lo irreducible label-noise; 0-byte "
            "SHARED-structure; A/B arm NOT a claim (oracle ceiling dS~0.024); advisory until "
            "byte-close"
        ),
        lawrefs=lawrefs,
        constant_manifest=constant_manifest,
        runtime_receipt_schemas=receipt_schemas,
    )


def SegChromaBoundary(
    weight: float = 0.0, margin_band: float = 1.0, start_epoch: int = 0, window: int = 0,
) -> Lever:
    """LEVER-4c — ANNULUS-DIRECTED CHROMA-BOUNDARY MATCH (chroma DOF probe a3e9f0bd GREEN 2026-07-03; #276
    chroma-DOF; operator 2026-06-25 "Chroma too"; CLAUDE.md "Chroma is a d_seg lever"). DEFAULT-OFF.

    SegNet reads RGB ⇒ its per-pixel argmax depends on CHROMA. The probe MEASURED (n96, 100% L*-match to the
    frozen SegNet) that removing chroma (constant-luma) FLIPS 7.54% of Lane→Road + 4.38% of Movable→Undriv,
    with 93.4% of chroma-flips in the ``margin < 1`` fragile ANNULUS (→ 33.7% at margin<0.25); the flips are
    LUMA-INDEPENDENT (desat-only still flips 3.1%) and the SegNet margin-gradient energy is 78.8% luma /
    21.2% chroma — so chroma is a PROVEN INDEPENDENT d_seg BOUNDARY SHARPENER, ORTHOGONAL to the geometry
    levers. The witness UNDER-exploits it (its rendered chroma converges to a near per-class CONSTANT palette;
    nothing supervises per-pixel chroma).

    So this is an additive chroma-MATCH loss ``L_c = w · mean_{ann} ‖chroma(f1) − chroma(GT)‖²`` on the
    SHARED realized-through-R render ``f1`` (the SAME render the SegNet forward / ``_signed`` come from — NO
    2nd render, NO 2nd SegNet forward), over the θ-independent fragile annulus ``1[GT margin < band]``.
    Chroma := ``rgb − BT.601-luma`` ⇒ LUMA-INVARIANT by construction ⇒ ORTHOGONAL to every luma lever (NOT a
    full-RGB reconstruction). It pulls the per-pixel RGB head (``self.out``, which HAS per-pixel chroma
    capacity) to paint the boundary chroma the constant palette can't.

    The trainer flag ``--seg-chroma-boundary-weight`` is default 0.0 ⇒ byte-identical when not composed (the
    branch is gated on ``chroma_bnd_w > 0`` AND the GT-chroma/annulus providers stay None). ``window=0`` =
    loss-config lever; ``start_epoch`` is the stage-boundary engage (re-treats the spike-guard; NEVER
    per-step). ``MicroBatch(B>1)`` consumes the same GT-chroma and annulus providers in a fused Metal
    map, preserving each pair's own weighted denominator before the batch mean. Sister of
    :func:`HorizonWeightedMargin` (both 0-byte SHARED-structure annulus levers on the SAME render;
    DIFFERENT quantity — chroma appearance-match vs the ``_signed`` margin hinge).

    **A/B arm, NOT a claim** — the 7.54%/4.38%/93.4% are a MEASURED chroma-REMOVAL ABLATION (the DOF
    EXISTENCE proof, eq ``chroma_decides_lane_and_movable_at_annulus_v1``), NOT the ADD-BACK score ΔS of THIS
    chroma-match term through the witness. That ΔS is UNMEASURED; the exit criterion is a CONVERGED n600
    byte-close A/B (the surviving annulus flips must shift toward the GT chroma, else terminal-finding).
    Advisory until byte-closed; this training lever cannot move the canonical ``reports/latest.md``
    pointer."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"SegChromaBoundary: weight must be >= 0, got {weight!r}")
    if not (float(margin_band) > 0.0):
        raise ValueError(
            f"SegChromaBoundary: margin_band ({margin_band!r}) must be > 0 — a non-positive band selects NO "
            "annulus pixel (the chroma-match term would silently do nothing; LEVER-4c measured band 1.0).")
    if not (int(start_epoch) >= 0):
        raise ValueError(f"SegChromaBoundary: start_epoch must be >= 0, got {start_epoch!r}")
    return Lever(
        "seg_chroma_boundary",
        overrides={"--seg-chroma-boundary-weight": float(weight),
                   "--seg-chroma-boundary-margin-band": float(margin_band),
                   "--seg-chroma-boundary-start-epoch": int(start_epoch)},
        epochs_delta=window,
        notes=(
            "LEVER-4c annulus-directed chroma-boundary match (additive w*mean_ann "
            "||chroma(f1)-chroma(GT)||^2 on the SHARED realized-through-R render, annulus "
            f"|GT margin|<{float(margin_band):.2f}); chroma=rgb-BT.601-luma = LUMA-INVARIANT "
            "=> ORTHOGONAL to luma levers; PROVEN independent d_seg boundary SHARPENER "
            "(probe a3e9f0bd: 93.4% of chroma-flips in margin<1); micro-batch routed with "
            "per-pair normalization + one fused Metal map; 0-byte SHARED-structure; "
            "start_epoch stage-gate; A/B arm NOT a claim (removal-ablation != add-back dS); "
            "advisory until byte-close"
        ),
    )


def TieLocusDisplacement(
    weight: float = 0.3, start_epoch: int = 0, v_band: float = 1.0,
    edge_weight_source: str = "pa_flipmass",
    edge_weight_path: str = "reports/pa_edge_weights.json",
    ref_domain: str = "seg384", window: int = 0,
) -> Lever:
    """P0 FORCE 3 — TIE-LOCUS NORMAL-DISPLACEMENT (derivation
    ``.omx/research/p0_forces_derivation_20260708.md`` §FORCE 3; task #360). DEFAULT-OFF.

    The d_seg currency IS boundary displacement (FEED-PA: 100% of the achievable floor is boundary
    placement). The machinery is ALREADY BUILT — the ``subpix`` term (trainer ~L4640) supervises the
    witness sub-pixel margin ratio ``t_wit = M_w/(M_w+M_q)`` toward the GT sub-pixel ratio ``t_ref``
    over genuine-V inter-class straddles (fully differentiable through the through-R ``_signed`` field).
    ``δn = |t_wit − t_ref|`` IS the sub-pixel normal-displacement error.

    This lever WRAPS the existing subpix flags (``--seg-subpix-boundary-weight/-start-epoch/-v-band``)
    and adds the MISSING piece: the flips are NOT uniform over straddles — they concentrate on
    Road-adjacent edges (Road hub; Road↔Lane = 41% of Road's flips, FEED-PA destination matrix). It
    weights each straddle by its adjacency-edge flip-mass share ``W_e[c_a,c_b]`` (a 5×5 symmetric matrix
    STAMPED from the measured P-A artifact — ``edge_weight_source=pa_flipmass`` reads
    ``edge_weight_path``; falls back to ``uniform`` + a LOUD WARN if the artifact is absent — NEVER a
    hardcoded guess). ``edge_weight_source=uniform`` = the pre-existing subpix behaviour, byte-identical.

    ``ref_domain`` (§FORCE 4 fold — NOT a second term): ``seg384`` (default) computes ``t_ref`` at 384
    (correct for the TRAINING loss, which is already post-R — the R-phase is handled by training-through-
    R). ``camera874_dphase`` reserves the 874-res D-sampling-phase ``t_ref`` for the decode-time render-
    placement Consumer B (SPEC-ONLY, not built) — the TRAINING loss is domain-invariant by derivation
    (already through-R) so it is IDENTICAL to seg384 for training; the flag records the operator's decode
    domain intent (telemetry-stamped) for the future consumer.

    The trainer flag ``--seg-subpix-boundary-weight`` is default 0.0 ⇒ byte-identical when not composed;
    ``--seg-subpix-edge-weight-source`` default ``uniform`` in the trainer keeps the incumbent subpix
    byte-identical, so this lever ELECTS pa_flipmass explicitly. Highest-EV of the P0 forces (the
    precision counter-force the FEED-roadfloor bug named). ``window=0`` = loss-config lever. Advisory
    until byte-closed (pointer 0.19110 UNMOVED)."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"TieLocusDisplacement: weight must be >= 0, got {weight!r}")
    if not (float(v_band) > 0.0):
        raise ValueError(f"TieLocusDisplacement: v_band must be > 0, got {v_band!r}")
    if str(edge_weight_source) not in ("uniform", "pa_flipmass"):
        raise ValueError(
            f"TieLocusDisplacement: edge_weight_source must be 'uniform' or 'pa_flipmass', "
            f"got {edge_weight_source!r}")
    if str(ref_domain) not in ("seg384", "camera874_dphase"):
        raise ValueError(
            f"TieLocusDisplacement: ref_domain must be 'seg384' or 'camera874_dphase', got {ref_domain!r}")
    return Lever(
        "tie_locus_displacement",
        overrides={"--seg-subpix-boundary-weight": float(weight),
                   "--seg-subpix-boundary-start-epoch": int(start_epoch),
                   "--seg-subpix-boundary-v-band": float(v_band),
                   "--seg-subpix-edge-weight-source": str(edge_weight_source),
                   "--seg-subpix-edge-weight-path": str(edge_weight_path),
                   "--seg-subpix-ref-domain": str(ref_domain)},
        epochs_delta=window,
        notes=("P0 FORCE 3 tie-locus normal-displacement (WRAPS the built subpix term; adds flip-density "
               "edge weighting W_e stamped from the FEED-PA destination matrix, source="
               + str(edge_weight_source) + "; Road-hub / Road<->Lane heaviest); ref_domain="
               + str(ref_domain) + " (#4 R-phase FOLDS in, training is post-R); highest-EV precision "
               "counter-force; default-OFF; advisory until byte-closed"))


def PhaseAdvectionConsistency(  # T1 cross-pair phase-advection (flicker deep-dive); DSL keyword-cased
    weight: float = 0.0, start_epoch: int = 0, classes: str = "0,1,2", band: float = 2.0,
    gap_xi: str = "interp", ref: str = "gt_advected", window: int = 0,
    start_event: str | None = None,
) -> Lever:
    """T1 — CROSS-PAIR PHASE-ADVECTION CONSISTENCY (flicker deep-dive design memo
    ``.omx/research/flicker_transform_geometry_term_design_20260710.md`` §4 T1). DEFAULT-OFF.

    THE d_seg endgame lever. The witness has converged to the temporal-majority ORACLE floor: the GT
    stride-2 SPIKE rate is 0.005318 ≈ the converged residual ~0.005 (L67). sub-0.15's d_seg
    (0.0008–0.0012) is 4.5–7× BELOW that floor ⇒ NO temporally-smooth-in-LABEL witness reaches it. The
    floor is pierceable ONLY by APPEARANCE-PHASE faithfulness: the spikes are DETERMINISTIC per-pair
    functions of the camera frame, so a witness that carries the boundary's sub-pixel PHASE inherits
    them for free (existence proof: real-frame content through R reaches d_seg 0.00086 < 0.00532).

    T1 is the genuinely-missing operator: (Force-1's se(3) ξ-transport A_ξ) ∘ (Force-3's sub-pixel tie
    coordinate t_wit) on the support NEITHER covers — the CROSS-PAIR × SCORED-frame (stride-2 f1)
    sequence. Force-3 supervises ``t_wit(p) → the RAW per-pair GT tie`` (noisy, σ≈0.09–0.43 px through
    the jittering chain); T1 supervises ``t_wit(p) → the ξ-ADVECTED GT tie of the PREVIOUS scored pair
    p-1``. Composed, the two terms implement the optimal SHRINKAGE of the noisy per-pair phase targets
    toward the ξ-advected (predictable) trajectory — the witness fits the predictable flicker channel
    without STORING it (the phase trajectory is low-dim and mostly generated by ξ).

    ``L_phase = w_p · mean_{annulus∩active, GROUND} ( t_wit(p) − warp_Aξ(GT_tie[p-1]) )²``.

    **Cross-pair wiring (verdict): ZERO batching change.** The entire cross-pair coupling lives in a
    θ-INDEPENDENT precomputed target (``warp_Aξ(GT_tie[p-1])`` — cached poses + cached GT margins), so
    the in-loss term is per-pair-LOCAL: it reads ONLY pair p's realized ``_signed`` + pair p's
    precomputed target. It therefore fits the incumbent random-permutation per-pair ``value_and_grad``
    with no change (respects #240 verdict-batch chunking + the launcher memory-preflight; no OOM class).
    ``ref='gt_advected'`` (DEFAULT, READY) is this mode. ``ref='gt_advected_with_own_tie_fallback'``
    is the EVENT-FALLBACK phase supervision force (SPEC_v10 §13.1 row 1; FEED-lane-gain §4b):
    ``t_ref := where(ref_active, advected_prev_tie, own_gt_tie)`` with
    ``weight := ann ∧ ground ∧ (ref_active ∨ own_active)`` — advect-where-persistent,
    target-where-born, covering the MEASURED 26.3% birth/fast-moved straddle coverage gap the
    transport-only T1 leaves phase-unsupervised (birth-SILENT). STATELESS by construction: NO
    per-island persistence hold (the memo anti-scope — a hold would fight GT's genuine deaths);
    same θ-independent per-pair-local containment as T1 (zero batching change). ``ref='witness_cached'`` (the memo's
    fully-differentiable witness-self-consistency formula ``t_wit(p) vs advected t_wit(p-1)``) is
    SPECIFIED but OWED (needs a per-pair stop-grad tie cache OR sequential-pair batching) — the trainer
    fails loud if selected.

    Reuses the SHARED realized ``_signed`` (NO 2nd SegNet forward). Primitives:
    ``tac.boundary_math.phase_primitives`` (t_wit + A_ξ advect + GT tie targets + residual) — the SAME
    code Force-1/Force-3 use, not duplicated. f0 is UNTOUCHED (Force-1 owns f0↔f1); GROUND classes
    {0,1,2} only (the ground homography is wrong on Movable/MyCar).

    ``weight`` derived at gradient-share ≈ 0.4× the subpix term (blink-back 0.42 / L67 0.44); cap
    ≤10% of total loss (term-domination guard L4). Ramp at STAGE BOUNDARIES ONLY. ``start_epoch`` MUST
    be ≥ l7 (needs a formed partition + tie field to advect). ``gap_xi='interp'`` (DEFAULT) estimates
    the odd→even inter-pair gap screw as the se(3)-composed ``ξ_gap≈½(ξ_pi+ξ_pi+1)`` (a train-time
    regularizer target; gap approximation tolerated per memo). The trainer flag
    ``--seg-phase-advect-weight`` is default 0.0 ⇒ byte-identical when not composed. Because the
    cross-pair context is already baked into theta-independent per-pair providers, the micro-batch
    twin stacks those rows and evaluates the phase residual in one fused Metal dispatch across B.
    ``window=0`` = loss-config lever, no epoch budget. Advisory until byte-closed; this training lever
    cannot move the canonical ``reports/latest.md`` pointer."""
    if not (float(weight) >= 0.0):
        raise ValueError(f"PhaseAdvectionConsistency: weight must be >= 0, got {weight!r}")
    if not (float(band) > 0.0):
        raise ValueError(f"PhaseAdvectionConsistency: band must be > 0, got {band!r}")
    _cls = [c.strip() for c in str(classes).split(",") if c.strip() != ""]
    if not _cls or any(c not in ("0", "1", "2") for c in _cls):
        raise ValueError(
            f"PhaseAdvectionConsistency: classes must be a non-empty subset of GROUND {{0,1,2}}, "
            f"got {classes!r} (Movable/MyCar are non-ground -> the ground homography is wrong for them)")
    if str(gap_xi) not in ("interp", "offline_homography"):
        raise ValueError(
            f"PhaseAdvectionConsistency: gap_xi must be 'interp' or 'offline_homography', got {gap_xi!r}")
    if start_event is not None and str(start_event) not in ("label_floor", "ncde_dseg"):
        raise ValueError(
            "PhaseAdvectionConsistency: start_event must be None, 'label_floor' (law-5 floor->phase-"
            "tail hand-off) or 'ncde_dseg' (SPEC_v10 §13.2 per-force event entry — the #344 NCDE "
            f"d_seg slope-flatten/basin), got {start_event!r}")
    if str(ref) not in ("gt_advected", "gt_advected_with_own_tie_fallback", "witness_cached"):
        raise ValueError(
            "PhaseAdvectionConsistency: ref must be 'gt_advected', "
            f"'gt_advected_with_own_tie_fallback' or 'witness_cached', got {ref!r}")
    return Lever(
        "phase_advection_consistency",
        overrides={"--seg-phase-advect-weight": float(weight),
                   "--seg-phase-advect-start-epoch": int(start_epoch),
                   "--seg-phase-advect-classes": str(classes),
                   "--seg-phase-advect-band": float(band),
                   "--seg-phase-advect-gap-xi": str(gap_xi),
                   "--seg-phase-advect-ref": str(ref),
                   **({"--seg-phase-advect-start-event": str(start_event)}
                      if start_event is not None else {})},
        epochs_delta=window,
        notes=("T1 cross-pair phase-advection consistency (t_wit(p) -> ξ-advected GT tie of p-1; "
               "composes Force-1's A_ξ ∘ Force-3's tie coordinate on the cross-pair scored-frame "
               "support); THE d_seg endgame lever (pierces the 0.0053 temporal-majority floor via "
               "appearance-phase correlation); shrinkage toward the ξ-advected predictable channel; "
               "gap_xi=" + str(gap_xi) + " ref=" + str(ref) + " (gt_advected=READY θ-indep target, "
               "micro-batch routed via stacked providers + fused Metal map; witness_cached=OWED); "
               "GROUND classes only; f0 untouched; default-OFF; advisory until byte-closed"))


def AACoverageRender(mode: str = "supersample", ss: int = 2,
                     grid_h: int = 384, grid_w: int = 512,
                     ipe_footprint: float = 1.0, window: int = 100) -> Lever:
    """FEED-07b lever #1 (#220): AA COVERAGE-INTEGRATED render + grid≥384 — the gate's "#1 MEASURED
    islands lever" (own gate verdict, ``tools/levelset_gate_discriminators_n600.py`` DAG FEED-ly,
    ``[macOS-CPU advisory] NON-PROMOTABLE``). POINT-sampling the witness render grid ERASES the
    finest-scale island (class-1 LANE dash) structure — it aliases out below the render Nyquist;
    FOOTPRINT/COVERAGE-integrated rendering RECOVERS it (oracle-R d_seg floor 0.00091 @ 384x512
    area/AA vs 0.00247 @ 192x256 point; class-1 LANE recall +0.38, 0.56→0.94). ~0-RATE: the AA
    render is a DETERMINISTIC decode-time op (inflate.py runs it FREE within the 30-min budget,
    rule 118); the archive weights+codes are UNCHANGED — AA moves d_seg WITHOUT the rate term.

    ``mode`` ∈ {"supersample", "ipe"} (#220 completion 2026-07-09 — the factory now expresses BOTH
    AA charts, killing the ``render_aa: "ipe"`` raw-override config-orphan in witness_autoconfig
    v6 so the DSL HOLDS the whole lever):
      * ``supersample`` (default; the AUTHORITY): render at ss×grid then AREA/box-downsample to the
        base grid (EXACT ss×ss block-mean = ground-truth pixel-footprint integration). ``ss=1`` is
        byte-identical (downsample is identity). Cost ss² × the witness forward.
      * ``ipe``: mip-NeRF cone attenuation of the curvelet basis (analytical, ~0-compute; the cheap
        decode-time PROXY to supersample, and the SELF-ORIENT-compatible chart — supersample is
        fail-closed against --self-orient's per-pair fine dir-feats / against --dseg-aware-taper's
        untapered fine grid; ipe attenuates the SAME base curvelet columns AFTER, so it composes).

    ``grid_h``/``grid_w``: the base (H,W) the R operator sees — ENFORCED ≥384 per axis (the "grid≥384"
    half of #220: R bicubic-UPsamples the render grid to camera 874×1164, so the achievable-through-R
    floor drops monotonically as the grid rises; <384 defeats the lever's own measured floor and is
    REFUSED). Default 384x512 = the levelset trainer's own default (byte-identical grid contribution).
    Raising it (512x768, ...) trades forward compute for a lower floor and REQUIRES a matching GT L*
    cache resolution (the trainer's lstar_shape check fails closed otherwise) — set only with a
    matching --gt-cache.

    Mechanism: ``tac.boundary_math.aa_sdf_observation_render`` (numpy-fp32 authority + MLX twin,
    parity ≥0.9997); law: ``tac.canonical_equations`` ``aa_supersample_lane_recall_lift_v1``. Trainer
    flags LANDED (#224 wire-in; default-off byte-identity proven by
    ``tools/wire_in_224_byte_identical_smoke.py``) — this factory is the DSL Lever that ARMS them;
    UN-composed the trainer defaults ``--render-aa none`` = byte-identical. STRUCTURAL render lever:
    the F2 resume-divergence guard refuses a resume that adds/changes ``render_aa`` /
    ``aa_supersample`` (``__cfg_render_aa`` / ``__cfg_aa_supersample``), so an A/B fires it from a
    FRESH run (or a matching resume).

    COMPOSE GUARD RESOLVED (#220 unblock, 2026-07-07): ``aa_sdf_observation_render`` invokes
    ``compose_fn`` AFTER the box-downsample (at the BASE grid), so supersample COMPOSES with the
    base-grid compose levers ``--residual-mode`` / ``--seed-islands`` / ``--lane-render-band`` by
    construction (``_validate_aa_compose_compat`` retained; accepts every tracked combination).
    CAVEAT unchanged: the ORTHOGONAL --self-orient × supersample fine-dir-feats memory/wall-clock
    gate (``--aa-self-orient-fine-mode`` refuse default) still applies over a --self-orient base
    (use ``mode="ipe"`` there).

    VERDICT SCOPE: the lane-recall lift is oracle-R MEASURED; the through-witness-training ΔS is
    ASSUMED_AWAITING_VERIFICATION until a converged byte-close A/B measures it (means ≠ ends: this
    factory BUILDS the arm; makes NO score claim; pointer UNMOVED)."""
    if mode not in ("supersample", "ipe"):
        raise ValueError(f"AACoverageRender: mode must be 'supersample' or 'ipe', got {mode!r}")
    if int(ss) < 1:
        raise ValueError(f"AACoverageRender: ss must be >= 1, got {ss!r}")
    if int(grid_h) < 384 or int(grid_w) < 384:
        raise ValueError(
            f"AACoverageRender: grid must be >=384 per axis (islands need grid>=384 through R; the "
            f"'grid>=384' half of #220), got ({int(grid_h)}x{int(grid_w)})")
    overrides: dict = {"--render-aa": mode,
                       "--render-h": int(grid_h),
                       "--render-w": int(grid_w)}
    if mode == "supersample":
        overrides["--aa-supersample"] = int(ss)
    else:  # ipe
        overrides["--aa-ipe-footprint"] = float(ipe_footprint)
    return Lever(
        "FEED_07b_aa_coverage_render",
        overrides=overrides,
        epochs_delta=window,
        notes=(f"#220 AA coverage render ({mode}) + grid>=384 — #1 measured islands lever; "
               "composes with residual-mode/seed-islands/lane-render-band (compose-after-"
               "downsample landed 2026-07-07); self-orient fine-mode gate still applies to "
               "supersample (use mode=ipe over a self-orient base)"),
    )


def StepNativeActivation(beta_start: float = 1.0, beta_end: float = 8.0,
                         anneal: str = "linear", window: int = 100,
                         *, basis: str = "annealed_hosc", omega: float = 1.0,
                         finer_bias_init: bool = False, finer_bias_k: float = 10.0,
                         scientific_declaration: bool = False) -> Lever:
    """FEED-07b lever #2 (#310, capstone lever #5 — duty-to-measure #2, NEVER-FIRED): the step-native
    activation chart, deep-math L∞-at-edge optimal for the piecewise-constant argmax target (error
    confined to the flip band, O(1) params/edge, no Gibbs). §OPERATOR PRIORITY measured-lever #5.

    REAL-FLAG ROUTE (never-invent-flags): this trainer's ``--activation`` choices are ``wire|hosc|relu``
    — there is NO ``step_basis`` token — so the step-native route is the hosc BETA-ANNEAL (FEED-fb):
    hosc = ``tanh(beta*sin(omega*u))`` and beta→∞ IS the step limit (a square-wave / partition
    indicator); ``--hosc-beta-end > --hosc-beta`` step-sharpens as the SDF partition forms. This
    factory maps ONLY to EXISTING trainer flags (--activation/--hosc-beta/--hosc-beta-end/
    --hosc-beta-anneal/--hosc-omega/--finer-bias-init/--finer-bias-k), so ``WitnessProgram.validate``
    passes.

    **NEVER a FIXED beta** (fail-closed, the #1 constraint): a CONSTANT beta (``beta_end ==
    beta_start``) is the MEASURED saturation-death — ``tanh(4*sin)`` saturates → gradient vanishes →
    AdamW random-walks → d_seg RISES (DAG FEED 2026-06-25a + FEED-ly). The guard now routes through the
    shared ``tac.boundary_math.step_native_activation.validate_step_native_config`` predicate, which
    REJECTS a constant beta (the prior ``beta_start <= beta_end`` guard permitted it), an invented
    basis/anneal, a non-positive beta, and ``basis='step_basis'`` WITHOUT FINER. The default
    ``beta_start=1.0`` starts where ``tanh(sin)`` is near-linear (gradients flow) and anneals UP — never
    starting in the saturating regime (the prior default 4.0 started AT the measured-divergent point).

    ``basis`` ∈ {"annealed_hosc" (default), "step_basis"}: ``step_basis`` REQUIRES the FINER++
    variable-periodic first-layer bias (``finer_bias_init=True``) — the published stability fix (arXiv
    2407.19434) that makes the sharper step the "stable trainable-slope survivor"; ``annealed_hosc``
    leaves FINER optional. ``omega`` = the periodic frequency; ``finer_bias_k`` = the FINER bias range
    U(-k, k). Setting ``finer_bias_init=True`` here arms the FINER flags in the SAME lever (a
    self-sufficient measured-safe preset — no need to separately compose :func:`FinerBiasInit`, which
    remains available for FINER-only A/Bs). window = the epoch budget for the step-sharpen span.

    means != ends: this factory ARMS the mechanism; it makes NO score claim; pointer UNMOVED. The
    step-native d_seg effect is ASSUMED_AWAITING_VERIFICATION until a byte-close A/B lands (the #310
    sweep is the owed anchor). Equations leg: ``step_native_activation_edge_optimality_v1``. Mechanism +
    shared safety predicate: ``tac.boundary_math.step_native_activation``."""
    from tac.boundary_math.step_native_activation import validate_step_native_config

    problems = validate_step_native_config(
        basis=basis, beta_start=beta_start, beta_end=beta_end, beta_anneal=anneal,
        omega=omega, finer_bias_init=finer_bias_init)
    if problems:
        raise ValueError("StepNativeActivation: " + "; ".join(problems))
    overrides = {"--activation": "hosc",
                 "--hosc-beta": float(beta_start),
                 "--hosc-beta-end": float(beta_end),
                 "--hosc-beta-anneal": anneal,
                 "--hosc-omega": float(omega)}
    if finer_bias_init:
        # arm the FINER++ stabilizer in the SAME lever (byte-identical when off: the flags are simply
        # not emitted). --finer-bias-k only meaningful with --finer-bias-init on.
        overrides["--finer-bias-init"] = True
        overrides["--finer-bias-k"] = float(finer_bias_k)
    lawrefs: dict = {}
    constant_manifest: dict = {}
    receipt_schemas: dict = {}
    if scientific_declaration:
        # Numeric activation-shape constants get executable LawRefs.  The two
        # categorical tokens (activation/anneal) remain parser-enforced tokens;
        # numeric InputRef must not be stretched to smuggle strings.
        numeric = {
            flag: value for flag, value in overrides.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        lawrefs, constant_manifest = _v9_scientific_constant_custody(
            "step_native_activation_edge_optimality_v1",
            numeric,
            provenance=(
                "V9 STEP isolation declaration: rank-3 34.2%; annealed HOSC beta "
                "1->8 is a distinct treatment from the sealed event endpoint 3.177"
            ),
        )
        receipt_schemas = dict.fromkeys(overrides, "v9_config_compile.v1")
    return Lever(
        "FEED_07b_step_native_activation",
        overrides=overrides,
        epochs_delta=window,
        notes=(f"#310 step-native chart via hosc beta-anneal ({basis}: beta {beta_start}->{beta_end} "
               f"@omega {omega}, FINER {finer_bias_init}; beta->inf = step limit; NEVER fixed-beta — "
               "anneal only); L-inf-at-edge optimality; duty-to-measure #2, #310 sweep owed"),
        lawrefs=lawrefs,
        constant_manifest=constant_manifest,
        runtime_receipt_schemas=receipt_schemas,
    )


def MuonWarmStart(lr_final_frac: float = 0.1, window: int = 0) -> Lever:
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


def FilmPolarChartSPELManifoldMuon(
    window: int = 0,
    *,
    start_epoch: int | None = None,
) -> Lever:
    """Default-OFF FiLM-only polar-chart MCSD/SPEL finisher.

    This is the honest round-2 fallback for the registered
    ``film_polar_chart_exact_manifold_muon_finisher`` ticket.  It reuses the
    typed Muon-stage learning-rate, momentum, Newton--Schulz, and schedule
    settings, but removes ``film.weight`` from ambient Muon dynamics and
    updates the function-preserving chart ``W=QH0`` with a single-loop
    Riemannian-gradient / projected spectral-LMO step.  It is not the exact
    nested tangent-dual solver and must not be reported as such.

    ``window=0`` means this lever changes optimizer geometry only; the owning
    program supplies the finisher window.  The trainer refuses the flag unless
    a real ``--muon-start-epoch`` is also compiled, preventing a silent no-op.
    """

    if int(window) < 0:
        raise ValueError("FilmPolarChartSPELManifoldMuon: window must be >= 0")
    overrides: dict = {"--film-polar-chart-spel": True}
    if start_epoch is not None:
        if int(start_epoch) < 1:
            raise ValueError("FilmPolarChartSPELManifoldMuon: start_epoch must be >= 1")
        overrides["--muon-start-epoch"] = int(start_epoch)
    return Lever(
        "film_polar_chart_spel_manifold_muon_finisher",
        overrides=overrides,
        epochs_delta=int(window),
        notes=("round-2 FiLM W=QH0 MCSD/SPEL fallback; registry-resumable Q/H0/"
               "tangent-momentum/Q-EMA; exact nested tangent-dual remains ticketed; "
               "n600 matched finishing-stage verdict owed"),
    )


def WitnessStability(
    preset: str = "amber",
    grad_clip: float | None = None,
    pose_grad_coeff_max: float | None = None,
    grad_normalize: str | None = None,
) -> Lever:
    """#146/#211 AMBER deep-unroll STABILITY: arm the training-collapse cures for the divergence-prone
    per-pair (batch=1) deep-unroll / joint-descent arm. The DIAGNOSED collapse (SETTLED, DAG
    FEED-amber-unblock; do NOT re-diagnose) is an optimizer-divergence DEAD-SATURATION LOCK — d_seg
    drops then the loss SPIKES 57->1070 and FREEZES at EXACTLY 1070.0802 (finite, not NaN): the
    score-domain pose term ``sqrt(10*d_pose+eps)`` has gradient coefficient ``5/sqrt(10*d_pose+eps)``
    -> 5e4 for easy pairs, EXPOSED by batch=1 (no averaging), x w_seg=100 x no-clip.

    The incumbent default already tames it (--grad-clip 1.0 + --pose-eps 1e-2 => coeff 50); this lever
    composes the TIGHTER batch=1 cure via the trainer's REAL flags (never-invent-flags; grep-verified:
    --grad-clip / --pose-grad-coeff-max / --stability-preset / --grad-normalize). ``preset='amber'`` =
    grad-clip 0.5 + pose-grad-coeff-max 25 (eps floor 4e-2) + per-group-grad-clip. Explicit ``grad_clip``
    / ``pose_grad_coeff_max`` override the preset's values (an explicit coeff bound always wins).
    ``grad_normalize='per-param'`` arms the Cells2Pixels per-parameter grad normalization (a DISTINCT
    A/B arm — it ALTERS the seg-vs-pose gradient scale ratio, so it is NOT in amber's default which uses
    the global clip; memo cells2pixels_deepdive_bridge_20260709).

    DEFAULT-OFF at the trainer: un-composed, the trainer runs --stability-preset none + coeff-max 0 =>
    the incumbent pose_eps/grad_clip UNCHANGED (byte-identical; #205 resumes unperturbed). This factory
    is what ARMS the tightening.

    OPTIMIZER-CONFIG lever (window 0): changes the grad-clip budget + effective pose_eps, NOT the
    trained basis, so it may be added on a fresh AMBER run. VERDICT SCOPE: duty-to-measure, NEVER-FIRED
    — the un-collapse A/B that MEASURES whether AMBER un-locks the deep-unroll arm at n600 is a SEPARATE
    operator-GO heavy launch (means != ends; this factory BUILDS the cure + PROVES it caps the
    coefficient on a synthetic blowup; it makes NO score claim; pointer 0.19110 UNMOVED). Mechanism:
    ``tac.witness_stability``; law: ``tac.canonical_equations.witness_pose_grad_coeff_stability_20260709``."""
    from tac.witness_stability import PRESETS, resolve_stability_config

    if str(preset) not in PRESETS:
        raise ValueError(
            f"WitnessStability: unknown preset {preset!r} (known: {sorted(PRESETS)})")
    if grad_normalize is not None and str(grad_normalize) not in ("none", "per-param"):
        raise ValueError(
            f"WitnessStability: grad_normalize must be none|per-param, got {grad_normalize!r}")
    # Resolve against the incumbent defaults so the emitted overrides reflect the actual applied config
    # and the factory fail-closes on an incoherent request (e.g. non-positive coeff bound).
    _incoming_gc = 1.0 if grad_clip is None else float(grad_clip)
    _incoming_cm = 0.0 if pose_grad_coeff_max is None else float(pose_grad_coeff_max)
    if pose_grad_coeff_max is not None and _incoming_cm <= 0.0:
        raise ValueError(
            f"WitnessStability: pose_grad_coeff_max must be > 0 when set, got {pose_grad_coeff_max!r}")
    cfg = resolve_stability_config(
        grad_clip=_incoming_gc, pose_eps=1e-2, pose_grad_coeff_max=_incoming_cm,
        stability_preset=str(preset), per_group_grad_clip=False)
    overrides: dict = {"--stability-preset": str(preset)}
    if grad_clip is not None:
        overrides["--grad-clip"] = float(grad_clip)
    if pose_grad_coeff_max is not None:
        overrides["--pose-grad-coeff-max"] = float(pose_grad_coeff_max)
    # The Cells2Pixels per-param grad normalize is a DISTINCT A/B arm (alters the seg-vs-pose gradient
    # SCALE ratio => owed A/B), NOT part of amber's default (amber uses the global clip). Emit it ONLY
    # when explicitly requested so the flag is DSL-HELD (no config-orphan) while amber stays clip-based.
    if grad_normalize is not None:
        overrides["--grad-normalize"] = str(grad_normalize)
    return Lever(
        "witness_stability_amber",
        overrides=overrides,
        epochs_delta=0,
        notes=(f"#146/#211 AMBER deep-unroll collapse-fix (preset={preset}: grad_clip "
               f"{cfg.grad_clip} + pose coeff<={cfg.pose_grad_coeff_max} => eff pose_eps "
               f"{cfg.effective_pose_eps:g}); un-collapse A/B operator-GO-owed; pointer UNMOVED"),
    )


def LengthSigma(spec: str = "fitted-20260707", window: int = 0) -> Lever:
    """Per-class-PAIR sigma_ij weighting of the Chan-Vese length term — the consumption path for
    the MEASURED Young's-law junction fit (``junction_young_angle_sigma_fit_v1``; fit JSON
    ``experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json``, commit
    3571e5b65). The trainer default ``--length-sigma-matrix all-ones`` is the uniform weight =
    the BYTE-IDENTICAL control (default-off); this lever's ``fitted-20260707`` preset is the
    TREATMENT: sigma[Road-Lane]=0.377 [0.317, 0.441] — the uniform length weight over-penalizes
    lane boundary length ~2.7x vs the frozen scorer's own junction geometry (a named lane-erasure
    mechanism; the sigma_ij IS the Imbert-Monneau flux-limiter DOF, hunt §7).

    WINDOW/STAGE SEMANTICS: the sigma weighting is a reparametrization of the length regularizer
    itself — active whenever the length term is (every epoch at constant ``--length-weight``;
    there is no start-epoch gate), across ALL curriculum stages. ``window=0`` (default) = a
    loss-geometry config change with no epoch budget of its own (same convention as
    :func:`MuonWarmStart`). ``spec`` may be ``"fitted-20260707"`` or a path to a 5x5 JSON (raw
    list or the fit tool's own JSON; NaN unobserved pairs filled with 1.0); it is VALIDATED here
    fail-closed (non-symmetric / wrong shape / non-positive off-diagonal refused) via the
    canonical resolver ``tac.boundary_math.length_sigma``. ``"all-ones"`` is REFUSED: emitting
    the trainer default is a silent no-op lever — the control arm is the lever's ABSENCE.
    A/B (sigma-weighted vs uniform, n600 through R, junction-local d_seg attribution) = the
    registered OWED anchor / duty-to-measure."""
    from tac.boundary_math.length_sigma import PRESET_ALL_ONES, resolve_length_sigma_matrix

    s = str(spec).strip()
    if s == PRESET_ALL_ONES:
        raise ValueError(
            "LengthSigma: 'all-ones' is the trainer DEFAULT (byte-identical control) — emitting "
            "it is a silent no-op lever. The control arm is the lever's absence; use "
            "'fitted-20260707' or a 5x5 JSON path for a treatment.")
    try:
        resolve_length_sigma_matrix(s)  # fail-closed content validation (shape/symmetry/positive)
    except ValueError as exc:
        raise ValueError(f"LengthSigma: {exc}") from exc
    return Lever(
        "FEED_08a_length_sigma",
        overrides={"--length-sigma-matrix": s},
        epochs_delta=window,
        notes=("sigma_ij per-class-pair length weight (junction_young_angle_sigma_fit_v1: "
               "sigma[Road-Lane]=0.377 — uniform over-penalizes lane boundary ~2.7x); "
               "faithful per-interface gather sigma[top1,top2] at {m=0}; A/B owed"),
    )


def PersistenceTopology(weight: float = 1.0, warmup_epochs: int = 0,
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


def MarginFieldHead(weight: float = 1.0, window: int = 100,
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


def HeadOffsetSolver(mode: str = "ot_newton", tau: float = 1.0) -> Lever:
    """#288 (solve-don't-train inventory row 2): SELECTABLE per-class head-bias offset SOLVER — the
    decode-time Laguerre / power-diagram reweight of the argmax cells that attacks minority-class
    (Lane 0.59% / Movable 1.56%) erasure (~57% of flips are Lane↔Road). BYTE-FREE: the solved
    ``b*`` folds into the already-counted ``out_sdf.bias`` (5 floats) → zero extra archive bytes.

    Four REAL mechanisms (NO-FAKE — each does its own work on real inputs):
      * ``mode="menon"`` — the priors-only ``b_k = -tau*log(pi_k)`` heuristic
        (:func:`tac.boundary_math.laguerre_logit_offset.menon_logit_adjustment_offsets`).
      * ``mode="ot_newton"`` — the damped-Newton semi-discrete OT solve (Kitagawa-Merigot-Thibert
        2019) that finds the ``b*`` whose Laguerre-reweighted SOFT cell masses EQUAL the GT class
        frequencies, accounting for THIS witness's boundary geometry the log-freq heuristic ignores
        (:func:`tac.boundary_math.laguerre_logit_offset.damped_newton_ot_offsets`). N-1 MEASURED
        (n600) this HURTS realized d_seg — the AREA objective is wrong (d_seg is Hamming, not
        Wasserstein); kept as the falsified baseline the #386 reformulations must beat.
      * ``mode="flip_weighted"`` (#386) — the SAME OT solve, but targeting the per-class FLIP SHARE
        (``flip_share_by_class``, the boundary-annulus residual mass) instead of GT area frequency.
        UN-ANALYZED (crucible-3 P3 F3): OT still mass-MATCHES, so it may re-inherit N-1's
        cell-inflation — the through-R n600 gate is the arbiter.
      * ``mode="flip_median"`` (#386) — S1's Hamming-optimal per-edge MEDIAN threshold
        (:func:`tac.boundary_math.laguerre_logit_offset.flip_median_offsets`), a DISTINCT closed-form
        path (NOT expressible through the OT target-mass machinery). d_seg is Hamming, whose
        L1-optimal 1-D threshold is the flip-margin median, not a mass-match.

    Trainer leg: ``--head-offset-solver {off,menon,ot_newton,flip_weighted,flip_median}`` (trainer
    default ``off`` = byte-identical) + ``--head-offset-solver-tau``. The trainer computes the WITNESS
    phi field at the EMA verdict (the decode-time site where phi exists — NOT the pre-loop
    margin-field-head setup, which has no phi), solves ``b*`` (area masses for ot_newton; the GT
    argmax + phi for the flip modes), folds it byte-free into a COPY of ``out_sdf.bias`` and emits an
    ADVISORY realized-through-R d_seg delta row. It NEVER mutates the EMA shadow / shipped / resumed
    weights → the live run is untouched. Mechanism + canonical entry:
    ``laguerre_logit_offset.solve_head_offsets``; law:
    ``tac.canonical_equations.laguerre_ot_head_offset_20260709``.

    VERDICT SCOPE (#288 $0 gate, mod32cap ep650): MEASURED-through-R the OT area-mass-matching offset
    made realized d_seg WORSE than both no-offset and the Menon prior at this operating point
    (cell-mass-matching over-inflates the rare Lane cell → over-prediction the SegNet re-read
    penalises; verdict_scope FORMULATION — the OBJECTIVE, not the solver). The #386 flip-weighted /
    flip-median reformulations target the FLIP mass instead; each is a RANKING PROXY re-confirmed
    through R (the n600 gate) before any promotion. means != ends: BUILDS + MEASURES the mechanism;
    makes NO score claim; pointer 0.19110 UNMOVED."""
    if str(mode) not in ("off", "menon", "ot_newton", "flip_weighted", "flip_median"):
        raise ValueError(
            "HeadOffsetSolver: mode must be off/menon/ot_newton/flip_weighted/flip_median, "
            f"got {mode!r}")
    if not (float(tau) > 0.0):
        raise ValueError(f"HeadOffsetSolver: tau must be > 0, got {tau!r}")
    return Lever(
        "head_offset_solver",
        overrides={"--head-offset-solver": str(mode),
                   "--head-offset-solver-tau": float(tau)},
        notes="#288/#386 selectable decode-time Laguerre head-bias offset solver (menon prior / "
              "ot_newton OT area-mass / flip_weighted OT flip-share / flip_median Hamming median); "
              "byte-free out_sdf.bias fold; advisory realized-through-R readout; OT area-match "
              "MEASURED-worse at mod32cap ep650, flip reformulations = the n600 gate arbiter",
    )


def ResumeLRWarmup(beta2: float = 0.999, steps_per_epoch: int = 75, c: float = 2.0,
                   floor: float = 0.1, shape: str = "linear") -> Lever:
    """p0_resume_warmup_geometry_20260717 item 2 (#518): beta2-DERIVED LR-rewarmup window.

    The window length resolves through the canonical law ``adam_v_variance_warmup_length_v1``
    (``warmup_epochs = ceil(c/(1-beta2)/steps_per_epoch)``; RAdam variance-rectification
    rationale, arXiv 1908.03265): with fresh/reset AdamW moments the early steps are
    quasi-isotropic (v~=0), so the LR ramp must SPAN the second-moment memory ``1/(1-beta2)``
    steps — ``c=1`` reproduces the sister bound ``rewarmup_beta2_memory_window_v1`` exactly;
    ``c~=2`` (default) is the conservative multiple. The config-of-record constant 8 (mod32cap/
    c2 launch.sh — UNDER the c=1 bound, the sister law's "cert8_satisfies: false" anchor) is
    preserved as the LawRef FALLBACK on the value-provenance ladder (DERIVED > CONFIG).

    Trainer legs: ``--stage-transition-rewarmup-epochs/-floor/-shape`` consumed by
    ``_stage_rewarmup_factor`` at every registered boundary — the item-1 widened resume
    trigger, the curriculum/tau-octave boundaries, and the item-6a pose-engage boundary.
    ADDITIVE lever: not part of any sealed spec (the live c2 spec keeps its compiled hash);
    composing it OVERRIDES the config constant with the derived value. c2 defaults:
    steps_per_epoch = 75 (600 pairs / accum 8, the MEASURED run value) => 27 epochs.
    means != ends: config-derivation law; pointer moves only through exact eval."""
    if not 0.0 < float(beta2) < 1.0:
        raise ValueError(f"ResumeLRWarmup: beta2 must be in (0,1), got {beta2!r}")
    if int(steps_per_epoch) <= 0:
        raise ValueError(f"ResumeLRWarmup: steps_per_epoch must be > 0, got {steps_per_epoch!r}")
    if not float(c) > 0.0:
        raise ValueError(f"ResumeLRWarmup: c must be > 0, got {c!r}")
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError(f"ResumeLRWarmup: floor must be in [0,1], got {floor!r}")
    if str(shape) not in ("linear", "cosine"):
        raise ValueError(f"ResumeLRWarmup: shape must be linear|cosine, got {shape!r}")
    from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
        CONFIG_OF_RECORD_REWARMUP_EPOCHS,
        adam_v_variance_warmup_epochs,
    )
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG, InputRef, LawRef

    epochs = int(adam_v_variance_warmup_epochs(float(beta2), int(steps_per_epoch), c=float(c)))
    ref = LawRef(
        equation_id="adam_v_variance_warmup_length_v1",
        inputs={
            "beta2": InputRef.literal(
                float(beta2), "AdamW beta2 (trainer default 0.999 / --adam-beta2)"),
            "steps_per_epoch": InputRef.literal(
                int(steps_per_epoch),
                "MEASURED optimizer steps/epoch = ceil(pairs/accum) (c2: 600/8 = 75; the "
                "sister law's run-2 anchor value)"),
            "c": InputRef.literal(
                float(c), "conservative multiple of the 1/(1-beta2) memory (RAdam rationale; "
                          "c=1 == the sister rewarmup_beta2_memory_window_v1 bound)"),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
        fallback=int(CONFIG_OF_RECORD_REWARMUP_EPOCHS),
        fallback_waiver_reason=(
            "config-of-record CONFIG-rung constant 8 (mod32cap/c2 launch.sh "
            "--stage-transition-rewarmup-epochs 8) — value-provenance ladder DERIVED > CONFIG; "
            "the 8 sits UNDER the c=1 memory bound (sister anchor cert8_satisfies=false)"),
    )
    return Lever(
        "resume_lr_rewarmup",
        overrides={"--stage-transition-rewarmup-epochs": epochs,
                   "--stage-transition-rewarmup-floor": float(floor),
                   "--stage-transition-rewarmup-shape": str(shape)},
        lawrefs={"--stage-transition-rewarmup-epochs": ref},
        notes=(f"#518 item 2: beta2-derived LR-rewarmup window ({epochs} ep = "
               f"ceil({c}/(1-{beta2})/{steps_per_epoch})) via adam_v_variance_warmup_length_v1; "
               "config-of-record 8 preserved as the LawRef fallback (DERIVED > CONFIG)"),
    )


def StageTransitionSoftVelocityBlend(
    *,
    enabled: bool = False,
    beta2: float = 0.999,
    c: float = 2.0,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    shape: str = "linear",
    clip_rms: float | None = None,
) -> Lever:
    """FA1/FB1 default-off declaration for the soft first-moment boundary blend.

    The treatment is the named optimizer-state arithmetic
    ``m_new=(1-alpha(t))*m_mapped+alpha(t)*m_fresh`` over a beta2-derived optimizer-step
    window.  There is intentionally no trainer flag here yet: the levelset trainer has no
    soft-blend consumer, and emitting an invented flag would violate the DSL contract.  The
    ON state therefore refuses until a captured replay corpus and trainer consumer land.
    """

    from tac.optimization.stage_transition_soft_velocity_blend import (
        StageTransitionSoftVelocityBlendConfig,
    )

    cfg = StageTransitionSoftVelocityBlendConfig.from_beta2(
        beta2,
        c=c,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        shape=shape,
        clip_rms=clip_rms,
    )
    if enabled:
        raise ValueError(
            "StageTransitionSoftVelocityBlend(enabled=True) has no levelset trainer consumer yet; "
            "capture a stage-boundary optimizer-state replay corpus and land the trainer hook "
            "before enabling"
        )
    clip_note = "off" if cfg.clip_rms is None else str(float(cfg.clip_rms))
    return Lever(
        "stage_transition_soft_velocity_blend_off",
        overrides={},
        notes=(
            "ddm_fb1 FA1 soft velocity blend default OFF; no trainer flags emitted "
            f"(never-invent-flags). Arithmetic: m_new=(1-a(t))*m_mapped+a(t)*m_fresh over "
            f"{cfg.window_steps} optimizer steps derived as ceil({c}/(1-{beta2})); "
            f"alpha={cfg.alpha_start}->{cfg.alpha_end} shape={cfg.shape}; clip_rms={clip_note}. "
            "ON requires captured previous optimizer state plus recorded post-boundary gradients "
            "and a named levelset trainer consumer."
        ),
    )


def PoseEngageWPoseRamp() -> Lever:
    """p0_resume_warmup_geometry_20260717 item 6b (#518): cosine w_pose ramp-in at pose engage.

    The in-loop ``pose_finish_engage`` flips ``_w_pose_now`` 0 -> full w_pose as a STEP — a
    stiff term landing FULL-WEIGHT on AdamW moments trained without it (the H5-F3 level-shift
    class the C11 resume stiff-drift detector catches at RESUME but never sees in-loop, at
    ~ep1000 on the c2 plant). Trainer leg: ``--pose-engage-wpose-ramp`` cosine-ramps w_pose
    over ``--stage-transition-rewarmup-epochs`` from the engage epoch (the same window the
    item-6a boundary registration anchors the LR ramp to). DEFAULT OFF trainer-side => the
    incumbent step is byte-identical; this lever is the tracked ON state (duty-to-measure:
    never-fired until an A/B run consumes it)."""
    return Lever(
        "pose_engage_wpose_ramp",
        overrides={"--pose-engage-wpose-ramp": True},
        notes="#518 item 6b: cosine w_pose 0->full over the rewarmup window from pose-engage "
              "(kills the stiff-step-onto-cold-moments H5-F3 class at the ~ep1000 boundary); "
              "pairs with the item-6a last_boundary_epoch registration (LR ramp at engage)",
    )


def ForkHeadSolve(mode: str = "flip_median", tau: float = 1.0, freeze_epochs: int = 0) -> Lever:
    """p0_resume_warmup_geometry_20260717 item 3 (#518): APPLIED head solve at a warm-start fork.

    Runs the EXISTING #288/#386 rank-4 head machinery
    (:func:`tac.boundary_math.laguerre_logit_offset.solve_head_offsets`) on the RESTORED
    weights BEFORE the first training step and APPLIES the solved zero-sum b* to the LIVE
    ``out_sdf.bias`` AND the EMA shadow — unlike the advisory :func:`HeadOffsetSolver` (which
    folds into a deploy COPY and never mutates the run). The pre-loop v0 verdict immediately
    after the solve is the MEASURED receipt (the item-5 schedule positioning runs first, so phi
    is computed at the checkpoint's true tau/beta). ``freeze_epochs > 0`` zeroes the out_sdf
    gradients for N epochs from the resume start so the solved offsets survive the cold-moment
    transient. Trainer legs: ``--fork-head-solve {off,menon,ot_newton,flip_weighted,
    flip_median}`` + ``--fork-head-solve-tau`` + ``--fork-head-freeze-epochs``; requires
    ``--resume-from`` (fail-closed). VERDICT SCOPE inherited from #288: ot_newton area-matching
    MEASURED-worse at mod32cap ep650 (FORMULATION); flip_median (default here) is S1's
    Hamming-optimal per-edge median — the n600 gate is the arbiter. Default OFF trainer-side =>
    byte-identical; duty-to-measure until fired."""
    if str(mode) not in ("menon", "ot_newton", "flip_weighted", "flip_median"):
        raise ValueError(
            f"ForkHeadSolve: mode must be menon/ot_newton/flip_weighted/flip_median, got {mode!r}")
    if not (float(tau) > 0.0):
        raise ValueError(f"ForkHeadSolve: tau must be > 0, got {tau!r}")
    if int(freeze_epochs) < 0:
        raise ValueError(f"ForkHeadSolve: freeze_epochs must be >= 0, got {freeze_epochs!r}")
    return Lever(
        "fork_head_solve",
        overrides={"--fork-head-solve": str(mode),
                   "--fork-head-solve-tau": float(tau),
                   "--fork-head-freeze-epochs": int(freeze_epochs)},
        notes="#518 item 3: solved Laguerre head offsets APPLIED to live+EMA out_sdf.bias at "
              "the warm-start fork (v0 verdict = measured receipt); optional out_sdf grad "
              "freeze through the transient; byte-free (bias already counted)",
    )


def MarginStepCap(cap: float, window: int = -1) -> Lever:
    """p0_resume_warmup_geometry_20260717 item 4 (#518): per-group ||dW|| trust-region cap.

    Caps the APPLIED per-GROUP update norm at ``cap`` per accepted step, ACTIVE ONLY inside
    the post-boundary ramp window (``window`` epochs; -1 => the LR-rewarmup window). Geometry:
    the SegNet head is EXACT rank-4 linear with flip distance d = |m|/||dW||
    (``segnet_head_rank4_linear_flipdist_v1``) and the margin field IS the Fisher surrogate
    (0.978) — a step is flip-safe when ||dW|| stays below the margin scale. NO cached
    margin-field surface is consulted at runtime (none is cached at the update site); the cap
    is PARAMETERIZED — derive it from the run's measured margin telemetry (e.g. a low quantile
    of |m| over the boundary annulus divided by the head-normal scale) and record the
    derivation with the run. Trainer legs: ``--margin-step-cap`` + ``--margin-step-cap-window``
    (post-update per-group projection w = w_pre + dW*cap/||dW||, before Stiefel re-projection +
    EMA). Default OFF trainer-side => byte-identical; duty-to-measure until fired."""
    if not (float(cap) > 0.0):
        raise ValueError(f"MarginStepCap: cap must be > 0, got {cap!r}")
    return Lever(
        "margin_step_cap",
        overrides={"--margin-step-cap": float(cap),
                   "--margin-step-cap-window": int(window)},
        notes="#518 item 4: per-group ||dW|| trust-region projection during the ramp window "
              "(flip d=|m|/||dW|| rank-4 head law; cold-Adam quasi-isotropic steps bounded)",
    )


def ForkEmaClearance() -> Lever:
    """p0_resume_warmup_geometry_20260717 item 7 (#518): post-fork EMA clearance window.

    At a warm-start fork the EMA shadow restarts as a POINT MASS at the restored weights;
    verdicts inside the first ``ema_warmup_updates(decay)`` shadow updates (~1/(1-decay) ~ 333
    at 0.997) score a transient blend, not a settled shadow. Trainer leg:
    ``--fork-ema-clearance`` suppresses BEST-banking inside the window (verdicts still emitted,
    stamped ``ema_warmup: true``) — extends the existing ``ema_warmup_updates`` machinery
    (telemetry-gating only before this) into a banking gate. Default OFF trainer-side =>
    byte-identical; duty-to-measure until fired. Residual (honest): event detectors that
    consume ``history`` are NOT yet warmup-filtered — rows are stamped, filtering is a named
    follow-up."""
    return Lever(
        "fork_ema_clearance",
        overrides={"--fork-ema-clearance": True},
        notes="#518 item 7: BEST-banking suppressed + verdict rows stamped ema_warmup=true "
              "inside the post-fork shadow warmup window (~1/(1-decay) updates)",
    )


def WarmStartRestoreBoundaryState() -> Lever:
    """p0_resume_warmup_geometry_20260717 item 8 partial (#518): boundary-state at the fork.

    MEASURED at the c2 fork (run 20260717T113932Z run.log): ``"hardness_restored": false`` —
    the warm-start path intentionally SKIPS the persisted ``__hardness_prob`` oversample
    baseline (the #403 restore is gated off under warm-start/lever-drift), so the fork
    re-seeded the oversample distribution. RNG streams restore UNCONDITIONALLY when the source
    carries ``__rng_*`` keys (the c2 ``"np_global_restored": false`` was a KEYLESS SOURCE —
    the fork resumed from an EMA-BEST npz, which persists no run state). Trainer leg:
    ``--warm-start-restore-boundary-state`` opts BACK IN to the hardness baseline under a
    warm-start when the source has it. Default OFF trainer-side => incumbent warm-start
    semantics (byte-identical). Residual (honest): BEST/deploy npz checkpoints still do NOT
    carry the boundary-state keys (the async snapshot threading needed to add them race-free
    is a named follow-up), so a fork from a BEST npz has nothing to restore regardless."""
    return Lever(
        "warm_start_restore_boundary_state",
        overrides={"--warm-start-restore-boundary-state": True},
        notes="#518 item 8 (partial): restore the persisted hardness-oversample baseline under "
              "a warm-start fork when the source sidecar carries it; RNG keys already restore "
              "unconditionally when present",
    )


def FinerBiasInit(k: float = 10.0, window: int = 0) -> Lever:
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


def FreShInitControl(
    n_dir_freqs: int = 4,
    freq_across: float = 32.0,
    freq_along: float = 8.0,
) -> Lever:
    """Matched cold-init control for :func:`FreshFrequencyShift`.

    This factory intentionally changes only the directional/SIREN basis shared
    by both arms.  It does *not* enable FreSh, so the control retains the exact
    cold zero first-layer bias and configured along-tangent frequency.
    """

    if (
        isinstance(n_dir_freqs, bool)
        or not isinstance(n_dir_freqs, Integral)
        or int(n_dir_freqs) <= 0
    ):
        raise ValueError(
            "FreShInitControl: n_dir_freqs must be a positive integer, "
            f"got {n_dir_freqs!r}"
        )
    across = float(freq_across)
    along = float(freq_along)
    if not math.isfinite(across) or across <= 0.0:
        raise ValueError(f"FreShInitControl: freq_across must be finite and > 0, got {freq_across!r}")
    if not math.isfinite(along) or along <= 0.0:
        raise ValueError(f"FreShInitControl: freq_along must be finite and > 0, got {freq_along!r}")
    return Lever(
        "fresh_init_control",
        overrides={
            "--activation": "hosc",
            "--siren-init": True,
            "--self-orient": True,
            "--n-dir-freqs": int(n_dir_freqs),
            "--freq-across": across,
            "--freq-along": along,
            # The live V9 seed is pair-dependent and is built after the init
            # seam.  Both matched arms disable it so the candidate callback is
            # the exact epoch-0 bare-witness through-R surface; validation
            # fails closed if a caller re-enables an unrouted composer.
            "--seed-islands": False,
            "--seed-island-eased": False,
            "--witness-alone-island-loss": False,
            "--fresh-init-control": True,
        },
        notes=("P0 FreSh matched control: identical periodic directional basis, cold frequency/bias; "
               "no spectral selection"),
    )


def FreShFixedQualitySlice(
    eval_every: int = 1,
    ckpt_every: int = 1,
) -> Lever:
    """Matched fixed-quality measurement cadence for the FreSh n8/n64 slice.

    This lever owns only observability and checkpoint cadence.  It is composed
    identically onto the control and treatment; it never changes the model,
    loss, optimizer, or FreSh candidate law.  Production arms leave it OFF.
    """

    if (
        isinstance(eval_every, bool)
        or not isinstance(eval_every, Integral)
        or int(eval_every) != 1
    ):
        raise ValueError(
            "FreShFixedQualitySlice: eval_every must be exactly 1 so the first "
            f"fixed-quality crossing is observed, got {eval_every!r}"
        )
    if (
        isinstance(ckpt_every, bool)
        or not isinstance(ckpt_every, Integral)
        or int(ckpt_every) != 1
    ):
        raise ValueError(
            "FreShFixedQualitySlice: ckpt_every must be exactly 1 so every "
            f"measured epoch is resumable and preserved, got {ckpt_every!r}"
        )
    return Lever(
        "fresh_fixed_quality_slice",
        overrides={
            "--eval-every": int(eval_every),
            "--ckpt-every": int(ckpt_every),
            "--stage-checkpoints": True,
        },
        notes=(
            "FreSh fixed-quality slice protocol: per-epoch realized d_seg plus preserved "
            "per-epoch resume checkpoints; compose identically on control/treatment"
        ),
    )


def FreshFrequencyShift(
    spectrum_size: int = 64,
    sample_pairs: int = 10,
    reference_freq_along: float = 8.0,
    tangent_deficit: float = 3.2,
    bias_k_min: float = 0.0,
    bias_k_max: float = 3.0,
    bias_k_step: float = 0.1,
    n_dir_freqs: int = 4,
    freq_across: float = 32.0,
    freq_along: float = 8.0,
) -> Lever:
    """FreSh init-only spectral alignment over tangent frequency and bias width.

    The sweep is backprop-free and default-OFF.  It scores the exact seeded
    initialized witness through R and the frozen MLX SegNet, selects by FreSh's
    boundary-spectrum Wasserstein distance, then hands that state to the
    otherwise unchanged training loop.  Exact score authority remains
    ``upstream/evaluate.py`` on archive bytes.
    """

    if (
        isinstance(spectrum_size, bool)
        or not isinstance(spectrum_size, Integral)
        or int(spectrum_size) <= 0
    ):
        raise ValueError(
            "FreshFrequencyShift: spectrum_size must be a positive integer, "
            f"got {spectrum_size!r}"
        )
    if (
        isinstance(sample_pairs, bool)
        or not isinstance(sample_pairs, Integral)
        or int(sample_pairs) <= 0
    ):
        raise ValueError(
            "FreshFrequencyShift: sample_pairs must be a positive integer, "
            f"got {sample_pairs!r}"
        )
    reference = float(reference_freq_along)
    deficit = float(tangent_deficit)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError(
            "FreshFrequencyShift: reference_freq_along must be finite and > 0, "
            f"got {reference_freq_along!r}")
    if not math.isfinite(deficit) or deficit <= 1.0:
        raise ValueError(
            f"FreshFrequencyShift: tangent_deficit must be finite and > 1, got {tangent_deficit!r}")
    # Reuse the runtime's decimal-stable validation so the factory and trainer
    # cannot disagree about inclusivity or divisibility of the bias grid.
    from tac.witness_init.fresh_frequency_shift import (
        inclusive_bias_width_grid,
        tangent_frequency_candidates,
    )
    from tac.witness_init.fresh_trainer_contract import MAX_FRESH_CANDIDATES

    bias_grid = inclusive_bias_width_grid(bias_k_min, bias_k_max, bias_k_step)
    if bias_grid[0] != 0.0:
        raise ValueError("FreshFrequencyShift: bias_k_min must be 0 for the exact cold control")
    control = FreShInitControl(
        n_dir_freqs=n_dir_freqs,
        freq_across=freq_across,
        freq_along=freq_along,
    )
    treatment_basis = dict(control.overrides)
    treatment_basis.pop("--fresh-init-control")
    frequencies = tangent_frequency_candidates(
        treatment_basis["--freq-along"],
        reference_frequency=reference,
        tangent_deficit=deficit,
    )
    candidate_count = len(frequencies) * len(bias_grid)
    if candidate_count > MAX_FRESH_CANDIDATES:
        raise ValueError(
            "FreshFrequencyShift: candidate grid has "
            f"{candidate_count} candidates; maximum is {MAX_FRESH_CANDIDATES}"
        )
    return Lever(
        "fresh_frequency_shift_init",
        overrides={
            **treatment_basis,
            "--fresh-init": True,
            "--fresh-spectrum-size": int(spectrum_size),
            "--fresh-sample-pairs": int(sample_pairs),
            "--fresh-reference-freq-along": reference,
            "--fresh-tangent-deficit": deficit,
            "--fresh-bias-k-min": float(bias_k_min),
            "--fresh-bias-k-max": float(bias_k_max),
            "--fresh-bias-k-step": float(bias_k_step),
        },
        notes=("P0 FreSh init-only boundary-spectrum Wasserstein selection over the measured "
               "along-tangent residual and first-layer bias; fixed-quality slice A/B required; "
               "n600 governed validation owed"),
    )


def LogitAdjust(tau: float = 1.0, window: int = 100) -> Lever:
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
    the loss adapter is the SAME object = byte-identical); ROUTED into the ``--micro-batch-pairs>1``
    batched twin (#D15: the offset is added to the BASE seg-form logits only and the live
    birth-completion cell is re-read on every call; row-local math is functionally equivalent but the
    upstream batched scorer need not be bit-exact). SISTER (do not confuse): the
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
               "surface only, deployed argmax unchanged); static/live offsets routed into the "
               "micro-batch twin under functional parity; #218 A/B owed"),
    )


def WeightEntropyPenaltyMLX(lam: float = 15.0, window: int = 0) -> Lever:
    """Ballé rate-in-the-loss WEIGHT-ENTROPY penalty — the council-draft-20260707 §22(2) fold:
    the torch vehicle's ``--weight-entropy-penalty-lambda`` PORTED to the levelset trainer as a
    DETERMINISTIC soft-histogram symbol-entropy surrogate over the COUNTED witness weights
    (``tac.boundary_math.weight_entropy_penalty_mlx``; free bank ``B``/``*_B`` excluded, rule
    118; the exact ``quantize_levelset_blob`` int8 grid). Adds ``λ·rate_term`` (contest rate
    scale, 25·bits/8/N) ONCE per opt step (per-MODEL, the code_nuc pattern; routed into BOTH the
    serial ``total_loss_fn`` and the ``--micro-batch-pairs`` batched twin's ``_once_terms``).
    State-free + no RNG => resume-safe; ``λ=0`` (trainer default) is a true no-op branch.

    BORROWED-NUMBER FIREWALL (NO-FAKE #8): the −19.6% live-decoder byte cut was MEASURED on the
    TORCH vehicle's LEARNED-prior term (2026-06-20; EMA-lag caveat; ema0.9 translation proof) —
    it does NOT transfer to this lever. This MLX lever is NEVER-FIRED (activation ledger) until
    its own byte-closed n600 A/B lands (λ-on vs λ=0 at equal d_seg/d_pose, real
    ``quantize_levelset_blob`` bytes). The torch λ* is open in {5,15,30} (λ50 overshoots into
    d_seg harm) — default 15.0 = the open-range center, a STARTING arm value, not a tuned
    optimum. NO-FAKE headline metric for the A/B:
    ``weight_entropy_penalty_mlx.measured_symbol_entropy_bits_numpy`` (hard codec-grid entropy),
    plus the real archive bytes. Equations leg: ``weight_entropy_rate_in_loss_lever_v1``."""
    _l = float(lam)
    if not (_l > 0.0) or _l != _l or _l in (float("inf"), float("-inf")):
        raise ValueError(
            f"WeightEntropyPenaltyMLX: lam must be a positive finite float (trainer default 0.0 "
            f"= OFF; construct the lever only to turn the penalty ON), got {lam!r}")
    return Lever(
        "WeightEntropyPenaltyMLX",
        overrides={"--weight-entropy-penalty-lambda": _l},
        epochs_delta=window,
        notes=("Ballé rate-in-the-loss weight-entropy penalty (MLX port of the torch-vehicle "
               "lever; deterministic soft-histogram surrogate, counted weights only); torch "
               "-19.6% does NOT transfer — n600 A/B owed (duty-to-measure)"),
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


def sealed_205_program(
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
