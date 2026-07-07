"""Auto-generated DSL completeness surface — the DSL self-reports which trainer flags
it holds, so it (not the operator's memory) is the single source of truth.

Operator 2026-07-06: *"we need a system ... single source of truth ... inheritable and
observable and transparent ... everything is missing from the DSL and you've been doing
it all by hand."* This module makes the DSL account for **every** trainer flag WITHOUT
any hand-typed registry: it reads the flags straight from the trainer argparse AND the
lever flags straight from the DSL's own factory bodies (static AST — no execution, no
required-arg problem, deterministic), then reports the mapping and the gaps.

Scope discipline (post adversarial review 2026-07-06): this surface reports COVERAGE only —
what the DSL references vs what the trainer exposes — because that is derivable exactly.
It deliberately does NOT report whether a lever is "on" in a given program: flag-presence
cannot decide it (``PoseDecouple`` is ON when ``--w-pose==0``; ``WarpRealLumaFrame0`` is ON
when ``--w-pose>0`` — the SAME flag, inverse meaning; carrier levers are code-wired, not
flag-gated). That belongs in #332 when levers carry explicit on-predicates in the DSL.

Functions, all derived, none remembered:
  1. ``lever_factories()`` — {factory_name: flags it emits}, discovered by AST (a factory is
     a module-level ``def`` returning a ``Lever`` or a tuple of ``Lever`` s, incl. composites
     that delegate to other factories). No list to maintain.
  2. ``dsl_emitted_flags()`` — every ``--flag`` the DSL actually EMITS to the trainer (factory
     overrides + BASELINE ``flag_dict`` + program constructors). Used for the drift check.
  3. ``dsl_referenced_flags()`` — every ``--flag`` the DSL mentions ANYWHERE (its coverage).
  4. ``completeness(trainer_path)`` — trainer ∩/∖ DSL → the UNMAPPED gaps (trainer flags the
     DSL does not hold) + STALE (DSL-EMITTED flags absent from the trainer = real dead/typo).
  5. ``name_composable_levers()`` / ``resolve_composable_lever(name)`` — the ``--dsl-lever``
     composability predicate + typed resolution (CLASS-fix, review 2026-07-06: Muon requires
     args → TypeError; DM1Minimal returns a tuple → AttributeError; both previously crashed the
     config generator with a raw traceback). NOTE: unlike the coverage scan, the predicate
     deliberately DOES execute nilary factories (cheap pure dataclass construction) — calling
     and isinstance-checking is the honest, deterministic composability decision.

Deterministic (sorted); no RNG, no clock. Consumed by ``test_lever_registry`` (kept honest).
Memory: [[triality_ran_on_one_and_a_half_legs_dsl_equations_never_proactive_20260706]] +
[[config_orphan_confound_permanent_fix_lever_registry_20260706]] (re-homed here, into the DSL).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from tac.witness_dsl import curriculum_dsl as _cd
from tac.witness_dsl.curriculum_dsl import BASELINE, real_trainer_flags

_FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
# The ``--no-`` regex artifact: curriculum_dsl.validate() builds ``--no-<X>`` from a
# store_true flag to describe an INVALID compile; it is never a real flag.
_NO_ARTIFACT = re.compile(r"^--no-")
# Both sync and async ``def`` count as factories/constructors (r5 review: an ``async def``
# factory returning a ``Lever`` was previously invisible to the AST scan).
_FUNC_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_source() -> str:
    return Path(_cd.__file__).read_text()


def _flags_in_node(node: ast.AST) -> frozenset[str]:
    """Every real ``--flag`` string constant lexically inside an AST node (drops ``--no-*``)."""
    return frozenset(
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and _FLAG_RE.match(n.value) and not _NO_ARTIFACT.match(n.value))


def _constructs(node: ast.FunctionDef, name: str) -> bool:
    """True if the function body calls ``name(...)`` (e.g. ``Lever`` / ``WitnessProgram``)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            return True
    return False


def _anno_is_lever(a: ast.AST) -> bool:
    """STRUCTURAL check that an annotation IS ``Lever`` or a tuple/collection OF ``Lever`` —
    NOT a mere substring match (which would falsely accept ``dict[str, Lever]``, a non-factory
    returning a dict; round-2 LOW-1 fix)."""
    if isinstance(a, ast.Name):
        return a.id == "Lever"
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        s = a.value.strip()
        if s == "Lever":
            return True
        # A fully-stringized composite, e.g. ``-> "tuple[Lever, Lever]"`` (r5 review): parse the
        # string as an expression and recurse. Bad/partial strings fail closed (False, no raise).
        try:
            return _anno_is_lever(ast.parse(s, mode="eval").body)
        except (SyntaxError, ValueError):
            return False
    if isinstance(a, ast.Subscript):
        base = a.value.id if isinstance(a.value, ast.Name) else None
        if base not in ("tuple", "Tuple", "list", "List"):      # dict[..], Optional[..], etc. are NOT
            return False
        sl = a.slice
        elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        # tuple[Lever, ...] / tuple[Lever, Lever] — every non-Ellipsis element must be Lever
        real = [e for e in elts if not (isinstance(e, ast.Constant) and e.value is Ellipsis)]
        return bool(real) and all(_anno_is_lever(e) for e in real)
    return False


def _returns_lever(node: ast.FunctionDef) -> bool:
    """True if the return annotation IS ``Lever`` or a tuple/list OF ``Lever`` (catches
    ``-> Lever`` AND the composite ``-> tuple[Lever, Lever]``)."""
    return node.returns is not None and _anno_is_lever(node.returns)


def _called_names(node: ast.FunctionDef) -> set[str]:
    return {n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def lever_factories() -> dict[str, frozenset[str]]:
    """{factory_name: flags it emits} — discovered by AST, deterministic, zero maintenance.

    A LEVER FACTORY is a module-level ``def`` that returns a ``Lever`` (direct construction OR
    return annotation mentioning ``Lever``, e.g. the ``tuple[Lever, Lever]`` composite
    ``DM1Minimal``) and is NOT a ``WitnessProgram`` constructor. A composite's flags are its
    own PLUS the flags of any lever factory it delegates to (one level), so a delegating
    factory is never reported as flag-less."""
    tree = ast.parse(_module_source())
    fdefs = [n for n in tree.body if isinstance(n, _FUNC_DEFS)]   # sync + async (r5)
    direct: dict[str, frozenset[str]] = {}   # factory -> its OWN flags
    calls: dict[str, set[str]] = {}          # factory -> factory names it calls
    for node in fdefs:
        if _constructs(node, "WitnessProgram"):
            continue
        if _constructs(node, "Lever") or _returns_lever(node):
            direct[node.name] = _flags_in_node(node)
            calls[node.name] = _called_names(node)
    # TRANSITIVE closure over delegation (a composite calls sub-factories, which may call
    # more) so a multi-level composite never silently drops a delegated flag (round-2 LOW-2).
    out: dict[str, frozenset[str]] = {}
    for name in direct:
        seen: set[str] = set()
        stack = [name]
        flags: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            flags |= direct.get(cur, frozenset())
            for callee in calls.get(cur, ()):
                if callee in direct and callee not in seen:
                    stack.append(callee)
        out[name] = frozenset(flags)
    return dict(sorted(out.items()))


def program_constructors() -> dict[str, frozenset[str]]:
    """{name: flags it references} for the module-level WitnessProgram constructors."""
    tree = ast.parse(_module_source())
    return dict(sorted(
        (n.name, _flags_in_node(n)) for n in tree.body
        if isinstance(n, _FUNC_DEFS) and _constructs(n, "WitnessProgram")))


def dsl_referenced_flags() -> frozenset[str]:
    """Every real ``--flag`` the DSL references ANYWHERE (module-wide) — its coverage set."""
    return _flags_in_node(ast.parse(_module_source()))


def dsl_emitted_flags() -> frozenset[str]:
    """Every ``--flag`` the DSL actually EMITS to the trainer: factory overrides + BASELINE
    ``flag_dict`` + program-constructor flags. This EXCLUDES docstring mentions, ``Contain``/
    launcher/daemon fields, and the ``--no-`` artifact — so ``stale`` computed against it is
    real drift (a DSL-emitted flag the trainer does not accept), not reporter noise (M3 fix)."""
    emitted: set[str] = set()
    for fl in lever_factories().values():
        emitted |= fl
    for fl in program_constructors().values():
        emitted |= fl
    emitted |= {f for f in BASELINE.flag_dict() if _FLAG_RE.match(f) and not _NO_ARTIFACT.match(f)}
    return frozenset(emitted)


@dataclass
class Completeness:
    trainer_total: int
    dsl_referenced: int
    mapped: list[str] = field(default_factory=list)      # trainer flags the DSL references
    unmapped: list[str] = field(default_factory=list)    # trainer flags NO DSL construct references (GAPS)
    stale: list[str] = field(default_factory=list)       # DSL-EMITTED flags absent from the trainer (drift)

    @property
    def coverage_frac(self) -> float:
        return len(self.mapped) / max(self.trainer_total, 1)


def completeness(trainer_path: str | Path | None = None) -> Completeness:
    """Reconcile the trainer's flags against the DSL. ``unmapped`` = the gap the operator wants
    visible (trainer flags the DSL does not hold); ``stale`` = DSL-EMITTED flags the trainer
    rejects (real drift). Deterministic, sorted."""
    # A bad explicit trainer_path is a caller error — fail LOUD with a clear typed message
    # (r5 review) rather than a raw OSError from deep inside real_trainer_flags. The default
    # (trainer_path=None) path is the normal case and never hits this.
    tp = Path(trainer_path) if trainer_path else None
    if tp is not None and not tp.is_file():
        raise FileNotFoundError(f"completeness(): trainer_path is not a file: {tp}")
    # Drop the ``--no-*`` BooleanOptionalAction negations on BOTH sides symmetrically (round-2
    # LOW-3), so if the trainer ever exposes one it cannot show as a phantom unmapped/stale.
    trainer = {f for f in real_trainer_flags(tp) if not _NO_ARTIFACT.match(f)}
    referenced = set(dsl_referenced_flags())
    emitted = set(dsl_emitted_flags())
    return Completeness(
        trainer_total=len(trainer),
        dsl_referenced=len(referenced),
        mapped=sorted(trainer & referenced),
        unmapped=sorted(trainer - referenced),
        stale=sorted(emitted - trainer),
    )


class LeverCompositionError(ValueError):
    """A ``--dsl-lever NAME`` that cannot be composed by bare name.

    Raised (with a clear operator-actionable message, never a raw traceback) when a
    requested lever name is unknown, requires explicit args (e.g. ``Muon(start_epoch)``),
    returns a composite (e.g. ``DM1Minimal`` → ``tuple[Lever, Lever]``), or fails to
    construct. Subclasses ``ValueError`` so pre-existing ``except ValueError`` callers
    keep working. CLASS-fix for the Muon/DM1Minimal ``--dsl-lever`` crash family
    (review 2026-07-06)."""


def _composability_check(name: str) -> tuple[object | None, str | None]:
    """Internal honest check: ``(lever, None)`` when ``name`` is composable by bare name,
    else ``(None, why_not)``. A name is composable iff its factory is callable with ZERO
    required args (every parameter has a default — checked via ``inspect.signature``, so
    ``Muon(start_epoch)`` is refused up front) AND actually returns a single ``Lever``
    (the nilary factory IS called — cheap pure dataclass construction — and the result
    isinstance-checked; deterministic and honest, so the ``tuple[Lever, Lever]`` composite
    ``DM1Minimal`` is refused too). Resolves via ``getattr`` so runtime-registered
    factories (tests) behave identically to source-defined ones."""
    import inspect

    from tac.witness_dsl.curriculum_dsl import Lever

    fac = getattr(_cd, name, None)
    if fac is None or not callable(fac):
        # keep the historical phrase so pre-existing matchers still recognize the class
        return None, "is not a curriculum_dsl Lever factory"
    try:
        sig = inspect.signature(fac)
    except (TypeError, ValueError):
        return None, "has an uninspectable signature — not composable via --dsl-lever"
    required = [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        return None, f"requires explicit args {required} — not composable via --dsl-lever"
    try:
        lever = fac()
    except Exception as exc:  # residual construction failure → the same clear typed error
        return None, f"failed to construct ({type(exc).__name__}: {exc})"
    if not isinstance(lever, Lever):
        return None, (
            f"returns {type(lever).__name__} (a composite / non-Lever) — not composable "
            "via --dsl-lever (compose its parts individually, e.g. StiefelW + "
            "CodeSpectralEntropy instead of DM1Minimal)")
    return lever, None


def name_composable_levers() -> tuple[str, ...]:
    """Sorted names of the lever factories composable by bare name (``--dsl-lever NAME``):
    the subset of :func:`lever_factories` that are BOTH zero-required-arg AND return a
    single ``Lever`` — decided by the SAME honest check as :func:`resolve_composable_lever`
    (nilary factories are actually called and isinstance-checked; a factory whose emitted
    value later fails the trainer argparse is caught by the parse-test in
    ``test_lever_registry``). This is the SoT the launcher's ``--dsl-lever`` help text and
    the CI parse-test enumerate — never a hand-typed list."""
    return tuple(sorted(
        fname for fname in lever_factories() if _composability_check(fname)[1] is None))


def resolve_composable_lever(name: str):
    """Resolve ``name`` → a single ``Lever`` instance, or raise :class:`LeverCompositionError`
    whose message names the reason AND the full composable set (operator-actionable, never a
    raw traceback). THE single runtime resolution path for ``--dsl-lever NAME`` — consumed by
    ``tac.witness_autoconfig._merge_dsl_levers`` and eagerly by the launcher, so a
    non-composable name is refused BEFORE any gate/spawn work."""
    lever, why = _composability_check(name)
    if why is not None:
        raise LeverCompositionError(
            f"--dsl-lever {name!r} {why}; composable (zero-required-arg, single-Lever) "
            f"factories: {', '.join(name_composable_levers())}")
    return lever


def emit_stub_lever(flag: str) -> str:
    """Candidate DSL lever factory source for an unmapped flag — so completing the DSL is a
    review-and-accept, never hand-typed from scratch. The human fills the relevance note + the
    correct override VALUE (store_true vs value is left generic: overrides {flag: True})."""
    if not _FLAG_RE.match(flag) or _NO_ARTIFACT.match(flag):
        raise ValueError(f"emit_stub_lever expects a real trainer flag, got {flag!r}")
    name = "".join(p.capitalize() for p in flag.lstrip("-").split("-"))
    return (
        f"def {name}(window: int = 100) -> Lever:  # noqa: N802 — AUTO-STUB, review before use\n"
        f'    """AUTO-STUB for {flag}. Fill the relevance/notes + the correct override VALUE\n'
        f'    (True for store_true, else a value). Carries a warm-start window (else dead-arm\n'
        f'    on resume, DSL review C1)."""\n'
        f'    return Lever("{flag.lstrip("-").replace("-", "_")}",\n'
        f'                 overrides={{"{flag}": True}}, epochs_delta=window,\n'
        f'                 notes="AUTO-STUB — classify + set correct override value")\n'
    )


__all__ = [
    "lever_factories", "program_constructors", "dsl_referenced_flags", "dsl_emitted_flags",
    "Completeness", "completeness", "emit_stub_lever", "BASELINE",
    "LeverCompositionError", "name_composable_levers", "resolve_composable_lever",
]
