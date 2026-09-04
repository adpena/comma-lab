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
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from tac.witness_dsl import curriculum_dsl as _cd
from tac.witness_dsl.curriculum_dsl import BASELINE, real_trainer_flags

_FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
# The ``--no-`` regex artifact: curriculum_dsl.validate() builds ``--no-<X>`` from a
# store_true flag to describe an INVALID compile; it is never a real flag.
_NO_ARTIFACT = re.compile(r"^--no-")
# Both sync and async ``def`` count as factories/constructors (r5 review: an ``async def``
# factory returning a ``Lever`` was previously invisible to the AST scan).
_FUNC_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


class MetricResolutionError(ValueError):
    """Raised when a non-trainer canonical metric cannot resolve exactly once."""


@dataclass(frozen=True, slots=True)
class CanonicalMetricDescriptor:
    """One canonical non-trainer metric known to the witness DSL."""

    metric_id: str
    binding_artifact: str


_CANONICAL_METRICS = (
    CanonicalMetricDescriptor(
        metric_id="argmax_native_vjp_fidelity_v1",
        binding_artifact=".omx/research/bregman_v9_all_surfaces_binding_20260714.json",
    ),
)


def canonical_metric_ids() -> tuple[str, ...]:
    """Return the deterministic canonical metric IDs held by this registry."""

    return tuple(sorted(metric.metric_id for metric in _CANONICAL_METRICS))


def resolve_canonical_metric(metric_id: str) -> CanonicalMetricDescriptor:
    """Resolve exactly one canonical metric, failing on unknown or duplicate IDs."""

    if not isinstance(metric_id, str) or not metric_id.strip():
        raise MetricResolutionError("metric_id must be a non-empty string")
    matches = [metric for metric in _CANONICAL_METRICS if metric.metric_id == metric_id]
    if len(matches) != 1:
        reason = "unknown" if not matches else "duplicated"
        raise MetricResolutionError(
            f"canonical metric {metric_id!r} is {reason}; registered ids: "
            f"{', '.join(canonical_metric_ids())}"
        )
    descriptor = matches[0]
    if not Path(descriptor.binding_artifact).is_file():
        raise MetricResolutionError(
            f"canonical metric {metric_id!r} is orphaned: missing binding artifact "
            f"{descriptor.binding_artifact!r}"
        )
    return descriptor


def _module_source(path: Path | None = None) -> str:
    return Path(path if path is not None else _cd.__file__).read_text()


# ── PACKAGE-WIDE SCAN (the vacuous-gate repair; ddm_sb2 task #819, 2026-07-31) ────────
# MEASURED BUG (`.omx/research/ddm_cn3_week_coherence_audit_20260731.md` L135-143): every
# function above ASTs ONE file — ``curriculum_dsl.py`` — so ~170 sibling ``witness_dsl/*.py``
# modules were invisible. ``completeness().stale == []`` was therefore VACUOUSLY clean, and the
# grade-(1) DESIGNED-STUB levers (e.g. the five fh1 adapted forces) could never surface: the
# registry could not see the file they live in. A gate returning a clean marker while scanning
# 0.6% of its domain is NO-FAKE forbidden class #1 at the gate layer.
#
# The repair is NOT to widen ``lever_factories()`` in place. Different lever modules target
# DIFFERENT trainers (``spec_tr1_renderer_20260728`` declares its own ``TRAINER_RELPATH``), so a
# single widened flag-set compared against the levelset trainer would report every tr1 flag as
# "stale" — swapping a vacuous PASS for a false FAIL. Instead the package-wide surface below
# resolves EACH module's own trainer and reports per-factory build grades. The historical
# ``lever_factories`` / ``completeness`` contract (levelset-trainer coverage) is preserved
# verbatim so its existing consumers are unaffected.
_PKG_DIR = Path(__file__).resolve().parent
_TRAINER_RELPATH_RE = re.compile(r'^TRAINER_RELPATH\s*=\s*"([^"]+)"', re.MULTILINE)
# Plural form (ddm_lr2, 2026-08-03): a module that legitimately binds to MORE THAN ONE trainer
# — ``curriculum_dsl`` holds flags from the levelset entry point AND the base it imports its
# primitives from (MEASURED: 35 flags live only on the base) — could not declare that fact, so
# it was forced to rely on the silent default. Forcing it into the singular form would have
# dropped those 35 flags and manufactured false "missing flag" grades: a false FAIL replacing a
# silent PASS, the exact trade this registry's own repair note refuses. ``TRAINER_RELPATH``
# stays the one-trainer spelling; neither regex can match the other (``TRAINER_RELPATHS`` has
# no ``=`` where the singular pattern demands one).
_TRAINER_RELPATHS_RE = re.compile(r"^TRAINER_RELPATHS\s*=\s*\(([^)]*)\)", re.MULTILINE)
_QUOTED_RE = re.compile(r'"([^"]+)"')
_REPO_ROOT = _PKG_DIR.parents[2]
# A factory whose docstring/body announces itself as a stub. Detected but never TRUSTED: the
# authoritative grade is whether the emitted flags exist on the module's trainer.
_STUB_MARKER_RE = re.compile(r"DESIGNED-STUB|AUTO-STUB")


def package_lever_modules(pkg_dir: Path | None = None) -> tuple[Path, ...]:
    """Every ``witness_dsl`` module that could define a ``Lever`` factory (sorted, deterministic).

    ``pkg_dir`` overrides the scanned directory. It exists so the sister refusal gate can be
    exercised against a SYNTHETIC fixture (ddm_lr2, 2026-08-03): a gate whose scan surface is
    hard-wired to the installed package can only ever be observed returning zero on a clean
    tree, which is indistinguishable from a gate that cannot fire at all. Default is unchanged.
    """
    base = Path(pkg_dir) if pkg_dir is not None else _PKG_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(p for p in base.glob("*.py") if not p.name.startswith("_")))


def module_declares_trainer(path: Path) -> bool:
    """Whether a lever module states its own trainer binding, instead of inheriting the default.

    The distinction this exposes is the whole point (ddm_lr2, 2026-08-03). An UNDECLARED module
    is not "bound to the levelset trainer" — it is bound to *whatever the default happens to be*,
    and the reader cannot tell an intentional levelset binding from an author who never
    considered the question. MEASURED consequence: three modules written FOR the TR1 vehicle
    (``fh1_adapted_force_levers``, ``ph3_s10_frontloaded_levers``, ``ax1_derived_levers`` — each
    naming TR1 in its own docstring) were graded against the RETIRED trainer, so no TR1-scoped
    query could surface their 8 factories at all. A silent default is an orphan generator, per
    CLAUDE.md "'Off' is a tracked queue, never a forgotten default"; this predicate is what makes
    the defaulted state TRACKED rather than invisible.
    """
    src = _module_source(path)
    return bool(_TRAINER_RELPATH_RE.search(src) or _TRAINER_RELPATHS_RE.search(src))


def module_trainer_paths(path: Path) -> tuple[Path, ...]:
    """The trainer(s) a lever module's flags must exist on.

    A module that declares its own ``TRAINER_RELPATH`` (the ``spec_tr1_renderer`` pattern) binds
    to that trainer; ``TRAINER_RELPATHS = (...)`` declares a multi-trainer binding explicitly;
    every other module falls back to the canonical levelset entry point plus the base it imports
    its primitives from. Resolving this per-module is what keeps the widened scan HONEST rather
    than merely louder. Use :func:`module_declares_trainer` to tell a DECLARED binding from a
    defaulted one — the two are indistinguishable in this function's return value, and that
    indistinguishability is the bug class it once hid.
    """
    src = _module_source(path)
    m = _TRAINER_RELPATH_RE.search(src)
    if m:
        return (_REPO_ROOT / m.group(1),)
    plural = _TRAINER_RELPATHS_RE.search(src)
    if plural:
        rels = _QUOTED_RE.findall(plural.group(1))
        return tuple(_REPO_ROOT / r for r in rels)
    from tac.witness_dsl.curriculum_dsl import TRAINER_PATH
    base = TRAINER_PATH.parent / "train_witness_realized_through_R_mlx.py"
    return tuple(p for p in (TRAINER_PATH, base) if p.is_file())


def _flags_of(paths: tuple[Path, ...]) -> frozenset[str]:
    out: set[str] = set()
    for p in paths:
        if p.is_file():
            out |= set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', p.read_text()))
    return frozenset(out)


@dataclass(frozen=True, slots=True)
class FactoryBuild:
    """One lever factory's BUILD grade — the dimension coverage/activation could not express."""

    module: str
    factory: str
    flags: tuple[str, ...]
    missing_flags: tuple[str, ...]   # emitted flags absent from THIS module's trainer argparse
    trainer: str
    stub_marker: bool                # the factory SAYS it is a stub
    # Did the module STATE that trainer, or merely inherit the default? (ddm_lr2, 2026-08-03.)
    # Without this field a defaulted binding and a declared one are byte-identical to every
    # reader, which is how 8 TR1-targeted factories were graded against the retired trainer for
    # a week. Default True so any pre-existing positional construction keeps its meaning.
    trainer_declared: bool = True

    @property
    def is_stub(self) -> bool:
        """A DESIGNED-STUB: it emits at least one flag its trainer does not declare.

        Structural, not label-based — a factory that forgot to say "DESIGNED-STUB" is still a
        stub, and a factory that says so while its flags exist is a stale label, not a stub.
        """
        return bool(self.missing_flags)

    @property
    def label_drift(self) -> bool:
        """The factory's self-declared stub status disagrees with its measured build state."""
        return self.stub_marker != self.is_stub

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module, "factory": self.factory, "flags": list(self.flags),
            "missing_flags": list(self.missing_flags), "trainer": self.trainer,
            "stub_marker": self.stub_marker, "is_stub": self.is_stub,
            "label_drift": self.label_drift, "trainer_declared": self.trainer_declared,
            "trainer_binding_is_verdict_relevant": self.trainer_binding_is_verdict_relevant,
        }

    @property
    def trainer_binding_is_verdict_relevant(self) -> bool:
        """The factory's build GRADE depends on which trainer it was resolved against.

        True exactly when the factory is graded a stub while its module never stated its
        trainer: the ``is_stub`` verdict is then an artifact of a default nobody chose. This is
        the narrow, live-checkable scope the sister gate refuses on — narrow because a factory
        whose flags all exist needs no binding argument to be graded, and MEASURED to have no
        blind spot on this tree (every undeclared module's flags sit on the retired trainer,
        consistent with its default; see ddm_lr2 §1).
        """
        return self.is_stub and not self.trainer_declared


def _factories_in_source(src: str) -> dict[str, tuple[frozenset[str], bool]]:
    """{factory: (flags, stub_marker)} for one module's source (same predicate as ``lever_factories``)."""
    tree = ast.parse(src)
    fdefs = [n for n in tree.body if isinstance(n, _FUNC_DEFS)]
    direct: dict[str, frozenset[str]] = {}
    calls: dict[str, set[str]] = {}
    marks: dict[str, bool] = {}
    for node in fdefs:
        if _constructs(node, "WitnessProgram"):
            continue
        if _constructs(node, "Lever") or _returns_lever(node):
            direct[node.name] = _flags_in_node(node)
            calls[node.name] = _called_names(node)
            seg = ast.get_source_segment(src, node) or ""
            marks[node.name] = bool(_STUB_MARKER_RE.search(seg))
    out: dict[str, tuple[frozenset[str], bool]] = {}
    for name in direct:  # transitive delegation closure, as in lever_factories()
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
        out[name] = (frozenset(flags), marks[name])
    return out


def _mtime_fingerprint(pkg_dir: Path | None = None) -> tuple[tuple[str, int, int], ...]:
    """(name, size, mtime_ns) per scanned module — the cache key, so an edit invalidates it."""
    out = []
    for p in package_lever_modules(pkg_dir):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((p.name, st.st_size, st.st_mtime_ns))
    return tuple(out)


@lru_cache(maxsize=8)
def _package_lever_factories_cached(_fingerprint: object,
                                    pkg_dir: str | None = None) -> tuple[FactoryBuild, ...]:
    return _scan_package_lever_factories(Path(pkg_dir) if pkg_dir else None)


def package_lever_factories(pkg_dir: Path | None = None) -> tuple[FactoryBuild, ...]:
    """EVERY lever factory in the whole ``witness_dsl`` package, graded against ITS OWN trainer.

    This is the surface the stub sweep needs and the one the registry never had. Deterministic
    (sorted by module then factory); pure AST + argparse text, no imports, no execution.

    CACHED on the scanned modules' (size, mtime) fingerprint. The uncached scan re-parses ~170
    modules — including the 375 KB ``curriculum_dsl.py`` — and takes seconds; a preflight gate
    that costs seconds per call is a gate that gets turned off, which is how the vacuous-scan
    bug survived in the first place. An edit to any scanned module changes the fingerprint and
    invalidates the entry, so the cache can never serve a stale grade.
    """
    return _package_lever_factories_cached(
        _mtime_fingerprint(pkg_dir), str(pkg_dir) if pkg_dir is not None else None
    )


def _scan_package_lever_factories(pkg_dir: Path | None = None) -> tuple[FactoryBuild, ...]:
    out: list[FactoryBuild] = []
    for mod in package_lever_modules(pkg_dir):
        try:
            src = _module_source(mod)
            facs = _factories_in_source(src)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        if not facs:
            continue
        trainers = module_trainer_paths(mod)
        declared = module_declares_trainer(mod)
        tflags = _flags_of(trainers)
        tname = ", ".join(str(t.relative_to(_REPO_ROOT)) if t.is_relative_to(_REPO_ROOT) else str(t)
                          for t in trainers)
        for name, (flags, marker) in sorted(facs.items()):
            out.append(FactoryBuild(
                module=mod.name, factory=name, flags=tuple(sorted(flags)),
                missing_flags=tuple(sorted(f for f in flags if f not in tflags)),
                trainer=tname, stub_marker=marker, trainer_declared=declared,
            ))
    return tuple(sorted(out, key=lambda f: (f.module, f.factory)))


@dataclass(frozen=True, slots=True)
class BuildCompleteness:
    """Package-wide build-grade summary — the answer to 'which levers are hollow?'."""

    factories: tuple[FactoryBuild, ...]

    @property
    def total(self) -> int:
        return len(self.factories)

    @property
    def stubs(self) -> tuple[FactoryBuild, ...]:
        return tuple(f for f in self.factories if f.is_stub)

    @property
    def silent_stubs(self) -> tuple[FactoryBuild, ...]:
        """Stubs that do NOT announce themselves — the worst grade: they present as BUILT."""
        return tuple(f for f in self.factories if f.is_stub and not f.stub_marker)

    @property
    def label_drift(self) -> tuple[FactoryBuild, ...]:
        return tuple(f for f in self.factories if f.label_drift)

    @property
    def modules_scanned(self) -> int:
        return len({f.module for f in self.factories})

    @property
    def undeclared_trainer_factories(self) -> tuple[FactoryBuild, ...]:
        """Factories whose module never STATED its trainer (ddm_lr2, 2026-08-03).

        Reported as a tracked state, not an error: for most of these the default is correct
        (MEASURED — their flags all live on the retired trainer). The point is that "correct by
        default" and "correct by intent" are now distinguishable to a reader.
        """
        return tuple(f for f in self.factories if not f.trainer_declared)

    @property
    def verdict_relevant_undeclared(self) -> tuple[FactoryBuild, ...]:
        """The refusable subset: graded a STUB *because of* a trainer binding nobody declared."""
        return tuple(f for f in self.factories if f.trainer_binding_is_verdict_relevant)

    def by_trainer(self) -> dict[str, int]:
        """Factory count per resolved trainer — the vehicle census a reader needs before quoting
        any coverage number. A total with no vehicle attached is the defect this answers."""
        out: dict[str, int] = {}
        for f in self.factories:
            out[f.trainer] = out.get(f.trainer, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_factories": self.total,
            "modules_with_factories": self.modules_scanned,
            "modules_globbed": len(package_lever_modules()),
            "stub_count": len(self.stubs),
            "silent_stub_count": len(self.silent_stubs),
            "label_drift_count": len(self.label_drift),
            "undeclared_trainer_factory_count": len(self.undeclared_trainer_factories),
            "verdict_relevant_undeclared_count": len(self.verdict_relevant_undeclared),
            "factories_by_trainer": self.by_trainer(),
            "stubs": [f.to_dict() for f in self.stubs],
        }


def build_completeness(pkg_dir: Path | None = None) -> BuildCompleteness:
    """Grade every lever factory in the package as BUILT or DESIGNED-STUB. Deterministic.

    ``pkg_dir`` overrides the scanned directory (see :func:`package_lever_modules`); the default
    is unchanged, so every existing caller keeps its exact behaviour."""
    return BuildCompleteness(package_lever_factories(pkg_dir))


def _flags_in_node(node: ast.AST) -> frozenset[str]:
    """Every real ``--flag`` string constant lexically inside an AST node (drops ``--no-*``)."""
    return frozenset(
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and _FLAG_RE.match(n.value) and not _NO_ARTIFACT.match(n.value))


def _constructs(node: ast.FunctionDef, name: str) -> bool:
    """True if the function body calls ``name(...)`` (e.g. ``Lever`` / ``WitnessProgram``)."""
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        for n in ast.walk(node)
    )


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
    # WHICH VEHICLE this coverage describes (ddm_lr2, 2026-08-03). The default call scopes to the
    # RETIRED levelset trainer and reports a reassuring ~82% that has been read as a live-vehicle
    # health number. The number is a correct computation about a vehicle we no longer ship; what
    # was missing is that it never said so. Defaults to "" so any positional construction keeps
    # working, and ``describes_live_vehicle`` fails CLOSED (unknown vehicle is not live).
    trainer_path: str = ""

    # The vehicle we actually ship. One place, so a reader and a gate cannot disagree.
    LIVE_TRAINER_BASENAME: ClassVar[str] = "train_tr1_partition_renderer_mlx.py"

    @property
    def coverage_frac(self) -> float:
        return len(self.mapped) / max(self.trainer_total, 1)

    @property
    def describes_live_vehicle(self) -> bool:
        """Whether this coverage number is about the vehicle we ship. Fails closed on unknown."""
        return Path(self.trainer_path).name == self.LIVE_TRAINER_BASENAME if self.trainer_path else False

    @property
    def vehicle_label(self) -> str:
        """A label a reader cannot drop: every coverage number carries the vehicle it measured."""
        if not self.trainer_path:
            return "[vehicle UNKNOWN]"
        return f"[{'LIVE' if self.describes_live_vehicle else 'RETIRED'} vehicle: {Path(self.trainer_path).name}]"


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
    from tac.witness_dsl.curriculum_dsl import TRAINER_PATH
    return Completeness(
        trainer_total=len(trainer),
        dsl_referenced=len(referenced),
        mapped=sorted(trainer & referenced),
        unmapped=sorted(trainer - referenced),
        stale=sorted(emitted - trainer),
        trainer_path=str(tp if tp is not None else TRAINER_PATH),
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
    # ``--dsl-lever NAME`` composes onto the live trainer's ARGV.  A Lever whose overrides are
    # dotted CONFIG keys targets a compiled-JSON vehicle instead (the QBR1/born cells:
    # ``AreaCapBornRareClass`` -> ``area_cap.*``, ``ExpectedFlipTauBandMsafe`` ->
    # ``expected_flip_tau.*``) and cannot be rendered as argv at all -- ``_render_argv`` would
    # emit bare words the trainer argparse refuses.  Excluding them STRUCTURALLY, by the shape of
    # what they emit, rather than accidentally by whether their factory happens to take a
    # required argument: ddm_ng3 found that a config-surface lever with all-default parameters
    # slips into the composable set and fails the CI parse-test with an unrecognized-argument
    # error that names neither the lever nor the reason.
    non_flag = sorted(key for key in lever.overrides if not str(key).startswith("--"))
    if non_flag:
        return None, (
            f"emits CONFIG-surface keys {non_flag[:3]} rather than trainer flags — not "
            "composable via --dsl-lever (it targets a compiled-JSON cell; compile it through "
            "its own config compiler, e.g. compile_qbr1_tau_band_config)")
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


def campaign_activation_nag(campaign_state: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the canonical DDM campaign nag without rebuilding its truth.

    ``activation_ledger`` imports this registry, so the adapter accepts the
    already-composed campaign state and uses the campaign module's digest guard;
    it never imports the activation ledger or performs a second derivation.
    """

    from tac.ddm_campaign_costate import campaign_consumer_view

    return campaign_consumer_view(campaign_state, "activation_nag")


__all__ = [
    "BASELINE",
    "BuildCompleteness",
    "CanonicalMetricDescriptor",
    "Completeness",
    "FactoryBuild",
    "LeverCompositionError",
    "MetricResolutionError",
    "build_completeness",
    "campaign_activation_nag",
    "canonical_metric_ids",
    "completeness",
    "dsl_emitted_flags",
    "dsl_referenced_flags",
    "emit_stub_lever",
    "lever_factories",
    "module_trainer_paths",
    "name_composable_levers",
    "package_lever_factories",
    "package_lever_modules",
    "program_constructors",
    "resolve_canonical_metric",
    "resolve_composable_lever",
]
