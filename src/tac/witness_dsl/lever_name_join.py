# SPDX-License-Identifier: MIT
"""ddm_pt2 — the factory<->instance lever-name JOIN (the prerequisite ``ddm_lr2`` measured and owed).

``ddm_lr2`` §2.4 MEASURED the defect this module closes: the activation ledger keys on DSL
**factory** names (:func:`tac.witness_dsl.activation_ledger.known_levers`, e.g.
``lever_token_grid``) while a sealed launch ticket records the constructed **instance** name
(``Lever.name``, e.g. ``tr1_token_grid_D16_c4``), and the live spec builds those names with
f-strings parameterised by the factory's own arguments.  The two key spaces are structurally
disjoint — **0 of 43 live instance names joined**, MEASURED — so wiring ``record_activation``
into ``tools/launch_tr1_run.py`` FIRST would pile rows under keys nothing joins to and leave
``never_fired()`` reporting every factory as never-fired forever: a SECOND cross-store
contradiction wearing the costume of a fix.  lr2 declined to fake it and named the join as the
highest-value owed row.  This module is that row.

**Mechanism (static, no imports, no execution).**  Each lever factory is AST-scanned for the
``name=`` argument of every ``Lever(...)`` it constructs.  Three shapes occur in the live tree:

* ``ast.Constant``     -> a LITERAL name (151 of 177 MEASURED).
* ``ast.JoinedStr``    -> a TEMPLATE; each interpolation becomes a ``(?:.+)`` hole and the
  literal segments are ``re.escape``d, giving a regex that matches exactly the family of
  instance names that factory can emit (21 MEASURED).
* ``ast.Call`` whose ``func`` is an attribute of a ``JoinedStr`` (``f"...".rstrip("0")``)
  -> unwrapped to the underlying JoinedStr.  ``lever_ema_decay`` is the live instance of this
  shape; without the unwrap its two real launched names do not resolve (MEASURED before/after).

**Precedence is LITERAL-first, and that is load-bearing, not cosmetic.**  ``lever_seg_physics``
emits ``f"tr1_seg_{form_start}"``, whose regex ``^tr1_seg_(.+)$`` also matches the LITERAL name
``tr1_seg_margin_weight`` emitted by ``lever_seg_margin_weight``.  Resolving literals first makes
that case exact; the residual template-vs-template overlap is REPORTED as ``AMBIGUOUS``, never
silently collapsed to a first match.

**A name that does not resolve is a FINDING, not a join bug.**  :data:`UNRESOLVED_NOT_A_FACTORY`
is returned for an instance name no factory in the package can emit — which is exactly what a
hand-constructed ``Lever(...)`` built outside the DSL looks like.  MEASURED on the live tickets:
``tr1_coupling_field_only`` is built inline in ``experiments/ddm_pa1r_seal_and_tickets.py`` and
``qa86_live_config_pin`` has no repo provenance at all.  Both are config-orphans of exactly the
class CLAUDE.md's "the DSL HOLDS every designed lever" law forbids, and the join surfaces them
instead of hiding them under a fallback.

Evidence axis: static analysis of committed source; ``score_claim=False``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.witness_dsl.lever_registry import _module_source, package_lever_modules

#: resolution kinds returned by :func:`resolve_instance`.
RESOLVED_LITERAL = "literal"
RESOLVED_TEMPLATE = "template"
AMBIGUOUS = "ambiguous"
UNRESOLVED_NOT_A_FACTORY = "unresolved_not_a_dsl_factory"

#: the interpolation hole. ``.+`` (not ``[^_]+``) because real instance names carry underscores
#: INSIDE a hole — MEASURED: ``tr1_token_temporal_shared_base``, ``tr1_token_init_solve_project``,
#: ``tr1_solve_frame_distill_kd_logits``. A narrower hole would silently fail to resolve them.
_HOLE = "(?:.+)"


@dataclass(frozen=True, slots=True)
class FactoryNameForm:
    """One ``Lever(name=...)`` expression found inside one factory."""

    module: str
    factory: str
    kind: str          # RESOLVED_LITERAL | RESOLVED_TEMPLATE
    literal: str | None
    pattern: str | None

    @property
    def ref(self) -> tuple[str, str]:
        return (self.module, self.factory)

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "factory": self.factory, "kind": self.kind,
                "literal": self.literal, "pattern": self.pattern}


@dataclass(frozen=True, slots=True)
class InstanceResolution:
    """The join answer for ONE instance name, with its basis attached."""

    instance: str
    kind: str                      # one of the four module constants
    factory_refs: tuple[tuple[str, str], ...]

    @property
    def resolved(self) -> bool:
        return len(self.factory_refs) == 1

    @property
    def factory(self) -> str | None:
        return self.factory_refs[0][1] if self.resolved else None

    @property
    def module(self) -> str | None:
        return self.factory_refs[0][0] if self.resolved else None

    def to_dict(self) -> dict[str, Any]:
        return {"instance": self.instance, "kind": self.kind, "resolved": self.resolved,
                "factory": self.factory, "module": self.module,
                "factory_refs": [f"{m}::{f}" for m, f in self.factory_refs]}


def _unwrap_name_expr(expr: ast.AST) -> ast.AST:
    """``f"...".rstrip("0")`` -> the underlying ``JoinedStr``; anything else unchanged.

    Method calls on an f-string only ever TRIM the rendered name (``rstrip``/``strip``/``replace``
    are the shapes in tree), so the family regex still matches the trimmed result.  Not unwrapping
    costs real resolutions: ``lever_ema_decay``'s two launched names are MEASURED unresolvable
    without this.
    """
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        inner = expr.func.value
        if isinstance(inner, (ast.JoinedStr, ast.Constant)):
            return inner
    return expr


def _name_exprs(fn: ast.AST) -> list[ast.AST]:
    """Every ``name=`` argument passed to a ``Lever(...)`` call inside this function def."""
    out: list[ast.AST] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        is_lever = ((isinstance(node.func, ast.Name) and node.func.id == "Lever")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "Lever"))
        if not is_lever:
            continue
        kw = [k.value for k in node.keywords if k.arg == "name"]
        if kw:
            out.extend(kw)
        elif node.args:           # positional: Lever("name", overrides=...) — the fh1/ph3 idiom
            out.append(node.args[0])
    return out


def _form_of(module: str, factory: str, expr: ast.AST) -> FactoryNameForm | None:
    expr = _unwrap_name_expr(expr)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return FactoryNameForm(module, factory, RESOLVED_LITERAL, expr.value, None)
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(re.escape(v.value))
            else:
                parts.append(_HOLE)
        return FactoryNameForm(module, factory, RESOLVED_TEMPLATE, None, "^" + "".join(parts) + "$")
    return None  # a fully dynamic name (a bare Name/Call) — cannot be joined statically; reported


def factory_name_forms(pkg_dir: Path | None = None) -> tuple[FactoryNameForm, ...]:
    """Every statically-joinable ``Lever`` name form in the whole ``witness_dsl`` package.

    Deterministic (sorted by module, factory, then the literal/pattern text).  Pure AST over
    committed source: no imports, no MLX, no execution — the same discipline
    ``package_lever_factories`` follows, and for the same reason.
    """
    out: list[FactoryNameForm] = []
    for mod in package_lever_modules(pkg_dir):
        try:
            tree = ast.parse(_module_source(mod))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for expr in _name_exprs(node):
                form = _form_of(mod.name, node.name, expr)
                if form is not None:
                    out.append(form)
    return tuple(sorted(out, key=lambda f: (f.module, f.factory, f.literal or f.pattern or "")))


def dynamic_name_factories(pkg_dir: Path | None = None) -> tuple[tuple[str, str], ...]:
    """Factories whose ``Lever`` name is NEITHER a literal nor an f-string — the honest residue.

    These cannot be joined statically by construction.  Reporting them is the denominator: a join
    that silently omitted them would look complete while being blind to a named subset.
    """
    out: set[tuple[str, str]] = set()
    for mod in package_lever_modules(pkg_dir):
        try:
            tree = ast.parse(_module_source(mod))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for expr in _name_exprs(node):
                if _form_of(mod.name, node.name, expr) is None:
                    out.add((mod.name, node.name))
    return tuple(sorted(out))


def resolve_instance(instance: str,
                     forms: tuple[FactoryNameForm, ...] | None = None) -> InstanceResolution:
    """Map ONE constructed ``Lever.name`` back to the factory that can emit it.

    LITERAL matches take precedence over TEMPLATE matches (see the module docstring: a template
    hole is ``.+`` and would otherwise swallow a sibling factory's literal name).  Multiple
    surviving candidates return :data:`AMBIGUOUS` with ALL of them — never a silent first-match.
    """
    forms = factory_name_forms() if forms is None else forms
    lit = sorted({f.ref for f in forms if f.kind == RESOLVED_LITERAL and f.literal == instance})
    if len(lit) == 1:
        return InstanceResolution(instance, RESOLVED_LITERAL, tuple(lit))
    if len(lit) > 1:
        return InstanceResolution(instance, AMBIGUOUS, tuple(lit))
    tmpl = sorted({f.ref for f in forms
                   if f.kind == RESOLVED_TEMPLATE and re.match(f.pattern or r"(?!)", instance)})
    if len(tmpl) == 1:
        return InstanceResolution(instance, RESOLVED_TEMPLATE, tuple(tmpl))
    if len(tmpl) > 1:
        return InstanceResolution(instance, AMBIGUOUS, tuple(tmpl))
    return InstanceResolution(instance, UNRESOLVED_NOT_A_FACTORY, ())


def resolve_instances(instances: tuple[str, ...] | list[str],
                      forms: tuple[FactoryNameForm, ...] | None = None
                      ) -> tuple[InstanceResolution, ...]:
    """Vectorised :func:`resolve_instance` — builds the (expensive) form table exactly once."""
    forms = factory_name_forms() if forms is None else forms
    return tuple(resolve_instance(i, forms) for i in instances)


def live_launch_factory_names(roots: tuple[str, ...] | None = None) -> tuple[dict[str, int], dict]:
    """The live vehicle's activation history, keyed by FACTORY name — what the ledger could not say.

    Reads the governed TR1 launch receipts' sealed tickets (via
    ``activation_ledger.live_launch_lever_names``) and joins each recorded instance name back to
    its factory.  Returns ``(factory -> launch count, diagnostics)``.  The diagnostics carry the
    DENOMINATOR and every unresolved/ambiguous name by name, so an empty or partial scan can never
    read as a clean one (an unmounted SSD is reported, not assumed empty).
    """
    from tac.witness_dsl.activation_ledger import live_launch_lever_names

    counts, diag = live_launch_lever_names(roots)
    forms = factory_name_forms()
    by_factory: dict[str, int] = {}
    unresolved: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for inst, n in counts.items():
        r = resolve_instance(inst, forms)
        if r.resolved:
            by_factory[r.factory] = by_factory.get(r.factory, 0) + int(n)
        elif r.kind == AMBIGUOUS:
            ambiguous[inst] = [f"{m}::{f}" for m, f in r.factory_refs]
        else:
            unresolved.append(inst)
    out_diag = dict(diag)
    out_diag.update({
        "join_forms_scanned": len(forms),
        "instances_seen": len(counts),
        "instances_joined_to_factory": len(counts) - len(unresolved) - len(ambiguous),
        "instances_ambiguous": ambiguous,
        "instances_unresolved_not_a_dsl_factory": sorted(unresolved),
        "distinct_factories_launched": len(by_factory),
        "dynamic_name_factories": [f"{m}::{f}" for m, f in dynamic_name_factories()],
    })
    return dict(sorted(by_factory.items())), out_diag
