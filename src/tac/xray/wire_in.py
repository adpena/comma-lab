"""Canonical solver-stack wire-in surface for xray primitives.

Per CLAUDE.md "Subagent coherence-by-default" 6-hook NON-NEGOTIABLE, every
xray primitive declares which of the 6 canonical hooks (sensitivity_map /
pareto_constraint / bit_allocator / cathedral_autopilot / continual_learning
/ probe_disambiguator) it engages. This module provides a NARROW typed API
for solver-stack consumers to:

1. Discover which xray primitives feed their hook.
2. Construct primitive instances from the canonical inventory.
3. Run primitives in a batch and collect their results into a wire-in
   bundle the consumer can iterate over.

The wire-in bundle is the canonical hand-off between this xray package and
the four major downstream consumer surfaces:

- ``tac.sensitivity_map.*`` (8 primitives wire in)
- ``tac.optimization.bit_allocator_end_to_end`` (7 primitives wire in)
- ``tac.optimization.autopilot_dispatch_ranking`` (3 primitives wire in)
- ``tac.optimization.field_equation_planner`` (3 primitives wire in)
- ``tac.continual_learning`` (1 primitive wires in)

Each consumer's adapter code iterates :func:`wire_in_for_hook` and consumes
:class:`tac.xray.base.XRayPrimitiveResult` rows in the canonical format —
no consumer-side primitive-specific code required.

Lane: ``lane_xray_canon_math_findings_wire_in_20260514``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from tac.xray.base import (
    CANONICAL_WIRE_IN_HOOKS,
    WireInHook,
    XRayPrimitive,
    XRayPrimitiveResult,
)
from tac.xray.registry import (
    XRayPrimitiveSpec,
    canonical_xray_primitive_inventory,
    specs_by_hook,
)


@dataclass(frozen=True)
class XRayWireInBundle:
    """Typed bundle the solver stack consumes for a single hook.

    Iterating over :attr:`results` yields one :class:`XRayPrimitiveResult`
    per primitive that engaged the hook. Each result carries its own
    ``primitive_name``, ``evidence_grade``, ``confidence_band`` so the
    consumer can rank / threshold / weight them.
    """

    hook: WireInHook
    n_primitives: int
    results: tuple[XRayPrimitiveResult, ...]
    skipped_primitives: tuple[tuple[str, str], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if self.hook not in CANONICAL_WIRE_IN_HOOKS:
            raise ValueError(
                f"hook {self.hook!r} must be one of {CANONICAL_WIRE_IN_HOOKS}"
            )
        if self.n_primitives < 0:
            raise ValueError("n_primitives must be non-negative")
        if len(self.results) > self.n_primitives:
            raise ValueError(
                "results length cannot exceed declared n_primitives"
            )


def instantiate_primitive(spec: XRayPrimitiveSpec) -> XRayPrimitive:
    """Instantiate the primitive class declared by ``spec``.

    Imports the canonical module + symbol declared in the registry and
    returns a fresh primitive instance (zero-arg construction is the
    canonical contract).
    """
    module = import_module(spec.canonical_module)
    cls = getattr(module, spec.canonical_symbol)
    return cls()


def wire_in_for_hook(
    hook: WireInHook,
    *,
    targets: Mapping[str, Any] | None = None,
    skip_on_error: bool = True,
) -> XRayWireInBundle:
    """Discover + run xray primitives that engage ``hook``.

    Parameters
    ----------
    hook : WireInHook
        One of the 6 canonical hooks.
    targets : Mapping[str, Any] | None
        Per-primitive target/kwarg map. The mapping key is the primitive's
        ``name`` and the value is the kwargs dict to pass to ``compute``.
        Primitives whose target is missing are skipped (recorded in
        ``skipped_primitives`` rather than failing the whole bundle).
    skip_on_error : bool
        When True, primitive .compute exceptions are caught and the
        primitive is added to ``skipped_primitives`` with the error
        string. When False, exceptions propagate.

    Returns
    -------
    XRayWireInBundle
        Bundle containing successful primitive results + skip log.
    """
    if hook not in CANONICAL_WIRE_IN_HOOKS:
        raise ValueError(
            f"hook {hook!r} must be one of {CANONICAL_WIRE_IN_HOOKS}"
        )
    if targets is None:
        targets = {}

    specs = specs_by_hook(hook)
    results: list[XRayPrimitiveResult] = []
    skipped: list[tuple[str, str]] = []

    for spec in specs:
        if spec.primitive_name not in targets:
            skipped.append(
                (spec.primitive_name, "no target provided")
            )
            continue
        try:
            primitive = instantiate_primitive(spec)
        except Exception as exc:
            if not skip_on_error:
                raise
            skipped.append(
                (spec.primitive_name, f"instantiation failed: {exc}")
            )
            continue
        try:
            kwargs = dict(targets[spec.primitive_name])
            target_value = kwargs.pop("target", None)
            result = primitive.compute(target_value, **kwargs)
            results.append(result)
        except Exception as exc:
            if not skip_on_error:
                raise
            skipped.append(
                (spec.primitive_name, f"compute failed: {exc}")
            )

    return XRayWireInBundle(
        hook=hook,
        n_primitives=len(specs),
        results=tuple(results),
        skipped_primitives=tuple(skipped),
    )


def discover_primitives_by_hook() -> dict[WireInHook, list[str]]:
    """Return ``{hook -> [primitive_name, ...]}`` for all 6 hooks.

    Solver-stack consumers use this to introspect the wire-in surface
    at startup (e.g., to print "the sensitivity-map is fed by these
    11 xray primitives").
    """
    out: dict[WireInHook, list[str]] = {}
    for hook in CANONICAL_WIRE_IN_HOOKS:
        out[hook] = [s.primitive_name for s in specs_by_hook(hook)]
    return out


def aggregate_hook_evidence_grade(
    bundle: XRayWireInBundle,
) -> str:
    """Return the WEAKEST evidence grade across all results in the bundle.

    Used by the consumer to determine the effective evidence grade of any
    decision derived from the hook's combined signal. The precedence is the
    same as :class:`tac.xray.base.ComposedXRayPrimitive`.
    """
    if not bundle.results:
        return "proxy"
    precedence = [
        "proxy",
        "council-deliberation",
        "structural-code-contract",
        "first-principles-bound",
        "mathematical-derivation",
        "empirical-anchor",
    ]
    weakest_idx = len(precedence) - 1
    for r in bundle.results:
        if r.evidence_grade in precedence:
            weakest_idx = min(weakest_idx, precedence.index(r.evidence_grade))
    return precedence[weakest_idx]


__all__ = [
    "XRayWireInBundle",
    "aggregate_hook_evidence_grade",
    "discover_primitives_by_hook",
    "instantiate_primitive",
    "wire_in_for_hook",
]
