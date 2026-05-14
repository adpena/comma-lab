# SPDX-License-Identifier: MIT
"""Canonical xray (archive/substrate/scorer analysis) primitive package.

Per operator directive 2026-05-14 ("keep enhancing xray capabilities and add
math findings to canonical reusable composable tools that are wired and
integrated"), this package promotes one-off math findings + research-memo
derivations into canonical reusable composable :class:`XRayPrimitive`
implementations.

Architectural contrast with :mod:`tac.composition.registry`:

- :mod:`tac.composition.registry` enumerates PACKET-COMPILER primitives
  (PR101 GOLD, sign-encoding, schema-elision, brotli, CompressAI) —
  primitives that COMPOSE INTO archive bytes.
- :mod:`tac.xray` enumerates ANALYTIC primitives — primitives that
  ANALYZE archives / substrates / scorers and produce typed
  :class:`XRayPrimitiveResult` rows that the 6 canonical solver-stack
  hooks consume (sensitivity_map / Pareto / bit_allocator / autopilot /
  continual_learning / probe-disambiguator).

The two registries are sister surfaces — both implement Catalog #169's
canonical-primitive-inventory pattern.

**Canonical contract.** Every xray primitive:

1. Implements :class:`tac.xray.base.XRayPrimitive` (Protocol).
2. Returns :class:`tac.xray.base.XRayPrimitiveResult` from ``.compute()``.
3. Declares non-empty ``wire_in_hooks_engaged`` (orphan-work fail-closed).
4. Is registered in :func:`tac.xray.registry.canonical_xray_primitive_inventory`.
5. Composes via :meth:`XRayPrimitive.compose_with` returning a new primitive
   (immutable composition).

**Lane:** ``lane_xray_canon_math_findings_wire_in_20260514``.

**Memory:** ``feedback_xray_canon_math_findings_wire_in_landed_20260514.md``.

**Cross-references:**

- Master math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md``
- Zen-floor council: ``.omx/research/zen_floor_field_medal_grade_council_20260514.md``
- Time-traveler architecture: ``.omx/research/time_traveler_architecture_reverse_engineered_20260513.md``
"""

from tac.xray.base import (
    CANONICAL_WIRE_IN_HOOKS,
    XRAY_PRIMITIVE_SCHEMA_VERSION,
    ComposedXRayPrimitive,
    EvidenceGrade,
    WireInHook,
    XRayPrimitive,
    XRayPrimitiveResult,
)
from tac.xray.registry import (
    XRAY_REGISTRY_SCHEMA_VERSION,
    XRayPrimitiveSpec,
    canonical_xray_primitive_inventory,
    get_xray_primitive_spec,
    serialize_xray_inventory,
    specs_by_category,
    specs_by_hook,
)
from tac.xray.wire_in import (
    XRayWireInBundle,
    aggregate_hook_evidence_grade,
    discover_primitives_by_hook,
    instantiate_primitive,
    wire_in_for_hook,
)

__all__ = [
    "CANONICAL_WIRE_IN_HOOKS",
    "ComposedXRayPrimitive",
    "EvidenceGrade",
    "WireInHook",
    "XRAY_PRIMITIVE_SCHEMA_VERSION",
    "XRAY_REGISTRY_SCHEMA_VERSION",
    "XRayPrimitive",
    "XRayPrimitiveResult",
    "XRayPrimitiveSpec",
    "XRayWireInBundle",
    "aggregate_hook_evidence_grade",
    "canonical_xray_primitive_inventory",
    "discover_primitives_by_hook",
    "get_xray_primitive_spec",
    "instantiate_primitive",
    "serialize_xray_inventory",
    "specs_by_category",
    "specs_by_hook",
    "wire_in_for_hook",
]
