"""Canonical xray primitive base protocol + result dataclass.

This module defines the foundation every ``tac.xray.*`` primitive implements.
The contract is intentionally narrow: ONE compute method, ONE typed result.
Composition is via ``compose_with`` returning a new primitive — never via
mutation.

**Why xray primitives are a distinct class from composition primitives.**

The composition registry at :mod:`tac.composition.registry` enumerates
PACKET-COMPILER primitives (PR101 GOLD, sign-encoding, schema-elision,
brotli, CompressAI codecs, etc.) — primitives that *compose into archive
bytes*. Xray primitives are different: they ANALYZE an existing archive,
substrate, or scorer to extract a sensitivity / bound / coverage / margin
report that the solver stack consumes (sensitivity-map, Pareto, bit-allocator,
autopilot, continual-learning, probe-disambiguator).

Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in
NON-NEGOTIABLE, every xray primitive MUST declare which of the 6 hooks it
engages via :attr:`XRayPrimitiveResult.wire_in_hooks_engaged`. Silent
omission is the orphan-work failure mode.

Lane: ``lane_xray_canon_math_findings_wire_in_20260514``.

Cross-references
----------------
- Master math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md``
- Zen-floor council: ``.omx/research/zen_floor_field_medal_grade_council_20260514.md``
- Subagent coherence non-negotiable: ``CLAUDE.md`` "Mandatory wire-in for every landing"

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim`` (primitives never emit score claims)
- ``no_mps_authoritative`` (no MPS gates anywhere in xray)
- ``no_tmp_paths`` (results never reference /tmp/)
- ``xray_canonical_primitive_protocol_v1``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

XRAY_PRIMITIVE_SCHEMA_VERSION = "tac_xray_primitive_v1"

EvidenceGrade = Literal[
    "mathematical-derivation",
    "first-principles-bound",
    "empirical-anchor",
    "proxy",
    "council-deliberation",
    "structural-code-contract",
]

WireInHook = Literal[
    "sensitivity_map",
    "pareto_constraint",
    "bit_allocator",
    "cathedral_autopilot",
    "continual_learning",
    "probe_disambiguator",
]


# The 6 canonical wire-in hooks per CLAUDE.md "Subagent coherence-by-default".
# Subset of these MUST appear in every XRayPrimitiveResult.wire_in_hooks_engaged.
CANONICAL_WIRE_IN_HOOKS: tuple[WireInHook, ...] = (
    "sensitivity_map",
    "pareto_constraint",
    "bit_allocator",
    "cathedral_autopilot",
    "continual_learning",
    "probe_disambiguator",
)


@dataclass(frozen=True)
class XRayPrimitiveResult:
    """Canonical typed result every xray primitive returns.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable,
    the dataclass is frozen + small + machine-checkable. Every field has a
    clear purpose:

    Attributes
    ----------
    primitive_name : str
        Canonical name (e.g., ``"mdl_scorer_conditional"``,
        ``"shannon_vector_r_d"``). Must match the primitive's ``.name``
        attribute. Used as registry key.
    archive_or_video_path : Path | None
        Path to the archive / video / substrate the primitive analyzed.
        ``None`` if the primitive is purely derivational (e.g., a Shannon
        bound that doesn't depend on a specific archive).
    archive_sha256 : str | None
        Hex sha256 of the archive bytes when ``archive_or_video_path``
        points at an archive. ``None`` otherwise. Catalog #92 / Catalog
        #115 require this for any provenance claim.
    primitive_value : Any
        The primitive's typed output — a float (e.g., Lipschitz constant),
        a tensor (e.g., sensitivity map), a dict (e.g., per-section
        breakdown), or a sub-dataclass. Caller introspects via
        ``isinstance``.
    evidence_grade : EvidenceGrade
        One of the 6 canonical evidence grades. Forbids tag-only
        promotion per Catalog #127.
    confidence_band : tuple[float, float] | None
        ``(lower, upper)`` for scalar primitive values. ``None`` for
        tensor / dict outputs. The band is used by the Pareto solver
        for trust-region construction.
    composes_with : tuple[str, ...]
        Other primitive ``name`` strings this primitive can compose with
        via ``compose_with``. Empty tuple = primitive stands alone.
        Used by the composition planner to surface allowed pairings.
    wire_in_hooks_engaged : tuple[WireInHook, ...]
        Non-empty subset of :data:`CANONICAL_WIRE_IN_HOOKS`. Declares
        which of the 6 NON-NEGOTIABLE hooks the primitive's output is
        wired into. Silent empty = construct-time refusal.
    metadata : Mapping[str, Any]
        Free-form per-primitive metadata (e.g., scorer device, sample
        count, sub-section breakdown). Frozen via ``MappingProxyType``
        when constructed.
    schema_version : str
        Pinned to :data:`XRAY_PRIMITIVE_SCHEMA_VERSION`.
    """

    primitive_name: str
    archive_or_video_path: Path | None
    archive_sha256: str | None
    primitive_value: Any
    evidence_grade: EvidenceGrade
    confidence_band: tuple[float, float] | None
    composes_with: tuple[str, ...]
    wire_in_hooks_engaged: tuple[WireInHook, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = XRAY_PRIMITIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.primitive_name:
            raise ValueError("primitive_name must be non-empty")
        if not self.wire_in_hooks_engaged:
            raise ValueError(
                f"primitive {self.primitive_name!r} declared zero wire-in hooks "
                "— this is the orphan-work failure mode per CLAUDE.md "
                "'Subagent coherence-by-default' non-negotiable. Declare at "
                "least one of: " + ", ".join(CANONICAL_WIRE_IN_HOOKS)
            )
        for hook in self.wire_in_hooks_engaged:
            if hook not in CANONICAL_WIRE_IN_HOOKS:
                raise ValueError(
                    f"primitive {self.primitive_name!r} declared unknown "
                    f"wire-in hook {hook!r}; must be one of "
                    f"{CANONICAL_WIRE_IN_HOOKS}"
                )
        if self.confidence_band is not None:
            lo, hi = self.confidence_band
            if lo > hi:
                raise ValueError(
                    f"primitive {self.primitive_name!r} confidence_band has "
                    f"lower > upper ({lo} > {hi})"
                )

    def hook_count(self) -> int:
        """Return number of distinct wire-in hooks engaged."""
        return len(set(self.wire_in_hooks_engaged))


@runtime_checkable
class XRayPrimitive(Protocol):
    """Canonical protocol every xray primitive implements.

    Three methods:

    - :meth:`name` — returns canonical primitive name (matches registry key).
    - :meth:`compute` — runs the analysis, returns :class:`XRayPrimitiveResult`.
    - :meth:`compose_with` — returns a new composed primitive (no mutation).
    """

    @property
    def name(self) -> str:
        """Canonical primitive name (registry key)."""
        ...

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        """Declare which of the 6 hooks this primitive engages.

        Non-empty subset of :data:`CANONICAL_WIRE_IN_HOOKS`. Used by the
        registry to surface coverage gaps.
        """
        ...

    def compute(
        self,
        target: Path | Any,
        **kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Run the primitive's analysis and return a typed result.

        Parameters
        ----------
        target : Path | Any
            Archive path, video path, substrate model, tensor, etc.
            Per-primitive type contract.
        **kwargs : Any
            Per-primitive optional configuration.

        Returns
        -------
        XRayPrimitiveResult
            Typed result with non-empty ``wire_in_hooks_engaged``.

        Raises
        ------
        ValueError
            If target is not the expected type or required kwargs are missing.
        """
        ...

    def compose_with(self, other: "XRayPrimitive") -> "XRayPrimitive":
        """Return a new primitive that runs both and combines outputs.

        The default composition (provided by :class:`ComposedXRayPrimitive`)
        runs each primitive independently and aggregates their results.
        Primitives that have non-trivial composition (e.g., a Pareto
        constraint composed with a Lipschitz bound producing a tighter
        Pareto vertex) MAY override.
        """
        ...


@dataclass
class ComposedXRayPrimitive:
    """Default composition: run two primitives independently, aggregate results.

    The aggregated :class:`XRayPrimitiveResult` is constructed with:

    - ``primitive_name = f"{left.name}+{right.name}"``
    - ``primitive_value = {"left": left_result, "right": right_result}``
    - ``wire_in_hooks_engaged = union(left.hooks, right.hooks)``
    - ``composes_with = ()`` (composed primitives don't auto-compose further)
    - ``evidence_grade`` = weakest of the two (precedence:
       ``empirical-anchor`` > ``mathematical-derivation`` >
       ``first-principles-bound`` > ``structural-code-contract`` >
       ``council-deliberation`` > ``proxy``)
    - ``confidence_band`` = ``None`` (composed primitives don't have a
       single scalar band).
    """

    left: XRayPrimitive
    right: XRayPrimitive

    @property
    def name(self) -> str:
        return f"{self.left.name}+{self.right.name}"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        seen: list[WireInHook] = []
        for h in self.left.wire_in_hooks + self.right.wire_in_hooks:
            if h not in seen:
                seen.append(h)
        return tuple(seen)

    def compute(
        self,
        target: Path | Any,
        **kwargs: Any,
    ) -> XRayPrimitiveResult:
        left_result = self.left.compute(target, **kwargs)
        right_result = self.right.compute(target, **kwargs)
        # Weakest-evidence-precedence aggregator: weaker grade wins.
        precedence = [
            "proxy",
            "council-deliberation",
            "structural-code-contract",
            "first-principles-bound",
            "mathematical-derivation",
            "empirical-anchor",
        ]
        lidx = precedence.index(left_result.evidence_grade)
        ridx = precedence.index(right_result.evidence_grade)
        weakest = precedence[min(lidx, ridx)]
        seen_hooks: list[WireInHook] = []
        for h in (
            left_result.wire_in_hooks_engaged
            + right_result.wire_in_hooks_engaged
        ):
            if h not in seen_hooks:
                seen_hooks.append(h)
        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=left_result.archive_or_video_path
            or right_result.archive_or_video_path,
            archive_sha256=left_result.archive_sha256
            or right_result.archive_sha256,
            primitive_value={"left": left_result, "right": right_result},
            evidence_grade=weakest,  # type: ignore[arg-type]
            confidence_band=None,
            composes_with=(),
            wire_in_hooks_engaged=tuple(seen_hooks),
            metadata={
                "composed_left": left_result.primitive_name,
                "composed_right": right_result.primitive_name,
            },
        )

    def compose_with(self, other: XRayPrimitive) -> XRayPrimitive:
        # Right-associative chain.
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "CANONICAL_WIRE_IN_HOOKS",
    "ComposedXRayPrimitive",
    "EvidenceGrade",
    "WireInHook",
    "XRAY_PRIMITIVE_SCHEMA_VERSION",
    "XRayPrimitive",
    "XRayPrimitiveResult",
]
