# SPDX-License-Identifier: MIT
"""CompressTimePassContract — canonical schema for a single compress-time pass.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5.3-§5.5 + §G compress-time techniques: every compress-time pass MUST declare
a frozen ``CompressTimePassContract`` that captures its identity, position in
the pipeline, what it consumes / emits, its correction kind + resolution, and
production-hardening invariants (deterministic, sensitivity_weighted,
max_wallclock_seconds, rate_budget_bytes, distortion_budget).

Mirrors the canonical ``BoostStageContract`` from ``tac.boosting.contract`` at
the compress-time-pass surface — same frozen-dataclass + field-level validator
pattern, raises ``CompressTimePassContractError`` on any violation at
construction time so malformed passes fail at IMPORT time (not at dispatch
time).

Distinguishing fields unique to this namespace (per PV-7 canonical-vs-unique
decision per layer):
  - ``max_wallclock_seconds: int | None`` — None is LEGAL (compress-time is
    unbounded per CLAUDE.md §G). For inflate-time stages this would be capped
    at 30 min — but ``stage_phase='inflate'`` is FORBIDDEN in this namespace.
  - ``rate_budget_bytes: int | None`` — optional explicit rate cap; the
    pipeline filter ``with_rate_budget(bytes=N)`` consumes this.
  - ``distortion_budget: float | None`` — optional explicit distortion cap.
  - ``seed: int | None`` — required when the function declares randomness
    AND ``deterministic=True``; the decorator's signature inspector enforces.

Per CLAUDE.md "Beauty, simplicity, and developer experience":
narrow API — one dataclass, one error class. The decorator + pipeline modules
consume this contract; consumers DO NOT subclass it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from tac.compress_time_optimization.errors import (
    CompressTimePassContractError,
    InflatePhaseForbiddenError,
)

__all__ = [
    "CompressTimePassContract",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_MERGE_POLICY",
    "LEGAL_STAGE_PHASE",
    "NOT_APPLICABLE_WITH_RATIONALE",
]


# ---------------------------------------------------------------------------
# Enum-like legal value sets (frozensets for O(1) membership; immutable).
# ---------------------------------------------------------------------------

LEGAL_CORRECTION_KIND: frozenset[str] = frozenset(
    {
        "refinement",         # generic TTO refinement (e.g. SGD on params)
        "residual_correction",# delta cascade between passes
        "search",             # coordinate / SA / bisection search
        "bisection",          # iterated bisection on R-D knee
        "transform",          # deterministic transform (e.g. wavelet)
        "passthrough",        # seed/no-op stage
    }
)

LEGAL_CORRECTION_RESOLUTION: frozenset[str] = frozenset(
    {
        "per_frame",
        "per_pair",
        "per_pixel",
        "per_block",
        "per_byte",
        "per_stream",
        "per_tensor",
        "global",
    }
)

# This namespace is COMPRESS-TIME ONLY. ``inflate`` is FORBIDDEN at decoration
# (raised in __post_init__). The sister namespace
# ``tac.inflate_time_post_processing`` (deferred per spec §5.2) covers
# inflate-time stages. Per the spec compress-time and archive-build are the
# admissible phases.
LEGAL_STAGE_PHASE: frozenset[str] = frozenset(
    {
        "compress",       # runs at compress time; primary phase
        "archive_build",  # runs at archive-pack time; assembles final bytes
    }
)

# Forbidden phase tokens that raise InflatePhaseForbiddenError at decoration.
_FORBIDDEN_STAGE_PHASES: frozenset[str] = frozenset({"inflate", "post_process"})

LEGAL_MERGE_POLICY: frozenset[str] = frozenset(
    {
        "last_writer_wins",
        "first_writer_wins",
        "additive",
        "concatenate",
        "explicit",
    }
)

NOT_APPLICABLE_WITH_RATIONALE = "not_applicable_with_rationale"

LEGAL_HOOK_SENSITIVITY: frozenset[str] = frozenset(
    {
        "master_gradient_v1",
        "scorer_conditional_entropy_map_v1",
        "axis_weights_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_PARETO: frozenset[str] = frozenset(
    {
        "rate_distortion_v1",
        "cost_band_envelope_v1",
        "compress_time_pareto_front_tracker_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_BIT_ALLOCATOR: frozenset[str] = frozenset(
    {
        "per_tensor_uniform",
        "per_channel_lsq",
        "ibps_kkt",
        "sensitivity_weighted_water_filling",
        "iterated_bisection",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_AUTOPILOT: frozenset[str] = frozenset(
    {
        "cathedral_autopilot_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_CONTINUAL_LEARNING: frozenset[str] = frozenset(
    {
        "paired_axis",
        "cuda_only",
        "cpu_only",
        "macos_cpu_advisory",
        "compress_time_optimization_pass_outcomes_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, kw_only=True)
class CompressTimePassContract:
    """Canonical contract declared by every compress-time pass via
    ``@compress_time_pass``.

    Mirrors ``BoostStageContract`` at the compress-time-pass surface — one
    frozen dataclass, field-level validators, no inheritance.

    Field groups:
      1. Identity + pipeline ordering (id, parent_pass_id, stage_phase,
         stage_order, description)
      2. Wire contract (consumes, emits, correction_kind, correction_resolution)
      3. Production hardening (deterministic, scorer_free, sensitivity_weighted,
         max_wallclock_seconds, rate_budget_bytes, distortion_budget, seed,
         merge_policy)
      4. 6-hook wire-in (Catalog #125)
      5. Provenance (lane_id, design_memo, canonical_vs_unique_decision)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": every consumer (Pipeline,
    persistence ledger, autopilot ranker) reads from this contract via
    explicit field access — no hidden state, no class hierarchy. The
    namespace does NOT share BoostStageContract: tac.boosting and
    tac.compress_time_optimization are STRUCTURALLY INDEPENDENT
    (per PV-7 + Catalog #290).
    """

    # 1. Identity + pipeline ordering (5)
    id: str
    parent_pass_id: str | None = None
    stage_phase: str = "compress"
    stage_order: int | None = None
    description: str = ""

    # 2. Wire contract (4)
    consumes: frozenset[str] = field(default_factory=frozenset)
    emits: frozenset[str] = field(default_factory=frozenset)
    correction_kind: str = "refinement"
    correction_resolution: str = "per_pair"

    # 3. Production hardening (8)
    deterministic: bool = True
    scorer_free: bool = True
    sensitivity_weighted: bool = False
    max_wallclock_seconds: int | None = None
    rate_budget_bytes: int | None = None
    distortion_budget: float | None = None
    seed: int | None = None
    merge_policy: str = "last_writer_wins"

    # 4. 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_pareto_constraint: str = "rate_distortion_v1"
    hook_bit_allocator_class: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_autopilot_ranker: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_continual_learning_anchor_kind: str = (
        "compress_time_optimization_pass_outcomes_v1"
    )
    hook_probe_disambiguator: str | None = None
    hook_not_applicable_rationale: dict[str, str] = field(default_factory=dict)

    # 5. Provenance (3)
    lane_id: str | None = None
    design_memo: str | None = None
    canonical_vs_unique_decision: str | None = None

    # ------------------------------------------------------------------
    # Validation (runs in __post_init__)
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # ---- 1. Identity validation
        if not isinstance(self.id, str) or not _ID_PATTERN.match(self.id):
            raise CompressTimePassContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/"
            )
        if self.parent_pass_id is not None:
            if not isinstance(self.parent_pass_id, str) or not _ID_PATTERN.match(
                self.parent_pass_id
            ):
                raise CompressTimePassContractError(
                    f"parent_pass_id={self.parent_pass_id!r} must be None or "
                    "match /^[a-z][a-z0-9_]*$/"
                )
            if self.parent_pass_id == self.id:
                raise CompressTimePassContractError(
                    f"parent_pass_id={self.parent_pass_id!r} must differ from "
                    f"id={self.id!r} (a pass cannot be its own parent)"
                )

        # Inflate phase is FORBIDDEN in this namespace per spec §5.2.
        if self.stage_phase in _FORBIDDEN_STAGE_PHASES:
            raise InflatePhaseForbiddenError(
                f"stage_phase={self.stage_phase!r} is FORBIDDEN in the "
                f"tac.compress_time_optimization namespace. This namespace is "
                f"COMPRESS-TIME ONLY. Use the sister namespace "
                f"tac.inflate_time_post_processing (per spec §5.2 build queue) "
                f"for inflate-time stages."
            )
        if self.stage_phase not in LEGAL_STAGE_PHASE:
            raise CompressTimePassContractError(
                f"stage_phase={self.stage_phase!r} not in "
                f"{sorted(LEGAL_STAGE_PHASE)}"
            )

        if self.stage_order is not None:
            if not isinstance(self.stage_order, int) or isinstance(
                self.stage_order, bool
            ):
                raise CompressTimePassContractError(
                    f"stage_order={self.stage_order!r} must be None or int"
                )
            if self.stage_order < 0:
                raise CompressTimePassContractError(
                    f"stage_order={self.stage_order} must be >= 0"
                )

        # ---- 2. Wire contract validation
        for fname, value in (("consumes", self.consumes), ("emits", self.emits)):
            if not isinstance(value, frozenset):
                try:
                    converted = frozenset(value)
                except TypeError as exc:
                    raise CompressTimePassContractError(
                        f"{fname}={value!r} must be a frozenset/set/list of str"
                    ) from exc
                object.__setattr__(self, fname, converted)
                value = converted
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise CompressTimePassContractError(
                        f"{fname} contains non-string or empty entry: {item!r}"
                    )

        # Overlap check: forbidden for non-passthrough kinds. Passthrough
        # passes (correction_kind="passthrough") REQUIRE emits == consumes
        # so the overlap is structurally legal and we exempt them.
        overlap = self.consumes & self.emits
        if overlap and self.correction_kind != "passthrough":
            raise CompressTimePassContractError(
                f"consumes ∩ emits is non-empty: {sorted(overlap)!r}. A pass "
                "that consumes a key cannot also emit the same key (use a "
                "versioned emit name like '<key>_v1' to express in-place "
                "refinement)."
            )

        if self.correction_kind not in LEGAL_CORRECTION_KIND:
            raise CompressTimePassContractError(
                f"correction_kind={self.correction_kind!r} not in "
                f"{sorted(LEGAL_CORRECTION_KIND)}"
            )
        if self.correction_resolution not in LEGAL_CORRECTION_RESOLUTION:
            raise CompressTimePassContractError(
                f"correction_resolution={self.correction_resolution!r} not in "
                f"{sorted(LEGAL_CORRECTION_RESOLUTION)}"
            )

        # ---- 3. Production hardening
        for fname, value in (
            ("deterministic", self.deterministic),
            ("scorer_free", self.scorer_free),
            ("sensitivity_weighted", self.sensitivity_weighted),
        ):
            if not isinstance(value, bool):
                raise CompressTimePassContractError(f"{fname} must be bool")

        # max_wallclock_seconds: None LEGAL per CLAUDE.md §G (compress unbounded);
        # int must be >= 1 if provided.
        if self.max_wallclock_seconds is not None:
            if isinstance(self.max_wallclock_seconds, bool) or not isinstance(
                self.max_wallclock_seconds, int
            ):
                raise CompressTimePassContractError(
                    f"max_wallclock_seconds={self.max_wallclock_seconds!r} must "
                    f"be None or int (None = unbounded per CLAUDE.md §G compress-"
                    f"time is unbounded)"
                )
            if self.max_wallclock_seconds < 1:
                raise CompressTimePassContractError(
                    f"max_wallclock_seconds={self.max_wallclock_seconds} must "
                    f"be >= 1 (or None for unbounded)"
                )

        # rate_budget_bytes: None LEGAL; int must be >= 0
        if self.rate_budget_bytes is not None:
            if isinstance(self.rate_budget_bytes, bool) or not isinstance(
                self.rate_budget_bytes, int
            ):
                raise CompressTimePassContractError(
                    f"rate_budget_bytes={self.rate_budget_bytes!r} must be None "
                    f"or int"
                )
            if self.rate_budget_bytes < 0:
                raise CompressTimePassContractError(
                    f"rate_budget_bytes={self.rate_budget_bytes} must be >= 0"
                )

        # distortion_budget: None LEGAL; float must be >= 0
        if self.distortion_budget is not None:
            if isinstance(self.distortion_budget, bool) or not isinstance(
                self.distortion_budget, (int, float)
            ):
                raise CompressTimePassContractError(
                    f"distortion_budget={self.distortion_budget!r} must be None "
                    f"or float"
                )
            if self.distortion_budget < 0:
                raise CompressTimePassContractError(
                    f"distortion_budget={self.distortion_budget} must be >= 0"
                )

        # seed: None LEGAL; int must be >= 0 if provided
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise CompressTimePassContractError(
                    f"seed={self.seed!r} must be None or int"
                )
            if self.seed < 0:
                raise CompressTimePassContractError(
                    f"seed={self.seed} must be >= 0"
                )

        if self.merge_policy not in LEGAL_MERGE_POLICY:
            raise CompressTimePassContractError(
                f"merge_policy={self.merge_policy!r} not in "
                f"{sorted(LEGAL_MERGE_POLICY)}"
            )

        # Cross-field invariant: archive_build-time passes MUST be deterministic
        # (per Catalog #158). Non-deterministic archive bytes break byte-stable
        # replay + the no-op detector (Catalog #105/#139).
        if self.stage_phase == "archive_build" and not self.deterministic:
            raise CompressTimePassContractError(
                f"stage_phase='archive_build' requires deterministic=True (per "
                f"Catalog #158 deterministic_compiler discipline). Got "
                f"deterministic=False on pass id={self.id!r}."
            )

        # Cross-field invariant: passthrough passes emit exactly what they
        # consume (sentinel: empty intersection illegal for passthrough).
        if self.correction_kind == "passthrough" and self.emits and self.consumes:
            if self.emits != self.consumes:
                raise CompressTimePassContractError(
                    f"correction_kind='passthrough' requires emits == consumes; "
                    f"got emits={sorted(self.emits)!r} consumes="
                    f"{sorted(self.consumes)!r} on pass id={self.id!r}."
                )

        # ---- 4. 6-hook wire-in validation
        hook_validators = {
            "hook_sensitivity_contribution": (
                self.hook_sensitivity_contribution,
                LEGAL_HOOK_SENSITIVITY,
            ),
            "hook_pareto_constraint": (
                self.hook_pareto_constraint,
                LEGAL_HOOK_PARETO,
            ),
            "hook_bit_allocator_class": (
                self.hook_bit_allocator_class,
                LEGAL_HOOK_BIT_ALLOCATOR,
            ),
            "hook_autopilot_ranker": (
                self.hook_autopilot_ranker,
                LEGAL_HOOK_AUTOPILOT,
            ),
            "hook_continual_learning_anchor_kind": (
                self.hook_continual_learning_anchor_kind,
                LEGAL_HOOK_CONTINUAL_LEARNING,
            ),
        }
        for fname, (value, legal) in hook_validators.items():
            if value not in legal:
                raise CompressTimePassContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise CompressTimePassContractError(
                        f"{fname}=not_applicable_with_rationale requires a "
                        f"non-empty entry in "
                        f"hook_not_applicable_rationale[{fname!r}]"
                    )

        # Hook 6 (probe-disambiguator): None requires rationale; non-None must
        # look like a path/identifier.
        if self.hook_probe_disambiguator is None:
            rationale = self.hook_not_applicable_rationale.get(
                "hook_probe_disambiguator"
            )
            if not rationale or not rationale.strip():
                raise CompressTimePassContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if (
                not isinstance(self.hook_probe_disambiguator, str)
                or not self.hook_probe_disambiguator.strip()
            ):
                raise CompressTimePassContractError(
                    "hook_probe_disambiguator must be a non-empty path/"
                    "identifier or None"
                )

        # hook_not_applicable_rationale dict keys must match valid hook names
        legal_rationale_keys = {
            "hook_sensitivity_contribution",
            "hook_pareto_constraint",
            "hook_bit_allocator_class",
            "hook_autopilot_ranker",
            "hook_continual_learning_anchor_kind",
            "hook_probe_disambiguator",
        }
        for k in self.hook_not_applicable_rationale:
            if k not in legal_rationale_keys:
                raise CompressTimePassContractError(
                    f"hook_not_applicable_rationale has illegal key {k!r}; "
                    f"legal={sorted(legal_rationale_keys)}"
                )

        # ---- 5. Provenance soft-validation (None tolerated; non-empty if set)
        for fname, value in (
            ("lane_id", self.lane_id),
            ("design_memo", self.design_memo),
            ("canonical_vs_unique_decision", self.canonical_vs_unique_decision),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise CompressTimePassContractError(
                    f"{fname}={value!r} must be None or a non-empty string"
                )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (sorted keys for byte-stable downstream
        consumers + JSON round-trip safety).

        Frozensets serialize as sorted lists for byte-stable JSON output.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, frozenset):
                out[f.name] = sorted(v)
            elif isinstance(v, dict):
                out[f.name] = dict(v)
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressTimePassContract":
        """Reconstruct a contract from a dict (e.g. JSON round-trip).

        Lists (from JSON deserialization) are re-frozen into frozensets per
        the dataclass field types. Missing fields use the dataclass defaults
        so partial dicts (e.g. legacy persistence rows) still construct.
        """
        converted: dict[str, Any] = dict(data)
        for fname in ("consumes", "emits"):
            if fname in converted and isinstance(converted[fname], (list, set)):
                converted[fname] = frozenset(converted[fname])
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in converted.items() if k in known}
        return cls(**filtered)
