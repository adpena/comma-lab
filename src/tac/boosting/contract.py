# SPDX-License-Identifier: MIT
"""BoostStageContract — canonical schema for a single boost stage.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.3-§5.5, every boost stage MUST declare a frozen ``BoostStageContract``
that captures its identity, position in the cascade, what it consumes /
emits, its correction kind + resolution, and the production-hardening
invariants (deterministic, sensitivity_weighted, max_bytes_added).

Mirrors the canonical ``SubstrateContract`` from
``tac.substrate_registry.contract`` at the boost-stage surface — same
frozen-dataclass + field-level validator pattern, raises
``BoostStageContractError`` on any violation at construction time so
malformed stages fail at IMPORT time (not at dispatch time).

Per CLAUDE.md "Beauty, simplicity, and developer experience":
narrow API — one dataclass, one error class. The decorator + pipeline
modules consume this contract; consumers DO NOT subclass it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from tac.boosting.errors import BoostStageContractError

__all__ = [
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
    "BoostStageContract",
]


# ---------------------------------------------------------------------------
# Enum-like legal value sets (frozensets for O(1) membership; immutable).
# ---------------------------------------------------------------------------

LEGAL_CORRECTION_KIND: frozenset[str] = frozenset(
    {
        "additive",        # e.g. PR106 format0d 2-pass additive correction
        "multiplicative",  # e.g. per-pair gain modulation
        "gated",           # e.g. SegNet-class-conditioned correction
        "replace",         # e.g. per-pair decoder selection (no residual)
        "passthrough",     # e.g. raw_decoder seed stage (no correction)
    }
)

LEGAL_CORRECTION_RESOLUTION: frozenset[str] = frozenset(
    {
        "per_frame",   # one correction per RGB frame
        "per_pair",    # one correction per (frame_0, frame_1) pair (PR106 default)
        "per_pixel",   # per-pixel correction (most expressive; highest byte cost)
        "per_block",   # per-block within tensor (block-FP analog)
        "global",      # one correction shared across all 600 pairs (baseline)
    }
)

LEGAL_STAGE_PHASE: frozenset[str] = frozenset(
    {
        "compress",      # runs at compress time; produces archive bytes
        "archive_build", # runs at archive-pack time; assembles final bytes
        "inflate",       # runs at inflate time; reconstructs frames (scorer-free!)
        "post_process",  # runs after inflate; e.g. deterministic image-domain filter
    }
)

LEGAL_MERGE_POLICY: frozenset[str] = frozenset(
    {
        "last_writer_wins",  # downstream stage value overrides upstream
        "first_writer_wins", # upstream value preserved; downstream discarded
        "additive",          # values summed (only legal for numeric types)
        "concatenate",       # values appended (only legal for sequences/bytes)
        "explicit",          # caller must supply merge_callable
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
        "boosting_pareto_front_tracker_v1",
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
        "boosting_stage_outcomes_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, kw_only=True)
class BoostStageContract:
    """Canonical contract declared by every boost stage via ``@boost_stage``.

    Mirrors ``SubstrateContract`` at the boost-stage surface — one frozen
    dataclass, field-level validators, no inheritance.

    Field groups:
      1. Identity + cascade ordering (id, parent_stage_id, stage_phase)
      2. Wire contract (consumes, emits, correction_kind, correction_resolution)
      3. Production hardening (deterministic, scorer_free, sensitivity_weighted,
         max_bytes_added)
      4. 6-hook wire-in (Catalog #125)
      5. Provenance (lane_id, design_memo)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": every consumer (Pipeline,
    ParetoFrontTracker, persistence ledger) reads from this contract via
    explicit field access — no hidden state, no class hierarchy.
    """

    # 1. Identity + cascade ordering (4)
    id: str
    parent_stage_id: str | None = None
    stage_phase: str = "compress"
    description: str = ""

    # 2. Wire contract (4)
    consumes: frozenset[str] = field(default_factory=frozenset)
    emits: frozenset[str] = field(default_factory=frozenset)
    correction_kind: str = "additive"
    correction_resolution: str = "per_pair"

    # 3. Production hardening (5)
    deterministic: bool = True
    scorer_free: bool = True
    sensitivity_weighted: bool = False
    max_bytes_added: int | None = None
    merge_policy: str = "last_writer_wins"

    # 4. 6-hook wire-in (Catalog #125) — sensitivity / pareto / bit-allocator /
    # autopilot / continual-learning / probe-disambiguator
    hook_sensitivity_contribution: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_pareto_constraint: str = "rate_distortion_v1"
    hook_bit_allocator_class: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_autopilot_ranker: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_continual_learning_anchor_kind: str = "boosting_stage_outcomes_v1"
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
            raise BoostStageContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/"
            )
        if self.parent_stage_id is not None:
            if not isinstance(self.parent_stage_id, str) or not _ID_PATTERN.match(
                self.parent_stage_id
            ):
                raise BoostStageContractError(
                    f"parent_stage_id={self.parent_stage_id!r} must be None or "
                    "match /^[a-z][a-z0-9_]*$/"
                )
            if self.parent_stage_id == self.id:
                raise BoostStageContractError(
                    f"parent_stage_id={self.parent_stage_id!r} must differ from "
                    f"id={self.id!r} (a stage cannot be its own parent)"
                )
        if self.stage_phase not in LEGAL_STAGE_PHASE:
            raise BoostStageContractError(
                f"stage_phase={self.stage_phase!r} not in "
                f"{sorted(LEGAL_STAGE_PHASE)}"
            )

        # ---- 2. Wire contract validation
        # consumes / emits must be frozenset (immutability + hashability for
        # downstream sets-of-sets comparisons in the pipeline builder)
        for fname, value in (("consumes", self.consumes), ("emits", self.emits)):
            if not isinstance(value, frozenset):
                # Auto-convert tuple/set/list to frozenset BUT only if the
                # caller passed an iterable of strings. The dataclass declares
                # ``frozenset[str]`` so a list/set passed at construction is
                # tolerated by re-freezing via object.__setattr__.
                try:
                    converted = frozenset(value)
                except TypeError as exc:
                    raise BoostStageContractError(
                        f"{fname}={value!r} must be a frozenset/set/list of str"
                    ) from exc
                # frozen dataclass requires object.__setattr__ to mutate
                object.__setattr__(self, fname, converted)
                value = converted
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise BoostStageContractError(
                        f"{fname} contains non-string or empty entry: {item!r}"
                    )

        overlap = self.consumes & self.emits
        if overlap:
            raise BoostStageContractError(
                f"consumes ∩ emits is non-empty: {sorted(overlap)!r}. A stage "
                "that consumes a key cannot also emit the same key (use a "
                "versioned emit name like '<key>_v1' to express in-place "
                "refinement)."
            )

        if self.correction_kind not in LEGAL_CORRECTION_KIND:
            raise BoostStageContractError(
                f"correction_kind={self.correction_kind!r} not in "
                f"{sorted(LEGAL_CORRECTION_KIND)}"
            )
        if self.correction_resolution not in LEGAL_CORRECTION_RESOLUTION:
            raise BoostStageContractError(
                f"correction_resolution={self.correction_resolution!r} not in "
                f"{sorted(LEGAL_CORRECTION_RESOLUTION)}"
            )

        # ---- 3. Production hardening
        if not isinstance(self.deterministic, bool):
            raise BoostStageContractError("deterministic must be bool")
        if not isinstance(self.scorer_free, bool):
            raise BoostStageContractError("scorer_free must be bool")
        if not isinstance(self.sensitivity_weighted, bool):
            raise BoostStageContractError("sensitivity_weighted must be bool")
        if self.max_bytes_added is not None:
            if not isinstance(self.max_bytes_added, int):
                raise BoostStageContractError(
                    f"max_bytes_added={self.max_bytes_added!r} must be int or None"
                )
            if self.max_bytes_added < 0:
                raise BoostStageContractError(
                    f"max_bytes_added={self.max_bytes_added} must be >= 0"
                )
        if self.merge_policy not in LEGAL_MERGE_POLICY:
            raise BoostStageContractError(
                f"merge_policy={self.merge_policy!r} not in "
                f"{sorted(LEGAL_MERGE_POLICY)}"
            )

        # Cross-field invariant: inflate-time stages MUST be scorer-free
        # (per CLAUDE.md "Strict scorer rule" non-negotiable). Surfaces at
        # decoration time so the violation never reaches a paid GPU job.
        if self.stage_phase == "inflate" and not self.scorer_free:
            raise BoostStageContractError(
                f"stage_phase='inflate' requires scorer_free=True (per CLAUDE.md "
                f"'Strict scorer rule' non-negotiable). Got scorer_free=False on "
                f"stage id={self.id!r}."
            )

        # Cross-field invariant: archive_build-time stages MUST be deterministic
        # (per Catalog #158). Non-deterministic archive bytes break byte-stable
        # replay + the no-op detector (Catalog #105/#139).
        if self.stage_phase == "archive_build" and not self.deterministic:
            raise BoostStageContractError(
                f"stage_phase='archive_build' requires deterministic=True (per "
                f"Catalog #158 deterministic_compiler discipline). Got "
                f"deterministic=False on stage id={self.id!r}."
            )

        # Cross-field invariant: passthrough stages emit exactly what they
        # consume (sentinel: empty intersection is illegal for passthrough).
        if (
            self.correction_kind == "passthrough"
            and self.emits
            and self.consumes
            and self.emits != self.consumes
        ):
            raise BoostStageContractError(
                f"correction_kind='passthrough' requires emits == consumes; "
                f"got emits={sorted(self.emits)!r} consumes="
                f"{sorted(self.consumes)!r} on stage id={self.id!r}."
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
                raise BoostStageContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise BoostStageContractError(
                        f"{fname}=not_applicable_with_rationale requires a "
                        f"non-empty entry in "
                        f"hook_not_applicable_rationale[{fname!r}]"
                    )

        # Hook 6 (probe-disambiguator): None requires rationale; non-None
        # must look like a path/identifier.
        if self.hook_probe_disambiguator is None:
            rationale = self.hook_not_applicable_rationale.get(
                "hook_probe_disambiguator"
            )
            if not rationale or not rationale.strip():
                raise BoostStageContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if (
                not isinstance(self.hook_probe_disambiguator, str)
                or not self.hook_probe_disambiguator.strip()
            ):
                raise BoostStageContractError(
                    "hook_probe_disambiguator must be a non-empty path/identifier "
                    "or None"
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
                raise BoostStageContractError(
                    f"hook_not_applicable_rationale has illegal key {k!r}; "
                    f"legal={sorted(legal_rationale_keys)}"
                )

        # ---- 5. Provenance soft-validation (None tolerated; non-empty if set)
        for fname, value in (
            ("lane_id", self.lane_id),
            ("design_memo", self.design_memo),
            ("canonical_vs_unique_decision", self.canonical_vs_unique_decision),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise BoostStageContractError(
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
    def from_dict(cls, data: dict[str, Any]) -> BoostStageContract:
        """Reconstruct a contract from a dict (e.g. JSON round-trip).

        Lists (from JSON deserialization) are re-frozen into frozensets per
        the dataclass field types. Missing fields use the dataclass defaults
        so partial dicts (e.g. legacy persistence rows) still construct.
        """
        # Re-freeze list-style frozensets if present
        converted: dict[str, Any] = dict(data)
        for fname in ("consumes", "emits"):
            if fname in converted and isinstance(converted[fname], (list, set)):
                converted[fname] = frozenset(converted[fname])
        # Filter to known field names (drop extras)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in converted.items() if k in known}
        return cls(**filtered)
