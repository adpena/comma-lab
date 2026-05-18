# SPDX-License-Identifier: MIT
"""InflateTimePostProcessingContract — canonical schema for a single
inflate-time post-processing pass.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5.3-§5.5 + §G inflate-time techniques: every inflate-time pass MUST
declare a frozen ``InflateTimePostProcessingContract`` that captures its
identity, position in the pipeline, what frames it consumes / emits, its
correction kind + resolution, and production-hardening invariants
(deterministic, scorer_free, max_wallclock_seconds, archive_bytes_added=0,
score_axis_affected).

Mirrors ``CompressTimePassContract`` from ``tac.compress_time_optimization``
at the inflate-time-pass surface — same frozen-dataclass + field-level
validator pattern, raises ``InflateTimePassContractError`` (or a more
specific subclass) on any violation at construction time so malformed
passes fail at IMPORT time (not at dispatch time).

UNIQUE fields per this namespace (per PV-7 canonical-vs-unique decision per
layer):

  - ``stage_phase: Literal["inflate"]`` (REQUIRED, default 'inflate'; raises
    ``CompressPhaseForbiddenError`` for any other value — sister namespace
    ``tac.compress_time_optimization`` is the canonical home for compress
    and archive_build phases).

  - ``max_wallclock_seconds: float`` (REQUIRED — None raises
    ``WallclockBudgetRequiredError``; inflate-time has the 30-minute T4
    ceiling per spec §G).

  - ``inflate_compute_budget_seconds: float`` (≤ 1800.0; default 1800.0 —
    the 30-minute T4 ceiling).

  - ``applies_to_frames: Literal["all", "pairs_only", "odd_only",
    "even_only"]`` (declares which frames the pass operates on; the
    pipeline composer uses this to detect ambiguous emit overlaps).

  - ``archive_bytes_added: int = 0`` (MUST be 0; raises
    ``ArchiveBytesViolation`` if > 0 — inflate-time passes don't change
    archive bytes; that's compress-time territory).

  - ``score_axis_affected: tuple[str, ...]`` (subset of ``("seg", "pose")``;
    declares which scorer axes the technique targets).

  - ``requires_scorer_surrogate: bool`` (True if the pass uses a
    Hinton-distilled surrogate at inflate time per Catalog #527; surrogates
    are CPU-only + small + baked into the archive bytes).

  - ``requires_cpu_only: bool = True`` (most inflate-time techniques are
    CPU-only per contest inflate runtime; False is legal but operator must
    document why in canonical_vs_unique_decision).

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow API —
one dataclass, one error class hierarchy. The decorator + pipeline modules
consume this contract; consumers DO NOT subclass it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from tac.inflate_time_post_processing.errors import (
    ArchiveBytesViolation,
    CompressPhaseForbiddenError,
    InflateTimePassContractError,
    ScorerAccessForbiddenError,
    WallclockBudgetRequiredError,
)

__all__ = [
    "InflateTimePostProcessingContract",
    "LEGAL_APPLIES_TO_FRAMES",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_MERGE_POLICY",
    "LEGAL_SCORE_AXIS",
    "LEGAL_STAGE_PHASE",
    "MAX_INFLATE_COMPUTE_BUDGET_SECONDS",
    "NOT_APPLICABLE_WITH_RATIONALE",
]


# ---------------------------------------------------------------------------
# Hard ceiling: 30-minute T4 wall budget per spec §G.
# ---------------------------------------------------------------------------

MAX_INFLATE_COMPUTE_BUDGET_SECONDS: float = 1800.0


# ---------------------------------------------------------------------------
# Enum-like legal value sets (frozensets for O(1) membership; immutable).
# ---------------------------------------------------------------------------

LEGAL_CORRECTION_KIND: frozenset[str] = frozenset(
    {
        "denoise",            # per-frame noise removal (bilateral / NLM / learned)
        "sharpen",            # per-frame edge enhancement
        "smooth",             # temporal/spatial smoothing
        "upscale",            # resolution upscale (avoid eval_roundtrip resize loss)
        "select",             # multi-variant selection per deterministic rule
        "refine",             # multi-pass refinement with surrogate
        "transform",          # deterministic seed/transform stage (renames a key)
        "passthrough",        # seed/no-op stage (emits == consumes by invariant)
    }
)

LEGAL_CORRECTION_RESOLUTION: frozenset[str] = frozenset(
    {
        "per_frame",
        "per_pair",
        "per_pixel",
        "per_block",
        "per_stream",
        "global",
    }
)

# This namespace is INFLATE-TIME ONLY. ``compress`` + ``archive_build`` are
# FORBIDDEN at decoration (raise CompressPhaseForbiddenError pointing to the
# sister namespace tac.compress_time_optimization).
LEGAL_STAGE_PHASE: frozenset[str] = frozenset({"inflate"})

# Forbidden phase tokens that raise CompressPhaseForbiddenError at decoration.
_FORBIDDEN_STAGE_PHASES: frozenset[str] = frozenset(
    {"compress", "archive_build", "training"}
)

LEGAL_APPLIES_TO_FRAMES: frozenset[str] = frozenset(
    {
        "all",         # every decoded frame
        "pairs_only",  # only the per-pair (frame_0, frame_1) pairs the scorer evaluates
        "odd_only",    # odd-indexed frames (e.g. frame_2k+1)
        "even_only",   # even-indexed frames (e.g. frame_2k)
    }
)

LEGAL_SCORE_AXIS: frozenset[str] = frozenset({"seg", "pose"})

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
        "scorer_surrogate_axis_weights_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_PARETO: frozenset[str] = frozenset(
    {
        "rate_distortion_v1",          # rare for inflate (archive bytes are frozen)
        "inflate_wallclock_envelope_v1",
        "frame_quality_pareto_v1",
        "cost_band_envelope_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_BIT_ALLOCATOR: frozenset[str] = frozenset(
    {
        # Inflate-time rarely allocates bits (archive frozen at compress); most
        # passes are not_applicable_with_rationale. Surrogate-driven variant
        # selection is the exception.
        "scorer_surrogate_variant_selector",
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
        "inflate_time_post_processing_pass_outcomes_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, kw_only=True)
class InflateTimePostProcessingContract:
    """Canonical contract declared by every inflate-time post-processing
    pass via ``@inflate_time_post_filter``.

    Mirrors ``CompressTimePassContract`` at the inflate-time-pass surface —
    one frozen dataclass, field-level validators, no inheritance.

    Field groups:
      1. Identity + pipeline ordering (id, parent_pass_id, stage_phase,
         stage_order, description)
      2. Wire contract (consumes, emits, correction_kind,
         correction_resolution, applies_to_frames)
      3. Production hardening (deterministic, scorer_free,
         max_wallclock_seconds [REQUIRED], inflate_compute_budget_seconds,
         archive_bytes_added=0, score_axis_affected,
         requires_scorer_surrogate, requires_cpu_only, seed, merge_policy)
      4. 6-hook wire-in (Catalog #125)
      5. Provenance (lane_id, design_memo, canonical_vs_unique_decision)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": every consumer
    (Pipeline, persistence ledger, autopilot ranker) reads from this
    contract via explicit field access — no hidden state, no class
    hierarchy. The namespace does NOT share CompressTimePassContract:
    tac.compress_time_optimization and tac.inflate_time_post_processing
    are STRUCTURALLY INDEPENDENT (per PV-7 + Catalog #290).
    """

    # 1. Identity + pipeline ordering (5)
    id: str
    parent_pass_id: str | None = None
    stage_phase: str = "inflate"
    stage_order: int | None = None
    description: str = ""

    # 2. Wire contract (5)
    consumes: frozenset[str] = field(default_factory=frozenset)
    emits: frozenset[str] = field(default_factory=frozenset)
    correction_kind: str = "denoise"
    correction_resolution: str = "per_frame"
    applies_to_frames: str = "all"

    # 3. Production hardening (10)
    deterministic: bool = True
    scorer_free: bool = True
    # REQUIRED (no None): inflate-time has a 30-min ceiling per spec §G.
    max_wallclock_seconds: float | None = None
    inflate_compute_budget_seconds: float = MAX_INFLATE_COMPUTE_BUDGET_SECONDS
    archive_bytes_added: int = 0
    score_axis_affected: tuple[str, ...] = ()
    requires_scorer_surrogate: bool = False
    requires_cpu_only: bool = True
    seed: int | None = None
    merge_policy: str = "last_writer_wins"

    # 4. 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_pareto_constraint: str = "inflate_wallclock_envelope_v1"
    hook_bit_allocator_class: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_autopilot_ranker: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_continual_learning_anchor_kind: str = (
        "inflate_time_post_processing_pass_outcomes_v1"
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
            raise InflateTimePassContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/"
            )
        if self.parent_pass_id is not None:
            if not isinstance(self.parent_pass_id, str) or not _ID_PATTERN.match(
                self.parent_pass_id
            ):
                raise InflateTimePassContractError(
                    f"parent_pass_id={self.parent_pass_id!r} must be None or "
                    "match /^[a-z][a-z0-9_]*$/"
                )
            if self.parent_pass_id == self.id:
                raise InflateTimePassContractError(
                    f"parent_pass_id={self.parent_pass_id!r} must differ from "
                    f"id={self.id!r} (a pass cannot be its own parent)"
                )

        # Compress / archive_build phase is FORBIDDEN in this namespace.
        if self.stage_phase in _FORBIDDEN_STAGE_PHASES:
            raise CompressPhaseForbiddenError(
                f"stage_phase={self.stage_phase!r} is FORBIDDEN in the "
                f"tac.inflate_time_post_processing namespace. This namespace "
                f"is INFLATE-TIME ONLY. Use the sister namespace "
                f"tac.compress_time_optimization for compress-time and "
                f"archive_build stages (per spec §5.2)."
            )
        if self.stage_phase not in LEGAL_STAGE_PHASE:
            raise InflateTimePassContractError(
                f"stage_phase={self.stage_phase!r} not in "
                f"{sorted(LEGAL_STAGE_PHASE)}"
            )

        if self.stage_order is not None:
            if not isinstance(self.stage_order, int) or isinstance(
                self.stage_order, bool
            ):
                raise InflateTimePassContractError(
                    f"stage_order={self.stage_order!r} must be None or int"
                )
            if self.stage_order < 0:
                raise InflateTimePassContractError(
                    f"stage_order={self.stage_order} must be >= 0"
                )

        # ---- 2. Wire contract validation
        for fname, value in (("consumes", self.consumes), ("emits", self.emits)):
            if not isinstance(value, frozenset):
                try:
                    converted = frozenset(value)
                except TypeError as exc:
                    raise InflateTimePassContractError(
                        f"{fname}={value!r} must be a frozenset/set/list of str"
                    ) from exc
                object.__setattr__(self, fname, converted)
                value = converted
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise InflateTimePassContractError(
                        f"{fname} contains non-string or empty entry: {item!r}"
                    )

        # Overlap check: forbidden for non-passthrough kinds.
        overlap = self.consumes & self.emits
        if overlap and self.correction_kind != "passthrough":
            raise InflateTimePassContractError(
                f"consumes ∩ emits is non-empty: {sorted(overlap)!r}. A pass "
                "that consumes a key cannot also emit the same key (use a "
                "versioned emit name like '<key>_v1' to express in-place "
                "refinement)."
            )

        if self.correction_kind not in LEGAL_CORRECTION_KIND:
            raise InflateTimePassContractError(
                f"correction_kind={self.correction_kind!r} not in "
                f"{sorted(LEGAL_CORRECTION_KIND)}"
            )
        if self.correction_resolution not in LEGAL_CORRECTION_RESOLUTION:
            raise InflateTimePassContractError(
                f"correction_resolution={self.correction_resolution!r} not in "
                f"{sorted(LEGAL_CORRECTION_RESOLUTION)}"
            )

        if self.applies_to_frames not in LEGAL_APPLIES_TO_FRAMES:
            raise InflateTimePassContractError(
                f"applies_to_frames={self.applies_to_frames!r} not in "
                f"{sorted(LEGAL_APPLIES_TO_FRAMES)}"
            )

        # ---- 3. Production hardening
        for fname, value in (
            ("deterministic", self.deterministic),
            ("scorer_free", self.scorer_free),
            ("requires_scorer_surrogate", self.requires_scorer_surrogate),
            ("requires_cpu_only", self.requires_cpu_only),
        ):
            if not isinstance(value, bool):
                raise InflateTimePassContractError(f"{fname} must be bool")

        # Scorer access is FORBIDDEN per CLAUDE.md "Strict scorer rule".
        # scorer_free=False raises so the operator cannot accidentally bake
        # the contest scorer into inflate.py. A Hinton-distilled surrogate
        # (requires_scorer_surrogate=True) is the LEGAL alternative.
        if not self.scorer_free:
            raise ScorerAccessForbiddenError(
                f"scorer_free=False is FORBIDDEN on pass id={self.id!r}. "
                f"Per CLAUDE.md 'Strict scorer rule' non-negotiable: NO "
                f"loading PoseNet or SegNet at inflate time (~73 MB rate "
                f"hit). If the pass needs a scorer-like signal, set "
                f"requires_scorer_surrogate=True and use a CPU-trained "
                f"Hinton-distilled surrogate per Catalog #527 (the "
                f"surrogate's weights ship as part of the archive bytes "
                f"via the COMPRESS-time grammar, not loaded ad-hoc at "
                f"inflate)."
            )

        # max_wallclock_seconds: REQUIRED per spec §G; None raises.
        if self.max_wallclock_seconds is None:
            raise WallclockBudgetRequiredError(
                f"max_wallclock_seconds is REQUIRED on inflate-time pass "
                f"id={self.id!r}. Inflate-time has a 30-min hard ceiling on "
                f"T4 per spec §G; passes MUST commit to a wallclock budget "
                f"at decoration so the pipeline composer can sum per-pass "
                f"budgets and refuse compositions that would exceed "
                f"{MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s at build time."
            )
        if isinstance(self.max_wallclock_seconds, bool) or not isinstance(
            self.max_wallclock_seconds, (int, float)
        ):
            raise InflateTimePassContractError(
                f"max_wallclock_seconds={self.max_wallclock_seconds!r} must "
                f"be float or int (REQUIRED; None forbidden per spec §G)"
            )
        if self.max_wallclock_seconds <= 0:
            raise InflateTimePassContractError(
                f"max_wallclock_seconds={self.max_wallclock_seconds} must be > 0"
            )
        if self.max_wallclock_seconds > MAX_INFLATE_COMPUTE_BUDGET_SECONDS:
            raise InflateTimePassContractError(
                f"max_wallclock_seconds={self.max_wallclock_seconds} exceeds the "
                f"30-min T4 ceiling ({MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s) per "
                f"spec §G. Either lower the pass's per-frame compute OR split "
                f"into multiple passes that each commit to a sub-budget."
            )

        # inflate_compute_budget_seconds: cap at the 30-min ceiling.
        if isinstance(
            self.inflate_compute_budget_seconds, bool
        ) or not isinstance(self.inflate_compute_budget_seconds, (int, float)):
            raise InflateTimePassContractError(
                f"inflate_compute_budget_seconds="
                f"{self.inflate_compute_budget_seconds!r} must be float or int"
            )
        if self.inflate_compute_budget_seconds <= 0:
            raise InflateTimePassContractError(
                f"inflate_compute_budget_seconds="
                f"{self.inflate_compute_budget_seconds} must be > 0"
            )
        if (
            self.inflate_compute_budget_seconds
            > MAX_INFLATE_COMPUTE_BUDGET_SECONDS
        ):
            raise InflateTimePassContractError(
                f"inflate_compute_budget_seconds="
                f"{self.inflate_compute_budget_seconds} exceeds the 30-min T4 "
                f"ceiling ({MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s) per spec §G."
            )
        # Per-pass budget cannot exceed the contract's declared compute budget.
        if self.max_wallclock_seconds > self.inflate_compute_budget_seconds:
            raise InflateTimePassContractError(
                f"max_wallclock_seconds={self.max_wallclock_seconds} exceeds "
                f"inflate_compute_budget_seconds="
                f"{self.inflate_compute_budget_seconds} on pass "
                f"id={self.id!r}. Per-pass budget cannot exceed the contract's "
                f"declared compute budget."
            )

        # archive_bytes_added: MUST be 0 (this is INFLATE-TIME post-processing).
        if isinstance(self.archive_bytes_added, bool) or not isinstance(
            self.archive_bytes_added, int
        ):
            raise InflateTimePassContractError(
                f"archive_bytes_added={self.archive_bytes_added!r} must be int"
            )
        if self.archive_bytes_added != 0:
            raise ArchiveBytesViolation(
                f"archive_bytes_added={self.archive_bytes_added} is FORBIDDEN "
                f"on inflate-time pass id={self.id!r}. Inflate-time "
                f"post-processing operates on FRAMES (after the renderer "
                f"decodes them) — it does NOT mutate archive BYTES. Adding "
                f"archive bytes is by definition a COMPRESS-time technique "
                f"and belongs in tac.compress_time_optimization."
            )

        # score_axis_affected: tuple of LEGAL_SCORE_AXIS members.
        if not isinstance(self.score_axis_affected, tuple):
            try:
                converted = tuple(self.score_axis_affected)
            except TypeError as exc:
                raise InflateTimePassContractError(
                    f"score_axis_affected={self.score_axis_affected!r} must "
                    f"be a tuple of str"
                ) from exc
            object.__setattr__(self, "score_axis_affected", converted)
        for axis in self.score_axis_affected:
            if axis not in LEGAL_SCORE_AXIS:
                raise InflateTimePassContractError(
                    f"score_axis_affected entry {axis!r} not in "
                    f"{sorted(LEGAL_SCORE_AXIS)}"
                )
        if len(self.score_axis_affected) != len(set(self.score_axis_affected)):
            raise InflateTimePassContractError(
                f"score_axis_affected has duplicates: "
                f"{self.score_axis_affected!r}"
            )

        # seed: None LEGAL; int must be >= 0 if provided
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise InflateTimePassContractError(
                    f"seed={self.seed!r} must be None or int"
                )
            if self.seed < 0:
                raise InflateTimePassContractError(
                    f"seed={self.seed} must be >= 0"
                )

        if self.merge_policy not in LEGAL_MERGE_POLICY:
            raise InflateTimePassContractError(
                f"merge_policy={self.merge_policy!r} not in "
                f"{sorted(LEGAL_MERGE_POLICY)}"
            )

        # Cross-field invariant: deterministic=True is REQUIRED for inflate
        # (per Catalog #158 deterministic-compiler discipline). Non-deterministic
        # inflate breaks byte-stable replay + invalidates contest scorer eval.
        if not self.deterministic:
            raise InflateTimePassContractError(
                f"deterministic=False is FORBIDDEN on inflate-time pass "
                f"id={self.id!r}. Per Catalog #158 deterministic-compiler "
                f"discipline + CLAUDE.md 'Bit-level deconstruction' "
                f"non-negotiable: inflate MUST produce byte-identical frames "
                f"deterministically per run, OR the contest scorer's eval is "
                f"non-reproducible."
            )

        # Cross-field invariant: passthrough passes emit exactly what they
        # consume (sentinel: empty intersection illegal for passthrough).
        if self.correction_kind == "passthrough" and self.emits and self.consumes:
            if self.emits != self.consumes:
                raise InflateTimePassContractError(
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
                raise InflateTimePassContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise InflateTimePassContractError(
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
                raise InflateTimePassContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if (
                not isinstance(self.hook_probe_disambiguator, str)
                or not self.hook_probe_disambiguator.strip()
            ):
                raise InflateTimePassContractError(
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
                raise InflateTimePassContractError(
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
                raise InflateTimePassContractError(
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
            elif isinstance(v, tuple):
                out[f.name] = list(v)
            elif isinstance(v, dict):
                out[f.name] = dict(v)
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "InflateTimePostProcessingContract":
        """Reconstruct a contract from a dict (e.g. JSON round-trip).

        Lists (from JSON deserialization) are re-frozen into frozensets per
        the dataclass field types. Missing fields use the dataclass defaults
        so partial dicts (e.g. legacy persistence rows) still construct.
        """
        converted: dict[str, Any] = dict(data)
        for fname in ("consumes", "emits"):
            if fname in converted and isinstance(converted[fname], (list, set)):
                converted[fname] = frozenset(converted[fname])
        if "score_axis_affected" in converted and isinstance(
            converted["score_axis_affected"], list
        ):
            converted["score_axis_affected"] = tuple(
                converted["score_axis_affected"]
            )
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in converted.items() if k in known}
        return cls(**filtered)
