# SPDX-License-Identifier: MIT
"""SideInfoBakerContract — canonical schema for a single side-information
baker pass.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J side-information / pre-processing / per-pair input conditioning + the
parent-prompt 5-builder enumeration: every side-information baker MUST
declare a frozen ``SideInfoBakerContract`` that captures its identity,
position in the pipeline, what it consumes / emits, its side-information
source, the BOTH compress-time AND inflate-time byte budgets it consumes,
and the Wyner-Ziv reproducibility invariants.

Mirrors the canonical ``BoostStageContract`` from ``tac.boosting.contract``
and ``CompressTimePassContract`` from
``tac.compress_time_optimization.contract`` at the side-info-baker surface
— same frozen-dataclass + field-level validator pattern, raises
``SideInfoBakerContractError`` on any violation at construction time so
malformed bakers fail at IMPORT time (not at dispatch time).

Fields UNIQUE to this namespace per PV-7 (canonical-vs-unique decision
per layer):
  - ``side_info_source: str`` — one of 6 values (scorer_weights /
    comma2k19_distilled / imagenet_statistics / dashcam_domain /
    wyner_ziv_residual / custom).
  - ``side_info_reproducible: bool`` — REQUIRED True per the Wyner-Ziv +
    contest-reproducibility contract. Bakers that set it False raise
    NonReproducibleSideInfoViolation at decoration.
  - ``archive_bytes_added: int >= 0`` — bytes that show up in the contest
    archive (Wyner-Ziv residuals add bytes; shared priors typically don't).
  - ``inflate_runtime_bytes_added: int >= 0`` — bytes that get baked into
    inflate.py CONSTANTS (counts against HNeRV parity L4 ≤ 100 LOC inflate
    budget).
  - ``requires_canonical_comma2k19_cache: bool`` — when True, the baker
    distills priors from Comma2k19 chunks; CANONICAL-COMMA2K19-CACHE-REQUIRED
    raised at decoration if the helper module is unimportable.
  - ``wyner_ziv_correlation_estimate: float | None`` — predicted
    I(X;Y)/H(X) ratio; informs cathedral autopilot ranking. None when the
    baker emits a shared-prior-only artifact with no Wyner-Ziv residual
    role (e.g. ImageNetStatisticsPrior).
  - ``stage_phase: Literal["compress", "inflate", "both"]`` — side info
    can apply at compress (baking) and/or inflate (consumption); BOTH
    is the canonical case for shared-prior bakers.

Per CLAUDE.md "Beauty, simplicity, and developer experience":
narrow API — one dataclass, one error class. The decorator + pipeline
modules consume this contract; consumers DO NOT subclass it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

from tac.side_information.errors import (
    CanonicalComma2k19CacheRequiredViolation,
    NonReproducibleSideInfoViolation,
    SideInfoBakerContractError,
    WynerZivCorrelationInvalidError,
)

__all__ = [
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_MERGE_POLICY",
    "LEGAL_SIDE_INFO_SOURCE",
    "LEGAL_STAGE_PHASE",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "SideInfoBakerContract",
]


# ---------------------------------------------------------------------------
# Enum-like legal value sets (frozensets for O(1) membership; immutable).
# ---------------------------------------------------------------------------

LEGAL_SIDE_INFO_SOURCE: frozenset[str] = frozenset(
    {
        # Frozen contest scorer weights (PoseNet + SegNet) used as a shared
        # prior (Atick-Redlich cooperative-receiver framing).
        "scorer_weights",
        # Palette distilled from MIT-licensed Comma2k19 dataset at compress
        # time; shipped as inflate.py constant (Catalog #213 + #146).
        "comma2k19_distilled",
        # Public ImageNet/Kinetics statistics (means, variances, etc.).
        "imagenet_statistics",
        # Public dashcam-domain priors (road texture, lane statistics).
        "dashcam_domain",
        # Wyner-Ziv classical residual encoding X - f(Y) where f is
        # computable by the decoder from side info Y.
        "wyner_ziv_residual",
        # Other publicly-reproducible side info (operator-attested).
        "custom",
    }
)

LEGAL_CORRECTION_KIND: frozenset[str] = frozenset(
    {
        # Bake a precomputed prior at compress; consume it at inflate.
        "shared_prior_bake",
        # Encode residual X - f(Y) against a shared-prior function f(Y).
        "wyner_ziv_residual_encode",
        # Distill a small palette/dictionary from public dataset(s).
        "palette_distillation",
        # Extract per-pair features (e.g. optical flow, ego-motion).
        "feature_extraction",
        # Compute per-pair conditional entropy lower bound.
        "entropy_estimation",
        # Pure compute-only transform (deterministic, idempotent).
        "transform",
        # Seed / passthrough stage.
        "passthrough",
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
        "per_class",
        "global",
    }
)

# Side information can apply at compress time (baking priors / encoding
# residuals), at inflate time (consuming baked constants), or BOTH (the
# canonical shared-prior bake-and-consume cycle).
LEGAL_STAGE_PHASE: frozenset[str] = frozenset(
    {
        "compress",  # runs at compress only (e.g. residual encode)
        "inflate",   # runs at inflate only (e.g. constant lookup)
        "both",      # runs at both (e.g. shared-prior bake + consume)
    }
)

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
        "scorer_weights_shared_prior_v1",  # unique: side-info uses scorer
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_PARETO: frozenset[str] = frozenset(
    {
        "rate_distortion_v1",
        "cost_band_envelope_v1",
        "side_info_pareto_front_tracker_v1",
        "wyner_ziv_rate_distortion_v1",  # unique: Wyner-Ziv R(D|Y) bound
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
        "wyner_ziv_residual_allocator",  # unique: bits-for-residual only
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
        "side_information_baker_outcomes_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _canonical_comma2k19_cache_importable() -> bool:
    """Return True if the canonical Comma2k19 cache helper is importable.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    against" + Catalog #213: bakers that distill priors from Comma2k19
    chunks MUST route through this canonical helper. The decorator checks
    this at construction time so a baker with
    ``requires_canonical_comma2k19_cache=True`` cannot be registered in a
    runtime env where the helper is missing (e.g. partial checkout).

    Returns True when the import succeeds; False on ImportError /
    ModuleNotFoundError. Any other exception propagates so import bugs
    in the helper surface immediately.
    """
    try:
        # Local import so the side_information namespace does NOT take a
        # hard dependency on pretrained_driving_prior at module load.
        from tac.substrates.pretrained_driving_prior.local_chunk_cache import (  # noqa: F401
            Comma2k19LocalCache,
        )
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@dataclass(frozen=True, kw_only=True)
class SideInfoBakerContract:
    """Canonical contract declared by every side-information baker via
    ``@side_info_baker``.

    Mirrors ``BoostStageContract`` (tac.boosting) and
    ``CompressTimePassContract`` (tac.compress_time_optimization) at the
    side-info-baker surface — one frozen dataclass, field-level validators,
    no inheritance.

    Field groups:
      1. Identity + pipeline ordering (id, parent_baker_id, stage_phase,
         stage_order, description)
      2. Wire contract (consumes, emits, correction_kind,
         correction_resolution)
      3. Side-information invariants (side_info_source,
         side_info_reproducible, requires_canonical_comma2k19_cache,
         wyner_ziv_correlation_estimate)
      4. Production hardening (deterministic, scorer_free,
         archive_bytes_added, inflate_runtime_bytes_added, seed,
         merge_policy)
      5. 6-hook wire-in (Catalog #125)
      6. Provenance (lane_id, design_memo, canonical_vs_unique_decision)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": every consumer
    (Pipeline, persistence ledger, autopilot ranker) reads from this
    contract via explicit field access — no hidden state, no class
    hierarchy. The namespace does NOT share BoostStageContract or
    CompressTimePassContract: each namespace is STRUCTURALLY INDEPENDENT
    (per PV-7 + Catalog #290 canonical-vs-unique decision per layer).
    """

    # 1. Identity + pipeline ordering (5)
    id: str
    parent_baker_id: str | None = None
    stage_phase: str = "both"
    stage_order: int | None = None
    description: str = ""

    # 2. Wire contract (4)
    consumes: frozenset[str] = field(default_factory=frozenset)
    emits: frozenset[str] = field(default_factory=frozenset)
    correction_kind: str = "shared_prior_bake"
    correction_resolution: str = "per_pair"

    # 3. Side-information invariants (4) — UNIQUE to this namespace
    side_info_source: str = "custom"
    side_info_reproducible: bool = True
    requires_canonical_comma2k19_cache: bool = False
    wyner_ziv_correlation_estimate: float | None = None

    # 4. Production hardening (6)
    deterministic: bool = True
    scorer_free: bool = True
    archive_bytes_added: int = 0
    inflate_runtime_bytes_added: int = 0
    seed: int | None = None
    merge_policy: str = "last_writer_wins"

    # 5. 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_pareto_constraint: str = "rate_distortion_v1"
    hook_bit_allocator_class: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_autopilot_ranker: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_continual_learning_anchor_kind: str = (
        "side_information_baker_outcomes_v1"
    )
    hook_probe_disambiguator: str | None = None
    hook_not_applicable_rationale: dict[str, str] = field(default_factory=dict)

    # 6. Provenance (3)
    lane_id: str | None = None
    design_memo: str | None = None
    canonical_vs_unique_decision: str | None = None

    # ------------------------------------------------------------------
    # Validation (runs in __post_init__)
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # ---- 1. Identity validation
        if not isinstance(self.id, str) or not _ID_PATTERN.match(self.id):
            raise SideInfoBakerContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/"
            )
        if self.parent_baker_id is not None:
            if not isinstance(
                self.parent_baker_id, str
            ) or not _ID_PATTERN.match(self.parent_baker_id):
                raise SideInfoBakerContractError(
                    f"parent_baker_id={self.parent_baker_id!r} must be None "
                    "or match /^[a-z][a-z0-9_]*$/"
                )
            if self.parent_baker_id == self.id:
                raise SideInfoBakerContractError(
                    f"parent_baker_id={self.parent_baker_id!r} must differ "
                    f"from id={self.id!r} (a baker cannot be its own parent)"
                )

        if self.stage_phase not in LEGAL_STAGE_PHASE:
            raise SideInfoBakerContractError(
                f"stage_phase={self.stage_phase!r} not in "
                f"{sorted(LEGAL_STAGE_PHASE)}"
            )

        if self.stage_order is not None:
            if not isinstance(self.stage_order, int) or isinstance(
                self.stage_order, bool
            ):
                raise SideInfoBakerContractError(
                    f"stage_order={self.stage_order!r} must be None or int"
                )
            if self.stage_order < 0:
                raise SideInfoBakerContractError(
                    f"stage_order={self.stage_order} must be >= 0"
                )

        # ---- 2. Wire contract validation
        for fname, value in (("consumes", self.consumes), ("emits", self.emits)):
            if not isinstance(value, frozenset):
                try:
                    converted = frozenset(value)
                except TypeError as exc:
                    raise SideInfoBakerContractError(
                        f"{fname}={value!r} must be a frozenset/set/list of str"
                    ) from exc
                object.__setattr__(self, fname, converted)
                value = converted
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise SideInfoBakerContractError(
                        f"{fname} contains non-string or empty entry: {item!r}"
                    )

        # Overlap check: forbidden for non-passthrough kinds. Passthrough
        # bakers (correction_kind="passthrough") REQUIRE emits == consumes
        # so the overlap is structurally legal.
        overlap = self.consumes & self.emits
        if overlap and self.correction_kind != "passthrough":
            raise SideInfoBakerContractError(
                f"consumes ∩ emits is non-empty: {sorted(overlap)!r}. A baker "
                "that consumes a key cannot also emit the same key (use a "
                "versioned emit name like '<key>_v1' to express in-place "
                "refinement)."
            )

        if self.correction_kind not in LEGAL_CORRECTION_KIND:
            raise SideInfoBakerContractError(
                f"correction_kind={self.correction_kind!r} not in "
                f"{sorted(LEGAL_CORRECTION_KIND)}"
            )
        if self.correction_resolution not in LEGAL_CORRECTION_RESOLUTION:
            raise SideInfoBakerContractError(
                f"correction_resolution={self.correction_resolution!r} not "
                f"in {sorted(LEGAL_CORRECTION_RESOLUTION)}"
            )

        # ---- 3. Side-information invariants
        if self.side_info_source not in LEGAL_SIDE_INFO_SOURCE:
            raise SideInfoBakerContractError(
                f"side_info_source={self.side_info_source!r} not in "
                f"{sorted(LEGAL_SIDE_INFO_SOURCE)}"
            )

        if not isinstance(self.side_info_reproducible, bool):
            raise SideInfoBakerContractError(
                "side_info_reproducible must be bool"
            )
        if not self.side_info_reproducible:
            raise NonReproducibleSideInfoViolation(
                f"Baker id={self.id!r} declares "
                f"side_info_reproducible=False. Per the contest rules + "
                f"CLAUDE.md 'Apples-to-apples evidence discipline' + "
                f"Wyner-Ziv 1976 source-coding-with-side-information "
                f"theorem, every piece of side information used by the "
                f"contest decoder MUST be derivable from public sources. "
                f"Non-reproducible side info (proprietary datasets, "
                f"license-restricted statistics, secret tables) violates "
                f"contest reproducibility and is a structural rule "
                f"violation. Re-source the prior from a public dataset "
                f"OR re-frame as Wyner-Ziv residual against a public "
                f"shared prior."
            )

        if not isinstance(self.requires_canonical_comma2k19_cache, bool):
            raise SideInfoBakerContractError(
                "requires_canonical_comma2k19_cache must be bool"
            )
        if self.requires_canonical_comma2k19_cache:
            # Verify the canonical Comma2k19 cache helper is importable;
            # baker depending on it is otherwise structurally invalid.
            if not _canonical_comma2k19_cache_importable():
                raise CanonicalComma2k19CacheRequiredViolation(
                    f"Baker id={self.id!r} declares "
                    f"requires_canonical_comma2k19_cache=True but the "
                    f"canonical helper module "
                    f"tac.substrates.pretrained_driving_prior."
                    f"local_chunk_cache is NOT importable in this runtime "
                    f"environment. Per STRICT preflight Catalog #213 and "
                    f"CLAUDE.md 'Bugs must be permanently fixed AND "
                    f"self-protected against' non-negotiables, bakers that "
                    f"distill priors from Comma2k19 chunks MUST route "
                    f"through Comma2k19LocalCache.fetch_chunk(...). "
                    f"Verify the helper module is on the worker filesystem "
                    f"OR set requires_canonical_comma2k19_cache=False if "
                    f"this baker no longer uses Comma2k19."
                )

        if self.wyner_ziv_correlation_estimate is not None:
            value = self.wyner_ziv_correlation_estimate
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WynerZivCorrelationInvalidError(
                    f"wyner_ziv_correlation_estimate={value!r} must be None "
                    f"or numeric in [0.0, 1.0]"
                )
            if math.isnan(value) or math.isinf(value):
                raise WynerZivCorrelationInvalidError(
                    f"wyner_ziv_correlation_estimate={value!r} must be a "
                    f"finite number in [0.0, 1.0]; got NaN/inf"
                )
            if not 0.0 <= value <= 1.0:
                raise WynerZivCorrelationInvalidError(
                    f"wyner_ziv_correlation_estimate={value!r} must be in "
                    f"[0.0, 1.0] (Wyner-Ziv I(X;Y)/H(X) ratio is by "
                    f"construction in this interval; values outside cannot "
                    f"be a meaningful correlation)"
                )

        # ---- 4. Production hardening
        for fname, value in (
            ("deterministic", self.deterministic),
            ("scorer_free", self.scorer_free),
        ):
            if not isinstance(value, bool):
                raise SideInfoBakerContractError(f"{fname} must be bool")

        for fname, value in (
            ("archive_bytes_added", self.archive_bytes_added),
            ("inflate_runtime_bytes_added", self.inflate_runtime_bytes_added),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SideInfoBakerContractError(
                    f"{fname}={value!r} must be a non-negative int"
                )
            if value < 0:
                raise SideInfoBakerContractError(
                    f"{fname}={value} must be >= 0"
                )

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise SideInfoBakerContractError(
                    f"seed={self.seed!r} must be None or int"
                )
            if self.seed < 0:
                raise SideInfoBakerContractError(
                    f"seed={self.seed} must be >= 0"
                )

        if self.merge_policy not in LEGAL_MERGE_POLICY:
            raise SideInfoBakerContractError(
                f"merge_policy={self.merge_policy!r} not in "
                f"{sorted(LEGAL_MERGE_POLICY)}"
            )

        # Cross-field invariant: any baker that adds archive bytes MUST be
        # deterministic (byte-stable archive per Catalog #158).
        if self.archive_bytes_added > 0 and not self.deterministic:
            raise SideInfoBakerContractError(
                f"archive_bytes_added>0 requires deterministic=True (per "
                f"Catalog #158 deterministic_compiler discipline). Got "
                f"deterministic=False with archive_bytes_added="
                f"{self.archive_bytes_added} on baker id={self.id!r}."
            )

        # Cross-field invariant: passthrough bakers emit exactly what they
        # consume.
        if (
            self.correction_kind == "passthrough"
            and self.emits
            and self.consumes
        ):
            if self.emits != self.consumes:
                raise SideInfoBakerContractError(
                    f"correction_kind='passthrough' requires emits == "
                    f"consumes; got emits={sorted(self.emits)!r} consumes="
                    f"{sorted(self.consumes)!r} on baker id={self.id!r}."
                )

        # Cross-field invariant: wyner_ziv_residual_encode SHOULD declare
        # a wyner_ziv_correlation_estimate (None is allowed but flagged
        # via field's hook rationale; the autopilot ranker downgrades
        # such bakers because the predicted gain is unknown).
        # We do NOT raise on this — operator may intentionally skip the
        # estimate when measuring is more expensive than the dispatch.

        # Cross-field invariant: scorer_weights side_info_source MUST set
        # scorer_free=False (the baker DOES load scorer weights at compress
        # to extract features). At INFLATE the scorer is forbidden per
        # CLAUDE.md "Strict scorer rule" — but compress-time is allowed.
        if self.side_info_source == "scorer_weights" and self.scorer_free:
            raise SideInfoBakerContractError(
                f"Baker id={self.id!r} declares "
                f"side_info_source='scorer_weights' but scorer_free=True. "
                f"A baker that extracts features from scorer weights MUST "
                f"set scorer_free=False (it loads the scorer at compress "
                f"time). The strict-scorer rule (CLAUDE.md non-negotiable) "
                f"forbids scorer load at INFLATE time; this baker must "
                f"declare stage_phase in {{'compress', 'both'}} and the "
                f"inflate-side artifact must be a precomputed constant, "
                f"NOT a runtime scorer call."
            )

        # ---- 5. 6-hook wire-in validation
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
                raise SideInfoBakerContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise SideInfoBakerContractError(
                        f"{fname}=not_applicable_with_rationale requires a "
                        f"non-empty entry in "
                        f"hook_not_applicable_rationale[{fname!r}]"
                    )

        # Hook 6 (probe-disambiguator): None requires rationale.
        if self.hook_probe_disambiguator is None:
            rationale = self.hook_not_applicable_rationale.get(
                "hook_probe_disambiguator"
            )
            if not rationale or not rationale.strip():
                raise SideInfoBakerContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if (
                not isinstance(self.hook_probe_disambiguator, str)
                or not self.hook_probe_disambiguator.strip()
            ):
                raise SideInfoBakerContractError(
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
                raise SideInfoBakerContractError(
                    f"hook_not_applicable_rationale has illegal key {k!r}; "
                    f"legal={sorted(legal_rationale_keys)}"
                )

        # ---- 6. Provenance soft-validation (None tolerated; non-empty if set)
        for fname, value in (
            ("lane_id", self.lane_id),
            ("design_memo", self.design_memo),
            ("canonical_vs_unique_decision", self.canonical_vs_unique_decision),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise SideInfoBakerContractError(
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
    def from_dict(cls, data: dict[str, Any]) -> "SideInfoBakerContract":
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
