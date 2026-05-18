# SPDX-License-Identifier: MIT
"""SearchContract — canonical schema for a single search strategy.

Per ``.omx/research/tac_search_namespace_design_20260517.md`` + the §7.6
spec at ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§F search-method coverage: every search strategy MUST declare a frozen
``SearchContract`` that captures its identity, search kind, candidate
budget, parallelism class, objective-function routing, and production-
hardening invariants (deterministic, seed, predicted_search_cost_usd).

Mirrors the canonical contract pattern from
``tac.boosting.contract.BoostStageContract`` and
``tac.compress_time_optimization.contract.CompressTimePassContract`` —
frozen kw_only dataclass + field-level validators raising
``SearchContractError`` at construction time so malformed strategies fail
at IMPORT time (not at dispatch time).

Distinguishing fields unique to this namespace (per design memo §10
canonical-vs-unique decision per layer):
  - ``search_kind: Literal[continuous, discrete, mixed, multi_objective]``
    — engine selection depends on it.
  - ``n_candidate_evaluations_max: int`` — the search budget; canonical
    rate-limiter per evaluation count (NOT wallclock; sister namespaces
    use wallclock).
  - ``parallelism: Literal[serial, vectorized, process_pool]`` — pinned
    at contract time so callers know whether process_pool is safe.
  - ``requires_objective_function: bool`` — hard-pinned True; refusing
    False at validation prevents undefined-search bugs.
  - ``objective_is_surrogate: bool`` — routes through the Hinton
    surrogate vs the real contest scorer per Catalog #527.
  - ``predicted_search_cost_usd: float`` — informs cathedral autopilot
    ranking.
  - ``seed: int | None`` — required when ``deterministic=True``;
    ``SeedRequiredViolation`` raised at decoration when the function
    signature lacks `seed` AND ``contract.seed is None``.

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow API
— one dataclass, one error class. The decorator + pipeline modules
consume this contract; consumers DO NOT subclass it.

Cross-references: design memo §9 (cargo-cult audit) + §10 (canonical-vs-
unique per layer) + §11 (observability surface) + §12 (6-hook wire-in).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from tac.search.errors import SearchContractError

__all__ = [
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_PARALLELISM",
    "LEGAL_SEARCH_KIND",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "SearchContract",
]


# ---------------------------------------------------------------------------
# Enum-like legal value sets (frozensets for O(1) membership; immutable).
# ---------------------------------------------------------------------------

LEGAL_SEARCH_KIND: frozenset[str] = frozenset(
    {
        "continuous",        # CMA-ES, BoTorch GP, BayesianOptimizationGP
        "discrete",          # MCTS, RashomonEnsembleCommittee on categorical
        "mixed",             # Optuna TPE (handles both)
        "multi_objective",   # NSGA-II family (deferred to future builder)
    }
)

LEGAL_PARALLELISM: frozenset[str] = frozenset(
    {
        "serial",            # MCTS (state-dependent across trials)
        "vectorized",        # CMA-ES (full population evaluated per gen)
        "process_pool",      # RandomSearch, embarrassingly-parallel TPE
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
        "multi_objective_pareto_front_v1",
        "search_strategy_pareto_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_BIT_ALLOCATOR: frozenset[str] = frozenset(
    {
        # Search strategies discover parameter values; bit allocation is
        # downstream. The full enum is preserved so multi-objective
        # strategies that DO produce bit-allocation candidates can name
        # them; otherwise N/A is the default.
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
        "search_strategy_outcomes_v1",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, kw_only=True)
class SearchContract:
    """Canonical contract declared by every search strategy via
    ``@search_strategy``.

    Mirrors ``BoostStageContract`` / ``CompressTimePassContract`` at the
    search-strategy surface — one frozen dataclass, field-level validators,
    no inheritance.

    Field groups:
      1. Identity (id, parent_strategy_id, description)
      2. Search semantics (search_kind, n_candidate_evaluations_max,
         parallelism, requires_objective_function, objective_is_surrogate)
      3. Production hardening (deterministic, seed, predicted_search_cost_usd,
         max_wallclock_seconds)
      4. 6-hook wire-in (Catalog #125)
      5. Provenance (lane_id, design_memo, canonical_vs_unique_decision)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": every consumer (Pipeline,
    persistence ledger, autopilot ranker) reads from this contract via
    explicit field access — no hidden state, no class hierarchy. The
    namespace does NOT share BoostStageContract / CompressTimePassContract:
    sister namespaces are STRUCTURALLY INDEPENDENT.
    """

    # 1. Identity (3)
    id: str
    parent_strategy_id: str | None = None
    description: str = ""

    # 2. Search semantics (5)
    search_kind: str = "continuous"
    n_candidate_evaluations_max: int = 100
    parallelism: str = "serial"
    requires_objective_function: bool = True  # HARD-PINNED True at validation
    objective_is_surrogate: bool = False

    # 3. Production hardening (4)
    deterministic: bool = True
    seed: int | None = None
    predicted_search_cost_usd: float = 0.0
    max_wallclock_seconds: int | None = None

    # 4. 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_pareto_constraint: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_bit_allocator_class: str = NOT_APPLICABLE_WITH_RATIONALE
    hook_autopilot_ranker: str = "cathedral_autopilot_v1"
    hook_continual_learning_anchor_kind: str = "search_strategy_outcomes_v1"
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
            raise SearchContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/ (snake_case "
                f"required; kebab-case rejected to mirror sister namespace "
                f"conventions)"
            )
        if self.parent_strategy_id is not None:
            if not isinstance(self.parent_strategy_id, str) or not _ID_PATTERN.match(
                self.parent_strategy_id
            ):
                raise SearchContractError(
                    f"parent_strategy_id={self.parent_strategy_id!r} must be "
                    "None or match /^[a-z][a-z0-9_]*$/"
                )
            if self.parent_strategy_id == self.id:
                raise SearchContractError(
                    f"parent_strategy_id={self.parent_strategy_id!r} must "
                    f"differ from id={self.id!r} (a strategy cannot be its "
                    "own parent)"
                )

        # ---- 2. Search semantics
        if self.search_kind not in LEGAL_SEARCH_KIND:
            raise SearchContractError(
                f"search_kind={self.search_kind!r} not in "
                f"{sorted(LEGAL_SEARCH_KIND)}"
            )
        if self.parallelism not in LEGAL_PARALLELISM:
            raise SearchContractError(
                f"parallelism={self.parallelism!r} not in "
                f"{sorted(LEGAL_PARALLELISM)}"
            )

        if not isinstance(self.n_candidate_evaluations_max, int) or isinstance(
            self.n_candidate_evaluations_max, bool
        ):
            raise SearchContractError(
                f"n_candidate_evaluations_max="
                f"{self.n_candidate_evaluations_max!r} must be a positive int "
                f"(the canonical search budget per evaluation count)"
            )
        if self.n_candidate_evaluations_max < 1:
            raise SearchContractError(
                f"n_candidate_evaluations_max={self.n_candidate_evaluations_max} "
                f"must be >= 1"
            )

        # requires_objective_function is HARD-PINNED True per design memo §4;
        # refusing False here prevents undefined-search bugs.
        if not isinstance(self.requires_objective_function, bool):
            raise SearchContractError(
                "requires_objective_function must be a bool"
            )
        if not self.requires_objective_function:
            raise SearchContractError(
                "requires_objective_function=False is FORBIDDEN: every search "
                "strategy needs an objective. The field exists for explicit "
                "contract declaration + serialization only; the value MUST "
                "be True. (Per design memo §4 + §9 cargo-cult audit.)"
            )

        if not isinstance(self.objective_is_surrogate, bool):
            raise SearchContractError(
                "objective_is_surrogate must be a bool"
            )

        # ---- 3. Production hardening
        if not isinstance(self.deterministic, bool):
            raise SearchContractError("deterministic must be a bool")

        # seed: None LEGAL; int must be >= 0 if provided.
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise SearchContractError(
                    f"seed={self.seed!r} must be None or int"
                )
            if self.seed < 0:
                raise SearchContractError(
                    f"seed={self.seed} must be >= 0 (or None for "
                    "function-signature-derived seeds)"
                )

        if not isinstance(self.predicted_search_cost_usd, (int, float)) or isinstance(
            self.predicted_search_cost_usd, bool
        ):
            raise SearchContractError(
                f"predicted_search_cost_usd="
                f"{self.predicted_search_cost_usd!r} must be a float >= 0"
            )
        if float(self.predicted_search_cost_usd) < 0:
            raise SearchContractError(
                f"predicted_search_cost_usd="
                f"{self.predicted_search_cost_usd} must be >= 0"
            )

        if self.max_wallclock_seconds is not None:
            if isinstance(self.max_wallclock_seconds, bool) or not isinstance(
                self.max_wallclock_seconds, int
            ):
                raise SearchContractError(
                    f"max_wallclock_seconds={self.max_wallclock_seconds!r} "
                    "must be None or int"
                )
            if self.max_wallclock_seconds < 1:
                raise SearchContractError(
                    f"max_wallclock_seconds={self.max_wallclock_seconds} must "
                    "be >= 1 (or None for unbounded)"
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
                raise SearchContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise SearchContractError(
                        f"{fname}=not_applicable_with_rationale requires a "
                        "non-empty entry in "
                        f"hook_not_applicable_rationale[{fname!r}]"
                    )

        # Hook 6 (probe-disambiguator)
        if self.hook_probe_disambiguator is None:
            rationale = self.hook_not_applicable_rationale.get(
                "hook_probe_disambiguator"
            )
            if not rationale or not rationale.strip():
                raise SearchContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if (
                not isinstance(self.hook_probe_disambiguator, str)
                or not self.hook_probe_disambiguator.strip()
            ):
                raise SearchContractError(
                    "hook_probe_disambiguator must be a non-empty path/"
                    "identifier or None"
                )

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
                raise SearchContractError(
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
                raise SearchContractError(
                    f"{fname}={value!r} must be None or a non-empty string"
                )

    # ------------------------------------------------------------------
    # Serialization helpers (byte-stable JSON round-trip per design memo §11)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (sorted keys for byte-stable downstream
        consumers + JSON round-trip safety).
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
    def from_dict(cls, data: dict[str, Any]) -> "SearchContract":
        """Reconstruct a contract from a dict (e.g. JSON round-trip).

        Missing fields use the dataclass defaults so partial dicts (e.g.
        legacy persistence rows) still construct.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
