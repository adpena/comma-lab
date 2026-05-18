# SPDX-License-Identifier: MIT
"""RashomonEnsembleCommittee — re-export wrapper around
``tac.autopilot_rudin_daubechies.RashomonEnsembleRanker``.

Per `.omx/research/tac_search_namespace_design_20260517.md` §6 + §10 +
PV-4 + §13 mandatory assumption statement:

  The K=8 Rashomon ensemble is CANONICAL infrastructure per Catalog #252.
  Re-implementing it in tac.search would violate the operator's standing
  directive (`feedback_canonical_share_when_serves_unique_when_suppresses`)
  because the same continual-learning anchor store backs the preflight
  risk scorer + the autopilot ranker; forking would silently desync the
  three surfaces.

This builder WRAPS the existing RashomonEnsembleRanker as a search
strategy. The "search" semantics here are CANDIDATE RANKING by ensemble
consensus + disagreement, not parameter-space exploration. The strategy:

  1. Takes a list of candidates (each is a ProxyPanel describing a
     parameterized substrate configuration)
  2. Predicts a consensus score + disagreement std-dev across the K=8
     SLIM members
  3. Returns the BEST candidate (lowest consensus score) as best_params
  4. Records the HIGH-disagreement candidates as a probe-disambiguator
     surface — those candidates are the next-experiment queue per Catalog
     #252

This builder is DISTINCT from the other 4 in that the "objective function"
is replaced by a candidate POOL (the ranker's view) + the SLIM panel
features. The `objective_fn` parameter is REPURPOSED: it returns a
ProxyPanel given a candidate id, so the strategy can stamp each candidate
with its panel features at runtime.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + the design memo
§10 canonical-vs-unique decision per layer: this is the only builder
that ADOPTS canonical rather than UNIQUE-FORKING. The canonical-ness IS
the design intent — the K=8 ensemble must remain the single source of
truth across preflight + autopilot + this new search surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tac.search.contract import SearchContract
from tac.search.decorator import search_strategy
from tac.search.errors import ObjectiveFunctionError
from tac.search.pipeline import SearchHistory, SearchResult, SearchTrial

__all__ = [
    "RashomonEnsembleCommittee",
    "RashomonEnsembleCommitteeSpec",
]


@dataclass(frozen=True)
class RashomonEnsembleCommitteeSpec:
    """Specification for a Rashomon committee re-export wrapper.

    The committee evaluates a candidate POOL (each candidate = ProxyPanel
    + candidate_id) and returns the consensus-best candidate.

    Args:
      strategy_id: namespace id
      candidate_pool: list of candidate dicts; each dict MUST have a
        `candidate_id: str` field; the objective_fn receives the dict and
        returns a ProxyPanel
      ensemble_size: K=8 by default (canonical)
      bootstrap_seed_base: base seed for bootstrap resampling
      sparsity_target: SLIM L0 sparsity target per member
      integer_coefficient_bound: SLIM integer-coefficient bound per member
      anchor_store_path: optional .omx/state/slim_anchor_store.jsonl path
        (None → use the canonical default from the underlying ranker)
    """

    strategy_id: str
    candidate_pool: list[Mapping[str, Any]]
    ensemble_size: int = 8
    bootstrap_seed_base: int = 42
    sparsity_target: int = 5
    integer_coefficient_bound: int = 10
    anchor_store_path: str | None = None
    description: str = ""
    lane_id: str | None = None
    objective_is_surrogate: bool = True  # SLIM is a surrogate by construction
    predicted_search_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_pool, list) or not self.candidate_pool:
            raise ValueError(
                f"candidate_pool={self.candidate_pool!r} must be a "
                "non-empty list of candidate dicts"
            )
        for i, c in enumerate(self.candidate_pool):
            if not isinstance(c, Mapping) or "candidate_id" not in c:
                raise ValueError(
                    f"candidate_pool[{i}]={c!r}: each candidate must be a "
                    "Mapping with a 'candidate_id' string field"
                )
        if self.ensemble_size < 1:
            raise ValueError(
                f"ensemble_size={self.ensemble_size} must be >= 1"
            )
        if self.sparsity_target < 1:
            raise ValueError(
                f"sparsity_target={self.sparsity_target} must be >= 1"
            )
        if self.integer_coefficient_bound < 1:
            raise ValueError(
                f"integer_coefficient_bound="
                f"{self.integer_coefficient_bound} must be >= 1"
            )
        if self.bootstrap_seed_base < 0:
            raise ValueError(
                f"bootstrap_seed_base={self.bootstrap_seed_base} must be "
                ">= 0"
            )


class RashomonEnsembleCommittee:
    """Builder for a Rashomon committee re-export wrapper.

    The committee REUSES the canonical
    ``tac.autopilot_rudin_daubechies.RashomonEnsembleRanker`` — it does
    NOT re-implement the K=8 SLIM ensemble. The continual-learning anchor
    store remains the single source of truth across preflight risk
    scorer + autopilot ranker + this search surface.

    Usage::

        from tac.search import (
            RashomonEnsembleCommittee, RashomonEnsembleCommitteeSpec,
        )

        committee = RashomonEnsembleCommittee(
            spec=RashomonEnsembleCommitteeSpec(
                strategy_id="rashomon_committee_pr101_candidates",
                candidate_pool=[
                    {"candidate_id": "fec6_K8", "K": 8, ...},
                    {"candidate_id": "fec6_K16", "K": 16, ...},
                    ...
                ],
                ensemble_size=8,
                bootstrap_seed_base=42,
                lane_id="lane_my_substrate_20260601",
            )
        )
        committee.register()

        # The strategy fn takes objective_fn that returns ProxyPanel
        from tac.search import run_search_over_pipeline
        result = run_search_over_pipeline(
            pipeline_with_search,
            objective_fn=my_panel_extractor,  # candidate dict -> ProxyPanel
        )
    """

    def __init__(self, *, spec: RashomonEnsembleCommitteeSpec) -> None:
        if not isinstance(spec, RashomonEnsembleCommitteeSpec):
            raise TypeError(
                f"spec must be RashomonEnsembleCommitteeSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SearchContract:
        rationale: dict[str, str] = {
            "hook_pareto_constraint": (
                "Rashomon ensemble ranks by consensus score; multi-objective "
                "Pareto is a sister consumer (probe-disambiguator queue)."
            ),
            "hook_bit_allocator_class": (
                "Rashomon ranks candidates; bit allocation is downstream."
            ),
            "hook_sensitivity_contribution": (
                "Rashomon uses SLIM panel features (Taylor proxies) as its "
                "sensitivity surface; the master_gradient consumer is "
                "downstream (next-probe selection)."
            ),
        }
        return SearchContract(
            id=self.spec.strategy_id,
            parent_strategy_id=None,
            description=(
                self.spec.description
                or (
                    f"Rashomon ensemble committee wrapping "
                    f"tac.autopilot_rudin_daubechies.RashomonEnsembleRanker: "
                    f"K={self.spec.ensemble_size}, "
                    f"sparsity={self.spec.sparsity_target}, "
                    f"|pool|={len(self.spec.candidate_pool)}."
                )
            ),
            search_kind="discrete",  # candidate-pool ranking
            n_candidate_evaluations_max=len(self.spec.candidate_pool),
            parallelism="serial",
            requires_objective_function=True,
            objective_is_surrogate=self.spec.objective_is_surrogate,
            deterministic=True,
            seed=self.spec.bootstrap_seed_base,
            predicted_search_cost_usd=self.spec.predicted_search_cost_usd,
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_pareto_constraint="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind="search_strategy_outcomes_v1",
            hook_probe_disambiguator=(
                "tools/cathedral_autopilot_autonomous_loop.py"
                ":disagreement_queue_consumer"
            ),
            hook_not_applicable_rationale=rationale,
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/tac_search_namespace_design_20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL_BECAUSE_SERVES: re-uses "
                "tac.autopilot_rudin_daubechies.RashomonEnsembleRanker per "
                "Catalog #252 + the operator's standing directive "
                "`feedback_canonical_share_when_serves_unique_when_suppresses`. "
                "The K=8 SLIM ensemble remains the single source of truth "
                "across preflight + autopilot + this search surface."
            ),
        )

    def register(self) -> SearchContract:
        contract = self.build_contract()
        spec = self.spec

        @search_strategy(contract)
        def _rashomon_committee_run(
            objective_fn: Callable[[Mapping[str, Any]], Any],
            *,
            bounds: Any = None,
            seed: int = spec.bootstrap_seed_base,
            warm_start: Any = None,
            **_ignored: Any,
        ) -> SearchResult:
            return _run_rashomon_committee(
                contract=contract,
                spec=spec,
                objective_fn=objective_fn,
                seed=seed,
            )

        return contract


def _run_rashomon_committee(
    *,
    contract: SearchContract,
    spec: RashomonEnsembleCommitteeSpec,
    objective_fn: Callable[[Mapping[str, Any]], Any],
    seed: int,
) -> SearchResult:
    """Execute the Rashomon committee ranking; return SearchResult.

    The canonical surface uses RashomonEnsembleRanker.predict_with_disagreement
    on each candidate's ProxyPanel; the best candidate (lowest consensus
    score) becomes best_params; the per-candidate (score, disagreement)
    rows become the history.
    """
    from tac.autopilot_rudin_daubechies import RashomonEnsembleRanker
    from tac.autopilot_rudin_daubechies.slim_ranker import ProxyPanel

    anchor_store_path: Path | None = (
        Path(spec.anchor_store_path) if spec.anchor_store_path else None
    )
    ranker = RashomonEnsembleRanker(
        ensemble_size=spec.ensemble_size,
        integer_bound=spec.integer_coefficient_bound,
        sparsity_target=spec.sparsity_target,
        rng_seed=seed,
        store_path=anchor_store_path,
    )

    trials: list[SearchTrial] = []
    best_score = float("inf")
    best_params: dict[str, Any] = {}
    start = time.monotonic()

    for i, candidate in enumerate(spec.candidate_pool):
        try:
            panel_or_panel_dict = objective_fn(candidate)
        except Exception as exc:
            raise ObjectiveFunctionError(
                f"objective_fn raised on candidate {i}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Accept either a ProxyPanel directly OR a dict mapping panel
        # fields to values.
        if isinstance(panel_or_panel_dict, ProxyPanel):
            panel = panel_or_panel_dict
        elif isinstance(panel_or_panel_dict, Mapping):
            try:
                panel = ProxyPanel(
                    candidate_id=str(candidate.get("candidate_id", f"c{i}")),
                    **{
                        k: v
                        for k, v in panel_or_panel_dict.items()
                        if k != "candidate_id"
                    },
                )
            except TypeError as exc:
                raise ObjectiveFunctionError(
                    f"objective_fn returned a dict that does not match "
                    f"ProxyPanel schema on candidate {i}: {exc}"
                ) from exc
        else:
            raise ObjectiveFunctionError(
                f"objective_fn returned {type(panel_or_panel_dict).__name__} "
                f"on candidate {i}; expected ProxyPanel or panel-field dict."
            )

        consensus, disagreement = ranker.predict_with_disagreement(panel)
        trial_params = {
            **dict(candidate),
            "rashomon_consensus_score": consensus,
            "rashomon_disagreement_stddev": disagreement,
        }
        trials.append(
            SearchTrial(
                trial_index=i,
                params=trial_params,
                score=float(consensus),
            )
        )
        if consensus < best_score:
            best_score = float(consensus)
            best_params = dict(candidate)
            best_params["rashomon_consensus_score"] = float(consensus)
            best_params["rashomon_disagreement_stddev"] = float(disagreement)

    elapsed = time.monotonic() - start
    return SearchResult(
        strategy_id=spec.strategy_id,
        best_params=best_params,
        best_score=best_score,
        n_evaluations=len(spec.candidate_pool),
        elapsed_seconds=elapsed,
        history=SearchHistory(trials=tuple(trials)),
        objective_is_surrogate=contract.objective_is_surrogate,
        seed=seed,
        notes="rashomon_committee_consensus_rank",
    )
