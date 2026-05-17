# SPDX-License-Identifier: MIT
"""ComposableBoostingPipeline — immutable pipeline-of-stages with operator
composition (`|` sequential, `&` parallel-merge, `@` attach-search).

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.4 composition primitives:

  - ``A | B`` — sequential: runs A then B (canonical chain)
  - ``A & B`` — parallel-merge: runs both then merges by per-byte policy
  - ``A @ search`` — attach a search strategy from ``tac.search.*``
    (when that namespace lands; for now the pipeline accepts the attach
    but the search strategy is None / placeholder)

Every compose operation returns a NEW immutable pipeline (no mutation, no
surprise side-effects, easy to test). Pipeline objects are JSON-serializable
so the cathedral autopilot can rank candidate pipelines without instantiating
them and the operator can audit ranked candidates as plain text.

Per CLAUDE.md "Beauty, simplicity, and developer experience":
  - immutable construction → no mid-build state races
  - all errors at .build()/.run() surface AmbiguousCompositionError or
    BoostingPipelineError with named conflicting stages (no silent stage
    ordering)
  - JSON-serializable representation for cathedral autopilot ranking
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from tac.boosting.contract import BoostStageContract  # noqa: TC001
from tac.boosting.decorator import (
    _REGISTERED_STAGES,
    get_stage_function,
)
from tac.boosting.errors import (
    AmbiguousCompositionError,
    BoostingPipelineError,
)
from tac.boosting.pareto_front import ParetoFrontTracker

__all__ = [
    "BoostingPipelineResult",
    "ComposableBoostingPipeline",
    "PipelineStageRef",
]


@dataclass(frozen=True)
class PipelineStageRef:
    """A single stage reference in a pipeline (id + optional parameters).

    Frozen so pipeline composition is structurally immutable. Pipeline
    operators return new pipelines with new tuples of references — the
    refs themselves are never mutated.
    """

    stage_id: str
    parameters: tuple[tuple[str, Any], ...] = ()
    # Each PipelineStageRef has a kind that mirrors how it was added:
    #   - "sequential": added via `|`
    #   - "parallel": added via `&` (paired with prior stage by position)
    #   - "search_attached": added via `@`; the parameters tuple carries
    #     the search strategy descriptor
    composition_kind: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "parameters": list(self.parameters),
            "composition_kind": self.composition_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStageRef:
        params = tuple(
            (k, v) for k, v in data.get("parameters", [])
        )
        return cls(
            stage_id=data["stage_id"],
            parameters=params,
            composition_kind=data.get("composition_kind", "sequential"),
        )


@dataclass(frozen=True)
class BoostingPipelineResult:
    """Result returned by ``ComposableBoostingPipeline.run``.

    Carries the final state dict, the per-stage outcome log, and a frontier
    snapshot when ``with_pareto_growth`` was active. Frozen for safe
    consumer-side audit / serialization.
    """

    final_state: Mapping[str, Any]
    per_stage_outcomes: tuple[Mapping[str, Any], ...] = ()
    pareto_snapshot: Mapping[str, Any] | None = None
    rejected_stages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": dict(self.final_state),
            "per_stage_outcomes": [dict(o) for o in self.per_stage_outcomes],
            "pareto_snapshot": (
                dict(self.pareto_snapshot)
                if self.pareto_snapshot is not None
                else None
            ),
            "rejected_stages": list(self.rejected_stages),
        }


@dataclass(frozen=True)
class ComposableBoostingPipeline:
    """Immutable pipeline-of-stages with operator composition.

    Construction is via the canonical ``|`` operator chaining starting from
    an empty pipeline::

        pipeline = (
            ComposableBoostingPipeline()
            | "raw_decoder"
            | "cascade_pose_residual"
            | "cascade_seg_residual"
        )

    Or alternatively from a list (imperative form for callers who prefer
    not to chain)::

        pipeline = ComposableBoostingPipeline.from_stage_ids(
            ["raw_decoder", "cascade_pose_residual"]
        )

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the pipeline carries
    NO hidden state — every behavior is visible via the ``stages`` tuple
    and the ``pareto_growth_filter`` callable. Two pipelines with equal
    ``stages`` + ``pareto_growth_filter_descriptor`` are equivalent
    (the Pareto filter itself is a callable — equivalence is checked via
    a string descriptor that serializes to JSON for cathedral autopilot
    ranking).
    """

    stages: tuple[PipelineStageRef, ...] = ()
    # Pareto-growth filter is OPT-IN; default None.
    # The filter is a callable (rate, distortion) -> bool. We carry both
    # the callable AND a JSON-serializable descriptor so the pipeline can
    # round-trip through JSON without losing the filter intent.
    pareto_growth_filter: Callable[[float, float], bool] | None = None
    pareto_growth_filter_descriptor: str | None = None
    # Search strategy attached via `@`. None when no strategy is attached.
    # When the tac.search namespace lands the descriptor here will name
    # a registered strategy.
    search_strategy_descriptor: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_stage_ids(
        cls, stage_ids: list[str], **kwargs: Any
    ) -> ComposableBoostingPipeline:
        """Build a pipeline from a flat list of stage ids (imperative form).

        Equivalent to ``ComposableBoostingPipeline() | stage_ids[0] | stage_ids[1] | ...``.
        """
        pipeline = cls(**kwargs)
        for sid in stage_ids:
            pipeline = pipeline | sid
        return pipeline

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposableBoostingPipeline:
        """Reconstruct a pipeline from a JSON-deserialized dict.

        The Pareto filter callable cannot be restored from JSON; only its
        descriptor survives. Callers that need the filter active after
        round-trip must re-attach via ``with_pareto_growth``.
        """
        stages = tuple(
            PipelineStageRef.from_dict(s) for s in data.get("stages", [])
        )
        return cls(
            stages=stages,
            pareto_growth_filter=None,  # cannot serialize a callable
            pareto_growth_filter_descriptor=data.get(
                "pareto_growth_filter_descriptor"
            ),
            search_strategy_descriptor=data.get("search_strategy_descriptor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "pareto_growth_filter_descriptor": self.pareto_growth_filter_descriptor,
            "search_strategy_descriptor": self.search_strategy_descriptor,
        }

    def to_json(self) -> str:
        """JSON-serialize the pipeline (sorted keys for byte-stable output)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        """Human-readable representation usable in operator audit + log lines."""
        if not self.stages:
            return "ComposableBoostingPipeline(<empty>)"
        chain = " | ".join(s.stage_id for s in self.stages)
        suffix = ""
        if self.pareto_growth_filter_descriptor is not None:
            suffix = (
                f".with_pareto_growth({self.pareto_growth_filter_descriptor})"
            )
        if self.search_strategy_descriptor is not None:
            suffix += f" @ {self.search_strategy_descriptor}"
        return f"ComposableBoostingPipeline({chain}){suffix}"

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(self, stage: str | PipelineStageRef) -> ComposableBoostingPipeline:
        """Sequential composition (`A | B` runs A then B).

        Accepts either a bare stage id (str) or a fully-formed PipelineStageRef
        (the latter is used by the search namespace to attach parameters).

        Returns a NEW pipeline; the original is unchanged.
        """
        ref = (
            stage
            if isinstance(stage, PipelineStageRef)
            else PipelineStageRef(stage_id=stage, composition_kind="sequential")
        )
        return replace(self, stages=(*self.stages, ref))

    def __and__(
        self, stage: str | PipelineStageRef
    ) -> ComposableBoostingPipeline:
        """Parallel-merge composition (`A & B` runs both then merges by
        per-byte policy declared in each stage's contract).

        Per spec §5.4: parallel-merge stages execute side-by-side and their
        emit dicts are merged per the prior stage's ``merge_policy``. The
        operator constructs a paired PipelineStageRef with composition_kind
        ``"parallel"`` so the runtime knows to dispatch the prior stage and
        this stage simultaneously.
        """
        if not self.stages:
            raise BoostingPipelineError(
                "`&` (parallel-merge) requires at least one prior stage; "
                "use `|` for the first stage."
            )
        ref = (
            replace(stage, composition_kind="parallel")
            if isinstance(stage, PipelineStageRef)
            else PipelineStageRef(stage_id=stage, composition_kind="parallel")
        )
        return replace(self, stages=(*self.stages, ref))

    def __matmul__(self, search_descriptor: str) -> ComposableBoostingPipeline:
        """Attach a search strategy (`pipeline @ "cma_es_over_K_palette"`).

        Per spec §5.4: the search strategy is a tac.search.* registered
        strategy that sweeps a stage's hyperparameters. This operator stores
        the descriptor; the actual search engine lives in tac.search.*
        (which will land in a future subagent slot).

        For now the pipeline accepts the attach but execution is a no-op
        (the search is not yet wired). The descriptor IS serialized through
        to_json so the cathedral autopilot ranker can see + rank pipelines
        with attached strategies.
        """
        if not isinstance(search_descriptor, str) or not search_descriptor.strip():
            raise BoostingPipelineError(
                f"`@` (attach search) requires a non-empty descriptor string; "
                f"got {search_descriptor!r}"
            )
        return replace(self, search_strategy_descriptor=search_descriptor)

    def with_pareto_growth(
        self,
        *,
        reject_if_worsens_axis: str = "rate",
        tracker: ParetoFrontTracker | None = None,
    ) -> ComposableBoostingPipeline:
        """Attach a Pareto-front-aware growth filter.

        Per spec §5.4: each stage's contribution must Pareto-improve the
        (rate, distortion) frontier; stages that worsen the chosen axis
        are rejected at run time and recorded in
        ``BoostingPipelineResult.rejected_stages``.

        The tracker is OPT-IN. If None is supplied a fresh tracker is
        created (axis defaults to the first stage's
        hook_pareto_constraint label OR "[proxy]"). Callers that already
        have a populated tracker (e.g. seeded with PR101 GOLD / PR102 /
        PR103 anchors) can pass it directly.
        """
        if tracker is None:
            tracker = ParetoFrontTracker(axis="[proxy]")
        if not isinstance(tracker, ParetoFrontTracker):
            raise BoostingPipelineError(
                f"tracker must be a ParetoFrontTracker or None; got "
                f"{type(tracker).__name__}"
            )
        filter_callable = tracker.for_pareto_growth_filter(
            reject_if_worsens_axis=reject_if_worsens_axis
        )
        descriptor = (
            f"axis={tracker.axis!r},reject_if_worsens_axis="
            f"{reject_if_worsens_axis!r}"
        )
        return replace(
            self,
            pareto_growth_filter=filter_callable,
            pareto_growth_filter_descriptor=descriptor,
        )

    # ------------------------------------------------------------------
    # Build + Run
    # ------------------------------------------------------------------

    def build(self) -> ComposableBoostingPipeline:
        """Validate the pipeline's structural correctness without running.

        Surfaces every error class at build time so dispatch never
        encounters a structurally invalid pipeline:

          - Unknown stage id (not registered via @boost_stage)
          - Ambiguous emit key (two stages emit same key without explicit
            merge) → AmbiguousCompositionError
          - Cycle in parent_stage_id chain → BoostingPipelineError

        Returns self (the pipeline is already immutable; build() is a
        validation pass). The validated pipeline is then safe to .run().
        """
        # 1. Every stage id must be registered
        for ref in self.stages:
            if ref.stage_id not in _REGISTERED_STAGES:
                raise BoostingPipelineError(
                    f"Pipeline references stage id={ref.stage_id!r} which is "
                    f"not registered via @boost_stage. Registered ids: "
                    f"{sorted(_REGISTERED_STAGES)}"
                )

        # 2. Detect ambiguous emit keys.
        # An emit key is ambiguous when two sequential stages emit it
        # without an intermediate consumer. Parallel-merge stages
        # (composition_kind="parallel") are EXPECTED to emit overlapping
        # keys (that's the merge), so they are excluded from this check.
        seen_emits: dict[str, str] = {}  # key -> stage_id that emitted
        consumed_since_emit: set[str] = set()
        for ref in self.stages:
            contract = _REGISTERED_STAGES[ref.stage_id]
            # Pre-mark this stage's consumes as having consumed any prior emits
            for key in contract.consumes:
                consumed_since_emit.add(key)
            for key in contract.emits:
                if (
                    key in seen_emits
                    and key not in consumed_since_emit
                    and ref.composition_kind != "parallel"
                ):
                    prior = seen_emits[key]
                    raise AmbiguousCompositionError(
                        f"Pipeline emits key {key!r} twice without "
                        f"intermediate consumer: first by stage "
                        f"{prior!r}, then by stage {ref.stage_id!r}. "
                        f"Either insert a stage that consumes {key!r} "
                        f"between them, OR use `&` (parallel-merge) to "
                        f"declare explicit merge intent, OR rename one "
                        f"stage's emit to {key!r}_v2."
                    )
                seen_emits[key] = ref.stage_id
                consumed_since_emit.discard(key)

        # 3. Cycle detection in parent_stage_id chain
        for ref in self.stages:
            contract = _REGISTERED_STAGES[ref.stage_id]
            seen = {contract.id}
            cursor = contract.parent_stage_id
            while cursor is not None:
                if cursor in seen:
                    raise BoostingPipelineError(
                        f"Cycle detected in parent_stage_id chain starting "
                        f"from stage id={contract.id!r}: cycle through {cursor!r}"
                    )
                seen.add(cursor)
                parent_contract = _REGISTERED_STAGES.get(cursor)
                if parent_contract is None:
                    # Unregistered parent — surface a structural error.
                    raise BoostingPipelineError(
                        f"Stage id={contract.id!r} declares "
                        f"parent_stage_id={cursor!r} which is not registered. "
                        f"Either register the parent OR set parent_stage_id=None."
                    )
                cursor = parent_contract.parent_stage_id

        return self

    def run(
        self,
        seed_state: Mapping[str, Any] | None = None,
        *,
        master_gradient: Any | None = None,
        policy: Mapping[str, Any] | None = None,
    ) -> BoostingPipelineResult:
        """Execute the pipeline left-to-right against ``seed_state``.

        For each stage:
          1. Resolve the registered stage function from the decorator registry
          2. Invoke ``fn(state, master_gradient=..., policy=...)``; the
             stage returns a new state dict (additive merge with prior state)
          3. If a Pareto growth filter is attached and the state carries
             ``rate`` + ``distortion`` keys, evaluate the filter; reject
             the stage's contribution if the filter returns False

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
          - state is a plain dict (no hidden ABC)
          - the function signature is uniform across all stages
          - rejected stages are RECORDED, not silently dropped, so the
            operator can audit the rejection log

        The default behavior is OPT-IN persistence — callers that want
        outcomes appended to the canonical ``.omx/state/
        boosting_stage_outcomes.jsonl`` ledger must wrap the call in
        ``persistence.append_stage_outcome_locked(...)`` per CLAUDE.md
        Catalog #128/#131 sister discipline (the namespace does not
        auto-persist).
        """
        # Validate before running
        self.build()

        state: dict[str, Any] = dict(seed_state) if seed_state is not None else {}
        per_stage_outcomes: list[dict[str, Any]] = []
        rejected: list[str] = []

        index = 0
        while index < len(self.stages):
            root_ref = self.stages[index]
            if root_ref.composition_kind == "parallel":
                raise BoostingPipelineError(
                    f"Stage id={root_ref.stage_id!r} is marked parallel but has "
                    "no sequential root in this execution group."
                )

            group_refs = [root_ref]
            index += 1
            while (
                index < len(self.stages)
                and self.stages[index].composition_kind == "parallel"
            ):
                group_refs.append(self.stages[index])
                index += 1

            # Parallel siblings must observe the SAME pre-group input state.
            # Otherwise `A & B` silently degenerates into `A | B`, which makes
            # stack-of-stacks probes depend on accidental stage order.
            group_input_state = dict(state)
            merged_stage_output: dict[str, Any] = {}

            for ref in group_refs:
                contract = _REGISTERED_STAGES[ref.stage_id]
                stage_output = self._invoke_stage(
                    ref,
                    group_input_state,
                    master_gradient=master_gradient,
                    policy=policy,
                )

                if stage_output is None:
                    per_stage_outcomes.append(
                        {
                            "stage_id": ref.stage_id,
                            "status": "no_op",
                            "emitted_keys": [],
                        }
                    )
                    continue
                if not isinstance(stage_output, Mapping):
                    raise BoostingPipelineError(
                        f"Stage id={ref.stage_id!r} returned "
                        f"{type(stage_output).__name__}; expected a Mapping "
                        "(dict-like) or None."
                    )

                if self._pareto_rejects(
                    stage_output=stage_output,
                    state=state,
                    stage_id=ref.stage_id,
                    per_stage_outcomes=per_stage_outcomes,
                    rejected=rejected,
                ):
                    continue

                self._merge_stage_output(
                    merged_stage_output,
                    dict(stage_output),
                    policy=contract.merge_policy,
                    stage_id=ref.stage_id,
                )
                per_stage_outcomes.append(
                    {
                        "stage_id": ref.stage_id,
                        "status": "accepted",
                        "emitted_keys": sorted(stage_output.keys()),
                    }
                )

            if merged_stage_output:
                new_state = dict(state)
                new_state.update(merged_stage_output)
                state = new_state

        return BoostingPipelineResult(
            final_state=state,
            per_stage_outcomes=tuple(per_stage_outcomes),
            pareto_snapshot=(
                {"descriptor": self.pareto_growth_filter_descriptor}
                if self.pareto_growth_filter is not None
                else None
            ),
            rejected_stages=tuple(rejected),
        )

    def _invoke_stage(
        self,
        ref: PipelineStageRef,
        state: Mapping[str, Any],
        *,
        master_gradient: Any | None,
        policy: Mapping[str, Any] | None,
    ) -> Any:
        contract = _REGISTERED_STAGES[ref.stage_id]
        fn = get_stage_function(ref.stage_id)
        kwargs: dict[str, Any] = {"policy": dict(policy) if policy else {}}
        if contract.sensitivity_weighted:
            # Auto-thread master_gradient when the contract declares
            # sensitivity_weighted=True (per CLAUDE.md H-row autowire).
            kwargs["master_gradient"] = master_gradient
        for k, v in ref.parameters:
            kwargs[k] = v

        try:
            return fn(state, **kwargs)
        except Exception as exc:
            raise BoostingPipelineError(
                f"Stage id={ref.stage_id!r} raised during pipeline.run: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _pareto_rejects(
        self,
        *,
        stage_output: Mapping[str, Any],
        state: Mapping[str, Any],
        stage_id: str,
        per_stage_outcomes: list[dict[str, Any]],
        rejected: list[str],
    ) -> bool:
        if self.pareto_growth_filter is None:
            return False
        rate = stage_output.get("rate", state.get("rate"))
        distortion = stage_output.get("distortion", state.get("distortion"))
        if rate is None or distortion is None:
            return False
        if self.pareto_growth_filter(float(rate), float(distortion)):
            return False
        rejected.append(stage_id)
        per_stage_outcomes.append(
            {
                "stage_id": stage_id,
                "status": "rejected_by_pareto_growth_filter",
                "rate": rate,
                "distortion": distortion,
                "pareto_descriptor": self.pareto_growth_filter_descriptor,
            }
        )
        return True

    @staticmethod
    def _merge_stage_output(
        merged_stage_output: dict[str, Any],
        incoming: dict[str, Any],
        *,
        policy: str,
        stage_id: str,
    ) -> None:
        for key, value in incoming.items():
            if key not in merged_stage_output:
                merged_stage_output[key] = value
                continue
            prior = merged_stage_output[key]
            if policy == "last_writer_wins":
                merged_stage_output[key] = value
            elif policy == "first_writer_wins":
                continue
            elif policy == "additive":
                if not isinstance(prior, (int, float)) or not isinstance(
                    value, (int, float)
                ):
                    raise BoostingPipelineError(
                        f"Stage id={stage_id!r} requested additive merge for "
                        f"key={key!r}, but prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are not both numeric."
                    )
                merged_stage_output[key] = prior + value
            elif policy == "concatenate":
                if type(prior) is type(value) and isinstance(
                    prior, (bytes, tuple, list, str)
                ):
                    merged_stage_output[key] = prior + value
                else:
                    raise BoostingPipelineError(
                        f"Stage id={stage_id!r} requested concatenate merge for "
                        f"key={key!r}, but prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are incompatible."
                    )
            elif policy == "explicit":
                raise BoostingPipelineError(
                    f"Stage id={stage_id!r} emits key={key!r} that conflicts "
                    "with a parallel sibling, but merge_policy='explicit' has "
                    "no merge callable in ComposableBoostingPipeline. Rename "
                    "one output key or use a concrete merge policy."
                )
            else:  # contract validation should make this unreachable
                raise BoostingPipelineError(
                    f"Stage id={stage_id!r} has unknown merge_policy={policy!r}"
                )

    # ------------------------------------------------------------------
    # Introspection / equality
    # ------------------------------------------------------------------

    def stage_contracts(self) -> tuple[BoostStageContract, ...]:
        """Return the contracts of every stage in the pipeline (in order)."""
        return tuple(
            _REGISTERED_STAGES[ref.stage_id] for ref in self.stages
        )
