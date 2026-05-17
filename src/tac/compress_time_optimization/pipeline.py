# SPDX-License-Identifier: MIT
"""ComposableCompressPipeline — immutable pipeline-of-passes with operator
composition (`|` sequential, `&` parallel-merge, `@` attach-search).

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5.4 composition primitives:

  - ``A | B`` — sequential: runs A then B (canonical chain)
  - ``A & B`` — parallel-merge: runs both then merges by per-byte policy
  - ``A @ search`` — attach a search strategy from ``tac.search.*`` (when
    that namespace lands; for now the pipeline accepts the attach but the
    search strategy is a descriptor-only placeholder)

Every compose operation returns a NEW immutable pipeline (no mutation, no
surprise side-effects, easy to test). Pipeline objects are JSON-serializable
so the cathedral autopilot can rank candidate pipelines without
instantiating them and the operator can audit ranked candidates as plain
text.

UNIQUE to this namespace per PV-7 + spec §5.5 + §G compress-time:
  - ``with_rate_budget(bytes=N)`` — rejects passes whose cumulative byte
    contribution would exceed N (rate-side guardrail).
  - ``with_wallclock_budget(seconds=N)`` — rejects passes whose elapsed
    wallclock would exceed N (compress-time smoke-before-full discipline
    per Catalog #167; default None = unbounded per CLAUDE.md §G).

Per CLAUDE.md "Beauty, simplicity, and developer experience":
  - immutable construction → no mid-build state races
  - all errors at .build()/.run() surface AmbiguousCompositionError or
    CompressTimePipelineError with named conflicting passes
  - JSON-serializable representation for cathedral autopilot ranking

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this pipeline does NOT
import tac.boosting.pipeline. Sister namespaces are structurally
independent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from tac.compress_time_optimization.decorator import (
    _REGISTERED_PASSES,
    get_pass_function,
)
from tac.compress_time_optimization.errors import (
    AmbiguousCompositionError,
    CompressTimeBudgetExceededError,
    CompressTimePipelineError,
    RateBudgetViolation,
)

__all__ = [
    "ComposableCompressPipeline",
    "CompressTimePipelineResult",
    "PipelineStageRef",
]

if TYPE_CHECKING:
    from tac.compress_time_optimization.contract import CompressTimePassContract


@dataclass(frozen=True)
class PipelineStageRef:
    """A single pass reference in a pipeline (id + optional parameters).

    Frozen so pipeline composition is structurally immutable. Pipeline
    operators return new pipelines with new tuples of references — the
    refs themselves are never mutated.

    Mirrors ``tac.boosting.pipeline.PipelineStageRef`` at the
    compress-time-pass surface. Per PV-7: the two classes are
    STRUCTURALLY INDEPENDENT (no import / no shared base) so each
    namespace can evolve independently.
    """

    pass_id: str
    parameters: tuple[tuple[str, Any], ...] = ()
    # Each PipelineStageRef has a kind that mirrors how it was added:
    #   - "sequential": added via `|`
    #   - "parallel": added via `&` (paired with prior pass by position)
    #   - "search_attached": added via `@`; the parameters tuple carries
    #     the search strategy descriptor
    composition_kind: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "parameters": list(self.parameters),
            "composition_kind": self.composition_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStageRef:
        params = tuple((k, v) for k, v in data.get("parameters", []))
        return cls(
            pass_id=data["pass_id"],
            parameters=params,
            composition_kind=data.get("composition_kind", "sequential"),
        )


@dataclass(frozen=True)
class CompressTimePipelineResult:
    """Result returned by ``ComposableCompressPipeline.run``.

    Carries the final state dict, the per-pass outcome log, rejected passes
    (rate / wallclock / pareto filters), and elapsed wallclock totals.
    Frozen for safe consumer-side audit / serialization.
    """

    final_state: Mapping[str, Any]
    per_pass_outcomes: tuple[Mapping[str, Any], ...] = ()
    rejected_passes: tuple[str, ...] = ()
    elapsed_seconds_total: float = 0.0
    cumulative_bytes_added: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": dict(self.final_state),
            "per_pass_outcomes": [dict(o) for o in self.per_pass_outcomes],
            "rejected_passes": list(self.rejected_passes),
            "elapsed_seconds_total": self.elapsed_seconds_total,
            "cumulative_bytes_added": self.cumulative_bytes_added,
        }


@dataclass(frozen=True)
class ComposableCompressPipeline:
    """Immutable pipeline-of-passes with operator composition.

    Construction is via the canonical ``|`` operator chaining starting from
    an empty pipeline::

        pipeline = (
            ComposableCompressPipeline()
            | "raw_quant"
            | "sensitivity_weighted_tto_refinement"
            | "iterated_bisection_rate_knee"
        )

    Or alternatively from a list (imperative form for callers who prefer
    not to chain)::

        pipeline = ComposableCompressPipeline.from_pass_ids(
            ["raw_quant", "sensitivity_weighted_tto_refinement"]
        )

    Two compress-time-unique filters:

        # Rate-side guardrail: reject any pass whose contribution would
        # push cumulative archive bytes above 200 KB.
        pipeline = pipeline.with_rate_budget(bytes=200_000)

        # Wallclock-side guardrail per Catalog #167 smoke-before-full.
        # By default compress-time is UNBOUNDED per CLAUDE.md §G; this
        # filter is opt-in only.
        pipeline = pipeline.with_wallclock_budget(seconds=3600)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the pipeline carries
    NO hidden state — every behavior is visible via the ``passes`` tuple
    + the two budget fields. Two pipelines with equal ``passes`` +
    ``rate_budget_bytes`` + ``wallclock_budget_seconds`` +
    ``search_strategy_descriptor`` are equivalent.
    """

    passes: tuple[PipelineStageRef, ...] = ()
    # Rate-side guardrail: None = unbounded
    rate_budget_bytes: int | None = None
    # Wallclock-side guardrail (compress-time): None = UNBOUNDED per CLAUDE.md
    # §G; opt-in only for smoke-before-full discipline (Catalog #167).
    wallclock_budget_seconds: int | None = None
    # Search strategy attached via `@`. None when no strategy is attached.
    # When the tac.search namespace lands the descriptor here will name
    # a registered strategy.
    search_strategy_descriptor: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pass_ids(
        cls, pass_ids: list[str], **kwargs: Any
    ) -> ComposableCompressPipeline:
        """Build a pipeline from a flat list of pass ids (imperative form).

        Equivalent to
        ``ComposableCompressPipeline() | pass_ids[0] | pass_ids[1] | ...``.
        """
        pipeline = cls(**kwargs)
        for pid in pass_ids:
            pipeline = pipeline | pid
        return pipeline

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposableCompressPipeline:
        """Reconstruct a pipeline from a JSON-deserialized dict."""
        passes = tuple(
            PipelineStageRef.from_dict(s) for s in data.get("passes", [])
        )
        return cls(
            passes=passes,
            rate_budget_bytes=data.get("rate_budget_bytes"),
            wallclock_budget_seconds=data.get("wallclock_budget_seconds"),
            search_strategy_descriptor=data.get("search_strategy_descriptor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": [s.to_dict() for s in self.passes],
            "rate_budget_bytes": self.rate_budget_bytes,
            "wallclock_budget_seconds": self.wallclock_budget_seconds,
            "search_strategy_descriptor": self.search_strategy_descriptor,
        }

    def to_json(self) -> str:
        """JSON-serialize the pipeline (sorted keys for byte-stable output)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        """Human-readable representation usable in operator audit + log lines."""
        if not self.passes:
            return "ComposableCompressPipeline(<empty>)"
        chain = " | ".join(s.pass_id for s in self.passes)
        suffix = ""
        if self.rate_budget_bytes is not None:
            suffix += f".with_rate_budget(bytes={self.rate_budget_bytes})"
        if self.wallclock_budget_seconds is not None:
            suffix += (
                f".with_wallclock_budget("
                f"seconds={self.wallclock_budget_seconds})"
            )
        if self.search_strategy_descriptor is not None:
            suffix += f" @ {self.search_strategy_descriptor}"
        return f"ComposableCompressPipeline({chain}){suffix}"

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(
        self, pass_ref: str | PipelineStageRef
    ) -> ComposableCompressPipeline:
        """Sequential composition (`A | B` runs A then B).

        Accepts either a bare pass id (str) or a fully-formed PipelineStageRef
        (the latter is used by the search namespace to attach parameters).

        Returns a NEW pipeline; the original is unchanged.
        """
        ref = (
            pass_ref
            if isinstance(pass_ref, PipelineStageRef)
            else PipelineStageRef(
                pass_id=pass_ref, composition_kind="sequential"
            )
        )
        return replace(self, passes=(*self.passes, ref))

    def __and__(
        self, pass_ref: str | PipelineStageRef
    ) -> ComposableCompressPipeline:
        """Parallel-merge composition (`A & B` runs both then merges by
        per-byte policy declared in each pass's contract).

        Per spec §5.4: parallel-merge passes execute side-by-side and their
        emit dicts are merged per the incoming sibling pass's
        ``merge_policy``. The operator constructs a paired PipelineStageRef
        with composition_kind
        ``"parallel"`` so the runtime knows to dispatch the prior pass and
        this pass simultaneously.

        Operator-precedence note: Python evaluates `&` BEFORE `|`, so
        ``A | B & C`` parses as ``A | (B & C)``. Use parentheses for the
        common pattern ``(A | B) & C``.
        """
        if not self.passes:
            raise CompressTimePipelineError(
                "`&` (parallel-merge) requires at least one prior pass; "
                "use `|` for the first pass."
            )
        ref = (
            replace(pass_ref, composition_kind="parallel")
            if isinstance(pass_ref, PipelineStageRef)
            else PipelineStageRef(
                pass_id=pass_ref, composition_kind="parallel"
            )
        )
        return replace(self, passes=(*self.passes, ref))

    def __matmul__(
        self, search_descriptor: str
    ) -> ComposableCompressPipeline:
        """Attach a search strategy (`pipeline @ "cma_es_over_K_palette"`).

        Per spec §5.4: the search strategy is a tac.search.* registered
        strategy that sweeps a pass's hyperparameters. This operator stores
        the descriptor; the actual search engine lives in tac.search.*
        (which will land in a future subagent slot per spec §5.2).

        For now the pipeline accepts the attach but execution is a no-op
        (the search is not yet wired). The descriptor IS serialized through
        to_json so the cathedral autopilot ranker can see + rank pipelines
        with attached strategies.
        """
        if not isinstance(search_descriptor, str) or not search_descriptor.strip():
            raise CompressTimePipelineError(
                f"`@` (attach search) requires a non-empty descriptor string; "
                f"got {search_descriptor!r}"
            )
        return replace(self, search_strategy_descriptor=search_descriptor)

    def with_rate_budget(
        self, *, bytes: int | None
    ) -> ComposableCompressPipeline:
        """Attach a rate-side guardrail.

        Per spec §5.4: each pass's emitted byte contribution must keep the
        cumulative byte count <= bytes; passes that would exceed the budget
        are rejected at run time and recorded in
        ``CompressTimePipelineResult.rejected_passes``.

        Pass ``bytes=None`` to clear the rate budget.
        """
        if bytes is not None:
            if isinstance(bytes, bool) or not isinstance(bytes, int):
                raise CompressTimePipelineError(
                    f"with_rate_budget bytes={bytes!r} must be None or int"
                )
            if bytes < 0:
                raise CompressTimePipelineError(
                    f"with_rate_budget bytes={bytes} must be >= 0"
                )
        return replace(self, rate_budget_bytes=bytes)

    def with_wallclock_budget(
        self, *, seconds: int | None
    ) -> ComposableCompressPipeline:
        """Attach a wallclock guardrail (Catalog #167 smoke-before-full).

        Per CLAUDE.md §G: compress-time compute is unbounded by default
        (``seconds=None``). This filter is opt-in for callers that want
        to bound the cumulative wallclock spend before triggering a full
        run — useful when chaining many TTO / SA / coordinate-search passes
        whose cumulative cost is hard to predict.

        Passes are rejected at run time when their elapsed wallclock would
        push the cumulative total above ``seconds``.

        Pass ``seconds=None`` to clear the wallclock budget.
        """
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(seconds, int):
                raise CompressTimePipelineError(
                    f"with_wallclock_budget seconds={seconds!r} must be None "
                    f"or int"
                )
            if seconds < 1:
                raise CompressTimePipelineError(
                    f"with_wallclock_budget seconds={seconds} must be >= 1"
                )
        return replace(self, wallclock_budget_seconds=seconds)

    # ------------------------------------------------------------------
    # Build + Run
    # ------------------------------------------------------------------

    def build(self) -> ComposableCompressPipeline:
        """Validate the pipeline's structural correctness without running.

        Surfaces every error class at build time so dispatch never
        encounters a structurally invalid pipeline:

          - Unknown pass id (not registered via @compress_time_pass)
          - Ambiguous emit key (two passes emit same key without explicit
            merge) → AmbiguousCompositionError
          - Cycle in parent_pass_id chain → CompressTimePipelineError

        Returns self (the pipeline is already immutable; build() is a
        validation pass). The validated pipeline is then safe to .run().
        """
        # 1. Every pass id must be registered
        for ref in self.passes:
            if ref.pass_id not in _REGISTERED_PASSES:
                raise CompressTimePipelineError(
                    f"Pipeline references pass id={ref.pass_id!r} which is "
                    f"not registered via @compress_time_pass. Registered ids: "
                    f"{sorted(_REGISTERED_PASSES)}"
                )

        # 2. Detect ambiguous emit keys (mirror tac.boosting algorithm).
        # An emit key is ambiguous when two sequential passes emit it
        # without an intermediate consumer. Parallel-merge passes
        # (composition_kind="parallel") are EXPECTED to emit overlapping
        # keys (that's the merge), so they are excluded from this check.
        seen_emits: dict[str, str] = {}
        consumed_since_emit: set[str] = set()
        for ref in self.passes:
            contract = _REGISTERED_PASSES[ref.pass_id]
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
                        f"intermediate consumer: first by pass "
                        f"{prior!r}, then by pass {ref.pass_id!r}. "
                        f"Either insert a pass that consumes {key!r} "
                        f"between them, OR use `&` (parallel-merge) to "
                        f"declare explicit merge intent, OR rename one "
                        f"pass's emit to {key!r}_v2."
                    )
                seen_emits[key] = ref.pass_id
                consumed_since_emit.discard(key)

        # 3. Cycle detection in parent_pass_id chain
        for ref in self.passes:
            contract = _REGISTERED_PASSES[ref.pass_id]
            seen = {contract.id}
            cursor = contract.parent_pass_id
            while cursor is not None:
                if cursor in seen:
                    raise CompressTimePipelineError(
                        f"Cycle detected in parent_pass_id chain starting "
                        f"from pass id={contract.id!r}: cycle through "
                        f"{cursor!r}"
                    )
                seen.add(cursor)
                parent_contract = _REGISTERED_PASSES.get(cursor)
                if parent_contract is None:
                    raise CompressTimePipelineError(
                        f"Pass id={contract.id!r} declares "
                        f"parent_pass_id={cursor!r} which is not registered. "
                        f"Either register the parent OR set "
                        f"parent_pass_id=None."
                    )
                cursor = parent_contract.parent_pass_id

        return self

    def run(
        self,
        seed_state: Mapping[str, Any] | None = None,
        *,
        master_gradient: Any | None = None,
        policy: Mapping[str, Any] | None = None,
        wallclock_strict: bool = False,
        rate_strict: bool = False,
    ) -> CompressTimePipelineResult:
        """Execute the pipeline left-to-right against ``seed_state``.

        For each pass:
          1. Resolve the registered pass function from the decorator registry
          2. Invoke ``fn(state, master_gradient=..., policy=..., seed=...)``;
             the pass returns a new state dict (additive merge with prior
             state)
          3. If a rate budget is attached, check the cumulative
             ``bytes_added`` against the budget; reject the pass if it
             would exceed
          4. If a wallclock budget is attached, check the cumulative
             elapsed seconds against the budget; reject the pass if it
             would exceed

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
          - state is a plain dict (no hidden ABC)
          - the function signature is uniform across all passes
          - rejected passes are RECORDED, not silently dropped, so the
            operator can audit the rejection log

        wallclock_strict=True: when a pass exceeds the wallclock budget,
        raise CompressTimeBudgetExceededError instead of recording and
        continuing. Default False (continue with rejection record).

        rate_strict=True: when a pass exceeds the rate budget, raise
        RateBudgetViolation instead of recording. Default False.

        The default behavior is OPT-IN persistence — callers that want
        outcomes appended to the canonical
        ``.omx/state/compress_time_optimization_pass_outcomes.jsonl``
        ledger must wrap the call in
        ``persistence.append_pass_outcome_locked(...)`` per CLAUDE.md
        Catalog #128/#131 sister discipline (the namespace does not
        auto-persist).
        """
        self.build()

        state: dict[str, Any] = (
            dict(seed_state) if seed_state is not None else {}
        )
        per_pass_outcomes: list[dict[str, Any]] = []
        rejected: list[str] = []
        elapsed_total = 0.0
        cumulative_bytes = 0

        index = 0
        while index < len(self.passes):
            root_ref = self.passes[index]
            if root_ref.composition_kind == "parallel":
                raise CompressTimePipelineError(
                    f"Pass id={root_ref.pass_id!r} is marked parallel but has "
                    "no sequential root in this execution group."
                )

            group_refs = [root_ref]
            index += 1
            while (
                index < len(self.passes)
                and self.passes[index].composition_kind == "parallel"
            ):
                group_refs.append(self.passes[index])
                index += 1

            # Parallel siblings must observe the same pre-group input state.
            # Otherwise `A & B` silently degenerates into `A | B`, making
            # stack-of-stacks search order-dependent and non-scientific.
            group_input_state = dict(state)
            merged_pass_output: dict[str, Any] = {}

            for ref in group_refs:
                contract = _REGISTERED_PASSES[ref.pass_id]
                start = time.monotonic()
                pass_output = self._invoke_pass(
                    ref,
                    group_input_state,
                    master_gradient=master_gradient,
                    policy=policy,
                )
                elapsed_this = time.monotonic() - start

                if pass_output is None:
                    per_pass_outcomes.append(
                        {
                            "pass_id": ref.pass_id,
                            "status": "no_op",
                            "emitted_keys": [],
                            "elapsed_seconds": elapsed_this,
                        }
                    )
                    elapsed_total += elapsed_this
                    continue
                if not isinstance(pass_output, Mapping):
                    raise CompressTimePipelineError(
                        f"Pass id={ref.pass_id!r} returned "
                        f"{type(pass_output).__name__}; expected a Mapping "
                        f"(dict-like) or None."
                    )

                # Wallclock budget check (cumulative across all accepted
                # passes). Even for logical parallel groups we conservatively
                # charge observed local wallclock so dry runs cannot undercount.
                if self.wallclock_budget_seconds is not None:
                    projected_total = elapsed_total + elapsed_this
                    if projected_total > self.wallclock_budget_seconds:
                        if wallclock_strict:
                            raise CompressTimeBudgetExceededError(
                                f"Pass id={ref.pass_id!r} would push cumulative "
                                f"elapsed seconds to {projected_total:.2f}s, "
                                f"exceeding budget of "
                                f"{self.wallclock_budget_seconds}s. Per Catalog "
                                f"#167 smoke-before-full discipline + the "
                                f"with_wallclock_budget filter."
                            )
                        rejected.append(ref.pass_id)
                        per_pass_outcomes.append(
                            {
                                "pass_id": ref.pass_id,
                                "status": "rejected_by_wallclock_budget",
                                "elapsed_seconds": elapsed_this,
                                "cumulative_seconds_projected": projected_total,
                                "wallclock_budget_seconds": (
                                    self.wallclock_budget_seconds
                                ),
                            }
                        )
                        continue

                # Rate budget check.
                bytes_added_this = 0
                if "bytes_added" in pass_output:
                    try:
                        bytes_added_this = int(pass_output["bytes_added"])
                    except (TypeError, ValueError) as exc:
                        raise CompressTimePipelineError(
                            f"Pass id={ref.pass_id!r} returned "
                            f"bytes_added={pass_output['bytes_added']!r} "
                            f"which is not coercible to int."
                        ) from exc
                if (
                    self.rate_budget_bytes is not None
                    and (cumulative_bytes + bytes_added_this)
                    > self.rate_budget_bytes
                ):
                    if rate_strict:
                        raise RateBudgetViolation(
                            f"Pass id={ref.pass_id!r} would push cumulative "
                            f"archive bytes to "
                            f"{cumulative_bytes + bytes_added_this}, exceeding "
                            f"budget of {self.rate_budget_bytes} bytes."
                        )
                    rejected.append(ref.pass_id)
                    per_pass_outcomes.append(
                        {
                            "pass_id": ref.pass_id,
                            "status": "rejected_by_rate_budget",
                            "bytes_added": bytes_added_this,
                            "cumulative_bytes_projected": (
                                cumulative_bytes + bytes_added_this
                            ),
                            "rate_budget_bytes": self.rate_budget_bytes,
                            "elapsed_seconds": elapsed_this,
                        }
                    )
                    continue

                self._merge_pass_output(
                    merged_pass_output,
                    dict(pass_output),
                    policy=contract.merge_policy,
                    pass_id=ref.pass_id,
                )
                cumulative_bytes += bytes_added_this
                elapsed_total += elapsed_this
                per_pass_outcomes.append(
                    {
                        "pass_id": ref.pass_id,
                        "status": "accepted",
                        "emitted_keys": sorted(pass_output.keys()),
                        "elapsed_seconds": elapsed_this,
                        "bytes_added": bytes_added_this,
                        "cumulative_bytes": cumulative_bytes,
                    }
                )

            if merged_pass_output:
                new_state = dict(state)
                new_state.update(merged_pass_output)
                state = new_state

        return CompressTimePipelineResult(
            final_state=state,
            per_pass_outcomes=tuple(per_pass_outcomes),
            rejected_passes=tuple(rejected),
            elapsed_seconds_total=elapsed_total,
            cumulative_bytes_added=cumulative_bytes,
        )

    def _invoke_pass(
        self,
        ref: PipelineStageRef,
        state: Mapping[str, Any],
        *,
        master_gradient: Any | None,
        policy: Mapping[str, Any] | None,
    ) -> Any:
        contract = _REGISTERED_PASSES[ref.pass_id]
        fn = get_pass_function(ref.pass_id)
        kwargs: dict[str, Any] = {"policy": dict(policy) if policy else {}}
        if contract.sensitivity_weighted:
            # Auto-thread master_gradient when contract declares
            # sensitivity_weighted=True (mirrors tac.boosting H-row
            # autowire pattern).
            kwargs["master_gradient"] = master_gradient
        if contract.seed is not None:
            # Auto-thread the contract-pinned seed for byte-stable
            # reproducibility per Catalog #158.
            kwargs["seed"] = contract.seed
        for k, v in ref.parameters:
            kwargs[k] = v

        try:
            return fn(state, **kwargs)
        except Exception as exc:
            raise CompressTimePipelineError(
                f"Pass id={ref.pass_id!r} raised during pipeline.run: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _merge_pass_output(
        merged_pass_output: dict[str, Any],
        incoming: dict[str, Any],
        *,
        policy: str,
        pass_id: str,
    ) -> None:
        for key, value in incoming.items():
            if key not in merged_pass_output:
                merged_pass_output[key] = value
                continue
            prior = merged_pass_output[key]
            if policy == "last_writer_wins":
                merged_pass_output[key] = value
            elif policy == "first_writer_wins":
                continue
            elif policy == "additive":
                if not isinstance(prior, (int, float)) or not isinstance(
                    value, (int, float)
                ):
                    raise CompressTimePipelineError(
                        f"Pass id={pass_id!r} requested additive merge for "
                        f"key={key!r}, but prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are not both numeric."
                    )
                merged_pass_output[key] = prior + value
            elif policy == "concatenate":
                if type(prior) is type(value) and isinstance(
                    prior, (bytes, tuple, list, str)
                ):
                    merged_pass_output[key] = prior + value
                else:
                    raise CompressTimePipelineError(
                        f"Pass id={pass_id!r} requested concatenate merge for "
                        f"key={key!r}, but prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are incompatible."
                    )
            elif policy == "explicit":
                raise CompressTimePipelineError(
                    f"Pass id={pass_id!r} emits key={key!r} that conflicts "
                    "with a parallel sibling, but merge_policy='explicit' has "
                    "no merge callable in ComposableCompressPipeline. Rename "
                    "one output key or use a concrete merge policy."
                )
            else:  # contract validation should make this unreachable
                raise CompressTimePipelineError(
                    f"Pass id={pass_id!r} has unknown merge_policy={policy!r}"
                )

    # ------------------------------------------------------------------
    # Introspection / equality
    # ------------------------------------------------------------------

    def pass_contracts(self) -> tuple[CompressTimePassContract, ...]:
        """Return the contracts of every pass in the pipeline (in order)."""
        return tuple(
            _REGISTERED_PASSES[ref.pass_id] for ref in self.passes
        )
