# SPDX-License-Identifier: MIT
"""ComposableInflatePipeline — immutable pipeline-of-passes with operator
composition (`|` sequential, `&` parallel-merge, `@` attach-search).

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5.4 composition primitives:

  - ``A | B`` — sequential: runs A then B (canonical chain)
  - ``A & B`` — parallel-merge: runs both then merges by per-frame policy
  - ``A @ search`` — attach a search strategy from ``tac.search.*`` (when
    that namespace lands; for now the pipeline accepts the attach but the
    search strategy is a descriptor-only placeholder)

Every compose operation returns a NEW immutable pipeline (no mutation, no
surprise side-effects, easy to test). Pipeline objects are JSON-serializable
so the cathedral autopilot can rank candidate pipelines without
instantiating them and the operator can audit ranked candidates as plain
text.

UNIQUE to this namespace per PV-7 + spec §G inflate-time:

  - ``with_inflate_compute_budget(seconds=N)`` — caps cumulative wallclock
    at N <= MAX_INFLATE_COMPUTE_BUDGET_SECONDS (1800.0; the 30-min T4
    ceiling per spec §G). Default uses each pass's own
    ``inflate_compute_budget_seconds`` value (typically 1800.0).

  - ``with_max_frames(n=N)`` — caps the number of frames each pass processes
    (useful for smoke runs that exercise the pipeline without paying for
    the full 1200-frame video decode).

  - NO ``with_rate_budget`` filter — inflate-time passes have
    archive_bytes_added=0 by contract invariant.

Per CLAUDE.md "Beauty, simplicity, and developer experience":
  - immutable construction → no mid-build state races
  - all errors at .build()/.run() surface AmbiguousCompositionError or
    InflateTimePipelineError with named conflicting passes
  - JSON-serializable representation for cathedral autopilot ranking

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this pipeline does NOT
import tac.compress_time_optimization.pipeline. Sister namespaces are
structurally independent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from tac.inflate_time_post_processing.contract import (
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
)
from tac.inflate_time_post_processing.decorator import (
    _REGISTERED_PASSES,
    get_pass_function,
)
from tac.inflate_time_post_processing.errors import (
    AmbiguousCompositionError,
    InflateBudgetExceededError,
    InflateTimePipelineError,
)

__all__ = [
    "ComposableInflatePipeline",
    "InflateTimePipelineResult",
    "PipelineStageRef",
]

if TYPE_CHECKING:
    from tac.inflate_time_post_processing.contract import (
        InflateTimePostProcessingContract,
    )


@dataclass(frozen=True)
class PipelineStageRef:
    """A single pass reference in a pipeline (id + optional parameters).

    Frozen so pipeline composition is structurally immutable. Pipeline
    operators return new pipelines with new tuples of references — the
    refs themselves are never mutated.

    Mirrors ``tac.compress_time_optimization.pipeline.PipelineStageRef`` at
    the inflate-time-pass surface. Per PV-7: the classes are STRUCTURALLY
    INDEPENDENT (no import / no shared base) so each namespace can evolve
    independently.
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
class InflateTimePipelineResult:
    """Result returned by ``ComposableInflatePipeline.run``.

    Carries the final state dict, the per-pass outcome log, rejected
    passes (wallclock / frame-count filters), and elapsed wallclock totals.
    Frozen for safe consumer-side audit / serialization.
    """

    final_state: Mapping[str, Any]
    per_pass_outcomes: tuple[Mapping[str, Any], ...] = ()
    rejected_passes: tuple[str, ...] = ()
    elapsed_seconds_total: float = 0.0
    frames_processed_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": dict(self.final_state),
            "per_pass_outcomes": [dict(o) for o in self.per_pass_outcomes],
            "rejected_passes": list(self.rejected_passes),
            "elapsed_seconds_total": self.elapsed_seconds_total,
            "frames_processed_total": self.frames_processed_total,
        }


@dataclass(frozen=True)
class ComposableInflatePipeline:
    """Immutable pipeline-of-passes with operator composition.

    Construction is via the canonical ``|`` operator chaining starting from
    an empty pipeline::

        pipeline = (
            ComposableInflatePipeline()
            | "bilateral_denoise_per_frame"
            | "nlm_denoise_per_pair"
            | "learned_post_filter_per_frame"
        )

    Or alternatively from a list (imperative form for callers who prefer
    not to chain)::

        pipeline = ComposableInflatePipeline.from_pass_ids(
            ["bilateral_denoise_per_frame", "nlm_denoise_per_pair"]
        )

    Two inflate-time-unique filters:

        # Cumulative wallclock guardrail per spec §G (30-min T4 ceiling).
        pipeline = pipeline.with_inflate_compute_budget(seconds=1500)

        # Max-frames guardrail for smoke runs.
        pipeline = pipeline.with_max_frames(n=120)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the pipeline carries
    NO hidden state — every behavior is visible via the ``passes`` tuple +
    the budget fields. Two pipelines with equal ``passes`` +
    ``inflate_compute_budget_seconds`` + ``max_frames`` +
    ``search_strategy_descriptor`` are equivalent.
    """

    passes: tuple[PipelineStageRef, ...] = ()
    # Wallclock-side guardrail (inflate-time): default uses the per-pass
    # contract value (typically 1800.0). MUST NOT exceed
    # MAX_INFLATE_COMPUTE_BUDGET_SECONDS.
    inflate_compute_budget_seconds: float | None = None
    # Frame-count guardrail: None = unbounded (process every frame).
    max_frames: int | None = None
    # Search strategy attached via `@`. None when no strategy is attached.
    search_strategy_descriptor: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pass_ids(
        cls, pass_ids: list[str], **kwargs: Any
    ) -> ComposableInflatePipeline:
        """Build a pipeline from a flat list of pass ids (imperative form).

        Equivalent to
        ``ComposableInflatePipeline() | pass_ids[0] | pass_ids[1] | ...``.
        """
        pipeline = cls(**kwargs)
        for pid in pass_ids:
            pipeline = pipeline | pid
        return pipeline

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposableInflatePipeline:
        """Reconstruct a pipeline from a JSON-deserialized dict."""
        passes = tuple(
            PipelineStageRef.from_dict(s) for s in data.get("passes", [])
        )
        return cls(
            passes=passes,
            inflate_compute_budget_seconds=data.get(
                "inflate_compute_budget_seconds"
            ),
            max_frames=data.get("max_frames"),
            search_strategy_descriptor=data.get("search_strategy_descriptor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": [s.to_dict() for s in self.passes],
            "inflate_compute_budget_seconds": (
                self.inflate_compute_budget_seconds
            ),
            "max_frames": self.max_frames,
            "search_strategy_descriptor": self.search_strategy_descriptor,
        }

    def to_json(self) -> str:
        """JSON-serialize the pipeline (sorted keys for byte-stable output)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        """Human-readable representation usable in operator audit + log lines."""
        if not self.passes:
            return "ComposableInflatePipeline(<empty>)"
        chain = " | ".join(s.pass_id for s in self.passes)
        suffix = ""
        if self.inflate_compute_budget_seconds is not None:
            suffix += (
                f".with_inflate_compute_budget("
                f"seconds={self.inflate_compute_budget_seconds})"
            )
        if self.max_frames is not None:
            suffix += f".with_max_frames(n={self.max_frames})"
        if self.search_strategy_descriptor is not None:
            suffix += f" @ {self.search_strategy_descriptor}"
        return f"ComposableInflatePipeline({chain}){suffix}"

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(
        self, pass_ref: str | PipelineStageRef
    ) -> ComposableInflatePipeline:
        """Sequential composition (`A | B` runs A then B).

        Accepts either a bare pass id (str) or a fully-formed
        PipelineStageRef (the latter is used by the search namespace to
        attach parameters).

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
    ) -> ComposableInflatePipeline:
        """Parallel-merge composition (`A & B` runs both then merges by
        per-frame policy declared in each pass's contract).

        Per spec §5.4: parallel-merge passes execute side-by-side and their
        emit dicts are merged per the incoming sibling pass's
        ``merge_policy``. The operator constructs a paired PipelineStageRef
        with composition_kind ``"parallel"`` so the runtime knows to
        dispatch the prior pass and this pass simultaneously.

        Operator-precedence note: Python evaluates `&` BEFORE `|`, so
        ``A | B & C`` parses as ``A | (B & C)``. Use parentheses for the
        common pattern ``(A | B) & C``.
        """
        if not self.passes:
            raise InflateTimePipelineError(
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
    ) -> ComposableInflatePipeline:
        """Attach a search strategy (`pipeline @ "cma_es_over_bilateral_sigma"`).

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
            raise InflateTimePipelineError(
                f"`@` (attach search) requires a non-empty descriptor string; "
                f"got {search_descriptor!r}"
            )
        return replace(self, search_strategy_descriptor=search_descriptor)

    def with_inflate_compute_budget(
        self, *, seconds: float | None
    ) -> ComposableInflatePipeline:
        """Attach a cumulative wallclock guardrail (spec §G 30-min ceiling).

        Per spec §G: inflate-time has a 30-min hard ceiling on T4. Each
        pass declares its own ``max_wallclock_seconds`` on the contract;
        this filter caps the CUMULATIVE total across all passes. Passes
        are rejected at run time when their elapsed wallclock would push
        the cumulative total above ``seconds``.

        Pass ``seconds=None`` to clear the budget (defaults back to the
        per-pass contract values + MAX_INFLATE_COMPUTE_BUDGET_SECONDS).
        """
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(
                seconds, (int, float)
            ):
                raise InflateTimePipelineError(
                    f"with_inflate_compute_budget seconds={seconds!r} must "
                    f"be None or numeric"
                )
            if seconds <= 0:
                raise InflateTimePipelineError(
                    f"with_inflate_compute_budget seconds={seconds} must be > 0"
                )
            if seconds > MAX_INFLATE_COMPUTE_BUDGET_SECONDS:
                raise InflateTimePipelineError(
                    f"with_inflate_compute_budget seconds={seconds} exceeds "
                    f"the 30-min T4 ceiling "
                    f"({MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s) per spec §G."
                )
        return replace(self, inflate_compute_budget_seconds=seconds)

    def with_max_frames(self, *, n: int | None) -> ComposableInflatePipeline:
        """Attach a max-frames guardrail (smoke-run constraint).

        Pass ``n=None`` to clear the constraint (process every frame).
        Useful for cheap smoke runs that exercise pipeline composition
        without paying for full-video frame processing.
        """
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int):
                raise InflateTimePipelineError(
                    f"with_max_frames n={n!r} must be None or int"
                )
            if n < 1:
                raise InflateTimePipelineError(
                    f"with_max_frames n={n} must be >= 1"
                )
        return replace(self, max_frames=n)

    # ------------------------------------------------------------------
    # Build + Run
    # ------------------------------------------------------------------

    def build(self) -> ComposableInflatePipeline:
        """Validate the pipeline's structural correctness without running.

        Surfaces every error class at build time so dispatch never
        encounters a structurally invalid pipeline:

          - Unknown pass id (not registered via @inflate_time_post_filter)
          - Ambiguous emit key (two passes emit same key without explicit
            merge) → AmbiguousCompositionError
          - Cycle in parent_pass_id chain → InflateTimePipelineError
          - Sum of per-pass max_wallclock_seconds exceeds pipeline budget

        Returns self (the pipeline is already immutable; build() is a
        validation pass). The validated pipeline is then safe to .run().
        """
        # 1. Every pass id must be registered
        for ref in self.passes:
            if ref.pass_id not in _REGISTERED_PASSES:
                raise InflateTimePipelineError(
                    f"Pipeline references pass id={ref.pass_id!r} which is "
                    f"not registered via @inflate_time_post_filter. "
                    f"Registered ids: {sorted(_REGISTERED_PASSES)}"
                )

        # 2. Detect ambiguous emit keys (mirror sister algorithm).
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
                    raise InflateTimePipelineError(
                        f"Cycle detected in parent_pass_id chain starting "
                        f"from pass id={contract.id!r}: cycle through "
                        f"{cursor!r}"
                    )
                seen.add(cursor)
                parent_contract = _REGISTERED_PASSES.get(cursor)
                if parent_contract is None:
                    raise InflateTimePipelineError(
                        f"Pass id={contract.id!r} declares "
                        f"parent_pass_id={cursor!r} which is not registered. "
                        f"Either register the parent OR set "
                        f"parent_pass_id=None."
                    )
                cursor = parent_contract.parent_pass_id

        # 4. Sum-of-per-pass-wallclock budget guard.
        # If pipeline.inflate_compute_budget_seconds is set, the sum of
        # contract.max_wallclock_seconds across all sequential passes
        # MUST NOT exceed it. (Parallel passes still consume cumulative
        # wallclock since they share the same provider GPU; we sum them.)
        if self.inflate_compute_budget_seconds is not None:
            total = sum(
                _REGISTERED_PASSES[ref.pass_id].max_wallclock_seconds
                for ref in self.passes
            )
            if total > self.inflate_compute_budget_seconds:
                raise InflateTimePipelineError(
                    f"Sum of per-pass max_wallclock_seconds ({total:.1f}s) "
                    f"exceeds the pipeline's inflate_compute_budget_seconds "
                    f"({self.inflate_compute_budget_seconds}s). Either lower "
                    f"a pass's max_wallclock_seconds OR raise the pipeline "
                    f"budget (up to "
                    f"{MAX_INFLATE_COMPUTE_BUDGET_SECONDS}s 30-min T4 ceiling)."
                )

        return self

    def run(
        self,
        decoded_frames: Mapping[str, Any] | None = None,
        *,
        scorer_surrogate: Any | None = None,
        policy: Mapping[str, Any] | None = None,
        wallclock_strict: bool = False,
    ) -> InflateTimePipelineResult:
        """Execute the pipeline left-to-right against ``decoded_frames``.

        For each pass:
          1. Resolve the registered pass function from the decorator registry
          2. Invoke ``fn(state, scorer_surrogate=..., policy=..., seed=...)``;
             the pass returns a new state dict (additive merge with prior
             state)
          3. Check the cumulative elapsed seconds against the budget; reject
             the pass if it would exceed
          4. Check the per-pass frames-processed count against max_frames;
             pass it as a kwarg so the pass body can respect it

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
          - state is a plain dict (no hidden ABC)
          - the function signature is uniform across all passes
          - rejected passes are RECORDED, not silently dropped, so the
            operator can audit the rejection log

        wallclock_strict=True: when a pass exceeds the cumulative wallclock
        budget, raise InflateBudgetExceededError instead of recording and
        continuing. Default False (continue with rejection record).

        The default behavior is OPT-IN persistence — callers that want
        outcomes appended to the canonical
        ``.omx/state/inflate_time_post_processing_pass_outcomes.jsonl``
        ledger must wrap the call in
        ``persistence.append_pass_outcome_locked(...)`` per CLAUDE.md
        Catalog #128/#131 sister discipline (the namespace does not
        auto-persist).
        """
        self.build()

        state: dict[str, Any] = (
            dict(decoded_frames) if decoded_frames is not None else {}
        )
        per_pass_outcomes: list[dict[str, Any]] = []
        rejected: list[str] = []
        elapsed_total = 0.0
        frames_processed_total = 0

        # Resolve effective wallclock budget.
        effective_budget = self.inflate_compute_budget_seconds
        if effective_budget is None:
            effective_budget = MAX_INFLATE_COMPUTE_BUDGET_SECONDS

        index = 0
        while index < len(self.passes):
            root_ref = self.passes[index]
            if root_ref.composition_kind == "parallel":
                raise InflateTimePipelineError(
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

            group_input_state = dict(state)
            merged_pass_output: dict[str, Any] = {}

            for ref in group_refs:
                contract = _REGISTERED_PASSES[ref.pass_id]
                start = time.monotonic()
                pass_output = self._invoke_pass(
                    ref,
                    group_input_state,
                    scorer_surrogate=scorer_surrogate,
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
                    raise InflateTimePipelineError(
                        f"Pass id={ref.pass_id!r} returned "
                        f"{type(pass_output).__name__}; expected a Mapping "
                        f"(dict-like) or None."
                    )

                # Wallclock budget check (cumulative across all passes).
                projected_total = elapsed_total + elapsed_this
                if projected_total > effective_budget:
                    if wallclock_strict:
                        raise InflateBudgetExceededError(
                            f"Pass id={ref.pass_id!r} would push cumulative "
                            f"elapsed seconds to {projected_total:.2f}s, "
                            f"exceeding inflate-time budget of "
                            f"{effective_budget}s (30-min T4 ceiling per spec "
                            f"§G + Catalog #167 smoke-before-full discipline)."
                        )
                    rejected.append(ref.pass_id)
                    per_pass_outcomes.append(
                        {
                            "pass_id": ref.pass_id,
                            "status": "rejected_by_inflate_compute_budget",
                            "elapsed_seconds": elapsed_this,
                            "cumulative_seconds_projected": projected_total,
                            "inflate_compute_budget_seconds": effective_budget,
                        }
                    )
                    continue

                # Frame-count tracking (informational; passes are
                # responsible for honoring max_frames internally via the
                # kwarg threaded through _invoke_pass).
                frames_added = 0
                if "frames_processed" in pass_output:
                    try:
                        frames_added = int(pass_output["frames_processed"])
                    except (TypeError, ValueError) as exc:
                        raise InflateTimePipelineError(
                            f"Pass id={ref.pass_id!r} returned "
                            f"frames_processed="
                            f"{pass_output['frames_processed']!r} which is "
                            f"not coercible to int."
                        ) from exc

                self._merge_pass_output(
                    merged_pass_output,
                    dict(pass_output),
                    policy=contract.merge_policy,
                    pass_id=ref.pass_id,
                )
                frames_processed_total += frames_added
                elapsed_total += elapsed_this
                per_pass_outcomes.append(
                    {
                        "pass_id": ref.pass_id,
                        "status": "accepted",
                        "emitted_keys": sorted(pass_output.keys()),
                        "elapsed_seconds": elapsed_this,
                        "frames_processed": frames_added,
                        "cumulative_seconds": elapsed_total,
                    }
                )

            if merged_pass_output:
                new_state = dict(state)
                new_state.update(merged_pass_output)
                state = new_state

        return InflateTimePipelineResult(
            final_state=state,
            per_pass_outcomes=tuple(per_pass_outcomes),
            rejected_passes=tuple(rejected),
            elapsed_seconds_total=elapsed_total,
            frames_processed_total=frames_processed_total,
        )

    def _invoke_pass(
        self,
        ref: PipelineStageRef,
        state: Mapping[str, Any],
        *,
        scorer_surrogate: Any | None,
        policy: Mapping[str, Any] | None,
    ) -> Any:
        contract = _REGISTERED_PASSES[ref.pass_id]
        fn = get_pass_function(ref.pass_id)
        kwargs: dict[str, Any] = {"policy": dict(policy) if policy else {}}
        if contract.requires_scorer_surrogate:
            # Auto-thread the scorer surrogate (per Catalog #527) for passes
            # that ranked variants via a CPU-trained Hinton-distilled
            # surrogate. The contest scorer is NEVER threaded — that's
            # forbidden by ScorerAccessForbiddenError.
            kwargs["scorer_surrogate"] = scorer_surrogate
        if contract.seed is not None:
            kwargs["seed"] = contract.seed
        if self.max_frames is not None:
            kwargs["max_frames"] = self.max_frames
        for k, v in ref.parameters:
            kwargs[k] = v

        try:
            return fn(state, **kwargs)
        except Exception as exc:
            raise InflateTimePipelineError(
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
                    raise InflateTimePipelineError(
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
                    raise InflateTimePipelineError(
                        f"Pass id={pass_id!r} requested concatenate merge for "
                        f"key={key!r}, but prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are incompatible."
                    )
            elif policy == "explicit":
                raise InflateTimePipelineError(
                    f"Pass id={pass_id!r} emits key={key!r} that conflicts "
                    "with a parallel sibling, but merge_policy='explicit' has "
                    "no merge callable in ComposableInflatePipeline. Rename "
                    "one output key or use a concrete merge policy."
                )
            else:  # contract validation should make this unreachable
                raise InflateTimePipelineError(
                    f"Pass id={pass_id!r} has unknown merge_policy={policy!r}"
                )

    # ------------------------------------------------------------------
    # Introspection / equality
    # ------------------------------------------------------------------

    def pass_contracts(self) -> tuple[InflateTimePostProcessingContract, ...]:
        """Return the contracts of every pass in the pipeline (in order)."""
        return tuple(
            _REGISTERED_PASSES[ref.pass_id] for ref in self.passes
        )
