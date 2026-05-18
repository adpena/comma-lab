# SPDX-License-Identifier: MIT
"""ComposableSideInfoPipeline — immutable pipeline-of-bakers with operator
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

UNIQUE to this namespace per PV-7 + spec §J:
  - ``with_archive_budget(bytes=N)`` — rejects bakers whose cumulative
    archive-byte contribution would exceed N (rate-side guardrail).
  - ``with_inflate_runtime_budget(bytes=N)`` — rejects bakers whose
    cumulative inflate-runtime-byte contribution would exceed N (per
    CLAUDE.md HNeRV parity discipline lesson 4 ≤ 100 LOC inflate
    budget; inflate-runtime constants count against the inflate.py size
    budget).

The TWO-BUDGET pattern is the structural distinguishing feature of this
pipeline vs sibling pipelines (compress_time has rate + wallclock; boosting
has bytes_added + decoder_overhead_bytes). The side-information pipeline
must separately budget BOTH archive bytes (which show up in the contest
archive ZIP) AND inflate runtime bytes (which inflate inside inflate.py as
Python constants).

Per CLAUDE.md "Beauty, simplicity, and developer experience":
  - immutable construction → no mid-build state races
  - all errors at .build()/.run() surface AmbiguousCompositionError or
    SideInfoPipelineError with named conflicting bakers
  - JSON-serializable representation for cathedral autopilot ranking

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this pipeline does
NOT import sibling pipelines. Sister namespaces are structurally
independent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from tac.side_information.decorator import (
    _REGISTERED_BAKERS,
    get_baker_function,
)
from tac.side_information.errors import (
    AmbiguousCompositionError,
    InflateRuntimeBudgetExceededError,
    SideInfoArchiveBudgetViolation,
    SideInfoPipelineError,
)

__all__ = [
    "ComposableSideInfoPipeline",
    "PipelineBakerRef",
    "SideInfoPipelineResult",
]

if TYPE_CHECKING:
    from tac.side_information.contract import SideInfoBakerContract


@dataclass(frozen=True)
class PipelineBakerRef:
    """A single baker reference in a pipeline (id + optional parameters).

    Frozen so pipeline composition is structurally immutable. Pipeline
    operators return new pipelines with new tuples of references — the
    refs themselves are never mutated.

    Mirrors ``tac.compress_time_optimization.pipeline.PipelineStageRef`` at
    the side-info-baker surface. Per PV-7: the two classes are
    STRUCTURALLY INDEPENDENT (no import / no shared base) so each
    namespace can evolve independently.
    """

    baker_id: str
    parameters: tuple[tuple[str, Any], ...] = ()
    # Each PipelineBakerRef has a kind that mirrors how it was added:
    #   - "sequential": added via `|`
    #   - "parallel": added via `&` (paired with prior baker by position)
    #   - "search_attached": added via `@`; the parameters tuple carries
    #     the search strategy descriptor
    composition_kind: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baker_id": self.baker_id,
            "parameters": list(self.parameters),
            "composition_kind": self.composition_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineBakerRef:
        params = tuple((k, v) for k, v in data.get("parameters", []))
        return cls(
            baker_id=data["baker_id"],
            parameters=params,
            composition_kind=data.get("composition_kind", "sequential"),
        )


@dataclass(frozen=True)
class SideInfoPipelineResult:
    """Result returned by ``ComposableSideInfoPipeline.run``.

    Carries the final state dict, the per-baker outcome log, rejected
    bakers (archive / inflate-runtime / pareto filters), and elapsed
    wallclock totals. Frozen for safe consumer-side audit / serialization.

    Per the §J pattern (both phases): the result records both
    ``cumulative_archive_bytes_added`` (bytes that go into the archive ZIP)
    and ``cumulative_inflate_runtime_bytes_added`` (bytes baked into
    inflate.py constants) separately so the operator can audit each
    budget envelope per CLAUDE.md "Beauty, simplicity, and developer
    experience".
    """

    final_state: Mapping[str, Any]
    per_baker_outcomes: tuple[Mapping[str, Any], ...] = ()
    rejected_bakers: tuple[str, ...] = ()
    elapsed_seconds_total: float = 0.0
    cumulative_archive_bytes_added: int = 0
    cumulative_inflate_runtime_bytes_added: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_state": dict(self.final_state),
            "per_baker_outcomes": [dict(o) for o in self.per_baker_outcomes],
            "rejected_bakers": list(self.rejected_bakers),
            "elapsed_seconds_total": self.elapsed_seconds_total,
            "cumulative_archive_bytes_added": (
                self.cumulative_archive_bytes_added
            ),
            "cumulative_inflate_runtime_bytes_added": (
                self.cumulative_inflate_runtime_bytes_added
            ),
        }


@dataclass(frozen=True)
class ComposableSideInfoPipeline:
    """Immutable pipeline-of-bakers with operator composition.

    Construction is via the canonical ``|`` operator chaining starting from
    an empty pipeline::

        pipeline = (
            ComposableSideInfoPipeline()
            | "scorer_weights_segnet_class_summary"
            | "comma2k19_chroma_palette_k16"
            | "wyner_ziv_residual_per_pair"
        )

    Or alternatively from a list (imperative form for callers who prefer
    not to chain)::

        pipeline = ComposableSideInfoPipeline.from_baker_ids(
            ["scorer_weights_segnet_class_summary",
             "comma2k19_chroma_palette_k16"]
        )

    Two side-info-unique filters:

        # Archive-side guardrail: reject any baker whose contribution
        # would push cumulative archive bytes above 4 KB (typical
        # Wyner-Ziv residual budget).
        pipeline = pipeline.with_archive_budget(bytes=4_000)

        # Inflate-runtime-side guardrail per HNeRV parity L4 ≤ 100 LOC
        # inflate budget. Default is None (unbounded); operators that
        # want to bound the inflate-runtime size opt in.
        pipeline = pipeline.with_inflate_runtime_budget(bytes=1_024)

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": the pipeline carries
    NO hidden state — every behavior is visible via the ``bakers`` tuple
    + the two budget fields + the search strategy descriptor. Two
    pipelines with equal fields are equivalent.
    """

    bakers: tuple[PipelineBakerRef, ...] = ()
    # Archive-side guardrail: None = unbounded
    archive_budget_bytes: int | None = None
    # Inflate-runtime-side guardrail per HNeRV parity L4. None = unbounded.
    inflate_runtime_budget_bytes: int | None = None
    # Search strategy attached via `@`. None when no strategy is attached.
    search_strategy_descriptor: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_baker_ids(
        cls, baker_ids: list[str], **kwargs: Any
    ) -> ComposableSideInfoPipeline:
        """Build a pipeline from a flat list of baker ids (imperative form).

        Equivalent to
        ``ComposableSideInfoPipeline() | baker_ids[0] | baker_ids[1] | ...``.
        """
        pipeline = cls(**kwargs)
        for bid in baker_ids:
            pipeline = pipeline | bid
        return pipeline

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposableSideInfoPipeline:
        """Reconstruct a pipeline from a JSON-deserialized dict."""
        bakers = tuple(
            PipelineBakerRef.from_dict(s) for s in data.get("bakers", [])
        )
        return cls(
            bakers=bakers,
            archive_budget_bytes=data.get("archive_budget_bytes"),
            inflate_runtime_budget_bytes=data.get(
                "inflate_runtime_budget_bytes"
            ),
            search_strategy_descriptor=data.get("search_strategy_descriptor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bakers": [s.to_dict() for s in self.bakers],
            "archive_budget_bytes": self.archive_budget_bytes,
            "inflate_runtime_budget_bytes": self.inflate_runtime_budget_bytes,
            "search_strategy_descriptor": self.search_strategy_descriptor,
        }

    def to_json(self) -> str:
        """JSON-serialize the pipeline (sorted keys for byte-stable output)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def __str__(self) -> str:
        """Human-readable representation usable in operator audit + log lines."""
        if not self.bakers:
            return "ComposableSideInfoPipeline(<empty>)"
        chain = " | ".join(s.baker_id for s in self.bakers)
        suffix = ""
        if self.archive_budget_bytes is not None:
            suffix += (
                f".with_archive_budget(bytes={self.archive_budget_bytes})"
            )
        if self.inflate_runtime_budget_bytes is not None:
            suffix += (
                f".with_inflate_runtime_budget("
                f"bytes={self.inflate_runtime_budget_bytes})"
            )
        if self.search_strategy_descriptor is not None:
            suffix += f" @ {self.search_strategy_descriptor}"
        return f"ComposableSideInfoPipeline({chain}){suffix}"

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(
        self, baker_ref: str | PipelineBakerRef
    ) -> ComposableSideInfoPipeline:
        """Sequential composition (`A | B` runs A then B).

        Accepts either a bare baker id (str) or a fully-formed
        PipelineBakerRef (the latter is used by the search namespace to
        attach parameters).

        Returns a NEW pipeline; the original is unchanged.
        """
        ref = (
            baker_ref
            if isinstance(baker_ref, PipelineBakerRef)
            else PipelineBakerRef(
                baker_id=baker_ref, composition_kind="sequential"
            )
        )
        return replace(self, bakers=(*self.bakers, ref))

    def __and__(
        self, baker_ref: str | PipelineBakerRef
    ) -> ComposableSideInfoPipeline:
        """Parallel-merge composition (`A & B` runs both then merges by
        per-byte policy declared in each baker's contract).

        Per spec §5.4: parallel-merge bakers execute side-by-side and their
        emit dicts are merged per the incoming sibling baker's
        ``merge_policy``. The operator constructs a paired PipelineBakerRef
        with composition_kind ``"parallel"`` so the runtime knows to
        dispatch the prior baker and this baker simultaneously.

        Operator-precedence note: Python evaluates `&` BEFORE `|`, so
        ``A | B & C`` parses as ``A | (B & C)``. Use parentheses for the
        common pattern ``(A | B) & C``.
        """
        if not self.bakers:
            raise SideInfoPipelineError(
                "`&` (parallel-merge) requires at least one prior baker; "
                "use `|` for the first baker."
            )
        ref = (
            replace(baker_ref, composition_kind="parallel")
            if isinstance(baker_ref, PipelineBakerRef)
            else PipelineBakerRef(
                baker_id=baker_ref, composition_kind="parallel"
            )
        )
        return replace(self, bakers=(*self.bakers, ref))

    def __matmul__(
        self, search_descriptor: str
    ) -> ComposableSideInfoPipeline:
        """Attach a search strategy (`pipeline @ "cma_es_over_palette_K"`).

        Per spec §5.4: the search strategy is a tac.search.* registered
        strategy that sweeps a baker's hyperparameters. This operator
        stores the descriptor; the actual search engine lives in
        tac.search.* (which will land in a future subagent slot per spec
        §5.2).

        For now the pipeline accepts the attach but execution is a no-op
        (the search is not yet wired). The descriptor IS serialized through
        to_json so the cathedral autopilot ranker can see + rank pipelines
        with attached strategies.
        """
        if (
            not isinstance(search_descriptor, str)
            or not search_descriptor.strip()
        ):
            raise SideInfoPipelineError(
                f"`@` (attach search) requires a non-empty descriptor "
                f"string; got {search_descriptor!r}"
            )
        return replace(self, search_strategy_descriptor=search_descriptor)

    def with_archive_budget(
        self, *, bytes: int | None
    ) -> ComposableSideInfoPipeline:
        """Attach an archive-byte guardrail.

        Per spec §J: each baker's emitted archive-byte contribution must
        keep the cumulative archive-byte count <= bytes; bakers that
        would exceed the budget are rejected at run time and recorded in
        ``SideInfoPipelineResult.rejected_bakers``.

        Pass ``bytes=None`` to clear the archive budget.
        """
        if bytes is not None:
            if isinstance(bytes, bool) or not isinstance(bytes, int):
                raise SideInfoPipelineError(
                    f"with_archive_budget bytes={bytes!r} must be None or int"
                )
            if bytes < 0:
                raise SideInfoPipelineError(
                    f"with_archive_budget bytes={bytes} must be >= 0"
                )
        return replace(self, archive_budget_bytes=bytes)

    def with_inflate_runtime_budget(
        self, *, bytes: int | None
    ) -> ComposableSideInfoPipeline:
        """Attach an inflate-runtime-byte guardrail.

        Per CLAUDE.md HNeRV parity discipline lesson 4 (≤ 100 LOC inflate
        budget) — inflate-runtime constants (palettes, statistics tables,
        priors baked into inflate.py) count against the inflate.py size
        budget. This filter is opt-in for callers that want to bound the
        cumulative inflate-runtime spend before triggering a full build.

        Bakers are rejected at run time when their cumulative
        ``inflate_runtime_bytes_added`` would push past ``bytes``.

        Pass ``bytes=None`` to clear the inflate-runtime budget.
        """
        if bytes is not None:
            if isinstance(bytes, bool) or not isinstance(bytes, int):
                raise SideInfoPipelineError(
                    f"with_inflate_runtime_budget bytes={bytes!r} must be "
                    f"None or int"
                )
            if bytes < 0:
                raise SideInfoPipelineError(
                    f"with_inflate_runtime_budget bytes={bytes} must be >= 0"
                )
        return replace(self, inflate_runtime_budget_bytes=bytes)

    # ------------------------------------------------------------------
    # Build + Run
    # ------------------------------------------------------------------

    def build(self) -> ComposableSideInfoPipeline:
        """Validate the pipeline's structural correctness without running.

        Surfaces every error class at build time so dispatch never
        encounters a structurally invalid pipeline:

          - Unknown baker id (not registered via @side_info_baker)
          - Ambiguous emit key (two bakers emit same key without explicit
            merge) → AmbiguousCompositionError
          - Cycle in parent_baker_id chain → SideInfoPipelineError

        Returns self (the pipeline is already immutable; build() is a
        validation pass). The validated pipeline is then safe to .run().
        """
        # 1. Every baker id must be registered
        for ref in self.bakers:
            if ref.baker_id not in _REGISTERED_BAKERS:
                raise SideInfoPipelineError(
                    f"Pipeline references baker id={ref.baker_id!r} which "
                    f"is not registered via @side_info_baker. Registered "
                    f"ids: {sorted(_REGISTERED_BAKERS)}"
                )

        # 2. Detect ambiguous emit keys (mirror tac.boosting algorithm).
        # An emit key is ambiguous when two sequential bakers emit it
        # without an intermediate consumer. Parallel-merge bakers
        # (composition_kind="parallel") are EXPECTED to emit overlapping
        # keys (that's the merge), so they are excluded from this check.
        seen_emits: dict[str, str] = {}
        consumed_since_emit: set[str] = set()
        for ref in self.bakers:
            contract = _REGISTERED_BAKERS[ref.baker_id]
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
                        f"intermediate consumer: first by baker "
                        f"{prior!r}, then by baker {ref.baker_id!r}. "
                        f"Either insert a baker that consumes {key!r} "
                        f"between them, OR use `&` (parallel-merge) to "
                        f"declare explicit merge intent, OR rename one "
                        f"baker's emit to {key!r}_v2."
                    )
                seen_emits[key] = ref.baker_id
                consumed_since_emit.discard(key)

        # 3. Cycle detection in parent_baker_id chain
        for ref in self.bakers:
            contract = _REGISTERED_BAKERS[ref.baker_id]
            seen = {contract.id}
            cursor = contract.parent_baker_id
            while cursor is not None:
                if cursor in seen:
                    raise SideInfoPipelineError(
                        f"Cycle detected in parent_baker_id chain starting "
                        f"from baker id={contract.id!r}: cycle through "
                        f"{cursor!r}"
                    )
                seen.add(cursor)
                parent_contract = _REGISTERED_BAKERS.get(cursor)
                if parent_contract is None:
                    raise SideInfoPipelineError(
                        f"Baker id={contract.id!r} declares "
                        f"parent_baker_id={cursor!r} which is not "
                        f"registered. Either register the parent OR set "
                        f"parent_baker_id=None."
                    )
                cursor = parent_contract.parent_baker_id

        return self

    def run(
        self,
        seed_state: Mapping[str, Any] | None = None,
        *,
        master_gradient: Any | None = None,
        policy: Mapping[str, Any] | None = None,
        archive_strict: bool = False,
        inflate_runtime_strict: bool = False,
    ) -> SideInfoPipelineResult:
        """Execute the pipeline left-to-right against ``seed_state``.

        For each baker:
          1. Resolve the registered baker function from the decorator
             registry
          2. Invoke ``fn(state, master_gradient=..., policy=..., seed=...)``;
             the baker returns a new state dict (additive merge with prior
             state)
          3. If an archive budget is attached, check the cumulative
             ``archive_bytes_added`` against the budget; reject the baker
             if it would exceed
          4. If an inflate-runtime budget is attached, check the cumulative
             ``inflate_runtime_bytes_added`` against the budget; reject the
             baker if it would exceed

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
          - state is a plain dict (no hidden ABC)
          - the function signature is uniform across all bakers
          - rejected bakers are RECORDED, not silently dropped, so the
            operator can audit the rejection log

        archive_strict=True: when a baker exceeds the archive-byte budget,
        raise SideInfoArchiveBudgetViolation instead of recording and
        continuing. Default False (continue with rejection record).

        inflate_runtime_strict=True: when a baker exceeds the
        inflate-runtime-byte budget, raise
        InflateRuntimeBudgetExceededError instead of recording. Default
        False.

        The default behavior is OPT-IN persistence — callers that want
        outcomes appended to the canonical
        ``.omx/state/side_information_baker_outcomes.jsonl`` ledger must
        wrap the call in ``persistence.append_baker_outcome_locked(...)``
        per CLAUDE.md Catalog #128/#131 sister discipline (the namespace
        does not auto-persist).
        """
        self.build()

        state: dict[str, Any] = (
            dict(seed_state) if seed_state is not None else {}
        )
        per_baker_outcomes: list[dict[str, Any]] = []
        rejected: list[str] = []
        elapsed_total = 0.0
        cumulative_archive_bytes = 0
        cumulative_inflate_runtime_bytes = 0

        index = 0
        while index < len(self.bakers):
            root_ref = self.bakers[index]
            if root_ref.composition_kind == "parallel":
                raise SideInfoPipelineError(
                    f"Baker id={root_ref.baker_id!r} is marked parallel "
                    "but has no sequential root in this execution group."
                )

            group_refs = [root_ref]
            index += 1
            while (
                index < len(self.bakers)
                and self.bakers[index].composition_kind == "parallel"
            ):
                group_refs.append(self.bakers[index])
                index += 1

            # Parallel siblings must observe the same pre-group input
            # state. Otherwise `A & B` silently degenerates into `A | B`,
            # making stack-of-stacks search order-dependent and
            # non-scientific.
            group_input_state = dict(state)
            merged_baker_output: dict[str, Any] = {}

            for ref in group_refs:
                contract = _REGISTERED_BAKERS[ref.baker_id]
                start = time.monotonic()
                baker_output = self._invoke_baker(
                    ref,
                    group_input_state,
                    master_gradient=master_gradient,
                    policy=policy,
                )
                elapsed_this = time.monotonic() - start

                if baker_output is None:
                    per_baker_outcomes.append(
                        {
                            "baker_id": ref.baker_id,
                            "status": "no_op",
                            "emitted_keys": [],
                            "elapsed_seconds": elapsed_this,
                        }
                    )
                    elapsed_total += elapsed_this
                    continue
                if not isinstance(baker_output, Mapping):
                    raise SideInfoPipelineError(
                        f"Baker id={ref.baker_id!r} returned "
                        f"{type(baker_output).__name__}; expected a "
                        f"Mapping (dict-like) or None."
                    )

                # Determine archive + inflate-runtime byte contributions.
                # Bakers may declare them in the output (per-run dynamic
                # values) OR fall back to the contract's static
                # declarations (for purely deterministic constant-size
                # outputs).
                archive_bytes_this = self._extract_int_field(
                    baker_output,
                    "archive_bytes_added",
                    contract.archive_bytes_added,
                    ref.baker_id,
                )
                inflate_runtime_bytes_this = self._extract_int_field(
                    baker_output,
                    "inflate_runtime_bytes_added",
                    contract.inflate_runtime_bytes_added,
                    ref.baker_id,
                )

                # Archive-byte budget check.
                if (
                    self.archive_budget_bytes is not None
                    and (cumulative_archive_bytes + archive_bytes_this)
                    > self.archive_budget_bytes
                ):
                    if archive_strict:
                        raise SideInfoArchiveBudgetViolation(
                            f"Baker id={ref.baker_id!r} would push "
                            f"cumulative archive bytes to "
                            f"{cumulative_archive_bytes + archive_bytes_this}, "
                            f"exceeding budget of "
                            f"{self.archive_budget_bytes} bytes."
                        )
                    rejected.append(ref.baker_id)
                    per_baker_outcomes.append(
                        {
                            "baker_id": ref.baker_id,
                            "status": "rejected_by_archive_budget",
                            "archive_bytes_added": archive_bytes_this,
                            "cumulative_archive_bytes_projected": (
                                cumulative_archive_bytes + archive_bytes_this
                            ),
                            "archive_budget_bytes": self.archive_budget_bytes,
                            "elapsed_seconds": elapsed_this,
                        }
                    )
                    elapsed_total += elapsed_this
                    continue

                # Inflate-runtime-byte budget check.
                if (
                    self.inflate_runtime_budget_bytes is not None
                    and (
                        cumulative_inflate_runtime_bytes
                        + inflate_runtime_bytes_this
                    )
                    > self.inflate_runtime_budget_bytes
                ):
                    if inflate_runtime_strict:
                        raise InflateRuntimeBudgetExceededError(
                            f"Baker id={ref.baker_id!r} would push "
                            f"cumulative inflate-runtime bytes to "
                            f"{cumulative_inflate_runtime_bytes + inflate_runtime_bytes_this}, "
                            f"exceeding budget of "
                            f"{self.inflate_runtime_budget_bytes} bytes. "
                            f"Per CLAUDE.md HNeRV parity discipline L4 "
                            f"(≤ 100 LOC inflate budget); inflate-runtime "
                            f"constants count against the inflate.py size "
                            f"budget."
                        )
                    rejected.append(ref.baker_id)
                    per_baker_outcomes.append(
                        {
                            "baker_id": ref.baker_id,
                            "status": "rejected_by_inflate_runtime_budget",
                            "inflate_runtime_bytes_added": (
                                inflate_runtime_bytes_this
                            ),
                            "cumulative_inflate_runtime_bytes_projected": (
                                cumulative_inflate_runtime_bytes
                                + inflate_runtime_bytes_this
                            ),
                            "inflate_runtime_budget_bytes": (
                                self.inflate_runtime_budget_bytes
                            ),
                            "elapsed_seconds": elapsed_this,
                        }
                    )
                    elapsed_total += elapsed_this
                    continue

                self._merge_baker_output(
                    merged_baker_output,
                    dict(baker_output),
                    policy=contract.merge_policy,
                    baker_id=ref.baker_id,
                )
                cumulative_archive_bytes += archive_bytes_this
                cumulative_inflate_runtime_bytes += inflate_runtime_bytes_this
                elapsed_total += elapsed_this
                per_baker_outcomes.append(
                    {
                        "baker_id": ref.baker_id,
                        "status": "accepted",
                        "emitted_keys": sorted(baker_output.keys()),
                        "elapsed_seconds": elapsed_this,
                        "archive_bytes_added": archive_bytes_this,
                        "inflate_runtime_bytes_added": (
                            inflate_runtime_bytes_this
                        ),
                        "cumulative_archive_bytes": cumulative_archive_bytes,
                        "cumulative_inflate_runtime_bytes": (
                            cumulative_inflate_runtime_bytes
                        ),
                    }
                )

            if merged_baker_output:
                new_state = dict(state)
                new_state.update(merged_baker_output)
                state = new_state

        return SideInfoPipelineResult(
            final_state=state,
            per_baker_outcomes=tuple(per_baker_outcomes),
            rejected_bakers=tuple(rejected),
            elapsed_seconds_total=elapsed_total,
            cumulative_archive_bytes_added=cumulative_archive_bytes,
            cumulative_inflate_runtime_bytes_added=(
                cumulative_inflate_runtime_bytes
            ),
        )

    def _invoke_baker(
        self,
        ref: PipelineBakerRef,
        state: Mapping[str, Any],
        *,
        master_gradient: Any | None,
        policy: Mapping[str, Any] | None,
    ) -> Any:
        contract = _REGISTERED_BAKERS[ref.baker_id]
        fn = get_baker_function(ref.baker_id)
        kwargs: dict[str, Any] = {"policy": dict(policy) if policy else {}}
        # Auto-thread master_gradient when contract declares
        # hook_sensitivity_contribution=master_gradient_v1 (sensitivity
        # surface) OR scorer_weights_shared_prior_v1 (the
        # ScorerWeightsAsSharedPrior unique hook). This mirrors the
        # tac.boosting H-row autowire pattern.
        if contract.hook_sensitivity_contribution in (
            "master_gradient_v1",
            "scorer_weights_shared_prior_v1",
        ):
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
            raise SideInfoPipelineError(
                f"Baker id={ref.baker_id!r} raised during pipeline.run: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_int_field(
        baker_output: Mapping[str, Any],
        field_name: str,
        contract_default: int,
        baker_id: str,
    ) -> int:
        """Extract a non-negative int field from baker output with a
        contract-level fallback.

        If the baker output explicitly declares the field, the dynamic
        value wins (so per-run sizes are accurate). If absent, the
        contract's static declaration is used (good for fixed-size constant
        bakers). Negative values raise SideInfoPipelineError because the
        budget arithmetic depends on non-negative contributions.
        """
        if field_name in baker_output:
            try:
                value = int(baker_output[field_name])
            except (TypeError, ValueError) as exc:
                raise SideInfoPipelineError(
                    f"Baker id={baker_id!r} returned "
                    f"{field_name}={baker_output[field_name]!r} which is "
                    f"not coercible to int."
                ) from exc
            if value < 0:
                raise SideInfoPipelineError(
                    f"Baker id={baker_id!r} returned {field_name}={value} "
                    f"which is negative; budget arithmetic requires >= 0."
                )
            return value
        return contract_default

    @staticmethod
    def _merge_baker_output(
        merged_baker_output: dict[str, Any],
        incoming: dict[str, Any],
        *,
        policy: str,
        baker_id: str,
    ) -> None:
        for key, value in incoming.items():
            if key not in merged_baker_output:
                merged_baker_output[key] = value
                continue
            prior = merged_baker_output[key]
            if policy == "last_writer_wins":
                merged_baker_output[key] = value
            elif policy == "first_writer_wins":
                continue
            elif policy == "additive":
                if not isinstance(prior, (int, float)) or not isinstance(
                    value, (int, float)
                ):
                    raise SideInfoPipelineError(
                        f"Baker id={baker_id!r} requested additive merge "
                        f"for key={key!r}, but prior={type(prior).__name__}"
                        f" and incoming={type(value).__name__} are not "
                        f"both numeric."
                    )
                merged_baker_output[key] = prior + value
            elif policy == "concatenate":
                if type(prior) is type(value) and isinstance(
                    prior, (bytes, tuple, list, str)
                ):
                    merged_baker_output[key] = prior + value
                else:
                    raise SideInfoPipelineError(
                        f"Baker id={baker_id!r} requested concatenate "
                        f"merge for key={key!r}, but "
                        f"prior={type(prior).__name__} and "
                        f"incoming={type(value).__name__} are incompatible."
                    )
            elif policy == "explicit":
                raise SideInfoPipelineError(
                    f"Baker id={baker_id!r} emits key={key!r} that "
                    "conflicts with a parallel sibling, but "
                    "merge_policy='explicit' has no merge callable in "
                    "ComposableSideInfoPipeline. Rename one output key or "
                    "use a concrete merge policy."
                )
            else:  # contract validation should make this unreachable
                raise SideInfoPipelineError(
                    f"Baker id={baker_id!r} has unknown "
                    f"merge_policy={policy!r}"
                )

    # ------------------------------------------------------------------
    # Introspection / equality
    # ------------------------------------------------------------------

    def baker_contracts(self) -> tuple[SideInfoBakerContract, ...]:
        """Return the contracts of every baker in the pipeline (in order)."""
        return tuple(
            _REGISTERED_BAKERS[ref.baker_id] for ref in self.bakers
        )
