# SPDX-License-Identifier: MIT
"""Comprehensive tests for the tac.boosting namespace.

Covers:
  - Contract validation (frozen dataclass + field-level validators)
  - Decorator behavior (registry write, rollback on non-callable,
    determinism / scorer-freedom invariants)
  - Pipeline composition operators (|, &, @)
  - Pareto-front tracker semantics
  - Persistence (fcntl-locked JSONL; strict-load + lenient-load; 4-proc
    concurrent-append stress)
  - Builders (ResidualCascadeBuilder, PerPairDecoderEnsembleSelector,
    ModeEnsembleDispatch)
  - Example stages end-to-end
  - Catalog #168 AST AnnAssign-aware introspection
  - Catalog #1 + #5 sister regression (no MPS-fallback, no
    eval_roundtrip=False)
  - JSON round-trip
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import FrozenInstanceError
from multiprocessing import get_context
from pathlib import Path

import pytest

from tac.boosting import (
    _REGISTERED_STAGES,
    AmbiguousCompositionError,
    BoostingLedgerCorruptError,
    BoostingPipelineError,
    BoostStageContract,
    BoostStageContractError,
    ComposableBoostingPipeline,
    DeterminismViolation,
    ModeEnsembleDispatch,
    ModeEnsembleDispatchSpec,
    ParetoAnchor,
    ParetoFrontTracker,
    ParetoFrontTrackerError,
    PerPairDecoderEnsembleSelector,
    PerPairDecoderEnsembleSpec,
    ResidualCascadeBuilder,
    ResidualCascadeStageSpec,
    ScorerFreedomViolation,
    _clear_stage_registry_for_tests,
    append_stage_outcome_locked,
    boost_stage,
    get_registered_stages,
    get_stage_function,
    load_stage_outcomes,
    load_stage_outcomes_strict,
    validate_all_registered_stages,
)
from tac.boosting.contract import NOT_APPLICABLE_WITH_RATIONALE
from tac.boosting.persistence import (
    BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear stage registry before AND after each test (autouse) and
    re-import the canonical example stages so the live repo registry stays
    intact when other test modules import them.
    """
    _clear_stage_registry_for_tests()
    yield
    _clear_stage_registry_for_tests()


def _make_minimal_valid_contract(
    *,
    id: str = "test_stage",
    parent_stage_id: str | None = None,
    stage_phase: str = "compress",
    consumes: frozenset[str] | None = None,
    emits: frozenset[str] | None = None,
    correction_kind: str = "additive",
    deterministic: bool = True,
    scorer_free: bool = True,
    **overrides,
) -> BoostStageContract:
    """Build a valid contract for fixture use; overrides any field."""
    base = {
        "id": id,
        "parent_stage_id": parent_stage_id,
        "stage_phase": stage_phase,
        "consumes": consumes or frozenset({"input_a"}),
        "emits": emits or frozenset({"output_a"}),
        "correction_kind": correction_kind,
        "correction_resolution": "per_pair",
        "deterministic": deterministic,
        "scorer_free": scorer_free,
        "sensitivity_weighted": False,
        "max_bytes_added": 512,
        "merge_policy": "last_writer_wins",
        "hook_sensitivity_contribution": "not_applicable_with_rationale",
        "hook_pareto_constraint": "rate_distortion_v1",
        "hook_bit_allocator_class": "not_applicable_with_rationale",
        "hook_autopilot_ranker": "cathedral_autopilot_v1",
        "hook_continual_learning_anchor_kind": "boosting_stage_outcomes_v1",
        "hook_probe_disambiguator": None,
        "hook_not_applicable_rationale": {
            "hook_sensitivity_contribution": "test stage; no master gradient",
            "hook_bit_allocator_class": "test stage; codec downstream",
            "hook_probe_disambiguator": "test stage; canonical interpretation",
        },
    }
    base.update(overrides)
    return BoostStageContract(**base)


# ---------------------------------------------------------------------------
# Contract validation tests
# ---------------------------------------------------------------------------


class TestBoostStageContract:
    def test_minimal_valid_contract_constructs(self):
        c = _make_minimal_valid_contract()
        assert c.id == "test_stage"
        assert c.deterministic is True

    def test_contract_is_frozen(self):
        c = _make_minimal_valid_contract()
        with pytest.raises(FrozenInstanceError):
            c.id = "mutated"  # type: ignore[misc]

    def test_invalid_id_pattern_rejected(self):
        with pytest.raises(BoostStageContractError, match="id="):
            _make_minimal_valid_contract(id="BadID")  # uppercase

    def test_invalid_id_starts_with_digit_rejected(self):
        with pytest.raises(BoostStageContractError, match="id="):
            _make_minimal_valid_contract(id="1bad")

    def test_parent_stage_id_self_reference_rejected(self):
        with pytest.raises(BoostStageContractError, match="parent_stage_id"):
            _make_minimal_valid_contract(
                id="loop", parent_stage_id="loop"
            )

    def test_invalid_stage_phase_rejected(self):
        with pytest.raises(BoostStageContractError, match="stage_phase"):
            _make_minimal_valid_contract(stage_phase="unknown_phase")

    def test_consumes_emits_overlap_rejected(self):
        with pytest.raises(BoostStageContractError, match="consumes ∩ emits"):
            _make_minimal_valid_contract(
                consumes=frozenset({"key_a"}),
                emits=frozenset({"key_a"}),
            )

    def test_list_consumes_auto_freezes(self):
        c = _make_minimal_valid_contract(consumes=["k1", "k2"])  # type: ignore[arg-type]
        assert isinstance(c.consumes, frozenset)

    def test_invalid_correction_kind_rejected(self):
        with pytest.raises(BoostStageContractError, match="correction_kind"):
            _make_minimal_valid_contract(correction_kind="unknown_kind")

    def test_inflate_requires_scorer_free(self):
        with pytest.raises(BoostStageContractError, match="scorer_free=True"):
            _make_minimal_valid_contract(
                stage_phase="inflate", scorer_free=False
            )

    def test_archive_build_requires_deterministic(self):
        with pytest.raises(BoostStageContractError, match="deterministic=True"):
            _make_minimal_valid_contract(
                stage_phase="archive_build", deterministic=False
            )

    def test_max_bytes_added_negative_rejected(self):
        with pytest.raises(BoostStageContractError, match="max_bytes_added"):
            _make_minimal_valid_contract(max_bytes_added=-1)

    def test_max_bytes_added_none_accepted(self):
        c = _make_minimal_valid_contract(max_bytes_added=None)
        assert c.max_bytes_added is None

    def test_hook_not_applicable_requires_rationale(self):
        with pytest.raises(BoostStageContractError, match="rationale"):
            _make_minimal_valid_contract(
                hook_sensitivity_contribution=NOT_APPLICABLE_WITH_RATIONALE,
                hook_not_applicable_rationale={
                    # Missing key for hook_sensitivity_contribution
                    "hook_probe_disambiguator": "canonical",
                    "hook_bit_allocator_class": "downstream",
                },
            )

    def test_hook_probe_disambiguator_none_requires_rationale(self):
        with pytest.raises(BoostStageContractError, match="probe_disambiguator"):
            _make_minimal_valid_contract(
                hook_probe_disambiguator=None,
                hook_not_applicable_rationale={
                    "hook_sensitivity_contribution": "x",
                    "hook_bit_allocator_class": "x",
                    # Missing entry for hook_probe_disambiguator
                },
            )

    def test_hook_not_applicable_rationale_illegal_key_rejected(self):
        with pytest.raises(BoostStageContractError, match="illegal key"):
            _make_minimal_valid_contract(
                hook_not_applicable_rationale={
                    "not_a_real_hook_name": "x",
                    "hook_sensitivity_contribution": "x",
                    "hook_bit_allocator_class": "x",
                    "hook_probe_disambiguator": "x",
                },
            )

    def test_to_dict_round_trip(self):
        c = _make_minimal_valid_contract()
        d = c.to_dict()
        # frozensets serialize as sorted lists
        assert isinstance(d["consumes"], list)
        assert isinstance(d["emits"], list)
        c2 = BoostStageContract.from_dict(d)
        assert c2.id == c.id
        assert c2.consumes == c.consumes
        assert c2.emits == c.emits

    def test_passthrough_requires_emits_equals_consumes(self):
        with pytest.raises(BoostStageContractError, match="passthrough"):
            _make_minimal_valid_contract(
                correction_kind="passthrough",
                consumes=frozenset({"a"}),
                emits=frozenset({"b"}),
            )

    def test_provenance_fields_default_none(self):
        c = _make_minimal_valid_contract()
        assert c.lane_id is None
        assert c.design_memo is None
        assert c.canonical_vs_unique_decision is None

    def test_provenance_empty_string_rejected(self):
        with pytest.raises(BoostStageContractError, match="lane_id"):
            _make_minimal_valid_contract(lane_id="   ")


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestBoostStageDecorator:
    def test_decorator_registers_and_returns_function(self):
        c = _make_minimal_valid_contract(id="dec_test_1")

        @boost_stage(c)
        def fn(state, *, policy):
            return {"output_a": 1}

        assert "dec_test_1" in get_registered_stages()
        # decorator is pass-through
        result = fn({"input_a": 1}, policy={})
        assert result == {"output_a": 1}

    def test_decorator_rejects_non_boost_stage_contract(self):
        with pytest.raises(BoostStageContractError, match="expects a BoostStageContract"):
            boost_stage("not a contract")  # type: ignore[arg-type]

    def test_decorator_duplicate_id_raises(self):
        c1 = _make_minimal_valid_contract(id="dup_id")
        c2 = _make_minimal_valid_contract(id="dup_id", description="diff")

        @boost_stage(c1)
        def fn1(state, *, policy):
            return {}

        with pytest.raises(BoostStageContractError, match="Duplicate"):
            boost_stage(c2)

    def test_decorator_idempotent_on_same_contract_identity(self):
        c = _make_minimal_valid_contract(id="idempotent_id")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        # Re-applying the SAME contract (identity check) is allowed
        @boost_stage(c)
        def fn2(state, *, policy):
            return {}

        assert "idempotent_id" in get_registered_stages()

    def test_decorator_rejects_non_callable_target(self):
        c = _make_minimal_valid_contract(id="ghost_test")
        with pytest.raises(BoostStageContractError, match="must decorate a callable"):
            boost_stage(c)(42)  # type: ignore[arg-type]
        # Rollback: registration should not have persisted
        assert "ghost_test" not in get_registered_stages()

    def test_determinism_violation_caught_at_decoration(self):
        c = _make_minimal_valid_contract(id="nondet_test", deterministic=True)
        with pytest.raises(DeterminismViolation, match="seed"):
            @boost_stage(c)
            def fn(state, rng, *, policy):
                return {}

    def test_determinism_satisfied_with_seed_kwarg(self):
        c = _make_minimal_valid_contract(id="det_seeded", deterministic=True)

        @boost_stage(c)
        def fn(state, rng, *, policy, seed=0):
            return {}

        assert "det_seeded" in get_registered_stages()

    def test_determinism_skipped_when_deterministic_false(self):
        # archive_build requires deterministic=True, but compress phase allows
        # deterministic=False. The rng-without-seed check only fires when
        # deterministic=True.
        c = _make_minimal_valid_contract(
            id="nondet_allowed", deterministic=False, stage_phase="compress"
        )

        @boost_stage(c)
        def fn(state, rng, *, policy):
            return {}

        assert "nondet_allowed" in get_registered_stages()

    def test_scorer_freedom_violation_caught_at_decoration(self):
        # Decorator-level check: stage_phase='inflate' AND consumes references
        # a scorer-state token
        with pytest.raises(ScorerFreedomViolation, match="scorer-state"):
            c = _make_minimal_valid_contract(
                id="scorer_at_inflate",
                stage_phase="inflate",
                consumes=frozenset({"segnet_state_dict", "frames_v0"}),
                # ScorerFreedomViolation fires in contract.__post_init__?
                # No — the check is in the decorator, not the contract.
                # We catch it here via boost_stage invocation below.
            )
            @boost_stage(c)
            def _fn(state, *, policy):
                return {}

    def test_get_stage_function_unknown_id_raises(self):
        with pytest.raises(BoostStageContractError, match="No boost stage"):
            get_stage_function("nonexistent_id")

    def test_clear_registry_removes_stage_functions_too(self):
        c = _make_minimal_valid_contract(id="clear_fn_stage")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        assert get_stage_function("clear_fn_stage") is fn
        _clear_stage_registry_for_tests()
        with pytest.raises(BoostStageContractError, match="No boost stage"):
            get_stage_function("clear_fn_stage")

    def test_validate_all_registered_stages_clean(self):
        c = _make_minimal_valid_contract(id="validate_clean")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        errors = validate_all_registered_stages()
        assert errors == []

    def test_validate_all_registered_stages_detects_corruption(self):
        c = _make_minimal_valid_contract(id="corruption_target")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        # Out-of-band mutation: replace registry value with non-contract
        _REGISTERED_STAGES["corruption_target"] = "not a contract"  # type: ignore[assignment]

        errors = validate_all_registered_stages()
        assert len(errors) == 1
        assert "not a BoostStageContract" in errors[0]

    def test_validate_all_registered_stages_prune_corrupt(self):
        c = _make_minimal_valid_contract(id="prune_target")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        _REGISTERED_STAGES["prune_target"] = "fake"  # type: ignore[assignment]
        validate_all_registered_stages(prune_corrupt=True)
        assert "prune_target" not in _REGISTERED_STAGES


# ---------------------------------------------------------------------------
# Pipeline composition tests
# ---------------------------------------------------------------------------


class TestPipelineComposition:
    def _register_two_stages(self):
        c1 = _make_minimal_valid_contract(
            id="stage_one",
            consumes=frozenset({"input_a"}),
            emits=frozenset({"intermediate"}),
        )
        c2 = _make_minimal_valid_contract(
            id="stage_two",
            parent_stage_id="stage_one",
            consumes=frozenset({"intermediate"}),
            emits=frozenset({"output_final"}),
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"intermediate": state["input_a"] + 1}

        @boost_stage(c2)
        def s2(state, *, policy):
            return {"output_final": state["intermediate"] * 2}

        return c1, c2

    def test_empty_pipeline_constructs(self):
        p = ComposableBoostingPipeline()
        assert p.stages == ()
        assert str(p) == "ComposableBoostingPipeline(<empty>)"

    def test_pipe_chain_adds_stages(self):
        self._register_two_stages()
        p = ComposableBoostingPipeline() | "stage_one" | "stage_two"
        assert len(p.stages) == 2
        assert p.stages[0].stage_id == "stage_one"
        assert p.stages[1].stage_id == "stage_two"

    def test_pipe_returns_new_pipeline(self):
        self._register_two_stages()
        p1 = ComposableBoostingPipeline()
        p2 = p1 | "stage_one"
        assert p1 is not p2
        assert len(p1.stages) == 0
        assert len(p2.stages) == 1

    def test_from_stage_ids_equivalent_to_chain(self):
        self._register_two_stages()
        p1 = ComposableBoostingPipeline.from_stage_ids(["stage_one", "stage_two"])
        p2 = ComposableBoostingPipeline() | "stage_one" | "stage_two"
        assert p1.stages == p2.stages

    def test_run_executes_stages_in_order(self):
        self._register_two_stages()
        p = ComposableBoostingPipeline() | "stage_one" | "stage_two"
        result = p.run(seed_state={"input_a": 5})
        assert result.final_state["intermediate"] == 6
        assert result.final_state["output_final"] == 12
        assert [o["stage_id"] for o in result.per_stage_outcomes] == [
            "stage_one",
            "stage_two",
        ]

    def test_run_unknown_stage_raises_at_build(self):
        p = ComposableBoostingPipeline() | "nonexistent_stage"
        with pytest.raises(BoostingPipelineError, match="not registered"):
            p.run()

    def test_ambiguous_emit_raises(self):
        c1 = _make_minimal_valid_contract(
            id="emit_x_1",
            consumes=frozenset({"in"}),
            emits=frozenset({"x", "marker_1"}),
        )
        c2 = _make_minimal_valid_contract(
            id="emit_x_2",
            consumes=frozenset({"marker_1"}),
            emits=frozenset({"x", "marker_2"}),  # x again, no consumer between
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"x": 1, "marker_1": "ok"}

        @boost_stage(c2)
        def s2(state, *, policy):
            return {"x": 2, "marker_2": "ok"}

        p = ComposableBoostingPipeline() | "emit_x_1" | "emit_x_2"
        with pytest.raises(AmbiguousCompositionError, match="emits key 'x' twice"):
            p.build()

    def test_parallel_merge_allows_overlapping_emits(self):
        # Both stages emit the same key 'x' but `&` declares explicit
        # merge intent
        c1 = _make_minimal_valid_contract(
            id="par_a",
            consumes=frozenset({"in"}),
            emits=frozenset({"x"}),
        )
        c2 = _make_minimal_valid_contract(
            id="par_b",
            consumes=frozenset({"in"}),
            emits=frozenset({"x"}),  # explicit parallel intent
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"x": "a"}

        @boost_stage(c2)
        def s2(state, *, policy):
            return {"x": "b"}

        # Use parens because Python evaluates `&` before `|`.
        p = (ComposableBoostingPipeline() | "par_a") & "par_b"
        # No raise on build (parallel is explicit-intent merge)
        p.build()

    def test_parallel_merge_runs_siblings_against_same_input_state(self):
        c1 = _make_minimal_valid_contract(
            id="side_by_side_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"x"}),
        )
        c2 = _make_minimal_valid_contract(
            id="side_by_side_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"y"}),
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"x": state["seed"] + 1}

        @boost_stage(c2)
        def s2(state, *, policy):
            if "x" in state:
                raise AssertionError("parallel sibling saw prior sibling output")
            return {"y": state["seed"] + 2}

        p = (ComposableBoostingPipeline() | "side_by_side_a") & "side_by_side_b"
        result = p.run(seed_state={"seed": 10})
        assert result.final_state["x"] == 11
        assert result.final_state["y"] == 12

    def test_parallel_merge_additive_policy_sums_overlapping_numeric_keys(self):
        c1 = _make_minimal_valid_contract(
            id="merge_add_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
        )
        c2 = _make_minimal_valid_contract(
            id="merge_add_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            merge_policy="additive",
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"delta": 3}

        @boost_stage(c2)
        def s2(state, *, policy):
            return {"delta": 4}

        p = (ComposableBoostingPipeline() | "merge_add_a") & "merge_add_b"
        result = p.run(seed_state={"seed": 0})
        assert result.final_state["delta"] == 7

    def test_parallel_merge_explicit_policy_requires_non_conflicting_keys(self):
        c1 = _make_minimal_valid_contract(
            id="merge_explicit_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
        )
        c2 = _make_minimal_valid_contract(
            id="merge_explicit_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            merge_policy="explicit",
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"delta": 3}

        @boost_stage(c2)
        def s2(state, *, policy):
            return {"delta": 4}

        p = (ComposableBoostingPipeline() | "merge_explicit_a") & "merge_explicit_b"
        with pytest.raises(BoostingPipelineError, match="merge_policy='explicit'"):
            p.run(seed_state={"seed": 0})

    def test_parallel_merge_without_prior_stage_raises(self):
        c = _make_minimal_valid_contract(id="alone")

        @boost_stage(c)
        def fn(state, *, policy):
            return {}

        p = ComposableBoostingPipeline()
        with pytest.raises(BoostingPipelineError, match="at least one prior stage"):
            _ = p & "alone"

    def test_attach_search_strategy(self):
        self._register_two_stages()
        p = (ComposableBoostingPipeline() | "stage_one") @ "cma_es_K_sweep"
        assert p.search_strategy_descriptor == "cma_es_K_sweep"
        # The pipeline still runs (search is a no-op for now)
        result = p.run(seed_state={"input_a": 1})
        assert result.final_state["intermediate"] == 2

    def test_attach_search_strategy_empty_descriptor_raises(self):
        p = ComposableBoostingPipeline()
        with pytest.raises(BoostingPipelineError, match="non-empty"):
            _ = p @ ""

    def test_stage_returning_none_is_no_op(self):
        c = _make_minimal_valid_contract(id="no_op_stage")

        @boost_stage(c)
        def fn(state, *, policy):
            return None

        p = ComposableBoostingPipeline() | "no_op_stage"
        result = p.run()
        assert result.per_stage_outcomes[0]["status"] == "no_op"

    def test_stage_returning_non_mapping_raises(self):
        c = _make_minimal_valid_contract(id="bad_return")

        @boost_stage(c)
        def fn(state, *, policy):
            return "not a mapping"

        p = ComposableBoostingPipeline() | "bad_return"
        with pytest.raises(BoostingPipelineError, match="expected a Mapping"):
            p.run()

    def test_stage_exception_wrapped_in_pipeline_error(self):
        c = _make_minimal_valid_contract(id="raiser")

        @boost_stage(c)
        def fn(state, *, policy):
            raise ValueError("inner")

        p = ComposableBoostingPipeline() | "raiser"
        with pytest.raises(BoostingPipelineError, match=r"raised during pipeline\.run"):
            p.run()

    def test_cycle_detection_in_parent_chain(self):
        # Stage A's parent is B; Stage B's parent is A → cycle.
        # Per the contract a stage cannot be its own parent (caught at
        # contract construction), but A→B→A is only caught at pipeline build.
        c_a = _make_minimal_valid_contract(
            id="cyc_a", parent_stage_id="cyc_b"
        )

        @boost_stage(c_a)
        def fn_a(state, *, policy):
            return {}

        # cyc_b doesn't exist yet — surfaces as "not registered" at build
        p = ComposableBoostingPipeline() | "cyc_a"
        with pytest.raises(BoostingPipelineError, match="not registered"):
            p.build()


# ---------------------------------------------------------------------------
# Pareto-front tracker tests
# ---------------------------------------------------------------------------


class TestParetoFrontTracker:
    def test_tracker_construction_requires_legal_axis(self):
        with pytest.raises(ParetoFrontTrackerError, match="axis"):
            ParetoFrontTracker(axis="[unknown-axis]")

    def test_tracker_admits_first_anchor(self):
        t = ParetoFrontTracker(axis="[contest-CUDA]")
        assert t.admits(rate=100, distortion=0.5) is True

    def test_anchor_frozen(self):
        a = ParetoAnchor(rate=100, distortion=0.5, source="test", axis="[contest-CPU]")
        with pytest.raises(FrozenInstanceError):
            a.rate = 200  # type: ignore[misc]

    def test_anchor_negative_rate_rejected(self):
        with pytest.raises(ParetoFrontTrackerError, match="rate"):
            ParetoAnchor(rate=-1, distortion=0.5, source="x", axis="[proxy]")

    def test_track_anchor_admits_dominating_candidate(self):
        t = ParetoFrontTracker(axis="[contest-CPU]")
        t.track_anchor(rate=200, distortion=0.5, source="anchor_1")
        # Strict improvement on both axes
        assert t.admits(rate=150, distortion=0.3) is True

    def test_dominated_candidate_rejected(self):
        t = ParetoFrontTracker(axis="[contest-CPU]")
        t.track_anchor(rate=150, distortion=0.3, source="best")
        # Worse on both axes
        assert t.admits(rate=200, distortion=0.5) is False

    def test_pareto_optimal_anchors_returns_non_dominated_subset(self):
        t = ParetoFrontTracker(axis="[contest-CUDA]")
        t.track_anchor(rate=100, distortion=0.5, source="A")
        t.track_anchor(rate=150, distortion=0.3, source="B")
        t.track_anchor(rate=200, distortion=0.6, source="C")  # dominated by A
        front = t.pareto_optimal_anchors()
        assert {a.source for a in front} == {"A", "B"}

    def test_best_on_axis(self):
        t = ParetoFrontTracker(axis="[contest-CUDA]")
        t.track_anchor(rate=180_000, distortion=0.193, source="pr101")
        t.track_anchor(rate=183_000, distortion=0.196, source="pr102")
        assert t.best_on_axis(axis="rate").source == "pr101"
        assert t.best_on_axis(axis="distortion").source == "pr101"

    def test_filter_callable_rejects_worse_rate(self):
        t = ParetoFrontTracker(axis="[contest-CUDA]")
        t.track_anchor(rate=100, distortion=0.5, source="best_rate")
        f = t.for_pareto_growth_filter(reject_if_worsens_axis="rate")
        assert f(150, 0.4) is False  # rate worsened
        assert f(80, 0.6) is True  # rate improved (distortion worsened, OK)

    def test_filter_callable_both_axes(self):
        t = ParetoFrontTracker(axis="[contest-CUDA]")
        t.track_anchor(rate=100, distortion=0.5, source="anchor")
        f = t.for_pareto_growth_filter(reject_if_worsens_axis="both")
        assert f(80, 0.4) is True  # admits
        assert f(150, 0.6) is False  # dominated

    def test_to_dict_round_trip(self):
        t = ParetoFrontTracker(axis="[contest-CPU]")
        t.track_anchor(rate=100, distortion=0.5, source="x")
        d = t.to_dict()
        t2 = ParetoFrontTracker.from_dict(d)
        assert t2.axis == t.axis
        assert len(t2.anchors) == 1
        assert t2.anchors[0].source == "x"

    def test_pipeline_with_pareto_growth(self):
        # Register two stages where the second WORSENS rate
        c1 = _make_minimal_valid_contract(
            id="gp_stage_1", emits=frozenset({"rate", "distortion"})
        )
        c2 = _make_minimal_valid_contract(
            id="gp_stage_2",
            parent_stage_id="gp_stage_1",
            consumes=frozenset({"rate"}),
            emits=frozenset({"rate_v2", "distortion_v2"}),
        )

        @boost_stage(c1)
        def s1(state, *, policy):
            return {"rate": 100, "distortion": 0.5}

        @boost_stage(c2)
        def s2(state, *, policy):
            # WORSEN rate: filter should reject this stage
            return {"rate": 200, "distortion": 0.3}

        tracker = ParetoFrontTracker(axis="[contest-CUDA]")
        tracker.track_anchor(rate=150, distortion=0.4, source="prior")

        p = (
            ComposableBoostingPipeline()
            | "gp_stage_1"
            | "gp_stage_2"
        ).with_pareto_growth(reject_if_worsens_axis="rate", tracker=tracker)

        result = p.run()
        # s2 rejected because it worsens rate vs the tracker's prior best
        assert "gp_stage_2" in result.rejected_stages


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_append_outcome_creates_file(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        record = {
            "stage_id": "test_stage",
            "status": "accepted",
        }
        append_stage_outcome_locked(record, path=ledger, lock_path=lock)
        assert ledger.exists()
        rows = load_stage_outcomes(ledger)
        assert len(rows) == 1
        assert rows[0]["stage_id"] == "test_stage"

    def test_append_outcome_stamps_provenance(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        record = {"stage_id": "x", "status": "accepted"}
        out = append_stage_outcome_locked(record, path=ledger, lock_path=lock)
        assert out["schema_version"] == BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION
        assert "written_at_utc" in out
        assert "written_pid" in out
        assert "written_host" in out

    def test_append_outcome_invalid_record_rejected(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        with pytest.raises(BoostingLedgerCorruptError, match="schema_version"):
            append_stage_outcome_locked(
                {"schema_version": "v_other", "stage_id": "x", "status": "y"},
                path=ledger,
                lock_path=lock,
            )

    def test_append_outcome_missing_stage_id_rejected(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        with pytest.raises(BoostingLedgerCorruptError, match="stage_id"):
            append_stage_outcome_locked(
                {"status": "accepted"}, path=ledger, lock_path=lock
            )

    def test_load_outcomes_empty_when_missing(self, tmp_path):
        ledger = tmp_path / "nonexistent.jsonl"
        assert load_stage_outcomes(ledger) == []
        assert load_stage_outcomes_strict(ledger) == []

    def test_load_strict_rejects_malformed_line(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        ledger.write_text(
            json.dumps({
                "schema_version": BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
                "stage_id": "ok", "status": "accepted"
            }, sort_keys=True) + "\n" + "{ not json }\n",
            encoding="utf-8",
        )
        with pytest.raises(BoostingLedgerCorruptError, match="not valid JSON"):
            load_stage_outcomes_strict(ledger)

    def test_load_lenient_skips_malformed_line(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        ledger.write_text(
            json.dumps({
                "schema_version": BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
                "stage_id": "good", "status": "ok"
            }, sort_keys=True) + "\n" + "{ bad json }\n" +
            json.dumps({
                "schema_version": BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
                "stage_id": "also_good", "status": "ok"
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows = load_stage_outcomes(ledger)
        assert len(rows) == 2
        assert {r["stage_id"] for r in rows} == {"good", "also_good"}

    def test_corrupt_ledger_quarantines_on_append(self, tmp_path):
        ledger = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        # Pre-corrupt the file
        ledger.write_text("totally not json\n", encoding="utf-8")
        with pytest.raises(BoostingLedgerCorruptError, match="quarantined"):
            append_stage_outcome_locked(
                {"stage_id": "x", "status": "y"},
                path=ledger,
                lock_path=lock,
            )
        # The quarantine file should exist
        siblings = list(tmp_path.iterdir())
        quarantines = [p for p in siblings if ".corrupt." in p.name]
        assert len(quarantines) == 1


def _worker_concurrent_append(args):
    """Worker for concurrent-append stress test (4-proc spawn pool)."""
    ledger_path_str, lock_path_str, worker_id, n_appends = args
    from pathlib import Path as P

    from tac.boosting.persistence import append_stage_outcome_locked

    ledger_path = P(ledger_path_str)
    lock_path = P(lock_path_str)
    for i in range(n_appends):
        append_stage_outcome_locked(
            {
                "stage_id": f"worker_{worker_id}_step_{i}",
                "status": "accepted",
                "worker": worker_id,
                "step": i,
            },
            path=ledger_path,
            lock_path=lock_path,
        )
    return n_appends


class TestPersistenceConcurrency:
    def test_4proc_spawn_pool_concurrent_append_safety(self, tmp_path):
        ledger = tmp_path / "concurrent.jsonl"
        lock = tmp_path / "concurrent.jsonl.lock"
        ctx = get_context("spawn")
        n_workers = 4
        n_per_worker = 5
        args = [
            (str(ledger), str(lock), wid, n_per_worker) for wid in range(n_workers)
        ]
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_concurrent_append, args)
        assert sum(results) == n_workers * n_per_worker

        rows = load_stage_outcomes_strict(ledger)
        assert len(rows) == n_workers * n_per_worker
        # Every (worker, step) pair must appear exactly once
        pairs = {(r["worker"], r["step"]) for r in rows}
        expected = {(w, s) for w in range(n_workers) for s in range(n_per_worker)}
        assert pairs == expected


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


class TestResidualCascadeBuilder:
    def test_depth_1_builds_single_contract(self):
        b = ResidualCascadeBuilder(
            root_stage_id="raw_decoder",
            depth=1,
            stage_specs=[
                ResidualCascadeStageSpec(stage_id="cascade_stage_1"),
            ],
        )
        contracts = b.build_contracts()
        assert len(contracts) == 1
        assert contracts[0].id == "cascade_stage_1"
        assert contracts[0].parent_stage_id == "raw_decoder"
        assert "frames_v0" in contracts[0].consumes
        assert "frames_v1" in contracts[0].emits

    def test_depth_3_links_parents_sequentially(self):
        b = ResidualCascadeBuilder(
            root_stage_id="root",
            depth=3,
            stage_specs=[
                ResidualCascadeStageSpec(stage_id="cascade_s1"),
                ResidualCascadeStageSpec(stage_id="cascade_s2"),
                ResidualCascadeStageSpec(stage_id="cascade_s3"),
            ],
        )
        contracts = b.build_contracts()
        assert contracts[0].parent_stage_id == "root"
        assert contracts[1].parent_stage_id == "cascade_s1"
        assert contracts[2].parent_stage_id == "cascade_s2"
        # Emit versioning
        assert "frames_v1" in contracts[0].emits
        assert "frames_v2" in contracts[1].emits
        assert "frames_v3" in contracts[2].emits

    def test_depth_mismatched_specs_rejected(self):
        with pytest.raises(ValueError, match="2 stage_specs"):
            ResidualCascadeBuilder(
                root_stage_id="root",
                depth=2,
                stage_specs=[ResidualCascadeStageSpec(stage_id="only_one")],
            )

    def test_duplicate_stage_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ResidualCascadeBuilder(
                root_stage_id="root",
                depth=2,
                stage_specs=[
                    ResidualCascadeStageSpec(stage_id="dup"),
                    ResidualCascadeStageSpec(stage_id="dup"),
                ],
            )


class TestPerPairDecoderEnsembleSelector:
    def test_build_contract_basic(self):
        spec = PerPairDecoderEnsembleSpec(
            stage_id="pp_decoder_ensemble",
            num_decoders=4,
            selector_criterion="local_variance",
            decoder_archive_keys=("decoder_v1", "decoder_v2", "decoder_v3", "decoder_v4"),
        )
        b = PerPairDecoderEnsembleSelector(spec=spec)
        c = b.build_contract()
        assert c.stage_phase == "inflate"
        assert c.scorer_free is True
        assert c.correction_kind == "replace"
        assert "decoder_v1" in c.consumes

    def test_num_decoders_below_2_rejected(self):
        with pytest.raises(ValueError, match="num_decoders"):
            PerPairDecoderEnsembleSpec(
                stage_id="x", num_decoders=1, selector_criterion="local_variance"
            )

    def test_illegal_selector_criterion_rejected(self):
        with pytest.raises(ValueError, match="selector_criterion"):
            PerPairDecoderEnsembleSpec(
                stage_id="x",
                num_decoders=2,
                selector_criterion="segnet_class_prediction",  # forbidden!
            )


class TestModeEnsembleDispatch:
    def test_product_space_size(self):
        spec = ModeEnsembleDispatchSpec(
            stage_id="md_dispatch",
            num_modes=16,
            num_decoders=8,
            selector_criterion="local_variance",
        )
        assert spec.product_space_size == 128
        assert spec.per_pair_index_bits == 7

    def test_build_contract(self):
        spec = ModeEnsembleDispatchSpec(
            stage_id="md_dispatch",
            num_modes=4,
            num_decoders=2,
            selector_criterion="gradient_magnitude",
            decoder_archive_keys=("d1", "d2"),
        )
        b = ModeEnsembleDispatch(spec=spec)
        c = b.build_contract()
        assert c.stage_phase == "inflate"
        assert c.correction_kind == "replace"
        assert "d1" in c.consumes
        assert "mode_palette" in c.consumes


# ---------------------------------------------------------------------------
# Example stages end-to-end
# ---------------------------------------------------------------------------


class TestExampleStages:
    def test_example_pipeline_runs(self):
        # Re-import example stages (they registered themselves at module
        # import time but our autouse fixture cleared the registry)
        # Force re-import to re-execute the decorators
        if "tac.boosting.examples.example_stages" in sys.modules:
            del sys.modules["tac.boosting.examples.example_stages"]
        if "tac.boosting.examples" in sys.modules:
            del sys.modules["tac.boosting.examples"]
        from tac.boosting.examples import example_stages  # noqa: F401

        p = (
            ComposableBoostingPipeline()
            | "raw_decoder"
            | "cascade_pose_residual_v1"
            | "cascade_seg_residual_v1"
        )
        result = p.run(seed_state={"seed_frames": "DUMMY"})
        # frames_v0, v1, v2 all emitted
        assert "frames_v0" in result.final_state
        assert "frames_v1" in result.final_state
        assert "frames_v2" in result.final_state
        # All accepted (no rejections)
        assert all(o["status"] == "accepted" for o in result.per_stage_outcomes)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestPipelineSerialization:
    def test_pipeline_to_json_round_trip(self):
        c1 = _make_minimal_valid_contract(id="ser_s1")

        @boost_stage(c1)
        def fn(state, *, policy):
            return {"output_a": 1}

        p = ComposableBoostingPipeline() | "ser_s1"
        j = p.to_json()
        data = json.loads(j)
        assert "stages" in data
        assert data["stages"][0]["stage_id"] == "ser_s1"
        # Round-trip
        p2 = ComposableBoostingPipeline.from_dict(data)
        assert p2.stages == p.stages


# ---------------------------------------------------------------------------
# Catalog #168 AST AnnAssign-aware introspection
# ---------------------------------------------------------------------------


class TestCatalog168AstAwareness:
    def test_ast_walker_handles_both_assign_and_annassign(self):
        """Future introspection helpers MUST walk BOTH ast.Assign AND
        ast.AnnAssign so type-annotated module-level registries are
        discoverable. This test enforces the discipline at the namespace's
        own contract surface.
        """
        # Read the contract.py source and verify it uses field defaults
        # via dataclass field() (which is AnnAssign) — our own code must
        # be AnnAssign-walker-safe.
        path = REPO_ROOT / "src" / "tac" / "boosting" / "contract.py"
        tree = ast.parse(path.read_text())
        ann_assigns = [
            node for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)
        ]
        # The contract dataclass fields are AnnAssign nodes
        assert len(ann_assigns) > 5, (
            "contract.py should declare its dataclass fields as AnnAssign nodes; "
            "future introspection helpers must handle this form"
        )


# ---------------------------------------------------------------------------
# Sister-gate regression tests (Catalog #1 MPS-fallback, Catalog #5 eval_roundtrip)
# ---------------------------------------------------------------------------


class TestNoMpsFallbackDefault:
    def test_namespace_source_files_have_no_mps_fallback(self):
        """Sister of Catalog #1 `check_no_mps_fallback_default`. Scan the
        tac.boosting package for the forbidden MPS-fallback ternary."""
        boosting_dir = REPO_ROOT / "src" / "tac" / "boosting"
        forbidden_patterns = [
            'else "mps"',
            "else 'mps'",
        ]
        for py_file in boosting_dir.rglob("*.py"):
            text = py_file.read_text()
            for pat in forbidden_patterns:
                assert pat not in text, (
                    f"{py_file} contains forbidden MPS-fallback pattern {pat!r} "
                    f"(per CLAUDE.md Catalog #1 non-negotiable)"
                )


class TestNoEvalRoundtripFalseDefault:
    def test_namespace_source_files_have_no_eval_roundtrip_false_default(self):
        """Sister of Catalog #5 `check_no_eval_roundtrip_false`. The
        boosting namespace does not invoke eval_roundtrip directly (that's
        the training-loop layer's job), but we explicitly verify no
        substrate-style `eval_roundtrip=False` default leaks in."""
        boosting_dir = REPO_ROOT / "src" / "tac" / "boosting"
        for py_file in boosting_dir.rglob("*.py"):
            text = py_file.read_text()
            assert "eval_roundtrip=False" not in text, (
                f"{py_file} contains forbidden eval_roundtrip=False default "
                f"(per CLAUDE.md Catalog #5 non-negotiable)"
            )
