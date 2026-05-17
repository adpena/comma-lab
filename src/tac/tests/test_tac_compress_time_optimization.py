# SPDX-License-Identifier: MIT
"""Tests for the tac.compress_time_optimization namespace.

Per CLAUDE.md "Subagent coherence-by-default" + the per-namespace test
discipline: 60-90 tests across contract validation, decorator semantics,
pipeline composition, persistence, builders, serialization, and a
cross-namespace composition stub regression.

Mirror coverage density of the sister tac.boosting namespace (83 tests).

Test organization:
  1. CompressTimePassContract validation + edge cases (~20)
  2. @compress_time_pass decorator + registry (~10)
  3. ComposableCompressPipeline composition + run (~18)
  4. Persistence (fcntl lock + JSONL append + strict load) (~10)
  5. Per-builder smoke tests (~10)
  6. Serialization round-trip (~5)
  7. AST introspection regression (Catalog #168) (~3)
  8. Cross-namespace composition stub (~3)
  9. Examples integration (~3)
"""

from __future__ import annotations

import ast
import json
import multiprocessing as mp
from pathlib import Path

import pytest

from tac.compress_time_optimization import (
    COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION,
    LEGAL_CORRECTION_KIND,
    AmbiguousCompositionError,
    ComposableCompressPipeline,
    CompressTimeBudgetExceededError,
    CompressTimeLedgerCorruptError,
    CompressTimeOptimizationError,
    CompressTimePassContract,
    CompressTimePassContractError,
    CompressTimePipelineError,
    DeterminismViolation,
    GenericTTOHarness,
    GenericTTOHarnessSpec,
    InflatePhaseForbiddenError,
    IteratedBisectionRateKnee,
    IteratedBisectionRateKneeSpec,
    MultipassRefinement,
    MultipassRefinementSpec,
    PerPairCoordinateSearch,
    PerPairCoordinateSearchSpec,
    PipelineStageRef,
    RateBudgetViolation,
    SeedRequiredViolation,
    SimulatedAnnealingOnDiscreteCodes,
    SimulatedAnnealingSpec,
    _clear_pass_registry_for_tests,
    append_pass_outcome_locked,
    compress_time_pass,
    get_pass_function,
    get_registered_passes,
    load_pass_outcomes,
    load_pass_outcomes_strict,
    validate_all_registered_passes,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _hook_rationale() -> dict[str, str]:
    """Canonical rationale dict satisfying every required hook entry."""
    return {
        "hook_sensitivity_contribution": "test fixture: not applicable",
        "hook_bit_allocator_class": "test fixture: not applicable",
        "hook_autopilot_ranker": "test fixture: not applicable",
        "hook_probe_disambiguator": "test fixture: single canonical interpretation",
    }


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear the per-test registry so tests don't pollute each other."""
    _clear_pass_registry_for_tests()
    yield
    _clear_pass_registry_for_tests()


# ---------------------------------------------------------------------------
# 1. CompressTimePassContract validation + edge cases (~20)
# ---------------------------------------------------------------------------


class TestContractValidation:
    def test_minimal_valid_contract(self):
        c = CompressTimePassContract(
            id="x",
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.id == "x"
        assert c.stage_phase == "compress"
        assert c.deterministic is True

    def test_id_must_match_pattern(self):
        with pytest.raises(CompressTimePassContractError, match="must match"):
            CompressTimePassContract(
                id="Bad-Id",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_id_empty_rejected(self):
        with pytest.raises(CompressTimePassContractError):
            CompressTimePassContract(
                id="",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_parent_pass_id_cannot_equal_id(self):
        with pytest.raises(CompressTimePassContractError, match="differ from"):
            CompressTimePassContract(
                id="x",
                parent_pass_id="x",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_stage_phase_inflate_forbidden(self):
        with pytest.raises(InflatePhaseForbiddenError) as exc:
            CompressTimePassContract(
                id="x",
                stage_phase="inflate",
                hook_not_applicable_rationale=_hook_rationale(),
            )
        assert "tac.inflate_time_post_processing" in str(exc.value)

    def test_stage_phase_post_process_forbidden(self):
        with pytest.raises(InflatePhaseForbiddenError):
            CompressTimePassContract(
                id="x",
                stage_phase="post_process",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_stage_phase_compress_legal(self):
        c = CompressTimePassContract(
            id="x",
            stage_phase="compress",
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.stage_phase == "compress"

    def test_stage_phase_archive_build_legal(self):
        c = CompressTimePassContract(
            id="x",
            stage_phase="archive_build",
            deterministic=True,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.stage_phase == "archive_build"

    def test_stage_phase_unknown_rejected(self):
        with pytest.raises(CompressTimePassContractError, match="not in"):
            CompressTimePassContract(
                id="x",
                stage_phase="something_else",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_consumes_emits_overlap_rejected(self):
        with pytest.raises(CompressTimePassContractError, match="non-empty"):
            CompressTimePassContract(
                id="x",
                consumes=frozenset({"a", "b"}),
                emits=frozenset({"b", "c"}),
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_passthrough_overlap_is_legal(self):
        # passthrough is the canonical exception: emits == consumes
        c = CompressTimePassContract(
            id="x",
            correction_kind="passthrough",
            consumes=frozenset({"k"}),
            emits=frozenset({"k"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.consumes == c.emits

    def test_consumes_list_auto_converts_to_frozenset(self):
        c = CompressTimePassContract(
            id="x",
            consumes=["a", "b"],  # list, auto-converted
            emits=frozenset({"out"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.consumes == frozenset({"a", "b"})

    def test_correction_kind_invalid(self):
        with pytest.raises(CompressTimePassContractError, match="correction_kind"):
            CompressTimePassContract(
                id="x",
                correction_kind="bogus",
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_all_legal_correction_kinds_accepted(self):
        for kind in LEGAL_CORRECTION_KIND:
            extras = {}
            if kind == "passthrough":
                extras = {"consumes": frozenset({"k"}), "emits": frozenset({"k"})}
            CompressTimePassContract(
                id="x",
                correction_kind=kind,
                hook_not_applicable_rationale=_hook_rationale(),
                **extras,
            )

    def test_max_wallclock_seconds_none_legal(self):
        c = CompressTimePassContract(
            id="x",
            max_wallclock_seconds=None,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.max_wallclock_seconds is None

    def test_max_wallclock_seconds_zero_rejected(self):
        with pytest.raises(CompressTimePassContractError, match=">= 1"):
            CompressTimePassContract(
                id="x",
                max_wallclock_seconds=0,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_max_wallclock_seconds_float_rejected(self):
        with pytest.raises(CompressTimePassContractError):
            CompressTimePassContract(
                id="x",
                max_wallclock_seconds=1.5,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_rate_budget_bytes_negative_rejected(self):
        with pytest.raises(CompressTimePassContractError, match=">= 0"):
            CompressTimePassContract(
                id="x",
                rate_budget_bytes=-1,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_distortion_budget_negative_rejected(self):
        with pytest.raises(CompressTimePassContractError):
            CompressTimePassContract(
                id="x",
                distortion_budget=-0.5,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_distortion_budget_int_accepted(self):
        c = CompressTimePassContract(
            id="x",
            distortion_budget=5,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        assert c.distortion_budget == 5

    def test_seed_negative_rejected(self):
        with pytest.raises(CompressTimePassContractError, match=">= 0"):
            CompressTimePassContract(
                id="x",
                seed=-1,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_seed_bool_rejected(self):
        with pytest.raises(CompressTimePassContractError):
            CompressTimePassContract(
                id="x",
                seed=True,  # bool subclass of int — must reject
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_archive_build_requires_deterministic(self):
        with pytest.raises(CompressTimePassContractError, match="deterministic"):
            CompressTimePassContract(
                id="x",
                stage_phase="archive_build",
                deterministic=False,
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_passthrough_consumes_must_equal_emits(self):
        with pytest.raises(CompressTimePassContractError, match="passthrough"):
            CompressTimePassContract(
                id="x",
                correction_kind="passthrough",
                consumes=frozenset({"a"}),
                emits=frozenset({"b"}),
                hook_not_applicable_rationale=_hook_rationale(),
            )

    def test_hook_not_applicable_requires_rationale(self):
        rationale = _hook_rationale()
        # Remove the required rationale for sensitivity (which defaults to N/A)
        del rationale["hook_sensitivity_contribution"]
        with pytest.raises(CompressTimePassContractError, match="rationale"):
            CompressTimePassContract(
                id="x",
                hook_not_applicable_rationale=rationale,
            )

    def test_hook_not_applicable_rationale_illegal_key_rejected(self):
        rationale = _hook_rationale()
        rationale["bogus_hook_name"] = "rationale"
        with pytest.raises(CompressTimePassContractError, match="illegal key"):
            CompressTimePassContract(
                id="x",
                hook_not_applicable_rationale=rationale,
            )

    def test_hook_probe_disambiguator_path_accepted(self):
        rationale = _hook_rationale()
        del rationale["hook_probe_disambiguator"]
        c = CompressTimePassContract(
            id="x",
            hook_probe_disambiguator="tools/probe_x.py",
            hook_not_applicable_rationale=rationale,
        )
        assert c.hook_probe_disambiguator == "tools/probe_x.py"


# ---------------------------------------------------------------------------
# 2. @compress_time_pass decorator + registry (~10)
# ---------------------------------------------------------------------------


class TestDecoratorAndRegistry:
    def test_decorator_registers_pass(self):
        contract = CompressTimePassContract(
            id="reg_test",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(contract)
        def reg_test(state, *, policy, seed=0):
            return {"x": 1}

        assert "reg_test" in get_registered_passes()

    def test_get_pass_function_returns_callable(self):
        contract = CompressTimePassContract(
            id="fn_test",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(contract)
        def fn_test(state, *, policy, seed=0):
            return {"out": 42}

        fn = get_pass_function("fn_test")
        assert fn({}, policy={}) == {"out": 42}

    def test_get_pass_function_missing_raises(self):
        with pytest.raises(CompressTimePassContractError, match="No compress-time"):
            get_pass_function("does_not_exist")

    def test_decorator_rejects_non_contract(self):
        with pytest.raises(CompressTimePassContractError, match="expects"):
            compress_time_pass("not_a_contract")

    def test_decorator_rejects_non_callable(self):
        contract = CompressTimePassContract(
            id="x",
            hook_not_applicable_rationale=_hook_rationale(),
        )
        with pytest.raises(CompressTimePassContractError, match="callable"):
            compress_time_pass(contract)(42)

    def test_decorator_rolls_back_on_non_callable(self):
        contract = CompressTimePassContract(
            id="rollback_test",
            hook_not_applicable_rationale=_hook_rationale(),
        )
        with pytest.raises(CompressTimePassContractError):
            compress_time_pass(contract)(42)
        assert "rollback_test" not in get_registered_passes()

    def test_failed_same_contract_redecoration_preserves_existing_registration(self):
        contract = CompressTimePassContract(
            id="preserve_existing",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(contract)
        def preserve_existing(state, *, policy, seed=0):
            return {"ok": True}

        with pytest.raises(CompressTimePassContractError):
            compress_time_pass(contract)(42)

        assert "preserve_existing" in get_registered_passes()
        assert get_pass_function("preserve_existing")({}, policy={}) == {"ok": True}

    def test_duplicate_id_with_different_contract_rejected(self):
        c1 = CompressTimePassContract(
            id="dup_test",
            hook_not_applicable_rationale=_hook_rationale(),
        )
        c2 = CompressTimePassContract(
            id="dup_test",
            description="different",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c1)
        def dup_test1(state, *, policy, seed=0):
            return {}

        with pytest.raises(CompressTimePassContractError, match="Duplicate"):
            compress_time_pass(c2)

    def test_same_contract_reregistration_is_idempotent(self):
        c = CompressTimePassContract(
            id="idem_test",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def f(state, *, policy, seed=0):
            return {}

        # Re-decorating with the SAME contract identity is fine
        @compress_time_pass(c)
        def f_again(state, *, policy, seed=0):
            return {}

    def test_determinism_violation_on_rng_without_seed(self):
        c = CompressTimePassContract(
            id="bad_rng",
            deterministic=True,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        with pytest.raises(DeterminismViolation, match="randomness"):
            @compress_time_pass(c)
            def bad_rng(state, *, policy, rng):
                return {}

    def test_determinism_rollback_on_violation(self):
        c = CompressTimePassContract(
            id="bad_rng_rollback",
            deterministic=True,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        with pytest.raises(DeterminismViolation):
            @compress_time_pass(c)
            def bad_rng_rollback(state, *, policy, noise):
                return {}
        assert "bad_rng_rollback" not in get_registered_passes()

    def test_seed_required_violation_on_deterministic_pass_without_seed(self):
        c = CompressTimePassContract(
            id="missing_seed",
            deterministic=True,
            hook_not_applicable_rationale=_hook_rationale(),
        )
        with pytest.raises(SeedRequiredViolation, match=r"contract\.seed"):
            @compress_time_pass(c)
            def missing_seed(state, *, policy):
                return {}
        assert "missing_seed" not in get_registered_passes()

    def test_seed_violation_same_contract_redecoration_preserves_existing_registration(self):
        c = CompressTimePassContract(
            id="seed_preserve_existing",
            deterministic=True,
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def seed_preserve_existing(state, *, policy, seed=0):
            return {"ok": seed}

        with pytest.raises(SeedRequiredViolation):
            @compress_time_pass(c)
            def seed_preserve_existing_bad(state, *, policy):
                return {"ok": False}

        assert "seed_preserve_existing" in get_registered_passes()
        assert get_pass_function("seed_preserve_existing")({}, policy={}) == {"ok": 0}

    def test_validate_all_registered_passes_clean(self):
        c = CompressTimePassContract(
            id="clean",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def clean(state, *, policy, seed=0):
            return {}

        assert validate_all_registered_passes() == []

    def test_validate_all_registered_detects_corruption(self):
        from tac.compress_time_optimization.decorator import _REGISTERED_PASSES
        _REGISTERED_PASSES["corrupt"] = "not_a_contract"  # type: ignore
        errors = validate_all_registered_passes(prune_corrupt=True)
        assert any("not a CompressTimePassContract" in e for e in errors)
        assert "corrupt" not in _REGISTERED_PASSES


# ---------------------------------------------------------------------------
# 3. ComposableCompressPipeline composition + run (~18)
# ---------------------------------------------------------------------------


def _register_seed_pass(pid: str = "seed_p"):
    c = CompressTimePassContract(
        id=pid,
        correction_kind="transform",
        consumes=frozenset({"in"}),
        emits=frozenset({"out_a"}),
        hook_not_applicable_rationale=_hook_rationale(),
    )

    @compress_time_pass(c)
    def fn(state, *, policy, seed=0):
        return {"out_a": state["in"], "bytes_added": 10}

    return c


def _register_refine_pass(pid: str = "refine_p", consumes="out_a", emits="out_b", bytes_added=20):
    c = CompressTimePassContract(
        id=pid,
        correction_kind="refinement",
        consumes=frozenset({consumes}),
        emits=frozenset({emits}),
        hook_not_applicable_rationale=_hook_rationale(),
    )

    @compress_time_pass(c)
    def fn(state, *, policy, seed=0):
        return {emits: state[consumes] + 1, "bytes_added": bytes_added}

    return c


class TestPipelineComposition:
    def test_empty_pipeline_str(self):
        p = ComposableCompressPipeline()
        assert "empty" in str(p)

    def test_pipe_operator_appends(self):
        _register_seed_pass("a1")
        _register_refine_pass("b1", consumes="out_a", emits="out_b")
        p = ComposableCompressPipeline() | "a1" | "b1"
        assert [s.pass_id for s in p.passes] == ["a1", "b1"]

    def test_pipe_returns_new_pipeline(self):
        _register_seed_pass("a2")
        p = ComposableCompressPipeline()
        p2 = p | "a2"
        assert p.passes == ()
        assert p2.passes != p.passes

    def test_and_operator_requires_prior_stage(self):
        _register_seed_pass("a3")
        with pytest.raises(CompressTimePipelineError, match="at least one prior"):
            ComposableCompressPipeline() & "a3"

    def test_and_operator_marks_parallel(self):
        # Operator precedence: `&` binds tighter than `|`, so parens are
        # required to mean (pipeline | A) & B.
        _register_seed_pass("a4")
        _register_refine_pass("b4")
        p = (ComposableCompressPipeline() | "a4") & "b4"
        assert p.passes[1].composition_kind == "parallel"

    def test_matmul_attaches_search_descriptor(self):
        # Operator precedence: `@` binds tighter than `|`; use parens.
        _register_seed_pass("a5")
        p = (ComposableCompressPipeline() | "a5") @ "cma_es"
        assert p.search_strategy_descriptor == "cma_es"

    def test_matmul_rejects_empty_string(self):
        p = ComposableCompressPipeline()
        with pytest.raises(CompressTimePipelineError, match="non-empty"):
            p @ ""

    def test_build_unknown_pass_id_raises(self):
        p = ComposableCompressPipeline() | "does_not_exist"
        with pytest.raises(CompressTimePipelineError, match="not registered"):
            p.build()

    def test_build_detects_ambiguous_emit(self):
        _register_seed_pass("a6")
        _register_refine_pass("b6a", consumes="out_a", emits="ambig")
        _register_refine_pass("b6b", consumes="out_a", emits="ambig")
        p = ComposableCompressPipeline() | "a6" | "b6a" | "b6b"
        with pytest.raises(AmbiguousCompositionError, match="twice"):
            p.build()

    def test_build_parallel_emit_overlap_allowed(self):
        _register_seed_pass("a7")
        _register_refine_pass("b7a", consumes="out_a", emits="ambig")
        _register_refine_pass("b7b", consumes="out_a", emits="ambig")
        # `&` marks parallel; ambiguous-emit detector skips parallel.
        # Parens required because `&` binds tighter than `|`.
        p = (ComposableCompressPipeline() | "a7" | "b7a") & "b7b"
        # parallel-merge does NOT raise (it's the explicit "merge intent")
        p.build()

    def test_parallel_merge_runs_siblings_against_same_input_state(self):
        c1 = CompressTimePassContract(
            id="side_by_side_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"x"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        c2 = CompressTimePassContract(
            id="side_by_side_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"y"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c1)
        def side_by_side_a(state, *, policy, seed=0):
            return {"x": state["seed"] + 1, "bytes_added": 1}

        @compress_time_pass(c2)
        def side_by_side_b(state, *, policy, seed=0):
            if "x" in state:
                raise AssertionError("parallel sibling saw prior sibling output")
            return {"y": state["seed"] + 2, "bytes_added": 2}

        p = (ComposableCompressPipeline() | "side_by_side_a") & "side_by_side_b"
        result = p.run(seed_state={"seed": 10})
        assert result.final_state["x"] == 11
        assert result.final_state["y"] == 12
        assert result.cumulative_bytes_added == 3

    def test_parallel_merge_additive_policy_sums_overlapping_numeric_keys(self):
        c1 = CompressTimePassContract(
            id="merge_add_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        c2 = CompressTimePassContract(
            id="merge_add_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            merge_policy="additive",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c1)
        def merge_add_a(state, *, policy, seed=0):
            return {"delta": 3, "bytes_added": 1}

        @compress_time_pass(c2)
        def merge_add_b(state, *, policy, seed=0):
            return {"delta": 4, "bytes_added": 2}

        p = (ComposableCompressPipeline() | "merge_add_a") & "merge_add_b"
        result = p.run(seed_state={"seed": 0})
        assert result.final_state["delta"] == 7
        assert result.final_state["bytes_added"] == 3
        assert result.cumulative_bytes_added == 3

    def test_parallel_merge_explicit_policy_requires_non_conflicting_keys(self):
        c1 = CompressTimePassContract(
            id="merge_explicit_a",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        c2 = CompressTimePassContract(
            id="merge_explicit_b",
            consumes=frozenset({"seed"}),
            emits=frozenset({"delta"}),
            merge_policy="explicit",
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c1)
        def merge_explicit_a(state, *, policy, seed=0):
            return {"delta": 3}

        @compress_time_pass(c2)
        def merge_explicit_b(state, *, policy, seed=0):
            return {"delta": 4}

        p = (ComposableCompressPipeline() | "merge_explicit_a") & "merge_explicit_b"
        with pytest.raises(CompressTimePipelineError, match="merge_policy='explicit'"):
            p.run(seed_state={"seed": 0})

    def test_build_consumer_breaks_ambiguity(self):
        _register_seed_pass("a8")
        _register_refine_pass("b8a", consumes="out_a", emits="mid")
        _register_refine_pass("b8b", consumes="mid", emits="final")
        _register_refine_pass("b8c", consumes="final", emits="mid")  # re-emit OK
        p = (
            ComposableCompressPipeline()
            | "a8" | "b8a" | "b8b" | "b8c"
        )
        p.build()  # mid is consumed by b8b before b8c re-emits

    def test_run_executes_passes_in_order(self):
        _register_seed_pass("a9")
        _register_refine_pass("b9", consumes="out_a", emits="out_b", bytes_added=5)
        p = ComposableCompressPipeline() | "a9" | "b9"
        r = p.run(seed_state={"in": 100})
        assert r.final_state["out_b"] == 101
        assert r.cumulative_bytes_added == 15

    def test_run_records_per_pass_outcomes(self):
        _register_seed_pass("a10")
        p = ComposableCompressPipeline() | "a10"
        r = p.run(seed_state={"in": 1})
        assert r.per_pass_outcomes[0]["pass_id"] == "a10"
        assert r.per_pass_outcomes[0]["status"] == "accepted"

    def test_run_no_op_when_pass_returns_none(self):
        c = CompressTimePassContract(
            id="noop",
            consumes=frozenset({"x"}),
            emits=frozenset({"y"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def noop(state, *, policy, seed=0):
            return None

        p = ComposableCompressPipeline() | "noop"
        r = p.run(seed_state={"x": 1})
        assert r.per_pass_outcomes[0]["status"] == "no_op"

    def test_run_rejects_non_mapping_return(self):
        c = CompressTimePassContract(
            id="bad_return",
            consumes=frozenset({"x"}),
            emits=frozenset({"y"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def bad_return(state, *, policy, seed=0):
            return [1, 2, 3]

        p = ComposableCompressPipeline() | "bad_return"
        with pytest.raises(CompressTimePipelineError, match="expected a Mapping"):
            p.run(seed_state={"x": 1})

    def test_with_rate_budget_rejects_excess(self):
        _register_seed_pass("a11")
        _register_refine_pass("b11", consumes="out_a", emits="out_b", bytes_added=100)
        p = (ComposableCompressPipeline() | "a11" | "b11").with_rate_budget(bytes=50)
        r = p.run(seed_state={"in": 1})
        assert "b11" in r.rejected_passes
        # 'a11' bytes_added=10 accepted; b11 100 rejected
        assert r.cumulative_bytes_added == 10

    def test_with_rate_budget_strict_raises(self):
        _register_seed_pass("a12")
        _register_refine_pass("b12", consumes="out_a", emits="out_b", bytes_added=100)
        p = (ComposableCompressPipeline() | "a12" | "b12").with_rate_budget(bytes=50)
        with pytest.raises(RateBudgetViolation):
            p.run(seed_state={"in": 1}, rate_strict=True)

    def test_with_rate_budget_negative_rejected(self):
        p = ComposableCompressPipeline()
        with pytest.raises(CompressTimePipelineError):
            p.with_rate_budget(bytes=-1)

    def test_with_wallclock_budget_unbounded_default(self):
        p = ComposableCompressPipeline()
        assert p.wallclock_budget_seconds is None

    def test_with_wallclock_budget_zero_rejected(self):
        p = ComposableCompressPipeline()
        with pytest.raises(CompressTimePipelineError):
            p.with_wallclock_budget(seconds=0)

    def test_with_wallclock_budget_set_and_clear(self):
        p = ComposableCompressPipeline().with_wallclock_budget(seconds=60)
        assert p.wallclock_budget_seconds == 60
        p2 = p.with_wallclock_budget(seconds=None)
        assert p2.wallclock_budget_seconds is None

    def test_pass_function_exception_wrapped(self):
        c = CompressTimePassContract(
            id="raiser",
            consumes=frozenset({"x"}),
            emits=frozenset({"y"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )

        @compress_time_pass(c)
        def raiser(state, *, policy, seed=0):
            raise RuntimeError("boom")

        p = ComposableCompressPipeline() | "raiser"
        with pytest.raises(CompressTimePipelineError, match="raised during"):
            p.run(seed_state={"x": 1})

    def test_pass_contracts_returns_in_order(self):
        _register_seed_pass("ax")
        _register_refine_pass("bx")
        p = ComposableCompressPipeline() | "ax" | "bx"
        contracts = p.pass_contracts()
        assert [c.id for c in contracts] == ["ax", "bx"]


# ---------------------------------------------------------------------------
# 4. Persistence (~10)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_append_creates_file(self, tmp_path: Path):
        path = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        rec = {"pass_id": "p1", "status": "accepted"}
        append_pass_outcome_locked(rec, path=path, lock_path=lock)
        assert path.exists()
        rows = load_pass_outcomes(path)
        assert len(rows) == 1
        assert rows[0]["pass_id"] == "p1"

    def test_append_stamps_provenance(self, tmp_path: Path):
        path = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        rec = {"pass_id": "p2", "status": "accepted"}
        out = append_pass_outcome_locked(rec, path=path, lock_path=lock)
        assert "written_at_utc" in out
        assert "written_pid" in out
        assert "written_host" in out
        assert out["schema_version"] == COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION

    def test_strict_load_empty_returns_empty(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        assert load_pass_outcomes_strict(path) == []

    def test_strict_load_raises_on_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not_valid_json\n", encoding="utf-8")
        with pytest.raises(CompressTimeLedgerCorruptError):
            load_pass_outcomes_strict(path)

    def test_strict_load_raises_on_non_dict_row(self, tmp_path: Path):
        path = tmp_path / "bad2.jsonl"
        path.write_text("[1,2,3]\n", encoding="utf-8")
        with pytest.raises(CompressTimeLedgerCorruptError, match="not a dict"):
            load_pass_outcomes_strict(path)

    def test_lenient_load_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"pass_id":"good","status":"accepted"}\n'
            "garbage line\n"
            '{"pass_id":"good2","status":"accepted"}\n',
            encoding="utf-8",
        )
        rows = load_pass_outcomes(path)
        assert len(rows) == 2

    def test_append_quarantines_on_corrupt(self, tmp_path: Path):
        path = tmp_path / "corrupt.jsonl"
        lock = tmp_path / "corrupt.jsonl.lock"
        path.write_text("not_valid_json\n", encoding="utf-8")
        rec = {"pass_id": "p", "status": "accepted"}
        with pytest.raises(CompressTimeLedgerCorruptError, match="quarantined"):
            append_pass_outcome_locked(rec, path=path, lock_path=lock)
        # Quarantined file should exist
        quarantines = list(tmp_path.glob("corrupt.jsonl.corrupt.*"))
        assert len(quarantines) == 1

    def test_atomic_write_no_tmp_leakage(self, tmp_path: Path):
        path = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        for i in range(5):
            append_pass_outcome_locked(
                {"pass_id": f"p{i}", "status": "accepted"},
                path=path,
                lock_path=lock,
            )
        # No .tmp.* files should remain
        tmps = list(tmp_path.glob("*.tmp.*"))
        assert tmps == []

    def test_append_rejects_record_missing_pass_id(self, tmp_path: Path):
        path = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        with pytest.raises(CompressTimeLedgerCorruptError, match="pass_id"):
            append_pass_outcome_locked(
                {"status": "accepted"},  # missing pass_id
                path=path,
                lock_path=lock,
            )

    def test_append_rejects_wrong_schema_version(self, tmp_path: Path):
        path = tmp_path / "outcomes.jsonl"
        lock = tmp_path / "outcomes.jsonl.lock"
        rec = {
            "pass_id": "p",
            "status": "accepted",
            "schema_version": "wrong_version",
        }
        with pytest.raises(CompressTimeLedgerCorruptError, match="schema_version"):
            append_pass_outcome_locked(rec, path=path, lock_path=lock)


def _persistence_worker(args):
    """Helper for the 4-proc spawn-pool stress test."""
    path_str, lock_str, pid, n = args
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from tac.compress_time_optimization import append_pass_outcome_locked
    for i in range(n):
        append_pass_outcome_locked(
            {"pass_id": f"proc{pid}_row{i}", "status": "accepted"},
            path=Path(path_str),
            lock_path=Path(lock_str),
        )
    return pid


class TestPersistenceConcurrency:
    def test_4_proc_spawn_pool_concurrent_append(self, tmp_path: Path):
        path = tmp_path / "concurrent.jsonl"
        lock = tmp_path / "concurrent.jsonl.lock"
        ctx = mp.get_context("spawn")
        n_rows_each = 5
        with ctx.Pool(processes=4) as pool:
            pool.map(
                _persistence_worker,
                [(str(path), str(lock), pid, n_rows_each) for pid in range(4)],
            )
        rows = load_pass_outcomes_strict(path)
        # Expect 4 x 5 = 20 rows, all distinct
        assert len(rows) == 20
        ids = {r["pass_id"] for r in rows}
        assert len(ids) == 20


# ---------------------------------------------------------------------------
# 5. Per-builder smoke tests (~10)
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_generic_tto_harness_build_contract(self):
        spec = GenericTTOHarnessSpec(
            pass_id="tto_test",
            target_kind="parameter_tensor",
            num_steps=100,
            seed=42,
        )
        contract = GenericTTOHarness(spec=spec).build_contract()
        assert contract.id == "tto_test"
        assert contract.correction_kind == "refinement"
        assert contract.seed == 42
        assert contract.deterministic is True

    def test_generic_tto_spec_invalid_target_kind(self):
        with pytest.raises(ValueError, match="target_kind"):
            GenericTTOHarnessSpec(
                pass_id="x",
                target_kind="bogus",
            )

    def test_generic_tto_spec_invalid_optimizer(self):
        with pytest.raises(ValueError, match="optimizer"):
            GenericTTOHarnessSpec(
                pass_id="x",
                target_kind="parameter_tensor",
                optimizer="bogus",
            )

    def test_multipass_refinement_build_contract(self):
        spec = MultipassRefinementSpec(
            pass_id="mp_test",
            depth=3,
            residual_termination_threshold=1e-4,
        )
        contract = MultipassRefinement(spec=spec).build_contract()
        assert contract.id == "mp_test"
        assert contract.correction_kind == "refinement"
        assert "residual_norm_final" in contract.emits

    def test_multipass_depth_zero_rejected(self):
        with pytest.raises(ValueError, match="depth"):
            MultipassRefinementSpec(pass_id="x", depth=0)

    def test_simulated_annealing_build_contract(self):
        spec = SimulatedAnnealingSpec(
            pass_id="sa_test",
            discrete_target="selector_indices",
            num_steps=100,
            sensitivity_weighted=True,
        )
        contract = SimulatedAnnealingOnDiscreteCodes(spec=spec).build_contract()
        assert contract.id == "sa_test"
        assert contract.correction_kind == "search"
        assert contract.sensitivity_weighted is True
        assert contract.hook_sensitivity_contribution == "master_gradient_v1"

    def test_sa_spec_invalid_temp_schedule(self):
        with pytest.raises(ValueError, match="temp_schedule"):
            SimulatedAnnealingSpec(
                pass_id="x",
                discrete_target="selector_indices",
                temp_schedule="bogus",
            )

    def test_sa_spec_invalid_cooling_alpha(self):
        with pytest.raises(ValueError):
            SimulatedAnnealingSpec(
                pass_id="x",
                discrete_target="selector_indices",
                cooling_alpha=2.0,  # > 1.0
            )

    def test_per_pair_coordinate_search_build_contract(self):
        spec = PerPairCoordinateSearchSpec(
            pass_id="cs_test",
            num_pairs=600,
            num_modes=16,
            num_palette_entries=8,
            num_pose_deltas=1,
        )
        contract = PerPairCoordinateSearch(spec=spec).build_contract()
        assert contract.correction_kind == "search"
        assert contract.correction_resolution == "per_pair"

    def test_per_pair_product_space_size(self):
        spec = PerPairCoordinateSearchSpec(
            pass_id="x",
            num_modes=16,
            num_palette_entries=8,
            num_pose_deltas=4,
        )
        assert spec.product_space_size_per_pair == 16 * 8 * 4
        assert spec.total_candidate_evaluations == 600 * 16 * 8 * 4

    def test_iterated_bisection_build_contract(self):
        spec = IteratedBisectionRateKneeSpec(
            pass_id="ib_test",
            granularity="per_block",
            num_outer_iterations=4,
        )
        contract = IteratedBisectionRateKnee(spec=spec).build_contract()
        assert contract.correction_kind == "bisection"
        assert contract.hook_bit_allocator_class == "iterated_bisection"

    def test_iterated_bisection_invalid_granularity(self):
        with pytest.raises(ValueError, match="granularity"):
            IteratedBisectionRateKneeSpec(
                pass_id="x",
                granularity="bogus",
            )

    def test_iterated_bisection_invalid_scale_range(self):
        with pytest.raises(ValueError, match="low < high"):
            IteratedBisectionRateKneeSpec(
                pass_id="x",
                granularity="per_tensor",
                scale_range_log10=(2.0, 1.0),  # low > high
            )

    def test_builder_rejects_non_spec(self):
        with pytest.raises(TypeError):
            GenericTTOHarness(spec="not_a_spec")


# ---------------------------------------------------------------------------
# 6. Serialization round-trip (~5)
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_contract_to_dict_from_dict_round_trip(self):
        c = CompressTimePassContract(
            id="ser_test",
            consumes=frozenset({"a", "b"}),
            emits=frozenset({"out"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        d = c.to_dict()
        c2 = CompressTimePassContract.from_dict(d)
        assert c2.id == c.id
        assert c2.consumes == c.consumes
        assert c2.emits == c.emits

    def test_contract_frozensets_serialize_as_sorted_lists(self):
        c = CompressTimePassContract(
            id="x",
            consumes=frozenset({"z", "a", "m"}),
            emits=frozenset({"out"}),
            hook_not_applicable_rationale=_hook_rationale(),
        )
        d = c.to_dict()
        assert d["consumes"] == ["a", "m", "z"]  # sorted

    def test_pipeline_to_json_byte_stable(self):
        _register_seed_pass("ser_a")
        p = ComposableCompressPipeline() | "ser_a"
        j1 = p.to_json()
        j2 = p.to_json()
        assert j1 == j2

    def test_pipeline_from_dict_round_trip(self):
        _register_seed_pass("ser_b")
        p = (
            ComposableCompressPipeline()
            | "ser_b"
        ).with_rate_budget(bytes=1000).with_wallclock_budget(seconds=60)
        d = json.loads(p.to_json())
        p2 = ComposableCompressPipeline.from_dict(d)
        assert [s.pass_id for s in p2.passes] == ["ser_b"]
        assert p2.rate_budget_bytes == 1000
        assert p2.wallclock_budget_seconds == 60

    def test_pipeline_stage_ref_to_dict_round_trip(self):
        ref = PipelineStageRef(
            pass_id="x",
            parameters=(("k", "v"),),
            composition_kind="parallel",
        )
        d = ref.to_dict()
        ref2 = PipelineStageRef.from_dict(d)
        assert ref2.pass_id == "x"
        assert ref2.composition_kind == "parallel"
        assert ref2.parameters == (("k", "v"),)


# ---------------------------------------------------------------------------
# 7. AST introspection regression (Catalog #168) (~3)
# ---------------------------------------------------------------------------


class TestASTIntrospectionCatalog168:
    def test_contract_fields_are_introspectable_via_ast(self):
        """Catalog #168: dataclass field declarations must be reachable by
        AST walkers that handle BOTH ast.Assign AND ast.AnnAssign."""
        contract_file = Path(__file__).resolve().parents[1] / (
            "compress_time_optimization/contract.py"
        )
        text = contract_file.read_text(encoding="utf-8")
        tree = ast.parse(text)
        # Find the CompressTimePassContract class
        target_cls = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "CompressTimePassContract"
            ):
                target_cls = node
                break
        assert target_cls is not None, "Class not found in source"
        # Count both Assign and AnnAssign field declarations
        annassign_count = sum(
            1 for n in target_cls.body if isinstance(n, ast.AnnAssign)
        )
        # Modern frozen dataclass uses AnnAssign exclusively for typed fields
        assert annassign_count >= 20, (
            f"Expected >= 20 AnnAssign field declarations; got {annassign_count}"
        )

    def test_walker_handles_both_assign_forms(self):
        """Sanity: a walker that ONLY checks ast.Assign would miss
        type-annotated assignments — verify our intent is to handle both."""
        sample = """
x = 1
y: int = 2
"""
        tree = ast.parse(sample)
        assign_count = sum(
            1 for n in ast.walk(tree) if isinstance(n, ast.Assign)
        )
        annassign_count = sum(
            1 for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
        )
        assert assign_count == 1
        assert annassign_count == 1

    def test_all_builder_specs_use_frozen_dataclass(self):
        """Each builder's spec class must be a frozen dataclass — verified
        by attempting mutation (frozen raises FrozenInstanceError)."""
        from dataclasses import FrozenInstanceError

        specs = [
            GenericTTOHarnessSpec(pass_id="x", target_kind="parameter_tensor"),
            MultipassRefinementSpec(pass_id="x", depth=1),
            SimulatedAnnealingSpec(
                pass_id="x", discrete_target="selector_indices"
            ),
            PerPairCoordinateSearchSpec(pass_id="x"),
            IteratedBisectionRateKneeSpec(pass_id="x", granularity="per_tensor"),
        ]
        for spec in specs:
            with pytest.raises(FrozenInstanceError):
                spec.pass_id = "modified"  # type: ignore


# ---------------------------------------------------------------------------
# 8. Cross-namespace composition stub (~3)
# ---------------------------------------------------------------------------


class TestCrossNamespaceComposition:
    def test_boosting_and_compress_time_coexist_without_collision(self):
        """tac.boosting.BoostStageContract and
        tac.compress_time_optimization.CompressTimePassContract can be
        imported and referenced in the same module without naming collision.
        """
        from tac.boosting import BoostStageContract

        # Both contract classes have a `to_dict` method but are distinct
        # dataclasses with distinct field sets.
        assert BoostStageContract is not CompressTimePassContract

    def test_boosting_pipeline_and_compress_pipeline_independent(self):
        """ComposableBoostingPipeline and ComposableCompressPipeline are
        STRUCTURALLY INDEPENDENT — no shared base, no import collision."""
        from tac.boosting import ComposableBoostingPipeline

        bp = ComposableBoostingPipeline()
        cp = ComposableCompressPipeline()
        # Verify they have distinct field sets (boost uses 'stages', compress
        # uses 'passes')
        assert "stages" in bp.__dataclass_fields__
        assert "passes" in cp.__dataclass_fields__
        assert "stages" not in cp.__dataclass_fields__

    def test_hybrid_pipeline_concept_documented(self):
        """The cross-namespace composition contract is documented in
        per CLAUDE.md "Subagent coherence-by-default" — sister namespaces are
        independently usable + composable at the pipeline level.

        For now we verify the contracts can be MIXED (both registered in
        their own namespaces) without runtime collision.
        """
        from tac.boosting import (
            BoostStageContract,
            boost_stage,
            get_registered_stages,
        )
        from tac.boosting.decorator import _clear_stage_registry_for_tests

        _clear_stage_registry_for_tests()
        try:
            boost_contract = BoostStageContract(
                id="boost_stage_x",
                consumes=frozenset({"a"}),
                emits=frozenset({"b"}),
                hook_not_applicable_rationale={
                    "hook_sensitivity_contribution": "r",
                    "hook_bit_allocator_class": "r",
                    "hook_autopilot_ranker": "r",
                    "hook_probe_disambiguator": "r",
                },
            )

            @boost_stage(boost_contract)
            def boost_stage_x(state, *, policy):
                return {"b": state["a"]}

            compress_contract = CompressTimePassContract(
                id="compress_pass_y",
                consumes=frozenset({"a"}),
                emits=frozenset({"b"}),
                hook_not_applicable_rationale=_hook_rationale(),
            )

            @compress_time_pass(compress_contract)
            def compress_pass_y(state, *, policy, seed=0):
                return {"b": state["a"]}

            # Both should appear in their respective registries
            assert "boost_stage_x" in get_registered_stages()
            assert "compress_pass_y" in get_registered_passes()
        finally:
            _clear_stage_registry_for_tests()


# ---------------------------------------------------------------------------
# 9. Examples integration (~3)
# ---------------------------------------------------------------------------


def _reimport_examples():
    """Force a fresh import of the examples module so the @compress_time_pass
    side-effects re-register the example passes after the autouse fixture
    clears the registry."""
    import importlib
    import sys
    # Drop both the package and any cached submodule
    sys.modules.pop("tac.compress_time_optimization.examples", None)
    sys.modules.pop(
        "tac.compress_time_optimization.examples.example_passes", None
    )
    importlib.import_module("tac.compress_time_optimization.examples")


class TestExamplesIntegration:
    def test_examples_module_registers_six_passes(self):
        _reimport_examples()
        registered = get_registered_passes()
        expected = {
            "raw_quant_example",
            "tto_pose_per_pair_example",
            "multipass_quant_depth_3_example",
            "sa_fec6_selector_indices_example",
            "coord_search_fec6_k16_palette8_example",
            "bisect_int8_scale_per_block_example",
        }
        assert expected.issubset(set(registered.keys()))

    def test_example_pipeline_runs(self):
        _reimport_examples()
        p = (
            ComposableCompressPipeline()
            | "raw_quant_example"
            | "tto_pose_per_pair_example"
        )
        r = p.run(seed_state={"seed_archive": b"X" * 64})
        assert all(o["status"] == "accepted" for o in r.per_pass_outcomes)
        assert r.cumulative_bytes_added > 0

    def test_example_seed_missing_raises(self):
        _reimport_examples()
        p = ComposableCompressPipeline() | "raw_quant_example"
        with pytest.raises(CompressTimePipelineError, match="raised during"):
            p.run(seed_state={})


# ---------------------------------------------------------------------------
# 10. Errors hierarchy
# ---------------------------------------------------------------------------


class TestErrorsHierarchy:
    def test_all_errors_inherit_from_root(self):
        for exc_cls in [
            CompressTimePassContractError,
            DeterminismViolation,
            SeedRequiredViolation,
            InflatePhaseForbiddenError,
            CompressTimePipelineError,
            AmbiguousCompositionError,
            RateBudgetViolation,
            CompressTimeBudgetExceededError,
            CompressTimeLedgerCorruptError,
        ]:
            assert issubclass(exc_cls, CompressTimeOptimizationError), exc_cls
