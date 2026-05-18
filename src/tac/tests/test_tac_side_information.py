# SPDX-License-Identifier: MIT
"""Tests for the tac.side_information namespace.

Per CLAUDE.md "Subagent coherence-by-default" + the per-namespace test
discipline: ≥ 90 tests across contract validation, decorator semantics,
pipeline composition, persistence, builders, serialization, and a
cross-namespace composition stub regression.

Mirror coverage density of the sister tac.boosting (83 tests) and
tac.compress_time_optimization (~120 tests) namespaces.

Test organization:
  1. SideInfoBakerContract validation + edge cases
  2. @side_info_baker decorator + registry
  3. ComposableSideInfoPipeline composition + run
  4. Persistence (fcntl lock + JSONL append + strict load)
  5. Per-builder smoke tests (5 builders)
  6. Serialization round-trip
  7. Examples integration
  8. Wyner-Ziv invariants
  9. Catalog #213 Comma2k19 cache invariants
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

from tac.side_information import (
    SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION,
    AmbiguousCompositionError,
    CanonicalComma2k19CacheRequiredViolation,
    Comma2k19DerivedPriorPalette,
    Comma2k19DerivedPriorPaletteSpec,
    ComposableSideInfoPipeline,
    DashcamDomainPrior,
    DashcamDomainPriorSpec,
    ImageNetStatisticsPrior,
    ImageNetStatisticsPriorSpec,
    InflateRuntimeBudgetExceededError,
    LEGAL_DASHCAM_PRIOR_KIND,
    LEGAL_FEATURE_EXTRACTION_KIND,
    LEGAL_PALETTE_KIND,
    LEGAL_RECONSTRUCTION_FN,
    LEGAL_RESIDUAL_CODE,
    LEGAL_SIDE_INFO_SOURCE,
    LEGAL_SOURCE_DATASET,
    LEGAL_STATISTIC_KIND,
    NonReproducibleSideInfoViolation,
    PipelineBakerRef,
    ScorerWeightsAsSharedPrior,
    ScorerWeightsAsSharedPriorSpec,
    SideInfoArchiveBudgetViolation,
    SideInfoBakerContract,
    SideInfoBakerContractError,
    SideInfoLedgerCorruptError,
    SideInfoPipelineError,
    SideInformationError,
    WynerZivCorrelationInvalidError,
    WynerZivResidualEncoder,
    WynerZivResidualEncoderSpec,
    _clear_baker_registry_for_tests,
    append_baker_outcome_locked,
    get_baker_function,
    get_registered_bakers,
    load_baker_outcomes,
    load_baker_outcomes_strict,
    side_info_baker,
    validate_all_registered_bakers,
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
    _clear_baker_registry_for_tests()
    yield
    _clear_baker_registry_for_tests()


# ---------------------------------------------------------------------------
# 1. SideInfoBakerContract validation + edge cases
# ---------------------------------------------------------------------------


class TestContractValidation:
    def test_minimal_valid_contract(self):
        c = SideInfoBakerContract(
            id="x",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.id == "x"
        assert c.side_info_reproducible is True
        assert c.archive_bytes_added == 0
        assert c.inflate_runtime_bytes_added == 0

    def test_id_rejects_uppercase(self):
        with pytest.raises(SideInfoBakerContractError, match="^id="):
            SideInfoBakerContract(
                id="X",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_id_rejects_leading_digit(self):
        with pytest.raises(SideInfoBakerContractError, match="^id="):
            SideInfoBakerContract(
                id="1x",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_id_rejects_dash(self):
        with pytest.raises(SideInfoBakerContractError, match="^id="):
            SideInfoBakerContract(
                id="x-y",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_parent_baker_id_self_reference_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="must differ"):
            SideInfoBakerContract(
                id="x",
                parent_baker_id="x",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_stage_phase_compress_legal(self):
        c = SideInfoBakerContract(
            id="x", stage_phase="compress",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.stage_phase == "compress"

    def test_stage_phase_inflate_legal(self):
        c = SideInfoBakerContract(
            id="x", stage_phase="inflate",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.stage_phase == "inflate"

    def test_stage_phase_both_default(self):
        c = SideInfoBakerContract(
            id="x",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.stage_phase == "both"

    def test_stage_phase_unknown_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="stage_phase"):
            SideInfoBakerContract(
                id="x", stage_phase="garbage",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_consume_emit_overlap_rejected_non_passthrough(self):
        with pytest.raises(
            SideInfoBakerContractError, match=r"consumes ∩ emits"
        ):
            SideInfoBakerContract(
                id="x",
                consumes=frozenset({"a"}),
                emits=frozenset({"a"}),
                correction_kind="shared_prior_bake",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_consume_emit_overlap_allowed_passthrough(self):
        c = SideInfoBakerContract(
            id="x",
            consumes=frozenset({"a"}),
            emits=frozenset({"a"}),
            correction_kind="passthrough",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.consumes == c.emits

    def test_passthrough_must_emit_what_consumes(self):
        with pytest.raises(SideInfoBakerContractError, match="passthrough"):
            SideInfoBakerContract(
                id="x",
                consumes=frozenset({"a"}),
                emits=frozenset({"b"}),
                correction_kind="passthrough",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_non_reproducible_raises_specific_violation(self):
        with pytest.raises(NonReproducibleSideInfoViolation):
            SideInfoBakerContract(
                id="x",
                side_info_reproducible=False,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_non_reproducible_inherits_from_base(self):
        with pytest.raises(SideInfoBakerContractError):
            SideInfoBakerContract(
                id="x",
                side_info_reproducible=False,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_side_info_source_unknown_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="side_info_source"):
            SideInfoBakerContract(
                id="x",
                side_info_source="garbage",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_all_legal_side_info_sources_accepted(self):
        for src in LEGAL_SIDE_INFO_SOURCE:
            # scorer_weights requires scorer_free=False (cross-field
            # invariant). Custom is default-accepted.
            kwargs: dict = dict(
                id="x",
                side_info_source=src,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )
            if src == "scorer_weights":
                kwargs["scorer_free"] = False
            SideInfoBakerContract(**kwargs)

    def test_archive_bytes_added_negative_rejected(self):
        with pytest.raises(SideInfoBakerContractError):
            SideInfoBakerContract(
                id="x", archive_bytes_added=-1,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_inflate_runtime_bytes_added_negative_rejected(self):
        with pytest.raises(SideInfoBakerContractError):
            SideInfoBakerContract(
                id="x", inflate_runtime_bytes_added=-1,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_archive_bytes_with_non_deterministic_rejected(self):
        with pytest.raises(
            SideInfoBakerContractError, match="requires deterministic=True"
        ):
            SideInfoBakerContract(
                id="x",
                archive_bytes_added=100,
                deterministic=False,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_zero_archive_bytes_with_non_deterministic_allowed(self):
        # Zero archive bytes + non-deterministic is allowed (no bytes ⇒
        # no byte-stable invariant to violate).
        SideInfoBakerContract(
            id="x",
            archive_bytes_added=0,
            deterministic=False,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

    def test_scorer_weights_requires_scorer_free_false(self):
        with pytest.raises(SideInfoBakerContractError, match="scorer_free"):
            SideInfoBakerContract(
                id="x",
                side_info_source="scorer_weights",
                scorer_free=True,  # invalid combo
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_wyner_ziv_correlation_negative_rejected(self):
        with pytest.raises(WynerZivCorrelationInvalidError):
            SideInfoBakerContract(
                id="x",
                wyner_ziv_correlation_estimate=-0.1,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_wyner_ziv_correlation_above_one_rejected(self):
        with pytest.raises(WynerZivCorrelationInvalidError):
            SideInfoBakerContract(
                id="x",
                wyner_ziv_correlation_estimate=1.1,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_wyner_ziv_correlation_nan_rejected(self):
        with pytest.raises(WynerZivCorrelationInvalidError):
            SideInfoBakerContract(
                id="x",
                wyner_ziv_correlation_estimate=float("nan"),
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_wyner_ziv_correlation_inf_rejected(self):
        with pytest.raises(WynerZivCorrelationInvalidError):
            SideInfoBakerContract(
                id="x",
                wyner_ziv_correlation_estimate=float("inf"),
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_wyner_ziv_correlation_boundary_zero_accepted(self):
        c = SideInfoBakerContract(
            id="x", wyner_ziv_correlation_estimate=0.0,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.wyner_ziv_correlation_estimate == 0.0

    def test_wyner_ziv_correlation_boundary_one_accepted(self):
        c = SideInfoBakerContract(
            id="x", wyner_ziv_correlation_estimate=1.0,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.wyner_ziv_correlation_estimate == 1.0

    def test_wyner_ziv_correlation_bool_rejected(self):
        with pytest.raises(WynerZivCorrelationInvalidError):
            SideInfoBakerContract(
                id="x",
                wyner_ziv_correlation_estimate=True,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_correction_kind_unknown_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="correction_kind"):
            SideInfoBakerContract(
                id="x", correction_kind="garbage",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_merge_policy_unknown_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="merge_policy"):
            SideInfoBakerContract(
                id="x", merge_policy="garbage",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_seed_negative_rejected(self):
        with pytest.raises(SideInfoBakerContractError):
            SideInfoBakerContract(
                id="x", seed=-1,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_hook_not_applicable_requires_rationale(self):
        with pytest.raises(
            SideInfoBakerContractError, match="hook_sensitivity_contribution"
        ):
            SideInfoBakerContract(
                id="x",
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
                hook_not_applicable_rationale={
                    # missing hook_sensitivity_contribution rationale
                    "hook_bit_allocator_class": "ok",
                    "hook_autopilot_ranker": "ok",
                    "hook_probe_disambiguator": "ok",
                },
            )

    def test_hook_probe_disambiguator_none_requires_rationale(self):
        with pytest.raises(
            SideInfoBakerContractError, match="hook_probe_disambiguator"
        ):
            SideInfoBakerContract(
                id="x",
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
                hook_not_applicable_rationale={
                    "hook_sensitivity_contribution": "ok",
                    "hook_bit_allocator_class": "ok",
                    "hook_autopilot_ranker": "ok",
                    # missing hook_probe_disambiguator rationale
                },
            )

    def test_hook_probe_disambiguator_string_accepted(self):
        c = SideInfoBakerContract(
            id="x",
            hook_probe_disambiguator="tools/probe_x.py",
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": "ok",
                "hook_bit_allocator_class": "ok",
                "hook_autopilot_ranker": "ok",
            },
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        assert c.hook_probe_disambiguator == "tools/probe_x.py"

    def test_hook_probe_disambiguator_empty_string_rejected(self):
        with pytest.raises(SideInfoBakerContractError):
            SideInfoBakerContract(
                id="x",
                hook_probe_disambiguator="",
                hook_not_applicable_rationale={
                    "hook_sensitivity_contribution": "ok",
                    "hook_bit_allocator_class": "ok",
                    "hook_autopilot_ranker": "ok",
                },
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_hook_rationale_dict_illegal_key_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="illegal key"):
            SideInfoBakerContract(
                id="x",
                hook_not_applicable_rationale={
                    "hook_sensitivity_contribution": "ok",
                    "hook_bit_allocator_class": "ok",
                    "hook_autopilot_ranker": "ok",
                    "hook_probe_disambiguator": "ok",
                    "garbage_key": "should be rejected",
                },
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_to_dict_serializable_frozensets_as_lists(self):
        c = SideInfoBakerContract(
            id="x",
            consumes=frozenset({"a", "b"}),
            emits=frozenset({"c"}),
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        d = c.to_dict()
        assert d["consumes"] == ["a", "b"]
        assert d["emits"] == ["c"]

    def test_to_dict_round_trip(self):
        c1 = SideInfoBakerContract(
            id="x",
            consumes=frozenset({"a"}),
            emits=frozenset({"b"}),
            inflate_runtime_bytes_added=256,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        c2 = SideInfoBakerContract.from_dict(c1.to_dict())
        assert c1 == c2

    def test_to_dict_json_serializable(self):
        c = SideInfoBakerContract(
            id="x",
            consumes=frozenset({"a"}),
            emits=frozenset({"b"}),
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )
        serialized = json.dumps(c.to_dict(), sort_keys=True)
        assert "x" in serialized

    def test_lane_id_empty_string_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="lane_id"):
            SideInfoBakerContract(
                id="x", lane_id="",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_lane_id_whitespace_rejected(self):
        with pytest.raises(SideInfoBakerContractError, match="lane_id"):
            SideInfoBakerContract(
                id="x", lane_id="   ",
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )


# ---------------------------------------------------------------------------
# 2. @side_info_baker decorator + registry
# ---------------------------------------------------------------------------


def _make_contract(baker_id: str = "x") -> SideInfoBakerContract:
    return SideInfoBakerContract(
        id=baker_id,
        hook_not_applicable_rationale=_hook_rationale(),
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="not_applicable_with_rationale",
    )


class TestDecorator:
    def test_register_simple_baker(self):
        @side_info_baker(_make_contract("baker_a"))
        def baker_a(state, *, policy, seed=42):
            return {}
        bakers = get_registered_bakers()
        assert "baker_a" in bakers

    def test_decorator_rejects_non_contract(self):
        with pytest.raises(
            SideInfoBakerContractError, match="SideInfoBakerContract"
        ):
            @side_info_baker("not a contract")  # type: ignore[arg-type]
            def f(state, *, policy):
                return {}

    def test_decorator_rejects_non_callable(self):
        contract = _make_contract("baker_x")
        deco = side_info_baker(contract)
        with pytest.raises(
            SideInfoBakerContractError, match="must decorate a callable"
        ):
            deco("not a callable")  # type: ignore[arg-type]

    def test_decorator_rejects_non_callable_rolls_back_registry(self):
        contract = _make_contract("baker_x")
        deco = side_info_baker(contract)
        with pytest.raises(SideInfoBakerContractError):
            deco("not a callable")  # type: ignore[arg-type]
        bakers = get_registered_bakers()
        assert "baker_x" not in bakers

    def test_duplicate_id_rejected(self):
        @side_info_baker(_make_contract("baker_d"))
        def baker_d_v1(state, *, policy):
            return {}
        with pytest.raises(SideInfoBakerContractError, match="Duplicate"):
            @side_info_baker(_make_contract("baker_d"))
            def baker_d_v2(state, *, policy):
                return {}

    def test_re_import_idempotent_on_identity(self):
        c = _make_contract("baker_id")
        @side_info_baker(c)
        def baker(state, *, policy):
            return {}
        # Re-decorating with SAME contract object should not raise
        @side_info_baker(c)
        def baker(state, *, policy):  # noqa: F811
            return {}

    def test_get_baker_function(self):
        @side_info_baker(_make_contract("baker_g"))
        def baker_g(state, *, policy):
            return {"k": 1}
        fn = get_baker_function("baker_g")
        assert fn(state={}, policy={}) == {"k": 1}

    def test_get_baker_function_unknown_raises(self):
        with pytest.raises(
            SideInfoBakerContractError, match="No side-info baker"
        ):
            get_baker_function("nonexistent")

    def test_validate_all_clean(self):
        @side_info_baker(_make_contract("baker_v"))
        def baker_v(state, *, policy):
            return {}
        errs = validate_all_registered_bakers()
        assert errs == []

    def test_determinism_violation_randomness_no_seed(self):
        c = _make_contract("baker_r")
        with pytest.raises(
            SideInfoBakerContractError, match="randomness parameter"
        ):
            @side_info_baker(c)
            def baker_r(state, *, policy, rng):
                return {}

    def test_determinism_violation_rolls_back_registry(self):
        c = _make_contract("baker_r")
        with pytest.raises(SideInfoBakerContractError):
            @side_info_baker(c)
            def baker_r(state, *, policy, rng):
                return {}
        bakers = get_registered_bakers()
        assert "baker_r" not in bakers

    def test_non_deterministic_with_randomness_allowed(self):
        c = SideInfoBakerContract(
            id="baker_nd",
            deterministic=False,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c)
        def baker_nd(state, *, policy, rng):
            return {}
        assert "baker_nd" in get_registered_bakers()

    def test_attaches_contract_to_function(self):
        c = _make_contract("baker_attr")
        @side_info_baker(c)
        def baker_attr(state, *, policy):
            return {}
        assert baker_attr.__side_info_baker_contract__ is c


# ---------------------------------------------------------------------------
# 3. ComposableSideInfoPipeline composition + run
# ---------------------------------------------------------------------------


def _register_seed_baker():
    """Register a passthrough baker used as the head of pipelines."""
    c = SideInfoBakerContract(
        id="root",
        consumes=frozenset({"seed"}),
        emits=frozenset({"side_info_palette_v1"}),
        correction_kind="palette_distillation",
        inflate_runtime_bytes_added=32,
        hook_not_applicable_rationale=_hook_rationale(),
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="not_applicable_with_rationale",
    )

    @side_info_baker(c)
    def root(state, *, policy, seed=42):
        return {
            "side_info_palette_v1": b"\x00" * 32,
            "inflate_runtime_bytes_added": 32,
        }
    return c


def _register_residual_baker(archive_bytes: int = 512):
    """Register a residual-encoder-like baker."""
    c = SideInfoBakerContract(
        id="residual",
        consumes=frozenset({"side_info_palette_v1"}),
        emits=frozenset({"archive_residual_bytes_v1"}),
        correction_kind="wyner_ziv_residual_encode",
        archive_bytes_added=archive_bytes,
        side_info_source="wyner_ziv_residual",
        hook_not_applicable_rationale=_hook_rationale(),
        hook_sensitivity_contribution="not_applicable_with_rationale",
        hook_bit_allocator_class="not_applicable_with_rationale",
        hook_autopilot_ranker="not_applicable_with_rationale",
    )

    @side_info_baker(c)
    def residual(state, *, policy, seed=42):
        return {
            "archive_residual_bytes_v1": b"\x00" * archive_bytes,
            "archive_bytes_added": archive_bytes,
        }
    return c


class TestPipeline:
    def test_empty_pipeline(self):
        p = ComposableSideInfoPipeline()
        assert p.bakers == ()

    def test_str_empty_pipeline(self):
        p = ComposableSideInfoPipeline()
        assert "empty" in str(p)

    def test_sequential_composition(self):
        _register_seed_baker()
        _register_residual_baker()
        p = (
            ComposableSideInfoPipeline()
            | "root"
            | "residual"
        )
        assert len(p.bakers) == 2
        assert p.bakers[0].baker_id == "root"
        assert p.bakers[1].baker_id == "residual"

    def test_from_baker_ids(self):
        _register_seed_baker()
        _register_residual_baker()
        p = ComposableSideInfoPipeline.from_baker_ids(["root", "residual"])
        assert len(p.bakers) == 2

    def test_immutable_composition(self):
        _register_seed_baker()
        _register_residual_baker()
        p1 = ComposableSideInfoPipeline() | "root"
        p2 = p1 | "residual"
        assert p1 is not p2
        assert len(p1.bakers) == 1
        assert len(p2.bakers) == 2

    def test_parallel_requires_prior_baker(self):
        _register_seed_baker()
        p = ComposableSideInfoPipeline()
        with pytest.raises(SideInfoPipelineError, match="prior"):
            p & "root"

    def test_parallel_composition(self):
        _register_seed_baker()
        _register_residual_baker()
        # `&` binds tighter than `|` so parentheses are required for the
        # `(seed) & sibling` pattern.
        p = (ComposableSideInfoPipeline() | "root") & "residual"
        assert p.bakers[1].composition_kind == "parallel"

    def test_attach_search(self):
        _register_seed_baker()
        p = (ComposableSideInfoPipeline() | "root") @ "my_strategy"
        assert p.search_strategy_descriptor == "my_strategy"

    def test_attach_search_empty_rejected(self):
        _register_seed_baker()
        p = ComposableSideInfoPipeline() | "root"
        with pytest.raises(SideInfoPipelineError, match="non-empty"):
            p @ ""

    def test_build_unknown_id_rejected(self):
        p = ComposableSideInfoPipeline() | "ghost"
        with pytest.raises(SideInfoPipelineError, match="not registered"):
            p.build()

    def test_build_ambiguous_emit_rejected(self):
        _register_seed_baker()

        # Register a second baker that ALSO emits side_info_palette_v1
        c2 = SideInfoBakerContract(
            id="root_dup",
            consumes=frozenset({"seed"}),
            emits=frozenset({"side_info_palette_v1"}),
            correction_kind="palette_distillation",
            inflate_runtime_bytes_added=32,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c2)
        def root_dup(state, *, policy, seed=42):
            return {"side_info_palette_v1": b""}
        p = ComposableSideInfoPipeline() | "root" | "root_dup"
        with pytest.raises(AmbiguousCompositionError):
            p.build()

    def test_run_basic_pipeline(self):
        _register_seed_baker()
        _register_residual_baker(archive_bytes=256)
        p = ComposableSideInfoPipeline() | "root" | "residual"
        result = p.run({"seed": True})
        assert result.cumulative_archive_bytes_added == 256
        assert result.cumulative_inflate_runtime_bytes_added == 32
        assert len(result.per_baker_outcomes) == 2

    def test_run_seed_state_none(self):
        _register_seed_baker()
        p = ComposableSideInfoPipeline() | "root"
        result = p.run()
        assert "side_info_palette_v1" in result.final_state

    def test_archive_budget_rejects(self):
        _register_seed_baker()
        _register_residual_baker(archive_bytes=1000)
        p = (
            ComposableSideInfoPipeline()
            | "root"
            | "residual"
        ).with_archive_budget(bytes=500)
        result = p.run({"seed": True})
        assert "residual" in result.rejected_bakers
        # archive bytes shouldn't have been added since baker was rejected
        assert result.cumulative_archive_bytes_added == 0

    def test_archive_budget_strict_raises(self):
        _register_seed_baker()
        _register_residual_baker(archive_bytes=1000)
        p = (
            ComposableSideInfoPipeline()
            | "root"
            | "residual"
        ).with_archive_budget(bytes=500)
        with pytest.raises(SideInfoArchiveBudgetViolation):
            p.run({"seed": True}, archive_strict=True)

    def test_inflate_runtime_budget_rejects(self):
        _register_seed_baker()
        _register_residual_baker()
        # root has 32 inflate bytes; budget too tight allows nothing
        p = (
            ComposableSideInfoPipeline()
            | "root"
        ).with_inflate_runtime_budget(bytes=16)
        result = p.run({"seed": True})
        assert "root" in result.rejected_bakers

    def test_inflate_runtime_budget_strict_raises(self):
        _register_seed_baker()
        p = (
            ComposableSideInfoPipeline()
            | "root"
        ).with_inflate_runtime_budget(bytes=16)
        with pytest.raises(InflateRuntimeBudgetExceededError):
            p.run({"seed": True}, inflate_runtime_strict=True)

    def test_with_archive_budget_negative_rejected(self):
        p = ComposableSideInfoPipeline()
        with pytest.raises(SideInfoPipelineError):
            p.with_archive_budget(bytes=-1)

    def test_with_inflate_runtime_budget_negative_rejected(self):
        p = ComposableSideInfoPipeline()
        with pytest.raises(SideInfoPipelineError):
            p.with_inflate_runtime_budget(bytes=-1)

    def test_pipeline_to_dict_roundtrip(self):
        _register_seed_baker()
        _register_residual_baker()
        p1 = (
            ComposableSideInfoPipeline()
            | "root"
            | "residual"
        ).with_archive_budget(bytes=1000)
        p2 = ComposableSideInfoPipeline.from_dict(p1.to_dict())
        assert p1.bakers == p2.bakers
        assert p1.archive_budget_bytes == p2.archive_budget_bytes

    def test_pipeline_to_json_byte_stable(self):
        _register_seed_baker()
        p = ComposableSideInfoPipeline() | "root"
        json_a = p.to_json()
        json_b = p.to_json()
        assert json_a == json_b

    def test_pipeline_str_repr(self):
        _register_seed_baker()
        _register_residual_baker()
        p = (
            ComposableSideInfoPipeline()
            | "root"
            | "residual"
        ).with_archive_budget(bytes=1000)
        s = str(p)
        assert "root" in s
        assert "residual" in s
        assert "archive_budget" in s

    def test_pipeline_str_inflate_budget(self):
        _register_seed_baker()
        p = (
            ComposableSideInfoPipeline()
            | "root"
        ).with_inflate_runtime_budget(bytes=4096)
        s = str(p)
        assert "inflate_runtime_budget" in s

    def test_run_baker_raises_propagates(self):
        c = SideInfoBakerContract(
            id="boom",
            consumes=frozenset({"in"}),
            emits=frozenset({"out"}),
            correction_kind="shared_prior_bake",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c)
        def boom(state, *, policy):
            raise ValueError("kaboom")
        p = ComposableSideInfoPipeline() | "boom"
        with pytest.raises(SideInfoPipelineError, match="kaboom"):
            p.run({})

    def test_baker_returns_non_mapping_rejected(self):
        c = SideInfoBakerContract(
            id="bad_return",
            consumes=frozenset({"in"}),
            emits=frozenset({"out"}),
            correction_kind="shared_prior_bake",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c)
        def bad_return(state, *, policy):
            return 42  # not a mapping
        p = ComposableSideInfoPipeline() | "bad_return"
        with pytest.raises(SideInfoPipelineError, match="expected a Mapping"):
            p.run({})

    def test_baker_returns_none_passthrough(self):
        c = SideInfoBakerContract(
            id="returns_none",
            consumes=frozenset({"in"}),
            emits=frozenset({"out"}),
            correction_kind="shared_prior_bake",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c)
        def returns_none(state, *, policy):
            return None
        p = ComposableSideInfoPipeline() | "returns_none"
        result = p.run({})
        assert result.per_baker_outcomes[0]["status"] == "no_op"

    def test_baker_returns_negative_bytes_rejected(self):
        c = SideInfoBakerContract(
            id="neg_bytes",
            consumes=frozenset({"in"}),
            emits=frozenset({"out"}),
            correction_kind="shared_prior_bake",
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

        @side_info_baker(c)
        def neg_bytes(state, *, policy):
            return {"archive_bytes_added": -1}
        p = ComposableSideInfoPipeline() | "neg_bytes"
        with pytest.raises(SideInfoPipelineError, match="negative"):
            p.run({})

    def test_pipeline_baker_contracts(self):
        _register_seed_baker()
        _register_residual_baker()
        p = ComposableSideInfoPipeline() | "root" | "residual"
        contracts = p.baker_contracts()
        assert len(contracts) == 2
        assert contracts[0].id == "root"


# ---------------------------------------------------------------------------
# 4. Persistence (fcntl lock + JSONL append + strict load)
# ---------------------------------------------------------------------------


def _ledger_record(baker_id: str = "x", status: str = "accepted") -> dict:
    return {
        "schema_version": SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION,
        "baker_id": baker_id,
        "status": status,
    }


class TestPersistence:
    def test_append_then_load(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        rec = _ledger_record("a")
        append_baker_outcome_locked(rec, path=path, lock_path=lock_path)
        rows = load_baker_outcomes(path=path)
        assert len(rows) == 1
        assert rows[0]["baker_id"] == "a"

    def test_append_two_records(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        append_baker_outcome_locked(
            _ledger_record("a"), path=path, lock_path=lock_path
        )
        append_baker_outcome_locked(
            _ledger_record("b"), path=path, lock_path=lock_path
        )
        rows = load_baker_outcomes(path=path)
        assert [r["baker_id"] for r in rows] == ["a", "b"]

    def test_append_adds_timestamp_pid_host(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        rec = {"baker_id": "a", "status": "accepted"}
        append_baker_outcome_locked(rec, path=path, lock_path=lock_path)
        rows = load_baker_outcomes(path=path)
        assert "written_at_utc" in rows[0]
        assert "written_pid" in rows[0]
        assert "written_host" in rows[0]
        assert "schema_version" in rows[0]

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        assert load_baker_outcomes(path=path) == []
        assert load_baker_outcomes_strict(path=path) == []

    def test_load_lenient_skips_malformed(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            '{"schema_version": "side_information_baker_outcomes_v1", '
            '"baker_id": "a", "status": "ok"}\nnotjson\n'
        )
        rows = load_baker_outcomes(path=path)
        assert len(rows) == 1

    def test_load_strict_raises_on_malformed(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            '{"schema_version": "side_information_baker_outcomes_v1", '
            '"baker_id": "a", "status": "ok"}\nnotjson\n'
        )
        with pytest.raises(SideInfoLedgerCorruptError):
            load_baker_outcomes_strict(path=path)

    def test_load_strict_raises_on_non_dict_row(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("[1, 2, 3]\n")
        with pytest.raises(SideInfoLedgerCorruptError, match="not a dict"):
            load_baker_outcomes_strict(path=path)

    def test_append_corrupt_ledger_quarantines(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        path.write_text("notjson\n")
        with pytest.raises(SideInfoLedgerCorruptError, match="quarantined"):
            append_baker_outcome_locked(
                _ledger_record("a"), path=path, lock_path=lock_path
            )
        # Quarantined file should exist
        assert not path.exists()
        siblings = list(tmp_path.glob("ledger.jsonl.corrupt.*"))
        assert len(siblings) == 1

    def test_append_invalid_record_schema_rejected(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        with pytest.raises(SideInfoLedgerCorruptError):
            append_baker_outcome_locked(
                {"baker_id": "a", "status": "ok", "schema_version": "wrong"},
                path=path,
                lock_path=lock_path,
            )

    def test_append_invalid_record_missing_baker_id(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        with pytest.raises(SideInfoLedgerCorruptError):
            append_baker_outcome_locked(
                {"status": "ok"}, path=path, lock_path=lock_path
            )

    def test_append_invalid_record_missing_status(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        with pytest.raises(SideInfoLedgerCorruptError):
            append_baker_outcome_locked(
                {"baker_id": "a"}, path=path, lock_path=lock_path
            )


def _proc_append(args):
    """Subprocess append for the concurrency stress test."""
    path, lock_path, baker_id = args
    from tac.side_information import (
        SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION,
        append_baker_outcome_locked,
    )
    for i in range(5):
        rec = {
            "schema_version": SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION,
            "baker_id": f"{baker_id}_{i}",
            "status": "accepted",
        }
        append_baker_outcome_locked(
            rec, path=Path(path), lock_path=Path(lock_path)
        )
    return baker_id


class TestPersistenceConcurrency:
    def test_4_proc_spawn_concurrent_append(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        lock_path = tmp_path / "ledger.jsonl.lock"
        ctx = mp.get_context("spawn")
        with ctx.Pool(4) as pool:
            results = pool.map(
                _proc_append,
                [
                    (str(path), str(lock_path), "a"),
                    (str(path), str(lock_path), "b"),
                    (str(path), str(lock_path), "c"),
                    (str(path), str(lock_path), "d"),
                ],
            )
        assert sorted(results) == ["a", "b", "c", "d"]
        # Should be 20 rows total (4 procs * 5 rows each)
        rows = load_baker_outcomes(path=path)
        assert len(rows) == 20


# ---------------------------------------------------------------------------
# 5. Per-builder smoke tests
# ---------------------------------------------------------------------------


class TestScorerWeightsAsSharedPrior:
    def test_build_contract(self):
        spec = ScorerWeightsAsSharedPriorSpec(
            baker_id="b1",
            feature_extraction_kind="segnet_per_class_centroids",
        )
        c = ScorerWeightsAsSharedPrior(spec=spec).build_contract()
        assert c.id == "b1"
        assert c.side_info_source == "scorer_weights"
        assert c.scorer_free is False  # critical invariant
        assert c.archive_bytes_added == 0
        assert c.inflate_runtime_bytes_added == 256

    def test_invalid_feature_extraction_kind(self):
        with pytest.raises(ValueError, match="feature_extraction_kind"):
            ScorerWeightsAsSharedPriorSpec(
                baker_id="b1", feature_extraction_kind="garbage"
            )

    def test_negative_inflate_runtime_bytes(self):
        with pytest.raises(ValueError):
            ScorerWeightsAsSharedPriorSpec(
                baker_id="b1",
                feature_extraction_kind="segnet_per_class_centroids",
                inflate_runtime_bytes_added=-1,
            )

    def test_wyner_ziv_correlation_validated(self):
        with pytest.raises(ValueError):
            ScorerWeightsAsSharedPriorSpec(
                baker_id="b1",
                feature_extraction_kind="segnet_per_class_centroids",
                wyner_ziv_correlation_estimate=1.5,
            )

    def test_all_legal_feature_kinds_accepted(self):
        for kind in LEGAL_FEATURE_EXTRACTION_KIND:
            spec = ScorerWeightsAsSharedPriorSpec(
                baker_id="b1", feature_extraction_kind=kind
            )
            c = ScorerWeightsAsSharedPrior(spec=spec).build_contract()
            assert c is not None

    def test_seed_negative_rejected(self):
        with pytest.raises(ValueError):
            ScorerWeightsAsSharedPriorSpec(
                baker_id="b1",
                feature_extraction_kind="segnet_per_class_centroids",
                seed=-1,
            )

    def test_spec_type_check(self):
        with pytest.raises(TypeError):
            ScorerWeightsAsSharedPrior(spec="not_a_spec")  # type: ignore[arg-type]

    def test_contract_has_scorer_weights_shared_prior_hook(self):
        spec = ScorerWeightsAsSharedPriorSpec(
            baker_id="b1",
            feature_extraction_kind="segnet_per_class_centroids",
        )
        c = ScorerWeightsAsSharedPrior(spec=spec).build_contract()
        assert (
            c.hook_sensitivity_contribution
            == "scorer_weights_shared_prior_v1"
        )


class TestComma2k19DerivedPriorPalette:
    def test_build_contract(self):
        spec = Comma2k19DerivedPriorPaletteSpec(
            baker_id="b1",
            palette_kind="chroma_anchors",
            num_palette_entries=16,
        )
        c = Comma2k19DerivedPriorPalette(spec=spec).build_contract()
        assert c.id == "b1"
        assert c.side_info_source == "comma2k19_distilled"
        assert c.requires_canonical_comma2k19_cache is True
        # Per Catalog #213 the canonical cache must be importable
        # otherwise this would raise; in the test env the helper is
        # present so build_contract succeeds.

    def test_invalid_palette_kind(self):
        with pytest.raises(ValueError, match="palette_kind"):
            Comma2k19DerivedPriorPaletteSpec(
                baker_id="b1", palette_kind="garbage", num_palette_entries=16
            )

    def test_num_palette_entries_too_small(self):
        with pytest.raises(ValueError, match="num_palette_entries"):
            Comma2k19DerivedPriorPaletteSpec(
                baker_id="b1",
                palette_kind="chroma_anchors",
                num_palette_entries=1,
            )

    def test_bytes_per_entry_zero_rejected(self):
        with pytest.raises(ValueError):
            Comma2k19DerivedPriorPaletteSpec(
                baker_id="b1",
                palette_kind="chroma_anchors",
                num_palette_entries=16,
                bytes_per_entry=0,
            )

    def test_total_palette_bytes_property(self):
        spec = Comma2k19DerivedPriorPaletteSpec(
            baker_id="b1",
            palette_kind="chroma_anchors",
            num_palette_entries=16,
            bytes_per_entry=3,
        )
        assert spec.total_palette_bytes == 48

    def test_all_legal_palette_kinds_accepted(self):
        for kind in LEGAL_PALETTE_KIND:
            spec = Comma2k19DerivedPriorPaletteSpec(
                baker_id="b1", palette_kind=kind, num_palette_entries=16
            )
            assert spec.palette_kind == kind

    def test_spec_type_check(self):
        with pytest.raises(TypeError):
            Comma2k19DerivedPriorPalette(spec="bad")  # type: ignore[arg-type]


class TestImageNetStatisticsPrior:
    def test_build_contract(self):
        spec = ImageNetStatisticsPriorSpec(
            baker_id="b1",
            statistic_kind="rgb_mean_std",
            inflate_runtime_bytes_added=24,
            source_url="https://pytorch.org/vision",
            license_tag="BSD-3-Clause",
        )
        c = ImageNetStatisticsPrior(spec=spec).build_contract()
        assert c.id == "b1"
        assert c.side_info_source == "imagenet_statistics"
        assert c.scorer_free is True

    def test_invalid_statistic_kind(self):
        with pytest.raises(ValueError, match="statistic_kind"):
            ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind="garbage",
                inflate_runtime_bytes_added=24,
                source_url="https://x.com",
            )

    def test_source_url_required_http_or_https(self):
        with pytest.raises(ValueError, match="http://"):
            ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind="rgb_mean_std",
                inflate_runtime_bytes_added=24,
                source_url="ftp://example.com",
            )

    def test_source_url_empty_rejected(self):
        with pytest.raises(ValueError):
            ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind="rgb_mean_std",
                inflate_runtime_bytes_added=24,
                source_url="",
            )

    def test_inflate_runtime_bytes_zero_rejected(self):
        with pytest.raises(ValueError):
            ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind="rgb_mean_std",
                inflate_runtime_bytes_added=0,
                source_url="https://x.com",
            )

    def test_license_tag_empty_rejected(self):
        with pytest.raises(ValueError):
            ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind="rgb_mean_std",
                inflate_runtime_bytes_added=24,
                source_url="https://x.com",
                license_tag="",
            )

    def test_all_legal_statistic_kinds_accepted(self):
        for kind in LEGAL_STATISTIC_KIND:
            spec = ImageNetStatisticsPriorSpec(
                baker_id="b1",
                statistic_kind=kind,
                inflate_runtime_bytes_added=10,
                source_url="https://x.com",
            )
            assert spec.statistic_kind == kind

    def test_spec_type_check(self):
        with pytest.raises(TypeError):
            ImageNetStatisticsPrior(spec="bad")  # type: ignore[arg-type]


class TestDashcamDomainPrior:
    def test_build_contract(self):
        spec = DashcamDomainPriorSpec(
            baker_id="b1",
            prior_kind="sky_horizon_split_prior",
            source_dataset="bdd100k",
            inflate_runtime_bytes_added=384,
        )
        c = DashcamDomainPrior(spec=spec).build_contract()
        assert c.id == "b1"
        assert c.side_info_source == "dashcam_domain"
        # bdd100k does NOT require comma2k19 cache
        assert c.requires_canonical_comma2k19_cache is False

    def test_comma2k19_source_requires_canonical_cache(self):
        spec = DashcamDomainPriorSpec(
            baker_id="b1",
            prior_kind="sky_horizon_split_prior",
            source_dataset="comma2k19",
            inflate_runtime_bytes_added=384,
            license_tag="MIT",
        )
        c = DashcamDomainPrior(spec=spec).build_contract()
        # MUST set requires_canonical_comma2k19_cache=True
        assert c.requires_canonical_comma2k19_cache is True

    def test_invalid_prior_kind(self):
        with pytest.raises(ValueError, match="prior_kind"):
            DashcamDomainPriorSpec(
                baker_id="b1",
                prior_kind="garbage",
                source_dataset="bdd100k",
                inflate_runtime_bytes_added=384,
            )

    def test_invalid_source_dataset(self):
        with pytest.raises(ValueError, match="source_dataset"):
            DashcamDomainPriorSpec(
                baker_id="b1",
                prior_kind="sky_horizon_split_prior",
                source_dataset="garbage",
                inflate_runtime_bytes_added=384,
            )

    def test_all_legal_prior_kinds_accepted(self):
        for kind in LEGAL_DASHCAM_PRIOR_KIND:
            spec = DashcamDomainPriorSpec(
                baker_id="b1",
                prior_kind=kind,
                source_dataset="bdd100k",
                inflate_runtime_bytes_added=10,
            )
            assert spec.prior_kind == kind

    def test_all_legal_source_datasets_accepted(self):
        for src in LEGAL_SOURCE_DATASET:
            spec = DashcamDomainPriorSpec(
                baker_id="b1",
                prior_kind="sky_horizon_split_prior",
                source_dataset=src,
                inflate_runtime_bytes_added=10,
            )
            assert spec.source_dataset == src

    def test_spec_type_check(self):
        with pytest.raises(TypeError):
            DashcamDomainPrior(spec="bad")  # type: ignore[arg-type]


class TestWynerZivResidualEncoder:
    def test_build_contract(self):
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent_baker",
            reconstruction_fn="linear_predictor",
            residual_code="arithmetic",
            archive_bytes_added=2048,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert c.id == "b1"
        assert c.parent_baker_id == "parent_baker"
        assert c.side_info_source == "wyner_ziv_residual"
        assert c.archive_bytes_added == 2048

    def test_invalid_reconstruction_fn(self):
        with pytest.raises(ValueError, match="reconstruction_fn"):
            WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="parent",
                reconstruction_fn="garbage",
                residual_code="arithmetic",
                archive_bytes_added=2048,
            )

    def test_invalid_residual_code(self):
        with pytest.raises(ValueError, match="residual_code"):
            WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="parent",
                reconstruction_fn="linear_predictor",
                residual_code="garbage",
                archive_bytes_added=2048,
            )

    def test_archive_bytes_zero_rejected(self):
        # Wyner-Ziv residual encoder MUST contribute archive bytes (>= 1)
        with pytest.raises(ValueError, match="archive_bytes_added"):
            WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="parent",
                reconstruction_fn="linear_predictor",
                residual_code="arithmetic",
                archive_bytes_added=0,
            )

    def test_shared_prior_baker_id_empty_rejected(self):
        with pytest.raises(ValueError, match="shared_prior_baker_id"):
            WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="",
                reconstruction_fn="linear_predictor",
                residual_code="arithmetic",
                archive_bytes_added=2048,
            )

    def test_all_legal_reconstruction_fns_accepted(self):
        for fn in LEGAL_RECONSTRUCTION_FN:
            spec = WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="parent",
                reconstruction_fn=fn,
                residual_code="arithmetic",
                archive_bytes_added=2048,
            )
            assert spec.reconstruction_fn == fn

    def test_all_legal_residual_codes_accepted(self):
        for code in LEGAL_RESIDUAL_CODE:
            spec = WynerZivResidualEncoderSpec(
                baker_id="b1",
                shared_prior_baker_id="parent",
                reconstruction_fn="linear_predictor",
                residual_code=code,
                archive_bytes_added=2048,
            )
            assert spec.residual_code == code

    def test_sensitivity_weighted_consumes_master_gradient(self):
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent",
            reconstruction_fn="linear_predictor",
            residual_code="arithmetic",
            archive_bytes_added=2048,
            sensitivity_weighted=True,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert "master_gradient" in c.consumes

    def test_custom_reconstruction_no_probe_disambiguator(self):
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent",
            reconstruction_fn="custom",
            residual_code="arithmetic",
            archive_bytes_added=2048,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert c.hook_probe_disambiguator is None

    def test_non_custom_reconstruction_has_probe_disambiguator(self):
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent",
            reconstruction_fn="linear_predictor",
            residual_code="arithmetic",
            archive_bytes_added=2048,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert c.hook_probe_disambiguator is not None

    def test_spec_type_check(self):
        with pytest.raises(TypeError):
            WynerZivResidualEncoder(spec="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. PipelineBakerRef + serialization round-trip
# ---------------------------------------------------------------------------


class TestPipelineBakerRef:
    def test_to_dict_round_trip(self):
        r = PipelineBakerRef(
            baker_id="x",
            parameters=(("k", 1),),
            composition_kind="sequential",
        )
        r2 = PipelineBakerRef.from_dict(r.to_dict())
        assert r == r2

    def test_from_dict_default_composition_kind(self):
        r = PipelineBakerRef.from_dict({"baker_id": "x"})
        assert r.composition_kind == "sequential"
        assert r.parameters == ()


# ---------------------------------------------------------------------------
# 7. Examples integration regression test
# ---------------------------------------------------------------------------


def _load_examples_into_registry() -> None:
    """Load example bakers freshly into the cleared registry.

    Module-level decorator state survives across `importlib.reload` because
    the existing contract instances are captured by closures. The cleanest
    way to repopulate the cleared registry per-test is to re-evaluate the
    examples module body in a fresh namespace.
    """
    examples_path = (
        Path(__file__).resolve().parent.parent
        / "side_information"
        / "examples"
        / "example_priors.py"
    )
    src = examples_path.read_text()
    namespace: dict = {"__name__": "example_priors_test_clone"}
    exec(compile(src, str(examples_path), "exec"), namespace)


class TestExamplesIntegration:
    def test_example_priors_registers_5_bakers(self):
        _load_examples_into_registry()
        bakers = get_registered_bakers()
        # The 5 canonical example bakers
        expected = {
            "scorer_segnet_centroids_example",
            "comma2k19_chroma_palette_example",
            "imagenet_rgb_mean_std_example",
            "dashcam_sky_horizon_example",
            "wz_residual_linear_predictor_example",
        }
        assert expected.issubset(set(bakers))

    def test_example_pipeline_runs(self):
        _load_examples_into_registry()
        # ImageNet + WZ residual is a self-contained 2-baker pipeline
        p = (
            ComposableSideInfoPipeline()
            | "imagenet_rgb_mean_std_example"
            | "wz_residual_linear_predictor_example"
        )
        result = p.run({"seed": True})
        assert result.cumulative_archive_bytes_added == 2048
        # imagenet 24 + wz inflate 512
        assert result.cumulative_inflate_runtime_bytes_added == 24 + 512


# ---------------------------------------------------------------------------
# 8. Cross-namespace structural-independence regression
# ---------------------------------------------------------------------------


class TestStructuralIndependence:
    def test_does_not_import_boost_stage_contract(self):
        """tac.side_information must NOT import tac.boosting per PV-7."""
        # Read the package __init__.py and assert no tac.boosting import
        init_path = (
            Path(__file__).resolve().parent.parent
            / "side_information"
            / "__init__.py"
        )
        src = init_path.read_text()
        assert "from tac.boosting" not in src
        assert "import tac.boosting" not in src

    def test_does_not_import_compress_time_pass_contract(self):
        """tac.side_information must NOT import tac.compress_time_optimization."""
        init_path = (
            Path(__file__).resolve().parent.parent
            / "side_information"
            / "__init__.py"
        )
        src = init_path.read_text()
        assert "from tac.compress_time_optimization" not in src
        assert "import tac.compress_time_optimization" not in src

    def test_root_error_hierarchy(self):
        """Every typed error must subclass SideInformationError."""
        for cls in (
            SideInfoBakerContractError,
            NonReproducibleSideInfoViolation,
            CanonicalComma2k19CacheRequiredViolation,
            WynerZivCorrelationInvalidError,
            SideInfoPipelineError,
            AmbiguousCompositionError,
            SideInfoArchiveBudgetViolation,
            InflateRuntimeBudgetExceededError,
            SideInfoLedgerCorruptError,
        ):
            assert issubclass(cls, SideInformationError)


# ---------------------------------------------------------------------------
# 9. Wyner-Ziv + Catalog #213 invariants
# ---------------------------------------------------------------------------


class TestWynerZivInvariants:
    def test_residual_encoder_parent_set_to_shared_prior(self):
        """Wyner-Ziv encoder's contract MUST declare its shared-prior
        parent so the pipeline cycle detector can verify ordering.
        """
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent",
            reconstruction_fn="linear_predictor",
            residual_code="arithmetic",
            archive_bytes_added=2048,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert c.parent_baker_id == "parent"

    def test_residual_encoder_emits_archive_residual_bytes(self):
        spec = WynerZivResidualEncoderSpec(
            baker_id="b1",
            shared_prior_baker_id="parent",
            reconstruction_fn="linear_predictor",
            residual_code="arithmetic",
            archive_bytes_added=2048,
        )
        c = WynerZivResidualEncoder(spec=spec).build_contract()
        assert "archive_residual_bytes_v1" in c.emits

    def test_all_shared_prior_bakers_emit_zero_archive_bytes(self):
        # The 4 shared-prior bakers do NOT contribute archive bytes
        scorer_spec = ScorerWeightsAsSharedPriorSpec(
            baker_id="b1",
            feature_extraction_kind="segnet_per_class_centroids",
        )
        assert (
            ScorerWeightsAsSharedPrior(spec=scorer_spec)
            .build_contract()
            .archive_bytes_added
            == 0
        )

        palette_spec = Comma2k19DerivedPriorPaletteSpec(
            baker_id="b2",
            palette_kind="chroma_anchors",
            num_palette_entries=16,
        )
        assert (
            Comma2k19DerivedPriorPalette(spec=palette_spec)
            .build_contract()
            .archive_bytes_added
            == 0
        )

        imagenet_spec = ImageNetStatisticsPriorSpec(
            baker_id="b3",
            statistic_kind="rgb_mean_std",
            inflate_runtime_bytes_added=24,
            source_url="https://x.com",
        )
        assert (
            ImageNetStatisticsPrior(spec=imagenet_spec)
            .build_contract()
            .archive_bytes_added
            == 0
        )

        dashcam_spec = DashcamDomainPriorSpec(
            baker_id="b4",
            prior_kind="sky_horizon_split_prior",
            source_dataset="bdd100k",
            inflate_runtime_bytes_added=384,
        )
        assert (
            DashcamDomainPrior(spec=dashcam_spec)
            .build_contract()
            .archive_bytes_added
            == 0
        )

    def test_canonical_comma2k19_cache_requirement_checked_at_decoration(
        self, monkeypatch
    ):
        """When the canonical helper is unimportable,
        requires_canonical_comma2k19_cache=True raises a specific violation.
        """
        # Monkeypatch the importer to return False
        import tac.side_information.contract as contract_mod
        monkeypatch.setattr(
            contract_mod,
            "_canonical_comma2k19_cache_importable",
            lambda: False,
        )
        with pytest.raises(CanonicalComma2k19CacheRequiredViolation):
            SideInfoBakerContract(
                id="b1",
                requires_canonical_comma2k19_cache=True,
                hook_not_applicable_rationale=_hook_rationale(),
                hook_sensitivity_contribution="not_applicable_with_rationale",
                hook_bit_allocator_class="not_applicable_with_rationale",
                hook_autopilot_ranker="not_applicable_with_rationale",
            )

    def test_canonical_comma2k19_cache_not_required_does_not_check(
        self, monkeypatch
    ):
        # Even with helper unimportable, contracts that don't require it
        # must construct cleanly.
        import tac.side_information.contract as contract_mod
        monkeypatch.setattr(
            contract_mod,
            "_canonical_comma2k19_cache_importable",
            lambda: False,
        )
        # Should NOT raise:
        SideInfoBakerContract(
            id="b1",
            requires_canonical_comma2k19_cache=False,
            hook_not_applicable_rationale=_hook_rationale(),
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_bit_allocator_class="not_applicable_with_rationale",
            hook_autopilot_ranker="not_applicable_with_rationale",
        )

    def test_dashcam_with_comma2k19_source_AND_missing_cache_rejects(
        self, monkeypatch
    ):
        # The DashcamDomainPrior builder sets
        # requires_canonical_comma2k19_cache=True when source_dataset==
        # comma2k19; when the helper is unimportable, build_contract
        # raises.
        import tac.side_information.contract as contract_mod
        monkeypatch.setattr(
            contract_mod,
            "_canonical_comma2k19_cache_importable",
            lambda: False,
        )
        spec = DashcamDomainPriorSpec(
            baker_id="b1",
            prior_kind="sky_horizon_split_prior",
            source_dataset="comma2k19",
            inflate_runtime_bytes_added=384,
        )
        with pytest.raises(CanonicalComma2k19CacheRequiredViolation):
            DashcamDomainPrior(spec=spec).build_contract()
