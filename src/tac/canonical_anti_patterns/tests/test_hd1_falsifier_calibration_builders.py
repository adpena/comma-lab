# SPDX-License-Identifier: MIT
"""Controls for the HD1 ensemble-calibrated-falsifier anti-pattern."""
from __future__ import annotations

from pathlib import Path

from tac.canonical_anti_patterns import (
    PARADIGM_RIGOR_LOSS,
    SEVERITY_HIGH,
    build_all_hd1_falsifier_calibration_anti_patterns,
    build_single_seed_falsifier_on_stochastic_endpoint_v1,
    populate_hd1_falsifier_calibration_anti_patterns,
    query_anti_patterns,
)


def test_the_wave_registers_exactly_the_expected_id() -> None:
    ids = {ap.anti_pattern_id for ap in build_all_hd1_falsifier_calibration_anti_patterns()}
    assert ids == {"single_seed_falsifier_on_stochastic_endpoint_v1"}


def test_row_is_well_formed_and_non_promotable() -> None:
    row = build_single_seed_falsifier_on_stochastic_endpoint_v1()

    assert len(row.description) <= 600
    assert row.paradigm_class == PARADIGM_RIGOR_LOSS
    assert row.severity == SEVERITY_HIGH
    # A design-time class carries no score authority of any kind.
    assert row.provenance.promotion_eligible is False
    assert row.provenance.score_claim_valid is False
    assert row.empirical_falsifications == ()


def test_the_band_names_the_seed_counts_that_moved_the_verdict() -> None:
    band = build_single_seed_falsifier_on_stochastic_endpoint_v1().falsification_band

    assert band["min_control_seeds_for_a_band"] == 2.0
    assert band["lr_ladder_control_seeds_used"] == 1.0
    assert band["lr_ladder_regrade_seeds_used"] == 2.0
    # The whole point of the anchor: the verdict moved with NO new treatment run.
    assert band["new_treatment_runs_required_for_regrade"] == 0.0
    assert band["verdicts_moved_on_regrade"] == 2.0


def test_predicate_and_unwind_name_the_actual_mechanism() -> None:
    row = build_single_seed_falsifier_on_stochastic_endpoint_v1()

    assert "endpoint_is_stochastic" in row.forbidden_pattern_predicate
    assert "single_control_run" in row.forbidden_pattern_predicate
    assert "seed ensemble" in row.canonical_unwind_path.lower()
    # The unwind must also cover the case where an ensemble is unaffordable.
    assert "INSTANCE" in row.canonical_unwind_path
    assert "LR6E5" in row.canonical_source_anchor


def test_row_has_consumers_so_it_is_not_an_orphan() -> None:
    row = build_single_seed_falsifier_on_stochastic_endpoint_v1()

    assert row.canonical_consumers
    assert row.canonical_producers
    assert any("match_stack_against_anti_patterns" in c for c in row.canonical_consumers)


def test_populate_round_trips_through_a_temporary_registry(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"

    populate_hd1_falsifier_calibration_anti_patterns(path=registry, lock_path=lock, agent="test")

    loaded = {ap.anti_pattern_id for ap in query_anti_patterns(path=registry)}
    assert "single_seed_falsifier_on_stochastic_endpoint_v1" in loaded
