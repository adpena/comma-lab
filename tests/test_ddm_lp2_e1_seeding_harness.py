from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_lp2_e1_seeding_harness import (
    LANE_CLASS_INDEX,
    SCHEMA,
    SURVIVAL_FALSIFIER_FRACTION,
    E1SeedingError,
    ErasedComponent,
    StubLocalTokenSolver,
    TokenGridGeometry,
    assemble_seeding_report,
    evaluate_survival_falsifier,
    extract_covering_tokens,
    report_to_row,
    run_local_seed_solves,
    stub_hard_oracle,
    verify_seed_init,
)

_GEOM = TokenGridGeometry(image_height=384, image_width=512, token_rows=24, token_cols=32)


def _component(cid: int, size_px: int, pixels) -> ErasedComponent:
    return ErasedComponent(component_id=cid, size_px=size_px, pixel_support=tuple(pixels))


# --- constants ------------------------------------------------------------------------------------


def test_constants() -> None:
    assert LANE_CLASS_INDEX == 1
    assert SCHEMA == "ddm_lp2_e1_seeding_harness.v1"
    assert SURVIVAL_FALSIFIER_FRACTION == 0.50  # gc12 §3 rung-2 (e1) anchor


# --- covering-token extraction (real geometry) ----------------------------------------------------


def test_pixel_to_token_corners() -> None:
    assert _GEOM.pixel_to_token(0, 0) == 0
    assert _GEOM.pixel_to_token(383, 511) == _GEOM.n_tokens - 1  # 24*32-1


def test_pixel_out_of_range_raises() -> None:
    with pytest.raises(E1SeedingError, match="outside image"):
        _GEOM.pixel_to_token(400, 0)


def test_extract_covering_tokens_local() -> None:
    # a small component near the top-left maps to a single token cell (no dilation)
    comp = _component(1, 9, [(2, 2), (2, 3), (3, 2)])
    cov = extract_covering_tokens(comp, _GEOM, dilation_cells=0)
    assert cov.component_id == 1
    assert cov.token_indices == (0,)  # all pixels fall in token cell (0,0)


def test_extract_covering_tokens_dilation_adds_ring() -> None:
    comp = _component(1, 9, [(200, 256)])  # interior pixel -> one cell, dilation adds its 8-neighborhood
    base = extract_covering_tokens(comp, _GEOM, dilation_cells=0)
    dil = extract_covering_tokens(comp, _GEOM, dilation_cells=1)
    assert len(base.token_indices) == 1
    assert len(dil.token_indices) == 9  # 3x3 ring around an interior cell
    assert set(base.token_indices).issubset(set(dil.token_indices))


def test_extract_empty_support_raises() -> None:
    with pytest.raises(E1SeedingError, match="empty pixel_support"):
        extract_covering_tokens(_component(1, 9, []), _GEOM)


def test_negative_dilation_raises() -> None:
    with pytest.raises(E1SeedingError, match="dilation_cells"):
        extract_covering_tokens(_component(1, 9, [(2, 2)]), _GEOM, dilation_cells=-1)


# --- #208 / #532 init verification ----------------------------------------------------------------


def test_init_verification_pass() -> None:
    v = verify_seed_init([0.30, 0.01, 0.40, 0.05, 0.24], [0.30, 0.006, 0.40, 0.05, 0.24])
    assert v.passed is True
    assert v.lane_channel_live is True
    assert v.dead_classes == ()


def test_init_verification_dead_lane_fails_208() -> None:
    # Lane channel collapsed to 0 -> #208 rare-class-protected catch (the fp1 dead-Lane ridge init).
    v = verify_seed_init([0.30, 0.0, 0.41, 0.05, 0.24], [0.30, 0.006, 0.40, 0.05, 0.24])
    assert v.lane_channel_live is False
    assert LANE_CLASS_INDEX in v.dead_classes
    assert v.passed is False


def test_init_verification_mass_shape_mismatch_raises() -> None:
    with pytest.raises(E1SeedingError, match="equal-length"):
        verify_seed_init([0.3, 0.1], [0.3, 0.1, 0.2])


# --- harness run (build-only, stub solver + stub oracle) -------------------------------------------


def test_run_local_seed_solves_skips_sub_nucleus() -> None:
    comps = [
        _component(1, 3, [(2, 2)]),  # sub-nucleus (<=5px) -> skipped
        _component(2, 40, [(100, 100), (100, 101)]),  # super-nucleus
    ]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), stub_hard_oracle)
    assert len(results) == 1
    assert results[0].component_id == 2
    assert results[0].status == "SEED_ACCEPTED"
    assert results[0].reachable is True


def test_reachability_fraction_rg3_analog() -> None:
    comps = [_component(i, 40, [(10 * i, 10 * i)]) for i in range(1, 5)]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), stub_hard_oracle)
    verification = verify_seed_init([0.3, 0.01, 0.4, 0.05, 0.24], [0.3, 0.006, 0.4, 0.05, 0.24])
    report = assemble_seeding_report(comps, results, verification)
    assert report.n_super_nucleus == 4
    assert report.reachable_fraction == 1.0
    assert set(report.marked_for_reconcile) == {1, 2, 3, 4}
    assert report.schema == SCHEMA


def test_failed_init_marks_nothing() -> None:
    comps = [_component(1, 40, [(10, 10)])]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), stub_hard_oracle)
    dead_lane = verify_seed_init([0.3, 0.0, 0.41, 0.05, 0.24], [0.3, 0.006, 0.4, 0.05, 0.24])
    report = assemble_seeding_report(comps, results, dead_lane)
    assert report.marked_for_reconcile == ()
    assert any("init verification FAILED" in c for c in report.caveats)


def test_no_init_verification_marks_nothing_fail_closed() -> None:
    comps = [_component(1, 40, [(10, 10)])]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), stub_hard_oracle)
    report = assemble_seeding_report(comps, results, None)
    assert report.marked_for_reconcile == ()
    assert any("NOT run" in c for c in report.caveats)


def test_report_serializes() -> None:
    comps = [_component(1, 40, [(10, 10)])]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), stub_hard_oracle)
    v = verify_seed_init([0.3, 0.01, 0.4, 0.05, 0.24], [0.3, 0.006, 0.4, 0.05, 0.24])
    row = report_to_row(assemble_seeding_report(comps, results, v))
    assert row["schema"] == SCHEMA
    assert row["score_claim"] is False
    assert "build-only" in row["evidence_axis"].lower()
    assert "ANCHOR" in row["provenance"]["survival_falsifier_fraction"]


def test_stub_solver_stalls_when_not_reachable() -> None:
    # An oracle where the seed makes things worse -> not reachable -> STALLED_UNREACHABLE.
    def worse_oracle(deltas: np.ndarray) -> float:
        return float(100.0 + np.abs(deltas).sum())

    comps = [_component(1, 40, [(10, 10)])]
    results = run_local_seed_solves(comps, _GEOM, StubLocalTokenSolver(), worse_oracle)
    assert results[0].status == "STALLED_UNREACHABLE"
    assert results[0].reachable is False


# --- preregistered survival falsifier -------------------------------------------------------------


def test_survival_falsifier_fires_below_threshold() -> None:
    v = evaluate_survival_falsifier(seeded_delta_s=0.10, survived_delta_s=0.04)  # 40% < 50%
    assert v.survival_fraction == pytest.approx(0.4)
    assert v.e1_closes_at_formulation_scope is True


def test_survival_falsifier_survives_above_threshold() -> None:
    v = evaluate_survival_falsifier(seeded_delta_s=0.10, survived_delta_s=0.08)  # 80% >= 50%
    assert v.e1_closes_at_formulation_scope is False


def test_survival_falsifier_rejects_nonpositive_seed() -> None:
    with pytest.raises(E1SeedingError, match="seeded_delta_s must be positive"):
        evaluate_survival_falsifier(seeded_delta_s=0.0, survived_delta_s=0.0)


def test_survival_falsifier_rejects_bad_threshold() -> None:
    with pytest.raises(E1SeedingError, match="threshold_fraction"):
        evaluate_survival_falsifier(0.1, 0.05, threshold_fraction=1.5)


# --- determinism ----------------------------------------------------------------------------------


def test_deterministic() -> None:
    comps = [_component(i, 40, [(10 * i, 10 * i)]) for i in range(1, 4)]
    solver = StubLocalTokenSolver()
    r1 = run_local_seed_solves(comps, _GEOM, solver, stub_hard_oracle)
    r2 = run_local_seed_solves(comps, _GEOM, solver, stub_hard_oracle)
    assert r1 == r2
