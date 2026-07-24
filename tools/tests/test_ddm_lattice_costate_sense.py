from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_lattice_costate_sense import (
    FACTOR_SCHEMA,
    PAIR_SCHEMA,
    LatticeSenseError,
    build_lattice_sense_pair,
    factorize_lattice_sense,
    write_sense_jsonl_atomic,
)


def _row(pair_id: int) -> dict:
    selected = np.full((2, 8, 12, 3), 100 + pair_id, dtype=np.uint8)
    origin = np.full_like(selected, 99)
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[:, 3:] = 1
    margins = np.linspace(0.0, 0.2, labels.size).reshape(labels.shape)
    dimensions = np.asarray([[3, 2], [1, 0]], dtype=np.uint8)
    return build_lattice_sense_pair(
        pair_id=pair_id,
        selected=selected,
        origin=origin,
        labels=labels,
        winner_rival_margins=margins,
        canonical_member_bytes=1000 + pair_id,
        selected_residual_bytes=800 + 2 * pair_id,
        active_tolerance=0.05,
        basis_norms=(1.0, 2.0, 3.0),
        local_facet_dimensions=dimensions,
    ).to_dict()


def test_pair_row_exposes_active_sets_rate_and_honest_missing_duals() -> None:
    row = _row(0)
    assert row["schema"] == PAIR_SCHEMA
    assert row["rate"]["partition"]["COUNTED"] == 800
    assert row["active_set"]["active_constraint_count"] > 0
    assert set(row["active_set"]["per_stratum"]) == {"cell", "edge", "saddle"}
    assert row["shadow_prices"]["available"] is False
    assert row["degeneracy"]["histogram"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert row["score_claim"] is False


def test_pair_row_accepts_precomputed_basis_distribution() -> None:
    selected = np.full((2, 8, 12, 3), 100, dtype=np.uint8)
    row = build_lattice_sense_pair(
        pair_id=1,
        selected=selected,
        origin=np.zeros_like(selected),
        labels=np.zeros((4, 6), dtype=np.uint8),
        winner_rival_margins=np.ones((4, 6), dtype=np.float64),
        canonical_member_bytes=100,
        selected_residual_bytes=90,
        active_tolerance=0.1,
        basis_norms={
            "count": 12,
            "norm_min": 1.0,
            "norm_p50": 2.0,
            "norm_p95": 3.0,
            "norm_max": 4.0,
        },
    ).to_dict()
    assert row["basis"]["count"] == 12
    assert row["basis"]["norm_p95"] == 3.0


def test_factorization_is_deterministic_and_noise_floor_gated() -> None:
    rows = [_row(index) for index in range(4)]
    first = factorize_lattice_sense(rows, coder_noise_floor_bytes=1)
    second = factorize_lattice_sense(rows, coder_noise_floor_bytes=1)
    assert first == second
    assert first["schema"] == FACTOR_SCHEMA
    assert first["pair_count"] == 4
    assert "v18_column_generation" in first["routes"]
    assert first["distilled_factor_count"] == 0
    assert all(
        factor["representation"]["status"]
        == "BLOCKED_NO_PER_STRATUM_CODER_RACE"
        for factor in first["factors"]
    )


def test_factorization_tags_only_strict_measured_coder_race_winners() -> None:
    rows = [_row(index) for index in range(4)]
    result = factorize_lattice_sense(
        rows,
        coder_noise_floor_bytes=0,
        maximum_factors=3,
        per_factor_coder_race={
            0: {"stratum": "edge", "skeleton_bytes": 7, "fiber_bytes": 11},
            1: {"stratum": "saddle", "skeleton_bytes": 13, "fiber_bytes": 5},
            2: {"stratum": "cell", "skeleton_bytes": 3, "fiber_bytes": 3},
        },
    )
    assert result["factors"][0]["representation"]["tag"] == "SKELETON"
    assert result["factors"][1]["representation"]["tag"] == "FIBER"
    assert result["factors"][2]["representation"]["status"] == "BLOCKED_CODER_RACE_TIE"
    assert result["factors"][2]["representation"]["tag"] is None


def test_jsonl_writer_is_atomic_and_refuses_untyped_rows(tmp_path) -> None:
    path = write_sense_jsonl_atomic([_row(0), _row(1)], tmp_path / "sense.jsonl")
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(LatticeSenseError, match="typed pair"):
        write_sense_jsonl_atomic([{"schema": "wrong"}], tmp_path / "bad.jsonl")
