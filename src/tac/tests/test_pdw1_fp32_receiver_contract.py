"""Tests for the PDW1 frozen-fp32 receiver contract (task #543 closure)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.pdw1_fp32_receiver_contract import (
    CONTRACT_ID,
    FRAME195_CACHED_LSTAR,
    FRAME195_GENERIC_F64_ARGMAX,
    FRAME195_QUOTIENT,
    contract_f32_assign,
    contract_f32_power_scores,
)
from tac.boundary_math.power_diagram_witness import (
    PowerDiagramWitnessError,
    make_power_diagram_target,
    power_assign,
)


def _toy_target():
    sites = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    weights = np.asarray([0.1, -0.05, -0.05], dtype=np.float32)
    weights = weights - weights.mean()
    return make_power_diagram_target(
        sites, weights, adjacency=((0, 1), (0, 2), (1, 2))
    )


def test_contract_id_is_pinned() -> None:
    assert CONTRACT_ID == "pdw1-native-f32-power-first-max.v1"


def test_scores_match_reference_formula() -> None:
    target = _toy_target()
    z = np.asarray([[0.25, 0.75], [2.0, -1.0]], dtype=np.float32)
    scores = contract_f32_power_scores(z, target)
    assert scores.dtype == np.float32
    assert scores.shape == (2, 3)
    sites = target.sites.astype(np.float32)
    weights = target.weights.astype(np.float32)
    for row, point in enumerate(z):
        for k in range(3):
            dot = np.sum(sites[k] * point, dtype=np.float32)
            norm = np.sum(sites[k] * sites[k], dtype=np.float32)
            expected = np.float32(2.0) * dot + weights[k] - norm
            assert scores[row, k] == expected


def test_assign_agrees_with_generic_away_from_ties() -> None:
    target = _toy_target()
    rng = np.random.default_rng(7)
    z = rng.normal(size=(4096, 2)).astype(np.float32)
    ours = contract_f32_assign(z, target)
    generic = power_assign(z.astype(np.float64), target)
    # Random Gaussian points essentially never land on an fp32 tie.
    assert np.array_equal(ours, generic)


def test_first_max_tie_rule_is_lowest_class() -> None:
    # Symmetric two-site geometry: the midpoint scores tie exactly in fp32.
    sites = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    weights = np.asarray([0.0, 0.0], dtype=np.float32)
    target = make_power_diagram_target(sites, weights, adjacency=((0, 1),))
    z = np.zeros((1, 2), dtype=np.float32)
    scores = contract_f32_power_scores(z, target)
    assert scores[0, 0] == scores[0, 1]
    assert contract_f32_assign(z, target)[0] == 0


def test_refuses_non_float32_points() -> None:
    target = _toy_target()
    with pytest.raises(PowerDiagramWitnessError, match="float32"):
        contract_f32_power_scores(np.zeros((1, 2), dtype=np.float64), target)


def test_refuses_bad_trailing_dimension_and_nonfinite() -> None:
    target = _toy_target()
    with pytest.raises(PowerDiagramWitnessError, match="trailing"):
        contract_f32_power_scores(np.zeros((1, 3), dtype=np.float32), target)
    bad = np.full((1, 2), np.nan, dtype=np.float32)
    with pytest.raises(PowerDiagramWitnessError, match="non-finite"):
        contract_f32_power_scores(bad, target)


def test_frame195_fixture_closes_under_contract() -> None:
    """The canonical #543 blocker pixel: f64 flips to class 1; the contract
    produces an exact fp32 tie and first-max class 0 == cached L*."""

    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    receipt_path = (
        repo_root / ".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json"
    )
    if not receipt_path.is_file():
        pytest.skip("sealed frame-195 diagnostic receipt not present on this host")
    from tac.boundary_math.power_diagram_witness import decode_pdw1

    receipt = json.loads(receipt_path.read_text())
    target = decode_pdw1(bytes.fromhex(receipt["frozen_target"]["pdw1_hex"]))
    z = np.asarray([FRAME195_QUOTIENT], dtype=np.float32)
    scores = contract_f32_power_scores(z, target)
    assert scores[0, 0] == scores[0, 1]  # exact fp32 tie
    assert int(contract_f32_assign(z, target)[0]) == FRAME195_CACHED_LSTAR
    generic = int(power_assign(z.astype(np.float64), target)[0])
    assert generic == FRAME195_GENERIC_F64_ARGMAX  # the reproduced f64 flip
