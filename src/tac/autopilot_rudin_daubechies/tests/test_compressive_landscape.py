# SPDX-License-Identifier: MIT
"""Tests for compressive sensing landscape recovery (Phase 4)."""
from __future__ import annotations

import pytest

from tac.autopilot_rudin_daubechies import (
    CompressiveSensingLandscapeRecovery,
    LandscapeCell,
)


def test_landscape_construction_validates_lengths():
    with pytest.raises(ValueError, match="length"):
        CompressiveSensingLandscapeRecovery(
            cell_coordinates=((0.0,), (1.0,)),
            cell_ids=("a",),  # length mismatch
        )


def test_landscape_from_substrate_axis():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(
        ["z3", "z4", "d1", "d4"]
    )
    assert lr.n_cells == 4
    assert lr.cell_ids == ("z3", "z4", "d1", "d4")


def test_landscape_no_anchors_returns_uniform_zero():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a", "b", "c"])
    cells = lr.reconstruct()
    assert len(cells) == 3
    for c in cells:
        assert c.predicted_score == 0.0
        assert c.measured is False


def test_landscape_anchor_recovers_exactly_at_measured_cell():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a", "b", "c"])
    lr.add_anchor("b", 0.20)
    cells = lr.reconstruct()
    measured = [c for c in cells if c.measured]
    assert len(measured) == 1
    assert measured[0].cell_id == "b"
    assert measured[0].predicted_score == 0.20
    assert measured[0].uncertainty_band == 0.0


def test_landscape_unobserved_cells_carry_uncertainty():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a", "b", "c"])
    lr.add_anchor("b", 0.20)
    cells = lr.reconstruct()
    unobserved = [c for c in cells if not c.measured]
    assert len(unobserved) == 2
    for c in unobserved:
        assert c.uncertainty_band > 0.0


def test_landscape_uncertainty_decreases_with_more_anchors():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(
        ["a", "b", "c", "d", "e", "f", "g", "h"]
    )
    lr.add_anchor("a", 0.20)
    cells_one = lr.reconstruct()
    band_one = next(c.uncertainty_band for c in cells_one if c.cell_id == "h")
    # Add 3 more anchors.
    lr.add_anchor("b", 0.21)
    lr.add_anchor("c", 0.19)
    lr.add_anchor("d", 0.20)
    cells_four = lr.reconstruct()
    band_four = next(c.uncertainty_band for c in cells_four if c.cell_id == "h")
    # More anchors -> smaller uncertainty.
    assert band_four < band_one


def test_landscape_predict_cell_returns_pair():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["x", "y"])
    lr.add_anchor("x", 0.18)
    pred, band = lr.predict_cell("y")
    assert isinstance(pred, float)
    assert band > 0.0


def test_landscape_predict_unknown_cell_raises():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a"])
    with pytest.raises(ValueError, match="not in landscape"):
        lr.predict_cell("nonexistent")


def test_landscape_add_anchor_rejects_unknown():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a"])
    with pytest.raises(ValueError, match="not in landscape"):
        lr.add_anchor("nonexistent", 0.20)


def test_landscape_add_anchor_rejects_inf_score():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a"])
    with pytest.raises(ValueError, match="finite"):
        lr.add_anchor("a", float("inf"))


def test_landscape_add_anchor_replaces_previous_for_same_cell():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a"])
    lr.add_anchor("a", 0.20)
    lr.add_anchor("a", 0.25)  # replace
    pred, _ = lr.predict_cell("a")
    assert pred == 0.25
    assert lr.n_anchors == 1


def test_landscape_l1_reconstruction_smooth_between_anchors():
    """Cells between two close-anchored values should interpolate smoothly."""
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(
        ["a", "b", "c", "d", "e"]
    )
    lr.add_anchor("a", 0.10)
    lr.add_anchor("e", 0.30)
    cells = lr.reconstruct()
    pred_a = next(c.predicted_score for c in cells if c.cell_id == "a")
    pred_c = next(c.predicted_score for c in cells if c.cell_id == "c")
    pred_e = next(c.predicted_score for c in cells if c.cell_id == "e")
    # End cells exact; middle cell between the anchors.
    assert pred_a == 0.10
    assert pred_e == 0.30
    assert 0.10 < pred_c < 0.30


def test_landscape_confidence_tag_carries_n_anchors():
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(["a"])
    lr.add_anchor("a", 0.20)
    tag = lr.confidence_tag()
    assert "compressive-sensing-L1-reconstruction" in tag
    assert "n=1-anchor-posterior" in tag


def test_landscape_continual_learning_loop():
    """Simulate the continual-learning loop: add anchors over time, check
    each query reflects the latest evidence."""
    lr = CompressiveSensingLandscapeRecovery.from_substrate_axis(
        [f"s{i}" for i in range(10)]
    )
    # Round 1
    lr.add_anchor("s0", 0.30)
    pred_round1, _ = lr.predict_cell("s5")
    # Round 2: add a contradictory anchor
    lr.add_anchor("s5", 0.18)
    pred_round2, _ = lr.predict_cell("s5")
    assert pred_round2 == 0.18  # measured cells are exact
    # Round 3: add more anchors to refine
    lr.add_anchor("s9", 0.25)
    cells = lr.reconstruct()
    assert len(cells) == 10
