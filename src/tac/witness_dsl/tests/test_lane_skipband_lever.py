# SPDX-License-Identifier: MIT
"""DSL-leg tests for the ARM-C #524 LaneSkipBand Lever factory (triality: DSL holds the lever)."""
from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import LaneSkipBand, Lever
from tac.witness_dsl.lever_registry import lever_factories


def test_lane_skipband_returns_lever_with_expected_flags():
    lv = LaneSkipBand()
    assert isinstance(lv, Lever)
    assert lv.name == "lane_skipband"
    assert set(lv.overrides) == {
        "--lane-skipband-weight", "--lane-skipband-start-epoch", "--lane-skipband-dilate"}


def test_lane_skipband_default_values_and_types():
    lv = LaneSkipBand()
    assert lv.overrides["--lane-skipband-weight"] == 0.05
    assert lv.overrides["--lane-skipband-dilate"] == 2
    assert lv.overrides["--lane-skipband-start-epoch"] == 0
    assert isinstance(lv.overrides["--lane-skipband-weight"], float)
    assert isinstance(lv.overrides["--lane-skipband-dilate"], int)


def test_lane_skipband_custom_values():
    lv = LaneSkipBand(weight=0.2, dilate=3, start_epoch=300, window=50)
    assert lv.overrides["--lane-skipband-weight"] == 0.2
    assert lv.overrides["--lane-skipband-dilate"] == 3
    assert lv.overrides["--lane-skipband-start-epoch"] == 300
    assert lv.epochs_delta == 50


def test_lane_skipband_notes_carry_run_gated_honesty():
    assert "RUN-GATED" in LaneSkipBand().notes


def test_lever_registry_discovers_lane_skipband():
    facs = lever_factories()
    assert "LaneSkipBand" in facs
    assert facs["LaneSkipBand"] == frozenset(
        {"--lane-skipband-weight", "--lane-skipband-start-epoch", "--lane-skipband-dilate"})


def test_lane_skipband_flags_exist_in_trainer_argparse():
    # never-invent-flags: every emitted flag must exist in the LIVE levelset trainer argparse.
    from pathlib import Path
    src = (Path(__file__).resolve().parents[4]
           / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()
    for flag in LaneSkipBand().overrides:
        assert f'"{flag}"' in src, f"{flag} not found in trainer argparse"
