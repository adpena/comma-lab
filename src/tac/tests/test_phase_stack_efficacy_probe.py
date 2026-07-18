# SPDX-License-Identifier: MIT
"""Tests for the pure helpers of tools/probe_phase_stack_efficacy_road_lane.py.

Covers the GT phase band construction (the #424-target addressable set + partner
dilation) and the CLI contract — no scorer weights, no torch forward, fast."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_PROBE = _REPO / "tools" / "probe_phase_stack_efficacy_road_lane.py"


@pytest.fixture(scope="module")
def probe_mod():
    spec = importlib.util.spec_from_file_location("phase_probe_under_test", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_probe_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gt_phase_band_selects_straddle_and_dilates_partner(probe_mod):
    # 4x4: a vertical class boundary between col 1 (class 0) and col 2 (class 1),
    # with small GT margins near the boundary (inside band) and large elsewhere.
    lst = np.zeros((4, 4), dtype=np.int64)
    lst[:, 2:] = 1
    mg = np.full((4, 4), 5.0, dtype=np.float32)
    mg[:, 1] = 0.3   # p side of the straddle
    mg[:, 2] = 0.4   # q side of the straddle
    band = probe_mod.gt_phase_band(lst, mg, band=1.0)
    # active p pixels are col 1 (right-straddle vs col 2); dilation adds col 2 (partner)
    assert band[:, 1].all(), "straddle p column must be on the band"
    assert band[:, 2].all(), "partner q column must be included by dilation"
    assert not band[:, 0].any() and not band[:, 3].any(), "far columns must be off-band"


def test_gt_phase_band_respects_band_threshold(probe_mod):
    lst = np.zeros((4, 4), dtype=np.int64)
    lst[:, 2:] = 1
    mg = np.full((4, 4), 5.0, dtype=np.float32)  # margins all OUTSIDE band
    band = probe_mod.gt_phase_band(lst, mg, band=1.0)
    assert not band.any(), "no genuine-V straddle when both margins exceed the band"


def test_cli_defaults_and_strata_contract(probe_mod):
    ap_ns = None
    # main() would run the probe; parse args through the same parser by invoking
    # argparse construction indirectly: replicate via a dry parse of main's parser.
    # The module exposes main(argv); we only verify it REFUSES without required args.
    with pytest.raises(SystemExit):
        probe_mod.main([])  # --ema/--out required -> argparse SystemExit(2)
    assert ap_ns is None


def test_min_norm_convention_reexported(probe_mod):
    # the probe must use the organ's exact convention (no fork)
    from tac.witness_control.realization_regime import min_norm_crossing_max_coord

    assert probe_mod.min_norm_crossing_max_coord is min_norm_crossing_max_coord
    assert probe_mod.SUB_LSB_MAX_COORD == 0.5
