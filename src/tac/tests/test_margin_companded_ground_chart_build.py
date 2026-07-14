"""Build-only guards for the default-OFF S1 margin-companded ground chart."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tac.boundary_math.inverse_depth_compander import (
    MEASURED_HORIZON_ROW,
    MEASURED_SOFTENING_OFFSET_ROWS,
)
from tac.witness_control.resume_registry import DIRECT_CONTROLLER_NAMES
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl.lever_registry import (
    lever_factories,
    name_composable_levers,
    resolve_composable_lever,
)

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("_margin_compander_trainer", TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dsl_factory_is_visible_default_off_and_uses_only_real_flags() -> None:
    lever = cd.MarginCompandedGroundChart()
    assert lever.name == "margin_companded_ground_chart"
    assert lever.overrides == {
        "--ground-frame-chart": True,
        "--margin-companded-ground-chart": True,
        "--margin-compander-horizon-row": MEASURED_HORIZON_ROW,
        "--margin-compander-softening-offset-rows": MEASURED_SOFTENING_OFFSET_ROWS,
        "--margin-compander-seed": 0,
    }
    assert "MarginCompandedGroundChart" in lever_factories()
    assert "MarginCompandedGroundChart" in name_composable_levers()
    assert resolve_composable_lever("MarginCompandedGroundChart") == lever
    assert set(lever.overrides) <= set(cd.real_trainer_flags())


def test_dsl_factory_round_trips_through_real_trainer_parser() -> None:
    lever = cd.MarginCompandedGroundChart()
    argv = ["--out-dir", "experiments/results/test_margin_compander_parser"]
    for flag, value in lever.overrides.items():
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            argv.extend((flag, str(value)))
    parsed = cd.build_real_trainer_parser().parse_args(argv)
    assert parsed.ground_frame_chart is True
    assert parsed.margin_companded_ground_chart is True
    assert parsed.margin_compander_horizon_row == MEASURED_HORIZON_ROW
    assert parsed.margin_compander_softening_offset_rows == MEASURED_SOFTENING_OFFSET_ROWS
    assert parsed.margin_compander_seed == 0


def test_trainer_composes_after_ground_chart_and_registers_resume_identity() -> None:
    source = TRAINER.read_text()
    build_at = source.index("_gfc_chart = GroundFrameChart.build(")
    compose_at = source.index("_input_chart = MarginCompandedGroundChart.compose(")
    register_at = source.index('"margin_compander", MARGIN_COMPANDER_RESUME_PREFIX')
    assert build_at < compose_at < register_at
    assert "margin_compander" in DIRECT_CONTROLLER_NAMES


def test_resume_guard_detects_compander_identity_drift() -> None:
    trainer = _load_trainer()
    args = SimpleNamespace(
        ground_frame_chart=True,
        margin_companded_ground_chart=True,
        margin_compander_horizon_row=MEASURED_HORIZON_ROW,
        margin_compander_softening_offset_rows=MEASURED_SOFTENING_OFFSET_ROWS,
        margin_compander_seed=0,
    )
    cfg = {
        "__cfg_ground_frame_chart": np.int64(1),
        "__cfg_margin_companded_ground_chart": np.int64(1),
        "__cfg_margin_compander_horizon_row": np.float64(MEASURED_HORIZON_ROW),
        "__cfg_margin_compander_softening_offset_rows": np.float64(
            MEASURED_SOFTENING_OFFSET_ROWS
        ),
        "__cfg_margin_compander_seed": np.int64(0),
    }
    assert trainer._resume_lever_divergences(cfg, args) == []
    args.margin_compander_softening_offset_rows += 1.0
    assert any(
        "margin_compander_softening_offset_rows" in row
        for row in trainer._resume_lever_divergences(cfg, args)
    )


def test_default_parser_keeps_compander_and_base_chart_off() -> None:
    parsed = cd.build_real_trainer_parser().parse_args(
        ["--out-dir", "experiments/results/test_margin_compander_default"]
    )
    assert parsed.ground_frame_chart is False
    assert parsed.margin_companded_ground_chart is False
