from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "experiments" / "ddm_dt1_ans_decode_wallclock.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ddm_dt1_wallclock_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fit_scaling_selects_exact_linear_form_and_reports_residuals() -> None:
    module = load_module()
    result = module.fit_scaling([(2, 5.0), (8, 17.0), (32, 65.0), (120, 241.0)])

    assert result["selected_form"] == "linear"
    assert result["linear"]["slope_s_per_frame"] == pytest.approx(2.0)
    assert result["linear"]["intercept_s"] == pytest.approx(1.0)
    assert result["selected_prediction_n600_s"] == pytest.approx(1201.0)
    assert max(abs(value) for value in result["linear"]["residuals_s"].values()) < 1e-9


def test_fit_scaling_keeps_signed_delta_on_linear_surface() -> None:
    module = load_module()
    result = module.fit_scaling([(2, -0.02), (8, 0.01), (32, 0.05), (120, 0.12)])

    assert result["selected_form"] == "linear"
    assert result["power"] is None
    assert set(result["linear"]["residuals_s"]) == {"2", "8", "32", "120"}


def test_retention_boundaries_include_verdict_points_and_periodic_checkpoints() -> None:
    module = load_module()
    boundaries = module.retention_boundaries()

    assert boundaries[0] == 0
    assert boundaries[-1] == 600
    assert {2, 8, 32, 120, 600}.issubset(boundaries)
    assert max(right - left for left, right in itertools.pairwise(boundaries)) <= 24


def test_probe_rejects_a_two_point_scaling_curve() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="at least three"):
        module.fit_scaling([(2, 1.0), (8, 4.0)])


def test_recover_completed_decode_requires_downstream_render_proof(tmp_path: Path) -> None:
    module = load_module()
    progress = {
        "completed_frames": 600,
        "elapsed_s": 12.5,
        "components_s": {
            "prepare_frame_context_s": 1.0,
            "selected_logits_s": 9.0,
            "probability_table_s": 1.0,
            "coder_s": 0.5,
            "state_update_s": 0.5,
        },
    }
    (tmp_path / "range_decode_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    bundle = SimpleNamespace(raw_tokens=torch.zeros((600, 1, 1), dtype=torch.uint8))

    assert module._recover_completed_range_decode(bundle, tmp_path) is None

    render = {"master_completed": 24, "carrier_completed": 0}
    (tmp_path / "render_state.json").write_text(json.dumps(render), encoding="utf-8")
    recovered = module._recover_completed_range_decode(bundle, tmp_path)

    assert recovered is not None
    assert recovered["exact_target_equality"] is True
    assert recovered["recovered_from_prior_completed_stage"] is True
    assert (tmp_path / "range_decode_result.json").exists()
