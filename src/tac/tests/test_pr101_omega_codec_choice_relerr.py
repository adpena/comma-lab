"""Regression tests for PR101 omega codec-choice proxy aggregation."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "pr101_omega_opt_per_tensor_codec_choice_empirical.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "pr101_omega_opt_per_tensor_codec_choice_empirical_under_test",
        TOOL_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_greedy_rel_err_is_element_weighted_not_tensor_weighted():
    mod = _load_tool()
    per_tensor = [
        {
            "tensor": "large",
            "n_elements": 1_000_000,
            "codecs": [
                {"label": "lossless", "bytes": 1000, "rel_err": 0.0, "alpha": 0.0},
                {"label": "lossy", "bytes": 100, "rel_err": 0.1, "alpha": 0.3},
            ],
        },
        {
            "tensor": "tiny",
            "n_elements": 1,
            "codecs": [
                {"label": "lossless", "bytes": 1000, "rel_err": 0.0, "alpha": 0.0},
                {"label": "lossy", "bytes": 100, "rel_err": 0.0, "alpha": 0.3},
            ],
        },
    ]

    selected = mod.greedy_select_under_rel_err_budget(per_tensor, 0.2)

    assert selected["rel_err_form"] == "element_weighted_l2_over_per_tensor_rel_err"
    assert selected["total_elements"] == 1_000_001
    assert selected["achieved_total_rel_err_element_weighted_l2"] == selected[
        "achieved_total_rel_err_l2_avg"
    ]
    assert abs(selected["achieved_total_rel_err_l2_avg"] - 0.09999995) < 1e-8
