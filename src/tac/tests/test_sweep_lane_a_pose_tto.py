"""Tests for experiments/sweep_lane_a_pose_tto.py.

Coverage:
  * search space well-defined (all bounds, kinds, citations)
  * non-negotiables enforced (eval_roundtrip True, device cuda)
  * factory build_sweep returns a properly-configured BayesianSweep
  * predicted band tagged on the sweep (provenance discipline)
  * --smoke runs end-to-end without GPU
  * trial result parser handles malformed inputs (no silent failures)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SWEEP_PATH = REPO / "experiments" / "sweep_lane_a_pose_tto.py"


@pytest.fixture(scope="module")
def lane_a_module():
    """Load experiments/sweep_lane_a_pose_tto.py as a module."""
    spec = importlib.util.spec_from_file_location("sweep_lane_a_pose_tto", SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sweep_lane_a_pose_tto"] = module
    spec.loader.exec_module(module)
    return module


def test_search_space_has_required_keys(lane_a_module):
    space = lane_a_module.LANE_A_SEARCH_SPACE
    for required in ("tto_steps", "batch_pairs", "tto_lr", "posetto_noise_std",
                     "eval_roundtrip", "device"):
        assert required in space, f"missing key {required!r} in LANE_A_SEARCH_SPACE"


def test_eval_roundtrip_pinned_true(lane_a_module):
    er = lane_a_module.LANE_A_SEARCH_SPACE["eval_roundtrip"]
    assert er == ("fixed", True), f"eval_roundtrip must be fixed True, got {er!r}"


def test_device_pinned_cuda(lane_a_module):
    dev = lane_a_module.LANE_A_SEARCH_SPACE["device"]
    assert dev == ("fixed", "cuda"), f"device must be fixed cuda, got {dev!r}"


def test_search_space_bounds_literature_informed(lane_a_module):
    """Bounds must match the docstring rationale."""
    space = lane_a_module.LANE_A_SEARCH_SPACE
    assert space["tto_steps"] == ("int", 200, 1500)
    assert space["batch_pairs"] == ("int", 4, 16)
    assert space["tto_lr"] == ("loguniform", 1e-3, 1e-1)
    assert space["posetto_noise_std"] == ("uniform", 0.1, 1.0)


def test_predicted_band_tagged(lane_a_module):
    band = lane_a_module.LANE_A_PREDICTED_BAND
    assert isinstance(band, tuple) and len(band) == 2
    assert band[0] < band[1]
    assert band == (0.95, 1.15)


def test_baseline_hand_tuned_recorded(lane_a_module):
    # CLAUDE.md "no signal loss" — the hand-tuned anchor MUST be in the file.
    assert lane_a_module.LANE_A_BASELINE_HAND_TUNED == 1.15


def test_template_path_exists(lane_a_module):
    template = lane_a_module.LANE_A_TEMPLATE
    assert template.exists(), f"template script missing: {template}"


def test_build_sweep_returns_configured_sweep(lane_a_module, tmp_path):
    sweep = lane_a_module.build_sweep(tmp_path / "out", n_trials=4)
    assert sweep.name == "lane_a_pose_tto"
    assert sweep.objective == "auth_score"
    assert sweep.direction == "minimize"
    assert sweep.n_trials == 4
    assert sweep.predicted_band == (0.95, 1.15)
    assert sweep.search_space_hash and len(sweep.search_space_hash) == 16


def test_build_sweep_template_has_all_placeholders(lane_a_module, tmp_path):
    """The remote template MUST contain every search-space placeholder."""
    sweep = lane_a_module.build_sweep(tmp_path / "out", n_trials=2)
    text = sweep._read_template()  # raises if any placeholder missing
    assert "__PARAM_TTO_STEPS__" in text
    assert "__PARAM_BATCH_PAIRS__" in text
    assert "__PARAM_TTO_LR__" in text
    assert "__PARAM_POSETTO_NOISE_STD__" in text


def test_smoke_mode_runs_end_to_end(lane_a_module, tmp_path):
    pytest.importorskip("optuna")
    rc = lane_a_module.main([
        "--n-trials", "3",
        "--output-dir", str(tmp_path / "out"),
        "--smoke",
    ])
    assert rc == 0
    results_path = tmp_path / "out" / "lane_a_sweep_results.json"
    assert results_path.exists()
    summary = json.loads(results_path.read_text())
    assert summary["sweep_name"] == "lane_a_pose_tto"
    assert summary["n_trials"] == 3
    assert summary["best_value"] is not None
    assert summary["predicted_band"] == [0.95, 1.15]


def test_smoke_writes_trial_history(lane_a_module, tmp_path):
    pytest.importorskip("optuna")
    rc = lane_a_module.main([
        "--n-trials", "2",
        "--output-dir", str(tmp_path / "out"),
        "--smoke",
    ])
    assert rc == 0
    history = (tmp_path / "out" / "trial_history.jsonl").read_text().splitlines()
    assert len(history) == 2
    parsed = [json.loads(line) for line in history]
    for record in parsed:
        assert record["sweep_name"] == "lane_a_pose_tto"
        assert "tto_steps" in record["params"]
        assert "tto_lr" in record["params"]


def test_optimized_remote_script_emit_pure(lane_a_module, tmp_path):
    """_emit_optimized_remote_script must NOT execute, just write."""
    out = lane_a_module._emit_optimized_remote_script(
        tmp_path,
        best_params={
            "tto_steps": 800,
            "batch_pairs": 12,
            "tto_lr": 0.005,
            "posetto_noise_std": 0.4,
            "eval_roundtrip": True,
            "device": "cuda",
        },
        best_value=1.05,
    )
    assert out.exists()
    text = out.read_text()
    # Concrete values rendered in.
    assert "800" in text and "0.005" in text
    # Sidecar JSON also written.
    sidecar = out.with_suffix(".params.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["best_value"] == 1.05
    assert payload["sweep_name"] == "lane_a_pose_tto"


def test_argparse_does_not_invent_flags(lane_a_module):
    """Defensive: every parse_args entry maps to a known concept.

    Mitigates the dead-flag wiring trap from CLAUDE.md (memory:
    feedback_dead_flag_wiring_pattern).
    """
    ns = lane_a_module.parse_args([])
    # Defaults for every documented flag.
    assert hasattr(ns, "n_trials")
    assert hasattr(ns, "objective")
    assert hasattr(ns, "output_dir")
    assert hasattr(ns, "smoke")
    # Sane defaults.
    assert ns.n_trials == 30
    assert ns.objective == "auth_score"
    assert ns.smoke is False


def test_parse_remote_result_handles_malformed_json(lane_a_module, tmp_path):
    """Malformed sidecar JSON should fail loud, not silent."""
    sweep = lane_a_module.build_sweep(tmp_path / "out", n_trials=1)
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    sidecar = script_path.with_suffix(".result.json")
    sidecar.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        sweep.parse_remote_result(script_path)


def test_parse_remote_result_handles_missing_file(lane_a_module, tmp_path):
    sweep = lane_a_module.build_sweep(tmp_path / "out", n_trials=1)
    script_path = tmp_path / "out" / "trial.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# ...")
    with pytest.raises(FileNotFoundError):
        sweep.parse_remote_result(script_path)
