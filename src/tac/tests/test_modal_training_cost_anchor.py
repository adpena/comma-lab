from __future__ import annotations

import json
from pathlib import Path

from tac.cost_band_calibration import load_anchors
from tac.deploy.modal.training_cost import (
    append_modal_training_cost_anchor,
    estimate_modal_training_cost_usd,
    normalize_modal_gpu,
)


def _metadata() -> dict:
    return {
        "label": "t1_balle_cheap_config_20260512T120000Z",
        "gpu": "T4",
        "score_claim": False,
        "promotion_eligible": False,
        "cost_band_anchor": {
            "schema": "modal_training_cost_anchor_metadata_v1",
            "trainer": "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
            "epochs": 3000,
            "batch_size": 16,
            "all_flags_on": True,
            "predicted_cost_usd_low": 1.0,
            "predicted_cost_usd_high": 3.0,
        },
    }


def test_estimate_modal_training_cost_uses_canonical_gpu_bucket() -> None:
    assert normalize_modal_gpu("a100-80gb") == "A100-80GB"
    assert normalize_modal_gpu("h100-smoke") == "H100"
    cost, rate = estimate_modal_training_cost_usd("T4", 1800.0)
    assert rate == 0.59
    assert cost == 0.295


def test_append_modal_training_cost_anchor_is_idempotent(tmp_path: Path) -> None:
    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    out_dir = tmp_path / "lane_t1_modal"  # FAKE_LANE_OK:test fixture directory name
    result = {
        "returncode": 0,
        "timed_out": False,
        "elapsed_seconds": 7200.0,
    }

    first = append_modal_training_cost_anchor(
        out_dir=out_dir,
        metadata=_metadata(),
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )
    second = append_modal_training_cost_anchor(
        out_dir=out_dir,
        metadata=_metadata(),
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )

    anchors = load_anchors(posterior)
    assert len(anchors) == 1
    assert anchors[0].dispatch_label == "t1_balle_cheap_config_20260512T120000Z"
    assert anchors[0].platform == "modal"
    assert anchors[0].gpu == "T4"
    assert anchors[0].actual_wall_clock_sec == 7200.0
    assert anchors[0].actual_cost_usd == 1.18
    assert anchors[0].prediction_in_band is True
    assert "cost_estimate_source=modal_elapsed_seconds_x_configured_hourly_rate" in anchors[0].notes
    assert first["appended"] is True
    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert first["cost_estimate"] is True
    assert second["appended"] is True
    assert second["already_appended"] is True
    marker = json.loads((out_dir / "cost_band_anchor_appended.json").read_text())
    assert marker["estimated_cost_usd"] == 1.18


def test_append_modal_training_cost_anchor_skips_without_metadata(tmp_path: Path) -> None:
    manifest = append_modal_training_cost_anchor(
        out_dir=tmp_path,
        metadata={"label": "no_cost_anchor", "gpu": "T4"},
        result={"elapsed_seconds": 60.0},
    )
    assert manifest["appended"] is False
    assert manifest["reason"] == "metadata_missing_cost_band_anchor"
    marker = json.loads((tmp_path / "cost_band_anchor_appended.json").read_text())
    assert marker["appended"] is False
    assert marker["score_claim"] is False
    assert marker["promotion_eligible"] is False


def test_append_modal_training_cost_anchor_skips_without_elapsed(tmp_path: Path) -> None:
    manifest = append_modal_training_cost_anchor(
        out_dir=tmp_path,
        metadata=_metadata(),
        result={"returncode": 0},
    )
    assert manifest["appended"] is False
    assert manifest["reason"] == "result_missing_numeric_elapsed_seconds"
    marker = json.loads((tmp_path / "cost_band_anchor_appended.json").read_text())
    assert marker["appended"] is False
    assert marker["score_claim"] is False
    assert marker["promotion_eligible"] is False
