from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/compare_ddm_cl1_hpac_capacity_mps.py"
SPEC = importlib.util.spec_from_file_location("compare_ddm_cl1_hpac_capacity_mps", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _result(device: str) -> dict:
    history = []
    for epoch, phase in ((1, "continuous"), (2, "continuous"), (4, "discrete_qat"), (6, "discrete_qat")):
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "bpp": 0.01 * epoch,
                "top1_error": 0.001 * epoch,
                "estimated_joint_bytes": 100_000 + epoch,
            }
        )
    return {
        "schema": "ddm_cl1_hpac_capacity_trainer_result.v1",
        "score_claim": False,
        "config": {
            "profile": "rx2_mc36",
            "epochs": 6,
            "qat_fraction": 0.5,
            "eval_every": 2,
            "device": device,
            "save": f"/{device}/save.pt",
            "out": f"/{device}/result.json",
            "seed": 20260716,
        },
        "run_identity": {
            "mps_trained": device == "mps",
            "port_mode": f"parity-{device}",
        },
        "history": history,
    }


def test_validate_results_and_matched_config() -> None:
    cpu = _result("cpu")
    mps = _result("mps")
    compare._validate_result(cpu, expected_device="cpu")
    compare._validate_result(mps, expected_device="mps")
    compare._validate_matched_config(cpu, mps)


def test_config_mismatch_is_refused() -> None:
    cpu = _result("cpu")
    mps = _result("mps")
    mps["config"]["seed"] = 3
    with pytest.raises(compare.ParityError, match="configs differ"):
        compare._validate_matched_config(cpu, mps)


def test_trajectory_reports_max_relative_divergence() -> None:
    cpu = _result("cpu")
    mps = _result("mps")
    mps["history"][2]["bpp"] *= 1.1
    rows, maxima = compare._trajectory(cpu, mps)
    assert len(rows) == 4
    assert maxima["bpp"] == pytest.approx(0.1)
    assert maxima["top1_error"] == 0.0
    assert maxima["estimated_joint_bytes"] == 0.0


def test_zero_relative_delta_is_defined() -> None:
    assert compare._relative_delta(0.0, 0.0) == 0.0
    assert compare._relative_delta(0.0, 1.0) == float("inf")


def test_comparator_sources_are_content_pinned() -> None:
    compare._validate_source_pins()


def test_race_projection_is_one_subtraction() -> None:
    result = compare._race_projection(
        cpu_elapsed_s=7200.0,
        mps_elapsed_s=1800.0,
        live_cpu_epoch=12,
    )
    assert result["measured_port_speedup"] == 4.0
    assert result["projected_cpu_remaining_hours"] == 16.0
    assert result["projected_full_mps_hours"] == 5.0
    assert result["finish_margin_hours"] == 11.0
    assert result["mps_finishes_first_if_cadence_holds"] is True


def test_timing_receipt_and_live_epoch_are_measured(tmp_path: Path) -> None:
    receipt = tmp_path / "parity.done"
    receipt.write_text(
        json.dumps(
            {
                "schema": "detached_local_process_done.v2",
                "rc": 0,
                "elapsed_s": 123.5,
            }
        )
    )
    log = tmp_path / "live.log"
    log.write_text(
        "noise\n"
        + json.dumps(
            {
                "epoch": 8,
                "phase": "continuous",
                "estimated_joint_bytes": 142000,
            }
        )
        + "\n"
    )
    assert compare._completed_elapsed_s(receipt) == 123.5
    assert compare._latest_live_cpu_epoch(log) == 8
