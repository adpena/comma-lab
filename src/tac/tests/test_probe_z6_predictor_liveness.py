# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_z6_predictor_liveness.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("z6_liveness_tool", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_z6_liveness_probe_finds_zero_ego_cargo_cult_condition() -> None:
    tool = _load_tool()

    payload = tool.build_probe_payload(seed=0)
    rows = {row["mode"]: row for row in payload["rows"]}

    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["paradigm_claim_allowed"] is False
    assert payload["verdict"] == (
        "z6_predictor_live_ramp_smoke_exercises_film_synthetic_only"
    )

    zero = rows["full_film_zero_ego"]
    assert zero["predictor_param_count"] > 0
    assert zero["predictor_gradient_l2"] > 0.0
    assert zero["predictor_gradient_live"] is True
    assert zero["ego_motion_nonzero_fraction"] == 0.0
    assert zero["film_conditioning_exercised"] is False

    ramp = rows["full_film_ramp_ego"]
    assert ramp["predictor_param_count"] == zero["predictor_param_count"]
    assert ramp["predictor_gradient_l2"] > 0.0
    assert ramp["ego_motion_nonzero_fraction"] > 0.0
    assert ramp["predictor_output_delta_vs_zero_ego_l2"] > 0.0
    assert ramp["film_conditioning_exercised"] is True


def test_z6_liveness_probe_identity_control_has_no_predictor_params() -> None:
    tool = _load_tool()

    row = tool.measure_liveness_row(
        ego_motion_mode="zero",
        identity_predictor=True,
        seed=0,
    )

    assert row["mode"] == "identity_predictor_control"
    assert row["predictor_param_count"] == 0
    assert row["predictor_gradient_l2"] == 0.0
    assert row["predictor_gradient_live"] is False
    assert row["verdict"] == "identity_control_no_predictor_params"


def test_z6_liveness_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / ".omx/research/z6_liveness.json"
    output_md = tmp_path / ".omx/research/z6_liveness.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json.relative_to(tmp_path)),
            "--output-md",
            str(output_md.relative_to(tmp_path)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == (
        "z6_predictor_live_ramp_smoke_exercises_film_synthetic_only"
    )
    assert output_md.read_text(encoding="utf-8").startswith(
        "# L5 v2 Z6 predictor liveness probe"
    )
    assert "score_claim=false" in proc.stdout
