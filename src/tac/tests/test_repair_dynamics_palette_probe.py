# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS
from tac.optimization.repair_dynamics_palette_probe import (
    REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA,
    build_repair_dynamics_palette_probe_matrix,
    repair_mode_id_to_postfilter_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _work_order() -> dict[str, object]:
    return {
        "schema": "frontier_rate_attack_targeted_component_correction_work_order.v1",
        **_false_authority(),
        "acquisition_id": "targeted_component_correction_pr110_palette",
        "candidate_id": "pr110_palette_candidate",
        "correction_family": "repair_dynamics_frame0_palette_interaction_waterfill",
        "archive_path": "submissions/pr110/archive.zip",
        "repair_dynamics_palette_prior": {
            "schema": "frontier_rate_attack_repair_dynamics_palette_prior.v1",
            **_false_authority(),
            "source": "unit_test_fec6_k16",
            "palette_modes": list(FEC6_FIXED_K16_MODE_IDS),
            "mode_count": 16,
            "mode_family_counts": {
                "blue_chroma": 2,
                "geometry_roll": 1,
                "identity": 1,
                "luma_bias": 4,
                "rgb_bias": 8,
            },
            "zero_frame1_modes": True,
        },
    }


def test_repair_mode_id_to_postfilter_mode_maps_fec6_palette() -> None:
    assert repair_mode_id_to_postfilter_mode("none")["postfilter_mode"] == "none"
    assert repair_mode_id_to_postfilter_mode("frame0_luma_bias_+1")[
        "postfilter_mode"
    ] == "even_bias:1"
    assert repair_mode_id_to_postfilter_mode("frame0_rgb_bias_m2_p1_p1")[
        "postfilter_mode"
    ] == "even_rgb_bias:-2,1,1"
    assert repair_mode_id_to_postfilter_mode("frame0_blue_chroma_amp_3")[
        "postfilter_mode"
    ] == "even_grain_chroma:3"
    assert repair_mode_id_to_postfilter_mode("frame0_roll_dx+0_dy+1")[
        "postfilter_mode"
    ] == "even_translate:1,0"
    assert repair_mode_id_to_postfilter_mode("frame1_luma_bias_-2")[
        "postfilter_mode"
    ] == "odd_bias:-2"


def test_build_repair_dynamics_palette_probe_matrix_groups_modes() -> None:
    matrix = build_repair_dynamics_palette_probe_matrix(
        work_order=_work_order(),
        work_order_path="work_order.json",
        probe_output_dir="repair_probe",
        n_pairs=24,
        max_modes=80,
    )

    assert matrix["schema"] == REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA
    assert matrix["score_claim"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    assert matrix["palette_mode_count"] == 16
    assert matrix["unsupported_mode_count"] == 0
    assert "even_bias:1" in matrix["postfilter_modes"]
    assert "even_rgb_bias:-2,1,1" in matrix["postfilter_modes"]
    assert "even_grain_chroma:1" in matrix["postfilter_modes"]
    assert "even_translate:1,0" in matrix["postfilter_modes"]
    assert any(
        group["group_id"] == "frame1_counterfactual_null_probe_modes"
        and any(str(mode).startswith("odd_") for mode in group["modes"])
        for group in matrix["probe_groups"]
    )
    command = matrix["commands"][0]["command"]
    assert command[:2] == [
        ".venv/bin/python",
        "tools/run_mlx_scorer_response_from_local_advisory.py",
    ]
    assert "--allow-gpu-research-signal" in command
    assert "--allow-local-cpu-advisory-cache-identity" in command
    assert matrix["commands"][0]["resources"]["kind"] == "local_mlx"
    assert matrix["device"] == "mlx"


def test_repair_dynamics_palette_probe_matrix_cli_writes_artifact(tmp_path: Path) -> None:
    work_order_path = tmp_path / "work_order.json"
    matrix_path = tmp_path / "matrix.json"
    work_order_path.write_text(json.dumps(_work_order()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_repair_dynamics_palette_probe_matrix.py",
            "--work-order",
            str(work_order_path),
            "--matrix-out",
            str(matrix_path),
            "--probe-output-dir",
            str(tmp_path / "probe"),
            "--n-pairs",
            "8",
            "--max-modes",
            "48",
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["schema"] == REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA
    assert matrix["n_pairs"] == 8
    assert matrix["device"] == "mlx"
