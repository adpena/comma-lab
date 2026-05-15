# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.optimization.hdm8_cuda_selector_probe_plan import (
    HDM8CudaSelectorProbePlanError,
    build_hdm8_cuda_selector_probe_plan,
)

REPO = Path(__file__).resolve().parents[3]


def _sweep() -> dict[str, object]:
    return {
        "axis": "modal-t4-cuda-proxy-prefix",
        "archive_bytes": 186395,
        "archive_sha256": "a" * 64,
        "n_pairs": 4,
        "modes": [
            {
                "mode": "none",
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [0.001, 0.001, 0.001, 0.001],
                "pair_segnet_dist": [0.001, 0.001, 0.001, 0.001],
            },
            {
                "mode": "even_bias:1",
                "avg_posenet_dist": 0.000775,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [0.0001, 0.001, 0.001, 0.001],
                "pair_segnet_dist": [0.001, 0.001, 0.001, 0.001],
            },
            {
                "mode": "even_rgb_bias:-1,0.5,0.5",
                "avg_posenet_dist": 0.00055,
                "avg_segnet_dist": 0.001,
                "pair_posenet_dist": [0.001, 0.0001, 0.0001, 0.001],
                "pair_segnet_dist": [0.001, 0.001, 0.001, 0.001],
            },
        ],
    }


def test_hdm8_cuda_selector_probe_plan_emits_sparse_selector_configs() -> None:
    plan = build_hdm8_cuda_selector_probe_plan(
        _sweep(),
        evidence_source_path="cuda_prefix_fixture.json",
        max_atoms=4,
        prefix_sizes=[1, 2],
    )

    assert plan["schema"] == "hdm8_cuda_selector_probe_plan_v1"
    assert plan["score_claim"] is False
    assert plan["axis_rankable_for_probe_planning"] is True
    assert plan["candidate_atom_count"] == 3
    assert len(plan["probe_configs"]) == 2

    top1 = plan["probe_configs"][0]
    assert top1["name"] == "sparse_cuda_prefix_top001"
    assert top1["config"]["mode"] == "selector"
    assert top1["config"]["score_claim"] is False
    assert len(top1["config"]["selector_indices"]) == 4
    assert top1["config"]["proxy"]["positive"] is True
    assert top1["config"]["proxy"]["axis"] == "modal-t4-cuda-proxy-prefix"
    assert top1["config"]["proxy"]["selected_pair_count"] == 1

    top2 = plan["probe_configs"][1]
    assert top2["config"]["proxy"]["selected_pair_count"] == 2
    assert sum(1 for idx in top2["config"]["selector_indices"] if idx != 0) == 2
    assert "exact_cuda_auth_eval_missing_for_each_probe" in plan["dispatch_blockers"]


def test_hdm8_cuda_selector_probe_plan_cli_writes_configs(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(_sweep()), encoding="utf-8")
    out_dir = tmp_path / "plan"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hdm8_cuda_selector_probe_plan.py"),
            "--sweep-json",
            str(sweep_path),
            "--output-dir",
            str(out_dir),
            "--prefix-sizes",
            "1,2",
            "--max-atoms",
            "4",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads((out_dir / "probe_plan.json").read_text(encoding="utf-8"))
    assert len(plan["probe_configs"]) == 2
    for row in plan["probe_configs"]:
        assert (REPO / row["selector_config_path"]).is_file()
        assert "--selector-config-json" in row["packet_build_command"]
        assert "--pack-selector-into-archive" in row["packet_build_command"]
    assert (out_dir / "probe_plan.md").is_file()


def test_hdm8_cuda_selector_probe_plan_rejects_mps_axis() -> None:
    sweep = _sweep()
    sweep["axis"] = "local-mps-proxy-prefix"

    try:
        build_hdm8_cuda_selector_probe_plan(sweep, max_atoms=4, prefix_sizes=[1, 2])
    except HDM8CudaSelectorProbePlanError as exc:
        assert "CUDA-prefix" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("MPS proxy sweeps must not produce CUDA selector plans")
