# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "plan_pr106_component_moving_cells.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_pr106_component_moving_cells", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_latent_plan_ranks_charged_cells_and_keeps_false_authority(tmp_path: Path) -> None:
    module = _load_module()
    table_path = tmp_path / "score_table.npy"
    manifest_path = tmp_path / "score_table_manifest.json"
    xray_path = tmp_path / "pair_component_xray.json"
    table = np.array(
        [
            [1.0, 0.90, 0.80, 0.70, 0.60],
            [2.0, 1.99, 1.98, 1.97, 1.96],
        ],
        dtype=np.float32,
    )
    np.save(table_path, table, allow_pickle=False)
    _write_json(
        manifest_path,
        {
            "manifest_schema": "pr106_latent_score_table_manifest_v1",
            "score_claim": False,
            "ready_for_builder": True,
            "ready_for_exact_eval_dispatch": False,
            "device": "cuda:0",
            "latent_dim": 2,
            "delta_radius": 1,
            "score_table_shape": [2, 5],
            "noop_candidate_index": 0,
        },
    )
    _write_json(
        xray_path,
        {
            "schema": "pair_component_error_xray_v1",
            "rows": [
                {
                    "pair_idx": 0,
                    "component_score_no_rate": 0.4,
                    "pose_score_contribution": 0.3,
                    "seg_score_contribution": 0.1,
                    "frame0_l1": 10.0,
                    "frame1_l1": 11.0,
                }
            ],
        },
    )

    plan = module.build_plan(
        score_table_npy=table_path,
        score_table_manifest=manifest_path,
        xray_json=xray_path,
        top_k=2,
        cell_byte_delta=0.0,
        label="fixture",
    )

    assert plan["schema"] == "pr106_component_moving_cell_plan_v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["kind"] == "latent_sidecar"
    assert plan["axis_labels"]["source_score_table"] == "[compress-time CUDA scorer table]"
    assert plan["table_summary"]["component_improving_cell_count"] == 8
    assert plan["top_cells"][0]["cell_id"] == "latent_sidecar:row0:candidate4"
    assert plan["top_cells"][0]["candidate"] == {"dim": 1, "delta_q": 1}
    assert plan["top_cells"][0]["component_score_delta_no_rate"] == pytest.approx(-0.4)
    assert plan["top_cells"][0]["net_score_delta_charged"] == pytest.approx(-0.4)
    assert plan["top_cells"][0]["xray_pair_context"]["component_score_no_rate"] == 0.4
    assert plan["top_cells"][0]["false_authority"]["rank_or_kill_eligible"] is False
    assert "requires_paired_contest_cuda_auth_eval" in plan["dispatch_blockers"]


def test_yshift_plan_infers_frame_locations_and_rate_delta(tmp_path: Path) -> None:
    module = _load_module()
    table_path = tmp_path / "score_table.npy"
    manifest_path = tmp_path / "score_table_manifest.json"
    candidates = module._build_yshift_candidate_grid(1)
    zero_idx = int(np.flatnonzero((candidates == 0).all(axis=1))[0])
    table = np.full((3, candidates.shape[0]), 1.0, dtype=np.float32)
    table[:, zero_idx] = 0.5
    table[1, zero_idx + 1] = 0.1
    np.save(table_path, table, allow_pickle=False)
    _write_json(
        manifest_path,
        {
            "manifest_schema": "pr106_yshift_score_table_manifest_v1",
            "score_claim": False,
            "ready_for_builder": True,
            "ready_for_exact_eval_dispatch": False,
            "device": "cuda:0",
            "candidate_radius": 1,
            "score_table_shape": [3, int(candidates.shape[0])],
            "zero_candidate_index": zero_idx,
        },
    )

    plan = module.build_plan(
        score_table_npy=table_path,
        score_table_manifest=manifest_path,
        top_k=1,
        cell_byte_delta=3.0,
    )

    top = plan["top_cells"][0]
    expected_rate = 25.0 * 3.0 / 37_545_489
    assert plan["kind"] == "yshift"
    assert top["row_idx"] == 1
    assert top["pair_idx"] == 0
    assert top["frame_slot"] == 1
    assert top["component_score_delta_no_rate"] == pytest.approx(-0.4)
    assert top["rate_score_delta"] == pytest.approx(expected_rate)
    assert top["net_score_delta_charged"] == pytest.approx(-0.4 + expected_rate)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    table_path = tmp_path / "score_table.npy"
    manifest_path = tmp_path / "score_table_manifest.json"
    out_json = tmp_path / "plan.json"
    out_md = tmp_path / "plan.md"
    np.save(
        table_path,
        np.array([[1.0, 0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        allow_pickle=False,
    )
    _write_json(
        manifest_path,
        {
            "manifest_schema": "pr106_latent_score_table_manifest_v1",
            "score_claim": False,
            "ready_for_builder": True,
            "ready_for_exact_eval_dispatch": False,
            "device": "cuda:0",
            "latent_dim": 2,
            "delta_radius": 1,
            "score_table_shape": [1, 5],
            "noop_candidate_index": 0,
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--score-table-npy",
            str(table_path),
            "--score-table-manifest",
            str(manifest_path),
            "--top-k",
            "1",
            "--cell-byte-delta",
            "0",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["top_cells"][0]["cell_id"] == "latent_sidecar:row0:candidate4"
    assert "PR106 Component-Moving Cell Plan" in out_md.read_text(encoding="utf-8")
