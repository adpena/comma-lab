# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "backfill_master_gradient_score_axis_dominance.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "backfill_master_gradient_score_axis_dominance",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backfill_tool_appends_locked_anchor_and_writes_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    archive = "f" * 64
    gradient = np.array([[10.0, 0.01, 0.0], [0.01, 10.0, 0.0]], dtype=np.float64)
    gradient_path = tmp_path / "aggregate.npy"
    np.save(gradient_path, gradient)
    anchor = {
        "archive_sha256": archive,
        "scored_archive_sha256": archive,
        "scored_archive_bytes": 12345,
        "gradient_array_path": str(gradient_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection",
        "measurement_call_id": "fixture-call",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 2,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
        "written_at_utc": "2026-05-18T01:00:01Z",
    }
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "backfill_manifest.json"

    payload = tool.build_backfill_manifest(
        archive_sha256=archive,
        anchor_path=ledger,
        manifest_path=manifest_path,
    )

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert payload["schema"] == tool.SCHEMA
    assert payload["append_performed"] is True
    assert payload["score_claim"] is False
    assert payload["corrected_score_axis_dominance"]["pose_axis_dominant_byte_count"] == 1
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["anchor_sha256_before"] != manifest["anchor_sha256_after"]


def test_backfill_tool_skips_existing_dominance_without_force(tmp_path: Path) -> None:
    tool = _load_tool()
    archive = "a" * 64
    gradient_path = tmp_path / "aggregate.npy"
    np.save(gradient_path, np.ones((1, 3), dtype=np.float64))
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(gradient_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 1,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "score_axis_dominance": {"schema": "master_gradient_score_axis_dominance_v1"},
    }
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(anchor) + "\n", encoding="utf-8")

    payload = tool.build_backfill_manifest(
        archive_sha256=archive,
        anchor_path=ledger,
        manifest_path=tmp_path / "manifest.json",
    )

    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert payload["append_performed"] is False
    assert payload["skip_reason"] == "score_axis_dominance_already_present"
