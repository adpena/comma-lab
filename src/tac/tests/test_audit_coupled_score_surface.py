# SPDX-License-Identifier: MIT
"""Tests for the canonical-pointer-driven coupled score-surface audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "audit_coupled_score_surface.py"
SPEC = importlib.util.spec_from_file_location("audit_coupled_score_surface", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_reads_target_from_pointer_and_composes_all_axes(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.json"
    manifest = tmp_path / "manifest.json"
    _write_json(
        pointer,
        {
            "effective_frontier": {
                "score": 0.172,
                "axis": "official_leaderboard",
                "custody": "external target only",
                "evidence_grade": "[official-leaderboard display]",
                "score_precision": "official_display",
                "source": "upstream_official_leaderboard",
                "source_kind": "external_public_leaderboard_target",
            }
        },
    )
    _write_json(
        manifest,
        {
            "points": [
                {
                    "id": "rate_heavy",
                    "d_seg": 0.0002,
                    "d_pose": 0.00002331,
                    "archive_bytes": 200_000,
                    "custody": "derived planning point",
                }
            ],
            "transitions": [
                {
                    "id": "seg_worse_rate_better",
                    "before": {
                        "d_seg": 0.00055961,
                        "d_pose": 0.00002942,
                        "archive_bytes": 177_169,
                    },
                    "after": {
                        "d_seg": 0.0006,
                        "d_pose": 0.00002942,
                        "archive_bytes": 167_169,
                    },
                }
            ],
        },
    )

    report = MODULE.build_report(manifest_path=manifest, pointer_path=pointer)

    assert report["target"]["score"] == 0.172
    assert report["target"]["axis"] == "official_leaderboard"
    assert report["independent_component_thresholds_are_admission_rules"] is False
    assert report["points"][0]["audit"]["inside_strict_sublevel"] is True
    transition = report["transitions"][0]["audit"]
    assert transition["seg_term_delta"] > 0.0
    assert transition["rate_term_delta"] < 0.0
    assert transition["improves_score"] is True
    assert report["score_claim"] is False


def test_report_fails_closed_without_effective_pointer_score(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.json"
    manifest = tmp_path / "manifest.json"
    _write_json(pointer, {"effective_frontier": {}})
    _write_json(manifest, {"points": [], "transitions": []})

    with pytest.raises(MODULE.CoupledScoreSurfaceError, match="lacks score"):
        MODULE.build_report(manifest_path=manifest, pointer_path=pointer)


def test_report_refuses_malformed_point_instead_of_inventing_axis_zero(
    tmp_path: Path,
) -> None:
    pointer = tmp_path / "pointer.json"
    manifest = tmp_path / "manifest.json"
    _write_json(pointer, {"effective_frontier": {"score": 0.172}})
    _write_json(
        manifest,
        {"points": [{"id": "missing_pose", "d_seg": 0.0, "archive_bytes": 1}]},
    )

    with pytest.raises(MODULE.CoupledScoreSurfaceError, match="requires numeric"):
        MODULE.build_report(manifest_path=manifest, pointer_path=pointer)
