# SPDX-License-Identifier: MIT
"""Tests for Alpha primitive mask diagnostics."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_primitive_mask_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("alpha_primitive_mask_diagnostics", MODULE_PATH)
assert SPEC is not None
diagnostics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnostics
SPEC.loader.exec_module(diagnostics)


def _safe_archive(tmp_path: Path, mask_bytes: bytes = b"mask-bytes") -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
    return archive


def _tiny_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0, 0, 1, 1, 1],
                [0, 2, 2, 1, 1],
                [3, 3, 2, 4, 4],
                [3, 0, 4, 4, 4],
            ],
            [
                [0, 1, 1, 1, 1],
                [0, 2, 2, 2, 1],
                [3, 3, 2, 4, 4],
                [3, 0, 0, 4, 4],
            ],
        ],
        dtype=torch.int64,
    )


def _base_report(tmp_path: Path) -> dict[str, Any]:
    archive = _safe_archive(tmp_path)
    _member_data, source_meta = diagnostics._read_archive_member(archive, "masks.mkv")
    config = diagnostics.DiagnosticConfig(max_components_per_class=1)
    return diagnostics._build_diagnostic_report_from_masks(
        masks=_tiny_masks(),
        source_meta=source_meta,
        config=config,
        command=["alpha_primitive_mask_diagnostics.py", "--unit-test"],
    )


def test_read_archive_member_rejects_requested_traversal(tmp_path: Path) -> None:
    archive = _safe_archive(tmp_path)

    with pytest.raises(ValueError, match="unsafe archive member path"):
        diagnostics._read_archive_member(archive, "../masks.mkv")


def test_read_archive_member_rejects_zip_slip_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("../escape", b"bad")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        diagnostics._read_archive_member(archive, "masks.mkv")


def test_read_archive_member_rejects_hidden_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("._masks.mkv", b"sidecar")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        diagnostics._read_archive_member(archive, "masks.mkv")


def test_diagnostic_report_is_deterministic_and_records_primitives(tmp_path: Path) -> None:
    report_a = _base_report(tmp_path)
    report_b = _base_report(tmp_path)

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
    assert report_a["schema"] == "alpha_primitive_mask_diagnostics_v1"
    assert report_a["evidence_grade"] == "empirical"
    assert report_a["source"]["decoded_masks"]["shape"] == [2, 4, 5]
    assert report_a["source"]["analyzed_masks"]["class_histogram"] == {
        "0": 8,
        "1": 10,
        "2": 7,
        "3": 6,
        "4": 9,
    }

    primitives = report_a["diagnostics"]
    assert primitives["summary"]["total_components_by_class"] == {
        "0": 4,
        "1": 2,
        "2": 2,
        "3": 2,
        "4": 2,
    }
    assert primitives["temporal"]["pair_count"] == 1
    assert primitives["temporal"]["total_changed_pixels"] == 3
    assert primitives["temporal"]["changed_pixels_by_pair"] == [
        {"from_frame": 0, "to_frame": 1, "changed_pixels": 3, "changed_fraction": 0.15}
    ]

    frame0_class0 = primitives["frames"][0]["classes"]["0"]
    assert frame0_class0["component_count"] == 2
    assert frame0_class0["emitted_component_count"] == 1
    assert frame0_class0["omitted_component_count"] == 1
    assert frame0_class0["emitted_components"][0]["area"] == 3
    assert frame0_class0["emitted_components"][0]["bbox_xyxy_exclusive"] == [0, 0, 2, 2]
    assert frame0_class0["emitted_components"][0]["centroid_xy"] == [0.333333, 0.333333]


def test_report_cannot_promote_or_claim_score(tmp_path: Path) -> None:
    report = _base_report(tmp_path)

    def walk(node: Any) -> list[dict[str, Any]]:
        if isinstance(node, dict):
            out = [node]
            for value in node.values():
                out.extend(walk(value))
            return out
        if isinstance(node, list):
            out = []
            for value in node:
                out.extend(walk(value))
            return out
        return []

    for item in walk(report):
        if "score_claim" in item:
            assert item["score_claim"] is False
        if "promotion_eligible" in item:
            assert item["promotion_eligible"] is False
        if "evidence_grade" in item:
            assert item["evidence_grade"] == "empirical"

    assert report["local_diagnostic_only"] is True
    assert report["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in report["canonical_score_source_required"]
    assert "Exact CUDA auth eval" in report["score_claim_warning"]


def test_cli_default_is_bounded() -> None:
    parser = diagnostics._build_arg_parser()
    args = parser.parse_args([])

    assert args.max_frames == diagnostics.DEFAULT_MAX_FRAMES
    assert args.all_frames is False
