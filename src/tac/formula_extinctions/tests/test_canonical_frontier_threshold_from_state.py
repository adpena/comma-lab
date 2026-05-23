# SPDX-License-Identifier: MIT
"""Tests for Row #6 — Frontier threshold from canonical state."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tac.formula_extinctions.canonical_frontier_threshold_from_state import (
    FrontierThresholdInput,
    canonical_frontier_threshold_from_state,
)


def test_invalid_axis_raises():
    """Bad axis raises."""
    with pytest.raises(ValueError, match="axis"):
        FrontierThresholdInput(repo_root=Path("/tmp"), axis="mps")  # type: ignore[arg-type]


def test_invalid_repo_root_type_raises():
    """Non-Path repo_root raises."""
    with pytest.raises(ValueError, match="repo_root"):
        FrontierThresholdInput(repo_root="/tmp", axis="cpu")  # type: ignore[arg-type]


def test_delegates_to_frontier_scan():
    """Helper delegates to tac.frontier_scan.collect_all_anchors + best_per_axis."""

    class FakeAnchor:
        score = 0.19205
        lane_id = "fake_lane"
        hardware_substrate = "linux_x86_64_cpu"
        archive_sha256 = "fake_sha"

    with (
        patch("tac.frontier_scan.collect_all_anchors", return_value=[FakeAnchor()]),
        patch("tac.frontier_scan.best_per_axis", return_value={"contest_cpu": [FakeAnchor()]}),
    ):
        r = canonical_frontier_threshold_from_state(
            FrontierThresholdInput(repo_root=Path("/tmp"), axis="cpu")
        )
        assert r.solved_value == 0.19205
        assert r.intermediate_values["anchor_lane_id"] == "fake_lane"


def test_raises_when_no_qualifying_anchor():
    """RuntimeError when no qualifying anchor found."""
    with (
        patch("tac.frontier_scan.collect_all_anchors", return_value=[]),
        patch("tac.frontier_scan.best_per_axis", return_value={}),
        pytest.raises(RuntimeError, match="No qualifying"),
    ):
        canonical_frontier_threshold_from_state(
            FrontierThresholdInput(repo_root=Path("/tmp"), axis="cpu")
        )


def test_citation_catalog_316():
    """Catalog #316 citation present."""

    class FakeAnchor:
        score = 0.19205
        lane_id = "fake_lane"
        hardware_substrate = "linux_x86_64_cpu"
        archive_sha256 = "fake_sha"

    with (
        patch("tac.frontier_scan.collect_all_anchors", return_value=[FakeAnchor()]),
        patch("tac.frontier_scan.best_per_axis", return_value={"contest_cuda": [FakeAnchor()]}),
    ):
        r = canonical_frontier_threshold_from_state(
            FrontierThresholdInput(repo_root=Path("/tmp"), axis="cuda")
        )
        assert "Catalog #316" in r.literature_citation
