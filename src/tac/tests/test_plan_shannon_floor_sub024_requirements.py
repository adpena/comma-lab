# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_shannon_floor_sub024_requirements.py"
SPEC = importlib.util.spec_from_file_location("plan_shannon_floor_sub024_requirements", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def test_sub024_byte_requirement_at_c067_distortion() -> None:
    component = {
        "score": 0.31561703078448233,
        "archive_bytes": 276_214,
        "distortion_score": 0.13169753078448234,
        "rate_score": 0.1839195,
    }

    requirement = planner.target_requirements(component, 0.24)

    assert requirement["feasible_by_rate_only_at_current_distortion"] is True
    assert requirement["max_archive_bytes_if_distortion_unchanged"] == 162_650
    assert requirement["bytes_to_remove_if_distortion_unchanged"] == 113_564
    assert requirement["feasible_by_distortion_only_at_current_bytes"] is True
    assert requirement["distortion_reduction_needed_at_current_bytes"] > 0.075


def test_sub024_stream_budget_requires_mask_after_renderer_to_11k() -> None:
    component = {
        "score": 0.31561703078448233,
        "archive_bytes": 276_214,
        "distortion_score": 0.13169753078448234,
        "rate_score": 0.1839195,
    }
    profile = {
        "streams": [
            {"name": "masks.mkv", "encoded_bytes": 219_472},
            {"name": "renderer.bin", "encoded_bytes": 55_965},
            {"name": "optimized_poses.bin", "encoded_bytes": 677},
        ]
    }
    requirement = planner.target_requirements(component, 0.24)

    scenarios = planner.stream_budget_scenarios(component, profile, [requirement])
    renderer_stack = next(item for item in scenarios if item["scenario"] == "renderer_to_11000_then_mask")

    assert renderer_stack["renderer_savings_bytes"] == 44_965
    assert renderer_stack["remaining_mask_savings_bytes"] == 68_599
    assert renderer_stack["target_mask_bytes"] == 150_873
    assert renderer_stack["stream_can_cover"] is True


def test_sub020_not_reachable_by_rate_only_at_current_distortion() -> None:
    component = {
        "score": 0.31561703078448233,
        "archive_bytes": 276_214,
        "distortion_score": 0.13169753078448234,
        "rate_score": 0.1839195,
    }

    requirement = planner.target_requirements(component, 0.20)

    assert requirement["feasible_by_rate_only_at_current_distortion"] is True
    assert requirement["bytes_to_remove_if_distortion_unchanged"] > 170_000
