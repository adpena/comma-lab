from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "profile_preflight_latency.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("profile_preflight_latency_test", TOOL)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_called_preflight_check_names_filters_recursive_and_preserves_order() -> None:
    module = _load_tool()

    def check_alpha(*, strict: bool) -> None:
        _ = strict

    def preflight_beta(*, verbose: bool) -> None:
        _ = verbose

    def preflight_all() -> None:
        pass

    def helper() -> None:
        pass

    def sample_preflight_all() -> None:
        check_alpha(strict=True)
        preflight_beta(verbose=False)
        check_alpha(strict=False)
        preflight_all()
        helper()

    names = module._called_preflight_check_names(sample_preflight_all)

    assert names == ["check_alpha", "preflight_beta"]


def test_build_report_sorts_hot_steps_and_keeps_surface_step_count() -> None:
    module = _load_tool()
    slow = module.StepTiming("surface-b", "slow", 2.0, "passed")
    fast = module.StepTiming("surface-b", "fast", 0.1, "passed")
    failed = module.StepTiming("surface-a", "failed", 1.5, "failed", detail="boom")
    surfaces = [
        module.SurfaceProfile("surface-b", "passed", 2.1, [fast, slow]),
        module.SurfaceProfile("surface-a", "failed", 1.5, [failed], error_type="RuntimeError", error="boom"),
    ]

    report = module._build_report(
        surfaces,
        top=2,
        max_step_records=1,
        generated_at="2026-05-08T00:00:00Z",
    )

    assert report["schema"] == module.PROFILE_SCHEMA
    assert report["generated_at"] == "2026-05-08T00:00:00Z"
    assert report["failed_surface_count"] == 1
    assert [row["name"] for row in report["hot_steps"]] == ["slow", "failed"]

    by_surface = {row["name"]: row for row in report["surfaces"]}
    assert by_surface["surface-b"]["step_count"] == 2
    assert [row["name"] for row in by_surface["surface-b"]["steps"]] == ["slow"]


def test_print_report_handles_minimal_report(capsys) -> None:
    module = _load_tool()
    surface = module.SurfaceProfile(
        "dispatch-hazards",
        "passed",
        0.25,
        [module.StepTiming("dispatch-hazards", "discover scan files", 0.01, "passed")],
        metadata={"file_count": 4, "hazard_count": 0},
    )
    report = module._build_report(
        [surface],
        top=1,
        max_step_records=10,
        generated_at="2026-05-08T00:00:00Z",
    )

    module._print_report(report, top=1)

    out = capsys.readouterr().out
    assert "Preflight latency profile" in out
    assert "dispatch-hazards" in out
    assert "hazard_count=0" in out


def test_failed_step_summary_names_failed_all_lanes_rows() -> None:
    module = _load_tool()
    failed = [
        module.StepTiming("all-lanes", "GATE #10: untracked source inventory", 0.4, "failed"),
        module.StepTiming("all-lanes", "LANE #3: sidechannels", 0.1, "failed"),
    ]

    summary = module._failed_step_summary(failed)

    assert "Failed all-lanes step(s):" in summary
    assert "GATE #10: untracked source inventory (failed)" in summary
    assert "LANE #3: sidechannels (failed)" in summary
