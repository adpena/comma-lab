# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
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


def test_called_preflight_check_names_omits_self_recursive_gate_call() -> None:
    module = _load_tool()

    def check_alpha(*, strict: bool) -> None:
        _ = strict

    def preflight_developer() -> None:
        preflight_developer()
        check_alpha(strict=True)

    names = module._called_preflight_check_names(preflight_developer)

    assert names == ["check_alpha"]


def test_called_preflight_calls_preserves_constant_strictness() -> None:
    module = _load_tool()

    def check_alpha(*, strict: bool, verbose: bool) -> None:
        _ = (strict, verbose)

    def check_beta(*, strict: bool, verbose: bool) -> None:
        _ = (strict, verbose)

    def sample_preflight_all() -> None:
        check_alpha(strict=True, verbose=False)
        check_beta(strict=False, verbose=False)

    calls = module._called_preflight_calls(sample_preflight_all)

    assert [(call.name, call.constant_kwargs["strict"]) for call in calls] == [
        ("check_alpha", True),
        ("check_beta", False),
    ]


def test_called_preflight_calls_records_local_import_module() -> None:
    module = _load_tool()

    def sample_preflight_all() -> None:
        from tac.preflight_runtime_refs import check_shell_script_runtime_refs_resolve

        check_shell_script_runtime_refs_resolve(strict=False, verbose=True)

    calls = module._called_preflight_calls(sample_preflight_all)

    assert len(calls) == 1
    assert calls[0].name == "check_shell_script_runtime_refs_resolve"
    assert calls[0].import_module == "tac.preflight_runtime_refs"


def test_select_preflight_calls_accepts_substring_filters_in_preflight_order() -> None:
    module = _load_tool()
    calls = [
        module.PreflightCall("preflight_filename_contract"),
        module.PreflightCall("preflight_loader_format_safety"),
        module.PreflightCall("check_no_bare_round_in_eval_roundtrip"),
        module.PreflightCall("check_no_eval_roundtrip_false"),
    ]

    selected, matches = module._select_preflight_calls(
        calls,
        ["filename_contract,no_bare_round", "eval_roundtrip"],
    )

    assert [call.name for call in selected] == [
        "preflight_filename_contract",
        "check_no_bare_round_in_eval_roundtrip",
        "check_no_eval_roundtrip_false",
    ]
    assert matches["filename_contract"] == ["preflight_filename_contract"]
    assert matches["eval_roundtrip"] == [
        "check_no_bare_round_in_eval_roundtrip",
        "check_no_eval_roundtrip_false",
    ]


def test_profile_preflight_checks_runs_selected_check_with_profile_kwargs(
    monkeypatch,
) -> None:
    module = _load_tool()
    events: list[tuple[str, bool, bool]] = []

    def check_alpha(*, strict: bool, verbose: bool) -> None:
        events.append(("alpha", strict, verbose))

    def check_beta(*, strict: bool, verbose: bool) -> None:
        events.append(("beta", strict, verbose))

    def preflight_all() -> None:
        check_alpha(strict=True, verbose=True)
        check_beta(strict=False, verbose=True)

    fake_preflight = SimpleNamespace(
        preflight_all=preflight_all,
        check_alpha=check_alpha,
        check_beta=check_beta,
    )
    real_import_module = module.importlib.import_module

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: (
            fake_preflight
            if name == "tac.preflight"
            else real_import_module(name)
        ),
    )

    profile = module._profile_preflight_checks(
        check_filters=["beta"],
        use_fs_cache=False,
        timeout_s=None,
        check_timeout_s=None,
    )

    assert profile.status == "passed"
    assert [step.name for step in profile.steps] == ["check_beta"]
    assert events == [("beta", False, False)]
    assert profile.metadata["selected_checks"] == ["check_beta"]


def test_profile_preflight_checks_resolves_local_imported_check(monkeypatch) -> None:
    module = _load_tool()
    events: list[tuple[str, bool, bool]] = []

    def local_check(*, strict: bool, verbose: bool) -> None:
        events.append(("local", strict, verbose))

    def preflight_all() -> None:
        # preflight-test-fixture-import: `local_check` is provided by the
        # `fake_runtime_refs` SimpleNamespace below via monkeypatched
        # `importlib.import_module`. The static `from ... import local_check`
        # form is REQUIRED so `tools/profile_preflight_latency.py`'s AST visitor
        # in `_called_preflight_calls` can detect this as a preflight check.
        from tac.preflight_runtime_refs import local_check

        local_check(strict=False, verbose=True)

    fake_preflight = SimpleNamespace(preflight_all=preflight_all)
    fake_runtime_refs = SimpleNamespace(local_check=local_check)
    real_import_module = module.importlib.import_module

    def fake_import_module(name: str):
        if name == "tac.preflight":
            return fake_preflight
        if name == "tac.preflight_runtime_refs":
            return fake_runtime_refs
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    profile = module._profile_preflight_checks(
        check_filters=["local_check"],
        use_fs_cache=False,
        timeout_s=None,
        check_timeout_s=None,
    )

    assert profile.status == "passed"
    assert [step.name for step in profile.steps] == ["local_check"]
    assert events == [("local", False, False)]


def test_profile_preflight_dev_cli_runs_bounded_developer_gate(monkeypatch) -> None:
    module = _load_tool()
    events: list[tuple[str, bool, bool]] = []

    def check_alpha(*, strict: bool, verbose: bool) -> None:
        events.append(("alpha", strict, verbose))

    fake_preflight = SimpleNamespace(DEFAULT_PREFLIGHT_CLI_TIMEOUT_S=30.0)
    fake_preflight.check_alpha = check_alpha

    def preflight_developer(**kwargs: object) -> None:
        assert kwargs["check_codebase"] is True
        assert kwargs["verbose"] is False
        assert kwargs["use_fs_cache"] is True
        assert kwargs["wall_clock_budget_s"] is None
        fake_preflight.check_alpha(strict=True, verbose=False)

    fake_preflight.preflight_developer = preflight_developer
    real_import_module = module.importlib.import_module
    monkeypatch.setattr(
        module,
        "_called_preflight_check_names",
        lambda func: ["check_alpha"] if func is preflight_developer else [],
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: (
            fake_preflight
            if name == "tac.preflight"
            else real_import_module(name)
        ),
    )

    profile = module._profile_preflight_dev_cli(use_fs_cache=True, timeout_s=None)

    assert profile.status == "passed"
    assert profile.metadata["timeout_s"] == 30.0
    assert profile.metadata["mirrors_cli_scope"] == "dev"
    assert [step.name for step in profile.steps] == ["check_alpha"]
    assert events == [("alpha", True, False)]


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


def test_profile_zig_source_scan_records_native_leaf_metadata(monkeypatch, tmp_path: Path) -> None:
    module = _load_tool()
    binary = tmp_path / "source_needle_scan"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    events: list[str] = []

    def fake_build_zig_source_scanner() -> Path:
        events.append("build")
        return binary

    def fake_run_zig_source_scan(**kwargs: object) -> dict[str, object]:
        events.append("scan")
        assert kwargs["build"] is False
        assert Path(kwargs["binary_path"]) == binary
        assert kwargs["dirs"] == module.ZIG_SOURCE_SCAN_DIRS
        return {
            "schema": "pact.zig_source_needle_scan.v1",
            "file_count": 12,
            "match_count": 7,
            "matches": [],
        }

    fake_tool = SimpleNamespace(
        build_zig_source_scanner=fake_build_zig_source_scanner,
        run_zig_source_scan=fake_run_zig_source_scan,
    )
    monkeypatch.setattr(module, "_load_tool_module", lambda _path, _name: fake_tool)

    profile = module._profile_zig_source_scan()

    assert profile.status == "passed"
    assert events == ["build", "scan"]
    assert [step.name for step in profile.steps] == [
        "build native scanner",
        "scan source substrings",
    ]
    assert profile.metadata["file_count"] == 12
    assert profile.metadata["match_count"] == 7
