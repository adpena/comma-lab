from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "dispatch_cpu_eval_via_github_actions.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_dispatch_cpu_eval_via_github_actions", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gha_cpu_dispatch_requires_pr_number_for_nonbaseline_runtime() -> None:
    tool = _load_tool()

    error = tool.submission_runtime_contract_error(
        "pr102_hnerv_lc_v2_scale095_rplus1_cpu_20260508T175932Z",
        None,
    )

    assert error is not None
    assert "--pr-number" in error
    assert "inflate.sh" in error
    assert "downloads only archive.zip" in error


def test_gha_cpu_dispatch_allows_baseline_without_pr_number() -> None:
    tool = _load_tool()

    assert tool.submission_runtime_contract_error("baseline", None) is None


def test_gha_cpu_dispatch_allows_pr_backed_submission_runtime() -> None:
    tool = _load_tool()

    assert tool.submission_runtime_contract_error("new_submission", "123") is None
