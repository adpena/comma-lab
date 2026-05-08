from __future__ import annotations

import importlib.util
import json
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


def test_gha_cpu_dispatch_adjudicated_record_uses_requested_repo(tmp_path: Path) -> None:
    tool = _load_tool()
    parsed = {
        "canonical_score": 0.19663589,
        "avg_segnet_dist": 0.000589,
        "avg_posenet_dist": 0.0000358,
        "compression_rate": 0.00475136,
        "score_recomputed_from_components": 0.19663589,
        "n_samples": 1200,
        "report_text": "score: 0.19663589\n",
    }

    out = tool.write_adjudicated(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_path=tmp_path / "archive.zip",
        archive_sha="a" * 64,
        archive_size=178_981,
        parsed=parsed,
        run_id=25571618194,
        run_url="https://github.com/example/fork/actions/runs/25571618194",
        release_tag="cpu-eval-test",
        asset_url="https://github.com/example/fork/releases/download/cpu-eval-test/archive.zip",
        runner_os_release="ubuntu-24.04",
        evaluate_py_sha="b" * 64,
        submission_name="submission_on_pr_branch",
        dispatched_at="2026-05-08T18:09:40+00:00",
        completed_at="2026-05-08T18:39:40+00:00",
        repo="example/fork",
    )

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["fork_repo"] == "example/fork"
