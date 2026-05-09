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
        "canonical_score_recomputed": 0.19663589,
        "canonical_score_source": "score_recomputed_from_components",
        "reported_final_score_display_rounded": 0.2,
        "score_rounding_abs_delta": 0.00336411,
        "score_reported_rounded_differs_from_canonical": True,
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


def test_gha_cpu_parse_report_uses_recomputed_score_as_canonical(tmp_path: Path) -> None:
    tool = _load_tool()
    report = tmp_path / "report.txt"
    report.write_text(
        "\n".join(
            [
                "=== Evaluation results over 600 samples ===",
                "  Average PoseNet Distortion: 0.00003286",
                "  Average SegNet Distortion: 0.00056023",
                "  Submission file size: 178,262 bytes",
                "  Original uncompressed size: 37,545,489 bytes",
                "  Compression Rate: 0.00474789",
                "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19",
            ]
        ),
        encoding="utf-8",
    )

    parsed = tool.parse_report(report)

    assert parsed["reported_final_score_display_rounded"] == 0.19
    assert parsed["canonical_score_source"] == "score_recomputed_from_components"
    assert parsed["canonical_score"] == parsed["score_recomputed_from_components"]
    assert parsed["canonical_score"] == parsed["canonical_score_recomputed"]
    assert parsed["canonical_score"] > 0.192
    assert parsed["score_reported_rounded_differs_from_canonical"] is True


def test_gha_cpu_dispatch_argparse_exposes_auto_create_fork_pr_flags() -> None:
    """Smoke test: argparse must expose --submission-dir and --auto-create-fork-pr."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr!r}"
    out = proc.stdout
    assert "--submission-dir" in out
    assert "--auto-create-fork-pr" in out
    assert "create_fork_pr_for_submission.py" in out
