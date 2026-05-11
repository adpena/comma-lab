from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
        runtime_manifest={
            "runtime_tree_sha256": "c" * 64,
            "runtime_content_tree_sha256": "d" * 64,
        },
    )

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["fork_repo"] == "example/fork"
    assert record["runtime_tree_sha256"] == "c" * 64
    assert record["runtime_content_tree_sha256"] == "d" * 64
    assert record["provenance"]["device"] == "cpu"
    assert record["provenance"]["inflate_runtime_manifest"]["runtime_tree_sha256"] == "c" * 64


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


def test_gha_cpu_run_log_matching_rejects_concurrent_wrong_submission(monkeypatch) -> None:
    tool = _load_tool()

    def fake_run(cmd, check, capture_output, text):
        assert "--log" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout="submission_dir: submissions/other_concurrent_submission\n",
        )

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    assert tool.run_log_mentions_submission(
        123,
        "example/fork",
        "wanted_submission",
    ) is False


def test_gha_cpu_run_log_matching_accepts_requested_submission(monkeypatch) -> None:
    tool = _load_tool()

    def fake_run(cmd, check, capture_output, text):
        assert "--log" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout="uv run evaluate.sh --submission-dir ./submissions/wanted_submission\n",
        )

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    assert tool.run_log_mentions_submission(
        123,
        "example/fork",
        "wanted_submission",
    ) is True


def test_gha_cpu_trigger_accepts_single_new_run_when_logs_unavailable(
    monkeypatch,
) -> None:
    tool = _load_tool()
    run_gh_calls: list[list[str]] = []

    def fake_run_gh(args, capture=True):
        run_gh_calls.append(args)
        if args[:2] == ["run", "list"] and args[-1] == "databaseId":
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"databaseId": 100}]),
                stderr="",
            )
        if args[:2] == ["workflow", "run"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["run", "list"] and args[-1] == "databaseId,status,createdAt":
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "databaseId": 200,
                            "status": "in_progress",
                            "createdAt": "2026-05-10T05:36:01Z",
                        },
                        {
                            "databaseId": 100,
                            "status": "completed",
                            "createdAt": "2026-05-10T05:00:00Z",
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected gh call: {args!r}")

    def fake_run(cmd, check, capture_output, text):
        assert "--log" in cmd
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="logs are unavailable until the run completes",
        )

    monkeypatch.setattr(tool, "run_gh", fake_run_gh)
    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)

    assert (
        tool.trigger_workflow(
            "wanted_submission",
            "https://example.invalid/archive.zip",
            "ubuntu-latest",
            "example/fork",
        )
        == 200
    )
    assert any(call[:2] == ["workflow", "run"] for call in run_gh_calls)


def test_gha_cpu_download_artifact_fallback_selects_matching_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run_gh(args, capture=True):
        calls.append(args)
        if "-n" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="no artifact")
        wrong = tmp_path / "eval-wrong"
        right = tmp_path / "eval-wanted_submission"
        wrong.mkdir(parents=True, exist_ok=True)
        right.mkdir(parents=True, exist_ok=True)
        (wrong / "report.txt").write_text(
            "submission_dir: submissions/wrong_submission\n",
            encoding="utf-8",
        )
        (right / "report.txt").write_text(
            "submission_dir: submissions/wanted_submission\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool, "run_gh", fake_run_gh)

    report = tool.download_artifact(
        123,
        "wanted_submission",
        "example/fork",
        tmp_path,
    )

    assert report == tmp_path / "eval-wanted_submission" / "report.txt"
    assert any("-n" in call for call in calls)


# ──────────────────────────────────────────────────────────────────────────
# HIGH 1 (codex round-2 review, 2026-05-09): exact-identity matching.
# ──────────────────────────────────────────────────────────────────────────


def test_apogee_does_not_cross_match_apogee_stack_b100() -> None:
    """`apogee` must NOT match a log mentioning `apogee_stack_b100`."""
    tool = _load_tool()
    log_text = (
        "uv run evaluate.sh --submission-dir ./submissions/apogee_stack_b100\n"
        "submission_dir: submissions/apogee_stack_b100\n"
    )
    assert tool._text_mentions_submission_exactly(log_text, "apogee") is False


def test_apogee_stack_b100_does_not_cross_match_apogee_stack_b100_v2() -> None:
    """`apogee_stack_b100` must NOT match a log for `apogee_stack_b100_v2`."""
    tool = _load_tool()
    log_text = (
        "submission_dir: submissions/apogee_stack_b100_v2\n"
        "uv run evaluate.sh --submission-dir submissions/apogee_stack_b100_v2\n"
    )
    assert (
        tool._text_mentions_submission_exactly(log_text, "apogee_stack_b100")
        is False
    )


def test_exact_match_still_works() -> None:
    """Exact identity match continues to succeed."""
    tool = _load_tool()
    log_text = (
        "submission_dir: submissions/apogee_stack_b100\n"
        "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19\n"
    )
    assert (
        tool._text_mentions_submission_exactly(log_text, "apogee_stack_b100")
        is True
    )


def test_exact_match_works_for_multiple_pattern_forms() -> None:
    """All four canonical pattern forms feed into the same token set."""
    tool = _load_tool()
    cases = [
        "submission_dir: submissions/apogee\n",
        "uv run evaluate.sh --submission-dir submissions/apogee\n",
        "uv run evaluate.sh --submission-dir=./submissions/apogee\n",
        "submission_name=apogee\n",
    ]
    for text in cases:
        assert tool._text_mentions_submission_exactly(text, "apogee") is True
    # And none of them give a false match for `apo` or `apogee_stack_b100`.
    for text in cases:
        assert tool._text_mentions_submission_exactly(text, "apo") is False
        assert (
            tool._text_mentions_submission_exactly(text, "apogee_stack_b100")
            is False
        )


def test_empty_submission_name_rejected() -> None:
    """Empty submission name MUST raise rather than degenerate-match anything."""
    import pytest as _pytest

    tool = _load_tool()
    with _pytest.raises(ValueError, match="non-empty"):
        tool._validate_submission_name("")
    with _pytest.raises(ValueError, match="non-empty"):
        tool._validate_submission_name("   ")
    with _pytest.raises(ValueError, match="None"):
        tool._validate_submission_name(None)  # type: ignore[arg-type]


def test_path_separator_in_submission_name_rejected() -> None:
    """A slash-bearing name (e.g., `submissions/apogee`) is rejected up front."""
    import pytest as _pytest

    tool = _load_tool()
    with _pytest.raises(ValueError, match="path separators"):
        tool._validate_submission_name("submissions/apogee")
    with _pytest.raises(ValueError, match="path separators"):
        tool._validate_submission_name("a/b")


def test_run_log_uses_exact_identity_for_concurrent_dispatch(monkeypatch) -> None:
    """Concurrent run that mentions `apogee_stack_b100` must NOT attach to `apogee`."""
    tool = _load_tool()

    def fake_run(cmd, check, capture_output, text):
        assert "--log" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "uv run evaluate.sh --submission-dir submissions/apogee_stack_b100\n"
                "Submission file size: 178,981 bytes\n"
            ),
        )

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    # The previous substring matcher returned True here. The exact-identity
    # matcher must return False.
    assert tool.run_log_mentions_submission(
        123,
        "example/fork",
        "apogee",
    ) is False


def test_download_artifact_fail_closed_on_ambiguity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Two distinct report.txt files each claiming the same submission_name → raise.

    Per HIGH 1 fix policy: never silently pick one when ambiguity is present.
    """
    import pytest as _pytest

    tool = _load_tool()

    def fake_run_gh(args, capture=True):
        if "-n" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="no artifact")
        # Plant two ambiguous reports that BOTH carry "submission_dir:
        # submissions/apogee" so the ambiguity check fires.
        a = tmp_path / "eval-apogee-a"
        b = tmp_path / "eval-apogee-b"
        a.mkdir(parents=True, exist_ok=True)
        b.mkdir(parents=True, exist_ok=True)
        (a / "report.txt").write_text(
            "submission_dir: submissions/apogee\nFinal score: 0.19\n",
            encoding="utf-8",
        )
        (b / "report.txt").write_text(
            "submission_dir: submissions/apogee\nFinal score: 0.21\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool, "run_gh", fake_run_gh)

    with _pytest.raises(tool.AmbiguousSubmissionMatchError) as excinfo:
        tool.download_artifact(
            123,
            "apogee",
            "example/fork",
            tmp_path,
        )
    err = excinfo.value
    assert err.submission_name == "apogee"
    assert len(err.candidates) == 2
    assert "report.txt" in err.candidates[0]


def test_download_artifact_does_not_cross_match_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`apogee` must NOT pick a report whose canonical token is `apogee_stack_b100`."""
    tool = _load_tool()

    def fake_run_gh(args, capture=True):
        if "-n" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="no artifact")
        long_dir = tmp_path / "eval-apogee_stack_b100"
        long_dir.mkdir(parents=True, exist_ok=True)
        (long_dir / "report.txt").write_text(
            "submission_dir: submissions/apogee_stack_b100\n"
            "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.21\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool, "run_gh", fake_run_gh)

    # Pre-fix, this would have picked the apogee_stack_b100 report. Post-fix
    # it must exit non-zero (no exact-identity match for `apogee`).
    import pytest as _pytest
    with _pytest.raises(SystemExit) as excinfo:
        tool.download_artifact(
            123,
            "apogee",
            "example/fork",
            tmp_path,
        )
    assert excinfo.value.code == 3


def test_extract_submission_dir_tokens_handles_path_suffix() -> None:
    """A captured group containing path noise must reduce to its basename."""
    tool = _load_tool()
    text = "submission_dir: submissions/apogee_stack_b100\n"
    tokens = tool._extract_submission_dir_tokens(text)
    assert tokens == {"apogee_stack_b100"}


def test_extract_submission_dir_tokens_collects_distinct_names() -> None:
    """Mixed log lines yield the union of all canonical names present."""
    tool = _load_tool()
    text = (
        "submission_dir: submissions/apogee\n"
        "submission_dir: submissions/apogee_stack_b100\n"
        "submission_name=baseline\n"
    )
    tokens = tool._extract_submission_dir_tokens(text)
    assert tokens == {"apogee", "apogee_stack_b100", "baseline"}


def test_run_log_accepts_when_log_lists_multiple_subs_including_ours(
    monkeypatch,
) -> None:
    """If the log mentions our submission AND others, we still match."""
    tool = _load_tool()

    def fake_run(cmd, check, capture_output, text):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "submission_dir: submissions/apogee_stack_b100\n"
                "submission_dir: submissions/apogee\n"
                "submission_name=baseline\n"
            ),
        )

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    assert tool.run_log_mentions_submission(
        123, "example/fork", "apogee"
    ) is True
    assert tool.run_log_mentions_submission(
        123, "example/fork", "apogee_stack_b100"
    ) is True
    assert tool.run_log_mentions_submission(
        123, "example/fork", "baseline"
    ) is True
    assert tool.run_log_mentions_submission(
        123, "example/fork", "apogee_stack_b100_v2"
    ) is False
