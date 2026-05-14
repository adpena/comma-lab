"""Tests for tools/trigger_gha_cpu_eval.py + tools/harvest_gha_cpu_eval.py.

These cover the queue-infrastructure layer that wraps the existing canonical
GHA dispatcher (tools/dispatch_cpu_eval_via_github_actions.py). The test
strategy is to inject lightweight fakes for gh CLI calls and subprocess
boundaries so we can assert behavior without hitting the GitHub API.

Per CLAUDE.md "Public Disclosure Hygiene" + "Forbidden score claims":
particular attention is paid to:
  - URL validation refuses non-release-asset URLs
  - hardware label / lane_tag is gated on runner == 'ubuntu-latest'
  - score_claim_valid is False until the harvest tool parses a real score
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
TRIGGER_TOOL = REPO / "tools" / "trigger_gha_cpu_eval.py"
HARVEST_TOOL = REPO / "tools" / "harvest_gha_cpu_eval.py"


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_tool():
    return _load(TRIGGER_TOOL, "_trigger_gha_cpu_eval_test")


@pytest.fixture
def harvest_tool():
    return _load(HARVEST_TOOL, "_harvest_gha_cpu_eval_test")


# --------------------------------------------------------------------------- #
# trigger_gha_cpu_eval.py validation                                           #
# --------------------------------------------------------------------------- #


def test_trigger_validate_archive_url_accepts_public_release_asset(trigger_tool):
    # Public, well-formed release asset URL on a public GH repo.
    trigger_tool.validate_archive_url(
        "https://github.com/adpena/comma_video_compression_challenge/"
        "releases/download/cpu-eval-test-20260512T120000Z/archive.zip"
    )


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "http://github.com/foo/bar/releases/download/tag/archive.zip",  # not https
        "https://example.com/foo/bar.zip",
        "https://github.com/foo/bar/blob/main/archive.zip",  # blob not release
        "https://github.com/foo/bar/raw/main/archive.zip",
        "https://gitlab.com/foo/bar/releases/download/tag/archive.zip",  # not github
    ],
)
def test_trigger_validate_archive_url_refuses_non_release_asset(trigger_tool, bad_url):
    with pytest.raises(SystemExit) as exc:
        trigger_tool.validate_archive_url(bad_url)
    assert "VALIDATION_ERROR" in str(exc.value)


def test_trigger_validate_archive_sha256_requires_64_hex(trigger_tool):
    # Canonical 64-hex digest (A1's actual archive sha) is accepted.
    trigger_tool.validate_archive_sha256(
        "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
    )
    for bad in [
        "",
        "abc",
        "x" * 64,  # non-hex
        "a" * 63,  # too short
        "a" * 65,  # too long
    ]:
        with pytest.raises(SystemExit):
            trigger_tool.validate_archive_sha256(bad)


def test_trigger_validate_archive_size_bytes_positive_int(trigger_tool):
    trigger_tool.validate_archive_size_bytes(178_262)
    for bad in [0, -1, "178262"]:  # type: ignore[list-item]
        with pytest.raises(SystemExit):
            trigger_tool.validate_archive_size_bytes(bad)


def test_trigger_validate_label_only_allows_safe_chars(trigger_tool):
    trigger_tool.validate_label("t1_balle_cheap_config_20260512T171203Z")
    trigger_tool.validate_label("a-b.c_1")
    for bad in ["", " ", "a/b", "a b", "a$b"]:
        with pytest.raises(SystemExit):
            trigger_tool.validate_label(bad)


def test_trigger_workflow_registered_parses_workflow_list(trigger_tool, monkeypatch):
    """workflow_registered returns True iff the workflow path ends with file."""
    fake_rows = [
        {"path": ".github/workflows/eval.yml", "name": "eval", "state": "active"},
        {"path": ".github/workflows/other.yml", "name": "other", "state": "active"},
    ]
    monkeypatch.setattr(
        trigger_tool,
        "run_gh",
        lambda args: SimpleNamespace(returncode=0, stdout=json.dumps(fake_rows), stderr=""),
    )
    assert trigger_tool.workflow_registered("any/repo", "eval.yml") is True
    assert trigger_tool.workflow_registered("any/repo", "missing.yml") is False


def test_trigger_workflow_registered_returns_false_on_gh_failure(trigger_tool, monkeypatch):
    monkeypatch.setattr(
        trigger_tool,
        "run_gh",
        lambda args: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    assert trigger_tool.workflow_registered("any/repo", "eval.yml") is False


def test_trigger_write_dispatch_metadata_round_trips_schema(tmp_path, trigger_tool):
    out_dir = tmp_path / "gha_dispatch"
    path = trigger_tool.write_dispatch_metadata(
        out_dir,
        label="t1_balle_test",
        submission_name="t1_balle_test",
        archive_url="https://github.com/foo/bar/releases/download/tag/archive.zip",
        archive_sha256="b" * 64,
        archive_size_bytes=178_262,
        repo="foo/bar",
        workflow_file="eval.yml",
        runner="ubuntu-latest",
        pr_number=None,
        run_id=25588422622,
        run_url="https://github.com/foo/bar/actions/runs/25588422622",
        dispatched_at_utc="2026-05-12T18:30:00Z",
        instance_job_id="gha_cpu_eval_t1_balle_test_20260512T183000Z",
        lane_id="gha_cpu_eval_t1_balle_test",
        trigger_status="ok",
    )
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "pact.gha_cpu_eval_dispatch_metadata.v1"
    assert data["score_claim_valid"] is False
    assert data["evidence_grade"] == "pending_harvest"
    assert data["lane_tag"] == "pending_harvest"
    assert data["promotion_eligible"] is False
    assert data["workflow_run_id"] == 25588422622
    assert data["fork_repo"] == "foo/bar"
    assert data["runner"] == "ubuntu-latest"


def test_trigger_skip_trigger_closes_claim_without_run(tmp_path, trigger_tool, monkeypatch):
    claims: list[dict] = []
    closes: list[dict] = []
    monkeypatch.setattr(trigger_tool, "workflow_registered", lambda *_: True)
    monkeypatch.setattr(
        trigger_tool,
        "claim_lane",
        lambda **kwargs: claims.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        trigger_tool,
        "close_lane_claim",
        lambda **kwargs: closes.append(kwargs) or 0,
    )

    rc = trigger_tool.main(
        [
            "--archive-url",
            "https://github.com/foo/bar/releases/download/tag/archive.zip",
            "--archive-sha256",
            "a" * 64,
            "--archive-size-bytes",
            "178262",
            "--label",
            "skip_trigger_test",
            "--output-dir",
            str(tmp_path / "gha_dispatch"),
            "--skip-trigger",
        ]
    )

    metadata = json.loads((tmp_path / "gha_dispatch" / "dispatch_metadata.json").read_text())
    assert rc == 0
    assert metadata["workflow_run_id"] is None
    assert metadata["trigger_status"] == "skipped_trigger_diagnostic"
    assert len(claims) == 1
    assert len(closes) == 1
    assert closes[0]["lane_id"] == claims[0]["lane_id"]
    assert closes[0]["instance_job_id"] == claims[0]["instance_job_id"]
    assert closes[0]["status"] == "refused_dispatch_skip_trigger_diagnostic"


def test_trigger_failure_closes_claim_without_run(tmp_path, trigger_tool, monkeypatch):
    claims: list[dict] = []
    closes: list[dict] = []
    monkeypatch.setattr(trigger_tool, "workflow_registered", lambda *_: True)
    monkeypatch.setattr(
        trigger_tool,
        "claim_lane",
        lambda **kwargs: claims.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        trigger_tool,
        "close_lane_claim",
        lambda **kwargs: closes.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        trigger_tool,
        "trigger_workflow_dispatch",
        lambda **_: (None, "could not identify new run id within 60s"),
    )

    rc = trigger_tool.main(
        [
            "--archive-url",
            "https://github.com/foo/bar/releases/download/tag/archive.zip",
            "--archive-sha256",
            "b" * 64,
            "--archive-size-bytes",
            "178262",
            "--label",
            "trigger_failure_test",
            "--output-dir",
            str(tmp_path / "gha_dispatch"),
        ]
    )

    metadata = json.loads((tmp_path / "gha_dispatch" / "dispatch_metadata.json").read_text())
    assert rc == 4
    assert metadata["workflow_run_id"] is None
    assert metadata["trigger_status"] == "could not identify new run id within 60s"
    assert len(claims) == 1
    assert len(closes) == 1
    assert closes[0]["lane_id"] == claims[0]["lane_id"]
    assert closes[0]["instance_job_id"] == claims[0]["instance_job_id"]
    assert closes[0]["status"] == "failed_gha_cpu_eval_trigger_no_run_id"


def test_trigger_workflow_dispatch_returns_unique_new_run(trigger_tool, monkeypatch):
    """trigger_workflow_dispatch identifies the new run via run-list delta."""
    call_state = {"step": 0}

    def fake_run_gh(args):
        cmd = " ".join(args)
        if cmd.startswith("run list"):
            # Step 0: pre-dispatch snapshot
            if call_state["step"] == 0:
                call_state["step"] = 1
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([{"databaseId": 100}, {"databaseId": 101}]),
                    stderr="",
                )
            # Step 1+: post-dispatch — new run 200 visible
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"databaseId": 200, "status": "queued", "createdAt": "x"},
                        {"databaseId": 100, "status": "completed", "createdAt": "x"},
                        {"databaseId": 101, "status": "completed", "createdAt": "x"},
                    ]
                ),
                stderr="",
            )
        if cmd.startswith("workflow run"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unhandled")

    monkeypatch.setattr(trigger_tool, "run_gh", fake_run_gh)
    monkeypatch.setattr(trigger_tool.time, "sleep", lambda *_: None)

    run_id, msg = trigger_tool.trigger_workflow_dispatch(
        repo="foo/bar",
        workflow_file="eval.yml",
        submission_name="t1_balle_test",
        submission_url="https://github.com/foo/bar/releases/download/tag/archive.zip",
        runner="ubuntu-latest",
        pr_number=None,
    )
    assert run_id == 200
    assert msg == "ok"


def test_trigger_build_run_url(trigger_tool):
    url = trigger_tool.build_run_url("adpena/comma_video_compression_challenge", 25588422622)
    assert (
        url
        == "https://github.com/adpena/comma_video_compression_challenge/actions/runs/25588422622"
    )


# --------------------------------------------------------------------------- #
# harvest_gha_cpu_eval.py                                                      #
# --------------------------------------------------------------------------- #


def test_harvest_parse_report_recomputes_canonical_score(tmp_path, harvest_tool):
    """parse_report matches the A1 reference (0.19284757...) within float epsilon."""
    report = tmp_path / "report.txt"
    report.write_text(
        "=== Evaluation results over 600 samples ===\n"
        "  Average PoseNet Distortion: 0.00003286\n"
        "  Average SegNet Distortion: 0.00056023\n"
        "  Submission file size: 178,262 bytes\n"
        "  Original uncompressed size: 37,545,489 bytes\n"
        "  Compression Rate: 0.00474789\n"
        "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19\n",
        encoding="utf-8",
    )
    parsed = harvest_tool.parse_report(report)
    expected = (
        100.0 * 0.00056023
        + math.sqrt(10.0 * 0.00003286)
        + 25.0 * 0.00474789
    )
    assert parsed["n_samples"] == 600
    assert math.isclose(parsed["canonical_score"], expected, rel_tol=0.0, abs_tol=1e-12)
    # Canonical (recomputed) != reported display (rounded to 2 decimals)
    assert parsed["score_reported_rounded_differs_from_canonical"] is True
    # Sanity against the actual A1 contest_auth_eval.cpu.json value.
    assert math.isclose(parsed["canonical_score"], 0.19284757743677347, rel_tol=0.0, abs_tol=1e-9)


def test_harvest_parse_report_raises_on_missing_field(tmp_path, harvest_tool):
    report = tmp_path / "report.txt"
    report.write_text("=== Evaluation results over 600 samples ===\n  no metrics\n", encoding="utf-8")
    with pytest.raises(ValueError):
        harvest_tool.parse_report(report)


def test_harvest_is_contest_compliant_runner_gates_lane_tag(harvest_tool):
    assert harvest_tool.is_contest_compliant_runner("ubuntu-latest") is True
    assert harvest_tool.is_contest_compliant_runner("linux-nvidia-t4") is False
    assert harvest_tool.is_contest_compliant_runner("macos-latest") is False
    assert harvest_tool.is_contest_compliant_runner("") is False


def test_harvest_build_cpu_json_matches_a1_reference_schema(harvest_tool):
    """Schema parity check: every key in submissions/a1/contest_auth_eval.cpu.json
    that the harvest tool is expected to produce IS produced by build_cpu_json.

    The harvest's record will not carry release_tag (the wrapper script owns
    the release upload, not the trigger tool); the schema accepts it as None
    when not provided. Per A1 reference, the canonical_score field comes from
    score_recomputed_from_components.
    """
    metadata = {
        "label": "t1_balle_test",
        "submission_name": "t1_balle_test",
        "lane_id": "gha_cpu_eval_t1_balle_test",
        "instance_job_id": "gha_cpu_eval_t1_balle_test_20260512T183000Z",
        "archive_url": "https://github.com/foo/bar/releases/download/tag/archive.zip",
        "archive_sha256": "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5",
        "archive_size_bytes": 178_262,
        "fork_repo": "adpena/comma_video_compression_challenge",
        "runner": "ubuntu-latest",
        "workflow_run_id": 25588422622,
        "workflow_run_url": (
            "https://github.com/adpena/comma_video_compression_challenge/"
            "actions/runs/25588422622"
        ),
        "dispatched_at_utc": "2026-05-12T18:30:00Z",
        "release_tag": "cpu-eval-t1_balle_test-20260512T183000Z",
    }
    parsed = {
        "avg_posenet_dist": 3.286e-05,
        "avg_segnet_dist": 0.00056023,
        "compression_rate": 0.00474789,
        "n_samples": 600,
        "report_text": "stub-report",
        "reported_final_score_display_rounded": 0.19,
        "canonical_score": 0.19284757743677347,
        "canonical_score_recomputed": 0.19284757743677347,
        "score_recomputed_from_components": 0.19284757743677347,
        "canonical_score_source": "score_recomputed_from_components",
        "score_rounding_abs_delta": 0.0028475774367734685,
        "score_reported_rounded_differs_from_canonical": True,
    }
    record = harvest_tool.build_cpu_json(
        metadata=metadata,
        parsed=parsed,
        runner_os="Image: ubuntu-24.04",
        completed_at_utc="2026-05-12T18:50:00Z",
    )
    assert record["evidence_grade"] == "contest-CPU-1to1"
    assert record["lane_tag"] == "[contest-CPU]"
    assert record["hardware"] == "github-actions-ubuntu-latest-x86_64"
    assert record["runner_arch"] == "x86_64"
    assert record["score_claim_valid"] is True
    assert record["device"] == "cpu"
    assert record["n_samples"] == 600
    assert math.isclose(
        record["canonical_score"], 0.19284757743677347, rel_tol=0.0, abs_tol=1e-12
    )
    # Sanity: all reference-schema fields are present.
    expected_keys = {
        "archive_size_bytes",
        "archive_sha256",
        "asset_url",
        "avg_posenet_dist",
        "avg_segnet_dist",
        "canonical_score",
        "canonical_score_recomputed",
        "canonical_score_source",
        "completed_at_utc",
        "compression_rate",
        "device",
        "dispatched_at_utc",
        "evidence_grade",
        "fork_repo",
        "hardware",
        "lane_tag",
        "n_samples",
        "release_tag",
        "report_text",
        "reported_final_score_display_rounded",
        "runner_os_release",
        "score_recomputed_from_components",
        "score_reported_rounded_differs_from_canonical",
        "score_rounding_abs_delta",
        "submission_name",
        "workflow_run_id",
        "workflow_run_url",
    }
    assert expected_keys.issubset(record.keys()), sorted(expected_keys - record.keys())


def test_harvest_build_cpu_json_demotes_advisory_for_nonubuntulatest_runner(harvest_tool):
    """A T4 runner is a CUDA axis — NOT [contest-CPU] even though same workflow."""
    metadata = {
        "label": "t4_run",
        "submission_name": "t4_run",
        "lane_id": "gha_cpu_eval_t4_run",
        "instance_job_id": "ij",
        "archive_url": "https://github.com/foo/bar/releases/download/tag/archive.zip",
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 1,
        "fork_repo": "foo/bar",
        "runner": "linux-nvidia-t4",
        "workflow_run_id": 1,
        "workflow_run_url": "x",
        "dispatched_at_utc": "2026-05-12T18:30:00Z",
    }
    parsed = {
        "avg_posenet_dist": 0.0,
        "avg_segnet_dist": 0.0,
        "compression_rate": 0.0,
        "n_samples": 600,
        "report_text": "",
        "reported_final_score_display_rounded": 0.0,
        "canonical_score": 0.0,
        "canonical_score_recomputed": 0.0,
        "score_recomputed_from_components": 0.0,
        "canonical_score_source": "score_recomputed_from_components",
        "score_rounding_abs_delta": 0.0,
        "score_reported_rounded_differs_from_canonical": False,
    }
    record = harvest_tool.build_cpu_json(
        metadata=metadata,
        parsed=parsed,
        runner_os="Image: ubuntu-24.04",
        completed_at_utc="2026-05-12T18:50:00Z",
    )
    assert record["evidence_grade"] == "advisory"
    assert record["lane_tag"] == "[advisory]"
    assert record["score_claim_valid"] is False
    assert record["hardware"] == "github-actions-linux-nvidia-t4"


def test_harvest_load_dispatch_metadata_validates_schema(tmp_path, harvest_tool):
    good = {
        "schema": "pact.gha_cpu_eval_dispatch_metadata.v1",
        "label": "x",
        "submission_name": "x",
        "lane_id": "x",
        "instance_job_id": "x",
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 1,
        "fork_repo": "foo/bar",
        "runner": "ubuntu-latest",
        "workflow_run_id": 1,
    }
    good_path = tmp_path / "ok.json"
    good_path.write_text(json.dumps(good))
    parsed = harvest_tool.load_dispatch_metadata(good_path)
    assert parsed["workflow_run_id"] == 1

    bad_schema = dict(good, schema="other.v1")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad_schema))
    with pytest.raises(SystemExit):
        harvest_tool.load_dispatch_metadata(bad_path)

    null_run = dict(good, workflow_run_id=None)
    null_path = tmp_path / "null_run.json"
    null_path.write_text(json.dumps(null_run))
    with pytest.raises(SystemExit) as exc:
        harvest_tool.load_dispatch_metadata(null_path)
    assert "workflow_run_id" in str(exc.value)


def test_harvest_poll_run_returns_when_status_completed(harvest_tool, monkeypatch):
    """poll_run loops until status==completed, surfacing in-progress step names."""
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "status": "in_progress",
                        "conclusion": None,
                        "jobs": [
                            {
                                "name": "test",
                                "steps": [{"name": "Evaluate", "status": "in_progress"}],
                            }
                        ],
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "status": "completed",
                        "conclusion": "success",
                        "jobs": [],
                    }
                ),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(harvest_tool, "run_gh", lambda args: next(responses))
    # Inject fake monotonic + sleep so the loop never blocks on wall-clock.
    tick = {"now": 0.0}

    def fake_monotonic():
        tick["now"] += 1
        return tick["now"]

    info = harvest_tool.poll_run(
        12345,
        "foo/bar",
        poll_interval_sec=0,
        poll_timeout_sec=1_000_000,
        monotonic=fake_monotonic,
        sleep=lambda *_: None,
    )
    assert info["status"] == "completed"
    assert info["conclusion"] == "success"


def test_harvest_poll_run_raises_on_timeout(harvest_tool, monkeypatch):
    monkeypatch.setattr(
        harvest_tool,
        "run_gh",
        lambda args: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "in_progress", "conclusion": None, "jobs": []}),
            stderr="",
        ),
    )
    tick = {"now": 0.0}

    def fake_monotonic():
        tick["now"] += 100
        return tick["now"]

    with pytest.raises(TimeoutError):
        harvest_tool.poll_run(
            42,
            "foo/bar",
            poll_interval_sec=0,
            poll_timeout_sec=50,
            monotonic=fake_monotonic,
            sleep=lambda *_: None,
        )
