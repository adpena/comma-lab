from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "promote_optimizer_candidate_for_exact_eval.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> tuple[int, str]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _write_claims(path: Path, rows: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "# Active lane dispatch claims - fixture\n",
        "\n",
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    body.extend(rows or [])
    path.write_text("".join(body), encoding="utf-8")
    return path


def _active_claim_row(lane_id: str, *, job: str = "fixture_job") -> str:
    return (
        "| 2099-01-01T00:00:00Z | test | "
        f"{lane_id} | modal | {job} |  | active_dispatching | fixture |\n"
    )


def _make_ready_fixture(tmp_path: Path, *, lane_id: str = "fixture_lane") -> dict[str, Path]:
    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("custody report\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
            "score_claim": False,
        },
    )
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "evaluate.py").write_text("# fixture\n", encoding="utf-8")
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "fixture",
                    "lane_id": lane_id,
                    "archive_path": (submission / "archive.zip").relative_to(tmp_path).as_posix(),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "score_affecting_payload_changed": True,
                    "charged_bits_changed": True,
                    "score_claim": False,
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "requires_exact_eval_readiness_gate",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )
    claims = _write_claims(tmp_path / "claims.md")
    return {
        "queue": queue,
        "claims": claims,
        "submission": submission,
    }


def _run_tool(
    fixture: dict[str, Path],
    *,
    output: Path,
    report_output: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(TOOL_PATH),
        "--repo-root",
        str(fixture["submission"].parent),
        "--queue",
        str(fixture["queue"]),
        "--candidate-id",
        "fixture",
        "--output",
        str(output),
        "--dispatch-claims-path",
        str(fixture["claims"]),
        "--active-floor-archive-bytes",
        "999999",
    ]
    if report_output is not None:
        cmd.extend(["--report-output", str(report_output)])
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_refuses_to_write_outputs_inside_submission_runtime_tree(tmp_path: Path) -> None:
    fixture = _make_ready_fixture(tmp_path)
    submission = fixture["submission"]

    proc = _run_tool(
        fixture,
        output=submission / "exact_ready_queue.json",
        report_output=submission / "exact_ready_report.json",
    )

    assert proc.returncode == 2
    assert "output_inside_submission_dir_would_mutate_runtime_tree" in proc.stderr
    assert not (submission / "exact_ready_queue.json").exists()
    assert not (submission / "exact_ready_report.json").exists()


def test_cli_fail_closes_deprecated_skip_active_claim_check(tmp_path: Path) -> None:
    fixture = _make_ready_fixture(tmp_path)
    output = tmp_path / "out" / "exact_ready_queue.json"

    proc = _run_tool(
        fixture,
        output=output,
        extra_args=["--skip-active-claim-check"],
    )

    assert proc.returncode == 2
    assert "--skip-active-claim-check is disabled" in proc.stderr
    assert not output.exists()


def test_cli_fail_closes_missing_dispatch_claim_ledger(tmp_path: Path) -> None:
    fixture = _make_ready_fixture(tmp_path)
    fixture["claims"].unlink()
    output = tmp_path / "out" / "exact_ready_queue.json"

    proc = _run_tool(fixture, output=output)

    assert proc.returncode == 2
    assert "dispatch claim ledger missing or not a file" in proc.stderr
    assert not output.exists()


def test_cli_blocks_active_same_lane_claim_before_exact_ready_output(tmp_path: Path) -> None:
    fixture = _make_ready_fixture(tmp_path, lane_id="fixture_lane")
    _write_claims(fixture["claims"], [_active_claim_row("fixture_lane")])
    output = tmp_path / "out" / "exact_ready_queue.json"

    proc = _run_tool(fixture, output=output)

    assert proc.returncode == 2
    assert "active dispatch claim check blocked exact-eval promotion" in proc.stderr
    assert "same_lane_active_dispatch_claim:fixture_lane:fixture_job:active_dispatching" in proc.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("source_lane_id", "claimed_lane_id"),
    [
        ("fixture_alias", "lane_fixture_alias"),
        ("lane_fixture_alias", "fixture_alias"),
    ],
)
def test_cli_blocks_active_lane_prefix_alias_claims(
    tmp_path: Path,
    source_lane_id: str,
    claimed_lane_id: str,
) -> None:
    fixture = _make_ready_fixture(tmp_path, lane_id=source_lane_id)
    _write_claims(fixture["claims"], [_active_claim_row(claimed_lane_id)])
    output = tmp_path / "out" / "exact_ready_queue.json"

    proc = _run_tool(fixture, output=output)

    assert proc.returncode == 2
    assert f"same_lane_active_dispatch_claim:{claimed_lane_id}:fixture_job" in proc.stderr
    assert not output.exists()


def test_cli_writes_failure_report_outside_submission_tree(tmp_path: Path) -> None:
    fixture = _make_ready_fixture(tmp_path)
    payload = json.loads(fixture["queue"].read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["score_affecting_payload_changed"] = False
    row["charged_bits_changed"] = False
    fixture["queue"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "out" / "exact_ready_queue.json"
    report_output = tmp_path / "out" / "exact_ready_report.json"

    proc = _run_tool(fixture, output=output, report_output=report_output)

    assert proc.returncode == 2
    assert "candidate is not exact-eval dispatch ready" in proc.stderr
    assert "score_affecting_change_proof_missing" in proc.stderr
    assert not output.exists()
    report = json.loads(report_output.read_text(encoding="utf-8"))
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "score_affecting_change_proof_missing" in report["blockers"]
