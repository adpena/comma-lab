from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from tac.preflight import check_dispatch_claim_helper_present


REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "tools" / "claim_lane_dispatch.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("claim_lane_dispatch_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_claim_helper_records_newest_first(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_a",
            "--platform",
            "lightning",
            "--instance-job-id",
            "job_1",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T01:00Z",
            "--now-utc",
            "2026-05-02T00:00Z",
            "--notes",
            "first claim",
        ]
    )

    assert rc == 0
    assert "CLAIM_RECORDED" in capsys.readouterr().out
    text = claims.read_text()
    assert "| 2026-05-02T00:00:00Z | codex:test | lane_a | lightning | job_1 |" in text


def test_claim_helper_is_executable_for_shell_runbooks() -> None:
    assert os.access(HELPER, os.X_OK)


def test_dispatch_claim_preflight_covers_helper_and_lightning_launcher() -> None:
    assert check_dispatch_claim_helper_present(REPO, strict=True, verbose=False) == []


def test_claim_helper_blocks_active_lane_conflict(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:00:00Z | claude:test | q_faithful | vast.ai | 123 | 2026-05-02T03:00Z | training | active |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "q_faithful",
            "--platform",
            "vast.ai",
            "--instance-job-id",
            "456",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T00:30Z",
        ]
    )

    assert rc == 2
    assert "REFUSING_DISPATCH" in capsys.readouterr().err
    assert "456" not in claims.read_text()


def test_claim_helper_rejects_empty_required_fields(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"

    with pytest.raises(SystemExit, match="instance_job_id must not be empty"):
        helper.main(
            [
                "claim",
                "--claims-path",
                str(claims),
                "--lane-id",
                "lane_a",
                "--platform",
                "lightning",
                "--instance-job-id",
                "",
                "--agent",
                "codex:test",
                "--predicted-eta-utc",
                "2026-05-02T01:00Z",
            ]
        )

    capsys.readouterr()
    assert not claims.exists()


def test_claim_helper_rejects_table_breaking_required_fields(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"

    with pytest.raises(
        SystemExit, match="lane_id must not contain markdown table separators"
    ):
        helper.main(
            [
                "claim",
                "--claims-path",
                str(claims),
                "--lane-id",
                "lane_a | lightning | job_a",
                "--platform",
                "lightning",
                "--instance-job-id",
                "job_a",
                "--agent",
                "codex:test",
                "--predicted-eta-utc",
                "2026-05-02T01:00Z",
            ]
        )

    capsys.readouterr()
    assert not claims.exists()


def test_claim_helper_allows_audited_parallel_child_claim(tmp_path: Path) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:00:00Z | codex:test | lane_qzs3 | vast.ai | h100_sweep | 2026-05-02T03:00Z | eval | active sweep |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_qzs3",
            "--platform",
            "lightning",
            "--instance-job-id",
            "t4_promotion",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T00:30Z",
            "--allow-parallel",
            "--child-of",
            "h100_sweep",
            "--parallel-reason",
            "promote_completed_child_candidate",
        ]
    )

    text = claims.read_text()
    assert rc == 0
    assert "| 2026-05-02T00:30:00Z | codex:test | lane_qzs3 | lightning | t4_promotion |" in text
    assert "child_of=h100_sweep" in text
    assert "parallel_reason=promote_completed_child_candidate" in text


def test_claim_helper_allows_human_readable_parallel_reason(tmp_path: Path) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:00:00Z | codex:test | lane_qzs3 | lightning | primary | 2026-05-02T03:00Z | eval | active |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_qzs3",
            "--platform",
            "lightning",
            "--instance-job-id",
            "bounded_hedge",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T00:30Z",
            "--allow-parallel",
            "--child-of",
            "primary",
            "--parallel-reason",
            "bounded T4 queue hedge",
        ]
    )

    text = claims.read_text()
    assert rc == 0
    assert "bounded_hedge" in text
    assert "parallel_reason=bounded T4 queue hedge" in text


def test_claim_helper_rejects_parallel_child_without_matching_parent(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:00:00Z | codex:test | lane_qzs3 | vast.ai | h100_sweep | 2026-05-02T03:00Z | eval | active sweep |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_qzs3",
            "--platform",
            "lightning",
            "--instance-job-id",
            "t4_promotion",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T00:30Z",
            "--allow-parallel",
            "--child-of",
            "other_sweep",
            "--parallel-reason",
            "promote_completed_child_candidate",
        ]
    )

    assert rc == 2
    assert "--child-of does not match" in capsys.readouterr().err
    assert "t4_promotion" not in claims.read_text()


def test_claim_helper_allows_terminal_or_stale_claim(tmp_path: Path) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-01T00:00:00Z | claude:test | q_faithful | vast.ai | old | 2026-05-01T03:00Z | training | stale by time |\n"
        + "| 2026-05-02T00:00:00Z | codex:test | q_faithful | lightning | done | 2026-05-02T01:00Z | completed_score=0.3 | done |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "q_faithful",
            "--platform",
            "vast.ai",
            "--instance-job-id",
            "new",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T01:01Z",
            "--ttl-hours",
            "24",
        ]
    )

    assert rc == 0
    assert "| 2026-05-02T01:01:00Z | codex:test | q_faithful | vast.ai | new |" in claims.read_text()


def test_claim_helper_treats_completed_status_family_as_terminal(tmp_path: Path) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:00:00Z | codex:test | lane_qzs3 | vast.ai | sweep | 2026-05-02T01:00Z | completed_b32_only_survivor | completed custom status |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_qzs3",
            "--platform",
            "vast.ai",
            "--instance-job-id",
            "next_sweep",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T04:00Z",
            "--now-utc",
            "2026-05-02T00:30Z",
        ]
    )

    assert rc == 0
    assert "| 2026-05-02T00:30:00Z | codex:test | lane_qzs3 | vast.ai | next_sweep |" in claims.read_text()


def test_claim_helper_treats_stale_refused_and_stopped_families_as_terminal(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    terminal_statuses = [
        "stale_superseded_by_l40s_negative",
        "refused_dispatch_blockers_unchanged_since_prior_refusal",
        "stopped_duplicate_same_decoded_mask",
    ]
    for idx, status in enumerate(terminal_statuses):
        claims = tmp_path / f"claims_{idx}.md"
        claims.write_text(
            helper.HEADER
            + f"| 2026-05-02T00:00:00Z | codex:test | lane_renderer | lightning | old_{idx} | 2026-05-02T01:00Z | {status} | terminal vocabulary drift |\n"
        )

        rc = helper.main(
            [
                "claim",
                "--claims-path",
                str(claims),
                "--lane-id",
                "lane_renderer",
                "--platform",
                "lightning",
                "--instance-job-id",
                f"new_{idx}",
                "--agent",
                "codex:test",
                "--predicted-eta-utc",
                "2026-05-02T04:00Z",
                "--now-utc",
                "2026-05-02T00:30Z",
            ]
        )

        assert rc == 0
        assert f"| 2026-05-02T00:30:00Z | codex:test | lane_renderer | lightning | new_{idx} |" in claims.read_text()


def test_claim_helper_terminal_row_closes_matching_older_active_job(tmp_path: Path) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:40:00Z | codex:test | lane_cmg2 | lightning | job_1 | 2026-05-02T01:00Z | completed_no_frontier | terminal row |\n"
        + "| 2026-05-02T00:00:00Z | codex:test | lane_cmg2 | lightning | job_1 | 2026-05-02T01:00Z | eval | older active row |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_cmg2",
            "--platform",
            "lightning",
            "--instance-job-id",
            "job_2",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T02:00Z",
            "--now-utc",
            "2026-05-02T00:45Z",
        ]
    )

    assert rc == 0
    assert "| 2026-05-02T00:45:00Z | codex:test | lane_cmg2 | lightning | job_2 |" in claims.read_text()


def test_claim_helper_terminal_row_does_not_close_different_active_job(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    claims = tmp_path / "claims.md"
    claims.write_text(
        helper.HEADER
        + "| 2026-05-02T00:40:00Z | codex:test | lane_cmg2 | lightning | job_done | 2026-05-02T01:00Z | completed_no_frontier | terminal row |\n"
        + "| 2026-05-02T00:00:00Z | codex:test | lane_cmg2 | lightning | job_still_active | 2026-05-02T01:00Z | eval | older active row |\n"
    )

    rc = helper.main(
        [
            "claim",
            "--claims-path",
            str(claims),
            "--lane-id",
            "lane_cmg2",
            "--platform",
            "lightning",
            "--instance-job-id",
            "job_2",
            "--agent",
            "codex:test",
            "--predicted-eta-utc",
            "2026-05-02T02:00Z",
            "--now-utc",
            "2026-05-02T00:45Z",
        ]
    )

    assert rc == 2
    assert "job_still_active" in capsys.readouterr().err
    assert "job_2" not in claims.read_text()
