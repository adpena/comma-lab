"""Tests for ``tools.auto_promote_contest_cuda`` — auto-promotion automation."""

from __future__ import annotations

import importlib.util
import json
import pathlib


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "auto_promote_contest_cuda.py"
    spec = importlib.util.spec_from_file_location("auto_promote_contest_cuda", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_artifact_dir(
    tmp_path: pathlib.Path,
    *,
    lane_dir_name: str,
    score: float,
    evidence_filename: str = "pre_submission_compliance.contest_final.json",
    archive_bytes: int = 185_578,
    archive_sha256: str = "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce",
) -> pathlib.Path:
    """Create a fake repo with one lane_dir containing archive + evidence."""
    repo = tmp_path / "fake_repo"
    lane_dir = repo / "experiments/results" / lane_dir_name
    lane_dir.mkdir(parents=True)
    archive = lane_dir / "archive.zip"
    archive.write_bytes(b"\x00" * archive_bytes)
    evidence_payload = {
        "auth_eval": {"score": score, "archive_sha256": archive_sha256},
        "score": score,
    }
    (lane_dir / evidence_filename).write_text(json.dumps(evidence_payload))
    return repo


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_returns_empty_when_no_results_dir(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    assert mod.discover_contest_cuda_artifacts(tmp_path) == []


def test_discover_finds_one_artifact(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(
        tmp_path,
        lane_dir_name="pr103_repack_pr106_standalone_20260507",
        score=0.20898,
    )
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    assert len(artifacts) == 1
    assert artifacts[0].score == 0.20898
    assert artifacts[0].evidence_grade == "[contest-CUDA]"


def test_discover_prefers_strict_formula_score(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = tmp_path / "fake_repo"
    lane_dir = repo / "experiments/results/pr103_strict"
    lane_dir.mkdir(parents=True)
    (lane_dir / "archive.zip").write_bytes(b"\x00")
    (lane_dir / "pre_submission_compliance.contest_final.json").write_text(
        json.dumps(
            {
                "auth_eval": {
                    "score": 0.20898105277982337,
                    "strict_formula": {"score": 0.2089810755823297},
                }
            }
        )
    )

    artifacts = mod.discover_contest_cuda_artifacts(repo)

    assert len(artifacts) == 1
    assert artifacts[0].score == 0.2089810755823297


def test_discover_skips_lane_dir_without_archive(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = tmp_path / "fake_repo"
    lane_dir = repo / "experiments/results/lane_no_archive"
    lane_dir.mkdir(parents=True)
    (lane_dir / "pre_submission_compliance.contest_final.json").write_text(
        json.dumps({"auth_eval": {"score": 0.18}})
    )
    assert mod.discover_contest_cuda_artifacts(repo) == []


def test_discover_skips_lane_dir_without_evidence(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = tmp_path / "fake_repo"
    lane_dir = repo / "experiments/results/lane_no_evidence"
    lane_dir.mkdir(parents=True)
    (lane_dir / "archive.zip").write_bytes(b"\x00")
    assert mod.discover_contest_cuda_artifacts(repo) == []


def test_discover_handles_corrupt_evidence_json(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = tmp_path / "corrupt"
    lane_dir = repo / "experiments/results/lane_corrupt"
    lane_dir.mkdir(parents=True)
    (lane_dir / "archive.zip").write_bytes(b"\x00")
    (lane_dir / "pre_submission_compliance.contest_final.json").write_text("{bad json")
    # Corrupt evidence → skip silently (no exception)
    assert mod.discover_contest_cuda_artifacts(repo) == []


# ---------------------------------------------------------------------------
# Lane id derivation
# ---------------------------------------------------------------------------

def test_lane_id_from_dir_strips_lane_prefix_and_utc_suffix() -> None:
    mod = _load_tool_module()
    assert mod.lane_id_from_dir("experiments/results/lane_op2_20260507T120000Z") == "op2"


def test_lane_id_from_dir_handles_no_lane_prefix() -> None:
    mod = _load_tool_module()
    # Real anchor directory (no lane_ prefix) — uses basename
    assert mod.lane_id_from_dir(
        "experiments/results/pr103_repack_pr106_standalone_20260507"
    ) == "pr103_repack_pr106_standalone_20260507"


# ---------------------------------------------------------------------------
# Atom ledger append
# ---------------------------------------------------------------------------

def test_atom_ledger_appends_record(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.18)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    assert len(artifacts) == 1
    ledger_path = mod.append_to_atom_ledger(repo, artifacts[0], "test_lane")
    record = json.loads(ledger_path.read_text().strip().splitlines()[-1])
    assert record["lane_id"] == "test_lane"
    assert record["contest_cuda_score"] == 0.18
    assert record["evidence_grade"] == "[contest-CUDA]"


# ---------------------------------------------------------------------------
# Reports auto-update
# ---------------------------------------------------------------------------

def test_reports_update_skips_when_not_a_frontier(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.30)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    reports = repo / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "# latest\n\n"
        "## prior\n\n"
        "**existing best**: score=0.193 [contest-CUDA T4]\n"
    )
    result = mod.update_reports_latest_if_frontier(repo, artifacts[0], "test", dry_run=False)
    assert result["status"] == "not_a_frontier"


def test_reports_update_inserts_frontier_note(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.18)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    reports = repo / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text("# latest\n\n## prior\n\n**existing best**: score=0.193 [contest-CUDA T4]\n")
    result = mod.update_reports_latest_if_frontier(repo, artifacts[0], "phase1_pr100", dry_run=False)
    assert result["status"] == "frontier_updated"
    new = (reports / "latest.md").read_text()
    assert "Auto-promotion" in new
    assert "phase1_pr100" in new
    assert "0.18" in new


def test_reports_update_dry_run_does_not_write(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.18)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    reports = repo / "reports"
    reports.mkdir()
    initial = "# latest\n\n## prior\n\n**existing best**: score=0.193 [contest-CUDA T4]\n"
    (reports / "latest.md").write_text(initial)
    result = mod.update_reports_latest_if_frontier(repo, artifacts[0], "phase1", dry_run=True)
    assert result["status"] == "would_update"
    # File unchanged
    assert (reports / "latest.md").read_text() == initial


# ---------------------------------------------------------------------------
# Gate marking (dry-run only — avoid mutating real lane registry)
# ---------------------------------------------------------------------------

def test_mark_lane_gates_dry_run_emits_three_actions(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.18)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    # Need a fake tools/lane_maturity.py for the dry-run path to exit "BLOCKED" or "ATTEMPTED"
    fake_cli = repo / "tools/lane_maturity.py"
    fake_cli.parent.mkdir(parents=True, exist_ok=True)
    fake_cli.write_text("# fake")
    result = mod.mark_lane_gates(repo, "test_lane", artifacts[0], dry_run=True)
    assert result["status"] == "ATTEMPTED"
    assert len(result["actions"]) == 3
    gates = {a["gate"] for a in result["actions"]}
    assert gates == {"impl_complete", "real_archive_empirical", "contest_cuda"}
    for action in result["actions"]:
        assert action["executed"] is False  # dry-run skips actual execution


def test_mark_lane_gates_blocks_when_cli_missing(tmp_path: pathlib.Path) -> None:
    mod = _load_tool_module()
    repo = _make_artifact_dir(tmp_path, lane_dir_name="lane_test_20260507T000000Z", score=0.18)
    artifacts = mod.discover_contest_cuda_artifacts(repo)
    # Don't create lane_maturity.py
    result = mod.mark_lane_gates(repo, "test_lane", artifacts[0])
    assert result["status"] == "BLOCKED"
