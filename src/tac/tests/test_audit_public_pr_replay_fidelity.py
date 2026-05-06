from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "audit_public_pr_replay_fidelity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_public_pr_replay_fidelity_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def test_replay_fidelity_blocks_public_score_mismatch(tmp_path: Path) -> None:
    intake, eval_dir, archive_meta = _fixture_dirs(tmp_path)
    (intake / "pr_metadata.json").write_text(
        json.dumps(
            {
                "pr_number": 103,
                "leaderboard_name": "hnerv_lc_ac",
                "leaderboard_score": 0.195,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (eval_dir / "adjudication.log").write_text(
        "ADJUDICATION_JSON: "
        + json.dumps(
            {
                "archive_bytes": archive_meta["archive_bytes"],
                "archive_sha256": archive_meta["archive_sha256"],
                "score_recomputed": 0.227764971422447,
            },
            sort_keys=True,
        )
        + "\n"
    )

    report = audit.audit_public_pr_replay_fidelity(intake_dir=intake, eval_dir=eval_dir)

    assert report.ready is False
    assert "public_leaderboard_score_mismatch" in report.blockers
    assert report.summary["classification"] == "public_runtime_or_eval_fidelity_mismatch"
    assert report.summary["score_delta_vs_public"] > 0.03
    assert report.score_claim is False
    assert report.dispatch_attempted is False


def test_replay_fidelity_accepts_matching_public_score(tmp_path: Path) -> None:
    intake, eval_dir, archive_meta = _fixture_dirs(tmp_path)
    (intake / "pr_metadata.json").write_text(
        json.dumps({"leaderboard_score": 0.195}, sort_keys=True) + "\n"
    )
    (eval_dir / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": archive_meta["archive_bytes"],
                "provenance": {"archive_sha256": archive_meta["archive_sha256"]},
                "score_recomputed_from_components": 0.1954,
            },
            sort_keys=True,
        )
        + "\n"
    )

    report = audit.audit_public_pr_replay_fidelity(intake_dir=intake, eval_dir=eval_dir)

    assert report.ready is True
    assert report.blockers == ()
    assert report.summary["classification"] == "public_replay_fidelity_closed"


def test_cli_json_reports_mismatch_without_score_claim(tmp_path: Path) -> None:
    intake, eval_dir, archive_meta = _fixture_dirs(tmp_path)
    (intake / "pr_metadata.json").write_text(
        json.dumps({"leaderboard_score": 0.195}, sort_keys=True) + "\n"
    )
    (eval_dir / "adjudication.log").write_text(
        "ADJUDICATION_JSON: "
        + json.dumps(
            {
                "archive_bytes": archive_meta["archive_bytes"],
                "archive_sha256": archive_meta["archive_sha256"],
                "score_recomputed": 0.227,
            },
            sort_keys=True,
        )
        + "\n"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--intake-dir",
            str(intake),
            "--eval-dir",
            str(eval_dir),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["replay_fidelity_closed"] is False
    assert payload["score_claim"] is False
    assert "public_leaderboard_score_mismatch" in payload["blockers"]


def _fixture_dirs(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    intake = tmp_path / "intake"
    eval_dir = tmp_path / "eval"
    intake.mkdir()
    eval_dir.mkdir()
    archive_meta = _write_archive(intake / "archive.zip", b"payload")
    (eval_dir / "archive.zip").write_bytes((intake / "archive.zip").read_bytes())
    return intake, eval_dir, archive_meta


def _write_archive(path: Path, payload: bytes) -> dict[str, object]:
    import hashlib

    info = ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.external_attr = 0o644 << 16
    with ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    blob = path.read_bytes()
    return {"archive_bytes": len(blob), "archive_sha256": hashlib.sha256(blob).hexdigest()}
