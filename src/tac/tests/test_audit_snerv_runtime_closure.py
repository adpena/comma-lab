# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools import audit_snerv_runtime_closure as cli


def test_audit_snerv_runtime_closure_profiles_reachability_and_blocks_launch(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "runtime_package" / "submission"
    _write_runtime_file(
        submission / "inflate.py",
        "from tac.foo.runtime import run\n",
    )
    _write_runtime_file(submission / "src/tac/__init__.py", "")
    _write_runtime_file(submission / "src/tac/foo/__init__.py", "")
    _write_runtime_file(
        submission / "src/tac/foo/runtime.py",
        "from tac.foo.used import helper\ndef run():\n    return helper()\n",
    )
    _write_runtime_file(
        submission / "src/tac/foo/used.py",
        "def helper():\n    return 7\n",
    )
    _write_runtime_file(
        submission / "src/tac/foo/unused.py",
        "UNUSED = 'x' * 1024\n",
    )
    archive_zip = tmp_path / "archive.zip"
    _write_archive_zip(
        archive_zip,
        {
            "0.bin": b"packet-payload",
            "inflate.py": (submission / "inflate.py").read_bytes(),
            "src/tac/__init__.py": b"",
            "src/tac/foo/__init__.py": b"",
            "src/tac/foo/runtime.py": (submission / "src/tac/foo/runtime.py").read_bytes(),
            "src/tac/foo/used.py": (submission / "src/tac/foo/used.py").read_bytes(),
            "src/tac/foo/unused.py": (submission / "src/tac/foo/unused.py").read_bytes(),
        },
    )

    report = cli.audit_snerv_runtime_closure(
        archive_zip_path=archive_zip,
        runtime_package_dir=tmp_path / "runtime_package",
        run_import_smoke=True,
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["schema"] == cli.SCHEMA
    assert report["import_smoke"]["passed"] is True
    assert report["archive_zip"]["member_count"] == 7
    contract = report["upstream_contest_bundle_contract"]
    assert contract["upstream_rate_uses_archive_zip_stat_only"] is True
    assert contract["upstream_inflate_sh_runs_from_submission_dir_not_archive_member"] is True
    assert contract["current_archive_contains_runtime_members"] is True
    assert contract["data_only_archive_zip_estimate"]["zip_bytes"] < report[
        "archive_zip"
    ]["bytes"]
    assert report["byte_accounting"]["runtime_member_compressed_bytes"] > 0
    assert report["byte_accounting"]["payload_member_compressed_bytes"] > 0
    minify = report["source_minification_estimates"]
    assert minify["runtime_python_member_count"] == 6
    assert minify["materialized"] is False
    assert minify["identifier_renaming_required_for_no_human_symbols"] is True
    assert "runtime_source_minification_not_materialized" in minify["blockers"]
    assert report["byte_accounting"]["unused_runtime_member_compressed_bytes"] > 0
    unused = report["runtime_reachability"]["unused_runtime_members"]
    assert [row["filename"] for row in unused] == ["src/tac/foo/unused.py"]
    assert "minimal_snerv_runtime_closure_not_materialized" in report["blockers"]
    assert "runtime_source_minification_not_materialized" in report["blockers"]
    assert report["launchability"]["candidate_package_launchable"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    prune = report["materialization_candidates"][0]
    assert prune["id"] == "snerv_runtime_static_unreachable_member_prune"
    assert prune["receiver_replay_required"] is True
    assert prune["score_claim"] is False


def test_audit_snerv_runtime_closure_fails_closed_on_missing_tac_import(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "runtime_package" / "submission"
    _write_runtime_file(
        submission / "inflate.py",
        "from tac.foo.missing import run\n",
    )
    _write_runtime_file(submission / "src/tac/__init__.py", "")
    _write_runtime_file(submission / "src/tac/foo/__init__.py", "")
    archive_zip = tmp_path / "archive.zip"
    _write_archive_zip(
        archive_zip,
        {
            "0.bin": b"packet-payload",
            "inflate.py": (submission / "inflate.py").read_bytes(),
            "src/tac/__init__.py": b"",
            "src/tac/foo/__init__.py": b"",
        },
    )

    report = cli.audit_snerv_runtime_closure(
        archive_zip_path=archive_zip,
        runtime_package_dir=tmp_path / "runtime_package",
        run_import_smoke=False,
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["static_import_graph"]["missing_tac_imports"] == ["tac.foo.missing"]
    assert "snerv_runtime_static_import_closure_missing_members" in report["blockers"]
    assert report["launchability"]["blocked_long_training_rows_must_not_launch"] is True


def _write_runtime_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_archive_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, payload in members.items():
            info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)


def test_audit_snerv_runtime_closure_cli_writes_json(tmp_path: Path) -> None:
    submission = tmp_path / "runtime_package" / "submission"
    _write_runtime_file(submission / "inflate.py", "VALUE = 1\n")
    archive_zip = tmp_path / "archive.zip"
    _write_archive_zip(
        archive_zip,
        {
            "0.bin": b"packet-payload",
            "inflate.py": (submission / "inflate.py").read_bytes(),
        },
    )
    output_json = tmp_path / "audit.json"

    assert (
        cli.main(
            [
                "--archive-zip",
                str(archive_zip),
                "--runtime-package-dir",
                str(tmp_path / "runtime_package"),
                "--output-json",
                str(output_json),
                "--no-import-smoke",
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == cli.SCHEMA
    assert payload["archive_zip"]["sha256"]
