# SPDX-License-Identifier: MIT
"""Tests for ``tools/score_macos_cpu_advisory_proxy.py`` CLI.

Per operator routing 2026-05-13. The CLI runs ``inflate.sh`` + ``upstream/
evaluate.py --device cpu`` on Darwin ARM64 and emits a non-promotable
advisory-signal manifest. These tests run the CLI in dry-run mode (no real
GPU / archive needed) and validate the manifest contract.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORER_SCRIPT = REPO_ROOT / "tools" / "score_macos_cpu_advisory_proxy.py"


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCORER_SCRIPT), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd or REPO_ROOT),
    )


def _make_fake_archive(path: Path, contents: bytes = b"PK\x05\x06" + b"\x00" * 18) -> Path:
    """Create a minimal valid zipfile at `path`."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", b"fake_renderer_bytes_for_test")
    return path


def test_cli_dry_run_emits_advisory_manifest_with_score_none(tmp_path: Path) -> None:
    archive = _make_fake_archive(tmp_path / "archive.zip")
    result = _run_cli(
        [
            "--archive", str(archive),
            "--family", "test_family",
            "--variant-id", "test_v1",
            "--run-id", "smoke_run_1",
            "--dry-run",
            "--allow-non-darwin",
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    # The CLI emits a structured JSON payload to stdout (last non-banner block).
    # Parse it by finding the first '{' line and reading from there.
    stdout = result.stdout
    json_start = stdout.find("{")
    assert json_start >= 0, f"no JSON in stdout: {stdout!r}"
    # Find a complete JSON block - tail until the closing brace at column 0
    json_text = stdout[json_start:]
    # The script's output is one JSON document. Try to parse it.
    payload = json.loads(json_text)
    assert payload["evidence_grade"] == "macOS-CPU-advisory"
    assert payload["evidence_tag"] == "[macOS-CPU advisory only]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ranking_only"] is True
    rows = payload["rows"]
    assert len(rows) == 1
    assert rows[0]["family"] == "test_family"
    assert rows[0]["variant_id"] == "test_v1"
    # Dry-run mode: score is the sentinel value 0.0 (the row is still tagged
    # non-promotable). Callers detect dry-run via the missing real evaluate
    # payload; the row's promotability flags remain False either way.
    assert rows[0]["score_macos_cpu"] == 0.0


def test_cli_dry_run_manifest_output_file(tmp_path: Path) -> None:
    archive = _make_fake_archive(tmp_path / "archive.zip")
    manifest_out = tmp_path / "manifest.json"
    result = _run_cli(
        [
            "--archive", str(archive),
            "--family", "fam_a",
            "--variant-id", "v_a",
            "--run-id", "smoke_run_2",
            "--dry-run",
            "--allow-non-darwin",
            "--manifest-output", str(manifest_out),
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert manifest_out.is_file()
    manifest = json.loads(manifest_out.read_text())
    assert manifest["schema"].startswith("macos_cpu_advisory_signal_manifest")
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False


def test_cli_dry_run_jsonl_append(tmp_path: Path) -> None:
    archive = _make_fake_archive(tmp_path / "archive.zip")
    jsonl_path = tmp_path / "agg.jsonl"
    # First invocation.
    r1 = _run_cli(
        [
            "--archive", str(archive),
            "--family", "fam_b",
            "--variant-id", "v_b1",
            "--run-id", "smoke_3a",
            "--dry-run",
            "--allow-non-darwin",
            "--jsonl-append", str(jsonl_path),
        ]
    )
    assert r1.returncode == 0
    # Second invocation appends.
    r2 = _run_cli(
        [
            "--archive", str(archive),
            "--family", "fam_b",
            "--variant-id", "v_b2",
            "--run-id", "smoke_3b",
            "--dry-run",
            "--allow-non-darwin",
            "--jsonl-append", str(jsonl_path),
        ]
    )
    assert r2.returncode == 0
    lines = [line for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["ranking_only"] is True


def test_cli_refuses_tmp_manifest_output(tmp_path: Path) -> None:
    archive = _make_fake_archive(tmp_path / "archive.zip")
    result = _run_cli(
        [
            "--archive", str(archive),
            "--family", "fam_c",
            "--variant-id", "v_c",
            "--run-id", "smoke_4",
            "--dry-run",
            "--allow-non-darwin",
            "--manifest-output", "/tmp/forbidden.json",
        ]
    )
    assert result.returncode == 5, f"expected rc=5, got {result.returncode}; stderr: {result.stderr}"


def test_cli_requires_darwin_arm64_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI refuses to run on non-Darwin hosts without --allow-non-darwin.

    We simulate by writing a tiny wrapper that imports the module and patches
    ``platform.system`` to return 'Linux'.
    """
    archive = _make_fake_archive(tmp_path / "archive.zip")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        f'''
import platform, sys
platform.system = lambda: "Linux"
platform.machine = lambda: "x86_64"
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
sys.path.insert(0, {str(REPO_ROOT / "tools")!r})
sys.argv = [
    "score_macos_cpu_advisory_proxy.py",
    "--archive", {str(archive)!r},
    "--family", "f",
    "--variant-id", "v",
    "--run-id", "rid",
    "--dry-run",
]
exec(open({str(SCORER_SCRIPT)!r}).read())
'''
    )
    result = subprocess.run(
        [sys.executable, str(wrapper)],
        capture_output=True,
        text=True,
        check=False,
    )
    # rc=2 is the non-Darwin platform refuse code in the CLI.
    assert result.returncode == 2, (
        f"expected rc=2 on simulated Linux; got rc={result.returncode}, "
        f"stdout={result.stdout[-200:]!r}, stderr={result.stderr[-200:]!r}"
    )


def test_cli_glob_mode_evaluates_multiple_archives(tmp_path: Path) -> None:
    _make_fake_archive(tmp_path / "a_one.zip")
    _make_fake_archive(tmp_path / "a_two.zip")
    _make_fake_archive(tmp_path / "a_three.zip")
    result = _run_cli(
        [
            "--archives", f"{tmp_path}/a_*.zip",
            "--family", "glob_fam",
            "--run-id", "glob_run",
            "--dry-run",
            "--allow-non-darwin",
        ]
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    json_start = result.stdout.find("{")
    payload = json.loads(result.stdout[json_start:])
    assert len(payload["rows"]) == 3
    # The variant_id defaults to the archive filename stem.
    stems = sorted(row["variant_id"] for row in payload["rows"])
    assert stems == ["a_one", "a_three", "a_two"]


def test_cli_glob_with_no_matches_errors_out(tmp_path: Path) -> None:
    result = _run_cli(
        [
            "--archives", f"{tmp_path}/nothing_*.zip",
            "--family", "fam",
            "--run-id", "no_match",
            "--dry-run",
            "--allow-non-darwin",
        ]
    )
    assert result.returncode == 4


def test_cli_help_prints_advisory_disclaimer() -> None:
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "macOS-CPU" in result.stdout or "advisory" in result.stdout.lower()
    assert "samples" in result.stdout.lower()


def test_cli_uses_canonical_safe_zip_extractor() -> None:
    text = SCORER_SCRIPT.read_text(encoding="utf-8")
    assert "safe_extract_zip(" in text
    assert ".extractall(" not in text
