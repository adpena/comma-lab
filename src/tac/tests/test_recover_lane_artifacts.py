"""Tests for tools/recover_lane_artifacts.py.

These tests exercise the artifact-recovery flow without requiring a live
Vast.ai instance. SSH/SCP/CLI subprocess calls are stubbed via
unittest.mock.patch on the ``_run`` helper.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "recover_lane_artifacts.py"


def _load_module():
    """Import the tool by file path to avoid `tools/` package shenanigans."""
    spec = importlib.util.spec_from_file_location(
        "recover_lane_artifacts_mod", str(TOOL_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recover_lane_artifacts_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_module_imports_clean(mod):
    """Module loads without raising."""
    assert hasattr(mod, "recover_artifacts")
    assert hasattr(mod, "recover_before_destroy")
    assert hasattr(mod, "RecoveryReport")
    assert hasattr(mod, "RecoveredArtifact")


def test_sanitize_label_strips_unsafe_chars(mod):
    assert mod._sanitize_label("lane/with spaces!") == "lane_with_spaces"
    assert mod._sanitize_label("") == "unlabeled"
    # Multiple unsafe runs collapse and strip leading/trailing underscores.
    assert mod._sanitize_label("___foo___") == "foo"


def test_recovery_dir_is_per_instance(mod):
    d = mod._recovery_dir_for(12345, "lane_rm_d")
    assert d.name == "recovered_12345_lane_rm_d"
    assert d.parent == REPO_ROOT / "experiments" / "results"


def test_recover_artifacts_handles_unresolvable_instance(mod, tmp_path):
    """No SSH details + missing vastai CLI => returns RecoveryReport with note."""
    # Force VASTAI to a non-existent path so _resolve_ssh_details returns None.
    # Redirect RECOVERY_BASE to tmp_path so the test does NOT pollute the
    # repo's real ``experiments/results/recovered_99999_phantom/`` artifact.
    with patch.object(mod, "VASTAI", tmp_path / "missing_vastai"), \
         patch.object(mod, "RECOVERY_BASE", tmp_path):
        report = mod.recover_artifacts(
            instance_id=99999, lane_label="phantom",
        )
    assert isinstance(report, mod.RecoveryReport)
    assert report.ssh_reachable is False
    assert any("Could not resolve SSH details" in n for n in report.notes)
    # Recovery dir is created even on failure (so we can write metadata).
    assert Path(report.recovery_dir).exists()
    meta = Path(report.recovery_dir) / "recovery_metadata.json"
    assert meta.exists()
    data = json.loads(meta.read_text())
    # v2 append-only schema: ssh_reachable lives inside attempts[-1].
    assert data["schema_version"] == "recovery_metadata.v2_attempts"
    assert isinstance(data["attempts"], list) and len(data["attempts"]) >= 1
    assert data["attempts"][-1]["ssh_reachable"] is False


def test_recover_artifacts_handles_ssh_unreachable(mod, tmp_path):
    """SSH probe failure => recovery returns clean report with note."""
    def fake_run(cmd, timeout):
        # First call: vastai show => return valid host/port
        if cmd[0].endswith("vastai") or "vastai" in str(cmd):
            return 0, json.dumps({"ssh_host": "ghost", "ssh_port": 12345}), ""
        # All ssh probes fail.
        return 255, "", "Connection refused"

    with (
        patch.object(mod, "VASTAI", tmp_path / "vastai"),
        patch.object(mod, "RECOVERY_BASE", tmp_path),
        patch.object(mod, "_run", side_effect=fake_run),
    ):
        # Provide explicit host/port to bypass the vastai-CLI probe.
        report = mod.recover_artifacts(
            instance_id=42, lane_label="dead",
            ssh_host="ghost", ssh_port=12345,
        )
    assert report.ssh_reachable is False
    assert any("SSH reachability probe" in n for n in report.notes)


def test_write_report_appends_new_attempt_with_audit_log(mod, tmp_path):
    rec_dir = tmp_path / "recovered_1_lane"
    first = mod.RecoveryReport(
        instance_id=1,
        lane_label="lane",
        recovery_dir=str(rec_dir),
        started_at_utc="2026-05-08T00:00:00+00:00",
        elapsed_seconds=1.0,
        ssh_reachable=False,
    )
    second = mod.RecoveryReport(
        instance_id=1,
        lane_label="lane",
        recovery_dir=str(rec_dir),
        started_at_utc="2026-05-08T00:01:00+00:00",
        elapsed_seconds=2.0,
        ssh_reachable=False,
    )

    mod._write_report(rec_dir, first)
    mod._write_report(rec_dir, second)

    payload = json.loads((rec_dir / "recovery_metadata.json").read_text())
    assert payload["schema_version"] == "recovery_metadata.v2_attempts"
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["attempt_kind"] == "initial"
    assert payload["attempts"][1]["attempt_kind"] == "force-rerun"
    command_log = payload["attempts"][1]["command_log_path"]
    assert command_log
    assert Path(command_log).is_file()


def test_write_report_replaces_same_in_progress_attempt(mod, tmp_path):
    rec_dir = tmp_path / "recovered_1_lane"
    report = mod.RecoveryReport(
        instance_id=1,
        lane_label="lane",
        recovery_dir=str(rec_dir),
        started_at_utc="2026-05-08T00:00:00+00:00",
        elapsed_seconds=1.0,
        ssh_reachable=False,
    )
    updated = mod.RecoveryReport(
        instance_id=1,
        lane_label="lane",
        recovery_dir=str(rec_dir),
        started_at_utc=report.started_at_utc,
        elapsed_seconds=3.0,
        ssh_reachable=True,
    )

    mod._write_report(rec_dir, report)
    mod._write_report(rec_dir, updated)

    payload = json.loads((rec_dir / "recovery_metadata.json").read_text())
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["attempt_kind"] == "initial"
    assert payload["attempts"][0]["ssh_reachable"] is True
    assert payload["attempts"][0]["elapsed_seconds"] == 3.0


def test_recover_artifacts_finds_and_scps_artifacts(mod, tmp_path):
    """Happy path: SSH ok, find returns 3 files, all SCP successfully."""
    # Redirect recovery base to tmp so we don't litter the repo.
    with patch.object(mod, "RECOVERY_BASE", tmp_path):
        # Build the canonical fake responses sequence.
        ssh_calls = {"count": 0}
        find_output = (
            "1024\t/workspace/renderer.bin\n"
            "2048\t/workspace/masks.mkv\n"
            "512\t/workspace/optimized_poses.pt\n"
        )

        def fake_run(cmd, timeout):
            joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            # Probe: ssh ... echo recover_ok
            if "echo recover_ok" in joined:
                return 0, "recover_ok\n", ""
            # find listing
            if "find " in joined or "-printf" in joined or " find" in joined:
                return 0, find_output, ""
            # scp call
            if cmd[0] == "scp":
                # Create the local file so size detection works.
                target = Path(cmd[-1])
                target.parent.mkdir(parents=True, exist_ok=True)
                # Write the same number of bytes that find reported, so size
                # classification is exercised.
                size_by_remote = {
                    "/workspace/renderer.bin": 1024,
                    "/workspace/masks.mkv": 2048,
                    "/workspace/optimized_poses.pt": 512,
                }
                remote = cmd[-2].split(":", 1)[1]
                target.write_bytes(b"\0" * size_by_remote.get(remote, 1))
                return 0, "", ""
            ssh_calls["count"] += 1
            return 0, "", ""

        with patch.object(mod, "_run", side_effect=fake_run):
            report = mod.recover_artifacts(
                instance_id=111, lane_label="happy",
                ssh_host="hp", ssh_port=22,
            )
    assert report.ssh_reachable is True
    assert len(report.artifacts) == 3
    # Classification should pick the largest archive-like / renderer-like.
    assert report.renderer_bin is not None
    assert report.renderer_bin.endswith("renderer.bin")
    assert report.masks_mkv is not None
    assert report.masks_mkv.endswith("masks.mkv")
    assert report.poses_pt is not None
    assert report.poses_pt.endswith("optimized_poses.pt")
    # Total bytes = 1024 + 2048 + 512 = 3584.
    assert report.total_bytes() == 3584
    # Metadata file is written.
    meta = Path(report.recovery_dir) / "recovery_metadata.json"
    assert meta.exists()


def test_recover_before_destroy_disabled_returns_none(mod):
    """enabled=False short-circuits to None (the --no-recover path)."""
    out = mod.recover_before_destroy(
        instance_id=1, lane_label="x", enabled=False,
    )
    assert out is None


def test_recover_before_destroy_swallows_exceptions(mod, tmp_path):
    """Exceptions during recovery must NOT propagate (destroy must proceed)."""
    def boom(*args, **kwargs):
        raise RuntimeError("simulated SCP storm")

    with (
        patch.object(mod, "RECOVERY_BASE", tmp_path),
        patch.object(mod, "recover_artifacts", side_effect=boom),
    ):
        out = mod.recover_before_destroy(
            instance_id=1, lane_label="x", enabled=True,
        )
    # Best-effort: returns None on exception, doesn't raise.
    assert out is None


def test_classify_picks_largest_matching_files(mod, tmp_path):
    """_classify favours largest archive*.zip and renderer.bin."""
    a = tmp_path / "archive_small.zip"
    a.write_bytes(b"\0" * 100)
    b = tmp_path / "archive_big.zip"
    b.write_bytes(b"\0" * 1000)
    r = tmp_path / "renderer.bin"
    r.write_bytes(b"\0" * 50)
    m = tmp_path / "masks.mkv"
    m.write_bytes(b"\0")
    cls = mod._classify([a, b, r, m])
    assert cls["archive_zip"] == str(b)
    assert cls["renderer_bin"] == str(r)
    assert cls["masks_mkv"] == str(m)
    assert cls["poses_pt"] is None


def test_classify_handles_empty_input(mod):
    cls = mod._classify([])
    assert cls == {
        "archive_zip": None,
        "renderer_bin": None,
        "masks_mkv": None,
        "poses_pt": None,
    }


def test_summary_renders_human_friendly_lines(mod, tmp_path):
    rec = mod.RecoveryReport(
        instance_id=42,
        lane_label="x",
        recovery_dir=str(tmp_path),
        started_at_utc="2026-04-28T00:00:00+00:00",
        elapsed_seconds=12.3,
        ssh_reachable=True,
        archive_zip="/path/archive.zip",
        renderer_bin="/path/renderer.bin",
    )
    out = rec.summary()
    assert "instance=42" in out
    assert "label=x" in out
    assert "archive_zip:  /path/archive.zip" in out
    assert "renderer_bin: /path/renderer.bin" in out


def test_overall_timeout_aborts_remaining_scps(mod, tmp_path):
    """When deadline is exceeded mid-loop, remaining files are skipped."""
    # 5 candidates; deadline=0 means we always hit the timeout check.
    find_output = "\n".join(f"100\t/workspace/file{i}.pt" for i in range(5))

    def fake_run(cmd, timeout):
        joined = " ".join(cmd)
        if "echo recover_ok" in joined:
            return 0, "recover_ok\n", ""
        if "find " in joined or "-printf" in joined:
            return 0, find_output + "\n", ""
        if cmd[0] == "scp":
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_bytes(b"\0" * 100)
            return 0, "", ""
        return 0, "", ""

    with (
        patch.object(mod, "RECOVERY_BASE", tmp_path),
        patch.object(mod, "_run", side_effect=fake_run),
    ):
        report = mod.recover_artifacts(
            instance_id=222, lane_label="timeout",
            ssh_host="h", ssh_port=22,
            overall_timeout_s=0,  # immediate timeout
        )
    # We may get 0 or 1 file before the deadline check fires; either way,
    # NOT all 5 files, and the timeout note must be present.
    assert len(report.artifacts) < 5
    assert any("Overall timeout" in n for n in report.notes)


def test_recovery_patterns_cover_all_archive_artifacts(mod):
    """Sanity: the canonical archive artifacts (renderer/masks/poses/zip) are
    all in RECOVERY_PATTERNS."""
    patterns = set(mod.RECOVERY_PATTERNS)
    # Match by exact basename.
    assert "renderer.bin" in patterns
    assert "masks.mkv" in patterns
    assert "optimized_poses.pt" in patterns
    assert "archive.zip" in patterns
    # Globs for variant naming.
    assert "archive_*.zip" in patterns
    # Logs are also pulled.
    assert "run.log" in patterns
    assert "heartbeat.log" in patterns
    assert "provenance.json" in patterns
