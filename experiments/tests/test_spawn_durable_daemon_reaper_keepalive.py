"""Fleet-reaper guard: assess the stable process image, not a transient shim.

MEASURED 2026-08-22: three consecutive jo1 r9 daemon deaths at ~5 min, no exit
receipt, 95% memory free — the launchd reaper matching ``claude`` in a PATH
that remained visible in safe_run's argv.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.fleet_reaper_guard import (
    FleetReaperLaunchRefusal,
    assert_detached_argv_reaper_safe,
    assess_detached_argv,
)
from tools.spawn_durable_daemon import spawn_detached_verified

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "tools" / "launch_detached_process.py"
SAFE_RUN = REPO / "tools" / "safe_run.py"


def _r9_shape(path_value: str) -> list[str]:
    return [
        sys.executable,
        str(SAFE_RUN),
        "--",
        "/usr/bin/env",
        f"PATH={path_value}",
        sys.executable,
        "-c",
        "pass",
    ]


def test_exact_r9_stable_argv_is_refused() -> None:
    cmd = _r9_shape(f"{REPO}/tools/host_shims:/Users/u/.claude/plugins/bin")
    assessment = assess_detached_argv(cmd)
    assert assessment.refused
    assert assessment.matched_tokens == ("claude",)
    with pytest.raises(FleetReaperLaunchRefusal):
        assert_detached_argv_reaper_safe(cmd)


def test_host_shim_alone_is_not_the_reaper_trigger() -> None:
    assessment = assert_detached_argv_reaper_safe(
        _r9_shape(f"{REPO}/tools/host_shims:/usr/bin:/bin")
    )
    assert not assessment.refused
    assert not assessment.matched_tokens


def test_transient_env_prefix_is_assessed_after_exec() -> None:
    assessment = assert_detached_argv_reaper_safe(
        ["/usr/bin/env", "REAPER_KEEPALIVE=1", "PATH=/Users/u/.claude/bin", "/bin/sleep", "1"]
    )
    assert assessment.stable_argv == ("/bin/sleep", "1")
    assert not assessment.exemption_hits


def test_unparsed_env_options_cannot_turn_a_transient_marker_into_an_exemption() -> None:
    assessment = assess_detached_argv(
        ["/usr/bin/env", "-i", "REAPER_KEEPALIVE=1", "/opt/codex/bin/worker"]
    )
    assert assessment.refused
    assert not assessment.exemption_hits


def test_persistent_exemption_in_safe_run_argv_is_honored() -> None:
    cmd = _r9_shape(f"{REPO}/tools/host_shims:/Users/u/.claude/plugins/bin")
    cmd.insert(3, "REAPER_KEEPALIVE=1")
    assessment = assert_detached_argv_reaper_safe(cmd)
    assert assessment.matched_tokens == ("claude",)
    assert assessment.exemption_hits == ("REAPER_KEEPALIVE",)


@pytest.mark.parametrize(
    "exemption_arg, expected_hit",
    [
        ("/repo/.omx/tmp/codex_runs/job.done", "codex_runs/"),
        ("REAPER_KEEPALIVE=1", "REAPER_KEEPALIVE"),
        ("/Applications/ChatGPT.app/Contents/MacOS/helper", "/Applications/ChatGPT.app/"),
    ],
)
def test_all_exact_source_exemptions_are_honored(
    exemption_arg: str, expected_hit: str
) -> None:
    assessment = assert_detached_argv_reaper_safe(
        ["/opt/codex/bin/worker", "--receipt", exemption_arg]
    )
    assert assessment.matched_tokens == ("codex",)
    assert expected_hit in assessment.exemption_hits


def test_spawn_core_refuses_before_popen(tmp_path: Path) -> None:
    with pytest.raises(FleetReaperLaunchRefusal):
        spawn_detached_verified(
            _r9_shape(f"{REPO}/tools/host_shims:/Users/u/.claude/plugins/bin"),
            tmp_path / "never-created.log",
            verify_s=0,
        )
    assert not (tmp_path / "never-created.log").exists()


def test_launch_detached_positive_and_negative_dry_controls(tmp_path: Path) -> None:
    positive = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--output-dir",
            str(tmp_path / "positive"),
            "--cwd",
            str(REPO),
            "--dry-run",
            "--",
            *_r9_shape(f"{REPO}/tools/host_shims:/Users/u/.claude/plugins/bin"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert positive.returncode == 5
    refusal = json.loads(positive.stderr)
    assert refusal["refused"] is True
    assert refusal["matched_tokens"] == ["claude"]
    assert not (tmp_path / "positive").exists()

    negative = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--output-dir",
            str(tmp_path / "negative"),
            "--cwd",
            str(REPO),
            "--dry-run",
            "--",
            *_r9_shape(f"{REPO}/tools/host_shims:/usr/bin:/bin"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert negative.returncode == 0, negative.stderr
    manifest = json.loads((tmp_path / "negative" / "launch_manifest.json").read_text())
    assert manifest["fleet_reaper_guard"]["refused"] is False
    assert manifest["reaper_predicate_hits"] == []


def test_deprecated_allow_flag_cannot_bypass_the_guard(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--output-dir",
            str(tmp_path / "refused"),
            "--cwd",
            str(REPO),
            "--allow-reaper-name-match",
            "--dry-run",
            "--",
            *_r9_shape(f"{REPO}/tools/host_shims:/Users/u/.claude/plugins/bin"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 5
    refusal = json.loads(result.stderr)
    assert refusal["allow_reaper_name_match_requested"] is True
    assert refusal["refused"] is True
