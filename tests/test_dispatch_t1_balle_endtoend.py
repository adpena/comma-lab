from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "dispatch_t1_balle_endtoend.py"
SPEC = importlib.util.spec_from_file_location("dispatch_t1_balle_endtoend", MODULE_PATH)
assert SPEC is not None
t1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = t1
SPEC.loader.exec_module(t1)


def test_vastai_default_gpu_tier_is_4090() -> None:
    assert t1.DEFAULT_GPU_TIER_BY_PROVIDER["vastai"] == "4090"
    assert t1.hourly_rate("vastai", "4090") == pytest.approx(0.42)


def test_unknown_provider_tier_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported provider/gpu-tier"):
        t1.hourly_rate("vastai", "h100")


def test_build_remote_command_enforces_main_before_remote_script(tmp_path: Path) -> None:
    plan = t1.DispatchPlan(
        provider="vastai",
        gpu_tier="4090",
        estimated_hours=24.0,
        cost_cap_usd=80.0,
        estimated_cost_usd=10.08,
        output_dir=tmp_path,
        epochs=3000,
    )
    command = t1.build_remote_command(plan)
    shell = command[-1]

    assert 'test "$(git branch --show-current)" = main' in shell
    assert "git pull --ff-only origin main" in shell
    assert "bash scripts/remote_lane_t1_balle_endtoend.sh" in shell


def test_non_dry_run_refuses_before_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "dry-run only" in result.stdout
    assert not (tmp_path / "vastai_metadata.json").exists()


def test_dry_run_vastai_writes_metadata_default_4090(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metadata = json.loads((tmp_path / "vastai_metadata.json").read_text())
    assert metadata["provider"] == "vastai"
    assert metadata["gpu_tier"] == "4090"
    assert metadata["estimated_cost_usd"] == pytest.approx(10.08)
    assert metadata["instance_id"] == "DRY_RUN"


def test_unsupported_vastai_h100_writes_no_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            ".venv/bin/python",
            str(MODULE_PATH),
            "--provider",
            "vastai",
            "--gpu-tier",
            "h100",
            "--dry-run",
            "--skip-claim",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "unsupported provider/gpu-tier" in result.stdout
    assert not (tmp_path / "vastai_metadata.json").exists()
