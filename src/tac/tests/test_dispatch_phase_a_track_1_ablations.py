from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_phase_a_track_1_ablations.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("dispatch_phase_a_track_1_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lightning_staging_validates_claim_shape_without_opening_active_claim(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    commands: list[list[str]] = []

    def fake_run(cmd, *, cwd, capture_output, text):
        commands.append(list(cmd))
        assert cwd == tool.REPO_ROOT
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout="dry-run ok\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    manifest = tool.dispatch_lightning_t4(tool.PHASE_A_DISPATCHES["A1"], tmp_path)

    assert manifest["status"] == "staged_pending_operator_launch"
    assert manifest["active_lane_claim_opened"] is False
    assert manifest["claim_dry_run_validated"] is True
    claim = commands[0]
    assert claim[:3] == [".venv/bin/python", "tools/claim_lane_dispatch.py", "claim"]
    assert "--dry-run" in claim
    assert "--platform" in claim
    assert claim[claim.index("--platform") + 1] == "lightning"
    assert "--instance-job-id" in claim
    assert claim[claim.index("--instance-job-id") + 1].startswith(
        "track1_phase_a1_score_gradient_"
    )
    assert "--predicted-eta-utc" in claim
    assert (Path(manifest["lane_dir"]) / "heartbeat.log").is_file()


def test_a2_dispatch_invokes_real_tool_locally_without_remote_claim(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    commands: list[list[str]] = []

    def fake_run(cmd, *, cwd, capture_output, text):
        commands.append(list(cmd))
        assert cwd == tool.REPO_ROOT
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout="manifest: a2\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    manifest = tool.dispatch_local_cpu(tool.PHASE_A_DISPATCHES["A2"], tmp_path)

    assert manifest["exit_code"] == 0
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["remote_dispatch_allowed"] is False
    assert manifest["dispatch_attempted"] is False
    command = commands[0]
    assert command[:2] == [".venv/bin/python", "tools/sensitivity_weighted_lossy_coarsening.py"]
    assert "--local-only" in command
    assert "--allow-diagnostic-sensitivity" in command
    assert "--output" in command
    assert "tools/claim_lane_dispatch.py" not in command
    assert (Path(manifest["lane_dir"]) / "build_manifest.json").is_file()
