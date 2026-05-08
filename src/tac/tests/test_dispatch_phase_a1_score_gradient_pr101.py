from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_phase_a1_score_gradient_pr101.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "dispatch_phase_a1_score_gradient_pr101_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_success_without_session_id_closes_claim_and_manifest_is_terminal(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    def fake_dispatch_lightning(**_kwargs):
        return None, True

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)
    monkeypatch.setattr(tool, "dispatch_lightning", fake_dispatch_lightning)

    assert tool.main(
        [
            "--pr101-archive",
            "CLAUDE.md",
            "--video-path",
            "CLAUDE.md",
            "--pr101-source-dir",
            "src",
            "--provider",
            "lightning",
            "--gpu-tier",
            "T4",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0

    statuses = [claim.get("status", "active_dispatching") for claim in claims]
    assert statuses == ["active_dispatching", "fired_no_session_id_verify_manually"]
    assert claims[1]["force"] is True

    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))

    assert manifest["session_id"] is None
    assert manifest["dispatch_status"] == "fired_no_session_id_verify_manually"
    assert manifest["dispatch_blockers"] == ["fired_no_session_id_verify_manually"]
    assert manifest["evidence_grade"] == "[advisory only — no dispatch]"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "verify platform state manually" in manifest["harvest_command_hint"]

    captured = capsys.readouterr()
    assert "Lane claim closed as 'fired_no_session_id_verify_manually'" in captured.err


def test_submit_batch_print_only_builds_valid_spec(monkeypatch, tmp_path: Path) -> None:
    """--submit-batch --dry-run-batch --print-only should build a spec that
    passes LightningBatchJobSpec.validate() without contacting Lightning."""
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)

    rc = tool.main(
        [
            "--submit-batch",
            "--dry-run-batch",
            "--print-only",
            "--pr101-archive",
            "CLAUDE.md",
            "--video-path",
            "CLAUDE.md",
            "--pr101-source-dir",
            "src",
            "--provider",
            "lightning",
            "--gpu-tier",
            "T4",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    # In --print-only mode no claim_lane should have fired.
    assert claims == []


def test_build_batch_spec_passes_validate(tmp_path: Path) -> None:
    """The constructed spec must pass LightningBatchJobSpec.validate()."""
    tool = _load_tool()
    import argparse

    args = argparse.Namespace(
        lane_id="track1_phase_a1_score_gradient",
        pr101_archive_rel="experiments/results/public_pr_intake_full/x/archive.zip",
        pr101_source_rel="experiments/results/public_pr_intake_full/x/src",
        video_path_rel="upstream/videos/0.mkv",
        epochs=200,
        predicted_low=0.15,
        predicted_high=0.22,
        machine="g4dn.2xlarge",
        studio=None,
        teamspace=None,
        user=None,
        cloud_account=None,
        max_runtime_seconds=4 * 60 * 60,
        remote_pact="/teamspace/studios/this_studio/pact",
        python_bin=".venv/bin/python",
    )
    spec = tool.build_batch_spec(args, "track1_phase_a1_score_gradient_test_id")
    spec.validate()
    assert spec.role == "track1_phase_a1_score_gradient_cuda"
    assert spec.queue_metadata["score_claim"] == "false"
    assert spec.queue_metadata["lane"] == "track1_phase_a1_score_gradient"
    # Heredocs are validated by spec.validate() above.
    assert "remote_track1_phase_a1_score_gradient_pr101.sh" in spec.command
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in spec.command
    assert "track1_phase_a1_batch_summary.json" in spec.command
