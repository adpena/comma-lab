# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_phase_a1_score_gradient_pr101.py"
REMOTE_SCRIPT = REPO_ROOT / "scripts" / "remote_track1_phase_a1_score_gradient_pr101.sh"


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
            "--allow-legacy-studio",
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


def test_unknown_gpu_tier_cost_estimate_fails_closed() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="unsupported GPU cost estimate key"):
        tool.estimated_cost_usd("lightning", "H100", 1.0)


def test_unknown_gpu_tier_refuses_before_claim(monkeypatch, tmp_path: Path, capsys) -> None:
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
            "--pr101-archive",
            "CLAUDE.md",
            "--video-path",
            "CLAUDE.md",
            "--pr101-source-dir",
            "src",
            "--provider",
            "lightning",
            "--gpu-tier",
            "H100",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert rc == 3
    assert claims == []
    assert "unsupported GPU cost estimate key 'lightning_h100'" in capsys.readouterr().err


def test_submit_batch_dry_run_manifest_is_not_fired(monkeypatch, tmp_path: Path) -> None:
    """A Lightning Batch dry-run must not look like an in-flight CUDA job."""
    tool = _load_tool()
    claims: list[dict] = []

    class FakeLightningBatchJobsClient:
        def submit(self, spec, *, dry_run: bool):
            spec.validate()
            assert dry_run is True
            return {"status": "DRY_RUN", "name": spec.name}

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    monkeypatch.setattr(tool, "LightningBatchJobsClient", FakeLightningBatchJobsClient)
    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)

    rc = tool.main(
        [
            "--submit-batch",
            "--dry-run-batch",
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
    statuses = [claim.get("status", "active_dispatching") for claim in claims]
    assert statuses == ["active_dispatching", "completed_dry_run"]

    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["session_id"] is None
    assert manifest["dispatch_attempted"] is False
    assert manifest["dispatch_status"] == "dry_run_batch_no_dispatch"
    assert manifest["dispatch_blockers"] == ["dry_run_batch_no_dispatch"]
    assert manifest["evidence_grade"] == "[advisory only — no dispatch]"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["harvest_command_hint"] is None


def test_submit_batch_known_pre_submit_failure_keeps_precise_terminal_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Known pre-submit failures should not add a generic terminal row."""
    tool = _load_tool()
    claims: list[dict] = []

    class FakeLightningBatchJobsClient:
        def submit(self, spec, *, dry_run: bool):
            spec.validate()
            assert dry_run is False
            raise ValueError("could not resolve teamspace")

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    monkeypatch.setattr(tool, "LightningBatchJobsClient", FakeLightningBatchJobsClient)
    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)

    rc = tool.main(
        [
            "--submit-batch",
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

    assert rc == 6
    statuses = [claim.get("status", "active_dispatching") for claim in claims]
    assert statuses == [
        "active_dispatching",
        "failed_batch_submit",
    ]
    assert "failed_dispatch_submission" not in statuses

    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["session_id"] is None
    assert manifest["dispatch_attempted"] is True
    assert manifest["dispatch_status"] == "failed_batch_submit"
    assert manifest["dispatch_blockers"] == ["failed_batch_submit"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


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
    assert "DISPATCH_INSTANCE_JOB_ID=track1_phase_a1_score_gradient_test_id" in spec.command
    assert "tools/claim_lane_dispatch.py claim" in spec.command
    assert "DISPATCH_CLAIMS_PATH" in spec.command
    assert "close_remote_claim()" in spec.command
    assert "failed_batch_worker_rc_${rc}" in spec.command
    assert "REMOTE_SCRIPT_RC=${PIPESTATUS[0]}" in spec.command
    assert "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in spec.command
    assert "track1_phase_a1_batch_summary.json" in spec.command


def test_legacy_studio_dispatch_refuses_without_explicit_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)

    rc = tool.main(
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
    )

    assert rc == 7
    assert claims == []


def test_legacy_studio_command_passes_remote_claim_identity(capsys) -> None:
    tool = _load_tool()

    session_id, fired = tool.dispatch_lightning(
        lane_id="track1_phase_a1_score_gradient",
        instance_job_id="track1_phase_a1_score_gradient_test_id",
        pr101_archive=REPO_ROOT / "CLAUDE.md",
        video_path=REPO_ROOT / "CLAUDE.md",
        pr101_source_dir=REPO_ROOT / "src",
        epochs=1,
        predicted_low=0.15,
        predicted_high=0.22,
        estimated_cost_usd=2.64,
        gpu_tier="T4",
        allow_gpu_mismatch=False,
        print_only=True,
    )

    assert session_id is None
    assert fired is False
    out = capsys.readouterr().out
    assert "--env DISPATCH_INSTANCE_JOB_ID=track1_phase_a1_score_gradient_test_id" in out
    assert "--env DISPATCH_CLAIMS_PATH=.omx/state/active_lane_dispatch_claims.md" in out


def test_submit_batch_dry_run_non_print_writes_no_dispatch_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    class FakeClient:
        def submit(self, spec, *, dry_run=False):
            assert dry_run is True
            return {"status": "DRY_RUN", "spec": spec.asdict()}

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)
    monkeypatch.setattr(tool, "LightningBatchJobsClient", FakeClient)

    rc = tool.main(
        [
            "--submit-batch",
            "--dry-run-batch",
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
    assert [claim.get("status", "active_dispatching") for claim in claims] == [
        "active_dispatching",
        "completed_dry_run",
    ]
    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["session_id"] is None
    assert manifest["dispatch_attempted"] is False
    assert manifest["dispatch_status"] == "dry_run_batch_no_dispatch"
    assert manifest["dispatch_blockers"] == ["dry_run_batch_no_dispatch"]


def test_submit_batch_exception_keeps_claim_nonterminal_for_reconciliation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    class FakeClient:
        def submit(self, spec, *, dry_run=False):
            raise RuntimeError("sdk timeout after submit boundary")

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)
    monkeypatch.setattr(tool, "LightningBatchJobsClient", FakeClient)

    rc = tool.main(
        [
            "--submit-batch",
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

    assert rc == 6
    statuses = [claim.get("status", "active_dispatching") for claim in claims]
    assert statuses == [
        "active_dispatching",
        "submit_status_unknown_reconcile_before_refire",
    ]
    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["dispatch_attempted"] is True
    assert manifest["dispatch_status"] == "submit_status_unknown_reconcile_before_refire"


def test_remote_pr101_script_requires_dispatch_claim_before_cuda_work() -> None:
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "require_active_dispatch_claim" in text
    assert text.index("require_active_dispatch_claim") < text.index("=== Stage 0: bootstrap deps")
    assert "DISPATCH_INSTANCE_JOB_ID is required" in text
    assert "claim_lane_dispatch.py" in text
    assert "close_dispatch_claim()" in text
    assert "completed_contest_cuda_exact_eval" in text
    assert "failed_exact_eval_rc_${EVAL_RC}" in text
    assert "failed_manifest_adjudication_rc_${REPORT_RC}" in text


def test_remote_pr101_pipelines_disable_pipefail_around_rc_capture() -> None:
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert (
        "set +e\n"
        "\"$PYBIN\" \"$WORKSPACE/experiments/train_score_gradient_pr101_finetune.py\""
        in text
    )
    assert "TRAIN_RC=${PIPESTATUS[0]}\nset -e" in text
    assert (
        "set +e\n"
        "\"$PYBIN\" \"$WORKSPACE/tools/build_pr101_finetuned_archive.py\""
        in text
    )
    assert "BUILD_RC=${PIPESTATUS[0]}\nset -e" in text
    assert (
        "set +e\n"
        "\"$PYBIN\" \"$WORKSPACE/experiments/contest_auth_eval.py\""
        in text
    )
    assert "EVAL_RC=${PIPESTATUS[0]}\nset -e" in text


def test_remote_pr101_trainer_receives_pr101_source_dir_for_archive_closure() -> None:
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")
    train_invocation = text[
        text.index('"$PYBIN" "$WORKSPACE/experiments/train_score_gradient_pr101_finetune.py"'):
        text.index('2>&1 | tee "$LOG_DIR/train.log"')
    ]

    assert '--pr101-source-dir "$WORKSPACE/$PR101_SOURCE_DIR"' in train_invocation


def test_remote_pr101_script_uses_auth_eval_semantics_before_contest_cuda_claim() -> None:
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "required_contest_cuda_evidence_blockers" in text
    assert "['contest_cpu_eval_pending'] if score_claim else metric_blockers" in text
    assert "manifest['ready_for_exact_eval_dispatch'] = False" in text
    assert "manifest['exact_cuda_eval_complete'] = score_claim" in text
    assert "manifest['promotion_eligible'] = False" in text
    assert "manifest['rank_or_kill_eligible'] = False" in text
    assert "TRACK1_A1_DONE [contest-CUDA] (rc=$EVAL_RC)" not in text
    assert "TRACK1_A1_DONE [contest-CUDA] (rc=0)" in text
    assert "TRACK1_A1_FAILED exact eval rc=$EVAL_RC; no score claim" in text
    assert "[contest-CUDA failed" not in text
