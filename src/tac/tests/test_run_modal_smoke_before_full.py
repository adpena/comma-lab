# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from tac.auth_eval_result import recompute_contest_score_from_payload
from tools.run_modal_smoke_before_full import (
    _expected_auth_artifact_markers,
    _extract_recipe_budget_floor_usd,
    _paid_session_authorization_error,
    _paid_session_authorization_error_for_budget,
    _recipe_requests_smoke_only,
    _resolve_recipe_path,
    _resolve_smoke_band,
    _smoke_validation_contract_from_recipe,
    _spawn_smoke_dispatch,
    _validate_smoke_result,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _archive_zip_artifact(payload: bytes) -> tuple[bytes, str, int]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", payload)
    archive = buf.getvalue()
    return archive, hashlib.sha256(archive).hexdigest(), len(archive)


def _auth_payload(*, score_axis: str = "contest_cuda") -> dict:
    payload = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.0004,
        "archive_size_bytes": 150_000,
        "score_axis": score_axis,
        "lane_tag": "[contest-CUDA]"
        if score_axis == "contest_cuda"
        else "[diagnostic-auth-eval]",
        "evidence_grade": "contest-CUDA" if score_axis == "contest_cuda" else "B",
        "exact_cuda_eval_complete": score_axis == "contest_cuda",
        "score_claim": score_axis == "contest_cuda",
        "score_claim_valid": score_axis == "contest_cuda",
    }
    payload["canonical_score"] = recompute_contest_score_from_payload(payload)
    return payload


def test_resolve_recipe_path_accepts_basename_and_paths(tmp_path: Path) -> None:
    recipes = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes.mkdir(parents=True)
    recipe = recipes / "substrate_x.yaml"
    recipe.write_text("schema_version: 1\n", encoding="utf-8")

    assert _resolve_recipe_path("substrate_x", tmp_path) == recipe
    assert _resolve_recipe_path("substrate_x.yaml", tmp_path) == recipe
    assert (
        _resolve_recipe_path(
            ".omx/operator_authorize_recipes/substrate_x.yaml", tmp_path
        )
        == recipe
    )
    assert _resolve_recipe_path(str(recipe), tmp_path) == recipe


def test_smoke_validation_accepts_canonical_contest_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(_auth_payload()),
            },
        }
    )

    assert green is True
    assert "SMOKE GREEN" in diagnostic


def test_smoke_validation_rejects_diagnostic_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload(score_axis="diagnostic_cuda")
                ),
            },
        }
    )

    assert green is False
    assert "did not contain any finite component-coherent contest-CUDA" in diagnostic


def test_smoke_validation_scans_all_auth_eval_artifacts() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "a_diagnostic_auth_eval_cuda.json": json.dumps(
                    _auth_payload(score_axis="diagnostic_cuda")
                ),
                "z_contest_auth_eval_cuda.json": json.dumps(_auth_payload()),
            },
        }
    )

    assert green is True
    assert "z_contest_auth_eval_cuda.json" in diagnostic


def test_smoke_validation_rejects_stale_auth_eval_artifact_when_marker_required() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
                "submissions/robust_current/auth_eval_renderer_fp4.json": json.dumps(
                    _auth_payload()
                ),
            },
        },
        required_artifact_markers=("lane_substrate_siren_results/output/",),
    )

    assert green is False
    assert "refusing stale evidence" in diagnostic


def test_smoke_validation_accepts_current_output_marker() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
                "lane_substrate_siren_results/output/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
            },
        },
        required_artifact_markers=("lane_substrate_siren_results/output/",),
    )

    assert green is True
    assert "lane_substrate_siren_results/output/contest_auth_eval_cuda.json" in diagnostic


def test_training_artifact_contract_accepts_current_false_authority_manifest() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {
            "training_mode": "smoke",
            "archive_bytes": len(payload),
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_zip_bytes": archive_zip_bytes,
            "archive_zip_sha256": archive_zip_sha,
        },
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        required_artifact_markers=("lane_research_results/output/",),
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is True
    assert "research-only training_artifact_v1" in diagnostic
    assert "archive.zip contains exactly 0.bin" in diagnostic
    assert "score/promotion/rank/readiness claims are false" in diagnostic


def test_training_artifact_contract_accepts_pair_capped_smoke_manifest() -> None:
    """Pair-capped real-video smokes are valid non-score training artifacts."""

    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_pair_capped_smoke",
        "training_mode": "pair_capped_smoke",
        "research_only": True,
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result": {
            "training_mode": "pair_capped_smoke",
            "archive_bytes": len(payload),
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_zip_bytes": archive_zip_bytes,
            "archive_zip_sha256": archive_zip_sha,
        },
    }

    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_pair_capped/output/manifest.json": json.dumps(manifest),
                "lane_pair_capped/output/archive.zip": archive_zip,
            },
        },
        required_artifact_markers=("lane_pair_capped/output/",),
        validation_contract="training_artifact_v1",
        required_lane_id="lane_pair_capped_smoke",
    )

    assert green is True
    assert "training_mode=pair_capped_smoke" in diagnostic
    assert "score/promotion/rank/readiness claims are false" in diagnostic


def test_training_artifact_contract_rejects_stale_manifest_marker() -> None:
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": 12,
        "archive_sha256": "a" * 64,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": b"PK\x03\x04payload",
            },
        },
        required_artifact_markers=("lane_research_results/output/",),
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "refusing stale evidence" in diagnostic


def test_training_artifact_contract_rejects_score_authority_manifest() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": True,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "false_authority_flags_not_false" in diagnostic


def test_training_artifact_contract_rejects_missing_archive_sha() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "archive_sha256_invalid" in diagnostic


def test_training_artifact_contract_rejects_archive_member_mismatch() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(b"wrong")
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "member_0bin" in diagnostic


def test_expected_auth_artifact_markers_parse_workspace_output_dir() -> None:
    recipe = """
env_overrides:
  SIREN_OUTPUT_DIR: /workspace/pact/lane_substrate_siren_results/output
"""

    markers = _expected_auth_artifact_markers(recipe, instance_job_id="job123")

    assert "job123" in markers
    assert "results/job123/" in markers
    assert "lane_substrate_siren_results/output/" in markers


def test_resolve_smoke_band_reads_siren_recipe_prediction_band() -> None:
    lo, hi = _resolve_smoke_band("predicted_band: [0.130, 0.165]\n")

    assert lo == 0.1125
    assert hi == 0.1825


def test_resolve_smoke_band_prefers_explicit_smoke_score_band() -> None:
    lo, hi = _resolve_smoke_band(
        "predicted_band: [0.130, 0.165]\nsmoke_score_band: [0.050, 5.000]\n"
    )

    assert lo == 0.05
    assert hi == 5.0


def test_recipe_requests_smoke_only_reads_top_level_flag() -> None:
    assert _recipe_requests_smoke_only("lane_id: lane_x\nsmoke_only: true\n") is True
    assert _recipe_requests_smoke_only("lane_id: lane_x\nsmoke_only: false\n") is False
    assert _recipe_requests_smoke_only("# smoke_only: true in comment\n") is False


def test_smoke_validation_contract_defaults_to_contest_cuda_auth_eval() -> None:
    assert (
        _smoke_validation_contract_from_recipe("lane_id: lane_x\n")
        == "contest_cuda_auth_eval_v1"
    )
    assert (
        _smoke_validation_contract_from_recipe(
            "smoke_validation_contract: training_artifact_v1\n"
        )
        == "training_artifact_v1"
    )


def test_main_rejects_training_artifact_contract_without_recipe_smoke_only(
    tmp_path: Path,
) -> None:
    recipe_dir = tmp_path / ".omx/operator_authorize_recipes"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "unsafe.yaml").write_text(
        "schema_version: 1\n"
        "name: unsafe\n"
        "lane_id: lane_unsafe\n"
        "smoke_validation_contract: training_artifact_v1\n",
        encoding="utf-8",
    )

    rc = main(["--repo-root", str(tmp_path), "--recipe", "unsafe", "--smoke-only"])

    assert rc == 8


def test_main_rejects_unknown_smoke_validation_contract(tmp_path: Path) -> None:
    recipe_dir = tmp_path / ".omx/operator_authorize_recipes"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "unknown.yaml").write_text(
        "schema_version: 1\n"
        "name: unknown\n"
        "lane_id: lane_unknown_contract\n"
        "smoke_only: true\n"
        "smoke_validation_contract: typo_contract\n",
        encoding="utf-8",
    )

    rc = main(["--repo-root", str(tmp_path), "--recipe", "unknown"])

    assert rc == 7


def test_main_rejects_paid_dispatch_without_session_budget(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    recipe_dir = tmp_path / ".omx/operator_authorize_recipes"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "paid.yaml").write_text(
        "schema_version: 1\n"
        "name: paid\n"
        "lane_id: lane_paid\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE", raising=False)
    monkeypatch.delenv("OPERATOR_AUTHORIZE_SESSION_BUDGET_USD", raising=False)

    rc = main(["--repo-root", str(tmp_path), "--recipe", "paid", "--smoke-only"])

    assert rc == 9
    assert "paid session authorization missing" in capsys.readouterr().err


def test_paid_session_authorization_requires_paired_budget(monkeypatch) -> None:
    monkeypatch.setenv("OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE", "1")
    monkeypatch.delenv("OPERATOR_AUTHORIZE_SESSION_BUDGET_USD", raising=False)
    assert "SESSION_BUDGET" in (_paid_session_authorization_error() or "")

    monkeypatch.setenv("OPERATOR_AUTHORIZE_SESSION_BUDGET_USD", "2.083")
    assert _paid_session_authorization_error() is None


def test_paid_session_authorization_rejects_budget_below_recipe_floor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE", "1")
    monkeypatch.setenv("OPERATOR_AUTHORIZE_SESSION_BUDGET_USD", "2.083")

    msg = _paid_session_authorization_error_for_budget(13.0)

    assert msg is not None
    assert "below the recipe budget floor" in msg


def test_extract_recipe_budget_floor_uses_declared_cost_band() -> None:
    text = """
schema_version: 1
cost_band:
  epochs: 300
  predicted_cost_usd: 13.0
  hand_calibrated_fallback_p50_usd: 10.0
"""
    assert _extract_recipe_budget_floor_usd(text) == 13.0


def test_s2sbs_recipe_is_smoke_only_scaffold_guard() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_s2sbs_byte_stuffing_modal_t4_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert 'S2SBS_SMOKE: "1"' in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_pretrained_driving_prior_recipe_is_smoke_only_scaffold_guard() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert "status: scaffold_only_no_full_dispatch" in text
    assert "contest_exact_eval" not in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_s2sbs_remote_script_does_not_emit_contest_cuda_claim() -> None:
    text = (
        REPO_ROOT / "scripts/remote_lane_substrate_s2sbs_byte_stuffing.sh"
    ).read_text(encoding="utf-8")

    assert "LANE_S2SBS_DONE [smoke-only-no-score-claim]" in text
    assert "TRAINER_SMOKE_ARGS+=(--smoke)" in text
    assert "LANE_S2SBS_DONE [contest-CUDA]" not in text


def test_dpp_remote_script_verifies_active_claim_and_false_authority() -> None:
    text = (
        REPO_ROOT / "scripts/remote_lane_substrate_pretrained_driving_prior.sh"
    ).read_text(encoding="utf-8")

    assert "stage_0b_dispatch_claim_verified" in text
    assert "claim_lane_dispatch.py\" summary" in text
    assert "validate_dispatch_required_inputs.py" in text
    assert "PYBIN not set or not executable after bootstrap" in text
    assert '"$PYBIN" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py"' in text
    assert '"$WORKSPACE/.venv/bin/python" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py"' not in text
    assert '"rank_or_kill_eligible": false' in text


def test_pr95plus_recipe_is_smoke_only_until_full_trainer_lands() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert "PR95PLUS_SMOKE: \"1\"" in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_sane_hnerv_modal_recipe_is_training_artifact_smoke_only() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_sane_hnerv_modal_a100_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"
    assert "AUTH_EVAL_DEVICE=cpu" in text
    assert "canonical exact-eval dispatchers" in text


def test_smoke_wrapper_uses_explicit_noninteractive_authorization_flag() -> None:
    text = (REPO_ROOT / "tools/run_modal_smoke_before_full.py").read_text(encoding="utf-8")

    assert '"tools/operator_authorize.py",' in text
    assert '"--yes",' in text


def test_spawn_smoke_dispatch_threads_cost_band_gpu_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, **kwargs):
        # DX-POLISH-WAVE 2026-05-15 (Catalog #238 / DX-4): the smoke
        # wrapper now also calls `_count_dirty_paths(repo_root)` BEFORE
        # the operator_authorize subprocess. That helper invokes
        # `git status --porcelain` via subprocess.run WITHOUT an `env=`
        # kwarg. Skip recording the porcelain call and only capture the
        # operator_authorize subprocess invocation.
        if list(cmd[:2]) == ["git", "status"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="",
                stderr="",
            )
        calls.append({"cmd": list(cmd), "env": dict(kwargs.get("env") or {})})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="✓ DISPATCHED via .spawn() - call_id=fc-test\ninstance_job_id=job-test\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    call_id, instance_job_id = _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )

    assert (call_id, instance_job_id) == ("fc-test", "job-test")
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--cost-band-epochs-override") + 1] == "100"
    assert cmd[cmd.index("--cost-band-gpu-override") + 1] == "T4"
    env = calls[0]["env"]
    assert env["MODAL_GPU"] == "T4"
    assert env["UNIT_EPOCHS"] == "100"


# ---------------------------------------------------------------------------
# OP-4 fix tests (codex chunk 5, 2026-05-15): paired-env auto-bypass
# anti-pattern. Per CLAUDE.md "Comment-only contracts — FORBIDDEN" + Catalog
# #199/#202 paired-env discipline: the wrapper MUST refuse bare-intent
# without rationale (do NOT auto-fabricate the rationale).
# ---------------------------------------------------------------------------


def _fake_dirty_run_factory(dirty_count: int):
    """Build a fake subprocess.run that pretends git status reports N dirty files.

    Returns ``--porcelain`` lines that ``_count_dirty_paths`` parses. The
    operator_authorize subprocess invocation (cmd[0] != git) is captured for
    inspection by the caller.
    """

    def fake_run(cmd, **kwargs):
        if list(cmd[:2]) == ["git", "status"]:
            stdout = "\n".join(f" M file_{i}.py" for i in range(dirty_count))
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=stdout,
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="✓ DISPATCHED via .spawn() - call_id=fc-test\ninstance_job_id=job-test\n",
            stderr="",
        )

    return fake_run


def test_op4_clean_tree_runs_normally_without_paired_env(
    monkeypatch, tmp_path: Path
) -> None:
    """OP-4: clean tree → no bypass env vars → standard dispatch path."""

    monkeypatch.delenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", raising=False)
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    monkeypatch.setattr(subprocess, "run", _fake_dirty_run_factory(dirty_count=0))

    call_id, _ = _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert call_id == "fc-test"


def test_op4_dirty_tree_bare_intent_without_rationale_refuses(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """OP-4: dirty tree + bare intent (no paired rationale) → SystemExit."""

    monkeypatch.setenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "1")
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    monkeypatch.setattr(subprocess, "run", _fake_dirty_run_factory(dirty_count=3))

    import pytest

    with pytest.raises(SystemExit) as excinfo:
        _spawn_smoke_dispatch(
            tmp_path / "unit_recipe.yaml",
            epoch_env_var="UNIT_EPOCHS",
            smoke_epochs=100,
            smoke_gpu="T4",
            smoke_timeout_hours=1.0,
            operator_handle="codex:test",
            repo_root=REPO_ROOT,
        )

    msg = str(excinfo.value)
    assert "OP-4" in msg
    assert "Catalog #199/#202" in msg or "paired-env" in msg
    assert "WITHOUT" in msg or "refuses" in msg.lower()


def test_op4_dirty_tree_paired_env_both_set_honored(
    monkeypatch, tmp_path: Path
) -> None:
    """OP-4: dirty tree + BOTH env vars set → bypass honored, dispatch proceeds."""

    monkeypatch.setenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "1")
    monkeypatch.setenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED",
        "operator-attests-sentinel-clean-2026-05-15",
    )
    monkeypatch.setattr(subprocess, "run", _fake_dirty_run_factory(dirty_count=5))

    call_id, _ = _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert call_id == "fc-test"


def test_op4_dirty_tree_neither_env_set_runs_with_warning(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """OP-4: dirty tree + neither var set → standard --require-clean-head path."""

    monkeypatch.delenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", raising=False)
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    monkeypatch.setattr(subprocess, "run", _fake_dirty_run_factory(dirty_count=2))

    call_id, _ = _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert call_id == "fc-test"
    captured = capsys.readouterr()
    # The warning text mentions OP-4 + Catalog #202 + the operator-action
    # path to set both vars to take the bypass.
    assert "OP-4" in captured.err
    assert "Catalog #202" in captured.err


def test_op4_does_not_auto_set_rationale_env_var_in_subprocess(
    monkeypatch, tmp_path: Path
) -> None:
    """OP-4 regression guard: rationale env var MUST NOT be auto-set even when
    intent is set on dirty tree."""

    monkeypatch.delenv("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", raising=False)
    monkeypatch.delenv(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED", raising=False
    )
    captured_envs: list[dict] = []

    def fake_run(cmd, **kwargs):
        if list(cmd[:2]) == ["git", "status"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=" M dirty_file.py",
                stderr="",
            )
        captured_envs.append(dict(kwargs.get("env") or {}))
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="✓ DISPATCHED via .spawn() - call_id=fc-test\ninstance_job_id=job-test\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    _spawn_smoke_dispatch(
        tmp_path / "unit_recipe.yaml",
        epoch_env_var="UNIT_EPOCHS",
        smoke_epochs=100,
        smoke_gpu="T4",
        smoke_timeout_hours=1.0,
        operator_handle="codex:test",
        repo_root=REPO_ROOT,
    )
    assert len(captured_envs) == 1
    env = captured_envs[0]
    # CRITICAL OP-4 invariant: neither env var should be auto-fabricated
    # when neither was supplied externally.
    assert env.get("OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK") in (None, "")
    assert env.get("OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED") in (None, "")
