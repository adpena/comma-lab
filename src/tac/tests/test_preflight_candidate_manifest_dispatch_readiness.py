from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "preflight_candidate_manifest_dispatch_readiness.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "candidate_manifest_dispatch_readiness_test", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_blocks_builder_specific_local_only_dispatch_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "stack_candidate",
            "score_claim": False,
            "dispatch_gate": "blocked_local_only_until_standalone_exact_positives_and_lane_claim",
            "dispatch_unlocked": False,
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is False
    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert "dispatch_gate_blocked" in codes
    assert "dispatch_unlocked_false" in codes


def test_blocks_runtime_changing_manifest_missing_exact_runtime_contract(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "pr90_stbm1br_lossless_pr85_mask_recode",
            "score_claim": False,
            "fail_closed_preflight": {
                "status": "passed",
                "exact_eval_requires_lane_claim": True,
            },
            "runtime_support": {
                "support_scope": "local_runtime_only",
                "format": "STBM1BR",
            },
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert any(
        blocker["code"]
        == "exact_eval_runtime_contract:missing_for_runtime_changing_candidate"
        for blocker in payload["blockers"]
    )


def test_allows_byte_closed_manifest_but_records_lane_claim_warning(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "byte_closed_candidate",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is True
    assert not payload["blockers"]
    assert any(
        warning["code"] == "lane_claim_still_required"
        for warning in payload["warnings"]
    )


def test_preflight_output_is_deterministic_without_claims_path(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "byte_closed_candidate",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )

    first = module.build_preflight(manifest)
    second = module.build_preflight(manifest)

    assert first == second
    assert first["lane_claim"] == {
        "checked": False,
        "lane_id": None,
        "claims_path": None,
        "active_conflicts": [],
    }


def test_claims_path_requires_explicit_now_utc_for_reproducibility(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "byte_closed_candidate",
            "lane_id": "same_lane",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "# claims\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T08:22:20Z | codex:gpt-5.5 | same_lane | lightning | exact_eval_active | 2026-05-04T09:37:20Z | eval | active |\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="--now-utc is required"):
        module.build_preflight(manifest, claims_path=claims)


def test_blocks_apogee_manifest_without_distortion_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "apogee_int4_repack",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
            "predicted_score_band": "[0.155, 0.180]",
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert any(blocker["code"] == "missing_distortion_model_gate" for blocker in payload["blockers"])


def test_score_affecting_payload_change_requires_source_and_distortion_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "new_rate_codec",
            "score_affecting_payload_changed": True,
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is False
    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert "missing_distortion_model_gate" in codes
    assert "source_archive_sha256_missing_for_score_affecting_change" in codes


def test_score_affecting_payload_change_can_pass_with_source_and_distortion_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "new_rate_codec",
            "score_affecting_payload_changed": True,
            "source_archive_sha256": "a" * 64,
            "distortion_model_status": "passed",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )

    payload = module.build_preflight(manifest)

    assert payload["ready_for_exact_eval_dispatch"] is True
    assert not payload["blockers"]


def test_blocks_pre_dispatch_manifest_with_score_or_dispatch_claims(tmp_path: Path) -> None:
    for field in ("score_claim", "dispatch_performed", "remote_jobs_dispatched"):
        manifest = _write_manifest(
            tmp_path / f"{field}.json",
            {
                "candidate_id": f"bad_{field}",
                "score_claim": False,
                "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
                "dispatch_unlocked": True,
                "ready_for_exact_eval_dispatch_claim": True,
                "fixed_runtime_preflight": {
                    "ready_for_fixed_runtime_exact_eval": True,
                    "remaining_blockers": [],
                },
                field: True,
            },
        )

        payload = module.build_preflight(manifest)

        assert payload["ready_for_exact_eval_dispatch"] is False
        assert any(field in blocker["code"] for blocker in payload["blockers"])


def test_blocks_active_same_lane_claim_when_claims_path_is_supplied(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "byte_closed_candidate",
            "lane_id": "pr85_stbm1br_pr92_rmb1_randmulti",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "# claims\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T08:22:20Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_active | 2026-05-04T09:37:20Z | eval | active |\n",
        encoding="utf-8",
    )

    payload = module.build_preflight(
        manifest,
        claims_path=claims,
        now_utc="2026-05-04T08:26:36Z",
    )

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert any(
        blocker["code"] == "active_lane_claim_conflict"
        for blocker in payload["blockers"]
    )


def test_terminal_newer_claim_closes_older_active_same_job(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "byte_closed_candidate",
            "lane_id": "pr85_stbm1br_pr92_rmb1_randmulti",
            "score_claim": False,
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "dispatch_unlocked": True,
            "ready_for_exact_eval_dispatch_claim": True,
            "fixed_runtime_preflight": {
                "ready_for_fixed_runtime_exact_eval": True,
                "remaining_blockers": [],
            },
        },
    )
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "# claims\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T08:30:00Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_closed | 2026-05-04T08:30:00Z | completed_score_0.2535 | done |\n"
        "| 2026-05-04T08:22:20Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_closed | 2026-05-04T09:37:20Z | eval | old active |\n",
        encoding="utf-8",
    )

    payload = module.build_preflight(
        manifest,
        claims_path=claims,
        now_utc="2026-05-04T08:31:00Z",
    )

    assert payload["ready_for_exact_eval_dispatch"] is True
    assert not payload["blockers"]


def test_cli_fail_if_not_ready_returns_two_and_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        {
            "candidate_id": "planning_candidate",
            "score_claim": False,
            "dispatch_gate": "planning_only/no_remote_dispatch",
        },
    )
    out = tmp_path / "report.json"

    rc = module.main(
        [
            "--manifest",
            str(manifest),
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["candidate_id"] == "planning_candidate"
    assert "planning_candidate" in captured.out
