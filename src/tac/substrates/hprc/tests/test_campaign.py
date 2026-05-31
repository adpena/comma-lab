# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import tac.substrates.hprc.archive_candidate as hprc_candidate
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    build_archive_bound_candidate_runtime_package,
)
from tac.substrates.hprc.campaign import (
    HPRC_CAMPAIGN_MANIFEST_SCHEMA,
    HPRC_EXACT_READINESS_REFUSAL_SCHEMA,
    materialize_minimal_hprc_campaign,
)
from tools import package_hprc_minimal_candidate as hprc_tool


def _fake_emit_runtime_package(**kwargs):
    proof = {
        "schema": kwargs["proof_schema"],
        "proof_path": "receiver_proof/hprc_receiver_proof.json",
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return build_archive_bound_candidate_runtime_package(
        adapter_id=kwargs["adapter_id"],
        candidate_family=kwargs["candidate_family"],
        candidate_id_prefix=kwargs["candidate_id_prefix"],
        transform_kind=kwargs["transform_kind"],
        archive_zip_path=kwargs["archive_zip_path"],
        archive_sha256=kwargs["archive_sha256"],
        archive_bytes=kwargs["archive_bytes"],
        submission_dir=kwargs["submission_dir"],
        output_dir=kwargs["output_dir"],
        repo_root=kwargs["repo_root"],
        receiver_proof=proof,
        receiver_contract_kind=kwargs["receiver_contract_kind"],
        runtime_adapter_manifest_extra=kwargs["runtime_adapter_manifest_extra"],
        candidate_row_schema=kwargs["candidate_row_schema"],
        wrapper_schema=kwargs["wrapper_schema"],
        input_artifacts=kwargs["input_artifacts"],
        extra_blockers=kwargs["extra_blockers"],
        mlx_triage_argv=kwargs["mlx_triage_argv"],
    )


def test_hprc_campaign_emits_refusal_and_resolution_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )
    storage_plan_path = tmp_path / "hprc_storage_plan.json"
    storage_plan_path.write_text(json.dumps({"schema": "test_storage_plan.v1"}))

    result = materialize_minimal_hprc_campaign(
        repo_root=repo,
        output_dir=tmp_path / "explicit",
        run_id="unit_hprc_campaign",
        storage_plan_path=storage_plan_path,
    )

    output_dir = Path(result.output_dir)
    assert output_dir == tmp_path / "explicit"
    assert Path(result.archive_zip_path).is_file()
    assert result.score_claim is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.storage_plan_path == storage_plan_path.as_posix()

    refusal = json.loads(Path(result.exact_readiness_refusal_path).read_text())
    assert refusal["schema"] == HPRC_EXACT_READINESS_REFUSAL_SCHEMA
    assert refusal["ready"] is False
    assert "trained_receiver_export_missing" in refusal["blockers"]
    assert "contest_resolution_contract_not_proven_by_full_frame_inflate" in refusal["blockers"]
    assert refusal["promotion_eligible"] is False

    manifest = json.loads(Path(result.campaign_manifest_path).read_text())
    assert manifest["schema"] == HPRC_CAMPAIGN_MANIFEST_SCHEMA
    assert manifest["storage_plan_path"] == storage_plan_path.as_posix()
    assert manifest["phase_status"]["receiver_scaffold"] == "runnable_non_promotable"
    assert manifest["phase_status"]["trained_receiver"] == "missing"
    assert manifest["phase_status"]["resolution_contract"] == "declared_not_proven"
    assert manifest["resolution_contract"]["contest_output"]["width"] == 1164
    assert manifest["resolution_contract"]["contest_output"]["height"] == 874
    assert manifest["resolution_contract"]["contest_output"]["pair_count"] == 600
    assert manifest["resolution_contract"]["scorer_preprocess"]["width"] == 512
    assert manifest["resolution_contract"]["scorer_preprocess"]["height"] == 384
    assert manifest["resolution_contract"]["posenet"]["frames_per_sample"] == 2
    assert manifest["queue_next_actions"][0]["id"] == "hprc_v1_train_export_archive"
    assert manifest["campaign_taxonomy"]["score_claim"] is False


def test_hprc_campaign_explicit_output_dir_skips_storage_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )

    result = materialize_minimal_hprc_campaign(
        repo_root=tmp_path,
        output_dir=tmp_path / "explicit",
        run_id="explicit",
    )

    assert result.storage_plan_path is None
    assert Path(result.archive_bound_package_path).is_file()
    manifest = json.loads(Path(result.campaign_manifest_path).read_text())
    assert manifest["storage_plan_path"] is None


def test_hprc_packaging_cli_uses_storage_waterfall(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    fast = tmp_path / "fast"
    repo.mkdir()
    fast.mkdir()
    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        _fake_emit_runtime_package,
    )

    exit_code = hprc_tool.main(
        [
            "--repo-root",
            repo.as_posix(),
            "--run-id",
            "unit_hprc_cli",
            "--storage-tier",
            f"fast={fast}",
            "--storage-reserve-free-gb",
            "0",
            "--storage-expected-bytes",
            "0",
            "--allow-local-output-dir",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = fast / hprc_tool.DEFAULT_HPRC_WORKLOAD_SUBDIR / "unit_hprc_cli"
    assert Path(payload["output_dir"]) == output_dir
    storage_plan = json.loads((output_dir / "hprc_storage_plan.json").read_text())
    assert storage_plan["schema"] == hprc_tool.HPRC_STORAGE_PLAN_SCHEMA
    assert storage_plan["storage_plan"]["selected_tier"] == "fast"
    assert storage_plan["score_claim"] is False
    manifest = json.loads(Path(payload["campaign_manifest_path"]).read_text())
    assert manifest["storage_plan_path"] == (output_dir / "hprc_storage_plan.json").as_posix()
