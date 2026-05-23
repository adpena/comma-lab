# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimization.representation_training_probe_integration import (
    adapt_representation_training_manifest_to_candidate,
    validate_representation_training_manifest,
)
from tac.substrates._shared.trainer_skeleton import (
    build_representation_training_probe_manifest,
    write_representation_training_probe_manifest,
)


def test_build_representation_training_manifest_is_false_authority() -> None:
    manifest = build_representation_training_probe_manifest(
        candidate_id="hnerv_muon_smoke_seed17",
        representation_family="hnerv",
        substrate_family="nerv_family",
        lane_id="offline_hnerv_training_probe",
        candidate_family="hnerv_optimizer_probe",
        seed=17,
        device_selected="mlx",
        stages=[{"index": 1, "module": "stage1"}],
        results=[{"stage_index": 1, "stage_module": "stage1", "best_score": 0.2}],
        archive_zip={
            "path": "experiments/results/hnerv/archive.zip",
            "bytes": 180000,
            "sha256": "a" * 64,
        },
        auth_eval_bridge={
            "ok": True,
            "score_axis": "macOS-CPU advisory",
            "auth_eval_json_sha256": "b" * 64,
            "auth_eval_canonical_score": 0.201,
            "score_comparable": False,
        },
    )

    validate_representation_training_manifest(manifest)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["auth_eval_bridge"]["score_claim"] is False


def test_writer_output_adapts_to_proxy_candidate_queue_row(tmp_path: Path) -> None:
    sidecar = tmp_path / "representation_training_manifest.json"

    manifest = write_representation_training_probe_manifest(
        sidecar,
        candidate_id="siren_optimizer_smoke_seed3",
        representation_family="siren",
        substrate_family="non_nerv_learned",
        candidate_family="siren_optimizer_probe",
        seed=3,
        stages=[{"index": 1, "module": "smoke"}],
        results=[{"stage_index": 1, "stage_module": "smoke", "best_score": 0.19}],
        optimizer_recipe={"id": "adamw_control"},
        scheduler_recipe={"id": "wsd"},
    )

    assert json.loads(sidecar.read_text(encoding="utf-8")) == manifest
    row = adapt_representation_training_manifest_to_candidate(
        manifest,
        source_path=sidecar,
        repo_root=tmp_path,
    )

    assert row["candidate_id"] == "siren_optimizer_smoke_seed3"
    assert row["rank_score"] == 0.19
    assert row["rank_score_field"] == "training_best_score_proxy_not_authority"
    assert "representation_training_archive_export_missing" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []


def test_writer_rejects_extra_fields_that_try_to_reenable_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="score_claim"):
        write_representation_training_probe_manifest(
            tmp_path / "bad.json",
            candidate_id="bad",
            representation_family="hnerv",
            substrate_family="nerv_family",
            extra_fields={"score_claim": True},
        )

