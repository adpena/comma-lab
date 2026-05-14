# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tac.analysis.driving_prior_readiness import (
    LANE_ID,
    SOURCE_COMMIT,
    PenultimateHookTarget,
    build_driving_prior_readiness_manifest,
    check_penultimate_hook_targets,
    validate_readiness_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeHandle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeModule:
    def __init__(self) -> None:
        self.hook_count = 0

    def register_forward_hook(self, _hook):
        self.hook_count += 1
        return _FakeHandle()


class _FakeModel:
    def __init__(self, modules: dict[str, _FakeModule]) -> None:
        self._modules_for_test = modules

    def named_modules(self):
        return [("", self), *self._modules_for_test.items()]


def test_manifest_is_proxy_safe_and_records_2032_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COMMA2K19_ROOT", str(tmp_path))

    manifest = build_driving_prior_readiness_manifest(
        REPO_ROOT,
        probe_scorer=False,
        env=os.environ,
    )

    assert manifest["schema"].startswith("tac_2032_driving_prior_readiness")
    assert manifest["lane_id"] == LANE_ID
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["research_only"] is True
    assert manifest["downloads_attempted"] is False
    assert manifest["network_access_attempted"] is False
    assert manifest["scorer_readiness"]["inflate_time_scorer_load_allowed"] is False

    sources = {
        row["dataset_id"]: row
        for row in manifest["dataset_pretraining_plan"]["sources"]
    }
    assert set(sources) == {"comma2k19", "bdd100k", "waymo_open_dataset"}
    assert sources["comma2k19"]["local_status"] == "local_path_present"
    assert all(row["download_attempted"] is False for row in sources.values())
    assert all(row["network_access_attempted"] is False for row in sources.values())

    budget = manifest["archive_budget_estimate"]
    assert budget["estimate_only_no_archive_materialized"] is True
    assert 40_000 <= budget["total_estimated_bytes_low"] <= 60_000
    assert 70_000 <= budget["total_estimated_bytes_high"] <= 90_000
    assert budget["rate_score_low"] > 0.0
    assert "byte_closed_archive_not_materialized" in manifest["dispatch_blockers"]

    assert validate_readiness_manifest(manifest) == []


def test_validate_rejects_score_authority_leak() -> None:
    manifest = build_driving_prior_readiness_manifest(REPO_ROOT, probe_scorer=False)
    bad = dict(manifest)
    bad["score_claim"] = True
    bad["ready_for_exact_eval_dispatch"] = True

    violations = validate_readiness_manifest(bad)

    assert "score_claim_must_be_false" in violations
    assert "ready_for_exact_eval_dispatch_must_be_false" in violations


def test_penultimate_hook_probe_registers_required_targets() -> None:
    posenet_summary = _FakeModule()
    segnet_encoder = _FakeModule()
    targets = (
        PenultimateHookTarget(
            target_id="pose_summary",
            scorer_id="posenet",
            module_path="summarizer",
            required=True,
            saliency_role="pose_saliency",
            expected_feature="summary",
        ),
        PenultimateHookTarget(
            target_id="seg_encoder",
            scorer_id="segnet",
            module_path="encoder",
            required=True,
            saliency_role="seg_saliency",
            expected_feature="encoder",
        ),
    )

    rows = check_penultimate_hook_targets(
        {
            "posenet": _FakeModel({"summarizer": posenet_summary}),
            "segnet": _FakeModel({"encoder": segnet_encoder}),
        },
        targets=targets,
    )

    assert [row["status"] for row in rows] == ["hook_registerable", "hook_registerable"]
    assert all(row["hook_registerable"] is True for row in rows)
    assert posenet_summary.hook_count == 1
    assert segnet_encoder.hook_count == 1


def test_penultimate_hook_probe_marks_missing_required_target() -> None:
    target = PenultimateHookTarget(
        target_id="pose_summary",
        scorer_id="posenet",
        module_path="summarizer",
        required=True,
        saliency_role="pose_saliency",
        expected_feature="summary",
    )

    rows = check_penultimate_hook_targets(
        {"posenet": _FakeModel({})},
        targets=(target,),
    )

    assert rows[0]["module_present"] is False
    assert rows[0]["hook_registerable"] is False
    assert rows[0]["status"] == "module_missing"


def test_cli_writes_manifest_without_scorer_probe(tmp_path) -> None:
    output = tmp_path / "readiness.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/probe_driving_prior_readiness.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--skip-scorer-probe",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output.read_text())
    stdout_payload = json.loads(proc.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["manifest_validation"]["passed"] is True
    assert stdout_payload["source_commit"] == SOURCE_COMMIT
