# SPDX-License-Identifier: MIT
"""Tests for tools/dispatch_hf_jobs_vision_training.py + tac.deploy.hf_jobs (Catalog #342)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_dispatcher_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "dispatch_hf_jobs_vision_training.py"
    mod_name = "dispatch_hf_jobs"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dispatcher_module():
    return _load_dispatcher_module()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Flavor cost table — canonical HF Jobs pricing per plugin directive #6
# --------------------------------------------------------------------------


def test_flavor_cost_table_canonical(dispatcher_module):
    table = dispatcher_module.HF_JOBS_FLAVOR_COSTS_USD_PER_HOUR
    assert "t4-small" in table
    assert "a10g-small" in table
    assert "a100-large" in table
    assert "cpu-basic" in table
    assert table["t4-small"] == pytest.approx(0.40)
    assert table["a10g-large"] == pytest.approx(1.50)
    assert table["a100-large"] == pytest.approx(2.50)
    assert table["cpu-basic"] == pytest.approx(0.01)
    # T4-small is the canonical default per plugin directive #6
    assert table["t4-small"] > 0
    # GPU flavors cost more than CPU flavors
    assert table["a100-large"] > table["t4-small"] > table["cpu-basic"]


# --------------------------------------------------------------------------
# plan_dispatch — pure function (no Hub contact)
# --------------------------------------------------------------------------


def test_plan_dispatch_canonical_invocation(dispatcher_module, tmp_path):
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
        model="timm/mobilenetv3_small_100.lamb_in1k",
        flavor="t4-small",
        num_epochs=100,
        timeout_seconds=7200,
        lane_id="lane_test_20260519",
        label="test_smoke",
    )
    assert plan.flavor == "t4-small"
    assert plan.timeout_seconds == 7200
    assert plan.estimated_cost_usd == pytest.approx(0.40 * 2.0)  # $0.40/hr * 2h
    assert plan.hub_dataset_repo == "adpena/test-dataset"
    assert plan.hub_model_repo == "adpena/test-model"
    assert plan.lane_id == "lane_test_20260519"


def test_plan_dispatch_required_flags_present(dispatcher_module, tmp_path):
    """Per plugin directive #4: required CLI flags MUST be in script_args."""
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
    )
    flags = " ".join(plan.script_args)
    assert "--no_remove_unused_columns" in flags
    assert "--push_to_hub" in flags
    assert "--metric_for_best_model eval_accuracy" in flags
    assert "--greater_is_better True" in flags
    assert "--hub_model_id adpena/test-model" in flags
    assert "--dataset_name adpena/test-dataset" in flags


def test_plan_dispatch_pins_dataset_revision_when_sha_declared(
    dispatcher_module,
    tmp_path,
):
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
        dataset_revision="52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a",
    )
    flags = " ".join(plan.script_args)
    assert "--dataset_revision 52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a" in flags


def test_plan_dispatch_rejects_invalid_flavor(dispatcher_module, tmp_path):
    with pytest.raises(ValueError, match="flavor"):
        dispatcher_module.plan_dispatch(
            script=tmp_path / "fake_script.py",
            hub_dataset_repo="adpena/test-dataset",
            hub_model_repo="adpena/test-model",
            flavor="not-a-real-flavor",
        )


def test_plan_dispatch_rejects_invalid_timeout(dispatcher_module, tmp_path):
    with pytest.raises(ValueError, match="timeout_seconds"):
        dispatcher_module.plan_dispatch(
            script=tmp_path / "fake_script.py",
            hub_dataset_repo="adpena/test-dataset",
            hub_model_repo="adpena/test-model",
            timeout_seconds=0,
        )


def test_plan_dispatch_extra_script_args_appended(dispatcher_module, tmp_path):
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
        extra_script_args=["--custom-flag", "value42"],
    )
    assert "--custom-flag" in plan.script_args
    assert "value42" in plan.script_args


def test_plan_dispatch_to_dict_serializable(dispatcher_module, tmp_path):
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
    )
    d = plan.to_dict()
    json.dumps(d, sort_keys=True)  # raises if non-serializable


# --------------------------------------------------------------------------
# CLI smoke
# --------------------------------------------------------------------------


def test_cli_dry_run_smoke(dispatcher_module, tmp_path):
    """--dry-run prints plan JSON without dispatching."""
    rc = dispatcher_module.main([
        "--script", str(tmp_path / "fake_script.py"),
        "--hub-dataset-repo", "adpena/test-ds",
        "--hub-model-repo", "adpena/test-mdl",
        "--flavor", "t4-small",
        "--num-epochs", "10",
        "--timeout-seconds", "3600",
        "--lane-id", "lane_test_20260519",
        "--label", "test_smoke",
        "--dry-run",
    ])
    assert rc == 0


# --------------------------------------------------------------------------
# Ledger smoke (sister tac.deploy.hf_jobs.job_id_ledger)
# --------------------------------------------------------------------------


def test_ledger_register_dispatched_happy_path(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        STATUS_DISPATCHED,
        query_by_hf_jobs_id,
        register_dispatched_hf_jobs_id,
    )

    ledger_path = tmp_path / "test_ledger.jsonl"
    lock_path = tmp_path / "test_ledger.lock"
    row = register_dispatched_hf_jobs_id(
        hf_jobs_id="hf_job_test_001",
        lane_id="lane_test_20260519",
        label="test_dispatch",
        flavor="t4-small",
        expected_cost_usd=2.0,
        expected_axis="cuda",
        path=ledger_path,
        lock_path=lock_path,
    )
    assert row["hf_jobs_id"] == "hf_job_test_001"
    assert row["status"] == STATUS_DISPATCHED
    assert row["flavor"] == "t4-small"
    assert row["expected_cost_usd"] == 2.0

    # Query back
    rows = query_by_hf_jobs_id("hf_job_test_001", path=ledger_path)
    assert len(rows) == 1
    assert rows[0]["hf_jobs_id"] == "hf_job_test_001"


def test_ledger_registers_intent_before_remote_job_id(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        STATUS_INTENT,
        query_by_lane,
        register_hf_jobs_dispatch_intent,
    )

    ledger_path = tmp_path / "test_ledger.jsonl"
    lock_path = tmp_path / "test_ledger.lock"
    row = register_hf_jobs_dispatch_intent(
        lane_id="lane_test_20260519",
        label="test_dispatch",
        flavor="t4-small",
        expected_axis="advisory",
        hub_dataset_sha="52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a",
        path=ledger_path,
        lock_path=lock_path,
    )
    assert row["event_type"] == "intent"
    assert row["status"] == STATUS_INTENT
    assert row["hf_jobs_id"] == "pending:test_dispatch"
    assert row["expected_axis"] == "advisory"
    assert row["hub_dataset_sha"] == "52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a"

    rows = query_by_lane("lane_test_20260519", path=ledger_path)
    assert [r["event_type"] for r in rows] == ["intent"]


def test_dispatch_launch_failure_records_failed_pending_intent(
    dispatcher_module,
    tmp_path,
    monkeypatch,
) -> None:
    plan = dispatcher_module.plan_dispatch(
        script=tmp_path / "fake_script.py",
        hub_dataset_repo="adpena/test-dataset",
        hub_model_repo="adpena/test-model",
        lane_id="lane_test_20260519",
        label="unit_launch_failure",
    )

    class FakeHfApi:
        def __init__(self, token):
            self.token = token

        def run_uv_job(self, **_kwargs):
            raise RuntimeError("402 Payment Required")

    import tac.deploy.hf_jobs.job_id_ledger as ledger

    intent_rows: list[dict] = []
    failed_rows: list[dict] = []

    def fake_register_intent(**kwargs):
        row = {
            "event_type": "intent",
            "hf_jobs_id": f"pending:{kwargs['label']}",
            **kwargs,
        }
        intent_rows.append(row)
        return row

    def fake_update_outcome(**kwargs):
        failed_rows.append(kwargs)
        return kwargs

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi),
    )
    monkeypatch.setattr(ledger, "register_hf_jobs_dispatch_intent", fake_register_intent)
    monkeypatch.setattr(ledger, "update_hf_jobs_outcome", fake_update_outcome)

    with pytest.raises(RuntimeError, match="failed before a job id"):
        dispatcher_module.dispatch(plan, expected_axis="advisory")

    assert intent_rows
    assert failed_rows
    assert failed_rows[0]["hf_jobs_id"] == "pending:unit_launch_failure"
    assert failed_rows[0]["status"] == ledger.EVENT_FAILED
    assert failed_rows[0]["rc"] == 1
    assert failed_rows[0]["cost_actual_usd"] == 0.0
    assert failed_rows[0]["evidence_grade"] == "remote_hf_jobs_launch_failed_before_job_id"
    assert "402 Payment Required" in failed_rows[0]["failure_reason"]


def test_ledger_update_outcome_appends_new_row(tmp_path):
    """Per HISTORICAL_PROVENANCE: outcomes are NEW rows, not mutations."""
    from tac.deploy.hf_jobs.job_id_ledger import (
        EVENT_HARVESTED,
        query_by_hf_jobs_id,
        register_dispatched_hf_jobs_id,
        update_hf_jobs_outcome,
    )

    ledger_path = tmp_path / "test_ledger.jsonl"
    lock_path = tmp_path / "test_ledger.lock"
    register_dispatched_hf_jobs_id(
        hf_jobs_id="hf_job_test_002",
        lane_id="lane_test_20260519",
        label="test_dispatch",
        path=ledger_path,
        lock_path=lock_path,
    )
    update_hf_jobs_outcome(
        hf_jobs_id="hf_job_test_002",
        status=EVENT_HARVESTED,
        rc=0,
        elapsed_seconds=3600.0,
        cost_actual_usd=0.50,
        path=ledger_path,
        lock_path=lock_path,
    )
    rows = query_by_hf_jobs_id("hf_job_test_002", path=ledger_path)
    assert len(rows) == 2
    assert rows[0]["status"] == "dispatched"
    assert rows[1]["status"] == EVENT_HARVESTED
    assert rows[1]["rc"] == 0


def test_ledger_rejects_empty_id(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import register_dispatched_hf_jobs_id

    with pytest.raises(ValueError, match="hf_jobs_id"):
        register_dispatched_hf_jobs_id(
            hf_jobs_id="",
            lane_id="lane_x",
            label="lbl",
            path=tmp_path / "l.jsonl",
            lock_path=tmp_path / "l.lock",
        )


def test_ledger_rejects_invalid_terminal_status(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import update_hf_jobs_outcome

    with pytest.raises(ValueError, match="status"):
        update_hf_jobs_outcome(
            hf_jobs_id="any",
            status="bogus_status",
            path=tmp_path / "l.jsonl",
            lock_path=tmp_path / "l.lock",
        )


def test_ledger_strict_load_raises_on_corrupt(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        HFJobsLedgerCorruptError,
        load_hf_jobs_strict,
    )

    ledger_path = tmp_path / "corrupt.jsonl"
    ledger_path.write_text('{"valid": "row"}\nnot json line\n')
    with pytest.raises(HFJobsLedgerCorruptError):
        load_hf_jobs_strict(path=ledger_path)


def test_ledger_lenient_load_skips_corrupt(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import load_hf_jobs

    ledger_path = tmp_path / "mixed.jsonl"
    ledger_path.write_text(
        '{"valid": "row"}\nnot json line\n{"another": "valid"}\n'
    )
    rows = load_hf_jobs(path=ledger_path)
    assert len(rows) == 2


def test_ledger_query_by_lane(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        query_by_lane,
        register_dispatched_hf_jobs_id,
    )

    ledger_path = tmp_path / "ledger.jsonl"
    lock_path = tmp_path / "ledger.lock"
    for i in range(3):
        register_dispatched_hf_jobs_id(
            hf_jobs_id=f"hf_job_{i}",
            lane_id="lane_alpha",
            label=f"job_{i}",
            path=ledger_path,
            lock_path=lock_path,
        )
    register_dispatched_hf_jobs_id(
        hf_jobs_id="hf_job_other",
        lane_id="lane_beta",
        label="other",
        path=ledger_path,
        lock_path=lock_path,
    )
    alpha_rows = query_by_lane("lane_alpha", path=ledger_path)
    assert len(alpha_rows) == 3
    beta_rows = query_by_lane("lane_beta", path=ledger_path)
    assert len(beta_rows) == 1


def test_ledger_latest_status_returns_terminal(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        EVENT_FAILED,
        latest_status_by_hf_jobs_id,
        register_dispatched_hf_jobs_id,
        update_hf_jobs_outcome,
    )

    ledger_path = tmp_path / "ledger.jsonl"
    lock_path = tmp_path / "ledger.lock"
    register_dispatched_hf_jobs_id(
        hf_jobs_id="hf_job_z",
        lane_id="lane_z",
        label="z",
        path=ledger_path,
        lock_path=lock_path,
    )
    assert latest_status_by_hf_jobs_id("hf_job_z", path=ledger_path) == "dispatched"
    update_hf_jobs_outcome(
        hf_jobs_id="hf_job_z",
        status=EVENT_FAILED,
        rc=1,
        path=ledger_path,
        lock_path=lock_path,
    )
    assert latest_status_by_hf_jobs_id("hf_job_z", path=ledger_path) == EVENT_FAILED


def test_ledger_reserved_field_collision_rejected(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import register_dispatched_hf_jobs_id

    with pytest.raises(ValueError, match="reserved"):
        register_dispatched_hf_jobs_id(
            hf_jobs_id="x",
            lane_id="lane_x",
            label="lbl",
            path=tmp_path / "l.jsonl",
            lock_path=tmp_path / "l.lock",
            status="overridden",  # collides with reserved field
        )


def test_ledger_extra_kwargs_attached(tmp_path):
    from tac.deploy.hf_jobs.job_id_ledger import (
        query_by_hf_jobs_id,
        register_dispatched_hf_jobs_id,
    )

    ledger_path = tmp_path / "l.jsonl"
    lock_path = tmp_path / "l.lock"
    register_dispatched_hf_jobs_id(
        hf_jobs_id="x",
        lane_id="lane_x",
        label="lbl",
        path=ledger_path,
        lock_path=lock_path,
        custom_note="operator review pending",
    )
    rows = query_by_hf_jobs_id("x", path=ledger_path)
    assert rows[0]["custom_note"] == "operator review pending"


def test_canonical_ledger_path_under_omx_state(repo_root):
    from tac.deploy.hf_jobs.job_id_ledger import HF_JOBS_CALL_ID_LEDGER_PATH

    assert repo_root / ".omx" / "state" / "hf_jobs_call_id_ledger.jsonl" == HF_JOBS_CALL_ID_LEDGER_PATH
