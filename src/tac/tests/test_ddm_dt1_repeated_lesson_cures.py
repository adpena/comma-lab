"""Incident-shaped tests for the DDM DT1 repeated-lesson cures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.modal import auth_eval
from tac.deploy.modal import single_flight as sf
from tac.deploy.worker_dependency_closure import (
    WorkerDependencyClosureError,
    require_worker_dependency_closure,
    scan_worker_dependency_closure,
)
from tac.preflight import (
    PreflightError,
    check_modal_dispatch_claim_guard_precedes_write,
    check_modal_dual_ledger_matching_is_call_id_first,
    check_worker_target_venv_dependency_closure_is_sealed,
)

REPO = Path(__file__).resolve().parents[3]
_CLAIMS_HEADER = (
    "# Active lane dispatch claims\n\n"
    "| timestamp_utc | agent | lane_id | platform | instance/job_id "
    "| predicted_eta_utc | status | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _write_claims(root: Path, rows: list[tuple[str, ...]]) -> None:
    path = root / ".omx/state/active_lane_dispatch_claims.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _CLAIMS_HEADER + "".join("| " + " | ".join(row) + " |\n" for row in rows),
        encoding="utf-8",
    )


def _claim(*, agent: str, lane: str, job: str, notes: str, status: str = "active_dispatched") -> tuple[str, ...]:
    return (
        "2026-08-14T00:00:00Z",
        agent,
        lane,
        "modal",
        job,
        "-",
        status,
        notes,
    )


def _write_synthetic_dependency_repo(root: Path) -> tuple[Path, Path]:
    worker = root / "workers/worker.py"
    helper = root / "src/localpkg/helper.py"
    worker.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    (helper.parent / "__init__.py").write_text("", encoding="utf-8")
    worker.write_text(
        "import numpy\nfrom localpkg import helper\nimport payload_runtime\n",
        encoding="utf-8",
    )
    helper.write_text(
        "import pydantic\nimport brotli\n\ndef unused_optional_path():\n    import constriction\n",
        encoding="utf-8",
    )
    lock = root / "upstream/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.11"\n\n[[package]]\nname = "numpy"\nversion = "2.3.4"\n',
        encoding="utf-8",
    )
    return worker, lock


def test_worker_closure_catches_transitive_deps_absent_from_target_lock(tmp_path: Path) -> None:
    worker, lock = _write_synthetic_dependency_repo(tmp_path)
    receipt = scan_worker_dependency_closure(
        repo_root=tmp_path,
        worker_entrypoints=(worker,),
        target_lock_path=lock,
        payload_provided_import_roots=("payload_runtime",),
    )
    assert receipt["required_third_party_import_roots"] == ["brotli", "numpy", "pydantic"]
    assert receipt["missing_import_roots"] == ["brotli", "pydantic"]
    assert receipt["payload_imports_observed"] == ["payload_runtime"]
    assert not receipt["passed"]
    with pytest.raises(WorkerDependencyClosureError, match=r"brotli.*pydantic"):
        require_worker_dependency_closure(
            repo_root=tmp_path,
            worker_entrypoints=(worker,),
            target_lock_path=lock,
            payload_provided_import_roots=("payload_runtime",),
        )


def test_worker_closure_accepts_only_when_same_extra_tuple_closes_target(tmp_path: Path) -> None:
    worker, lock = _write_synthetic_dependency_repo(tmp_path)
    receipt = require_worker_dependency_closure(
        repo_root=tmp_path,
        worker_entrypoints=(worker,),
        target_lock_path=lock,
        extra_target_dependencies=("pydantic==2.13.4", "Brotli==1.2.0"),
        payload_provided_import_roots=("payload_runtime",),
    )
    assert receipt["passed"]
    assert receipt["missing_import_roots"] == []


def test_ec2_incident_chain_is_missing_then_closed_by_pinned_image_tuple() -> None:
    kwargs = {
        "repo_root": REPO,
        "worker_entrypoints": (
            REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py",
            REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py",
            REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py",
        ),
        "target_lock_path": REPO / "upstream/uv.lock",
        "payload_provided_import_roots": (
            "ddm_ec1_implicit_edge_conditioning",
            "ddm_ec1_runtime",
            "modules",
            "runtime",
        ),
    }
    missing = scan_worker_dependency_closure(**kwargs)
    assert missing["missing_import_roots"] == ["brotli", "pydantic"]
    closed = require_worker_dependency_closure(
        **kwargs,
        extra_target_dependencies=("pydantic==2.13.4", "Brotli==1.2.0"),
    )
    assert closed["passed"]


def test_same_lane_manual_preclaim_is_not_mistaken_for_dispatcher_own_claim(
    tmp_path: Path,
) -> None:
    _write_claims(
        tmp_path,
        [
            _claim(
                agent="claude-main",
                lane="lane-a",
                job="pending_spawn",
                notes="manual preclaim",
            )
        ],
    )
    findings = sf.single_flight_findings(
        lane_id="lane-a",
        claim_agent="main:dispatcher",
        repo_root=tmp_path,
        check_cloud=False,
    )
    assert len(findings) == 1
    assert "lane-a" in findings[0]
    assert (
        sf.single_flight_findings(
            label="pending_spawn",
            lane_id="lane-a",
            claim_agent="claude-main",
            repo_root=tmp_path,
            check_cloud=False,
        )
        == []
    )
    stale_job = sf.single_flight_findings(
        label="modal:new-job",
        lane_id="lane-a",
        claim_agent="claude-main",
        repo_root=tmp_path,
        check_cloud=False,
    )
    assert len(stale_job) == 1


def test_dispatch_claim_helper_guards_before_it_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        sf,
        "assert_modal_single_flight",
        lambda **kwargs: order.append(("guard", kwargs)),
    )
    monkeypatch.setattr(
        auth_eval,
        "record_dispatch_claim",
        lambda **kwargs: order.append(("write", kwargs)),
    )
    spec = auth_eval.ClaimSpec(
        lane_id="lane-a",
        instance_job_id="modal:job-a",
        agent="main:dispatcher",
    )
    auth_eval.claim_modal_auth_eval_dispatch(
        repo_root=tmp_path,
        spec=spec,
        status="active_spawning",
    )
    assert [name for name, _ in order] == ["guard", "write"]
    assert order[0][1]["claim_agent"] == "main:dispatcher"


def test_terminal_matching_does_not_lane_match_newer_live_call(tmp_path: Path) -> None:
    _write_claims(
        tmp_path,
        [
            _claim(
                agent="main:ec2",
                lane="ddm-ec2",
                job="modal:ec2",
                notes="live call_id=fc-NEW456",
            )
        ],
    )
    assert (
        sf.dual_ledger_terminality_blockers(
            call_id="fc-OLD123",
            label="ec2",
            lane_id="ddm-ec2",
            repo_root=tmp_path,
        )
        == []
    )
    exact = sf.dual_ledger_terminality_blockers(
        call_id="fc-NEW456",
        label="ec2",
        lane_id="ddm-ec2",
        repo_root=tmp_path,
    )
    assert len(exact) == 1


def test_terminal_matching_keeps_legacy_lane_fallback_without_call_id(tmp_path: Path) -> None:
    _write_claims(
        tmp_path,
        [_claim(agent="legacy", lane="lane-a", job="job-a", notes="legacy row")],
    )
    blockers = sf.dual_ledger_terminality_blockers(
        call_id="fc-TERMINAL",
        lane_id="lane-a",
        repo_root=tmp_path,
    )
    assert len(blockers) == 1


def test_dt1_strict_gates_pass_live_repo() -> None:
    assert check_worker_target_venv_dependency_closure_is_sealed(repo_root=REPO) == []
    assert check_modal_dispatch_claim_guard_precedes_write(repo_root=REPO) == []
    assert check_modal_dual_ledger_matching_is_call_id_first(repo_root=REPO) == []


def test_dt1_claim_order_gate_rejects_write_before_guard(tmp_path: Path) -> None:
    path = tmp_path / "src/tac/deploy/modal/auth_eval.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "def claim_modal_auth_eval_dispatch():\n"
        "    record_dispatch_claim()\n"
        "    assert_modal_single_flight(label='job', claim_agent='agent')\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="before claim write"):
        check_modal_dispatch_claim_guard_precedes_write(repo_root=tmp_path, strict=True)


def test_dt1_call_id_gate_rejects_lane_only_matcher(tmp_path: Path) -> None:
    path = tmp_path / "src/tac/deploy/modal/single_flight.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "def dual_ledger_terminality_blockers():\n    tokens = []\n    return tokens\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="call-id-first"):
        check_modal_dual_ledger_matching_is_call_id_first(repo_root=tmp_path, strict=True)


def test_dt1_dependency_gate_rejects_unsealed_dispatcher(tmp_path: Path) -> None:
    dispatcher = tmp_path / "experiments/ddm_ec2_modal_oriented_adapter_trainer.py"
    helper = tmp_path / "src/tac/deploy/worker_dependency_closure.py"
    dispatcher.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    dispatcher.write_text(
        "def prepare():\n    return {}\n\ndef run_trainer():\n    return {}\n",
        encoding="utf-8",
    )
    helper.write_text("# present but not consumed\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="dependency seal"):
        check_worker_target_venv_dependency_closure_is_sealed(
            repo_root=tmp_path,
            strict=True,
        )
