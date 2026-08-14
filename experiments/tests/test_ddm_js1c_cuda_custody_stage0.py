from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_js1_stage0_per_edge as stage0
from experiments import ddm_js1c_cuda_custody_stage0 as js1c
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _summary(flips: int) -> dict:
    matrix = np.zeros((5, 5), dtype=np.int64)
    matrix[0, 0] = 1_000
    matrix[0, 1] = flips
    return stage0.matrix_summary(matrix, total_pixels=1_000 + flips)


def _measurement(candidate_record: dict | None = None) -> dict:
    candidate_record = candidate_record or {
        "bytes": 117_964_928,
        "path": "/remote/candidate_argmax_n600.npy",
        "sha256": js1c.PRIOR_FIELD_RECORDS["prior_t1r1_candidate"]["sha256"],
    }
    return {
        "execution_status": "MEASUREMENT_COMPLETE",
        "axis": js1c.AXIS,
        "candidate_archive": js1c.CANDIDATE_ARCHIVE_RECORD,
        "receiver": {"complete": True},
        "scorer": {"complete": True},
        "retained_prior_fields": {
            "gt": js1c.PRIOR_FIELD_RECORDS["gt"],
            "cp135_base": js1c.PRIOR_FIELD_RECORDS["cp135_base"],
        },
        "field_measurement": {
            "denominator_pixels": js1c.TOTAL_PIXELS,
            "base_flips_vs_gt": js1c.BASE_FLIPS,
            "candidate_flips_vs_gt": 55_807,
            "adjudicated_remotely": False,
        },
        "retention": {"candidate_argmax_field": candidate_record},
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _request() -> dict:
    return {
        "inputs": {"candidate_archive.zip": js1c.CANDIDATE_ARCHIVE_RECORD},
        "candidate_archive": js1c.CANDIDATE_ARCHIVE_RECORD,
    }


def test_build_request_is_one_candidate_seg_only_and_resumable() -> None:
    payloads, request = js1c.build_request()

    assert set(payloads) == {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "RE1X_FULL_N600_BLOCKER.json",
    }
    assert request["run_id"] == request["resume_from"] == js1c.RUN_ID
    assert request["candidate_archive"]["sha256"] == (
        js1c.CANDIDATE_ARCHIVE_RECORD["sha256"]
    )
    assert request["candidate_runtime"]["tree_sha256"] == (
        js1c.CANDIDATE_RUNTIME_TREE_SHA256
    )
    assert request["local_pose_delta"] == 0.0
    assert request["pose_unmeasured"] is True
    assert "retain_pose_vectors" not in request
    assert request["resume_required"] is True
    assert request["per_stage_checkpoints"] is True
    assert request["k_arithmetic"]["fits_30_minutes"] is True
    assert request["score_claim"] is False
    assert request["promotion_eligible"] is False
    assert request["pointer_moved"] is False


def test_remote_measurement_contract_requires_retained_same_axis_field() -> None:
    record = js1c.verify_remote_measurement(_measurement(), _request())
    assert record["sha256"] == js1c.PRIOR_FIELD_RECORDS["prior_t1r1_candidate"]["sha256"]

    bad = _measurement()
    bad["field_measurement"]["base_flips_vs_gt"] = js1c.BASE_FLIPS - 1
    with pytest.raises(js1c.JS1CError, match="base control"):
        js1c.verify_remote_measurement(bad, _request())


def test_stage0_result_adjudicates_matched_axis_without_local_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "RE1T_T4_REMOTE_RESULT.json"
    remote.write_text(json.dumps(_measurement()))
    sealed = tmp_path / "JS1C_T4_REQUEST.json"
    sealed.write_text("{}")
    monkeypatch.setattr(js1c, "REMOTE_RESULT", remote)
    monkeypatch.setattr(js1c, "SEALED_REQUEST", sealed)
    field_records = {
        "gt": js1c.PRIOR_FIELD_RECORDS["gt"],
        "cp135_base": js1c.PRIOR_FIELD_RECORDS["cp135_base"],
        "c1_target": js1c.PRIOR_FIELD_RECORDS["c1_target"],
        "candidate": js1c.PRIOR_FIELD_RECORDS["prior_t1r1_candidate"],
    }
    result = js1c.stage0_result(
        summaries={
            "cp135_base": _summary(34_970),
            "candidate": _summary(55_807),
            "c1_target": _summary(27_330),
        },
        field_records=field_records,
        request=_request(),
        measurement=_measurement(),
    )

    assert result["status"] == "NOT_ADMITTED_RHO_GATE"
    assert result["comparison"]["rho_measured"] == pytest.approx(
        -20_837 / 7_640
    )
    assert result["comparison"]["rho_gate_passed"] is False
    assert result["determinism"][
        "fresh_candidate_field_byte_identical_to_prior_js1b_run"
    ] is True
    assert result["score_claim"] is False
    assert result["pointer_moved"] is False


def test_follow_ons_fold_v0_v5_but_queue_1043_982_and_978(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "STAGE0_RESULT.json"
    result_path.write_text("{}")
    monkeypatch.setattr(js1c, "STORE", tmp_path)
    receipts = js1c.follow_on_receipts(
        {"comparison": {"rho_gate_passed": False}}
    )

    assert receipts["v0_v5"]["disposition"] == "FOLDED"
    assert receipts["task_1043"]["trigger_satisfied"] is True
    assert receipts["task_1043"]["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert {row["task_id"] for row in receipts["reroutes"]["routes"]} == {982, 978}
    assert all(
        row["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
        for row in receipts["reroutes"]["routes"]
    )


def test_atomic_copy_preserves_payload_and_rejects_conflict(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "retained/destination.bin"
    source.write_bytes(b"kept payload")

    record = js1c.atomic_copy(source, destination)
    assert destination.read_bytes() == b"kept payload"
    assert record["sha256"] == js1c.sha256_file(source)
    assert js1c.atomic_copy(source, destination) == record

    destination.write_bytes(b"different")
    with pytest.raises(js1c.JS1CError, match="byte count differs"):
        js1c.atomic_copy(source, destination)


def test_dispatcher_consumer_does_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=js1c.REPO,
        roots=[Path("experiments/ddm_js1c_cuda_custody_stage0.py")],
    )
    assert findings == []
