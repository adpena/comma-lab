from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_re1t_modal_t4_sign_gate as dispatch
from experiments import ddm_re1t_t4_sign_gate_worker as worker
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _runtime_with_pin(root: Path, *, digest: str, size: int) -> Path:
    root.mkdir(parents=True)
    (root / "inflate.py").write_text(
        "\n".join(
            [
                f'ARCHIVE_SHA256 = "{digest}"',
                f"ARCHIVE_BYTES = {size}",
                "def _verify_input(data_dir, archive_path):",
                "    if archive_path.sha != ARCHIVE_SHA256: raise ValueError('sha')",
                "    if archive_path.size != ARCHIVE_BYTES: raise ValueError('size')",
                "def main(data_dir, archive_path):",
                "    _verify_input(data_dir, archive_path)",
            ]
        )
        + "\n"
    )
    return root


def _request() -> dict[str, object]:
    return {
        "inputs": {
            "candidate_archive.zip": {
                "bytes": dispatch.CANDIDATE_BYTES,
                "sha256": dispatch.CANDIDATE_SHA256,
            }
        },
        "candidate_archive": {
            "bytes": dispatch.CANDIDATE_BYTES,
            "sha256": dispatch.CANDIDATE_SHA256,
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "pose_gate_note": "pose remains measured-follow-up debt",
    }


def _measurement(*, candidate_flips: int, identical: bool) -> dict[str, object]:
    changed_pixels = 0 if identical else 1
    return {
        "execution_status": "MEASUREMENT_COMPLETE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "remote_adjudication_performed": False,
        "axis": dispatch.AXIS,
        "candidate_archive": {
            "bytes": dispatch.CANDIDATE_BYTES,
            "sha256": dispatch.CANDIDATE_SHA256,
        },
        "retained_prior_fields": {
            "gt": dispatch.GT_FIELD_RECORD,
            "cp135_base": dispatch.BASE_FIELD_RECORD,
        },
        "field_measurement": {
            "denominator_pixels": dispatch.DENOMINATOR,
            "base_flips_vs_gt": dispatch.BASE_FLIPS,
            "candidate_flips_vs_gt": candidate_flips,
            "candidate_minus_base_flips": candidate_flips - dispatch.BASE_FLIPS,
            "candidate_changed_pixels_vs_cp135": changed_pixels,
            "candidate_field_identical_to_cp135": identical,
            "adjudicated_remotely": False,
        },
    }


def test_runtime_pin_requires_candidate_sha_bytes_and_live_guard(tmp_path: Path) -> None:
    record = {"sha256": "a" * 64, "bytes": 123}
    runtime = _runtime_with_pin(tmp_path / "runtime", digest=record["sha256"], size=record["bytes"])
    result = dispatch.verify_runtime_archive_pin(runtime, record)
    assert result["passed"] is True
    assert result["verify_input_calls"] == 1

    (runtime / "inflate.py").write_text(
        (runtime / "inflate.py").read_text().replace("a" * 64, "b" * 64)
    )
    with pytest.raises(dispatch.RE1TDispatchError, match="inflate pin differs"):
        dispatch.verify_runtime_archive_pin(runtime, record)


def test_blocker_rehash_deduplicates_records_and_fails_on_payload_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"kept")
    record = dispatch.js1b_dispatch.file_record(payload)
    receipt = tmp_path / "blocker.json"
    receipt.write_text(
        json.dumps(
            {
                "verdict": "BLOCKED_HASH_PINNED_PUBLIC_FRONT_DOOR_REQUIRES_CUDA",
                "candidate_archive": {
                    "bytes": dispatch.CANDIDATE_BYTES,
                    "sha256": dispatch.CANDIDATE_SHA256,
                },
                "candidate_runtime": {
                    "tree_sha256": dispatch.CANDIDATE_RUNTIME_TREE_SHA256,
                },
                "failure": {"runtime_error": "F26 inflation requires a CUDA-capable GPU"},
                "records": [record, record],
            },
            sort_keys=True,
        )
    )
    monkeypatch.setattr(dispatch, "BLOCKER_BYTES", receipt.stat().st_size)
    monkeypatch.setattr(dispatch, "BLOCKER_SHA256", dispatch.js1b_dispatch.sha256_file(receipt))
    monkeypatch.setattr(dispatch, "BLOCKER_DISTINCT_RECORDS", 1)
    result = dispatch.verify_blocker_receipt(receipt)
    assert result["distinct_records_rehashed"] == 1
    payload.write_bytes(b"lost")
    with pytest.raises(Exception, match="differs"):
        dispatch.verify_blocker_receipt(receipt)


def test_local_adjudication_closes_receiver_null_without_pose_fire() -> None:
    result = dispatch.adjudicate_measurement(
        _measurement(candidate_flips=dispatch.BASE_FLIPS, identical=True),
        _request(),
    )
    assert result["status"] == "DEAD_INSTANCE_RECEIVER_NULL_IDENTICAL_TO_CP135"
    assert result["disposition"] == "FOLDED"
    assert result["pose_unmeasured"] is True
    assert result["pose_job_may_fire"] is False
    assert result["score_claim"] is False


def test_local_adjudication_provisional_gain_keeps_pose_debt() -> None:
    result = dispatch.adjudicate_measurement(
        _measurement(candidate_flips=dispatch.BASE_FLIPS - 1, identical=False),
        _request(),
    )
    assert result["status"] == "PROVISIONALLY_ADMITTED_SEG_SIGN_GATE_POSE_MEASUREMENT_REQUIRED"
    assert result["provisional_seg_sign_admission"] is True
    assert result["pose_follow_up_required_before_composition"] is True
    assert result["pose_job_may_fire"] is True
    assert result["mixed_axis_delta_s_gate_only"] < 0.0
    assert result["score_claim"] is False


def test_local_adjudication_rejects_inconsistent_remote_reduction() -> None:
    measurement = _measurement(candidate_flips=dispatch.BASE_FLIPS - 1, identical=False)
    measurement["field_measurement"]["candidate_minus_base_flips"] = 0
    with pytest.raises(dispatch.RE1TDispatchError, match="flip delta is inconsistent"):
        dispatch.adjudicate_measurement(measurement, _request())


def test_worker_measurement_reports_identity_without_adjudicating() -> None:
    gt = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
    base = np.array([[[0, 0], [1, 0]]], dtype=np.uint8)
    result = worker.field_measurement(base.copy(), gt, base)
    assert result["base_flips_vs_gt"] == 1
    assert result["candidate_flips_vs_gt"] == 1
    assert result["candidate_field_identical_to_cp135"] is True
    assert result["adjudicated_remotely"] is False


def test_fire_order_has_exact_detached_command_and_fresh_id() -> None:
    request = {
        "run_id": dispatch.RUN_ID,
        "lane_id": dispatch.LANE_ID,
        "instance_job_id": dispatch.INSTANCE_JOB_ID,
        "runtime_archive_pin": {"passed": True},
        "blocker_rehash": {"distinct_records_rehashed": 28},
        "remote_scope": "decode and score only",
    }
    order = dispatch.build_fire_order(
        request,
        {"path": str(dispatch.SEALED_REQUEST), "bytes": 100, "sha256": "c" * 64},
    )
    assert order["exact_command_argv"][:3] == [".venv/bin/modal", "run", "--detach"]
    assert order["fresh_run_id"] == "ddm_re1_round1_t4_gate_20260813"
    assert order["pre_fire_verify"]["candidate_archive_sha256"] == dispatch.CANDIDATE_SHA256
    assert (
        order["pre_fire_verify"]["candidate_runtime_tree_sha256"]
        == dispatch.CANDIDATE_RUNTIME_TREE_SHA256
    )
    assert order["local_adjudication_after_harvest"] is True


def test_sealed_loader_refuses_source_drift(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    payloads = {
        "candidate_archive.zip": b"archive",
        "candidate_runtime.zip": b"runtime",
        "RE1X_FULL_N600_BLOCKER.json": b"blocker",
    }
    for name, payload in payloads.items():
        path = input_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    request = {
        "run_id": dispatch.RUN_ID,
        "resume_from": dispatch.RUN_ID,
        "lane_id": dispatch.LANE_ID,
        "instance_job_id": dispatch.INSTANCE_JOB_ID,
        "score_claim": False,
        "promotion_eligible": False,
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "candidate_archive": {
            "path": str(dispatch.CANDIDATE_ARCHIVE.resolve()),
            "bytes": dispatch.CANDIDATE_BYTES,
            "sha256": dispatch.CANDIDATE_SHA256,
        },
        "candidate_runtime": {
            "tree_sha256": dispatch.CANDIDATE_RUNTIME_TREE_SHA256,
            "file_count": dispatch.CANDIDATE_RUNTIME_FILE_COUNT,
        },
        "runtime_archive_pin": {"passed": True},
        "inputs": {
            name: dispatch.js1b_dispatch.payload_record(payload)
            for name, payload in payloads.items()
        },
        "dispatcher_source_sha256": "0" * 64,
        "worker_source_sha256": dispatch.js1b_dispatch.sha256_file(
            dispatch.REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": dispatch.js1b_dispatch.sha256_file(
            dispatch.REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    }
    request_path = tmp_path / "REQUEST.json"
    dispatch.js1b_dispatch.atomic_json(request_path, request)
    with pytest.raises(dispatch.RE1TDispatchError, match="source drift"):
        dispatch.load_sealed_inputs(
            sealed_request=request_path,
            fire_input_dir=input_root,
            expected_request_sha256=dispatch.js1b_dispatch.sha256_file(request_path),
        )


def test_worker_never_checkpoints_volatile_storage_preflight() -> None:
    source = Path(worker.__file__).read_text()
    tree = ast.parse(source)
    assert "sys.dont_write_bytecode = True" in source
    assert 'js1b.sha256_file(Path(__file__))' in source
    assert 'js1b.sha256_file(Path(js1b.__file__))' in source
    checkpoint_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "checkpoint_once"
    ]
    assert checkpoint_calls
    for call in checkpoint_calls:
        assert "storage_preflight" not in ast.dump(call).lower()


def test_dispatcher_and_worker_do_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=dispatch.REPO,
        roots=[
            Path("experiments/ddm_re1t_modal_t4_sign_gate.py"),
            Path("experiments/ddm_re1t_t4_sign_gate_worker.py"),
        ],
    )
    assert findings == []
