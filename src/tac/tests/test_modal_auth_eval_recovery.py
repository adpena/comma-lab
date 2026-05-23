# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import types
from pathlib import Path

from tac.deploy.modal.auth_eval import (
    function_call_id,
    recover_modal_auth_eval,
    write_spawn_metadata,
)


class _FakeFunctionCall:
    result: object = {}

    @classmethod
    def from_id(cls, call_id: str):
        assert call_id == "fc-test"
        return cls()

    def get(self, timeout: float = 0.0):
        if self.result == "pending":
            raise TimeoutError
        return self.result


def _fake_modal(result: object):
    class FunctionCall(_FakeFunctionCall):
        pass

    FunctionCall.result = result
    return types.SimpleNamespace(functions=types.SimpleNamespace(FunctionCall=FunctionCall))


def _contest_cuda_auth_eval_payload() -> bytes:
    return json.dumps(
        {
            "score_recomputed_from_components": 25.0,
            "archive_size_bytes": 37_545_489,
            "avg_segnet_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "score_axis": "contest_cuda",
            "evidence_grade": "contest-CUDA",
            "lane_tag": "[contest-CUDA Modal A100]",
            "exact_cuda_eval_complete": True,
            "score_claim": True,
            "score_claim_valid": True,
            "promotion_eligible": True,
        }
    ).encode("utf-8")


def _contest_cpu_auth_eval_payload() -> bytes:
    return json.dumps(
        {
            "score_recomputed_from_components": 25.0,
            "archive_size_bytes": 37_545_489,
            "avg_segnet_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "score_axis": "contest_cpu",
            "evidence_grade": "contest-CPU",
            "lane_tag": "[contest-CPU Modal Linux x86_64]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": True,
        }
    ).encode("utf-8")


def test_function_call_id_accepts_modal_sdk_object_id() -> None:
    assert function_call_id(types.SimpleNamespace(object_id="fc-test")) == "fc-test"
    assert function_call_id(types.SimpleNamespace(function_call_id="fc-alt")) == "fc-alt"


def test_write_spawn_metadata_and_recover_artifacts(tmp_path: Path) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={"archive_sha256": "a" * 64},
        result_json_name="modal_cuda_auth_eval_result.json",
        extra={"lane_id": "lane_test", "instance_job_id": "job_test"},
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "promotion_eligible": False,
        "score_recomputed_from_components": 0.208,
        "artifacts": {
            "contest_auth_eval.json": b'{"score_recomputed_from_components": 0.208}\n',
            "inflated_outputs_manifest.json": b'{"aggregate_sha256": "' + b"b" * 64 + b'"}\n',
        },
    }
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered"
    assert summary["passed"] is True
    assert summary["score_recomputed_from_components"] == 0.208
    assert (tmp_path / "contest_auth_eval.json").is_file()
    assert (tmp_path / "inflated_outputs_manifest.json").is_file()
    persisted = json.loads((tmp_path / "modal_cuda_auth_eval_result.json").read_text())
    assert "artifacts" not in persisted
    assert persisted["returncode"] == 0


def test_write_spawn_metadata_forces_false_authority_extra_fields(
    tmp_path: Path,
) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={"archive_sha256": "a" * 64},
        result_json_name="modal_cuda_auth_eval_result.json",
        extra={
            "score_claim": True,
            "score_claim_valid": True,
            "promotion_eligible": True,
            "promotable": True,
            "rank_or_kill_eligible": True,
            "ready_for_exact_eval_dispatch": True,
        },
    )

    payload = json.loads((tmp_path / "modal_auth_eval_spawn.json").read_text())

    assert payload["score_claim"] is False
    assert payload["score_claim_valid"] is False
    assert payload["promotion_eligible"] is False
    assert payload["promotable"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_recover_prefers_canonical_auth_eval_claim_flags(tmp_path: Path) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="diagnostic_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "promotion_eligible": True,
        "score_recomputed_from_components": 0.208,
        "artifacts": {
            "contest_auth_eval.json": (
                b'{'
                b'"score_recomputed_from_components": 0.208,'
                b'"score_claim": false,'
                b'"score_claim_valid": false,'
                b'"promotion_eligible": false,'
                b'"score_axis": "diagnostic_cuda",'
                b'"evidence_grade": "B",'
                b'"diagnostic_blockers": ["inflate_device_policy_cpu"],'
                b'"provenance": {"inflate_device_policy": "cpu"}'
                b'}\n'
            ),
        },
    }

    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered"
    assert summary["passed"] is True
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["score_axis"] == "diagnostic_cuda"
    assert summary["diagnostic_blockers"] == [
        "unsupported_modal_auth_eval_recovery_axis:diagnostic_cuda",
        "inflate_device_policy_cpu",
    ]
    assert summary["inflate_device_policy"] == "cpu"


def test_recover_refuses_diagnostic_axis_even_if_artifact_claims_score(
    tmp_path: Path,
) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="diagnostic_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    payload = json.loads(_contest_cuda_auth_eval_payload().decode("utf-8"))
    payload["score_axis"] = "diagnostic_cuda"
    payload["evidence_grade"] = "B"
    payload["promotion_eligible"] = True
    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "promotion_eligible": True,
        "artifacts": {"contest_auth_eval.json": json.dumps(payload).encode("utf-8")},
    }

    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["diagnostic_blockers"] == [
        "unsupported_modal_auth_eval_recovery_axis:diagnostic_cuda"
    ]


def test_recover_accepts_contest_cpu_leaderboard_artifact_as_cpu_score_claim(
    tmp_path: Path,
) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval_cpu.py",
        app="comma-auth-eval-cpu",
        axis="contest_cpu",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cpu_auth_eval_result.json",
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "promotion_eligible": True,
        "artifacts": {"contest_auth_eval.json": _contest_cpu_auth_eval_payload()},
    }
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered"
    assert summary["score_claim"] is True
    assert summary["promotion_eligible"] is False
    assert summary["score_axis"] == "contest_cpu"


def test_recover_pending_writes_pending_summary(tmp_path: Path) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval_cpu.py",
        app="comma-auth-eval-cpu",
        axis="contest_cpu",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cpu_auth_eval_result.json",
    )

    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal("pending"),
    )

    assert summary["status"] == "pending"
    assert summary["score_claim"] is False
    assert json.loads((tmp_path / "modal_auth_eval_recover_summary.json").read_text())[
        "status"
    ] == "pending"


def test_recover_remote_error_writes_fail_closed_summary(tmp_path: Path) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    class FunctionCall(_FakeFunctionCall):
        def get(self, timeout: float = 0.0):
            raise RuntimeError("remote failed before artifact return")

    modal_module = types.SimpleNamespace(functions=types.SimpleNamespace(FunctionCall=FunctionCall))
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=modal_module,
    )

    assert summary["status"] == "remote_error"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert "remote failed" in summary["error"]


def test_recover_contest_cuda_requires_canonical_auth_eval_artifact(
    tmp_path: Path,
) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": True,
        "score_recomputed_from_components": 25.0,
    }
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered_missing_canonical_auth_eval_artifact"
    assert summary["passed"] is False
    assert summary["returncode"] == 97
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["diagnostic_blockers"] == [
        "missing_canonical_contest_auth_eval_json"
    ]


def test_recover_contest_cuda_accepts_valid_canonical_auth_eval_artifact(
    tmp_path: Path,
) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": False,
        "promotion_eligible": False,
        "artifacts": {
            "contest_auth_eval.json": _contest_cuda_auth_eval_payload(),
        },
    }
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "recovered"
    assert summary["score_claim"] is True
    assert summary["promotion_eligible"] is True
    assert summary["score_axis"] == "contest_cuda"


def test_recover_rejects_unsafe_modal_artifact_paths(tmp_path: Path) -> None:
    write_spawn_metadata(
        out_dir=tmp_path,
        tool="experiments/modal_auth_eval.py",
        app="comma-auth-eval",
        axis="contest_cuda",
        call_id="fc-test",
        local_request={},
        result_json_name="modal_cuda_auth_eval_result.json",
    )

    result = {
        "passed": True,
        "returncode": 0,
        "score_claim": True,
        "artifacts": {
            ".": b"{}",
            "contest_auth_eval.json": _contest_cuda_auth_eval_payload(),
        },
    }
    summary = recover_modal_auth_eval(
        out_dir=tmp_path,
        timeout_s=0,
        modal_module=_fake_modal(result),
    )

    assert summary["status"] == "invalid_artifacts"
    assert summary["score_claim"] is False
    assert not (tmp_path / "contest_auth_eval.json").exists()
    assert not (tmp_path / "contest_auth_eval.json").exists()
