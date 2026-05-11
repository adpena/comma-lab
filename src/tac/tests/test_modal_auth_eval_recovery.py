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
