from __future__ import annotations

import types

from tools.modal_function_status import inspect_function_call


class _FakeStatus:
    name = "PENDING"


class _FakeInfo:
    input_id = "in-1"
    function_call_id = "fc-1"
    task_id = ""
    status = _FakeStatus()
    function_name = "run_t1_balle_modal"
    module_name = "modal_t1_balle_endtoend"
    children = []


class _FakeCall:
    def get_dashboard_url(self) -> str:
        return "https://modal.com/id/fc-1"

    def get_call_graph(self) -> list[_FakeInfo]:
        return [_FakeInfo()]

    def get(self, timeout: float = 0.0):
        raise TimeoutError()


class _FakeFunctionCall:
    @classmethod
    def from_id(cls, function_call_id: str) -> _FakeCall:
        assert function_call_id == "fc-1"
        return _FakeCall()


def test_inspect_function_call_reports_pending_without_blocking() -> None:
    fake_modal = types.SimpleNamespace(
        functions=types.SimpleNamespace(FunctionCall=_FakeFunctionCall)
    )

    payload = inspect_function_call("fc-1", modal_module=fake_modal)

    assert payload["schema_version"] == "modal_function_status_v1"
    assert payload["function_call_id"] == "fc-1"
    assert payload["dashboard_url"] == "https://modal.com/id/fc-1"
    assert payload["result_state"] == "pending"
    assert payload["call_graph"][0]["status"] == "pending"
    assert payload["call_graph"][0]["function_name"] == "run_t1_balle_modal"
