#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect a Modal FunctionCall without blocking on terminal recovery."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _status_text(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return str(value)


def _input_info_to_dict(info: Any) -> dict[str, Any]:
    children = getattr(info, "children", None) or []
    return {
        "input_id": getattr(info, "input_id", ""),
        "function_call_id": getattr(info, "function_call_id", ""),
        "task_id": getattr(info, "task_id", ""),
        "status": _status_text(getattr(info, "status", "")),
        "function_name": getattr(info, "function_name", ""),
        "module_name": getattr(info, "module_name", ""),
        "children": [_input_info_to_dict(child) for child in children],
    }


def inspect_function_call(
    function_call_id: str,
    *,
    get_timeout_s: float = 0.0,
    modal_module: Any | None = None,
) -> dict[str, Any]:
    """Return dashboard, call graph, and non-blocking result readiness."""

    if modal_module is None:
        import modal as modal_module  # type: ignore[no-redef]

    fc = modal_module.functions.FunctionCall.from_id(function_call_id)
    payload: dict[str, Any] = {
        "schema_version": "modal_function_status_v1",
        "function_call_id": function_call_id,
        "dashboard_url": fc.get_dashboard_url(),
        "call_graph": [_input_info_to_dict(info) for info in fc.get_call_graph()],
        "result_state": "unknown",
        "result_type": None,
        "result_error_type": None,
        "result_error": None,
    }
    try:
        result = fc.get(timeout=float(get_timeout_s))
    except TimeoutError as exc:
        payload["result_state"] = "pending"
        payload["result_error_type"] = type(exc).__name__
    except Exception as exc:  # pragma: no cover - depends on Modal service errors.
        payload["result_state"] = "error"
        payload["result_error_type"] = type(exc).__name__
        payload["result_error"] = str(exc)
    else:
        payload["result_state"] = "ready"
        payload["result_type"] = type(result).__name__
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Modal FunctionCall status without blocking for recover()."
    )
    parser.add_argument("function_call_id")
    parser.add_argument("--get-timeout-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    payload = inspect_function_call(
        args.function_call_id,
        get_timeout_s=args.get_timeout_s,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
