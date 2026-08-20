# SPDX-License-Identifier: MIT
"""JSON-safe projection of a Modal FunctionCall result.

`json.dumps(result, default=str)` is the defect this module exists to replace.
Modal returns `artifacts` as `dict[str, bytes]`; `default=str` calls `str(b"...")`,
so a receipt lands on disk as the Python *repr* of the bytes —

    "b'{\\n  \\"schema_version\\": 1,\\n  \\"final_score\\": 0.15,..."

— a string that looks like JSON to a human and raises `JSONDecodeError` for
`json.load`. It damaged every arm's `MODAL_REMOTE_RESULT.json`, including the
current frontier's (ddm_jg5, 2026-08-20). It is lossy for genuinely binary
payloads too, because `repr` of non-UTF-8 bytes is not reversible without
`ast.literal_eval` and nothing recorded that the encoding had happened.

The projection here never calls `repr`. Bytes become text when they decode as
UTF-8 and round-trip exactly, and base64 otherwise; both cases are RECORDED, so
a reader can always recover the original bytes and can always tell which
transform was applied. Values that are neither JSON-native nor bytes are
stringified — the payload is never dropped — but their paths are recorded too,
so a stringified object can never be silently mistaken for data.
"""
from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "BASE64_PATHS_KEY",
    "STRINGIFIED_PATHS_KEY",
    "TEXT_DECODED_PATHS_KEY",
    "decode_possibly_bytes_repr",
    "dump_modal_result_json",
    "json_safe_modal_result",
]

TEXT_DECODED_PATHS_KEY = "_utf8_decoded_paths"
BASE64_PATHS_KEY = "_binary_base64_paths"
STRINGIFIED_PATHS_KEY = "_stringified_paths"


def _project(
    value: Any,
    path: str,
    text_paths: list[str],
    b64_paths: list[str],
    str_paths: list[str],
) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None and text.encode("utf-8") == raw:
            text_paths.append(path)
            return text
        b64_paths.append(path)
        return base64.b64encode(raw).decode("ascii")
    if isinstance(value, Mapping):
        return {
            str(key): _project(
                item, f"{path}.{key}" if path else str(key), text_paths, b64_paths, str_paths
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [
            _project(item, f"{path}[{index}]", text_paths, b64_paths, str_paths)
            for index, item in enumerate(value)
        ]
    # Never drop a payload; never let it pass as data either.
    str_paths.append(path)
    return str(value)


def json_safe_modal_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return `result` with every value JSON-serialisable and every transform recorded."""

    text_paths: list[str] = []
    b64_paths: list[str] = []
    str_paths: list[str] = []
    projected = _project(dict(result), "", text_paths, b64_paths, str_paths)
    projected[TEXT_DECODED_PATHS_KEY] = sorted(text_paths)
    projected[BASE64_PATHS_KEY] = sorted(b64_paths)
    projected[STRINGIFIED_PATHS_KEY] = sorted(str_paths)
    return projected


def dump_modal_result_json(path: Path | str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write the JSON-safe projection; return what was written.

    Writing is atomic (tmp + `os.replace`) so a crash mid-write cannot leave a
    truncated receipt that the harvested-JSON gate would then report as damage.
    """

    target = Path(path)
    payload = json_safe_modal_result(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, sort_keys=False)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    # Fail closed: the receipt must be loadable by the next consumer.
    json.loads(target.read_text(encoding="utf-8"))
    return payload


def decode_possibly_bytes_repr(text: str) -> bytes | None:
    """Recover the original bytes from a Python bytes-repr string, or None.

    Repair path for receipts already on disk from the `default=str` era. Returns
    None unless the decode round-trips to exactly the input text, so a string
    that merely looks repr-ish is never silently reinterpreted.
    """

    import ast

    stripped = text.strip()
    if not (stripped.startswith(("b'", 'b"')) and stripped.endswith(("'", '"'))):
        return None
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, bytes) or repr(value) != stripped:
        return None
    return value
