# SPDX-License-Identifier: MIT
"""The harvested-receipt bytes-repr defect and its cure.

`tools/modal_harvest_poller.py` wrote `json.dumps(result, default=str)`. Modal
returns `artifacts` as `dict[str, bytes]`, so every artifact landed as the Python
repr of its bytes and `json.load` raised on it — including on the current
frontier's own receipt (ddm_jg5, 2026-08-20).

`test_bytes_repr_is_the_defect_and_is_not_loadable` is the POSITIVE CONTROL: it
reproduces the old writer and proves the output is unreadable, so the assertions
about the new writer are not measuring an absent hazard.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tac.deploy.modal.result_json import (
    BASE64_PATHS_KEY,
    STRINGIFIED_PATHS_KEY,
    TEXT_DECODED_PATHS_KEY,
    decode_possibly_bytes_repr,
    dump_modal_result_json,
    json_safe_modal_result,
)


def test_bytes_repr_is_the_defect_and_is_not_loadable(tmp_path: Path) -> None:
    """POSITIVE CONTROL: the old writer really does produce unreadable JSON."""

    result = {"artifacts": {"contest_auth_eval.json": b'{"final_score": 0.15}'}}
    old = tmp_path / "old.json"
    old.write_text(json.dumps(result, indent=2, default=str))
    reloaded = json.loads(old.read_text())
    value = reloaded["artifacts"]["contest_auth_eval.json"]
    assert value.startswith("b'"), value[:20]
    with pytest.raises(json.JSONDecodeError):
        json.loads(value)


def test_utf8_artifacts_round_trip_as_readable_text(tmp_path: Path) -> None:
    payload = b'{"schema_version": 1, "final_score": 0.15}'
    result = {"artifacts": {"contest_auth_eval.json": payload}}
    target = tmp_path / "MODAL_REMOTE_RESULT.json"
    dump_modal_result_json(target, result)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    inner = loaded["artifacts"]["contest_auth_eval.json"]
    # The whole point: the inner receipt is itself parseable now.
    assert json.loads(inner)["final_score"] == 0.15
    assert inner.encode("utf-8") == payload
    assert loaded[TEXT_DECODED_PATHS_KEY] == ["artifacts.contest_auth_eval.json"]
    assert loaded[BASE64_PATHS_KEY] == []


def test_binary_artifacts_are_base64_and_recorded_not_repr(tmp_path: Path) -> None:
    blob = bytes(range(256))  # not valid UTF-8
    target = tmp_path / "r.json"
    dump_modal_result_json(target, {"artifacts": {"archive.zip": blob}})

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded[BASE64_PATHS_KEY] == ["artifacts.archive.zip"]
    assert base64.b64decode(loaded["artifacts"]["archive.zip"]) == blob
    assert not loaded["artifacts"]["archive.zip"].startswith("b'")


def test_non_json_values_are_recorded_never_silently_data() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-object"

    projected = json_safe_modal_result({"handle": Opaque(), "n": 3})
    assert projected["handle"] == "opaque-object"
    # The payload is kept, but it can never be mistaken for real data.
    assert projected[STRINGIFIED_PATHS_KEY] == ["handle"]
    assert projected["n"] == 3


def test_nested_bytes_are_found_at_any_depth() -> None:
    projected = json_safe_modal_result(
        {"a": {"b": [{"c": b"deep"}]}, "artifacts": {}},
    )
    assert projected["a"]["b"][0]["c"] == "deep"
    assert projected[TEXT_DECODED_PATHS_KEY] == ["a.b[0].c"]


def test_dump_is_atomic_and_verifies_loadability(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "r.json"
    dump_modal_result_json(target, {"artifacts": {"x": b"ok"}})
    assert target.is_file()
    # No temp files left behind.
    assert [p.name for p in target.parent.iterdir()] == ["r.json"]
    json.loads(target.read_text(encoding="utf-8"))


def test_decode_possibly_bytes_repr_recovers_and_refuses() -> None:
    original = b'{"final_score": 0.15}'
    assert decode_possibly_bytes_repr(repr(original)) == original
    # A plain JSON string is not a repr and must not be reinterpreted.
    assert decode_possibly_bytes_repr('{"final_score": 0.15}') is None
    # A repr-ish string that does not round-trip exactly is refused.
    assert decode_possibly_bytes_repr("b'unterminated") is None
    assert decode_possibly_bytes_repr("") is None


def test_emitters_no_longer_use_default_str() -> None:
    """Anti-regression: the two known emitter sites must not reintroduce it."""

    repo = Path(__file__).resolve().parents[3]
    for relpath in (
        "tools/modal_harvest_poller.py",
        "tools/harvest_click_polish_run.py",
    ):
        text = (repo / relpath).read_text(encoding="utf-8")
        assert "dump_modal_result_json" in text, relpath
        # `default=str` next to a json dump is the defect signature. Comments
        # about the defect are documentation, not the defect, so only CODE
        # lines count — otherwise this test flags its own fix's explanation.
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if "default=str" in code and "json.dump" in code:
                pytest.fail(f"{relpath} reintroduced the bytes-repr writer: {line.strip()}")
