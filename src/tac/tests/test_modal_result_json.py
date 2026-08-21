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
import re
from pathlib import Path

import pytest

from tac.deploy.modal.result_json import (
    BASE64_PATHS_KEY,
    STRINGIFIED_PATHS_KEY,
    TEXT_DECODED_PATHS_KEY,
    BytesInSummaryError,
    bytes_safe_json_default,
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


# ======================================================================================
# CLASS PROTECTION.
#
# The test above names TWO files. A hand-typed denominator is instance protection, not
# class protection: a THIRD emitter is invisible to it, and there already was one —
# tools/harvest_modal_calls.py, the canonical harvester named in CLAUDE.md's "Modal
# .spawn() HARVEST OR LOSE", held a raw result and carried ELEVEN `default=str` dumps
# that no guard could see. The scan below DISCOVERS its own denominator instead.
#
# The rule: a production file that holds a raw Modal FunctionCall result in memory
# may not hand `default=str` to a json dump. `default=str` is the SILENCING mechanism —
# with no default, bytes raise TypeError, which is loud and unshippable; `default=str`
# turns that refusal into a repr that reads as data. Files legitimately needing
# Path/datetime coercion use `bytes_safe_json_default`, which keeps the coercion and
# refuses only the payload case.
# ======================================================================================

# `fc.get()` on a FunctionCall, or the poller's wrapper around it, are the two ways a
# raw Modal result (whose `artifacts` value is `dict[str, bytes]`) enters this repo.
_HOLDS_RAW_RESULT = re.compile(r"FunctionCall\.from_id|poll_modal_call\s*\(")
_SILENT_DUMP = re.compile(r"json\.dump")
_WAIVER = "MODAL_BYTES_REPR_OK:"


def _files_holding_a_raw_modal_result() -> list[Path]:
    repo = Path(__file__).resolve().parents[3]
    found: list[Path] = []
    for root in ("tools", "src", "experiments"):
        for path in sorted((repo / root).rglob("*.py")):
            rel = path.relative_to(repo).as_posix()
            # Vendored/exported snapshots are frozen third-party copies, and tests are
            # allowed to construct the defect on purpose (this file does).
            if "experiments/results/" in rel or "/tests/" in rel or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if _HOLDS_RAW_RESULT.search(text):
                found.append(path)
    return found


def test_the_class_scan_finds_a_real_population_not_an_empty_one() -> None:
    """A guard whose denominator is zero passes vacuously. Report the denominator."""

    holders = _files_holding_a_raw_modal_result()
    # Measured 2026-08-21: 33 production files hold a raw Modal result. The floor is
    # deliberately loose — it exists to catch the scan silently matching nothing (a
    # renamed Modal API, a moved tree), not to pin an exact inventory.
    assert len(holders) >= 20, f"scan matched only {len(holders)} files — it broke"
    names = {p.name for p in holders}
    for expected in ("harvest_modal_calls.py", "modal_harvest_poller.py"):
        assert expected in names, f"{expected} fell out of the scan"


def test_no_result_holder_silences_bytes_with_default_str() -> None:
    """THE CLASS RULE, over a discovered denominator."""

    repo = Path(__file__).resolve().parents[3]
    offenders: list[str] = []
    for path in _files_holding_a_raw_modal_result():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "default=str" in code and _SILENT_DUMP.search(code) and _WAIVER not in line:
                offenders.append(f"{path.relative_to(repo).as_posix()}:{number}: {line.strip()}")
    assert not offenders, (
        "these files hold a raw Modal result (artifacts = dict[str, bytes]) and encode "
        "with `default=str`, which writes a Python bytes-repr that reads as data and "
        "fails json.load. Use tac.deploy.modal.result_json.bytes_safe_json_default for "
        "summaries, or dump_modal_result_json for receipts:\n  " + "\n  ".join(offenders)
    )


def test_bytes_safe_default_refuses_payloads_and_still_coerces_paths() -> None:
    """INVERSE CONTROL PAIR: refuses the case that hurt, keeps the case that helped."""

    with pytest.raises(BytesInSummaryError):
        json.dumps({"artifacts": {"report.txt": b"score 0.15"}}, default=bytes_safe_json_default)
    # And the coercion `default=str` was actually there for still works.
    coerced = json.loads(json.dumps({"p": Path("/x/y")}, default=bytes_safe_json_default))
    assert coerced["p"].endswith("x/y")


def test_the_refusal_names_the_canonical_route_not_just_the_error() -> None:
    """A guard that only says 'no' teaches the next author nothing."""

    with pytest.raises(BytesInSummaryError) as caught:
        json.dumps(b"payload", default=bytes_safe_json_default)
    assert "dump_modal_result_json" in str(caught.value)


def test_default_str_would_have_written_the_repr_the_rule_forbids() -> None:
    """POSITIVE CONTROL for the rule itself: the forbidden default really is unsafe."""

    written = json.dumps({"artifacts": {"report.txt": b"score 0.15"}}, default=str)
    assert "b'score 0.15'" in written
    # And this is exactly what a downstream json.load then chokes on.
    assert json.loads(written)["artifacts"]["report.txt"] == "b'score 0.15'"
