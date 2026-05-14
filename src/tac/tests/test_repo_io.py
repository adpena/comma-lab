# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.repo_io import (
    json_line,
    json_text,
    read_json,
    repo_relative,
    sha256_bytes,
    sha256_file,
    write_json,
)


def test_json_text_is_stable_and_rejects_nan() -> None:
    assert json_text({"b": 1, "a": [2]}) == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'


def test_json_line_is_stable_single_line_jsonl() -> None:
    assert json_line({"b": 1, "a": [2]}) == '{"a":[2],"b":1}\n'


def test_write_read_json_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json(path, {"z": 1, "a": 2})

    assert read_json(path) == {"a": 2, "z": 1}
    assert path.read_text(encoding="utf-8") == json.dumps({"a": 2, "z": 1}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert sha256_file(path) == "4b8884e8891f3aaedfad4ff8f8b08c6159a253c2f93a8b8e1112bd22e18a4162"
    assert sha256_bytes(path.read_bytes()) == sha256_file(path)


def test_repo_relative_uses_posix_path(tmp_path: Path) -> None:
    child = tmp_path / "a" / "b.txt"
    child.parent.mkdir()
    child.write_text("x", encoding="utf-8")

    assert repo_relative(child, tmp_path) == "a/b.txt"
