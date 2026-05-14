# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.python_ast_indexer_bridge import index_python_top_level_names_native


def _write_fake_indexer(path: Path, payload: object, *, returncode: int = 0) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"payload = {json.dumps(payload)!r}\n"
        "print(payload)\n"
        f"sys.exit({returncode})\n"
    )
    path.chmod(0o755)
    return path


def _write_counting_fake_indexer(path: Path, counter: Path, payload: object) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib\n"
        f"counter = pathlib.Path({str(counter)!r})\n"
        "counter.write_text(str(int(counter.read_text() or '0') + 1) if counter.exists() else '1')\n"
        f"print({json.dumps(json.dumps(payload))})\n"
    )
    path.chmod(0o755)
    return path


def test_native_top_level_bridge_accepts_batch_success(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n")
    fake = _write_fake_indexer(
        tmp_path / "python-ast-indexer",
        [
            {
                "path": str(source),
                "parse_ok": True,
                "top_level_names": ["VALUE", "f"],
            }
        ],
    )

    result = index_python_top_level_names_native([source], binary_path=fake)

    assert result[source.resolve()] == {"VALUE", "f"}


def test_native_top_level_bridge_omits_parse_failures(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def nope(:\n")
    fake = _write_fake_indexer(
        tmp_path / "python-ast-indexer",
        [
            {
                "path": str(bad),
                "parse_ok": False,
                "error": "parse error",
                "top_level_names": [],
            }
        ],
        returncode=1,
    )

    assert index_python_top_level_names_native([bad], binary_path=fake) == {}


def test_native_top_level_bridge_missing_binary_is_empty(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n")

    assert index_python_top_level_names_native(
        [source],
        binary_path=tmp_path / "missing-indexer",
    ) == {}


def test_native_top_level_bridge_reuses_incremental_cache(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n")
    counter = tmp_path / "count.txt"
    fake = _write_counting_fake_indexer(
        tmp_path / "python-ast-indexer",
        counter,
        [
            {
                "path": str(source),
                "parse_ok": True,
                "top_level_names": ["VALUE"],
            }
        ],
    )

    assert index_python_top_level_names_native([source], binary_path=fake) == {
        source.resolve(): {"VALUE"}
    }
    assert index_python_top_level_names_native([source], binary_path=fake) == {
        source.resolve(): {"VALUE"}
    }
    assert counter.read_text() == "1"
