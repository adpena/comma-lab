from __future__ import annotations

import json
from pathlib import Path

from tools import preflight_cache


def test_preflight_cache_roundtrip_and_sha_invalidation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_cache, "REPO", tmp_path)
    monkeypatch.setattr(preflight_cache, "CACHE_DIR", tmp_path / ".omx/state/preflight_cache")

    source = tmp_path / "tool.py"
    source.write_text("print('v1')\n", encoding="utf-8")

    key = preflight_cache.build_cache_key(
        name="unit",
        files=[source],
        config={"mode": "fast"},
    )
    preflight_cache.write_pass_cache("unit", key, {"checks": 3})

    payload = preflight_cache.load_valid_cache("unit", key)
    assert payload is not None
    assert payload["result"] == {"checks": 3}

    source.write_text("print('v2')\n", encoding="utf-8")
    changed_key = preflight_cache.build_cache_key(
        name="unit",
        files=[source],
        config={"mode": "fast"},
    )
    assert preflight_cache.load_valid_cache("unit", changed_key) is None


def test_preflight_cache_rejects_nonpassing_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_cache, "REPO", tmp_path)
    monkeypatch.setattr(preflight_cache, "CACHE_DIR", tmp_path / ".omx/state/preflight_cache")

    source = tmp_path / "tool.py"
    source.write_text("print('v1')\n", encoding="utf-8")
    key = preflight_cache.build_cache_key(name="unit", files=[source])

    preflight_cache.CACHE_DIR.mkdir(parents=True)
    (preflight_cache.CACHE_DIR / "unit.json").write_text(
        json.dumps({"key": key, "passed": False, "result": {}}),
        encoding="utf-8",
    )

    assert preflight_cache.load_valid_cache("unit", key) is None
