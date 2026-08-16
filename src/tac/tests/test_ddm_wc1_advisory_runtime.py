from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
RUNTIME_PATH = REPO / "experiments/ddm_wc1_advisory_runtime.py"


def _runtime() -> ModuleType:
    name = "_ddm_wc1_advisory_runtime_for_tests"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_token_cache_populates_hits_and_rehashes_payload(tmp_path: Path) -> None:
    runtime = _runtime()
    source = tmp_path / "retained_tokens.u8"
    source.write_bytes(bytes(range(64)))
    binding = {
        "schema": "ddm_wc1_token_cache_binding.v1",
        "archive_sections": {"token_stream_sha256": "abc"},
        "decoder_code_sha256": "def",
        "thread_config": {"torch_num_threads": 4, "torch_interop_threads": 1},
        "pair_count": 1,
    }
    cache_root = tmp_path / "cache"

    payload, populated = runtime.publish_token_cache(
        cache_root,
        binding,
        source=source,
        expected_bytes=64,
        created={"token_decoder": {"decoder_bit_position": 7}},
    )
    assert populated["status"] == "POPULATED"
    assert payload.read_bytes() == source.read_bytes()

    hit = runtime.load_token_cache(cache_root, binding, expected_bytes=64)
    assert hit is not None
    hit_path, receipt = hit
    assert hit_path == payload
    assert receipt["status"] == "HIT"
    assert receipt["created"]["token_decoder"]["decoder_bit_position"] == 7

    payload.write_bytes(b"x" * 64)
    with pytest.raises(runtime.WC1AdvisoryError, match="payload_sha256"):
        runtime.load_token_cache(cache_root, binding, expected_bytes=64)


def test_token_cache_manifest_must_be_canonical(tmp_path: Path) -> None:
    runtime = _runtime()
    source = tmp_path / "tokens.u8"
    source.write_bytes(b"token-payload")
    binding = {"b": 2, "a": 1}
    cache_root = tmp_path / "cache"
    payload, _ = runtime.publish_token_cache(
        cache_root,
        binding,
        source=source,
        expected_bytes=13,
        created={"test": True},
    )
    manifest = payload.parent / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(runtime.WC1AdvisoryError, match="canonically encoded"):
        runtime.load_token_cache(cache_root, binding, expected_bytes=13)


def test_cache_key_is_order_stable_and_binds_decoder_inputs() -> None:
    runtime = _runtime()
    left = {"archive": {"token": "a", "hpac": "b"}, "threads": 4}
    reordered = {"threads": 4, "archive": {"hpac": "b", "token": "a"}}
    changed = {"threads": 4, "archive": {"hpac": "c", "token": "a"}}
    assert runtime.token_cache_key(left) == runtime.token_cache_key(reordered)
    assert runtime.token_cache_key(left) != runtime.token_cache_key(changed)


def test_worker_count_is_derived_from_live_cpu_and_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 18)
    monkeypatch.setattr(runtime, "_available_memory_bytes", lambda: (32 << 30, "test"))

    premeasure = runtime.resolve_render_workers("auto", per_process_threads=4)
    assert premeasure["selected"] == 4
    assert premeasure["cpu_capacity"] == 4
    assert premeasure["worker_rss_budget_source"].startswith("conservative")

    measured = runtime.resolve_render_workers(
        "auto", per_process_threads=4, measured_worker_rss_bytes=2 << 30
    )
    assert measured["selected"] == 4
    assert measured["worker_rss_budget_bytes"] == int(2.5 * (1 << 30))

    with pytest.raises(runtime.WC1AdvisoryError, match="exceeds live capacity"):
        runtime.resolve_render_workers("5", per_process_threads=4)
