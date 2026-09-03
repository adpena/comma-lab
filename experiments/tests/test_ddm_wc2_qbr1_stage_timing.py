from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import ddm_wc2_qbr1_stage_timing as wc2


def test_tree_fact_is_content_and_path_sensitive(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/data.bin").write_bytes(b"payload")
    first = wc2._tree_fact(tmp_path)
    assert first["regular_files"] == 1
    assert first["bytes"] == 7
    (tmp_path / "nested/data.bin").write_bytes(b"payloaD")
    assert wc2._tree_fact(tmp_path)["tree_sha256"] != first["tree_sha256"]


def test_profiler_aggregates_named_stages(tmp_path: Path) -> None:
    profiler = wc2.StageProfiler(tmp_path / "timing.json", synchronize_mps=False, flush_steps=2)
    profiler.record("scorer", 2_000_000_000)
    profiler.record("scorer", 4_000_000_000)
    row = profiler.payload(complete=False)["stages"]["training.scorer"]
    assert row["count"] == 2
    assert row["seconds"] == 6.0
    assert row["mean_seconds"] == 3.0


def test_profiling_restores_every_wrapped_symbol(tmp_path: Path) -> None:
    profiler = wc2.StageProfiler(tmp_path / "timing.json", synchronize_mps=False, flush_steps=2)
    watched = (
        (wc2.qbt, "_target_arrays"),
        (wc2.qbt.QBFLOWTorch, "forward"),
        (wc2.qbr, "fairform_objective"),
        (wc2.qbr, "_append_history"),
    )
    before = tuple(getattr(owner, name) for owner, name in watched)
    with wc2._profiling(profiler):
        inside = tuple(getattr(owner, name) for owner, name in watched)
        assert all(new is not old for new, old in zip(inside, before, strict=True))
    assert tuple(getattr(owner, name) for owner, name in watched) == before


def test_default_off_delegates_without_installing_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"complete": True}
    calls = []

    def fake_run(path: Path) -> dict[str, bool]:
        calls.append(path)
        return sentinel

    monkeypatch.setattr(wc2.qbr, "run_config", fake_run)
    config = Path("sealed.json")
    assert wc2.run_config(config, profile_stages=False, timing_output=None, flush_steps=16) is sentinel
    assert calls == [config]


def test_default_off_refuses_timing_output() -> None:
    with pytest.raises(wc2.WC2TimingError, match="requires --profile-stages"):
        wc2.run_config(
            Path("sealed.json"),
            profile_stages=False,
            timing_output=Path("timing.json"),
            flush_steps=16,
        )


def test_default_off_cpu_tree_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = tmp_path / "resume_smoke"
    tree.mkdir()
    receipt = tree / "RESUME_SMOKE_RESULT.json"
    receipt.write_text(
        json.dumps({"status": "PASS", "axis": "[macOS-CPU exact smoke]"}),
        encoding="utf-8",
    )
    expected_source_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    output = tmp_path / "identity.json"

    def atomic_json(path: Path, payload: object) -> dict[str, object]:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": str(path)}

    monkeypatch.setattr(wc2.qbt, "atomic_json", atomic_json)
    result = wc2.verify_default_off(tree, output)
    assert result["status"] == "PASS"
    assert result["tree_byte_identical"] is True
    assert result["wrappers_installed_when_off"] is False
    assert result["source_resume_smoke"]["sha256"] == expected_source_sha
