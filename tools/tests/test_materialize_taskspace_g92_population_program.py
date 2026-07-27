# SPDX-License-Identifier: MIT
"""Closed config, immutable plan, and fail-closed authority tests for G92."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "materialize_taskspace_g92_population_program.py"
SPEC = importlib.util.spec_from_file_location("g92_population_program_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

H = "a" * 64


def _identity() -> dict[str, object]:
    return {"path": "/does/not/exist", "bytes": 1, "sha256": H}


def _config_body() -> dict[str, object]:
    return {
        "schema": MODULE.CONFIG_SCHEMA,
        "output_root": "/Volumes/VertigoDataTier/pact/g92_test",
        "safety_reserve_bytes": 1,
        "semantic_archive": _identity(),
        "current_base_archive": _identity(),
        "g90_aggregate": _identity(),
        "g90_aggregate_self_sha256": H,
        "g51_teacher_receipt": _identity(),
    }


def test_config_is_closed_and_binds_all_required_custody(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config_body()), encoding="utf-8")
    config = MODULE.load_config(path)
    assert config.g90_aggregate_self_sha256 == H
    assert config.semantic_archive.sha256 == H
    assert config.current_base_archive.sha256 == H
    assert config.g51_provenance_receipt.sha256 == H


def test_config_refuses_scalarized_admission_threshold(tmp_path: Path) -> None:
    body = _config_body()
    body["admission_threshold"] = 0.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(MODULE.G92MaterializerError, match="schema/key set differs"):
        MODULE.load_config(path)


def test_production_config_refuses_local_bulk_output(tmp_path: Path) -> None:
    body = _config_body()
    body["output_root"] = str(tmp_path / "local")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(MODULE.G92MaterializerError, match="SSD waterfall"):
        MODULE.load_config(path)


def test_immutable_plan_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "program_plan.json"
    MODULE._atomic_write_bytes(path, b"first")
    MODULE._atomic_write_bytes(path, b"first")
    with pytest.raises(MODULE.G92MaterializerError, match="immutable checkpoint differs"):
        MODULE._atomic_write_bytes(path, b"second")


def test_atomic_publish_refuses_racing_different_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "program_plan.json"

    def racing_link(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"racing-writer")
        raise FileExistsError

    monkeypatch.setattr(MODULE.os, "link", racing_link)
    with pytest.raises(
        MODULE.G92MaterializerError,
        match="immutable checkpoint differs",
    ):
        MODULE._atomic_write_bytes(path, b"our-writer")
    assert path.read_bytes() == b"racing-writer"


def test_missing_g90_aggregate_writes_honest_blocker(tmp_path: Path) -> None:
    body = _config_body()
    for field, payload in (
        ("semantic_archive", b"semantic"),
        ("current_base_archive", b"current-base"),
        ("g51_teacher_receipt", b"receipt"),
    ):
        source = tmp_path / f"{field}.bin"
        source.write_bytes(payload)
        body[field] = {
            "path": str(source),
            "bytes": len(payload),
            "sha256": MODULE.sha256_bytes(payload),
        }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(body), encoding="utf-8")
    config = MODULE.load_config(config_path)
    object.__setattr__(config, "output_root", tmp_path / "out")
    config.output_root.mkdir()
    with pytest.raises(Exception, match="G90 aggregate exact bytes/SHA differ"):
        MODULE._compile_plan(config)
    blocker = MODULE._write_blocker(
        config,
        plan_path=None,
        exc=MODULE.G92MaterializerError("G90 aggregate exact bytes/SHA differ"),
    )
    value = json.loads(blocker.read_text(encoding="utf-8"))
    assert value["candidate_claim"] is False
    assert value["score_claim"] is False
    assert value["pointer_moved"] is False
    assert value["archive_emitted"] is False
    assert value["archive_priced"] is False
