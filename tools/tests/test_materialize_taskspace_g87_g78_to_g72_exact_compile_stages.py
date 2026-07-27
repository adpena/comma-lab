# SPDX-License-Identifier: MIT
"""Typed config tests for the G87 five-stage materializer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tac.witness_control.taskspace_g87_g78_to_g72_exact_compile_adapter_v1 import (
    G87ExactCompileAdapterError,
)

TOOL = Path(__file__).resolve().parents[1] / "materialize_taskspace_g87_g78_to_g72_exact_compile_stages.py"
SPEC = importlib.util.spec_from_file_location("g87_materializer_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _body() -> dict[str, str]:
    return {
        "schema": MODULE.CONFIG_SCHEMA,
        "aggregate_receipt_path": "/Volumes/VertigoDataTier/pact/g78.json",
        "aggregate_file_sha256": "a" * 64,
        "aggregate_self_sha256": "b" * 64,
        "output_root": "/Volumes/VertigoDataTier/pact/g87",
    }


def test_config_is_closed_key_and_typed(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_body()), encoding="utf-8")
    config = MODULE.load_config(path)
    assert config.aggregate_file_sha256 == "a" * 64
    assert config.output_root == Path("/Volumes/VertigoDataTier/pact/g87")


def test_extra_config_key_refuses(tmp_path: Path) -> None:
    body = _body()
    body["invented_threshold"] = "1"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(
        G87ExactCompileAdapterError,
        match="key set differs",
    ):
        MODULE.load_config(path)
