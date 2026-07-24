# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_measure_ddm_mc1_test", ROOT / "tools" / "measure_ddm_mc1_hood_static_reassert.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return {
        "schema": "DDMMC1HoodStaticReassertConfigV1",
        "run_id": "ddm_mc1_test_run",
        "menu1_config_path": ".omx/research/configs/menu1.json",
        "menu1_config_sha256": "a" * 64,
        "menu1_receipt_path": ".omx/research/menu1.json",
        "menu1_receipt_sha256": (
            "2fc12eb505aa7de140b5e785e5fb528c349fd72b423a1821f23b70ea21d6f29d"
        ),
        "checkpoint_root": "/Volumes/VertigoDataTier/pact/ddm_mc1_test",
    }


def test_config_fails_closed_on_authority_escalation() -> None:
    payload = _config()
    payload["score_claim"] = True
    with pytest.raises(ValidationError):
        MODULE.MC1Config.model_validate(payload)


def test_config_requires_primary_ssd() -> None:
    payload = _config()
    payload["checkpoint_root"] = "/tmp/ddm_mc1"
    with pytest.raises(ValidationError, match="primary SSD"):
        MODULE.MC1Config.model_validate(payload)


def test_stable_hash_ignores_no_fields() -> None:
    a = MODULE.MC1Config.model_validate(_config())
    payload = _config()
    payload["run_id"] = "ddm_mc1_test_run_changed"
    b = MODULE.MC1Config.model_validate(payload)
    assert a.stable_hash() != b.stable_hash()
