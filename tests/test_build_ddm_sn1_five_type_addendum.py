from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_ddm_sn1_five_type_addendum.py"
SPEC = importlib.util.spec_from_file_location("build_ddm_sn1_five_type_addendum", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_checked_in_addendum_rebuilds_exactly() -> None:
    payload = MODULE.build(ROOT)
    checked_in = json.loads((ROOT / ".omx/research/ddm_sn1_five_type_derivation_addendum_20260724.json").read_text())
    assert payload == checked_in
    assert payload["typed_stream_count"] == 16
    assert set(payload["representation_types_covered"]) == {
        "SKELETON",
        "CONNECTION",
        "FIBER",
        "GAUGE",
        "RESIDUAL",
    }
    assert payload["score_claim"] is False
    assert payload["pointer_moved"] is False


def test_every_emitted_row_has_derivation_home_and_first_rung() -> None:
    payload = MODULE.build(ROOT)
    for row in payload["rows"]:
        assert row["evaluate_recursion_level"].startswith(("L0_", "L1_", "L2_"))
        assert row["layer_home"].startswith(("L1_", "L2_", "L3_", "L4_", "L5_"))
        assert row["derivation"]
        assert row["first_rung"]
        assert row["verdict_scope"]
        assert row["identity_euclidean_control"] is False
        assert row["score_claim"] is False


def test_builder_refuses_pinned_receipt_drift() -> None:
    with pytest.raises(RuntimeError, match="pinned receipt SHA drift"):
        MODULE._load_pinned_receipt(
            ROOT,
            MODULE.SN1_RECEIPT,
            "0" * 64,
        )
