# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools/materialize_taskspace_pga_n2_receipt.py"
SPEC = importlib.util.spec_from_file_location("materialize_taskspace_pga_n2_receipt", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _minimal_receipt() -> bytes:
    value = {
        "schema": tool.SCHEMA,
        "lane_id": tool.LANE_ID,
        "scope": "test",
        "competitive_target": {},
        "git_head_before_landing": "a" * 40,
        "encoder_inputs": {},
        "counted_sections": {},
        "whole_object": {},
        "receiver": {},
        "measured_semantic_control": {},
        "open_blockers": [],
        "truth": {
            "candidate_archive_eligible": False,
            "exact_score_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
            "scorer_invoked": False,
            "standalone_runtime_closure": False,
            "through_r_target_realization_verified": False,
        },
        "implementation_custody": [],
    }
    return tool._canonical_json(value) + b"\n"


def test_receipt_strict_parse_reemit_and_truth_labels() -> None:
    payload = _minimal_receipt()
    parsed = tool.parse_materialization_receipt(payload)

    assert parsed["schema"] == tool.SCHEMA
    assert parsed["truth"]["candidate_archive_eligible"] is False
    assert tool._canonical_json(parsed) + b"\n" == payload


@pytest.mark.parametrize("payload", [b"{}", b"{}\n\n", b'{"schema":NaN}\n'])
def test_receipt_refuses_noncanonical_or_nonfinite_payload(payload: bytes) -> None:
    with pytest.raises(tool.TaskspacePGAMaterializationError):
        tool.parse_materialization_receipt(payload)


def test_receipt_refuses_duplicate_keys() -> None:
    payload = _minimal_receipt()
    value = json.loads(payload)
    prefix = b'{"schema":"' + tool.SCHEMA.encode("ascii") + b'","schema":"' + tool.SCHEMA.encode("ascii") + b'",'
    duplicate = prefix + tool._canonical_json(value)[1:]

    with pytest.raises(tool.TaskspacePGAMaterializationError, match="repeats key"):
        tool.parse_materialization_receipt(duplicate + b"\n")


def test_write_once_or_equal_never_overwrites_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    tool.write_once_or_equal(path, b"first")
    tool.write_once_or_equal(path, b"first")
    assert path.read_bytes() == b"first"

    with pytest.raises(tool.TaskspacePGAMaterializationError, match="refusing to overwrite"):
        tool.write_once_or_equal(path, b"second")
    assert path.read_bytes() == b"first"


def test_receiver_call_is_explicit_runtime_not_source_path() -> None:
    source = TOOL_PATH.read_text()
    receiver_region = source.split("receive_kwargs =", maxsplit=1)[1].split("decoded_second =", maxsplit=1)[0]

    assert '"predictor_runtime": source.runtime' in receiver_region
    assert "EP725_SOURCE_DIRECTORY" not in receiver_region
    assert "predictor_state=" not in receiver_region
