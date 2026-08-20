# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools/measure_taskspace_pga_n2_macos_cpu.py"
SPEC = importlib.util.spec_from_file_location("measure_taskspace_pga_n2_macos_cpu", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _minimal_receipt() -> bytes:
    value = {
        "schema": tool.SCHEMA,
        "axis": tool.AXIS,
        "scope": "test",
        "git_head_before_landing": "0" * 40,
        "implementation_custody": [],
        "supersedes_receipt": {},
        "competitive_target": {},
        "archive": {},
        "receiver": {},
        "target_custody": {},
        "scorer_custody": {},
        "measurement": {},
        "runtime": {},
        "truth": {
            "authoritative_contest_cpu_evaluation": False,
            "authoritative_contest_cuda_evaluation": False,
            "candidate_archive_eligible": False,
            "component_distances_measured": True,
            "n600_evaluation": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "research_only": True,
            "score_claim": False,
        },
    }
    return tool._canonical_json(value) + b"\n"


def test_measurement_receipt_is_canonical_and_authority_closed() -> None:
    payload = _minimal_receipt()
    parsed = tool.parse_measurement_receipt(payload)

    assert parsed["axis"] == tool.AXIS
    assert parsed["truth"]["n600_evaluation"] is False
    assert parsed["truth"]["score_claim"] is False
    assert tool._canonical_json(parsed) + b"\n" == payload


@pytest.mark.parametrize("payload", [b"{}", b"{}\n\n", b'{"schema":NaN}\n'])
def test_measurement_receipt_refuses_noncanonical_payload(payload: bytes) -> None:
    with pytest.raises(tool.TaskspacePGAN2MeasurementError):
        tool.parse_measurement_receipt(payload)


def test_measurement_receipt_refuses_duplicate_keys() -> None:
    duplicate = b'{"schema":"x","schema":"y"}\n'

    with pytest.raises(tool.TaskspacePGAN2MeasurementError, match="repeats key"):
        tool.parse_measurement_receipt(duplicate)


def test_write_once_or_equal_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    tool.write_once_or_equal(path, b"one")
    tool.write_once_or_equal(path, b"one")

    with pytest.raises(tool.TaskspacePGAN2MeasurementError, match="refusing to overwrite"):
        tool.write_once_or_equal(path, b"two")
    assert path.read_bytes() == b"one"
