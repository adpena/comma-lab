# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools/measure_taskspace_n2_stage_ablation.py"
SPEC = importlib.util.spec_from_file_location("measure_taskspace_n2_stage_ablation", TOOL_PATH)
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
        "competitive_target": {},
        "baseline_materialization": {},
        "target_custody": {},
        "scorer_custody": {},
        "realization_profile_custody": {},
        "semantic_control": {},
        "variant_order": list(tool.VARIANT_ORDER),
        "variants": {name: {} for name in tool.VARIANT_ORDER},
        "scorer_runtime": {},
        "runtime": {},
        "implementation_custody": [],
        "truth": {
            "authoritative_contest_cpu_evaluation": False,
            "authoritative_contest_cuda_evaluation": False,
            "candidate_archive_eligible": False,
            "component_distances_measured": True,
            "dense_frames_persisted": False,
            "n600_evaluation": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "research_only": True,
            "score_claim": False,
        },
    }
    return tool._canonical_json(value) + b"\n"


def test_receipt_is_closed_and_canonical() -> None:
    payload = _minimal_receipt()
    parsed = tool.parse_stage_ablation_receipt(payload)

    assert parsed["variant_order"] == list(tool.VARIANT_ORDER)
    assert parsed["truth"]["score_claim"] is False
    assert tool._canonical_json(parsed) + b"\n" == payload


@pytest.mark.parametrize("payload", [b"{}", b"{}\n\n", b'{"schema":NaN}\n'])
def test_receipt_rejects_noncanonical_payload(payload: bytes) -> None:
    with pytest.raises(tool.TaskspaceN2StageAblationError):
        tool.parse_stage_ablation_receipt(payload)


def test_receipt_rejects_duplicate_keys() -> None:
    with pytest.raises(tool.TaskspaceN2StageAblationError, match="repeats key"):
        tool.parse_stage_ablation_receipt(b'{"schema":"x","schema":"y"}\n')


def test_receipt_rejects_variant_omission() -> None:
    payload = _minimal_receipt()
    value = tool.parse_stage_ablation_receipt(payload)
    value["variants"].pop(tool.VARIANT_ORDER[-1])

    with pytest.raises(tool.TaskspaceN2StageAblationError, match="variant universe"):
        tool.parse_stage_ablation_receipt(tool._canonical_json(value) + b"\n")


def test_write_once_or_equal_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    tool.write_once_or_equal(path, b"one")
    tool.write_once_or_equal(path, b"one")

    with pytest.raises(tool.TaskspaceN2StageAblationError, match="refusing to overwrite"):
        tool.write_once_or_equal(path, b"two")
    assert path.read_bytes() == b"one"
