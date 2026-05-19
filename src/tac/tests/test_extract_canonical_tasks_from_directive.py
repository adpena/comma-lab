"""Tests for tools/extract_canonical_tasks_from_directive.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "extract_canonical_tasks_from_directive.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "extract_canonical_tasks_from_directive_under_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _item_ids(tasks: list[dict[str, object]]) -> list[str]:
    return [str(task["item_id"]) for task in tasks]


def test_extracts_current_directive_heading_dialects(tmp_path: Path) -> None:
    directive = tmp_path / "codex_routing_directive_heading_fixture.md"
    directive.write_text(
        "\n".join(
            [
                "# Fixture",
                "### Wire-in #3: per-pair difficulty atlas",
                "predicted_cost_usd: 0.0",
                "### Build #2: Z6-v2 Wave 2",
                "### OP-7 FIRST: Direct master-gradient pose-byte hoist",
                "#### Item 1: titlecase paid dispatch",
                "## CLUSTER B - META event-driven retroactive sweep gate",
                "## CLUSTER F - parent bundle should not become a task",
                "**Sub-cluster F.1 - sigma=15 per-substrate sweep design**:",
                "**Sub-cluster F.2 - 600-pair independence test**:",
            ]
        ),
        encoding="utf-8",
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == [
        "WIRE_IN_3",
        "BUILD_2",
        "OP_7",
        "ITEM_1",
        "CLUSTER_B",
        "CLUSTER_F1",
        "CLUSTER_F2",
    ]
    assert "CLUSTER_F" not in _item_ids(tasks)


def test_extracts_comprehensive_wire_in_live_directive() -> None:
    directive = (
        REPO
        / ".omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md"
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == [
        "WIRE_IN_1",
        "WIRE_IN_2",
        "WIRE_IN_3",
        "WIRE_IN_4",
        "WIRE_IN_5",
        "WIRE_IN_6",
        "BUILD_1",
        "BUILD_2",
        "BUILD_3",
    ]


def test_extracts_paid_batch_titlecase_items_live_directive() -> None:
    directive = (
        REPO
        / ".omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md"
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == ["ITEM_1", "ITEM_2", "ITEM_3", "ITEM_4", "ITEM_5", "ITEM_6"]
    assert tasks[0]["predicted_cost_usd"] is None


def test_extracts_bcef_clusters_and_subclusters_live_directive() -> None:
    directive = (
        REPO
        / ".omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md"
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == [
        "CLUSTER_B",
        "CLUSTER_C",
        "CLUSTER_E1",
        "CLUSTER_F1",
        "CLUSTER_F2",
    ]
    assert tasks[3]["predicted_delta_s_band"] == (-0.002, -0.0003)


def test_extracts_op_headings_from_live_canonical_package_directive() -> None:
    directive = (
        REPO
        / ".omx/research/codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md"
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == ["OP_1", "OP_2", "OP_3", "OP_4", "OP_5", "OP_6", "OP_7"]
    assert tasks[0]["title"] == "Layer 1 — canonical helper module"


def test_extracts_op_first_heading_from_live_pose_axis_directive() -> None:
    directive = (
        REPO
        / ".omx/research/codex_routing_directive_cheap_probe_wave_pose_axis_op1_op2_op6_op7_op10_20260518.md"
    )

    tasks = _load_tool().extract_tasks_from_directive(directive)

    assert _item_ids(tasks) == ["OP_7", "OP_2", "OP_10", "OP_1", "OP_6"]
    assert tasks[0]["title"] == "Direct master-gradient pose-byte hoist"
