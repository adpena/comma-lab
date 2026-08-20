# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.taskspace_outer_archive_codec import OuterArchiveEncoding
from tac.witness_dsl.taskspace_whole_archive_allocator import (
    TaskspaceMeasurementRequestV1,
    TaskspaceReceiverRequestV1,
)

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools/run_taskspace_a3_n2_allocator.py"
SPEC = importlib.util.spec_from_file_location("run_taskspace_a3_n2_allocator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _minimal_receipt() -> bytes:
    value = {
        "schema": tool.SCHEMA,
        "lane_id": tool.LANE_ID,
        "axis": tool.AXIS,
        "scope": "test",
        "git_head_before_landing": "0" * 40,
        "implementation_custody": [],
        "command": [],
        "competitive_target": {},
        "row_counts": [1, 4, 16],
        "encoder_custody": {},
        "acquisition": {},
        "allocation": {},
        "final_archive": {},
        "scorer_session": {},
        "runtime": {},
        "open_blockers": [],
        "truth": tool.TRUTH,
    }
    return tool._canonical_json(value) + b"\n"


def test_geometric_ladder_and_paired_interpretation_order() -> None:
    rows = tool.parse_row_counts("1,4,16")
    plans = tool.build_prefix_plans(rows)

    assert rows == tool.DEFAULT_ROW_COUNTS
    assert tuple(plan.interpretation.value for plan in plans) == (
        "TARGET_CONSTANT_RGB_V1",
        "CORRECTED_Y1_SUPPORT_COPY_V1",
    )
    assert all(plan.row_counts == rows for plan in plans)


@pytest.mark.parametrize("value", ["", "0,4", "4,1", "1,3,16", "1,1"])
def test_row_ladder_refuses_nonpositive_unordered_or_nongeometric(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tool.parse_row_counts(value)


def test_receipt_parse_reemit_and_authority_are_closed() -> None:
    payload = _minimal_receipt()
    parsed = tool.parse_allocation_receipt(payload)

    assert parsed["truth"]["research_only"] is True
    assert parsed["truth"]["n2_only"] is True
    assert parsed["truth"]["n600_evaluation"] is False
    assert parsed["truth"]["exact_score_claim"] is False
    assert tool._canonical_json(parsed) + b"\n" == payload


@pytest.mark.parametrize("payload", [b"{}", b"{}\n\n", b'{"schema":NaN}\n'])
def test_receipt_refuses_noncanonical_payload(payload: bytes) -> None:
    with pytest.raises(tool.TaskspaceA3N2AllocatorRunnerError):
        tool.parse_allocation_receipt(payload)


def test_receipt_refuses_duplicate_keys() -> None:
    with pytest.raises(tool.TaskspaceA3N2AllocatorRunnerError, match="repeats key"):
        tool.parse_allocation_receipt(b'{"schema":"x","schema":"y"}\n')


def test_write_once_or_equal_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    tool.write_once_or_equal(path, b"one")
    tool.write_once_or_equal(path, b"one")

    with pytest.raises(tool.TaskspaceA3N2AllocatorRunnerError, match="refusing to overwrite"):
        tool.write_once_or_equal(path, b"two")
    assert path.read_bytes() == b"one"


def test_ephemeral_cache_requires_exact_archive_output_foreign_keys() -> None:
    cache = tool.EphemeralFrameCache()
    archive = b"archive"
    member = b"member"
    receiver_request = TaskspaceReceiverRequestV1(
        stage_id="baseline",
        encoding=OuterArchiveEncoding.STORED,
        archive_bytes=archive,
        archive_sha256=tool._sha256(archive),
        archive_nbytes=len(archive),
        member_sha256=tool._sha256(member),
        member_nbytes=len(member),
    )
    frames = np.zeros((1, 2, 3, 4, 3), dtype=np.uint8)
    cached = cache.record(receiver_request, frames)
    measurement_request = TaskspaceMeasurementRequestV1(
        stage_id=receiver_request.stage_id,
        selected_encoding=receiver_request.encoding,
        archive_bytes=receiver_request.archive_bytes,
        archive_sha256=receiver_request.archive_sha256,
        archive_nbytes=receiver_request.archive_nbytes,
        member_sha256=receiver_request.member_sha256,
        member_nbytes=receiver_request.member_nbytes,
        decoded_output_sha256=cached.output_sha256,
        decoded_output_nbytes=cached.output_nbytes,
        receiver_receipt_sha256="a" * 64,
    )

    assert np.array_equal(cache.consume(measurement_request), frames)
    with pytest.raises(tool.TaskspaceA3N2AllocatorRunnerError, match="not cache-bound"):
        cache.consume(measurement_request)


def test_print_command_never_enters_real_allocation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("real allocation must not run")

    monkeypatch.setattr(tool, "run_real_allocation", forbidden)

    assert tool.main(["--print-authorized-command"]) == 0
    output = capsys.readouterr().out
    assert "--execute-reviewed" in output
    assert "--row-counts 1,4,16" in output
