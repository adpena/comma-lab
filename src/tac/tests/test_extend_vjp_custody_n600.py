# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "extend_vjp_custody_n600", ROOT / "tools/extend_vjp_custody_n600.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_chunks_isolates_known_refusal_and_caps_producer_batches() -> None:
    schedule = MODULE._chunks([11, *range(29, 60)], [11])
    assert schedule[0] == [11]
    assert all(1 <= len(chunk) <= MODULE.MAX_PAIRS for chunk in schedule)
    assert [value for chunk in schedule for value in chunk] == [11, *range(29, 60)]


def test_chunks_reschedules_every_nonrefused_tail_after_multiple_chunk_refusals() -> None:
    missing = [*range(245, 257), *range(277, 281)]
    schedule = MODULE._chunks(missing, [245, 277])
    assert schedule[:2] == [[245], [277]]
    assert all(245 not in chunk and 277 not in chunk for chunk in schedule[2:])
    assert sorted(value for chunk in schedule for value in chunk) == missing
    assert all(len(chunk) <= MODULE.MAX_PAIRS for chunk in schedule)


def test_refused_cached_chunk_becomes_native_retry_plus_complete_tail() -> None:
    recovery = MODULE._recovery_work(
        [245, 246, 247, 248], completed=set(), refused={245}
    )
    assert recovery == [
        ([245], "fresh-native"),
        ([246, 247, 248], "cached-verified"),
    ]

    with pytest.raises(MODULE.ExtensionError):
        MODULE._recovery_work([1, 2], completed=set(), refused={3})


def test_chunks_rejects_duplicate_or_out_of_range_ids() -> None:
    with pytest.raises(MODULE.ExtensionError):
        MODULE._chunks([1, 1], [])
    with pytest.raises(MODULE.ExtensionError):
        MODULE._chunks([600], [])


def test_chunk_directory_name_is_deterministic(tmp_path: Path) -> None:
    first = MODULE._chunk_dir(tmp_path, [29, 30, 31])
    second = MODULE._chunk_dir(tmp_path, [29, 30, 31])
    assert first == second
    assert first.name.startswith("chunk_0029_0031_")

    refreshed = MODULE._chunk_dir(tmp_path, [29, 30, 31], winner_policy="fresh-native")
    assert refreshed != first
    assert refreshed.name.endswith("_fresh_native")


def test_real_source_coverage_validates_sidecars_and_has_28_pairs() -> None:
    coverage, records = MODULE._source_coverage(list(MODULE.DEFAULT_SOURCES))
    assert coverage == set(range(29)) - {11}
    assert len(records) == 5
    assert all(len(record["sha256"]) == 64 for record in records)
