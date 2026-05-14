# SPDX-License-Identifier: MIT
"""Tests for tac.vastai_tracker.

Per CLAUDE.md non-negotiable + memory feedback_oneshot_vastai_subagent_failure_pattern:
every Vast.ai launch must register the instance ID into a tracker file so
orphans (instances created but never destroyed) can be detected by a
separate cleanup script.

These tests verify the centralized tracker module: it appends new records,
loads them back, removes them after destroy, and is robust to concurrent
writes via file lock.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tac.vastai_tracker import (
    list_instances,
    register_instance,
    remove_instance,
    tracker_path,
)


def test_register_instance_creates_file_and_appends(tmp_path: Path) -> None:
    """First register_instance creates the tracker file and writes one record."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    rec = register_instance(
        instance_id="123456",
        label="pact-test-1",
        metadata={"experiment": "lane_x", "dph": 0.25},
        repo_root=tmp_path,
    )
    assert rec["instance_id"] == "123456"
    assert rec["label"] == "pact-test-1"
    assert rec["metadata"]["dph"] == 0.25
    path = tracker_path(repo_root=tmp_path)
    assert path.exists(), "tracker file must exist after register"
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["instance_id"] == "123456"


def test_register_two_instances_appends_in_order(tmp_path: Path) -> None:
    """Sequential registrations accumulate; order preserved."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    register_instance("1", "label-1", repo_root=tmp_path)
    register_instance("2", "label-2", repo_root=tmp_path)
    records = list_instances(repo_root=tmp_path)
    assert len(records) == 2
    assert [r["instance_id"] for r in records] == ["1", "2"]


def test_remove_instance_drops_record(tmp_path: Path) -> None:
    """remove_instance drops the matching record and returns True."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    register_instance("aaa", "la", repo_root=tmp_path)
    register_instance("bbb", "lb", repo_root=tmp_path)
    found = remove_instance("aaa", repo_root=tmp_path)
    assert found is True
    rest = list_instances(repo_root=tmp_path)
    assert [r["instance_id"] for r in rest] == ["bbb"]


def test_remove_instance_returns_false_if_missing(tmp_path: Path) -> None:
    """remove_instance returns False when the id is not present."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    register_instance("aaa", "la", repo_root=tmp_path)
    found = remove_instance("zzz", repo_root=tmp_path)
    assert found is False
    assert len(list_instances(repo_root=tmp_path)) == 1


def test_register_rejects_empty_id(tmp_path: Path) -> None:
    """Empty instance_id is a programmer error — fail loudly."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    with pytest.raises(ValueError, match="instance_id"):
        register_instance("", "label", repo_root=tmp_path)


def test_register_rejects_non_serialisable_metadata(tmp_path: Path) -> None:
    """Non-JSON-safe metadata fails loudly so caller fixes it before launch."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    with pytest.raises(ValueError, match="JSON-serialisable"):
        register_instance(
            "1", "label",
            metadata={"bad": object()},
            repo_root=tmp_path,
        )


def test_concurrent_register_does_not_corrupt(tmp_path: Path) -> None:
    """Concurrent threads writing to the tracker should not lose records."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    n_threads = 8
    n_per_thread = 4

    def worker(thread_id: int) -> None:
        for i in range(n_per_thread):
            register_instance(
                f"t{thread_id}-i{i}",
                f"label-{thread_id}-{i}",
                repo_root=tmp_path,
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = list_instances(repo_root=tmp_path)
    # All n_threads * n_per_thread records must be present (no lost writes).
    assert len(records) == n_threads * n_per_thread
    seen_ids = {r["instance_id"] for r in records}
    expected_ids = {
        f"t{t}-i{i}" for t in range(n_threads) for i in range(n_per_thread)
    }
    assert seen_ids == expected_ids
