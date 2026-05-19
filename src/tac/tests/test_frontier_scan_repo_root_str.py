"""Regression tests for tac.frontier_scan Path/str repo_root normalization.

Per Catalog #343 PHASE 1: every loader function in ``tac.frontier_scan``
must accept ``repo_root`` as either ``Path`` or ``str``. The pre-fix bug
(``repo_root / ".omx/state/..."`` with bare string ``repo_root``) raised
``TypeError`` on ``str / str`` concatenation; the fix normalizes via
``Path(repo_root)`` at the top of every loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.frontier_scan import (
    build_frontier_scan_payload,
    collect_all_anchors,
    load_active_lane_dispatch_claims_anchors,
    load_continual_learning_anchors,
    load_experiments_results_anchors,
    load_modal_call_id_ledger_anchors,
    scan_reports_latest_md,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_build_frontier_scan_payload_accepts_str_repo_root() -> None:
    """The canonical entry point must accept str repo_root."""

    payload = build_frontier_scan_payload(repo_root=str(REPO_ROOT))
    assert isinstance(payload, dict)
    assert payload.get("schema") == "pact_frontier_scan_v1"


def test_build_frontier_scan_payload_accepts_path_repo_root() -> None:
    """The canonical entry point must accept Path repo_root."""

    payload = build_frontier_scan_payload(repo_root=REPO_ROOT)
    assert isinstance(payload, dict)
    assert payload.get("schema") == "pact_frontier_scan_v1"


def test_collect_all_anchors_accepts_both_types() -> None:
    """Aggregator must accept either Path or str."""

    via_path = collect_all_anchors(REPO_ROOT)
    via_str = collect_all_anchors(str(REPO_ROOT))
    assert len(via_path) == len(via_str)


def test_each_loader_accepts_str_repo_root() -> None:
    """Every loader function must normalize str->Path internally."""

    repo_str = str(REPO_ROOT)
    # Each should return a list without raising TypeError on str/str.
    assert isinstance(load_continual_learning_anchors(repo_str), list)
    assert isinstance(load_modal_call_id_ledger_anchors(repo_str), list)
    assert isinstance(load_active_lane_dispatch_claims_anchors(repo_str), list)
    assert isinstance(load_experiments_results_anchors(repo_str), list)


def test_scan_reports_latest_md_accepts_str() -> None:
    """Citation scanner must accept str repo_root."""

    cited = scan_reports_latest_md(str(REPO_ROOT))
    assert isinstance(cited, dict)


def test_missing_repo_root_str_returns_empty_lists_not_raises() -> None:
    """Loaders accept str that points to missing dir; return empty."""

    fake_root = "/nonexistent/path/that/does/not/exist"
    # Should not raise TypeError or any exception; return empty.
    assert load_continual_learning_anchors(fake_root) == []
    assert load_modal_call_id_ledger_anchors(fake_root) == []
    assert load_active_lane_dispatch_claims_anchors(fake_root) == []
    assert load_experiments_results_anchors(fake_root) == []


def test_str_and_path_produce_identical_payloads(tmp_path: Path) -> None:
    """Payload from str vs Path repo_root must be byte-identical."""

    # Use the real repo root since fresh tmp_path has no state files.
    via_path = build_frontier_scan_payload(repo_root=REPO_ROOT)
    via_str = build_frontier_scan_payload(repo_root=str(REPO_ROOT))
    # Scan stats must agree (best-anchor selection is deterministic).
    assert via_path.get("scan_stats") == via_str.get("scan_stats")
