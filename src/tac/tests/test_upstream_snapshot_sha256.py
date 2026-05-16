# SPDX-License-Identifier: MIT
"""Sanity tests for ``tac.contest_compliance.compute_upstream_snapshot_sha256``
+ the ``upstream_snapshot_sha256`` field plumbed through the 3 anchor
schemas (ContestResult / CostBandAnchor / Modal call_id ledger).

Per the 12-month strategic-foresight premortem item #2: anchors that lack
the upstream snapshot SHA cannot be cross-checked against a later upstream
rotation; this gate adds the field + canonical helper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.contest_compliance import compute_upstream_snapshot_sha256


def test_compute_returns_none_when_upstream_missing(tmp_path):
    # tmp_path has no upstream/ subdirectory.
    result = compute_upstream_snapshot_sha256(tmp_path)
    assert result is None


def test_compute_stable_under_repeat(tmp_path):
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "alpha.py").write_text("print('alpha')\n")
    (tmp_path / "upstream" / "beta.txt").write_text("beta-content\n")
    sha_a = compute_upstream_snapshot_sha256(tmp_path)
    sha_b = compute_upstream_snapshot_sha256(tmp_path)
    assert sha_a == sha_b
    assert sha_a is not None and len(sha_a) == 64


def test_compute_changes_on_content_mutation(tmp_path):
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "alpha.py").write_text("v1\n")
    sha_v1 = compute_upstream_snapshot_sha256(tmp_path)
    (tmp_path / "upstream" / "alpha.py").write_text("v2\n")
    sha_v2 = compute_upstream_snapshot_sha256(tmp_path)
    assert sha_v1 != sha_v2


def test_compute_skips_pycache_and_pyc(tmp_path):
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "alpha.py").write_text("alpha\n")
    sha_clean = compute_upstream_snapshot_sha256(tmp_path)
    # Add bytecode noise that MUST NOT change the snapshot hash.
    (tmp_path / "upstream" / "__pycache__").mkdir()
    (tmp_path / "upstream" / "__pycache__" / "alpha.cpython-311.pyc").write_bytes(
        b"\x03\xf3\x0d\x0a"
    )
    (tmp_path / "upstream" / "alpha.pyc").write_bytes(b"\x00\x01\x02")
    sha_with_noise = compute_upstream_snapshot_sha256(tmp_path)
    assert sha_clean == sha_with_noise


def test_compute_changes_on_path_rename(tmp_path):
    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream" / "alpha.py").write_text("body\n")
    sha_orig = compute_upstream_snapshot_sha256(tmp_path)
    (tmp_path / "upstream" / "alpha.py").rename(tmp_path / "upstream" / "beta.py")
    sha_renamed = compute_upstream_snapshot_sha256(tmp_path)
    assert sha_orig != sha_renamed


def test_contest_result_carries_field_with_default_none():
    from tac.continual_learning import ContestResult

    r = ContestResult(
        axis="cuda",
        hardware_substrate="linux_x86_64_t4",
        architecture_class="test_class",
        score_value=0.2,
        evidence_tag="[contest-CUDA]",
        archive_sha256="a" * 64,
        archive_bytes=100_000,
    )
    assert hasattr(r, "upstream_snapshot_sha256")
    assert r.upstream_snapshot_sha256 is None

    r2 = ContestResult(
        axis="cuda",
        hardware_substrate="linux_x86_64_t4",
        architecture_class="test_class",
        score_value=0.2,
        evidence_tag="[contest-CUDA]",
        archive_sha256="a" * 64,
        archive_bytes=100_000,
        upstream_snapshot_sha256="b" * 64,
    )
    assert r2.upstream_snapshot_sha256 == "b" * 64


def test_cost_band_anchor_carries_field_and_round_trips_through_jsonl(tmp_path):
    from tac.cost_band_calibration import (
        CostBandAnchor,
        SUCCESSFUL_DISPATCH,
        append_anchor,
        load_anchors,
    )

    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    anchor = CostBandAnchor(
        logged_at_utc="2026-05-16T00:00:00+00:00",
        dispatch_label="test_label",
        trainer="experiments/test.py",
        platform="modal",
        gpu="A100",
        epochs=100,
        batch_size=32,
        all_flags_on=True,
        actual_wall_clock_sec=1000.0,
        actual_cost_usd=2.5,
        outcome=SUCCESSFUL_DISPATCH,
        upstream_snapshot_sha256="c" * 64,
    )
    append_anchor(anchor, posterior_path=posterior, lock_path=lock)
    rows = load_anchors(posterior_path=posterior)
    assert len(rows) == 1
    assert rows[0].upstream_snapshot_sha256 == "c" * 64


def test_cost_band_anchor_legacy_row_loads_as_none(tmp_path):
    from tac.cost_band_calibration import SCHEMA_VERSION, load_anchors, SUCCESSFUL_DISPATCH

    posterior = tmp_path / "posterior.jsonl"
    # Pre-2026-05-16 row missing the new field.
    legacy = {
        "schema": SCHEMA_VERSION,
        "schema_version": 1,
        "logged_at_utc": "2026-05-13T00:00:00+00:00",
        "dispatch_label": "legacy",
        "trainer": "x.py",
        "platform": "modal",
        "gpu": "T4",
        "epochs": 50,
        "batch_size": 16,
        "all_flags_on": False,
        "actual_wall_clock_sec": 500.0,
        "actual_cost_usd": 0.5,
        "predicted_cost_usd_low": None,
        "predicted_cost_usd_high": None,
        "prediction_in_band": None,
        "outcome": SUCCESSFUL_DISPATCH,
        "returncode": None,
        "notes": "",
    }
    posterior.write_text(json.dumps(legacy) + "\n")
    rows = load_anchors(posterior_path=posterior)
    assert len(rows) == 1
    assert rows[0].upstream_snapshot_sha256 is None


def test_modal_call_id_ledger_accepts_and_persists_upstream_sha(tmp_path):
    from tac.deploy.modal.call_id_ledger import (
        register_dispatched_call_id,
    )

    ledger_path = tmp_path / "ledger.jsonl"
    lock_path = tmp_path / "ledger.lock"
    record = register_dispatched_call_id(
        call_id="fc-test-001",
        lane_id="lane_test_20260516",
        label="test_label",
        upstream_snapshot_sha256="d" * 64,
        path=ledger_path,
        lock_path=lock_path,
    )
    assert record["upstream_snapshot_sha256"] == "d" * 64
    # Round-trip through the JSONL file.
    lines = [
        line for line in ledger_path.read_text().splitlines() if line.strip()
    ]
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["upstream_snapshot_sha256"] == "d" * 64


def test_compute_returns_string_for_live_repo_upstream():
    """The live repo's upstream/ exists; the helper should return a 64-hex digest."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sha = compute_upstream_snapshot_sha256(repo_root)
    # The live repo HAS upstream/; this is a real receipt that the helper
    # works against the canonical contest snapshot.
    if (repo_root / "upstream").exists():
        assert sha is not None and len(sha) == 64 and all(
            c in "0123456789abcdef" for c in sha
        )
