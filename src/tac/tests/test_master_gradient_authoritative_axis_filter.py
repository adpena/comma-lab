# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.master_gradient import is_authoritative_axis_anchor, latest_anchor_for_archive


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _anchor(**overrides) -> dict:
    row = {
        "archive_sha256": "a" * 64,
        "gradient_array_path": ".omx/state/fake.npy",
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "autograd_per_parameter_projected_full",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "n_bytes": 16,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    row.update(overrides)
    return row


def test_authoritative_axis_anchor_rejects_advisory_hardware() -> None:
    row = _anchor(measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory")

    assert is_authoritative_axis_anchor(row) is False


def test_authoritative_axis_anchor_rejects_subset_pair_count() -> None:
    row = _anchor(n_pairs_used=8, n_pairs_total=600)

    assert is_authoritative_axis_anchor(row) is False


def test_latest_anchor_for_archive_filters_false_contest_axis_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    stale_advisory = _anchor(
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_utc="2026-05-18T01:00:00Z",
    )
    authoritative = _anchor(
        measurement_hardware="linux_x86_64_cpu",
        measurement_utc="2026-05-18T00:00:00Z",
        n_pairs_used=600,
        n_pairs_total=600,
    )
    _write_rows(ledger, [authoritative, stale_advisory])

    latest = latest_anchor_for_archive("a" * 64, path=ledger, axis="[contest-CPU]")

    assert latest is not None
    assert latest["measurement_hardware"] == "linux_x86_64_cpu"


def test_latest_anchor_for_archive_returns_none_when_only_false_axis_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    _write_rows(
        ledger,
        [
            _anchor(
                measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                n_pairs_used=8,
                n_pairs_total=600,
            )
        ],
    )

    assert latest_anchor_for_archive("a" * 64, path=ledger, axis="[contest-CPU]") is None

