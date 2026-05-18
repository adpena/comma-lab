# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.master_gradient import (
    contest_axis_authority_violation_reason,
    is_authoritative_axis_anchor,
    latest_anchor_for_archive,
    latest_rejected_contest_axis_anchor_for_archive,
    unresolved_contest_axis_authority_violations,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _anchor(**overrides) -> dict:
    row = {
        "archive_sha256": "a" * 64,
        "gradient_array_path": ".omx/state/fake.npy",
        "measurement_axis": "[contest-CPU]",
        "measurement_call_id": "call-test",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "autograd_per_parameter_projected_full",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "n_bytes": 16,
        "n_pairs_used": 600,
        "n_pairs_total": 600,
        "scored_archive_sha256": "a" * 64,
        "scored_archive_bytes": 123,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    row.update(overrides)
    return row


def test_authoritative_axis_anchor_rejects_advisory_hardware() -> None:
    row = _anchor(measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory")

    assert is_authoritative_axis_anchor(row) is False


def test_authoritative_axis_anchor_rejects_shorthand_contest_axis() -> None:
    row = _anchor(
        measurement_axis="contest_cuda",
        measurement_hardware="darwin_arm64_m5_max_macos_mps_advisory",
    )

    assert is_authoritative_axis_anchor(row) is False


def test_authoritative_axis_anchor_rejects_cuda_axis_on_cpu_hardware() -> None:
    row = _anchor(
        measurement_axis="[contest-CUDA]",
        measurement_hardware="linux_x86_64_cpu",
    )

    assert is_authoritative_axis_anchor(row) is False
    assert (
        contest_axis_authority_violation_reason(row)
        == "contest-CUDA axis requires CUDA/GPU hardware"
    )


def test_authoritative_axis_anchor_rejects_missing_pair_count_custody() -> None:
    row = _anchor()
    row.pop("n_pairs_used")

    assert is_authoritative_axis_anchor(row) is False
    assert (
        contest_axis_authority_violation_reason(row)
        == "contest axis missing pair-count custody"
    )


def test_authoritative_axis_anchor_rejects_missing_call_custody() -> None:
    row = _anchor(measurement_call_id="")

    assert is_authoritative_axis_anchor(row) is False
    assert (
        contest_axis_authority_violation_reason(row)
        == "contest axis missing measurement call/runtime custody"
    )


def test_authoritative_axis_anchor_rejects_missing_scored_archive_custody() -> None:
    row = _anchor()
    row.pop("scored_archive_sha256")

    assert is_authoritative_axis_anchor(row) is False
    assert (
        contest_axis_authority_violation_reason(row)
        == "contest axis missing scored archive SHA custody"
    )


def test_authoritative_axis_anchor_rejects_subset_pair_count() -> None:
    row = _anchor(n_pairs_used=8, n_pairs_total=600)

    assert is_authoritative_axis_anchor(row) is False
    assert contest_axis_authority_violation_reason(row) == "contest axis uses pair subset"


def test_authoritative_axis_anchor_rejects_subset_method_even_without_pair_fields() -> None:
    row = _anchor(measurement_method="autograd_per_parameter_projected_8pair_subset")

    assert is_authoritative_axis_anchor(row) is False
    assert (
        contest_axis_authority_violation_reason(row)
        == "contest axis uses subset measurement method"
    )


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
    assert latest_anchor_for_archive("a" * 64, path=ledger) is None


def test_latest_rejected_contest_axis_anchor_reports_reason(tmp_path: Path) -> None:
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    _write_rows(
        ledger,
        [
            _anchor(
                measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                measurement_utc="2026-05-18T00:00:00Z",
            ),
            _anchor(
                measurement_hardware="linux_x86_64_cpu",
                measurement_utc="2026-05-18T01:00:00Z",
                n_pairs_used=8,
                n_pairs_total=600,
            ),
        ],
    )

    rejected = latest_rejected_contest_axis_anchor_for_archive(
        "a" * 64, path=ledger, axis="[contest-CPU]"
    )

    assert rejected is not None
    row, reason = rejected
    assert row["measurement_utc"] == "2026-05-18T01:00:00Z"
    assert reason == "contest axis uses pair subset"


def test_unresolved_contest_axis_violations_respect_append_only_correction() -> None:
    stale = _anchor(
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="autograd_per_parameter_projected_8pair_subset",
        measurement_utc="2026-05-18T00:00:00Z",
    )
    correction = _anchor(
        measurement_axis="[macOS-CPU advisory]",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="autograd_per_parameter_projected_8pair_subset",
        measurement_utc="2026-05-18T01:00:00Z",
    )

    assert unresolved_contest_axis_authority_violations([stale, correction]) == []


def test_unresolved_contest_axis_violations_do_not_hide_different_artifact() -> None:
    stale = _anchor(
        gradient_array_path=".omx/state/stale.npy",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="autograd_per_parameter_projected_8pair_subset",
        measurement_utc="2026-05-18T00:00:00Z",
    )
    unrelated_cuda = _anchor(
        gradient_array_path=".omx/state/cuda.npy",
        measurement_axis="[contest-CUDA]",
        measurement_hardware="linux_x86_64_cuda_t4",
        measurement_utc="2026-05-18T01:00:00Z",
    )

    violations = unresolved_contest_axis_authority_violations([stale, unrelated_cuda])

    assert len(violations) == 1
    assert violations[0][0] == stale
    assert violations[0][1] == "contest axis uses advisory/local/proxy hardware"


def test_unresolved_contest_axis_violations_flag_latest_false_authority() -> None:
    stale = _anchor(
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="autograd_per_parameter_projected_8pair_subset",
        measurement_utc="2026-05-18T01:00:00Z",
    )
    correction = _anchor(
        measurement_axis="[macOS-CPU advisory]",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_method="autograd_per_parameter_projected_8pair_subset",
        measurement_utc="2026-05-18T00:00:00Z",
    )

    violations = unresolved_contest_axis_authority_violations([correction, stale])

    assert len(violations) == 1
    assert violations[0][0] == stale
    assert violations[0][1] == "contest axis uses advisory/local/proxy hardware"
