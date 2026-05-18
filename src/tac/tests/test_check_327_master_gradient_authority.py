# SPDX-License-Identifier: MIT
"""Catalog #327 coverage for master-gradient contest-axis custody."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import tac.preflight as preflight_module
from tac.preflight import (
    PreflightError,
    check_master_gradient_contest_axis_requires_authoritative_custody,
)


def _write_ledger(root: Path, rows: list[dict]) -> Path:
    ledger = root / ".omx" / "state" / "master_gradient_anchors.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return ledger


def _anchor(**overrides) -> dict:
    row = {
        "archive_sha256": "f" * 64,
        "gradient_array_path": ".omx/state/fake.npy",
        "measurement_axis": "[contest-CPU]",
        "measurement_call_id": "modal_cpu_123",
        "measurement_hardware": "linux_x86_64_modal_cpu",
        "measurement_method": "autograd_per_parameter_projected_full",
        "measurement_utc": "2026-05-18T00:00:00Z",
        "n_bytes": 178417,
        "n_pairs_used": 600,
        "n_pairs_total": 600,
        "scored_archive_sha256": "f" * 64,
        "scored_archive_bytes": 178517,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    row.update(overrides)
    return row


def test_check_327_blocks_latest_macos_advisory_contest_axis_row(
    tmp_path: Path,
) -> None:
    _write_ledger(
        tmp_path,
        [
            _anchor(
                measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                measurement_method="autograd_per_parameter_projected_8pair_subset",
                measurement_utc="2026-05-18T01:00:00Z",
                n_pairs_used=8,
                n_pairs_total=600,
            )
        ],
    )

    violations = check_master_gradient_contest_axis_requires_authoritative_custody(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "Catalog #327" not in violations[0]
    assert "advisory" in violations[0]
    assert "autograd_per_parameter_projected_8pair_subset" in violations[0]
    with pytest.raises(PreflightError, match="Catalog #327"):
        check_master_gradient_contest_axis_requires_authoritative_custody(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check_327_accepts_append_only_diagnostic_correction(tmp_path: Path) -> None:
    _write_ledger(
        tmp_path,
        [
            _anchor(
                measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                measurement_method="autograd_per_parameter_projected_8pair_subset",
                measurement_utc="2026-05-18T00:00:00Z",
                n_pairs_used=8,
                n_pairs_total=600,
            ),
            _anchor(
                measurement_axis="[macOS-CPU advisory]",
                measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
                measurement_method="autograd_per_parameter_projected_8pair_subset",
                measurement_utc="2026-05-18T01:00:00Z",
                n_pairs_used=8,
                n_pairs_total=600,
            ),
        ],
    )

    assert (
        check_master_gradient_contest_axis_requires_authoritative_custody(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_check_327_accepts_authoritative_full_contest_axis_row(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_anchor()])

    assert (
        check_master_gradient_contest_axis_requires_authoritative_custody(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_check_327_requires_consumer_authority_filter_token(tmp_path: Path) -> None:
    target = tmp_path / "src" / "tac" / "master_gradient_consumers.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "from tac.master_gradient import load_anchors_lenient\n"
        "def load():\n"
        "    return load_anchors_lenient()\n"
    )

    violations = check_master_gradient_contest_axis_requires_authoritative_custody(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert violations
    assert "is_authoritative_axis_anchor" in violations[0]
    with pytest.raises(PreflightError, match="Catalog #327"):
        check_master_gradient_contest_axis_requires_authoritative_custody(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check_327_is_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight_module.preflight_all)
    idx = source.find("check_master_gradient_contest_axis_requires_authoritative_custody")
    assert idx >= 0, "Catalog #327 must be wired into preflight_all"
    call_window = source[idx : idx + 220]
    assert "strict=True" in call_window
