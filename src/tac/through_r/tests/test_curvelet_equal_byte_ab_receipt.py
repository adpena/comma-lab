from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.through_r.equal_archive_budget import ArchiveBudgetError, file_sha256
from tools.curvelet_equal_byte_ab_receipt import (
    finalize_transfer,
    match_archives,
    tree_manifest_sha256,
)

PROGRAM_SHA = "1" * 64


def _zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("0.bin", payload)


def _measurement(
    *,
    family: str,
    archive: Path,
    d_seg: float,
    output_tree_sha256: str,
    basis_program_sha256: str | None = None,
) -> dict:
    row = {
        "family": family,
        "archive_sha256": file_sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "n_pairs": 600,
        "n_samples": 600,
        "scorer_batch_size": 32,
        "through_r": True,
        "official_evaluator": True,
        "parse_back": True,
        "output_tree_sha256": output_tree_sha256,
        "upstream_snapshot_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "evaluate_report_sha256": "6" * 64,
        "segnet_weights_sha256": "7" * 64,
        "posenet_weights_sha256": "8" * 64,
        "git_sha": "9" * 40,
        "measurement_utc": "2026-07-16T20:00:00Z",
        "hardware_substrate": "contest-linux-x86_64-cpu",
        "torch_version": "2.8.0",
        "device": "cpu",
        "evaluator_argv": ["python3", "upstream/evaluate.py", "--device", "cpu"],
        "seed": 0,
        "axis": "contest-CPU",
        "d_seg": d_seg,
        "d_pose": 0.02,
    }
    if basis_program_sha256 is not None:
        row["basis_program_sha256"] = basis_program_sha256
    return row


def test_two_phase_driver_rate_matches_and_finalizes_exact_output_receipt(tmp_path: Path) -> None:
    control_source, treatment_source = tmp_path / "c.zip", tmp_path / "t.zip"
    control_matched, treatment_matched = tmp_path / "cm.zip", tmp_path / "tm.zip"
    _zip(control_source, b"control")
    _zip(treatment_source, b"treatment-payload")
    budget = match_archives(
        control_source=control_source,
        treatment_source=treatment_source,
        control_matched=control_matched,
        treatment_matched=treatment_matched,
    )
    assert control_matched.stat().st_size == treatment_matched.stat().st_size

    output_dirs = [tmp_path / name for name in ("cs", "cm", "ts", "tm")]
    for output in output_dirs:
        output.mkdir()
        (output / "000000.png").write_bytes(b"same-output")
    control_tree_sha = tree_manifest_sha256(
        {"000000.png": file_sha256(output_dirs[0] / "000000.png")}
    )
    treatment_tree_sha = tree_manifest_sha256(
        {"000000.png": file_sha256(output_dirs[2] / "000000.png")}
    )
    receipt = finalize_transfer(
        control_matched=control_matched,
        treatment_matched=treatment_matched,
        equal_budget=budget,
        control_source_output=output_dirs[0],
        control_matched_output=output_dirs[1],
        treatment_source_output=output_dirs[2],
        treatment_matched_output=output_dirs[3],
        control_measurement=_measurement(
            family="legacy_fourier_ab_control",
            archive=control_matched,
            d_seg=0.004,
            output_tree_sha256=control_tree_sha,
        ),
        treatment_measurement=_measurement(
            family="literal_polar_curvelet",
            archive=treatment_matched,
            d_seg=0.003,
            output_tree_sha256=treatment_tree_sha,
            basis_program_sha256=PROGRAM_SHA,
        ),
        basis_program_sha256=PROGRAM_SHA,
    )
    assert receipt["verdict"]["status"] == "MEASURED_TRANSFER_FORMULATION_INSTANCE"
    assert receipt["verdict"]["family_verdict"] == "OPEN"
    assert receipt["pointer_delta"] == "ZERO"
    assert len(receipt["receipt_sha256"]) == 64


def test_finalize_refuses_changed_matched_output_tree(tmp_path: Path) -> None:
    control_source, treatment_source = tmp_path / "c.zip", tmp_path / "t.zip"
    control_matched, treatment_matched = tmp_path / "cm.zip", tmp_path / "tm.zip"
    _zip(control_source, b"control")
    _zip(treatment_source, b"treatment")
    budget = match_archives(
        control_source=control_source,
        treatment_source=treatment_source,
        control_matched=control_matched,
        treatment_matched=treatment_matched,
    )
    output_dirs = [tmp_path / name for name in ("cs", "cm", "ts", "tm")]
    for output in output_dirs:
        output.mkdir()
        (output / "000000.png").write_bytes(b"same-output")
    (output_dirs[3] / "000000.png").write_bytes(b"changed")
    with pytest.raises(ArchiveBudgetError, match="inflated outputs changed"):
        finalize_transfer(
            control_matched=control_matched,
            treatment_matched=treatment_matched,
            equal_budget=budget,
            control_source_output=output_dirs[0],
            control_matched_output=output_dirs[1],
            treatment_source_output=output_dirs[2],
            treatment_matched_output=output_dirs[3],
            control_measurement={},
            treatment_measurement={},
            basis_program_sha256=PROGRAM_SHA,
        )
