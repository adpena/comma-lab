# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV archive-bound runtime packages."""

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchiveError,
    pack_snerv_archive_snar2,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (
    expected_receiver_output_bytes_from_metadata,
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.inflate import CAMERA_HW
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    build_snerv_receiver_archive_proof,
)


def test_snerv_archive_bound_package_runs_receiver_proof(tmp_path: Path) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )

    package = export_snerv_archive_bound_candidate_package(
        packet=archive.packet,
        output_dir=tmp_path,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=120,
    )
    proof_payload = package["receiver_proof"]
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    manifest = row["runtime_adapter_manifest"]

    assert (tmp_path / "archive.zip").is_file()
    assert (tmp_path / "submission" / "inflate.sh").stat().st_mode & stat.S_IXUSR
    assert not (tmp_path / "submission" / "0.bin").exists()
    assert proof_payload["runtime_consumption_proof_passed"] is True
    assert proof_payload["receiver_contract_satisfied"] is True
    assert proof_payload["archive_dir_for_inflate"].endswith(
        "archive_extracted_for_receiver_proof"
    )
    assert proof_payload["receiver_output_retained"] is False
    assert proof_payload["receiver_output_bytes"] == 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3
    assert package["score_claim"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "snerv_packet_not_full_600_pairs" in row["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in row["blockers"]
    assert manifest["archive_zip_payload_only"] is True
    assert manifest["archive_member_name"] == "x"
    assert manifest["runtime_source_outside_archive_zip"] is True
    assert manifest["upstream_evaluate_rate_uses_archive_zip_stat_only"] is True

    with zipfile.ZipFile(tmp_path / "archive.zip") as zf:
        assert zf.namelist() == ["x"]
        assert zf.read("x") == archive.packet


def test_snerv_archive_bound_package_overwrite_refreshes_extract_dir(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        wavelet="haar",
        hw=(16, 24),
        full_frame_packet=True,
    )

    first = export_snerv_archive_bound_candidate_package(
        packet=archive.packet,
        output_dir=tmp_path,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=120,
    )
    assert first["receiver_proof"]["runtime_consumption_proof_passed"] is True

    with pytest.raises(ValueError, match="archive extraction target already exists"):
        export_snerv_archive_bound_candidate_package(
            packet=archive.packet,
            output_dir=tmp_path,
            retain_receiver_output=False,
            receiver_proof_timeout_seconds=120,
        )

    rerun = export_snerv_archive_bound_candidate_package(
        packet=archive.packet,
        output_dir=tmp_path,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=120,
        allow_overwrite=True,
    )

    assert rerun["receiver_proof"]["runtime_consumption_proof_passed"] is True
    assert (tmp_path / "archive_extracted_for_receiver_proof" / "x").read_bytes() == (
        archive.packet
    )


def test_haar_package_uses_numpy_receiver_dwt_without_pywavelets_blocker(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        wavelet="haar",
        hw=(16, 24),
        full_frame_packet=True,
    )

    package = export_snerv_archive_bound_candidate_package(
        packet=archive.packet,
        output_dir=tmp_path,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=120,
    )
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    manifest = row["runtime_adapter_manifest"]

    assert package["receiver_proof"]["runtime_consumption_proof_passed"] is True
    assert manifest["receiver_dwt_dependency"] == "numpy_haar_no_pywavelets"
    assert manifest["archive_packet_wire_format"] == "snar1"
    assert "pywavelets_runtime_dependency_not_contest_proven" not in row["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in row["blockers"]


def test_snar2_package_uses_snar2_transform_and_receiver_contract(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        wavelet="haar",
        hw=(16, 24),
        full_frame_packet=True,
    )
    decoded = unpack_snerv_archive(archive.packet)
    snar2 = pack_snerv_archive_snar2(
        metadata_payload=decoded.sections["metadata_payload"],
        lf_payload=decoded.sections["lf_payload"],
        decoder_payload=decoded.sections["decoder_payload"],
        step_map_packet=decoded.sections["step_map_packet"],
        metadata=decoded.metadata,
    )

    package = export_snerv_archive_bound_candidate_package(
        packet=snar2.packet,
        output_dir=tmp_path,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=120,
    )
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    manifest = row["runtime_adapter_manifest"]

    assert package["receiver_proof"]["runtime_consumption_proof_passed"] is True
    assert row["target_kind"] == "snerv_inverse_steg_snar2_archive"
    assert row["archive_native_transform_kind"] == "snerv_inverse_steg_snar2_archive"
    assert (
        row["receiver_contract_kind"]
        == "snerv_inverse_steg_snar2_inflate_decode_only_receiver"
    )
    assert manifest["archive_packet_schema"] == "snerv_inverse_steg_archive.snar2.v1"
    assert manifest["archive_packet_wire_format"] == "snar2"
    assert "snar1" not in row["target_kind"]
    assert "snar1" not in row["receiver_contract_kind"]


def test_expected_receiver_output_bytes_requires_contest_grouping() -> None:
    assert expected_receiver_output_bytes_from_metadata({"n_pairs": 2}) == (
        2 * 2 * 3 * CAMERA_HW[0] * CAMERA_HW[1]
    )
    with pytest.raises(SnervArchiveError, match="n_pairs"):
        expected_receiver_output_bytes_from_metadata({})
    with pytest.raises(SnervArchiveError, match="frames_per_pair"):
        expected_receiver_output_bytes_from_metadata(
            {"n_pairs": 1, "frames_per_pair": 1}
        )
    with pytest.raises(SnervArchiveError, match="channels"):
        expected_receiver_output_bytes_from_metadata({"n_pairs": 1, "channels": 1})


def test_receiver_proof_cli_package_option(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    cli_path = repo_root / "tools/prove_snerv_receiver_archive.py"
    report_path = tmp_path / "proof.json"
    packet_path = tmp_path / "proof.snar"
    package_dir = tmp_path / "package"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prove_snerv_receiver_archive_for_package_test",
        cli_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main(
        [
            "--bins",
            "4",
            "--levels",
            "1",
            "--height",
            "16",
            "--width",
            "24",
            "--out",
            str(report_path),
            "--packet-out",
            str(packet_path),
            "--package-dir",
            str(package_dir),
            "--package-timeout-seconds",
            "120",
        ]
    )
    payload = json.loads(report_path.read_text())

    assert rc == 0
    assert payload["full_frame_packet"] is True
    assert payload["runtime_package"]["receiver_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert (package_dir / "archive.zip").is_file()
    assert packet_path.is_file()
