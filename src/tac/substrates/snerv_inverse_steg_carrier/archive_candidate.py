# SPDX-License-Identifier: MIT
"""Archive-bound SNeRV receiver package helpers.

This bridge turns receiver-visible SNAR packets into a contest-shaped
``archive.zip`` plus generated ``inflate.sh`` proof. It is deliberately
false-authority: runtime consumption is proved, exact score authority is not.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.submission_archive import (
    MINIMAL_SINGLE_MEMBER_NAME,
    safe_extract_zip,
    write_minimal_single_member_archive,
)
from tac.substrates._shared.pact_nerv_full_main import (
    write_contest_runtime,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SNERV_ARCHIVE_SCHEMA,
    SNERV_ARCHIVE_SCHEMA_V2,
    SnervArchiveError,
    unpack_snerv_archive,
)

SNERV_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "snerv_inverse_steg_archive_bound_adapter_package.v1"
)
SNERV_RECEIVER_PROOF_SCHEMA = "snerv_inverse_steg_generated_receiver_proof.v1"
SNERV_ARCHIVE_BOUND_ADAPTER_ID = "snerv_inverse_steg_archive_export"
SNERV_ARCHIVE_CANDIDATE_FAMILY = "snerv_inverse_steg_carrier"
SNERV_ARCHIVE_TRANSFORM_KIND = "snerv_inverse_steg_snar1_archive"
SNERV_ARCHIVE_TRANSFORM_KIND_V2 = "snerv_inverse_steg_snar2_archive"
SNERV_ARCHIVE_WIRE_FORMAT = "snar1"
SNERV_ARCHIVE_WIRE_FORMAT_V2 = "snar2"
SNERV_RECEIVER_CONTRACT_KIND = "snerv_inverse_steg_snar1_inflate_decode_only_receiver"
SNERV_RECEIVER_CONTRACT_KIND_V2 = (
    "snerv_inverse_steg_snar2_inflate_decode_only_receiver"
)
CAMERA_HW: tuple[int, int] = (874, 1164)


def write_snerv_contest_runtime(
    submission_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Write the self-contained SNeRV inflate wrapper and receiver modules."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[4]
    write_contest_runtime(
        Path(submission_dir),
        substrate_pkg_name="snerv_inverse_steg_carrier",
        repo_root=root,
        runtime_module_files=(
            "inflate.py",
            "archive.py",
            "carrier.py",
            "dwt.py",
            "lf_payload_codec.py",
            "official_hfr.py",
            "official_mfu.py",
            "official_tub.py",
        ),
        vendor_extra_tac_subpackages=(
            ("analysis", ("snerv_step_map_coder.py",)),
            ("codec", ("receiver_integer_plane_codec.py",)),
            ("substrates._shared", ("int_stream_codec.py",)),
        ),
    )


def export_snerv_archive_bound_candidate_package(
    *,
    packet: bytes,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    mlx_triage_argv: Sequence[str] | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Emit packet custody, payload-only ``archive.zip``, runtime, and proof."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[4]
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_bytes = bytes(packet)
    decoded = unpack_snerv_archive(bin_bytes)
    transform_kind, receiver_contract_kind = _archive_contract_for_schema(decoded.schema)
    expected_output_bytes = expected_receiver_output_bytes_from_metadata(
        decoded.metadata
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_snerv_contest_runtime(submission_dir, repo_root=root)

    archive_zip_path = out_dir / "archive.zip"
    archive_build = write_minimal_single_member_archive(
        archive_zip_path,
        bin_bytes,
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes = archive_zip_path.stat().st_size
    archive_extract_dir = out_dir / "archive_extracted_for_receiver_proof"
    if allow_overwrite and archive_extract_dir.exists():
        shutil.rmtree(archive_extract_dir)
    safe_extract_zip(archive_zip_path, archive_extract_dir)

    receiver_dwt_dependency = _receiver_dwt_dependency_mode(decoded.metadata)
    extra_blockers = ["paired_contest_cpu_cuda_auth_eval_missing"]
    if receiver_dwt_dependency != "numpy_haar_no_pywavelets":
        extra_blockers.append("pywavelets_runtime_dependency_not_contest_proven")
    if int(decoded.metadata.get("n_pairs", 0)) != 600:
        extra_blockers.insert(0, "snerv_packet_not_full_600_pairs")

    return emit_archive_bound_candidate_runtime_package(
        adapter_id=SNERV_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=SNERV_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="snerv_inverse_steg",
        transform_kind=transform_kind,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        archive_dir_for_inflate=archive_extract_dir,
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind=receiver_contract_kind,
        proof_schema=SNERV_RECEIVER_PROOF_SCHEMA,
        proof_filename="snerv_inverse_steg_receiver_proof.json",
        candidate_label="snerv_inverse_steg",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=expected_output_bytes,
        retain_receiver_output=retain_receiver_output,
        timeout_seconds=receiver_proof_timeout_seconds,
        runtime_adapter_manifest_extra={
            "schema": "snerv_inverse_steg_runtime_adapter_manifest.v1",
            "archive_packet_schema": decoded.schema,
            "archive_packet_wire_format": _archive_wire_format_for_schema(
                decoded.schema
            ),
            "archive_packet_sha256": decoded.packet_sha256,
            "n_pairs": int(decoded.metadata["n_pairs"]),
            "frames_per_pair": int(decoded.metadata.get("frames_per_pair", 2)),
            "channels": int(decoded.metadata.get("channels", 3)),
            "camera_hw": list(CAMERA_HW),
            "numpy_receiver_raw_writer": True,
            "receiver_dwt_dependency": receiver_dwt_dependency,
            "archive_zip_build": archive_build,
            "archive_member_name": archive_build["member_name"],
            "archive_zip_payload_only": True,
            "runtime_source_outside_archive_zip": True,
            "upstream_evaluate_rate_uses_archive_zip_stat_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        candidate_row_schema="snerv_inverse_steg_archive_bound_candidate_row.v1",
        wrapper_schema=SNERV_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        input_artifacts=[
            archive_zip_path.as_posix(),
            bin_path.as_posix(),
            (archive_extract_dir / MINIMAL_SINGLE_MEMBER_NAME).as_posix(),
        ],
        extra_blockers=extra_blockers,
        mlx_triage_argv=mlx_triage_argv,
    )


def expected_receiver_output_bytes_from_metadata(metadata: dict[str, Any]) -> int:
    """Return expected contest raw byte count from SNAR1 grouping metadata."""

    try:
        n_pairs = int(metadata["n_pairs"])
    except KeyError as exc:
        raise SnervArchiveError("metadata missing 'n_pairs' for contest inflate") from exc
    frames_per_pair = int(metadata.get("frames_per_pair", 2))
    channels = int(metadata.get("channels", 3))
    if n_pairs < 1:
        raise SnervArchiveError(f"n_pairs must be >= 1, got {n_pairs}")
    if frames_per_pair != 2:
        raise SnervArchiveError(
            f"contest inflate expects frames_per_pair=2, got {frames_per_pair}"
        )
    if channels != 3:
        raise SnervArchiveError(f"contest inflate expects channels=3, got {channels}")
    return int(n_pairs) * frames_per_pair * channels * int(CAMERA_HW[0]) * int(CAMERA_HW[1])


def _receiver_dwt_dependency_mode(metadata: dict[str, Any]) -> str:
    wavelet = str(metadata.get("wavelet", "")).strip().lower()
    if wavelet in {"haar", "db1"}:
        return "numpy_haar_no_pywavelets"
    return "pywavelets_required_for_non_haar"


def _archive_contract_for_schema(schema: str) -> tuple[str, str]:
    if schema == SNERV_ARCHIVE_SCHEMA:
        return SNERV_ARCHIVE_TRANSFORM_KIND, SNERV_RECEIVER_CONTRACT_KIND
    if schema == SNERV_ARCHIVE_SCHEMA_V2:
        return SNERV_ARCHIVE_TRANSFORM_KIND_V2, SNERV_RECEIVER_CONTRACT_KIND_V2
    raise SnervArchiveError(f"unsupported SNeRV archive packet schema: {schema!r}")


def _archive_wire_format_for_schema(schema: str) -> str:
    if schema == SNERV_ARCHIVE_SCHEMA:
        return SNERV_ARCHIVE_WIRE_FORMAT
    if schema == SNERV_ARCHIVE_SCHEMA_V2:
        return SNERV_ARCHIVE_WIRE_FORMAT_V2
    raise SnervArchiveError(f"unsupported SNeRV archive packet schema: {schema!r}")


__all__ = [
    "SNERV_ARCHIVE_BOUND_ADAPTER_ID",
    "SNERV_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "SNERV_ARCHIVE_CANDIDATE_FAMILY",
    "SNERV_ARCHIVE_TRANSFORM_KIND",
    "SNERV_ARCHIVE_TRANSFORM_KIND_V2",
    "SNERV_ARCHIVE_WIRE_FORMAT",
    "SNERV_ARCHIVE_WIRE_FORMAT_V2",
    "SNERV_RECEIVER_CONTRACT_KIND",
    "SNERV_RECEIVER_CONTRACT_KIND_V2",
    "SNERV_RECEIVER_PROOF_SCHEMA",
    "expected_receiver_output_bytes_from_metadata",
    "export_snerv_archive_bound_candidate_package",
    "write_snerv_contest_runtime",
]
