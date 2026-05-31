# SPDX-License-Identifier: MIT
"""Archive-bound runtime bridge for Z8 hierarchical predictive-coding candidates.

Z8 is the terminal member of the provenance-clean predictive stack. This module
turns its Z8HPC1 ``0.bin`` bytes into the same shared archive-bound candidate
surface used by Z7 Mamba-2, DreamerV3 RSSM, Z6-v2, and Z4: deterministic
archive.zip, packaged decode-only runtime, receiver proof, replay metadata, and
false-authority candidate row.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    Z8CanonicalQuadrupleBinding,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
)
from tac.substrates.z8_hierarchical_predictive_coding.inflate import CONTEST_RAW_BYTES

Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "z8_hierarchical_predictive_coding_archive_bound_adapter_package.v1"
)
Z8_HPC_RECEIVER_PROOF_SCHEMA = (
    "z8_hierarchical_predictive_coding_generated_receiver_proof.v1"
)
Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID = "z8_hierarchical_predictive_coding_archive_export"
Z8_HPC_ARCHIVE_CANDIDATE_FAMILY = "z8_hierarchical_predictive_coding"
Z8_HPC_ARCHIVE_TRANSFORM_KIND = (
    "z8_hpc1_mamba_mallat_dreamer_wyner_ziv_predictive_coding_archive"
)

Z8_RUNTIME_MODULE_FILES: tuple[str, ...] = (
    "archive.py",
    "binding_contract.py",
    "canonical_quadruple_binding.py",
    "inflate.py",
    "loss.py",
    "mallat_dwt_adapter.py",
    "mamba2_adapter.py",
    "scorer_sensitivity_map.py",
    "wyner_ziv_coder.py",
)
Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("optimization", ("mamba2_predictor.py",)),
    ("symposium_impls", ("daubechies_wavelet_codec.py",)),
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path | None,
) -> tuple[Path, Path]:
    root = Path(repo_root) if repo_root is not None else _repo_root_from_here()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, out_dir


def _write_z8_runtime(
    *,
    submission_dir: Path,
    repo_root: Path,
    bin_bytes: bytes,
) -> None:
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="z8_hierarchical_predictive_coding",
        repo_root=repo_root,
        runtime_module_files=Z8_RUNTIME_MODULE_FILES,
        inflate_import_line=(
            "from tac.substrates.z8_hierarchical_predictive_coding.inflate "
            "import inflate_one_video"
        ),
        vendor_shared_inflate_runtime=True,
        vendor_extra_tac_subpackages=Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)


def export_z8hpc1_archive_bytes(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export already-built Z8HPC1 bytes as a contest-shaped archive."""

    root, out_dir = _resolve_output_dir(output_dir, repo_root=repo_root)
    bin_bytes = bytes(archive_bytes)
    (out_dir / "0.bin").write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    _write_z8_runtime(
        submission_dir=submission_dir,
        repo_root=root,
        bin_bytes=bin_bytes,
    )
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes_count = archive_zip_path.stat().st_size
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=Z8_HPC_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="z8_hpc1",
            transform_kind=Z8_HPC_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes_count,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind=(
                "z8_hpc1_generated_inflate_sh_decode_only_receiver"
            ),
            proof_schema=Z8_HPC_RECEIVER_PROOF_SCHEMA,
            proof_filename="z8_hpc1_receiver_proof.json",
            candidate_label="z8_hpc1",
            expected_receiver_output_bytes=CONTEST_RAW_BYTES,
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "z8_hpc1_runtime_adapter_manifest.v1",
                "predictive_coding_family": (
                    "mamba_mallat_dreamer_wyner_ziv_stack"
                ),
                "runtime_module_files": list(Z8_RUNTIME_MODULE_FILES),
                "vendored_tac_subpackages": [
                    subpkg for subpkg, _files in Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES
                ],
            },
            candidate_row_schema="z8_hpc1_archive_bound_candidate_row.v1",
            wrapper_schema=Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes_count)


def export_z8hpc1_archive_from_canonical_quadruple(
    binding: Z8CanonicalQuadrupleBinding,
    real_pair_rgb_frame_0: np.ndarray,
    real_pair_rgb_frame_1: np.ndarray,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Build Z8HPC1 bytes from canonical quadruple state and export a package."""

    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding,
        real_pair_rgb_frame_0,
        real_pair_rgb_frame_1,
    )
    return export_z8hpc1_archive_bytes(
        archive_bytes,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=emit_archive_bound_candidate_package,
        retain_receiver_proof_output=retain_receiver_proof_output,
        mlx_triage_argv=mlx_triage_argv,
    )


def export_z8hpc1_archive_bound_candidate_package(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Z8HPC1 bytes and emit the shared archive-bound package."""

    archive_zip_path, archive_sha256, archive_bytes_count = export_z8hpc1_archive_bytes(
        archive_bytes,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
    )
    root, out_dir = _resolve_output_dir(output_dir, repo_root=repo_root)
    submission_dir = out_dir / "submission"
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=Z8_HPC_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="z8_hpc1",
        transform_kind=Z8_HPC_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes_count,
        submission_dir=submission_dir,
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="z8_hpc1_generated_inflate_sh_decode_only_receiver",
        proof_schema=Z8_HPC_RECEIVER_PROOF_SCHEMA,
        proof_filename="z8_hpc1_receiver_proof.json",
        candidate_label="z8_hpc1",
        expected_receiver_output_bytes=CONTEST_RAW_BYTES,
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "z8_hpc1_runtime_adapter_manifest.v1",
            "predictive_coding_family": "mamba_mallat_dreamer_wyner_ziv_stack",
            "runtime_module_files": list(Z8_RUNTIME_MODULE_FILES),
            "vendored_tac_subpackages": [
                subpkg for subpkg, _files in Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES
            ],
        },
        candidate_row_schema="z8_hpc1_archive_bound_candidate_row.v1",
        wrapper_schema=Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID",
    "Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "Z8_HPC_ARCHIVE_CANDIDATE_FAMILY",
    "Z8_HPC_ARCHIVE_TRANSFORM_KIND",
    "Z8_HPC_RECEIVER_PROOF_SCHEMA",
    "Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES",
    "Z8_RUNTIME_MODULE_FILES",
    "export_z8hpc1_archive_bound_candidate_package",
    "export_z8hpc1_archive_bytes",
    "export_z8hpc1_archive_from_canonical_quadruple",
]
