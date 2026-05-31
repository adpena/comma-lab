# SPDX-License-Identifier: MIT
"""Archive-bound runtime bridge for Z8 hierarchical predictive-coding candidates.

Z8 is the terminal member of the provenance-clean predictive stack. This module
turns its Z8HPC1 ``0.bin`` bytes into the same shared archive-bound candidate
surface used by Z7 Mamba-2, DreamerV3 RSSM, Z6-v2, and Z4: deterministic
archive.zip, packaged decode-only runtime, receiver proof, replay metadata, and
false-authority candidate row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from tac.substrates.z8_hierarchical_predictive_coding.byte_mutation_proof import (
    probe_z8_archive_distinguishing_feature,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    Z8CanonicalQuadrupleBinding,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
)
from tac.substrates.z8_hierarchical_predictive_coding.inflate import CONTEST_RAW_BYTES
from tac.substrates.z8_hierarchical_predictive_coding.runtime_custody import (
    Z8_HPC_ARCHIVE_CANDIDATE_PIXEL_CONSUMED_SECTIONS,
    Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS,
    build_z8_runtime_custody_contract,
)
from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
    build_runtime_payload_bridge_report,
)

Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "z8_hierarchical_predictive_coding_archive_bound_adapter_package.v1"
)
Z8_HPC_RECEIVER_PROOF_SCHEMA = (
    "z8_hierarchical_predictive_coding_generated_receiver_proof.v1"
)
Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID = "z8_hierarchical_predictive_coding_archive_export"
Z8_HPC_ARCHIVE_CANDIDATE_FAMILY = "z8_hierarchical_predictive_coding"
Z8_HPC_ARCHIVE_TRANSFORM_KIND = (
    "z8_hpc1_mallat_wavelet_plus_mamba_wyner_ziv_top_ll_pixel_driver_partial_predictive_stack_archive"
)
Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS = (
    Z8_HPC_ARCHIVE_CANDIDATE_PIXEL_CONSUMED_SECTIONS
)

Z8_RUNTIME_MODULE_FILES: tuple[str, ...] = (
    "archive.py",
    "binding_contract.py",
    "canonical_quadruple_binding.py",
    "inflate.py",
    "loss.py",
    "mallat_dwt_adapter.py",
    "mamba2_adapter.py",
    "runtime_custody.py",
    "runtime_payload_bridge.py",
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


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


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


def _z8_runtime_adapter_manifest_extra(
    *,
    byte_mutation_proof: Mapping[str, Any] | None = None,
    runtime_payload_bridge_report: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    section_pixel_consumption_proofs: dict[str, Any] = {}
    predictive_stack_pixel_consumption: dict[str, Any] = {}
    if byte_mutation_proof is not None:
        raw_section_proofs = byte_mutation_proof.get(
            "section_pixel_consumption_proofs"
        )
        if isinstance(raw_section_proofs, Mapping):
            section_pixel_consumption_proofs = dict(raw_section_proofs)
        raw_predictive_stack = byte_mutation_proof.get(
            "predictive_stack_pixel_consumption"
        )
        if isinstance(raw_predictive_stack, Mapping):
            predictive_stack_pixel_consumption = dict(raw_predictive_stack)
    mamba_dreamer_wz_proven = (
        byte_mutation_proof is not None
        and byte_mutation_proof.get(
            "mamba_dreamer_wyner_ziv_pixel_consumption_proven"
        )
        is True
    )
    if mamba_dreamer_wz_proven:
        runtime_consumption_status = "pixel_consumed_proven"
    elif byte_mutation_proof is not None:
        runtime_consumption_status = (
            "archive_bound_custody_only_distinguishing_feature_mutation"
            "_proved_not_pixel_consumed"
        )
    else:
        runtime_consumption_status = (
            "archive_bound_custody_only_pending_distinguishing_feature"
            "_mutation_proofs"
        )
    manifest = {
        "schema": "z8_hpc1_runtime_adapter_manifest.v1",
        "runtime_custody_contract": build_z8_runtime_custody_contract(
            source="z8_hpc1_runtime_adapter_manifest",
            section_name_style="candidate_manifest",
            archive_bound_candidate_package_emitted=True,
            trained_mlx_renderer_archive_export_ready=False,
        ),
        "predictive_coding_family": (
            "z8_hpc1_mallat_wavelet_plus_wyner_ziv_top_state_pixel_consumed"
            "_predictive_stack_custody"
        ),
        "pixel_consumed_archive_sections": list(Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS),
        "stack_custody_not_yet_pixel_consumed_sections": list(
            Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS
        ),
        "section_pixel_consumption_proofs": section_pixel_consumption_proofs,
        "predictive_stack_pixel_consumption": predictive_stack_pixel_consumption,
        "full_stack_pixel_consumption_claim": False,
        "predictive_stack_runtime_consumption_status": (
            predictive_stack_pixel_consumption.get("status")
            or runtime_consumption_status
        ),
        "mamba_dreamer_wyner_ziv_runtime_consumption_status": (
            runtime_consumption_status
        ),
        "runtime_module_files": list(Z8_RUNTIME_MODULE_FILES),
        "vendored_tac_subpackages": [
            subpkg for subpkg, _files in Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES
        ],
    }
    if runtime_payload_bridge_report is not None:
        report_path = str(runtime_payload_bridge_report.get("report_path") or "")
        if report_path and repo_root is not None:
            report_path = _repo_relative(Path(report_path), repo_root)
        manifest["runtime_payload_bridge_report"] = {
            "schema": runtime_payload_bridge_report.get("schema"),
            "report_path": report_path or None,
            "wyner_ziv_top_state_decode_ready": bool(
                runtime_payload_bridge_report.get(
                    "wyner_ziv_top_state_decode_ready"
                )
            ),
            "wyner_ziv_top_state_count": runtime_payload_bridge_report.get(
                "wyner_ziv_top_state_count"
            ),
            "wyner_ziv_top_state_sha256": runtime_payload_bridge_report.get(
                "wyner_ziv_top_state_sha256"
            ),
            "state_to_pixel_projection_ready": bool(
                runtime_payload_bridge_report.get(
                    "state_to_pixel_projection_ready"
                )
            ),
            "pixel_consumption_proven": bool(
                runtime_payload_bridge_report.get("pixel_consumption_proven")
            ),
            "next_required_task": runtime_payload_bridge_report.get(
                "next_required_task"
            ),
        }
    if byte_mutation_proof is not None:
        proof_path = str(byte_mutation_proof.get("proof_path") or "")
        if proof_path and repo_root is not None:
            proof_path = _repo_relative(Path(proof_path), repo_root)
        manifest["byte_mutation_consumption_proof"] = {
            "schema": byte_mutation_proof.get("schema_version"),
            "proof_path": proof_path or None,
            "distinguishing_feature": byte_mutation_proof.get(
                "distinguishing_feature"
            ),
            "distinguishing_feature_consumed": bool(
                byte_mutation_proof.get("distinguishing_feature_consumed")
            ),
            "pixel_consumed_sections": list(
                byte_mutation_proof.get("pixel_consumed_sections") or []
            ),
            "custody_only_sections": list(
                byte_mutation_proof.get("custody_only_sections") or []
            ),
            "mamba_dreamer_wyner_ziv_pixel_consumption_proven": bool(
                byte_mutation_proof.get(
                    "mamba_dreamer_wyner_ziv_pixel_consumption_proven"
                )
            ),
            "wyner_ziv_payload_pixel_consumption_proven": bool(
                byte_mutation_proof.get("wyner_ziv_payload_pixel_consumption_proven")
            ),
            "stack_context_pixel_consumption_proven": bool(
                byte_mutation_proof.get("stack_context_pixel_consumption_proven")
            ),
            "section_pixel_consumption_proofs": section_pixel_consumption_proofs,
            "predictive_stack_pixel_consumption": predictive_stack_pixel_consumption,
        }
    return manifest


def export_z8hpc1_archive_bytes(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    emit_byte_mutation_proof: bool = True,
    emit_runtime_payload_bridge_report: bool = True,
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
    byte_mutation_proof: dict[str, Any] | None = None
    byte_mutation_proof_path = out_dir / "z8_hpc1_byte_mutation_proof.json"
    runtime_payload_bridge_report_path = (
        out_dir / "z8_hpc1_runtime_payload_bridge_report.json"
    )
    runtime_payload_bridge_report = (
        build_runtime_payload_bridge_report(
            bin_bytes,
            report_out=runtime_payload_bridge_report_path,
        )
        if emit_runtime_payload_bridge_report
        else None
    )
    if emit_byte_mutation_proof:
        byte_mutation_proof = probe_z8_archive_distinguishing_feature(
            out_dir / "0.bin",
            proof_out=byte_mutation_proof_path,
        )
        if byte_mutation_proof.get("distinguishing_feature_consumed") is not True:
            raise RuntimeError(
                "Z8HPC1 archive-bound export failed byte-mutation proof: "
                "wavelet_blob did not produce pixel-changing output"
            )
    if emit_archive_bound_candidate_package:
        input_artifacts = [
            _repo_relative(archive_zip_path, root),
            _repo_relative(submission_dir / "0.bin", root),
        ]
        if byte_mutation_proof is not None:
            input_artifacts.append(_repo_relative(byte_mutation_proof_path, root))
        if runtime_payload_bridge_report is not None:
            input_artifacts.append(
                _repo_relative(runtime_payload_bridge_report_path, root)
            )
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
            runtime_adapter_manifest_extra=_z8_runtime_adapter_manifest_extra(
                byte_mutation_proof=byte_mutation_proof,
                runtime_payload_bridge_report=runtime_payload_bridge_report,
                repo_root=root,
            ),
            candidate_row_schema="z8_hpc1_archive_bound_candidate_row.v1",
            wrapper_schema=Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            input_artifacts=input_artifacts,
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
    emit_byte_mutation_proof: bool = True,
    emit_runtime_payload_bridge_report: bool = True,
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
        emit_byte_mutation_proof=emit_byte_mutation_proof,
        emit_runtime_payload_bridge_report=emit_runtime_payload_bridge_report,
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
    byte_mutation_proof_path = out_dir / "z8_hpc1_byte_mutation_proof.json"
    runtime_payload_bridge_report_path = (
        out_dir / "z8_hpc1_runtime_payload_bridge_report.json"
    )
    byte_mutation_proof = (
        json.loads(byte_mutation_proof_path.read_text(encoding="utf-8"))
        if byte_mutation_proof_path.is_file()
        else None
    )
    runtime_payload_bridge_report = (
        json.loads(runtime_payload_bridge_report_path.read_text(encoding="utf-8"))
        if runtime_payload_bridge_report_path.is_file()
        else None
    )
    input_artifacts = [
        _repo_relative(archive_zip_path, root),
        _repo_relative(submission_dir / "0.bin", root),
    ]
    if byte_mutation_proof_path.is_file():
        input_artifacts.append(_repo_relative(byte_mutation_proof_path, root))
    if runtime_payload_bridge_report_path.is_file():
        input_artifacts.append(
            _repo_relative(runtime_payload_bridge_report_path, root)
        )
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
        runtime_adapter_manifest_extra=_z8_runtime_adapter_manifest_extra(
            byte_mutation_proof=byte_mutation_proof,
            runtime_payload_bridge_report=runtime_payload_bridge_report,
            repo_root=root,
        ),
        candidate_row_schema="z8_hpc1_archive_bound_candidate_row.v1",
        wrapper_schema=Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        input_artifacts=input_artifacts,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID",
    "Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "Z8_HPC_ARCHIVE_CANDIDATE_FAMILY",
    "Z8_HPC_ARCHIVE_TRANSFORM_KIND",
    "Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS",
    "Z8_HPC_RECEIVER_PROOF_SCHEMA",
    "Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS",
    "Z8_RUNTIME_EXTRA_TAC_SUBPACKAGES",
    "Z8_RUNTIME_MODULE_FILES",
    "export_z8hpc1_archive_bound_candidate_package",
    "export_z8hpc1_archive_bytes",
    "export_z8hpc1_archive_from_canonical_quadruple",
]
