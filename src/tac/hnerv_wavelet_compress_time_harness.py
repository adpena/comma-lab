"""Fail-closed compress-time harness scaffold for HNeRV WR01 wavelet atoms.

The harness records deterministic source custody and alpha-wavelet configuration
for a future train/select/apply path. It does not train atoms, emit candidate
archives, dispatch jobs, or claim scores.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    DEFAULT_WAVELET_SECTION,
    WAVELET_AUDIT_SECTIONS,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.repo_io import json_line, json_text, write_json

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_wavelet_compress_time_harness.build_wavelet_compress_time_harness"


class HnervWaveletCompressTimeHarnessError(ValueError):
    """Raised when a WR01 compress-time harness input is invalid."""


@dataclasses.dataclass(frozen=True)
class HnervWaveletCompressTimeConfig:
    """Deterministic WR01 compress-time harness configuration."""

    seed: int = 0
    target_sections: tuple[str, ...] = (DEFAULT_WAVELET_SECTION,)
    atom_budget: int = 32
    block_size: int = 64
    quant_step: float = 1.0
    train_steps: int = 0
    selection_mode: str = "deterministic_topk_placeholder"
    apply_mode: str = "not_implemented_fail_closed"

    def normalized(self) -> dict[str, Any]:
        """Return the canonical JSON config payload."""

        target_sections = _normalize_sections(self.target_sections)
        seed = int(self.seed)
        atom_budget = int(self.atom_budget)
        block_size = int(self.block_size)
        quant_step = float(self.quant_step)
        train_steps = int(self.train_steps)
        if seed < 0:
            raise HnervWaveletCompressTimeHarnessError(f"seed must be nonnegative, got {seed}")
        if atom_budget <= 0:
            raise HnervWaveletCompressTimeHarnessError(f"atom_budget must be positive, got {atom_budget}")
        if block_size < 2 or block_size & (block_size - 1):
            raise HnervWaveletCompressTimeHarnessError(f"block_size must be a power of two >= 2, got {block_size}")
        if quant_step <= 0:
            raise HnervWaveletCompressTimeHarnessError(f"quant_step must be positive, got {quant_step}")
        if train_steps < 0:
            raise HnervWaveletCompressTimeHarnessError(f"train_steps must be nonnegative, got {train_steps}")
        selection_mode = str(self.selection_mode)
        apply_mode = str(self.apply_mode)
        if not selection_mode:
            raise HnervWaveletCompressTimeHarnessError("selection_mode must be nonempty")
        if not apply_mode:
            raise HnervWaveletCompressTimeHarnessError("apply_mode must be nonempty")
        return {
            "seed": seed,
            "target_sections": list(target_sections),
            "atom_budget": atom_budget,
            "block_size": block_size,
            "quant_step": quant_step,
            "train_steps": train_steps,
            "selection_mode": selection_mode,
            "apply_mode": apply_mode,
            "rng_state_mutated": False,
            "determinism_contract": (
                "seed is captured for future atom training/selection; this "
                "scaffold does not mutate Python, NumPy, torch, or CUDA RNG state"
            ),
        }


def build_wavelet_compress_time_harness(
    *,
    source_archive: str | Path,
    source_label: str,
    output_dir: str | Path | None = None,
    target_sections: Sequence[str] = (DEFAULT_WAVELET_SECTION,),
    seed: int = 0,
    atom_budget: int = 32,
    block_size: int = 64,
    quant_step: float = 1.0,
    train_steps: int = 0,
    expected_source_archive_sha256: str | None = None,
    expected_source_archive_bytes: int | None = None,
    exact_decode_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic fail-closed WR01 compress-time harness manifest."""

    config = HnervWaveletCompressTimeConfig(
        seed=seed,
        target_sections=tuple(target_sections),
        atom_budget=atom_budget,
        block_size=block_size,
        quant_step=quant_step,
        train_steps=train_steps,
    ).normalized()
    config_sha256 = sha256_bytes(json_line(config).encode("utf-8"))
    archive = read_strict_single_member_zip(source_archive)
    _validate_expected_source_identity(
        archive_sha256=archive.archive_sha256,
        archive_bytes=archive.archive_bytes,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_source_archive_bytes=expected_source_archive_bytes,
    )
    packed = parse_ff_packed_brotli_hnerv(archive.payload)
    source_custody = {
        "source_label": str(source_label),
        "source_archive_path": str(source_archive),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_member_bytes": archive.member_bytes,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "source_archive_custody_mode": (
            "operator_expected_archive_identity_verified"
            if expected_source_archive_sha256 is not None
            else "measured_source_archive_identity"
        ),
        "score_claim": False,
    }
    section_records, section_blockers = _source_section_records(
        packed=packed,
        target_sections=tuple(config["target_sections"]),
    )
    decode_validation = _decode_validation_placeholder(
        exact_decode_validation=exact_decode_validation,
        source_archive_sha256=archive.archive_sha256,
        target_sections=tuple(config["target_sections"]),
    )
    pipeline_blockers = [
        "compress_time_atom_training_not_implemented",
        "compress_time_atom_selection_not_implemented",
        "compress_time_atom_apply_not_implemented",
        "wr01_candidate_archive_not_emitted",
        "requires_archive_manifest_preflight",
        "requires_exact_cuda_auth_eval",
    ]
    blockers = [
        *section_blockers,
        *decode_validation["blockers"],
        *pipeline_blockers,
    ]
    input_manifest = {
        "schema": "hnerv_wavelet_compress_time_input.v1",
        "source": source_custody,
        "config": config,
        "config_sha256": config_sha256,
        "wr01_contract": {
            "sidechannel_magic": "WR01",
            "wrapper_magic": "0xfa",
            "target_sections": list(config["target_sections"]),
            "score_claim": False,
        },
    }
    output_manifest = {
        "schema": "hnerv_wavelet_compress_time_output.v1",
        "trained_atoms_manifest_path": None,
        "selected_atoms_manifest_path": None,
        "wavelet_sidechannel_archive_path": None,
        "applied_candidate_archive_path": None,
        "candidate_archive_sha256": None,
        "candidate_archive_bytes": None,
        "decode_validation": decode_validation,
        "score_claim": False,
        "dispatch_attempted": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "config_sha256": config_sha256,
        "input_manifest": input_manifest,
        "output_manifest": output_manifest,
        "source_sections": section_records,
        "ready_for_compress_time_training": False,
        "ready_for_wavelet_sidechannel_candidate": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "dispatch_blockers": [
            "compress_time_harness_scaffold_only",
            "no_dispatch_without_lane_claim",
            "requires_trained_wr01_atom_manifest",
            "requires_selected_wr01_atom_manifest",
            "requires_wr01_runtime_apply_path",
            "requires_exact_decode_validation_manifest",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
        "candidate_required_proof": [
            "source_archive_sha256",
            "source_archive_bytes",
            "trained_atoms_manifest_sha256",
            "selected_atoms_manifest_sha256",
            "wavelet_sidechannel_sha256",
            "applied_candidate_archive_sha256",
            "decode_validation_exact",
            "archive_manifest_preflight",
            "contest_cuda_auth_eval_json",
        ],
    }
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "hnerv_wavelet_compress_time_harness.json"
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(manifest)
        write_json(manifest_path, manifest)
    else:
        manifest["manifest_path"] = None
        manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(manifest)
    return manifest


def _source_section_records(
    *,
    packed: Any,
    target_sections: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for section_name in target_sections:
        if section_name not in WAVELET_AUDIT_SECTIONS:
            blockers.append(f"unknown_wr01_target_section:{section_name}")
            continue
        section_bytes = packed.section_bytes(section_name)
        record: dict[str, Any] = {
            "section_name": section_name,
            "section_bytes": len(section_bytes),
            "section_sha256": sha256_bytes(section_bytes),
            "score_claim": False,
        }
        if section_name == "packed_header_ff_len24":
            record.update(
                {
                    "transform_domain": "section_wire_bytes",
                    "raw_bytes": len(section_bytes),
                    "raw_sha256": sha256_bytes(section_bytes),
                    "decode_probe_status": "not_required_for_header_section",
                }
            )
        else:
            try:
                raw = brotli.decompress(section_bytes)
            except brotli.error:
                record.update(
                    {
                        "transform_domain": "brotli_decompressed_section",
                        "raw_bytes": None,
                        "raw_sha256": None,
                        "decode_probe_status": "brotli_decompress_failed",
                    }
                )
                blockers.append(f"source_section_brotli_decode_failed:{section_name}")
            else:
                record.update(
                    {
                        "transform_domain": "brotli_decompressed_section",
                        "raw_bytes": len(raw),
                        "raw_sha256": sha256_bytes(raw),
                        "decode_probe_status": "local_brotli_decode_only_not_exact_validation",
                    }
                )
        records.append(record)
    if not records:
        blockers.append("no_source_sections_recorded")
    return records, blockers


def _decode_validation_placeholder(
    *,
    exact_decode_validation: Mapping[str, Any] | None,
    source_archive_sha256: str,
    target_sections: tuple[str, ...],
) -> dict[str, Any]:
    if exact_decode_validation is None:
        return {
            "status": "blocked",
            "ready": False,
            "exact_validation_available": False,
            "fail_closed": True,
            "validation_manifest_sha256": None,
            "validated_source_archive_sha256": None,
            "validated_sections": [],
            "blockers": ["missing_exact_decode_validation_manifest"],
            "score_claim": False,
        }

    blockers: list[str] = []
    validation = dict(exact_decode_validation)
    validation_sha256 = sha256_bytes(json_line(validation).encode("utf-8"))
    validation_source = str(
        validation.get("source_archive_sha256") or validation.get("validated_source_archive_sha256") or ""
    )
    if validation_source != source_archive_sha256:
        blockers.append("exact_decode_validation_source_archive_sha256_mismatch")
    if validation.get("score_claim") is not False:
        blockers.append("exact_decode_validation_score_claim_not_false")
    exact_flag = bool(
        validation.get("exact_decode_validation") is True
        or validation.get("validation_mode") == "exact_wr01_decode_validation"
    )
    if not exact_flag:
        blockers.append("exact_decode_validation_flag_missing")
    validated_sections = [str(section) for section in validation.get("validated_sections") or []]
    for section_name in target_sections:
        if section_name not in validated_sections:
            blockers.append(f"exact_decode_validation_missing_section:{section_name}")
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "exact_validation_available": not blockers,
        "fail_closed": bool(blockers),
        "validation_manifest_sha256": validation_sha256,
        "validated_source_archive_sha256": validation_source or None,
        "validated_sections": validated_sections,
        "blockers": blockers,
        "score_claim": False,
    }


def _normalize_sections(sections: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(section) for section in sections))
    if not normalized:
        raise HnervWaveletCompressTimeHarnessError("at least one target section is required")
    if any(not section for section in normalized):
        raise HnervWaveletCompressTimeHarnessError("target section names must be nonempty")
    return normalized


def _validate_expected_source_identity(
    *,
    archive_sha256: str,
    archive_bytes: int,
    expected_source_archive_sha256: str | None,
    expected_source_archive_bytes: int | None,
) -> None:
    if (expected_source_archive_sha256 is None) != (expected_source_archive_bytes is None):
        raise HnervWaveletCompressTimeHarnessError(
            "expected_source_archive_sha256 and expected_source_archive_bytes must be provided together"
        )
    if expected_source_archive_sha256 is None:
        return
    if len(expected_source_archive_sha256) != 64 or not all(
        c in "0123456789abcdef" for c in expected_source_archive_sha256
    ):
        raise HnervWaveletCompressTimeHarnessError(
            "expected_source_archive_sha256 must be a 64-char lowercase hex digest"
        )
    if expected_source_archive_sha256 != archive_sha256:
        raise HnervWaveletCompressTimeHarnessError(
            "expected_source_archive_sha256 does not match measured source archive"
        )
    if int(expected_source_archive_bytes) != archive_bytes:
        raise HnervWaveletCompressTimeHarnessError(
            "expected_source_archive_bytes does not match measured source archive"
        )


def _manifest_sha256_excluding_self(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256_excluding_self"}
    return sha256_bytes(json_text(payload).encode("utf-8"))


__all__ = [
    "SCHEMA_VERSION",
    "TOOL",
    "HnervWaveletCompressTimeConfig",
    "HnervWaveletCompressTimeHarnessError",
    "build_wavelet_compress_time_harness",
]
