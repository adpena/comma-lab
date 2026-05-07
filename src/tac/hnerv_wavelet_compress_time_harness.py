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
ATOM_PLAN_SCHEMA = "hnerv_wavelet_compress_time_atom_plan.v1"
ATOM_PLAN_FILENAME = "hnerv_wavelet_compress_time_atom_plan.json"
SELECTED_ATOMS_SCHEMA = "hnerv_wavelet_compress_time_selected_atoms.v1"
SELECTED_ATOMS_FILENAME = "hnerv_wavelet_compress_time_selected_atoms.json"
WR01_FIXED_ATOM_WIRE_BYTES = 17


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
    atom_plan_path = None if output_dir is None else Path(output_dir) / ATOM_PLAN_FILENAME
    selected_atoms_path = None if output_dir is None else Path(output_dir) / SELECTED_ATOMS_FILENAME
    atom_plan_manifest, atom_plan_blockers = _build_atom_plan_manifest(
        source_archive=source_archive,
        source_label=source_label,
        archive=archive,
        packed=packed,
        config=config,
        config_sha256=config_sha256,
        manifest_path=atom_plan_path,
    )
    selected_atoms_manifest, selected_atoms_blockers = _build_selected_atoms_manifest(
        source_archive=source_archive,
        source_label=source_label,
        archive=archive,
        atom_plan=atom_plan_manifest,
        manifest_path=selected_atoms_path,
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
        *atom_plan_blockers,
        *selected_atoms_blockers,
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
        "atom_plan_manifest_path": str(atom_plan_path) if atom_plan_path is not None else None,
        "atom_plan_manifest_sha256": atom_plan_manifest["manifest_sha256_excluding_self"],
        "atom_plan_schema": ATOM_PLAN_SCHEMA,
        "selected_atoms_manifest_path": str(selected_atoms_path) if selected_atoms_path is not None else None,
        "selected_atoms_manifest_sha256": selected_atoms_manifest["manifest_sha256_excluding_self"],
        "selected_atoms_schema": SELECTED_ATOMS_SCHEMA,
        "trained_atoms_manifest_path": None,
        "wavelet_sidechannel_archive_path": None,
        "applied_candidate_archive_path": None,
        "candidate_archive_sha256": None,
        "candidate_archive_bytes": None,
        "decode_validation": decode_validation,
        "apply_readiness": {
            "status": "blocked",
            "ready": False,
            "selected_atoms_manifest_path": (
                str(selected_atoms_path) if selected_atoms_path is not None else None
            ),
            "selected_atoms_manifest_sha256": selected_atoms_manifest[
                "manifest_sha256_excluding_self"
            ],
            "blockers": list(selected_atoms_manifest["dispatch_blockers"]),
            "score_claim": False,
            "dispatch_attempted": False,
        },
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
        "atom_plan_manifest": atom_plan_manifest,
        "selected_atoms_manifest": selected_atoms_manifest,
        "source_sections": section_records,
        "ready_for_compress_time_training": False,
        "ready_for_atom_plan_review": not atom_plan_blockers,
        "ready_for_selected_atom_review": not selected_atoms_blockers,
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
        write_json(atom_plan_path, atom_plan_manifest)
        write_json(selected_atoms_path, selected_atoms_manifest)
        manifest_path = output_root / "hnerv_wavelet_compress_time_harness.json"
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(manifest)
        write_json(manifest_path, manifest)
    else:
        manifest["manifest_path"] = None
        manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(manifest)
    return manifest


def _build_atom_plan_manifest(
    *,
    source_archive: str | Path,
    source_label: str,
    archive: Any,
    packed: Any,
    config: Mapping[str, Any],
    config_sha256: str,
    manifest_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    section_spans = _packed_section_spans(packed)
    target_sections = tuple(str(section) for section in config["target_sections"])
    sections: list[dict[str, Any]] = []
    global_atom_ids: list[str] = []
    total_emitted_atoms = 0
    total_candidate_atoms_before_budget = 0
    for section_name in target_sections:
        if section_name not in WAVELET_AUDIT_SECTIONS:
            blockers.append(f"atom_plan_unknown_wr01_target_section:{section_name}")
            continue
        section_bytes = packed.section_bytes(section_name)
        raw, transform_domain, raw_blockers = _atom_plan_transform_bytes(
            section_name=section_name,
            section_bytes=section_bytes,
        )
        blockers.extend(raw_blockers)
        if raw is None:
            sections.append(
                _blocked_atom_plan_section(
                    section_name=section_name,
                    section_bytes=section_bytes,
                    section_span=section_spans[section_name],
                    transform_domain=transform_domain,
                    blockers=raw_blockers,
                )
            )
            continue
        atoms, candidate_count = _enumerate_candidate_atoms(
            raw=raw,
            section_name=section_name,
            source_section_sha256=sha256_bytes(section_bytes),
            atom_budget=int(config["atom_budget"]),
            block_size=int(config["block_size"]),
            quant_step=float(config["quant_step"]),
            train_steps=int(config["train_steps"]),
        )
        if not atoms:
            blockers.append(f"atom_plan_no_atoms_within_budget:{section_name}")
        atom_ids = [str(atom["atom_id"]) for atom in atoms]
        global_atom_ids.extend(atom_ids)
        total_emitted_atoms += len(atoms)
        total_candidate_atoms_before_budget += candidate_count
        sections.append(
            {
                "section_name": section_name,
                "source_section_bytes": len(section_bytes),
                "source_section_sha256": sha256_bytes(section_bytes),
                "source_section_payload_span": {
                    "start": section_spans[section_name][0],
                    "end": section_spans[section_name][1],
                },
                "transform_domain": transform_domain,
                "raw_bytes": len(raw),
                "raw_sha256": sha256_bytes(raw),
                "block_size": int(config["block_size"]),
                "quant_step": float(config["quant_step"]),
                "candidate_atoms_before_budget": candidate_count,
                "emitted_atom_count": len(atoms),
                "candidate_atom_ids": atom_ids,
                "candidate_atoms": atoms,
                "score_claim": False,
                "dispatch_attempted": False,
            }
        )
    if not sections:
        blockers.append("atom_plan_no_sections_recorded")
    atom_plan = {
        "schema": ATOM_PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_path": str(source_archive),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "config_sha256": config_sha256,
        "budget": {
            "atom_count_budget_per_section": int(config["atom_budget"]),
            "atom_wire_bytes_budget_per_atom": WR01_FIXED_ATOM_WIRE_BYTES,
            "atom_wire_bytes_budget_per_section": int(config["atom_budget"]) * WR01_FIXED_ATOM_WIRE_BYTES,
            "train_step_budget": int(config["train_steps"]),
            "block_size": int(config["block_size"]),
            "quant_step": float(config["quant_step"]),
            "budget_is_dispatch_clearance": False,
        },
        "target_sections": list(target_sections),
        "total_candidate_atoms_before_budget": total_candidate_atoms_before_budget,
        "total_emitted_atom_count": total_emitted_atoms,
        "atom_ids": global_atom_ids,
        "sections": sections,
        "ready_for_train_select_apply": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            *blockers,
            "compress_time_atom_training_not_implemented",
            "compress_time_atom_selection_not_implemented",
            "compress_time_atom_apply_not_implemented",
            "requires_exact_decode_validation_manifest",
            "requires_wr01_runtime_apply_path",
        ],
        "dispatch_blockers": [
            "atom_plan_manifest_is_planning_only",
            "no_candidate_archive_emitted",
            "requires_exact_decode_validation_manifest",
            "requires_wr01_runtime_apply_path",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
        "candidate_required_proof": [
            "source_archive_sha256",
            "source_archive_bytes",
            "source_section_sha256",
            "raw_sha256",
            "trained_atoms_manifest_sha256",
            "selected_atoms_manifest_sha256",
            "runtime_apply_manifest_sha256",
            "exact_decode_validation_manifest_sha256",
        ],
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }
    atom_plan["plan_sha256"] = _atom_plan_digest(atom_plan)
    atom_plan["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(atom_plan)
    return atom_plan, blockers


def _build_selected_atoms_manifest(
    *,
    source_archive: str | Path,
    source_label: str,
    archive: Any,
    atom_plan: Mapping[str, Any],
    manifest_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    structural_blockers: list[str] = []
    selected_sections: list[dict[str, Any]] = []
    selected_atom_ids: list[str] = []
    seen_atom_ids: set[str] = set()
    total_estimated_wire_bytes = 0
    plan_manifest_sha256 = str(atom_plan.get("manifest_sha256_excluding_self") or "")
    if len(plan_manifest_sha256) != 64:
        structural_blockers.append("selected_atoms_missing_atom_plan_manifest_sha256")
    plan_sha256 = str(atom_plan.get("plan_sha256") or "")
    if len(plan_sha256) != 64:
        structural_blockers.append("selected_atoms_missing_atom_plan_sha256")

    for section in atom_plan.get("sections") or []:
        if not isinstance(section, Mapping):
            structural_blockers.append("selected_atoms_invalid_atom_plan_section")
            continue
        section_name = str(section.get("section_name") or "")
        section_blockers: list[str] = []
        candidates = [
            dict(atom)
            for atom in section.get("candidate_atoms") or []
            if isinstance(atom, Mapping)
        ]
        if not section_name:
            section_blockers.append("selected_atoms_section_name_missing")
        if not candidates:
            section_blockers.append(f"selected_atoms_no_candidate_atoms:{section_name or 'unknown'}")

        atoms: list[dict[str, Any]] = []
        for selection_rank, atom in enumerate(sorted(candidates, key=_atom_selection_sort_key), start=1):
            atom_id = str(atom.get("atom_id") or "")
            if not atom_id:
                section_blockers.append(f"selected_atoms_candidate_missing_atom_id:{section_name}")
                continue
            if atom_id in seen_atom_ids:
                section_blockers.append(f"selected_atoms_duplicate_atom_id:{atom_id}")
                continue
            seen_atom_ids.add(atom_id)
            atom_wire_bytes = int(atom.get("atom_wire_bytes_budget") or WR01_FIXED_ATOM_WIRE_BYTES)
            selected_atom = {
                "atom_id": atom_id,
                "section_name": section_name,
                "source_section_sha256": section.get("source_section_sha256"),
                "raw_sha256": section.get("raw_sha256"),
                "raw_offset": int(atom.get("raw_offset", 0)),
                "raw_end": int(atom.get("raw_end", 0)),
                "level": int(atom.get("level", 0)),
                "coefficient_index": int(atom.get("coefficient_index", 0)),
                "coefficient_quantized": int(atom.get("coefficient_quantized", 0)),
                "abs_coefficient_quantized": int(atom.get("abs_coefficient_quantized", 0)),
                "budget_rank": int(atom.get("budget_rank", selection_rank)),
                "selection_rank": selection_rank,
                "selection_policy": "atom_plan_budget_rank_order",
                "estimated_wire_bytes": atom_wire_bytes,
                "byte_delta": atom_wire_bytes,
                "budget_status": "selected_from_atom_plan_for_apply_readiness_scaffold",
                "selected_for_apply_readiness": True,
                "selected_for_runtime_apply": False,
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_blockers": [
                    "selected_atom_is_planning_only",
                    "requires_trained_wr01_atom_manifest",
                    "requires_wr01_runtime_apply_path",
                    "requires_exact_decode_validation_manifest",
                    "no_candidate_archive_emitted",
                ],
            }
            atoms.append(selected_atom)
            selected_atom_ids.append(atom_id)
            total_estimated_wire_bytes += atom_wire_bytes

        structural_blockers.extend(section_blockers)
        selected_sections.append(
            {
                "section_name": section_name,
                "source_section_bytes": section.get("source_section_bytes"),
                "source_section_sha256": section.get("source_section_sha256"),
                "source_section_payload_span": section.get("source_section_payload_span"),
                "transform_domain": section.get("transform_domain"),
                "raw_bytes": section.get("raw_bytes"),
                "raw_sha256": section.get("raw_sha256"),
                "selected_atom_count": len(atoms),
                "selected_atom_ids": [str(atom["atom_id"]) for atom in atoms],
                "estimated_wire_bytes": sum(int(atom["estimated_wire_bytes"]) for atom in atoms),
                "atoms": atoms,
                "selected_atoms": atoms,
                "ready_for_runtime_apply": False,
                "ready_for_archive_preflight": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [
                    *section_blockers,
                    "selected_atoms_section_is_planning_only",
                    "requires_wr01_runtime_apply_path",
                    "requires_exact_decode_validation_manifest",
                ],
                "score_claim": False,
                "dispatch_attempted": False,
            }
        )

    if not selected_sections:
        structural_blockers.append("selected_atoms_no_sections_recorded")
    if not selected_atom_ids:
        structural_blockers.append("selected_atoms_no_atoms_selected")

    blocking_status = [
        "selected_atoms_manifest_is_planning_only",
        "requires_trained_wr01_atom_manifest",
        "requires_wr01_runtime_apply_path",
        "requires_exact_decode_validation_manifest",
        "no_candidate_archive_emitted",
        "requires_archive_manifest_preflight",
        "requires_exact_cuda_auth_eval",
    ]
    selected_manifest = {
        "schema": SELECTED_ATOMS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_path": str(source_archive),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "atom_plan_schema": str(atom_plan.get("schema") or ""),
        "atom_plan_manifest_path": str(atom_plan.get("manifest_path") or ""),
        "atom_plan_manifest_sha256": plan_manifest_sha256,
        "atom_plan_sha256": plan_sha256,
        "selection_mode": "atom_plan_budget_rank_order",
        "selection_input": {
            "atom_plan_atom_count": len(atom_plan.get("atom_ids") or []),
            "atom_plan_atom_ids": list(atom_plan.get("atom_ids") or []),
            "budget_is_dispatch_clearance": False,
        },
        "target_sections": list(atom_plan.get("target_sections") or []),
        "total_selected_atom_count": len(selected_atom_ids),
        "selected_atom_ids": selected_atom_ids,
        "estimated_total_atom_wire_bytes": total_estimated_wire_bytes,
        "sections": selected_sections,
        "ready_for_selected_atom_review": not structural_blockers,
        "ready_for_train_select_apply": False,
        "ready_for_runtime_apply": False,
        "ready_for_wavelet_sidechannel_candidate": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "trained_atoms_manifest_path": None,
        "trained_atoms_manifest_sha256": None,
        "wavelet_sidechannel_archive_path": None,
        "applied_candidate_archive_path": None,
        "candidate_archive_path": None,
        "candidate_archive_sha256": None,
        "candidate_archive_bytes": None,
        "blockers": [*structural_blockers, *blocking_status],
        "dispatch_blockers": blocking_status,
        "candidate_required_proof": [
            "source_archive_sha256",
            "source_archive_bytes",
            "atom_plan_manifest_sha256",
            "selected_atoms_manifest_sha256",
            "trained_atoms_manifest_sha256",
            "wr01_runtime_apply_manifest_sha256",
            "exact_decode_validation_manifest_sha256",
            "archive_manifest_preflight",
            "contest_cuda_auth_eval_json",
        ],
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }
    selected_manifest["selection_sha256"] = _selected_atoms_digest(selected_manifest)
    selected_manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(
        selected_manifest
    )
    return selected_manifest, structural_blockers


def _blocked_atom_plan_section(
    *,
    section_name: str,
    section_bytes: bytes,
    section_span: tuple[int, int],
    transform_domain: str,
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "section_name": section_name,
        "source_section_bytes": len(section_bytes),
        "source_section_sha256": sha256_bytes(section_bytes),
        "source_section_payload_span": {"start": section_span[0], "end": section_span[1]},
        "transform_domain": transform_domain,
        "raw_bytes": None,
        "raw_sha256": None,
        "candidate_atoms_before_budget": 0,
        "emitted_atom_count": 0,
        "candidate_atom_ids": [],
        "candidate_atoms": [],
        "blockers": list(blockers),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _atom_plan_transform_bytes(
    *,
    section_name: str,
    section_bytes: bytes,
) -> tuple[bytes | None, str, list[str]]:
    if section_name == "packed_header_ff_len24":
        return section_bytes, "section_wire_bytes", []
    try:
        return brotli.decompress(section_bytes), "brotli_decompressed_section", []
    except brotli.error:
        return None, "brotli_decompressed_section", [f"atom_plan_brotli_decode_failed:{section_name}"]


def _packed_section_spans(packed: Any) -> dict[str, tuple[int, int]]:
    decoder_start = len(packed.header)
    decoder_end = decoder_start + len(packed.decoder_packed_brotli)
    latent_end = decoder_end + len(packed.latents_and_sidecar_brotli)
    return {
        "packed_header_ff_len24": (0, decoder_start),
        "decoder_packed_brotli": (decoder_start, decoder_end),
        "latents_and_sidecar_brotli": (decoder_end, latent_end),
    }


def _enumerate_candidate_atoms(
    *,
    raw: bytes,
    section_name: str,
    source_section_sha256: str,
    atom_budget: int,
    block_size: int,
    quant_step: float,
    train_steps: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    for block_start in range(0, len(raw), block_size):
        chunk = raw[block_start : block_start + block_size]
        if len(chunk) < block_size:
            continue
        current = [float(byte) - 128.0 for byte in chunk]
        width = block_size
        level = 0
        while width >= 2:
            next_level: list[float] = []
            for pair_index in range(width // 2):
                left = current[2 * pair_index]
                right = current[2 * pair_index + 1]
                avg = (left + right) * 0.5
                detail = (left - right) * 0.5
                quantized = round(detail / quant_step)
                if quantized:
                    support = 1 << (level + 1)
                    support_start = block_start + pair_index * support
                    support_end = support_start + support
                    if support_end <= len(raw):
                        candidates.append(
                            {
                                "section_name": section_name,
                                "raw_offset": support_start,
                                "raw_end": support_end,
                                "level": level,
                                "coefficient_index": pair_index,
                                "coefficient_quantized": quantized,
                                "abs_coefficient_quantized": abs(quantized),
                            }
                        )
                next_level.append(avg)
            current = next_level
            width //= 2
            level += 1
    candidates.sort(
        key=lambda atom: (
            -int(atom["abs_coefficient_quantized"]),
            int(atom["raw_offset"]),
            int(atom["level"]),
            int(atom["coefficient_index"]),
            int(atom["coefficient_quantized"]),
        )
    )
    emitted: list[dict[str, Any]] = []
    for budget_rank, atom in enumerate(candidates[:atom_budget], start=1):
        atom_id = _atom_id(
            section_name=section_name,
            source_section_sha256=source_section_sha256,
            atom=atom,
        )
        emitted.append(
            {
                **atom,
                "atom_id": atom_id,
                "budget_rank": budget_rank,
                "budget_status": "inside_planning_budget_not_selected_for_apply",
                "atom_wire_bytes_budget": WR01_FIXED_ATOM_WIRE_BYTES,
                "train_step_budget": train_steps,
                "selected_for_apply": False,
                "score_claim": False,
                "dispatch_attempted": False,
            }
        )
    return emitted, len(candidates)


def _atom_id(*, section_name: str, source_section_sha256: str, atom: Mapping[str, Any]) -> str:
    section_slug = section_name.replace("_", "-")
    payload = (
        f"{section_name}:{source_section_sha256}:"
        f"{atom['raw_offset']}:{atom['raw_end']}:"
        f"{atom['level']}:{atom['coefficient_index']}:{atom['coefficient_quantized']}"
    )
    return f"wr01-{section_slug}-{sha256_bytes(payload.encode('utf-8'))[:16]}"


def _atom_plan_digest(atom_plan: Mapping[str, Any]) -> str:
    payload = {
        "schema": atom_plan.get("schema"),
        "source_archive_sha256": atom_plan.get("source_archive_sha256"),
        "config_sha256": atom_plan.get("config_sha256"),
        "atom_ids": atom_plan.get("atom_ids"),
        "sections": [
            {
                "section_name": section.get("section_name"),
                "source_section_sha256": section.get("source_section_sha256"),
                "raw_sha256": section.get("raw_sha256"),
                "candidate_atoms": section.get("candidate_atoms"),
            }
            for section in atom_plan.get("sections") or []
            if isinstance(section, Mapping)
        ],
    }
    return sha256_bytes(json_line(payload).encode("utf-8"))


def _atom_selection_sort_key(atom: Mapping[str, Any]) -> tuple[int, int, int, int, int, str]:
    return (
        int(atom.get("budget_rank", 1 << 30)),
        int(atom.get("raw_offset", 0)),
        int(atom.get("level", 0)),
        int(atom.get("coefficient_index", 0)),
        int(atom.get("coefficient_quantized", 0)),
        str(atom.get("atom_id") or ""),
    )


def _selected_atoms_digest(selected_manifest: Mapping[str, Any]) -> str:
    payload = {
        "schema": selected_manifest.get("schema"),
        "source_archive_sha256": selected_manifest.get("source_archive_sha256"),
        "source_archive_bytes": selected_manifest.get("source_archive_bytes"),
        "atom_plan_manifest_sha256": selected_manifest.get("atom_plan_manifest_sha256"),
        "selected_atom_ids": selected_manifest.get("selected_atom_ids"),
        "sections": [
            {
                "section_name": section.get("section_name"),
                "source_section_sha256": section.get("source_section_sha256"),
                "raw_sha256": section.get("raw_sha256"),
                "atoms": section.get("atoms"),
            }
            for section in selected_manifest.get("sections") or []
            if isinstance(section, Mapping)
        ],
    }
    return sha256_bytes(json_line(payload).encode("utf-8"))


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
    "ATOM_PLAN_FILENAME",
    "ATOM_PLAN_SCHEMA",
    "SCHEMA_VERSION",
    "SELECTED_ATOMS_FILENAME",
    "SELECTED_ATOMS_SCHEMA",
    "TOOL",
    "WR01_FIXED_ATOM_WIRE_BYTES",
    "HnervWaveletCompressTimeConfig",
    "HnervWaveletCompressTimeHarnessError",
    "build_wavelet_compress_time_harness",
]
