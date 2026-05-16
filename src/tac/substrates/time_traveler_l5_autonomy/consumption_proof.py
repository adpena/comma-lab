# SPDX-License-Identifier: MIT
"""Artifact producer for TT5L temporal side-info consumption proofs.

The L5-v2 staircase gate requires more than unit tests saying side-info is
wired. This module emits a small, byte-stable JSON proof that a deterministic
TT5L packet:

* parses its temporal side-info section,
* changes inflated raw output when side-info bytes change,
* changes inflated raw output when AC-state bytes change,
* records all archive, section, runtime, and output hashes needed by the
  L5-v2 gate validator.

It intentionally does not claim score movement. AC-state is still residual
calibration in TT5L v1; replacing it with a real range/ANS decoder remains a
separate score-lowering task.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.substrates.time_traveler_l5_autonomy.architecture import (
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_FMT,
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
    parse_archive,
    parse_tt5l_archive_bytes,
    serialize_deterministic_state_dict_blob,
)
from tac.substrates.time_traveler_l5_autonomy.inflate import inflate_one_video

TT5L_SIDEINFO_CONSUMPTION_PROOF_SCHEMA = "tt5l_sideinfo_consumption_proof_v1"
TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID = (
    "tt5l_byte_closed_temporal_sideinfo_consumption_v1"
)
TT5L_SIDEINFO_CONSUMPTION_GATE_ID = "byte_closed_temporal_sideinfo_consumption"
TT5L_SIDEINFO_CONSUMPTION_DEFAULT_ARTIFACT = (
    ".omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json"
)
TT5L_SIDEINFO_CONSUMPTION_DEFAULT_MANIFEST = (
    ".omx/research/tt5l_sideinfo_consumption_manifest_20260516_codex.json"
)
TT5L_SIDEINFO_CONSUMPTION_DEFAULT_WORK_DIR = (
    "experiments/results/time_traveler_l5_v2/"
    "tt5l_sideinfo_consumption_proof_20260516_codex"
)
TT5L_CONTEST_PAIR_COUNT = 600
TT5L_CONTEST_FRAME_COUNT = 1200

_RUNTIME_TREE_FILES: tuple[str, ...] = (
    "src/tac/substrates/time_traveler_l5_autonomy/archive.py",
    "src/tac/substrates/time_traveler_l5_autonomy/architecture.py",
    "src/tac/substrates/time_traveler_l5_autonomy/inflate.py",
)


@dataclass(frozen=True)
class TT5LConsumptionProofResult:
    """Paths and payload emitted by the proof builder."""

    proof: dict[str, Any]
    manifest: dict[str, Any]
    proof_path: Path
    manifest_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _runtime_tree_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relpath in _RUNTIME_TREE_FILES:
        path = repo_root / relpath
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_repo_custody_path(
    value: str | Path | None,
    default_value: str,
    *,
    repo_root: Path,
    label: str,
) -> Path:
    raw_path = Path(value or default_value)
    resolved = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside repo root: {resolved}") from exc
    return resolved


def _resolve_existing_repo_path(
    value: str | Path,
    *,
    repo_root: Path,
    label: str,
    expect_dir: bool = False,
) -> Path:
    raw_path = Path(value)
    resolved = raw_path if raw_path.is_absolute() else repo_root / raw_path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside repo root: {resolved}") from exc
    if expect_dir:
        if not resolved.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {resolved}")
    elif not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def _toy_config_and_state_dict() -> tuple[TimeTravelerConfig, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    cfg = TimeTravelerConfig(
        hidden_dim=16,
        num_hidden_layers=2,
        output_height=64,
        output_width=96,
        num_pairs=1,
    )
    substrate = TimeTravelerSubstrate(cfg)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in substrate.state_dict().items()
    }
    return cfg, state_dict


def _toy_meta(cfg: TimeTravelerConfig) -> dict[str, Any]:
    return {
        "coord_dim": cfg.coord_dim,
        "coord_feature_freqs": cfg.coord_feature_freqs,
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "int8_scale": 64.0,
        "markov_transition_band": cfg.markov_transition_band,
    }


def _pack_with_world_blob(
    *,
    world_blob: bytes,
    cfg: TimeTravelerConfig,
    side_info: np.ndarray,
    meta: dict[str, Any],
    ac_state: bytes,
) -> bytes:
    side_blob = bytes(
        brotli.compress(np.ascontiguousarray(side_info).tobytes(), quality=9)
    )
    ac_blob = bytes(brotli.compress(ac_state, quality=9)) if ac_state else b""
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    header = struct.pack(
        TT5L_HEADER_FMT,
        TT5L_MAGIC,
        TT5L_SCHEMA_VERSION,
        cfg.num_pairs,
        cfg.hidden_dim,
        cfg.num_hidden_layers,
        cfg.output_height,
        cfg.output_width,
        cfg.foveation_grid_h,
        cfg.foveation_grid_w,
        cfg.pose_dim,
        cfg.per_pair_side_info_bytes,
        len(world_blob),
        len(side_blob),
        len(ac_blob),
        len(meta_bytes),
    )
    return header + world_blob + side_blob + ac_blob + meta_bytes


def _build_toy_archive(
    *,
    cfg: TimeTravelerConfig,
    state_dict: dict[str, torch.Tensor],
    side_info: np.ndarray,
    ac_state: bytes,
) -> bytes:
    return _pack_with_world_blob(
        world_blob=serialize_deterministic_state_dict_blob(state_dict),
        cfg=cfg,
        side_info=side_info,
        meta=_toy_meta(cfg),
        ac_state=ac_state,
    )


def _repack_with_fixed_world_blob(
    *,
    baseline_archive: bytes,
    cfg: TimeTravelerConfig,
    side_info: np.ndarray,
    meta: dict[str, Any],
    ac_state: bytes,
) -> bytes:
    """Rebuild a TT5L blob while preserving baseline WORLD_MODEL_BLOB bytes."""

    world_offset, world_len = parse_tt5l_archive_bytes(baseline_archive)[
        "world_model_blob"
    ]
    world_blob = baseline_archive[world_offset : world_offset + world_len]
    return _pack_with_world_blob(
        world_blob=world_blob,
        cfg=cfg,
        side_info=side_info,
        meta=meta,
        ac_state=ac_state,
    )


def _archive_section_payload(blob: bytes, section_name: str) -> tuple[int, bytes]:
    sections = parse_tt5l_archive_bytes(blob)
    start, length = sections[section_name]
    return start, blob[start : start + length]


def _differing_offsets_within_section(
    baseline_blob: bytes,
    mutated_blob: bytes,
    *,
    section_name: str,
) -> list[int]:
    baseline_start, baseline_section = _archive_section_payload(
        baseline_blob,
        section_name,
    )
    mutated_start, mutated_section = _archive_section_payload(mutated_blob, section_name)
    if baseline_start != mutated_start:
        return [baseline_start]
    limit = min(len(baseline_section), len(mutated_section))
    offsets = [
        baseline_start + idx
        for idx in range(limit)
        if baseline_section[idx] != mutated_section[idx]
    ]
    if not offsets and len(baseline_section) != len(mutated_section):
        offsets.append(baseline_start + max(0, len(baseline_section) - 1))
    return offsets or [baseline_start]


def _inflate_and_hash(
    archive_bytes: bytes,
    output_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    frames = inflate_one_video(archive_bytes, output_path, device="cpu")
    raw_sha = _sha256_file(output_path)
    return {
        "path": _display_path(output_path, repo_root),
        "frames": frames,
        "bytes": output_path.stat().st_size,
        "sha256": raw_sha,
    }


def _file_list_entries(path: Path) -> tuple[str, ...]:
    entries = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not entries:
        raise ValueError("file_list is empty")
    for entry in entries:
        rel = Path(entry)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"file_list entry must be repo-relative safe path: {entry!r}")
    return entries


def _raw_outputs_aggregate(
    *,
    output_dir: Path,
    file_list_entries: tuple[str, ...],
    repo_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for entry in file_list_entries:
        path = (output_dir / entry).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"output path escapes output_dir: {entry!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"raw output listed in file_list not found: {path}")
        records.append(
            {
                "file_list_entry": entry,
                "path": _display_path(path, repo_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    aggregate_payload = {
        "schema": "tt5l_contest_raw_output_aggregate_v1",
        "files": records,
    }
    return {
        "aggregate_sha256": _sha256_bytes(_canonical_json_bytes(aggregate_payload)),
        "total_frames": len(records),
        "files": records,
    }


def _component_proof(
    *,
    baseline_blob: bytes,
    mutated_blob: bytes,
    baseline_output: dict[str, Any],
    mutated_output: dict[str, Any],
    section_name: str,
    logical_section: str,
    parser_consumed_bytes: bool,
    consumption_mode: str,
    runtime_tree_sha256: str,
) -> dict[str, Any]:
    section_offset, section_payload = _archive_section_payload(baseline_blob, section_name)
    mutated_section_offset, mutated_section_payload = _archive_section_payload(
        mutated_blob,
        section_name,
    )
    changed_offsets = _differing_offsets_within_section(
        baseline_blob,
        mutated_blob,
        section_name=section_name,
    )
    return {
        "section": logical_section,
        "archive_section_name": section_name,
        "consumption_mode": consumption_mode,
        "parser_consumed_bytes": parser_consumed_bytes,
        "output_changed": baseline_output["sha256"] != mutated_output["sha256"],
        "mutated_byte_offsets": changed_offsets,
        "section_offset": section_offset,
        "section_nbytes": len(section_payload),
        "section_sha256": _sha256_bytes(section_payload),
        "mutated_section_offset": mutated_section_offset,
        "mutated_section_nbytes": len(mutated_section_payload),
        "mutated_section_sha256": _sha256_bytes(mutated_section_payload),
        "baseline_archive_sha256": _sha256_bytes(baseline_blob),
        "mutated_archive_sha256": _sha256_bytes(mutated_blob),
        "runtime_tree_sha256": runtime_tree_sha256,
        "baseline_inflate_sha256": baseline_output["sha256"],
        "mutated_inflate_sha256": mutated_output["sha256"],
    }


def build_tt5l_sideinfo_consumption_proof(
    *,
    artifact_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> TT5LConsumptionProofResult:
    """Build and write the byte-closed TT5L side-info consumption proof."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    proof_path = _resolve_repo_custody_path(
        artifact_path,
        TT5L_SIDEINFO_CONSUMPTION_DEFAULT_ARTIFACT,
        repo_root=root,
        label="artifact_path",
    )
    manifest_out = _resolve_repo_custody_path(
        manifest_path,
        TT5L_SIDEINFO_CONSUMPTION_DEFAULT_MANIFEST,
        repo_root=root,
        label="manifest_path",
    )
    work = _resolve_repo_custody_path(
        work_dir,
        TT5L_SIDEINFO_CONSUMPTION_DEFAULT_WORK_DIR,
        repo_root=root,
        label="work_dir",
    )
    work.mkdir(parents=True, exist_ok=True)

    cfg, state_dict = _toy_config_and_state_dict()
    zero_side = np.zeros((1, 45), dtype=np.int8)
    side_mutated = zero_side.copy()
    side_mutated[0, 36:45] = 64
    ac_state_a = bytes([0, 64, 128, 192, 255] * 4)
    ac_state_b = bytes([255, 192, 128, 64, 0] * 4)

    meta = _toy_meta(cfg)
    baseline_archive = _build_toy_archive(
        cfg=cfg,
        state_dict=state_dict,
        side_info=zero_side,
        ac_state=ac_state_a,
    )
    sideinfo_archive = _repack_with_fixed_world_blob(
        baseline_archive=baseline_archive,
        cfg=cfg,
        side_info=side_mutated,
        meta=meta,
        ac_state=ac_state_a,
    )
    ac_state_archive = _repack_with_fixed_world_blob(
        baseline_archive=baseline_archive,
        cfg=cfg,
        side_info=side_mutated,
        meta=meta,
        ac_state=ac_state_b,
    )

    for name, blob in (
        ("baseline_0.bin", baseline_archive),
        ("sideinfo_mutated_0.bin", sideinfo_archive),
        ("ac_state_mutated_0.bin", ac_state_archive),
    ):
        (work / name).write_bytes(blob)

    baseline_output = _inflate_and_hash(
        baseline_archive,
        work / "baseline.raw",
        repo_root=root,
    )
    sideinfo_output = _inflate_and_hash(
        sideinfo_archive,
        work / "sideinfo_mutated.raw",
        repo_root=root,
    )
    ac_state_output = _inflate_and_hash(
        ac_state_archive,
        work / "ac_state_mutated.raw",
        repo_root=root,
    )

    parsed_baseline = parse_archive(baseline_archive)
    parsed_sideinfo = parse_archive(sideinfo_archive)
    parsed_ac_state = parse_archive(ac_state_archive)
    runtime_sha = _runtime_tree_sha256(root)

    sideinfo_proof = _component_proof(
        baseline_blob=baseline_archive,
        mutated_blob=sideinfo_archive,
        baseline_output=baseline_output,
        mutated_output=sideinfo_output,
        section_name="per_pair_side_info_blob",
        logical_section="tt5l_temporal_sideinfo",
        parser_consumed_bytes=bool(
            np.array_equal(parsed_baseline.per_pair_side_info, zero_side)
            and np.array_equal(parsed_sideinfo.per_pair_side_info, side_mutated)
        ),
        consumption_mode="per_pair_residual_decode",
        runtime_tree_sha256=runtime_sha,
    )
    ac_state_proof = _component_proof(
        baseline_blob=sideinfo_archive,
        mutated_blob=ac_state_archive,
        baseline_output=sideinfo_output,
        mutated_output=ac_state_output,
        section_name="ac_state_blob",
        logical_section="tt5l_ac_state_residual_calibration",
        parser_consumed_bytes=(
            parsed_sideinfo.ac_state == ac_state_a
            and parsed_ac_state.ac_state == ac_state_b
        ),
        consumption_mode="residual_calibration_not_entropy_decode",
        runtime_tree_sha256=runtime_sha,
    )

    output_records = {
        "baseline": baseline_output,
        "sideinfo_mutated": sideinfo_output,
        "ac_state_mutated": ac_state_output,
    }
    aggregate_sha = _sha256_bytes(_canonical_json_bytes(output_records))
    manifest = {
        "schema": "tt5l_sideinfo_consumption_inflated_outputs_manifest_v1",
        "aggregate_sha256": aggregate_sha,
        "raw_output_aggregate_sha256": aggregate_sha,
        "outputs": output_records,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    proof = {
        "schema": TT5L_SIDEINFO_CONSUMPTION_PROOF_SCHEMA,
        "gate_id": TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
        "predicate_id": TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID,
        "predicate_passed": bool(
            sideinfo_proof["parser_consumed_bytes"]
            and sideinfo_proof["output_changed"]
            and ac_state_proof["parser_consumed_bytes"]
            and ac_state_proof["output_changed"]
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "proof_scope": (
            "local_no_gpu_toy_tt5l_archive_parser_and_inflate_consumption_only"
        ),
        "byte_mutation_proof": {
            **sideinfo_proof,
            "inflated_outputs_manifest_path": _display_path(manifest_out, root),
            "inflated_raw_output_aggregate_sha256": aggregate_sha,
            "inflate_command": (
                "src/tac/substrates/time_traveler_l5_autonomy/inflate.sh "
                "<archive_dir> <output_dir> <file_list>"
            ),
        },
        "component_proofs": {
            "per_pair_side_info": sideinfo_proof,
            "ac_state": ac_state_proof,
        },
        "ac_state_status": (
            "consumed_as_residual_calibration_not_real_range_or_ans_entropy_decoder"
        ),
        "runtime_tree_files": list(_RUNTIME_TREE_FILES),
        "runtime_tree_sha256": runtime_sha,
        "artifact_work_dir": _display_path(work, root),
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_bytes(_canonical_json_bytes(manifest))
    proof_path.write_bytes(_canonical_json_bytes(proof))
    return TT5LConsumptionProofResult(
        proof=proof,
        manifest=manifest,
        proof_path=proof_path,
        manifest_path=manifest_out,
    )


def build_tt5l_contest_full_frame_sideinfo_consumption_proof(
    *,
    baseline_archive_path: str | Path,
    mutated_archive_path: str | Path,
    baseline_output_dir: str | Path,
    mutated_output_dir: str | Path,
    file_list_path: str | Path,
    artifact_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> TT5LConsumptionProofResult:
    """Build the real L5-v2 gate artifact from contest-shaped TT5L outputs.

    This is the operational counterpart to the local toy proof above. It does
    not run inflate itself; it consumes two already-inflated TT5L output trees
    plus their shared contest file list, verifies the TT5L side-info section
    changed and parses, and emits the contest-full-frame proof shape accepted by
    the L5 v2 staircase gate.
    """

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    baseline_archive = _resolve_existing_repo_path(
        baseline_archive_path,
        repo_root=root,
        label="baseline_archive_path",
    )
    mutated_archive = _resolve_existing_repo_path(
        mutated_archive_path,
        repo_root=root,
        label="mutated_archive_path",
    )
    baseline_outputs = _resolve_existing_repo_path(
        baseline_output_dir,
        repo_root=root,
        label="baseline_output_dir",
        expect_dir=True,
    )
    mutated_outputs = _resolve_existing_repo_path(
        mutated_output_dir,
        repo_root=root,
        label="mutated_output_dir",
        expect_dir=True,
    )
    file_list = _resolve_existing_repo_path(
        file_list_path,
        repo_root=root,
        label="file_list_path",
    )
    proof_path = _resolve_repo_custody_path(
        artifact_path,
        str(artifact_path),
        repo_root=root,
        label="artifact_path",
    )
    manifest_out = _resolve_repo_custody_path(
        manifest_path,
        str(manifest_path),
        repo_root=root,
        label="manifest_path",
    )

    entries = _file_list_entries(file_list)
    baseline_blob = baseline_archive.read_bytes()
    mutated_blob = mutated_archive.read_bytes()
    parsed_baseline = parse_archive(baseline_blob)
    parsed_mutated = parse_archive(mutated_blob)
    baseline_aggregate = _raw_outputs_aggregate(
        output_dir=baseline_outputs,
        file_list_entries=entries,
        repo_root=root,
    )
    mutated_aggregate = _raw_outputs_aggregate(
        output_dir=mutated_outputs,
        file_list_entries=entries,
        repo_root=root,
    )
    runtime_sha = _runtime_tree_sha256(root)
    pair_aggregate_payload = {
        "baseline_raw_output_aggregate_sha256": baseline_aggregate[
            "aggregate_sha256"
        ],
        "mutated_raw_output_aggregate_sha256": mutated_aggregate[
            "aggregate_sha256"
        ],
        "file_list_sha256": _sha256_file(file_list),
    }
    pair_aggregate_sha = _sha256_bytes(_canonical_json_bytes(pair_aggregate_payload))
    n_pairs_hashed = parsed_baseline.num_pairs
    total_frames = len(entries)
    parser_consumed = bool(
        parsed_baseline.num_pairs == parsed_mutated.num_pairs
        and parsed_baseline.per_pair_bytes == parsed_mutated.per_pair_bytes
        and parsed_baseline.per_pair_side_info.shape
        == parsed_mutated.per_pair_side_info.shape
        and not np.array_equal(
            parsed_baseline.per_pair_side_info,
            parsed_mutated.per_pair_side_info,
        )
    )
    output_changed = (
        baseline_aggregate["aggregate_sha256"] != mutated_aggregate["aggregate_sha256"]
    )
    section_offset, section_payload = _archive_section_payload(
        baseline_blob,
        "per_pair_side_info_blob",
    )
    mutated_section_offset, mutated_section_payload = _archive_section_payload(
        mutated_blob,
        "per_pair_side_info_blob",
    )
    byte_mutation_proof = {
        "section": "tt5l_temporal_sideinfo",
        "archive_section_name": "per_pair_side_info_blob",
        "consumption_mode": "contest_full_frame_sideinfo_mutation",
        "parser_consumed_bytes": parser_consumed,
        "output_changed": output_changed,
        "mutated_byte_offsets": _differing_offsets_within_section(
            baseline_blob,
            mutated_blob,
            section_name="per_pair_side_info_blob",
        ),
        "section_offset": section_offset,
        "section_nbytes": len(section_payload),
        "section_sha256": _sha256_bytes(section_payload),
        "mutated_section_offset": mutated_section_offset,
        "mutated_section_nbytes": len(mutated_section_payload),
        "mutated_section_sha256": _sha256_bytes(mutated_section_payload),
        "baseline_archive_path": _display_path(baseline_archive, root),
        "mutated_archive_path": _display_path(mutated_archive, root),
        "baseline_archive_sha256": _sha256_bytes(baseline_blob),
        "mutated_archive_sha256": _sha256_bytes(mutated_blob),
        "runtime_tree_sha256": runtime_sha,
        "baseline_inflate_sha256": baseline_aggregate["aggregate_sha256"],
        "mutated_inflate_sha256": mutated_aggregate["aggregate_sha256"],
        "inflated_outputs_manifest_path": _display_path(manifest_out, root),
        "inflated_raw_output_aggregate_sha256": pair_aggregate_sha,
        "baseline_raw_output_aggregate_sha256": baseline_aggregate[
            "aggregate_sha256"
        ],
        "mutated_raw_output_aggregate_sha256": mutated_aggregate[
            "aggregate_sha256"
        ],
        "source_raw_output_aggregate_sha256": baseline_aggregate[
            "aggregate_sha256"
        ],
        "candidate_raw_output_aggregate_sha256": mutated_aggregate[
            "aggregate_sha256"
        ],
        "file_list_path": _display_path(file_list, root),
        "file_list_sha256": _sha256_file(file_list),
        "n_pairs_hashed": n_pairs_hashed,
        "total_frames": total_frames,
        "inflate_command": (
            "src/tac/substrates/time_traveler_l5_autonomy/inflate.sh "
            "<archive_dir> <output_dir> <file_list>"
        ),
    }
    predicate_passed = bool(
        parser_consumed
        and output_changed
        and n_pairs_hashed == TT5L_CONTEST_PAIR_COUNT
        and total_frames == TT5L_CONTEST_FRAME_COUNT
    )
    manifest = {
        "schema": "tt5l_contest_full_frame_sideinfo_outputs_manifest_v1",
        "raw_output_aggregate_sha256": pair_aggregate_sha,
        "aggregate_sha256": pair_aggregate_sha,
        "baseline_raw_output_aggregate_sha256": baseline_aggregate[
            "aggregate_sha256"
        ],
        "mutated_raw_output_aggregate_sha256": mutated_aggregate[
            "aggregate_sha256"
        ],
        "source_raw_output_aggregate_sha256": baseline_aggregate[
            "aggregate_sha256"
        ],
        "candidate_raw_output_aggregate_sha256": mutated_aggregate[
            "aggregate_sha256"
        ],
        "file_list_path": _display_path(file_list, root),
        "file_list_sha256": _sha256_file(file_list),
        "n_pairs_hashed": n_pairs_hashed,
        "total_frames": total_frames,
        "baseline_outputs": baseline_aggregate,
        "mutated_outputs": mutated_aggregate,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof = {
        "schema": TT5L_SIDEINFO_CONSUMPTION_PROOF_SCHEMA,
        "gate_id": TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
        "predicate_id": TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID,
        "predicate_passed": predicate_passed,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "proof_scope": "contest_full_frame_sideinfo_consumption_proof",
        "byte_mutation_proof": byte_mutation_proof,
        "component_proofs": {"per_pair_side_info": byte_mutation_proof},
        "runtime_tree_files": list(_RUNTIME_TREE_FILES),
        "runtime_tree_sha256": runtime_sha,
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_bytes(_canonical_json_bytes(manifest))
    proof_path.write_bytes(_canonical_json_bytes(proof))
    return TT5LConsumptionProofResult(
        proof=proof,
        manifest=manifest,
        proof_path=proof_path,
        manifest_path=manifest_out,
    )


__all__ = [
    "TT5L_SIDEINFO_CONSUMPTION_DEFAULT_ARTIFACT",
    "TT5L_SIDEINFO_CONSUMPTION_DEFAULT_MANIFEST",
    "TT5L_SIDEINFO_CONSUMPTION_DEFAULT_WORK_DIR",
    "TT5L_SIDEINFO_CONSUMPTION_GATE_ID",
    "TT5L_SIDEINFO_CONSUMPTION_PREDICATE_ID",
    "TT5L_SIDEINFO_CONSUMPTION_PROOF_SCHEMA",
    "TT5LConsumptionProofResult",
    "build_tt5l_contest_full_frame_sideinfo_consumption_proof",
    "build_tt5l_sideinfo_consumption_proof",
]
