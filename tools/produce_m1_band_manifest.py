#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce the real n600 M1 positive-anisotropic band artifact.

This is encoder-side, advisory custody.  It never invokes a scorer, trainer,
provider, or evaluator.  Each logical pair is first written as an immutable
checkpoint; the consumer arrays are then assembled through resumable memmaps
and admitted only after :class:`BandArtifact` parses them back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tac.boundary_math.integer_plane_banded_trainer import (  # noqa: E402
    BAND_SCHEMA,
    INACTIVE_BAND_RADIUS,
    LOGICAL_PAIR_COUNT,
    PLANE_SHAPE,
    RATE_SCORE_PER_BYTE,
    BandArtifact,
    canonical_json,
    sha256_file,
    storage_preflight,
)
from tac.boundary_math.integer_plane_emitter import factor2_operator  # noqa: E402
from tac.optimization.vjp_custody import sha256_array  # noqa: E402

SCHEMA = "m1_band_manifest_producer_receipt.v1"
PAIR_SCHEMA = "m1_band_manifest_pair_checkpoint.v1"
CONFIG_SCHEMA = "m1_band_manifest_producer_config.v1"
ASSEMBLY_SCHEMA = "m1_band_manifest_assembly_state.v1"
SEED = 20260720
PAIR_COUNT = 600
CANDIDATE_COUNT = 38_077
MAX_RGB_RADIUS = 8.0
SCALE = 1.0
LOCAL_LIPSCHITZ_FALLBACK = 1.0
AXIS = "[macOS-CPU advisory]"
POINTER = "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
DEFAULT_CACHE = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)
DEFAULT_COMPILE_GATE = REPO / ".omx/research/r1b5_r1b2_compile_gate_20260720.json"
DEFAULT_ORDERING = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/"
    "fisher_ev/fisher_ev_ordering_38077.jsonl.br"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_band_manifest_20260720"
)
DEFAULT_PDW2_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"
)
SSD_ROOTS = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))
PDW2_MEMBER = "seg_head_target.pdw2"
STRATA = {0: "road_lane_edge", 1: "other_edge", 2: "bulk_nonedge"}
EXPECTED_ORDERING_SHA256 = "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"
EXPECTED_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
LAW_REFS = [
    "frozen_scorer_fisher_curvature_margin_colocation_v1",
    "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
    "realization_necessity_preimage_per_stratum_v1",
    "resize_exploit_flip_fix_frontier_v1",
    "segnet_head_rank4_linear_flipdist_v1",
    "posenet_luma_chroma_sensitivity_asymmetry_v1",
    "flip_margin_step_law_v1",
    "instant_projected_input_adjoint_v1",
    "shearlet_nterm_upper_bounds_task_rate_v1",
    "curvelet_directional_basis_dseg_reduction_v1",
    "cgauge_curvelet_parabolic_bank_v1",
    "scorer_obligation_matrix_factorization_v1",
    "lane_band_ego_factorization_source_reparam_v1",
    "witness_measured_reverse_waterfill_v1",
    "meta_lagrangian_dual_solver_per_axis_kkt_residual_v1",
    "cgauge_master_action_v1",
]
ARTIFACT_NAMES = (
    "ranked_ev_field",
    "necessity",
    "resize",
    "channel_sensitivity",
    "kkt",
    "inner_jacobian_secant_qp",
    "curvelet_carrier",
    "xi_factorization",
    "gauge_binding",
)


class ProducerError(RuntimeError):
    """Fail-closed input, custody, resume, or artifact error."""


def _beneath_ssd(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for root in SSD_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    raise ProducerError(f"large band artifacts must remain on the SSD waterfall: {resolved}")


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ProducerError(f"overwrite refused: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    _atomic_bytes(path, canonical_json(value), replace=replace)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProducerError(f"overwrite refused: {path}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProducerError(f"overwrite refused: {path}")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Map one unencrypted ZIP_STORED NPY member without loading the cache."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise ProducerError(f"cache lacks {member}") from exc
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ProducerError(f"cache member must be unencrypted ZIP_STORED: {member}")
        offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(offset)
        header = handle.read(30)
        if len(header) != 30:
            raise ProducerError(f"truncated ZIP header: {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise ProducerError(f"invalid ZIP local header: {member}")
        handle.seek(offset + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ProducerError(f"unsupported NPY version for {member}: {version}")
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )


def quantize_band_widths(raw_widths: np.ndarray, *, max_radius: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return integer scorer-u8 widths and per-channel realizability.

    A positive width below one scorer uint8 step admits no nonzero integer
    displacement and is therefore dead.  This is a necessary structural gate,
    not proof that a camera-lattice/receiver realization exists.
    """

    raw = np.asarray(raw_widths, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ProducerError("raw band widths must be finite nonnegative Nx3")
    if max_radius <= 0 or max_radius >= int(INACTIVE_BAND_RADIUS):
        raise ProducerError("max radius leaves its valid domain")
    clipped = np.minimum(raw, float(max_radius))
    steps = np.floor(clipped).astype(np.uint8)
    return steps, steps >= np.uint8(1)


def derive_candidate_widths(
    margin: np.ndarray,
    pair_norm: np.ndarray,
    unit_pullback: np.ndarray,
    local_lipschitz: np.ndarray,
    *,
    scale: float = SCALE,
    max_radius: float = MAX_RGB_RADIUS,
) -> np.ndarray:
    """Exact vectorized specialization of ``derive_hyperplane_channel_band``."""

    margin64 = np.asarray(margin, dtype=np.float64)
    norms64 = np.asarray(pair_norm, dtype=np.float64)
    pull64 = np.asarray(unit_pullback, dtype=np.float64)
    lip64 = np.asarray(local_lipschitz, dtype=np.float64)
    count = len(margin64)
    if (
        margin64.shape != (count,)
        or norms64.shape != (count,)
        or pull64.shape != (count, 3)
        or lip64.shape != (count,)
        or not all(np.isfinite(value).all() for value in (margin64, norms64, pull64, lip64))
        or np.any(margin64 < 0)
        or np.any(norms64 <= 0)
        or np.any(lip64 < 0)
    ):
        raise ProducerError("candidate hyperplane fields leave their valid domain")
    distance = float(scale) * margin64 / norms64
    denominator = 3.0 * np.abs(pull64) * lip64[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        widths = np.where(denominator > 0, distance[:, None] / denominator, max_radius)
    return np.minimum(widths, max_radius)


def _load_ordering(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_ORDERING_SHA256:
        raise ProducerError("sealed Fisher ordering hash mismatch")
    lines = brotli.decompress(raw).splitlines()
    header = json.loads(lines[0])
    if (
        header.get("schema") != "r1b5_fisher_ev_ordering_jsonl.v1"
        or header.get("candidate_count") != CANDIDATE_COUNT
        or len(lines) != CANDIDATE_COUNT + 1
    ):
        raise ProducerError("sealed Fisher ordering header/count mismatch")
    columns = header.get("columns")
    rows = [dict(zip(columns, json.loads(line), strict=True)) for line in lines[1:]]
    cells = [(int(row["pair"]), int(row["row"]), int(row["col"])) for row in rows]
    if len(set(cells)) != CANDIDATE_COUNT or any(not 0 <= pair < 24 for pair, _, _ in cells):
        raise ProducerError("Fisher candidates are not the sealed unique n24 population")
    return rows, raw


def _load_compile_gate(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    document = json.loads(raw)
    campaign = document.get("vjp_campaign", {})
    rows = campaign.get("per_pair_declared_custody")
    if (
        campaign.get("status") != "COMPLETE_N600"
        or campaign.get("completed_pair_count") != PAIR_COUNT
        or not isinstance(rows, list)
        or [row.get("pair_id") for row in rows] != list(range(PAIR_COUNT))
    ):
        raise ProducerError("R1b2 compile gate lacks complete ordered n600 VJP custody")
    campaign_record = campaign.get("campaign", {})
    campaign_path = Path(str(campaign_record.get("path", ""))).resolve(strict=True)
    if (
        campaign_path.stat().st_size != campaign_record.get("bytes")
        or sha256_file(campaign_path) != campaign_record.get("sha256")
    ):
        raise ProducerError("VJP campaign receipt custody mismatch")
    return document, rows, raw


def _merkle_tensor(rows: Sequence[Mapping[str, Any]], field: str, *, active_margin: bool = False) -> str:
    leaves: list[list[Any]] = []
    for row in rows:
        hashes = row["tensor_hashes"]
        selected_field = (
            "native_margin" if active_margin and "cached_winner" in hashes else "cached_margin"
        ) if field == "margin" else field
        digest = hashes.get(selected_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ProducerError(f"missing tensor custody {selected_field} for pair {row['pair_id']}")
        leaves.append([int(row["pair_id"]), selected_field, digest])
    return hashlib.sha256(canonical_json(leaves)).hexdigest()


def _project_pair(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    operator = factor2_operator()
    result = np.empty(PLANE_SHAPE, dtype=np.uint8)
    for plane, frame in enumerate((frame0, frame1)):
        numerator, denominator = operator.apply_numerators(np.asarray(frame))
        result[plane] = np.clip(
            np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0
        ).astype(np.uint8)
    return result


def _load_active_fields(row: Mapping[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    path = Path(str(row["path"])).resolve(strict=True)
    hashes = row["tensor_hashes"]
    margin_name = "native_margin" if "cached_winner" in hashes else "cached_margin"
    names = (margin_name, "winner", "rival", "head_pair_norms", "seg_q", "seg_local_lipschitz")
    loaded: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in names:
            value = np.asarray(archive[name])
            if sha256_array(value) != hashes.get(name):
                raise ProducerError(f"pair {row['pair_id']} tensor hash mismatch: {name}")
            loaded[name] = value.reshape(-1, *value.shape[2:])[indices].copy() if value.ndim == 3 else value.reshape(-1)[indices].copy()
    loaded["margin"] = loaded.pop(margin_name)
    return loaded


def _pair_checkpoint(
    *,
    pair_id: int,
    source: np.ndarray,
    radii: np.ndarray,
    rank_indices: np.ndarray,
    raw_widths: np.ndarray,
    effective_steps: np.ndarray,
    realizability: np.ndarray,
    strata: np.ndarray,
    sidecar_sha256: str,
    config_sha256: str,
) -> dict[str, np.ndarray]:
    metadata = {
        "schema": PAIR_SCHEMA,
        "pair_id": pair_id,
        "config_sha256": config_sha256,
        "sidecar_sha256": sidecar_sha256,
        "source_sha256": sha256_array(source),
        "radii_sha256": sha256_array(radii),
        "candidate_count": len(rank_indices),
        "selected_pixel_count": int(np.count_nonzero(np.any(realizability, axis=1))),
    }
    return {
        "metadata_json": np.asarray(canonical_json(metadata).decode("ascii")),
        "source": source,
        "radii": radii,
        "rank_indices": rank_indices.astype(np.int32),
        "raw_widths": raw_widths.astype(np.float32),
        "effective_steps": effective_steps.astype(np.uint8),
        "realizability": realizability.astype(np.bool_),
        "strata": strata.astype(np.uint8),
    }


def _validate_pair_checkpoint(path: Path, pair_id: int, config_sha256: str, sidecar_sha256: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).reshape(())))
        source = np.asarray(archive["source"])
        radii = np.asarray(archive["radii"])
        if (
            metadata.get("schema") != PAIR_SCHEMA
            or metadata.get("pair_id") != pair_id
            or metadata.get("config_sha256") != config_sha256
            or metadata.get("sidecar_sha256") != sidecar_sha256
            or source.dtype != np.uint8
            or source.shape != PLANE_SHAPE
            or radii.dtype != np.float32
            or radii.shape != PLANE_SHAPE
            or sha256_array(source) != metadata.get("source_sha256")
            or sha256_array(radii) != metadata.get("radii_sha256")
        ):
            raise ProducerError(f"stale or malformed pair checkpoint: {pair_id}")
    return metadata


def _write_or_resume_pair_checkpoints(
    output: Path,
    cache: Path,
    vjp_rows: list[dict[str, Any]],
    ordering: list[dict[str, Any]],
    config_sha256: str,
) -> None:
    frame0 = stored_npy_memmap(cache, "gt_f0")
    frame1 = stored_npy_memmap(cache, "gt_f1")
    if frame0.shape != (PAIR_COUNT, 874, 1164, 3) or frame1.shape != frame0.shape:
        raise ProducerError("real cache frame geometry mismatch")
    by_pair: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for rank, row in enumerate(ordering):
        by_pair[int(row["pair"])].append((rank, row))
    checkpoints = output / "checkpoints"
    for pair_id in range(PAIR_COUNT):
        custody = vjp_rows[pair_id]
        sidecar = Path(str(custody["path"])).resolve(strict=True)
        actual_sidecar_sha = sha256_file(sidecar)
        if sidecar.stat().st_size != custody.get("bytes") or actual_sidecar_sha != custody.get("declared_sha256"):
            raise ProducerError(f"pair {pair_id} VJP sidecar byte custody mismatch")
        checkpoint = checkpoints / f"pair_{pair_id:04d}.band.npz"
        if checkpoint.exists():
            _validate_pair_checkpoint(checkpoint, pair_id, config_sha256, actual_sidecar_sha)
            continue
        source = _project_pair(frame0[pair_id], frame1[pair_id])
        radii = np.full(PLANE_SHAPE, np.float32(INACTIVE_BAND_RADIUS), dtype=np.float32)
        ranked = by_pair.get(pair_id, [])
        rank_indices = np.asarray([rank for rank, _ in ranked], dtype=np.int32)
        strata = np.asarray([row["necessity_edge_tier"] for _, row in ranked], dtype=np.uint8)
        if ranked:
            linear = np.asarray([int(row["row"]) * 512 + int(row["col"]) for _, row in ranked])
            fields = _load_active_fields(custody, linear)
            if np.any(fields["winner"] == fields["rival"]):
                raise ProducerError(f"pair {pair_id} contains identical active winner/rival")
            raw_widths = derive_candidate_widths(
                fields["margin"],
                fields["head_pair_norms"],
                fields["seg_q"],
                fields["seg_local_lipschitz"],
            )
            effective_steps, realizability = quantize_band_widths(raw_widths)
            pixel_realizable = np.any(realizability, axis=1)
            for local_index, (_rank, row) in enumerate(ranked):
                if pixel_realizable[local_index]:
                    radii[1, int(row["row"]), int(row["col"])] = effective_steps[local_index]
        else:
            raw_widths = np.empty((0, 3), dtype=np.float64)
            effective_steps = np.empty((0, 3), dtype=np.uint8)
            realizability = np.empty((0, 3), dtype=np.bool_)
        arrays = _pair_checkpoint(
            pair_id=pair_id,
            source=source,
            radii=radii,
            rank_indices=rank_indices,
            raw_widths=raw_widths,
            effective_steps=effective_steps,
            realizability=realizability,
            strata=strata,
            sidecar_sha256=actual_sidecar_sha,
            config_sha256=config_sha256,
        )
        _atomic_npz(checkpoint, **arrays)
        _validate_pair_checkpoint(checkpoint, pair_id, config_sha256, actual_sidecar_sha)
        print(f"pair-checkpoint {pair_id + 1}/{PAIR_COUNT}", flush=True)


def _assemble_arrays(output: Path, config_sha256: str, vjp_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    source_path = output / "source_planes.npy"
    radii_path = output / "radii_uint8_steps.npy"
    if source_path.exists() != radii_path.exists():
        raise ProducerError("consumer array finalization is asymmetric")
    if source_path.exists():
        return source_path, radii_path
    partial_source = output / ".source_planes.partial.npy"
    partial_radii = output / ".radii_uint8_steps.partial.npy"
    state_path = output / "assembly_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != ASSEMBLY_SCHEMA or state.get("config_sha256") != config_sha256:
            raise ProducerError("assembly resume state identity mismatch")
        next_pair = int(state["next_pair"])
        if not partial_source.exists() or not partial_radii.exists():
            raise ProducerError("assembly resume memmap is missing")
        source = np.lib.format.open_memmap(partial_source, mode="r+")
        radii = np.lib.format.open_memmap(partial_radii, mode="r+")
    else:
        source = np.lib.format.open_memmap(
            partial_source, mode="w+", dtype=np.uint8, shape=(PAIR_COUNT, *PLANE_SHAPE)
        )
        radii = np.lib.format.open_memmap(
            partial_radii, mode="w+", dtype=np.float32, shape=(PAIR_COUNT, *PLANE_SHAPE)
        )
        next_pair = 0
        state = {"schema": ASSEMBLY_SCHEMA, "config_sha256": config_sha256, "next_pair": 0}
        _atomic_json(state_path, state)
    if source.shape != (PAIR_COUNT, *PLANE_SHAPE) or radii.shape != source.shape:
        raise ProducerError("assembly memmap geometry mismatch")
    for pair_id in range(next_pair, PAIR_COUNT):
        checkpoint = output / "checkpoints" / f"pair_{pair_id:04d}.band.npz"
        _validate_pair_checkpoint(
            checkpoint,
            pair_id,
            config_sha256,
            str(vjp_rows[pair_id]["declared_sha256"]),
        )
        with np.load(checkpoint, allow_pickle=False) as archive:
            source[pair_id] = archive["source"]
            radii[pair_id] = archive["radii"]
        source.flush()
        radii.flush()
        state["next_pair"] = pair_id + 1
        _atomic_json(state_path, state, replace=True)
    del source, radii
    os.replace(partial_source, source_path)
    os.replace(partial_radii, radii_path)
    state["complete"] = True
    _atomic_json(state_path, state, replace=True)
    return source_path, radii_path


def _aggregate_candidate_arrays(output: Path, ordering: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "cells": output / "candidate_cells.npy",
        "raw": output / "raw_band_widths.npy",
        "steps": output / "effective_uint8_steps.npy",
        "flags": output / "realizability_flags.npy",
        "pixels": output / "pixel_realizable.npy",
        "strata": output / "candidate_strata.npy",
    }
    if all(path.exists() for path in paths.values()):
        raw = np.load(paths["raw"], mmap_mode="r")
        steps = np.load(paths["steps"], mmap_mode="r")
        flags = np.load(paths["flags"], mmap_mode="r")
        pixels = np.load(paths["pixels"], mmap_mode="r")
        strata = np.load(paths["strata"], mmap_mode="r")
    elif any(path.exists() for path in paths.values()):
        raise ProducerError("candidate aggregate finalization is incomplete")
    else:
        cells = np.empty((CANDIDATE_COUNT, 4), dtype=np.int32)
        raw = np.empty((CANDIDATE_COUNT, 3), dtype=np.float32)
        steps = np.empty((CANDIDATE_COUNT, 3), dtype=np.uint8)
        flags = np.empty((CANDIDATE_COUNT, 3), dtype=np.bool_)
        pixels = np.empty(CANDIDATE_COUNT, dtype=np.bool_)
        strata = np.empty(CANDIDATE_COUNT, dtype=np.uint8)
        filled = np.zeros(CANDIDATE_COUNT, dtype=np.bool_)
        for pair_id in range(PAIR_COUNT):
            checkpoint = output / "checkpoints" / f"pair_{pair_id:04d}.band.npz"
            with np.load(checkpoint, allow_pickle=False) as archive:
                ranks = np.asarray(archive["rank_indices"], dtype=np.int64)
                raw[ranks] = archive["raw_widths"]
                steps[ranks] = archive["effective_steps"]
                flags[ranks] = archive["realizability"]
                pixels[ranks] = np.any(archive["realizability"], axis=1)
                strata[ranks] = archive["strata"]
                filled[ranks] = True
        if not np.all(filled):
            raise ProducerError("candidate aggregate lacks sealed rank coverage")
        for rank, row in enumerate(ordering):
            cells[rank] = (int(row["pair"]), 1, int(row["row"]), int(row["col"]))
        for name, value in (
            ("cells", cells), ("raw", raw), ("steps", steps), ("flags", flags),
            ("pixels", pixels), ("strata", strata),
        ):
            _atomic_npy(paths[name], np.asarray(value))
    selected = int(np.count_nonzero(pixels))
    if not 0 < selected <= CANDIDATE_COUNT:
        raise ProducerError("quantization gate selected an invalid population")
    return {
        "paths": paths,
        "selected_pixel_count": selected,
        "dead_pixel_count": CANDIDATE_COUNT - selected,
        "channel_realizable_count": np.count_nonzero(flags, axis=0).astype(int).tolist(),
        "effective_step_histogram": {
            str(int(key)): int(value)
            for key, value in sorted(Counter(np.asarray(steps).reshape(-1).tolist()).items())
        },
        "stratum_candidate_count": {
            STRATA[key]: int(np.count_nonzero(strata == key)) for key in sorted(STRATA)
        },
        "stratum_selected_count": {
            STRATA[key]: int(np.count_nonzero((strata == key) & pixels)) for key in sorted(STRATA)
        },
    }


def _frame_summary(ordering: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    pixels = np.load(candidate["paths"]["pixels"], mmap_mode="r")
    frame_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for pair_id in range(PAIR_COUNT):
        for frame in (0, 1):
            frame_rows[(pair_id, frame)] = {
                "pair_id": pair_id,
                "frame": frame,
                "candidate_count": 0,
                "selected_pixel_count": 0,
                "dead_pixel_count": 0,
                "strata": {name: {"candidate_count": 0, "selected_pixel_count": 0} for name in STRATA.values()},
            }
    for rank, ordered in enumerate(ordering):
        row = frame_rows[(int(ordered["pair"]), 1)]
        name = STRATA[int(ordered["necessity_edge_tier"])]
        row["candidate_count"] += 1
        row["strata"][name]["candidate_count"] += 1
        if bool(pixels[rank]):
            row["selected_pixel_count"] += 1
            row["strata"][name]["selected_pixel_count"] += 1
        else:
            row["dead_pixel_count"] += 1
    return {
        "schema": "m1_band_manifest_frame_summary.v1",
        "authority": AXIS,
        "pair_count": PAIR_COUNT,
        "frame_count": PAIR_COUNT * 2,
        "candidate_population": "sealed_PDW1_n24_realization_mismatches_exact_38077",
        "n600_totals": {
            "candidate_count": CANDIDATE_COUNT,
            "selected_pixel_count": candidate["selected_pixel_count"],
            "dead_pixel_count": candidate["dead_pixel_count"],
            "stratum_candidate_count": candidate["stratum_candidate_count"],
            "stratum_selected_count": candidate["stratum_selected_count"],
        },
        "frames": list(frame_rows.values()),
    }


def _write_artifacts(
    output: Path,
    ordering_path: Path,
    ordering: list[dict[str, Any]],
    candidate: dict[str, Any],
    vjp_rows: list[dict[str, Any]],
    compile_gate_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    ranked = records_dir / "ranked_ev_field.jsonl.br"
    if not ranked.exists():
        temporary = ranked.with_name(f".{ranked.name}.{os.getpid()}.tmp")
        shutil.copyfile(ordering_path, temporary)
        os.replace(temporary, ranked)
    if sha256_file(ranked) != EXPECTED_ORDERING_SHA256:
        raise ProducerError("copied Fisher ordering hash mismatch")
    frame_summary = _frame_summary(ordering, candidate)
    summary_path = output / "frame_summary.json"
    if not summary_path.exists():
        _atomic_json(summary_path, frame_summary)
    elif json.loads(summary_path.read_text()) != frame_summary:
        raise ProducerError("frame summary resume mismatch")
    merkle = {
        "margins_sha256": _merkle_tensor(vjp_rows, "margin", active_margin=True),
        "winner_sha256": _merkle_tensor(vjp_rows, "winner"),
        "rival_sha256": _merkle_tensor(vjp_rows, "rival"),
        "unit_head_normal_pullback_rgb_sha256": _merkle_tensor(vjp_rows, "seg_q"),
        "pair_norms_sha256": _merkle_tensor(vjp_rows, "head_pair_norms"),
    }
    channel_rows = [
        {
            "pair_id": int(row["pair_id"]),
            "path": str(Path(str(row["path"])).resolve()),
            "bytes": int(row["bytes"]),
            "sha256": str(row["declared_sha256"]),
            "active_margin_tensor": (
                "native_margin" if "cached_winner" in row["tensor_hashes"] else "cached_margin"
            ),
            "tensor_hashes": row["tensor_hashes"],
        }
        for row in vjp_rows
    ]
    artifacts: dict[str, Any] = {
        "necessity": {
            "schema": "m1_band_necessity.v1",
            "population": "PDW1_n24_realization_mismatches_exact_38077",
            "scope": "candidate support is n24; source and derivative custody are n600",
            "strata": candidate["stratum_candidate_count"],
            "selected_by_stratum": candidate["stratum_selected_count"],
            "frame_summary": {"path": "../frame_summary.json", "sha256": sha256_file(summary_path)},
        },
        "resize": {
            "schema": "m1_band_resize_realizability.v1",
            "operator": "factor2_align_corners_false_disjoint_2x2",
            "source_projection": "round(exact_resize_numerator/common_denominator)_to_uint8",
            "quantization_rule": "effective_uint8_steps=floor(raw_scorer_rgb_radius)",
            "dead_rule": "all channel widths below one scorer uint8 step are unrealizable/dead",
            "verdict_scope": "necessary scorer-u8 gate; not sufficient camera-lattice or receiver proof",
            "candidate_sidecars": {
                name: {"path": f"../{path.name}", "sha256": sha256_file(path)}
                for name, path in candidate["paths"].items()
            },
            "selected_pixel_count": candidate["selected_pixel_count"],
            "dead_pixel_count": candidate["dead_pixel_count"],
            "effective_step_histogram": candidate["effective_step_histogram"],
        },
        "channel_sensitivity": {
            "schema": "m1_band_channel_sensitivity_custody.v1",
            "compile_gate": {"path": str(compile_gate_path.resolve()), "sha256": sha256_file(compile_gate_path)},
            "pair_count": PAIR_COUNT,
            "all_sidecar_bytes_rehashed_by_producer": True,
            "merkle_definition": "sha256(canonical_json([[pair_id,tensor_name,tensor_sha256],...]))",
            "merkle": merkle,
            "per_pair": channel_rows,
        },
        "kkt": {
            "schema": "m1_band_kkt_status.v1",
            "structural_quantization_stop": True,
            "structural_stop_meaning": "dead cells have zero attainable nonzero scorer-u8 step and hence zero structural EV/byte",
            "receiver_byte_admission": "BLOCKED",
            "blocker": "RECEIVER_BYTE_KKT_ADMISSION_PENDING_REALIZED_BACKBONE_SECANTS_AND_EXACT_PREFIX_BYTE_MARGINALS",
            "verdict_scope": "manifest selection is a band-training pre-admission set, not receiver-byte GO",
            "formalization": "FORMALIZATION_PENDING",
        },
        "inner_jacobian_secant_qp": {
            "schema": "m1_band_inner_jacobian_secant_qp_status.v1",
            "first_order_vjp": "MEASURED_REAL_N600",
            "realized_backbone_secants": "ABSENT",
            "qp_receiver_closure": "ABSENT",
            "blocker": "R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT",
            "verdict_scope": "first-order band geometry only",
            "formalization": "FORMALIZATION_PENDING",
        },
        "curvelet_carrier": {
            "schema": "m1_band_curvelet_carrier_status.v1",
            "catalog_502": "GENERIC_WINDOWED_CURVELET_SHEARLET_FRAME_EXISTS",
            "r1b4_receiver": "RECEIVER_BOUND_PACKET_APPLICATION_EXISTS",
            "c2_banded_trainer_binding": "ABSENT",
            "status": "#502_R1B4_RECEIVER_BOUND_C2_BANDED_TRAINER_BINDING_ABSENT",
            "verdict_scope": "curvelet family remains live; this exact C2 trainer topology is not carrier-bound",
        },
        "xi_factorization": {
            "schema": "m1_band_xi_factorization_status.v1",
            "pose_factorization": "single_se3_xi_twist",
            "status": "MEASURED_VJP_CUSTODY_PRESENT_NOT_RECEIVER_FACTORIZED",
            "verdict_scope": "no pose score or promotion claim",
        },
        "gauge_binding": {
            "schema": "m1_band_gauge_binding_status.v1",
            "status": "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED",
            "pdw2_spatial_receiver": "MODULE_MERGED",
            "c2_factor2_composition": "ABSENT",
            "blocker": "PDW2_SPATIAL_RECEIVER_MODULE_MERGED_C2_FACTOR2_COMPOSITION_ABSENT",
            "verdict_scope": "PDW2 target bytes parse, but do not yet close scorer-free spatial/RGB pullback in C2",
        },
    }
    artifact_records: dict[str, dict[str, str]] = {
        "ranked_ev_field": {"path": str(ranked.relative_to(output)), "sha256": sha256_file(ranked)}
    }
    for name, document in artifacts.items():
        path = records_dir / f"{name}.json"
        if not path.exists():
            _atomic_json(path, document)
        elif json.loads(path.read_text()) != document:
            raise ProducerError(f"artifact resume mismatch: {name}")
        artifact_records[name] = {"path": str(path.relative_to(output)), "sha256": sha256_file(path)}
    if set(artifact_records) != set(ARTIFACT_NAMES):
        raise ProducerError("artifact record set drifted")
    return artifact_records, merkle


def _extract_pdw2(source_archive: Path, output: Path) -> Path:
    destination = output / "inputs" / PDW2_MEMBER
    with zipfile.ZipFile(source_archive) as archive:
        try:
            payload = archive.read(PDW2_MEMBER)
        except KeyError as exc:
            raise ProducerError(f"PDW2 source archive lacks {PDW2_MEMBER}") from exc
    if not destination.exists():
        _atomic_bytes(destination, payload)
    elif destination.read_bytes() != payload:
        raise ProducerError("PDW2 materializer input resume mismatch")
    return destination


def _materialize_manifest(
    output: Path,
    source: Path,
    radii: Path,
    artifact_records: dict[str, dict[str, str]],
    merkle: dict[str, str],
    selected_count: int,
) -> Path:
    document = {
        "schema": BAND_SCHEMA,
        "mode": "positive_anisotropic",
        "pair_count": PAIR_COUNT,
        "geometry": list(PLANE_SHAPE),
        "source_planes": {"path": source.name, "sha256": sha256_file(source)},
        "radii": {"path": radii.name, "sha256": sha256_file(radii)},
        "custody": {
            "derivation": "derive_hyperplane_channel_band",
            **merkle,
            "config": {
                "scale": SCALE,
                "local_lipschitz": LOCAL_LIPSCHITZ_FALLBACK,
                "max_rgb_radius": MAX_RGB_RADIUS,
            },
            "ev_selection": {
                "policy": "measured_reverse_waterfill_highest_ev_first",
                "candidate_flip_count": CANDIDATE_COUNT,
                "selected_pixel_count": selected_count,
                "inactive_radius": float(INACTIVE_BAND_RADIUS),
                "rate_break_even_score_per_byte": RATE_SCORE_PER_BYTE,
                "stopped_below_break_even": True,
                "blanket_fix": False,
                "artifact_records": artifact_records,
                "law_refs": LAW_REFS,
                "metric": "fisher_top1_top2_margin",
                "carrier_basis": "cgauge_curvelet_parabolic_bank_v1",
                "realization_predictor": "first_order_plus_secant_plus_qp_inner_jacobian",
                "pose_factorization": "single_se3_xi_twist",
                "gauge_status": "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED",
            },
        },
    }
    path = output / "band_manifest.json"
    if not path.exists():
        _atomic_json(path, document)
    elif json.loads(path.read_text()) != document:
        raise ProducerError("band manifest resume mismatch")
    loaded = BandArtifact.load(path)
    if loaded.mode != "positive_anisotropic" or loaded.pair_count != LOGICAL_PAIR_COUNT:
        raise ProducerError("BandArtifact parse-back did not admit the final artifact")
    return path


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output = _beneath_ssd(args.output)
    output.mkdir(parents=True, exist_ok=True)
    completed_receipt = output / "receipt.json"
    if completed_receipt.exists():
        validated = check(completed_receipt)
        document = json.loads(completed_receipt.read_text())
        return {
            "receipt": str(completed_receipt),
            "receipt_sha256": validated["receipt_sha256"],
            "manifest": document["band_manifest"]["path"],
            "manifest_sha256": validated["manifest_sha256"],
            "selected_pixel_count": document["selected_pixel_count"],
            "dead_substep_pixel_count": document["dead_substep_pixel_count"],
            "status": "PASS_BANDARTIFACT_LOAD_BLOCKED_DRY_CONFIG_ONLY_RESUMED_COMPLETE",
        }
    preflight = storage_preflight(output, required_free_bytes=args.required_free_bytes)
    if not preflight["ok"]:
        raise ProducerError("SSD storage preflight refused the producer")
    cache = args.cache.expanduser().resolve(strict=True)
    ordering_path = args.ordering.expanduser().resolve(strict=True)
    compile_gate_path = args.compile_gate.expanduser().resolve(strict=True)
    pdw2_archive = args.pdw2_archive.expanduser().resolve(strict=True)
    if sha256_file(cache) != EXPECTED_CACHE_SHA256:
        raise ProducerError("real n600 cache hash mismatch")
    ordering, _ordering_raw = _load_ordering(ordering_path)
    _gate, vjp_rows, gate_raw = _load_compile_gate(compile_gate_path)
    config = {
        "schema": CONFIG_SCHEMA,
        "seed": args.seed,
        "pair_count": PAIR_COUNT,
        "cache": {"path": str(cache), "sha256": EXPECTED_CACHE_SHA256},
        "ordering": {"path": str(ordering_path), "sha256": EXPECTED_ORDERING_SHA256},
        "compile_gate": {
            "path": str(compile_gate_path),
            "sha256": hashlib.sha256(gate_raw).hexdigest(),
        },
        "scale": SCALE,
        "local_lipschitz_fallback": LOCAL_LIPSCHITZ_FALLBACK,
        "max_rgb_radius": MAX_RGB_RADIUS,
        "quantization": "floor_scorer_uint8_width_sub1_dead",
    }
    config_sha256 = hashlib.sha256(canonical_json(config)).hexdigest()
    config_path = output / "producer_config.json"
    if not config_path.exists():
        _atomic_json(config_path, config)
    elif hashlib.sha256(config_path.read_bytes()).hexdigest() != config_sha256:
        raise ProducerError("producer resume config mismatch")
    _write_or_resume_pair_checkpoints(output, cache, vjp_rows, ordering, config_sha256)
    source, radii = _assemble_arrays(output, config_sha256, vjp_rows)
    candidate = _aggregate_candidate_arrays(output, ordering)
    artifact_records, merkle = _write_artifacts(
        output, ordering_path, ordering, candidate, vjp_rows, compile_gate_path
    )
    pdw2_packet = _extract_pdw2(pdw2_archive, output)
    manifest = _materialize_manifest(
        output,
        source,
        radii,
        artifact_records,
        merkle,
        candidate["selected_pixel_count"],
    )
    loaded = BandArtifact.load(manifest)
    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "seed": args.seed,
        "git_head": _git_head(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "storage_preflight": preflight,
        "config": {"path": str(config_path), "sha256": config_sha256},
        "band_manifest": {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": loaded.manifest_sha256,
            "consumer": "tac.boundary_math.integer_plane_banded_trainer.BandArtifact.load",
            "parse_back": "PASS",
        },
        "source_planes": {"path": str(source), "bytes": source.stat().st_size, "sha256": loaded.source_sha256},
        "radii": {"path": str(radii), "bytes": radii.stat().st_size, "sha256": sha256_file(radii)},
        "candidate_count": CANDIDATE_COUNT,
        "selected_pixel_count": candidate["selected_pixel_count"],
        "dead_substep_pixel_count": candidate["dead_pixel_count"],
        "stratum_candidate_count": candidate["stratum_candidate_count"],
        "stratum_selected_count": candidate["stratum_selected_count"],
        "per_pair_checkpoint_count": PAIR_COUNT,
        "vjp_sidecars_rehashed": PAIR_COUNT,
        "pdw2_materializer_input": {
            "path": str(pdw2_packet), "bytes": pdw2_packet.stat().st_size, "sha256": sha256_file(pdw2_packet)
        },
        "readiness": "BLOCKED_DRY_CONFIG_ONLY",
        "blocking_gates": [
            "RECEIVER_BYTE_KKT_ADMISSION_PENDING_REALIZED_BACKBONE_SECANTS_AND_EXACT_PREFIX_BYTE_MARGINALS",
            "#502_R1B4_RECEIVER_BOUND_C2_BANDED_TRAINER_BINDING_ABSENT",
            "PDW2_SPATIAL_RECEIVER_MODULE_MERGED_C2_FACTOR2_COMPOSITION_ABSENT",
        ],
        "launch": False,
        "paid_dispatch": False,
        "pointer_mutation": False,
    }
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        receipt["created_at_utc"] = existing.get("created_at_utc")
        if existing != receipt:
            raise ProducerError("receipt resume mismatch")
    else:
        _atomic_json(receipt_path, receipt)
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "manifest": str(manifest),
        "manifest_sha256": loaded.manifest_sha256,
        "selected_pixel_count": candidate["selected_pixel_count"],
        "dead_substep_pixel_count": candidate["dead_pixel_count"],
        "status": "PASS_BANDARTIFACT_LOAD_BLOCKED_DRY_CONFIG_ONLY",
    }


def check(path: Path) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("score_claim") is not False:
        raise ProducerError("producer receipt authority mismatch")
    manifest_record = receipt.get("band_manifest", {})
    manifest = Path(str(manifest_record.get("path", ""))).resolve(strict=True)
    loaded = BandArtifact.load(manifest)
    if loaded.manifest_sha256 != manifest_record.get("sha256"):
        raise ProducerError("producer receipt band manifest custody mismatch")
    return {
        "schema": SCHEMA,
        "valid": True,
        "receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": loaded.manifest_sha256,
        "launch": False,
        "score_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce")
    produce.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    produce.add_argument("--compile-gate", type=Path, default=DEFAULT_COMPILE_GATE)
    produce.add_argument("--ordering", type=Path, default=DEFAULT_ORDERING)
    produce.add_argument("--pdw2-archive", type=Path, default=DEFAULT_PDW2_ARCHIVE)
    produce.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    produce.add_argument("--seed", type=int, default=SEED)
    produce.add_argument("--required-free-bytes", type=int, default=12_000_000_000)
    validate = subparsers.add_parser("check")
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check(args.receipt) if args.command == "check" else execute(args)
    except (ProducerError, OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"M1 band producer refusal: {exc}", file=sys.stderr)
        return 6
    print(canonical_json(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
