#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build CMG2 foveated/hard-pair AMR1 residual candidate archives.

This is a deterministic byte-screen scaffold.  It starts from an existing CMG2
candidate manifest, regenerates the CMG2 decoded mask tensor, builds AMR1
repair atoms only where CMG2 differs from the decoded frontier masks, ranks
those atoms by hard-pair/fovea prior per measured compressed byte, and emits
closed archive candidates containing every repair bit.

It does not load scorers and does not make score claims.  Any emitted archive
requires exact CUDA auth eval before ranking, promotion, or retirement.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CMG2_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg2_downsample_candidate.py"
ALPHA_BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
ALPHA_ARCHIVE_BUILDER_PATH = REPO_ROOT / "experiments" / "build_alpha_mask_replacement_archive.py"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"

DEFAULT_CMG2_MANIFEST = (
    REPO_ROOT / "experiments/results/c067_cmg2_downsample2x2_candidate_20260502T1010Z/build_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c067_cmg2_foveated_repair_candidates_20260502"
DEFAULT_COMPONENT_TRACE = (
    REPO_ROOT / "experiments/results/vast_harvest/c063_same_h100_component_trace_20260502T0700Z/component_trace.json"
)

SCHEMA = "cmg2_foveated_repair_candidate_builder_v1"
PLAN_NAME = "cmg2_foveated_repair_plan.json"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this builder. Atom rankings are byte/prior "
    "planning signals only; every emitted archive requires exact CUDA auth "
    "eval before ranking, promotion, retirement, or paper claims."
)


@dataclasses.dataclass(frozen=True)
class ResidualRun:
    frame_index: int
    y: int
    x0: int
    length: int
    class_id: int
    fovea_band: str


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def _resolve_existing(raw: str, *, base_dir: Path) -> Path:
    if not raw or "\x00" in raw:
        raise ValueError(f"unsafe path value: {raw!r}")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [base_dir / path, REPO_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"could not resolve existing path {raw!r}")


def _safe_zip_info(name: str, *, compression: int) -> zipfile.ZipInfo:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe zip member name: {name!r}")
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o644 << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_source_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            zf.writestr(_safe_zip_info(name, compression=zipfile.ZIP_STORED), data)


def _read_source_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        members: dict[str, bytes] = {}
        for info in zf.infolist():
            if info.is_dir():
                raise ValueError(f"source archive contains directory member {info.filename!r}")
            name = _safe_zip_info(info.filename, compression=zipfile.ZIP_STORED).filename
            if name in members:
                raise ValueError(f"duplicate source archive member {name!r}")
            members[name] = zf.read(info)
    required = {"renderer.bin", "masks.cmg2", "optimized_poses.bin"}
    missing = required - set(members)
    if missing:
        raise ValueError(f"CMG2 source archive missing members: {sorted(missing)}")
    return members


def _load_cmg2_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("score_claim") is not False:
        raise ValueError("CMG2 manifest must have score_claim=false")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("CMG2 manifest must have promotion_eligible=false")
    if "contest_auth_eval.py --device cuda" not in manifest.get("canonical_score_source_required", ""):
        raise ValueError("CMG2 manifest must preserve exact CUDA auth eval as score source")
    return manifest


def load_full_and_cmg2_masks(
    *,
    cmg2_manifest_path: Path,
    manifest: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = cmg2_manifest_path.resolve()
    manifest = _load_cmg2_manifest(manifest_path) if manifest is None else manifest
    cmg2_builder = _load_module(CMG2_BUILDER_PATH, "_cmg2_foveated_repair_cmg2_builder")
    decoded_record = manifest.get("decoded_mask_array")
    if not isinstance(decoded_record, dict):
        raise ValueError("CMG2 manifest missing decoded_mask_array")
    decoded_path = _resolve_existing(str(decoded_record.get("path", "")), base_dir=manifest_path.parent)
    full = cmg2_builder._load_decoded_masks(decoded_path)  # noqa: SLF001 - local builder helper
    if decoded_record.get("tensor_sha256") and decoded_record["tensor_sha256"] != _sha256_bytes(full.tobytes(order="C")):
        raise ValueError("decoded mask tensor SHA mismatch against CMG2 manifest")

    cmg2_record = manifest.get("cmg2")
    if not isinstance(cmg2_record, dict):
        raise ValueError("CMG2 manifest missing cmg2 record")
    scale = cmg2_record.get("scale")
    if not isinstance(scale, list) or len(scale) != 2:
        raise ValueError(f"CMG2 manifest has invalid scale {scale!r}")
    _low, recon, disagreement = cmg2_builder.downsample_block_mode(
        full,
        scale_y=int(scale[0]),
        scale_x=int(scale[1]),
    )
    recon_sha = _sha256_bytes(recon.tobytes(order="C"))
    if cmg2_record.get("reconstructed_tensor_sha256") and cmg2_record["reconstructed_tensor_sha256"] != recon_sha:
        raise ValueError("regenerated CMG2 reconstruction SHA mismatch against manifest")
    return full, recon, {
        "decoded_mask_array": {
            "path": str(decoded_path),
            "npy_sha256": _sha256_file(decoded_path),
            "tensor_sha256": _sha256_bytes(full.tobytes(order="C")),
            "shape": [int(v) for v in full.shape],
        },
        "cmg2_reconstruction": {
            "tensor_sha256": recon_sha,
            "pixel_disagreement_vs_full": float(disagreement),
            "scale": [int(scale[0]), int(scale[1])],
        },
    }


def _fovea_bands(height: int, width: int, *, inner_radius: float, mid_radius: float) -> np.ndarray:
    if not (0.0 < inner_radius < mid_radius):
        raise ValueError("fovea radii must satisfy 0 < inner < mid")
    y = (np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / (height / 2.0)
    x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / (width / 2.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    bands = np.full((height, width), 2, dtype=np.uint8)
    bands[rr <= mid_radius] = 1
    bands[rr <= inner_radius] = 0
    return bands


def _band_name(value: int) -> str:
    return ("fovea", "mid", "periphery")[int(value)]


def _row_residual_run_segments(
    *,
    changed_row: np.ndarray,
    target_row: np.ndarray,
    band_row: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Return contiguous changed x-runs split by target class and fovea band."""
    changed_x = np.flatnonzero(changed_row)
    if changed_x.size == 0:
        return []
    classes = target_row[changed_x].astype(np.int16, copy=False)
    bands = band_row[changed_x].astype(np.int16, copy=False)
    split = np.flatnonzero(
        (np.diff(changed_x) != 1)
        | (np.diff(classes) != 0)
        | (np.diff(bands) != 0)
    ) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), split))
    stops = np.concatenate((split, np.array([changed_x.size], dtype=np.int64)))
    return [
        (int(changed_x[start]), int(changed_x[stop - 1]) + 1, int(classes[start]), int(bands[start]))
        for start, stop in zip(starts, stops, strict=True)
    ]


def residual_runs_by_pair_class_band(
    target: np.ndarray,
    candidate: np.ndarray,
    *,
    inner_radius: float = 0.35,
    mid_radius: float = 0.70,
) -> dict[tuple[int, int, str], list[ResidualRun]]:
    if target.shape != candidate.shape:
        raise ValueError(f"target/candidate shape mismatch: {target.shape} != {candidate.shape}")
    if target.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("target and candidate masks must be uint8")
    frames, height, width = target.shape
    bands = _fovea_bands(height, width, inner_radius=inner_radius, mid_radius=mid_radius)
    grouped: dict[tuple[int, int, str], list[ResidualRun]] = defaultdict(list)
    changed = target != candidate
    for frame in range(frames):
        pair = frame // 2
        for y_idx in range(height):
            for x0, x1, class_id, band in _row_residual_run_segments(
                changed_row=changed[frame, y_idx],
                target_row=target[frame, y_idx],
                band_row=bands[y_idx],
            ):
                band_name = _band_name(band)
                grouped[(pair, class_id, band_name)].append(
                    ResidualRun(
                        frame_index=frame,
                        y=y_idx,
                        x0=x0,
                        length=x1 - x0,
                        class_id=class_id,
                        fovea_band=band_name,
                    )
                )
    return dict(grouped)


def _load_pair_priors(path: Path | None, *, pair_count: int) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "sha256": None,
            "source_schema": None,
            "signal_source": "uniform_no_component_trace",
            "pair_signal": [1.0] * pair_count,
            "hardest_pair_indices": [],
            "stats": {},
        }
    payload = _read_json(path)
    pair_signal = [0.0] * pair_count
    pose = [0.0] * pair_count
    seg = [0.0] * pair_count
    if isinstance(payload.get("samples"), list):
        samples = payload["samples"]
        seen: set[int] = set()
        avg_pose = _finite(payload.get("avg_posenet_dist"), field="avg_posenet_dist")
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: samples[{idx}] must be an object")
            pair = int(sample.get("pair_index"))
            if pair < 0 or pair >= pair_count:
                continue
            seen.add(pair)
            pose[pair] = _finite(sample.get("posenet_dist"), field=f"samples[{idx}].posenet_dist")
            seg[pair] = _finite(sample.get("segnet_dist"), field=f"samples[{idx}].segnet_dist")
            signal = sample.get("score_combined_contribution_first_order")
            if signal is None:
                seg_signal = 100.0 * seg[pair] / max(pair_count, 1)
                pose_signal = 0.0 if avg_pose <= 0.0 else (5.0 / math.sqrt(10.0 * avg_pose)) * (
                    pose[pair] / max(pair_count, 1)
                )
                signal = seg_signal + pose_signal
            pair_signal[pair] = max(0.0, _finite(signal, field=f"samples[{idx}].score_signal"))
        if not seen:
            raise ValueError(f"{path}: component trace contained no usable pair samples")
        hardest = sorted(range(pair_count), key=lambda i: (pair_signal[i], pose[i], seg[i], -i), reverse=True)
        return {
            "path": str(path),
            "sha256": _sha256_file(path),
            "source_schema": payload.get("schema") or payload.get("evidence_grade"),
            "signal_source": "score_combined_contribution_first_order",
            "pair_signal": pair_signal,
            "hardest_pair_indices": hardest[: min(100, pair_count)],
            "stats": {
                "avg_posenet_dist": payload.get("avg_posenet_dist"),
                "avg_segnet_dist": payload.get("avg_segnet_dist"),
                "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
            },
        }

    if payload.get("n_pairs") not in (None, pair_count):
        raise ValueError(f"{path}: n_pairs must be {pair_count}, got {payload.get('n_pairs')!r}")
    raw_signal = payload.get("per_pair_combined_score_signal")
    raw_pose = payload.get("per_pair_pose_dist")
    raw_seg = payload.get("per_pair_seg_dist")
    for pair in range(pair_count):
        if isinstance(raw_signal, list) and pair < len(raw_signal):
            pair_signal[pair] = max(0.0, _finite(raw_signal[pair], field=f"per_pair_combined_score_signal[{pair}]"))
        elif isinstance(raw_pose, list) and isinstance(raw_seg, list) and pair < len(raw_pose) and pair < len(raw_seg):
            pose[pair] = _finite(raw_pose[pair], field=f"per_pair_pose_dist[{pair}]")
            seg[pair] = _finite(raw_seg[pair], field=f"per_pair_seg_dist[{pair}]")
            pair_signal[pair] = 100.0 * seg[pair] + math.sqrt(max(0.0, 10.0 * pose[pair]))
        else:
            pair_signal[pair] = 1.0
    hardest = payload.get("hardest_pair_indices", [])
    if not isinstance(hardest, list):
        raise ValueError(f"{path}: hardest_pair_indices must be a list when provided")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "source_schema": payload.get("schema") or payload.get("schema_version"),
        "signal_source": "legacy_or_uniform_pair_signal",
        "pair_signal": pair_signal,
        "hardest_pair_indices": [int(i) for i in hardest if isinstance(i, int) and 0 <= i < pair_count],
        "stats": payload.get("stats", {}),
    }


def _to_alpha_runs(alpha_builder: Any, runs: Iterable[ResidualRun]) -> list[Any]:
    return [
        alpha_builder.RepairRun(
            frame_index=int(run.frame_index),
            y=int(run.y),
            x0=int(run.x0),
            length=int(run.length),
            class_id=int(run.class_id),
        )
        for run in runs
    ]


def _encode_repair_payload(
    *,
    alpha_builder: Any,
    runs: list[ResidualRun],
    shape: tuple[int, int, int],
    source_mask_sha256: str,
    candidate_mask_sha256: str,
    selection_meta: dict[str, Any],
) -> bytes:
    return alpha_builder._encode_repair_payload(  # noqa: SLF001 - reviewed local AMR1 encoder
        _to_alpha_runs(alpha_builder, runs),
        shape=shape,
        source_mask_sha256=source_mask_sha256,
        candidate_mask_sha256=candidate_mask_sha256,
        selection_meta=selection_meta,
    )


def _repair_selection_meta(
    *,
    policy_name: str,
    selected_runs: list[ResidualRun],
    total_residual_pixels: int,
    total_residual_runs: int,
    selected_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_pixels = sum(int(run.length) for run in selected_runs)
    return {
        "strategy": "cmg2_foveated_hard_pair_residual_repair_v1",
        "policy_name": policy_name,
        "policy_kind": "ranked_pair_class_fovea_atoms",
        "selected_atom_count": int(len(selected_atoms)),
        "selected_atoms": [
            {
                "atom_id": atom["atom_id"],
                "pair_index": atom["pair_index"],
                "class_id": atom["class_id"],
                "fovea_band": atom["fovea_band"],
            }
            for atom in selected_atoms
        ],
        "total_residual_pixels": int(total_residual_pixels),
        "total_residual_runs": int(total_residual_runs),
        "selected_repair_pixels": int(selected_pixels),
        "selected_repair_runs": int(len(selected_runs)),
        "residual_pixel_coverage": 0.0
        if total_residual_pixels == 0
        else round(selected_pixels / total_residual_pixels, 12),
        "partial_repair": bool(selected_pixels != total_residual_pixels),
        "fail_on_partial_repair": False,
        "score_claim": False,
    }


def build_atom_table(
    *,
    target: np.ndarray,
    candidate: np.ndarray,
    pair_priors: dict[str, Any],
    repair_compressor: str,
    inner_radius: float,
    mid_radius: float,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], list[ResidualRun]]]:
    alpha_builder = _load_module(ALPHA_BUILDER_PATH, "_cmg2_foveated_repair_alpha_builder")
    archive_builder = _load_module(ALPHA_ARCHIVE_BUILDER_PATH, "_cmg2_foveated_repair_alpha_archive_builder")
    groups = residual_runs_by_pair_class_band(
        target,
        candidate,
        inner_radius=inner_radius,
        mid_radius=mid_radius,
    )
    shape = tuple(int(v) for v in target.shape)
    source_sha = _sha256_bytes(target.tobytes(order="C"))
    candidate_sha = _sha256_bytes(candidate.tobytes(order="C"))
    total_pixels = int((target != candidate).sum())
    total_runs = sum(len(runs) for runs in groups.values())
    fovea_weight = {"fovea": 1.35, "mid": 1.0, "periphery": 0.65}
    hardest = set(int(i) for i in pair_priors.get("hardest_pair_indices", []))
    pair_signal = pair_priors["pair_signal"]
    atoms: list[dict[str, Any]] = []
    for pair, class_id, band in sorted(groups):
        runs = groups[(pair, class_id, band)]
        atom_id = f"pair{pair:03d}_class{class_id}_{band}"
        selected_pixels = sum(int(run.length) for run in runs)
        selection_meta = _repair_selection_meta(
            policy_name=f"atom_{atom_id}",
            selected_runs=runs,
            total_residual_pixels=total_pixels,
            total_residual_runs=total_runs,
            selected_atoms=[
                {
                    "atom_id": atom_id,
                    "pair_index": pair,
                    "class_id": class_id,
                    "fovea_band": band,
                }
            ],
        )
        raw_payload = _encode_repair_payload(
            alpha_builder=alpha_builder,
            runs=runs,
            shape=shape,
            source_mask_sha256=source_sha,
            candidate_mask_sha256=candidate_sha,
            selection_meta=selection_meta,
        )
        member_name, compressed = archive_builder._compress_repair_payload(raw_payload, repair_compressor)
        compressed_bytes = len(compressed)
        prior = float(pair_signal[pair]) * fovea_weight[band]
        if pair in hardest:
            prior *= 1.15
        atoms.append(
            {
                "atom_id": atom_id,
                "score_claim": False,
                "pair_index": int(pair),
                "class_id": int(class_id),
                "fovea_band": band,
                "selected_repair_pixels": int(selected_pixels),
                "selected_repair_runs": int(len(runs)),
                "residual_pixel_coverage": 0.0
                if total_pixels == 0
                else round(selected_pixels / total_pixels, 12),
                "raw_amr1_bytes": int(len(raw_payload)),
                "raw_amr1_sha256": _sha256_bytes(raw_payload),
                "compressed_member_name": member_name,
                "compressed_bytes": int(compressed_bytes),
                "compressed_sha256": _sha256_bytes(compressed),
                "rate_term_cost": round(25.0 * compressed_bytes / 37_545_489, 12),
                "pair_signal_prior": round(float(pair_signal[pair]), 12),
                "fovea_weight": fovea_weight[band],
                "hard_pair_prior_applied": bool(pair in hardest),
                "weighted_prior": round(prior, 12),
                "weighted_prior_per_compressed_byte": round(prior / compressed_bytes, 12)
                if compressed_bytes
                else None,
                "pixels_per_compressed_byte": round(selected_pixels / compressed_bytes, 12)
                if compressed_bytes
                else None,
            }
        )

    def sort_key(atom: dict[str, Any]) -> tuple[float, float, int, int, str]:
        return (
            float(atom["weighted_prior_per_compressed_byte"] or -1.0),
            float(atom["pixels_per_compressed_byte"] or -1.0),
            int(atom["selected_repair_pixels"]),
            -int(atom["compressed_bytes"]),
            str(atom["atom_id"]),
        )

    return sorted(atoms, key=sort_key, reverse=True), groups


def _selected_runs_for_atoms(
    groups: dict[tuple[int, int, str], list[ResidualRun]],
    atoms: list[dict[str, Any]],
) -> list[ResidualRun]:
    selected: list[ResidualRun] = []
    for atom in atoms:
        key = (int(atom["pair_index"]), int(atom["class_id"]), str(atom["fovea_band"]))
        selected.extend(groups[key])
    return selected


def _apply_runs(candidate: np.ndarray, runs: list[ResidualRun]) -> np.ndarray:
    repaired = candidate.copy()
    for run in runs:
        repaired[run.frame_index, run.y, run.x0 : run.x0 + run.length] = run.class_id
    return repaired


def _build_candidate_archive(
    *,
    output_dir: Path,
    policy_name: str,
    cmg2_source_members: dict[str, bytes],
    target: np.ndarray,
    candidate: np.ndarray,
    selected_atoms: list[dict[str, Any]],
    selected_runs: list[ResidualRun],
    repair_compressor: str,
    total_residual_pixels: int,
    total_residual_runs: int,
    force: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"candidate output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_builder = _load_module(ALPHA_BUILDER_PATH, "_cmg2_foveated_repair_alpha_builder_for_archive")
    archive_builder = _load_module(ALPHA_ARCHIVE_BUILDER_PATH, "_cmg2_foveated_repair_alpha_archive_for_archive")
    packer = _load_module(PACKER_PATH, "_cmg2_foveated_repair_packer")
    shape = tuple(int(v) for v in target.shape)
    source_sha = _sha256_bytes(target.tobytes(order="C"))
    candidate_sha = _sha256_bytes(candidate.tobytes(order="C"))
    selection_meta = _repair_selection_meta(
        policy_name=policy_name,
        selected_runs=selected_runs,
        total_residual_pixels=total_residual_pixels,
        total_residual_runs=total_residual_runs,
        selected_atoms=selected_atoms,
    )
    raw_payload = _encode_repair_payload(
        alpha_builder=alpha_builder,
        runs=selected_runs,
        shape=shape,
        source_mask_sha256=source_sha,
        candidate_mask_sha256=candidate_sha,
        selection_meta=selection_meta,
    )
    repair_member_name, repair_member_bytes = archive_builder._compress_repair_payload(
        raw_payload,
        repair_compressor,
    )
    source_archive = output_dir / "cmg2_repair_source_members.zip"
    _write_source_archive(
        source_archive,
        [
            ("renderer.bin", cmg2_source_members["renderer.bin"]),
            ("masks.cmg2", cmg2_source_members["masks.cmg2"]),
            ("optimized_poses.bin", cmg2_source_members["optimized_poses.bin"]),
            (repair_member_name, repair_member_bytes),
        ],
    )
    archive_path = output_dir / "archive.zip"
    packed_meta = packer.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=packer.POSE_QP1_CODEC,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )
    repaired = _apply_runs(candidate, selected_runs)
    manifest = {
        "schema": f"{SCHEMA}.candidate",
        "policy_name": policy_name,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "repair": {
            "archive_member": repair_member_name,
            "compressor": repair_compressor,
            "raw_amr1_bytes": int(len(raw_payload)),
            "raw_amr1_sha256": _sha256_bytes(raw_payload),
            "compressed_bytes": int(len(repair_member_bytes)),
            "compressed_sha256": _sha256_bytes(repair_member_bytes),
            "selection": selection_meta,
        },
        "mask_tensor_shas": {
            "target_frontier_sha256": source_sha,
            "cmg2_candidate_sha256": candidate_sha,
            "repaired_candidate_sha256": _sha256_bytes(repaired.tobytes(order="C")),
        },
        "agreement": {
            "residual_pixels_before": int(total_residual_pixels),
            "residual_pixels_after": int((target != repaired).sum()),
            "pixel_disagreement_before": round(float((target != candidate).mean()), 12),
            "pixel_disagreement_after": round(float((target != repaired).mean()), 12),
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256_file(source_archive),
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
        },
        "packed_payload": packed_meta,
    }
    (output_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _policy_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(token) for token in value.split(",") if token.strip())
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return counts


def build_candidates(
    *,
    cmg2_manifest_path: Path,
    output_dir: Path,
    component_trace: Path | None,
    repair_compressor: str,
    top_policy_counts: tuple[int, ...],
    max_atoms: int,
    plan_only: bool,
    force: bool,
    inner_radius: float = 0.35,
    mid_radius: float = 0.70,
) -> dict[str, Any]:
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")
    cmg2_manifest_path = cmg2_manifest_path.resolve()
    cmg2_manifest = _load_cmg2_manifest(cmg2_manifest_path)
    target, candidate, mask_meta = load_full_and_cmg2_masks(
        cmg2_manifest_path=cmg2_manifest_path,
        manifest=cmg2_manifest,
    )
    pair_count = int(target.shape[0] + 1) // 2
    pair_priors = _load_pair_priors(component_trace.resolve() if component_trace is not None else None, pair_count=pair_count)
    atoms, groups = build_atom_table(
        target=target,
        candidate=candidate,
        pair_priors=pair_priors,
        repair_compressor=repair_compressor,
        inner_radius=inner_radius,
        mid_radius=mid_radius,
    )
    total_residual_pixels = int((target != candidate).sum())
    total_residual_runs = sum(len(runs) for runs in groups.values())
    top_atoms = atoms[:max_atoms]
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmg2_source_record = cmg2_manifest.get("source_archive")
    if not isinstance(cmg2_source_record, dict):
        raise ValueError("CMG2 manifest missing source_archive")
    cmg2_source_path = _resolve_existing(str(cmg2_source_record.get("path", "")), base_dir=cmg2_manifest_path.parent)
    cmg2_source_members = _read_source_members(cmg2_source_path)
    if _sha256_file(cmg2_source_path) != cmg2_source_record.get("sha256"):
        raise ValueError("CMG2 source archive SHA mismatch against manifest")

    policies: list[dict[str, Any]] = []
    candidate_manifests: list[dict[str, Any]] = []
    seen_policy_keys: set[tuple[str, ...]] = set()
    for count in top_policy_counts:
        selected_atoms = top_atoms[: min(count, len(top_atoms))]
        if not selected_atoms:
            continue
        policy_key = tuple(str(atom["atom_id"]) for atom in selected_atoms)
        if policy_key in seen_policy_keys:
            continue
        seen_policy_keys.add(policy_key)
        selected_runs = _selected_runs_for_atoms(groups, selected_atoms)
        estimated_atom_bytes = sum(int(atom["compressed_bytes"]) for atom in selected_atoms)
        policy_name = f"top{len(selected_atoms):03d}_foveated_hardpair_atoms"
        policy = {
            "policy_name": policy_name,
            "selected_atom_count": int(len(selected_atoms)),
            "selected_atom_ids": [atom["atom_id"] for atom in selected_atoms],
            "selected_pair_indices": sorted({int(atom["pair_index"]) for atom in selected_atoms}),
            "selected_repair_pixels": int(sum(run.length for run in selected_runs)),
            "selected_repair_runs": int(len(selected_runs)),
            "estimated_atom_compressed_bytes_sum": int(estimated_atom_bytes),
            "note": "Atom-local bytes are not additive; built archive bytes are authoritative.",
        }
        policies.append(policy)
        if not plan_only:
            candidate_manifest = _build_candidate_archive(
                output_dir=output_dir / policy_name,
                policy_name=policy_name,
                cmg2_source_members=cmg2_source_members,
                target=target,
                candidate=candidate,
                selected_atoms=selected_atoms,
                selected_runs=selected_runs,
                repair_compressor=repair_compressor,
                total_residual_pixels=total_residual_pixels,
                total_residual_runs=total_residual_runs,
                force=force,
            )
            policy["built_archive"] = candidate_manifest["output_archive"]
            policy["built_repair"] = candidate_manifest["repair"]
            policy["built_agreement"] = candidate_manifest["agreement"]
            candidate_manifests.append(candidate_manifest)

    plan = {
        "schema": SCHEMA,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": "experiments/build_cmg2_foveated_repair_candidates.py",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "cmg2_manifest": {
            "path": str(cmg2_manifest_path),
            "sha256": _sha256_file(cmg2_manifest_path),
        },
        "cmg2_source_archive": {
            "path": str(cmg2_source_path),
            "bytes": cmg2_source_path.stat().st_size,
            "sha256": _sha256_file(cmg2_source_path),
        },
        "mask_inputs": mask_meta,
        "component_trace": {
            "path": pair_priors["path"],
            "sha256": pair_priors["sha256"],
            "source_schema": pair_priors["source_schema"],
            "signal_source": pair_priors["signal_source"],
            "hardest_pair_indices": pair_priors["hardest_pair_indices"],
            "stats": pair_priors["stats"],
        },
        "repair_compressor": repair_compressor,
        "fovea_policy": {
            "inner_radius": inner_radius,
            "mid_radius": mid_radius,
            "weights": {"fovea": 1.35, "mid": 1.0, "periphery": 0.65},
        },
        "residual_summary": {
            "total_residual_pixels": total_residual_pixels,
            "total_residual_runs": total_residual_runs,
            "atom_count": len(atoms),
            "top_atom_count": len(top_atoms),
        },
        "top_atoms": top_atoms,
        "recommended_policies": policies,
        "candidate_manifests": [
            {
                "path": str(Path(item["output_archive"]["path"]).parent / "build_manifest.json"),
                "archive": item["output_archive"],
            }
            for item in candidate_manifests
        ],
        "plan_only": bool(plan_only),
        "required_next_steps": [
            "byte-screen built archive bytes and residual coverage from build_manifest.json",
            "run local unpack/inflate smoke only for runtime routing validation",
            "run experiments/contest_auth_eval.py --device cuda before any score claim",
        ],
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }
    (output_dir / PLAN_NAME).write_bytes(_json_bytes(plan))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmg2-manifest", type=Path, default=DEFAULT_CMG2_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--component-trace",
        type=Path,
        default=DEFAULT_COMPONENT_TRACE if DEFAULT_COMPONENT_TRACE.exists() else None,
    )
    parser.add_argument(
        "--repair-compressor",
        choices=("raw", "zlib", "lzma_xz", "brotli"),
        default="lzma_xz",
    )
    parser.add_argument("--top-policy-counts", type=_policy_counts, default=(4, 8, 16, 32))
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--inner-radius", type=float, default=0.35)
    parser.add_argument("--mid-radius", type=float, default=0.70)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_candidates(
        cmg2_manifest_path=args.cmg2_manifest,
        output_dir=args.output_dir,
        component_trace=args.component_trace,
        repair_compressor=args.repair_compressor,
        top_policy_counts=tuple(args.top_policy_counts),
        max_atoms=args.max_atoms,
        plan_only=bool(args.plan_only),
        force=bool(args.force),
        inner_radius=float(args.inner_radius),
        mid_radius=float(args.mid_radius),
    )
    print(
        json.dumps(
            {
                "plan": str(Path(args.output_dir).resolve() / PLAN_NAME),
                "atom_count": plan["residual_summary"]["atom_count"],
                "policies": len(plan["recommended_policies"]),
                "built_archives": len(plan["candidate_manifests"]),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
