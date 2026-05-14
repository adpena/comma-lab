#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a charged postdecode AMR1 mask-repair candidate for C067.

This builder consumes C067 runtime members plus a lossy legacy ``masks.mkv``
candidate and emits a deterministic contest archive:

    renderer.bin
    masks.mkv
    alpha4_residual_repair.amr1[.zlib|.xz|.br]
    optimized_poses.bin

The repair payload is charged inside ``archive.zip`` and uses the existing
AMR1 inflate contract. It does not run scorers and always records
``score_claim=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.submission_archive import (  # noqa: E402
    validate_archive_member_name,
    write_deterministic_zip_member,
)


ALPHA_BUILDER_PATH = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
PROTECTED_BUILDER_PATH = REPO_ROOT / "experiments" / "build_protected_mask_reencode_candidate.py"
SCHEMA = "c067_postdecode_mask_repair_candidate_v1"
TOOL = "experiments/build_c067_postdecode_mask_repair_candidate.py"
DEFAULT_BASE_RUNTIME_DIR = REPO_ROOT / "experiments/results/c067_fixedslice_unpacked_runtime_20260502/unpacked"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c067_postdecode_mask_repair_candidate_20260502"
REPAIR_MEMBER_RAW = "alpha4_residual_repair.amr1"
REPAIR_COMPRESSORS = ("raw", "zlib", "lzma_xz", "brotli")
RUNTIME_MEMBERS = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
ORIGINAL_VIDEO_BYTES = 37_545_489
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class RepairAtom:
    atom_id: str
    frames: tuple[int, ...]
    pair_indices: tuple[int, ...]
    class_id: int
    changed_pixels: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "frames": [int(v) for v in self.frames],
            "pair_indices": [int(v) for v in self.pair_indices],
            "class_id": int(self.class_id),
            "changed_pixels": int(self.changed_pixels),
        }


@dataclass(frozen=True)
class RepairAtomPolicy:
    policy: str = "top_pixels"
    atom_granularity: str = "pair_class"
    max_atoms: int | None = 256
    max_repair_payload_bytes: int | None = None
    pair_indices: tuple[int, ...] = ()
    frame_indices: tuple[int, ...] = ()
    class_ids: tuple[int, ...] = ()
    label: str = "c067_postdecode_amr1_repair"

    def as_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "policy": self.policy,
            "atom_granularity": self.atom_granularity,
            "max_atoms": self.max_atoms,
            "max_repair_payload_bytes": self.max_repair_payload_bytes,
            "pair_indices": [int(v) for v in self.pair_indices],
            "frame_indices": [int(v) for v in self.frame_indices],
            "class_ids": [int(v) for v in self.class_ids],
        }


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_alpha_builder() -> Any:
    return _load_module("alpha_mask_candidate_builder_for_c067_postdecode_repair", ALPHA_BUILDER_PATH)


def _load_protected_builder() -> Any:
    return _load_module("protected_mask_reencode_builder_for_c067_postdecode_repair", PROTECTED_BUILDER_PATH)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _parse_int_set(value: str | None, *, field: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    values: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"{field} range has end before start: {token!r}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if any(v < 0 for v in values):
        raise ValueError(f"{field} values must be nonnegative")
    return tuple(sorted(values))


def _read_runtime_members_from_dir(runtime_dir: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    for name in RUNTIME_MEMBERS:
        validate_archive_member_name(name)
        path = runtime_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"runtime dir missing required member {name!r}: {path}")
        data = path.read_bytes()
        members[name] = data
        inventory.append({"name": name, "size_bytes": len(data), "sha256": _sha256_bytes(data)})
    return members, inventory


def _read_runtime_members_from_archive(
    archive: Path,
    *,
    source_mask_member: str = "masks.mkv",
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    allowed = {"renderer.bin", source_mask_member, "optimized_poses.bin"}
    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        seen: set[str] = set()
        for info in zf.infolist():
            name = validate_archive_member_name(info.filename)
            if info.is_dir():
                raise ValueError(f"directory member not allowed in runtime archive: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate runtime archive member: {name!r}")
            if name not in allowed:
                raise ValueError(f"unexpected runtime archive member {name!r}; allowed={sorted(allowed)!r}")
            seen.add(name)
            data = zf.read(info)
            canonical_name = "masks.mkv" if name == source_mask_member else name
            members[canonical_name] = data
            inventory.append(
                {
                    "name": name,
                    "canonical_name": canonical_name,
                    "size_bytes": len(data),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    missing = [name for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin") if name not in members]
    if missing:
        raise ValueError(f"runtime archive missing required member(s): {missing}")
    return members, inventory


def _read_lossy_mask_stream(*, lossy_mask: Path | None, lossy_archive: Path | None, member: str) -> tuple[bytes, dict[str, Any]]:
    if (lossy_mask is None) == (lossy_archive is None):
        raise ValueError("exactly one of lossy_mask or lossy_archive must be provided")
    if lossy_mask is not None:
        data = lossy_mask.read_bytes()
        return data, {
            "source": "file",
            "path": str(lossy_mask.resolve()),
            "member": member,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
    assert lossy_archive is not None
    wanted = validate_archive_member_name(member)
    with zipfile.ZipFile(lossy_archive, "r") as zf:
        seen = [info for info in zf.infolist() if info.filename == wanted]
        if len(seen) != 1:
            raise ValueError(f"expected exactly one {wanted!r} in lossy archive, found {len(seen)}")
        data = zf.read(seen[0])
    return data, {
        "source": "archive",
        "path": str(lossy_archive.resolve()),
        "member": wanted,
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _decode_legacy_mask_stream(data: bytes, *, member: str, max_frames: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    protected = _load_protected_builder()
    return protected._decode_source_masks(data, member, max_frames=max_frames)


def _tensor_u8_sha256(masks: torch.Tensor) -> str:
    alpha = _load_alpha_builder()
    return alpha._tensor_u8_sha256(masks)


def _validate_masks(masks: torch.Tensor) -> torch.Tensor:
    alpha = _load_alpha_builder()
    return alpha._validate_masks(masks)


def _compress_repair_payload(raw_payload: bytes, compressor: str) -> tuple[str, bytes]:
    if compressor not in REPAIR_COMPRESSORS:
        raise ValueError(f"unknown repair compressor {compressor!r}")
    if compressor == "raw":
        return REPAIR_MEMBER_RAW, raw_payload
    if compressor == "zlib":
        return f"{REPAIR_MEMBER_RAW}.zlib", zlib.compress(raw_payload, level=9)
    if compressor == "lzma_xz":
        return f"{REPAIR_MEMBER_RAW}.xz", lzma.compress(
            raw_payload,
            format=lzma.FORMAT_XZ,
            preset=9 | lzma.PRESET_EXTREME,
        )
    try:
        import brotli  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("brotli repair compression requested but brotli is unavailable") from exc
    return f"{REPAIR_MEMBER_RAW}.br", brotli.compress(raw_payload, quality=11, lgwin=24)


def _diff_atom_summaries(
    source: torch.Tensor,
    candidate: torch.Tensor,
    *,
    granularity: str,
) -> tuple[list[RepairAtom], dict[str, Any]]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    if tuple(source.shape) != tuple(candidate.shape):
        raise ValueError(f"source mask shape {tuple(source.shape)} != candidate {tuple(candidate.shape)}")
    src = source.cpu().numpy().astype(np.uint8, copy=False)
    cand = candidate.cpu().numpy().astype(np.uint8, copy=False)
    diff = src != cand
    total_pixels = int(diff.sum())
    atoms: dict[tuple[int, int], int] = {}
    if total_pixels:
        frames, _ys, _xs = np.nonzero(diff)
        classes = src[diff].astype(np.int64, copy=False)
        if granularity == "pair_class":
            groups = frames.astype(np.int64) // 2
        elif granularity == "frame_class":
            groups = frames.astype(np.int64)
        else:
            raise ValueError("atom_granularity must be one of pair_class or frame_class")
        for group_id, class_id in zip(groups.tolist(), classes.tolist(), strict=True):
            key = (int(group_id), int(class_id))
            atoms[key] = atoms.get(key, 0) + 1

    out: list[RepairAtom] = []
    for (group_id, class_id), pixels in atoms.items():
        if granularity == "pair_class":
            frames_tuple = (2 * group_id, 2 * group_id + 1)
            pairs_tuple = (group_id,)
            atom_id = f"pair{group_id:04d}_class{class_id}"
        else:
            frames_tuple = (group_id,)
            pairs_tuple = (group_id // 2,)
            atom_id = f"frame{group_id:04d}_class{class_id}"
        out.append(
            RepairAtom(
                atom_id=atom_id,
                frames=frames_tuple,
                pair_indices=pairs_tuple,
                class_id=class_id,
                changed_pixels=pixels,
            )
        )
    out.sort(key=lambda atom: (-atom.changed_pixels, atom.atom_id))
    stats = {
        "total_residual_pixels": total_pixels,
        "total_atoms": len(out),
        "atom_granularity": granularity,
    }
    return out, stats


def _select_atoms(atoms: list[RepairAtom], policy: RepairAtomPolicy) -> tuple[list[RepairAtom], dict[str, Any]]:
    if policy.policy not in {"full", "top_pixels", "pair_indices", "frame_indices", "class_ids"}:
        raise ValueError("policy must be one of full, top_pixels, pair_indices, frame_indices, class_ids")
    selected = list(atoms)
    if policy.policy == "pair_indices":
        requested = set(policy.pair_indices)
        if not requested:
            raise ValueError("pair_indices policy requires at least one pair index")
        selected = [atom for atom in selected if any(pair in requested for pair in atom.pair_indices)]
    elif policy.policy == "frame_indices":
        requested = set(policy.frame_indices)
        if not requested:
            raise ValueError("frame_indices policy requires at least one frame index")
        selected = [atom for atom in selected if any(frame in requested for frame in atom.frames)]
    elif policy.policy == "class_ids":
        requested = set(policy.class_ids)
        if not requested:
            raise ValueError("class_ids policy requires at least one class id")
        selected = [atom for atom in selected if atom.class_id in requested]

    if policy.policy == "top_pixels" and policy.max_atoms is not None:
        if policy.max_atoms < 0:
            raise ValueError("max_atoms must be nonnegative")
        selected = selected[: policy.max_atoms]

    selected_ids = {atom.atom_id for atom in selected}
    summary = {
        "policy": policy.as_manifest(),
        "selected_atom_count": len(selected),
        "selected_repair_pixels_before_rle": int(sum(atom.changed_pixels for atom in selected)),
        "selected_atoms": [atom.as_manifest() for atom in selected],
        "rejected_atom_count": len(atoms) - len(selected_ids),
    }
    return selected, summary


def _build_repair_runs(
    source: torch.Tensor,
    candidate: torch.Tensor,
    selected_atoms: list[RepairAtom],
    *,
    granularity: str,
) -> list[Any]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    src = source.cpu().numpy().astype(np.uint8, copy=False)
    cand = candidate.cpu().numpy().astype(np.uint8, copy=False)
    selected_ids = {atom.atom_id for atom in selected_atoms}
    alpha = _load_alpha_builder()
    runs: list[Any] = []
    t, h, w = [int(v) for v in src.shape]
    for frame_index in range(t):
        group_id = frame_index // 2 if granularity == "pair_class" else frame_index
        for y in range(h):
            x = 0
            while x < w:
                class_id = int(src[frame_index, y, x])
                atom_id = (
                    f"pair{group_id:04d}_class{class_id}"
                    if granularity == "pair_class"
                    else f"frame{group_id:04d}_class{class_id}"
                )
                if src[frame_index, y, x] == cand[frame_index, y, x] or atom_id not in selected_ids:
                    x += 1
                    continue
                x0 = x
                x += 1
                while (
                    x < w
                    and src[frame_index, y, x] != cand[frame_index, y, x]
                    and int(src[frame_index, y, x]) == class_id
                ):
                    x += 1
                runs.append(
                    alpha.RepairRun(
                        frame_index=frame_index,
                        y=y,
                        x0=x0,
                        length=x - x0,
                        class_id=class_id,
                    )
                )
    return runs


def _build_repair_runs_by_atom(
    source: torch.Tensor,
    candidate: torch.Tensor,
    selected_atoms: list[RepairAtom],
    *,
    granularity: str,
) -> dict[str, list[Any]]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    src = source.cpu().numpy().astype(np.uint8, copy=False)
    cand = candidate.cpu().numpy().astype(np.uint8, copy=False)
    selected_ids = {atom.atom_id for atom in selected_atoms}
    alpha = _load_alpha_builder()
    runs_by_atom: dict[str, list[Any]] = {atom.atom_id: [] for atom in selected_atoms}
    t, h, w = [int(v) for v in src.shape]
    for frame_index in range(t):
        group_id = frame_index // 2 if granularity == "pair_class" else frame_index
        for y in range(h):
            x = 0
            while x < w:
                class_id = int(src[frame_index, y, x])
                atom_id = (
                    f"pair{group_id:04d}_class{class_id}"
                    if granularity == "pair_class"
                    else f"frame{group_id:04d}_class{class_id}"
                )
                if src[frame_index, y, x] == cand[frame_index, y, x] or atom_id not in selected_ids:
                    x += 1
                    continue
                x0 = x
                x += 1
                while (
                    x < w
                    and src[frame_index, y, x] != cand[frame_index, y, x]
                    and int(src[frame_index, y, x]) == class_id
                ):
                    x += 1
                runs_by_atom[atom_id].append(
                    alpha.RepairRun(
                        frame_index=frame_index,
                        y=y,
                        x0=x0,
                        length=x - x0,
                        class_id=class_id,
                    )
                )
    return runs_by_atom


def _flatten_runs_for_atoms(runs_by_atom: dict[str, list[Any]], atoms: list[RepairAtom]) -> list[Any]:
    runs: list[Any] = []
    for atom in atoms:
        runs.extend(runs_by_atom.get(atom.atom_id, ()))
    return runs


def _archive_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            data = zf.read(info)
            rows.append(
                {
                    "name": info.filename,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "compress_type": int(info.compress_type),
                    "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                    "sha256": _sha256_bytes(data),
                }
            )
    return rows


def _write_candidate_archive(output_archive: Path, members: list[tuple[str, bytes]]) -> None:
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in members:
            write_deterministic_zip_member(zf, name, data, compresslevel=9)


def build_candidate(
    *,
    base_runtime_dir: Path | None = DEFAULT_BASE_RUNTIME_DIR,
    base_archive: Path | None = None,
    lossy_mask: Path | None = None,
    lossy_archive: Path | None = None,
    lossy_mask_member: str = "masks.mkv",
    output_archive: Path,
    manifest_json: Path | None = None,
    policy: RepairAtomPolicy = RepairAtomPolicy(),
    repair_compressor: str = "lzma_xz",
    max_frames: int | None = None,
) -> dict[str, Any]:
    if (base_runtime_dir is None) == (base_archive is None):
        raise ValueError("exactly one of base_runtime_dir or base_archive must be provided")
    if base_archive is not None:
        runtime_members, runtime_inventory = _read_runtime_members_from_archive(base_archive)
        base_record = {
            "source": "archive",
            "path": str(base_archive.resolve()),
            "size_bytes": base_archive.stat().st_size,
            "sha256": _sha256_file(base_archive),
        }
    else:
        assert base_runtime_dir is not None
        runtime_members, runtime_inventory = _read_runtime_members_from_dir(base_runtime_dir)
        base_record = {"source": "runtime_dir", "path": str(base_runtime_dir.resolve())}

    lossy_mask_bytes, lossy_record = _read_lossy_mask_stream(
        lossy_mask=lossy_mask,
        lossy_archive=lossy_archive,
        member=lossy_mask_member,
    )
    source_masks, source_decode = _decode_legacy_mask_stream(
        runtime_members["masks.mkv"],
        member="masks.mkv",
        max_frames=max_frames,
    )
    candidate_masks, candidate_decode = _decode_legacy_mask_stream(
        lossy_mask_bytes,
        member=lossy_mask_member,
        max_frames=max_frames,
    )
    source_masks = _validate_masks(source_masks)
    candidate_masks = _validate_masks(candidate_masks)
    if tuple(source_masks.shape) != tuple(candidate_masks.shape):
        raise ValueError(
            f"source mask shape {tuple(source_masks.shape)} != lossy candidate {tuple(candidate_masks.shape)}"
        )

    all_atoms, residual_stats = _diff_atom_summaries(
        source_masks,
        candidate_masks,
        granularity=policy.atom_granularity,
    )
    selected_atoms, selector_summary = _select_atoms(all_atoms, policy)
    runs_by_atom = _build_repair_runs_by_atom(
        source_masks,
        candidate_masks,
        selected_atoms,
        granularity=policy.atom_granularity,
    )
    alpha = _load_alpha_builder()
    source_sha = _tensor_u8_sha256(source_masks)
    candidate_sha = _tensor_u8_sha256(candidate_masks)
    source_shape = tuple(int(value) for value in source_masks.shape)

    def materialize_payload(atoms: list[RepairAtom]) -> tuple[list[Any], dict[str, Any], bytes, str, bytes]:
        runs = _flatten_runs_for_atoms(runs_by_atom, atoms)
        selected_pixels = int(sum(int(run.length) for run in runs))
        selected_run_count = int(len(runs))
        partial_repair = selected_pixels != int(residual_stats["total_residual_pixels"])
        selection_meta = {
            "strategy": "c067_postdecode_mask_repair_atoms_v1",
            "score_claim": False,
            "policy": policy.as_manifest(),
            "total_residual_pixels": int(residual_stats["total_residual_pixels"]),
            "total_candidate_atoms": int(residual_stats["total_atoms"]),
            "selected_repair_pixels": selected_pixels,
            "selected_repair_runs": selected_run_count,
            "selected_atom_count": int(len(atoms)),
            "selected_atoms": [atom.as_manifest() for atom in atoms],
            "partial_repair": partial_repair,
            "fail_on_partial_repair": False,
        }
        raw_payload = alpha._encode_repair_payload(
            runs,
            shape=source_shape,
            source_mask_sha256=source_sha,
            candidate_mask_sha256=candidate_sha,
            selection_meta=selection_meta,
        )
        repair_member, repair_bytes = _compress_repair_payload(raw_payload, repair_compressor)
        return runs, selection_meta, raw_payload, repair_member, repair_bytes

    budget_summary: dict[str, Any] = {
        "budget_applied": False,
        "candidate_atoms_before_budget": int(len(selected_atoms)),
        "max_repair_payload_bytes": policy.max_repair_payload_bytes,
        "atoms_rejected_by_budget": 0,
    }
    if policy.max_repair_payload_bytes is not None:
        if policy.max_repair_payload_bytes < 0:
            raise ValueError("max_repair_payload_bytes must be nonnegative")
        low = 0
        high = len(selected_atoms)
        best: tuple[int, list[Any], dict[str, Any], bytes, str, bytes] | None = None
        while low <= high:
            mid = (low + high) // 2
            runs_mid, meta_mid, raw_mid, member_mid, bytes_mid = materialize_payload(selected_atoms[:mid])
            if len(bytes_mid) <= policy.max_repair_payload_bytes:
                best = (mid, runs_mid, meta_mid, raw_mid, member_mid, bytes_mid)
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            raise ValueError(
                "empty AMR1 repair header exceeds max_repair_payload_bytes="
                f"{policy.max_repair_payload_bytes}"
            )
        best_count, runs, selection_meta, raw_payload, repair_member, repair_bytes = best
        selected_atoms = selected_atoms[:best_count]
        budget_summary = {
            "budget_applied": True,
            "candidate_atoms_before_budget": int(selector_summary["selected_atom_count"]),
            "max_repair_payload_bytes": int(policy.max_repair_payload_bytes),
            "atoms_rejected_by_budget": int(selector_summary["selected_atom_count"] - len(selected_atoms)),
            "compressed_repair_bytes_after_budget": int(len(repair_bytes)),
        }
    else:
        runs, selection_meta, raw_payload, repair_member, repair_bytes = materialize_payload(selected_atoms)

    selected_pixels = int(selection_meta["selected_repair_pixels"])
    selected_run_count = int(selection_meta["selected_repair_runs"])
    partial_repair = bool(selection_meta["partial_repair"])
    selector_summary = {
        **selector_summary,
        "selected_atom_count": int(len(selected_atoms)),
        "selected_repair_pixels_before_rle": selected_pixels,
        "selected_atoms": [atom.as_manifest() for atom in selected_atoms],
        "rejected_atom_count": int(len(all_atoms) - len({atom.atom_id for atom in selected_atoms})),
        "compressed_byte_budget": budget_summary,
    }

    archive_members = [
        ("renderer.bin", runtime_members["renderer.bin"]),
        ("masks.mkv", lossy_mask_bytes),
        (repair_member, repair_bytes),
        ("optimized_poses.bin", runtime_members["optimized_poses.bin"]),
    ]
    _write_candidate_archive(output_archive, archive_members)

    archive_size = output_archive.stat().st_size
    base_archive_bytes = (
        base_archive.stat().st_size
        if base_archive is not None
        else sum(len(runtime_members[name]) for name in RUNTIME_MEMBERS)
    )
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "cuda_jobs_launched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "archive": {
            "path": str(output_archive.resolve()),
            "size_bytes": archive_size,
            "sha256": _sha256_file(output_archive),
            "members": _archive_inventory(output_archive),
            "delta_vs_base_bytes": int(archive_size - base_archive_bytes),
            "rate_term_delta_vs_base": 25.0 * (archive_size - base_archive_bytes) / ORIGINAL_VIDEO_BYTES,
        },
        "base": {
            **base_record,
            "runtime_member_inventory": runtime_inventory,
            "source_mask_stream": {
                "member": "masks.mkv",
                "size_bytes": len(runtime_members["masks.mkv"]),
                "sha256": _sha256_bytes(runtime_members["masks.mkv"]),
                "decoded_class_u8_sha256": source_sha,
                "decode": source_decode,
            },
        },
        "lossy_mask_candidate": {
            **lossy_record,
            "archive_member": "masks.mkv",
            "decoded_class_u8_sha256": candidate_sha,
            "decode": candidate_decode,
        },
        "repair_selector": {
            **residual_stats,
            **selector_summary,
            "selected_repair_runs": selected_run_count,
            "selected_repair_pixels": selected_pixels,
            "partial_repair": partial_repair,
        },
        "repair_payload": {
            "archive_member": repair_member,
            "compressor": repair_compressor,
            "raw_amr1_size_bytes": len(raw_payload),
            "raw_amr1_sha256": _sha256_bytes(raw_payload),
            "compressed_size_bytes": len(repair_bytes),
            "compressed_sha256": _sha256_bytes(repair_bytes),
            "selection": selection_meta,
        },
        "charged_member_accounting": {
            repair_member: {
                "role": "charged_postdecode_amr1_mask_repair",
                "bytes": len(repair_bytes),
                "sha256": _sha256_bytes(repair_bytes),
            }
        },
        "runtime_contract": {
            "archive_members": [name for name, _data in archive_members],
            "legacy_masks_member": "masks.mkv",
            "repair_member": repair_member,
            "inflate_hook": (
                "submissions/robust_current/inflate_renderer.py applies charged AMR1 repair "
                "after decoding legacy masks.mkv when grayscale.mkv is absent"
            ),
            "payload_closure": "repair payload is inside archive.zip and source-affecting sidecars are forbidden",
        },
    }
    if manifest_json is None:
        manifest_json = output_archive.with_name("c067_postdecode_mask_repair_manifest.json")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--base-runtime-dir", type=Path, default=DEFAULT_BASE_RUNTIME_DIR)
    base.add_argument("--base-archive", type=Path)
    lossy = parser.add_mutually_exclusive_group(required=True)
    lossy.add_argument("--lossy-mask", type=Path)
    lossy.add_argument("--lossy-archive", type=Path)
    parser.add_argument("--lossy-mask-member", default="masks.mkv")
    parser.add_argument("--output-archive", type=Path, default=DEFAULT_OUTPUT_DIR / "archive.zip")
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--repair-compressor", choices=REPAIR_COMPRESSORS, default="lzma_xz")
    parser.add_argument("--policy", choices=("full", "top_pixels", "pair_indices", "frame_indices", "class_ids"), default="top_pixels")
    parser.add_argument("--atom-granularity", choices=("pair_class", "frame_class"), default="pair_class")
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument(
        "--max-repair-payload-bytes",
        type=int,
        help=(
            "Choose the largest deterministic prefix of the selected atom order "
            "whose compressed AMR1 repair member fits this byte budget."
        ),
    )
    parser.add_argument("--pair-indices", default="")
    parser.add_argument("--frame-indices", default="")
    parser.add_argument("--class-ids", default="")
    parser.add_argument("--label", default="c067_postdecode_amr1_repair")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    policy = RepairAtomPolicy(
        policy=args.policy,
        atom_granularity=args.atom_granularity,
        max_atoms=None if args.max_atoms < 0 else int(args.max_atoms),
        max_repair_payload_bytes=args.max_repair_payload_bytes,
        pair_indices=_parse_int_set(args.pair_indices, field="pair_indices"),
        frame_indices=_parse_int_set(args.frame_indices, field="frame_indices"),
        class_ids=_parse_int_set(args.class_ids, field="class_ids"),
        label=args.label,
    )
    manifest = build_candidate(
        base_runtime_dir=None if args.base_archive is not None else args.base_runtime_dir,
        base_archive=args.base_archive,
        lossy_mask=args.lossy_mask,
        lossy_archive=args.lossy_archive,
        lossy_mask_member=args.lossy_mask_member,
        output_archive=args.output_archive,
        manifest_json=args.manifest_json,
        policy=policy,
        repair_compressor=args.repair_compressor,
        max_frames=args.max_frames,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
