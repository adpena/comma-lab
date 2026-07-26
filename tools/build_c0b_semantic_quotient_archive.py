#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the original E1 semantic-base plus exact C1 two-plane quotient.

The full command is intentionally fixed to the original-work-only donors.  It
never imports a scorer and never evaluates a score.  The counted E1 semantic
packet is rendered by the hash-bound generic DDM receiver, downsampled with the
same exact integer operator as the production V10 receiver, and differenced
against all 50 custody-bound C1 target chunks.  The resulting dense quotient
is a non-promotable scientific seam baseline.

A full invocation is storage-heavy and is not run by this file's unit tests.
It is crash-resumable through ``--resume-from`` and immutable per-stage
checkpoints.  Optional success-only cleanup certifies assembled raw and
extracted packet copies before deleting them while preserving every stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.witness_dsl.c0b_semantic_quotient import (  # noqa: E402
    TARGET_TEACHER_CUSTODY_SCHEMA,
    PlaneChunk,
    RendererIdentity,
    SemanticQuotientError,
    build_semantic_quotient_archive,
    canonical_json,
    exact_resize_round_u8,
    parse_semantic_quotient_archive,
    sha256_file,
    storage_preflight,
    write_once_or_equal,
)

PAIR_COUNT = 600
CHUNK_PAIRS = 12
CHUNK_COUNT = 50
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
CHANNELS = 3
E1_PACKET_SCHEMA = "ddm_e1_runtime_archive.v1"
E1_PACKET_SHA256 = "05775433089d6aa2ae6800f2f8551358252d91288dcc1f1dbbfcc0d5517f26c1"
E1_CAMERA_RAW_SHA256 = "5936308b2a37221ed33f743463889c66f0f59863045cb753104922ec295ac838"
E1_RENDERER_ID = "tac.optimization.ddm_runtime_receiver.inflate:e1.v1"
E1_RENDERER_SOURCE_SHA256 = "0cb60d6a4ac2a19ba2a369bee6ee09cc8ab59c9199b2ba98e2a62650c1c1f2ee"
E1_PACKET_DEFAULT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/ddm_e1_runtime_exporter_20260723/upstream_harness/submission/archive.zip"
)
C1_ROOT_DEFAULT = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719")
C1_PREPARE_RECEIPT_SHA256 = "f5a0334002b0c212a994c1bc8135da449a3be247eec9b75e9b7830a92ed54183"
C1_ARCHIVE_SHA256 = "e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42"
C1_Y0_SHA256 = "5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566"
C1_Y1_SHA256 = "6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc"
C1_SOURCE_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_E1_MEMBERS = ("manifest.json", "base/chart.ddb", "semantic/composed.dds")
TOOL_RECEIPT_SCHEMA = "tac.c0b_semantic_quotient_tool_receipt.v1"
CLEANUP_CERTIFICATE_SCHEMA = "tac.c0b_semantic_quotient_cleanup_certificate.v1"
CLEANUP_COMPLETE_SCHEMA = "tac.c0b_semantic_quotient_cleanup_complete.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticQuotientError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise SemanticQuotientError(f"{label} is not a canonical JSON object")
    return value


def _safe_extract_e1_packet(packet: bytes, output_root: Path) -> Mapping[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(packet), "r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != EXPECTED_E1_MEMBERS:
                raise SemanticQuotientError("E1 semantic packet member grammar differs")
            members: dict[str, bytes] = {}
            for info in infos:
                mode = (info.external_attr >> 16) & 0o170000
                if info.is_dir() or info.flag_bits & 0x1 or mode not in (0, 0o100000):
                    raise SemanticQuotientError("E1 semantic packet member framing is unsafe")
                members[info.filename] = archive.read(info)
    except SemanticQuotientError:
        raise
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise SemanticQuotientError("E1 semantic packet is not a readable ZIP") from exc
    manifest_payload = members["manifest.json"]
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticQuotientError("E1 semantic manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != E1_PACKET_SCHEMA:
        raise SemanticQuotientError("E1 semantic packet schema differs")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("bytes") != PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * CHANNELS
        or output.get("sha256") != E1_CAMERA_RAW_SHA256
    ):
        raise SemanticQuotientError("E1 semantic packet output custody differs")
    for name in EXPECTED_E1_MEMBERS:
        write_once_or_equal(output_root.joinpath(*name.split("/")), members[name])
    return manifest


class E1SemanticPlaneRenderer:
    """Hash-bound adapter from the counted original E1 packet to base planes."""

    def __init__(self, *, runtime_source: Path, expected_runtime_sha256: str) -> None:
        try:
            resolved_source = runtime_source.resolve(strict=True)
        except OSError as exc:
            raise SemanticQuotientError("E1 generic renderer source is absent") from exc
        if runtime_source.is_symlink() or not resolved_source.is_file():
            raise SemanticQuotientError("E1 generic renderer source must be one real file")
        source_sha = sha256_file(resolved_source)
        if source_sha != expected_runtime_sha256:
            raise SemanticQuotientError("E1 generic renderer source SHA-256 differs")
        self._runtime_source = resolved_source
        self._identity = RendererIdentity(
            renderer_id=E1_RENDERER_ID,
            renderer_source_sha256=source_sha,
            semantic_packet_schema=E1_PACKET_SCHEMA,
            expected_semantic_packet_sha256=E1_PACKET_SHA256,
            expected_camera_raw_sha256=E1_CAMERA_RAW_SHA256,
        )

    @property
    def identity(self) -> RendererIdentity:
        return self._identity

    def render_chunks(
        self,
        semantic_packet: bytes,
        *,
        work_root: Path,
        chunk_pairs: int,
        resume: bool,
    ) -> Iterable[PlaneChunk]:
        if _sha256(semantic_packet) != E1_PACKET_SHA256:
            raise SemanticQuotientError("E1 renderer received a different semantic packet")
        if chunk_pairs != CHUNK_PAIRS:
            raise SemanticQuotientError("full E1/C1 renderer requires canonical 12-pair chunks")
        if work_root.exists() and not resume and any(work_root.iterdir()):
            raise SemanticQuotientError("fresh E1 render refuses a non-empty work root")
        source_sha_pre = sha256_file(self._runtime_source)
        if source_sha_pre != self._identity.renderer_source_sha256:
            raise SemanticQuotientError("E1 renderer source changed before execution")
        try:
            runtime_module = importlib.import_module("tac.optimization.ddm_runtime_receiver")
            executed_module_path = Path(runtime_module.__file__).resolve(strict=True)
            inflate_source_path = Path(runtime_module.inflate.__code__.co_filename).resolve(strict=True)
        except (AttributeError, OSError, TypeError) as exc:
            raise SemanticQuotientError("cannot bind the executed E1 renderer module source") from exc
        if executed_module_path != self._runtime_source or inflate_source_path != self._runtime_source:
            raise SemanticQuotientError("executed E1 renderer module path differs from the intended source")
        if sha256_file(executed_module_path) != source_sha_pre:
            raise SemanticQuotientError("executed E1 renderer live source hash differs")
        extracted = work_root / "extracted"
        output = work_root / "output"
        names = work_root / "video_names.txt"
        _safe_extract_e1_packet(semantic_packet, extracted)
        write_once_or_equal(names, b"0.mkv\n")
        try:
            receipt = runtime_module.inflate(extracted, output, names)
        except Exception as exc:
            raise SemanticQuotientError("original E1 generic receiver failed") from exc
        source_sha_post = sha256_file(self._runtime_source)
        if source_sha_post != source_sha_pre:
            raise SemanticQuotientError("E1 renderer source changed during execution")
        raw_path = output / "0.raw"
        expected_bytes = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * CHANNELS
        if (
            not raw_path.is_file()
            or raw_path.stat().st_size != expected_bytes
            or sha256_file(raw_path) != E1_CAMERA_RAW_SHA256
            or receipt.get("final", {}).get("sha256") != E1_CAMERA_RAW_SHA256
        ):
            raise SemanticQuotientError("E1 rendered camera bytes differ from sealed custody")
        stage = {
            "schema": "tac.c0b_e1_semantic_renderer_stage.v1",
            "semantic_packet_sha256": E1_PACKET_SHA256,
            "renderer_id": E1_RENDERER_ID,
            "renderer_source_sha256": self._identity.renderer_source_sha256,
            "executed_renderer_module_path": str(executed_module_path),
            "executed_renderer_source_sha256_pre": source_sha_pre,
            "executed_renderer_source_sha256_post": source_sha_post,
            "renderer_source_pinned_pre_post": True,
            "camera_raw_bytes": expected_bytes,
            "camera_raw_sha256": E1_CAMERA_RAW_SHA256,
            "resumable": True,
            "score_claim": False,
            "promotion_eligible": False,
        }
        write_once_or_equal(work_root / "renderer_stage.json", canonical_json(stage))
        raw = np.memmap(
            raw_path,
            mode="r",
            dtype=np.uint8,
            shape=(PAIR_COUNT, 2, CAMERA_HW[0], CAMERA_HW[1], CHANNELS),
        )
        operator = DisjointResizeOperator.build(
            camera_h=CAMERA_HW[0],
            camera_w=CAMERA_HW[1],
            scorer_h=SCORER_HW[0],
            scorer_w=SCORER_HW[1],
        )
        for chunk_index, start in enumerate(range(0, PAIR_COUNT, chunk_pairs)):
            stop = min(start + chunk_pairs, PAIR_COUNT)
            y0 = np.stack(
                [exact_resize_round_u8(operator, np.asarray(raw[pair_id, 0])) for pair_id in range(start, stop)]
            )
            y1 = np.stack(
                [exact_resize_round_u8(operator, np.asarray(raw[pair_id, 1])) for pair_id in range(start, stop)]
            )
            yield PlaneChunk(chunk_index, tuple(range(start, stop)), y0, y1)


class C1TargetTeacher:
    """Exact loader for the 50 manifest-bound original C1 target chunks."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)
        self.receipt_path = self.root / "prepare_receipt.json"
        if sha256_file(self.receipt_path) != C1_PREPARE_RECEIPT_SHA256:
            raise SemanticQuotientError("C1 prepare receipt SHA-256 differs")
        receipt = _read_canonical_json(self.receipt_path, label="C1 prepare receipt")
        required = {
            "schema": "v10_two_plane_receiver_prepare.v1",
            "completed": True,
            "test_only_small_fixture": False,
            "pair_count": PAIR_COUNT,
            "chunk_count": CHUNK_COUNT,
            "chunk_pairs": CHUNK_PAIRS,
            "camera_hw": list(CAMERA_HW),
            "scorer_hw": list(SCORER_HW),
            "archive_sha256": C1_ARCHIVE_SHA256,
            "y0_sha256": C1_Y0_SHA256,
            "y1_sha256": C1_Y1_SHA256,
            "frame0_policy_id": "description-frame0.v1",
            "y_codec_id": "predictor-residual-u8.v1",
            "predictor_mode_id": "spatial-smooth-121.v1",
            "strict_parseback_identical": True,
            "combined_without_recompression": True,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise SemanticQuotientError("C1 prepare receipt scientific custody differs")
        source_cache = receipt.get("source_cache")
        if not isinstance(source_cache, dict) or source_cache.get("sha256") != C1_SOURCE_CACHE_SHA256:
            raise SemanticQuotientError("C1 source-cache custody differs")
        archive_path = self.root / "archive.zip"
        if (
            not archive_path.is_file()
            or archive_path.stat().st_size != receipt.get("archive_bytes")
            or sha256_file(archive_path) != C1_ARCHIVE_SHA256
        ):
            raise SemanticQuotientError("C1 control archive bytes differ from prepare custody")
        manifest_hashes = receipt.get("prepare_chunk_manifest_sha256")
        if not isinstance(manifest_hashes, list) or len(manifest_hashes) != CHUNK_COUNT:
            raise SemanticQuotientError("C1 prepare receipt chunk-manifest registry differs")
        self.receipt = receipt
        self.manifest_hashes = tuple(manifest_hashes)
        declared_chunks: list[dict[str, Any]] = []
        expected_pair = 0
        chunk_root = self.root / "prepare_chunks"
        for chunk_index, expected_manifest_sha in enumerate(self.manifest_hashes):
            manifest_path = chunk_root / f"chunk-{chunk_index:04d}.manifest.json"
            if sha256_file(manifest_path) != expected_manifest_sha:
                raise SemanticQuotientError("C1 chunk manifest SHA-256 differs while binding custody")
            manifest = _read_canonical_json(manifest_path, label=f"C1 chunk {chunk_index} manifest")
            pair_ids = list(range(expected_pair, min(expected_pair + CHUNK_PAIRS, PAIR_COUNT)))
            if manifest.get("chunk_index") != chunk_index or manifest.get("pair_ids") != pair_ids:
                raise SemanticQuotientError("C1 chunk manifest geometry differs while binding custody")
            y0_sha = manifest.get("y0_sha256")
            y1_sha = manifest.get("y1_sha256")
            if not (isinstance(y0_sha, str) and len(y0_sha) == 64 and isinstance(y1_sha, str) and len(y1_sha) == 64):
                raise SemanticQuotientError("C1 chunk manifest target hashes are malformed")
            declared_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "pair_ids": pair_ids,
                    "y0_sha256": y0_sha,
                    "y1_sha256": y1_sha,
                }
            )
            expected_pair += len(pair_ids)
        self.declared_chunk_targets = tuple(declared_chunks)

    def custody(self) -> Mapping[str, Any]:
        consumed = [dict(row) for row in self.declared_chunk_targets]
        return {
            "schema": TARGET_TEACHER_CUSTODY_SCHEMA,
            "teacher_id": "original-c1-independent-source-planes.v1",
            "pair_count": PAIR_COUNT,
            "chunk_count": CHUNK_COUNT,
            "chunk_pairs": CHUNK_PAIRS,
            "scorer_hw": list(SCORER_HW),
            "channels": CHANNELS,
            "y0_sha256": C1_Y0_SHA256,
            "y1_sha256": C1_Y1_SHA256,
            "consumed_chunk_target_hashes": consumed,
            "consumed_chunk_target_hashes_sha256": _sha256(canonical_json(consumed)),
            "provenance": {
                "prepare_receipt_bytes": self.receipt_path.stat().st_size,
                "prepare_receipt_sha256": C1_PREPARE_RECEIPT_SHA256,
                "prepare_chunk_tree_sha256": self.receipt["prepare_chunk_tree_sha256"],
                "prepare_chunk_manifest_sha256": list(self.manifest_hashes),
                "source_cache": {
                    "bytes": self.receipt["source_cache"]["bytes"],
                    "sha256": C1_SOURCE_CACHE_SHA256,
                },
                "control_archive": {
                    "bytes": self.receipt["archive_bytes"],
                    "sha256": C1_ARCHIVE_SHA256,
                },
            },
            "all_video_derived_metadata_counted": True,
            "score_claim": False,
            "promotion_eligible": False,
        }

    def chunks(self) -> Iterator[PlaneChunk]:
        y0_digest = hashlib.sha256()
        y1_digest = hashlib.sha256()
        expected_pair = 0
        chunk_root = self.root / "prepare_chunks"
        for chunk_index, expected_manifest_sha in enumerate(self.manifest_hashes):
            stem = chunk_root / f"chunk-{chunk_index:04d}"
            manifest_path = stem.with_suffix(".manifest.json")
            if sha256_file(manifest_path) != expected_manifest_sha:
                raise SemanticQuotientError("C1 chunk manifest SHA-256 differs")
            manifest = _read_canonical_json(manifest_path, label=f"C1 chunk {chunk_index} manifest")
            pair_ids = list(range(expected_pair, expected_pair + CHUNK_PAIRS))
            expected = {
                "schema": "v10_two_plane_receiver_prepare_chunk.v1",
                "complete": True,
                "chunk_index": chunk_index,
                "pair_ids": pair_ids,
                "pair_count": CHUNK_PAIRS,
                "camera_hw": list(CAMERA_HW),
                "scorer_hw": list(SCORER_HW),
                "source_cache_sha256": C1_SOURCE_CACHE_SHA256,
                "frame0_policy_id": "description-frame0.v1",
                "y_codec_id": "predictor-residual-u8.v1",
                "predictor_mode_id": "spatial-smooth-121.v1",
                "y0_bytes": CHUNK_PAIRS * SCORER_HW[0] * SCORER_HW[1] * CHANNELS,
                "y1_bytes": CHUNK_PAIRS * SCORER_HW[0] * SCORER_HW[1] * CHANNELS,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise SemanticQuotientError("C1 chunk manifest content differs")
            y0_path = stem.with_suffix(".y0.bin")
            y1_path = stem.with_suffix(".y1.bin")
            try:
                y0_bytes = y0_path.read_bytes()
                y1_bytes = y1_path.read_bytes()
            except OSError as exc:
                raise SemanticQuotientError("C1 target chunk bytes are absent") from exc
            expected_bytes = CHUNK_PAIRS * SCORER_HW[0] * SCORER_HW[1] * CHANNELS
            if (
                len(y0_bytes) != expected_bytes
                or len(y1_bytes) != expected_bytes
                or _sha256(y0_bytes) != manifest.get("y0_sha256")
                or _sha256(y1_bytes) != manifest.get("y1_sha256")
            ):
                raise SemanticQuotientError("C1 target chunk bytes/hash differ")
            y0_digest.update(y0_bytes)
            y1_digest.update(y1_bytes)
            shape = (CHUNK_PAIRS, SCORER_HW[0], SCORER_HW[1], CHANNELS)
            yield PlaneChunk(
                chunk_index,
                tuple(pair_ids),
                np.frombuffer(y0_bytes, dtype=np.uint8).reshape(shape),
                np.frombuffer(y1_bytes, dtype=np.uint8).reshape(shape),
            )
            expected_pair += CHUNK_PAIRS
        if (
            expected_pair != PAIR_COUNT
            or y0_digest.hexdigest() != C1_Y0_SHA256
            or y1_digest.hexdigest() != C1_Y1_SHA256
        ):
            raise SemanticQuotientError("C1 aggregate target-plane custody differs")


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SemanticQuotientError("cannot resolve Git HEAD for provenance") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SemanticQuotientError("Git HEAD is not a full lowercase commit")
    return value


def _implementation_sources() -> Mapping[str, Any]:
    paths = {
        "semantic_quotient": REPO_ROOT / "src/tac/witness_dsl/c0b_semantic_quotient.py",
        "builder": Path(__file__).resolve(),
        "e1_semantic_renderer": REPO_ROOT / "src/tac/optimization/ddm_runtime_receiver.py",
        "v10_production_receiver": REPO_ROOT / "src/tac/witness_dsl/v10_production_receiver.py",
        "uint8_lattice_solver": REPO_ROOT / "src/tac/optimization/uint8_lattice_feasibility.py",
    }
    return {
        key: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }


def _tree_rows(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise SemanticQuotientError("cleanup root must be one real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise SemanticQuotientError("cleanup certification refuses links or special files")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            rows.append({"path": relative, "type": "directory"})
        else:
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def certify_and_cleanup_renderer_scratch(
    work_root: Path,
    *,
    archive_sha256: str,
    renderer_identity: RendererIdentity,
) -> Mapping[str, Any]:
    """Delete assembled scratch while preserving every per-stage checkpoint."""

    scratch_names = ("semantic_renderer", "decode-pass-1", "decode-pass-2")
    roots = [work_root / name for name in scratch_names if (work_root / name).exists()]
    targets: list[dict[str, Any]] = []
    preserved_stage_roots: list[str] = []
    for root in roots:
        if root.parent.resolve() != work_root.resolve() or root.name not in scratch_names:
            raise SemanticQuotientError("cleanup target escaped the exact renderer scratch set")
        checkpoint_root = root / ".ddm_runtime_checkpoints"
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise SemanticQuotientError("cleanup requires preserved E1 per-stage checkpoints")
        preserved_stage_roots.append(str(checkpoint_root))
        raw_path = root / "output" / "0.raw"
        if raw_path.exists() and (raw_path.is_symlink() or not raw_path.is_file()):
            raise SemanticQuotientError("cleanup assembled raw path is unsafe")
        if raw_path.is_file() and not raw_path.is_symlink():
            targets.append(
                {
                    "kind": "assembled_raw_file",
                    "original_path": str(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "sha256": sha256_file(raw_path),
                }
            )
        extracted = root / "extracted"
        if extracted.exists() and (extracted.is_symlink() or not extracted.is_dir()):
            raise SemanticQuotientError("cleanup extracted path is unsafe")
        if extracted.is_dir() and not extracted.is_symlink():
            rows = _tree_rows(extracted)
            targets.append(
                {
                    "kind": "extracted_counted_packet_copy",
                    "original_path": str(extracted),
                    "tree_rows": rows,
                    "tree_sha256": _sha256(canonical_json(rows)),
                    "bytes": sum(row.get("bytes", 0) for row in rows),
                }
            )
    certificate = {
        "schema": CLEANUP_CERTIFICATE_SCHEMA,
        "archive_sha256": archive_sha256,
        "semantic_packet_sha256": E1_PACKET_SHA256,
        "renderer": renderer_identity.as_manifest(),
        "rebuild_command": "rerun this tool with the same counted archive inputs and --resume-from work root",
        "reason": "success-only assembled raw and extracted packet copies are deterministically rebuildable",
        "targets": targets,
        "preserved_per_stage_checkpoint_roots": preserved_stage_roots,
        "per_stage_checkpoints_preserved": True,
        "delete_authorized": True,
        "score_claim": False,
    }
    certificate_path = work_root / "cleanup" / "renderer_scratch_certificate.json"
    write_once_or_equal(certificate_path, canonical_json(certificate))
    removed_paths: list[str] = []
    for row in targets:
        target = Path(row["original_path"])
        if row["kind"] == "assembled_raw_file":
            if target.parent.name != "output" or target.name != "0.raw":
                raise SemanticQuotientError("cleanup raw target shape differs")
            target.unlink()
            try:
                target.parent.rmdir()
            except OSError:
                pass
        elif row["kind"] == "extracted_counted_packet_copy":
            if target.name != "extracted":
                raise SemanticQuotientError("cleanup extracted target shape differs")
            shutil.rmtree(target)
        else:
            raise SemanticQuotientError("cleanup target kind differs")
        removed_paths.append(str(target))
    complete = {
        "schema": CLEANUP_COMPLETE_SCHEMA,
        "certificate_sha256": sha256_file(certificate_path),
        "removed_paths": removed_paths,
        "all_absent_after_cleanup": all(not Path(path).exists() for path in removed_paths),
        "preserved_per_stage_checkpoint_roots": preserved_stage_roots,
        "all_per_stage_checkpoints_preserved": all(Path(path).is_dir() for path in preserved_stage_roots),
        "recoverable_by_deterministic_rebuild": True,
        "score_claim": False,
    }
    if complete["all_absent_after_cleanup"] is not True or complete["all_per_stage_checkpoints_preserved"] is not True:
        raise SemanticQuotientError("renderer scratch cleanup was incomplete")
    write_once_or_equal(work_root / "cleanup" / "renderer_scratch_complete.json", canonical_json(complete))
    return complete


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    implementation_sources_pre = _implementation_sources()
    work_root = args.resume_from.expanduser().resolve(strict=False)
    archive_path = (args.archive_path or work_root / "output" / "archive.zip").expanduser().resolve(strict=False)
    semantic_packet_path = args.semantic_packet.expanduser().resolve(strict=True)
    if sha256_file(semantic_packet_path) != E1_PACKET_SHA256:
        raise SemanticQuotientError("selected semantic packet is not the sealed original E1 packet")
    semantic_packet = semantic_packet_path.read_bytes()
    runtime_source = REPO_ROOT / "src/tac/optimization/ddm_runtime_receiver.py"
    renderer = E1SemanticPlaneRenderer(
        runtime_source=runtime_source,
        expected_runtime_sha256=args.expected_renderer_source_sha256,
    )
    teacher = C1TargetTeacher(args.c1_root.expanduser().resolve(strict=True))
    camera_raw_bytes = PAIR_COUNT * 2 * CAMERA_HW[0] * CAMERA_HW[1] * CHANNELS
    target_plane_bytes = PAIR_COUNT * SCORER_HW[0] * SCORER_HW[1] * CHANNELS
    # Three renderer roots are preserved (build plus two independent decode
    # passes), and each E1 root contains both per-stage raw and assembled raw.
    required_bytes = 6 * camera_raw_bytes + 4 * target_plane_bytes + (2 << 30)
    storage = storage_preflight(
        work_root,
        required_bytes=required_bytes,
        allow_local_storage=args.allow_local_storage,
    )
    result = build_semantic_quotient_archive(
        semantic_packet,
        renderer,
        teacher.chunks(),
        archive_path=archive_path,
        work_root=work_root,
        target_teacher_custody=teacher.custody(),
        camera_hw=CAMERA_HW,
        scorer_hw=SCORER_HW,
        channels=CHANNELS,
        pair_count=PAIR_COUNT,
        chunk_pairs=CHUNK_PAIRS,
        resume=True,
        allow_local_storage=args.allow_local_storage,
    )
    implementation_sources_post = _implementation_sources()
    if implementation_sources_post != implementation_sources_pre:
        raise SemanticQuotientError("implementation sources changed during the C0B build")
    parsed = parse_semantic_quotient_archive(result.archive_path)
    tool_receipt: dict[str, Any] = {
        "schema": TOOL_RECEIPT_SCHEMA,
        "scientific_label": parsed.manifest["scientific_label"],
        "git_sha": _git_head(),
        "argv": list(args.semantic_argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation_sources": {
            "pre": implementation_sources_pre,
            "post": implementation_sources_post,
            "byte_identical": True,
        },
        "archive": {
            "path": str(result.archive_path),
            "bytes": result.archive_bytes,
            "sha256": result.archive_sha256,
        },
        "semantic_packet": {
            "path": str(semantic_packet_path),
            "bytes": len(semantic_packet),
            "sha256": E1_PACKET_SHA256,
        },
        "renderer": renderer.identity.as_manifest(),
        "target_teacher_custody": teacher.custody(),
        "storage_preflight": {
            "schema": storage["schema"],
            "selected_tier": storage["selected_tier"],
            "required_bytes": storage["required_bytes"],
            "passed": storage["passed"],
            "allow_local_storage": storage["allow_local_storage"],
        },
        "build_receipt_sha256": sha256_file(result.receipt_path),
        "no_scorer_invocation": True,
        "dense_nonpromotable_baseline": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if args.cleanup_renderer_scratch:
        tool_receipt["cleanup"] = certify_and_cleanup_renderer_scratch(
            work_root,
            archive_sha256=result.archive_sha256,
            renderer_identity=renderer.identity,
        )
    else:
        tool_receipt["cleanup"] = {
            "performed": False,
            "reason": "renderer stages preserved for resume; use --cleanup-renderer-scratch after successful build",
        }
    write_once_or_equal(work_root / "tool_receipt.json", canonical_json(tool_receipt))
    return tool_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the nonpromotable original E1 semantic-base plus exact C1 two-plane quotient"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="durable stage root; created for a fresh build and revalidated for every resume",
    )
    parser.add_argument("--archive-path", type=Path, help="default: <resume-from>/output/archive.zip")
    parser.add_argument("--semantic-packet", type=Path, default=E1_PACKET_DEFAULT)
    parser.add_argument("--c1-root", type=Path, default=C1_ROOT_DEFAULT)
    parser.add_argument(
        "--expected-renderer-source-sha256",
        default=E1_RENDERER_SOURCE_SHA256,
        help="fail-closed exact generic E1 runtime source identity",
    )
    parser.add_argument(
        "--allow-local-storage",
        action="store_true",
        help="explicitly opt into local storage only when SSD tiers cannot be used",
    )
    parser.add_argument(
        "--cleanup-renderer-scratch",
        action="store_true",
        help="after double-decode, certify/delete assembled scratch while preserving every stage checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    semantic_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(semantic_argv)
    args.semantic_argv = semantic_argv
    try:
        receipt = build(args)
    except SemanticQuotientError as exc:
        raise SystemExit(f"C0B semantic quotient build refused: {exc}") from exc
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
