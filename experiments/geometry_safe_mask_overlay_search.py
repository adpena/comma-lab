#!/usr/bin/env python3
"""Search local geometry-safe PR75 mask byte-cut candidates.

The tool is intentionally local-only.  It reads a single-member PR75/QP1
archive, preserves renderer/pose/tile-action streams, and searches finite mask
stream transforms:

* lossless Brotli resweep of the decoded AV1 OBU mask stream;
* bounded AV1 OBU re-encodes with decoded class-pixel disagreement metrics;
* exact CDO1 overlay economics for lossy bases, without dispatching GPU work.

Generated archives remain ``score_claim=false`` until exact CUDA auth eval runs
on the exact bytes and matching runtime tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
    "c067_pr75_actions_lag_eval_top67_p6/archive.zip"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/geometry_safe_mask_overlay_search_20260503_worker"
)
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
TOOL = "experiments/geometry_safe_mask_overlay_search.py"
MEMBER_NAME = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
MASK_SHAPE = (600, 384, 512)
MASK_CLASS_SCALE = 255 // 4
CDO1_MAGIC = b"CDO1"
CDO1_VERSION = 1
CDO1_HEADER_STRUCT = struct.Struct("<4sHI")
CDO1_RECORD_STRUCT = struct.Struct("<HHHHB")
CDO1_RECORD_STRUCT_NAME = "u16_frame_u16_y_u16_x0_u16_length_u8_value_le"


@dataclass(frozen=True)
class P6Slices:
    mask_br: bytes
    model_br: bytes
    actions_br: bytes
    pose_br: bytes
    record_count: int


@dataclass(frozen=True)
class EncodePolicy:
    policy_id: str
    crf: int
    preset: int
    keyint: int = 9999
    fps: int = 20
    svt_params: str = "enable-restoration=0:enable-cdef=0"


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    quality: int
    mode: int
    lgwin: int
    lgblock: int

    @property
    def params(self) -> dict[str, int]:
        return {
            "quality": int(self.quality),
            "mode": int(self.mode),
            "lgwin": int(self.lgwin),
            "lgblock": int(self.lgblock),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any] | list[Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_zip_member_name(name: str) -> str:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_zip_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def write_single_p_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(MEMBER_NAME), payload)


def read_single_p_archive(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly one {MEMBER_NAME!r}; got {names!r}")
        return zf.read(infos[0])


def load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("geometry_safe_pr75_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_p6_payload(payload: bytes) -> P6Slices:
    if not payload.startswith(b"P6"):
        raise ValueError(f"source payload must be P6 for this worker; got {payload[:2]!r}")
    header_size = 2 + struct.calcsize("<IHHH")
    if len(payload) <= header_size:
        raise ValueError("P6 payload too short")
    mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
    if min(mask_len, model_len, actions_len, record_count) <= 0:
        raise ValueError("P6 payload contains an empty required slice")
    cursor = header_size
    if cursor + mask_len + model_len + actions_len >= len(payload):
        raise ValueError("P6 slice lengths leave no pose stream")
    mask_br = payload[cursor : cursor + mask_len]
    cursor += mask_len
    model_br = payload[cursor : cursor + model_len]
    cursor += model_len
    actions_br = payload[cursor : cursor + actions_len]
    cursor += actions_len
    return P6Slices(
        mask_br=mask_br,
        model_br=model_br,
        actions_br=actions_br,
        pose_br=payload[cursor:],
        record_count=int(record_count),
    )


def build_p6_payload(slices: P6Slices) -> bytes:
    return (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(slices.mask_br),
            len(slices.model_br),
            len(slices.actions_br),
            int(slices.record_count),
        )
        + slices.mask_br
        + slices.model_br
        + slices.actions_br
        + slices.pose_br
    )


def focused_brotli_grid() -> list[tuple[int, int, int, int]]:
    """Small deterministic Brotli grid around the observed PR75 mask optimum."""
    params: list[tuple[int, int, int, int]] = []
    for quality in (11, 10):
        for mode in (0, 1, 2):
            for lgwin in (18, 19, 20):
                for lgblock in (0, 17, 18, 19):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def best_brotli(raw: bytes, source: bytes, params: Iterable[tuple[int, int, int, int]]) -> BrotliChoice:
    best = BrotliChoice(source, -1, -1, -1, -1)
    for param in params:
        quality, mode, lgwin, lgblock = param
        candidate = brotli.compress(
            raw,
            quality=quality,
            mode=mode,
            lgwin=lgwin,
            lgblock=lgblock,
        )
        if len(candidate) < len(best.data):
            best = BrotliChoice(candidate, quality, mode, lgwin, lgblock)
    if brotli.decompress(best.data) != raw:
        raise ValueError("selected Brotli stream failed roundtrip")
    return best


def mask_tensor_sha256(classes: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(classes.astype(np.uint8, copy=False)).tobytes())


def _ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for mask decode/re-encode search")
    return ffmpeg


def _decode_video_to_gray(video_bytes: bytes, *, suffix: str = ".obu") -> np.ndarray:
    ffmpeg = _ffmpeg_binary()
    with tempfile.TemporaryDirectory(prefix="geometry_mask_decode_") as tmp:
        path = Path(tmp) / f"masks{suffix}"
        path.write_bytes(video_bytes)
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "pipe:1",
            ],
            capture_output=True,
            timeout=300,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg mask decode failed: "
            + proc.stderr.decode("utf-8", errors="replace")[:2000]
        )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = math.prod(MASK_SHAPE)
    if raw.size != expected:
        raise ValueError(f"decoded mask bytes={raw.size} expected={expected} for shape={MASK_SHAPE}")
    return raw.reshape(MASK_SHAPE)


def gray_to_classes(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(gray.astype(np.float32) / float(MASK_CLASS_SCALE)), 0, 4).astype(np.uint8)


def classes_to_gray(classes: np.ndarray) -> np.ndarray:
    return np.clip(classes.astype(np.uint16) * MASK_CLASS_SCALE, 0, 255).astype(np.uint8)


def encode_classes_to_obu(classes: np.ndarray, policy: EncodePolicy, output_path: Path) -> bytes:
    ffmpeg = _ffmpeg_binary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{MASK_SHAPE[2]}x{MASK_SHAPE[1]}",
            "-pix_fmt",
            "gray",
            "-r",
            str(policy.fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libsvtav1",
            "-crf",
            str(policy.crf),
            "-preset",
            str(policy.preset),
            "-g",
            str(policy.keyint),
            "-svtav1-params",
            policy.svt_params,
            "-pix_fmt",
            "gray",
            "-an",
            "-f",
            "obu",
            str(output_path),
        ],
        input=classes_to_gray(classes).tobytes(),
        capture_output=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encode failed for {policy.policy_id}: "
            + proc.stderr.decode("utf-8", errors="replace")[:2000]
        )
    return output_path.read_bytes()


def compare_classes(source: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if source.shape != candidate.shape:
        raise ValueError(f"mask shape mismatch: {source.shape} != {candidate.shape}")
    diff = source != candidate
    frame_counts = diff.reshape(diff.shape[0], -1).sum(axis=1).astype(np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for src_class in range(5):
        src_mask = source == src_class
        for cand_class in range(5):
            confusion[src_class, cand_class] = int(np.count_nonzero(src_mask & (candidate == cand_class)))
    changed = int(diff.sum())
    total = int(diff.size)
    return {
        "bounded_pixel_disagreement": changed > 0,
        "changed_frame_count": int(np.count_nonzero(frame_counts)),
        "changed_pixel_count": changed,
        "changed_pixel_fraction": float(changed / total),
        "max_changed_pixels_per_frame": int(frame_counts.max(initial=0)),
        "mean_changed_pixels_per_changed_frame": (
            float(frame_counts[frame_counts > 0].mean()) if np.any(frame_counts > 0) else 0.0
        ),
        "source_to_candidate_confusion_5x5": confusion.tolist(),
        "total_pixel_count": total,
        "zero_disagreement": changed == 0,
    }


def cdo1_runs(base: np.ndarray, target: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    if base.shape != target.shape:
        raise ValueError(f"CDO1 base/target shape mismatch: {base.shape} != {target.shape}")
    diff = base != target
    runs: list[tuple[int, int, int, int, int]] = []
    changed_rows = np.argwhere(diff.any(axis=2))
    for frame_index, y in changed_rows:
        xs = np.flatnonzero(diff[frame_index, y])
        values = target[frame_index, y, xs]
        if xs.size == 0:
            continue
        boundaries = np.flatnonzero((np.diff(xs) != 1) | (np.diff(values) != 0)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [xs.size]))
        for start, end in zip(starts, ends, strict=True):
            x0 = int(xs[start])
            x1 = int(xs[end - 1])
            runs.append(
                (
                    int(frame_index),
                    int(y),
                    x0,
                    int(x1 - x0 + 1),
                    int(values[start]),
                )
            )
    return runs


def encode_cdo1_payload(
    *,
    base: np.ndarray,
    target: np.ndarray,
    runs: Sequence[tuple[int, int, int, int, int]],
    producer: str,
    policy_id: str,
) -> bytes:
    header = {
        "base_mask_tensor_sha256": mask_tensor_sha256(base),
        "candidate_policy_id": policy_id,
        "pair_index_basis": "half_frame_pair_index",
        "producer": producer,
        "reconstructed_mask_u8_sha256": mask_tensor_sha256(target),
        "run_count": len(runs),
        "run_struct": CDO1_RECORD_STRUCT_NAME,
        "schema": "c067_decoded_delta_overlay_payload_v1",
        "score_claim": False,
        "selected_pair_indices": sorted({int(run[0]) for run in runs}),
        "selected_pixel_count": int(sum(int(run[3]) for run in runs)),
        "shape": [int(value) for value in base.shape],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray(CDO1_MAGIC + struct.pack("<HI", CDO1_VERSION, len(header_bytes)) + header_bytes)
    for run in runs:
        out.extend(CDO1_RECORD_STRUCT.pack(*run))
    return bytes(out)


def _member_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in sorted(decoded.items())
    }


def _validate_pr75_payload_parse(payload: bytes, unpacker: Any) -> tuple[str, dict[str, bytes]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    return str(header.get("payload_format")), decoded


def _candidate_manifest(
    *,
    candidate_id: str,
    source_archive: Path,
    source_payload: bytes,
    output_archive: Path,
    output_payload: bytes,
    payload_format: str,
    source_decoded: dict[str, bytes],
    candidate_decoded: dict[str, bytes],
    mask_metrics: dict[str, Any],
    stream_choices: dict[str, Any],
    candidate_kind: str,
    dispatch_recommendation: str,
) -> dict[str, Any]:
    source_archive_bytes = source_archive.stat().st_size
    archive_bytes = output_archive.stat().st_size
    delta_bytes = archive_bytes - source_archive_bytes
    preserved_members = {
        name: candidate_decoded.get(name) == source_decoded.get(name)
        for name in sorted(set(source_decoded) | set(candidate_decoded))
        if name != "masks.mkv"
    }
    return {
        "archive_delta_bytes_vs_source": int(delta_bytes),
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_recommendation": dispatch_recommendation,
        "evidence_grade": (
            "empirical_lossless_byte_transform"
            if mask_metrics["zero_disagreement"]
            else "empirical_bounded_mask_disagreement"
        ),
        "formula_only_rate_score_delta_vs_source": float(delta_bytes * RATE_SCORE_PER_BYTE),
        "mask_geometry": mask_metrics,
        "non_mask_decoded_members_preserved": preserved_members,
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(output_archive),
            "sha256": _sha256_file(output_archive),
        },
        "payload": {
            "bytes": len(output_payload),
            "format": payload_format,
            "member": MEMBER_NAME,
            "sha256": _sha256_bytes(output_payload),
        },
        "promotion_eligible": False,
        "schema": "geometry_safe_mask_overlay_candidate_manifest_v1",
        "score_claim": False,
        "source_archive": {
            "bytes": source_archive_bytes,
            "path": str(source_archive),
            "sha256": _sha256_file(source_archive),
        },
        "source_payload": {
            "bytes": len(source_payload),
            "sha256": _sha256_bytes(source_payload),
        },
        "stream_choices": stream_choices,
        "tool": TOOL,
    }


def build_lossless_candidate(
    *,
    source_archive: Path,
    source_payload: bytes,
    source_slices: P6Slices,
    source_decoded: dict[str, bytes],
    source_classes: np.ndarray,
    output_dir: Path,
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    best_mask = best_brotli(
        source_decoded["masks.mkv"],
        source_slices.mask_br,
        focused_brotli_grid(),
    )
    candidate_id = "lossless_mask_obu_brotli_resweep_p6"
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    payload = build_p6_payload(
        P6Slices(
            mask_br=best_mask.data,
            model_br=source_slices.model_br,
            actions_br=source_slices.actions_br,
            pose_br=source_slices.pose_br,
            record_count=source_slices.record_count,
        )
    )
    payload_format, candidate_decoded = _validate_pr75_payload_parse(payload, unpacker)
    if candidate_decoded["masks.mkv"] != source_decoded["masks.mkv"]:
        raise ValueError("lossless candidate changed decoded mask OBU bytes")
    write_single_p_archive(archive_path, payload)
    metrics = compare_classes(source_classes, source_classes)
    manifest = _candidate_manifest(
        candidate_id=candidate_id,
        source_archive=source_archive,
        source_payload=source_payload,
        output_archive=archive_path,
        output_payload=payload,
        payload_format=payload_format,
        source_decoded=source_decoded,
        candidate_decoded=candidate_decoded,
        mask_metrics=metrics,
        stream_choices={
            "masks.mkv": {
                "brotli_bytes": len(best_mask.data),
                "brotli_params": best_mask.params if best_mask.quality >= 0 else "source",
                "raw_bytes": len(source_decoded["masks.mkv"]),
                "raw_sha256": _sha256_bytes(source_decoded["masks.mkv"]),
            }
        },
        candidate_kind="lossless_mask_stream_rebrotli",
        dispatch_recommendation=(
            "exact-ready but standalone value is too small; only dispatch if bundled "
            "with a larger exact-safe change"
        ),
    )
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_bounded_reencode_candidate(
    *,
    policy: EncodePolicy,
    source_archive: Path,
    source_payload: bytes,
    source_slices: P6Slices,
    source_decoded: dict[str, bytes],
    source_classes: np.ndarray,
    output_dir: Path,
    unpacker: Any,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = f"bounded_mask_reencode_{policy.policy_id}"
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    mask_path = candidate_dir / "masks.obu"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    t0 = time.monotonic()
    mask_obu = encode_classes_to_obu(source_classes, policy, mask_path)
    decoded_gray = _decode_video_to_gray(mask_obu)
    candidate_classes = gray_to_classes(decoded_gray)
    metrics = compare_classes(source_classes, candidate_classes)
    best_mask = best_brotli(mask_obu, brotli.compress(mask_obu, quality=11), focused_brotli_grid())
    payload = build_p6_payload(
        P6Slices(
            mask_br=best_mask.data,
            model_br=source_slices.model_br,
            actions_br=source_slices.actions_br,
            pose_br=source_slices.pose_br,
            record_count=source_slices.record_count,
        )
    )
    payload_format, candidate_decoded = _validate_pr75_payload_parse(payload, unpacker)
    if candidate_decoded["renderer.bin"] != source_decoded["renderer.bin"]:
        raise ValueError(f"{candidate_id} changed renderer.bin")
    if candidate_decoded["optimized_poses.qp1"] != source_decoded["optimized_poses.qp1"]:
        raise ValueError(f"{candidate_id} changed optimized_poses.qp1")
    if candidate_decoded["seg_tile_actions.bin"] != source_decoded["seg_tile_actions.bin"]:
        raise ValueError(f"{candidate_id} changed seg_tile_actions.bin")
    write_single_p_archive(archive_path, payload)
    archive_delta = archive_path.stat().st_size - source_archive.stat().st_size
    recommendation = (
        "do_not_dispatch: decoded mask tensor changes; needs local component trace or "
        "operator-approved exact eval after dispatch claim"
    )
    if metrics["changed_pixel_fraction"] <= 0.001 and archive_delta <= -1024:
        recommendation = (
            "candidate_watchlist: >1KB saved with <=0.1% mask disagreement, but still "
            "not promotable without CUDA exact eval"
        )
    manifest = _candidate_manifest(
        candidate_id=candidate_id,
        source_archive=source_archive,
        source_payload=source_payload,
        output_archive=archive_path,
        output_payload=payload,
        payload_format=payload_format,
        source_decoded=source_decoded,
        candidate_decoded=candidate_decoded,
        mask_metrics=metrics,
        stream_choices={
            "masks.mkv": {
                "brotli_bytes": len(best_mask.data),
                "brotli_params": best_mask.params if best_mask.quality >= 0 else "source",
                "encoded_obu_bytes": len(mask_obu),
                "encoded_obu_sha256": _sha256_bytes(mask_obu),
                "ffmpeg_policy": {
                    "crf": policy.crf,
                    "fps": policy.fps,
                    "keyint": policy.keyint,
                    "preset": policy.preset,
                    "svt_params": policy.svt_params,
                },
            }
        },
        candidate_kind="bounded_lossy_mask_reencode",
        dispatch_recommendation=recommendation,
    )
    manifest["wall_time_seconds"] = round(time.monotonic() - t0, 3)
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest, {
        "candidate_classes": candidate_classes,
        "mask_obu": mask_obu,
        "policy": policy,
    }


def overlay_economics(
    *,
    policy_id: str,
    candidate_classes: np.ndarray,
    source_classes: np.ndarray,
) -> dict[str, Any]:
    runs = cdo1_runs(candidate_classes, source_classes)
    raw_payload = encode_cdo1_payload(
        base=candidate_classes,
        target=source_classes,
        runs=runs,
        producer=TOOL,
        policy_id=policy_id,
    )
    compressed = {
        "brotli_q11_bytes": len(brotli.compress(raw_payload, quality=11)),
        "raw_bytes": len(raw_payload),
        "zlib9_bytes": len(zlib.compress(raw_payload, level=9)),
    }
    return {
        "exact_overlay_reconstructs_source_mask_sha256": mask_tensor_sha256(source_classes),
        "overlay_payload_sha256": _sha256_bytes(raw_payload),
        "run_count": len(runs),
        "selected_pixel_count": int(sum(int(run[3]) for run in runs)),
        "compressed_sizes": compressed,
        "verdict": "negative_too_large_for_byte_cut",
    }


def default_policies() -> list[EncodePolicy]:
    return [
        EncodePolicy("crf52_preset13_g9999", crf=52, preset=13),
        EncodePolicy("crf55_preset13_g9999", crf=55, preset=13),
        EncodePolicy("crf58_preset13_g9999", crf=58, preset=13),
        EncodePolicy("crf60_preset13_g9999", crf=60, preset=13),
    ]


def parse_policy(raw: str) -> EncodePolicy:
    parts = raw.split(":")
    if len(parts) not in {2, 3, 4}:
        raise argparse.ArgumentTypeError("policy must be id:crf[:preset[:keyint]]")
    policy_id = parts[0].strip()
    if not policy_id:
        raise argparse.ArgumentTypeError("policy id must be nonempty")
    crf = int(parts[1])
    preset = int(parts[2]) if len(parts) >= 3 else 13
    keyint = int(parts[3]) if len(parts) >= 4 else 9999
    return EncodePolicy(policy_id, crf=crf, preset=preset, keyint=keyint)


def run_search(
    *,
    source_archive: Path,
    output_dir: Path,
    policies: Sequence[EncodePolicy],
    force: bool = False,
    overlay_probe_limit: int = 2,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unpacker = load_unpacker()
    source_payload = read_single_p_archive(source_archive)
    source_slices = parse_p6_payload(source_payload)
    source_header, source_decoded = unpacker._parse_payload(source_payload)  # noqa: SLF001
    source_payload_format = str(source_header.get("payload_format"))
    if source_payload_format != "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        raise ValueError(f"expected PR75 P6 source payload, got {source_payload_format!r}")
    source_gray = _decode_video_to_gray(source_decoded["masks.mkv"])
    source_classes = gray_to_classes(source_gray)

    manifests: list[dict[str, Any]] = []
    manifests.append(
        build_lossless_candidate(
            source_archive=source_archive,
            source_payload=source_payload,
            source_slices=source_slices,
            source_decoded=source_decoded,
            source_classes=source_classes,
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        )
    )

    bounded_runtime: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for policy in policies:
        manifest, runtime = build_bounded_reencode_candidate(
            policy=policy,
            source_archive=source_archive,
            source_payload=source_payload,
            source_slices=source_slices,
            source_decoded=source_decoded,
            source_classes=source_classes,
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        )
        manifests.append(manifest)
        bounded_runtime.append((manifest, runtime))

    overlay_rows = []
    bounded_sorted = sorted(
        bounded_runtime,
        key=lambda item: (
            item[0]["mask_geometry"]["changed_pixel_fraction"],
            item[0]["archive_delta_bytes_vs_source"],
        ),
    )
    for manifest, runtime in bounded_sorted[: max(0, int(overlay_probe_limit))]:
        overlay_rows.append(
            {
                "candidate_id": manifest["candidate_id"],
                "archive_delta_bytes_vs_source_without_overlay": manifest[
                    "archive_delta_bytes_vs_source"
                ],
                "mask_geometry_without_overlay": manifest["mask_geometry"],
                **overlay_economics(
                    policy_id=runtime["policy"].policy_id,
                    candidate_classes=runtime["candidate_classes"],
                    source_classes=source_classes,
                ),
            }
        )
    _write_json(output_dir / "overlay_economics.json", {"overlays": overlay_rows, "score_claim": False})

    rows = [
        {
            "archive_bytes": int(m["output_archive"]["bytes"]),
            "archive_delta_bytes_vs_source": int(m["archive_delta_bytes_vs_source"]),
            "archive_path": m["output_archive"]["path"],
            "archive_sha256": m["output_archive"]["sha256"],
            "candidate_id": m["candidate_id"],
            "candidate_kind": m["candidate_kind"],
            "changed_pixel_count": int(m["mask_geometry"]["changed_pixel_count"]),
            "changed_pixel_fraction": float(m["mask_geometry"]["changed_pixel_fraction"]),
            "dispatch_recommendation": m["dispatch_recommendation"],
            "formula_only_rate_score_delta_vs_source": m[
                "formula_only_rate_score_delta_vs_source"
            ],
            "manifest_path": str(output_dir / m["candidate_id"] / "manifest.json"),
            "score_claim": False,
            "zero_disagreement": bool(m["mask_geometry"]["zero_disagreement"]),
        }
        for m in manifests
    ]
    summary = {
        "best_lossless": min(
            (row for row in rows if row["zero_disagreement"]),
            key=lambda row: row["archive_bytes"],
        ),
        "best_bounded_by_bytes": min(rows, key=lambda row: row["archive_bytes"]),
        "candidate_count": len(rows),
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "evidence_grade": "empirical_local_mask_geometry_screen",
        "overlay_economics_path": str(output_dir / "overlay_economics.json"),
        "promotion_eligible": False,
        "schema": "geometry_safe_mask_overlay_search_summary_v1",
        "score_claim": False,
        "source_archive": {
            "bytes": source_archive.stat().st_size,
            "path": str(source_archive),
            "sha256": _sha256_file(source_archive),
        },
        "source_decoded_members": _member_summary(source_decoded),
        "source_mask_tensor": {
            "shape": [int(v) for v in source_classes.shape],
            "sha256": mask_tensor_sha256(source_classes),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--policy",
        action="append",
        type=parse_policy,
        help="Additional or replacement encode policy as id:crf[:preset[:keyint]].",
    )
    parser.add_argument(
        "--overlay-probe-limit",
        type=int,
        default=2,
        help="Run exact CDO1 overlay economics for the N lowest-disagreement bounded candidates.",
    )
    args = parser.parse_args(argv)
    policies = args.policy if args.policy else default_policies()
    summary = run_search(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        policies=policies,
        force=bool(args.force),
        overlay_probe_limit=args.overlay_probe_limit,
    )
    print(
        json.dumps(
            {
                "best_bounded_by_bytes": summary["best_bounded_by_bytes"],
                "best_lossless": summary["best_lossless"],
                "candidate_count": summary["candidate_count"],
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
