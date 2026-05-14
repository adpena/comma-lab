#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan C091/PR75 mask-packer big moves without dispatching GPU work.

This tool is a deterministic local byte screen.  It does not load scorer
models, does not edit dispatch state, and does not claim score.  It exists to
separate exact-lossless packer opportunities from lossy mask-transcode probes
around the current C091/PR75/PR77/C089 frontier.
"""
from __future__ import annotations

import argparse
import bz2
import importlib.util
import json
import lzma
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.repo_io import read_json, sha256_bytes, sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c091_mask_packer_bigmove.py"
SCHEMA = "c091_mask_packer_bigmove_plan_v1"
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489.0
DEFAULT_TARGET_SCORE = 0.31
FRONTIER_SCORE_FALLBACK = 0.31516575028285976
FRONTIER_BYTES_FALLBACK = 276_481
FRONTIER_SHA256_FALLBACK = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"

DEFAULT_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_FRONTIER_EVAL = DEFAULT_FRONTIER_ARCHIVE.with_name("contest_auth_eval.adjudicated.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c091_mask_packer_bigmove_20260503_codex"
DEFAULT_CANDIDATE_MATRICES = (
    REPO_ROOT / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/candidate_matrix.json",
    REPO_ROOT / "experiments/results/pr75_lossless_micro_packer_worker_20260503/candidate_matrix.json",
    REPO_ROOT / "experiments/results/pr75_minp_p6_stream_mix_worker_20260503/candidate_matrix.json",
)
DEFAULT_ACTIVE_CANDIDATE_ID = "pr77_actions_pr75mask_renderer_c089pose_fixedslice"


@dataclass(frozen=True)
class StreamSlices:
    payload_format: str
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    action_record_count: int | None = None


FIXED_SLICE_TABLE: dict[int, tuple[int, int, int, str]] = {
    276_381: (219_472, 55_756, 255, "pr75_minp_fixed_actions255_model55756"),
    276_379: (219_472, 55_756, 253, "pr75_minp_fixed_actions253_model55756"),
    276_451: (219_472, 55_756, 325, "pr77_fixed_actions325_model55756"),
    276_520: (219_472, 55_914, 236, "pr75_fixed_actions236_model55914"),
    276_641: (219_472, 56_034, 236, "pr75_fixed_actions236_model56034"),
}


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_zip_member_name(name: str) -> str:
    parts = Path(name).parts
    if not name or name.startswith("/") or ".." in parts or len(parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if name.startswith(".") or name.startswith("._") or "/." in name or "/._" in name:
        raise ValueError(f"hidden/system ZIP member is forbidden: {name!r}")
    return name


def read_single_payload_zip(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read a strict single-member stored payload archive."""
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        infos: list[zipfile.ZipInfo] = []
        for info in zf.infolist():
            name = _safe_zip_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                infos.append(info)
        names = [info.filename for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly member {MEMBER_NAME!r}; got {names!r}")
        info = infos[0]
        payload = zf.read(info)
    return payload, {
        "archive_path": _rel(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_count": 1,
        "member_name": info.filename,
        "member_compress_type": int(info.compress_type),
        "member_compress_size": int(info.compress_size),
        "member_file_size": int(info.file_size),
        "member_crc": int(info.CRC),
        "zip_overhead_bytes": int(path.stat().st_size - len(payload)),
    }


def parse_payload_slices(payload: bytes) -> StreamSlices:
    """Parse C091/PR75/PR77/C089 payload container forms used by this lane."""
    action_record_count: int | None = None
    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        payload_format = "public_pr75_qzs3_qp1_segactions_p3"
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, action_record_count = struct.unpack_from("<IHHH", payload, 2)
        payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    elif payload.startswith(b"P5"):
        cursor = 2 + struct.calcsize("<IHHHH")
        mask_len, renderer_len, dict_len, actions_len, action_record_count = struct.unpack_from(
            "<IHHHH",
            payload,
            2,
        )
        if dict_len:
            raise ValueError("P5 action-dict payloads are not a mask big-move target for this planner")
        payload_format = "public_pr75_qzs3_qp1_segactions_p5_no_dict"
    elif len(payload) in FIXED_SLICE_TABLE:
        mask_len, renderer_len, actions_len, label = FIXED_SLICE_TABLE[len(payload)]
        cursor = 0
        payload_format = f"public_pr75_qzs3_qp1_segactions_fixed_slices:{label}"
    else:
        raise ValueError(f"unsupported payload form prefix={payload[:4]!r} bytes={len(payload)}")

    if min(mask_len, renderer_len, actions_len) <= 0:
        raise ValueError("payload contains an empty required encoded stream")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("payload stream lengths leave no pose stream")
    return StreamSlices(
        payload_format=payload_format,
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=payload[renderer_end:actions_end],
        pose_br=payload[actions_end:],
        action_record_count=action_record_count,
    )


def _decode_brotli(label: str, data: bytes) -> bytes:
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise ValueError(f"{label} did not Brotli-decode") from exc


def _load_unpacker() -> Any:
    path = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
    spec = importlib.util.spec_from_file_location("c091_mask_packer_unpacker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runtime_parse_summary(payload: bytes) -> dict[str, Any]:
    unpacker = _load_unpacker()
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    return {
        "ok": True,
        "payload_format": header.get("payload_format", header.get("schema")),
        "members": {
            name: {
                "decoded_bytes": len(blob),
                "decoded_sha256": sha256_bytes(blob),
                "decoded_prefix_hex": blob[:8].hex(),
            }
            for name, blob in sorted(decoded.items())
        },
        "encoded_members": {
            item["name"]: {
                "bytes": int(item["bytes"]),
                "codec": str(item["codec"]),
                "sha256": str(item["sha256"]),
                "decoded_bytes": int(item["decoded_bytes"]),
                "decoded_sha256": str(item["decoded_sha256"]),
            }
            for item in header.get("members", [])
        },
    }


def stream_profile(archive: Path) -> dict[str, Any]:
    payload, zip_meta = read_single_payload_zip(archive)
    slices = parse_payload_slices(payload)
    decoded = {
        "masks.mkv": _decode_brotli("masks.mkv", slices.mask_br),
        "renderer.bin": _decode_brotli("renderer.bin", slices.renderer_br),
        "seg_tile_actions.bin.wire": _decode_brotli("seg_tile_actions.bin", slices.actions_br),
        "optimized_poses.qp1": _decode_brotli("optimized_poses.qp1", slices.pose_br),
    }
    encoded = {
        "masks.mkv": slices.mask_br,
        "renderer.bin": slices.renderer_br,
        "seg_tile_actions.bin": slices.actions_br,
        "optimized_poses.qp1": slices.pose_br,
    }
    return {
        "zip": zip_meta,
        "payload": {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "payload_format": slices.payload_format,
            "action_record_count": slices.action_record_count,
        },
        "encoded_streams": {
            name: {"bytes": len(blob), "sha256": sha256_bytes(blob)}
            for name, blob in sorted(encoded.items())
        },
        "decoded_streams": {
            name: {
                "bytes": len(blob),
                "sha256": sha256_bytes(blob),
                "prefix_hex": blob[:8].hex(),
            }
            for name, blob in sorted(decoded.items())
        },
        "runtime_parse": runtime_parse_summary(payload),
    }


def focused_brotli_params() -> list[tuple[int, int, int, int]]:
    """Small deterministic grid that includes the observed mask optimum."""
    params = [
        (11, 0, 19, 17),
        (11, 0, 16, 0),
        (11, 0, 24, 0),
        (11, 1, 19, 17),
        (11, 2, 19, 17),
    ]
    for quality in (10, 9, 6, 4, 0):
        params.append((quality, 0, 19, 17))
    return params


def mask_lossless_probes(mask_obu: bytes, source_mask_br: bytes) -> dict[str, Any]:
    """Probe exact-lossless repacking options for the decoded mask OBU stream."""
    rows: list[dict[str, Any]] = [
        {
            "codec": "source_brotli",
            "bytes": len(source_mask_br),
            "sha256": sha256_bytes(source_mask_br),
            "delta_bytes_vs_source": 0,
            "roundtrip_equal": True,
            "params": "source",
        },
        {
            "codec": "raw_obu",
            "bytes": len(mask_obu),
            "sha256": sha256_bytes(mask_obu),
            "delta_bytes_vs_source": len(mask_obu) - len(source_mask_br),
            "roundtrip_equal": True,
            "params": "none",
        },
    ]
    for quality, mode, lgwin, lgblock in focused_brotli_params():
        encoded = brotli.compress(mask_obu, quality=quality, mode=mode, lgwin=lgwin, lgblock=lgblock)
        rows.append(
            {
                "codec": "brotli",
                "bytes": len(encoded),
                "sha256": sha256_bytes(encoded),
                "delta_bytes_vs_source": len(encoded) - len(source_mask_br),
                "roundtrip_equal": brotli.decompress(encoded) == mask_obu,
                "params": {
                    "quality": quality,
                    "mode": mode,
                    "lgwin": lgwin,
                    "lgblock": lgblock,
                },
            }
        )
    for codec, encoded, decode in (
        ("zlib_9", zlib.compress(mask_obu, 9), zlib.decompress),
        ("lzma_preset9", lzma.compress(mask_obu, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
        ("bz2_9", bz2.compress(mask_obu, compresslevel=9), bz2.decompress),
    ):
        rows.append(
            {
                "codec": codec,
                "bytes": len(encoded),
                "sha256": sha256_bytes(encoded),
                "delta_bytes_vs_source": len(encoded) - len(source_mask_br),
                "roundtrip_equal": decode(encoded) == mask_obu,
                "params": "default_max",
            }
        )
    rows.sort(key=lambda row: (int(row["bytes"]), str(row["codec"])))
    best = rows[0]
    return {
        "source_mask_brotli_bytes": len(source_mask_br),
        "source_mask_brotli_sha256": sha256_bytes(source_mask_br),
        "decoded_mask_obu_bytes": len(mask_obu),
        "decoded_mask_obu_sha256": sha256_bytes(mask_obu),
        "probe_count": len(rows),
        "best": best,
        "best_savings_bytes": max(0, len(source_mask_br) - int(best["bytes"])),
        "rows": rows,
        "dispatch_implication": (
            "The only exact-lossless mask-stream win in this focused grid is too small "
            "for fixed-slice public payloads because the runtime fixed-slice parser "
            "assumes a 219472-byte mask segment; self-describing P3/P6 payloads can "
            "consume the shorter stream but pay their existing header overhead."
        ),
    }


def read_frontier_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "score": FRONTIER_SCORE_FALLBACK,
            "bytes": FRONTIER_BYTES_FALLBACK,
            "sha256": FRONTIER_SHA256_FALLBACK,
            "source": "fallback_constants",
        }
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {
        "score": float(data.get("canonical_score", data.get("score_recomputed_from_components", FRONTIER_SCORE_FALLBACK))),
        "bytes": int(data.get("archive_size_bytes", FRONTIER_BYTES_FALLBACK)),
        "sha256": str(data.get("provenance", {}).get("archive_sha256", "")) or FRONTIER_SHA256_FALLBACK,
        "avg_segnet_dist": data.get("avg_segnet_dist"),
        "avg_posenet_dist": data.get("avg_posenet_dist"),
        "n_samples": data.get("n_samples"),
        "score_rate_contribution": data.get("score_rate_contribution"),
        "score_seg_contribution": data.get("score_seg_contribution"),
        "score_pose_contribution": data.get("score_pose_contribution"),
        "source": _rel(path),
    }


def score_math(
    *,
    frontier_score: float,
    frontier_bytes: int,
    candidate_bytes: int,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    delta_bytes = int(candidate_bytes - frontier_bytes)
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = frontier_score + rate_delta
    component_gain_needed = max(0.0, score_if_components_unchanged - target_score)
    equivalent_bytes_needed = (
        int(math.ceil(component_gain_needed / RATE_SCORE_PER_BYTE))
        if component_gain_needed > 0
        else 0
    )
    return {
        "target_score": target_score,
        "delta_bytes_vs_frontier": delta_bytes,
        "rate_score_delta_vs_frontier": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "target_component_score_improvement_needed": component_gain_needed,
        "target_equivalent_bytes_needed_after_candidate": equivalent_bytes_needed,
        # Legacy keys are preserved for old ledgers/tests, but they now follow
        # the explicit target_score instead of hard-coding the obsolete 0.314.
        "sub314_component_score_improvement_needed": component_gain_needed,
        "sub314_equivalent_bytes_needed_after_candidate": equivalent_bytes_needed,
    }


def _load_json_if_exists(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def collect_candidate_rows(
    matrix_paths: list[Path],
    *,
    frontier: dict[str, Any],
    active_candidate_id: str,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for matrix_path in matrix_paths:
        if not matrix_path.exists():
            continue
        matrix = read_json(matrix_path)
        if not isinstance(matrix, dict):
            raise ValueError(f"{matrix_path} must contain a JSON object")
        for raw in matrix.get("candidates", []):
            archive_bytes = raw.get("archive_bytes")
            candidate_id = str(raw.get("candidate_id", ""))
            archive_sha = str(raw.get("archive_sha256", ""))
            if not candidate_id or archive_bytes is None or not archive_sha:
                continue
            key = f"{candidate_id}:{archive_sha}"
            if key in seen:
                continue
            seen.add(key)
            manifest = _load_json_if_exists(raw.get("manifest_path"))
            math_row = score_math(
                frontier_score=float(frontier["score"]),
                frontier_bytes=int(frontier["bytes"]),
                candidate_bytes=int(archive_bytes),
                target_score=target_score,
            )
            is_noop = bool(raw.get("noop")) or str(raw.get("noop_status", "")).startswith("noop")
            is_active = candidate_id == active_candidate_id
            exact_eval_justified = (
                bool(raw.get("dispatchable_after_gate"))
                and not is_noop
                and math_row["target_equivalent_bytes_needed_after_candidate"] == 0
            )
            if is_active:
                exact_eval_status = "already_active_do_not_touch_or_duplicate"
            elif exact_eval_justified:
                exact_eval_status = "rate_only_target_after_required_gate"
            elif bool(raw.get("dispatchable_after_gate")) and not is_noop:
                exact_eval_status = "not_target_without_component_gain"
            else:
                exact_eval_status = "not_ready_or_noop"
            row = {
                "candidate_id": candidate_id,
                "archive_path": raw.get("archive_path"),
                "archive_bytes": int(archive_bytes),
                "archive_sha256": archive_sha,
                "source_matrix": _rel(matrix_path),
                "manifest_path": raw.get("manifest_path"),
                "semantic_contract": raw.get("semantic_contract"),
                "dispatchable_after_gate": bool(raw.get("dispatchable_after_gate")),
                "next_dispatch_safety_gate": raw.get("next_dispatch_safety_gate"),
                "noop": is_noop,
                "active_elsewhere": is_active,
                "score_claim": bool(raw.get("score_claim")),
                "exact_eval_justified_from_this_plan": exact_eval_justified and not is_active,
                "exact_eval_status": exact_eval_status,
                "break_even_vs_c091": math_row,
                "manifest_stream_packing": (manifest or {}).get("stream_packing"),
                "manifest_runtime_parse_status": (manifest or {}).get("runtime_parse_validation", {}).get("status")
                or (manifest or {}).get("runtime_parse_validation", {}).get("decoded_parity_status"),
            }
            rows.append(row)
    rows.sort(key=lambda item: (item["break_even_vs_c091"]["target_equivalent_bytes_needed_after_candidate"], item["archive_bytes"]))
    return rows


def _run_checked(cmd: list[str], *, timeout: int = 300, capture: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, check=True, capture_output=capture, timeout=timeout)


def _decode_gray_frames(path: Path, frames: int) -> np.ndarray:
    result = _run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-pix_fmt",
            "gray",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout=300,
    )
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = 384 * 512
    if raw.size % frame_size:
        raise ValueError(f"decoded raw frame bytes {raw.size} not divisible by {frame_size}")
    return raw.reshape(raw.size // frame_size, 384, 512)


def _class_ids(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(gray.astype(np.float32) / 63.0).astype(np.int16), 0, 4)


def lossy_transcode_probe(
    *,
    mask_obu: bytes,
    output_dir: Path,
    frames: int,
    cq_values: list[int],
) -> dict[str, Any]:
    """Run a local short-sample AV1 mask transcode screen with class parity."""
    if frames <= 0:
        return {"status": "skipped", "reason": "frames <= 0"}
    missing = [name for name in ("ffmpeg", "aomenc") if shutil.which(name) is None]
    if missing:
        return {"status": "skipped", "reason": f"missing executables: {', '.join(missing)}"}

    probe_dir = output_dir / "lossy_transcode_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mask-transcode-", dir=str(probe_dir)) as tmp_text:
        tmp = Path(tmp_text)
        source_obu = tmp / "source.obu"
        source_obu.write_bytes(mask_obu)
        source_first = tmp / "source_first.obu"
        y4m = tmp / "source_gray.y4m"
        _run_checked(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_obu),
                "-frames:v",
                str(frames),
                "-c",
                "copy",
                "-f",
                "obu",
                str(source_first),
            ],
            timeout=300,
        )
        _run_checked(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_obu),
                "-frames:v",
                str(frames),
                "-pix_fmt",
                "gray",
                "-strict",
                "-1",
                "-f",
                "yuv4mpegpipe",
                str(y4m),
            ],
            timeout=300,
        )
        source_gray = _decode_gray_frames(source_obu, frames)
        source_classes = _class_ids(source_gray)
        rows: list[dict[str, Any]] = []
        for cq in cq_values:
            encoded = tmp / f"aom_cq{cq:02d}.obu"
            cmd = [
                "aomenc",
                "-y",
                "--quiet",
                "--codec=av1",
                "--obu",
                "--monochrome",
                "--cpu-used=6",
                "--end-usage=q",
                f"--cq-level={cq}",
                "--kf-max-dist=9999",
                "--disable-kf",
                "--lag-in-frames=35",
                "--enable-palette=1",
                "--enable-intrabc=1",
                "--enable-cdef=0",
                "--enable-restoration=0",
                "--threads=4",
                "-o",
                str(encoded),
                str(y4m),
            ]
            try:
                _run_checked(cmd, timeout=600)
                candidate_gray = _decode_gray_frames(encoded, frames)
                if candidate_gray.shape != source_gray.shape:
                    raise ValueError(f"shape mismatch: {candidate_gray.shape} != {source_gray.shape}")
                candidate_classes = _class_ids(candidate_gray)
                class_mismatch = candidate_classes != source_classes
                pixel_abs = np.abs(candidate_gray.astype(np.int16) - source_gray.astype(np.int16))
                rows.append(
                    {
                        "cq": cq,
                        "encoded_obu_bytes": encoded.stat().st_size,
                        "encoded_obu_sha256": sha256_file(encoded),
                        "delta_bytes_vs_source_first_frames": encoded.stat().st_size - source_first.stat().st_size,
                        "class_mismatch_count": int(class_mismatch.sum()),
                        "class_mismatch_rate": float(class_mismatch.mean()),
                        "max_pixel_abs_delta": int(pixel_abs.max()) if pixel_abs.size else 0,
                        "geometry_safe_class_parity": bool(not class_mismatch.any()),
                        "byte_saving_on_sample": encoded.stat().st_size < source_first.stat().st_size,
                        "command": " ".join(cmd),
                    }
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                rows.append(
                    {
                        "cq": cq,
                        "status": "failed",
                        "error": str(exc),
                        "geometry_safe_class_parity": False,
                        "byte_saving_on_sample": False,
                    }
                )
        rows.sort(key=lambda row: (not bool(row.get("byte_saving_on_sample")), int(row.get("encoded_obu_bytes", 10**12))))
        safe_byte_saving = [row for row in rows if row.get("geometry_safe_class_parity") and row.get("byte_saving_on_sample")]
        return {
            "status": "completed",
            "frames": frames,
            "source_first_frames_obu_bytes": source_first.stat().st_size,
            "source_first_frames_obu_sha256": sha256_file(source_first),
            "cq_values": cq_values,
            "rows": rows,
            "geometry_safe_byte_saving_candidate_found": bool(safe_byte_saving),
            "best_geometry_safe_byte_saving": safe_byte_saving[0] if safe_byte_saving else None,
            "interpretation": (
                "A lossy row is not geometry-safe unless class_mismatch_count is zero. "
                "Rows that save bytes but flip mask classes are planner signal only and "
                "must not be dispatched as exact-score candidates."
            ),
        }


def build_plan(
    *,
    frontier_archive: Path,
    frontier_eval: Path,
    candidate_matrices: list[Path],
    output_dir: Path,
    lossy_probe_frames: int,
    cq_values: list[int],
    active_candidate_id: str,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict[str, Any]:
    frontier_eval_data = read_frontier_eval(frontier_eval)
    profile = stream_profile(frontier_archive)
    mask_obu = _decode_brotli("frontier mask", parse_payload_slices(read_single_payload_zip(frontier_archive)[0]).mask_br)
    mask_br = parse_payload_slices(read_single_payload_zip(frontier_archive)[0]).mask_br
    lossless = mask_lossless_probes(mask_obu, mask_br)
    candidates = collect_candidate_rows(
        candidate_matrices,
        frontier=frontier_eval_data,
        active_candidate_id=active_candidate_id,
        target_score=target_score,
    )
    lossy = lossy_transcode_probe(
        mask_obu=mask_obu,
        output_dir=output_dir,
        frames=lossy_probe_frames,
        cq_values=cq_values,
    )
    rate_only_gap_score = float(frontier_eval_data["score"]) - target_score
    rate_only_gap_bytes = int(math.ceil(max(0.0, rate_only_gap_score) / RATE_SCORE_PER_BYTE))
    best_lossless_candidate = next(
        (
            row
            for row in candidates
            if row["archive_bytes"] < int(frontier_eval_data["bytes"])
            and not row["noop"]
            and row["dispatchable_after_gate"]
        ),
        None,
    )
    decision = {
        "score_claim": False,
        "remote_gpu_dispatch_performed": False,
        "dispatch_state_touched": False,
        "exact_eval_justified_now": False,
        "target_score": target_score,
        "reason": (
            f"No candidate in this mask/packer screen reaches target {target_score:.6f} "
            "by exact-lossless bytes. "
            "The mask stream has only a 7-byte exact-lossless Brotli improvement, and the "
            "best byte-saving rows still require material component gain or are already active elsewhere."
        ),
        "next_command": (
            f".venv/bin/python {TOOL} --frontier-archive {frontier_archive} "
            f"--frontier-eval-json {frontier_eval} --output-dir {output_dir} --lossy-probe-frames 600"
        ),
        "dispatch_if_active_result_condition": (
            "After the active PR77 action + C089 pose exact eval completes, only consider a follow-up "
            "P3/P6 mask-packer eval if that exact score is within 3-7 bytes of the target or the "
            "component result proves a real positive basin. Claim the lane first."
        ),
    }
    if best_lossless_candidate:
        needed = best_lossless_candidate["break_even_vs_c091"]["target_equivalent_bytes_needed_after_candidate"]
        decision["best_existing_byte_screen_candidate"] = {
            "candidate_id": best_lossless_candidate["candidate_id"],
            "archive_bytes": best_lossless_candidate["archive_bytes"],
            "archive_sha256": best_lossless_candidate["archive_sha256"],
            "target_equivalent_bytes_needed_after_candidate": needed,
            "sub314_equivalent_bytes_needed_after_candidate": needed,
            "exact_eval_status": best_lossless_candidate["exact_eval_status"],
        }
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": "empirical_deterministic_byte_screen",
        "score_claim": False,
        "remote_gpu_dispatch_performed": False,
        "dispatch_state_touched": False,
        "frontier": {
            **frontier_eval_data,
            "archive_path": _rel(frontier_archive),
            "target_score": target_score,
            "rate_only_gap_to_target_bytes": rate_only_gap_bytes,
            "rate_only_gap_to_target_score": rate_only_gap_score,
            "rate_only_gap_to_sub314_bytes": rate_only_gap_bytes,
            "rate_only_gap_to_sub314_score": rate_only_gap_score,
        },
        "frontier_stream_profile": profile,
        "mask_lossless_probe": lossless,
        "lossy_transcode_probe": lossy,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "decision": decision,
    }


def parse_cq_values(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one CQ value is required")
    for value in values:
        if not 0 <= value <= 63:
            raise ValueError(f"CQ value out of range [0, 63]: {value}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-archive", type=Path, default=DEFAULT_FRONTIER_ARCHIVE)
    parser.add_argument("--frontier-eval-json", type=Path, default=DEFAULT_FRONTIER_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-matrix",
        type=Path,
        action="append",
        default=None,
        help="Existing local candidate_matrix.json to ingest; may be repeated.",
    )
    parser.add_argument("--active-candidate-id", default=DEFAULT_ACTIVE_CANDIDATE_ID)
    parser.add_argument(
        "--target-score",
        type=float,
        default=DEFAULT_TARGET_SCORE,
        help="Contest score target used for rate-only break-even math. Defaults to the active <=0.31 target.",
    )
    parser.add_argument("--lossy-probe-frames", type=int, default=60)
    parser.add_argument("--aom-cq-values", default="63,50,40,35,30,25,20,15,10,5,0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrices = args.candidate_matrix if args.candidate_matrix is not None else list(DEFAULT_CANDIDATE_MATRICES)
    plan = build_plan(
        frontier_archive=args.frontier_archive,
        frontier_eval=args.frontier_eval_json,
        candidate_matrices=matrices,
        output_dir=args.output_dir,
        lossy_probe_frames=args.lossy_probe_frames,
        cq_values=parse_cq_values(args.aom_cq_values),
        active_candidate_id=args.active_candidate_id,
        target_score=args.target_score,
    )
    _write_json(args.output_dir / "plan.json", plan)
    _write_json(args.output_dir / "candidate_matrix.json", {"schema": SCHEMA, "candidates": plan["candidates"]})
    _write_json(args.output_dir / "mask_lossless_probe.json", plan["mask_lossless_probe"])
    _write_json(args.output_dir / "lossy_transcode_probe.json", plan["lossy_transcode_probe"])
    print(json.dumps({"output_dir": _rel(args.output_dir), "decision": plan["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
