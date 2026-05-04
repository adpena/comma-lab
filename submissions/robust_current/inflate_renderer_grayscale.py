#!/usr/bin/env python3
"""Lane MM dispatch: existing renderer + Selfcomp grayscale-LUT mask decode.

This is the rate-attack inflate path for Lane MM. The archive is identical
to a Lane A archive EXCEPT that masks.mkv is replaced by grayscale.mkv —
a single-plane AV1 monochrome stream where each pixel value encodes a
class via the Selfcomp gray targets [0, 255, 64, 192, 128] (sigma=15
Gaussian softmax LUT at decode time).

Decoding pipeline:

    grayscale.mkv  -> PyAV decode (uint8 gray)
                  -> create_gaussian_softmax_lut[256, 5] (sigma=15)
                  -> embedding lookup -> (B, 5, H, W) probability map
                  -> argmax            -> (B, H, W) int64 class id
                  -> 5-channel one-hot
                  -> EXISTING renderer.bin (MaskRenderer / ASYM)
                  -> frames at scorer resolution (384x512)
                  -> bicubic upsample to camera resolution (1164, 874)
                  -> rgb24 .raw

Lane MM hypothesis: grayscale-LUT mask cuts rate ~50% with no quality
loss because (1) AV1 monochrome skips chroma planes and (2) the spread
gray targets [0, 64, 128, 192, 255] are 51-pixel-spaced, so AV1
quantization noise of ±10-15 levels stays within the nearest-neighbour
basin and decodes back to the correct class.

This script reuses the existing Lane A inflate_renderer.py renderer load
+ frame generation logic — only the mask decode stage swaps in. The
implementation calls into the Lane A inflate_renderer module's
``_load_renderer_and_masks`` + ``_generate_and_write`` indirectly via a
mask-source override.

Usage (called from inflate.sh PYTHON_INFLATE=renderer_grayscale arm):

    python inflate_renderer_grayscale.py <archive_dir> <inflated_dir> <names_file>

Strict-scorer-rule compliance: this path does NOT load PoseNet or SegNet
at inflate time. The renderer.bin is the only neural component.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "upstream"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Lazy import: tac.mask_grayscale_lut is light-weight; the rest is loaded
# on-demand to avoid pulling in the full inflate_renderer module if the
# user is just running the codec by itself.
from tac.mask_grayscale_lut import create_gaussian_softmax_lut  # noqa: E402

_REPAIR_MAGIC = b"AMR1"
_REPAIR_SCHEMA = "alpha4_residual_repair_amr1_v1"
_REPAIR_HEADER_STRUCT = ">I"
_REPAIR_RECORD_STRUCT = ">IHHHB"
_REPAIR_RECORD_SIZE = struct.calcsize(_REPAIR_RECORD_STRUCT)
_OPTIONAL_REPAIR_MEMBERS = (
    ("alpha4_residual_repair.amr1", "raw"),
    ("alpha4_residual_repair.amr1.xz", "lzma_xz"),
    ("alpha4_residual_repair.amr1.zlib", "zlib"),
    ("alpha4_residual_repair.amr1.br", "brotli"),
)


def _class_tensor_sha256(classes: torch.Tensor) -> str:
    data = classes.to(torch.uint8).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _decompress_repair_payload(path: Path, codec: str) -> bytes:
    payload = path.read_bytes()
    if codec == "raw":
        return payload
    if codec == "lzma_xz":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    if codec == "zlib":
        return zlib.decompress(payload)
    if codec == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:  # pragma: no cover - contest env dependent
            raise RuntimeError(
                "alpha4_residual_repair.amr1.br requires brotli in the inflate environment"
            ) from exc
        return brotli.decompress(payload)
    raise RuntimeError(f"unsupported repair codec {codec!r}")


def _load_optional_repair_payload(archive_dir: Path) -> tuple[str, bytes] | None:
    matches: list[tuple[str, str, Path]] = []
    for member_name, codec in _OPTIONAL_REPAIR_MEMBERS:
        path = archive_dir / member_name
        if path.exists():
            matches.append((member_name, codec, path))
    if len(matches) > 1:
        names = ", ".join(name for name, _, _ in matches)
        raise RuntimeError(f"multiple Alpha residual repair payloads present: {names}")
    if not matches:
        return None
    member_name, codec, path = matches[0]
    return member_name, _decompress_repair_payload(path, codec)


def _decode_amr1_repair_payload(payload: bytes) -> tuple[dict, list[tuple[int, int, int, int, int]]]:
    if not payload.startswith(_REPAIR_MAGIC):
        raise RuntimeError("Alpha residual repair payload missing AMR1 magic")
    offset = len(_REPAIR_MAGIC)
    if len(payload) < offset + struct.calcsize(_REPAIR_HEADER_STRUCT):
        raise RuntimeError("Alpha residual repair payload missing header length")
    (header_length,) = struct.unpack(_REPAIR_HEADER_STRUCT, payload[offset : offset + 4])
    offset += 4
    header_end = offset + int(header_length)
    if header_end > len(payload):
        raise RuntimeError("Alpha residual repair header extends past payload")
    header = json.loads(payload[offset:header_end].decode("utf-8"))
    offset = header_end
    if header.get("schema") != _REPAIR_SCHEMA:
        raise RuntimeError(f"unsupported Alpha residual repair schema {header.get('schema')!r}")
    if header.get("record_struct") != _REPAIR_RECORD_STRUCT:
        raise RuntimeError(
            f"unsupported Alpha residual repair record struct {header.get('record_struct')!r}"
        )
    shape = header.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise RuntimeError(f"Alpha residual repair header has invalid shape {shape!r}")
    t, h, w = [int(value) for value in shape]
    if t <= 0 or h <= 0 or w <= 0:
        raise RuntimeError(f"Alpha residual repair header has nonpositive shape {shape!r}")
    record_count = int(header.get("record_count", -1))
    if record_count < 0:
        raise RuntimeError(f"Alpha residual repair record_count invalid: {record_count}")
    expected = offset + record_count * _REPAIR_RECORD_SIZE
    if expected != len(payload):
        raise RuntimeError(
            f"Alpha residual repair payload size mismatch: expected {expected}, got {len(payload)}"
        )
    runs: list[tuple[int, int, int, int, int]] = []
    for _ in range(record_count):
        frame_index, y, x0, length, class_id = struct.unpack(
            _REPAIR_RECORD_STRUCT,
            payload[offset : offset + _REPAIR_RECORD_SIZE],
        )
        offset += _REPAIR_RECORD_SIZE
        frame_index = int(frame_index)
        y = int(y)
        x0 = int(x0)
        length = int(length)
        class_id = int(class_id)
        if not (0 <= frame_index < t):
            raise RuntimeError(f"Alpha repair frame out of range: {frame_index}")
        if not (0 <= y < h):
            raise RuntimeError(f"Alpha repair row out of range: {y}")
        if not (0 <= x0 < w):
            raise RuntimeError(f"Alpha repair x0 out of range: {x0}")
        if length <= 0 or x0 + length > w:
            raise RuntimeError(f"Alpha repair run length out of range: x0={x0} length={length}")
        if not (0 <= class_id < 5):
            raise RuntimeError(f"Alpha repair class id out of range: {class_id}")
        runs.append((frame_index, y, x0, length, class_id))
    return header, runs


def _apply_amr1_repair(classes: torch.Tensor, payload: bytes, *, source_name: str) -> torch.Tensor:
    header, runs = _decode_amr1_repair_payload(payload)
    expected_shape = tuple(int(value) for value in header["shape"])
    if tuple(int(value) for value in classes.shape) != expected_shape:
        raise RuntimeError(
            f"Alpha residual repair shape {expected_shape} does not match decoded classes "
            f"{tuple(int(value) for value in classes.shape)}"
        )
    expected_candidate_sha = header.get("candidate_mask_u8_sha256")
    if expected_candidate_sha:
        actual_candidate_sha = _class_tensor_sha256(classes)
        if actual_candidate_sha != expected_candidate_sha:
            raise RuntimeError(
                f"Alpha residual repair candidate SHA mismatch for {source_name}: "
                f"{actual_candidate_sha} != {expected_candidate_sha}"
            )
    repaired = classes.clone()
    for frame_index, y, x0, length, class_id in runs:
        repaired[frame_index, y, x0 : x0 + length] = class_id
    selection = header.get("selection") if isinstance(header.get("selection"), dict) else {}
    if selection.get("partial_repair") is False and header.get("source_mask_u8_sha256"):
        actual_source_sha = _class_tensor_sha256(repaired)
        if actual_source_sha != header["source_mask_u8_sha256"]:
            raise RuntimeError(
                f"Alpha residual repair source SHA mismatch for {source_name}: "
                f"{actual_source_sha} != {header['source_mask_u8_sha256']}"
            )
    return repaired


def _maybe_apply_residual_repair(archive_dir: Path, classes: torch.Tensor) -> torch.Tensor:
    loaded = _load_optional_repair_payload(archive_dir)
    if loaded is None:
        return classes
    member_name, payload = loaded
    repaired = _apply_amr1_repair(classes, payload, source_name=member_name)
    print(
        f"[lane-mm] applied Alpha residual repair {member_name}: "
        f"{len(payload):,} raw AMR1 bytes",
        file=sys.stderr,
    )
    return repaired


def _decode_grayscale_mkv_to_classes(
    grayscale_mkv: Path,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Decode grayscale.mkv -> (N, target_h, target_w) int64 class ids.

    Uses ffmpeg subprocess to dump raw uint8 gray frames (matches the
    existing inflate_renderer.py mask-decode pattern), then bicubic-resamples
    to (target_h, target_w) and argmaxes the LUT-soft-projection.

    Args:
        grayscale_mkv: path to the 1-channel AV1 monochrome stream.
        target_h, target_w: scorer-input resolution (typically 384, 512).

    Returns:
        int64 (N, target_h, target_w) tensor with class ids in [0, NUM_CLASSES).
    """
    if not grayscale_mkv.exists():
        raise FileNotFoundError(f"grayscale.mkv not found: {grayscale_mkv}")

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            str(grayscale_mkv),
        ],
        capture_output=True, text=True, timeout=30, check=True,
    )
    parts = probe.stdout.strip().split(",")
    src_w, src_h = int(parts[0]), int(parts[1])

    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(grayscale_mkv),
            "-f", "rawvideo", "-pix_fmt", "gray", "-v", "error",
            "pipe:1",
        ],
        capture_output=True, timeout=300, check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = src_h * src_w
    n = len(raw) // frame_size
    if len(raw) % frame_size != 0:
        raise ValueError(
            f"decoded gray data {len(raw)} not divisible by {src_h}x{src_w}={frame_size}"
        )
    pixels = torch.from_numpy(raw.reshape(n, src_h, src_w).copy())  # (N, H_src, W_src) uint8

    # Resample to scorer resolution if needed (bicubic on float, then round).
    if (src_h, src_w) != (target_h, target_w):
        pix_f = pixels.to(torch.float32).unsqueeze(1)  # (N, 1, H, W)
        pix_f = F.interpolate(
            pix_f, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        pixels = pix_f.round().clamp(0, 255).squeeze(1).to(torch.uint8)

    # Selfcomp Gaussian-LUT projection -> (N, target_h, target_w, NUM_CLASSES) prob
    # then argmax -> (N, H, W) int64.
    # Lane FR-MM (sigma sweep): operator can override LUT sigma via
    # LANE_MM_SIGMA env var (set by lane scripts, sourced via config.env).
    # Defaults to LUT_DEFAULT_SIGMA (15.0) for the canonical Lane MM path.
    sigma_env = os.environ.get("LANE_MM_SIGMA")
    if sigma_env is not None and sigma_env.strip():
        try:
            sigma = float(sigma_env)
            if sigma <= 0:
                raise ValueError("sigma must be > 0")
            lut = create_gaussian_softmax_lut(sigma=sigma)
            print(
                f"[inflate-grayscale] LANE_MM_SIGMA override active: sigma={sigma}",
                file=sys.stderr,
            )
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"LANE_MM_SIGMA env var present but invalid ({sigma_env!r}): {exc}"
            ) from exc
    else:
        lut = create_gaussian_softmax_lut()  # (256, 5) sigma=LUT_DEFAULT_SIGMA
    gray_long = pixels.to(torch.long)
    probability_map = F.embedding(gray_long, lut)  # (N, H, W, 5)
    masks = probability_map.argmax(dim=-1).to(torch.int64)
    return masks


def inflate_renderer_grayscale(
    archive_dir: Path, inflated_dir: Path, video_names_file: Path
) -> None:
    """Lane MM inflate: grayscale-LUT mask decode -> existing renderer.

    The renderer load + frame generation reuses inflate_renderer.py's
    well-tested helpers via a monkey-patch on its mask-source resolver.
    We avoid duplicating ~1000 lines of renderer-loading code that is
    already audit-passed in the Lane A path.
    """
    # Defer the heavy import until we have the masks ready.
    import importlib.util

    archive_dir = Path(archive_dir)
    inflated_dir = Path(inflated_dir)

    grayscale_mkv = archive_dir / "grayscale.mkv"
    classes = _decode_grayscale_mkv_to_classes(
        grayscale_mkv, target_h=384, target_w=512
    )
    classes = _maybe_apply_residual_repair(archive_dir, classes)
    print(
        f"[lane-mm] decoded grayscale -> {tuple(classes.shape)} "
        f"unique classes={sorted(classes.unique().tolist())}",
        file=sys.stderr,
    )

    # Stash the decoded masks back into the archive directory in the
    # legacy Lane A format so the existing inflate_renderer.py can pick
    # them up via its native (class * 255 // 4) mask-decode path.
    # Strategy: write a small grayscale.mkv-equivalent in the legacy
    # encoding so the existing path needs zero changes.
    # CLAUDE.md "no scorers at inflate" still holds (no SegNet load).
    legacy_mkv = archive_dir / "masks.mkv"
    if not legacy_mkv.exists():
        # Build a temporary masks.mkv from the decoded class ids using the
        # legacy encoding (class * 63). We use ffmpeg's rawvideo input.
        from tac.mask_codec import encode_masks
        encode_masks(classes, legacy_mkv, crf=50, fps=20)
        print(
            f"[lane-mm] wrote translated masks.mkv ({legacy_mkv.stat().st_size:,} B) "
            f"from grayscale.mkv for legacy renderer path",
            file=sys.stderr,
        )

    # Now hand off to the existing Lane A inflate_renderer.py.
    inflate_renderer_path = Path(__file__).resolve().parent / "inflate_renderer.py"
    if not inflate_renderer_path.exists():
        raise FileNotFoundError(f"inflate_renderer.py not found at {inflate_renderer_path}")
    spec = importlib.util.spec_from_file_location(
        "inflate_renderer", str(inflate_renderer_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load inflate_renderer.py from {inflate_renderer_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("inflate_renderer", mod)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "inflate_renderer"):
        raise AttributeError("inflate_renderer.py missing inflate_renderer() entry point")
    mod.inflate_renderer(str(archive_dir), str(inflated_dir), str(video_names_file))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: inflate_renderer_grayscale.py <archive_dir> <inflated_dir> <video_names_file>",
            file=sys.stderr,
        )
        sys.exit(2)
    inflate_renderer_grayscale(
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    )
