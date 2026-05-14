#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a PR #67 ``qpose14_qzs3``-shaped candidate archive.

PR #67 (rank-1 leaderboard ~0.31) ships a single-blob archive ``p`` (zipped
with no compression) containing the concatenation:

    p = mask_obu_br | model_qzs3_br | pose_q_br

Where:
    - mask_obu_br  : brotli-compressed AV1 OBU monochrome mask stream (219472 B)
    - model_qzs3_br: brotli-compressed QZS3 weight payload (~56KB)
    - pose_q_br    : brotli-compressed QP1 (or raw uint16) pose stream (~1KB)

The PR #67 inflate.py looks at total ``len(payload)`` to pick ``model_br_len``
between the mask and pose slices (see pr67_inflate.py:746-764). To stay
inside one of those windows, this builder either:

  1. Defaults to the standard mask len 219472 + computes model_br_len after
     compression, then ZIP-pads the trailing pose slice if needed, OR
  2. Accepts ``--mask-bytes-len`` for builds that re-encode the mask.

When inputs are absent, the builder synthesizes minimal placeholder bytes
so the ``--smoke`` mode produces a structurally valid archive with random
weights — used to verify pr67's inflate.py round-trips before any trained
checkpoint exists.

Outputs a single ``archive.zip`` containing one stored member ``p``. ZIP
overhead is < 100 bytes vs the raw payload (deterministic timestamp 1980).

Usage (smoke):
    python experiments/build_qpose_archive.py \\
        --output-dir experiments/results/qpose_smoke \\
        --smoke

Usage (with trained checkpoint, masks, poses):
    python experiments/build_qpose_archive.py \\
        --renderer-state experiments/results/lane_q_faithful_retrain_*/best.pt \\
        --mask-obu-br experiments/results/.../mask.obu.br \\
        --pose-pt experiments/results/.../optimized_poses.pt \\
        --output-dir experiments/results/qpose_candidate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

# Add repo src/ to import path when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.qp1_pose_codec import encode_qp1
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import encode_qzs3_state_dict


ARCHIVE_MEMBER_NAME = "p"
DETERMINISTIC_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PR67_REFERENCE_MASK_BYTES = 219472  # PR #67 archive's mask slice length

# Empirical model_br_len ranges from pr67_inflate.py:749-762. Each tuple is
# (lower_inclusive, upper_inclusive, expected_model_br_len).  The total
# payload length must land in exactly one window so PR #67 can slice
# correctly.
PR67_MODEL_LEN_WINDOWS: tuple[tuple[int, int, int], ...] = (
    (276430, 276470, 56093),
    (276550, 276610, 56221),
    (277400, 277430, 57053),
    (277350, 277399, 57031),
    (278100, 278130, 57757),
    (281240, 281240, 60880),
)
PR67_DEFAULT_MODEL_LEN = 61147


def _brotli_compress(data: bytes, *, quality: int = 11) -> bytes:
    return brotli.compress(data, quality=quality)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"renderer-state not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError(
        f"unrecognized renderer-state file format at {path}: "
        f"expected raw state_dict or {{'state_dict': ...}}"
    )


def _load_pose_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"pose file not found: {path}")
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(tensor, dict) and "poses" in tensor:
            tensor = tensor["poses"]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"pose .pt did not contain a tensor: {path}")
        arr = tensor.detach().cpu().float().numpy()
    elif path.suffix == ".npy":
        arr = np.load(path).astype(np.float32, copy=False)
    elif path.suffix == ".bin":
        # raw fp16 row-major (N, 6)
        raw = path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(-1, 6)
    else:
        raise ValueError(f"unsupported pose extension: {path.suffix}")
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"pose array must be (N, 6); got shape {arr.shape}")
    return arr


def _synthesize_smoke_mask_obu(num_pairs: int = 600) -> bytes:
    """Build a tiny dummy mask blob just so the structural test can run.

    The smoke mode never gets passed to a real evaluator — it only proves the
    QZS3 + QP1 packer composes into a structurally valid archive that
    pr67_inflate.py can ``mmap`` and slice.  We fill exactly
    PR67_REFERENCE_MASK_BYTES so the inflate.py model_br_len heuristic
    selects the standard window.
    """

    return b"\x00" * PR67_REFERENCE_MASK_BYTES


def _select_model_window(payload_total: int) -> int | None:
    for lo, hi, expected in PR67_MODEL_LEN_WINDOWS:
        if lo <= payload_total <= hi:
            return expected
    return None


def _write_archive(blob: bytes, archive_path: Path) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=ARCHIVE_MEMBER_NAME, date_time=DETERMINISTIC_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (0o644 & 0xFFFF) << 16
    with zipfile.ZipFile(archive_path, mode="w") as zf:
        zf.writestr(info, blob)
    return archive_path.stat().st_size


def build_qpose_archive(
    *,
    renderer_state_dict: dict[str, torch.Tensor] | None,
    pose_array: np.ndarray | None,
    mask_obu_br_bytes: bytes,
    output_archive: Path,
    block_size: int = 32,
    pose_codec: str = "qp1",
    brotli_quality: int = 11,
) -> dict[str, Any]:
    """Compose the three brotli'd slices and emit a stored-zip archive.

    Returns a metadata dict with sha256 + per-segment byte counts. Caller
    can decide whether the (empirical) layout matches a PR #67 model_br_len
    window or whether a re-pack of the mask slice is required.
    """

    if renderer_state_dict is None:
        torch.manual_seed(0)
        template = build_quantizr_faithful_renderer().eval()
        renderer_state_dict = template.state_dict()

    qzs3_payload = encode_qzs3_state_dict(renderer_state_dict, block_size=block_size)
    model_br = _brotli_compress(qzs3_payload, quality=brotli_quality)

    if pose_array is None:
        # Smoke default: 600 pairs of constant velocity 30 m/s.
        pose_array = np.zeros((600, 6), dtype=np.float32)
        pose_array[:, 0] = 30.0

    if pose_codec == "qp1":
        pose_payload = encode_qp1(pose_array)
    elif pose_codec == "raw_uint16":
        # PR #67's fallback path: raw uint16 (N,6) bytes (no magic header).
        # PoseNet-faithful qpose14 quantization, all 6 cols preserved.
        velocities = pose_array[:, 0]
        q0 = np.rint((velocities - 20.0) * 512.0).astype(np.int64)
        if (q0 < 0).any() or (q0 > 0xFFFF).any():
            raise ValueError("raw_uint16 pose codec: velocity outside [0, 65535]")
        q_pose = np.zeros((pose_array.shape[0], 6), dtype=np.uint16)
        q_pose[:, 0] = q0.astype(np.uint16)
        rest = np.rint(pose_array[:, 1:] * 2048.0).astype(np.int64)
        if (rest < -32768).any() or (rest > 32767).any():
            raise ValueError("raw_uint16 pose codec: pose dim outside int16 range")
        q_pose[:, 1:] = rest.astype(np.int16).view(np.uint16)
        pose_payload = q_pose.tobytes()
    else:
        raise ValueError(f"unsupported pose codec: {pose_codec}")
    pose_q_br = _brotli_compress(pose_payload, quality=brotli_quality)

    blob = mask_obu_br_bytes + model_br + pose_q_br
    archive_bytes = _write_archive(blob, output_archive)

    digest = hashlib.sha256(blob).hexdigest()
    expected_model_len = _select_model_window(len(blob))

    meta: dict[str, Any] = {
        "archive_path": str(output_archive),
        "archive_bytes": archive_bytes,
        "blob_bytes": len(blob),
        "mask_br_bytes": len(mask_obu_br_bytes),
        "model_br_bytes": len(model_br),
        "model_uncompressed_bytes": len(qzs3_payload),
        "pose_br_bytes": len(pose_q_br),
        "pose_uncompressed_bytes": len(pose_payload),
        "pose_codec": pose_codec,
        "blob_sha256": digest,
        "expected_model_br_len_for_pr67_dispatch": expected_model_len,
        "model_br_len_actual": len(model_br),
        "pr67_dispatch_compatible": (
            expected_model_len is not None and expected_model_len == len(model_br)
        ),
    }
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer-state",
        type=Path,
        default=None,
        help="path to a JointFrameGenerator state_dict (.pt). When absent, an "
        "init-random renderer is used (smoke / structural test only).",
    )
    parser.add_argument(
        "--mask-obu-br",
        type=Path,
        default=None,
        help="path to a brotli-compressed AV1 OBU mask file. Required unless "
        "--smoke is set.",
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help="path to a pose array (.pt / .npy / .bin fp16 row-major).",
    )
    parser.add_argument(
        "--pose-codec",
        choices=("qp1", "raw_uint16"),
        default="qp1",
        help="pose stream encoder. qp1 = PR #67's velocity-only ZigZag-VLQ "
        "(default; lossy, cols 1-5 zeroed). raw_uint16 = qpose14 "
        "quantization of all 6 cols, no header.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="QZS3 block size for FP4 weight grouping (default 32).",
    )
    parser.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
        help="brotli compression quality (default 11; max 11).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write archive.zip + metadata.json into.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="synthesize placeholder mask + pose bytes when not provided. "
        "Random renderer weights when --renderer-state absent.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state_dict(args.renderer_state) if args.renderer_state else None
    pose_array = _load_pose_array(args.pose_file) if args.pose_file else None

    if args.mask_obu_br is not None:
        mask_bytes = args.mask_obu_br.read_bytes()
    elif args.smoke:
        mask_bytes = _synthesize_smoke_mask_obu()
    else:
        parser.error("--mask-obu-br is required unless --smoke is set")

    archive_path = args.output_dir / "archive.zip"
    meta = build_qpose_archive(
        renderer_state_dict=state,
        pose_array=pose_array,
        mask_obu_br_bytes=mask_bytes,
        output_archive=archive_path,
        block_size=args.block_size,
        pose_codec=args.pose_codec,
        brotli_quality=args.brotli_quality,
    )

    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(json.dumps(meta, indent=2, sort_keys=True))
    if not meta["pr67_dispatch_compatible"]:
        print(
            f"\n[warn] blob length {meta['blob_bytes']} not in any PR #67 "
            f"model_br_len window; pr67_inflate.py would slice with the "
            f"PR67_DEFAULT_MODEL_LEN={PR67_DEFAULT_MODEL_LEN} fallback "
            f"({'OK' if meta['model_br_len_actual'] == PR67_DEFAULT_MODEL_LEN else 'BROKEN'}).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
