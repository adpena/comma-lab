# SPDX-License-Identifier: MIT
"""cascade_c_prime_frame_1_segnet_waterfill inflate runtime — numpy-portable per HNeRV parity L4.

Per HNeRV parity discipline lessons L4 + L5 + strict-scorer-rule:

- ~180 LOC inflate budget (HNeRV parity L4 + L7 substrate_engineering exception
  per Catalog #270 substrate trainer scope).
- <=2 external deps: numpy + brotli (NO torch, NO smp, NO scorers).
- NO scorer load per CLAUDE.md "Strict scorer rule" non-negotiable.
- Catalog #205 inline-device-fork via canonical helper :func:`select_inflate_device`
  (local helper mirroring the canonical contract at
  ``tac.substrates._shared.inflate_runtime.select_inflate_device``).
- Catalog #295 self-contained: NO PYTHONPATH-shim or sister-substrate imports;
  archive.py routing decision decoder is local.

v1 SCAFFOLD forward path:

1. Read ``<archive_dir>/0.bin`` bytes; call :func:`parse_archive`.
2. Decode routing decision via brotli + np.unpackbits (sister of architecture.py).
3. For each pair:
   - If routing=FRAME_0: lookup frame-0 menu index + apply frame-0 mode perturbation.
   - If routing=FRAME_1: lookup frame-1 menu index + apply frame-1 mode perturbation.
   - Apply 6-DOF affine warp from pose delta to derive frame_1 from frame_0.
4. Write contest raw stream to ``<output_stem>.raw``.

**SCAFFOLD STATUS** per Catalog #220: `score_improvement_mechanism_status =
SCAFFOLD_DEFERRED_INTEGRATION`. The frame-0/frame-1 mode perturbation lookup
tables are NOT yet wired (operator-routable 7th-order: build per-pair mode
lookup tables via trainer at COMPRESS time + ship as inflate-time fixed tables
per Daubechies multi-scale partition discovery pattern).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from .archive import (
    CCPF_MAGIC,
    CCPF_VERSION_V1,
    POSE_DIMS,
    CascadeCPrimeArchive,
    parse_archive,
)


def select_inflate_device(env_var: str = "PACT_INFLATE_DEVICE") -> str:
    """Canonical inflate device selector mirroring tac.substrates._shared.inflate_runtime.

    Per Catalog #205: operator-pinnable via PACT_INFLATE_DEVICE env var (auto/cpu/cuda);
    `mps` explicitly refused per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    raw = os.environ.get(env_var, "auto").strip().lower()
    if raw == "mps":
        raise RuntimeError(
            f"{env_var}=mps refused per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable"
        )
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        # CPU default for numpy-portable inflate; CUDA path would require torch
        return "cpu"
    raise ValueError(f"{env_var}={raw!r} unrecognized; expected one of (auto, cpu, cuda)")


def _dequantize_pose_deltas(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode uint8-quantized per-pair pose deltas.

    Returns (n_pairs, POSE_DIMS) float32 array of 6-DOF pose deltas.

    SCAFFOLD: linear dequantization with fixed scale (1.0 / 127.5; range [-1, 1]).
    Production trainer ships per-pair scale + offset tables for adaptive quantization
    per sister NSCS06 v8 pattern.
    """
    arr = np.frombuffer(arc.pose_delta_stream, dtype=np.uint8).reshape(
        arc.n_pairs, POSE_DIMS
    ).astype(np.float32)
    return (arr - 127.5) / 127.5


def _decode_frame_0_menu_indices(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode 4-bit-packed frame-0 menu indices.

    Returns (n_pairs,) uint8 array of frame-0 menu indices in [0, frame_0_menu_size).
    """
    packed = np.frombuffer(arc.frame_0_menu_index_stream, dtype=np.uint8)
    # Low nibble first; high nibble second (sister of pack_archive)
    low_nibble = packed & 0x0F
    high_nibble = (packed >> 4) & 0x0F
    interleaved = np.empty(packed.shape[0] * 2, dtype=np.uint8)
    interleaved[0::2] = low_nibble
    interleaved[1::2] = high_nibble
    return interleaved[: arc.n_pairs]


def _decode_frame_1_menu_indices(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode byte-aligned frame-1 menu indices (SCAFFOLD: 8-bit per pair).

    Returns (n_pairs,) uint8 array of frame-1 menu indices in [0, frame_1_menu_size).
    7th-order iteration packs to 3-bit per pair for K=8 menu.
    """
    return np.frombuffer(arc.frame_1_menu_index_stream, dtype=np.uint8)


def _affine_warp_frame1_from_frame0(
    frame_0: np.ndarray, pose_delta: np.ndarray
) -> np.ndarray:
    """Apply 6-DOF affine warp to derive frame_1 from frame_0.

    SCAFFOLD: identity warp (returns frame_0 unchanged). The 7th-order trainer
    ships a per-pair 6-DOF affine warp parameterized by the dequantized pose
    delta (sister of NSCS06 v8 ``_affine_warp_frame1_from_frame0``).

    Inputs:
        frame_0: (H, W, 3) uint8 RGB frame at output resolution
        pose_delta: (POSE_DIMS,) float32 6-DOF pose delta vector

    Returns:
        frame_1: (H, W, 3) uint8 RGB frame
    """
    # SCAFFOLD identity warp; 7th-order iteration ships affine warp.
    # The identity warp at SCAFFOLD time is the EXPLICIT no-op per
    # Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION — pose deltas are parsed
    # (Catalog #139 no-op detector) but not yet OPERATIONALLY applied.
    return frame_0.copy()


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one video from CH-CCP-FRAME1-WATERFILL archive bytes.

    SCAFFOLD: emits placeholder raw stream of correct shape. The 7th-order
    trainer + production frame-0/frame-1 menu lookup tables + ROI warp ship
    the actual rendered frames.

    Per CLAUDE.md "Forbidden score claims": SCAFFOLD output produces
    structurally-valid archive bytes consumable by inflate but NOT a real
    score-improving rendered video. The pre-symposium per Catalog #325 gates
    promotion to a real rendering path.
    """
    arc = parse_archive(archive_bytes)
    routing = arc.routing_decision  # (n_pairs,) int8
    pose_deltas = _dequantize_pose_deltas(arc)
    f0_indices = _decode_frame_0_menu_indices(arc)
    f1_indices = _decode_frame_1_menu_indices(arc)

    # Verify routing-decision sidecar bytes are structurally consumed
    # (Catalog #139 no-op detector + Catalog #272 distinguishing-feature integration contract)
    n_frame_0 = int(np.sum(routing == 0))
    n_frame_1 = int(np.sum(routing == 1))
    assert n_frame_0 + n_frame_1 == arc.n_pairs

    # SCAFFOLD output: structurally-valid raw stream
    # Production trainer ships actual rendered frames via per-pair mode lookup
    # SCAFFOLD_DEFERRED_INTEGRATION per Catalog #220
    H, W = 384, 512
    n_frames = arc.n_pairs * 2  # 2 frames per pair (frame_0 + frame_1)
    placeholder_frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    out_path = output_stem.with_suffix(".raw")
    placeholder_frames.tofile(out_path)
    return out_path


def main_cli() -> int:
    """CLI entry point matching contest inflate.sh contract.

    Usage: python -m tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate
               <archive_dir> <output_dir> <file_list>

    Per Catalog #146 contest 3-arg signature.
    """
    if len(sys.argv) != 4:
        print(
            "usage: python -m tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate "
            "<archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2

    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_inflate_device()
    print(f"[cascade_c_prime_frame_1_segnet_waterfill] inflate device={device}", file=sys.stderr)

    file_list = file_list_path.read_text().splitlines()
    for line in file_list:
        line = line.strip()
        if not line:
            continue
        archive_path = archive_dir / "0.bin"
        if not archive_path.exists():
            print(f"FATAL: archive missing at {archive_path}", file=sys.stderr)
            return 3
        archive_bytes = archive_path.read_bytes()
        output_stem = output_dir / Path(line).stem
        out_path = inflate_one_video(archive_bytes, output_stem)
        print(f"[cascade_c_prime_frame_1_segnet_waterfill] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
