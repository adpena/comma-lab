# SPDX-License-Identifier: MIT
"""Golden-vector generator for `tac-levelset-inflate`.

Produces the committed golden vectors for the level-set inflate's #257 store-nothing
POSE-CARRIER ξ decode (the pure-integer arithmetic range-decoder + column reconstruction).
The inputs are REAL: the per-pair ego-motion ξ derived from the frozen contest gt_poses
(``experiments/results/mlx_fleet_gt_cache/gt_n600.npz``) via the SAME canonical helpers the
shipped byte-close/inflate uses (``tac.boundary_math.warp_real_luma_frame0.xi_from_pose_calibration``
+ ``tac.boundary_math.xi_pose_coder.{quantize_xi, serialize_xi_payload}``). No synthetic RNG.

Each vector pins the SHA-256 of the DECODED OUTPUT (int64 little-endian) the Rust port must
reproduce bit-for-bit, plus the input byte fixtures the Rust port reads:

  * ``levelset_xi_range_decode_v1``  — the raw static-frequency range decoder
    (``tac.lossless.range_coder.decode_static_symbols``): stream + uint32 frequencies + count ->
    symbols. Digest = sha256(symbols as ``<i8`` LE).
  * ``levelset_xi_column_delta_v1``  — the on-path pose-carrier column decode
    (``tac.boundary_math.xi_pose_coder._channel_delta_decode`` minus the brotli-of-counts step,
    which is a separately-covered primitive): arithmetic decode -> ``+lo`` -> cumsum -> ``seed``.
    Digest = sha256(int64 column as ``<i8`` LE).

Run:  .venv/bin/python runtime-rs/crates/tac-levelset-inflate/generate_golden_vectors.py

Determinism: the gt cache is frozen; every downstream transform is deterministic, so the
committed fixtures reproduce byte-for-byte. Regenerate only when the ξ coder grammar changes.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GV = HERE / "golden_vectors"
REPO = HERE.parents[2]  # runtime-rs/crates/tac-levelset-inflate -> repo root
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math import warp_real_luma_frame0 as wrl  # noqa: E402
from tac.boundary_math import xi_pose_coder as xip  # noqa: E402
from tac.lossless.range_coder import decode_static_symbols  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _i64le(arr) -> bytes:
    return np.asarray(arr, dtype="<i8").tobytes(order="C")


def _real_xi_payload():
    """Real per-pair ξ payload bytes from the frozen contest gt_poses (delta_ar coder)."""
    d = np.load(GT_CACHE)
    gt_poses = np.asarray(d["gt_poses"])  # (P, 6)
    p_count = int(gt_poses.shape[0])
    pitch = 0.0
    s_t = s_r = 1.0
    xi = np.stack(
        [wrl.xi_from_pose_calibration(gt_poses[p], s_t, s_r, pitch) for p in range(p_count)], 0
    )
    q, scales = xip.quantize_xi(xi)
    blob = xip.serialize_xi_payload(q, scales, coder="delta_ar")
    return blob, p_count


def _iter_columns(blob: bytes):
    """Yield per-column (k, seed, lo, hi, counts uint32, stream bytes) from a delta_ar ξ payload."""
    import brotli

    magic = b"XIP2"
    assert blob[: len(magic)] == magic, "bad ξ-payload magic"
    off = len(magic)
    cid, p_count, d_dim = struct.unpack_from("<BHB", blob, off)
    off += 4
    assert cid == 1, f"expected delta_ar coder id 1, got {cid}"
    off += d_dim * 4  # skip scales
    for k in range(d_dim):
        seed, lo, hi = struct.unpack_from("<iii", blob, off)
        off += 12
        (mlen,) = struct.unpack_from("<I", blob, off)
        off += 4
        counts = (
            np.frombuffer(brotli.decompress(blob[off : off + mlen]), dtype=np.uint32)
            if mlen
            else None
        )
        off += mlen
        (slen,) = struct.unpack_from("<I", blob, off)
        off += 4
        stream = blob[off : off + slen]
        off += slen
        yield k, seed, lo, hi, counts, stream, p_count


def _write(name: str, blob: bytes) -> None:
    (GV / name).write_bytes(blob)


def main() -> int:
    GV.mkdir(parents=True, exist_ok=True)
    if not GT_CACHE.exists():
        print(f"gt cache not found: {GT_CACHE}", file=sys.stderr)
        return 2
    blob, p_count = _real_xi_payload()
    cols = list(_iter_columns(blob))
    # Pick two REAL columns with distinct alphabet sizes for coverage: the widest and a narrow one.
    usable = [c for c in cols if c[3] > c[2] and c[4] is not None]  # hi>lo and counts present
    usable.sort(key=lambda c: len(c[4]))
    narrow = usable[0]
    wide = usable[-1]

    # --- Vector A: raw static-frequency range decoder (widest alphabet). ---
    k, seed, lo, hi, counts, stream, pc = wide
    freqs = counts.astype(np.uint32)
    count = pc - 1
    symbols = decode_static_symbols(stream, count=count, frequencies=freqs.tolist())
    _write("levelset_xi_range_decode_v1_stream.bin", stream)
    _write("levelset_xi_range_decode_v1_frequencies.bin", freqs.astype("<u4").tobytes())
    manifest_a = {
        "schema": "levelset_xi_range_decode.v1",
        "sha256": _sha(_i64le(symbols)),
        "count": int(count),
        "n_frequencies": int(freqs.shape[0]),
        "total": int(freqs.sum()),
        "stream_bytes": int(len(stream)),
        "source": "real gt_poses -> xi_from_pose_calibration -> quantize_xi -> serialize_xi_payload(delta_ar), column %d" % k,
        "oracle": "tac.lossless.range_coder.decode_static_symbols",
        "output_encoding": "decoded symbols as little-endian int64 (<i8), C-order",
    }
    (GV / "levelset_xi_range_decode_v1.json").write_text(json.dumps(manifest_a, indent=2, sort_keys=True) + "\n")

    # --- Vector B: on-path pose-carrier column decode (narrow alphabet). ---
    k2, seed2, lo2, hi2, counts2, stream2, pc2 = narrow
    freqs2 = counts2.astype(np.uint32)
    count2 = pc2 - 1
    syms2 = np.asarray(decode_static_symbols(stream2, count=count2, frequencies=freqs2.tolist()), dtype=np.int64)
    deltas2 = syms2 + int(lo2)
    col2 = np.empty(pc2, dtype=np.int64)
    col2[0] = int(seed2)
    col2[1:] = int(seed2) + np.cumsum(deltas2)
    _write("levelset_xi_column_delta_v1_stream.bin", stream2)
    _write("levelset_xi_column_delta_v1_frequencies.bin", freqs2.astype("<u4").tobytes())
    manifest_b = {
        "schema": "levelset_xi_column_delta.v1",
        "sha256": _sha(_i64le(col2)),
        "p_count": int(pc2),
        "seed": int(seed2),
        "lo": int(lo2),
        "hi": int(hi2),
        "n_frequencies": int(freqs2.shape[0]),
        "total": int(freqs2.sum()),
        "stream_bytes": int(len(stream2)),
        "source": "real gt_poses -> ... -> serialize_xi_payload(delta_ar), column %d" % k2,
        "oracle": "tac.boundary_math.xi_pose_coder._channel_delta_decode (sans brotli-of-counts)",
        "output_encoding": "reconstructed int64 column (P,) as little-endian int64 (<i8), C-order",
    }
    (GV / "levelset_xi_column_delta_v1.json").write_text(json.dumps(manifest_b, indent=2, sort_keys=True) + "\n")

    print("wrote golden vectors to", GV)
    print("  A range_decode : count=%d n_freq=%d total=%d stream=%dB sha=%s" % (
        manifest_a["count"], manifest_a["n_frequencies"], manifest_a["total"],
        manifest_a["stream_bytes"], manifest_a["sha256"][:16]))
    print("  B column_delta : P=%d seed=%d lo=%d n_freq=%d total=%d stream=%dB sha=%s" % (
        manifest_b["p_count"], manifest_b["seed"], manifest_b["lo"], manifest_b["n_frequencies"],
        manifest_b["total"], manifest_b["stream_bytes"], manifest_b["sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
