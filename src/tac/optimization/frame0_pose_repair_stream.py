# SPDX-License-Identifier: MIT
"""Base-agnostic frame_0 pose-repair stream: the reusable carriage for the THIRD pose route.

WHY THIS EXISTS (the general mechanism, DERIVED + measured by ddm_bz1/js1).  SegNet reads ONLY the
last frame (`upstream/modules.py`: ``x = x[:, -1, ...]``), so **frame_0 is structurally
seg-invisible**: any additive correction placed there costs ZERO seg flips, by construction, on
every pair.  That makes frame_0 a FREE pose canvas -- a pose-carrying route distinct from bo1's
dead frame_1 post-hoc family (m09/#881) and from joint descent (#366).  Whenever a seg base has
edited frame_1 (moving what PoseNet sees) and thereby DAMAGED pose, the damage can be repaired on
frame_0 without touching the seg field.

WHY A GENERIC DCT BASIS (rule 118).  The repair delta is written as ``delta = coef @ A`` where
``A`` is a rank-k separable 2-D DCT synthesis basis.  The basis is DETERMINISTICALLY GENERATED at
inflate (``dct_atoms``) so it is FREE; only the ``k*k*3`` int16 coefficients per pair are COUNTED.
ddm_bz1 measured k=4 (48 coefficients = 96 B/pair): d_pose repaired to <= shipped across a damage
spectrum from 1.06x to 123.85x, int16 quantiser degradation < 5e-6 (naive rounding suffices; no
adaptive/QAT needed), and seg exactly preserved on every pair.

BASE-AGNOSTIC BY DESIGN.  This module holds the SECTION only.  It takes coefficients solved against
ANY seg base's damaged frame_1 and emits/parses a self-describing byte section that either archive
grammar (ix2 ``build_payload`` joint group, or the TR1 packet directory) can embed as one COUNTED
section.  The seg base is a SEPARATE, SWAPPABLE section -- swap ep854 / #827 / a phase field without
re-plumbing this one.  ``section_ledger`` reports the real per-section bytes for
``tac.submission_chain.build_byte_ledger`` to close against.

CARRIAGE NOTE (measured, honest).  The counted payload is ``k*k*3`` int16 per repaired pair.  Raw
that is 96 B/pair at k=4; the stream is additionally LZMA1-coded here and the SMALLER of raw/coded is
what ships, so ``section_ledger`` never over-counts.  The seg budget that funds the repair is
<=157.7 B/pair (1 B/pair == 0.00039952 S at n600), so k<=5 is the admissible range; k=4 is the
measured operating point.

NOT A SCORE.  This is deterministic carriage.  A d_pose only becomes a score through
``upstream/evaluate.py`` on contest hardware over the exact shipped bytes.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "MAGIC",
    "PoseRepairSection",
    "apply_pose_repair_scorer",
    "dct_atoms",
    "decode_pose_repair_stream",
    "encode_pose_repair_stream",
    "section_ledger",
]

MAGIC: Final = b"F0PR1!\x00\x00"          # frame_0 pose repair, v1
_INT16_MIN, _INT16_MAX = -32768, 32767
# LZMA1 raw, matching the coder the offset-field ledger uses (deterministic, self-contained).
_LZMA_FILTERS: Final = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]


def dct_atoms(k: int, seg_h: int, seg_w: int) -> np.ndarray:
    """The generic separable 2-D DCT-III synthesis atoms -- FREE at inflate (rule 118).

    ``atoms[a*k+b]`` is the (seg_h, seg_w) image whose only nonzero DCT-II coefficient is (a, b).
    ``delta = coef @ atoms.reshape(k*k, -1)`` reconstructs the repair.  scipy is used at BUILD time;
    an inflate implementation may ship a vendored idctn (still generic) with no counted bytes.
    """
    from scipy.fft import idctn

    if k <= 0:
        raise ValueError("k must be positive")
    atoms = np.zeros((k * k, seg_h, seg_w), np.float32)
    for a in range(k):
        for b in range(k):
            c = np.zeros((seg_h, seg_w))
            c[a, b] = 1.0
            atoms[a * k + b] = idctn(c, type=2, norm="ortho", axes=(-2, -1))
    return atoms


def apply_pose_repair_scorer(base0_scorer: np.ndarray, coef_i16: np.ndarray,
                             atoms: np.ndarray) -> np.ndarray:
    """Apply an int16 coefficient repair to a scorer-lattice frame_0 (H, W, 3) float/uint8.

    Deterministic and additive: ``round(clamp(base0 + (coef @ A), 0, 255))`` -- byte-identical to
    the object ddm_bz1/js1 scored.  Returns uint8 (H, W, 3).  NO scorer is consulted.
    """
    k2, hw = atoms.reshape(atoms.shape[0], -1).shape
    if coef_i16.shape != (3, k2):
        raise ValueError(f"coef shape {coef_i16.shape} != (3, {k2})")
    seg_h, seg_w = atoms.shape[1], atoms.shape[2]
    A = atoms.reshape(k2, -1).astype(np.float32)
    # macOS Accelerate raises a SPURIOUS FPE flag ("divide by zero encountered in matmul") for this
    # (3,k*k)@(k*k, H*W) shape even though inputs are finite and the result is byte-identical to the
    # float reference (asserted in tests).  Silence the platform quirk locally, not globally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        delta = (coef_i16.astype(np.float32) @ A).reshape(3, seg_h, seg_w)
    base = np.asarray(base0_scorer, np.float32)
    if base.shape == (seg_h, seg_w, 3):
        base = base.transpose(2, 0, 1)
    cur = np.clip(base + delta, 0.0, 255.0)
    return np.rint(cur).astype(np.uint8).transpose(1, 2, 0)


@dataclass(frozen=True)
class PoseRepairSection:
    """Parsed stream: coefficients keyed by pair, plus the geometry needed to rebuild the atoms."""

    k: int
    seg_h: int
    seg_w: int
    coefs: dict[int, np.ndarray]     # pair -> (3, k*k) int16

    @property
    def bytes_per_pair_raw(self) -> int:
        return self.k * self.k * 3 * 2


def encode_pose_repair_stream(coefs: dict[int, np.ndarray], *, k: int, seg_h: int,
                              seg_w: int) -> bytes:
    """Serialize {pair -> (3, k*k) int16 coef} into one self-describing COUNTED section.

    Layout (all little-endian):
        MAGIC(8) | k(u16) seg_h(u16) seg_w(u16) count(u16) coder(u8) |
        [ pair(u32) ]*count  |  coder(int16 body)
    ``coder`` = 0 stored, 1 LZMA1.  The smaller of stored/LZMA1 is chosen so the section never
    over-counts.  Pairs whose repair is all-zero (no damage / all-zero solve) may be omitted by the
    caller -- they carry 0 bytes, which is the correct rate for an unrepaired pair.
    """
    pairs = sorted(coefs)
    k2 = k * k
    body = bytearray()
    for p in pairs:
        c = np.asarray(coefs[p])
        if c.shape != (3, k2):
            raise ValueError(f"pair {p}: coef shape {c.shape} != (3, {k2})")
        if (c < _INT16_MIN).any() or (c > _INT16_MAX).any():
            raise ValueError(f"pair {p}: coefficient exceeds int16 range (need wider dtype/scale)")
        body += c.astype("<i2").tobytes()
    body_bytes = bytes(body)
    coded = lzma.compress(body_bytes, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    if len(coded) < len(body_bytes):
        coder, payload = 1, coded
    else:
        coder, payload = 0, body_bytes
    head = MAGIC + struct.pack("<HHHHB", k, seg_h, seg_w, len(pairs), coder)
    idx = b"".join(struct.pack("<I", p) for p in pairs)
    return head + idx + payload


def decode_pose_repair_stream(blob: bytes) -> PoseRepairSection:
    """Inverse of :func:`encode_pose_repair_stream`, closing exactly on the final byte."""
    if blob[:8] != MAGIC:
        raise ValueError("bad magic for frame_0 pose repair stream")
    k, seg_h, seg_w, count, coder = struct.unpack_from("<HHHHB", blob, 8)
    off = 8 + 9
    pairs = [struct.unpack_from("<I", blob, off + 4 * i)[0] for i in range(count)]
    off += 4 * count
    payload = blob[off:]
    k2 = k * k
    raw_len = count * k2 * 3 * 2
    if coder == 1:
        body = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    elif coder == 0:
        body = payload
    else:
        raise ValueError(f"unknown coder id {coder}")
    if len(body) != raw_len:
        raise ValueError(f"body {len(body)} != expected {raw_len}")
    arr = np.frombuffer(body, dtype="<i2").reshape(count, 3, k2).astype(np.int16)
    return PoseRepairSection(k=k, seg_h=seg_h, seg_w=seg_w,
                             coefs={p: arr[i].copy() for i, p in enumerate(pairs)})


def section_ledger(blob: bytes) -> dict:
    """The COUNTED byte accounting for this section, for build_byte_ledger to close against."""
    sec = decode_pose_repair_stream(blob)
    n = len(sec.coefs)
    return {
        "name": "frame0_pose_repair",
        "magic": MAGIC.rstrip(b"\x00").decode("ascii", "replace"),
        "counted_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "k": sec.k,
        "n_pairs_repaired": n,
        "bytes_per_pair_raw": sec.bytes_per_pair_raw,
        "counted_bytes_per_repaired_pair": (len(blob) / n) if n else 0.0,
        "note": "int16 k*k*3 coefficients per repaired pair; generic DCT basis is FREE (rule 118)",
    }
