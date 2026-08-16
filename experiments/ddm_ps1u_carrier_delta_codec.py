#!/usr/bin/env python3
"""ddm_ps1u — counted sparse codec for frame-0 int12 carrier deltas.

WHY A NEW CODEC AND NOT ``Q2C1``
--------------------------------
The shipped compensation overlay (``ddm_qs2_compensation_overlay_runtime``) is
structurally too small for this payload: it admits **1–15 pairs** with deltas in
**[-3, 4]** (3 bits). The ps1u pose solve produces **60 pairs** with deltas measured in
**[-29, 48]** across ~8.2 of 12 dimensions. Widening Q2C1's fields would change the
shipped format for its existing 7-pair consumer, so this is a sibling format, not an edit.

FORMAT ``P1D1`` (all big-endian bit order)
------------------------------------------
    magic  "P1D1"                      4 B
    u8     version(4) | reserved(4)
    u16    pair_count
    then, bit-packed:
      per pair, ascending: 10-bit DELTA of the pair index from the previous one
                           (first pair absolute); the list is strictly increasing,
                           so gaps code smaller than raw indices
      per pair: 12-bit support mask (which dimensions are nonzero)
      per set bit, in dimension order: Exp-Golomb order-0 of zigzag(value)

Exp-Golomb is used rather than a fixed width because the delta magnitude distribution is
heavy-headed (median |Δ| ~5) with a long tail (max 48): a fixed 7-bit field would truncate
and a fixed 12-bit field would waste ~40% of the payload. No trained table ships — the code
is a fixed, generic construction, so nothing here is video-derived.

INVARIANTS (all fail closed)
----------------------------
* pair indices strictly increasing, inside [0, 600)
* zero-support pairs are non-canonical (they would code a no-op)
* zero values inside the support are non-canonical
* decode must reproduce the exact input; ``encode`` self-verifies before returning
* trailing padding bits must be zero, and the payload length must be exact

AXIS. ``[local-CPU $0 exact codec]``. This module measures a SECTION size exactly; it does
NOT measure an archive delta — the byte half of any admitted row still comes from re-running
the real coder stack and diffing real archive bytes.
"""

from __future__ import annotations

import json
import hashlib
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np

MAGIC = b"P1D1"
VERSION = 1
PAIR_COUNT = 600
DIMENSIONS = 12
INT12_MIN = -2048
INT12_MAX = 2047


class CarrierDeltaCodecError(ValueError):
    """The counted overlay or its receiving int12 lattice is invalid."""


class _BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def put(self, value: int, width: int) -> None:
        if width <= 0 or not 0 <= value < (1 << width):
            raise CarrierDeltaCodecError(f"bit field {value} does not fit {width} bits")
        self.bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))

    def put_exp_golomb(self, value: int) -> None:
        """Order-0 Exp-Golomb of a NON-NEGATIVE integer."""
        if value < 0:
            raise CarrierDeltaCodecError("exp-golomb takes non-negative values")
        code = value + 1
        width = code.bit_length()
        self.bits.extend([0] * (width - 1))
        self.put(code, width)

    def tobytes(self) -> bytes:
        if not self.bits:
            return b""
        return np.packbits(
            np.asarray(self.bits, dtype=np.uint8), bitorder="big"
        ).tobytes()


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.bits = np.unpackbits(
            np.frombuffer(payload, dtype=np.uint8), bitorder="big"
        )
        self.cursor = 0

    def take(self, width: int) -> int:
        if width <= 0 or self.cursor + width > self.bits.size:
            raise CarrierDeltaCodecError("truncated carrier delta bitstream")
        value = 0
        for bit in self.bits[self.cursor : self.cursor + width]:
            value = (value << 1) | int(bit)
        self.cursor += width
        return value

    def take_exp_golomb(self) -> int:
        zeros = 0
        while True:
            if self.cursor >= self.bits.size:
                raise CarrierDeltaCodecError("truncated exp-golomb prefix")
            if int(self.bits[self.cursor]) == 1:
                break
            zeros += 1
            self.cursor += 1
            if zeros > 32:
                raise CarrierDeltaCodecError("exp-golomb prefix exceeds domain")
        return self.take(zeros + 1) - 1


def _zigzag(value: int) -> int:
    return (value << 1) if value >= 0 else ((-value << 1) - 1)


def _unzigzag(code: int) -> int:
    return (code >> 1) if code % 2 == 0 else -((code + 1) >> 1)


def encode_carrier_deltas(
    pair_indices: Sequence[int], deltas: np.ndarray
) -> bytes:
    """Encode a sorted sparse set of int12 carrier deltas. Self-verifies."""
    pairs = [int(v) for v in pair_indices]
    values = np.asarray(deltas, dtype=np.int64)
    if values.shape != (len(pairs), DIMENSIONS):
        raise CarrierDeltaCodecError(f"overlay geometry differs: {values.shape}")
    if not pairs:
        raise CarrierDeltaCodecError("empty overlay is non-canonical")
    if sorted(set(pairs)) != pairs:
        raise CarrierDeltaCodecError("pair indices must be sorted and unique")
    if pairs[0] < 0 or pairs[-1] >= PAIR_COUNT:
        raise CarrierDeltaCodecError("pair index exceeds the n600 domain")
    if len(pairs) > 0xFFFF:
        raise CarrierDeltaCodecError("pair count exceeds the u16 header field")

    writer = _BitWriter()
    previous = 0
    for index, pair in enumerate(pairs):
        gap = pair if index == 0 else pair - previous - 1
        if not 0 <= gap < 1024:
            raise CarrierDeltaCodecError("pair index gap exceeds 10 bits")
        writer.put(gap, 10)
        previous = pair
    masks: list[int] = []
    for row in range(len(pairs)):
        mask = 0
        for dimension in range(DIMENSIONS):
            if int(values[row, dimension]):
                mask |= 1 << dimension
        if not mask:
            raise CarrierDeltaCodecError("zero-support pair is non-canonical")
        masks.append(mask)
        writer.put(mask, DIMENSIONS)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if mask & (1 << dimension):
                writer.put_exp_golomb(_zigzag(int(values[row, dimension])))
    payload = MAGIC + bytes((VERSION << 4,)) + struct.pack(">H", len(pairs)) + writer.tobytes()
    got_pairs, got_values = decode_carrier_deltas(payload)
    if got_pairs != pairs or not np.array_equal(got_values, values.astype(np.int32)):
        raise CarrierDeltaCodecError("encode/decode round-trip differs")
    return payload


def decode_carrier_deltas(payload: bytes) -> tuple[list[int], np.ndarray]:
    """Decode and reject truncation, disorder, zero support, and nonzero padding."""
    if len(payload) < 7 or not payload.startswith(MAGIC):
        raise CarrierDeltaCodecError("invalid carrier delta magic or length")
    version = payload[4] >> 4
    if version != VERSION or (payload[4] & 0x0F):
        raise CarrierDeltaCodecError("unsupported carrier delta header")
    (count,) = struct.unpack(">H", payload[5:7])
    if not 1 <= count <= PAIR_COUNT:
        raise CarrierDeltaCodecError("carrier delta pair count out of domain")
    reader = _BitReader(payload[7:])
    pairs: list[int] = []
    previous = -1
    for index in range(count):
        gap = reader.take(10)
        pair = gap if index == 0 else previous + gap + 1
        if pair >= PAIR_COUNT or pair <= previous:
            raise CarrierDeltaCodecError("pair index order or domain differs")
        pairs.append(pair)
        previous = pair
    masks = [reader.take(DIMENSIONS) for _ in range(count)]
    if any(mask == 0 for mask in masks):
        raise CarrierDeltaCodecError("zero-support pair is non-canonical")
    values = np.zeros((count, DIMENSIONS), dtype=np.int32)
    for row, mask in enumerate(masks):
        for dimension in range(DIMENSIONS):
            if mask & (1 << dimension):
                value = _unzigzag(reader.take_exp_golomb())
                if value == 0:
                    raise CarrierDeltaCodecError("zero value inside support")
                values[row, dimension] = value
    if int(reader.bits[reader.cursor :].sum()) != 0:
        raise CarrierDeltaCodecError("nonzero padding bits")
    if reader.bits.size - reader.cursor >= 8:
        raise CarrierDeltaCodecError("carrier delta overlay has trailing bytes")
    return pairs, values


def apply_carrier_deltas(codes: np.ndarray, payload: bytes) -> np.ndarray:
    """Apply a counted overlay to the real 600x12 signed-int12 lattice."""
    lattice = np.asarray(codes, dtype=np.int32).copy()
    if lattice.shape != (PAIR_COUNT, DIMENSIONS):
        raise CarrierDeltaCodecError("receiving lattice geometry differs")
    pairs, values = decode_carrier_deltas(payload)
    updated = lattice[pairs].astype(np.int64) + values.astype(np.int64)
    if np.any(updated < INT12_MIN) or np.any(updated > INT12_MAX):
        raise CarrierDeltaCodecError("overlay drives a coefficient outside signed-int12")
    lattice[pairs] = updated.astype(np.int32)
    return lattice


def price(pair_indices: Sequence[int], deltas: np.ndarray) -> dict[str, Any]:
    payload = encode_carrier_deltas(pair_indices, deltas)
    n = len(list(pair_indices))
    return {
        "schema": "ddm_ps1u_carrier_delta_section_price.v1",
        "axis": "[local-CPU $0 exact codec section size]",
        "score_claim": False,
        "format": MAGIC.decode(),
        "pairs": n,
        "section_bytes": len(payload),
        "bytes_per_pair": len(payload) / n,
        "section_sha256": hashlib.sha256(payload).hexdigest(),
        "rate_delta_s_if_section_is_the_archive_delta": len(payload) * 25.0 / 37_545_489,
        "caveat": (
            "this is the exact SECTION size, not the archive delta; the byte half of any "
            "admitted row must come from re-running the real coder stack and diffing real "
            "archive.zip bytes (the qs2 precedent maps a 36 B section ~1:1 onto archive "
            "bytes for this family, but that is a precedent, not this candidate's receipt)"
        ),
    }


def build_candidate(rows_glob: str) -> dict[str, Any]:
    """Assemble the counted section + the PRE-REGISTERED admission arithmetic.

    Every number here is either exact (section bytes, rate) or explicitly labelled as an
    advisory-axis measurement that is NOT the shipping axis (the pose reduction). The
    admission bar is stated BEFORE the T4 row exists."""
    import glob

    rows: list[dict[str, Any]] = []
    for path in glob.glob(rows_glob):
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda r: int(r["pair"]))
    active = [r for r in rows if any(r["code_delta"])]
    if not active:
        raise CarrierDeltaCodecError("no pair carries a nonzero delta")
    pairs = [int(r["pair"]) for r in active]
    deltas = np.asarray([r["code_delta"] for r in active], dtype=np.int64)
    section = encode_carrier_deltas(pairs, deltas)
    section_bytes = len(section)
    rate_delta_s = section_bytes * 25.0 / 37_545_489

    base = np.asarray([r["base_objective"] for r in active], dtype=np.float64)
    final = np.asarray([r["final_objective"] for r in active], dtype=np.float64)
    advisory_base_dpose = 1.474653494795297e-04
    advisory_new_dpose = advisory_base_dpose - (base.sum() - final.sum()) / PAIR_COUNT

    cuda_base_dpose = 6.885642960696714e-06
    cuda_base_contribution = float(np.sqrt(10.0 * cuda_base_dpose))
    required_contribution = cuda_base_contribution - rate_delta_s
    required_dpose = required_contribution**2 / 10.0
    required_reduction = 1.0 - required_dpose / cuda_base_dpose

    return {
        "schema": "ddm_ps1u_pose_candidate_bundle.v1",
        "axis": "[counted section exact; pose leg macOS-CPU advisory NON-PROMOTABLE]",
        "score_claim": False,
        "promotion_eligible": False,
        "pairs_edited": len(pairs),
        "pairs": pairs,
        "section_format": MAGIC.decode(),
        "section_bytes": section_bytes,
        "section_bytes_per_pair": section_bytes / len(pairs),
        "section_sha256": hashlib.sha256(section).hexdigest(),
        "rate_delta_s": rate_delta_s,
        "advisory_axis": {
            "note": (
                "the advisory chain decodes a DIFFERENT object than the shipping chain "
                "(DEVICE_DEPENDENT_DECODE_CONFIRMED: cpu raw e5539653… vs cuda raw "
                "9a6b75e5… on the same archive), so these numbers are DIRECTIONAL ONLY"
            ),
            "base_dpose_n600": advisory_base_dpose,
            "projected_dpose_n600": advisory_new_dpose,
            "reduction_of_total_n600_dpose": 1.0 - advisory_new_dpose / advisory_base_dpose,
            "mass_weighted_reduction_on_edited_pairs": float(
                1.0 - final.sum() / base.sum()
            ),
        },
        "PRE_REGISTERED_ADMISSION": {
            "authority": "upstream/evaluate.py, contest-CUDA T4, n600, exact archive bytes",
            "incumbent_S": 0.15959729295498598,
            "cuda_base_dpose": cuda_base_dpose,
            "cuda_base_pose_contribution": cuda_base_contribution,
            "pose_prize_ceiling_s": cuda_base_contribution,
            "rate_cost_s": rate_delta_s,
            "required_cuda_dpose_after": required_dpose,
            "required_cuda_dpose_reduction_fraction": required_reduction,
            "rule": (
                "ADMIT iff the T4 row's recomputed S < 0.15959729295498598 with the pose leg "
                "MEASURED (not projected). Equivalently the CUDA d_pose must fall by more "
                f"than {100 * required_reduction:.2f}% to pay for the {section_bytes} B "
                "section. The advisory row is directional only and CANNOT admit."
            ),
        },
        "transfer_is_the_measurement": (
            "the edit is to the int12 CODES, which both decoders consume; the qs2/re1 "
            "precedent is that CPU-solved candidates realized on T4 to ~6 significant "
            "figures despite the decode divergence, which is why this is worth one row — "
            "but that precedent is not a guarantee, and TRANSFER IS EXACTLY WHAT THIS ROW "
            "MEASURES"
        ),
    }


def _self_test() -> None:
    rng = np.random.default_rng(20260816)
    pairs = sorted(rng.choice(PAIR_COUNT, size=60, replace=False).tolist())
    values = rng.integers(-29, 49, size=(60, DIMENSIONS))
    mask = rng.random((60, DIMENSIONS)) < 0.68
    values = np.where(mask, values, 0)
    for row in range(60):
        if not values[row].any():
            values[row, 0] = 3
    payload = encode_carrier_deltas(pairs, values)
    got_pairs, got_values = decode_carrier_deltas(payload)
    assert got_pairs == pairs and np.array_equal(got_values, values.astype(np.int32))
    lattice = np.zeros((PAIR_COUNT, DIMENSIONS), dtype=np.int32)
    out = apply_carrier_deltas(lattice, payload)
    assert np.array_equal(out[pairs], values.astype(np.int32))
    assert bool((np.delete(out, pairs, axis=0) == 0).all())
    for corrupt, why in (
        (payload[:-1], "truncation"),
        (b"XXXX" + payload[4:], "magic"),
        (payload + b"\x00", "trailing bytes"),
    ):
        try:
            decode_carrier_deltas(corrupt)
        except CarrierDeltaCodecError:
            continue
        raise AssertionError(f"{why} was not refused")
    try:
        apply_carrier_deltas(np.full((PAIR_COUNT, DIMENSIONS), INT12_MAX, np.int32), payload)
    except CarrierDeltaCodecError:
        pass
    else:
        raise AssertionError("int12 overflow was not refused")
    print(json.dumps(price(pairs, values), indent=2))
    print("self-test: PASS (round-trip, apply, 3 refusals, int12 overflow refusal)")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("candidate", help="assemble the counted section + admission arithmetic")
    c.add_argument("--rows-glob", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--section-out", required=True)
    args = ap.parse_args(argv)
    if args.command == "selftest":
        _self_test()
        return 0
    bundle = build_candidate(args.rows_glob)
    import glob as _glob

    rows: list[dict[str, Any]] = []
    for path in _glob.glob(args.rows_glob):
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda r: int(r["pair"]))
    active = [r for r in rows if any(r["code_delta"])]
    section = encode_carrier_deltas(
        [int(r["pair"]) for r in active],
        np.asarray([r["code_delta"] for r in active], dtype=np.int64),
    )
    from pathlib import Path

    Path(args.section_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.section_out).write_bytes(section)
    Path(args.out).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    a = bundle["PRE_REGISTERED_ADMISSION"]
    print(f"[ps1u-cand] pairs {bundle['pairs_edited']} section {bundle['section_bytes']} B "
          f"({bundle['section_bytes_per_pair']:.2f} B/pair) sha {bundle['section_sha256'][:16]}")
    print(f"[ps1u-cand] rate cost {a['rate_cost_s']:+.6f} S ; pose ceiling "
          f"{a['pose_prize_ceiling_s']:.6f} S")
    print(f"[ps1u-cand] ADMISSION BAR: CUDA d_pose must fall > "
          f"{100 * a['required_cuda_dpose_reduction_fraction']:.2f}%")
    print(f"[ps1u-cand] -> {args.out} , {args.section_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
