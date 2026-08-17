"""ddm_ra2 — hv1 archive section census + carrier scale-field gauge status.

Measures, on the LIVE frontier archive (hv1 ep0634, 182,759 B, sha256
80d9c8c6...), the exact RX1M section census including raw body sizes, and
re-derives the strict sub-0.15 byte bar from the authority eval components.

WHAT THIS SETTLES
-----------------
*   The section census closes exactly against 182,759 B, with raw (decoded)
    body sizes per section.  Independent of any memo.
*   The strict INTEGRAL byte bar.  Several circulating figures are continuous
    (14,413.4 B) and are one byte short of crossing 0.15.

WHAT THIS DOES NOT SETTLE (stated, not worked around)
-----------------------------------------------------
The CPR1 basis/coefficient field census ("the 22,032 B vs 22,155 B pool
contradiction" between ``ddm_ra2`` and ``ddm_ra2crr``) is NOT resolved here.
The brotli-decoded carrier body (22,219 B) does not begin with ``CPR1``,
``CAP1`` or ``F0C1``, and ``carrier_repack.materialize_cpr1`` from the
hv1_base_control generation runtime refuses it when called directly on that
slice.  The generation receipt records ``receiver_closed=true`` and
``parser=legacy``, so the full receiver plainly does decode it: this extraction
is one legacy-parser step short, NOT evidence of a different wire.  The census
is therefore reported as UNRESOLVED with the exact blocker, rather than guessed.

Axis: [macOS-CPU advisory, scorer-free, exact byte + exact arithmetic].
score_claim=false.  No scorer forward, no render, no dispatch, no launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import struct
import zipfile
from pathlib import Path

import brotli

# The frontier pointer, re-hashed at use time (never trusted from a memo).
FRONTIER_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
FRONTIER_BYTES = 182_759

# Exact contest-CUDA components, read from the authority eval JSON at
# experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/
# MODAL_REMOTE_RESULT.json (score_axis="contest_cuda", score_claim=true).
FRONTIER_D_SEG = 0.00029611
FRONTIER_D_POSE = 6.88e-06
UNCOMPRESSED_SIZE = 37_545_489
RATE_COEFFICIENT = 25.0 / UNCOMPRESSED_SIZE
TARGET_SCORE = 0.15

# RX1M container contract, mirrored from
# tools/audit_archive_coder_axis.py::parse_rx1.
RX1_MAGIC = b"RX1M"
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_CODEC_XZ = 1
RX1_CODEC_BROTLI = 2
RESIDUAL_TABLE_BYTES = 96

CARRIER_MAGICS = {"CPR1": b"CPR1", "CAP1": b"CAP1", "F0C1": b"F0C1"}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def parse_rx1m(archive: Path) -> dict:
    """Exact section census of the RX1M container, closing to the file size."""
    raw = archive.read_bytes()
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if names != ["p"]:
            raise RuntimeError(f"expected exactly member 'p', found {names!r}")
        outer = handle.read("p")

    if not outer.startswith(RX1_MAGIC):
        raise RuntimeError("archive payload is not an RX1M container")
    (_m, version, codec, table_mode, reserved, hpac_n, semantic_n, carrier_n) = (
        RX1_HEADER.unpack_from(outer)
    )
    if version != 1 or codec not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI):
        raise RuntimeError("unsupported RX1M model header")

    offset = RX1_HEADER.size
    coded: dict[str, bytes] = {}
    for name, length in (
        ("hpac", hpac_n),
        ("semantic", semantic_n),
        ("carrier", carrier_n),
    ):
        if length <= 0 or offset + length > len(outer):
            raise RuntimeError(f"RX1M section {name} is truncated")
        coded[name] = outer[offset:offset + length]
        offset += length
    tail = outer[offset:]
    if len(tail) <= RESIDUAL_TABLE_BYTES:
        raise RuntimeError("RX1M residual/token tail is truncated")
    residual, tokens = tail[:RESIDUAL_TABLE_BYTES], tail[RESIDUAL_TABLE_BYTES:]

    def decode(name: str, blob: bytes) -> bytes:
        if name == "hpac" and codec == RX1_CODEC_XZ:
            return lzma.decompress(blob, format=lzma.FORMAT_XZ)
        return brotli.decompress(blob)

    bodies = {name: decode(name, blob) for name, blob in coded.items()}

    sections = {
        "zip_framing": len(raw) - len(outer),
        "rx1m_header": RX1_HEADER.size,
        "hpac_coded": len(coded["hpac"]),
        "semantic_coded": len(coded["semantic"]),
        "carrier_coded": len(coded["carrier"]),
        "residual_table": len(residual),
        "token_stream": len(tokens),
    }
    accounted = sum(sections.values())
    if accounted != len(raw):
        raise RuntimeError(f"section census does not close: {accounted} != {len(raw)}")

    return {
        "sections_coded": sections,
        "census_closes_exactly": True,
        "rx1m_codec": int(codec),
        "rx1m_table_mode": int(table_mode),
        "rx1m_reserved": int(reserved),
        "raw_bodies": {
            f"{name}_raw": len(body) for name, body in bodies.items()
        },
        "raw_body_magic": {
            name: body[:4].decode("latin1") if body[:4].isascii() else body[:4].hex()
            for name, body in bodies.items()
        },
        "bodies": bodies,
        "residual": residual,
        "tokens": tokens,
        "archive_bytes": len(raw),
        "archive_sha256": sha256_bytes(raw),
    }


def carrier_wire_status(carrier_body: bytes) -> dict:
    """Report which known carrier wire the shipped body presents, if any."""
    matched = [
        name for name, magic in CARRIER_MAGICS.items()
        if carrier_body.startswith(magic)
    ]
    return {
        "carrier_raw_bytes": len(carrier_body),
        "carrier_raw_sha256": sha256_bytes(carrier_body),
        "leading_4_bytes_hex": carrier_body[:4].hex(),
        "matches_known_wire": matched or None,
        "cpr1_field_census_resolved": bool(matched),
        "blocker": (
            None if matched else
            "brotli-decoded carrier body matches none of CPR1/CAP1/F0C1; "
            "materialize_cpr1 from the hv1_base_control generation runtime "
            "refuses this slice directly. GENERATION_RECEIPT records "
            "receiver_closed=true and parser=legacy, so the receiver does "
            "decode it -- this extraction is one legacy-parser step short. "
            "The 22,032 B vs 22,155 B pool contradiction between ddm_ra2 and "
            "ddm_ra2crr is therefore UNRESOLVED here."
        ),
    }


def strict_bar() -> dict:
    """Re-derive the strict INTEGRAL sub-0.15 byte bar from the components."""
    seg_term = 100.0 * FRONTIER_D_SEG
    pose_term = math.sqrt(10.0 * FRONTIER_D_POSE)
    distortion = seg_term + pose_term
    base_score = distortion + RATE_COEFFICIENT * FRONTIER_BYTES

    ceiling = FRONTIER_BYTES
    while distortion + RATE_COEFFICIENT * ceiling >= TARGET_SCORE:
        ceiling -= 1
    continuous = (TARGET_SCORE - distortion) / RATE_COEFFICIENT

    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "distortion_total": distortion,
        "rate_term": RATE_COEFFICIENT * FRONTIER_BYTES,
        "score_recomposed": base_score,
        "score_per_byte": RATE_COEFFICIENT,
        "archive_ceiling_bytes_strict": ceiling,
        "archive_ceiling_bytes_continuous": continuous,
        "bytes_required_strict": FRONTIER_BYTES - ceiling,
        "bytes_required_continuous": FRONTIER_BYTES - continuous,
        "note": (
            "archive bytes are integral and the target is strict: the "
            "continuous figure is one byte short of crossing 0.15."
        ),
    }


def price(delta_bytes: int, bar: dict) -> dict:
    """Price a ZERO-DISTORTION byte delta. d_seg and d_pose are unchanged."""
    return {
        "delta_bytes": delta_bytes,
        "delta_score": -RATE_COEFFICIENT * delta_bytes,
        "resulting_archive_bytes": FRONTIER_BYTES - delta_bytes,
        "resulting_score": bar["distortion_total"]
        + RATE_COEFFICIENT * (FRONTIER_BYTES - delta_bytes),
        "fraction_of_strict_bar": delta_bytes / bar["bytes_required_strict"],
        "clears_admission_bar_3p5e_6": RATE_COEFFICIENT * delta_bytes > 3.5e-6,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--retain-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    parsed = parse_rx1m(args.archive)
    if parsed["archive_sha256"] != FRONTIER_SHA256:
        raise SystemExit(
            f"REFUSED: not the frontier pointer (sha {parsed['archive_sha256']})"
        )
    if parsed["archive_bytes"] != FRONTIER_BYTES:
        raise SystemExit("REFUSED: archive byte count is not the frontier pointer")

    bodies = parsed.pop("bodies")
    residual = parsed.pop("residual")
    tokens = parsed.pop("tokens")
    bar = strict_bar()

    result = {
        "schema": "ddm_ra2_section_census.v1",
        "axis": "[macOS-CPU advisory, scorer-free, exact byte + exact arithmetic]",
        "score_claim": False,
        "promotable": False,
        "frontier_moved": False,
        "custody": {
            "archive_path": str(args.archive),
            "archive_sha256": parsed["archive_sha256"],
            "archive_bytes": parsed["archive_bytes"],
            "matches_frontier_pointer": True,
        },
        "census": parsed,
        "carrier_wire": carrier_wire_status(bodies["carrier"]),
        "strict_bar": bar,
        "zero_distortion_pricing": {
            "basis_scales_gauge_ceiling_48B": price(48, bar),
            "ra2_cpr1_inner_coder_realized_230B": price(230, bar),
            "bundle_278B": price(278, bar),
        },
        "operating_point": {
            "d_seg": FRONTIER_D_SEG,
            "d_pose": FRONTIER_D_POSE,
            "source": (
                "experiments/results/"
                "ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/"
                "MODAL_REMOTE_RESULT.json"
            ),
        },
    }

    if args.retain_dir is not None:
        args.retain_dir.mkdir(parents=True, exist_ok=True)
        retained = {}
        payloads = dict(bodies)
        payloads["residual_table"] = residual
        payloads["token_stream"] = tokens
        for name, blob in payloads.items():
            path = args.retain_dir / f"{name}.bin"
            path.write_bytes(blob)
            retained[name] = {
                "path": str(path),
                "bytes": len(blob),
                "sha256": sha256_bytes(blob),
            }
        result["retained_payloads"] = retained

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["census"], indent=2, sort_keys=True))
    print(json.dumps(result["carrier_wire"], indent=2, sort_keys=True))
    print(json.dumps(result["strict_bar"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
