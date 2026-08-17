"""ddm_rc4 Stage 0 - independent byte census of the LIVE hv1 frontier archive.

Re-derives every section boundary from the shipped receiver's own parse constants
(runtime/residual_archive.py), never from an inherited table. Persists every
materialized payload (ALWAYS KEEP THE PAYLOAD, P0).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10/archive.zip"
)
STORE = Path("/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816")
RETAINED = STORE / "retained" / "census_sections"

# Constants lifted verbatim from the SHIPPED receiver parse.
RX1_MAGIC = b"RX1M"
RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")
FIXED_MAGIC = b"RCF1"
FIXED_STATES = 25
FIXED_BITS = 6
NUM_CLASSES = 5


def packed_length(count: int, bits: int) -> int:
    return (count * bits + 7) // 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def order0_entropy_bits(data: bytes) -> float:
    """Shannon order-0 (memoryless) entropy of the byte stream, in BITS."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum(c * math.log2(c / total) for c in counts.values())


def main() -> int:
    raw = ARCHIVE.read_bytes()
    archive_bytes = len(raw)
    archive_sha = sha256(raw)

    with zipfile.ZipFile(ARCHIVE) as zf:
        names = zf.namelist()
        payload = zf.read("p")
    payload_bytes = len(payload)
    zip_framing = archive_bytes - payload_bytes

    if not payload.startswith(RX1_MAGIC):
        raise SystemExit("hv1 payload is not RX1M - re-derive the parse")

    magic, version, codec, table_mode, reserved, hpac_n, semantic_n, carrier_n = (
        RX1_MODEL_HEADER.unpack_from(payload)
    )
    header_bytes = RX1_MODEL_HEADER.size
    off = header_bytes
    hpac_stream = payload[off : off + hpac_n]
    off += hpac_n
    semantic_stream = payload[off : off + semantic_n]
    off += semantic_n
    carrier_stream = payload[off : off + carrier_n]
    off += carrier_n
    model_end = off
    section = payload[model_end:]

    fixed_size = len(FIXED_MAGIC) + 2 + packed_length(FIXED_STATES * NUM_CLASSES, FIXED_BITS)
    compact_size = fixed_size - len(FIXED_MAGIC)
    residual_compact = section[:compact_size]
    token_stream = section[compact_size:]

    RETAINED.mkdir(parents=True, exist_ok=True)
    sections = [
        ("rx1_header", payload[:header_bytes], "RX1M header: codec/table-mode/3 section lengths"),
        ("hpac_stream", hpac_stream, "shipped HPAC/IHS1 probability-model stream"),
        ("semantic_stream", semantic_stream, "shipped semantic renderer weights (brotli)"),
        ("carrier_stream", carrier_stream, "shipped frame-0 pose carrier + selector (brotli)"),
        ("residual_table", residual_compact, "RCF1 fixed boundary residual table (compact)"),
        ("token_stream", token_stream, "RC64 arithmetic-coded per-pixel label tokens"),
    ]

    rows = []
    for name, blob, what in sections:
        path = RETAINED / f"{name}.bin"
        path.write_bytes(blob)
        h = order0_entropy_bits(blob)
        rows.append(
            {
                "section": name,
                "bytes": len(blob),
                "sha256": sha256(blob),
                "what": what,
                "retained_path": str(path),
                "share_of_archive": len(blob) / archive_bytes,
                "rate_term": 25.0 * len(blob) / 37_545_489,
                "order0_byte_entropy_bits": h,
                "order0_byte_entropy_bytes": h / 8.0,
                "order0_slack_bytes": len(blob) - h / 8.0,
                "bits_per_byte": (h / len(blob)) if blob else 0.0,
            }
        )

    total_sections = sum(r["bytes"] for r in rows)
    accounted = total_sections + zip_framing

    census = {
        "arm": "ddm_rc4",
        "stage": "0_census",
        "archive_path": str(ARCHIVE),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "zip_member_names": names,
        "zip_framing_bytes": zip_framing,
        "payload_member_bytes": payload_bytes,
        "rx1_header_fields": {
            "magic": magic.decode(),
            "version": version,
            "codec": codec,
            "codec_name": {1: "xz", 2: "brotli"}.get(codec, "unknown"),
            "table_mode": table_mode,
            "reserved": reserved,
            "hpac_bytes": hpac_n,
            "semantic_bytes": semantic_n,
            "carrier_bytes": carrier_n,
        },
        "sections": rows,
        "section_total_bytes": total_sections,
        "accounted_bytes": accounted,
        "unaccounted_bytes": archive_bytes - accounted,
        "score_per_byte": 25.0 / 37_545_489,
        "recomposed_rate_term": 25.0 * archive_bytes / 37_545_489,
        "token_geometry": {
            "frames": 600,
            "eval_h": 384,
            "eval_w": 512,
            "alphabet": NUM_CLASSES,
            "symbol_count": 600 * 384 * 512,
            "samples_per_symbol": 600 * 384 * 512 / NUM_CLASSES,
            "shipped_bits_per_symbol": 8.0 * len(token_stream) / (600 * 384 * 512),
        },
    }

    STORE.mkdir(parents=True, exist_ok=True)
    out = STORE / "CENSUS_hv1.json"
    out.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")

    w = sys.stdout.write
    w(f"archive {archive_bytes} B  sha {archive_sha[:16]}...\n")
    w(f"zip members {names}  framing {zip_framing} B\n\n")
    w(f"{'section':<18}{'bytes':>9}{'share':>9}{'rate_term':>13}{'bits/byte':>11}{'H0 slack B':>12}\n")
    for r in rows:
        w(
            f"{r['section']:<18}{r['bytes']:>9}{100*r['share_of_archive']:>8.3f}%"
            f"{r['rate_term']:>13.7f}{r['bits_per_byte']:>11.4f}{r['order0_slack_bytes']:>12.1f}\n"
        )
    w(f"{'zip_framing':<18}{zip_framing:>9}{100*zip_framing/archive_bytes:>8.3f}%\n")
    w(f"{'TOTAL':<18}{accounted:>9}   unaccounted {archive_bytes - accounted}\n")
    w(f"\nrecomposed rate term {census['recomposed_rate_term']!r}\n")
    tg = census["token_geometry"]
    w(f"token geometry: {tg['symbol_count']} symbols, alphabet {tg['alphabet']}, "
      f"{tg['samples_per_symbol']:.3e} samples/symbol, "
      f"{tg['shipped_bits_per_symbol']:.6f} shipped bits/symbol\n")
    w(f"\ncensus -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
