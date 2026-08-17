#!/usr/bin/env python
"""Re-point the coder-axis closure at ANY archive, instead of citing an old base.

The bug this extincts: the "coder axis is CLOSED, all four sections vs their own
memoryless bound" table was measured once, on the PR130 base (191,052 B), and was
then cited on cp135, MC36, e480b and hv1 -- four bases downstream -- without ever
being re-measured.  That is the cross-regime constant-transfer genus.  This tool
makes re-measuring a one-command operation so the citation never has to be
inherited again.

For each RX1 section of an archive it reports, MEASURED on that archive:

  * shipped coded bytes, raw bytes, and samples-per-symbol (the governing
    variable: a conditional model can only amortize on a dense stream)
  * the order-0 byte-entropy bound and the order-1 conditional ORACLE
  * the order-0 and order-1 ADAPTIVE code lengths -- the exact sequential
    Krichevsky-Trofimov mixture cost, which pays its own learning cost and
    transmits no table, so it is what a real adaptive coder achieves
  * a real coder race (brotli-q11, LZMA2, LZMA1) on retained payloads

The order-1 ORACLE minus ADAPTIVE gap is the density trap: on a sparse section
the oracle shows a large apparent prize that no coder can collect.  Report both
or the reader will chase the mirage.

ALWAYS KEEP THE PAYLOAD: every decompressed section and every re-coded candidate
is written under --outdir with its sha256 and byte count.

Usage:
    .venv/bin/python tools/audit_archive_coder_axis.py \\
        --archive /path/to/archive.zip --outdir /Volumes/APDataStore/pact/<arm>/retained
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import gammaln

try:
    import brotli
except ImportError as error:  # pragma: no cover - environment guard
    raise SystemExit("this audit requires the brotli module") from error

RX1_MAGIC = b"RX1M"
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_CODEC_XZ = 1
RX1_CODEC_BROTLI = 2
BYTE_ALPHABET = 256
LN2 = float(np.log(2.0))
RESIDUAL_TABLE_BYTES = 96


class CoderAxisAuditError(RuntimeError):
    """The archive is not in a layout this audit can parse honestly."""


@dataclass
class SectionResult:
    section: str
    shipped_coded_bytes: int
    raw_bytes: int
    payloads: list[dict] = field(default_factory=list)
    race: list[dict] = field(default_factory=list)


def sequential_kt_bytes(counts: np.ndarray, alphabet: int, alpha: float = 0.5) -> float:
    """Exact code length of the sequential Dirichlet(alpha) mixture, in bytes.

    This is what an adaptive coder actually achieves: encoder and decoder adapt
    identically from the same history, so no table is transmitted and the cost
    of learning is already inside the number.
    """
    counts = np.atleast_2d(counts).astype(np.float64)
    totals = counts.sum(axis=1)
    used = totals > 0
    if not used.any():
        return 0.0
    rows = counts[used]
    term = gammaln(totals[used] + alphabet * alpha) - gammaln(alphabet * alpha)
    term -= (gammaln(rows + alpha) - gammaln(alpha)).sum(axis=1)
    return float(term.sum() / LN2 / 8.0)


def oracle_bytes(counts: np.ndarray) -> float:
    """Empirical conditional entropy: a LOWER BOUND that assumes a free table."""
    counts = np.atleast_2d(counts).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    probabilities = np.where(totals > 0, counts / safe, 0.0)
    # Build the log array by explicit assignment: np.log2(..., where=mask) leaves
    # the masked entries UNINITIALISED, which is a trap even when a later
    # np.where discards them.
    positive = probabilities > 0
    logs = np.zeros_like(probabilities)
    logs[positive] = np.log2(probabilities[positive])
    return float(-(counts * logs).sum() / 8.0)


def byte_counts(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    values = np.frombuffer(blob, dtype=np.uint8).astype(np.int64)
    order0 = np.bincount(values, minlength=BYTE_ALPHABET)
    if values.size < 2:
        return order0, np.zeros((BYTE_ALPHABET, BYTE_ALPHABET), dtype=np.int64)
    pairs = values[:-1] * BYTE_ALPHABET + values[1:]
    order1 = np.bincount(pairs, minlength=BYTE_ALPHABET**2).reshape(
        BYTE_ALPHABET, BYTE_ALPHABET
    )
    return order0, order1


def persist(outdir: Path, name: str, payload: bytes) -> dict:
    path = outdir / f"{name}.bin"
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def race_coders(outdir: Path, name: str, raw: bytes) -> list[dict]:
    """Real coders on the real payload; every candidate payload is retained."""
    candidates = {
        "brotli_q11_lgwin24": lambda data: brotli.compress(data, quality=11, lgwin=24),
        "lzma2_raw_xtreme_pb0_lc0": lambda data: lzma.compress(
            data,
            format=lzma.FORMAT_RAW,
            filters=[
                {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "pb": 0, "lc": 0}
            ],
        ),
        "lzma1_raw_xtreme_pb0_lc0": lambda data: lzma.compress(
            data,
            format=lzma.FORMAT_RAW,
            filters=[
                {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "pb": 0, "lc": 0}
            ],
        ),
    }
    rows = []
    for label, encode in candidates.items():
        payload = encode(raw)
        rows.append({"coder": label, **persist(outdir, f"{name}__{label}", payload)})
    return rows


def parse_rx1(archive_path: Path) -> tuple[dict, dict[str, bytes], bytes, bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != ["p"]:
            raise CoderAxisAuditError(f"expected exactly member 'p', found {names}")
        outer = archive.read("p")
    if len(outer) < RX1_HEADER.size or not outer.startswith(RX1_MAGIC):
        raise CoderAxisAuditError("archive payload is not an RX1 container")
    magic, version, codec, table_mode, reserved, hpac_n, semantic_n, carrier_n = (
        RX1_HEADER.unpack_from(outer)
    )
    if magic != RX1_MAGIC or version != 1 or codec not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI):
        raise CoderAxisAuditError("unsupported RX1 model header")
    offset = RX1_HEADER.size
    shipped: dict[str, bytes] = {}
    for name, length in (("hpac", hpac_n), ("semantic", semantic_n), ("carrier", carrier_n)):
        if length <= 0 or offset + length > len(outer):
            raise CoderAxisAuditError(f"RX1 section {name} is truncated")
        shipped[name] = outer[offset : offset + length]
        offset += length
    tail = outer[offset:]
    if len(tail) <= RESIDUAL_TABLE_BYTES:
        raise CoderAxisAuditError("RX1 residual/token tail is truncated")
    header = {
        "payload_member_bytes": len(outer),
        "zip_framing_bytes": archive_path.stat().st_size - len(outer),
        "rx1_header_bytes": RX1_HEADER.size,
        "rx1_codec": int(codec),
        "rx1_table_mode": int(table_mode),
        "rx1_reserved": int(reserved),
    }
    return header, shipped, tail[:RESIDUAL_TABLE_BYTES], tail[RESIDUAL_TABLE_BYTES:]


def decompress(name: str, blob: bytes, codec: int) -> bytes:
    if name == "hpac" and codec == RX1_CODEC_XZ:
        return lzma.decompress(blob, format=lzma.FORMAT_XZ)
    return brotli.decompress(blob)


def audit(archive_path: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    header, shipped, residual_table, token_stream = parse_rx1(archive_path)

    sections: list[dict] = []
    for name in ("hpac", "semantic", "carrier"):
        raw = decompress(name, shipped[name], header["rx1_codec"])
        order0, order1 = byte_counts(raw)
        used_contexts = int((order1.sum(axis=1) > 0).sum())
        coded = len(shipped[name])
        row = {
            "section": name,
            "shipped_coded_bytes": coded,
            "raw_bytes": len(raw),
            "alphabet": BYTE_ALPHABET,
            "samples_per_symbol": len(raw) / BYTE_ALPHABET,
            "order0_bound_bytes": oracle_bytes(order0),
            "order0_adaptive_bytes": sequential_kt_bytes(order0, BYTE_ALPHABET),
            "order1_contexts_used": used_contexts,
            "order1_oracle_bytes": oracle_bytes(order1),
            "order1_adaptive_bytes": sequential_kt_bytes(order1, BYTE_ALPHABET),
            "raw_payload": persist(outdir, f"raw_{name}", raw),
            "shipped_payload": persist(outdir, f"shipped_{name}", shipped[name]),
            "race": race_coders(outdir, name, raw),
        }
        row["shipped_minus_order0_bound"] = coded - row["order0_bound_bytes"]
        row["order1_density_trap_bytes"] = (
            row["order1_adaptive_bytes"] - row["order1_oracle_bytes"]
        )
        best = min(row["race"], key=lambda candidate: candidate["bytes"])
        row["best_race_coder"] = best["coder"]
        row["best_race_minus_shipped"] = best["bytes"] - coded
        sections.append(row)

    token_row = {
        "section": "token_stream",
        "shipped_coded_bytes": len(token_stream),
        "note": "already conditionally coded; a generic recode here tests only the wire",
        "shipped_payload": persist(outdir, "shipped_token_stream", token_stream),
        "race": race_coders(outdir, "token_stream", token_stream),
    }
    best_token = min(token_row["race"], key=lambda candidate: candidate["bytes"])
    token_row["best_race_coder"] = best_token["coder"]
    token_row["best_race_minus_shipped"] = best_token["bytes"] - len(token_stream)
    sections.append(token_row)
    sections.append(
        {
            "section": "residual_table",
            "shipped_coded_bytes": len(residual_table),
            "shipped_payload": persist(outdir, "shipped_residual_table", residual_table),
        }
    )

    best_total = sum(
        min(row.get("best_race_minus_shipped", 0), 0) for row in sections
    )
    return {
        "schema": "archive_coder_axis_audit.v1",
        "verdict": "coder_axis_open" if best_total < 0 else "coder_axis_closed",
        "best_available_saving_bytes": -best_total,
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": archive_path.stat().st_size,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            **header,
        },
        "sections": sections,
    }


def render(report: dict) -> str:
    lines = [
        f"archive {report['archive']['sha256'][:12]}… {report['archive']['bytes']:,} B",
        "",
        f"{'section':14s} {'shipped':>10s} {'raw':>9s} {'samp/sym':>9s} "
        f"{'H0 bound':>10s} {'H1 oracle':>10s} {'H1 adapt':>10s} {'best race':>10s}",
    ]
    for row in report["sections"]:
        if "raw_bytes" not in row:
            race = (
                f"{row['best_race_minus_shipped']:+10,d}"
                if "best_race_minus_shipped" in row
                else f"{'not raced':>10s}"
            )
            lines.append(
                f"{row['section']:14s} {row['shipped_coded_bytes']:10,d} "
                # raw(9) samp(9) H0(10) H1oracle(10) H1adapt(10) + 5 separators
                + " " * 53
                + race
            )
            continue
        lines.append(
            f"{row['section']:14s} {row['shipped_coded_bytes']:10,d} {row['raw_bytes']:9,d} "
            f"{row['samples_per_symbol']:9,.1f} {row['order0_bound_bytes']:10,.0f} "
            f"{row['order1_oracle_bytes']:10,.0f} {row['order1_adaptive_bytes']:10,.0f} "
            f"{row['best_race_minus_shipped']:+10,d}"
        )
    lines += [
        "",
        "H1 oracle assumes a free 65,536-cell table and is NOT achievable; H1 adapt pays",
        "its own learning cost and is. A large oracle-minus-adapt gap means the section is",
        "too sparse for conditional modelling -- that gap is the trap, not a prize.",
        "",
        f"VERDICT: {report['verdict']} "
        f"(best available saving {report['best_available_saving_bytes']:,} B)",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="durable directory for retained payloads (use the SSD tier, never /tmp)",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    arguments = parser.parse_args()

    report = audit(arguments.archive.resolve(), arguments.outdir.resolve())
    destination = arguments.outdir.resolve() / "coder_axis_audit.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True) if arguments.json else render(report))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
