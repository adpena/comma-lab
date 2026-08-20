"""ddm_hm1 -- exact counted-byte re-pricing of every fitted correction table.

The ladder prices a table as raw RCF1 packing: 6 bits per value plus a 4-byte magic and
one fp16 scale.  Two corrections are owed, and they push in opposite directions:

* UNDER-count -- the equal-mass margin bin edges are quantiles of THIS video's logit
  margins, so they are video-derived and must be counted with the table (rule 118).  A
  generic equal-width binning would be free but is not what was fitted.
* OVER-count -- the shipped archive brotli-compresses its model sections, and a large
  correction table is highly redundant, so raw 6-bit packing overstates its real cost.

This tool packs each retained table exactly as ``_decode_fixed_table`` would read it,
compresses it the way the archive compresses model sections, adds the counted bin edges,
and reports the true counted bytes.  Both corrections are applied so neither the pass nor
the fail can be blamed on convenient pricing.

Axis: ``[macOS-CPU advisory / scorer-free byte measurement]``.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.payload_retention import portable_retention_record, retain_candidates

NUM_CLASSES = 5
RCF1_TABLE_BITS = 6
RCF1_HEADER_BYTES = 6
# One fp16 threshold per interior bin edge.  The ladder's rung names carry their bin
# count; a rung with B equal-mass margin bins needs B-1 transmitted thresholds.
MARGIN_BINS_BY_RUNG = {
    "r0_no_table": 0,
    "r1_shipped_context": 0,
    "r2_margin4": 4,
    "r3_margin16": 16,
    "r4_bucket8_margin16": 16,
    "r5_bucket8_margin32": 32,
    "r6_prevclass_margin16": 16,
    "r7_prevclass_margin32": 32,
    "r8_second_choice": 8,
}


class RepriceError(RuntimeError):
    """Raised when a retained table cannot be priced."""


def pack_six_bit_codes(codes: np.ndarray, bits: int = RCF1_TABLE_BITS) -> bytes:
    """Bit-pack signed codes exactly the way ``runtime.bits.unpack_signed`` reads them.

    That unpacker is LITTLE-endian at the bit level -- it accumulates
    ``acc |= blob[index] << available`` and takes the low ``bits`` -- so value 0 occupies
    the LOW bits of byte 0, not the high ones.  It also refuses non-zero padding in the
    tail, which under little ordering is the HIGH bits of the final byte.  Packing
    MSB-first produces a blob the receiver decodes to different numbers, silently.
    """
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    if flat.size and (flat.min() < low or flat.max() > high):
        raise RepriceError(f"codes do not fit a signed {bits}-bit field")
    unsigned = (flat & ((1 << bits) - 1)).astype(np.uint8)
    stream = np.unpackbits(unsigned[:, None], axis=1, bitorder="little")[
        :, :bits
    ].reshape(-1)
    pad = (-stream.size) % 8
    if pad:
        stream = np.concatenate([stream, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(stream, bitorder="little").tobytes()


def price_table(path: Path, rung: str, retained: Path) -> dict[str, Any]:
    values = np.fromfile(path, dtype="<f4")
    if values.size % NUM_CLASSES:
        raise RepriceError(f"{path} is not a multiple of {NUM_CLASSES} values")
    table = values.reshape(-1, NUM_CLASSES)
    nonzero = np.abs(table[table != 0.0])
    if nonzero.size == 0:
        codes = np.zeros_like(table, dtype=np.int64)
        scale = np.float16(1.0)
    else:
        # Recover the scale the ladder actually used: every stored value is an integer
        # multiple of it, so the smallest nonzero magnitude is the step.
        scale = np.float16(float(nonzero.min()))
        codes = np.rint(table / np.float32(scale)).astype(np.int64)
        if np.abs(codes).max() > 32:
            raise RepriceError(f"{path} does not decode as 6-bit codes at scale {scale}")

    body = b"RCF1" + np.float16(scale).tobytes() + pack_six_bit_codes(codes)
    compressed = brotli.compress(body, quality=11, lgwin=24)
    # Retain BOTH candidates through the canonical helper, not only the one that wins
    # the min().  A length is not a payload, and the loser is exactly what a re-audit
    # needs to re-derive the choice -- the anchor's discarded ANS payload was later
    # measured at -2,120 B against the coder that had been kept.
    records = {
        name: portable_retention_record(record)
        for name, record in retain_candidates(
            retained,
            {f"table_{rung}.rcf1": body, f"table_{rung}.rcf1.br": compressed},
            suffix="",
        ).items()
    }
    raw_record = records[f"table_{rung}.rcf1"]
    compressed_record = records[f"table_{rung}.rcf1.br"]

    bins = MARGIN_BINS_BY_RUNG.get(rung, 0)
    edge_bytes = max(0, bins - 1) * 2  # fp16 thresholds, counted: they are video-derived
    counted = min(raw_record["bytes"], compressed_record["bytes"]) + edge_bytes
    return {
        "rung": rung,
        "cells": int(table.shape[0]),
        "table_scale": float(scale),
        "raw_rcf1": raw_record,
        "brotli_q11": compressed_record,
        "counted_margin_edge_bytes": edge_bytes,
        "counted_model_bytes": counted,
    }


def run(ladder_dir: Path) -> dict[str, Any]:
    ladder_path = ladder_dir / "ladder.json"
    if not ladder_path.is_file():
        raise RepriceError(f"missing ladder report: {ladder_path}")
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))

    retained = ladder_dir / "repriced"
    retained.mkdir(parents=True, exist_ok=True)
    priced: dict[str, dict[str, Any]] = {}
    for row in ladder["rows"]:
        rung = row["rung"]
        if row.get("free_table_oracle") or rung == "shipped_actual_table":
            continue
        payload = ladder_dir / f"table_{rung}.f32"
        if not payload.is_file():
            raise RepriceError(f"missing retained table payload: {payload}")
        priced[rung] = price_table(payload, rung, retained)

    rows: list[dict[str, Any]] = []
    baseline: float | None = None
    for row in ladder["rows"]:
        rung = row["rung"]
        if row.get("free_table_oracle") or rung == "shipped_actual_table":
            continue
        counted = priced[rung]["counted_model_bytes"]
        tokens = float(row["realized_token_bytes"])
        if baseline is None:
            baseline = tokens
        rows.append(
            {
                **priced[rung],
                "realized_token_bytes": tokens,
                "ladder_packed_model_bytes": row["packed_model_bytes"],
                "repriced_joint_bytes": tokens + counted,
                "tokens_saved_vs_no_table": baseline - tokens,
            }
        )

    for index in range(1, len(rows)):
        previous_row, row = rows[index - 1], rows[index]
        model_delta = row["counted_model_bytes"] - previous_row["counted_model_bytes"]
        token_delta = row["realized_token_bytes"] - previous_row["realized_token_bytes"]
        row["adjacent_model_byte_delta"] = model_delta
        row["adjacent_token_byte_delta"] = token_delta
        row["adjacent_slope"] = token_delta / model_delta if model_delta else None
        row["adjacent_pays"] = bool(model_delta and token_delta / model_delta < -1.0)

    best = min(rows, key=lambda row: row["repriced_joint_bytes"])
    report = {
        "schema": "ddm_hm1_repriced_ladder.v1",
        "axis": "[macOS-CPU advisory / scorer-free byte measurement]",
        "score_claim": False,
        "promotable": False,
        "frames": ladder["frames"],
        "note": (
            "counted_model_bytes = min(raw RCF1, brotli-q11 of the same body) + fp16 "
            "margin-bin edges, which are quantiles of this video and therefore counted"
        ),
        "rows": rows,
        "best_repriced_joint": {
            "rung": best["rung"],
            "repriced_joint_bytes": best["repriced_joint_bytes"],
            "counted_model_bytes": best["counted_model_bytes"],
            "realized_token_bytes": best["realized_token_bytes"],
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ladder_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(args.ladder_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    destination = args.report or (args.ladder_dir / "repriced_ladder.json")
    destination.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
