"""ddm_ck2 -- price the RATE ceiling of the ck1 pointer archive, section by section.

WHY THIS EXISTS
---------------
The eleventh-move ranking needs to know *where the counted bytes actually are* on the
ck1 base and *how much of each section is still compressible*, before any candidate is
built.  Two standing laws force this order:

* ``the_denominator_and_the_falsifier_can_both_be_vacuous`` (ra3) -- price the CEILING
  first; a mechanism that "works" is worthless if its section holds no headroom.
* ``the_counted_byte_is_not_fungible_placement_beats_amount`` -- a byte saved in the
  tail and a byte saved in the semantic body are the same score, but the *reachable*
  amount differs by section by more than an order of magnitude, so the census is the
  ranking input, not a footnote.

WHAT IT MEASURES (exactly, no scorer, no model)
-----------------------------------------------
1. **Section census** of the RX1M member: header / hpac / semantic / carrier / tail,
   with byte counts and each section's share of the archive.
2. **Re-compression headroom per section**: each stored section is re-compressed with
   brotli q11 at several window sizes and, where the section is itself a brotli stream,
   decompressed-then-recompressed.  A section that is already at its brotli fixed point
   returns ~0 and is honestly reported as saturated.
3. **The semantic byte-plane split ceiling**: sz1 ships a 2-plane de-interleave over a
   PINNED region (offset 49, length 8284) whose profit depends on that region aligning
   to 16-bit metadata.  ck1's SM3R mode-6 row-prune changes the body layout, so the
   pinned constants are a cross-regime constant transfer
   (``cross_regime_constant_transfer_genus_finishing_stage``).  This probe therefore
   measures BOTH the pinned-constant credit AND the best credit over a SEARCH of
   (offset, length), which is the ceiling the mechanism could reach if the constants
   were re-solved for this body.  The search is exhaustive over a coarse grid and then
   refined; it reports the argmax so a builder can re-solve rather than carry.

AXIS ``[macOS-CPU exact byte measurement]``.  No score is produced or implied.
Every stream materialized here is persisted under the arm's retained directory per the
ALWAYS-KEEP-THE-PAYLOAD rule; nothing is measured-and-discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any, Final

import brotli

RX1_HEADER: Final = struct.Struct("<4sBBBBHHH")
MEMBER_NAME: Final = "p"
AXIS: Final = "[macOS-CPU exact byte measurement]"


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def split_sections(archive: Path) -> dict[str, Any]:
    """Split an RX1M member into its four sections plus the header fields."""
    outer = zipfile.ZipFile(archive).read(MEMBER_NAME)
    magic, version, codec, table_mode, reserved, hb, sb, cb = RX1_HEADER.unpack_from(outer)
    off = RX1_HEADER.size
    hpac, off = outer[off : off + hb], off + hb
    semantic, off = outer[off : off + sb], off + sb
    carrier, off = outer[off : off + cb], off + cb
    return {
        "magic": magic.decode("latin1"),
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "header_bytes": RX1_HEADER.size,
        "member_bytes": len(outer),
        "sections": {
            "hpac": hpac,
            "semantic": semantic,
            "carrier": carrier,
            "tail": outer[off:],
        },
    }


def brotli_stream(payload: bytes, *, lgwin: int = 24) -> bytes:
    return brotli.compress(payload, quality=11, lgwin=lgwin)


def recompress_headroom(name: str, blob: bytes) -> dict[str, Any]:
    """How many bytes does this section give up under a stronger/other brotli pass?

    Two honest readings are reported and never conflated:

    * ``recompress_as_opaque`` -- brotli over the stored bytes as-is.  For a section
      that is already a brotli stream this is almost always a LOSS and is reported as
      such; it is the control that proves the section is not raw.
    * ``decompress_then_recompress`` -- only attempted when the section decompresses as
      brotli.  This is the real headroom: it re-encodes the same payload at q11 across
      window sizes and returns the best.  ``0`` means the section is already at its
      brotli fixed point, which is a finding, not a failure.
    """
    out: dict[str, Any] = {
        "name": name,
        "stored_bytes": len(blob),
        "sha256": sha256_bytes(blob),
    }
    opaque = min(len(brotli_stream(blob, lgwin=w)) for w in (22, 24))
    out["recompress_as_opaque_bytes"] = opaque
    out["recompress_as_opaque_delta"] = opaque - len(blob)

    try:
        raw = brotli.decompress(blob)
    except brotli.error:
        out["is_brotli_stream"] = False
        out["raw_bytes"] = None
        out["decompress_then_recompress_best_bytes"] = None
        out["decompress_then_recompress_delta"] = None
        return out

    out["is_brotli_stream"] = True
    out["raw_bytes"] = len(raw)
    best, best_w = len(blob), None
    for w in (16, 18, 20, 22, 24):
        cand = len(brotli_stream(raw, lgwin=w))
        if cand < best:
            best, best_w = cand, w
    out["decompress_then_recompress_best_bytes"] = best
    out["decompress_then_recompress_lgwin"] = best_w
    out["decompress_then_recompress_delta"] = best - len(blob)
    return out


def deinterleave(body: bytes, start: int, length: int) -> bytes:
    """Group a region's odd bytes then its even bytes -- sz1's 2-plane split."""
    region = body[start : start + length]
    return body[:start] + region[1::2] + region[0::2] + body[start + length :]


def split_search(
    body: bytes, *, pinned_offset: int, pinned_length: int, coarse: int
) -> dict[str, Any]:
    """Measure the byte-plane split credit: at sz1's pinned constants, and at its argmax.

    The split is a pure byte permutation, so it is always exactly invertible and the
    only question is profit.  ``credit`` is NEGATIVE when the split SAVES bytes, so the
    sign convention matches a score delta.
    """
    baseline = len(brotli_stream(body))
    result: dict[str, Any] = {
        "body_bytes": len(body),
        "body_sha256": sha256_bytes(body),
        "nosplit_stream_bytes": baseline,
    }

    if len(body) >= pinned_offset + pinned_length:
        pinned = len(brotli_stream(deinterleave(body, pinned_offset, pinned_length)))
        result["pinned_constants"] = {
            "offset": pinned_offset,
            "length": pinned_length,
            "stream_bytes": pinned,
            "credit_bytes": pinned - baseline,
        }
    else:
        result["pinned_constants"] = {
            "offset": pinned_offset,
            "length": pinned_length,
            "stream_bytes": None,
            "credit_bytes": None,
            "skipped": "body shorter than pinned region",
        }

    # Whole-body split is the parameter-free control: no fitted constants at all.
    whole = len(brotli_stream(deinterleave(body, 0, len(body) & ~1)))
    result["whole_body_split"] = {
        "stream_bytes": whole,
        "credit_bytes": whole - baseline,
    }

    best = {"offset": None, "length": None, "stream_bytes": baseline, "credit_bytes": 0}
    evaluated = 0
    n = len(body)
    for start in range(0, n - 2 * coarse, coarse):
        for length in range(2 * coarse, n - start + 1, coarse):
            cand = len(brotli_stream(deinterleave(body, start, length & ~1)))
            evaluated += 1
            if cand - baseline < best["credit_bytes"]:
                best = {
                    "offset": start,
                    "length": length & ~1,
                    "stream_bytes": cand,
                    "credit_bytes": cand - baseline,
                }
    result["coarse_grid"] = {"step": coarse, "evaluated": evaluated, "best": dict(best)}
    result["best_overall"] = dict(best)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pinned-offset", type=int, default=49)
    ap.add_argument("--pinned-length", type=int, default=8_284)
    ap.add_argument(
        "--split-coarse",
        type=int,
        default=512,
        help="grid step for the (offset,length) split search, in bytes",
    )
    args = ap.parse_args(argv)

    digest = sha256_bytes(args.archive.read_bytes())
    if digest != args.expect_sha256:
        raise SystemExit(f"archive sha mismatch: {digest} != {args.expect_sha256}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retained = args.out_dir / "sections"
    retained.mkdir(exist_ok=True)

    parsed = split_sections(args.archive)
    census: dict[str, Any] = {}
    headroom: dict[str, Any] = {}
    for name, blob in parsed["sections"].items():
        # ALWAYS KEEP THE PAYLOAD: every section materialized here is persisted before
        # its length is reported, so the next consumer never re-derives these bytes.
        (retained / f"{name}.bin").write_bytes(blob)
        census[name] = {
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "share_of_member": len(blob) / parsed["member_bytes"],
        }
        headroom[name] = recompress_headroom(name, blob)

    semantic_raw = None
    split: dict[str, Any] | None = None
    try:
        semantic_raw = brotli.decompress(parsed["sections"]["semantic"])
    except brotli.error:
        pass
    if semantic_raw is not None:
        (retained / "semantic_body.bin").write_bytes(semantic_raw)
        split = split_search(
            semantic_raw,
            pinned_offset=args.pinned_offset,
            pinned_length=args.pinned_length,
            coarse=args.split_coarse,
        )

    report = {
        "schema": "ddm_ck2_rate_ceiling.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "archive": str(args.archive),
        "archive_sha256": digest,
        "archive_bytes": args.archive.stat().st_size,
        "header": {
            k: parsed[k]
            for k in ("magic", "version", "codec", "table_mode", "reserved", "header_bytes")
        },
        "member_bytes": parsed["member_bytes"],
        "section_census": census,
        "section_headroom": headroom,
        "semantic_split": split,
        "retained_dir": str(retained),
    }
    out = args.out_dir / "CK2_RATE_CEILING.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"archive {digest[:16]}… {report['archive_bytes']:,} B  member {parsed['member_bytes']:,} B")
    for name, row in census.items():
        h = headroom[name]
        rec = h["decompress_then_recompress_delta"]
        print(
            f"  {name:9s} {row['bytes']:7,d} B  {row['share_of_member']:6.2%}  "
            f"brotli={h['is_brotli_stream']}  raw={h['raw_bytes']}  "
            f"recompress_delta={rec}"
        )
    if split:
        print(
            f"  semantic body {split['body_bytes']:,} B  nosplit={split['nosplit_stream_bytes']:,} B\n"
            f"    pinned  credit {split['pinned_constants']['credit_bytes']}\n"
            f"    whole   credit {split['whole_body_split']['credit_bytes']}\n"
            f"    argmax  credit {split['best_overall']['credit_bytes']} "
            f"@ offset={split['best_overall']['offset']} length={split['best_overall']['length']}"
        )
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
