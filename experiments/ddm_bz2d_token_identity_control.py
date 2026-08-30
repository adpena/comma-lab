#!/usr/bin/env python3
"""ddm_bz2d token-identity control.

THE CLAIM THIS CONTROL MAKES MEASURABLE
---------------------------------------
The measurement vehicle is bz2's TOKEN FIELD re-encoded through lb1's shipped HPAC
model, spliced into lb1's own body (lb1's semantic renderer + lb1's pose carrier,
both byte-identical by construction -- proven at offsets 13,529 / 44,385 inside the
live pointer archive).  Because the renderer is deterministic, IF the tokens the
shipping runtime decodes are byte-identical to bz2's own retained parse-back token
field, THEN the rendered frames are byte-identical to bz2's, and therefore the
d_seg / d_pose this advisory measures ARE bz2's distortion -- measured through the
real inflate/evaluate chain rather than transferred from a sibling.

Without this control the claim is an argument.  With it, it is a measurement.

WHAT IT COMPARES
----------------
  A: <inflated>/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8
     -- written by runtime/f26_inflate.py:584 during THIS advisory's decode.
  B: <bz2 retained>/archive_parseback_tokens.u8
     -- bz2's own parse-back token dump, 117,964,800 B (600 x 384 x 512 uint8).

Byte-identity is checked by size, sha256, and (on mismatch) a first-differing-byte
report plus a per-pair mismatch census, so a NEGATIVE result is diagnostic and not
merely a refusal.

Emits a typed JSON receipt.  Exit 0 = IDENTICAL, 3 = MISMATCH, 2 = INPUT MISSING.
No score claim is made or implied here; this is a custody control only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 1 << 22  # 4 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def first_difference(a: Path, b: Path) -> int | None:
    """Byte offset of the first difference, or None if the common prefix matches."""
    offset = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ca = fa.read(CHUNK)
            cb = fb.read(CHUNK)
            if not ca or not cb:
                return None
            if ca != cb:
                for i, (x, y) in enumerate(zip(ca, cb)):
                    if x != y:
                        return offset + i
                return offset + min(len(ca), len(cb))
            offset += len(ca)


def pair_census(a: Path, b: Path, pair_bytes: int) -> dict:
    """Per-pair mismatch census: how many of the 600 pairs differ, and by how much."""
    differing = []
    total_differing_bytes = 0
    idx = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ca = fa.read(pair_bytes)
            cb = fb.read(pair_bytes)
            if not ca or not cb:
                break
            if ca != cb:
                n = sum(1 for x, y in zip(ca, cb) if x != y)
                differing.append({"pair": idx, "differing_bytes": n})
                total_differing_bytes += n
            idx += 1
    return {
        "pairs_examined": idx,
        "pairs_differing": len(differing),
        "total_differing_bytes": total_differing_bytes,
        "first_20_differing_pairs": differing[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoded", type=Path, required=True,
                    help="tokens_cpu_stage_complete.u8 written by this advisory's decode")
    ap.add_argument("--reference", type=Path, required=True,
                    help="bz2's retained archive_parseback_tokens.u8")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=600)
    args = ap.parse_args()

    receipt: dict = {
        "control": "ddm_bz2d_token_identity_v1",
        "decoded_path": str(args.decoded),
        "reference_path": str(args.reference),
        "claim_under_test": (
            "the measurement vehicle's decoded token field is byte-identical to bz2's "
            "own retained parse-back token field, hence its render (and therefore its "
            "d_seg/d_pose) IS bz2's"
        ),
        "score_claim": False,
        "promotable": False,
    }

    for label, path in (("decoded", args.decoded), ("reference", args.reference)):
        if not path.is_file():
            receipt["verdict"] = "INPUT_MISSING"
            receipt["missing"] = label
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"INPUT_MISSING: {label} at {path}", file=sys.stderr)
            return 2

    # A control that CANNOT fail is theater (the vacuous-control class).  Comparing a
    # file against itself -- by identical path, by symlink, or by hardlink -- would
    # return IDENTICAL for free and prove nothing about the decode.  Refuse it.
    decoded_stat, reference_stat = args.decoded.stat(), args.reference.stat()
    same_inode = (
        decoded_stat.st_dev == reference_stat.st_dev
        and decoded_stat.st_ino == reference_stat.st_ino
    )
    if args.decoded.resolve() == args.reference.resolve() or same_inode:
        receipt["verdict"] = "VACUOUS_CONTROL_REFUSED"
        receipt["reason"] = (
            "decoded and reference resolve to the same file (path, symlink, or hardlink); "
            "a self-comparison cannot falsify the identity claim"
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print("VACUOUS_CONTROL_REFUSED: decoded and reference are the same file",
              file=sys.stderr)
        return 2

    a_bytes = args.decoded.stat().st_size
    b_bytes = args.reference.stat().st_size
    receipt["decoded_bytes"] = a_bytes
    receipt["reference_bytes"] = b_bytes

    a_sha = sha256_of(args.decoded)
    b_sha = sha256_of(args.reference)
    receipt["decoded_sha256"] = a_sha
    receipt["reference_sha256"] = b_sha

    if a_bytes == b_bytes and a_sha == b_sha:
        receipt["verdict"] = "IDENTICAL"
        receipt["consequence"] = (
            "the rendered frames are bz2's frames; this advisory's d_seg/d_pose are "
            "bz2's distortion, MEASURED through the real inflate/evaluate chain"
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"IDENTICAL: {a_bytes} B, sha {a_sha[:16]}")
        return 0

    receipt["verdict"] = "MISMATCH"
    receipt["size_equal"] = a_bytes == b_bytes
    receipt["first_difference_offset"] = first_difference(args.decoded, args.reference)
    if a_bytes == b_bytes and args.pairs > 0 and a_bytes % args.pairs == 0:
        receipt["pair_census"] = pair_census(
            args.decoded, args.reference, a_bytes // args.pairs
        )
    receipt["consequence"] = (
        "the render is NOT provably bz2's; the measured distortion belongs to the "
        "measurement vehicle alone and may NOT be attributed to bz2"
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"MISMATCH: decoded {a_bytes} B sha {a_sha[:16]} vs "
          f"reference {b_bytes} B sha {b_sha[:16]}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
