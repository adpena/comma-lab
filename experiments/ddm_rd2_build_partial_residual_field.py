"""ddm_rd2 Phase B -- materialise HG1's field at a PARTIAL residual, for scoring.

`ddm_bo2` measured exactly two points of the HG1 container: the full residual (exact by
construction, 460,408 B) and no residual at all (101,128 B, `dS_dist = +5.131079`).  This
builds the points in between, which nobody has ever materialised, so they can be scored
through the same instrument.

Method, and it is deliberately the receiver's own: take the shipped generated field, take
a value-ordered SUBSET of the shipped residual's 1,334,939 corrections, and apply that
subset with the shipped `apply_residual`.  Not a re-implementation -- the actual receiver
function, imported.  The output field is therefore exactly what a receiver decoding a
truncated-but-legal residual would reconstruct.

The subset payload is produced by `ddm_rd2_residual_byte_curve.py`, which measured its
real coded size with the real coders, so the (bytes, distortion) pair this feeds is
matched: both halves come from the same constructed object.

Run:
  .venv/bin/python experiments/ddm_rd2_build_partial_residual_field.py \
      --residual-payload <path from Phase A> --out <field.u8>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

N_PAIRS, HEIGHT, WIDTH = 600, 384, 512
TOTAL_POSITIONS = N_PAIRS * HEIGHT * WIDTH

AP = Path("/Volumes/APDataStore/pact")
GENERATED = AP / "ddm_hg1_heterogeneous_analytic_generator_gate/retained/generators/generated_tokens.u8"
GENERATED_SHA = "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"
# ddm_bo2 sec.1: dx2's own categorical token field, the exactness target.
DX2_FIELD_SHA = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-payload", type=Path, required=True,
                        help="uncompressed residual wire payload holding the chosen subset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expect-corrections", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_hg1_heterogeneous_analytic_generator_gate as hg1  # noqa: PLC0415

    # Custody first: a size match is not an identity.
    if not GENERATED.is_file():
        raise SystemExit(f"REFUSE: missing {GENERATED}")
    got = sha256_file(GENERATED)
    if got != GENERATED_SHA:
        raise SystemExit(f"REFUSE: generated field sha {got} != {GENERATED_SHA}")
    print(f"[rd2B] custody OK: generated field {GENERATED_SHA[:16]}...", flush=True)

    payload = args.residual_payload.read_bytes()
    field = np.fromfile(GENERATED, dtype=np.uint8).reshape(N_PAIRS, HEIGHT, WIDTH)

    # The SHIPPED receiver applies the corrections -- not a re-implementation. If the
    # subset were non-canonical or duplicated, this raises, which is the point.
    applied = hg1.apply_residual(payload, field)
    print(f"[rd2B] shipped receiver applied {applied:,} corrections", flush=True)
    if args.expect_corrections is not None and applied != args.expect_corrections:
        raise SystemExit(f"REFUSE: applied {applied:,}, expected {args.expect_corrections:,}")

    # ALWAYS KEEP THE PAYLOAD: the field is written before any verdict can raise.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    field.tofile(args.out)
    out_sha = sha256_file(args.out)
    out_bytes = args.out.stat().st_size
    print(f"[rd2B] wrote {args.out} ({out_bytes:,} B) sha {out_sha[:16]}...", flush=True)

    if out_bytes != TOTAL_POSITIONS:
        raise SystemExit(f"REFUSE: field is {out_bytes:,} B, expected {TOTAL_POSITIONS:,}")

    # Controls that make the field's IDENTITY checkable rather than asserted.
    generated = np.fromfile(GENERATED, dtype=np.uint8)
    changed = int((generated != field.reshape(-1)).sum())
    if changed != applied:
        raise SystemExit(
            f"REFUSE: {changed:,} positions differ from the generated field but "
            f"{applied:,} corrections were applied -- a correction rewrote an equal value"
        )
    if out_sha == GENERATED_SHA:
        raise SystemExit("REFUSE: output field is byte-identical to the generated field")
    if out_sha == DX2_FIELD_SHA:
        print("[rd2B] NOTE: partial field reached dx2 exactness (all damage corrected)", flush=True)

    manifest = {
        "schema": "ddm_rd2_partial_residual_field.v1",
        "arm": "ddm_rd2",
        "generated_field": {"path": str(GENERATED), "sha256": GENERATED_SHA},
        "residual_payload": {
            "path": str(args.residual_payload),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "corrections_applied": applied,
        "positions_changed_vs_generated": changed,
        "output_field": {"path": str(args.out), "bytes": out_bytes, "sha256": out_sha},
        "output_is_dx2_exact": out_sha == DX2_FIELD_SHA,
        "controls": {
            "generated_field_custody_verified": True,
            "applied_via_shipped_receiver_apply_residual": True,
            "changed_positions_equals_corrections": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"field_sha256": out_sha, "corrections": applied}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
