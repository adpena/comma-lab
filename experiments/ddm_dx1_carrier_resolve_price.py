#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""dx1 -- exact carrier re-code price of a pose re-solve, on the live body.

pq6 §B item 2 records that the rung-4 reopening is blocked on two doors.  Door 1 is the
pose residual.  Door 2 is rc4's byte budget: the re-coded carrier section must grow by
less than 4,873 B.  pq6 states the carrier's re-coding cost across 600 pairs is
UNMEASURED.

It is measurable exactly, and on the correct body, because jg5's own build already
performed a full carrier re-solve: ``body_before_resolve`` (180,580 B) -> ``best``
(180,625 B) over 454 admitted pairs.  Both archives are retained.  This tool prices the
CAP1 carrier section of each through the receiver's own Rice encoder, with the same
bit-exact positive control the race uses, and reports the exact section delta and the
per-pair rate.

This does not price a token DROP.  It prices the carrier RE-SOLVE, which is the leg
rc4's door 2 budgets, on the vehicle that ships.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_dx1_dxi_recode_race import (
    UNCOMPRESSED_BYTES,
    forward_ar1,
    load_shipped,
)


def price(archive: Path, runtime: Path) -> dict:
    (carrier_repack, _cap1, predictor, carrier_blob, info, model, codes) = load_shipped(
        archive, runtime
    )
    residuals = forward_ar1(codes, model, predictor)
    u = carrier_repack._zigzag(residuals)
    ks, payload, bits = carrier_repack._rice_encode(u, 1)
    control_ok = (
        int(bits) == int(info["rice_payload_bits"])
        and np.array_equal(
            ks.reshape(-1).astype(np.uint8),
            np.asarray(info["rice_ks"], dtype=np.uint8),
        )
        and bytes(payload)
        == carrier_blob[len(carrier_blob) - int(info["rice_payload_bytes"]) :]
    )
    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "carrier_blob_bytes": len(carrier_blob),
        "fixed_prefix_bytes": len(carrier_blob) - int(info["rice_payload_bytes"]),
        "rice_payload_bytes": int(info["rice_payload_bytes"]),
        "rice_payload_bits": int(info["rice_payload_bits"]),
        "rice_ks": np.asarray(info["rice_ks"], dtype=np.uint8).astype(int).tolist(),
        "control_bit_exact": bool(control_ok),
        "codes_sha256": hashlib.sha256(
            np.ascontiguousarray(codes.astype(np.int32)).tobytes()
        ).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="body before the carrier re-solve")
    ap.add_argument("--after", required=True, help="body after the carrier re-solve")
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--pairs-resolved", type=int, required=True)
    ap.add_argument("--budget-bytes", type=float, default=4873.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runtime = Path(args.runtime)
    before = price(Path(args.before), runtime)
    after = price(Path(args.after), runtime)
    if not (before["control_bit_exact"] and after["control_bit_exact"]):
        print("CONTROL FAILED on one of the bodies -- price is not admissible.")
        return 3

    d_carrier = after["carrier_blob_bytes"] - before["carrier_blob_bytes"]
    d_rice = after["rice_payload_bytes"] - before["rice_payload_bytes"]
    d_archive = after["archive_bytes"] - before["archive_bytes"]
    per_pair = d_carrier / args.pairs_resolved
    at_600 = per_pair * 600.0

    codes_moved = before["codes_sha256"] != after["codes_sha256"]
    out = {
        "schema": "ddm_dx1_carrier_resolve_price.v1",
        "axis": "[exact local byte arithmetic, no scorer]",
        "before": before,
        "after": after,
        "codes_actually_changed": codes_moved,
        "pairs_resolved": args.pairs_resolved,
        "delta_carrier_blob_bytes": d_carrier,
        "delta_rice_payload_bytes": d_rice,
        "delta_archive_bytes": d_archive,
        "carrier_bytes_per_resolved_pair": per_pair,
        "projected_carrier_bytes_at_600_pairs": at_600,
        "rc4_door2_budget_bytes": args.budget_bytes,
        "door2_margin_x": args.budget_bytes / at_600 if at_600 > 0 else None,
        "door2_clears": bool(at_600 < args.budget_bytes),
        "delta_S_of_measured_carrier_growth": 25.0 * d_carrier / UNCOMPRESSED_BYTES,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("before", "after")}, indent=2))
    print(f"\nbefore: archive {before['archive_bytes']} B  carrier {before['carrier_blob_bytes']} B"
          f"  rice {before['rice_payload_bytes']} B  control={before['control_bit_exact']}")
    print(f"after : archive {after['archive_bytes']} B  carrier {after['carrier_blob_bytes']} B"
          f"  rice {after['rice_payload_bytes']} B  control={after['control_bit_exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
