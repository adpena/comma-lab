#!/usr/bin/env python3
"""Rebuild the edited token planes from a ``ddm_jg3`` per-pair checkpoint.

**This exists to make an ALWAYS-KEEP-THE-PAYLOAD claim checkable rather than
asserted.**  ``ddm_jg3_joint_solve`` writes its npz mirror every ``--payload-every``
pairs, but it writes the JSONL every pair -- and the JSONL's ``accepted`` field is
the complete sparse edit list ``(y, x, value)`` against a sha-pinned base token
field.  So the edited planes are exactly reconstructible from the checkpoint alone,
at any moment, including mid-run.  This tool does that reconstruction and emits the
``.npz`` that ``experiments/ddm_jg2_tail_reencode.py --stage encode`` consumes, so
the rate leg can be MEASURED on a partial run without waiting for it to finish.
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

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_jg3_joint_solve as jg3  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N pairs of the run")
    args = parser.parse_args(argv)

    tokens = jg1.load_tokens()
    rows = [
        json.loads(line)
        for line in Path(args.checkpoint).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    planes: dict[str, np.ndarray] = {}
    changed = 0
    for row in rows:
        if not row["accepted"]:
            continue
        pair = int(row["pair"])
        plane = np.array(tokens[pair], dtype=np.uint8)
        for y, x, value in row["accepted"]:
            plane[int(y), int(x)] = int(value)
        # the checkpoint's own token count must match the reconstruction, or the
        # sparse record and the measured result have drifted apart
        actual = int((plane != tokens[pair]).sum())
        if actual != row["tokens_changed"]:
            raise SystemExit(
                f"pair {pair}: reconstructed {actual} changed tokens but the "
                f"checkpoint recorded {row['tokens_changed']}"
            )
        planes[str(pair)] = plane
        changed += actual

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **planes)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "pairs_in_checkpoint": len(rows),
                "pairs_edited": len(planes),
                "tokens_changed": changed,
                "out": str(out),
                "sha256": digest,
                "bytes": out.stat().st_size,
                "rate_prior_bits_per_token": jg3.RATE_PRIOR_BITS_PER_TOKEN,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
