"""Run one real structural treatment while the independent control completes.

The charter's prior landed-encoder reproduction is a hard prerequisite inside
hpac.prepare. This wrapper permits concurrent CPU encoding, NOT admission before
the additional live-loop control matches the source. All bytes are retained even
if that control fails. Neither this wrapper nor hpac.encode seals or scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_ntb2_hpac as hpac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=hpac.TREATMENTS[1:], required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    root = hpac.ROOT / args.treatment
    if args.resume_from.resolve() != root.resolve():
        raise ValueError("wrong resume root")
    # Retain the exact scheduling policy; the underlying measurement producer
    # separately pins the model mutation, archive, field and receiver sources.
    hpac.record(
        root / "PENDING_ADMISSION.json",
        {
            "producer": hpac.fact(Path(__file__)),
            "score_claim": False,
            "admitted": False,
            "control_required": str(hpac.ROOT / "control/PRICE.json"),
            "policy": "real bytes may be produced concurrently; no admission before full source identity",
        },
    )
    result = hpac.encode(args.treatment)
    control = hpac.ROOT / "control/PRICE.json"
    identity = False
    if control.exists():
        proof = json.loads(control.read_text())
        identity = proof["n"] == 600 and all(item["sha256"] == hpac.control.BASE_SHA for item in proof["twins"])
    hpac.record(
        root / "CONTROL_GATE_AT_HARVEST.json",
        {
            "control": hpac.fact(control) if control.exists() else None,
            "control_identity_verified": identity,
            "public_decode_verified": False,
            "admitted": False,
            "score_claim": False,
            "price": hpac.fact(root / "PRICE.json"),
            "status": "AWAIT_PUBLIC_IDENTITY" if identity else "HOLD_FOR_CAUSAL_CONTROL",
        },
    )
    print(
        json.dumps(
            {
                "treatment": args.treatment,
                "delta_bytes": result["delta_bytes"],
                "control_identity_verified": identity,
                "admitted": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
