#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the Task #494 ANE fixed-point build/refusal receipt.

This command is intentionally prepare-only.  It consumes the full-n600 QDQ
receipt and the settled #482 CoreML W8A8 receipt, writes a small atomic JSON
decision, and never launches CoreML conversion or a trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.local_acceleration.ane_fixedpoint_authority import (  # noqa: E402
    compile_ane_fixedpoint_ticket,
)
from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
)

DEFAULT_QDQ = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_n600.json"
)
SETTLED_R4 = REPO / "experiments/results/ane_unlock_correction_20260713/r4_variants.json"
DEFAULT_OUTPUT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "ane_fixedpoint_authority_ticket.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdq-receipt", type=Path, default=DEFAULT_QDQ)
    parser.add_argument("--settled-r4-receipt", type=Path, default=SETTLED_R4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--formulation-id",
        default="coreml_linear_symmetric_per_channel_w8a8_ptq",
    )
    args = parser.parse_args()
    for path in (args.qdq_receipt, args.settled_r4_receipt):
        if not path.is_file():
            parser.error(f"required receipt is absent: {path}")
    qdq = json.loads(args.qdq_receipt.read_text(encoding="utf-8"))
    r4 = json.loads(args.settled_r4_receipt.read_text(encoding="utf-8"))
    ticket = compile_ane_fixedpoint_ticket(
        qdq_receipt=qdq,
        settled_r4_receipt=r4,
        formulation_id=args.formulation_id,
    )
    payload = {
        "schema": "ane_fixedpoint_authority_ticket.v1",
        "lane_id": "throughput_authority_ladder",
        "task_id": 494,
        "axis": "[source-inspection + receipt compile; research-only MEANS]",
        "qdq_receipt": str(args.qdq_receipt.relative_to(REPO)),
        "qdq_receipt_sha256": _sha256(args.qdq_receipt),
        "settled_r4_receipt": str(args.settled_r4_receipt.relative_to(REPO)),
        "settled_r4_receipt_sha256": _sha256(args.settled_r4_receipt),
        "formulation_id": args.formulation_id,
        "ticket": ticket.to_dict(),
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
