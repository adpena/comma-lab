#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile Task #494's receipt-bound op/substrate/precision assignment.

This is a pure decision compiler. It never launches a trainer, Metal work, or
an evaluator. Missing or incomplete receipts produce explicit HELD_OWED rows;
they are never coerced into authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.witness_dsl.throughput_authority_policy_20260714 import (  # noqa: E402
    compile_throughput_authority_policy,
)

RESULTS = REPO / "experiments/results/throughput_authority_ladder_20260714"
DEFAULT_QDQ = RESULTS / "dynamic_fixedpoint_scorer_forward_n600.json"
DEFAULT_METAL = RESULTS / "metal_dynamic_fixedpoint_segnet_n600.json"
DEFAULT_INTEGER_R = RESULTS / "integer_r_backend_n600.json"
DEFAULT_OUTPUT = RESULTS / "throughput_authority_policy.json"


def _load_optional(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not path.is_file():
        return None, {"path": str(path), "status": "OWED"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {
        "path": str(path.resolve().relative_to(REPO.resolve())),
        "status": "PRESENT",
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdq", type=Path, default=DEFAULT_QDQ)
    parser.add_argument("--metal", type=Path, default=DEFAULT_METAL)
    parser.add_argument("--integer-r", type=Path, default=DEFAULT_INTEGER_R)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pose-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--pose-canary-every", type=int, default=8)
    parser.add_argument("--banked-r1-dpose", type=float, default=0.001610)
    parser.add_argument("--require-receipts", action="store_true")
    args = parser.parse_args()

    qdq, qdq_custody = _load_optional(args.qdq)
    metal, metal_custody = _load_optional(args.metal)
    integer_r, integer_r_custody = _load_optional(args.integer_r)
    policy = compile_throughput_authority_policy(
        fixedpoint_qdq_receipt=qdq,
        metal_fixedpoint_receipt=metal,
        integer_r_receipt=integer_r,
        pose_gate_enabled=args.pose_gate,
        pose_canary_every=args.pose_canary_every,
        banked_r1_dpose=args.banked_r1_dpose,
    )
    missing = [
        name
        for name, row in (
            ("qdq", qdq_custody),
            ("metal", metal_custody),
            ("integer_r", integer_r_custody),
        )
        if row["status"] != "PRESENT"
    ]
    payload = policy.to_dict()
    payload.update(
        {
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[decision compile; research-only MEANS]",
            "receipt_custody": {
                "qdq": qdq_custody,
                "metal": metal_custody,
                "integer_r": integer_r_custody,
            },
            "missing_receipts": missing,
            "terminal_authority": (
                "exact archive bytes through upstream/evaluate.py on contest CPU/CUDA"
            ),
        }
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2 if args.require_receipts and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
