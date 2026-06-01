#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a fail-closed SNeRV scorer-loop decoder/QAT contract artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.snerv_scorer_loop_decoder_qat_contract import (  # noqa: E402
    DEFAULT_LANE_ID,
    build_snerv_scorer_loop_decoder_qat_contract,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_scorer_loop_decoder_qat_contract_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pose_gate",
        help="SNeRV pose-guarded decoder gate JSON artifact.",
    )
    parser.add_argument("--out", default=None, help="Output contract JSON path.")
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument(
        "--dispatch-hold",
        default=None,
        help="Optional exact/full-video dispatch hold reason to preserve in blockers.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.pose_gate)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    source_bytes = in_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    pose_gate = json.loads(source_bytes.decode("utf-8"))

    contract = build_snerv_scorer_loop_decoder_qat_contract(
        pose_gate,
        source_gate_path=str(in_path),
        source_gate_sha256=source_sha,
        lane_id=args.lane_id,
        dispatch_hold_reason=args.dispatch_hold,
    )

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(contract.as_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[SNeRV scorer-loop decoder/QAT contract] false-authority")
    print(f"  source_gate: {in_path}")
    print(f"  source_sha256: {source_sha}")
    print(f"  lane_id: {contract.lane_id}")
    print(
        "  ready_for_scorer_loop_trainer_implementation: "
        f"{contract.ready_for_scorer_loop_trainer_implementation}"
    )
    print(f"  ready_for_local_training_smoke: {contract.ready_for_local_training_smoke}")
    print(f"  ready_for_exact_eval_dispatch: {contract.ready_for_exact_eval_dispatch}")
    print(f"  next_code_artifacts: {len(contract.next_code_artifacts)}")
    if contract.blockers:
        print(f"  blockers: {list(contract.blockers)}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
