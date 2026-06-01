#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a fail-closed pose-guarded SNeRV decoder gate artifact."""

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

from tac.analysis.snerv_pose_guarded_decoder_gate import (  # noqa: E402
    build_snerv_pose_guarded_decoder_gate,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_pose_guarded_decoder_gate_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="SNeRV advisory/sweep JSON payloads to gate.",
    )
    parser.add_argument("--out", default=None, help="Output gate JSON path.")
    parser.add_argument("--byte-slack", type=int, default=2048)
    parser.add_argument("--pose-slack", type=float, default=0.0)
    parser.add_argument("--seg-ceiling", type=float, default=0.02)
    args = parser.parse_args(argv)

    payloads = []
    source_paths = []
    source_hashes = {}
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        source_bytes = path.read_bytes()
        payloads.append(json.loads(source_bytes.decode("utf-8")))
        rel = str(path)
        source_paths.append(rel)
        source_hashes[rel] = hashlib.sha256(source_bytes).hexdigest()

    gate = build_snerv_pose_guarded_decoder_gate(
        payloads,
        source_paths=tuple(source_paths),
        byte_slack=args.byte_slack,
        pose_slack=args.pose_slack,
        seg_ceiling=args.seg_ceiling,
    )
    out_payload = gate.as_jsonable()
    out_payload["source_sha256"] = source_hashes

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")

    print("[SNeRV pose-guarded decoder gate] false-authority")
    print(f"  inputs: {len(args.inputs)}")
    print(f"  baseline: {gate.baseline_label}")
    print(f"  accepted_rows: {len(gate.accepted_rows)}")
    print(f"  closed_form_scalar_weighting_no_go: {gate.closed_form_scalar_weighting_no_go}")
    print(f"  verdict: {gate.verdict}")
    print(f"  next: {gate.next_action}")
    if gate.blockers:
        print(f"  blockers: {list(gate.blockers)}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
