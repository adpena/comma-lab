#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest TM1 cross-run telemetry frames from existing run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.crossrun_trajectory_mining import (  # noqa: E402
    analyze_frames,
    harvest_roots,
    load_frames_jsonl,
    write_frames_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="artifact root to scan; repeatable",
    )
    parser.add_argument("--frames-jsonl", type=Path)
    parser.add_argument(
        "--from-frames-jsonl",
        type=Path,
        help="load an existing frame JSONL and emit a fresh summary without rescanning roots",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-bytes-per-file", type=int, default=8_000_000)
    parser.add_argument("--max-records-per-file", type=int, default=50_000)
    parser.add_argument("--no-file-hash", action="store_true")
    args = parser.parse_args(argv)

    if args.from_frames_jsonl is not None:
        frames = load_frames_jsonl(args.from_frames_jsonl)
        summary = {
            "frames_jsonl": str(args.from_frames_jsonl),
            "harvest": {
                "schema_version": "crossrun_frame_reanalysis_v1_20260804",
                "frames_loaded": len(frames),
                "runs": len({f.run_id for f in frames}),
            },
            "analysis": analyze_frames(frames),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary["harvest"], indent=2, sort_keys=True))
        return 0

    if args.frames_jsonl is None:
        parser.error("--frames-jsonl is required unless --from-frames-jsonl is used")

    roots = args.root or [
        REPO / "experiments" / "results",
        REPO / ".omx" / "research",
        Path("/Volumes/VertigoDataTier/pact"),
    ]
    result = harvest_roots(
        roots,
        max_files=args.max_files,
        max_bytes_per_file=args.max_bytes_per_file,
        max_records_per_file=args.max_records_per_file,
        hash_files=not args.no_file_hash,
    )
    write_frames_jsonl(result.frames, args.frames_jsonl)
    summary = {
        "roots": [str(r) for r in roots],
        "harvest": result.to_summary(),
        "analysis": analyze_frames(result.frames),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["harvest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
