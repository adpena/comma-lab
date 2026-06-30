#!/usr/bin/env python3
"""Witness checkpoint ARCHIVER — thin CLI over ``tac.checkpoint_retention`` (operator 2026-06-30).

Keeps the WHOLE-RUN trajectory (every new best + periodic latest snaps), never overwriting, with a
moving-window->SSD-spill safety valve + append-only manifest. The reusable core (decorator + context
manager + direct API + policy) lives in ``src/tac/checkpoint_retention.py``; this is the manual/external
watcher for a RUNNING arm whose in-process save logic can't be changed.

  # one-shot: capture whatever best/latest exist now
  python tools/archive_witness_checkpoints.py --run-dir <run> --once
  # durable watcher (keep ALL local; snapshot latest every 60s):
  python tools/archive_witness_checkpoints.py --run-dir <run> --poll-sec 30 --latest-every-sec 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.checkpoint_retention import CheckpointArchiver, _utc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--archive-dir", type=Path, default=None)
    ap.add_argument("--poll-sec", type=float, default=60.0)
    ap.add_argument("--latest-every-sec", type=float, default=300.0, help="0 = do not snapshot latest")
    ap.add_argument("--keep-window", type=int, default=0, help="0 = keep ALL local; >0 = spill oldest latest to SSD")
    ap.add_argument("--no-spill", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)

    arch = CheckpointArchiver(args.archive_dir or (args.run_dir / "ckpt_archive"),
                              keep_window=args.keep_window, spill=not args.no_spill,
                              min_free_gb=args.min_free_gb)

    if args.once:
        r = arch.scan(args.run_dir, snapshot_latest=args.latest_every_sec > 0)
        print(json.dumps({"stage": "archive_once", "archive_dir": str(arch.archive_dir), **r}), flush=True)
        return 0

    print(json.dumps({"stage": "watch_start", "run_dir": str(args.run_dir),
                      "archive_dir": str(arch.archive_dir), "poll_sec": args.poll_sec,
                      "latest_every_sec": args.latest_every_sec, "keep_window": args.keep_window}), flush=True)
    last_latest = 0.0
    while True:
        snap = (args.latest_every_sec > 0) and (time.time() - last_latest >= args.latest_every_sec)
        try:
            r = arch.scan(args.run_dir, snapshot_latest=snap)
            if r.get("best") or r.get("latest") or r.get("best_error"):
                print(json.dumps({"stage": "archived", "ts": _utc(), **r}), flush=True)
        except Exception as e:
            print(json.dumps({"stage": "error", "ts": _utc(), "error": str(e)}), flush=True)
        if snap:
            last_latest = time.time()
        time.sleep(max(5.0, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
