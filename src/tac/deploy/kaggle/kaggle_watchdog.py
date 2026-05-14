# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_status_sync(manifest_path: Path) -> dict[str, object]:
    from kaggle_status_sync import sync_kaggle_status

    return sync_kaggle_status(manifest_path)


def _default_queue_tick(manifests_dir: Path, max_active: int) -> dict[str, object]:
    from kaggle_queue_tick import tick_queue

    return tick_queue(manifests_dir=manifests_dir, max_active=max_active)


def run_once(
    *,
    manifests_dir: Path,
    log_dir: Path,
    max_active: int = 2,
    status_sync: Callable[[Path], dict[str, object]] | None = None,
    queue_tick: Callable[[Path, int], dict[str, object]] | None = None,
) -> dict[str, object]:
    status_sync = status_sync or _default_status_sync
    queue_tick = queue_tick or _default_queue_tick

    synced: list[dict[str, object]] = []
    for manifest_path in sorted(manifests_dir.glob("kaggle-*.json")):
        synced.append(status_sync(manifest_path))

    queue_result = queue_tick(manifests_dir, max_active)
    payload = {
        "ts_utc": now_iso(),
        "synced": synced,
        "queue": queue_result,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_path = log_dir / "latest.json"
    history_path = log_dir / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2))
    with history_path.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-interval Kaggle status/queue watchdog.")
    parser.add_argument("--manifests-dir", type=Path, default=Path(".omx/logs/remote_jobs"))
    parser.add_argument("--log-dir", type=Path, default=Path("reports/raw/kaggle_watchdog"))
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    iteration = 0
    while True:
        run_once(
            manifests_dir=args.manifests_dir,
            log_dir=args.log_dir,
            max_active=args.max_active,
        )
        iteration += 1
        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
