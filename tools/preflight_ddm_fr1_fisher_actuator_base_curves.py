#!/usr/bin/env python3
"""Verify FR1 actuator/base custody and emit a fail-closed typed receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.ddm_fr1_fisher_preflight import (  # noqa: E402
    FR1PreflightError,
    build_preflight_receipt,
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FR1PreflightError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--expect-blocked",
        action="store_true",
        help="return success only when the typed preflight blocks execution",
    )
    return result


def execute(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_bytes())
    receipt = build_preflight_receipt(config, repo_root=REPO)
    receipt["typed_config"] = {
        "path": str(config_path.relative_to(REPO)),
        "bytes": config_path.stat().st_size,
        "sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
    }
    _atomic_write(args.output.expanduser().resolve(), _canonical_json(receipt))
    blocked = not receipt["execution_allowed"]
    if args.expect_blocked:
        return 0 if blocked else 4
    return 3 if blocked else 0


def main() -> int:
    try:
        return execute(parser().parse_args())
    except (FR1PreflightError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"DDM_FR1_PREFLIGHT_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
