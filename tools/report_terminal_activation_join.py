#!/usr/bin/env python3
"""Emit a scorer-free terminal compiled-config to activation-ledger join receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tac.witness_dsl.activation_ledger import (
    LEDGER_PATH,
    TerminalJoinStatus,
    terminal_activation_join,
)


def _read_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read compiled config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("compiled config JSON must be an object")
    return payload


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled_config", type=Path)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = terminal_activation_join(
            _read_config(args.compiled_config), path=args.ledger
        )
        body = json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            _atomic_write(args.output, body)
        sys.stdout.write(body)
        return 0 if receipt.status is TerminalJoinStatus.PASS else 2
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"REFUSE terminal activation join: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
