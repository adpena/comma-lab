#!/usr/bin/env python3
"""Measure the fail-closed shared PDW2/realization receiver intersection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from tac.boundary_math.shared_receiver_admission import (
    evaluate_shared_receiver_admission,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{name} is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must contain a JSON object: {resolved}")
    return value


def _atomic_output(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise SystemExit(f"stale output temporary requires review: {partial}")
    try:
        partial.write_text(payload)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _durable_display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdw2-receipt", required=True, type=Path)
    parser.add_argument("--pdw1-receipt", required=True, type=Path)
    parser.add_argument("--step2-summary", required=True, type=Path)
    parser.add_argument("--dense-section-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {output}")
    repo = Path(__file__).resolve().parents[1]
    inputs = {
        "pdw2": args.pdw2_receipt.expanduser().resolve(strict=True),
        "pdw1": args.pdw1_receipt.expanduser().resolve(strict=True),
        "step2": args.step2_summary.expanduser().resolve(strict=True),
    }
    if args.dense_section_receipt is not None:
        inputs["dense_section"] = args.dense_section_receipt.expanduser().resolve(strict=True)
    result = evaluate_shared_receiver_admission(
        pdw2_receipt=_load(inputs["pdw2"], "pdw2 receipt"),
        pdw1_receipt=_load(inputs["pdw1"], "pdw1 receipt"),
        step2_summary=_load(inputs["step2"], "step2 summary"),
        dense_section_receipt=None
        if "dense_section" not in inputs
        else _load(inputs["dense_section"], "dense spatial-section receipt"),
    )
    result["measurement"] = {
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git_head(repo),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "inputs": {
            key: {
                "path": _durable_display_path(path, repo),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for key, path in sorted(inputs.items())
        },
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    _atomic_output(output, payload)
    print(payload, end="")
    return 0 if result["success"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
