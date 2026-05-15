#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect a D1 sidecar for decoded overlay effect before auth eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.d1_segnet_margin_polytope import (  # noqa: E402
    analyze_d1_overlay_effect,
    parse_archive,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-bin", type=Path, required=True)
    parser.add_argument("--channel-policy", default=None)
    parser.add_argument("--amplitude-scale", type=float, default=None)
    parser.add_argument("--sign-policy", default=None)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    archive = parse_archive(args.d1_bin.read_bytes())
    diag = analyze_d1_overlay_effect(
        archive,
        channel_policy=args.channel_policy,
        amplitude_scale=args.amplitude_scale,
        sign_policy=args.sign_policy,
    )
    payload = {
        "d1_bin": args.d1_bin.as_posix(),
        "d1_bin_bytes": args.d1_bin.stat().st_size,
        "base_substrate_id": archive.base_substrate_id,
        "base_archive_sha256_truncated": archive.base_archive_sha256_truncated,
        "margin_map_resolution": [archive.height, archive.width],
        "d1_overlay_diagnostics": diag.to_json_dict(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": diag.dispatch_blockers,
    }
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
