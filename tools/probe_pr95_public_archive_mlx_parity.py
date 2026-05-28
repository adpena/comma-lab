#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe public PR95 archive-packet forward parity between PyTorch and MLX."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ZIP = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "archive.zip"
)
DEFAULT_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)


def _load_public_pr95_model(source_model: Path) -> Any:
    if not source_model.is_file():
        raise SystemExit(f"public PR95 model.py not found: {source_model}")
    spec = importlib.util.spec_from_file_location(
        "public_pr95_hnerv_model_for_mlx_parity_cli",
        source_model,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to import public PR95 model.py: {source_model}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, default=DEFAULT_ARCHIVE_ZIP)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--mlx-device",
        choices=("cpu", "gpu"),
        action="append",
        default=None,
        help="MLX device(s) to probe. Defaults to cpu.",
    )
    parser.add_argument(
        "--sample-indices",
        help="Comma-separated PR95 latent-row indices. Default: first, middle, last.",
    )
    parser.add_argument("--atol-max", type=float, default=2e-3)
    parser.add_argument("--atol-mean", type=float, default=1e-4)
    parser.add_argument(
        "--conv2d-accumulation-mode",
        choices=("optimized", "fixed_fp32", "kahan_fp32", "fixed_fp64"),
        default="optimized",
        help="PR95 MLX Conv2d accumulation path to probe.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero if any requested device exceeds configured tolerances.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from tac.local_acceleration.pr95_hnerv_mlx import (
        FALSE_AUTHORITY,
        compare_pr95_public_archive_forward_with_pytorch,
        parse_pr95_public_archive_zip,
    )

    args = _build_parser().parse_args(argv)
    model_module = _load_public_pr95_model(args.source_model)
    packet = parse_pr95_public_archive_zip(args.archive_zip)
    devices = args.mlx_device or ["cpu"]
    sample_indices = _parse_indices(args.sample_indices)
    results = [
        compare_pr95_public_archive_forward_with_pytorch(
            packet,
            model_module.HNeRVDecoder,
            sample_indices=sample_indices,
            mlx_device=device,
            atol_max=args.atol_max,
            atol_mean=args.atol_mean,
            conv2d_accumulation_mode=args.conv2d_accumulation_mode,
        )
        for device in devices
    ]
    payload = {
        "schema": "pr95_hnerv_public_archive_mlx_forward_parity_probe.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "archive_packet": packet.custody_manifest(),
        "source_model_path": args.source_model.as_posix(),
        "conv2d_accumulation_mode": args.conv2d_accumulation_mode,
        "results": results,
        "all_passed": all(result["parity"]["passed"] for result in results),
        "require_pass": bool(args.require_pass),
        **FALSE_AUTHORITY,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 1 if args.require_pass and not payload["all_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
