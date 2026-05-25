#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a PR95/HNeRV MLX public archive packet to PyTorch with parity proof."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
PR95_MLX_EXPORT_BRIDGE_SCHEMA = "pr95_mlx_archive_pytorch_export_bridge.v1"


def _load_public_pr95_model(source_model: Path) -> Any:
    if not source_model.is_file():
        raise FileNotFoundError(f"public PR95 model.py not found: {source_model}")
    spec = importlib.util.spec_from_file_location(
        "public_pr95_hnerv_model_for_archive_export_bridge",
        source_model,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import public PR95 model.py: {source_model}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "HNeRVDecoder"):
        raise RuntimeError(f"{source_model}: missing HNeRVDecoder")
    return module


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def export_pr95_public_archive_to_pytorch_state_dict(
    *,
    archive_zip: Path,
    output_pytorch_state_dict: Path,
    source_model: Path = DEFAULT_SOURCE_MODEL,
    report_out: Path | None = None,
    sample_indices: list[int] | None = None,
    mlx_device: str = "cpu",
    atol_max: float = 2e-3,
    atol_mean: float = 1e-4,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Parse the PR95 archive packet, write a .pt state_dict, and prove parity.

    The archive packet is the source of truth.  This helper intentionally avoids
    synthetic random-init checkpoint semantics; it delegates parsing, export,
    and forward parity to the canonical PR95 MLX helpers.
    """

    from tac.local_acceleration.pr95_hnerv_mlx import (
        FALSE_AUTHORITY,
        parse_pr95_public_archive_zip,
        write_pr95_public_archive_pytorch_export_forward_parity,
    )

    model_module = _load_public_pr95_model(Path(source_model))
    packet = parse_pr95_public_archive_zip(Path(archive_zip))
    run_id = (
        "pr95_mlx_archive_export_"
        f"{packet.archive_zip_sha256[:12]}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    parity = write_pr95_public_archive_pytorch_export_forward_parity(
        packet,
        model_module.HNeRVDecoder,
        output_pt_path=Path(output_pytorch_state_dict),
        run_id=run_id,
        sample_indices=sample_indices,
        mlx_device=mlx_device,
        atol_max=atol_max,
        atol_mean=atol_mean,
        overwrite=overwrite,
    )
    report = {
        "schema_version": PR95_MLX_EXPORT_BRIDGE_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/export_pr95_mlx_to_pytorch_state_dict.py",
        "source_model_path": Path(source_model).as_posix(),
        "archive_zip_path": Path(archive_zip).as_posix(),
        "output_pytorch_state_dict": Path(output_pytorch_state_dict).as_posix(),
        "run_id": run_id,
        "archive_packet": packet.custody_manifest(),
        "pytorch_export_forward_parity": parity,
        "pytorch_export_forward_parity_established": bool(
            parity.get("pytorch_export_forward_parity_established") is True
        ),
        "pt_path": parity.get("pt_path"),
        "pt_sha256": parity.get("pt_sha256"),
        "pt_bytes": parity.get("pt_bytes"),
        "sample_indices": parity.get("sample_indices"),
        "mlx_device": mlx_device,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "local_mlx_pytorch_export_parity_probe_is_not_contest_auth_eval",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }
    if report_out is not None:
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-pytorch-state-dict", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--sample-indices")
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--atol-max", type=float, default=2e-3)
    parser.add_argument("--atol-mean", type=float, default=1e-4)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to overwrite an existing output .pt.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when the forward parity proof fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = export_pr95_public_archive_to_pytorch_state_dict(
        archive_zip=args.archive_zip,
        output_pytorch_state_dict=args.output_pytorch_state_dict,
        source_model=args.source_model,
        report_out=args.report_out,
        sample_indices=_parse_indices(args.sample_indices),
        mlx_device=args.mlx_device,
        atol_max=args.atol_max,
        atol_mean=args.atol_mean,
        overwrite=not args.no_overwrite,
    )
    passed = report["pytorch_export_forward_parity_established"]
    parity = report["pytorch_export_forward_parity"]["forward_parity"]["parity"]
    print(f"[pr95-mlx-pytorch-export] parity_passed={passed}")
    print(f"[pr95-mlx-pytorch-export] pt={report['pt_path']}")
    print(f"[pr95-mlx-pytorch-export] report={args.report_out}")
    print(
        "[pr95-mlx-pytorch-export] "
        f"max_abs={parity['max_abs']:.6e} mean_abs={parity['mean_abs']:.6e}"
    )
    return 1 if args.require_pass and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
