#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run queue-owned PR95 MLX Conv2d drift scope acquisition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    FALSE_AUTHORITY,
    LANE_ID,
    PR95_MLX_CONV2D_ACCUMULATION_MODES,
    PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    compare_pr95_public_archive_forward_with_pytorch,
    parse_pr95_public_archive_zip,
    pr95_mlx_conv2d_scope_search_candidates,
    trace_pr95_public_archive_decoder_with_pytorch,
)

DEFAULT_PUBLIC_PR95_SOURCE_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
DEFAULT_PUBLIC_PR95_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "archive.zip"
)
SCOPE_SEARCH_SCHEMA = "pr95_hnerv_mlx_conv2d_drift_scope_search.v1"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _load_public_pr95_decoder_cls(model_path: Path) -> Any:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"public PR95 source model.py not found: {model_path}")
    spec = importlib.util.spec_from_file_location(
        f"public_pr95_hnerv_model_scope_search_{abs(hash(model_path.resolve()))}",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import public PR95 model.py: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.HNeRVDecoder
    except AttributeError as exc:
        raise RuntimeError(f"{model_path} does not define HNeRVDecoder") from exc


def _rank_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        float(row["max_abs"]),
        float(row["mean_abs"]),
        int(row["override_count"]),
        float(row["elapsed_seconds"]),
    )


def _minimal_scope_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(row["override_count"]),
        float(row["max_abs"]),
        float(row["mean_abs"]),
        float(row["elapsed_seconds"]),
    )


def _candidate_row(
    candidate: dict[str, Any],
    *,
    attestation: dict[str, Any],
    trace: dict[str, Any],
    output_dir: Path,
    write_candidate_artifacts: bool,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    candidate_dir = output_dir / candidate_id
    attestation_path = candidate_dir / "forward_drift_attestation.json"
    trace_path = candidate_dir / "decoder_trace.json"
    if write_candidate_artifacts:
        _write_json(attestation_path, attestation)
        _write_json(trace_path, trace)
    parity = attestation["parity"]
    drift_cliff = trace.get("drift_cliff")
    return {
        "candidate_id": candidate_id,
        "kind": candidate["kind"],
        "conv2d_accumulation_overrides": candidate[
            "conv2d_accumulation_overrides"
        ],
        "override_count": candidate["override_count"],
        "passed": bool(parity["passed"]),
        "max_abs": float(parity["max_abs"]),
        "mean_abs": float(parity["mean_abs"]),
        "p99_abs": float(parity["p99_abs"]),
        "p999_abs": float(parity["p999_abs"]),
        "elapsed_seconds": float(attestation["elapsed_seconds"])
        + float(trace["elapsed_seconds"]),
        "drift_cliff_name": drift_cliff.get("name") if isinstance(drift_cliff, dict) else None,
        "drift_cliff_max_abs_delta": (
            drift_cliff.get("max_abs_delta") if isinstance(drift_cliff, dict) else None
        ),
        "attestation_path": _rel(attestation_path) if write_candidate_artifacts else None,
        "trace_path": _rel(trace_path) if write_candidate_artifacts else None,
        **FALSE_AUTHORITY,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, default=DEFAULT_PUBLIC_PR95_ARCHIVE)
    parser.add_argument(
        "--public-pr95-source-model",
        type=Path,
        default=DEFAULT_PUBLIC_PR95_SOURCE_MODEL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", action="append", type=int)
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--conv2d-accumulation-mode",
        choices=PR95_MLX_CONV2D_ACCUMULATION_MODES,
        default=PR95_MLX_OPTIMIZED_CONV2D_ACCUMULATION_MODE,
    )
    parser.add_argument("--atol-max", type=float, default=1e-2)
    parser.add_argument("--atol-mean", type=float, default=1e-3)
    parser.add_argument("--cliff-threshold", type=float, default=1e-3)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--no-presets", action="store_true")
    parser.add_argument("--no-single-blocks", action="store_true")
    parser.add_argument("--no-prefix-blocks", action="store_true")
    parser.add_argument("--write-candidate-artifacts", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.allow_existing_output_dir:
        raise SystemExit(
            f"output directory already exists: {_rel(output_dir)} "
            "(pass --allow-existing-output-dir to append/overwrite manifests)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_zip = args.archive_zip.resolve()
    source_model = args.public_pr95_source_model.resolve()
    packet = parse_pr95_public_archive_zip(archive_zip)
    torch_decoder_cls = _load_public_pr95_decoder_cls(source_model)
    candidates = pr95_mlx_conv2d_scope_search_candidates(
        block_count=args.block_count,
        include_presets=not args.no_presets,
        include_single_blocks=not args.no_single_blocks,
        include_prefix_blocks=not args.no_prefix_blocks,
    )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        overrides = candidate["conv2d_accumulation_overrides"]
        attestation = compare_pr95_public_archive_forward_with_pytorch(
            packet,
            torch_decoder_cls,
            sample_indices=args.sample_index or [0],
            mlx_device=args.mlx_device,
            atol_max=args.atol_max,
            atol_mean=args.atol_mean,
            conv2d_accumulation_mode=args.conv2d_accumulation_mode,
            conv2d_accumulation_overrides=overrides,
        )
        trace = trace_pr95_public_archive_decoder_with_pytorch(
            packet,
            torch_decoder_cls,
            sample_indices=args.sample_index or [0],
            mlx_device=args.mlx_device,
            cliff_threshold=args.cliff_threshold,
            conv2d_accumulation_mode=args.conv2d_accumulation_mode,
            conv2d_accumulation_overrides=overrides,
        )
        rows.append(
            _candidate_row(
                candidate,
                attestation=attestation,
                trace=trace,
                output_dir=output_dir,
                write_candidate_artifacts=args.write_candidate_artifacts,
            )
        )

    ranked = sorted(rows, key=_rank_key)
    best = ranked[0] if ranked else None
    passed_rows = [row for row in rows if row["passed"]]
    no_cliff_rows = [
        row for row in rows if row["passed"] and row["drift_cliff_name"] is None
    ]
    minimal_passed = min(passed_rows, key=_minimal_scope_key) if passed_rows else None
    minimal_no_cliff = (
        min(no_cliff_rows, key=_minimal_scope_key) if no_cliff_rows else None
    )
    summary = {
        "schema": SCOPE_SEARCH_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": LANE_ID,
        "source_pr": 95,
        "submission": "hnerv_muon",
        "evidence_grade": "[macOS-MLX research-signal]",
        "archive_zip": _rel(archive_zip),
        "archive_sha256": _sha256_file(archive_zip),
        "public_pr95_source_model": _rel(source_model),
        "public_pr95_source_model_sha256": _sha256_file(source_model),
        "mlx_device": args.mlx_device,
        "conv2d_accumulation_mode": args.conv2d_accumulation_mode,
        "sample_indices": args.sample_index or [0],
        "candidate_count": len(rows),
        "candidate_artifacts_written": bool(args.write_candidate_artifacts),
        "best_by_delta_candidate": best,
        "minimal_passed_candidate": minimal_passed,
        "minimal_no_cliff_candidate": minimal_no_cliff,
        "rows": ranked,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "local_mlx_conv2d_scope_search_is_not_contest_auth_eval",
                "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }
    summary_path = output_dir / "scope_search_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({"ok": True, "summary": _rel(summary_path), **FALSE_AUTHORITY}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
