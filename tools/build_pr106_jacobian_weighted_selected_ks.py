#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build PR106 Jacobian-weighted selected-K planning manifest.

This is a CPU-safe selected-K producer.  It requires a future CUDA-authored,
non-diagnostic Jacobian/scorer-pullback importance JSON manifest, reduces
per-channel importance to phase-1 per-tensor scalars, runs the canonical
``JacobianWeightedAllocator`` on PR106 K curves, and emits
``weighted_k_allocations[].selected_Ks`` for the no-dead-K builder interface.

No scorer is loaded, no archive is built, no GPU job is launched, and the
output is not promotion/rank/kill evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr106_omega_opt_lagrangian_per_tensor_allocation_empirical import (  # noqa: E402
    DEFAULT_PR106_ARCHIVE,
    PR106_ARCHIVE_OVERHEAD_BYTES,
    PR106_DECODER_BROTLI_BASELINE_BYTES,
    _encode_decoder_brotli_with_per_tensor_K,
    _precompute_K_curves,
    collect_pr106_tensors,
)

from tac.codec.cost_curves import TensorBlob  # noqa: E402
from tac.optimization.jacobian_weighted_selected_k import (  # noqa: E402
    build_jacobian_selected_k_manifest,
    load_importance_manifest,
)
from tac.repo_io import repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/build_pr106_jacobian_weighted_selected_ks.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/raw/pr106_jacobian_weighted_selected_ks"


def _utc_ts() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _make_joint_encoder(pr106_tensors: list[Any]):
    def hook(selections: list[dict[str, Any]]) -> dict[str, Any]:
        selected_ks = [int(selection["K"]) for selection in selections]
        encoded = _encode_decoder_brotli_with_per_tensor_K(pr106_tensors, selected_ks)
        return {
            "total_bytes": int(encoded["archive_bytes"]),
            "rel_err": float(encoded["rel_err"]),
            "Ks": selected_ks,
            "decoder_brotli_bytes": int(encoded["decoder_brotli_bytes"]),
            "archive_overhead_bytes": int(encoded["archive_overhead_bytes"]),
        }

    return hook


def build_manifest(
    *,
    archive_path: Path,
    importance_manifest_path: Path,
    rms_targets: list[float],
    max_k: int,
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise SystemExit(f"PR106 archive not found: {archive_path}")
    if not importance_manifest_path.is_file():
        raise SystemExit(f"importance manifest not found: {importance_manifest_path}")
    if max_k < 1:
        raise SystemExit("--max-K must be >= 1")
    if max_k > 64:
        raise SystemExit("PR106 scaffold currently supports --max-K <= 64")

    pr106_tensors = collect_pr106_tensors(archive_path)
    tensors = [
        TensorBlob(name=tensor.name, raw=tensor.raw_i8.astype("int32"))
        for tensor in pr106_tensors
    ]
    curves = _precompute_K_curves(pr106_tensors)
    if max_k < 64:
        curves = [rows[:max_k] for rows in curves]

    importance_payload = load_importance_manifest(importance_manifest_path)
    return build_jacobian_selected_k_manifest(
        tensors=tensors,
        importance_payload=importance_payload,
        rms_targets=rms_targets,
        k_range=list(range(1, max_k + 1)),
        joint_encoder=_make_joint_encoder(pr106_tensors),
        curves=curves,
        importance_manifest_path=importance_manifest_path,
        producer_tool=TOOL_NAME,
        extra_inputs={
            "pr106_archive": repo_relative(archive_path, REPO_ROOT),
            "pr106_archive_sha256": sha256_file(archive_path),
            "producer_substrate": "pr106_decoder_packed_brotli_per_tensor_K",
            "pr106_decoder_brotli_baseline_bytes": PR106_DECODER_BROTLI_BASELINE_BYTES,
            "pr106_archive_overhead_bytes": PR106_ARCHIVE_OVERHEAD_BYTES,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--importance-manifest", type=Path, required=True)
    parser.add_argument("--rms-targets", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--max-K", type=int, default=64)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        archive_path=args.archive,
        importance_manifest_path=args.importance_manifest,
        rms_targets=list(args.rms_targets),
        max_k=int(args.max_K),
    )
    output_json = args.output_json
    if output_json is None:
        output_json = DEFAULT_OUTPUT_ROOT / _utc_ts() / "manifest.json"
    write_json(output_json, manifest)

    print(f"manifest: {output_json}")
    for row in manifest["weighted_k_allocations"]:
        print(
            "selected_Ks:"
            f" target={float(row['rms_target']):.4f}"
            f" bytes={int(row['total_bytes']):,}"
            f" rel_err={float(row['rel_err']):.6f}"
        )
    print("status: planning-only; no score claim, no dispatch, no promotion/rank/kill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
