#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an HPRC residual-protection surface from P18/P19 scorer priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np  # noqa: E402

from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.hprc.native_rate_surface import (  # noqa: E402
    FALSE_AUTHORITY,
    build_hprc_native_rate_residual_protection_surface,
    load_json_object,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p19-posenet-null-pairs", required=True, type=Path)
    parser.add_argument("--p18-segnet-region-waterfill", type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--residual-grid-h", required=True, type=int)
    parser.add_argument("--residual-grid-w", required=True, type=int)
    parser.add_argument("--gop-size", type=int, default=2)
    parser.add_argument("--default-protection", type=float, default=1.0)
    parser.add_argument("--p19-null-protection", type=float, default=0.15)
    parser.add_argument("--p18-region-protection", type=float, default=1.0)
    parser.add_argument("--output-npy", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    p19_path = _resolve(args.p19_posenet_null_pairs, repo_root=repo_root)
    p18_path = (
        None
        if args.p18_segnet_region_waterfill is None
        else _resolve(args.p18_segnet_region_waterfill, repo_root=repo_root)
    )
    protection, manifest = build_hprc_native_rate_residual_protection_surface(
        p19_posenet_null_pairs=load_json_object(p19_path),
        p18_segnet_region_waterfill=None if p18_path is None else load_json_object(p18_path),
        frames=int(args.frames),
        residual_grid_h=int(args.residual_grid_h),
        residual_grid_w=int(args.residual_grid_w),
        default_protection=float(args.default_protection),
        p19_null_protection=float(args.p19_null_protection),
        p18_region_protection=float(args.p18_region_protection),
        gop_size=int(args.gop_size),
    )
    output_npy = _resolve(args.output_npy, repo_root=repo_root)
    output_json = _resolve(args.out_json, repo_root=repo_root)
    if output_npy.exists() and not args.allow_overwrite:
        raise ArtifactWriteError(f"refusing to overwrite residual protection npy: {output_npy}")
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, protection)
    manifest = {
        **manifest,
        "output_npy": {
            "path": _repo_rel(output_npy, repo_root),
            "bytes": output_npy.stat().st_size,
            "sha256": sha256_file(output_npy),
        },
        "p19_posenet_null_pairs": _artifact_record(p19_path, repo_root=repo_root),
        "p18_segnet_region_waterfill": (
            None if p18_path is None else _artifact_record(p18_path, repo_root=repo_root)
        ),
        **FALSE_AUTHORITY,
    }
    expected = sha256_file(output_json) if output_json.exists() and args.allow_overwrite else None
    write_json_artifact(
        output_json,
        manifest,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=expected,
    )
    print(json.dumps({**manifest, "report_path": output_json.as_posix()}, sort_keys=True))
    return 0


def _resolve(path: Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_record(path: Path, *, repo_root: Path) -> dict[str, object]:
    return {
        "path": _repo_rel(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, FileNotFoundError, ValueError) as exc:
        print(f"build_hprc_native_rate_surface failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
