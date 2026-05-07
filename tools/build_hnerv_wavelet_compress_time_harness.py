#!/usr/bin/env python3
"""Build fail-closed WR01 compress-time harness and planning manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_wavelet_compress_time_harness import (  # noqa: E402
    build_wavelet_compress_time_harness,
)
from tac.repo_io import json_text, read_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for hnerv_wavelet_compress_time_harness.json, the "
            "planning-only atom plan, and the selected-atom apply-readiness scaffold."
        ),
    )
    parser.add_argument("--target-section", action="append", dest="target_sections")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atom-budget", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quant-step", type=float, default=1.0)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--expected-source-archive-sha256")
    parser.add_argument("--expected-source-archive-bytes", type=int)
    parser.add_argument(
        "--runtime-apply-manifest",
        type=Path,
        help=(
            "Optional manifest from build_hnerv_wavelet_apply_transform_candidate.py. "
            "When provided, the harness writes a fail-closed runtime/decode review "
            "manifest tying the applied WR01 atom IDs back to the selected atoms."
        ),
    )
    parser.add_argument("--exact-decode-validation", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-blocked", action="store_true")
    args = parser.parse_args(argv)

    runtime_apply_manifest = (
        read_json(args.runtime_apply_manifest) if args.runtime_apply_manifest is not None else None
    )
    exact_decode_validation = (
        read_json(args.exact_decode_validation) if args.exact_decode_validation is not None else None
    )
    manifest = build_wavelet_compress_time_harness(
        source_archive=args.source_archive,
        source_label=args.source_label,
        output_dir=args.output_dir,
        target_sections=tuple(args.target_sections or ("latents_and_sidecar_brotli",)),
        seed=args.seed,
        atom_budget=args.atom_budget,
        block_size=args.block_size,
        quant_step=args.quant_step,
        train_steps=args.train_steps,
        expected_source_archive_sha256=args.expected_source_archive_sha256,
        expected_source_archive_bytes=args.expected_source_archive_bytes,
        runtime_apply_manifest=runtime_apply_manifest,
        exact_decode_validation=exact_decode_validation,
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_blocked and manifest["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
