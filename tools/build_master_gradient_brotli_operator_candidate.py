#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a PR106 Brotli-section master-gradient operator candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.master_gradient_brotli_operator_candidate import (  # noqa: E402
    SUPPORTED_TARGET_SECTIONS,
    MasterGradientBrotliOperatorError,
    build_master_gradient_brotli_operator_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-section", choices=sorted(SUPPORTED_TARGET_SECTIONS), required=True)
    parser.add_argument(
        "--quality",
        type=int,
        action="append",
        dest="qualities",
        help="Brotli quality to try. Repeatable. Defaults to 0..11.",
    )
    parser.add_argument(
        "--lgwin",
        type=int,
        action="append",
        dest="lgwins",
        help="Brotli lgwin to try. Repeatable. Defaults to 10..24.",
    )
    parser.add_argument(
        "--allow-non-improving",
        action="store_true",
        help="Allow byte-neutral/regressive materialization for diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_master_gradient_brotli_operator_candidate(
            source_archive=args.source_archive,
            output_dir=args.output_dir,
            target_section=args.target_section,
            candidate_id=args.candidate_id,
            qualities=tuple(args.qualities) if args.qualities else tuple(range(12)),
            lgwin_values=tuple(args.lgwins) if args.lgwins else tuple(range(10, 25)),
            require_smaller=not args.allow_non_improving,
        )
    except (MasterGradientBrotliOperatorError, OSError) as exc:
        raise SystemExit(f"master-gradient Brotli operator build failed: {exc}") from None

    archive = manifest["candidate_archive"]
    replacement = manifest["replacement_payload"]
    print(
        f"wrote {archive['path']} ({archive['bytes']} bytes, "
        f"sha256={archive['sha256']})"
    )
    print(
        f"{args.target_section}: {manifest['source_section']['bytes']} -> "
        f"{replacement['bytes']} bytes "
        f"(section_delta={replacement['section_byte_delta']}, "
        f"archive_delta={archive['archive_byte_delta']})"
    )
    print(
        "score_claim=false promotion_eligible=false "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']}"
    )
    print(f"operator_manifest={args.output_dir / 'operator_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
