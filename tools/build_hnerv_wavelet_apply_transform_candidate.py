#!/usr/bin/env python3
"""Build a plain-HNeRV candidate by applying WR01 atoms offline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_wavelet_apply_transform import build_wavelet_apply_transform_candidate  # noqa: E402
from tac.repo_io import json_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--section-name", default="latents_and_sidecar_brotli")
    parser.add_argument("--strength-numerator", type=int, default=1)
    parser.add_argument("--strength-denominator", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-archive-preflight-ready", action="store_true")
    args = parser.parse_args(argv)

    payload = build_wavelet_apply_transform_candidate(
        wavelet_archive=args.wavelet_archive,
        output_dir=args.output_dir,
        source_label=args.source_label,
        section_name=args.section_name,
        strength_numerator=args.strength_numerator,
        strength_denominator=args.strength_denominator,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_not_archive_preflight_ready and not payload["ready_for_archive_preflight"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
