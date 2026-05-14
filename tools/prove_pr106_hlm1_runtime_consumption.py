#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove a PR106 HLM archive's fixed-latent section is runtime-consumed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr106_hlm1_runtime_consumption import (  # noqa: E402
    dumps_hlm1_runtime_consumption_manifest,
    prove_pr106_hlm_runtime_consumption,
    prove_pr106_hlm1_runtime_consumption,
)
from tac.repo_io import write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-codec",
        choices=("hlm1", "hlm2"),
        default="hlm1",
        help="HLM fixed-latent grammar expected in the archive. Defaults to HLM1 compatibility.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allowed_codec == "hlm1":
        manifest = prove_pr106_hlm1_runtime_consumption(
            archive_path=args.archive,
            runtime_dir=args.runtime_dir,
            repo_root=REPO_ROOT,
        )
        claim_key = "runtime_hlm1_decode_consumption_claim"
    else:
        manifest = prove_pr106_hlm_runtime_consumption(
            archive_path=args.archive,
            runtime_dir=args.runtime_dir,
            repo_root=REPO_ROOT,
            allowed_codecs=("hlm2",),
        )
        claim_key = "runtime_hlm_decode_consumption_claim"
    if args.output_json:
        write_json(args.output_json, manifest)
    sys.stdout.write(dumps_hlm1_runtime_consumption_manifest(manifest))
    return 0 if manifest.get(claim_key) else 2


if __name__ == "__main__":
    raise SystemExit(main())
