#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sweep compact receiver decoder codecs from an existing archive.zip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates._shared.compact_decoder_codec_sweep import (  # noqa: E402
    SUPPORTED_COMPACT_DECODER_CODECS,
    sweep_compact_decoder_codecs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--family",
        choices=("auto", "pact_nerv_vq", "pact_nerv_selector_v4", "hi_nerv"),
        default="auto",
    )
    parser.add_argument(
        "--decoder-codec",
        action="append",
        choices=SUPPORTED_COMPACT_DECODER_CODECS,
        dest="decoder_codecs",
        help=(
            "Decoder codec to materialize. May be repeated. Defaults to the "
            "full compact decoder codec portfolio."
        ),
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument(
        "--skip-receiver-proof",
        action="store_true",
        help="Materialize bytes only; leaves variants fail-closed.",
    )
    parser.add_argument(
        "--retain-receiver-proof-output",
        action="store_true",
        help="Keep inflated raw proof output instead of proof-and-delete.",
    )
    parser.add_argument("--receiver-proof-timeout-seconds", default=1800, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = sweep_compact_decoder_codecs(
        source_archive_zip=args.source_archive_zip,
        output_dir=args.output_dir,
        decoder_codecs=tuple(args.decoder_codecs or SUPPORTED_COMPACT_DECODER_CODECS),
        family=args.family,
        repo_root=args.repo_root,
        run_receiver_proof=not args.skip_receiver_proof,
        retain_receiver_proof_output=args.retain_receiver_proof_output,
        receiver_proof_timeout_seconds=args.receiver_proof_timeout_seconds,
        allow_overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "family": report["family"],
                "best_variant": report["best_variant"],
                "report_path": report["report_path"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
