#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a minimal HPRC archive-bound candidate.

This is the first runnable HPRC receiver scaffold. It is intentionally
false-authority: the packet proves the archive/runtime/receiver contract, not a
trained RNeRV/PACT-NeRV score result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.substrates.hprc.archive import HprcPacketConfig
from tac.substrates.hprc.archive_candidate import (
    build_minimal_hprc_v0_packet,
    export_hprc_archive_bytes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--decoder-family-id", type=int, default=95)
    parser.add_argument("--retain-receiver-output", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    packet = build_minimal_hprc_v0_packet(
        config=HprcPacketConfig(decoder_family_id=int(args.decoder_family_id)),
        decoder_family_id=int(args.decoder_family_id),
    )
    archive_zip_path, archive_sha256, archive_bytes = export_hprc_archive_bytes(
        packet,
        args.output_dir,
        repo_root=args.repo_root,
        retain_receiver_proof_output=bool(args.retain_receiver_output),
        mlx_triage_argv=sys.argv,
    )
    result = {
        "schema": "hprc_minimal_candidate_materialization_result.v1",
        "archive_zip_path": archive_zip_path.as_posix(),
        "archive_zip_sha256": archive_sha256,
        "archive_zip_bytes": int(archive_bytes),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_axis_blocker": "contest_cpu_cuda_exact_eval_not_executed",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
