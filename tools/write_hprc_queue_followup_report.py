#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the queue-owned HPRC replay/Z8 follow-up contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import ArtifactWriteError  # noqa: E402
from tac.substrates.hprc.campaign import (  # noqa: E402
    build_hprc_queue_followup_report,
    write_hprc_queue_followup_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", required=True, type=Path)
    parser.add_argument("--decode-pairs", required=True, type=int)
    parser.add_argument("--full-replay-min-pairs", type=int, default=600)
    parser.add_argument("--local-replay-summary-json", type=Path)
    parser.add_argument("--mlx-prefilter-gate-json", type=Path)
    parser.add_argument("--exact-auth-gate-json", type=Path)
    parser.add_argument("--z8-archive-bin", type=Path)
    parser.add_argument("--z8-surface", type=Path)
    parser.add_argument("--z8-reference-pairs-npy", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_hprc_queue_followup_report(
        training_result_path=args.training_result,
        decode_pairs=int(args.decode_pairs),
        full_replay_min_pairs=int(args.full_replay_min_pairs),
        local_replay_summary_path=args.local_replay_summary_json,
        mlx_prefilter_gate_path=args.mlx_prefilter_gate_json,
        exact_auth_gate_path=args.exact_auth_gate_json,
        z8_archive_bin_path=args.z8_archive_bin,
        z8_surface_path=args.z8_surface,
        z8_reference_pairs_npy_path=args.z8_reference_pairs_npy,
        repo_root=args.repo_root,
    )
    path = write_hprc_queue_followup_report(
        output_path=args.out_json,
        report=report,
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(json.dumps({**report, "report_path": path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, ValueError, FileNotFoundError) as exc:
        print(f"write_hprc_queue_followup_report failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
