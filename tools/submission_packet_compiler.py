#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI for the deterministic submission-packet compiler oracle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.submission_packet_compiler import (  # noqa: E402
    MODES,
    TARGET_PROFILES,
    PacketCompilerError,
    compile_packet,
    write_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="packet directory or archive.zip")
    parser.add_argument("--mode", choices=MODES, default="inspect")
    parser.add_argument("--target-profile", choices=TARGET_PROFILES, default="contest_one_video_replay")
    parser.add_argument("--output-dir", type=Path, default=None, help="required for identity mode")
    parser.add_argument("--json-out", type=Path, default=None, help="write manifest JSON here")
    parser.add_argument(
        "--zipwire-bin",
        type=Path,
        default=None,
        help="optional native zipwire executable for explicit ZIP conformance comparison",
    )
    args = parser.parse_args(argv)

    try:
        manifest = compile_packet(
            args.packet,
            mode=args.mode,
            target_profile=args.target_profile,
            output_dir=args.output_dir,
            zipwire_bin=args.zipwire_bin,
        )
    except PacketCompilerError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        write_manifest(manifest, args.json_out)
    blockers = manifest["contest_compliance"]["blockers"]
    print(
        f"[packet-compiler] mode={args.mode} target={args.target_profile} "
        f"blockers={len(blockers)} ready_for_exact_eval_dispatch=false"
    )
    if blockers:
        for blocker in blockers[:20]:
            print(f"  - {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
