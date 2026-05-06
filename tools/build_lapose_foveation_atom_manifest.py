#!/usr/bin/env python3
"""Build planning-only LA-POSE foveation transport atom rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.lapose_foveation_atoms import (  # noqa: E402
    build_foveation_transport_atom_manifest,
    records_from_json_payload,
)
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def _parse_pair(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must be 'x,y'")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be numeric") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--base-pose-dist", type=float, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frame-width", type=int, default=512)
    parser.add_argument("--frame-height", type=int, default=384)
    parser.add_argument("--foveal-center", type=_parse_pair, default=(256.0, 174.0))
    parser.add_argument("--center-gain", type=_parse_pair, default=(18.0, 10.0))
    parser.add_argument("--scalar-bytes", type=int, default=2)
    parser.add_argument("--pair-index-bytes", type=int, default=2)
    parser.add_argument("--opcode-bytes", type=int, default=1)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    manifest = build_foveation_transport_atom_manifest(
        records_from_json_payload(read_json(args.records_json)),
        base_pose_dist=args.base_pose_dist,
        source=args.source,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        foveal_center=args.foveal_center,
        center_gain=args.center_gain,
        scalar_bytes=args.scalar_bytes,
        pair_index_bytes=args.pair_index_bytes,
        opcode_bytes=args.opcode_bytes,
        max_atoms=args.max_atoms,
    )
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.records_json],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
