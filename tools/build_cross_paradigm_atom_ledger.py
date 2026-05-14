#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only cross-paradigm meta-Lagrangian atom ledger."""

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

from tac.optimization.cross_paradigm_atoms import (  # noqa: E402
    atoms_from_adapter_payload,
    build_cross_paradigm_atom_ledger,
    normalize_cross_paradigm_atom,
)
from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pose-dist", type=float, required=True)
    parser.add_argument("--source", default="cross_paradigm_manual")
    parser.add_argument("--atoms-json", action="append", type=Path, default=[])
    parser.add_argument("--hnerv-rate-recode-profile", action="append", type=Path, default=[])
    parser.add_argument("--wr01-wavelet-plan", action="append", type=Path, default=[])
    parser.add_argument("--categorical-mask-plan", action="append", type=Path, default=[])
    parser.add_argument("--lapose-plan", action="append", type=Path, default=[])
    parser.add_argument("--foveation-plan", action="append", type=Path, default=[])
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    atoms: list[dict] = []
    input_paths: list[Path] = []

    for path in args.atoms_json:
        payload = read_json(path)
        if not isinstance(payload, list):
            raise SystemExit("--atoms-json must contain a JSON list")
        atoms.extend(normalize_cross_paradigm_atom(atom) for atom in payload)
        input_paths.append(path)

    adapter_inputs = [
        ("hnerv_rate_recode_profile", args.hnerv_rate_recode_profile),
        ("wr01_wavelet_plan", args.wr01_wavelet_plan),
        ("categorical_openpilot_mask", args.categorical_mask_plan),
        ("lapose_plan", args.lapose_plan),
        ("foveation_plan", args.foveation_plan),
    ]
    for adapter, paths in adapter_inputs:
        for path in paths:
            payload = read_json(path)
            if not isinstance(payload, dict):
                raise SystemExit(f"{path}: adapter inputs must contain a JSON object")
            atoms.extend(
                atoms_from_adapter_payload(
                    adapter,
                    payload,
                    evidence_source_path=path.as_posix(),
                    evidence_source_sha256=sha256_file(path),
                )
            )
            input_paths.append(path)

    if not atoms:
        raise SystemExit("provide at least one atom source")

    ledger = build_cross_paradigm_atom_ledger(
        atoms,
        base_pose_dist=args.base_pose_dist,
        source=args.source,
    )
    ledger = attach_tool_run_manifest(
        ledger,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(ledger)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
