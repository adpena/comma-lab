# SPDX-License-Identifier: MIT
"""Shared CLI adapter for deterministic packet-compiler aliases.

Operator-facing packet compiler tools must route through
``tac.packet_compiler.deterministic_compiler`` rather than importing the
lower-level submission-packet oracle directly.  This module keeps the legacy
``tools/{submission,contest}_packet_compiler.py`` entry points as thin aliases
without duplicating parser or dispatch logic.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from tac.packet_compiler.deterministic_compiler import (
    COMPILER_MODES,
    MANIFEST_NAME,
    TARGET_PROFILES,
    DeterministicPacketCompilerError,
    compile_packet,
    inspect_packet_oracle,
)

CLI_MODES: tuple[str, ...] = ("inspect", *COMPILER_MODES)


def build_arg_parser(*, label: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"{label}: canonical alias for tac.packet_compiler."
            "deterministic_compiler. Inspect mode emits byte-custody vectors; "
            "rewrite modes delegate to the deterministic compiler."
        ),
    )
    parser.add_argument("packet", type=Path, help="packet directory or archive.zip")
    parser.add_argument("--mode", choices=CLI_MODES, default="inspect")
    parser.add_argument(
        "--target-profile",
        choices=TARGET_PROFILES,
        default="contest_one_video_replay",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="required for identity/canonicalize/optimize modes",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="write manifest JSON here")
    parser.add_argument(
        "--zipwire-bin",
        type=Path,
        default=None,
        help="optional native zipwire executable for inspect-mode conformance comparison",
    )
    return parser


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_result_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(output_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeterministicPacketCompilerError(
            f"{manifest_path} did not decode to a JSON object"
        )
    return payload


def _shape_blockers(manifest: dict[str, Any]) -> list[str]:
    contest = manifest.get("contest_compliance")
    if isinstance(contest, dict) and isinstance(contest.get("blockers"), list):
        return [str(item) for item in contest["blockers"]]
    blockers = manifest.get("blockers")
    return [str(item) for item in blockers] if isinstance(blockers, list) else []


def _score_dispatch_blockers(manifest: dict[str, Any]) -> list[str]:
    gate = manifest.get("score_dispatch_gate")
    if isinstance(gate, dict) and isinstance(gate.get("blockers"), list):
        return [str(item) for item in gate["blockers"]]
    blockers = manifest.get("score_dispatch_blockers")
    return [str(item) for item in blockers] if isinstance(blockers, list) else []


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(manifest), encoding="utf-8")


def _print_summary(*, label: str, mode: str, target_profile: str, manifest: dict[str, Any]) -> None:
    shape_blockers = _shape_blockers(manifest)
    score_blockers = _score_dispatch_blockers(manifest)
    if label == "contest-packet-compiler":
        print(
            f"[{label}] mode={mode} target={target_profile} "
            f"shape_blockers={len(shape_blockers)} "
            f"score_dispatch_blockers={len(score_blockers)} "
            "score_claim=false dispatchable=false "
            "ready_for_exact_eval_dispatch=false"
        )
    else:
        print(
            f"[{label}] mode={mode} target={target_profile} "
            f"blockers={len(shape_blockers)} "
            "ready_for_exact_eval_dispatch=false"
        )
    for blocker in shape_blockers[:20]:
        print(f"  - {blocker}")


def run_packet_compiler_cli(
    argv: list[str] | None = None,
    *,
    label: str = "packet-compiler",
) -> int:
    args = build_arg_parser(label=label).parse_args(argv)

    try:
        if args.mode == "inspect":
            manifest = inspect_packet_oracle(
                args.packet,
                target_profile=args.target_profile,
                zipwire_bin=args.zipwire_bin,
            )
        else:
            if args.output_dir is None:
                raise DeterministicPacketCompilerError(
                    f"{args.mode} mode requires --output-dir"
                )
            if args.zipwire_bin is not None:
                raise DeterministicPacketCompilerError(
                    "--zipwire-bin is inspect-mode only"
                )
            result = compile_packet(
                input_packet=args.packet,
                output_dir=args.output_dir,
                mode=args.mode,
                target_profile=args.target_profile,
            )
            manifest = _load_result_manifest(result.output_dir) or dataclasses.asdict(result)
    except (DeterministicPacketCompilerError, OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        _write_manifest(manifest, args.json_out)
    _print_summary(
        label=label,
        mode=args.mode,
        target_profile=args.target_profile,
        manifest=manifest,
    )
    return 0


def main(argv: list[str] | None = None, *, label: str = "packet-compiler") -> int:
    return run_packet_compiler_cli(argv, label=label)


__all__ = [
    "CLI_MODES",
    "build_arg_parser",
    "main",
    "run_packet_compiler_cli",
]
