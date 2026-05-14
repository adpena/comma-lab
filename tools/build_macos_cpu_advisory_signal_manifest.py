#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed macOS-CPU advisory-signal manifest.

This is the sanctioned path for using local macOS-CPU as a free first-class
advisory proxy. Per CLAUDE.md PR107 empirical calibration (M5 Max
``0.19664189`` matched GHA Linux x86_64 ``0.1966358879`` within ``6e-6``),
macOS-CPU is a high-fidelity proxy for the contest-CPU axis, but per
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable, it is NEVER 1:1 contest-compliant.

The output manifest carries ``score_claim=false``, ``promotion_eligible=false``,
and ``ready_for_exact_eval_dispatch=false``. It is consumed by the autopilot
ranker as a planning prior (``[macOS-CPU advisory]`` rows participate in
ranking only).

Operator routing 2026-05-13: "training is the real roadblock; we can prepare
and run things on macos and cpu". Cascade reframe: dev loop on macOS, deploy
to contest hardware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.macos_cpu_advisory_signal import (  # noqa: E402
    build_macos_cpu_advisory_signal_manifest,
    detect_macos_cpu_hardware_substrate,
    is_running_on_macos_arm64,
    json_text,
    load_calibration_model,
    load_observations,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="JSON or JSONL macOS-CPU observation rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Manifest JSON output path (refused under /tmp per CLAUDE.md)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run id for provenance",
    )
    parser.add_argument(
        "--source",
        help="Source label/path for the observation set",
    )
    parser.add_argument(
        "--hardware-substrate",
        default=None,
        help=(
            "Override the auto-detected hardware substrate string (default: "
            "auto-detected via sysctl machdep.cpu.brand_string). The validator "
            "refuses this substrate as macos_substrate by design."
        ),
    )
    parser.add_argument(
        "--calibration-model",
        type=Path,
        default=None,
        help=(
            "Path to a calibration_model.json (e.g. from the sister "
            "lane_macos_cpu_proxy_empirical_validation lane). When omitted "
            "the loader auto-discovers the latest matching file under "
            "experiments/results/ OR falls back to the PR107 placeholder."
        ),
    )
    parser.add_argument(
        "--allow-non-darwin",
        action="store_true",
        help=(
            "Skip the Darwin ARM64 platform check (intended for unit tests "
            "and CI replay paths only — production manifests should always "
            "run on the actual macOS host)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Fail-closed platform check. Per CLAUDE.md: the macOS-CPU tag refers
    # specifically to the Apple Silicon Darwin-ARM64 proxy axis. Running
    # the builder on Linux or x86_64 macOS produces malformed evidence.
    if not args.allow_non_darwin and not is_running_on_macos_arm64():
        print(
            "[macos-cpu-advisory] FATAL: this builder must run on Darwin ARM64. "
            "Use --allow-non-darwin only for unit tests or CI replay.",
            file=sys.stderr,
        )
        return 2

    observations = load_observations(args.observations)
    if not observations:
        print(
            "[macos-cpu-advisory] FATAL: observation file contained zero rows.",
            file=sys.stderr,
        )
        return 3

    if args.calibration_model is not None:
        try:
            import json as _json

            calibration_model = _json.loads(args.calibration_model.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(
                f"[macos-cpu-advisory] FATAL: could not load calibration model: {exc}",
                file=sys.stderr,
            )
            return 4
    else:
        # Auto-discover latest sister-subagent calibration model OR fall back
        # to the PR107 placeholder. Per CLAUDE.md "DEGRADE GRACEFULLY if it
        # doesn't exist yet" the placeholder is canonical until the sister
        # empirical-validation lane lands its output.
        calibration_model = load_calibration_model()

    substrate = args.hardware_substrate or detect_macos_cpu_hardware_substrate()

    manifest = build_macos_cpu_advisory_signal_manifest(
        observations,
        source=args.source or args.observations.as_posix(),
        run_id=args.run_id,
        hardware_substrate=substrate,
        calibration_model=calibration_model,
    )

    output_str = str(args.output)
    if (
        output_str.startswith("/tmp/")
        or "/private/tmp/" in output_str
        or "/var/tmp/" in output_str
    ):
        print(
            f"[macos-cpu-advisory] FATAL: refusing to write to /tmp path: {output_str!r}",
            file=sys.stderr,
        )
        return 5

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text(manifest), encoding="utf-8")

    print(f"[macos-cpu-advisory] wrote {args.output}")
    print(f"  schema = {manifest['schema']}")
    print(f"  evidence_grade = {manifest['evidence_grade']}")
    print(f"  evidence_tag = {manifest['evidence_tag']}")
    print(f"  hardware_substrate = {manifest['hardware_substrate']}")
    print(f"  rows = {manifest['row_count']}")
    print(f"  ranking_atoms = {len(manifest['ranking_atoms'])}")
    print(f"  promotion_eligible = {manifest['promotion_eligible']} (permanently False)")
    print(
        f"  calibration_status = {manifest['calibration_model'].get('calibration_status', '?')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
