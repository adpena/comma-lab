#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay public PR86 HPAC tokens locally and fail closed on byte-parity gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr86_hpac_codec import (  # noqa: E402
    DEFAULT_PR86_ARCHIVE,
    DEFAULT_PR86_MERGED_SOURCE_DIR,
    DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT,
    Pr86ArchiveContract,
    default_source_artifact_paths,
    run_pr86_hpac_probability_variant_matrix,
    run_pr86_hpac_replay,
    supported_hpac_probability_variant_names,
)


_WORKER_VOLATILE_REPORT_KEYS = frozenset({"recorded_at_utc", "elapsed_sec"})


def _deterministic_worker_report(value):
    if isinstance(value, dict):
        return {
            key: _deterministic_worker_report(item)
            for key, item in value.items()
            if key not in _WORKER_VOLATILE_REPORT_KEYS
        }
    if isinstance(value, list):
        return [_deterministic_worker_report(item) for item in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR86_ARCHIVE)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_PR86_MERGED_SOURCE_DIR,
        help="Current merged PR86 source intake directory used for contract classification.",
    )
    parser.add_argument(
        "--no-current-source-context",
        action="store_true",
        help="Skip current merged-source contract classification. Intended only for synthetic fixtures.",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        action="append",
        default=None,
        help="Replay intake JSON artifact. Defaults to the four captured PR86 intake JSONs.",
    )
    parser.add_argument(
        "--no-source-artifacts",
        action="store_true",
        help="Do not load default intake JSONs. Intended only for synthetic local fixtures.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Local debugging cap. Omit for the full 600-frame dispatch gate.",
    )
    parser.add_argument(
        "--probability-variant",
        action="append",
        choices=supported_hpac_probability_variant_names(),
        default=None,
        help=(
            "HPAC probability/categorical contract to probe. May be repeated; "
            "multiple values produce a variant-matrix report."
        ),
    )
    parser.add_argument(
        "--all-probability-variants",
        action="store_true",
        help="Run the source contract plus all named off-contract probability probes.",
    )
    parser.add_argument(
        "--allow-non-pr86-archive",
        action="store_true",
        help="Disable PR86 archive SHA/size/member-size constants for synthetic local fixtures.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path. JSON is also printed to stdout.",
    )
    parser.add_argument(
        "--worker-probability-contract-json-out",
        action="store_true",
        help=(
            "Write to the fixed PR86 probability-contract worker artifact path: "
            f"{DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.no_source_artifacts:
        artifacts: tuple[Path, ...] = ()
    else:
        artifacts = tuple(args.source_artifact) if args.source_artifact else default_source_artifact_paths()

    contract = Pr86ArchiveContract()
    if args.allow_non_pr86_archive:
        contract = Pr86ArchiveContract(
            expected_archive_bytes=None,
            expected_archive_sha256=None,
            expected_member_bytes={},
            expected_tokens_sha256=None,
        )

    if args.all_probability_variants:
        variants = supported_hpac_probability_variant_names()
    else:
        variants = tuple(args.probability_variant or ["source_float64_perfect_false"])

    if len(variants) == 1:
        report = run_pr86_hpac_replay(
            archive=args.archive,
            contract=contract,
            source_dir=None if args.no_current_source_context else args.source_dir,
            source_artifacts=artifacts,
            device="cpu",
            max_frames=args.max_frames,
            attempt_reencode=True,
            probability_variant=variants[0],
        )
    else:
        report = run_pr86_hpac_probability_variant_matrix(
            archive=args.archive,
            variants=variants,
            contract=contract,
            source_dir=None if args.no_current_source_context else args.source_dir,
            source_artifacts=artifacts,
            device="cpu",
            max_frames=args.max_frames,
            attempt_reencode=True,
        )
    json_out = DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT if args.worker_probability_contract_json_out else args.json_out
    if args.worker_probability_contract_json_out:
        report = _deterministic_worker_report(report)
        report["deterministic_worker_artifact"] = True
        report["volatile_fields_omitted"] = sorted(_WORKER_VOLATILE_REPORT_KEYS)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
