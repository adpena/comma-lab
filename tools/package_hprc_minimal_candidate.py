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

from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY, HPRC_RECEIVER_PROOF_SCRATCH_BYTES
from tac.substrates.hprc.campaign import materialize_minimal_hprc_campaign

HPRC_STORAGE_PLAN_SCHEMA = "hprc_campaign_storage_plan.v1"
DEFAULT_HPRC_WORKLOAD_SUBDIR = "experiments/results/hprc_campaign"
DEFAULT_HPRC_STORAGE_EXPECTED_BYTES = HPRC_RECEIVER_PROOF_SCRATCH_BYTES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. If omitted, choose the first eligible external "
            "SSD tier from the operator storage waterfall."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--decoder-family-id", type=int, default=95)
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--storage-tier", action="append", default=[], help="name=/path storage tier override")
    parser.add_argument("--storage-workload-subdir", default=DEFAULT_HPRC_WORKLOAD_SUBDIR)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--storage-expected-bytes", type=int, default=DEFAULT_HPRC_STORAGE_EXPECTED_BYTES)
    parser.add_argument(
        "--allow-local-output-dir",
        action="store_true",
        help="Allow local-disk fallback only by explicit opt-in.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir, storage_plan_path = _resolve_output_dir(args)
    result = materialize_minimal_hprc_campaign(
        repo_root=args.repo_root,
        output_dir=output_dir,
        run_id=args.run_id,
        decoder_family_id=int(args.decoder_family_id),
        retain_receiver_output=bool(args.retain_receiver_output),
        storage_plan_path=storage_plan_path,
        mlx_triage_argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
    )
    result = {
        "schema": "hprc_minimal_candidate_materialization_result.v1",
        **result.to_dict(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_axis_blocker": "contest_cpu_cuda_exact_eval_not_executed",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _resolve_output_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        return output_dir, None

    tiers = parse_storage_tier_specs(
        list(args.storage_tier),
        repo_root=repo_root,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=bool(args.allow_local_output_dir),
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=str(args.storage_workload_subdir),
        requested_bytes=int(args.storage_expected_bytes),
        create=True,
    )
    workload_root = require_selected_storage(plan)
    run_id = args.run_id or _utc_stamp()
    output_dir = workload_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_plan_path = output_dir / "hprc_storage_plan.json"
    _write_json_maybe_overwrite(
        storage_plan_path,
        {
            "schema": HPRC_STORAGE_PLAN_SCHEMA,
            "storage_plan": plan.to_dict(),
            "selected_campaign_output_dir": output_dir.as_posix(),
            **FALSE_AUTHORITY,
        },
    )
    return output_dir, storage_plan_path


def _write_json_maybe_overwrite(path: Path, payload: object) -> None:
    expected_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=expected_sha is not None,
        expected_existing_sha256=expected_sha,
    )


def _utc_stamp() -> str:
    import time

    return time.strftime("hprc_campaign_%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, StorageTierError) as exc:
        print(f"package_hprc_minimal_candidate failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
