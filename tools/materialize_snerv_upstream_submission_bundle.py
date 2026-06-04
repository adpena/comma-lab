#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an upstream-shaped SNeRV submission bundle.

The upstream evaluator charges ``submission_dir/archive.zip`` and executes
``submission_dir/inflate.sh`` outside the archive. This tool keeps the runtime
source outside the charged ZIP, writes a data-only archive containing ``x`` by
default, and can run an upstream-shaped receiver proof.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.archive_bound_candidate_runtime_bridge import (  # noqa: E402
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file, tree_sha256, write_json  # noqa: E402
from tac.submission_archive import (  # noqa: E402
    MINIMAL_SINGLE_MEMBER_NAME,
    safe_extract_zip,
    validate_archive_member_name,
    write_minimal_single_member_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    SNERV_RECEIVER_PROOF_SCHEMA,
    expected_receiver_output_bytes_from_metadata,
)

SCHEMA = "snerv_upstream_submission_bundle_materialization.v1"
AXIS_TAG = "[upstream-shaped-receiver-proof:false-authority]"

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

EXCLUDED_RUNTIME_NAMES = {
    "0.bin",
    "archive.zip",
    "archive_bound_candidate_adapter_package.json",
}
EXCLUDED_RUNTIME_DIRS = {"archive", "inflated", "receiver_proof", "__pycache__"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = materialize_upstream_submission_bundle(
        source_submission_dir=args.source_submission_dir,
        output_submission_dir=args.output_submission_dir,
        packet=args.packet,
        source_archive_zip=args.source_archive_zip,
        archive_member_name=args.archive_member_name,
        run_receiver_proof=bool(args.run_receiver_proof),
        retain_receiver_output=bool(args.retain_receiver_output),
        receiver_proof_timeout_seconds=int(args.receiver_proof_timeout_seconds),
        expected_receiver_output_bytes=args.expected_receiver_output_bytes,
        generated_utc=datetime.now(UTC).isoformat(),
    )
    write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": args.output_json.as_posix(),
                "output_submission_dir": args.output_submission_dir.as_posix(),
                "archive_zip_bytes": report["archive_zip"]["bytes"],
                "source_archive_zip_bytes": report["source_archive_zip"].get("bytes"),
                "archive_zip_delta_vs_source": report["archive_zip"].get(
                    "delta_bytes_vs_source_archive_zip"
                ),
                "receiver_proof_passed": report["receiver_proof"][
                    "runtime_consumption_proof_passed"
                ],
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def materialize_upstream_submission_bundle(
    *,
    source_submission_dir: str | Path,
    output_submission_dir: str | Path,
    packet: str | Path | None = None,
    source_archive_zip: str | Path | None = None,
    archive_member_name: str = MINIMAL_SINGLE_MEMBER_NAME,
    run_receiver_proof: bool,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    expected_receiver_output_bytes: int | None = None,
    generated_utc: str,
) -> dict[str, Any]:
    source_submission = Path(source_submission_dir).expanduser().resolve(strict=False)
    output_submission = Path(output_submission_dir).expanduser().resolve(strict=False)
    if not source_submission.is_dir():
        raise FileNotFoundError(f"source submission dir not found: {source_submission}")
    packet_path = _resolve_packet_path(source_submission, packet)
    source_archive_path = _resolve_source_archive_path(source_submission, source_archive_zip)
    packet_bytes = packet_path.read_bytes()
    expected_output_bytes = _expected_output_bytes(
        packet_bytes,
        override=expected_receiver_output_bytes,
    )
    archive_member_name = validate_archive_member_name(str(archive_member_name))

    if output_submission.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_submission}")
    output_submission.mkdir(parents=True)
    runtime_rows = _copy_external_runtime(
        source_submission=source_submission,
        output_submission=output_submission,
    )
    archive_zip = output_submission / "archive.zip"
    archive_build = write_minimal_single_member_archive(
        archive_zip,
        packet_bytes,
        member_name=archive_member_name,
    )
    receiver_proof = _receiver_proof_stub()
    if run_receiver_proof:
        receiver_proof = _run_upstream_shaped_receiver_proof(
            archive_zip=archive_zip,
            submission_dir=output_submission,
            expected_receiver_output_bytes=expected_output_bytes,
            retain_receiver_output=retain_receiver_output,
            timeout_seconds=receiver_proof_timeout_seconds,
        )

    source_archive = _source_archive_row(source_archive_path)
    archive_delta = (
        None
        if source_archive.get("bytes") is None
        else int(archive_zip.stat().st_size) - int(source_archive["bytes"])
    )
    blockers = _blockers(receiver_proof, run_receiver_proof=run_receiver_proof)
    report = {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "generated_utc": generated_utc,
        "operation": "snerv_upstream_data_only_archive_external_runtime_bundle",
        "upstream_contest_contract": {
            "rate_uses_submission_archive_zip_stat_only": True,
            "inflate_sh_runs_from_submission_dir": True,
            "archive_zip_unzipped_before_inflate": True,
            "runtime_source_outside_archive_zip": True,
            "archive_zip_payload_only": True,
            "archive_member_name_minimized": archive_member_name
            == MINIMAL_SINGLE_MEMBER_NAME,
            "sources_checked": [
                "upstream/README.md",
                "upstream/evaluate.py",
                "upstream/evaluate.sh",
                "upstream/.github/workflows/eval.yml",
            ],
        },
        "source_submission_dir": source_submission.as_posix(),
        "output_submission_dir": output_submission.as_posix(),
        "source_packet": {
            "path": packet_path.as_posix(),
            "bytes": len(packet_bytes),
            "sha256": sha256_file(packet_path),
        },
        "source_archive_zip": source_archive,
        "archive_zip": {
            **archive_build,
            "path": archive_zip.as_posix(),
            "data_only": True,
            "delta_bytes_vs_source_archive_zip": archive_delta,
            "score_authority": "false_until_exact_upstream_evaluate_replay",
        },
        "external_runtime": {
            "tree_sha256": tree_sha256(output_submission),
            "runtime_file_count": len(runtime_rows),
            "runtime_source_bytes": sum(int(row["bytes"]) for row in runtime_rows),
            "runtime_files": runtime_rows,
            "contains_unarchived_payload_packet": False,
            "runtime_source_minified": False,
            "runtime_source_rate_charged_by_upstream_evaluate_py": False,
            "identifier_renaming_applied": False,
        },
        "internal_rule_faithful_estimate": {
            "archive_zip_plus_external_runtime_source_bytes": int(
                archive_zip.stat().st_size
            )
            + sum(int(row["bytes"]) for row in runtime_rows),
            "note": (
                "This is a conservative local accounting view. Upstream "
                "evaluate.py score rate uses archive.zip bytes only."
            ),
        },
        "receiver_proof": receiver_proof,
        "launchability": {
            "candidate_package_launchable": False,
            "blocked_long_training_rows_must_not_launch": True,
            "reason": "exact paired contest eval and final compliance gate are missing",
        },
        "blockers": blockers,
        "next_actions": [
            "run_upstream_evaluate_sh_on_materialized_submission_dir",
            "run_paired_contest_cpu_cuda_auth_eval_before_score_claim",
            "optionally_minify_external_runtime_for_internal_rule_faithful_view_only",
        ],
        **FALSE_AUTHORITY,
    }
    return report


def _copy_external_runtime(
    *,
    source_submission: Path,
    output_submission: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in sorted(source_submission.rglob("*")):
        rel = src.relative_to(source_submission)
        if src.is_dir():
            continue
        if _excluded_runtime_relpath(rel):
            continue
        dst = output_submission / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append(
            {
                "path": rel.as_posix(),
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )
    required = ("inflate.sh", "inflate.py")
    missing = [name for name in required if not (output_submission / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"external runtime missing required files after copy: {missing}"
        )
    return rows


def _excluded_runtime_relpath(rel: Path) -> bool:
    if rel.name in EXCLUDED_RUNTIME_NAMES:
        return True
    return any(part in EXCLUDED_RUNTIME_DIRS for part in rel.parts)


def _run_upstream_shaped_receiver_proof(
    *,
    archive_zip: Path,
    submission_dir: Path,
    expected_receiver_output_bytes: int | None,
    retain_receiver_output: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    archive_dir = submission_dir / "receiver_proof" / "archive_extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_zip(archive_zip, archive_dir)
    return run_generated_inflate_receiver_proof(
        archive_zip_path=archive_zip,
        archive_sha256=sha256_file(archive_zip),
        archive_bytes=archive_zip.stat().st_size,
        submission_dir=submission_dir,
        archive_dir_for_inflate=archive_dir,
        output_dir=submission_dir,
        repo_root=REPO_ROOT,
        proof_schema=SNERV_RECEIVER_PROOF_SCHEMA,
        proof_filename="snerv_upstream_submission_receiver_proof.json",
        candidate_label="snerv_upstream_submission_bundle",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=expected_receiver_output_bytes,
        retain_receiver_output=retain_receiver_output,
        timeout_seconds=timeout_seconds,
    )


def _expected_output_bytes(packet_bytes: bytes, *, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    try:
        decoded = unpack_snerv_archive(packet_bytes)
        return expected_receiver_output_bytes_from_metadata(decoded.metadata)
    except Exception:
        return None


def _resolve_packet_path(source_submission: Path, packet: str | Path | None) -> Path:
    if packet is not None:
        return Path(packet).expanduser().resolve(strict=False)
    candidate = source_submission / "0.bin"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"packet not supplied and source submission has no 0.bin: {source_submission}"
    )


def _resolve_source_archive_path(
    source_submission: Path,
    source_archive_zip: str | Path | None,
) -> Path | None:
    if source_archive_zip is not None:
        return Path(source_archive_zip).expanduser().resolve(strict=False)
    candidate = source_submission.parent / "archive.zip"
    if candidate.is_file():
        return candidate
    candidate = source_submission / "archive.zip"
    return candidate if candidate.is_file() else None


def _source_archive_row(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": None if path is None else path.as_posix(), "bytes": None}
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _receiver_proof_stub() -> dict[str, Any]:
    return {
        "present": False,
        "runtime_consumption_proof_passed": False,
        "receiver_contract_satisfied": False,
        "blockers": ["snerv_upstream_submission_receiver_proof_not_requested"],
    }


def _blockers(
    receiver_proof: Mapping[str, Any],
    *,
    run_receiver_proof: bool,
) -> list[str]:
    blockers = [
        "full_video_scorer_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "pre_submission_compliance_gate_missing",
    ]
    if not run_receiver_proof:
        blockers.append("snerv_upstream_submission_receiver_proof_not_requested")
    elif receiver_proof.get("runtime_consumption_proof_passed") is not True:
        blockers.append("snerv_upstream_submission_receiver_proof_failed")
    if receiver_proof.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_upstream_submission_receiver_contract_missing")
    return _dedupe([*blockers, *list(receiver_proof.get("blockers") or ())])


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output-submission-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--source-archive-zip", type=Path)
    parser.add_argument(
        "--archive-member-name",
        default=MINIMAL_SINGLE_MEMBER_NAME,
        help=(
            "Charged archive.zip member name. Default 'x' minimizes ZIP name "
            "overhead; runtime must consume this member or receiver proof fails."
        ),
    )
    parser.add_argument("--run-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument("--receiver-proof-timeout-seconds", type=int, default=1800)
    parser.add_argument("--expected-receiver-output-bytes", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
