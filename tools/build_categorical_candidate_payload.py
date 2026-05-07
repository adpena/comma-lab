#!/usr/bin/env python3
"""Build a byte-closed local categorical payload candidate.

The output is a local archive/readiness artifact only. It never dispatches GPU
work and always remains blocked until decode/re-encode parity and runtime
output parity are proven.
"""

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

from tac.categorical_payload_candidate import (  # noqa: E402
    CategoricalPayloadCandidateError,
    RUNTIME_EXECUTION_PROOF_FILENAME,
    build_categorical_payload_candidate,
    extract_pr91_hpm1_categorical_payload,
)
from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE  # noqa: E402
from tac.repo_io import json_text, sha256_bytes, sha256_file, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--payload-source",
        choices=("pr91-hpm1-mask", "file"),
        default="pr91-hpm1-mask",
    )
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--categorical-payload", type=Path)
    parser.add_argument("--source-archive-sha256")
    return parser.parse_args(argv)


def _load_payload(args: argparse.Namespace) -> tuple[bytes, dict, str, list[Path]]:
    if args.payload_source == "pr91-hpm1-mask":
        payload, payload_source = extract_pr91_hpm1_categorical_payload(args.source_archive)
        return payload, payload_source, str(payload_source["source_archive_sha256"]), [args.source_archive]
    if args.categorical_payload is None:
        raise CategoricalPayloadCandidateError("--categorical-payload is required with --payload-source=file")
    if args.source_archive_sha256 is None:
        raise CategoricalPayloadCandidateError("--source-archive-sha256 is required with --payload-source=file")
    payload = args.categorical_payload.read_bytes()
    return (
        payload,
        {
            "kind": "local_payload_file",
            "path": args.categorical_payload.as_posix(),
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
        },
        args.source_archive_sha256,
        [args.categorical_payload],
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    try:
        payload, payload_source, source_archive_sha256, input_paths = _load_payload(args)
        result = build_categorical_payload_candidate(
            out_dir=args.out_dir,
            categorical_payload=payload,
            payload_source=payload_source,
            repo_root=REPO_ROOT,
            source_archive_sha256=source_archive_sha256,
        )
    except CategoricalPayloadCandidateError as exc:
        print(f"categorical candidate build failed: {exc}", file=sys.stderr)
        return 2

    summary_path = args.out_dir / "summary.json"
    archive_path = args.out_dir / "archive.zip"
    candidate_path = args.out_dir / "candidate.json"
    readiness_path = args.out_dir / "readiness.json"
    manifest_path = args.out_dir / "archive_member_manifest.json"
    hpm1_structural_inventory_path = args.out_dir / "hpm1_structural_inventory.json"
    runtime_execution_proof_path = args.out_dir / RUNTIME_EXECUTION_PROOF_FILENAME
    generated_input_paths = [
        *[path for path in input_paths if path.is_file()],
        archive_path,
        candidate_path,
        manifest_path,
        *(
            [hpm1_structural_inventory_path]
            if hpm1_structural_inventory_path.is_file()
            else []
        ),
        *(
            [runtime_execution_proof_path]
            if runtime_execution_proof_path.is_file()
            else []
        ),
    ]
    readiness = attach_tool_run_manifest(
        result["readiness"],
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=generated_input_paths,
        repo_root=REPO_ROOT,
        output_path=readiness_path,
    )
    write_json(readiness_path, readiness)
    summary_payload = {
        **result["summary"],
        "readiness_blockers": readiness["dispatch_blockers"],
    }
    summary = attach_tool_run_manifest(
        summary_payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[
            *generated_input_paths,
            readiness_path,
        ],
        repo_root=REPO_ROOT,
        output_path=summary_path,
    )
    summary["archive_sha256"] = sha256_file(archive_path)
    summary["ready_for_exact_eval_dispatch"] = False
    write_json(summary_path, summary)
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
