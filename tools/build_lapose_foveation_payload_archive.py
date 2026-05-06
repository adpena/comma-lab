#!/usr/bin/env python3
"""Build a byte-closed local LFV1 foveation payload archive.

The output is a local fail-closed archive/readiness artifact only. It never
dispatches GPU work and never claims score readiness.
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

from tac.analysis.lapose_foveation_payload import PAYLOAD_MEMBER  # noqa: E402
from tac.lapose_foveation_payload_candidate import (  # noqa: E402
    LaposeFoveationPayloadCandidateError,
    build_lapose_foveation_payload_archive_candidate,
)
from tac.repo_io import json_text, read_json, sha256_bytes, sha256_file, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lfv1-payload", type=Path, required=True)
    parser.add_argument("--source-readiness-json", type=Path)
    parser.add_argument("--source-archive-sha256", default="")
    return parser.parse_args(argv)


def _payload_source(args: argparse.Namespace, payload: bytes) -> tuple[dict, list[Path]]:
    source = {
        "kind": "local_lfv1_payload_file",
        "path": args.lfv1_payload.as_posix(),
        "member": PAYLOAD_MEMBER,
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }
    input_paths = [args.lfv1_payload]
    if args.source_readiness_json is None:
        return source, input_paths

    readiness = read_json(args.source_readiness_json)
    if not isinstance(readiness, dict):
        raise LaposeFoveationPayloadCandidateError("--source-readiness-json must contain an object")
    if readiness.get("member") != PAYLOAD_MEMBER:
        raise LaposeFoveationPayloadCandidateError("source readiness does not describe LFV1 member")
    if readiness.get("sha256") != source["payload_sha256"]:
        raise LaposeFoveationPayloadCandidateError("source readiness SHA-256 does not match LFV1 payload")
    if readiness.get("bytes") != source["payload_bytes"]:
        raise LaposeFoveationPayloadCandidateError("source readiness byte count does not match LFV1 payload")
    source.update(
        {
            "kind": "local_lfv1_payload_readiness",
            "source_readiness_path": args.source_readiness_json.as_posix(),
            "source_readiness_sha256": sha256_file(args.source_readiness_json),
            "source_schema": str(readiness.get("schema") or ""),
            "source_ready_for_exact_eval_dispatch": bool(
                readiness.get("ready_for_exact_eval_dispatch")
            ),
        }
    )
    input_paths.append(args.source_readiness_json)
    return source, input_paths


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    try:
        payload = args.lfv1_payload.read_bytes()
        payload_source, input_paths = _payload_source(args, payload)
        result = build_lapose_foveation_payload_archive_candidate(
            out_dir=args.out_dir,
            lfv1_payload=payload,
            payload_source=payload_source,
            repo_root=REPO_ROOT,
            source_archive_sha256=args.source_archive_sha256,
        )
    except (OSError, LaposeFoveationPayloadCandidateError, ValueError) as exc:
        print(f"lapose foveation payload archive build failed: {exc}", file=sys.stderr)
        return 2

    summary_path = args.out_dir / "summary.json"
    archive_path = args.out_dir / "archive.zip"
    candidate_path = args.out_dir / "candidate.json"
    readiness_path = args.out_dir / "readiness.json"
    manifest_path = args.out_dir / "archive_member_manifest.json"
    readiness = attach_tool_run_manifest(
        result["readiness"],
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[
            *[path for path in input_paths if path.is_file()],
            archive_path,
            candidate_path,
            manifest_path,
        ],
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
            *[path for path in input_paths if path.is_file()],
            archive_path,
            candidate_path,
            readiness_path,
            manifest_path,
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
