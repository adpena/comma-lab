#!/usr/bin/env python3
"""Build a monolithic replacement manifest from materialized CodecOp bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.monolithic_codec_op_replacement import (  # noqa: E402
    MonolithicCodecOpReplacementError,
    build_monolithic_codec_op_replacement_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--target-section", required=True)
    parser.add_argument(
        "--replacement-payload",
        type=Path,
        help=(
            "Raw replacement section payload. Optional when --evidence-json "
            "contains materialized_payload_path, replacement_payload_path, "
            "payload_path, or blob_path."
        ),
    )
    parser.add_argument("--output-replacement-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--section-payload-contract",
        default="raw_section_bytes",
        choices=[
            "raw_section_bytes",
            "pr106_brotli_section",
            "pr106_decoder_packed_brotli",
            "pr106_latents_and_sidecar_brotli",
        ],
    )
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--expected-source-archive-sha256")
    parser.add_argument("--expected-source-archive-bytes", type=int)
    args = parser.parse_args(argv)

    try:
        replacement_payload = args.replacement_payload or _payload_path_from_evidence_json(
            args.evidence_json
        )
        manifest = build_monolithic_codec_op_replacement_manifest(
            source_archive=args.source_archive,
            target_section=args.target_section,
            replacement_payload=replacement_payload,
            output_replacement_manifest=args.output_replacement_manifest,
            candidate_id=args.candidate_id,
            section_payload_contract=args.section_payload_contract,
            evidence_json=args.evidence_json,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            expected_source_archive_bytes=args.expected_source_archive_bytes,
        )
    except (
        MonolithicCodecOpReplacementError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"monolithic CodecOp replacement manifest failed: {exc}") from None

    replacement = manifest["replacements"][0]
    payload = manifest["replacement_payload"]
    print(
        f"wrote {args.output_replacement_manifest} for {replacement['section_name']}: "
        f"{replacement['expected_old_bytes']} -> {replacement['expected_new_bytes']} bytes "
        f"(delta {payload['byte_delta']})"
    )
    print(f"replacement sha256={replacement['expected_new_sha256']}")
    return 0


def _payload_path_from_evidence_json(path: Path | None) -> Path:
    if path is None:
        raise MonolithicCodecOpReplacementError(
            "--replacement-payload is required unless --evidence-json names a payload path"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonolithicCodecOpReplacementError("evidence_json must contain an object")
    for key in (
        "replacement_payload_path",
        "materialized_payload_path",
        "payload_path",
        "blob_path",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            return resolved
    raise MonolithicCodecOpReplacementError(
        "--replacement-payload is required; evidence_json has no materialized payload path"
    )


if __name__ == "__main__":
    raise SystemExit(main())
