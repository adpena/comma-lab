#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a lossless per-tensor entropy-recode candidate from an FP11 packet.

Applies PR #112's per-tensor adaptive range coder + per-dim AR latent coder
(``tac.packet_compiler.ctx_range_coder``) to the decoder-weight and latent
sections of an FP11 (PR #101/#110/R3) frontier packet, replacing the split-Brotli
decoder blob and raw-LZMA1 latent blob. The recode is lossless by construction
(verified fail-closed), the FECa/FEC8 selector + DQS1 tail are preserved verbatim,
and the win is rate-only.

Emits:
  <output-dir>/submission_dir/archive.zip   (the recoded packet, single STORED member)
  <output-dir>/byte_closure_proof.json
  <output-dir>/noop_detector.json
  <output-dir>/recode_manifest.json

The submission_dir runtime (inflate.py + src/codec_ctx.py + R3 src/encoder) must be
staged by the caller (this tool only rewrites archive.zip + proofs). Use the
canonical submission_dir at
``experiments/results/pr110_payload_entropy_recode_20260610/submission_dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler import pr110_payload_entropy_recode as rec  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
        help="FP11 frontier archive.zip to recode (single STORED member).",
    )
    parser.add_argument(
        "--source-submission-dir",
        required=True,
        type=Path,
        help="submission_dir whose src/codec.py declares DECODER_BLOB_LEN.",
    )
    parser.add_argument(
        "--output-archive",
        required=True,
        type=Path,
        help="Path to write the recoded archive.zip.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for byte_closure_proof.json / noop_detector.json / recode_manifest.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_archive.exists() and not args.overwrite:
        print(f"FATAL: {args.output_archive} exists (use --overwrite)", file=sys.stderr)
        return 2

    try:
        member_name, source_member = rec.read_single_member(args.source_archive)
        decoder_blob_len = rec.infer_decoder_blob_len(args.source_submission_dir)
        result = rec.recode_fp11_member(source_member, decoder_blob_len=decoder_blob_len)
    except (rec.Pr110PayloadEntropyRecodeError, OSError, ValueError) as exc:
        print(f"FATAL: payload entropy recode failed: {exc}", file=sys.stderr)
        return 2

    rec.write_stored_archive(
        args.output_archive, member_name=member_name, payload=result.recoded_member
    )

    source_archive_bytes = args.source_archive.read_bytes()
    recoded_archive_bytes = args.output_archive.read_bytes()

    # no-op detector: prove the targeted byte spans actually changed and the recoded
    # member still parses + decodes to the same raw sections (consumption proof).
    noop = {
        "schema": "pr110_payload_entropy_recode_noop_detector.v1",
        "source_member_sha256": hashlib.sha256(source_member).hexdigest(),
        "recoded_member_sha256": hashlib.sha256(result.recoded_member).hexdigest(),
        "member_bytes_changed": result.recoded_member != source_member,
        "member_delta_bytes": result.member_delta_bytes,
        "decoder_section_changed": result.metrics["decoder_delta_bytes"] != 0,
        "latent_section_changed": result.metrics["latent_delta_bytes"] != 0,
        "selector_payload_preserved": True,  # asserted inside recode_fp11_member
        "dqs1_tail_preserved": True,  # asserted inside recode_fp11_member
        "decoder_raw_consumption_byte_identical": result.lossless_proof[
            "decoder_raw_byte_identical"
        ],
        "latent_raw_consumption_byte_identical": result.lossless_proof[
            "latent_raw_byte_identical"
        ],
        "sidecar_consumption_byte_identical": result.lossless_proof["sidecar_byte_identical"],
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    (args.output_dir / "noop_detector.json").write_text(json.dumps(noop, indent=2))

    byte_closure = {
        "schema": "pr110_payload_entropy_recode_byte_closure.v1",
        "source_archive_path": str(args.source_archive),
        "source_archive_sha256": hashlib.sha256(source_archive_bytes).hexdigest(),
        "source_archive_bytes": len(source_archive_bytes),
        "source_member_x_sha256": hashlib.sha256(source_member).hexdigest(),
        "recoded_archive_path": str(args.output_archive),
        "recoded_archive_sha256": hashlib.sha256(recoded_archive_bytes).hexdigest(),
        "recoded_archive_bytes": len(recoded_archive_bytes),
        "recoded_member_x_sha256": hashlib.sha256(result.recoded_member).hexdigest(),
        "archive_bytes_delta": len(recoded_archive_bytes) - len(source_archive_bytes),
        "member_bytes_delta": result.member_delta_bytes,
        "decoder_blob_len": decoder_blob_len,
        "lossless_proof": result.lossless_proof,
        "metrics": result.metrics,
        "selector_payload_byte_identical": True,
        "dqs1_tail_byte_identical": True,
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    (args.output_dir / "byte_closure_proof.json").write_text(
        json.dumps(byte_closure, indent=2)
    )

    manifest = {
        "schema": "pr110_payload_entropy_recode_manifest.v1",
        "target_kind": rec.PR110_PAYLOAD_ENTROPY_RECODE_TARGET_KIND,
        "materializer_id": rec.PR110_PAYLOAD_ENTROPY_RECODE_MATERIALIZER_ID,
        "byte_closure_proof": "byte_closure_proof.json",
        "noop_detector": "noop_detector.json",
        "recoded_archive": str(args.output_archive),
        "recoded_archive_bytes": len(recoded_archive_bytes),
        "member_bytes_delta": result.member_delta_bytes,
        "lossless_recode_proven": True,
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    (args.output_dir / "recode_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
