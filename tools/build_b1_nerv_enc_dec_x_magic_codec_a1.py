#!/usr/bin/env python3
# ROUNDTRIP_TESTED:src/tac/tests/test_b1_nerv_enc_dec_x_magic_codec_a1_roundtrip.py
"""Build B1 composition cell: nerv_enc_dec_separated x magic_codec on A1.

Composition recipe:
1. Read A1 archive `x` inner blob and split into typed sections.
2. Apply magic_codec auto-selector to the decoder_blob bytes.
3. Re-pack inner blob with magic-codec-wrapped decoder section.
4. NO FILM slot (this cell composes nerv_enc_dec_separated with magic_codec -
   NeRV enc/dec separation is compress-time only and adds NO archive bytes).
5. Emit archive.zip + runtime_manifest.json + selection_manifest.json +
   no-op proof JSON.

Composition note:
``nerv_enc_dec_separated`` (per ``src/tac/nerv_enc_dec_separated.py``) is a
TRAINING-TIME bolt-on: an explicit encoder + decoder split where the encoder
embeds frames into latents at compress time, and ONLY the decoder ships at
inflate time. The pre-computed latents land in the latent_blob exactly like
Lane 12-v2 or A1's existing latent table. Therefore on the archive surface
this cell is BYTE-IDENTICAL to the singleton magic_codec on A1 - only the
composition_metadata changes (recording that NeRV enc/dec is the training-
side architecture that produced the substrate).

This makes the cell's "predicted delta band" effectively the magic_codec
overhead band on A1 (~+10..+500 bytes envelope wrap overhead) - there is no
additional architectural cost on disk. The empirical delta here will be exactly
the singleton magic_codec on A1's empirical delta.

CLAUDE.md compliance:
* No scorer load, no MPS, no /tmp.
* runtime_manifest.json declares the composition is compress-time + meta-codec.

Usage::

    python tools/build_b1_nerv_enc_dec_x_magic_codec_a1.py \\
        --output-dir experiments/results/b1_nerv_enc_dec_x_magic_codec_a1_<utc>/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from _b1_composition_on_a1_helper import (  # noqa: E402
    A1_ARCHIVE_PATH_DEFAULT,
    apply_magic_codec_to_decoder_blob,
    build_runtime_manifest,
    emit_b1_archive,
    emit_selection_manifest,
    predicted_band_for_cell,
    read_a1_inner_bytes,
    repack_a1_inner_with_decoder,
    split_a1_inner_sections,
    verify_a1_archive_sha,
)

from tac.output_path_policy import assert_not_temporary_output_dir  # noqa: E402

CELL_ID = "nerv_enc_dec_x_magic_codec"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_b1_nerv_enc_dec_x_magic_codec_a1",
        description=(
            "B1 composition cell: nerv_enc_dec_separated + magic_codec on "
            "the A1 score-gradient-trained substrate. NeRV enc/dec split "
            "is compress-time only; archive bytes match singleton magic_codec."
        ),
    )
    p.add_argument(
        "--source-archive",
        type=Path,
        default=A1_ARCHIVE_PATH_DEFAULT,
        help="Source A1 archive.zip.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory. Must NOT be under /tmp.",
    )
    p.add_argument(
        "--stream-type",
        default="weight_tensor",
        help="Stream-type hint for magic_codec.",
    )
    p.add_argument(
        "--selection-strategy",
        default="smallest_byte_count",
        choices=["smallest_byte_count", "entropy_estimate", "stacked_optimal"],
    )
    p.add_argument("--no-require-sha", action="store_true")
    p.add_argument("--operator", default=None)
    return p.parse_args(argv)


def _run_byte_mutation_no_op_proof(archive_path: Path) -> dict[str, object]:
    raw = archive_path.read_bytes()
    if len(raw) < 16:
        return {"no_op_proof_runnable": False, "reason": "archive too small"}
    mut_offset = len(raw) // 2
    mutated = bytearray(raw)
    mutated[mut_offset] = (mutated[mut_offset] + 1) & 0xFF
    return {
        "no_op_proof_runnable": True,
        "mutation_offset": mut_offset,
        "bytes_differ_after_mutation": bytes(mutated) != raw,
        "runtime_consumes_bytes": False,
        "no_op_detector_passed": False,
        "limitation": (
            "no inflate adapter for B1 composition cells; "
            "runtime_consumes_bytes cannot be verified executably yet"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assert_not_temporary_output_dir(
            args.output_dir, tool_name="build_b1_nerv_enc_dec_x_magic_codec_a1"
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    src_path = args.source_archive.resolve()
    if not src_path.exists():
        raise SystemExit(f"A1 source archive not found: {src_path}")
    _src_bytes, src_sha = verify_a1_archive_sha(
        src_path, require_sha=not args.no_require_sha
    )
    src_size_bytes = src_path.stat().st_size

    inner = read_a1_inner_bytes(src_path)
    sections = split_a1_inner_sections(inner)
    mc_result, mc_meta = apply_magic_codec_to_decoder_blob(
        sections.decoder_blob,
        stream_type=args.stream_type,
        selection_strategy=args.selection_strategy,
        quantize_bits=8,
    )
    new_inner = repack_a1_inner_with_decoder(sections, mc_result.payload)

    composition_metadata = {
        "cell_id": CELL_ID,
        "composition_kind": "orthogonal_pair_bolt_on_compress_time_plus_meta_codec",
        "primary_substrate": "A1 score-gradient-trained PR101 latent-aligned",
        "bolt_on_a_compress_time": "nerv_enc_dec_separated (training-only)",
        "compress_time_note": (
            "NeRV encoder+decoder split is a compress-time architectural "
            "bolt-on; ONLY the decoder ships at inflate time. The latent "
            "blob in this archive is the existing A1 latent table (not "
            "regenerated by a new encoder - that would require retraining)."
        ),
        "meta_codec_b": "magic_codec (auto-selector)",
        "magic_codec_selection": mc_meta,
        "byte_identity_to_singleton_magic_codec_on_a1": True,
        "predicted_delta_band_bytes": list(predicted_band_for_cell(CELL_ID)),
    }

    archive_path, archive_sha, archive_bytes = emit_b1_archive(
        cell_id=CELL_ID,
        out_dir=args.output_dir,
        inner_payload=new_inner,
        film_pose_slot=None,
        composition_metadata=composition_metadata,
    )

    parser_sections = [
        {
            "name": "x",
            "kind": "a1_inner_blob_with_magic_codec_decoder",
            "size_bytes": len(new_inner),
            "components": [
                {"name": "section_total_uint32", "offset": 0, "size": 4},
                {
                    "name": "decoder_blob_magic_codec",
                    "offset": 4,
                    "size": len(mc_result.payload),
                    "magic_codec_selected_primitive": mc_meta["selected_primitive"],
                },
                {
                    "name": "latent_blob",
                    "offset": 4 + len(mc_result.payload),
                    "size": len(sections.latent_blob),
                    "produced_by": "A1 trained latent table (not re-encoded)",
                },
                {
                    "name": "sidecar_blob",
                    "offset": 4 + len(mc_result.payload) + len(sections.latent_blob),
                    "size": len(sections.sidecar_blob),
                },
            ],
        },
    ]

    runtime_manifest = build_runtime_manifest(
        cell_id=CELL_ID,
        archive_path=archive_path,
        archive_sha256=archive_sha,
        archive_size_bytes=archive_bytes,
        source_archive_path=src_path,
        source_archive_sha256=src_sha,
        source_archive_size_bytes=src_size_bytes,
        parser_sections=parser_sections,
        composition_steps=[
            "read A1 archive 'x' inner blob",
            "split into [section_total][decoder][latent][sidecar]",
            "apply magic_codec auto-selector to decoder bytes",
            "repack inner with magic-codec decoder section",
            "(NeRV enc/dec split is compress-time only; no archive bytes added)",
            "emit deterministic zip",
        ],
    )
    (args.output_dir / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n"
    )

    empirical_delta = archive_bytes - src_size_bytes
    selection_manifest = emit_selection_manifest(
        cell_id=CELL_ID,
        composition_metadata=composition_metadata,
        predicted_delta_band_bytes=predicted_band_for_cell(CELL_ID),
        empirical_delta_bytes=empirical_delta,
        archive_path=archive_path,
        archive_sha256=archive_sha,
        archive_size_bytes=archive_bytes,
        source_archive_sha256=src_sha,
        source_archive_size_bytes=src_size_bytes,
        operator=args.operator,
    )
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    )

    proof = _run_byte_mutation_no_op_proof(archive_path)
    proof["generated_at_utc"] = datetime.now(UTC).isoformat()
    proof["cell_id"] = CELL_ID
    (args.output_dir / "no_op_proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n"
    )

    verdict = (
        "REGRESSION"
        if empirical_delta > 0
        else ("SAVINGS" if empirical_delta < 0 else "ZERO_DELTA")
    )
    print(
        f"[byte-proxy] cell={CELL_ID} {archive_path} "
        f"(sha={archive_sha[:8]}, {archive_bytes} bytes; "
        f"empirical delta vs A1 {empirical_delta:+d} bytes; verdict={verdict})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
