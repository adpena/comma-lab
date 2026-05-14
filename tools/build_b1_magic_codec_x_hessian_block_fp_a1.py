#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ROUNDTRIP_TESTED:src/tac/tests/test_b1_magic_codec_x_hessian_block_fp_a1_roundtrip.py
"""Build B1 composition cell: magic_codec x hessian_block_fp on A1.

Composition recipe (sequential pipeline):
1. Read A1 archive `x` inner blob and split into typed sections.
2. Apply Hessian-saliency-weighted bit allocation + lossy coarsening to
   A1's decoder state_dict.
3. Re-encode coarsened state_dict via PR101 split-Brotli codec.
4. Wrap the resulting coarsened decoder_blob in a magic_codec envelope.
5. Re-pack inner blob with magic-codec-wrapped coarsened decoder section.
6. NO FILM slot.
7. Emit archive.zip + runtime_manifest.json + selection_manifest.json +
   no-op proof JSON.

This is the OTHER cell predicted to show real byte savings on A1: the
hessian coarsening reduces decoder bytes, then magic_codec wraps the
already-Brotli-compressed result (might add envelope overhead or might
find a smaller primitive - that's what the auto-selector decides).

CLAUDE.md compliance:
* Catalog #123: requires score-aware saliency or explicit advisory opt-in.
* No scorer load, no MPS, no /tmp.

Usage::

    python tools/build_b1_magic_codec_x_hessian_block_fp_a1.py \\
        --output-dir experiments/results/b1_magic_codec_x_hessian_block_fp_a1_<utc>/ \\
        --target-decoder-bytes 155000 \\
        --proxy-acknowledged-non-score-aware
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
    coarsen_decoder_state_dict_by_hessian,
    emit_b1_archive,
    emit_selection_manifest,
    predicted_band_for_cell,
    read_a1_inner_bytes,
    repack_a1_inner_with_decoder,
    split_a1_inner_sections,
    synthesize_neutral_saliency_advisory,
    verify_a1_archive_sha,
)

from tac.output_path_policy import assert_not_temporary_output_dir  # noqa: E402

CELL_ID = "magic_codec_x_hessian_block_fp"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_b1_magic_codec_x_hessian_block_fp_a1",
        description=(
            "B1 composition cell: magic_codec wrapping hessian_block_fp "
            "coarsened decoder on A1 substrate."
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
        "--target-decoder-bytes",
        type=int,
        default=155_000,
        help=(
            "Target decoder-blob bytes pre-magic-codec-wrap (default 155000; "
            "the wrapper may add or remove a few hundred bytes)."
        ),
    )
    p.add_argument(
        "--floor-bits",
        type=int,
        default=4,
        help="Minimum per-tensor bit width.",
    )
    p.add_argument(
        "--ceiling-bits",
        type=int,
        default=8,
        help="Maximum per-tensor bit width.",
    )
    p.add_argument(
        "--stream-type",
        default="weight_tensor",
        help="Stream-type hint for magic_codec wrap of coarsened decoder.",
    )
    p.add_argument(
        "--selection-strategy",
        default="smallest_byte_count",
        choices=["smallest_byte_count", "entropy_estimate", "stacked_optimal"],
    )
    p.add_argument(
        "--proxy-acknowledged-non-score-aware",
        action="store_true",
        help=(
            "Explicit opt-in for uniform-saliency advisory proxy (no "
            "score-aware claim). Required per Catalog #123 unless "
            "a real score-gradient saliency is precomputed."
        ),
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
            args.output_dir, tool_name="build_b1_magic_codec_x_hessian_block_fp_a1"
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.proxy_acknowledged_non_score_aware:
        raise SystemExit(
            "FATAL: hessian_block_fp on A1 requires score-gradient saliency "
            "(Catalog #123). For a byte-proxy-only advisory build pass "
            "--proxy-acknowledged-non-score-aware."
        )

    src_path = args.source_archive.resolve()
    if not src_path.exists():
        raise SystemExit(f"A1 source archive not found: {src_path}")
    _src_bytes, src_sha = verify_a1_archive_sha(
        src_path, require_sha=not args.no_require_sha
    )
    src_size_bytes = src_path.stat().st_size

    inner = read_a1_inner_bytes(src_path)
    sections = split_a1_inner_sections(inner)

    # Step 1: Hessian-weighted lossy coarsening of decoder state_dict.
    saliency_proxy = synthesize_neutral_saliency_advisory(sections.decoder_blob)
    coarsened_decoder, bits_per_tensor, rel_errs = coarsen_decoder_state_dict_by_hessian(
        sections.decoder_blob,
        saliency_proxy=saliency_proxy,
        target_decoder_bytes=args.target_decoder_bytes,
        floor_bits=args.floor_bits,
        ceiling_bits=args.ceiling_bits,
    )

    # Step 2: magic_codec wrap of coarsened decoder bytes.
    mc_result, mc_meta = apply_magic_codec_to_decoder_blob(
        coarsened_decoder,
        stream_type=args.stream_type,
        selection_strategy=args.selection_strategy,
        quantize_bits=8,
    )
    new_decoder_payload = mc_result.payload
    new_inner = repack_a1_inner_with_decoder(sections, new_decoder_payload)

    composition_metadata = {
        "cell_id": CELL_ID,
        "composition_kind": "sequential_lossy_then_meta_codec",
        "primary_substrate": "A1 score-gradient-trained PR101 latent-aligned",
        "lossy_a": "hessian_block_fp (saliency-weighted per-tensor bit allocation)",
        "meta_codec_b": "magic_codec wrapping coarsened decoder",
        "saliency_source": "uniform_advisory_proxy_acknowledged_non_score_aware",
        "advisory_only": True,
        "target_decoder_bytes": args.target_decoder_bytes,
        "coarsened_decoder_bytes_pre_magic_codec": len(coarsened_decoder),
        "magic_codec_payload_bytes": len(new_decoder_payload),
        "magic_codec_envelope_overhead": len(new_decoder_payload) - len(coarsened_decoder),
        "floor_bits": args.floor_bits,
        "ceiling_bits": args.ceiling_bits,
        "bits_per_tensor": bits_per_tensor,
        "max_rel_err_per_tensor": max(rel_errs.values()) if rel_errs else 0.0,
        "magic_codec_selection": mc_meta,
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
            "kind": "a1_inner_blob_coarsened_then_magic_codec_wrapped",
            "size_bytes": len(new_inner),
            "components": [
                {"name": "section_total_uint32", "offset": 0, "size": 4},
                {
                    "name": "decoder_blob_coarsened_magic_codec",
                    "offset": 4,
                    "size": len(new_decoder_payload),
                    "coarsen_strategy": "hessian_saliency_water_filling",
                    "magic_codec_selected_primitive": mc_meta["selected_primitive"],
                    "advisory_only": True,
                },
                {
                    "name": "latent_blob",
                    "offset": 4 + len(new_decoder_payload),
                    "size": len(sections.latent_blob),
                },
                {
                    "name": "sidecar_blob",
                    "offset": 4 + len(new_decoder_payload) + len(sections.latent_blob),
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
            "split into typed sections",
            "apply hessian_block_fp saliency-weighted bit allocation + lossy coarsening",
            "re-encode coarsened state_dict via PR101 split-Brotli",
            "wrap coarsened decoder bytes in magic_codec envelope",
            "repack inner with magic-codec-wrapped coarsened decoder",
            "emit deterministic zip",
        ],
        measured_config_status="byte_proxy_only_advisory_saliency",
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
        f"[byte-proxy advisory] cell={CELL_ID} {archive_path} "
        f"(sha={archive_sha[:8]}, {archive_bytes} bytes; "
        f"empirical delta vs A1 {empirical_delta:+d} bytes; verdict={verdict})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
