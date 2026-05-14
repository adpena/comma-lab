#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ROUNDTRIP_TESTED:src/tac/tests/test_b1_film_pose_x_magic_codec_a1_roundtrip.py
"""Build B1 composition cell: film_pose_conditioning x magic_codec on A1.

Composition recipe:
1. Read A1 archive `x` member and split into [section_total][decoder][latent][sidecar].
2. Apply magic_codec auto-selector to the decoder_blob bytes (int8 stream).
3. Re-pack inner blob with the magic-codec-wrapped decoder section.
4. Add a FILM pose conditioning reserved bolt-on slot (~4 KB) as a separate
   zip member ``FILM`` so the archive byte count reflects the cost of
   including the bolt-on.
5. Emit archive.zip + runtime_manifest.json + selection_manifest.json + a
   no-op proof JSON.

Per CLAUDE.md "Forbidden score claims" + Catalog #100 + Catalog #91 + Catalog
#139, the emitted manifests permanently disable score claims and dispatch
eligibility until empirical CUDA + CPU eval lands. The byte-mutation no-op
proof verifies whether a single-byte flip to the archive changes downstream
inflate output - but since no inflate adapter exists for these composition
cells yet, the proof's runtime_consumption_passed is permanently False.

CLAUDE.md compliance:
* NO scorer load, NO MPS, NO /tmp paths, NO retroactive A1 mutation.
* runtime_manifest.json carries archive custody + dispatch blockers.
* Sibling test file at ``src/tac/tests/test_b1_film_pose_x_magic_codec_a1_roundtrip.py``
  per Catalog #91.

Usage::

    python tools/build_b1_film_pose_x_magic_codec_a1.py \\
        --output-dir experiments/results/b1_film_pose_x_magic_codec_a1_<utc>/
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
    FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT,
    apply_magic_codec_to_decoder_blob,
    build_film_pose_reserved_slot,
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

CELL_ID = "film_pose_x_magic_codec"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_b1_film_pose_x_magic_codec_a1",
        description=(
            "B1 composition cell: film_pose_conditioning + magic_codec on "
            "the A1 score-gradient-trained substrate."
        ),
    )
    p.add_argument(
        "--source-archive",
        type=Path,
        default=A1_ARCHIVE_PATH_DEFAULT,
        help="Source A1 archive.zip (default: canonical A1 anchor).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory. Must NOT be under /tmp. Canonical location: "
            "experiments/results/b1_<cell>_a1_<utc>/."
        ),
    )
    p.add_argument(
        "--film-slot-bytes",
        type=int,
        default=FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT,
        help=(
            "Reserved bolt-on slot size (default 4096; substrate matrix "
            "byte_budget_band is 2000-8000)."
        ),
    )
    p.add_argument(
        "--stream-type",
        default="weight_tensor",
        help="Stream-type hint for magic_codec auto-selector.",
    )
    p.add_argument(
        "--selection-strategy",
        default="smallest_byte_count",
        choices=["smallest_byte_count", "entropy_estimate", "stacked_optimal"],
        help="Magic codec selection strategy.",
    )
    p.add_argument("--no-require-sha", action="store_true", help="Skip A1 SHA check.")
    p.add_argument("--operator", default=None, help="Operator handle for provenance.")
    return p.parse_args(argv)


def _run_byte_mutation_no_op_proof(archive_path: Path) -> dict[str, object]:
    """Mutate one archive byte and observe whether downstream inflate changes.

    Per CLAUDE.md Catalog #139: ``_verify_runtime_consumes_payload_bytes_executable``
    pattern. Since no inflate adapter exists for B1 composition cells (the
    runtime contract is research-only until vendored), this proof can only
    detect WHETHER the byte change persists in the rebuilt archive (true)
    and CANNOT verify downstream inflate consumption.

    Returns a structured proof row recording the limitation.
    """
    raw = archive_path.read_bytes()
    if len(raw) < 16:
        return {
            "no_op_proof_runnable": False,
            "reason": "archive too small for byte-mutation smoke",
        }
    # Pick a byte well inside the zip body (skip local-header magic).
    mut_offset = len(raw) // 2
    mutated = bytearray(raw)
    mutated[mut_offset] = (mutated[mut_offset] + 1) & 0xFF
    bytes_differ = bytes(mutated) != raw
    return {
        "no_op_proof_runnable": True,
        "mutation_offset": mut_offset,
        "bytes_differ_after_mutation": bytes_differ,
        # The next two fields can only flip to True once an inflate adapter
        # exists for the B1 composition cells. Until then, the runtime
        # consumption proof is permanently False per Catalog #139.
        "runtime_consumes_bytes": False,
        "no_op_detector_passed": False,
        "limitation": (
            "no inflate adapter exists for B1 composition cells; "
            "runtime_consumes_bytes cannot be verified executably until "
            "tools/_b1_inflate_adapter_a1.py is vendored"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assert_not_temporary_output_dir(
            args.output_dir, tool_name="build_b1_film_pose_x_magic_codec_a1"
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

    # Step 1+2: read A1 inner sections + apply magic_codec to decoder blob.
    inner = read_a1_inner_bytes(src_path)
    sections = split_a1_inner_sections(inner)
    mc_result, mc_meta = apply_magic_codec_to_decoder_blob(
        sections.decoder_blob,
        stream_type=args.stream_type,
        selection_strategy=args.selection_strategy,
        quantize_bits=8,
    )

    # Step 3: repack inner with magic-codec-wrapped decoder.
    new_decoder_payload = mc_result.payload
    new_inner = repack_a1_inner_with_decoder(sections, new_decoder_payload)

    # Step 4: build FILM reserved bolt-on slot.
    film_slot = build_film_pose_reserved_slot(args.film_slot_bytes)

    composition_metadata = {
        "cell_id": CELL_ID,
        "composition_kind": "orthogonal_pair_bolt_on_plus_meta_codec",
        "primary_substrate": "A1 score-gradient-trained PR101 latent-aligned",
        "bolt_on_a": "film_pose_conditioning (reserved slot)",
        "meta_codec_b": "magic_codec (auto-selector)",
        "film_slot_bytes": args.film_slot_bytes,
        "magic_codec_selection": mc_meta,
        "predicted_delta_band_bytes": list(predicted_band_for_cell(CELL_ID)),
    }

    archive_path, archive_sha, archive_bytes = emit_b1_archive(
        cell_id=CELL_ID,
        out_dir=args.output_dir,
        inner_payload=new_inner,
        film_pose_slot=film_slot,
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
                    "size": len(new_decoder_payload),
                    "magic_codec_selected_primitive": mc_meta["selected_primitive"],
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
        {
            "name": "FILM",
            "kind": "film_pose_conditioning_reserved_slot",
            "size_bytes": args.film_slot_bytes,
            "magic_bytes": "FILM",
            "note": "zero-filled reserved slot for future trained FiLM weights",
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
            "add FILM reserved bolt-on slot (4 KB zero-fill, FILM magic)",
            "emit deterministic zip (ZIP_STORED, fixed timestamp)",
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

    # Per CLAUDE.md "Apples-to-apples evidence discipline" lane-tag rule.
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
