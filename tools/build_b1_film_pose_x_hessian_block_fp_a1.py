#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ROUNDTRIP_TESTED:src/tac/tests/test_b1_film_pose_x_hessian_block_fp_a1_roundtrip.py
"""Build B1 composition cell: film_pose_conditioning x hessian_block_fp on A1.

Composition recipe:
1. Read A1 archive `x` inner blob and split into typed sections.
2. Compute (or accept) per-tensor score-gradient saliency dict for A1's
   decoder state_dict per CLAUDE.md Catalog #123 (forbidden weight-domain
   saliency on score-gradient substrate).
3. Apply Hessian-saliency-weighted bit allocation + lossy coarsening to
   the decoder state_dict (water-filling KKT form).
4. Re-encode the coarsened state_dict via the PR101 split-Brotli codec.
5. Re-pack inner blob with the coarsened decoder section.
6. Add the FILM pose conditioning reserved bolt-on slot (~4 KB).
7. Emit archive.zip + runtime_manifest.json + selection_manifest.json +
   no-op proof JSON.

This is the most likely cell to show real byte savings on A1 because the
hessian_block_fp coarsening directly targets the decoder blob (largest
section) and A1's score-gradient training distributes saliency
non-uniformly across tensors.

CLAUDE.md compliance:
* Catalog #123: refuses weight-domain saliency on A1 unless caller passes
  ``--proxy-acknowledged-non-score-aware``.
* No scorer load at this builder layer (saliency is precomputed externally).
* No /tmp paths.

Usage::

    # Byte-only build with uniform-saliency advisory (no score-aware claim):
    python tools/build_b1_film_pose_x_hessian_block_fp_a1.py \\
        --output-dir experiments/results/b1_film_pose_x_hessian_block_fp_a1_<utc>/ \\
        --target-archive-bytes 175000 \\
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
    FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT,
    build_film_pose_reserved_slot,
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

CELL_ID = "film_pose_x_hessian_block_fp"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_b1_film_pose_x_hessian_block_fp_a1",
        description=(
            "B1 composition cell: film_pose_conditioning + hessian_block_fp "
            "on the A1 score-gradient-trained substrate."
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
        "--film-slot-bytes",
        type=int,
        default=FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT,
        help="Reserved bolt-on slot size (default 4096).",
    )
    p.add_argument(
        "--target-archive-bytes",
        type=int,
        default=175_000,
        help=(
            "Target FINAL archive bytes (after FiLM slot + coarsened "
            "decoder). The coarsener back-solves the decoder budget from "
            "this target."
        ),
    )
    p.add_argument(
        "--floor-bits",
        type=int,
        default=4,
        help="Minimum per-tensor bit width (default 4).",
    )
    p.add_argument(
        "--ceiling-bits",
        type=int,
        default=8,
        help="Maximum per-tensor bit width (default 8 = original PR101 INT8).",
    )
    p.add_argument(
        "--proxy-acknowledged-non-score-aware",
        action="store_true",
        help=(
            "Explicit opt-in for uniform-saliency advisory proxy when no "
            "score-gradient computation is available. The output is then "
            "tagged advisory-only with NO score claim. Required to "
            "satisfy Catalog #123 enforcement without computing real "
            "score-gradient saliency (~25 min CPU autograd)."
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
            "no inflate adapter exists for B1 composition cells; "
            "runtime_consumes_bytes cannot be verified executably yet"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assert_not_temporary_output_dir(
            args.output_dir, tool_name="build_b1_film_pose_x_hessian_block_fp_a1"
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.proxy_acknowledged_non_score_aware:
        raise SystemExit(
            "FATAL: hessian_block_fp on A1 requires a score-gradient "
            "saliency proxy (Catalog #123). For a byte-proxy-only build, "
            "pass --proxy-acknowledged-non-score-aware to acknowledge an "
            "advisory uniform-saliency proxy; the output will be tagged "
            "advisory-only with NO score claim. For a score-aware build, "
            "compute via tac.score_gradient_param_saliency.build_score_gradient_saliency_for_a1_archive "
            "and integrate it into a sibling builder."
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

    # Solve back the decoder byte budget from the FINAL archive target.
    # archive size = zip_overhead + len(inner) + zip_overhead + len(FILM slot)
    # inner size  = 4 + decoder + len(latent) + len(sidecar)
    # decoder_budget = target - everything-else
    fixed_overhead = (
        src_size_bytes
        - len(sections.decoder_blob)
        + args.film_slot_bytes
        # The FILM zip member adds ~70-100 bytes of zip metadata; budget
        # generously here so the bisection back-solves a feasible decoder.
        + 100
    )
    target_decoder_bytes = max(1, args.target_archive_bytes - fixed_overhead)

    # Get advisory uniform-saliency proxy (caller acknowledged non-score-aware).
    saliency_proxy = synthesize_neutral_saliency_advisory(sections.decoder_blob)

    new_decoder_blob, bits_per_tensor, rel_errs = coarsen_decoder_state_dict_by_hessian(
        sections.decoder_blob,
        saliency_proxy=saliency_proxy,
        target_decoder_bytes=target_decoder_bytes,
        floor_bits=args.floor_bits,
        ceiling_bits=args.ceiling_bits,
    )
    new_inner = repack_a1_inner_with_decoder(sections, new_decoder_blob)
    film_slot = build_film_pose_reserved_slot(args.film_slot_bytes)

    composition_metadata = {
        "cell_id": CELL_ID,
        "composition_kind": "orthogonal_pair_bolt_on_plus_lossy_coarsen",
        "primary_substrate": "A1 score-gradient-trained PR101 latent-aligned",
        "bolt_on_a": "film_pose_conditioning (reserved slot)",
        "lossy_b": "hessian_block_fp (saliency-weighted per-tensor bit allocation)",
        "saliency_source": "uniform_advisory_proxy_acknowledged_non_score_aware",
        "advisory_only": True,
        "film_slot_bytes": args.film_slot_bytes,
        "target_decoder_bytes": target_decoder_bytes,
        "target_archive_bytes": args.target_archive_bytes,
        "floor_bits": args.floor_bits,
        "ceiling_bits": args.ceiling_bits,
        "bits_per_tensor": bits_per_tensor,
        "max_rel_err_per_tensor": max(rel_errs.values()) if rel_errs else 0.0,
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
            "kind": "a1_inner_blob_with_coarsened_decoder",
            "size_bytes": len(new_inner),
            "components": [
                {"name": "section_total_uint32", "offset": 0, "size": 4},
                {
                    "name": "decoder_blob_coarsened",
                    "offset": 4,
                    "size": len(new_decoder_blob),
                    "coarsen_strategy": "hessian_saliency_water_filling",
                    "advisory_only": True,
                },
                {
                    "name": "latent_blob",
                    "offset": 4 + len(new_decoder_blob),
                    "size": len(sections.latent_blob),
                },
                {
                    "name": "sidecar_blob",
                    "offset": 4 + len(new_decoder_blob) + len(sections.latent_blob),
                    "size": len(sections.sidecar_blob),
                },
            ],
        },
        {
            "name": "FILM",
            "kind": "film_pose_conditioning_reserved_slot",
            "size_bytes": args.film_slot_bytes,
            "magic_bytes": "FILM",
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
            "apply hessian_block_fp saliency-weighted bit allocation",
            "coarsen decoder state_dict; re-encode via PR101 split-Brotli",
            "repack inner with coarsened decoder section",
            "add FILM reserved bolt-on slot",
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
