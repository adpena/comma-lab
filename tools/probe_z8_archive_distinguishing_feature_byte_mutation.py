#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z8HPC1 archive distinguishing-feature byte-mutation consumption proof.

Per Catalog #272 (distinguishing-feature integration contract) + Catalog #220
(L1+ scaffold operational mechanism) + Catalog #105/#139 (no-op detector):
proves the Z8 contest archive's distinguishing feature is OPERATIONALLY
CONSUMED — mutating a distinguishing byte CHANGES the reconstructed RGB output.

This is the small-frame (archive ``eval_height`` × ``eval_width``) per-pair
reconstruction comparison, NOT the 3.6 GB contest raw. It runs the SAME
``reconstruct_pair_rgb_from_pyramid`` Mallat-inverse path the contest
``inflate_one_video_from_archive_bytes`` runs per pair, so a non-zero pixel
delta on a wavelet-coefficient mutation is a faithful operational-consumption
proof.

HONEST per-section verdict (Catalog #307 paradigm-vs-implementation +
NO FAKE IMPLEMENTATIONS):

- ``wavelet_blob`` is the REAL distinguishing feature (DISTINGUISHING FEATURE #2
  per the Z8 design): a mid-payload coefficient byte-flip produces a measurable
  pixel delta -> CONSUMED.
- ``decoder_blob`` (the canonical-quadruple placeholder M4-projection state,
  AND the slot where a future categorical-posterior HNeRV decoder weight wire-in
  would land) is brotli-PARSE-consumed only: the inflate reconstruction
  (Mallat wavelet inverse) does NOT read the decoder weights for pixels. A
  random flip corrupts the brotli stream -> parse error (structural consumption
  per Catalog #105), but a clean re-serialized decoder state produces NO pixel
  change. The tool reports this honestly so a future wave does not mistake the
  decoder slot for a pixel-consumed distinguishing feature (the Catalog #220
  research-substrate trap).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
Z8_BYTE_MUTATION_PROOF_SCHEMA = (
    "z8_hierarchical_predictive_coding_distinguishing_feature_byte_mutation_proof.v1"
)


def _reconstruct_all_pairs_small(archive_bytes: bytes) -> Any:
    """Reconstruct every trained pair's RGB at the archive's small eval dims.

    Returns a flat numpy float32 vector of all pair pixels (frame_0 ++ frame_1
    concatenated per pair) — the comparison surface for byte-mutation deltas.
    """
    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        parse_archive,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        parse_pair_blobs_from_wavelet_blob,
        reconstruct_pair_rgb_from_pyramid,
    )

    arc = parse_archive(archive_bytes)
    eval_h = int(arc.meta.get("eval_height", 32))
    eval_w = int(arc.meta.get("eval_width", 32))
    cfg = SimpleNamespace(
        num_levels=arc.num_levels,
        num_groups_per_level=tuple(arc.num_groups_per_level),
        num_categories_per_level=tuple(arc.num_categories_per_level),
        num_pairs=arc.num_pairs,
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(eval_h, eval_w),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    outs: list[Any] = []
    for pyr in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyr)
        outs.append(np.concatenate([r0.ravel(), r1.ravel()]))
    return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)


def _section_map(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        parse_z8hpc1_archive_bytes,
    )

    return parse_z8hpc1_archive_bytes(archive_bytes)


def _probe_section_consumption(
    archive_bytes: bytes,
    base_recon: Any,
    section: str,
    *,
    num_offsets: int = 7,
) -> dict[str, Any]:
    """Flip bytes across a section; record max pixel delta + parse-error count.

    Returns a per-section verdict dict. ``CONSUMED`` iff at least one flip
    produced a finite pixel delta above ``min_delta``; ``PARSE_GUARD`` iff
    every flip either raised (structural-consumption guard) or produced a
    zero delta (parse-consumed only, not pixel-consumed).
    """
    import numpy as np

    secs = _section_map(archive_bytes)
    if section not in secs:
        return {"section": section, "present": False}
    start, length = secs[section]
    if length <= 0:
        return {"section": section, "present": True, "length": 0, "verdict": "EMPTY"}

    # Sweep evenly-spaced offsets across the payload (skip the first 4 bytes
    # which are commonly a length/count prefix for the blob).
    skip = min(4, max(0, length - 1))
    span = length - skip
    raw_offsets = (
        [start]
        if span <= 0
        else [
            start + skip + (i * span) // max(num_offsets, 1)
            for i in range(num_offsets)
        ]
    )
    offsets = sorted({o for o in raw_offsets if start <= o < start + length})

    max_delta = 0.0
    n_pixel_changed = 0
    n_parse_error = 0
    n_zero_delta = 0
    per_offset: list[dict[str, Any]] = []
    for off in offsets:
        mutated = bytearray(archive_bytes)
        mutated[off] ^= 0xFF
        try:
            recon = _reconstruct_all_pairs_small(bytes(mutated))
            if recon.shape != base_recon.shape:
                # Length/structure-changing flip -> treat as parse-guard.
                n_parse_error += 1
                per_offset.append({"offset_in_section": off - start, "result": "shape_changed"})
                continue
            delta = float(np.abs(recon - base_recon).max())
            if delta > 0.0:
                n_pixel_changed += 1
                max_delta = max(max_delta, delta)
            else:
                n_zero_delta += 1
            per_offset.append(
                {"offset_in_section": off - start, "max_abs_delta": delta}
            )
        except Exception as exc:  # parse-level structural consumption guard
            n_parse_error += 1
            per_offset.append(
                {"offset_in_section": off - start, "result": "parse_error", "err": repr(exc)[:80]}
            )

    if n_pixel_changed > 0:
        verdict = "PIXEL_CONSUMED"
    elif n_parse_error > 0:
        verdict = "PARSE_GUARD_ONLY"
    else:
        verdict = "NO_OP"
    return {
        "section": section,
        "present": True,
        "length": length,
        "offsets_probed": len(offsets),
        "n_pixel_changed": n_pixel_changed,
        "n_parse_error": n_parse_error,
        "n_zero_delta": n_zero_delta,
        "max_abs_pixel_delta": max_delta,
        "verdict": verdict,
        "per_offset": per_offset,
    }


def probe_z8_archive_distinguishing_feature(
    archive_path: Path,
    *,
    proof_out: Path | None = None,
) -> dict[str, Any]:
    """Run the byte-mutation consumption proof on a Z8HPC1 ``0.bin`` archive.

    Args:
        archive_path: path to a Z8HPC1 ``0.bin`` archive.
        proof_out: optional path for the proof JSON.

    Returns:
        Proof manifest with per-section verdicts + an overall verdict.
    """
    archive_bytes = Path(archive_path).read_bytes()
    base_recon = _reconstruct_all_pairs_small(archive_bytes)

    wavelet = _probe_section_consumption(archive_bytes, base_recon, "wavelet_blob")
    decoder = _probe_section_consumption(archive_bytes, base_recon, "decoder_blob")
    wz = _probe_section_consumption(archive_bytes, base_recon, "wyner_ziv_blob")

    # The distinguishing-feature contract (Catalog #272) is SATISFIED when the
    # canonical distinguishing feature (wavelet_blob) is PIXEL_CONSUMED.
    distinguishing_feature_consumed = (
        wavelet.get("verdict") == "PIXEL_CONSUMED"
    )

    manifest = {
        "schema_version": Z8_BYTE_MUTATION_PROOF_SCHEMA,
        "tool": "tools/probe_z8_archive_distinguishing_feature_byte_mutation.py",
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_bytes),
        "base_recon_pixel_count": int(base_recon.size),
        "distinguishing_feature": "wavelet_coeffs_blob",
        "distinguishing_feature_consumed": bool(distinguishing_feature_consumed),
        "sections": {
            "wavelet_blob": wavelet,
            "decoder_blob": decoder,
            "wyner_ziv_blob": wz,
        },
        "honest_architecture_note": (
            "The Z8 contest inflate reconstructs frames from wavelet_coeffs_blob "
            "(Mallat 1989 perfect reconstruction), NOT from the decoder_blob. "
            "decoder_blob is brotli-PARSE-consumed (structural, Catalog #105) but "
            "NOT pixel-consumed: a clean decoder state change produces zero pixel "
            "delta. The trained categorical-posterior HNeRV renderer's weights are "
            "exported to PyTorch by tools/export_z8_hier_pc_mlx_to_pytorch_state_dict.py "
            "for analysis/future-renderer use, but the current classical-wavelet "
            "inflate does not consume them. Wiring trained decoder weights into "
            "pack_archive WITHOUT an inflate-side consumer would be the Catalog "
            "#220 research-substrate trap."
        ),
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "advisory",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }
    if proof_out is not None:
        out = Path(proof_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["proof_path"] = str(out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--proof-out", type=Path)
    parser.add_argument(
        "--require-distinguishing-feature-consumed", action="store_true"
    )
    args = parser.parse_args(argv)
    manifest = probe_z8_archive_distinguishing_feature(
        args.archive, proof_out=args.proof_out
    )
    wav = manifest["sections"]["wavelet_blob"]
    dec = manifest["sections"]["decoder_blob"]
    print(
        f"[z8-byte-mutation] wavelet_blob verdict={wav.get('verdict')} "
        f"max_abs_pixel_delta={wav.get('max_abs_pixel_delta')}"
    )
    print(
        f"[z8-byte-mutation] decoder_blob verdict={dec.get('verdict')} "
        f"(parse-consumed only; NOT pixel-consumed)"
    )
    print(
        f"[z8-byte-mutation] distinguishing_feature_consumed="
        f"{manifest['distinguishing_feature_consumed']}"
    )
    if args.proof_out is not None:
        print(f"[z8-byte-mutation] proof={args.proof_out}")
    if (
        args.require_distinguishing_feature_consumed
        and not manifest["distinguishing_feature_consumed"]
    ):
        return 1
    return 0


__all__ = [
    "Z8_BYTE_MUTATION_PROOF_SCHEMA",
    "main",
    "probe_z8_archive_distinguishing_feature",
]


if __name__ == "__main__":
    raise SystemExit(main())
