#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A1 sub-frontier enumeration — macOS-CPU advisory only.

Operator directive 2026-05-13 LOCAL HARDWARE MAXIMIZATION SWEEP Stream 4.

The A1 baseline scores 0.192864 [macOS-CPU advisory] at 178,262 bytes. The
φ1 SABOR boundary audit empirically demonstrated 99.27% pixel argmax-stability
at ε=32 RGB uint8 perturbation, suggesting substantial "free bytes" capacity
exists in the HNeRV decoder + latent representation.

This tool enumerates the candidate sub-frontier mutation space for A1's
archive bytes:

1. **Inflate A1 archive** and inspect the byte layout of its three sections
   (decoder_blob, latent_blob, sidecar_blob) per the canonical PR101 grammar.
2. **Section-level entropy + redundancy analysis** — identifies how many
   bytes per section could plausibly be reduced via:
     - lossy weight coarsening (decoder_blob)
     - latent quantization (latent_blob)
     - sidecar pruning (sidecar_blob)
3. **Predicted savings table** with provenance:
     - Section sha256s
     - Byte ranges
     - Compression-headroom estimate
4. **Top-10 sub-frontier candidates** prioritized by EV/byte under the
   "preserve macOS-CPU score within 0.001" envelope (advisory bound only).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192:
- evidence_grade = "macOS-CPU-advisory"
- score_claim = False, promotion_eligible = False
- ranking_only = True

Note: this tool DOES NOT run mutations or re-eval. It catalogs the candidate
space + ranks by predicted EV/byte. Running mutations + paired CPU re-eval is
operator-routable downstream work (~10 min per mutation on macOS-CPU).

Usage:
    .venv/bin/python tools/enumerate_a1_sub_frontier_macos_cpu.py \\
        --a1-archive submissions/a1/archive.zip \\
        --output-dir experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EVIDENCE_GRADE = "macOS-CPU-advisory"
EVIDENCE_TAG = "[macOS-CPU advisory only]"
LANE_ID = "lane_local_hardware_maximization_sweep_20260513"

# Canonical PR101 grammar constants (verified against
# experiments/results/.../submission_dir/inflate.py)
LATENT_BLOB_LEN = 15_387


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _shannon_entropy_bits_per_byte(data: bytes) -> float:
    """Empirical 1st-order byte entropy in bits/byte."""
    if not data:
        return 0.0
    n = len(data)
    counts = Counter(data)
    e = 0.0
    for c in counts.values():
        p = c / n
        e -= p * math.log2(p)
    return e


def _parse_a1_archive(archive_path: Path) -> dict:
    """Parse A1 archive into typed sections per PR101 grammar.

    A1 archive zip contains a single member 'x' with this layout:
        uint32 LE: decoder_section_total_bytes (D)
        byte * (D - 4): encoded decoder blob (split-Brotli)
        byte * 15_387: latent_blob (PR101 ORIGINAL)
        byte * remaining: sidecar_blob (PR101 ORIGINAL)
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        member_names = zf.namelist()
        # PR101 grammar: single 'x' member
        if "x" not in member_names:
            return {
                "schema": "non_pr101_grammar",
                "member_names": member_names,
                "note": "expected single 'x' member per PR101 grammar; got other",
            }
        with zf.open("x") as f:
            inner = f.read()

    if len(inner) < 4 + LATENT_BLOB_LEN:
        return {
            "schema": "truncated_pr101_grammar",
            "inner_bytes": len(inner),
            "note": "inner blob too small to contain decoder + latent sections",
        }

    decoder_total_bytes = struct.unpack("<I", inner[:4])[0]
    if decoder_total_bytes > len(inner) or decoder_total_bytes < 4:
        return {
            "schema": "invalid_decoder_section_length",
            "decoder_total_bytes_decl": decoder_total_bytes,
            "inner_bytes": len(inner),
        }

    decoder_blob = inner[4:decoder_total_bytes]  # excludes the 4 uint32 header bytes
    latent_blob = inner[
        decoder_total_bytes : decoder_total_bytes + LATENT_BLOB_LEN
    ]
    sidecar_blob = inner[decoder_total_bytes + LATENT_BLOB_LEN :]

    return {
        "schema": "pr101_grammar_v1",
        "inner_bytes_total": len(inner),
        "sections": {
            "decoder_header_uint32": {
                "offset": 0,
                "length": 4,
                "value": decoder_total_bytes,
            },
            "decoder_blob": {
                "offset": 4,
                "length": len(decoder_blob),
                "sha256": _sha256_of(decoder_blob),
                "entropy_bits_per_byte": _shannon_entropy_bits_per_byte(decoder_blob),
            },
            "latent_blob": {
                "offset": decoder_total_bytes,
                "length": len(latent_blob),
                "sha256": _sha256_of(latent_blob),
                "entropy_bits_per_byte": _shannon_entropy_bits_per_byte(latent_blob),
            },
            "sidecar_blob": {
                "offset": decoder_total_bytes + LATENT_BLOB_LEN,
                "length": len(sidecar_blob),
                "sha256": _sha256_of(sidecar_blob),
                "entropy_bits_per_byte": _shannon_entropy_bits_per_byte(sidecar_blob),
            },
        },
    }


def _enumerate_candidates(parsed: dict, archive_bytes: int) -> list[dict]:
    """Enumerate sub-frontier mutation candidates with predicted EV/byte.

    Per first-principles cost model (rate = bytes / uncompressed_corpus):
        d(score)/d(bytes) = 25 / uncompressed_size_bytes
    where uncompressed_size = 1200 frames * 874 * 1164 * 3 ≈ 3.66e9 B.
        d(score)/d(bytes) ≈ 25 / 3.66e9 ≈ 6.83e-9 per byte saved.

    For a save of N bytes: score reduction (rate term only) ≈ N * 6.83e-9.
    A 1000-byte save → 6.83e-6 score reduction (within macOS-CPU proxy noise).
    A 10000-byte save → 6.83e-5 score reduction (just above proxy noise).

    Mutations that hold distortion constant therefore need:
        N >= 1500 bytes to be advisory-detectable.

    Candidates ranked by predicted EV (assuming distortion preservation
    holds — which requires empirical paired verification at the macOS-CPU axis).
    """
    if parsed.get("schema") != "pr101_grammar_v1":
        return []

    sec = parsed["sections"]
    candidates = []

    # Candidate 1: decoder weight QAT/lossy-coarsening
    decoder_len = sec["decoder_blob"]["length"]
    decoder_entropy = sec["decoder_blob"]["entropy_bits_per_byte"]
    # If decoder entropy is 8.0, the blob is uncompressible.
    # Entropy < 7.9 suggests residual structure.
    decoder_headroom_bytes = int(decoder_len * (8.0 - decoder_entropy) / 8.0)
    if decoder_headroom_bytes > 0:
        candidates.append({
            "candidate_id": "decoder_lossy_coarsening_block_int4",
            "section": "decoder_blob",
            "mechanism": "block-FP4 / lossy weight quantization",
            "current_section_bytes": decoder_len,
            "predicted_savings_bytes": decoder_headroom_bytes,
            "predicted_score_reduction_rate_only": (
                decoder_headroom_bytes * 25.0 / 3.66e9
            ),
            "distortion_risk": "MEDIUM — score-gradient training may flip on quant",
            "empirical_verification_cost_macos_cpu_min": 10,
            "ev_per_byte_advisory": (
                decoder_headroom_bytes * 25.0 / 3.66e9 / max(1, decoder_headroom_bytes)
            ),
            "cross_ref": "Catalog #14 weights_only_false; PR101 lossy coarsening",
        })

    # Candidate 2: latent blob — uniform structure / smaller block sizes
    latent_len = sec["latent_blob"]["length"]
    latent_entropy = sec["latent_blob"]["entropy_bits_per_byte"]
    latent_headroom_bytes = int(latent_len * (8.0 - latent_entropy) / 8.0)
    if latent_headroom_bytes > 0:
        candidates.append({
            "candidate_id": "latent_uint8_to_int4_smaller_block",
            "section": "latent_blob",
            "mechanism": "latent quantization (int8→int4) or smaller block size",
            "current_section_bytes": latent_len,
            "predicted_savings_bytes": latent_headroom_bytes,
            "predicted_score_reduction_rate_only": (
                latent_headroom_bytes * 25.0 / 3.66e9
            ),
            "distortion_risk": "HIGH — latent quant directly affects decoder output",
            "empirical_verification_cost_macos_cpu_min": 10,
            "ev_per_byte_advisory": (
                latent_headroom_bytes * 25.0 / 3.66e9 / max(1, latent_headroom_bytes)
            ),
            "cross_ref": "Catalog #123 forbidden-weight-saliency; PR106 lossy coarsening",
        })

    # Candidate 3: sidecar blob — pruning
    sidecar_len = sec["sidecar_blob"]["length"]
    sidecar_entropy = sec["sidecar_blob"]["entropy_bits_per_byte"]
    sidecar_headroom_bytes = int(sidecar_len * (8.0 - sidecar_entropy) / 8.0)
    if sidecar_headroom_bytes > 0:
        candidates.append({
            "candidate_id": "sidecar_pruning_or_re_entropy_coding",
            "section": "sidecar_blob",
            "mechanism": "sidecar pruning / re-entropy-coding",
            "current_section_bytes": sidecar_len,
            "predicted_savings_bytes": sidecar_headroom_bytes,
            "predicted_score_reduction_rate_only": (
                sidecar_headroom_bytes * 25.0 / 3.66e9
            ),
            "distortion_risk": "LOW — sidecar typically affects only auxiliary signal",
            "empirical_verification_cost_macos_cpu_min": 10,
            "ev_per_byte_advisory": (
                sidecar_headroom_bytes * 25.0 / 3.66e9 / max(1, sidecar_headroom_bytes)
            ),
            "cross_ref": "PR103 hnerv_lc_ac arithmetic coding gains",
        })

    # Candidate 4: removal of redundant 4-byte uint32 header (if duplicated by other framing)
    # NOT — header is decoded by inflate; cannot remove without grammar break.

    # Candidate 5: SABOR-style boundary-only RGB rewrite (Stream 4 hypothesis)
    # NOT directly applicable to A1 archive bytes; A1 doesn't encode pixel RGB.
    # But this is the SABOR substrate hypothesis space — flagged as
    # research_direction_not_a1_byte_mutation.
    candidates.append({
        "candidate_id": "sabor_boundary_only_substrate_NOT_A1_byte_mutation",
        "section": "n/a",
        "mechanism": "different substrate (boundary-only RGB) — would replace A1",
        "current_section_bytes": 0,
        "predicted_savings_bytes": 0,
        "predicted_score_reduction_rate_only": 0.0,
        "distortion_risk": "DIFFERENT_SUBSTRATE",
        "empirical_verification_cost_macos_cpu_min": 0,
        "ev_per_byte_advisory": 0.0,
        "cross_ref": ".omx/research/sabor_boundary_audit_20260513.md",
        "note": (
            "SABOR is a substrate-level alternative to A1, not an A1 byte "
            "mutation. Recorded here for cross-ref."
        ),
    })

    # Sort by predicted_savings_bytes descending (proxies for EV total)
    candidates.sort(key=lambda c: c["predicted_savings_bytes"], reverse=True)
    return candidates


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--a1-archive",
        type=Path,
        default=REPO_ROOT / "submissions" / "a1" / "archive.zip",
        help="Path to A1 archive.zip (default: submissions/a1/archive.zip).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out_str = str(out)
    if "/tmp/" in out_str or "/private/tmp/" in out_str or "/var/tmp/" in out_str:
        print(f"FATAL: refusing /tmp persisted output: {out_str}", file=sys.stderr)
        return 2

    archive_path = args.a1_archive.resolve()
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_of_file(archive_path)

    parsed = _parse_a1_archive(archive_path)
    candidates = _enumerate_candidates(parsed, archive_bytes)

    output = {
        "schema": "a1_sub_frontier_enumeration_v1",
        "lane_id": LANE_ID,
        "a1_archive_path": str(archive_path),
        "a1_archive_sha256": archive_sha,
        "a1_archive_bytes": archive_bytes,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "platform": {
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "stamped_at_utc": _utc_stamp(),
        "parsed_grammar": parsed,
        "candidates": candidates,
        "summary": {
            "num_candidates": len(candidates),
            "total_predicted_savings_bytes": sum(
                c["predicted_savings_bytes"] for c in candidates
            ),
            "best_candidate": candidates[0]["candidate_id"] if candidates else None,
            "note": (
                "Predictions are first-order Shannon-entropy bounds. Each "
                "candidate requires empirical paired CPU+CUDA verification "
                "(per CLAUDE.md \"Submission auth eval — BOTH CPU AND CUDA\") "
                "before any score claim. Sub-frontier discovery is "
                "operator-routable downstream work."
            ),
        },
    }

    out_path = out / "a1_sub_frontier_enumeration.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    print(f"parsed schema: {parsed.get('schema')}")
    print(f"num candidates: {len(candidates)}")
    if candidates:
        for c in candidates[:5]:
            print(
                f"  {c['candidate_id']:<50}  "
                f"section={c['section']:<14}  "
                f"savings={c['predicted_savings_bytes']:>6} B  "
                f"Δscore≈{c['predicted_score_reduction_rate_only']:.6e}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
