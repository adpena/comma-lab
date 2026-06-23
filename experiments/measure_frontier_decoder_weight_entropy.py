"""Measure the empirical Shannon entropy of the frontier decoder's quantized
weights vs the bytes the entropy coder actually spends.

DECISIVE $0 MEASUREMENT (Part C #4 of the PR95-vs-ours capacity-RD deep-math memo):
  Q: Is the frontier's decoder section (161,104 bytes) near the Shannon entropy
     of its quantized INT8 weight symbols, or is there ~2x rate headroom that a
     better entropy coder could recover (→ a pure recode toward sub-0.15)?

Authority: [macOS-CPU advisory] / [analysis] NON-PROMOTABLE. score_claim=false,
promotion_eligible=false, ready_for_exact_eval_dispatch=false. No MPS, no paid
dispatch, no pinned-upstream edits. Moves no pointer. ALL contest-score math via
tac.contest_score (never hand-rolled).

Method (NO FAKE — measures the REAL frontier archive bytes b46897267...):
  1. Inflate the frontier member 'x' to recover the byte-exact raw decoder stream
     (the INT8 weight symbols + fp16 per-tensor scales) via the submission's own
     inflate.py path. The byte-closure proof confirms this raw stream is exactly
     what the range coder encodes.
  2. Per tensor: split symbols from scales, compute the order-0 Shannon entropy
     H0 = -sum p_i log2 p_i over the 256-ary symbol alphabet, and the implied
     ideal bytes = numel * H0 / 8 + 2 (the fp16 scale is incompressible).
  3. Compare sum(ideal bytes) to the measured decoder section bytes (161,104).
  4. Recompute S at the entropy floor, holding d_seg/d_pose fixed at the frontier,
     and at 2x / 3x rate, via tac.contest_score.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FRONTIER_DIR = REPO / "experiments/results/pr110_payload_entropy_recode_20260610"
SUBMISSION_DIR = FRONTIER_DIR / "submission_dir"
ARCHIVE = SUBMISSION_DIR / "archive.zip"
EXPECTED_SHA = "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e"

# Measured byte budget (from byte_closure_proof.json metrics).
DEC_SEC_BYTES = 161104  # decoder section bytes (range-coded, post-recode)
LAT_SEC_BYTES = 15070   # latent section bytes
SIDECAR_LEN = 607
ARCHIVE_BYTES = 177169  # the frontier archive.zip total
TOTAL_VIDEO_BYTES = 37_545_489  # contest rate denominator (evaluate.py:64)

# Frontier distortion (from report.txt, recomputed exact).
FRONTIER_D_SEG = 0.00055978
FRONTIER_D_POSE = 0.00002942


def h0_bits_per_symbol(symbols: np.ndarray) -> float:
    """Order-0 Shannon entropy in bits/symbol over the empirical histogram."""
    if symbols.size == 0:
        return 0.0
    counts = np.bincount(symbols.astype(np.int64), minlength=1).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> int:
    import hashlib

    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert sha == EXPECTED_SHA, f"archive sha mismatch: {sha} != {EXPECTED_SHA}"

    # Wire the submission's own inflate module to recover the raw decoder stream.
    sys.path.insert(0, str(SUBMISSION_DIR))
    sys.path.insert(0, str(SUBMISSION_DIR / "src"))
    import zipfile

    import inflate as inf  # the frontier's own inflate.py

    with zipfile.ZipFile(ARCHIVE) as z:
        member = z.read("x")

    (
        decoder_sd,
        latents,
        meta,
        raw_joined,
        *_rest,
    ) = inf.parse_member(member)

    # Walk the raw decoder stream exactly as decode_decoder_state_from_raw does,
    # but capture the per-tensor INT8 SYMBOLS (uint8 as stored = post-byte-map),
    # which is exactly what the range coder models.
    from model import HNeRVDecoder

    probe = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    items = list(probe.state_dict().items())

    pos = 0
    per_tensor = []
    all_symbols_storage = []  # the uint8 stored stream (what the coder sees)
    all_symbols_signed = []   # the signed INT8 (post byte-map) value semantics
    raw = np.frombuffer(raw_joined, dtype=np.uint8)

    for idx in inf.DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        numel = int(tensor.numel())
        zz = raw[pos : pos + numel]  # the stored uint8 symbols
        pos += numel
        scale = float(np.frombuffer(raw_joined, dtype=np.float16, count=1, offset=pos)[0])
        pos += 2

        # The signed INT8 values the byte-map decodes to (post-map alphabet).
        signed = inf.decode_mapped_u8(zz, inf.DECODER_BYTE_MAPS.get(idx, "zig"))
        signed_u = (signed.astype(np.int64) + 128).astype(np.uint8)  # 0..255 for hist

        h_stored = h0_bits_per_symbol(zz)
        h_signed = h0_bits_per_symbol(signed_u)
        # The coder is per-tensor adaptive; the best order-0 model uses whichever
        # alphabet has lower entropy. Report both; use min for the ideal bound.
        h_best = min(h_stored, h_signed)
        ideal_bytes = numel * h_best / 8.0 + 2.0  # +2 fp16 scale (incompressible)

        per_tensor.append({
            "idx": int(idx),
            "name": name,
            "numel": numel,
            "scale": scale,
            "byte_map": inf.DECODER_BYTE_MAPS.get(idx, "zig"),
            "n_unique": int(np.unique(zz).size),
            "H0_stored_bits": h_stored,
            "H0_signed_bits": h_signed,
            "H0_best_bits": h_best,
            "ideal_bytes_H0": ideal_bytes,
        })
        all_symbols_storage.append(zz)
        all_symbols_signed.append(signed_u)

    assert pos == len(raw_joined), f"raw walk mismatch: {pos} != {len(raw_joined)}"

    # Global order-0 entropy (single shared model — what a non-adaptive coder gets).
    glob_stored = np.concatenate(all_symbols_storage)
    glob_signed = np.concatenate(all_symbols_signed)
    total_numel = int(glob_stored.size)
    H0_global_stored = h0_bits_per_symbol(glob_stored)
    H0_global_signed = h0_bits_per_symbol(glob_signed)

    # Per-tensor adaptive ideal (sum of per-tensor H0 ideal bytes) — the bound the
    # frontier's per-tensor adaptive 256-ary coder targets.
    sum_ideal_per_tensor = sum(t["ideal_bytes_H0"] for t in per_tensor)
    n_scale_bytes = 2 * len(per_tensor)
    sum_ideal_symbol_only = sum_ideal_per_tensor - n_scale_bytes

    # Global single-model ideal bytes (symbols only).
    global_ideal_symbol_only = total_numel * min(H0_global_stored, H0_global_signed) / 8.0

    # ---- Score arithmetic via tac.contest_score (NEVER hand-rolled) ----
    sys.path.insert(0, str(REPO / "src"))
    from tac.contest_score import compute_contest_score  # noqa

    def S(d_seg, d_pose, archive_bytes):
        return compute_contest_score(
            d_seg, d_pose, archive_bytes, uncompressed_size=TOTAL_VIDEO_BYTES
        )

    # Sanity: reproduce the frontier S at the real archive bytes.
    s_frontier = S(FRONTIER_D_SEG, FRONTIER_D_POSE, ARCHIVE_BYTES)

    # S at the per-tensor-H0 entropy floor for the DECODER section, holding
    # latent + sidecar + zip overhead fixed, d_seg/d_pose fixed.
    non_decoder_fixed = ARCHIVE_BYTES - DEC_SEC_BYTES  # latent + sidecar + zip/container overhead
    dec_at_floor = sum_ideal_per_tensor
    archive_at_floor = non_decoder_fixed + dec_at_floor
    s_at_floor = S(FRONTIER_D_SEG, FRONTIER_D_POSE, archive_at_floor)

    # S at 2x / 3x DECODER-section rate reduction (the parent's hypothesis lever).
    s_dec_2x = S(FRONTIER_D_SEG, FRONTIER_D_POSE, non_decoder_fixed + DEC_SEC_BYTES / 2)
    s_dec_3x = S(FRONTIER_D_SEG, FRONTIER_D_POSE, non_decoder_fixed + DEC_SEC_BYTES / 3)

    # S if the WHOLE archive (decoder+latent) shrank 2x / 3x.
    s_all_2x = S(FRONTIER_D_SEG, FRONTIER_D_POSE, ARCHIVE_BYTES / 2)
    s_all_3x = S(FRONTIER_D_SEG, FRONTIER_D_POSE, ARCHIVE_BYTES / 3)

    result = {
        "axis": "[macOS-CPU advisory] / [analysis]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_sha256": sha,
        "measured_byte_budget": {
            "archive_bytes": ARCHIVE_BYTES,
            "decoder_section_bytes": DEC_SEC_BYTES,
            "latent_section_bytes": LAT_SEC_BYTES,
            "sidecar_bytes": SIDECAR_LEN,
            "non_decoder_fixed_bytes": non_decoder_fixed,
        },
        "decoder_weight_stats": {
            "total_weight_params": total_numel,
            "n_tensors": len(per_tensor),
            "achieved_decoder_bytes": DEC_SEC_BYTES,
            "achieved_bits_per_param": DEC_SEC_BYTES * 8.0 / total_numel,
            "H0_per_tensor_adaptive": {
                "ideal_bytes_with_scales": sum_ideal_per_tensor,
                "ideal_bytes_symbols_only": sum_ideal_symbol_only,
                "ideal_bits_per_param": sum_ideal_per_tensor * 8.0 / total_numel,
                "achieved_vs_ideal_ratio": DEC_SEC_BYTES / sum_ideal_per_tensor,
                "headroom_bytes": DEC_SEC_BYTES - sum_ideal_per_tensor,
            },
            "H0_global_single_model": {
                "H0_global_stored_bits": H0_global_stored,
                "H0_global_signed_bits": H0_global_signed,
                "ideal_bytes_symbols_only": global_ideal_symbol_only,
                "ideal_bits_per_param": global_ideal_symbol_only * 8.0 / total_numel,
            },
        },
        "score_arithmetic_via_tac_contest_score": {
            "frontier_reproduced_S": s_frontier,
            "S_at_per_tensor_H0_decoder_floor": s_at_floor,
            "decoder_floor_archive_bytes": archive_at_floor,
            "delta_S_to_decoder_H0_floor": s_at_floor - s_frontier,
            "S_decoder_2x": s_dec_2x,
            "S_decoder_3x": s_dec_3x,
            "S_whole_archive_2x": s_all_2x,
            "S_whole_archive_3x": s_all_3x,
        },
        "per_tensor": per_tensor,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
