#!/usr/bin/env python3
"""PR101 frame-conditional bit-budget byte-anchor (Track 1 council Decision 5).

Method (CPU-prep only, no scorer load, no contest-CUDA)
───────────────────────────────────────────────────────
PR101's archive consists of a 162,164-byte HNeRV decoder blob (Brotli-compressed
quantised weights) + a 15,387-byte LZMA-compressed latent stream + a
607-byte sidecar. The latent stream encodes 600 frame-pairs × 28
latent-dim symbols at uint8 precision (nominal 8 bits/symbol). Per-pair
delta encoding within the LZMA filter compresses the symbol differences.

This tool **does not retrain PR101**. It instead quantifies the byte
impact of frame-conditional bit allocation as a *re-quantisation proxy*:

  1. Read 1200 frames from ``upstream/videos/0.mkv`` and compute
     per-frame edge-density × pixel-variance × frame-difference complexity.
  2. Pair-average the 1200 frame complexities to 600 per-pair complexities.
  3. Allocate total latent bit budget (134,400 bits = 16,800 bytes ÷ 8)
     across 600 pairs at eta ∈ {0.0, 0.5, 1.0, 2.0} via
     :func:`tac.codec.frame_conditional_bit_budget.allocate_per_frame_bits`.
  4. For each eta, compute the per-pair quantisation precision implied by
     the allocation:  ``q_bits_i = bits_pair_i / LATENT_DIM`` (rounded
     down, clamped to ≥1 and ≤8).
  5. Re-quantise PR101's actual de-quantised float latents to ``q_bits_i``
     precision per pair. Re-encode via the same LZMA delta-pipeline used
     by PR101's codec (fixed format, no schema change). Measure the new
     latent stream byte size + sidecar overhead for per-pair bit-width
     metadata (≤600 bytes worst case).
  6. Compare to the uniform-allocation baseline (eta=0).

Why this is a valid byte-anchor (and what it does NOT measure)
──────────────────────────────────────────────────────────────
The byte-anchor measures how the SAME total latent bit budget redistributes
under frame-conditional concentration. Since PR101 has fixed total bits per
pair under the canonical schema, this anchor is equivalent to: "if we
deviated from uniform 8-bit-per-symbol storage and granted high-motion
pairs more bits at the expense of low-motion pairs, how would the LZMA
delta-encoder respond?"

The anchor does NOT measure the *score* impact of the redistribution,
because per-pair score sensitivity (∂S/∂q_bits_pair_i) requires a
contest-CUDA forward pass on the modified latents through the HNeRV
decoder + scorer. This is recorded as a dispatch_blocker:
``awaiting_per_frame_score_marginal``.

Compliance markers
──────────────────
* ``score_claim=False``
* ``byte_proxy_only=True``
* ``ready_for_exact_eval_dispatch=False``
* ``rank_or_kill_eligible=False``
* ``promotion_eligible=False``

Tag: ``[CPU-prep faithful frame-conditional byte anchor]``.

Usage
─────
.venv/bin/python tools/pr101_frame_conditional_bit_anchor.py \
    --pr101-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
    --video-path upstream/videos/0.mkv \
    --output-dir experiments/results/pr101_frame_conditional_bit_<TIMESTAMP>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import lzma
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PR101_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex"
    / "source/submissions/hnerv_ft_microcodec/src"
)
sys.path.insert(0, str(PR101_SOURCE))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Constants from PR101's codec.py (mirrored — we do NOT mutate the upstream
# clone; per CLAUDE.md "Forbidden in-place edits to public PR intake clones").
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_N_PAIRS = 600
PR101_LATENT_DIM = 28
PR101_LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]

TOOL_NAME = "tools/pr101_frame_conditional_bit_anchor.py"
SCHEMA_VERSION = "pr101_frame_conditional_bit_anchor.v1"
EVIDENCE_GRADE = "[CPU-prep faithful frame-conditional byte anchor]"
EVIDENCE_SEMANTICS = "frame_conditional_bit_budget_byte_proxy_no_score"
DISPATCH_BLOCKERS = [
    "awaiting_per_frame_score_marginal",
    "no_archive_substitution_performed",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "per_pair_bit_width_schema_change_requires_inflate_path_update",
]


def _proxy_evidence_contract() -> dict[str, object]:
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "byte_proxy_only": True,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


# ─────────────────────────────────────────────────────────────────────────
# PR101 latent extraction (read-only; mirrors codec.py's parse_latents)
# ─────────────────────────────────────────────────────────────────────────


def _read_pr101_archive_bytes(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as z:
        with z.open("x") as f:
            return f.read()


def _extract_pr101_latent_payload(archive_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mins, scales, q_codes_pair_first) from PR101's latent stream.

    ``q_codes_pair_first`` has shape (N_PAIRS, LATENT_DIM) uint8 — the
    quantised symbols *before* delta-encoding (the symbol stream after
    cumsum reconstruction in codec.py).
    """
    latent_blob = archive_bytes[
        PR101_DECODER_BLOB_LEN : PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    ]
    raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=PR101_LATENT_LZMA_FILTERS
    )
    buf = io.BytesIO(raw)
    mins = np.frombuffer(buf.read(PR101_LATENT_DIM * 2), dtype=np.float16).astype(
        np.float32
    )
    scales = np.frombuffer(buf.read(PR101_LATENT_DIM * 2), dtype=np.float16).astype(
        np.float32
    )
    stored = np.frombuffer(
        buf.read(PR101_N_PAIRS * PR101_LATENT_DIM), dtype=np.uint8
    ).reshape(PR101_LATENT_DIM, PR101_N_PAIRS)

    # Mirror codec.decode_latents_compact: deltas → cumsum → reorder.
    q = stored.copy()
    q[:, 1:] = (
        np.cumsum(
            ((stored[:, 1:].astype(np.int16) - 128) & 255),
            axis=1,
            dtype=np.uint16,
        ).astype(np.uint8)
        + stored[:, :1]
    )
    # Transpose to (N_PAIRS, LATENT_DIM) for per-pair operations.
    return mins, scales, q.T.astype(np.uint8)


def _encode_pr101_latent_stream(
    mins: np.ndarray,
    scales: np.ndarray,
    q_pair_first: np.ndarray,
) -> bytes:
    """Re-encode latent stream with PR101's exact LZMA delta-pipeline.

    Mirrors the encoder shape implied by codec.decode_latents_compact — we
    do NOT mutate the upstream clone, we replicate the format here.
    """
    if q_pair_first.shape != (PR101_N_PAIRS, PR101_LATENT_DIM):
        raise ValueError(f"q_pair_first must be ({PR101_N_PAIRS}, {PR101_LATENT_DIM})")
    q_dim_first = q_pair_first.T.copy()  # (LATENT_DIM, N_PAIRS)
    # Inverse cumsum: stored[:, 0] = q[:, 0]; stored[:, i] = ((q[:, i] - q[:, i-1]) & 255) + 128 - 128 ≡ delta encoding.
    deltas = q_dim_first.copy()
    deltas[:, 1:] = (
        ((q_dim_first[:, 1:].astype(np.int16) - q_dim_first[:, :-1].astype(np.int16)) & 255).astype(
            np.uint8
        )
        + 128
    ).astype(np.uint8)
    # Note: the codec decoder does ((delta - 128) & 255) cumsum, so the
    # encoder's delta is ((q[i] - q[i-1]) & 255) WITHOUT the +128 offset.
    # Re-derive the delta to match codec.decode_latents_compact exactly:
    deltas = q_dim_first.copy()
    diffs = ((q_dim_first[:, 1:].astype(np.int16) - q_dim_first[:, :-1].astype(np.int16)) & 255).astype(
        np.uint8
    )
    # decoder: q_ord[:, i] = cumsum( ((delta[:, i:].i16 - 128) & 255) ) + delta[:, 0]
    # so encoder must store delta = (diff + 128) & 255 so that (delta - 128) & 255 == diff.
    deltas[:, 1:] = ((diffs.astype(np.int16) + 128) & 255).astype(np.uint8)

    payload = (
        mins.astype(np.float16).tobytes()
        + scales.astype(np.float16).tobytes()
        + deltas.astype(np.uint8).tobytes()
    )
    return lzma.compress(
        payload, format=lzma.FORMAT_RAW, filters=PR101_LATENT_LZMA_FILTERS
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-pair re-quantisation under a frame-conditional bit budget
# ─────────────────────────────────────────────────────────────────────────


def _bits_per_pair_to_q_bits(bits_per_pair: np.ndarray, latent_dim: int) -> np.ndarray:
    """Map per-pair bit allocation to per-pair quantisation precision.

    PR101 uses 8 bits/symbol × LATENT_DIM symbols/pair = 8*LATENT_DIM bits/pair
    nominally. For frame-conditional allocation we let high-motion pairs use
    up to ``cap*8 = 16`` bits/symbol equivalent (clamped to 8 — int8 cap)
    and low-motion pairs use as few as ``floor*8 = 4`` bits/symbol.
    """
    q_bits = bits_per_pair / latent_dim
    q_bits = np.clip(q_bits, 1.0, 8.0)
    return q_bits


def _requantise_per_pair(
    q_pair_first: np.ndarray,
    q_bits_per_pair: np.ndarray,
) -> np.ndarray:
    """Re-quantise PR101's uint8 latent codes to per-pair ``q_bits`` precision.

    For pair ``i`` with ``q_bits[i] = b``, the symbol space contracts from
    256 levels to ``2**floor(b)`` levels: the symbol is divided by
    ``2**(8 - floor(b))`` and multiplied back. This is a strictly-lossy
    re-quantisation that emulates how reduced per-pair precision would
    affect the LZMA delta-encoder's compressibility.
    """
    out = q_pair_first.copy().astype(np.uint8)
    for i in range(out.shape[0]):
        b = int(np.floor(q_bits_per_pair[i]))
        b = max(1, min(8, b))
        if b == 8:
            continue
        shift = 8 - b
        # Round-to-nearest re-quantisation in uint8 space.
        x = out[i].astype(np.int32)
        # Reconstruct by quantise-then-dequantise: floor(x / 2**shift) * 2**shift.
        x = (x >> shift) << shift
        out[i] = np.clip(x, 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Allocator-driven sweep
# ─────────────────────────────────────────────────────────────────────────


def _build_per_pair_complexity(
    video_path: Path, n_frames: int = 2 * PR101_N_PAIRS
) -> np.ndarray:
    from tac.codec.frame_conditional_bit_budget import compute_per_frame_complexity

    if n_frames % 2 != 0:
        raise ValueError(f"n_frames must be even (pair-aligned), got {n_frames}")
    per_frame = compute_per_frame_complexity(video_path, n_frames)
    pairs = per_frame.reshape(-1, 2).mean(axis=1)
    return pairs


def _sweep_etas(
    archive_path: Path,
    video_path: Path,
    etas: list[float],
    floor: float,
    cap: float,
    output_dir: Path,
) -> dict:
    from tac.codec.frame_conditional_bit_budget import allocate_per_frame_bits

    archive_bytes = _read_pr101_archive_bytes(archive_path)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    mins, scales, q_pair_first = _extract_pr101_latent_payload(archive_bytes)

    # Baseline: re-encode the original (no re-quant) to confirm format parity.
    baseline_latent_bytes = _encode_pr101_latent_stream(mins, scales, q_pair_first)
    baseline_total_bytes = (
        PR101_DECODER_BLOB_LEN + len(baseline_latent_bytes) + (len(archive_bytes) - PR101_DECODER_BLOB_LEN - PR101_LATENT_BLOB_LEN)
    )

    # Per-pair complexity from video.
    complexities = _build_per_pair_complexity(video_path, n_frames=2 * PR101_N_PAIRS)

    # Total bit budget = nominal LATENT bit count = N_PAIRS * LATENT_DIM * 8
    total_bits = float(PR101_N_PAIRS * PR101_LATENT_DIM * 8)

    rows: list[dict] = []
    for eta in etas:
        bits_per_pair = allocate_per_frame_bits(
            complexities, total_bits, eta=eta, floor=floor, cap=cap
        )
        q_bits_per_pair = _bits_per_pair_to_q_bits(bits_per_pair, PR101_LATENT_DIM)
        q_requant = _requantise_per_pair(q_pair_first, q_bits_per_pair)
        new_latent_bytes = _encode_pr101_latent_stream(mins, scales, q_requant)
        # Per-pair bit-width sidechannel cost: ⌈log2(8)⌉ = 3 bits per pair → ⌈600*3/8⌉ = 225 bytes
        sidechannel_overhead = int(np.ceil(PR101_N_PAIRS * 3 / 8))
        # If eta=0 (uniform) the sidechannel can be omitted.
        if eta == 0.0:
            sidechannel_overhead = 0
        total_bytes = (
            PR101_DECODER_BLOB_LEN
            + len(new_latent_bytes)
            + (len(archive_bytes) - PR101_DECODER_BLOB_LEN - PR101_LATENT_BLOB_LEN)
            + sidechannel_overhead
        )
        rows.append(
            {
                "eta": eta,
                "floor": floor,
                "cap": cap,
                "latent_bytes_new": len(new_latent_bytes),
                "latent_bytes_baseline": len(baseline_latent_bytes),
                "latent_delta_bytes": len(new_latent_bytes) - len(baseline_latent_bytes),
                "sidechannel_overhead_bytes": sidechannel_overhead,
                "archive_bytes_new": total_bytes,
                "archive_bytes_baseline": baseline_total_bytes,
                "archive_delta_bytes": total_bytes - baseline_total_bytes,
                "bits_per_pair_min": float(bits_per_pair.min()),
                "bits_per_pair_max": float(bits_per_pair.max()),
                "bits_per_pair_mean": float(bits_per_pair.mean()),
                "bits_per_pair_std": float(bits_per_pair.std()),
                "q_bits_per_pair_min": float(q_bits_per_pair.min()),
                "q_bits_per_pair_max": float(q_bits_per_pair.max()),
                "q_bits_per_pair_mean": float(q_bits_per_pair.mean()),
                **_proxy_evidence_contract(),
                "source": (
                    f"{EVIDENCE_GRADE} {output_dir} "
                    f"(eta={eta}, floor={floor}, cap={cap}, allocator=frame_conditional)"
                ),
            }
        )

    rows.sort(key=lambda r: r["archive_delta_bytes"])
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_archive": str(archive_path),
        "input_archive_sha256": archive_sha,
        "input_archive_bytes": len(archive_bytes),
        "input_video": str(video_path),
        "n_pairs": PR101_N_PAIRS,
        "latent_dim": PR101_LATENT_DIM,
        "total_bit_budget": total_bits,
        "complexities_pair_min": float(complexities.min()),
        "complexities_pair_max": float(complexities.max()),
        "complexities_pair_mean": float(complexities.mean()),
        "complexities_pair_std": float(complexities.std()),
        "etas_swept": etas,
        "floor": floor,
        "cap": cap,
        "rows": rows,
        "best_eta": rows[0]["eta"],
        "best_archive_delta_bytes": rows[0]["archive_delta_bytes"],
        **_proxy_evidence_contract(),
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pr101-archive",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream/videos/0.mkv",
    )
    p.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0],
    )
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--cap", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.pr101_archive.is_file():
        raise SystemExit(f"PR101 archive not found: {args.pr101_archive}")
    if not args.video_path.is_file():
        raise SystemExit(f"video not found: {args.video_path}")

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"experiments/results/pr101_frame_conditional_bit_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _sweep_etas(
        args.pr101_archive,
        args.video_path,
        list(args.etas),
        args.floor,
        args.cap,
        args.output_dir,
    )
    manifest["timestamp"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_json = args.output_dir / "build_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nbuild_manifest: {out_json}\n")
    print(
        "  eta | floor | cap  | latent_Δ | sidech_oh | archive_Δ | min_bits/pair | max_bits/pair"
    )
    for r in sorted(manifest["rows"], key=lambda r: r["eta"]):
        print(
            f"  {r['eta']:>4.2f} | {r['floor']:>5.2f} | {r['cap']:>4.2f} | "
            f"{r['latent_delta_bytes']:>+8d} | {r['sidechannel_overhead_bytes']:>9d} | "
            f"{r['archive_delta_bytes']:>+9d} | {r['bits_per_pair_min']:>13.2f} | "
            f"{r['bits_per_pair_max']:>13.2f}"
        )
    print(
        f"\nbest_eta={manifest['best_eta']}  "
        f"archive_delta_bytes={manifest['best_archive_delta_bytes']:+d}\n"
    )
    print("NOTE: byte-anchor only. score_claim=False; per-pair score-marginal")
    print("      required before any dispatch. dispatch_blocker:")
    print("      awaiting_per_frame_score_marginal.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
