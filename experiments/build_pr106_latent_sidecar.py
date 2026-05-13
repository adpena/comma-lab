#!/usr/bin/env python3
"""Build PR106 + per-pair latent-correction sidecar archive.

Pattern: ports PR100's hnerv_lc_v2 sidecar idea (1.2KB per-pair correction blob)
onto PR106's belt_and_suspenders archive (28-dim x 600-pair latents). The
PR100-vs-PR105 empirical sidecar gain (-0.00218 score at +1124B) is tracked as a
planning target, not as a prediction for heuristic smoke output.

Wire format (lane_pr106_latent_sidecar 0.bin):

    magic(1B) = 0xFE
    format_id(1B) = 0x01
    pr106_len(4B uint32 LE)
    pr106_bytes(pr106_len B)              # original PR106 0.bin bytes verbatim
    sidecar_len(2B uint16 LE)
    sidecar_bytes(sidecar_len B)          # brotli-compressed PR100 sidecar wire format:
                                          #   u16 n_pairs
                                          #   per pair: u8 dim_idx (0..27, or 255=no-op),
                                          #             i8 delta_quantized (real = i8 * 0.01)

Per-pair (dim, delta) selection strategy:
  Default: heuristic nonzero smoke signal. For each pair, choose the largest
    magnitude latent dimension and emit a one-quantum signed nudge toward zero.
    This validates the wrapper/sidecar/runtime path but is not a score-aware
    optimizer.

  Score-table mode: reduce a precomputed CUDA scorer table over (dim_idx,
    delta_q) candidates into charged sidecar bytes. The table is compress-time
    evidence only; generated artifacts remain score_claim=false and
    ready_for_exact_eval_dispatch=false until exact CUDA auth eval scores them.

CUDA REQUIRED per CLAUDE.md MPS-auth-eval-is-NOISE rule. CPU is acceptable
ONLY for smoke tests labeled [advisory only] via --device cpu --smoke.

Usage:
    .venv/bin/python experiments/build_pr106_latent_sidecar.py \\
        --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \\
        --output-dir experiments/results/lane_pr106_latent_sidecar_$(date -u +%Y%m%dT%H%M%SZ) \\
        --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import io
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
PR106_SRC_CANDIDATE_PATHS = (
    REPO_ROOT / "submissions" / "pr106_latent_sidecar" / "src",
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex"
    / "source/submissions/belt_and_suspenders/src",
)
for candidate in reversed(PR106_SRC_CANDIDATE_PATHS):
    if candidate.exists():
        sys.path.insert(0, str(candidate.resolve()))

# Imported AFTER path insertion: PR106's reference codec + decoder model.
from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]

from tac.packet_compiler.pr106_latent_sidecar_selection import (
    build_latent_candidate_grid,
    choose_latent_corrections_from_score_table_file,
    choose_latent_corrections_from_scores,
    latent_candidate_grid_npy_sha256,
    validate_score_table_manifest,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_NO_OP_DIM,
    decode_brotli_dim_delta_sidecar_payload,
    encode_brotli_dim_delta_sidecar_payload,
)
from tac.repo_io import json_text

SIDECAR_MAGIC = 0xFE
SIDECAR_FORMAT_ID = 0x01
DELTA_SCALE = 0.01  # [inherited:PR100/hnerv_lc_v2/sidecar.py] int8 quant: real_delta = i8 * 0.01 (range ±1.27)
NO_OP_DIM = PR106_NO_OP_DIM

CAMERA_H, CAMERA_W = 874, 1164
NATIVE_H, NATIVE_W = 384, 512

__all__ = [
    "DELTA_SCALE",
    "NO_OP_DIM",
    "apply_sidecar_corrections",
    "build_latent_candidate_grid",
    "build_sidecar_archive_blob",
    "choose_latent_corrections_from_score_table_file",
    "choose_latent_corrections_from_scores",
    "decode_sidecar_corrections",
    "encode_sidecar_corrections",
    "latent_candidate_grid_npy_sha256",
    "parse_sidecar_archive_blob",
    "validate_score_table_manifest",
]


# =====================================================================
# Sidecar wire format (mirrors PR100 hnerv_lc_v2/sidecar.py exactly)
# =====================================================================


def encode_sidecar_corrections(dim_arr: np.ndarray, delta_q_arr: np.ndarray) -> bytes:
    """Serialize per-pair (dim, delta) array to brotli-compressed blob.

    Mirrors PR100 hnerv_lc_v2/sidecar.py::encode_corrections wire format
    so future PR100-style submissions can adopt this archive shape.
    """
    return encode_brotli_dim_delta_sidecar_payload(dim_arr, delta_q_arr, quality=11)


def decode_sidecar_corrections(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_sidecar_corrections. Returns (dim_arr uint8, delta_q_arr int8)."""
    return decode_brotli_dim_delta_sidecar_payload(blob)


def apply_sidecar_corrections(
    latents: torch.Tensor,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    scale: float = DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor."""
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == NO_OP_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


# =====================================================================
# Archive wire format (PR106 verbatim + sidecar appended)
# =====================================================================


def build_sidecar_archive_blob(pr106_bytes: bytes, sidecar_blob: bytes) -> bytes:
    """Produce 0.bin bytes for lane_pr106_latent_sidecar.

    Layout:
        magic(1B) = 0xFE
        format_id(1B) = 0x01
        pr106_len(4B uint32 LE)
        pr106_bytes(pr106_len)
        sidecar_len(2B uint16 LE)
        sidecar_bytes(sidecar_len)

    pr106_bytes is held verbatim so the inflate path can simply slice + delegate
    to the PR106 parser; no re-encoding risk.
    """
    if len(sidecar_blob) > 0xFFFF:
        raise ValueError(
            f"sidecar blob too large for u16 length field: {len(sidecar_blob)} bytes"
        )
    out = io.BytesIO()
    out.write(bytes([SIDECAR_MAGIC]))
    out.write(bytes([SIDECAR_FORMAT_ID]))
    out.write(struct.pack("<I", len(pr106_bytes)))
    out.write(pr106_bytes)
    out.write(struct.pack("<H", len(sidecar_blob)))
    out.write(sidecar_blob)
    return out.getvalue()


def parse_sidecar_archive_blob(bin_bytes: bytes) -> tuple[bytes, bytes]:
    """Inverse of build_sidecar_archive_blob. Returns (pr106_bytes, sidecar_blob)."""
    if not bin_bytes:
        raise ValueError("empty archive")
    if bin_bytes[0] != SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{bin_bytes[0]:02X}, expected 0x{SIDECAR_MAGIC:02X}"
        )
    if bin_bytes[1] != SIDECAR_FORMAT_ID:
        raise ValueError(
            f"sidecar format_id mismatch: got 0x{bin_bytes[1]:02X}, expected 0x{SIDECAR_FORMAT_ID:02X}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    pr106_bytes = bin_bytes[pos : pos + pr106_len]
    pos += pr106_len
    if pos + 2 > len(bin_bytes):
        raise ValueError("sidecar archive truncated before sidecar_len")
    (sidecar_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    sidecar_blob = bin_bytes[pos : pos + sidecar_len]
    pos += sidecar_len
    if pos != len(bin_bytes):
        raise ValueError(
            f"sidecar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
        )
    return pr106_bytes, sidecar_blob


# =====================================================================
# Score scaffolding (predict + measure)
# =====================================================================


def _load_pr106_components(
    pr106_bytes: bytes,
) -> tuple[dict, torch.Tensor, dict]:
    """Re-use the PR106 parser to recover (state_dict, latents, meta)."""
    return parse_packed_archive(pr106_bytes)


def _build_decoder(state_dict: dict, meta: dict, device: torch.device) -> HNeRVDecoder:
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(state_dict)
    decoder.eval()
    return decoder


def _try_load_scorers(device: torch.device):
    """Try to load PoseNet+SegNet scorers for CUDA gradient-guided search.

    Returns (posenet, segnet) or (None, None) if not available (smoke / CPU paths
    don't strictly need the scorers — we fall back to a heuristic per-pair greedy
    that targets latent reconstruction self-consistency rather than scorer loss).
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "upstream"))
        from modules import PoseNet, SegNet  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"[sidecar-build] WARN: scorer modules unavailable ({exc}); "
              f"falling back to self-consistency heuristic.")
        return None, None
    try:
        posenet = PoseNet().to(device).eval()
        segnet = SegNet().to(device).eval()
        # Best effort: load weights if pinned in upstream; else random-init scorers
        # are still useful for smoke pipeline-correctness validation.
        return posenet, segnet
    except Exception as exc:
        print(f"[sidecar-build] WARN: scorer load failed ({exc}); "
              f"falling back to self-consistency heuristic.")
        return None, None


# =====================================================================
# Per-pair (dim, delta) search strategies
# =====================================================================


def _heuristic_self_consistency_search(
    decoder: HNeRVDecoder,
    latents: torch.Tensor,
    *,
    device: torch.device,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """For each pair, emit a small nonzero heuristic latent nudge.

    Without scorers loaded, we use a self-consistency proxy: minimize the L2
    distance between decoder(latents) and decoder(latents + correction). This
    isn't a real distortion target but exercises the full encode/decode path
    so the wire format and search loop are validated end-to-end. The CUDA
    auth-eval Stage 3 of remote_lane_pr106_latent_sidecar.sh provides the
    real measurement.
    """
    n_pairs, latent_dim = latents.shape
    dim_arr = np.zeros(n_pairs, dtype=np.uint8)
    delta_q_arr = np.zeros(n_pairs, dtype=np.int8)

    # Heuristic: pick the latent dimension with the largest absolute value
    # for each pair (high-magnitude dims drive the decoder hardest, so a
    # small correction there has the strongest delta in pixel space).
    abs_lat = latents.detach().cpu().abs().numpy()
    candidate_dim = abs_lat.argmax(axis=1).astype(np.uint8)

    # Quantize a one-step nudge toward zero. This is intentionally not a score
    # objective; it is a nontrivial wire-format smoke signal. Use the signed
    # latent values, not abs_lat, and avoid half-step rounding-to-zero.
    lat_np = latents.detach().cpu().numpy()
    selected_values = lat_np[np.arange(n_pairs), candidate_dim]
    delta_q = (-np.sign(selected_values)).astype(np.int8)

    # If top_k specified, only correct top-k pairs (by latent magnitude).
    if top_k is not None and top_k < n_pairs:
        # Pick pairs with largest peak |latent|.
        peak_per_pair = abs_lat.max(axis=1)
        top_pairs = np.argsort(peak_per_pair)[-top_k:]
        keep_mask = np.zeros(n_pairs, dtype=bool)
        keep_mask[top_pairs] = True
        candidate_dim = np.where(keep_mask, candidate_dim, NO_OP_DIM).astype(np.uint8)
        delta_q = np.where(keep_mask, delta_q, 0).astype(np.int8)

    dim_arr = candidate_dim
    delta_q_arr = delta_q

    histogram = {int(d): int((dim_arr == d).sum()) for d in np.unique(dim_arr)}
    correction_mask = (dim_arr != NO_OP_DIM) & (delta_q_arr != 0)
    n_corrections = int(correction_mask.sum())
    nonzero_delta_count = int(np.count_nonzero(delta_q_arr))
    diagnostics = {
        "search_mode": "self_consistency_heuristic",
        "n_pairs": int(n_pairs),
        "n_corrections": n_corrections,
        "n_no_op": int(n_pairs - n_corrections),
        "nonzero_delta_count": nonzero_delta_count,
        "dim_histogram": histogram,
        "delta_q_min": int(delta_q_arr.min()),
        "delta_q_max": int(delta_q_arr.max()),
        "delta_q_mean": float(delta_q_arr.mean()),
        "scorer_available": False,
    }
    return dim_arr, delta_q_arr, diagnostics


# =====================================================================
# Main
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True,
                        help="Path to PR106 archive.zip (containing 0.bin).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write sidecar_archive.zip + build_metadata.json.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="cuda required for production; cpu only for --smoke.")
    parser.add_argument("--smoke", action="store_true",
                        help="Skip CUDA gating; runs self-consistency heuristic only "
                             "(produces a wire-format-valid archive but score is not "
                             "measured here — Stage 3 of remote_lane runbook does that).")
    parser.add_argument("--top-k", type=int, default=600,
                        help="Limit how many pairs receive a correction (default 600 = all).")
    parser.add_argument("--search-mode", choices=["heuristic", "score_table"], default="heuristic",
                        help="Search strategy. 'heuristic' = self-consistency proxy "
                             "(fast scaffolding); 'score_table' = reduce a precomputed "
                             "CUDA scorer table into charged sidecar bytes.")
    parser.add_argument("--score-table-npy", type=Path, default=None,
                        help="Required for --search-mode score_table. Shape must be "
                             "(n_pairs, 1 + latent_dim * 2 * delta_radius).")
    parser.add_argument("--score-table-manifest", type=Path, default=None,
                        help="Optional JSON provenance for the CUDA scorer table. The "
                             "builder validates it when present; exact CUDA auth eval "
                             "is still required on the emitted archive.")
    parser.add_argument("--delta-radius", type=int, default=1,
                        help="Integer latent delta grid radius for score_table mode.")
    args = parser.parse_args()

    if args.device == "cpu" and not args.smoke:
        sys.exit(
            "FATAL: --device cpu requires --smoke flag (per CLAUDE.md MPS-auth-eval-is-NOISE: "
            "CPU output is [advisory only])."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit(
            "FATAL: --device cuda requested but torch.cuda.is_available()=False. "
            "Use --device cpu --smoke for local scaffolding."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    started_at = time.time()

    # Stage A: load PR106 archive bytes verbatim.
    if not args.source_archive.is_file():
        sys.exit(f"FATAL: source archive not found at {args.source_archive}")
    with zipfile.ZipFile(args.source_archive) as z:
        if "0.bin" not in z.namelist():
            sys.exit(f"FATAL: 0.bin missing from {args.source_archive}")
        pr106_bytes = z.read("0.bin")
    pr106_sha = hashlib.sha256(pr106_bytes).hexdigest()
    print(f"[sidecar-build] PR106 0.bin: {len(pr106_bytes)} bytes, sha256={pr106_sha[:16]}...")

    # Stage B: decode PR106 to recover (state_dict, latents, meta).
    state_dict, latents, meta = _load_pr106_components(pr106_bytes)
    print(f"[sidecar-build] PR106 decoded: {len(state_dict)} tensors, "
          f"latents shape={tuple(latents.shape)}, meta={meta}")

    # Stage C: build decoder (validates state_dict shape correctness).
    decoder = _build_decoder(state_dict, meta, device)
    latents = latents.to(device)
    print(f"[sidecar-build] HNeRV decoder built on {device}")

    # Stage D: run search to pick per-pair (dim, delta).
    if args.search_mode == "heuristic":
        dim_arr, delta_q_arr, diagnostics = _heuristic_self_consistency_search(
            decoder, latents, device=device, top_k=args.top_k
        )
        score_table_metadata: dict[str, object] | None = None
    elif args.search_mode == "score_table":
        if args.score_table_npy is None:
            sys.exit("FATAL: --score-table-npy is required when --search-mode score_table")
        dim_arr, delta_q_arr, diagnostics = choose_latent_corrections_from_score_table_file(
            args.score_table_npy,
            n_pairs=int(meta["n_pairs"]),
            latent_dim=int(meta["latent_dim"]),
            delta_radius=int(args.delta_radius),
            top_k=args.top_k,
        )
        score_table_manifest: dict[str, object] | None = None
        if args.score_table_manifest is not None:
            score_table_manifest = validate_score_table_manifest(
                args.score_table_manifest,
                score_table_npy=args.score_table_npy,
                source_archive=args.source_archive,
                n_pairs=int(meta["n_pairs"]),
                latent_dim=int(meta["latent_dim"]),
                delta_radius=int(args.delta_radius),
                candidate_count=int(diagnostics["candidate_count"]),
            )
        score_table_metadata = {
            "score_table_npy_path": str(args.score_table_npy),
            "score_table_npy_bytes": int(args.score_table_npy.stat().st_size),
            "score_table_npy_sha256": hashlib.sha256(args.score_table_npy.read_bytes()).hexdigest(),
            "score_table_manifest_path": str(args.score_table_manifest) if args.score_table_manifest else None,
            "score_table_manifest_sha256": (
                hashlib.sha256(args.score_table_manifest.read_bytes()).hexdigest()
                if args.score_table_manifest else None
            ),
            "score_table_manifest_validated": score_table_manifest is not None,
            "score_table_manifest_schema": (
                score_table_manifest.get("manifest_schema") if score_table_manifest is not None else None
            ),
            "validated_source_archive_sha256_match": (
                score_table_manifest.get("validated_source_archive_sha256_match")
                if score_table_manifest is not None
                else None
            ),
            "validated_source_zero_bin_sha256_match": (
                score_table_manifest.get("validated_source_zero_bin_sha256_match")
                if score_table_manifest is not None
                else None
            ),
            "score_table_is_score_claim": False,
            "score_table_required_provenance": (
                "CUDA scorer table must be generated against the exact source archive; "
                "this builder only reduces measured table entries into charged bytes."
            ),
        }
    else:  # pragma: no cover - argparse owns choices
        raise AssertionError(f"unhandled search_mode={args.search_mode!r}")

    print(f"[sidecar-build] search complete: {diagnostics['n_corrections']} pairs corrected, "
          f"{diagnostics['n_no_op']} no-op (top_k={args.top_k})")

    # Stage E: encode sidecar blob.
    sidecar_blob = encode_sidecar_corrections(dim_arr, delta_q_arr)
    print(f"[sidecar-build] sidecar blob: {len(sidecar_blob)} bytes (brotli q11)")

    # Stage F: encode/decode round-trip sanity (catches wire-format bugs locally).
    rt_dim, rt_delta = decode_sidecar_corrections(sidecar_blob)
    assert np.array_equal(rt_dim, np.where(delta_q_arr == 0, NO_OP_DIM, dim_arr)), \
        "sidecar round-trip dim_arr mismatch"
    assert np.array_equal(rt_delta, delta_q_arr), "sidecar round-trip delta_q_arr mismatch"
    print("[sidecar-build] sidecar encode/decode round-trip OK")

    # Stage G: build full archive blob.
    archive_blob = build_sidecar_archive_blob(pr106_bytes, sidecar_blob)
    pr106_back, sidecar_back = parse_sidecar_archive_blob(archive_blob)
    assert pr106_back == pr106_bytes, "PR106 bytes mutated by sidecar wrapper"
    assert sidecar_back == sidecar_blob, "sidecar bytes mutated by archive wrapper"
    print(f"[sidecar-build] archive blob: {len(archive_blob)} bytes; round-trip OK")

    # Stage H: write archive.zip with deterministic ZIP framing.
    archive_path = args.output_dir / "sidecar_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, archive_blob)
    archive_size = archive_path.stat().st_size
    pr106_size = args.source_archive.stat().st_size
    delta = archive_size - pr106_size
    score_delta = 25.0 * delta / 37545489.0
    print(f"[sidecar-build] wrote {archive_path}: {archive_size} bytes "
          f"(PR106 was {pr106_size}; delta {delta:+d}; rate-component score Δ {score_delta:+.6f})")

    # Stage I: write sidecar bin alongside (operator may attach to a different archive).
    sidecar_bin_path = args.output_dir / "sidecar.bin"
    sidecar_bin_path.write_bytes(sidecar_blob)

    # Stage J: write build metadata.
    elapsed = time.time() - started_at
    recorded_wall_clock = 0.0 if args.smoke else elapsed
    dispatch_blockers = [
        "requires_exact_cuda_auth_eval",
    ]
    if args.search_mode == "heuristic":
        dispatch_blockers.extend(
            [
                "cpu_smoke_or_heuristic_builder_output",
                "requires_scorer_backed_cuda_latent_search",
            ]
        )
    if args.search_mode == "score_table" and args.score_table_manifest is None:
        dispatch_blockers.append("missing_cuda_score_table_manifest")

    metadata = {
        "lane_id": "lane_pr106_latent_sidecar",
        "wall_clock_seconds": recorded_wall_clock,
        "wall_clock_seconds_note": (
            "omitted_for_deterministic_smoke_manifest" if args.smoke else "observed"
        ),
        "device": args.device,
        "smoke_mode": bool(args.smoke),
        "search_mode": args.search_mode,
        "top_k": int(args.top_k),
        "source_archive": str(args.source_archive),
        "source_archive_bytes": pr106_size,
        "source_archive_sha256": hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
        "pr106_bin_bytes": len(pr106_bytes),
        "pr106_bin_sha256": pr106_sha,
        "sidecar_path": str(sidecar_bin_path),
        "sidecar_bytes": len(sidecar_blob),
        "archive_path": str(archive_path),
        "archive_blob_bytes": len(archive_blob),
        "archive_zip_bytes": archive_size,
        "delta_bytes_vs_pr106": delta,
        "rate_component_score_delta_vs_pr106": score_delta,
        "planning_target_total_score_delta_vs_pr106": -0.00218,  # PR100-vs-PR105 extrapolation
        "planning_target_source": "PR100-vs-PR105 extrapolation; not a heuristic-smoke prediction",
        "predicted_total_score_delta_vs_pr106": None,
        "predicted_total_score": None,
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
        "score_table": score_table_metadata,
        "evidence_grade": "empirical_build_only",
        "diagnostics": diagnostics,
        "tag": "[design-validation]" if args.smoke else "[empirical:pending-stage-3]",
        "council_status": "PRE_REGISTERED",
        "next_step": (
            "Dispatch contest-CUDA via scripts/remote_lane_pr106_latent_sidecar.sh — "
            "Stage 3 runs the scorer-driven (dim, delta) refinement and contest_auth_eval."
        ),
    }
    metadata_path = args.output_dir / "build_metadata.json"
    metadata_path.write_text(json_text(metadata), encoding="utf-8")
    print(f"[sidecar-build] wrote {metadata_path}")
    print(f"[sidecar-build] DONE in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
