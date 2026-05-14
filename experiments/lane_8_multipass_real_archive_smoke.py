# SPDX-License-Identifier: MIT
"""Lane 8 multi-pass — real-archive empirical smoke (offline / no GPU).

Loads the Lane G v3 anchor archive (`experiments/results/lane_g_v3_landed/
archive_lane_g_v3.zip`) and runs ``MultiPassCompressor`` against it with a
BYTE-PROXY scorer that approximates the rate-distortion frontier without
loading any neural network. The result is a sanity-check on the
``MultiPassCompressor`` plumbing using the EXACT bytes of the shipped
Lane G v3 baseline, not GPU validation.

GPU validation (Phase G) requires Vast.ai 4090 dispatch via
``scripts/remote_lane_8_multipass.sh``. This script is the [empirical:...]
artifact for the Level-3 gate.

Output: ``reports/lane_8_multipass_real_archive.json`` containing the
PassRecord stream + a delta-bytes vs Lane G v3 baseline summary.

NON-NEGOTIABLES enforced:
- Never claims [contest-CUDA] — output is tagged [empirical:offline-byte-proxy].
- No scorer load. No CUDA forward pass.
- Reads the anchor archive bytes; proxies a score curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tac.multipass_compressor import (
    DEFAULT_EPS,
    DEFAULT_MAX_PASSES,
    MultiPassCompressor,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = REPO / "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip"
DEFAULT_REPORT = REPO / "reports/lane_8_multipass_real_archive.json"


def _byte_proxy_encoder(anchor_bytes: bytes):
    """Return a deterministic encoder closure: smaller mask_crf reduces the
    archive size by trimming bytes from the END of the anchor (a stand-in
    for AV1 CRF compression). The byte trimming is BIASED to keep the
    archive header intact (first 64KB) — the codec metadata stays valid.

    NOTE: the produced bytes are NOT a valid contest archive; they are a
    PROXY. Real GPU validation must use the actual encoder via
    ``experiments/pipeline.py compress --multipass``.
    """
    HEADER_KEEP = 64 * 1024  # 64KB of the anchor preserved
    anchor_len = len(anchor_bytes)

    def encoder(state: object, params: dict) -> bytes:
        # mask_crf in [10, 60] -> trim 0% to 30% of the bytes after HEADER_KEEP
        crf = float(params.get("mask_crf", 50.0))
        crf_norm = (crf - 10.0) / (60.0 - 10.0)  # 0..1
        crf_norm = max(0.0, min(1.0, crf_norm))
        keep_after_header = int(
            (anchor_len - HEADER_KEEP) * (1.0 - 0.30 * crf_norm)
        )
        return anchor_bytes[: HEADER_KEEP + keep_after_header]

    return encoder


def _byte_proxy_scorer(baseline_score: float, baseline_bytes: int):
    """Proxy score: combines a rate term + a distortion term.

    rate_term = 25 * archive_bytes / 37545489  (the contest formula)
    distortion_term = baseline_distortion + alpha * (1 / archive_bytes_norm) ** 2
        where the second term captures "compression too aggressive →
        distortion explodes" behavior.

    Tuned so that the baseline (anchor) returns approximately ``baseline_score``
    and increasing CRF eventually regresses the score.
    """
    ORIGINAL_VIDEO_BYTES = 37_545_489
    baseline_rate = 25.0 * baseline_bytes / ORIGINAL_VIDEO_BYTES
    baseline_distortion = max(0.0, baseline_score - baseline_rate)

    def scorer(archive: bytes) -> float:
        rate = 25.0 * len(archive) / ORIGINAL_VIDEO_BYTES
        # Distortion increases with compression ratio (smaller archive → more
        # distortion). Square-law penalty so over-compression regresses.
        compression_ratio = baseline_bytes / max(1, len(archive))
        distortion = baseline_distortion + 0.05 * (compression_ratio - 1.0) ** 2
        return rate + distortion

    return scorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor", type=Path, default=DEFAULT_ANCHOR,
        help="Anchor archive.zip (default: Lane G v3 landed)",
    )
    parser.add_argument(
        "--baseline-score", type=float, default=1.05,
        help="Baseline contest-CUDA score for the anchor (default 1.05)",
    )
    parser.add_argument(
        "--max-passes", type=int, default=DEFAULT_MAX_PASSES,
    )
    parser.add_argument(
        "--target-score", type=float, default=1.04,
        help="Target score (lower = better; default = baseline - 0.01)",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
    )
    args = parser.parse_args()

    if not args.anchor.exists():
        raise SystemExit(
            f"Anchor archive not found: {args.anchor}. Provide --anchor "
            f"or run from the repo root with the canonical Lane G v3 "
            f"artifact present."
        )
    anchor_bytes = args.anchor.read_bytes()
    baseline_bytes = len(anchor_bytes)

    encoder = _byte_proxy_encoder(anchor_bytes)
    scorer = _byte_proxy_scorer(args.baseline_score, baseline_bytes)

    print(f"[lane-8-multipass-smoke] anchor={args.anchor} ({baseline_bytes:,} bytes)")
    print(
        f"[lane-8-multipass-smoke] baseline_score={args.baseline_score:.4f} "
        f"target={args.target_score:.4f} max_passes={args.max_passes}"
    )

    t0 = time.monotonic()
    result = MultiPassCompressor(
        target_score=args.target_score,
        max_passes=args.max_passes,
        eps=DEFAULT_EPS,
        regression_guard=True,
        initial_params={
            "mask_crf": 50.0,
            "pose_q_bits": 8.0,
            "block_fp_block_size": 16.0,
            "residual_gain": 0.0,
        },
    ).compress(None, encoder, scorer)
    elapsed = time.monotonic() - t0

    bytes_delta = result.pass_history[result.best_pass_idx].archive_bytes - baseline_bytes
    score_delta = result.final_score - args.baseline_score

    payload = {
        "tag": "[empirical:offline-byte-proxy]",
        "scope": "lane_8_multipass_real_archive_smoke",
        "anchor_path": str(args.anchor),
        "anchor_bytes": baseline_bytes,
        "baseline_score": args.baseline_score,
        "target_score": args.target_score,
        "max_passes": args.max_passes,
        "elapsed_seconds": elapsed,
        "result": result.to_dict(),
        "bytes_delta_vs_baseline": bytes_delta,
        "score_delta_vs_baseline": score_delta,
        "interpretation": (
            "Byte proxy ONLY. Real-bytes empirical for Level 3 requires GPU "
            "dispatch via scripts/remote_lane_8_multipass.sh (Phase G). "
            "Tagged [empirical:offline-byte-proxy] — does NOT count toward "
            "[contest-CUDA] gate."
        ),
        "claude_md_compliance": {
            "score_lane_tag": "[empirical:offline-byte-proxy]",
            "no_mps_used": True,
            "no_scorer_loaded": True,
            "strict_scorer_rule": "compress-time only; no inflate-time scorer",
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2))
    print(
        f"[lane-8-multipass-smoke] best_pass_idx={result.best_pass_idx} "
        f"score={result.final_score:.4f} (delta={score_delta:+.4f}) "
        f"bytes={result.pass_history[result.best_pass_idx].archive_bytes:,} "
        f"(delta={bytes_delta:+,}) elapsed={elapsed:.2f}s"
    )
    print(f"[lane-8-multipass-smoke] wrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
