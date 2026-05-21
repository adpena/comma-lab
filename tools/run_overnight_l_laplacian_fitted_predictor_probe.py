# SPDX-License-Identifier: MIT
"""tools/run_overnight_l_laplacian_fitted_predictor_probe.py — OVERNIGHT-L pair #1 engineering probe.

LOCAL macOS-CPU MVP probe of the canonical TESTABLE HYPOTHESIS per the
ratification memo's operator-routable #2
(``.omx/research/canonical_equation_procedural_predictor_plus_residual_correction_ratification_landed_20260521.md``).

**Engineering hypothesis** (TESTABLE): pair #1 (DWT detail subbands × magic_codec
dense_streams residual) was empirically falsified at residual_zscore=38.8 because
the ``pcg64``-uniform predictor distribution mismatches the empirical Laplacian-
peaked DWT detail subband distribution. Per Shannon source coding bounds, a
distributional match between predictor and empirical should produce a near-zero-
peaked residual that brotli/lzma can compress to ~ε*N rather than ~N bytes.

**Engineering fix** (per Carmack MVP-first phasing per CLAUDE.md `be125b878`):
replace the pcg64-uniform synthetic int8 with a Laplacian-fitted synthetic int8
parameterized by (μ, b) per subband, where (μ, b) are fitted from the empirical
DWT detail subband distribution via Method-of-Moments (μ = sample median,
b = mean absolute deviation × sqrt(2)/2). The Laplacian-fitted synthetic must
still be DETERMINISTICALLY derived from a small seed (per Catalog #318 master-
gradient raw-byte-authority discipline + canonical equation #26's
``derive_codebook_from_seed`` contract).

**MVP scope** (per Carmack MVP-first 5-step):
1. FREE local CPU smoke (no GPU dispatch)
2. Falsifiably challenges the cargo-cult (pair #1 residual_zscore=38.8 baseline
   IS the cargo-cult; this probe tests whether the FIX restores Shannon-bound
   compression)
3. Catalog #344 cross-ref + RATIFY-4 EXCLUDED context respect (this probe
   operates in the residual-hybrid context with the NEW sister equation
   ``procedural_predictor_plus_residual_correction_savings_v1`` — Catalog #359
   structural protection of equation #26 is preserved; no new EXCLUDED contexts
   needed)
4. Landing verdict in same commit batch
5. Re-route operator priority queue within ~1h

**Apples-to-apples comparison** (the canonical measurement structure mirrors
the prior pair #1 smoke for direct verifiability):

    Configuration A (direct empirical brotli baseline):
        per-subband empirical-int8 bytes → brotli(q=11) → baseline_bytes
        rate-term cost: 25 * baseline_bytes / 37_545_489

    Configuration C-pcg64 (PRIOR pair #1 with pcg64-uniform predictor):
        per-subband 32 B seed + pcg64-uniform synthetic-int8
        + magic_codec_dense_streams residual
        rate-term cost: 25 * (32 + encoded_residual_bytes) / 37_545_489
        (PRIOR RESULT: bytes_saved = -55,275; empirical ΔS = +0.036805 = REGRESSION)

    Configuration C-laplacian (THIS probe — engineering FIX):
        per-subband (32 B seed + 8 B per-subband (μ, b) fitted params)
        + Laplacian-fitted synthetic-int8
        + magic_codec_dense_streams residual
        rate-term cost: 25 * (32 + 24 + encoded_residual_bytes) / 37_545_489
        (PREDICTED: bytes_saved > 0 if predictor distributional match holds)

**Verdict semantics**:
- HARD-EARNED-RESCUE if Configuration C-laplacian bytes < Configuration A bytes
  (predictor fix RESCUES the residual-hybrid stacking paradigm; ratification
  memo's operator-routable #2 IS the canonical research-path forward)
- CARGO-CULTED-STILL if Configuration C-laplacian bytes >= Configuration A bytes
  (predictor fix INSUFFICIENT; sister equation #26's posterior remains intact
  for any predictor-class extension; the residual-hybrid stacking paradigm
  requires structural redesign per Catalog #307 paradigm-level)

This is a MEASUREMENT, not a score claim. No new canonical equation registered
(canonical equation
``procedural_predictor_plus_residual_correction_savings_v1`` already exists per
sister codex landing 2026-05-21T01:05:18Z; THIS probe is a NEW empirical anchor
candidate for that equation if HARD-EARNED-RESCUE verdict).

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND
CUDA": this is a `[macOS-CPU advisory]` measurement; non-promotable; no score
claim; no autopilot dispatch routing.

Sister-DISJOINT verified:
* No active sister subagents per ToolSearch query at session start
* NEW file path (sister of pair #1 smoke; different filename); zero overlap
* Catalog #340 sister-checkpoint guard PROCEED required
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import brotli  # noqa: E402

CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
DEFAULT_FRAME_INDEX = 300
DEFAULT_SEED_BYTES = 32
DEFAULT_WAVELET = "haar"
DEFAULT_DWT_LEVEL = 2
BROTLI_QUALITY = 11
BROTLI_LGWIN = 22
# (μ, b) Laplacian params per subband: 2 float32s = 8 bytes
LAPLACIAN_PARAMS_BYTES_PER_SUBBAND = 8


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_now_filename() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decode_y_plane(video_path: Path, frame_index: int) -> tuple[np.ndarray, int, int]:
    """Decode the Y plane of `frame_index` from `video_path` via pyav.

    Returns (y_uint8, height, width). Sister of pair #1 smoke implementation
    for apples-to-apples comparison.
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                ndarray = frame.to_ndarray(format="yuv420p")
                h = frame.height
                w = frame.width
                y_plane = ndarray[:h, :w].astype(np.uint8)
                return y_plane, h, w
        raise RuntimeError(f"frame_index={frame_index} not reachable in {video_path}")
    finally:
        container.close()


def compute_dwt_detail_subbands(
    y_plane: np.ndarray, wavelet: str = DEFAULT_WAVELET, level: int = DEFAULT_DWT_LEVEL
) -> dict[str, np.ndarray]:
    """Apply ``pywt.wavedec2`` and return level-N detail subbands as float32."""
    import pywt

    y_float = y_plane.astype(np.float32)
    coeffs = pywt.wavedec2(y_float, wavelet, level=level)
    LH, HL, HH = coeffs[1]
    return {
        "LH": LH.astype(np.float32),
        "HL": HL.astype(np.float32),
        "HH": HH.astype(np.float32),
    }


def normalize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float32 detail subband to int8 via 99-percentile fit.

    Identical to pair #1 smoke ``normalize_to_int8_distribution`` for
    apples-to-apples comparison.
    """
    if arr.size == 0:
        return arr.astype(np.int8)
    center = float(np.median(arr))
    centered = arr - center
    abs99 = float(np.percentile(np.abs(centered), 99.0))
    if abs99 < 1e-6:
        return np.zeros(arr.shape, dtype=np.int8)
    scaled = centered / abs99 * 120.0
    clipped = np.clip(scaled, -128.0, 127.0)
    return clipped.astype(np.int8)


def fit_laplacian_params(empirical_int8: np.ndarray) -> tuple[float, float]:
    """Fit (μ, b) Laplacian distribution params from empirical int8 via MoM.

    Method-of-Moments:
        μ = sample median (Laplacian location parameter MLE)
        b = mean(|x - μ|) (Laplacian scale parameter MLE)

    Returns (μ, b) as Python floats serializable to 8 bytes (2 × float32).
    """
    flat = empirical_int8.astype(np.float64).flatten()
    mu = float(np.median(flat))
    b = float(np.mean(np.abs(flat - mu)))
    # Guard against degenerate b=0 (constant signal); use 1.0 as floor
    if b < 1e-3:
        b = 1.0
    return mu, b


def derive_laplacian_synthetic_int8(
    shape: tuple[int, int],
    seed_bytes: bytes,
    mu: float,
    b: float,
) -> np.ndarray:
    """Deterministically derive a Laplacian-distributed synthetic int8 array.

    The seed_bytes (32 B) seeds a numpy default_rng; the Laplacian samples
    are drawn from ``rng.laplace(loc=mu, scale=b, size=shape)`` then clipped
    to int8 [-128, 127]. The (μ, b) params are encoded inline in the archive
    (8 bytes total per subband: 2 × float32).

    Per Catalog #318 master-gradient raw-byte-authority discipline: this
    helper produces SYNTHETIC bytes deterministically; the inflate-time
    contract is (seed_bytes + (μ, b) params) → synthetic_int8 ∈ [-128, 127].
    """
    # Seed numpy rng deterministically from the SHA-256 of seed_bytes
    seed_int = int.from_bytes(seed_bytes[:8], "little", signed=False)
    rng = np.random.default_rng(seed_int)
    samples = rng.laplace(loc=mu, scale=b, size=shape)
    clipped = np.clip(samples, -128.0, 127.0)
    return clipped.astype(np.int8)


def derive_pcg64_synthetic_int8(
    shape: tuple[int, int],
    seed_bytes: bytes,
) -> np.ndarray:
    """Sister of pair #1 pcg64-uniform predictor for apples-to-apples baseline.

    Uses the CANONICAL ``tac.procedural_codebook_generator.derive_codebook_from_seed``
    to produce uniform-int8 bytes; reshaped to subband shape. The PRIOR pair
    #1 smoke used this distribution and was empirically falsified at
    residual_zscore=38.8.
    """
    from tac.procedural_codebook_generator import derive_codebook_from_seed

    raw = derive_codebook_from_seed(
        seed_bytes, output_shape=shape, dtype=np.uint8, generator_kind="pcg64"
    )
    # raw is a uint8 array in [0, 255]; convert to int8 in [-128, 127]
    raw_int16 = raw.astype(np.int16) - 128
    return raw_int16.astype(np.int8)


def derive_seed_for_subband(subband_name: str, base_seed_bytes: bytes) -> bytes:
    """Sister of pair #1 smoke (identical helper)."""
    return hashlib.sha256(base_seed_bytes + subband_name.encode("ascii")).digest()


def compute_residuals_int8(empirical_int8: np.ndarray, synthetic_int8: np.ndarray) -> np.ndarray:
    """Sister of pair #1 smoke (identical helper)."""
    raw = empirical_int8.astype(np.int16) - synthetic_int8.astype(np.int16)
    clipped = np.clip(raw, -128, 127)
    return clipped.astype(np.int8)


def encode_brotli(values: np.ndarray) -> int:
    """Sister of pair #1 ``encode_config_a_baseline_brotli`` (identical contract)."""
    raw = values.tobytes()
    return len(brotli.compress(raw, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN))


def run_probe(
    video_path: Path,
    frame_index: int,
    base_seed_bytes: bytes,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the OVERNIGHT-L Laplacian-fitted predictor probe.

    Returns the smoke result dict (also written to ``output_dir/smoke_result.json``).
    """
    started_at = _utc_now_iso()

    # Step 1: decode frame + DWT
    y_plane, h, w = decode_y_plane(video_path, frame_index)
    subbands_float = compute_dwt_detail_subbands(y_plane)

    per_subband: dict[str, dict[str, Any]] = {}
    aggregate_config_a_bytes = 0
    aggregate_config_c_pcg64_bytes = 0
    aggregate_config_c_laplacian_bytes = 0

    for subband_name, subband_float in subbands_float.items():
        # Step 2: normalize empirical to int8
        empirical_int8 = normalize_to_int8(subband_float)
        shape = empirical_int8.shape

        # Step 3: derive per-subband seed
        seed_for_subband = derive_seed_for_subband(subband_name, base_seed_bytes)

        # Configuration A: direct empirical brotli baseline
        config_a_bytes = encode_brotli(empirical_int8)
        aggregate_config_a_bytes += config_a_bytes

        # Configuration C-pcg64 (PRIOR pair #1 baseline; for apples-to-apples
        # re-verification of the cargo-culted falsification at residual_zscore=38.8)
        synthetic_pcg64 = derive_pcg64_synthetic_int8(shape, seed_for_subband)
        residual_pcg64 = compute_residuals_int8(empirical_int8, synthetic_pcg64)
        residual_pcg64_brotli_bytes = encode_brotli(residual_pcg64)
        config_c_pcg64_total = DEFAULT_SEED_BYTES + residual_pcg64_brotli_bytes
        aggregate_config_c_pcg64_bytes += config_c_pcg64_total

        # Configuration C-laplacian (THIS probe — engineering FIX)
        mu, b = fit_laplacian_params(empirical_int8)
        synthetic_laplacian = derive_laplacian_synthetic_int8(shape, seed_for_subband, mu, b)
        residual_laplacian = compute_residuals_int8(empirical_int8, synthetic_laplacian)
        residual_laplacian_brotli_bytes = encode_brotli(residual_laplacian)
        config_c_laplacian_total = (
            DEFAULT_SEED_BYTES
            + LAPLACIAN_PARAMS_BYTES_PER_SUBBAND
            + residual_laplacian_brotli_bytes
        )
        aggregate_config_c_laplacian_bytes += config_c_laplacian_total

        # Predictor-empirical KL divergence proxy: std of residual
        residual_pcg64_std = float(np.std(residual_pcg64.astype(np.float64)))
        residual_laplacian_std = float(np.std(residual_laplacian.astype(np.float64)))

        per_subband[subband_name] = {
            "shape": list(shape),
            "n_pixels": int(shape[0] * shape[1]),
            "fitted_mu": mu,
            "fitted_b": b,
            "config_a_brotli_bytes": config_a_bytes,
            "config_c_pcg64_total_bytes": config_c_pcg64_total,
            "config_c_pcg64_residual_brotli_bytes": residual_pcg64_brotli_bytes,
            "config_c_pcg64_residual_std": residual_pcg64_std,
            "config_c_laplacian_total_bytes": config_c_laplacian_total,
            "config_c_laplacian_residual_brotli_bytes": residual_laplacian_brotli_bytes,
            "config_c_laplacian_residual_std": residual_laplacian_std,
            "config_c_laplacian_overhead_bytes": (
                DEFAULT_SEED_BYTES + LAPLACIAN_PARAMS_BYTES_PER_SUBBAND
            ),
            "bytes_saved_c_laplacian_vs_a": (
                config_a_bytes - config_c_laplacian_total
            ),
            "bytes_saved_c_pcg64_vs_a": (
                config_a_bytes - config_c_pcg64_total
            ),
            "delta_laplacian_vs_pcg64": (
                config_c_pcg64_total - config_c_laplacian_total
            ),
        }

    # Aggregate verdicts
    aggregate_bytes_saved_c_laplacian_vs_a = (
        aggregate_config_a_bytes - aggregate_config_c_laplacian_bytes
    )
    aggregate_bytes_saved_c_pcg64_vs_a = (
        aggregate_config_a_bytes - aggregate_config_c_pcg64_bytes
    )
    aggregate_delta_laplacian_vs_pcg64 = (
        aggregate_config_c_pcg64_bytes - aggregate_config_c_laplacian_bytes
    )

    empirical_delta_s_c_laplacian = (
        -CANONICAL_RATE_MULTIPLIER * aggregate_bytes_saved_c_laplacian_vs_a
        / CANONICAL_RATE_DENOM_BYTES
    )
    empirical_delta_s_c_pcg64 = (
        -CANONICAL_RATE_MULTIPLIER * aggregate_bytes_saved_c_pcg64_vs_a
        / CANONICAL_RATE_DENOM_BYTES
    )

    if aggregate_bytes_saved_c_laplacian_vs_a > 0:
        verdict = "HARD_EARNED_RESCUE_LAPLACIAN_PREDICTOR_FIX_RESCUES_PAIR_1"
        verdict_detail = (
            "Laplacian-fitted predictor produces fewer bytes than direct brotli "
            "baseline; residual-hybrid stacking paradigm RESCUED by predictor "
            "distributional match. Sister equation "
            "procedural_predictor_plus_residual_correction_savings_v1 anchor candidate."
        )
    elif aggregate_bytes_saved_c_laplacian_vs_a > aggregate_bytes_saved_c_pcg64_vs_a:
        verdict = "PARTIAL_IMPROVEMENT_LAPLACIAN_BETTER_THAN_PCG64_BUT_STILL_REGRESSION"
        verdict_detail = (
            "Laplacian-fitted predictor improves on pcg64-uniform but STILL regresses "
            "vs direct brotli baseline; predictor distributional match is necessary "
            "but not sufficient; further engineering (per-class adaptive predictor / "
            "Anscombe-like variance stabilization) per ratification memo §6 needed."
        )
    else:
        verdict = "CARGO_CULTED_STILL_LAPLACIAN_NO_BETTER_THAN_PCG64"
        verdict_detail = (
            "Laplacian-fitted predictor NO BETTER than pcg64-uniform; predictor "
            "distributional match is INSUFFICIENT to rescue residual-hybrid stacking "
            "paradigm; structural redesign (per Catalog #307 paradigm-level "
            "falsification of the residual-hybrid stacking class for DWT detail "
            "subbands) is the canonical research path forward."
        )

    completed_at = _utc_now_iso()

    smoke_result = {
        "lane_id": "lane_overnight_l_magic_codec_pair_1_2_engineering_fix_20260521",
        "task_id": "OVERNIGHT-L",
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "local_cpu_smoke_advisory",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatched_at_utc": started_at,
        "completed_at_utc": completed_at,
        "video_path": str(video_path.relative_to(REPO_ROOT)),
        "frame_index": frame_index,
        "frame_height": h,
        "frame_width": w,
        "base_seed_bytes_sha256": _sha256_bytes(base_seed_bytes),
        "base_seed_bytes_len": len(base_seed_bytes),
        "wavelet": DEFAULT_WAVELET,
        "dwt_level": DEFAULT_DWT_LEVEL,
        "per_subband": per_subband,
        "aggregate_config_a_brotli_bytes": aggregate_config_a_bytes,
        "aggregate_config_c_pcg64_total_bytes": aggregate_config_c_pcg64_bytes,
        "aggregate_config_c_laplacian_total_bytes": aggregate_config_c_laplacian_bytes,
        "aggregate_bytes_saved_c_pcg64_vs_a": aggregate_bytes_saved_c_pcg64_vs_a,
        "aggregate_bytes_saved_c_laplacian_vs_a": aggregate_bytes_saved_c_laplacian_vs_a,
        "aggregate_delta_laplacian_vs_pcg64_bytes": aggregate_delta_laplacian_vs_pcg64,
        "empirical_delta_s_c_pcg64_vs_a": empirical_delta_s_c_pcg64,
        "empirical_delta_s_c_laplacian_vs_a": empirical_delta_s_c_laplacian,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "canonical_equation_referenced": (
            "procedural_predictor_plus_residual_correction_savings_v1"
        ),
        "canonical_equation_sister_negative_cross_reference": (
            "procedural_codebook_from_seed_compression_savings_v1"
        ),
        "in_domain_context": (
            "magic_codec_dense_streams_laplacian_fitted_predictor_residual_correction"
            "_on_dwt_detail_subbands"
        ),
        "engineering_fix_summary": (
            "Replace pcg64-uniform synthetic int8 predictor with Laplacian-fitted "
            "synthetic int8 parameterized by per-subband (μ, b) Method-of-Moments fit. "
            "Total overhead: 32 B seed + 8 B (μ, b) params per subband. Tests whether "
            "predictor distributional match rescues residual-hybrid stacking paradigm "
            "per ratification memo operator-routable #2."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "smoke_result.json"
    result_path.write_text(json.dumps(smoke_result, indent=2, sort_keys=True))

    return smoke_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OVERNIGHT-L Laplacian-fitted predictor probe")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to upstream contest video (default upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help="Frame index to decode (default 300; matches pair #1 smoke for apples-to-apples)",
    )
    parser.add_argument(
        "--base-seed-hex",
        type=str,
        default="65d2ab47bcd6e3a3ac9651b3511cf20fbbbc0017650132e3ef827697ab7f063b",
        help="Base seed bytes as hex (default matches pair #1 smoke for apples-to-apples)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default experiments/results/overnight_l_<utc>/)",
    )
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"overnight_l_laplacian_fitted_predictor_probe_{_utc_now_filename()}"
        )

    base_seed_bytes = bytes.fromhex(args.base_seed_hex)
    if len(base_seed_bytes) != DEFAULT_SEED_BYTES:
        print(
            f"FATAL: base_seed_hex must encode exactly {DEFAULT_SEED_BYTES} bytes "
            f"(got {len(base_seed_bytes)})",
            file=sys.stderr,
        )
        return 1

    if not args.video_path.exists():
        print(f"FATAL: video_path does not exist: {args.video_path}", file=sys.stderr)
        return 1

    print(f"[OVERNIGHT-L] running probe at {_utc_now_iso()}", file=sys.stderr)
    print(f"[OVERNIGHT-L] video_path: {args.video_path}", file=sys.stderr)
    print(f"[OVERNIGHT-L] frame_index: {args.frame_index}", file=sys.stderr)
    print(f"[OVERNIGHT-L] output_dir: {args.output_dir}", file=sys.stderr)

    smoke_result = run_probe(
        video_path=args.video_path,
        frame_index=args.frame_index,
        base_seed_bytes=base_seed_bytes,
        output_dir=args.output_dir,
    )

    print(f"\n=== OVERNIGHT-L VERDICT ===", file=sys.stderr)
    print(f"Verdict: {smoke_result['verdict']}", file=sys.stderr)
    print(f"\n=== AGGREGATE BYTES ===", file=sys.stderr)
    print(
        f"Configuration A (direct brotli baseline):   {smoke_result['aggregate_config_a_brotli_bytes']:>10,} B",
        file=sys.stderr,
    )
    print(
        f"Configuration C-pcg64 (PRIOR pair #1):      {smoke_result['aggregate_config_c_pcg64_total_bytes']:>10,} B",
        file=sys.stderr,
    )
    print(
        f"Configuration C-laplacian (ENGINEERING FIX):{smoke_result['aggregate_config_c_laplacian_total_bytes']:>10,} B",
        file=sys.stderr,
    )
    print(f"\n=== BYTES SAVED vs Configuration A ===", file=sys.stderr)
    print(
        f"C-pcg64 vs A:     {smoke_result['aggregate_bytes_saved_c_pcg64_vs_a']:>+10,} B",
        file=sys.stderr,
    )
    print(
        f"C-laplacian vs A: {smoke_result['aggregate_bytes_saved_c_laplacian_vs_a']:>+10,} B",
        file=sys.stderr,
    )
    print(
        f"C-laplacian gain over C-pcg64: {smoke_result['aggregate_delta_laplacian_vs_pcg64_bytes']:>+10,} B",
        file=sys.stderr,
    )
    print(f"\n=== ΔS (canonical contest formula) ===", file=sys.stderr)
    print(
        f"empirical ΔS (C-pcg64 vs A):     {smoke_result['empirical_delta_s_c_pcg64_vs_a']:+.6f}",
        file=sys.stderr,
    )
    print(
        f"empirical ΔS (C-laplacian vs A): {smoke_result['empirical_delta_s_c_laplacian_vs_a']:+.6f}",
        file=sys.stderr,
    )
    print(f"\nsmoke_result.json: {smoke_result.get('lane_id', 'unknown')}", file=sys.stderr)
    print(f"  written to: {args.output_dir / 'smoke_result.json'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
