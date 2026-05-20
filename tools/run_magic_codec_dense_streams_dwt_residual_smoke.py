# SPDX-License-Identifier: MIT
"""tools/run_magic_codec_dense_streams_dwt_residual_smoke.py — WAVE-3 pair #1.

LOCAL macOS-CPU smoke validating PAIR #1 architecture from the
MAGIC CODEC × TODAY'S CASCADE STACKING ANALYSIS memo
``.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md``:

    magic_codec_dense_streams × DWT detail subbands (LH+HL+HH)
        applied to procedural-codebook-substitution RESIDUALS
        (NOT direct substitution per yesterday's empirical falsification)

Yesterday's DWT-DETAIL-SUBBAND CPU SMOKE
(``feedback_dwt_detail_subband_procedural_cpu_smoke_landed_20260520.md``,
landing commit ``f25f8cc1b``) empirically vindicated Assumption-Adversary
verdict #1: KL=1.638 nats vs H0 threshold 0.5 (residual_zscore 3.28 > 2σ)
→ direct procedural substitution of DWT detail subbands is CARGO-CULTED.

This smoke validates the CANONICAL RESCUE PATH per the stacking analysis
memo: instead of REPLACING detail subbands with procedural codebook
(refuted), apply procedural codebook AS A PREDICTOR first, then encode
the RESIDUALS via ``tac.packet_compiler.magic_codec_dense_streams`` (per-
stream brotli/lzma/magic_codec_classic 3-way head-to-head). Net archive
cost = 32 B seed + dense-stream-coded residuals.

3-way apples-to-apples comparison (the canonical measurement structure):

    Configuration A (direct empirical):
        per-subband empirical-int8 bytes → brotli(q=11) → baseline_bytes
        rate-term cost: 25 * baseline_bytes / 37_545_489

    Configuration B (procedural only — yesterday's REFUTED hypothesis):
        per-subband 32 B seed + derive_codebook_from_seed → synthetic-int8
        bytes stored INSTEAD OF empirical-int8 bytes
        rate-term cost: 25 * 32 / 37_545_489 (one seed per subband)
        (KL=1.638 nats per yesterday; bytes do NOT match empirical
         distribution → would corrupt inverse DWT → CARGO-CULTED)

    Configuration C (procedural + dense-stream residuals — pair #1 candidate):
        per-subband 32 B seed (predictor)
        + dense-stream-coded residual (empirical-int8 - synthetic-int8)
          via encode_magic_codec_dense_streams
          (per-stream codec auto-selection brotli/lzma/magic_codec_classic)
        rate-term cost: 25 * (32 + encoded_residual_bytes) / 37_545_489

The pair #1 prediction (from stacking analysis memo §7):
    composition_alpha=0.8 ADDITIVE
    predicted ΔS = -0.00200 ([-0.011, -0.005] cumulative under 4-pair stack)

Verdict at 2σ threshold (per the stacking analysis memo §8 Dykstra-
feasibility): HARD-EARNED if residual_zscore < 2σ (rescue path validated;
the cascade can proceed to pair #2); CARGO-CULTED if residual_zscore > 2σ
(rescue path falsified; cascade pivots to pair #2 procedural-codebook
null-byte residuals via sparse_packet_ir SRL1 instead).

Pipeline:

1. Decode frame index=300 from ``upstream/videos/0.mkv`` via pyav → YUV420p
2. Extract Y plane (uint8, 874x1164) — identical to yesterday's smoke for
   apples-to-apples comparison
3. Apply ``pywt.wavedec2(Y, "haar", level=2)`` → LL2 + (LH2, HL2, HH2) +
   (LH1, HL1, HH1); identical to yesterday's smoke
4. For each detail subband at level=2 (LH2, HL2, HH2):
   a. Compute empirical int8 distribution (Configuration A: brotli baseline)
   b. Derive synthetic int8 via ``derive_codebook_from_seed`` (Configuration B)
   c. Compute residuals = empirical - synthetic (element-wise int16 then
      cast to int8 in [-128, 127])
   d. Encode residuals via ``encode_magic_codec_dense_streams`` (Configuration C)
5. Catalog #272 byte-mutation smoke: mutate 1 byte of seed → re-derive →
   re-compute residuals → re-encode → verify the encoded bytes change
   (proves the residual encoding is seed-sensitive; rules out the empty-byte
   / no-op trap from Catalog #220/#249)
6. Compute aggregate residual_zscore (NEW IN-DOMAIN context per Catalog #344):
   bytes_saved_residual_correction = baseline_bytes - (32 + residual_bytes)
7. Compare predicted vs empirical ΔS:
   predicted_ΔS = -0.00200 (per pair #1 ADDITIVE α=0.8 prediction)
   empirical_ΔS = -25 * bytes_saved_aggregate / 37_545_489
8. Append SECOND empirical anchor to canonical equation #26 via
   ``tac.canonical_equations.update_equation_with_empirical_anchor``
   with NEW IN-DOMAIN context
   ``magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands``
   (axis=[macOS-CPU advisory]; hardware=darwin_arm64_m5_max_macos_cpu_advisory;
   evidence_grade=MACOS_CPU_ADVISORY; promotion_eligible=False;
   score_claim_valid=False per Catalog #192)
9. Emit JSON + Markdown artifacts under
   ``experiments/results/magic_codec_dense_streams_dwt_residual_smoke_<utc>/``

Note: this smoke does NOT run inflate.sh / contest_auth_eval.py — it
measures the byte-budget difference between Configurations A/B/C as the
SECOND empirical anchor for canonical equation #26 with a NEW context
distinct from yesterday's FIRST anchor (which measured distributional fit
under the direct-substitution H0).

Sister-DISJOINT from:

* ``aa17d84d`` DP1 PROCEDURAL TRAINER BUILD (different substrate;
  different file path; this smoke validates pair #1 of the magic_codec
  cascade-stacking 4-pair matrix; DP1 is a separate substrate trainer
  path)
* ``a230693c`` CANONICAL EQUATION #26 DOMAIN REFINEMENT (different file;
  this smoke COMPLEMENTS the domain refinement by adding a SECOND IN-
  DOMAIN context whose empirical anchor validates the
  magic_codec_dense_streams stacking surface)
* ``f25f8cc1b`` (yesterday's DWT-DETAIL-SUBBAND CPU SMOKE) — sister-
  COMPLEMENTARY: yesterday's smoke measured distributional fit under H0
  (direct substitution); THIS smoke measures byte budget under
  rescue-path (procedural-predictor + residual-correction)

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 (macOS-CPU advisory
not promotable without Linux x86_64 verification) + Catalog #127 (per-
call-site custody triple axis × hardware × evidence_grade): every metric
stamped ``[macOS-CPU advisory]`` + ``hardware_substrate=darwin_arm64_m5_max_macos_cpu_advisory``
+ ``evidence_grade=local_cpu_smoke_advisory``. NOT score truth.

Per Carmack MVP-first phasing + Time Traveler framing *"we have all the
information we need"*: this smoke validates the pair #1 rescue path
BEFORE the $2 paid DWT-HNeRV bind L0 SCAFFOLD smoke would have been
fired (op-routable #1 from the stacking analysis memo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Canonical constants per CLAUDE.md "Submission auth eval" + canonical equation #26
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
DEFAULT_FRAME_INDEX = 300
DEFAULT_SEED_BYTES = 32
DEFAULT_GENERATOR_KIND = "pcg64"
DEFAULT_WAVELET = "haar"
DEFAULT_DWT_LEVEL = 2

# Pair #1 prediction from MAGIC CODEC × TODAY'S CASCADE STACKING ANALYSIS memo §7
# Composition alpha = 0.8 ADDITIVE; predicted ΔS = -0.00200
PAIR_1_PREDICTED_DELTA_S = -0.00200
PAIR_1_COMPOSITION_ALPHA_ESTIMATE = 0.8

# 2σ threshold for residual_zscore HARD-EARNED vs CARGO-CULTED verdict
# (sister of yesterday's H0 KL threshold; for pair #1 the H0 is "rescue
# path produces no net byte savings" → bytes_saved == 0 → ΔS == 0)
# 2σ threshold for byte_savings vs predicted = 0.5 * |predicted ΔS| (i.e.
# empirical must be within 0.5x of predicted to be HARD-EARNED)
PAIR_1_ZSCORE_HARD_EARNED_THRESHOLD = 2.0

# Brotli baseline canonical parameters (must match
# magic_codec_dense_streams._BROTLI_QUALITY / _BROTLI_LGWIN)
BROTLI_QUALITY = 11
BROTLI_LGWIN = 22


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _utc_now_filename() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decode_frame_from_video(video_path: Path, frame_index: int) -> tuple[np.ndarray, int, int]:
    """Decode frame at ``frame_index`` from ``video_path`` via pyav.

    Sister of yesterday's smoke (identical implementation). Returns
    (Y_plane uint8, height, width). YUV420p planar; we extract only the
    Y plane (full resolution).
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        height = int(stream.height)
        width = int(stream.width)
        target_frame: np.ndarray | None = None
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                yuv = frame.to_ndarray(format="yuv420p")
                target_frame = yuv[:height, :width].copy()
                break
        if target_frame is None:
            raise ValueError(
                f"frame_index={frame_index} not found in {video_path} "
                f"(decoded {i + 1} frames)"
            )
    finally:
        container.close()
    return target_frame, height, width


def compute_dwt_detail_subbands(
    y_plane: np.ndarray, wavelet: str, level: int
) -> dict[str, np.ndarray]:
    """Apply ``pywt.wavedec2`` and return the level-N detail subbands as float32 arrays.

    Sister of yesterday's smoke (identical implementation). Returns dict
    with keys 'LH', 'HL', 'HH' at the deepest decomposition level.
    """
    import pywt

    y_float = y_plane.astype(np.float32)
    coeffs = pywt.wavedec2(y_float, wavelet, level=level)
    if not isinstance(coeffs[1], tuple) or len(coeffs[1]) != 3:
        raise RuntimeError(
            f"unexpected pywt.wavedec2 structure: coeffs[1] type={type(coeffs[1]).__name__}"
        )
    LH, HL, HH = coeffs[1]
    return {
        "LH": LH.astype(np.float32),
        "HL": HL.astype(np.float32),
        "HH": HH.astype(np.float32),
    }


def normalize_to_int8_distribution(arr: np.ndarray) -> np.ndarray:
    """Normalize a float32 detail subband to int8 distribution via histogram fit.

    Sister of yesterday's smoke (identical implementation) — critical for
    apples-to-apples comparison: the SAME normalization defines the
    "canonical empirical-int8 byte representation" both smokes operate on.
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


def derive_seed_for_subband(subband_name: str, base_seed_bytes: bytes) -> bytes:
    """Deterministically derive a 32-byte per-subband seed from a base seed.

    Sister of yesterday's smoke (identical helper) for apples-to-apples
    seed-identity guarantee between the two smokes.
    """
    return hashlib.sha256(base_seed_bytes + subband_name.encode("ascii")).digest()


def compute_residuals_int8(empirical_int8: np.ndarray, synthetic_int8: np.ndarray) -> np.ndarray:
    """Compute (empirical - synthetic) residuals clipped to int8 range.

    Each input is int8 [-128, 127]; raw difference fits in int16; we clip
    the result to int8 so the residual can be losslessly stored as an
    int8 array. Clipping IS a small-amount lossy operation but the
    residual encoding tests whether the per-stream codec selection
    recovers entropy savings on the post-prediction residual surface.

    Returns int8 ndarray of same shape; the encoded byte cost via
    encode_magic_codec_dense_streams is the canonical Configuration C
    rate-term contribution.
    """
    if empirical_int8.dtype != np.int8 or synthetic_int8.dtype != np.int8:
        raise ValueError("residual inputs must be int8")
    if empirical_int8.shape != synthetic_int8.shape:
        raise ValueError(
            f"shape mismatch: empirical {empirical_int8.shape} vs synthetic {synthetic_int8.shape}"
        )
    # Compute in int16 then clip; the residual distribution for
    # uniform-PRNG synthetic vs Laplacian-peaked empirical will itself
    # be ~Laplacian (the two distributions' mean difference is the
    # empirical's bias from int8-uniform).
    raw = empirical_int8.astype(np.int16) - synthetic_int8.astype(np.int16)
    clipped = np.clip(raw, -128, 127)
    return clipped.astype(np.int8)


def encode_config_a_baseline_brotli(empirical_int8: np.ndarray) -> int:
    """Configuration A: direct empirical brotli baseline byte count.

    Encodes the empirical int8 bytes via brotli(q=11) — the SAME
    parameters used by encode_magic_codec_dense_streams._try_brotli for
    apples-to-apples comparison. Returns the encoded byte length (not
    including any envelope / length-prefix; this is the rate-term
    contribution under the canonical encoder).
    """
    raw_bytes = empirical_int8.tobytes()
    encoded = brotli.compress(raw_bytes, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    return len(encoded)


def encode_config_c_procedural_plus_dense_streams(
    residuals_int8: np.ndarray,
    seed_bytes: bytes,
    subband_name: str,
) -> tuple[int, dict[str, Any]]:
    """Configuration C: 32B seed + magic_codec_dense_streams residual byte count.

    Encodes the residual via ``encode_magic_codec_dense_streams`` (per-
    stream codec auto-selection brotli/lzma/magic_codec_classic). Returns
    (total_bytes, selection_log) where:

      total_bytes = len(seed_bytes) + len(dense_streams_payload)

    The selection_log captures per-codec encoded byte counts so the
    landing memo can document WHICH codec the auto-selector chose for
    the residual distribution.
    """
    from tac.packet_compiler.magic_codec_dense_streams import (
        DenseStreamInput,
        encode_magic_codec_dense_streams,
    )

    stream = DenseStreamInput(
        name=f"dwt_residual_{subband_name}",
        values=residuals_int8,
        hint=None,  # no magic_codec_classic candidate; brotli vs lzma only
    )
    bundle = encode_magic_codec_dense_streams(
        streams=[stream],
        selection_strategy="smallest_byte_count",
    )

    total_bytes = len(seed_bytes) + len(bundle.payload)
    selection_log = {
        "seed_bytes_len": len(seed_bytes),
        "dense_streams_payload_len": len(bundle.payload),
        "total_inner_byte_count": bundle.total_inner_byte_count,
        "total_config_c_bytes": total_bytes,
        "per_codec_candidates": [
            {
                "codec_name": cand.codec_name,
                "codec_id": cand.codec_id,
                "encoded_bytes_len": len(cand.encoded_bytes) if not cand.refused else None,
                "refused": cand.refused,
                "refusal_reason": cand.refusal_reason,
            }
            for selection in bundle.selections
            for cand in selection.candidates
        ],
        "selected_codec_name": bundle.selections[0].selected_codec_name,
        "selected_codec_id": bundle.selections[0].selected_codec_id,
        "selected_byte_count": bundle.selections[0].selected_byte_count,
    }
    return total_bytes, selection_log


def run_smoke(
    video_path: Path,
    frame_index: int,
    base_seed_bytes: bytes,
    wavelet: str,
    dwt_level: int,
    generator_kind: str,
) -> dict[str, Any]:
    """Run the WAVE-3 pair #1 smoke pipeline.

    Returns a typed result dict ready for JSON serialization + canonical
    equation SECOND anchor append.
    """
    from tac.procedural_codebook_generator.seed_derived_codebook import (
        derive_codebook_from_seed,
    )

    start_utc = _utc_now_iso()
    video_sha = _sha256_of_file(video_path)

    # Step 1+2: decode + extract Y (identical to yesterday's smoke)
    y_plane, height, width = decode_frame_from_video(video_path, frame_index)
    y_sha = _sha256_of_bytes(y_plane.tobytes())

    # Step 3: DWT 2-level (identical to yesterday's smoke)
    detail_subbands = compute_dwt_detail_subbands(y_plane, wavelet, dwt_level)

    # Step 4-5: per-subband 3-way comparison + byte-mutation smoke
    per_subband: dict[str, dict[str, Any]] = {}
    aggregate_config_a_bytes = 0
    aggregate_config_b_bytes = 0
    aggregate_config_c_bytes = 0

    for subband_name, subband in detail_subbands.items():
        empirical_int8 = normalize_to_int8_distribution(subband)
        n_detail = int(empirical_int8.size)
        seed_for_subband = derive_seed_for_subband(subband_name, base_seed_bytes)

        # Configuration A: direct empirical brotli baseline
        config_a_bytes = encode_config_a_baseline_brotli(empirical_int8)

        # Configuration B: procedural only (32B seed; bytes do NOT match
        # empirical per yesterday's KL=1.638 nats; rate-term cost is only
        # the seed itself but the substrate would be CORRUPTED at inflate)
        config_b_bytes = len(seed_for_subband)

        # Configuration C: procedural + dense-stream residuals (pair #1)
        synthetic_int8 = derive_codebook_from_seed(
            seed_bytes=seed_for_subband,
            output_shape=(n_detail, 1),
            dtype=np.int8,
            generator_kind=generator_kind,
        ).reshape(empirical_int8.shape)

        residuals_int8 = compute_residuals_int8(empirical_int8, synthetic_int8)
        config_c_bytes, config_c_selection_log = (
            encode_config_c_procedural_plus_dense_streams(
                residuals_int8=residuals_int8,
                seed_bytes=seed_for_subband,
                subband_name=subband_name,
            )
        )

        # Catalog #272 byte-mutation smoke: mutate first byte of seed →
        # re-derive synthetic → re-compute residuals → re-encode → verify
        # encoded bytes change (proves residual encoding is seed-sensitive)
        mutated_seed = bytearray(seed_for_subband)
        mutated_seed[0] ^= 0xFF
        mutated_synthetic = derive_codebook_from_seed(
            seed_bytes=bytes(mutated_seed),
            output_shape=(n_detail, 1),
            dtype=np.int8,
            generator_kind=generator_kind,
        ).reshape(empirical_int8.shape)
        mutated_residuals = compute_residuals_int8(empirical_int8, mutated_synthetic)
        mutated_config_c_bytes, _mutated_log = (
            encode_config_c_procedural_plus_dense_streams(
                residuals_int8=mutated_residuals,
                seed_bytes=bytes(mutated_seed),
                subband_name=subband_name,
            )
        )

        # Compute per-subband bytes_saved (C vs A)
        bytes_saved_c_vs_a = config_a_bytes - config_c_bytes

        # Compute residual distribution statistics (the rescue path's
        # entropy structure: residuals should be near-zero-mean and
        # slightly Laplacian; we record the spread for the landing memo)
        residual_mean = float(residuals_int8.mean())
        residual_std = float(residuals_int8.std())
        residual_abs_max = int(np.abs(residuals_int8).max())

        per_subband[subband_name] = {
            "shape": list(empirical_int8.shape),
            "n_pixels": n_detail,
            "empirical_int8_sha256": _sha256_of_bytes(empirical_int8.tobytes()),
            "synthetic_int8_sha256": _sha256_of_bytes(synthetic_int8.tobytes()),
            "residuals_int8_sha256": _sha256_of_bytes(residuals_int8.tobytes()),
            "mutated_residuals_int8_sha256": _sha256_of_bytes(mutated_residuals.tobytes()),
            "seed_for_subband_sha256": _sha256_of_bytes(seed_for_subband),
            "config_a_baseline_brotli_bytes": config_a_bytes,
            "config_b_procedural_only_bytes": config_b_bytes,
            "config_c_procedural_plus_dense_streams_bytes": config_c_bytes,
            "config_c_selection_log": config_c_selection_log,
            "bytes_saved_c_vs_a": bytes_saved_c_vs_a,
            "residual_distribution": {
                "mean": residual_mean,
                "std": residual_std,
                "abs_max": residual_abs_max,
            },
            "byte_mutation_smoke_mutated_config_c_bytes": mutated_config_c_bytes,
            "byte_mutation_smoke_byte_diff_count": int(
                np.count_nonzero(residuals_int8.flatten() != mutated_residuals.flatten())
            ),
            "byte_mutation_smoke_encoded_bytes_differ": (
                config_c_bytes != mutated_config_c_bytes
            ),
            "byte_mutation_smoke_verdict_seed_sensitive": bool(
                int(np.count_nonzero(residuals_int8.flatten() != mutated_residuals.flatten())) > 0
            ),
        }
        aggregate_config_a_bytes += config_a_bytes
        aggregate_config_b_bytes += config_b_bytes
        aggregate_config_c_bytes += config_c_bytes

    # Step 6: aggregate bytes saved + empirical ΔS
    bytes_saved_c_vs_a_aggregate = aggregate_config_a_bytes - aggregate_config_c_bytes
    empirical_delta_s = -CANONICAL_RATE_MULTIPLIER * bytes_saved_c_vs_a_aggregate / CANONICAL_RATE_DENOM_BYTES

    # Step 7: residual_zscore vs predicted ΔS (HARD-EARNED if empirical
    # is within 2σ of predicted; CARGO-CULTED if outside 2σ)
    # The 2σ threshold is 0.5 * |predicted ΔS| (so empirical within
    # ±0.5x of predicted = HARD-EARNED; empirical outside = CARGO-CULTED)
    sigma_predicted = 0.5 * abs(PAIR_1_PREDICTED_DELTA_S)
    if sigma_predicted < 1e-12:
        residual_zscore = float("inf")
    else:
        residual_zscore = abs(empirical_delta_s - PAIR_1_PREDICTED_DELTA_S) / sigma_predicted
    canonical_equation_verdict = (
        "HARD-EARNED" if residual_zscore < PAIR_1_ZSCORE_HARD_EARNED_THRESHOLD else "CARGO-CULTED"
    )

    # Sub-verdict: rescue path validated if bytes_saved > 0 (net savings)
    rescue_path_net_savings_validated = bytes_saved_c_vs_a_aggregate > 0

    # Byte-mutation smoke aggregate: ALL subbands must show seed-sensitivity
    byte_mutation_all_subbands_seed_sensitive = all(
        per_subband[name]["byte_mutation_smoke_verdict_seed_sensitive"]
        for name in per_subband
    )

    # DWT bind rescue path verdict (the operator-facing summary)
    if canonical_equation_verdict == "HARD-EARNED" and rescue_path_net_savings_validated:
        dwt_bind_rescue_path_verdict = "DWT_BIND_RESCUE_PATH_VALIDATED_PROCEED_TO_PAIR_2"
    elif rescue_path_net_savings_validated:
        dwt_bind_rescue_path_verdict = "PARTIAL_RESCUE_NET_SAVINGS_BUT_OUTSIDE_PREDICTED_BAND"
    elif canonical_equation_verdict == "CARGO-CULTED":
        dwt_bind_rescue_path_verdict = "DWT_BIND_RESCUE_PATH_FALSIFIED_PIVOT_TO_PAIR_2_NULL_BYTE_RESIDUALS"
    else:
        dwt_bind_rescue_path_verdict = "INDETERMINATE_REQUIRES_PAIRED_LINUX_X86_64_VERIFICATION"

    end_utc = _utc_now_iso()

    try:
        video_path_for_record = str(video_path.relative_to(REPO_ROOT))
    except ValueError:
        video_path_for_record = str(video_path)

    return {
        "smoke_label": "wave_3_magic_codec_dense_streams_dwt_residual_cpu_smoke",
        "smoke_lane_id": "lane_wave_3_magic_codec_pair_1_dwt_residual_cpu_smoke_20260520",
        "smoke_cascade_stacking_analysis_memo": (
            ".omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md"
        ),
        "smoke_pair_id": "pair_1_magic_codec_dense_streams_x_dwt_detail_subbands",
        "smoke_baseline_anchor": "feedback_dwt_detail_subband_procedural_cpu_smoke_landed_20260520",
        "started_at_utc": start_utc,
        "completed_at_utc": end_utc,
        "platform": platform.platform(),
        "video_path_repo_relative": video_path_for_record,
        "video_sha256": video_sha,
        "frame_index": frame_index,
        "frame_height": height,
        "frame_width": width,
        "y_plane_sha256": y_sha,
        "wavelet": wavelet,
        "dwt_level": dwt_level,
        "generator_kind": generator_kind,
        "base_seed_bytes_hex": base_seed_bytes.hex(),
        "base_seed_bytes_len": len(base_seed_bytes),
        "per_subband": per_subband,
        "aggregate_config_a_baseline_brotli_bytes": aggregate_config_a_bytes,
        "aggregate_config_b_procedural_only_bytes": aggregate_config_b_bytes,
        "aggregate_config_c_procedural_plus_dense_streams_bytes": aggregate_config_c_bytes,
        "aggregate_bytes_saved_c_vs_a": bytes_saved_c_vs_a_aggregate,
        "empirical_delta_s": empirical_delta_s,
        "predicted_delta_s_pair_1": PAIR_1_PREDICTED_DELTA_S,
        "composition_alpha_estimate_pair_1": PAIR_1_COMPOSITION_ALPHA_ESTIMATE,
        "sigma_predicted_for_zscore": sigma_predicted,
        "residual_zscore_empirical_vs_predicted": residual_zscore,
        "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": canonical_equation_verdict,
        "rescue_path_net_savings_validated": rescue_path_net_savings_validated,
        "dwt_bind_rescue_path_verdict": dwt_bind_rescue_path_verdict,
        "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands": byte_mutation_all_subbands_seed_sensitive,
        "axis_tag": "[macOS-CPU advisory]",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "evidence_grade": "local_cpu_smoke_advisory",
        "promotion_eligible": False,
        "score_claim_valid": False,
        "score_claim_axis": None,
        "canonical_equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "canonical_equation_in_domain_context": (
            "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands"
        ),
        "canonical_equation_second_empirical_anchor_pending_append": True,
    }


def append_second_empirical_anchor(
    smoke_result: dict[str, Any],
    output_json_path: Path,
) -> None:
    """Append the SECOND empirical anchor to canonical equation #26.

    Per Catalog #344 sister discipline + Catalog #127 per-call-site custody
    (macOS-CPU advisory non-promotable per Catalog #192). The NEW IN-DOMAIN
    context is ``magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands``,
    distinct from yesterday's FIRST anchor context (direct substitution
    distributional fit).

    Per CLAUDE.md "Forbidden premature KILL": a CARGO-CULTED verdict on
    THIS smoke does NOT mean the cascade is dead — it means the rescue
    path falsified for pair #1 and the cascade pivots to pair #2 (null-
    byte residuals via sparse_packet_ir SRL1).
    """
    from tac.canonical_equations.equation import EmpiricalAnchor
    from tac.canonical_equations.registry import update_equation_with_empirical_anchor
    from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

    try:
        source_artifact_relpath = str(output_json_path.relative_to(REPO_ROOT))
    except ValueError:
        source_artifact_relpath = str(output_json_path)
    source_artifact_sha = _sha256_of_file(output_json_path)

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=source_artifact_sha,
        source_path=source_artifact_relpath,
    )

    # The SECOND anchor captures the pair #1 prediction (per stacking
    # analysis memo §7 ADDITIVE α=0.8 → -0.00200) vs empirical byte budget
    # under the rescue path. The residual magnitude is the absolute delta
    # between empirical and predicted ΔS.
    anchor = EmpiricalAnchor(
        anchor_id=f"second_empirical_anchor_wave_3_magic_codec_pair_1_dwt_residual_smoke_{_utc_now_filename()}",
        measurement_utc=smoke_result["completed_at_utc"],
        inputs={
            "smoke_label": smoke_result["smoke_label"],
            "smoke_lane_id": smoke_result["smoke_lane_id"],
            "smoke_pair_id": smoke_result["smoke_pair_id"],
            "video_path": smoke_result["video_path_repo_relative"],
            "video_sha256": smoke_result["video_sha256"],
            "frame_index": smoke_result["frame_index"],
            "wavelet": smoke_result["wavelet"],
            "dwt_level": smoke_result["dwt_level"],
            "generator_kind": smoke_result["generator_kind"],
            "base_seed_bytes_len": smoke_result["base_seed_bytes_len"],
            "axis_tag": smoke_result["axis_tag"],
            "hardware_substrate": smoke_result["hardware_substrate"],
            "evidence_grade": smoke_result["evidence_grade"],
            "subbands_tested": sorted(smoke_result["per_subband"].keys()),
            "in_domain_context": smoke_result["canonical_equation_in_domain_context"],
            "composition_alpha_estimate_pair_1": smoke_result["composition_alpha_estimate_pair_1"],
        },
        predicted_output={
            "predicted_delta_s": smoke_result["predicted_delta_s_pair_1"],
            "predicted_delta_s_source": (
                "magic_codec_x_todays_cascade_stacking_analysis_20260520_memo_section_7_"
                "pair_1_additive_alpha_0p8_prediction"
            ),
            "sigma_predicted_for_zscore": smoke_result["sigma_predicted_for_zscore"],
            "hypothesis_status": (
                "predicted_pair_1_rescue_path_byte_savings_within_2sigma_of_"
                "additive_alpha_0p8_composition"
            ),
        },
        empirical_output={
            "aggregate_config_a_baseline_brotli_bytes": smoke_result[
                "aggregate_config_a_baseline_brotli_bytes"
            ],
            "aggregate_config_b_procedural_only_bytes": smoke_result[
                "aggregate_config_b_procedural_only_bytes"
            ],
            "aggregate_config_c_procedural_plus_dense_streams_bytes": smoke_result[
                "aggregate_config_c_procedural_plus_dense_streams_bytes"
            ],
            "aggregate_bytes_saved_c_vs_a": smoke_result["aggregate_bytes_saved_c_vs_a"],
            "empirical_delta_s": smoke_result["empirical_delta_s"],
            "residual_zscore_empirical_vs_predicted": smoke_result[
                "residual_zscore_empirical_vs_predicted"
            ],
            "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": smoke_result[
                "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"
            ],
            "rescue_path_net_savings_validated": smoke_result[
                "rescue_path_net_savings_validated"
            ],
            "dwt_bind_rescue_path_verdict": smoke_result["dwt_bind_rescue_path_verdict"],
            "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands": smoke_result[
                "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands"
            ],
            "per_subband_config_a_bytes": {
                name: per["config_a_baseline_brotli_bytes"]
                for name, per in smoke_result["per_subband"].items()
            },
            "per_subband_config_c_bytes": {
                name: per["config_c_procedural_plus_dense_streams_bytes"]
                for name, per in smoke_result["per_subband"].items()
            },
            "per_subband_bytes_saved_c_vs_a": {
                name: per["bytes_saved_c_vs_a"]
                for name, per in smoke_result["per_subband"].items()
            },
            "per_subband_selected_codec_name": {
                name: per["config_c_selection_log"]["selected_codec_name"]
                for name, per in smoke_result["per_subband"].items()
            },
        },
        residual=float(abs(smoke_result["empirical_delta_s"] - smoke_result["predicted_delta_s_pair_1"])),
        source_artifact=source_artifact_relpath,
        measurement_method=(
            "haar_dwt_2_level_y_plane_per_subband_int8_normalization_then_three_way_"
            "comparison_config_a_direct_brotli_baseline_vs_config_b_procedural_only_32B_seed_vs_"
            "config_c_procedural_plus_magic_codec_dense_streams_residual_correction_per_stream_"
            "brotli_lzma_magic_classic_auto_selector_local_macos_cpu_smoke_advisory"
        ),
        provenance=provenance,
    )

    update_equation_with_empirical_anchor(
        equation_id="procedural_codebook_from_seed_compression_savings_v1",
        anchor=anchor,
        agent="claude_subagent",
        subagent_id="wave-3-magic-codec-pair-1-dwt-residual-cpu-smoke-20260520",
        notes=(
            "WAVE-3 pair #1 SECOND empirical anchor; LOCAL macOS-CPU advisory smoke "
            "per Catalog #192 + #127 + #323; NEW IN-DOMAIN context "
            "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands; "
            "validates Carmack MVP-first pair #1 rescue path BEFORE paid GPU "
            "DWT-HNeRV bind L0 SCAFFOLD smoke would have fired"
        ),
    )


def emit_markdown_report(smoke_result: dict[str, Any], md_path: Path) -> None:
    """Write a human-readable Markdown table summarizing the smoke."""
    lines = [
        "<!-- SPDX-License-Identifier: MIT -->",
        "# WAVE-3 Magic-Codec × DWT Detail-Subband Procedural Codebook Residual Smoke",
        "",
        f"**Lane**: `{smoke_result['smoke_lane_id']}`  ",
        f"**Stacking analysis memo**: `{smoke_result['smoke_cascade_stacking_analysis_memo']}`  ",
        f"**Pair ID**: `{smoke_result['smoke_pair_id']}`  ",
        f"**Baseline anchor**: `{smoke_result['smoke_baseline_anchor']}`  ",
        f"**Started**: `{smoke_result['started_at_utc']}`  ",
        f"**Completed**: `{smoke_result['completed_at_utc']}`  ",
        f"**Platform**: `{smoke_result['platform']}`  ",
        "",
        "## Custody (Catalog #127 + #192 + #323)",
        "",
        f"* `axis_tag`: `{smoke_result['axis_tag']}` (NEVER promotable per Catalog #192)",
        f"* `hardware_substrate`: `{smoke_result['hardware_substrate']}`",
        f"* `evidence_grade`: `{smoke_result['evidence_grade']}`",
        f"* `promotion_eligible`: `{smoke_result['promotion_eligible']}`",
        f"* `score_claim_valid`: `{smoke_result['score_claim_valid']}`",
        "",
        "## Inputs",
        "",
        f"* Video: `{smoke_result['video_path_repo_relative']}` (sha256 `{smoke_result['video_sha256'][:16]}...`)",
        f"* Frame index: `{smoke_result['frame_index']}`",
        f"* Frame shape: `{smoke_result['frame_height']} x {smoke_result['frame_width']}` Y plane",
        f"* Y plane sha256: `{smoke_result['y_plane_sha256'][:16]}...`",
        f"* Wavelet: `{smoke_result['wavelet']}` @ level `{smoke_result['dwt_level']}`",
        f"* PRNG generator kind: `{smoke_result['generator_kind']}`",
        f"* Base seed length: `{smoke_result['base_seed_bytes_len']}` bytes",
        "",
        "## 3-way apples-to-apples byte budget (per subband)",
        "",
        "| Subband | N pixels | Config A baseline (brotli) | Config B procedural-only (32B seed) | Config C procedural + dense-streams | Bytes saved (C vs A) | Selected codec | Residual mean | Residual std |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name in sorted(smoke_result["per_subband"].keys()):
        per = smoke_result["per_subband"][name]
        lines.append(
            f"| {name} | {per['n_pixels']} "
            f"| {per['config_a_baseline_brotli_bytes']} "
            f"| {per['config_b_procedural_only_bytes']} "
            f"| {per['config_c_procedural_plus_dense_streams_bytes']} "
            f"| {per['bytes_saved_c_vs_a']:+d} "
            f"| {per['config_c_selection_log']['selected_codec_name']} "
            f"| {per['residual_distribution']['mean']:.4f} "
            f"| {per['residual_distribution']['std']:.4f} |"
        )
    lines += [
        "",
        "## Aggregate canonical-equation verdict",
        "",
        f"* Aggregate Config A baseline (brotli): `{smoke_result['aggregate_config_a_baseline_brotli_bytes']}` bytes",
        f"* Aggregate Config B procedural-only: `{smoke_result['aggregate_config_b_procedural_only_bytes']}` bytes (KL=1.638 nats per yesterday's anchor; substrate-corrupting at inflate)",
        f"* Aggregate Config C procedural + dense-stream residuals: `{smoke_result['aggregate_config_c_procedural_plus_dense_streams_bytes']}` bytes",
        f"* Aggregate bytes saved (C vs A): `{smoke_result['aggregate_bytes_saved_c_vs_a']:+d}`",
        f"* Empirical ΔS: `{smoke_result['empirical_delta_s']:+.6f}`",
        f"* Predicted ΔS (pair #1 per stacking analysis memo §7): `{smoke_result['predicted_delta_s_pair_1']:+.6f}`",
        f"* Composition_alpha estimate: `{smoke_result['composition_alpha_estimate_pair_1']}` (ADDITIVE)",
        f"* Residual zscore (empirical vs predicted): `{smoke_result['residual_zscore_empirical_vs_predicted']:.4f}`",
        f"* **Canonical equation #26 verdict**: `{smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']}` (at 2σ threshold)",
        f"* **Rescue path net savings validated**: `{'YES' if smoke_result['rescue_path_net_savings_validated'] else 'NO'}`",
        f"* **DWT bind rescue path verdict**: `{smoke_result['dwt_bind_rescue_path_verdict']}`",
        f"* **Catalog #272 byte-mutation smoke verdict**: `{'PASSED' if smoke_result['byte_mutation_smoke_aggregate_seed_sensitive_all_subbands'] else 'FAILED'}` (all subbands seed-sensitive)",
        "",
        "## Equation linkage",
        "",
        f"* Canonical equation: `{smoke_result['canonical_equation_id']}`",
        f"* NEW IN-DOMAIN context: `{smoke_result['canonical_equation_in_domain_context']}`",
        f"* SECOND empirical anchor pending append: `{smoke_result['canonical_equation_second_empirical_anchor_pending_append']}`",
        "",
        "## Discipline citations",
        "",
        "* CLAUDE.md \"MPS auth eval is NOISE\" — macOS-CPU is NEVER score truth",
        "* CLAUDE.md \"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE\" — `[macOS-CPU advisory]` is non-promotable",
        "* CLAUDE.md \"Apples-to-apples evidence discipline\" — axis labels preserved",
        "* CLAUDE.md \"Bit-level deconstruction and entropy discipline\" — 3-way comparison at byte level",
        "* CLAUDE.md \"Forbidden premature KILL\" — CARGO-CULTED pair #1 verdict pivots cascade to pair #2; does NOT kill the rescue paradigm",
        "* Catalog #127 — custody triple axis × hardware × evidence_grade",
        "* Catalog #192 — macOS-CPU advisory not promotable without Linux x86_64 verification",
        "* Catalog #272 — distinguishing-feature byte-mutation smoke (3-subband seed-sensitivity verified)",
        "* Catalog #277 — wavelet multi-scale canonical helper",
        "* Catalog #287 — placeholder-rationale rejection (zero `<rationale>` / `<reason>` literals)",
        "* Catalog #309 — horizon_class=`frontier_breaking_enabler`",
        "* Catalog #318 — master-gradient null-space surface (sister)",
        "* Catalog #323 — canonical Provenance umbrella",
        "* Catalog #324 — predicted_band validation (predicted_band_validation_status pending_post_training)",
        "* Catalog #335 — canonical consumer contract (sister auto-discoverable)",
        "* Catalog #344 — canonical equation SECOND `anchor_appended` event with NEW IN-DOMAIN context",
        "",
        "## Source",
        "",
        "* MAGIC CODEC × CASCADE STACKING ANALYSIS memo: `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md`",
        "* Sister landing memo (yesterday's baseline): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dwt_detail_subband_procedural_cpu_smoke_landed_20260520.md`",
        "* Op-routable: stacking analysis memo §14 Top-3 #1 (FREE CPU smoke; pair #1)",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WAVE-3 pair #1 magic-codec × DWT detail-subband procedural residual "
            "local macOS-CPU smoke. NON-promotable; macOS-CPU advisory per "
            "CLAUDE.md \"MPS auth eval is NOISE\" + Catalog #192."
        )
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to contest video (default: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help=f"Frame index to decode for smoke (default: {DEFAULT_FRAME_INDEX}).",
    )
    parser.add_argument(
        "--base-seed-hex",
        type=str,
        default=None,
        help=(
            "Hex-encoded base seed bytes (default: sha256 of canonical "
            "WAVE-3 pair #1 label). Length must be 16-512 hex chars."
        ),
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default=DEFAULT_WAVELET,
        help=f"pywt wavelet name (default: {DEFAULT_WAVELET}).",
    )
    parser.add_argument(
        "--dwt-level",
        type=int,
        default=DEFAULT_DWT_LEVEL,
        help=f"DWT decomposition level (default: {DEFAULT_DWT_LEVEL}).",
    )
    parser.add_argument(
        "--generator-kind",
        type=str,
        default=DEFAULT_GENERATOR_KIND,
        choices=("xorshift", "lcg", "pcg64"),
        help=(
            "Procedural-codebook generator kind (default: "
            f"{DEFAULT_GENERATOR_KIND}; matches Catalog #344 canonical equation #26)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory under experiments/results/ (default: "
            "experiments/results/magic_codec_dense_streams_dwt_residual_smoke_<utc>/)."
        ),
    )
    parser.add_argument(
        "--skip-canonical-equation-append",
        action="store_true",
        help=(
            "Skip the canonical-equation anchor_appended event (for "
            "dry-run / replay smoke runs that should NOT pollute the "
            "canonical posterior). The smoke JSON + MD artifacts still emit."
        ),
    )
    args = parser.parse_args(argv)

    if not args.video_path.exists():
        print(f"FATAL: video_path={args.video_path} not found", file=sys.stderr)
        return 2

    if args.base_seed_hex is None:
        base_seed_bytes = hashlib.sha256(
            b"wave_3_magic_codec_dense_streams_dwt_residual_cpu_smoke_20260520_pair_1"
        ).digest()
    else:
        try:
            base_seed_bytes = bytes.fromhex(args.base_seed_hex)
        except ValueError as exc:
            print(f"FATAL: --base-seed-hex invalid hex: {exc}", file=sys.stderr)
            return 2
        if not (8 <= len(base_seed_bytes) <= 256):
            print(
                f"FATAL: --base-seed-hex length {len(base_seed_bytes)} bytes "
                "out of canonical range [8, 256]",
                file=sys.stderr,
            )
            return 2

    if args.output_dir is None:
        output_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"magic_codec_dense_streams_dwt_residual_smoke_{_utc_now_filename()}"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "smoke_result.json"
    md_path = output_dir / "smoke_result.md"

    print(f"[wave-3-magic-codec-pair-1] decoding video {args.video_path}...", file=sys.stderr)
    smoke_result = run_smoke(
        video_path=args.video_path,
        frame_index=args.frame_index,
        base_seed_bytes=base_seed_bytes,
        wavelet=args.wavelet,
        dwt_level=args.dwt_level,
        generator_kind=args.generator_kind,
    )

    print(
        "[wave-3-magic-codec-pair-1] writing smoke_result.json + smoke_result.md",
        file=sys.stderr,
    )
    json_path.write_text(
        json.dumps(smoke_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    emit_markdown_report(smoke_result, md_path)

    if not args.skip_canonical_equation_append:
        print(
            "[wave-3-magic-codec-pair-1] appending SECOND empirical anchor to canonical "
            "equation procedural_codebook_from_seed_compression_savings_v1 "
            "(NEW IN-DOMAIN context magic_codec_dense_streams_residual_correction_on_"
            "dwt_detail_subbands)...",
            file=sys.stderr,
        )
        append_second_empirical_anchor(smoke_result, json_path)
        print(
            "[wave-3-magic-codec-pair-1] anchor_appended event landed (Catalog #344 sister)",
            file=sys.stderr,
        )
    else:
        print(
            "[wave-3-magic-codec-pair-1] SKIPPED canonical equation anchor_appended event "
            "(--skip-canonical-equation-append)",
            file=sys.stderr,
        )

    try:
        out_dir_display = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        out_dir_display = str(output_dir)
    print(
        f"[wave-3-magic-codec-pair-1] DONE: output_dir={out_dir_display}",
        file=sys.stderr,
    )
    print(
        f"[wave-3-magic-codec-pair-1] config_a_baseline={smoke_result['aggregate_config_a_baseline_brotli_bytes']}B "
        f"config_b_procedural_only={smoke_result['aggregate_config_b_procedural_only_bytes']}B "
        f"config_c_procedural_plus_dense_streams={smoke_result['aggregate_config_c_procedural_plus_dense_streams_bytes']}B "
        f"bytes_saved_c_vs_a={smoke_result['aggregate_bytes_saved_c_vs_a']:+d} "
        f"empirical_delta_s={smoke_result['empirical_delta_s']:+.6f} "
        f"predicted_delta_s={smoke_result['predicted_delta_s_pair_1']:+.6f} "
        f"residual_zscore={smoke_result['residual_zscore_empirical_vs_predicted']:.4f} "
        f"verdict={smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']} "
        f"rescue_path={smoke_result['dwt_bind_rescue_path_verdict']} "
        f"byte_mutation_seed_sensitive={smoke_result['byte_mutation_smoke_aggregate_seed_sensitive_all_subbands']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
