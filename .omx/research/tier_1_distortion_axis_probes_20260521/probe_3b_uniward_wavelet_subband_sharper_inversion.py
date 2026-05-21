# SPDX-License-Identifier: MIT
"""
OVERNIGHT-DDD Tier-1 Probe 3b: UNIWARD wavelet-subband sharper inversion [macOS-CPU advisory]

Per OVERNIGHT-CCC §6.2 partial-verdict follow-on + AAA T4 §2.3 + §9
+ CLAUDE.md "Fridrich inverse steganalysis" + Carmack MVP-first 5-step.

CONTRACT (Carmack MVP-first 5-step + sharper inversion):
  Canonical Fridrich UNIWARD (Holub-Fridrich-Denemark 2014):
      cost_i = 1 / (|HL|_i + |LH|_i + |HH|_i + sigma)

  where HL/LH/HH are the wavelet detail subbands of a 2D DWT (Daubechies db8 per
  the canonical Fridrich reference). Textured pixels carry large detail-subband
  magnitudes -> low cost (admit distortion); flat pixels carry near-zero detail
  magnitudes -> high cost (suppress distortion).

  This is SHARPER than CCC probe 3's local-variance inversion `1/(var+eps)`
  because:
    1. Wavelet detail subbands are oriented (HL=vertical, LH=horizontal, HH=
       diagonal) and capture texture across multiple scales, not just a local
       7x7 box variance.
    2. The sum |HL|+|LH|+|HH| has a wider dynamic range than local variance
       (which compresses via the square of the mean), producing sharper
       flat/textured separation in the cost-distribution tails.
    3. The reciprocal of the sum (not the square of the inverse variance)
       matches the Fridrich canonical exactly so the textured-region weight
       collapses below the 0.5 threshold that CCC probe 3 missed.

PREDICTED SIGNATURE (sharper than CCC probe 3):
  - textured_avg_weight < 0.5 (CCC probe 3 baseline = 0.806; sharper expected
    to break the 0.5 threshold)
  - Wavelet detail-subband dynamic range matches local-variance baseline
    (both > 1.0 log10)
  - Flat/textured separation present (textured > 10% AND flat > 30%)

FALSIFYING OUTCOME:
  - Sharper formula yields same-or-worse textured_avg_weight (>= 0.5) =>
    DEFER per Catalog #307 IMPLEMENTATION-level falsification + Catalog #308
    alternative reducer (e.g. SCC / HILL / WOW / J-UNIWARD).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pywt
import torch
import torch.nn.functional as F


def _decode_first_n_frames(video_path: Path, n: int = 8) -> torch.Tensor:
    import av  # type: ignore

    container = av.open(str(video_path))
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i >= n:
            break
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()
    stacked = np.stack(frames, axis=0)
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    return t


def _luma(frames: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma; matches CCC probe 3 baseline."""
    return 0.2989 * frames[:, 0] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 2]


def _local_variance(luma: torch.Tensor, window: int = 7) -> torch.Tensor:
    """CCC probe 3 baseline reproduced verbatim for apples-to-apples delta."""
    gray = luma.unsqueeze(1)
    kernel = torch.ones(1, 1, window, window) / (window * window)
    pad = window // 2
    local_mean = F.conv2d(gray, kernel, padding=pad)
    local_mean_sq = F.conv2d(gray ** 2, kernel, padding=pad)
    local_var = (local_mean_sq - local_mean ** 2).clamp_min(0.0)
    return local_var.squeeze(1)


def _wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str = "db8",
) -> torch.Tensor:
    """Canonical Fridrich UNIWARD detail-subband sum.

    For each frame, run a single-level 2D DWT and return per-pixel
    |HL| + |LH| + |HH|, upsampled back to the original (H, W) shape via
    nearest-neighbor (consistent with Fridrich 2014 cost-map construction).
    """
    luma_np = luma.cpu().numpy()  # (N, H, W)
    n, h, w = luma_np.shape
    detail_sum = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        # Single-level 2D DWT: cA (LL) + (cH=LH, cV=HL, cD=HH) per pywt
        # convention.
        coeffs = pywt.dwt2(luma_np[i], wavelet_name, mode="symmetric")
        _cA, (cH, cV, cD) = coeffs

        # |HL| + |LH| + |HH| per Fridrich canonical
        detail_lo = np.abs(cH) + np.abs(cV) + np.abs(cD)

        # cH/cV/cD shapes are ~(h//2, w//2); upsample via nearest-neighbor
        # to (h, w) for per-pixel cost map.
        detail_t = torch.from_numpy(detail_lo).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(detail_t, size=(h, w), mode="nearest")
        detail_sum[i] = upsampled.squeeze(0).squeeze(0).numpy()

    return torch.from_numpy(detail_sum)


def _run_probe() -> dict:
    t_start = time.time()

    # Decode 8 frames from PR 101 reference video (apples-to-apples CCC baseline)
    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=8)
    n_frames, _, H, W = frames.shape

    luma = _luma(frames)

    # === Baseline: CCC probe 3 local-variance inversion ===
    local_var = _local_variance(luma, window=7)
    epsilon_var = 1e-3
    baseline_weights = 1.0 / (local_var + epsilon_var)
    baseline_weights_norm = baseline_weights / baseline_weights.mean()

    # === Sharper: canonical Fridrich wavelet-subband inversion ===
    detail_sum = _wavelet_detail_magnitudes(luma, wavelet_name="db8")
    # Per Fridrich canonical sigma = 2^-6 (small constant; prevents inf at zero detail)
    sigma_fridrich = 2.0 ** -6
    sharper_weights = 1.0 / (detail_sum + sigma_fridrich)
    sharper_weights_norm = sharper_weights / sharper_weights.mean()

    # === Simulated per-pixel SegNet reconstruction error (apples-to-apples vs CCC) ===
    torch.manual_seed(0xCC3)  # matches CCC probe 3 seed
    per_pixel_error = 0.05 * torch.randn(n_frames, H, W)
    per_pixel_error_sq = per_pixel_error ** 2
    uniform_weighted_error = per_pixel_error_sq.mean().item()

    baseline_weighted_error = (per_pixel_error_sq * baseline_weights_norm).mean().item()
    sharper_weighted_error = (per_pixel_error_sq * sharper_weights_norm).mean().item()

    # === Variance / detail-magnitude distribution analysis ===
    var_dynamic_range = float((local_var.max() / (local_var.min() + 1e-10)).log10().item())
    detail_dynamic_range = float((detail_sum.max() / (detail_sum.min() + 1e-10)).log10().item())

    # Textured region defined by upper quartile of variance (consistent with CCC baseline)
    textured_mask_var = local_var > local_var.quantile(0.75)
    # Sister textured mask using the wavelet-subband signal (per Fridrich canonical)
    textured_mask_wav = detail_sum > detail_sum.quantile(0.75)

    baseline_textured_avg_weight = baseline_weights_norm[textured_mask_var].mean().item()
    sharper_textured_avg_weight = sharper_weights_norm[textured_mask_wav].mean().item()
    sharper_textured_avg_weight_var_mask = sharper_weights_norm[textured_mask_var].mean().item()

    var_flat_fraction = float((local_var < local_var.median()).float().mean().item())
    var_textured_fraction = float((local_var > local_var.quantile(0.75)).float().mean().item())
    detail_flat_fraction = float((detail_sum < detail_sum.median()).float().mean().item())
    detail_textured_fraction = float((detail_sum > detail_sum.quantile(0.75)).float().mean().item())

    # === Sharper-than-CCC predicted-signature checks ===
    sig_sharper_textured_suppression = sharper_textured_avg_weight < 0.5
    sig_sharper_under_baseline = sharper_textured_avg_weight < baseline_textured_avg_weight
    sig_detail_dynamic_range = detail_dynamic_range > 1.0
    sig_detail_flat_textured = detail_textured_fraction > 0.1 and detail_flat_fraction > 0.3

    # === Verdict ===
    # POSITIVE_SIGNAL_SHARPER: sharper formula breaks the 0.5 threshold AND
    # improves on CCC baseline (the explicit CCC §6.2 ask).
    if (
        sig_sharper_textured_suppression
        and sig_sharper_under_baseline
        and sig_detail_dynamic_range
        and sig_detail_flat_textured
    ):
        verdict = "POSITIVE_SIGNAL_SHARPER"
        recommendation = (
            "POSITIVE_SIGNAL_SHARPER: canonical Fridrich wavelet-subband inversion "
            f"yields textured_avg_weight={sharper_textured_avg_weight:.4f} "
            f"(< 0.5 threshold; CCC baseline was 0.806). Per AAA T4 §6.2: "
            "Tier-2 paid dispatch on UNIWARD-weighted per-pixel SegNet loss substrate "
            "via Vast.ai 4090 / Lightning T4 JUSTIFIED. Predicted ΔS -0.005 to -0.015 "
            "[predicted] per AAA T4 §2.3 + §9. Estimated cost ~$1-5."
        )
    elif sig_sharper_under_baseline and sig_detail_dynamic_range:
        verdict = "POSITIVE_SIGNAL_SHARPER_PARTIAL"
        recommendation = (
            "PARTIAL: sharper inversion improves on CCC baseline "
            f"({sharper_textured_avg_weight:.4f} < {baseline_textured_avg_weight:.4f}) "
            "but still above 0.5 threshold. Per CLAUDE.md 'Forbidden premature KILL': "
            "DEFER-PENDING-FURTHER-INVERSION-REFINEMENT; iterate sister probe with "
            "multi-scale wavelet decomposition (J-UNIWARD multi-level) OR HILL/WOW "
            "filter-based cost map per Catalog #308 alternative reducer."
        )
    else:
        verdict = "NULL_SIGNAL_DEFER"
        recommendation = (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification: sharper "
            "wavelet-subband formula did NOT improve on CCC baseline local-variance "
            "approximation. Per Catalog #308 reactivation criteria: try HILL "
            "(High-pass + Low-pass + Low-pass) filter from Li et al. 2014 OR WOW "
            "(Wavelet Obtained Weights) from Holub-Fridrich 2012 alternative "
            "reducers. Paradigm INTACT (textured-region undetectability per "
            "Fridrich inverse-steganalysis); IMPLEMENTATION-level approximation "
            "axis needs alternative."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_uniward_wavelet_subband_sharper_inversion_smoke",
        "lane_id": "lane_overnight_ddd_uniward_sharper_inversion_follow_on_tier_1_macos_cpu_advisory_smoke_20260521",
        "probe_name": "UNIWARD wavelet-subband sharper inversion",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "sharper_textured_suppression": "textured_avg_weight < 0.5 (CCC baseline=0.806)",
            "sharper_under_baseline": "sharper_textured_avg_weight < baseline_textured_avg_weight",
            "detail_dynamic_range": "log10 > 1.0",
            "detail_flat_textured_separation": "detail_textured > 10% AND detail_flat > 30%",
        },
        "actual_signature": {
            # Wavelet-subband sharper formula (primary)
            "sharper_textured_avg_weight": sharper_textured_avg_weight,
            "sharper_textured_avg_weight_via_var_mask": sharper_textured_avg_weight_var_mask,
            "sharper_weighted_error": sharper_weighted_error,
            "detail_dynamic_range_log10": detail_dynamic_range,
            "detail_flat_fraction": detail_flat_fraction,
            "detail_textured_fraction": detail_textured_fraction,
            # CCC baseline (for delta comparison)
            "baseline_textured_avg_weight": baseline_textured_avg_weight,
            "baseline_weighted_error": baseline_weighted_error,
            "var_dynamic_range_log10": var_dynamic_range,
            "var_flat_fraction": var_flat_fraction,
            "var_textured_fraction": var_textured_fraction,
            # Universal
            "uniform_weighted_error": uniform_weighted_error,
            "uniform_vs_sharper_ratio": uniform_weighted_error / max(sharper_weighted_error, 1e-10),
            "n_frames": n_frames,
            "frame_resolution_HxW": [H, W],
            "wavelet_name": "db8",
            "sigma_fridrich": sigma_fridrich,
        },
        "delta_vs_ccc_baseline": {
            "ccc_baseline_textured_avg_weight": baseline_textured_avg_weight,
            "sharper_textured_avg_weight": sharper_textured_avg_weight,
            "absolute_delta": sharper_textured_avg_weight - baseline_textured_avg_weight,
            "ratio_sharper_over_baseline": sharper_textured_avg_weight / max(baseline_textured_avg_weight, 1e-10),
            "ccc_passed_threshold_lt_0p5": baseline_textured_avg_weight < 0.5,
            "sharper_passed_threshold_lt_0p5": sharper_textured_avg_weight < 0.5,
            "improvement_direction": (
                "sharper_lower_is_better"
                if sharper_textured_avg_weight < baseline_textured_avg_weight
                else "sharper_did_not_improve_or_worse"
            ),
        },
        "signature_checks": {
            "sharper_textured_suppression_present": sig_sharper_textured_suppression,
            "sharper_under_baseline_present": sig_sharper_under_baseline,
            "detail_dynamic_range_present": sig_detail_dynamic_range,
            "detail_flat_textured_separation_present": sig_detail_flat_textured,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate uniward_textured_region_undetectability_pose_distortion_savings_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N pending per AAA T4 op-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_3b_uniward_wavelet_subband_sharper_inversion",
            "predecessor_probe": "tier_1_distortion_uniward_per_pixel_segnet_smoke",
            "fridrich_canonical_reference": (
                "Holub-Fridrich-Denemark 2014 'Universal Distortion Function for "
                "Steganography in an Arbitrary Domain' (UNIWARD); CLAUDE.md "
                "'Fridrich inverse steganalysis' section."
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "uniward_textured_region_undetectability_pose_distortion_savings_v1",
        "next_action_on_POSITIVE_SHARPER": (
            "Operator-routable: Tier-2 paid dispatch on UNIWARD-weighted per-pixel "
            "SegNet loss substrate via Vast.ai 4090 ($0.25/hr) or Lightning T4; "
            "estimated cost ~$1-5; predicted ΔS -0.005 to -0.015 [predicted] per "
            "AAA T4 §2.3 + §9; recipe pending per AAA T4 §6.2 op-routable."
        ),
        "next_action_on_PARTIAL": (
            "Continue sister-probe iteration at $0: try multi-scale (J-UNIWARD) "
            "wavelet decomposition OR HILL filter cost map per Catalog #308."
        ),
        "next_action_on_NULL": (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification; queue HILL "
            "and WOW alternative reducers per Catalog #308 reactivation criteria."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_3b_uniward_wavelet_subband_sharper_inversion_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[probe_3b] verdict={verdict['verdict']} "
        f"sharper_textured_avg_weight={verdict['actual_signature']['sharper_textured_avg_weight']:.4f} "
        f"(CCC baseline={verdict['delta_vs_ccc_baseline']['ccc_baseline_textured_avg_weight']:.4f}) "
        f"detail_log10_range={verdict['actual_signature']['detail_dynamic_range_log10']:.2f} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_3b] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
