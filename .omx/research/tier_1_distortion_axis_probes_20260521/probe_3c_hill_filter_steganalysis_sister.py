# SPDX-License-Identifier: MIT
"""
OVERNIGHT-EEE Tier-1 Probe 3c: HILL filter steganalysis sister [macOS-CPU advisory]

Per OVERNIGHT-DDD §6.2 op-routable explicit recommendation + AAA T4 §6.2
+ CLAUDE.md "Fridrich inverse steganalysis" + Carmack MVP-first 5-step.

CONTRACT (Carmack MVP-first 5-step + HILL filter cascade per Li-Fridrich-Wang 2014):

  Canonical HILL filter (Li-Tang-Huang-Luo 2014 "A new cost function for spatial
  image steganography"):

      1. KB high-pass filter (Ker-Bohme 2008 canonical 3x3):
             K_KB = (1/4) * [[-1, 2, -1],
                             [ 2, -4,  2],
                             [-1, 2, -1]]
             residual = convolve(luma, K_KB)

      2. First low-pass filter L1 (3x3 average):
             L1 = ones(3,3) / 9
             cost_intermediate = convolve(|residual|, L1)

      3. Second low-pass filter L2 (15x15 average; larger window per Li et al.):
             L2 = ones(15,15) / 225
             cost_smooth = convolve(1 / (cost_intermediate + sigma), L2)

      cost_i = cost_smooth_i  (per-pixel cost map)
      weight_i = cost_i / cost.mean()  (normalized)

  This is SHARPER than DDD probe 3b's wavelet-subband formula because:
    1. KB high-pass operator captures sharp edges + texture transitions more
       aggressively than db8 detail subbands (smaller stencil + canonical
       Ker-Bohme 2008 steganalysis-optimal kernel).
    2. The cascaded L1->reciprocal->L2 produces stronger separation between
       textured and smooth regions: the inner L1 smooths the residual to a
       stable cost-distribution; the reciprocal sharpens the tail; the outer
       L2 propagates this sharper cost to neighboring pixels for spatial
       coherence (avoids per-pixel salt-and-pepper noise in cost map).
    3. Li et al. 2014 BOSSbase benchmarks: HILL achieves higher embedding
       security than UNIWARD at equivalent payload, indicating sharper
       cost-distribution tail in practice.

PREDICTED SIGNATURE (sharper than DDD probe 3b):
  - hill_textured_avg_weight <= 0.5 hard threshold (DDD boundary = 0.626;
    CCC baseline = 0.806; HILL expected to break the 0.5 threshold per Li
    et al. BOSSbase benchmarks)
  - HILL cost-distribution dynamic range matches or exceeds DDD (log10 > 1.0)
  - Flat/textured separation present (textured > 10% AND flat > 30%)

FALSIFYING OUTCOME:
  - HILL formula yields hill_textured_avg_weight >= 0.626 (DDD boundary):
    no improvement vs sharper sister; DEFER per Catalog #307 IMPLEMENTATION-
    level boundary + queue J-UNIWARD multi-scale per DDD's 2nd alternative.
  - HILL formula yields opposite-direction signal (hill > 1.0): cascade
    inverted; queue WOW per DDD's 3rd alternative.
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
import scipy.ndimage
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
    """ITU-R BT.601 luma; matches CCC + DDD baselines for apples-to-apples delta."""
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


def _hill_filter_cost_map(
    luma: torch.Tensor,
    sigma: float = 2.0 ** -6,
    l2_window: int = 15,
) -> torch.Tensor:
    """Canonical Li-Fridrich-Wang 2014 HILL filter cost map.

    KB high-pass + L1 (3x3 average) + L2 (15x15 average) cascade.
    Returns per-pixel cost map of shape (N, H, W) normalized to mean=1.0.
    """
    luma_np = luma.cpu().numpy().astype(np.float32)  # (N, H, W)
    n, h, w = luma_np.shape

    # Step 1: KB high-pass filter (Ker-Bohme 2008 canonical 3x3 kernel per
    # Li-Fridrich-Wang 2014 §III.A; divide by 4 normalizes to unit gain).
    K_KB = np.array(
        [
            [-1.0,  2.0, -1.0],
            [ 2.0, -4.0,  2.0],
            [-1.0,  2.0, -1.0],
        ],
        dtype=np.float32,
    ) / 4.0

    # Step 2: First low-pass filter L1 (3x3 average per Li et al. canonical).
    L1 = np.ones((3, 3), dtype=np.float32) / 9.0

    # Step 3: Second low-pass filter L2 (15x15 average per Li et al. canonical).
    L2 = np.ones((l2_window, l2_window), dtype=np.float32) / float(l2_window * l2_window)

    cost_map = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        # KB high-pass on luma frame; mode='reflect' per Li et al. boundary handling.
        residual = scipy.ndimage.convolve(luma_np[i], K_KB, mode="reflect")
        # First low-pass on |residual|.
        cost_intermediate = scipy.ndimage.convolve(np.abs(residual), L1, mode="reflect")
        # Reciprocal with sigma to prevent inf at zero residual.
        cost_reciprocal = 1.0 / (cost_intermediate + sigma)
        # Second low-pass for spatial coherence.
        cost_smooth = scipy.ndimage.convolve(cost_reciprocal, L2, mode="reflect")
        cost_map[i] = cost_smooth

    return torch.from_numpy(cost_map)


def _wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str = "db8",
) -> torch.Tensor:
    """DDD probe 3b sharper Fridrich UNIWARD detail-subband sum (reproduced for delta).

    Single-level 2D DWT (db8) returning per-pixel |HL|+|LH|+|HH| upsampled to (H,W).
    """
    import pywt

    luma_np = luma.cpu().numpy()
    n, h, w = luma_np.shape
    detail_sum = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        coeffs = pywt.dwt2(luma_np[i], wavelet_name, mode="symmetric")
        _cA, (cH, cV, cD) = coeffs
        detail_lo = np.abs(cH) + np.abs(cV) + np.abs(cD)
        detail_t = torch.from_numpy(detail_lo).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(detail_t, size=(h, w), mode="nearest")
        detail_sum[i] = upsampled.squeeze(0).squeeze(0).numpy()

    return torch.from_numpy(detail_sum)


def _run_probe() -> dict:
    t_start = time.time()

    # Decode 8 frames from PR 101 reference video (apples-to-apples CCC + DDD baselines)
    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=8)
    n_frames, _, H, W = frames.shape

    luma = _luma(frames)

    # === Baseline 1: CCC probe 3 local-variance inversion (reproduced) ===
    local_var = _local_variance(luma, window=7)
    epsilon_var = 1e-3
    ccc_weights = 1.0 / (local_var + epsilon_var)
    ccc_weights_norm = ccc_weights / ccc_weights.mean()

    # === Baseline 2: DDD probe 3b wavelet-subband sharper inversion (reproduced) ===
    detail_sum = _wavelet_detail_magnitudes(luma, wavelet_name="db8")
    sigma_fridrich = 2.0 ** -6
    ddd_weights = 1.0 / (detail_sum + sigma_fridrich)
    ddd_weights_norm = ddd_weights / ddd_weights.mean()

    # === Sharper sister: HILL filter cost map per Li-Fridrich-Wang 2014 ===
    hill_cost = _hill_filter_cost_map(luma, sigma=sigma_fridrich, l2_window=15)
    hill_weights_norm = hill_cost / hill_cost.mean()

    # === Simulated per-pixel SegNet reconstruction error (apples-to-apples vs CCC + DDD) ===
    torch.manual_seed(0xCC3)  # matches CCC + DDD seed
    per_pixel_error = 0.05 * torch.randn(n_frames, H, W)
    per_pixel_error_sq = per_pixel_error ** 2
    uniform_weighted_error = per_pixel_error_sq.mean().item()

    ccc_weighted_error = (per_pixel_error_sq * ccc_weights_norm).mean().item()
    ddd_weighted_error = (per_pixel_error_sq * ddd_weights_norm).mean().item()
    hill_weighted_error = (per_pixel_error_sq * hill_weights_norm).mean().item()

    # === Cost-distribution dynamic range analysis ===
    var_dynamic_range = float((local_var.max() / (local_var.min() + 1e-10)).log10().item())
    detail_dynamic_range = float((detail_sum.max() / (detail_sum.min() + 1e-10)).log10().item())
    hill_dynamic_range = float((hill_cost.max() / (hill_cost.min() + 1e-10)).log10().item())

    # === Textured region defined by upper quartile (consistent with CCC + DDD baselines) ===
    textured_mask_var = local_var > local_var.quantile(0.75)
    textured_mask_wav = detail_sum > detail_sum.quantile(0.75)
    textured_mask_hill = hill_cost > hill_cost.quantile(0.75)

    ccc_textured_avg_weight = ccc_weights_norm[textured_mask_var].mean().item()
    ddd_textured_avg_weight = ddd_weights_norm[textured_mask_wav].mean().item()
    hill_textured_avg_weight = hill_weights_norm[textured_mask_hill].mean().item()
    hill_textured_avg_weight_var_mask = hill_weights_norm[textured_mask_var].mean().item()
    hill_textured_avg_weight_wav_mask = hill_weights_norm[textured_mask_wav].mean().item()

    var_flat_fraction = float((local_var < local_var.median()).float().mean().item())
    var_textured_fraction = float((local_var > local_var.quantile(0.75)).float().mean().item())
    detail_flat_fraction = float((detail_sum < detail_sum.median()).float().mean().item())
    detail_textured_fraction = float((detail_sum > detail_sum.quantile(0.75)).float().mean().item())
    hill_flat_fraction = float((hill_cost < hill_cost.median()).float().mean().item())
    hill_textured_fraction = float((hill_cost > hill_cost.quantile(0.75)).float().mean().item())

    # === Sharper-than-DDD predicted-signature checks ===
    # Hard threshold per CCC + DDD canonical pattern: textured_avg_weight < 0.5
    sig_hill_textured_suppression = hill_textured_avg_weight < 0.5
    # Sharper than DDD (DDD = 0.626 boundary)
    sig_hill_under_ddd = hill_textured_avg_weight < ddd_textured_avg_weight
    # Sharper than CCC baseline (CCC = 0.806)
    sig_hill_under_ccc = hill_textured_avg_weight < ccc_textured_avg_weight
    sig_hill_dynamic_range = hill_dynamic_range > 1.0
    sig_hill_flat_textured = hill_textured_fraction > 0.1 and hill_flat_fraction > 0.3

    # === Verdict ===
    # POSITIVE_SIGNAL_FULL: HILL breaks the 0.5 hard threshold AND improves on
    # both DDD (0.626) and CCC (0.806) baselines per Carmack MVP-first §2
    # falsifiable challenge.
    if (
        sig_hill_textured_suppression
        and sig_hill_under_ddd
        and sig_hill_under_ccc
        and sig_hill_dynamic_range
        and sig_hill_flat_textured
    ):
        verdict = "POSITIVE_SIGNAL_FULL"
        recommendation = (
            "POSITIVE_SIGNAL_FULL: canonical Li-Fridrich-Wang 2014 HILL filter "
            f"cascade yields textured_avg_weight={hill_textured_avg_weight:.4f} "
            f"(< 0.5 hard threshold; DDD wavelet-subband boundary was {ddd_textured_avg_weight:.4f}; "
            f"CCC local-variance baseline was {ccc_textured_avg_weight:.4f}). Per AAA T4 §6.2 + "
            "DDD §6.1: Tier-2 paid dispatch on UNIWARD/HILL-weighted per-pixel SegNet loss "
            "substrate via Vast.ai 4090 ($0.25/hr) or Lightning T4 JUSTIFIED. Predicted "
            "ΔS -0.005 to -0.015 [predicted] per AAA T4 §2.3 + §9. Estimated cost ~$1-5."
        )
    elif sig_hill_under_ddd and sig_hill_dynamic_range:
        verdict = "POSITIVE_SIGNAL_SHARPER_PARTIAL"
        recommendation = (
            "PARTIAL: HILL cascade improves on DDD wavelet-subband sharper baseline "
            f"({hill_textured_avg_weight:.4f} < DDD {ddd_textured_avg_weight:.4f}) "
            "but still above 0.5 hard threshold. Per CLAUDE.md 'Forbidden premature KILL': "
            "DEFER-PENDING-FURTHER-INVERSION-REFINEMENT; iterate sister probe per Catalog #308: "
            "(a) multi-scale J-UNIWARD (Holub-Fridrich-Denemark 2014 multi-level extension); "
            "(b) WOW directional filter banks (Holub-Fridrich 2012)."
        )
    elif sig_hill_under_ccc and sig_hill_dynamic_range:
        verdict = "POSITIVE_SIGNAL_PARTIAL_BELOW_DDD"
        recommendation = (
            f"PARTIAL_BELOW_DDD: HILL cascade ({hill_textured_avg_weight:.4f}) improves on "
            f"CCC baseline ({ccc_textured_avg_weight:.4f}) but is NOT sharper than DDD "
            f"wavelet-subband ({ddd_textured_avg_weight:.4f}). Two implementations both "
            "produce sub-canonical signal; queue multi-scale J-UNIWARD and WOW per Catalog "
            "#308 alternative reducers."
        )
    else:
        verdict = "NULL_SIGNAL_DEFER"
        recommendation = (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification: HILL filter "
            "cascade did NOT improve on DDD wavelet-subband sharper baseline OR CCC "
            "local-variance baseline. Per Catalog #308 reactivation criteria: queue "
            "multi-scale J-UNIWARD AND WOW alternative reducers; paradigm INTACT "
            "(Fridrich inverse-steganalysis textured-region undetectability per "
            "CLAUDE.md); IMPLEMENTATION-level approximation axis needs alternative."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_hill_filter_steganalysis_sister_smoke",
        "lane_id": "lane_overnight_eee_hill_filter_steganalysis_sister_tier_1_macos_cpu_advisory_smoke_20260521",
        "probe_name": "HILL filter steganalysis sister",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "hill_textured_suppression": "textured_avg_weight < 0.5 (DDD boundary=0.626; CCC baseline=0.806)",
            "hill_under_ddd": "hill_textured_avg_weight < ddd_textured_avg_weight (0.626)",
            "hill_under_ccc": "hill_textured_avg_weight < ccc_textured_avg_weight (0.806)",
            "hill_dynamic_range": "log10 > 1.0",
            "hill_flat_textured_separation": "hill_textured > 10% AND hill_flat > 30%",
        },
        "actual_signature": {
            # HILL filter cascade (primary)
            "hill_textured_avg_weight": hill_textured_avg_weight,
            "hill_textured_avg_weight_via_var_mask": hill_textured_avg_weight_var_mask,
            "hill_textured_avg_weight_via_wav_mask": hill_textured_avg_weight_wav_mask,
            "hill_weighted_error": hill_weighted_error,
            "hill_dynamic_range_log10": hill_dynamic_range,
            "hill_flat_fraction": hill_flat_fraction,
            "hill_textured_fraction": hill_textured_fraction,
            # DDD baseline (for delta comparison)
            "ddd_textured_avg_weight": ddd_textured_avg_weight,
            "ddd_weighted_error": ddd_weighted_error,
            "detail_dynamic_range_log10": detail_dynamic_range,
            "detail_flat_fraction": detail_flat_fraction,
            "detail_textured_fraction": detail_textured_fraction,
            # CCC baseline (for delta comparison)
            "ccc_textured_avg_weight": ccc_textured_avg_weight,
            "ccc_weighted_error": ccc_weighted_error,
            "var_dynamic_range_log10": var_dynamic_range,
            "var_flat_fraction": var_flat_fraction,
            "var_textured_fraction": var_textured_fraction,
            # Universal
            "uniform_weighted_error": uniform_weighted_error,
            "uniform_vs_hill_ratio": uniform_weighted_error / max(hill_weighted_error, 1e-10),
            "n_frames": n_frames,
            "frame_resolution_HxW": [H, W],
            "kb_kernel_normalization": "divided_by_4_unit_gain",
            "l1_window": 3,
            "l2_window": 15,
            "sigma_fridrich": sigma_fridrich,
        },
        "delta_vs_ddd_baseline": {
            "ddd_textured_avg_weight": ddd_textured_avg_weight,
            "hill_textured_avg_weight": hill_textured_avg_weight,
            "absolute_delta": hill_textured_avg_weight - ddd_textured_avg_weight,
            "ratio_hill_over_ddd": hill_textured_avg_weight / max(ddd_textured_avg_weight, 1e-10),
            "ddd_passed_threshold_lt_0p5": ddd_textured_avg_weight < 0.5,
            "hill_passed_threshold_lt_0p5": hill_textured_avg_weight < 0.5,
            "improvement_direction_vs_ddd": (
                "hill_sharper_lower_is_better"
                if hill_textured_avg_weight < ddd_textured_avg_weight
                else "hill_did_not_improve_or_worse"
            ),
        },
        "delta_vs_ccc_baseline": {
            "ccc_baseline_textured_avg_weight": ccc_textured_avg_weight,
            "hill_textured_avg_weight": hill_textured_avg_weight,
            "absolute_delta": hill_textured_avg_weight - ccc_textured_avg_weight,
            "ratio_hill_over_ccc": hill_textured_avg_weight / max(ccc_textured_avg_weight, 1e-10),
            "improvement_direction_vs_ccc": (
                "hill_sharper_lower_is_better"
                if hill_textured_avg_weight < ccc_textured_avg_weight
                else "hill_did_not_improve_or_worse"
            ),
        },
        "signature_checks": {
            "hill_textured_suppression_present": sig_hill_textured_suppression,
            "hill_under_ddd_present": sig_hill_under_ddd,
            "hill_under_ccc_present": sig_hill_under_ccc,
            "hill_dynamic_range_present": sig_hill_dynamic_range,
            "hill_flat_textured_separation_present": sig_hill_flat_textured,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate uniward_textured_region_undetectability_pose_distortion_savings_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N pending per AAA T4 op-routable; "
            "HILL extends UNIWARD via Li-Fridrich-Wang 2014 sharper-tail formulation)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_3c_hill_filter_steganalysis_sister",
            "predecessor_probes": [
                "tier_1_distortion_uniward_per_pixel_segnet_smoke",
                "tier_1_distortion_uniward_wavelet_subband_sharper_inversion_smoke",
            ],
            "hill_canonical_reference": (
                "Li-Tang-Huang-Luo 2014 'A new cost function for spatial image "
                "steganography' (HILL filter); KB high-pass kernel per Ker-Bohme 2008 "
                "'Revisiting weighted stego-image steganalysis'; CLAUDE.md 'Fridrich "
                "inverse steganalysis' section."
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "uniward_textured_region_undetectability_pose_distortion_savings_v1",
        "next_action_on_POSITIVE_FULL": (
            "Operator-routable: Tier-2 paid dispatch on UNIWARD/HILL-weighted per-pixel "
            "SegNet loss substrate via Vast.ai 4090 ($0.25/hr) or Lightning T4; "
            "estimated cost ~$1-5; predicted ΔS -0.005 to -0.015 [predicted] per "
            "AAA T4 §2.3 + §9 + DDD §6.1; recipe pending per AAA T4 §6.2 op-routable."
        ),
        "next_action_on_PARTIAL": (
            "Continue sister-probe iteration at $0 per DDD §6.2 + Catalog #308: try "
            "multi-scale J-UNIWARD (Holub-Fridrich-Denemark 2014 multi-level) OR WOW "
            "(Holub-Fridrich 2012 directional filter banks)."
        ),
        "next_action_on_NULL": (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification; queue "
            "multi-scale J-UNIWARD and WOW per DDD's queued alternatives + Catalog "
            "#308 reactivation criteria; paradigm INTACT."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_3c_hill_filter_steganalysis_sister_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[probe_3c] verdict={verdict['verdict']} "
        f"hill_textured_avg_weight={verdict['actual_signature']['hill_textured_avg_weight']:.4f} "
        f"(DDD baseline={verdict['delta_vs_ddd_baseline']['ddd_textured_avg_weight']:.4f}; "
        f"CCC baseline={verdict['delta_vs_ccc_baseline']['ccc_baseline_textured_avg_weight']:.4f}) "
        f"hill_log10_range={verdict['actual_signature']['hill_dynamic_range_log10']:.2f} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_3c] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
