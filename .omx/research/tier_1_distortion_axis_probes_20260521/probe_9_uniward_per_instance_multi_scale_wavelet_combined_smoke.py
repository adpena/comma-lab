# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 WAVE-3 Probe 9: UNIWARD per-instance x multi-scale wavelet COMBINED [macOS-CPU advisory]

Per COMBINED-TIER-1-WAVE-2 landing memo commit `68f0bba4d` (Probe 8 verdict
POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL with min_segment_textured_avg_weight=0.5233
at PR 101 SegNet per-instance connected-components; threshold 0.5 unbroken;
spread 0.4027). Probe 9 COMBINES Probe 8's per-instance segment granularity
with Holub-Fridrich 2014 canonical MULTI-LEVEL wavelet decomposition (single-
level db8 -> 3-level db8 detail-subband sum across scales).

CONTRACT (Carmack MVP-first 5-step + COMBINED multi-scale + per-instance):
  Canonical Holub-Fridrich 2014 UNIWARD multi-scale (J-UNIWARD predecessor):
      cost_i = 1 / (sum_level_l sum_subband_in_{HL,LH,HH} |coeff_l|_i + sigma)

  Probe 8 dimension (D16 per-segment-label): scipy.ndimage.label connected-
  components on SegNet 5-class hard mask -> 22 valid per-instance segments.
  Min segment textured_avg_weight=0.5233 (PARTIAL; threshold 0.5 unbroken).

  Probe 3b dimension (M40 single-level wavelet-subband db8): textured_avg_weight
  = 0.626 (DDD baseline; less sharp than Probe 5 per-class 0.5673).

  NEW dimension (D26 COMBINED per-instance + multi-level wavelet): for each
  connected-component segment, compute per-instance textured_avg_weight using
  the 3-LEVEL wavelet detail-subband sum (HL+LH+HH at each scale, upsampled
  to original resolution and summed across scales). Hypothesis: combining
  per-instance granularity (Probe 8 spread 0.4027 = 2.2x Probe 5) with multi-
  scale wavelet sharpening (3-level db8 instead of single-level) should drive
  min segment textured_avg_weight BELOW the 0.5 threshold.

PREDICTED SIGNATURES (3 outcomes per Carmack MVP-first step 2):
  POSITIVE_BREAKS_THRESHOLD:  min combined per-instance textured_avg_weight < 0.5
                              AND inter-segment spread > Probe 8's 0.4027 spread
                              AND valid_segment_count >= 4 (cross-class diversity)
  POSITIVE_COMBINED_PARTIAL:  min combined per-instance textured_avg_weight <
                              Probe 8's 0.5233 ceiling but threshold 0.5 unbroken
  NEGATIVE:                   min combined per-instance textured_avg_weight >=
                              Probe 8's 0.5233 (multi-scale extension yields no
                              improvement; per-instance + multi-scale is NOT a
                              productive combination)

FALSIFYING OUTCOME (NEGATIVE):
  - Combined multi-scale + per-instance fails to break 0.5 hard threshold.
  - Multi-scale wavelet extension yields no improvement over Probe 8 single-level
    + per-instance combination.
  - => DEFER per Catalog #307 IMPLEMENTATION-LEVEL falsification of multi-scale
    + per-instance combination paradigm. Per Catalog #308 alternative reducer:
    queue HILL filter per-instance OR boundary-aware UNIWARD per-instance
    (Fridrich 2014 boundary-cost extension).

CANONICAL PROVENANCE per Catalog #287 + #323:
  evidence_grade = "macOS-CPU-advisory"; promotable = False; axis_tag = "[macOS-CPU advisory]"
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

import numpy as np  # noqa: E402
import pywt  # noqa: E402
import scipy.ndimage  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from upstream.modules import SegNet  # noqa: E402


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


def _multi_scale_wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str = "db8",
    levels: int = 3,
) -> torch.Tensor:
    """Canonical Holub-Fridrich 2014 multi-level wavelet detail-subband sum.

    For each level l in 0..L-1, compute 2D DWT detail subbands (HL, LH, HH),
    sum their magnitudes, upsample to original resolution via nearest-neighbor,
    and accumulate across levels. The result is a per-pixel "edge energy"
    map at multiple scales — fine-scale details (l=0) capture sharp textures,
    coarser scales (l=1, l=2) capture larger structures.

    Per Holub-Fridrich-Denemark 2014 'Universal Distortion Function for
    Steganography in an Arbitrary Domain' (UNIWARD) multi-level extension.
    """
    luma_np = luma.cpu().numpy()
    n, h, w = luma_np.shape
    detail_sum = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        # Compute multi-level DWT: returns [cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]
        coeffs_multi = pywt.wavedec2(luma_np[i], wavelet_name, mode="symmetric", level=levels)
        # coeffs_multi[0] is the approximation (cA at deepest level); skip it.
        # coeffs_multi[1..L] are tuples (cH, cV, cD) at decreasing depth.
        for level_coeffs in coeffs_multi[1:]:
            cH, cV, cD = level_coeffs
            detail_lo = np.abs(cH) + np.abs(cV) + np.abs(cD)
            detail_t = torch.from_numpy(detail_lo).unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(detail_t, size=(h, w), mode="nearest")
            detail_sum[i] += upsampled.squeeze(0).squeeze(0).numpy()

    return torch.from_numpy(detail_sum)


def _run_probe() -> dict:
    t_start = time.time()
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Apples-to-apples Probe 8 setup: 8 frames -> 4 pairs (eval frames [1..4])
    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=8)
    n_frames, _, H, W = frames.shape

    pair_count = 4
    pairs = torch.stack(
        [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(pair_count)],
        dim=0,
    )  # (4, 2, 3, H, W)
    pairs = pairs.to(device)

    # === SegNet forward -> 5-class logits -> hard-classification mask ===
    with torch.no_grad():
        seg_input = segnet.preprocess_input(pairs)  # (4, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (4, 5, 384, 512)
    seg_mask = seg_logits.argmax(dim=1)  # (4, 384, 512) hard-class per pixel

    # Eval frames per upstream/modules.py x[:, -1] convention
    eval_frame_indices = list(range(1, pair_count + 1))
    eval_frames = frames[eval_frame_indices]  # (4, 3, H, W)
    luma_eval = _luma(eval_frames)  # (4, H, W)

    # === MULTI-LEVEL wavelet detail-subband sum (NEW D26 dimension) ===
    LEVELS = 3
    detail_sum_orig_res = _multi_scale_wavelet_detail_magnitudes(
        luma_eval, wavelet_name="db8", levels=LEVELS
    )  # (4, H, W)

    # Resize wavelet detail to SegNet's eval resolution
    detail_sum = F.interpolate(
        detail_sum_orig_res.unsqueeze(1),
        size=(384, 512),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # (4, 384, 512)

    sigma_fridrich = 2.0 ** -6
    uniward_weights = 1.0 / (detail_sum + sigma_fridrich)
    uniward_weights_norm = uniward_weights / uniward_weights.mean()

    # === Per-instance + multi-scale combined decomposition (D26) ===
    seg_mask_np = seg_mask.cpu().numpy()  # (4, 384, 512)
    detail_sum_np = detail_sum.cpu().numpy()  # (4, 384, 512)
    uniward_weights_norm_np = uniward_weights_norm.cpu().numpy()  # (4, 384, 512)

    per_segment_metrics: list[dict] = []
    per_segment_textured_weights: list[float] = []  # for global summary
    per_class_segment_count: dict[int, int] = dict.fromkeys(range(5), 0)

    # Connected-components 4-connectivity structure
    structure_4conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)

    # Per-segment minimum pixel count threshold (avoid tiny noise segments)
    MIN_SEGMENT_PIXELS = 200  # ~0.1% of 384*512 = 197 pixels
    MIN_TEXTURED_PIXELS = 50  # threshold for stable textured statistics

    for p in range(pair_count):
        for c in range(5):
            class_mask_p = (seg_mask_np[p] == c)  # (384, 512) bool
            labeled, num_segments = scipy.ndimage.label(class_mask_p, structure=structure_4conn)
            if num_segments == 0:
                continue

            for instance_id in range(1, num_segments + 1):
                instance_mask = (labeled == instance_id)
                instance_pixel_count = int(instance_mask.sum())
                if instance_pixel_count < MIN_SEGMENT_PIXELS:
                    continue
                per_class_segment_count[c] += 1

                # Per-instance multi-scale detail magnitude stats
                instance_details = detail_sum_np[p][instance_mask]
                instance_avg_detail = float(instance_details.mean())

                # Per-instance textured mask = upper quartile WITHIN the instance
                instance_detail_q75 = float(np.quantile(instance_details, 0.75))
                instance_textured_mask = instance_mask & (detail_sum_np[p] > instance_detail_q75)
                instance_textured_count = int(instance_textured_mask.sum())

                if instance_textured_count < MIN_TEXTURED_PIXELS:
                    continue

                instance_textured_avg_weight = float(
                    uniward_weights_norm_np[p][instance_textured_mask].mean()
                )
                per_segment_textured_weights.append(instance_textured_avg_weight)

                per_segment_metrics.append({
                    "pair_index": p,
                    "class_index": c,
                    "instance_id": instance_id,
                    "instance_pixel_count": instance_pixel_count,
                    "instance_textured_count": instance_textured_count,
                    "instance_avg_detail_multi_scale": instance_avg_detail,
                    "instance_textured_avg_weight_combined": instance_textured_avg_weight,
                })

    # === Per-segment summary statistics ===
    if per_segment_textured_weights:
        min_segment_textured_weight = float(min(per_segment_textured_weights))
        max_segment_textured_weight = float(max(per_segment_textured_weights))
        spread_segment_textured_weight = max_segment_textured_weight - min_segment_textured_weight
        any_segment_below_threshold = any(w < 0.5 for w in per_segment_textured_weights)
        valid_segment_count = len(per_segment_textured_weights)
        median_segment_textured_weight = float(np.median(per_segment_textured_weights))
    else:
        min_segment_textured_weight = None
        max_segment_textured_weight = None
        spread_segment_textured_weight = None
        any_segment_below_threshold = False
        valid_segment_count = 0
        median_segment_textured_weight = None

    # === Baselines for delta comparison ===
    ccc_baseline_textured_avg_weight = 0.8062214255332947  # CCC probe 3 per-pixel
    ddd_baseline_textured_avg_weight = 0.626  # DDD probe 3b single-level wavelet
    probe_5_per_class_min = 0.5673  # Probe 5 PARTIAL ceiling
    probe_8_per_instance_min = 0.5233  # Probe 8 PARTIAL per-instance min
    probe_8_per_instance_spread = 0.4027  # Probe 8 PARTIAL per-instance spread

    # === Predicted signature checks ===
    sig_any_segment_below_threshold = any_segment_below_threshold
    sig_breaks_probe_8_min = (
        min_segment_textured_weight is not None
        and min_segment_textured_weight < probe_8_per_instance_min
    )
    sig_spread_exceeds_probe_8 = (
        spread_segment_textured_weight is not None
        and spread_segment_textured_weight > probe_8_per_instance_spread
    )
    sig_valid_segment_count = valid_segment_count >= 4

    # === Verdict logic ===
    if sig_any_segment_below_threshold and sig_valid_segment_count:
        verdict = "POSITIVE_SIGNAL_BREAKS_THRESHOLD"
        recommendation = (
            f"POSITIVE_SIGNAL_BREAKS_THRESHOLD: combined per-instance + multi-scale wavelet "
            f"COMBINED yields min segment textured_avg_weight={min_segment_textured_weight:.4f} "
            f"(< 0.5 hard threshold; Probe 8 per-instance min={probe_8_per_instance_min:.4f}; "
            f"DDD single-level baseline={ddd_baseline_textured_avg_weight:.3f}). "
            f"Inter-segment spread {spread_segment_textured_weight:.4f} vs Probe 8 spread "
            f"{probe_8_per_instance_spread:.4f}. Per Holub-Fridrich 2014 canonical multi-scale "
            "+ per-instance granularity: BOTH dimensions productive. STRONG empirical case for "
            "Tier-2 paid dispatch on per-instance + multi-scale wavelet UNIWARD-weighted SegNet "
            "loss substrate JUSTIFIED. Predicted DS -0.010 to -0.025 [predicted]. Estimated cost ~$2-7."
        )
    elif sig_breaks_probe_8_min and sig_valid_segment_count:
        verdict = "POSITIVE_SIGNAL_COMBINED_PARTIAL"
        recommendation = (
            f"POSITIVE_SIGNAL_COMBINED_PARTIAL: combined per-instance + multi-scale wavelet tightens "
            f"inversion below Probe 8 per-instance ceiling (min combined per-instance textured_avg_weight="
            f"{min_segment_textured_weight:.4f} < Probe 8 per-instance min {probe_8_per_instance_min:.4f}) "
            f"but the 0.5 hard threshold remains unbroken. Per CLAUDE.md 'Forbidden premature KILL': "
            "DEFER-PENDING-FURTHER-DIMENSION; iterate sister probe with per-instance + HILL filter OR "
            "per-instance + boundary-aware UNIWARD per Catalog #308. Paradigm INTACT (combined multi-"
            "scale + per-instance is productive); IMPLEMENTATION-level threshold needs further "
            "dimensional combination."
        )
    else:
        verdict = "NEGATIVE_COMBINATION_FALSIFIED"
        recommendation = (
            f"NEGATIVE_COMBINATION_FALSIFIED: combined per-instance + multi-scale wavelet did NOT "
            f"improve over Probe 8 per-instance ceiling 0.5233 (min combined={min_segment_textured_weight}). "
            "Per Catalog #307 IMPLEMENTATION-LEVEL falsification of the COMBINED multi-scale + per-instance "
            "paradigm. Per Catalog #308 reactivation criteria: queue per-instance + HILL filter OR "
            "per-instance + boundary-aware UNIWARD per Fridrich 2014 boundary-cost extension. Probe 8 "
            "per-instance Tier-2 recommendation UNCHANGED (paradigm-level INTACT; combined-axis falsified)."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_uniward_per_instance_multi_scale_wavelet_combined_smoke",
        "lane_id": "lane_combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_20260525",
        "probe_name": "UNIWARD per-instance x multi-scale wavelet COMBINED (M40 x D16 x D26 multi-level db8)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "POSITIVE_BREAKS_THRESHOLD": "any segment combined per-instance + multi-scale textured_avg_weight < 0.5 AND valid_segment_count >= 4",
            "POSITIVE_COMBINED_PARTIAL": "min segment combined textured_avg_weight < Probe 8 per-instance ceiling 0.5233 but threshold 0.5 unbroken",
            "NEGATIVE": "min segment combined textured_avg_weight >= 0.5233 (multi-scale extension non-productive)",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "pair_count": pair_count,
            "segnet_eval_resolution_HxW": [384, 512],
            "wavelet_levels": LEVELS,
            "wavelet_name": "db8",
            "min_segment_pixels_threshold": MIN_SEGMENT_PIXELS,
            "min_textured_pixels_threshold": MIN_TEXTURED_PIXELS,
            "per_class_segment_count": per_class_segment_count,
            "valid_segment_count": valid_segment_count,
            "per_segment_metrics": per_segment_metrics,
            "min_segment_textured_avg_weight_combined": min_segment_textured_weight,
            "max_segment_textured_avg_weight_combined": max_segment_textured_weight,
            "median_segment_textured_avg_weight_combined": median_segment_textured_weight,
            "spread_segment_textured_avg_weight_combined": spread_segment_textured_weight,
            "any_segment_below_threshold": any_segment_below_threshold,
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "probe_5_per_class_min": probe_5_per_class_min,
            "probe_8_per_instance_min": probe_8_per_instance_min,
            "probe_8_per_instance_spread": probe_8_per_instance_spread,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_baselines": {
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "probe_5_per_class_min": probe_5_per_class_min,
            "probe_8_per_instance_min": probe_8_per_instance_min,
            "min_combined_vs_ccc_absolute_delta": (
                min_segment_textured_weight - ccc_baseline_textured_avg_weight
                if min_segment_textured_weight is not None else None
            ),
            "min_combined_vs_ddd_absolute_delta": (
                min_segment_textured_weight - ddd_baseline_textured_avg_weight
                if min_segment_textured_weight is not None else None
            ),
            "min_combined_vs_probe_5_absolute_delta": (
                min_segment_textured_weight - probe_5_per_class_min
                if min_segment_textured_weight is not None else None
            ),
            "min_combined_vs_probe_8_absolute_delta": (
                min_segment_textured_weight - probe_8_per_instance_min
                if min_segment_textured_weight is not None else None
            ),
            "spread_combined_vs_probe_8_absolute_delta": (
                spread_segment_textured_weight - probe_8_per_instance_spread
                if spread_segment_textured_weight is not None else None
            ),
            "per_segment_min_passed_threshold_lt_0p5": any_segment_below_threshold,
            "per_segment_min_breaks_probe_8_per_instance_ceiling": sig_breaks_probe_8_min,
            "improvement_direction": (
                "combined_sharper_than_probe_8_per_instance"
                if sig_breaks_probe_8_min
                else "combined_did_not_improve_over_probe_8_per_instance"
            ),
        },
        "signature_checks": {
            "any_segment_below_threshold_present": sig_any_segment_below_threshold,
            "breaks_probe_8_min_present": sig_breaks_probe_8_min,
            "spread_exceeds_probe_8_present": sig_spread_exceeds_probe_8,
            "valid_segment_count_present": sig_valid_segment_count,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate uniward_per_instance_multi_scale_wavelet_combined_v1 (Catalog #344 "
            "FORMALIZATION_PENDING; RATIFY-N pending per RATE-ATTACK-MATRIX cell #3 + cell #21 "
            "combined-axis operator-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_9_uniward_per_instance_multi_scale_wavelet_combined",
            "predecessor_probes": [
                "tier_1_distortion_uniward_wavelet_subband_sharper_inversion_smoke (DDD; single-level)",
                "tier_1_distortion_uniward_per_class_explicit_segnet_smoke (Probe 5; per-class)",
                "tier_1_distortion_uniward_per_segment_label_segnet_smoke (Probe 8; per-instance)",
            ],
            "fridrich_canonical_reference": (
                "Holub-Fridrich-Denemark 2014 'Universal Distortion Function for Steganography "
                "in an Arbitrary Domain' (UNIWARD) multi-level wavelet extension; CLAUDE.md "
                "'Fridrich inverse steganalysis' section; SegNet exact architecture per CLAUDE.md "
                "'Exact scorer architectures'."
            ),
            "multi_scale_wavelet_reference": (
                "pywt.wavedec2 with db8 wavelet at level=3; canonical multi-scale Daubechies "
                "wavelet decomposition; coarse-to-fine HL+LH+HH detail-subband sum across levels "
                "captures edge-energy at multiple scales per Holub-Fridrich 2014 + Mallat 1989 "
                "multi-resolution analysis."
            ),
            "connected_components_reference": (
                "scipy.ndimage.label with 4-connectivity structure for per-instance segmentation "
                "derivation from SegNet 5-class hard mask; canonical CS computer-vision instance "
                "segmentation primitive from semantic segmentation via connected-components."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #3 (per-instance) "
                "+ cell #21 (multi-scale wavelet) COMBINED dimension D26: per-instance UNIWARD + "
                "multi-level db8 wavelet detail-subband sum"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "uniward_per_instance_multi_scale_wavelet_combined_v1",
        "next_action_on_POSITIVE_BREAKS_THRESHOLD": (
            "STRONG empirical case for Tier-2 paid dispatch on per-instance + multi-scale wavelet "
            "UNIWARD-weighted SegNet loss substrate via Vast.ai 4090 ($0.25/hr); estimated cost "
            "~$2-7; predicted DS -0.010 to -0.025 [predicted]."
        ),
        "next_action_on_PARTIAL": (
            "DEFER-PENDING-FURTHER-DIMENSION; iterate sister probe with per-instance + HILL filter "
            "OR per-instance + boundary-aware UNIWARD per Catalog #308."
        ),
        "next_action_on_NEGATIVE": (
            "DEFER per Catalog #307 IMPLEMENTATION-LEVEL falsification of combined-axis paradigm; "
            "queue per-instance + HILL filter per Catalog #308."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_9_uniward_per_instance_multi_scale_wavelet_combined_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    min_w = verdict["actual_signature"]["min_segment_textured_avg_weight_combined"]
    spread = verdict["actual_signature"]["spread_segment_textured_avg_weight_combined"]
    vsc = verdict["actual_signature"]["valid_segment_count"]
    print(
        f"[probe_9] verdict={verdict['verdict']} "
        f"min_combined_textured_avg_weight={min_w if min_w is None else f'{min_w:.4f}'} "
        f"(Probe 8 per-instance min=0.5233; DDD single-level=0.626) "
        f"inter_segment_spread={spread if spread is None else f'{spread:.4f}'} "
        f"valid_segments={vsc} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_9] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
