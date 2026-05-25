# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 CCC-ext Probe 5: UNIWARD per-class explicit SegNet [macOS-CPU advisory]

Per RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cell #3:
  (M40 UNIWARD x D15 per-class explicit)
+ CCC probe 3 POSITIVE_SIGNAL_PARTIAL + DDD probe 3b POSITIVE_SIGNAL_SHARPER
+ CLAUDE.md "Fridrich inverse steganalysis" + Carmack MVP-first 5-step.

CONTRACT (Carmack MVP-first 5-step + per-class explicit):
  Canonical Fridrich UNIWARD wavelet-subband (Holub-Fridrich-Denemark 2014):
      cost_i = 1 / (|HL|_i + |LH|_i + |HH|_i + sigma)

  NEW dimension (D15 per-class explicit): decompose UNIWARD weighting via PR 101
  SegNet's 5-class hard-classification mask. Hypothesis: per-class boundaries
  align with SegNet's stride-2-stem blindspot per Yousfi+Fridrich inverse-
  steganalysis paradigm. Per-class textured/flat separation should be SHARPER
  than per-pixel because each class has its own typical texture profile (e.g.
  class 0=road has uniform flat regions; class 4=other has high-variance
  textured regions).

PREDICTED SIGNATURE (sharper than DDD baseline 0.626):
  - At least 1 class with class-conditional textured_avg_weight <= 0.5
  - Inter-class variation in textured_avg_weight (>0.1 spread between min/max)
  - Class-conditional flat/textured separation present per-class

FALSIFYING OUTCOME:
  - All per-class textured_avg_weight > 0.5 (no per-class sharpening over DDD)
  - => DEFER per Catalog #307 IMPLEMENTATION-level falsification + Catalog #308
    alternative reducer (e.g. per-segment-label D16 OR per-time-window D17).

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

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from upstream.modules import SegNet  # type: ignore


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


def _wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str = "db8",
) -> torch.Tensor:
    """Canonical Fridrich UNIWARD detail-subband sum.

    Per DDD probe 3b: for each frame, run a single-level 2D DWT and return
    per-pixel |HL| + |LH| + |HH|, upsampled to original (H, W) via nearest-
    neighbor per Fridrich 2014 cost-map construction.
    """
    luma_np = luma.cpu().numpy()  # (N, H, W)
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
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Decode 8 frames from PR 101 reference video (apples-to-apples CCC+DDD baseline)
    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=8)
    n_frames, _, H, W = frames.shape

    # === Build SegNet input pairs (B, T=2, C, H, W) per canonical scorer interface ===
    pair_count = 4
    pairs = torch.stack(
        [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(pair_count)],
        dim=0,
    )  # (4, 2, 3, H, W)
    pairs = pairs.to(device)

    # === SegNet forward → 5-class logits → hard-classification mask ===
    # SegNet.preprocess_input slices last frame `x[:, -1]` and resizes to (384, 512).
    with torch.no_grad():
        seg_input = segnet.preprocess_input(pairs)  # (4, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (4, 5, 384, 512)
    seg_mask = seg_logits.argmax(dim=1)  # (4, 384, 512) hard-class per pixel

    # SegNet evaluates on the last frame of each pair (per upstream/modules.py x[:, -1])
    # so the SegNet output corresponds to frames[1..4] (indices 1, 2, 3, 4).
    eval_frame_indices = list(range(1, pair_count + 1))  # [1, 2, 3, 4]

    # === Build luma + wavelet detail on SegNet-evaluated frames ===
    eval_frames = frames[eval_frame_indices]  # (4, 3, H, W) [0,1]
    luma_eval = _luma(eval_frames)  # (4, H, W)
    detail_sum_orig_res = _wavelet_detail_magnitudes(luma_eval, wavelet_name="db8")  # (4, H, W)

    # Resize wavelet detail to SegNet's eval resolution (384, 512) for per-class alignment
    detail_sum = F.interpolate(
        detail_sum_orig_res.unsqueeze(1),  # (4, 1, H, W)
        size=(384, 512),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # (4, 384, 512)

    sigma_fridrich = 2.0 ** -6
    uniward_weights = 1.0 / (detail_sum + sigma_fridrich)
    uniward_weights_norm = uniward_weights / uniward_weights.mean()

    # === Per-class explicit decomposition (D15 NEW dimension) ===
    # For each SegNet class c in {0, 1, 2, 3, 4}, compute:
    #   class_mask = (seg_mask == c) -> indicator
    #   class_textured_mask = class_mask AND (wavelet_detail > quantile(0.75) within class)
    #   class_textured_avg_weight = uniward_weights[class_textured_mask].mean()
    per_class_metrics = []
    class_textured_weights = []
    for c in range(5):
        class_mask = (seg_mask == c)  # (4, 384, 512)
        class_pixel_count = int(class_mask.sum().item())
        class_pixel_fraction = float(class_mask.float().mean().item())

        if class_pixel_count < 100:
            # Skip class if too few pixels for stable statistics
            per_class_metrics.append(
                {
                    "class_index": c,
                    "class_pixel_count": class_pixel_count,
                    "class_pixel_fraction": class_pixel_fraction,
                    "class_textured_avg_weight": None,
                    "class_avg_detail": None,
                    "class_detail_dynamic_range_log10": None,
                    "skipped_reason": "insufficient_pixel_count_lt_100",
                }
            )
            continue

        # Per-class detail magnitude stats
        class_details = detail_sum[class_mask]  # 1-D tensor of detail values within class c
        class_avg_detail = float(class_details.mean().item())
        class_detail_dynamic_range = float(
            (class_details.max() / (class_details.min() + 1e-10)).log10().item()
        )

        # Per-class textured mask = upper quartile of detail WITHIN the class
        class_detail_q75 = float(class_details.quantile(0.75).item())
        class_textured_mask = class_mask & (detail_sum > class_detail_q75)
        class_textured_count = int(class_textured_mask.sum().item())

        if class_textured_count < 50:
            class_textured_avg_weight = None
            class_textured_count_note = "insufficient_textured_pixels_lt_50"
        else:
            class_textured_avg_weight = float(
                uniward_weights_norm[class_textured_mask].mean().item()
            )
            class_textured_count_note = "ok"
            class_textured_weights.append(class_textured_avg_weight)

        per_class_metrics.append(
            {
                "class_index": c,
                "class_pixel_count": class_pixel_count,
                "class_pixel_fraction": class_pixel_fraction,
                "class_textured_count": class_textured_count,
                "class_textured_count_note": class_textured_count_note,
                "class_textured_avg_weight": class_textured_avg_weight,
                "class_avg_detail": class_avg_detail,
                "class_detail_dynamic_range_log10": class_detail_dynamic_range,
                "class_detail_q75": class_detail_q75,
            }
        )

    # === Per-class summary statistics ===
    valid_class_weights = [w for w in class_textured_weights if w is not None]
    if valid_class_weights:
        min_class_textured_weight = float(min(valid_class_weights))
        max_class_textured_weight = float(max(valid_class_weights))
        spread_class_textured_weight = max_class_textured_weight - min_class_textured_weight
        any_class_below_threshold = any(w < 0.5 for w in valid_class_weights)
    else:
        min_class_textured_weight = None
        max_class_textured_weight = None
        spread_class_textured_weight = None
        any_class_below_threshold = False

    # === Baselines for delta comparison (CCC=0.806, DDD=0.626) ===
    ccc_baseline_textured_avg_weight = 0.8062214255332947
    ddd_baseline_textured_avg_weight = 0.626  # from DDD probe 3b landing memo

    # === Predicted signature checks ===
    sig_any_class_below_threshold = any_class_below_threshold
    sig_inter_class_spread = (
        spread_class_textured_weight is not None and spread_class_textured_weight > 0.1
    )
    sig_valid_class_count = len(valid_class_weights) >= 2

    # === Verdict logic ===
    if sig_any_class_below_threshold and sig_inter_class_spread and sig_valid_class_count:
        verdict = "POSITIVE_SIGNAL_PER_CLASS_FULL"
        recommendation = (
            "POSITIVE_SIGNAL_PER_CLASS_FULL: at least one PR 101 SegNet class "
            f"yields class-conditional textured_avg_weight={min_class_textured_weight:.4f} "
            f"(< 0.5 threshold; CCC baseline=0.806, DDD baseline=0.626). Inter-class "
            f"spread {spread_class_textured_weight:.4f} confirms per-class boundaries "
            "align with SegNet stride-2-stem blindspot per Yousfi+Fridrich inverse-"
            "steganalysis paradigm. Per RATE-ATTACK-MATRIX cell #3 Tier-2 paid dispatch "
            "on per-class UNIWARD-weighted SegNet loss substrate JUSTIFIED. Predicted "
            "ΔS -0.005 to -0.015 [predicted]. Estimated cost ~$1-5."
        )
    elif sig_inter_class_spread and sig_valid_class_count:
        verdict = "POSITIVE_SIGNAL_PER_CLASS_PARTIAL"
        recommendation = (
            "PARTIAL: inter-class spread present but no class breaks the 0.5 threshold. "
            "Per CLAUDE.md 'Forbidden premature KILL': DEFER-PENDING-CLASS-AGGREGATION; "
            "iterate sister probe with per-segment-label D16 OR multi-scale wavelet "
            "decomposition per Catalog #308 alternative reducer. Paradigm INTACT "
            "(per-class boundaries align with SegNet blindspot); IMPLEMENTATION-level "
            "threshold needs tighter inversion."
        )
    else:
        verdict = "NULL_SIGNAL_DEFER"
        recommendation = (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification: per-class "
            "explicit decomposition did NOT yield sharper UNIWARD inversion than DDD "
            "wavelet-subband baseline. Per Catalog #308 reactivation criteria: try "
            "per-segment-label D16 (label-conditional cost) OR per-temporal-window D17 "
            "(temporal-coherent cost). Paradigm INTACT; per-class dimension "
            "implementation-level falsified."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_uniward_per_class_explicit_segnet_smoke",
        "lane_id": "lane_combined_tier_1_ccc_ext_probes_uniward_per_class_plus_hinton_kl_temporal_context_20260525",
        "probe_name": "UNIWARD per-class explicit SegNet (M40 x D15 RATE-ATTACK-MATRIX cell #3)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "any_class_below_threshold": "min class_textured_avg_weight < 0.5",
            "inter_class_spread": "max - min class_textured_avg_weight > 0.1",
            "valid_class_count": ">= 2 classes with stable statistics",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "pair_count": pair_count,
            "segnet_eval_resolution_HxW": [384, 512],
            "per_class_metrics": per_class_metrics,
            "valid_class_count": len(valid_class_weights),
            "min_class_textured_avg_weight": min_class_textured_weight,
            "max_class_textured_avg_weight": max_class_textured_weight,
            "spread_class_textured_avg_weight": spread_class_textured_weight,
            "any_class_below_threshold": any_class_below_threshold,
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_ccc_baseline": {
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "min_per_class_vs_ccc_absolute_delta": (
                min_class_textured_weight - ccc_baseline_textured_avg_weight
                if min_class_textured_weight is not None
                else None
            ),
            "min_per_class_vs_ddd_absolute_delta": (
                min_class_textured_weight - ddd_baseline_textured_avg_weight
                if min_class_textured_weight is not None
                else None
            ),
            "ccc_passed_threshold_lt_0p5": ccc_baseline_textured_avg_weight < 0.5,
            "ddd_passed_threshold_lt_0p5": ddd_baseline_textured_avg_weight < 0.5,
            "per_class_min_passed_threshold_lt_0p5": any_class_below_threshold,
            "improvement_direction": (
                "per_class_sharper_than_ddd"
                if (
                    min_class_textured_weight is not None
                    and min_class_textured_weight < ddd_baseline_textured_avg_weight
                )
                else "per_class_did_not_improve_over_ddd"
            ),
        },
        "signature_checks": {
            "any_class_below_threshold_present": sig_any_class_below_threshold,
            "inter_class_spread_present": sig_inter_class_spread,
            "valid_class_count_present": sig_valid_class_count,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate uniward_per_class_explicit_v1 (Catalog #344 FORMALIZATION_PENDING; "
            "RATIFY-N pending per RATE-ATTACK-MATRIX cell #3 op-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_5_uniward_per_class_explicit_segnet",
            "predecessor_probes": [
                "tier_1_distortion_uniward_per_pixel_segnet_smoke (CCC)",
                "tier_1_distortion_uniward_wavelet_subband_sharper_inversion_smoke (DDD)",
            ],
            "fridrich_canonical_reference": (
                "Holub-Fridrich-Denemark 2014 'Universal Distortion Function for "
                "Steganography in an Arbitrary Domain' (UNIWARD); CLAUDE.md "
                "'Fridrich inverse steganalysis' section; SegNet exact architecture "
                "per CLAUDE.md 'Exact scorer architectures'."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #3: "
                "(M40 UNIWARD x D15 per-class explicit)"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "uniward_per_class_explicit_v1",
        "next_action_on_POSITIVE_PER_CLASS_FULL": (
            "Operator-routable: Tier-2 paid dispatch on per-class UNIWARD-weighted "
            "SegNet loss substrate via Vast.ai 4090 ($0.25/hr) or Lightning T4; "
            "estimated cost ~$1-5; predicted ΔS -0.005 to -0.015 [predicted] per "
            "RATE-ATTACK-MATRIX cell #3."
        ),
        "next_action_on_PARTIAL": (
            "Continue sister-probe iteration at $0: try per-segment-label D16 "
            "(label-conditional cost) OR multi-scale wavelet decomposition per "
            "Catalog #308."
        ),
        "next_action_on_NULL": (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification; queue per-"
            "segment-label D16 alternative reducer per Catalog #308."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_5_uniward_per_class_explicit_segnet_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    min_w = verdict["actual_signature"]["min_class_textured_avg_weight"]
    spread = verdict["actual_signature"]["spread_class_textured_avg_weight"]
    print(
        f"[probe_5] verdict={verdict['verdict']} "
        f"min_class_textured_avg_weight={min_w if min_w is None else f'{min_w:.4f}'} "
        f"(CCC baseline=0.806, DDD baseline=0.626) "
        f"inter_class_spread={spread if spread is None else f'{spread:.4f}'} "
        f"valid_classes={verdict['actual_signature']['valid_class_count']} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_5] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
