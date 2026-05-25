# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 WAVE-2 Probe 8: UNIWARD x per-segment-label SegNet [macOS-CPU advisory]

Per COMBINED-TIER-1-CCC-EXT-PROBES landing memo commit `685fe6726` (probe 5
verdict POSITIVE_SIGNAL_PER_CLASS_PARTIAL with min_class_textured_avg_weight=0.5673
at PR 101 SegNet 5-class explicit decomposition; threshold 0.5 unbroken; spread=0.18).
Probe 8 explores Probe 5's sister-cascade per Catalog #308 alternative reducer:
replace per-CLASS hard-classification (5 classes) with per-INSTANCE connected-
components segmentation (per-segment-label D16 dimension).

CONTRACT (Carmack MVP-first 5-step + per-segment-label dimension):
  Canonical Fridrich UNIWARD wavelet-subband (Holub-Fridrich-Denemark 2014):
      cost_i = 1 / (|HL|_i + |LH|_i + |HH|_i + sigma)

  Probe 5 dimension (D15 per-class explicit): decompose UNIWARD weighting via
  PR 101 SegNet's 5-class hard mask. Min class textured_avg_weight=0.5673
  (PARTIAL; threshold 0.5 unbroken).

  NEW dimension (D16 per-segment-label): for each SegNet class c, run connected-
  components labeling on (seg_mask == c) to derive per-instance segments. Each
  connected segment gets its OWN UNIWARD weighting computed over that segment's
  pixels only. Hypothesis: per-instance boundary granularity is finer than per-
  class hard mask aggregation — different segments of the same class (e.g. two
  separate vehicles both class-4) have distinct local texture profiles, so per-
  instance textured_avg_weight should exhibit GREATER inter-segment spread than
  per-class inter-class spread (Probe 5: 0.1808), and at least one segment
  should break the 0.5 threshold.

PREDICTED SIGNATURES (3 outcomes per Carmack MVP-first step 2):
  POSITIVE_BREAKS_THRESHOLD:  ANY connected segment yields textured_avg_weight < 0.5
                              AND inter-segment spread > Probe 5's 0.18 spread
                              AND valid_segment_count >= 4 (cross-class diversity)
  POSITIVE_PER_SEGMENT_PARTIAL: min segment textured_avg_weight < Probe 5's 0.5673
                                ceiling AND spread > 0.18 but threshold 0.5 unbroken
  NEGATIVE:                   min segment textured_avg_weight >= 0.6 (no improvement
                              over Probe 5 PARTIAL ceiling)

FALSIFYING OUTCOME (NEGATIVE):
  - Per-segment-label fails to break 0.5 like per-class (Probe 5).
  - Per-segment textured_avg_weight ceiling >= 0.6 (no improvement over Probe 5).
  - => DEFER per Catalog #307 IMPLEMENTATION-level falsification of per-segment-
    label dimension. Per Catalog #308 alternative reducer: queue per-instance +
    UNIWARD multi-scale wavelet COMBINED (Probe 5 sister + multi-scale wavelet
    per Holub-Fridrich 2014).

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


def _wavelet_detail_magnitudes(luma: torch.Tensor, wavelet_name: str = "db8") -> torch.Tensor:
    """Canonical Fridrich UNIWARD detail-subband sum (per DDD probe 3b + Probe 5)."""
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
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Apples-to-apples Probe 5 setup: 8 frames -> 4 pairs (eval frames [1..4])
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
    detail_sum_orig_res = _wavelet_detail_magnitudes(luma_eval, wavelet_name="db8")  # (4, H, W)

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

    # === Per-segment-label decomposition (D16 NEW dimension) ===
    # For each pair p and each class c, run scipy.ndimage.label on (seg_mask[p] == c)
    # to derive connected-component instance segments. Each instance gets its own
    # UNIWARD textured_avg_weight computed over its pixels and the upper-quartile-
    # within-instance textured subset.
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

                # Per-instance detail magnitude stats
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
                    "instance_avg_detail": instance_avg_detail,
                    "instance_textured_avg_weight": instance_textured_avg_weight,
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
    ddd_baseline_textured_avg_weight = 0.626  # DDD probe 3b wavelet-subband
    probe_5_per_class_min = 0.5673  # Probe 5 PARTIAL ceiling
    probe_5_per_class_spread = 0.1808

    # === Predicted signature checks ===
    sig_any_segment_below_threshold = any_segment_below_threshold
    sig_breaks_probe_5_min = (
        min_segment_textured_weight is not None
        and min_segment_textured_weight < probe_5_per_class_min
    )
    sig_spread_exceeds_probe_5 = (
        spread_segment_textured_weight is not None
        and spread_segment_textured_weight > probe_5_per_class_spread
    )
    sig_valid_segment_count = valid_segment_count >= 4

    # === Verdict logic ===
    if sig_any_segment_below_threshold and sig_valid_segment_count and sig_spread_exceeds_probe_5:
        verdict = "POSITIVE_SIGNAL_BREAKS_THRESHOLD"
        recommendation = (
            f"POSITIVE_SIGNAL_BREAKS_THRESHOLD: at least one PR 101 SegNet per-instance "
            f"segment yields per-instance textured_avg_weight="
            f"{min_segment_textured_weight:.4f} (< 0.5 threshold; Probe 5 per-class min="
            f"{probe_5_per_class_min:.4f}; DDD wavelet baseline={ddd_baseline_textured_avg_weight:.3f}). "
            f"Inter-segment spread {spread_segment_textured_weight:.4f} > Probe 5 inter-class "
            f"spread {probe_5_per_class_spread:.4f} confirms per-instance granularity reveals "
            "tighter UNIWARD inversion than per-class hard mask per Yousfi+Fridrich inverse-"
            "steganalysis paradigm + per-instance ego-motion-coherent boundaries. Per RATE-"
            "ATTACK-MATRIX cell #3 Tier-2 paid dispatch on per-instance UNIWARD-weighted SegNet "
            "loss substrate JUSTIFIED. Predicted DS -0.005 to -0.015 [predicted]. Estimated cost ~$1-5."
        )
    elif sig_breaks_probe_5_min and sig_valid_segment_count:
        verdict = "POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL"
        recommendation = (
            f"POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL: per-instance segmentation tightens UNIWARD "
            f"inversion below Probe 5 per-class ceiling (min segment textured_avg_weight="
            f"{min_segment_textured_weight:.4f} < Probe 5 per-class min {probe_5_per_class_min:.4f}) "
            f"but the 0.5 threshold remains unbroken. Per CLAUDE.md 'Forbidden premature KILL': "
            "DEFER-PENDING-MULTI-SCALE; iterate sister probe with per-instance + UNIWARD multi-"
            "scale wavelet COMBINED (Probe 5 + multi-scale per Holub-Fridrich 2014) per "
            "Catalog #308 alternative reducer. Paradigm INTACT (per-instance granularity is "
            "real); IMPLEMENTATION-level threshold needs combined multi-scale + per-instance."
        )
    else:
        verdict = "NULL_SIGNAL_DEFER"
        recommendation = (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification: per-instance segmentation "
            f"did NOT improve over Probe 5 per-class ceiling (min={min_segment_textured_weight}). "
            "Per Catalog #308 reactivation criteria: try per-instance + UNIWARD multi-scale "
            "wavelet COMBINED OR boundary-aware UNIWARD (Fridrich 2014 boundary-cost extension). "
            "Probe 5 per-class Tier-2 recommendation UNCHANGED."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_uniward_per_segment_label_segnet_smoke",
        "lane_id": "lane_combined_tier_1_wave_2_hinton_kl_longer_temporal_plus_uniward_per_segment_label_20260525",
        "probe_name": "UNIWARD x per-segment-label SegNet (M40 x D16 connected-components per-instance)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "POSITIVE_BREAKS_THRESHOLD": "any segment textured_avg_weight < 0.5 AND spread > 0.18 AND valid_segment_count >= 4",
            "POSITIVE_PER_SEGMENT_PARTIAL": "min segment textured_avg_weight < Probe 5 ceiling 0.5673 AND spread > 0.18",
            "NEGATIVE": "min segment textured_avg_weight >= 0.6 (no improvement over Probe 5)",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "pair_count": pair_count,
            "segnet_eval_resolution_HxW": [384, 512],
            "min_segment_pixels_threshold": MIN_SEGMENT_PIXELS,
            "min_textured_pixels_threshold": MIN_TEXTURED_PIXELS,
            "per_class_segment_count": per_class_segment_count,
            "valid_segment_count": valid_segment_count,
            "per_segment_metrics": per_segment_metrics,
            "min_segment_textured_avg_weight": min_segment_textured_weight,
            "max_segment_textured_avg_weight": max_segment_textured_weight,
            "median_segment_textured_avg_weight": median_segment_textured_weight,
            "spread_segment_textured_avg_weight": spread_segment_textured_weight,
            "any_segment_below_threshold": any_segment_below_threshold,
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "probe_5_per_class_min": probe_5_per_class_min,
            "probe_5_per_class_spread": probe_5_per_class_spread,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_baselines": {
            "ccc_baseline_textured_avg_weight": ccc_baseline_textured_avg_weight,
            "ddd_baseline_textured_avg_weight": ddd_baseline_textured_avg_weight,
            "probe_5_per_class_min": probe_5_per_class_min,
            "min_per_segment_vs_ccc_absolute_delta": (
                min_segment_textured_weight - ccc_baseline_textured_avg_weight
                if min_segment_textured_weight is not None else None
            ),
            "min_per_segment_vs_ddd_absolute_delta": (
                min_segment_textured_weight - ddd_baseline_textured_avg_weight
                if min_segment_textured_weight is not None else None
            ),
            "min_per_segment_vs_probe_5_per_class_absolute_delta": (
                min_segment_textured_weight - probe_5_per_class_min
                if min_segment_textured_weight is not None else None
            ),
            "spread_per_segment_vs_probe_5_per_class_absolute_delta": (
                spread_segment_textured_weight - probe_5_per_class_spread
                if spread_segment_textured_weight is not None else None
            ),
            "per_segment_min_passed_threshold_lt_0p5": any_segment_below_threshold,
            "per_segment_min_breaks_probe_5_per_class_ceiling": sig_breaks_probe_5_min,
            "improvement_direction": (
                "per_segment_sharper_than_probe_5_per_class"
                if (sig_breaks_probe_5_min)
                else "per_segment_did_not_improve_over_probe_5_per_class"
            ),
        },
        "signature_checks": {
            "any_segment_below_threshold_present": sig_any_segment_below_threshold,
            "breaks_probe_5_min_present": sig_breaks_probe_5_min,
            "spread_exceeds_probe_5_present": sig_spread_exceeds_probe_5,
            "valid_segment_count_present": sig_valid_segment_count,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate uniward_per_segment_label_v1 (Catalog #344 FORMALIZATION_PENDING; "
            "RATIFY-N pending per RATE-ATTACK-MATRIX cell #3 alternative reducer op-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_8_uniward_per_segment_label_segnet",
            "predecessor_probes": [
                "tier_1_distortion_uniward_per_pixel_segnet_smoke (CCC)",
                "tier_1_distortion_uniward_wavelet_subband_sharper_inversion_smoke (DDD)",
                "tier_1_distortion_uniward_per_class_explicit_segnet_smoke (Probe 5)",
            ],
            "fridrich_canonical_reference": (
                "Holub-Fridrich-Denemark 2014 'Universal Distortion Function for "
                "Steganography in an Arbitrary Domain' (UNIWARD); CLAUDE.md "
                "'Fridrich inverse steganalysis' section; SegNet exact architecture "
                "per CLAUDE.md 'Exact scorer architectures'."
            ),
            "connected_components_reference": (
                "scipy.ndimage.label with 4-connectivity structure for per-instance "
                "segmentation derivation from 5-class hard mask; canonical CS computer-vision "
                "segmentation primitive (instance segmentation from semantic segmentation via "
                "connected-components)."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #3 alternative "
                "reducer: (M40 UNIWARD x D16 per-segment-label / per-instance connected-components)"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "uniward_per_segment_label_v1",
        "next_action_on_POSITIVE_BREAKS_THRESHOLD": (
            "STRONG empirical case for Tier-2 paid dispatch on per-instance UNIWARD-weighted "
            "SegNet loss substrate via Vast.ai 4090 ($0.25/hr); estimated cost ~$1-5; predicted "
            "DS -0.005 to -0.015 [predicted]."
        ),
        "next_action_on_PARTIAL": (
            "DEFER-PENDING-MULTI-SCALE; iterate sister probe with per-instance + UNIWARD multi-"
            "scale wavelet COMBINED per Catalog #308."
        ),
        "next_action_on_NULL": (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification; queue per-instance + "
            "UNIWARD multi-scale combined OR boundary-aware UNIWARD per Catalog #308."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_8_uniward_per_segment_label_segnet_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    min_w = verdict["actual_signature"]["min_segment_textured_avg_weight"]
    spread = verdict["actual_signature"]["spread_segment_textured_avg_weight"]
    vsc = verdict["actual_signature"]["valid_segment_count"]
    print(
        f"[probe_8] verdict={verdict['verdict']} "
        f"min_segment_textured_avg_weight={min_w if min_w is None else f'{min_w:.4f}'} "
        f"(Probe 5 per-class min=0.5673; DDD baseline=0.626) "
        f"inter_segment_spread={spread if spread is None else f'{spread:.4f}'} "
        f"valid_segments={vsc} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_8] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
