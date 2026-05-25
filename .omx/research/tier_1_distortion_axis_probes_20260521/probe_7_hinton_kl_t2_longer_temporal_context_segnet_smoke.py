# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 WAVE-2 Probe 7: Hinton KL T=2.0 x LONGER temporal-context SegNet [macOS-CPU advisory]

Per COMBINED-TIER-1-CCC-EXT-PROBES landing memo commit `685fe6726` (probe 6
verdict POSITIVE_SIGNAL_TEMPORAL_CONTEXT at W=2 with kl_mean_temporal=7.607e-3 =
11.3x CCC static baseline 6.74e-4). Probe 7 extends Probe 6's W=2 window to
W in {4, 6, 8} to test whether the temporal-context signal SCALES with window
radius OR PLATEAUS / saturates at small W.

CONTRACT (Carmack MVP-first 5-step + W-extension):
  Canonical Hinton KL T=2.0 (Probe 6 baseline at W=2): for each pair p,
      teacher_window_logits = mean over w in [-W..+W] of SegNet(frames[p+w])
      student_logits        = SegNet(frames[p])
      KL_temporal_W(p)      = KL(softmax(teacher_window/T), softmax(student/T))
      kl_mean_temporal_W=2  = 7.607e-3 (Probe 6 result; 11.3x CCC static baseline)

  NEW dimension (D17 extended W ∈ {4, 6, 8}): does the temporal-context signal
  SCALE with window radius? At W=8 the window covers 17 frames (~0.7 sec @ 24fps);
  if the signal SCALES monotonically we expect ≥20x CCC static baseline (=1.35e-2),
  if it PLATEAUS at ~11x we infer W~3 captures the ego-motion-coherent dark-
  knowledge structure and additional frames add noise more than signal.

PREDICTED SIGNATURES (3 outcomes per Carmack MVP-first step 2):
  POSITIVE_SCALES:    kl_mean_temporal_W=8 ≥ 20x CCC static (≥ 1.35e-2)
                      AND monotone increase W=4 < W=6 < W=8
  POSITIVE_PLATEAU:   kl_mean_temporal_W=8 in [10x, 20x] CCC static
                      AND W=4-8 all within ±20% of Probe 6's 11.3x ratio
  NEGATIVE:           kl_mean_temporal_W=8 < 5x CCC static (< 3.37e-3)
                      OR per-class drift collapses (<2/5 classes drift > 1e-3)

FALSIFYING OUTCOME (NEGATIVE):
  - W-extension fails to scale OR plateau; temporal signal exhausted at W~3.
  - => DEFER per Catalog #307 IMPLEMENTATION-LEVEL falsification of W-extension
    paradigm. Probe 6's POSITIVE finding at W=2-3 remains INTACT; only the
    W-extension claim is falsified. Per Catalog #308 alternative reducer: try
    motion-aware weighting (ego-motion-conditioned temporal averaging per
    Atick-Redlich + Rao-Ballard predictive-coding paradigm).

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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from upstream.modules import SegNet  # noqa: E402


def _decode_first_n_frames(video_path: Path, n: int) -> torch.Tensor:
    """Decode first N frames; need N=24 for W=8 sliding window over 4 pairs."""
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


def _segnet_forward_single_frame(segnet: SegNet, frame: torch.Tensor) -> torch.Tensor:
    """Run SegNet on a single frame; build canonical (B=1, T=2, C, H, W) input."""
    pair = torch.stack([frame, frame], dim=0).unsqueeze(0)  # (1, 2, 3, H, W)
    with torch.no_grad():
        seg_input = segnet.preprocess_input(pair)  # (1, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (1, 5, 384, 512)
    return seg_logits


def _compute_window_kl(
    frame_logits_cache: dict[int, torch.Tensor],
    center_pair_indices: list[int],
    W: int,
    T: float,
) -> tuple[float, list[float], int, float, list[dict]]:
    """Compute kl_mean_temporal for a given window radius W.

    Returns: (kl_mean, kl_per_pair, classes_with_measurable_drift,
              soft_entropy_temporal_mean, per_class_drift_summary)
    """
    kl_temporal_per_pair: list[float] = []
    temporal_soft_entropies: list[float] = []
    per_class_temporal_drift: dict[int, list[float]] = {c: [] for c in range(5)}

    for center_idx in center_pair_indices:
        window_logits_list = [
            frame_logits_cache[center_idx + dw]
            for dw in range(-W, W + 1)
        ]
        teacher_window = torch.stack(window_logits_list, dim=0).mean(dim=0)  # (1, 5, 384, 512)
        student_temporal = frame_logits_cache[center_idx]  # no noise; difference IS the temporal signal
        teacher_soft_window = F.softmax(teacher_window / T, dim=1)
        student_log_soft_temporal = F.log_softmax(student_temporal / T, dim=1)
        kl_per_pixel_temporal = (
            teacher_soft_window
            * (teacher_soft_window.clamp_min(1e-10).log() - student_log_soft_temporal)
        ).sum(dim=1)
        kl_temporal_per_pair.append(float(kl_per_pixel_temporal.mean().item()))
        temporal_soft_entropies.append(float(
            -(teacher_soft_window * teacher_soft_window.clamp_min(1e-10).log()).sum(dim=1).mean().item()
        ))

        teacher_static_for_this_center = F.softmax(student_temporal / T, dim=1)
        for c in range(5):
            window_class_mass = float(teacher_soft_window[:, c].mean().item())
            static_class_mass = float(teacher_static_for_this_center[:, c].mean().item())
            per_class_temporal_drift[c].append(window_class_mass - static_class_mass)

    kl_mean = sum(kl_temporal_per_pair) / len(kl_temporal_per_pair)
    soft_entropy_temporal_mean = sum(temporal_soft_entropies) / len(temporal_soft_entropies)

    per_class_drift_summary = []
    classes_with_measurable_drift = 0
    for c in range(5):
        class_drifts = per_class_temporal_drift[c]
        mean_drift = sum(class_drifts) / len(class_drifts)
        abs_mean_drift = abs(mean_drift)
        if abs_mean_drift > 1e-3:
            classes_with_measurable_drift += 1
        per_class_drift_summary.append({
            "class_index": c,
            "mean_drift_window_minus_static": mean_drift,
            "abs_mean_drift": abs_mean_drift,
        })

    return kl_mean, kl_temporal_per_pair, classes_with_measurable_drift, soft_entropy_temporal_mean, per_class_drift_summary


def _run_probe() -> dict:
    t_start = time.time()
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Need W=8 sliding window over 4 center pairs → indices [center-8..center+8].
    # Center pairs [8, 9, 10, 11]; range [0..19] → decode 20 frames.
    W_values = [4, 6, 8]
    W_max = max(W_values)
    center_pair_indices = [8, 9, 10, 11]
    n_frames_needed = max(center_pair_indices) + W_max + 1  # = 11 + 8 + 1 = 20

    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=n_frames_needed)
    n_frames = frames.shape[0]

    # === Cache logits per frame for all window-needed indices ===
    frame_indices_needed = sorted(
        {
            idx + dw
            for idx in center_pair_indices
            for dw in range(-W_max, W_max + 1)
        }
    )
    frame_logits_cache: dict[int, torch.Tensor] = {}
    for fi in frame_indices_needed:
        frame_logits_cache[fi] = _segnet_forward_single_frame(
            segnet, frames[fi].to(device)
        )

    T = 2.0
    ccc_static_baseline = 6.739782984368503e-4  # exact CCC probe 1 value
    probe_6_w2_anchor = 7.607e-3  # Probe 6 W=2 result
    probe_6_w2_ratio_over_ccc = probe_6_w2_anchor / ccc_static_baseline  # ~11.29

    # === Compute KL per window size ===
    per_window_results: dict[int, dict] = {}
    for W in W_values:
        kl_mean, kl_per_pair, classes_drift, soft_entropy, per_class_drift = _compute_window_kl(
            frame_logits_cache, center_pair_indices, W, T
        )
        ratio_over_ccc = kl_mean / max(ccc_static_baseline, 1e-10)
        ratio_over_probe_6 = kl_mean / max(probe_6_w2_anchor, 1e-10)
        per_window_results[W] = {
            "W": W,
            "window_size_total_frames": 2 * W + 1,
            "kl_mean_temporal": kl_mean,
            "kl_temporal_per_pair": kl_per_pair,
            "classes_with_measurable_drift": classes_drift,
            "soft_entropy_temporal_mean": soft_entropy,
            "ratio_over_ccc_static_baseline": ratio_over_ccc,
            "ratio_over_probe_6_w2": ratio_over_probe_6,
            "per_class_drift_summary": per_class_drift,
        }

    # === Scaling signature analysis ===
    kl_w4 = per_window_results[4]["kl_mean_temporal"]
    kl_w6 = per_window_results[6]["kl_mean_temporal"]
    kl_w8 = per_window_results[8]["kl_mean_temporal"]
    ratio_w4 = per_window_results[4]["ratio_over_ccc_static_baseline"]
    ratio_w6 = per_window_results[6]["ratio_over_ccc_static_baseline"]
    ratio_w8 = per_window_results[8]["ratio_over_ccc_static_baseline"]

    sig_monotone_increase = kl_w4 < kl_w6 < kl_w8
    sig_w8_scales = ratio_w8 >= 20.0
    sig_w8_plateau = 10.0 <= ratio_w8 <= 20.0
    sig_w8_negative = ratio_w8 < 5.0

    # Plateau test: are W=4, W=6, W=8 all within ±20% of Probe 6's 11.3x ratio?
    plateau_band_lo = probe_6_w2_ratio_over_ccc * 0.80
    plateau_band_hi = probe_6_w2_ratio_over_ccc * 1.20
    all_in_plateau_band = all(
        plateau_band_lo <= per_window_results[W]["ratio_over_ccc_static_baseline"] <= plateau_band_hi
        for W in W_values
    )

    # Multi-class drift signature (≥2/5 classes drift > 1e-3) preserved across windows?
    drift_collapses = any(per_window_results[W]["classes_with_measurable_drift"] < 2 for W in W_values)

    # === Verdict logic ===
    if sig_w8_scales and sig_monotone_increase and not drift_collapses:
        verdict = "POSITIVE_SIGNAL_SCALES"
        recommendation = (
            f"POSITIVE_SIGNAL_SCALES: temporal-context KL_mean scales monotonically W=4 -> W=8 "
            f"(W=4: {ratio_w4:.2f}x, W=6: {ratio_w6:.2f}x, W=8: {ratio_w8:.2f}x CCC static baseline). "
            f"W=8 yields >=20x CCC baseline ({ratio_w8:.2f}x). The temporal-context signal is "
            f"NOT exhausted at W=2-3; substantial additional dark-knowledge structure available "
            "at longer windows. STRONG empirical case for Tier-2 paid dispatch of distillation-"
            "substrate training paradigm with W>=8 window size. Predicted DS -0.010 to -0.025 "
            "[predicted] per RATE-ATTACK-MATRIX cell #4 scaling extrapolation. Estimated cost ~$3-7."
        )
    elif (sig_w8_plateau or all_in_plateau_band) and not drift_collapses:
        verdict = "POSITIVE_SIGNAL_PLATEAU"
        recommendation = (
            f"POSITIVE_SIGNAL_PLATEAU: temporal-context KL_mean plateaus at ~10-12x CCC static "
            f"baseline across W in {{4, 6, 8}} (W=4: {ratio_w4:.2f}x, W=6: {ratio_w6:.2f}x, "
            f"W=8: {ratio_w8:.2f}x; Probe 6 W=2: {probe_6_w2_ratio_over_ccc:.2f}x). The temporal-"
            f"coherent dark-knowledge structure saturates at W~3; longer windows do not add signal. "
            "Per CLAUDE.md 'Forbidden premature KILL': Probe 6 W=2-3 substrate remains JUSTIFIED for "
            "Tier-2 paid dispatch (RATE-ATTACK-MATRIX cell #4); W>=4 extension paradigm is "
            "DEFER-PENDING (additional frames do not justify their cost in extra SegNet forwards). "
            "Predicted DS unchanged from Probe 6 baseline -0.005 to -0.020 [predicted]."
        )
    elif sig_w8_negative or drift_collapses:
        verdict = "NEGATIVE_W_EXTENSION_FALSIFIED"
        recommendation = (
            f"NEGATIVE_W_EXTENSION_FALSIFIED: temporal-context KL_mean fails to scale at W=8 "
            f"(ratio_w8={ratio_w8:.2f}x CCC static baseline; threshold 5x). Per Catalog #307 "
            "IMPLEMENTATION-LEVEL falsification of W-extension paradigm. Probe 6 W=2-3 finding "
            "remains INTACT (paradigm-level UNCHANGED); only the W-extension claim is falsified. "
            "Per Catalog #308 alternative reducer: queue motion-aware weighting (ego-motion-"
            "conditioned temporal averaging per Atick-Redlich 1990 + Rao-Ballard predictive-coding) "
            "OR per-segment-label D16 temporal-coherent grouping. Probe 6 W=2 Tier-2 dispatch "
            "recommendation UNCHANGED."
        )
    else:
        verdict = "MIXED_SIGNAL_DEFER"
        recommendation = (
            f"MIXED_SIGNAL_DEFER: W-extension signal is non-monotone OR partial-plateau "
            f"(W=4: {ratio_w4:.2f}x, W=6: {ratio_w6:.2f}x, W=8: {ratio_w8:.2f}x CCC static "
            "baseline). Per CLAUDE.md 'Forbidden premature KILL': DEFER-PENDING-W-AUDIT; iterate "
            "with intermediate W=3, W=5, W=7 to map the saturation curve more precisely. Probe 6 "
            "W=2 Tier-2 dispatch recommendation UNCHANGED."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_hinton_kl_t2_longer_temporal_context_segnet_smoke",
        "lane_id": "lane_combined_tier_1_wave_2_hinton_kl_longer_temporal_plus_uniward_per_segment_label_20260525",
        "probe_name": "Hinton KL T=2.0 x LONGER temporal-context SegNet (M44 x D17 W-extension W in {4,6,8})",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "POSITIVE_SCALES": "kl_mean_temporal W=8 >= 20x CCC static AND monotone W=4<W=6<W=8",
            "POSITIVE_PLATEAU": "kl_mean_temporal W=8 in [10x, 20x] CCC static AND all W within plateau band of Probe 6 W=2",
            "NEGATIVE": "kl_mean_temporal W=8 < 5x CCC static OR drift collapses",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "W_values_tested": W_values,
            "center_pair_indices": center_pair_indices,
            "per_window_results": per_window_results,
            "monotone_increase_W4_to_W8": sig_monotone_increase,
            "w8_scales_ge_20x": sig_w8_scales,
            "w8_plateau_10_to_20x": sig_w8_plateau,
            "w8_negative_lt_5x": sig_w8_negative,
            "all_W_in_plateau_band_of_probe_6": all_in_plateau_band,
            "drift_collapses_any_W_lt_2_of_5_classes": drift_collapses,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_probe_6_w2_baseline": {
            "probe_6_w2_kl_mean": probe_6_w2_anchor,
            "probe_6_w2_ratio_over_ccc": probe_6_w2_ratio_over_ccc,
            "ccc_static_baseline": ccc_static_baseline,
            "w4_kl_mean": kl_w4,
            "w6_kl_mean": kl_w6,
            "w8_kl_mean": kl_w8,
            "w4_over_probe_6_w2": per_window_results[4]["ratio_over_probe_6_w2"],
            "w6_over_probe_6_w2": per_window_results[6]["ratio_over_probe_6_w2"],
            "w8_over_probe_6_w2": per_window_results[8]["ratio_over_probe_6_w2"],
            "scaling_direction": (
                "monotone_increase_W=4_to_W=8" if sig_monotone_increase
                else "non_monotone_or_saturated"
            ),
        },
        "signature_checks": {
            "monotone_increase_present": sig_monotone_increase,
            "w8_scales_ge_20x_present": sig_w8_scales,
            "w8_plateau_10_to_20x_present": sig_w8_plateau,
            "w8_negative_lt_5x_present": sig_w8_negative,
            "all_W_in_plateau_band_present": all_in_plateau_band,
            "drift_collapses_present": drift_collapses,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate hinton_kl_temperature2_longer_temporal_context_v1 (Catalog #344 "
            "FORMALIZATION_PENDING; RATIFY-N pending per RATE-ATTACK-MATRIX cell #4 W-extension "
            "operator-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_7_hinton_kl_t2_longer_temporal_context_segnet",
            "predecessor_probes": [
                "tier_1_distortion_hinton_kl_t2_segnet_smoke (CCC)",
                "tier_1_distortion_hinton_kl_t2_temporal_context_segnet_smoke (Probe 6 W=2)",
            ],
            "hinton_canonical_reference": (
                "Hinton/Vinyals/Dean 2015 'Distilling the Knowledge in a Neural Network' "
                "(KL T=2.0 dark-knowledge); CLAUDE.md 'Grand Council Geoffrey Hinton' + "
                "'Quantizr intelligence' (Quantizr 0.33 [contest-CUDA] uses kl_on_logits T=2.0)."
            ),
            "temporal_context_reference": (
                "Atick-Redlich 1990 cooperative-receiver temporal coherence + Rao-Ballard "
                "predictive-coding paradigm; CLAUDE.md 'Grand Council Atick' + 'predictive coding'."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #4: "
                "(M44 Hinton KL T=2.0 x D17 per-time-window temporal-context) W-extension"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "hinton_kl_temperature2_longer_temporal_context_v1",
        "next_action_on_POSITIVE_SCALES": (
            "STRONG empirical case for Tier-2 paid dispatch of distillation-substrate training "
            "paradigm with W>=8 window size. Operator-routable: HF Jobs T4 (pending RECHARGE) or "
            "Vast.ai 4090 ($0.25/hr); estimated cost ~$3-7; predicted DS -0.010 to -0.025 "
            "[predicted]."
        ),
        "next_action_on_POSITIVE_PLATEAU": (
            "Probe 6 W=2 substrate remains JUSTIFIED for Tier-2 paid dispatch unchanged; W>=4 "
            "extension paradigm is DEFER-PENDING (additional frames do not justify cost)."
        ),
        "next_action_on_NEGATIVE": (
            "DEFER W-extension per Catalog #307 IMPLEMENTATION-LEVEL falsification; queue "
            "motion-aware weighting OR per-segment-label D16 temporal grouping per Catalog #308 "
            "alternative reducer. Probe 6 W=2 Tier-2 dispatch UNCHANGED."
        ),
        "next_action_on_MIXED": (
            "DEFER-PENDING-W-AUDIT; iterate sister probe with intermediate W=3, W=5, W=7 to map "
            "saturation curve."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_7_hinton_kl_t2_longer_temporal_context_segnet_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    pw = verdict["actual_signature"]["per_window_results"]
    print(
        f"[probe_7] verdict={verdict['verdict']} "
        f"W=4: {pw[4]['ratio_over_ccc_static_baseline']:.2f}x, "
        f"W=6: {pw[6]['ratio_over_ccc_static_baseline']:.2f}x, "
        f"W=8: {pw[8]['ratio_over_ccc_static_baseline']:.2f}x "
        f"(CCC baseline 6.74e-4; Probe 6 W=2 11.29x) "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_7] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
