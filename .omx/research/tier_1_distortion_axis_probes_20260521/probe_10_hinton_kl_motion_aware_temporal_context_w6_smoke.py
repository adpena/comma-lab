# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 WAVE-3 Probe 10: Hinton KL T=2.0 motion-aware temporal-context W=6 [macOS-CPU advisory]

Per COMBINED-TIER-1-WAVE-2 landing memo commit `68f0bba4d` (Probe 7 verdict
POSITIVE_SIGNAL_PLATEAU; W=6 peak 21.20x CCC static baseline 6.74e-4). Probe 10
extends Probe 7's W=6 peak via motion-aware per-pair weighting: weight each
pair's temporal-context KL by ego-motion concentration (canonical
Atick-Redlich 1990 cooperative-receiver + Rao-Ballard predictive-coding
paradigm).

CONTRACT (Carmack MVP-first 5-step + motion-aware cooperative-receiver):
  Canonical Hinton KL T=2.0 W=6 (Probe 7 peak): for each pair p,
      teacher_window_logits = mean over w in [-6..+6] of SegNet(frames[p+w])
      student_logits        = SegNet(frames[p])
      KL_temporal_W6(p)     = KL(softmax(teacher_window/T), softmax(student/T))
      kl_mean_temporal_W=6  = 1.428e-2 (Probe 7 result; 21.20x CCC static baseline)

  NEW dimension (D27 motion-aware per-pair weighting): for each pair p, compute
  ego_motion_concentration(p) as the inter-frame L2 norm of luma frame
  differences within the W=6 window:
      ego_motion_norm(p) = (1/(2W+1)) * sum_{w in [-W..+W]} ||luma(p+w) - luma(p)||_2

  Then aggregate motion-weighted KL:
      motion_weighted_kl_W6 = (sum_p ego_motion_norm(p) * KL_temporal_W6(p)) / sum_p ego_motion_norm(p)
  vs uniform aggregate:
      uniform_kl_W6 = (1/N) * sum_p KL_temporal_W6(p) = Probe 7 W=6 mean

  Hypothesis: per Atick-Redlich 1990 + Rao-Ballard predictive-coding, the
  temporal-context dark-knowledge structure is CONCENTRATED in ego-motion-rich
  pairs (large optical-flow magnitude reveals more predictive coding bandwidth).
  Motion-weighted aggregate should be >= 1.5x uniform aggregate IF ego-motion
  concentration is predictive of KL density.

PREDICTED SIGNATURES (3 outcomes per Carmack MVP-first step 2):
  POSITIVE_MOTION_AMPLIFIED:  motion_weighted_kl_W6 / uniform_kl_W6 >= 1.5x
                              AND high-motion-pair KL > low-motion-pair KL (per
                              per-pair pearson correlation > 0.6)
  POSITIVE_MOTION_PARTIAL:    1.0x < ratio < 1.5x (motion concentration weakly
                              predictive)
  NEGATIVE_MOTION_NEUTRAL:    ratio <= 1.0x (motion concentration NOT predictive
                              of temporal-context dark-knowledge density;
                              hypothesis IMPLEMENTATION-LEVEL falsified)

FALSIFYING OUTCOME (NEGATIVE):
  - Motion-weighted aggregate is <= uniform aggregate -> motion concentration is
    NOT predictive of temporal-context KL density.
  - => DEFER per Catalog #307 IMPLEMENTATION-LEVEL falsification of the motion-
    aware cooperative-receiver hypothesis at W=6. Probe 7's W=6 21.20x peak
    finding remains INTACT (paradigm-level UNCHANGED); only the motion-
    concentration enhancement is falsified.
  - Per Catalog #308 alternative reducer: try per-segment-label D16 temporal
    grouping (motion-aware aggregation REPLACED by per-instance temporal averaging)
    OR Atick-Redlich pure-redundancy-reduction aggregation (motion-orthogonal
    spatial coherence).

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
    """ITU-R BT.601 luma; canonical primitive shared with Probe 3/5/8/9."""
    return 0.2989 * frames[:, 0] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 2]


def _segnet_forward_single_frame(segnet: SegNet, frame: torch.Tensor) -> torch.Tensor:
    """Run SegNet on a single frame; build canonical (B=1, T=2, C, H, W) input."""
    pair = torch.stack([frame, frame], dim=0).unsqueeze(0)  # (1, 2, 3, H, W)
    with torch.no_grad():
        seg_input = segnet.preprocess_input(pair)  # (1, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (1, 5, 384, 512)
    return seg_logits


def _compute_ego_motion_concentration_per_pair(
    luma_cache: dict[int, torch.Tensor],
    center_pair_indices: list[int],
    W: int,
) -> list[float]:
    """Compute per-pair ego-motion concentration via L2 luma frame-difference magnitude.

    For each center pair p, compute the mean L2 distance from the center luma
    to each window frame luma. This is a canonical proxy for ego-motion
    magnitude — larger values indicate more inter-frame motion within the W=6
    window. Per Atick-Redlich 1990 early visual processing + canonical
    computer-vision dense optical-flow magnitude approximation.
    """
    ego_motion_norms: list[float] = []
    for center_idx in center_pair_indices:
        center_luma = luma_cache[center_idx]  # (H, W)
        distances = []
        for dw in range(-W, W + 1):
            if dw == 0:
                continue
            window_luma = luma_cache[center_idx + dw]
            d = float(((center_luma - window_luma) ** 2).mean().sqrt().item())
            distances.append(d)
        ego_motion_norms.append(float(np.mean(distances)))
    return ego_motion_norms


def _compute_kl_per_pair_w6(
    frame_logits_cache: dict[int, torch.Tensor],
    center_pair_indices: list[int],
    W: int,
    T: float,
) -> list[float]:
    """Compute per-pair Hinton KL T=2.0 temporal-context at fixed W=6 (Probe 7 peak)."""
    kl_per_pair: list[float] = []
    for center_idx in center_pair_indices:
        window_logits_list = [
            frame_logits_cache[center_idx + dw]
            for dw in range(-W, W + 1)
        ]
        teacher_window = torch.stack(window_logits_list, dim=0).mean(dim=0)  # (1, 5, 384, 512)
        student_temporal = frame_logits_cache[center_idx]
        teacher_soft_window = F.softmax(teacher_window / T, dim=1)
        student_log_soft_temporal = F.log_softmax(student_temporal / T, dim=1)
        kl_per_pixel = (
            teacher_soft_window
            * (teacher_soft_window.clamp_min(1e-10).log() - student_log_soft_temporal)
        ).sum(dim=1)
        kl_per_pair.append(float(kl_per_pixel.mean().item()))
    return kl_per_pair


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    numer = float(((x - x_mean) * (y - y_mean)).sum())
    denom = float(np.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return numer / denom


def _run_probe() -> dict:
    t_start = time.time()
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Fix W=6 (Probe 7 peak); apples-to-apples Probe 7 layout.
    # Center pairs [6, 7, 8, 9, 10, 11] => 6 pairs (more than Probe 7's 4 for
    # better motion-vs-KL correlation statistics); range [0..17] => decode 18 frames.
    W = 6
    center_pair_indices = [6, 7, 8, 9, 10, 11]
    n_frames_needed = max(center_pair_indices) + W + 1  # = 11 + 6 + 1 = 18

    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=n_frames_needed)
    n_frames = frames.shape[0]

    # === Cache luma + logits per frame for all window-needed indices ===
    frame_indices_needed = sorted(
        {
            idx + dw
            for idx in center_pair_indices
            for dw in range(-W, W + 1)
        }
    )
    luma_all = _luma(frames)  # (n_frames, H, W)
    luma_cache: dict[int, torch.Tensor] = {fi: luma_all[fi] for fi in frame_indices_needed}

    frame_logits_cache: dict[int, torch.Tensor] = {}
    for fi in frame_indices_needed:
        frame_logits_cache[fi] = _segnet_forward_single_frame(
            segnet, frames[fi].to(device)
        )

    T = 2.0
    ccc_static_baseline = 6.739782984368503e-4
    probe_7_w6_anchor = 1.428e-2  # Probe 7 W=6 peak result
    probe_7_w6_ratio_over_ccc = probe_7_w6_anchor / ccc_static_baseline  # ~21.20

    # === Compute per-pair KL (W=6) and per-pair ego-motion concentration ===
    kl_per_pair = _compute_kl_per_pair_w6(frame_logits_cache, center_pair_indices, W, T)
    ego_motion_norms = _compute_ego_motion_concentration_per_pair(luma_cache, center_pair_indices, W)

    # === Aggregate uniform vs motion-weighted ===
    uniform_kl_W6 = float(np.mean(kl_per_pair))
    motion_weights_total = float(sum(ego_motion_norms))
    if motion_weights_total > 0.0:
        motion_weighted_kl_W6 = float(
            sum(em * kl for em, kl in zip(ego_motion_norms, kl_per_pair, strict=True)) / motion_weights_total
        )
    else:
        motion_weighted_kl_W6 = uniform_kl_W6

    motion_amplification_ratio = (
        motion_weighted_kl_W6 / uniform_kl_W6
        if uniform_kl_W6 > 0.0 else float("nan")
    )

    # === Per-pair correlation between ego-motion and KL ===
    motion_kl_pearson = _pearson_correlation(ego_motion_norms, kl_per_pair)

    # === High-motion vs low-motion partition (median split) ===
    median_motion = float(np.median(ego_motion_norms))
    high_motion_kls = [
        kl for em, kl in zip(ego_motion_norms, kl_per_pair, strict=True)
        if em >= median_motion
    ]
    low_motion_kls = [
        kl for em, kl in zip(ego_motion_norms, kl_per_pair, strict=True)
        if em < median_motion
    ]
    high_motion_kl_mean = float(np.mean(high_motion_kls)) if high_motion_kls else float("nan")
    low_motion_kl_mean = float(np.mean(low_motion_kls)) if low_motion_kls else float("nan")
    high_vs_low_motion_kl_ratio = (
        high_motion_kl_mean / low_motion_kl_mean
        if low_motion_kl_mean > 0.0
        else float("nan")
    )

    # === Ratios over CCC static baseline ===
    motion_weighted_ratio_over_ccc = motion_weighted_kl_W6 / max(ccc_static_baseline, 1e-10)
    uniform_ratio_over_ccc = uniform_kl_W6 / max(ccc_static_baseline, 1e-10)
    high_motion_ratio_over_ccc = (
        high_motion_kl_mean / ccc_static_baseline
        if high_motion_kl_mean == high_motion_kl_mean else float("nan")
    )

    # === Predicted signature checks ===
    sig_motion_amplified_ge_1p5x = (
        motion_amplification_ratio == motion_amplification_ratio  # not NaN
        and motion_amplification_ratio >= 1.5
    )
    sig_motion_partial_gt_1p0x = (
        motion_amplification_ratio == motion_amplification_ratio
        and 1.0 < motion_amplification_ratio < 1.5
    )
    sig_motion_neutral_le_1p0x = (
        motion_amplification_ratio == motion_amplification_ratio
        and motion_amplification_ratio <= 1.0
    )
    sig_pearson_positive_ge_0p6 = (
        motion_kl_pearson == motion_kl_pearson
        and motion_kl_pearson >= 0.6
    )

    # === Verdict logic ===
    if sig_motion_amplified_ge_1p5x and sig_pearson_positive_ge_0p6:
        verdict = "POSITIVE_SIGNAL_MOTION_AMPLIFIED"
        recommendation = (
            f"POSITIVE_SIGNAL_MOTION_AMPLIFIED: motion-weighted aggregate yields "
            f"{motion_weighted_kl_W6:.4e} = {motion_weighted_ratio_over_ccc:.2f}x CCC static "
            f"baseline (vs uniform W=6 aggregate {uniform_kl_W6:.4e} = "
            f"{uniform_ratio_over_ccc:.2f}x; amplification ratio "
            f"{motion_amplification_ratio:.2f}x >= 1.5x threshold). Per-pair pearson "
            f"correlation ego_motion vs KL = {motion_kl_pearson:.3f} >= 0.6. Per Atick-Redlich "
            "1990 + Rao-Ballard predictive-coding canonical: ego-motion concentration IS "
            "predictive of temporal-context dark-knowledge density. STRONG empirical case for "
            "Tier-2 paid dispatch on motion-weighted temporal Hinton-distilled scorer surrogate "
            "substrate JUSTIFIED. Predicted DS -0.010 to -0.025 [predicted]. Estimated cost ~$2-7."
        )
    elif sig_motion_amplified_ge_1p5x or (
        sig_motion_partial_gt_1p0x and sig_pearson_positive_ge_0p6
    ):
        verdict = "POSITIVE_SIGNAL_MOTION_PARTIAL"
        recommendation = (
            f"POSITIVE_SIGNAL_MOTION_PARTIAL: motion-aware weighting tightens or correlates with "
            f"the temporal Hinton KL signal but not strongly (amplification ratio "
            f"{motion_amplification_ratio:.2f}x; pearson {motion_kl_pearson:.3f}). Per CLAUDE.md "
            "'Forbidden premature KILL': DEFER-PENDING-WEIGHTING-AUDIT; iterate sister probe "
            "with per-segment-label temporal grouping OR Rao-Ballard hierarchical prediction-"
            "error aggregation per Catalog #308. Probe 7 W=6 baseline INTACT (motion concentration "
            "is weakly predictive)."
        )
    else:
        verdict = "NEGATIVE_MOTION_NEUTRAL"
        recommendation = (
            f"NEGATIVE_MOTION_NEUTRAL: motion concentration is NOT predictive of temporal-context "
            f"KL density (motion-weighted/uniform ratio={motion_amplification_ratio:.2f}x <= 1.0x; "
            f"pearson={motion_kl_pearson:.3f}). Per Catalog #307 IMPLEMENTATION-LEVEL falsification "
            "of the motion-aware cooperative-receiver enhancement. Probe 7 W=6 21.20x finding "
            "remains INTACT (paradigm-level UNCHANGED); only the motion-concentration enhancement "
            "is falsified. Per Catalog #308 alternative reducer: queue per-segment-label D16 "
            "temporal grouping OR Atick-Redlich pure-redundancy-reduction aggregation. Probe 7 "
            "W=6 Tier-2 dispatch recommendation UNCHANGED."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_hinton_kl_motion_aware_temporal_context_w6_smoke",
        "lane_id": "lane_combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_20260525",
        "probe_name": "Hinton KL T=2.0 motion-aware temporal-context W=6 (M44 x D17 x D27 ego-motion-weighted)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "POSITIVE_MOTION_AMPLIFIED": "motion_weighted / uniform >= 1.5x AND pearson(ego_motion, KL) >= 0.6",
            "POSITIVE_MOTION_PARTIAL": "1.0x < motion_weighted/uniform < 1.5x AND pearson >= 0.6",
            "NEGATIVE_MOTION_NEUTRAL": "motion_weighted/uniform <= 1.0x (motion NOT predictive of KL)",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "W_fixed": W,
            "center_pair_indices": center_pair_indices,
            "kl_per_pair": kl_per_pair,
            "ego_motion_norms_per_pair": ego_motion_norms,
            "uniform_kl_W6": uniform_kl_W6,
            "motion_weighted_kl_W6": motion_weighted_kl_W6,
            "motion_amplification_ratio": motion_amplification_ratio,
            "motion_kl_pearson": motion_kl_pearson,
            "median_motion": median_motion,
            "high_motion_kl_mean": high_motion_kl_mean,
            "low_motion_kl_mean": low_motion_kl_mean,
            "high_vs_low_motion_kl_ratio": high_vs_low_motion_kl_ratio,
            "uniform_ratio_over_ccc_static_baseline": uniform_ratio_over_ccc,
            "motion_weighted_ratio_over_ccc_static_baseline": motion_weighted_ratio_over_ccc,
            "high_motion_ratio_over_ccc_static_baseline": high_motion_ratio_over_ccc,
            "ccc_static_baseline": ccc_static_baseline,
            "probe_7_w6_anchor": probe_7_w6_anchor,
            "probe_7_w6_ratio_over_ccc": probe_7_w6_ratio_over_ccc,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_probe_7_w6_baseline": {
            "probe_7_w6_kl_mean": probe_7_w6_anchor,
            "probe_7_w6_ratio_over_ccc": probe_7_w6_ratio_over_ccc,
            "motion_weighted_vs_probe_7_absolute_delta": motion_weighted_kl_W6 - probe_7_w6_anchor,
            "motion_weighted_vs_probe_7_relative_ratio": (
                motion_weighted_kl_W6 / probe_7_w6_anchor if probe_7_w6_anchor > 0 else float("nan")
            ),
            "motion_amplification_direction": (
                "motion_amplifies_KL" if motion_amplification_ratio > 1.0 else "motion_neutral_or_dilutes_KL"
            ),
        },
        "signature_checks": {
            "motion_amplified_ge_1p5x_present": sig_motion_amplified_ge_1p5x,
            "motion_partial_gt_1p0x_present": sig_motion_partial_gt_1p0x,
            "motion_neutral_le_1p0x_present": sig_motion_neutral_le_1p0x,
            "pearson_positive_ge_0p6_present": sig_pearson_positive_ge_0p6,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate hinton_kl_motion_aware_temporal_context_v1 (Catalog #344 "
            "FORMALIZATION_PENDING; RATIFY-N pending per RATE-ATTACK-MATRIX cell #4 + cell #29 "
            "motion-aware aggregation operator-routable)"
        ),
        "catalog_references": ["#344", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_10_hinton_kl_motion_aware_temporal_context_w6",
            "predecessor_probes": [
                "tier_1_distortion_hinton_kl_t2_segnet_smoke (CCC)",
                "tier_1_distortion_hinton_kl_t2_temporal_context_segnet_smoke (Probe 6 W=2)",
                "tier_1_distortion_hinton_kl_t2_longer_temporal_context_segnet_smoke (Probe 7 W=6 peak)",
            ],
            "hinton_canonical_reference": (
                "Hinton/Vinyals/Dean 2015 'Distilling the Knowledge in a Neural Network' "
                "(KL T=2.0 dark-knowledge); CLAUDE.md 'Grand Council Geoffrey Hinton' + "
                "'Quantizr intelligence' (Quantizr 0.33 [contest-CUDA] uses kl_on_logits T=2.0)."
            ),
            "motion_aware_cooperative_receiver_reference": (
                "Atick-Redlich 1990 'Towards a Theory of Early Visual Processing' + "
                "Rao-Ballard 1999 'Predictive coding in the visual cortex' canonical "
                "cooperative-receiver + predictive-coding paradigm; CLAUDE.md 'Grand Council "
                "Atick' + 'predictive coding'. Ego-motion concentration weight derived from "
                "luma frame-difference L2 magnitude — canonical computer-vision dense optical-"
                "flow proxy without explicit Lucas-Kanade / RAFT solver."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #4 (temporal-"
                "context) + cell #29 (motion-aware aggregation) COMBINED dimension D27: per-"
                "pair ego-motion-weighted aggregation of Probe 7 W=6 per-pair KL values"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "hinton_kl_motion_aware_temporal_context_v1",
        "next_action_on_POSITIVE_MOTION_AMPLIFIED": (
            "STRONG empirical case for Tier-2 paid dispatch on motion-weighted temporal Hinton-"
            "distilled scorer surrogate substrate via Vast.ai 4090 ($0.25/hr); estimated cost "
            "~$2-7; predicted DS -0.010 to -0.025 [predicted]."
        ),
        "next_action_on_POSITIVE_PARTIAL": (
            "DEFER-PENDING-WEIGHTING-AUDIT; iterate sister probe with per-segment-label "
            "temporal grouping per Catalog #308 alternative reducer."
        ),
        "next_action_on_NEGATIVE": (
            "DEFER motion-aware enhancement per Catalog #307 IMPLEMENTATION-LEVEL falsification; "
            "queue per-segment-label D16 temporal grouping OR Atick-Redlich pure-redundancy-"
            "reduction aggregation per Catalog #308. Probe 7 W=6 Tier-2 dispatch UNCHANGED."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_10_hinton_kl_motion_aware_temporal_context_w6_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    a = verdict["actual_signature"]
    print(
        f"[probe_10] verdict={verdict['verdict']} "
        f"motion_amplification_ratio={a['motion_amplification_ratio']:.3f}x "
        f"pearson(ego_motion, KL)={a['motion_kl_pearson']:.3f} "
        f"high_motion_KL/low_motion_KL={a['high_vs_low_motion_kl_ratio']:.3f}x "
        f"uniform_W6_ratio_over_ccc={a['uniform_ratio_over_ccc_static_baseline']:.2f}x "
        f"motion_weighted_W6_ratio_over_ccc={a['motion_weighted_ratio_over_ccc_static_baseline']:.2f}x "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_10] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
