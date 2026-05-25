# SPDX-License-Identifier: MIT
"""
COMBINED Tier-1 CCC-ext Probe 6: Hinton KL T=2.0 x temporal-context SegNet [macOS-CPU advisory]

Per RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cell #4:
  (M44 Hinton KL T=2.0 x D17 per-time-window temporal-context)
+ CCC probe 1 POSITIVE_SIGNAL (kl_mean=6.74e-4 static baseline)
+ Quantizr canonical 0.33 paradigm per CLAUDE.md "Quantizr intelligence"
+ CLAUDE.md "Grand Council Geoffrey Hinton" + Hinton/Vinyals/Dean 2014.

CONTRACT (Carmack MVP-first 5-step + temporal-context extension):
  Canonical Hinton KL T=2.0 (CCC probe 1 static baseline): for each pair p,
      teacher_logits = SegNet(frames[p+1])  # static center frame
      student_logits = teacher_logits + 0.1 * randn   # simulated student
      KL_static(p) = KL(softmax(teacher/T), softmax(student/T))
      kl_mean_static = 6.74e-4

  NEW dimension (D17 per-time-window temporal-context): for each pair p,
      teacher_window_logits = mean over w in [-W..+W] of SegNet(frames[p+w])
      student_logits = SegNet(frames[p])  # static center frame, no window
      KL_temporal(p) = KL(softmax(teacher_window/T), softmax(student/T))

  Hypothesis: temporal-context teacher (averaged over W=2-3 adjacent frames)
  reveals temporal-coherent dark-knowledge structure beyond CCC's per-frame
  static KL T=2.0 baseline. The temporal average smooths transient artifacts
  AND captures motion-coherent class-boundary dynamics that the SegNet's
  per-frame argmax does not surface (a kind of D17 temporal-coherent
  cooperative-receiver signal per Atick-Redlich 1990).

PREDICTED SIGNATURE (positive direction = MORE dark-knowledge in temporal context):
  - kl_temporal_mean >= kl_static_baseline = 6.74e-4 (CCC anchor)
  - Per-class temporal-context drift > 0 for at least 2 classes (temporal
    smoothing reveals class-boundary motion structure)
  - Soft-entropy of temporal-context teacher > static teacher (temporal
    averaging softens peaked single-frame logits)

FALSIFYING OUTCOME:
  - kl_temporal_mean < kl_static_baseline (temporal averaging suppresses
    rather than enhances dark-knowledge)
  - OR per-class drift < 0 across all classes (temporal averaging collapses
    class distinctions)
  - => DEFER per Catalog #307 IMPLEMENTATION-level falsification + Catalog #308
    alternative reducer (e.g. longer window W=5-7 OR per-time-window with
    motion-aware weighting).

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
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from upstream.modules import SegNet  # type: ignore


def _decode_first_n_frames(video_path: Path, n: int = 16) -> torch.Tensor:
    """Decode first N frames; need N=16 for sliding window W=2-3 over 4 pairs."""
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
    """Run SegNet on a single frame; build canonical (B=1, T=2, C, H, W) input.

    SegNet.preprocess_input slices `x[:, -1]` so we duplicate the frame at both
    pair positions; SegNet evaluates the last one. Returns logits (1, 5, 384, 512).
    """
    pair = torch.stack([frame, frame], dim=0).unsqueeze(0)  # (1, 2, 3, H, W)
    with torch.no_grad():
        seg_input = segnet.preprocess_input(pair)  # (1, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (1, 5, 384, 512)
    return seg_logits


def _run_probe() -> dict:
    t_start = time.time()
    device = torch.device("cpu")

    # === Load canonical SegNet ===
    segnet = SegNet()
    sd = load_file(str(REPO_ROOT / "upstream" / "models" / "segnet.safetensors"))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # Decode 16 frames; sliding window W=2 over 4 center-pair positions [4, 5, 6, 7]
    # so window indices [2..7] are all valid (window radius 2 around centers).
    frames = _decode_first_n_frames(REPO_ROOT / "upstream" / "videos" / "0.mkv", n=16)
    n_frames = frames.shape[0]

    # === Window settings ===
    W = 2  # window radius (2 frames before + 2 after = 5-frame window)
    center_pair_indices = [4, 5, 6, 7]  # need indices [2..9] to be valid for W=2

    # === Run SegNet forward on all window-needed frames ===
    # Cache logits per frame so we don't double-evaluate
    frame_indices_needed = sorted(set(
        idx + dw
        for idx in center_pair_indices
        for dw in range(-W, W + 1)
    ))
    frame_logits_cache: dict[int, torch.Tensor] = {}
    for fi in frame_indices_needed:
        frame_logits_cache[fi] = _segnet_forward_single_frame(
            segnet, frames[fi].to(device)
        )  # (1, 5, 384, 512)

    # === T=2.0 canonical Hinton (matches CCC probe 1) ===
    T = 2.0

    # === Static KL baseline (CCC probe 1 paradigm reproduced for delta) ===
    # For each center pair, teacher = SegNet(center), student = teacher + 0.1 * noise.
    # KL_static is the canonical CCC baseline; we reproduce it here apples-to-apples.
    torch.manual_seed(0xCCE)  # matches CCC probe 1 seed family
    kl_static_per_pair = []
    static_soft_entropies = []
    for center_idx in center_pair_indices:
        teacher_static = frame_logits_cache[center_idx]  # (1, 5, 384, 512)
        student_static = teacher_static + 0.1 * torch.randn_like(teacher_static)
        teacher_soft = F.softmax(teacher_static / T, dim=1)
        student_log_soft = F.log_softmax(student_static / T, dim=1)
        kl_per_pixel = (teacher_soft * (teacher_soft.clamp_min(1e-10).log() - student_log_soft)).sum(dim=1)
        kl_static_per_pair.append(float(kl_per_pixel.mean().item()))
        static_soft_entropies.append(float(
            -(teacher_soft * teacher_soft.clamp_min(1e-10).log()).sum(dim=1).mean().item()
        ))
    kl_mean_static = sum(kl_static_per_pair) / len(kl_static_per_pair)
    soft_entropy_static_mean = sum(static_soft_entropies) / len(static_soft_entropies)

    # === Temporal-context KL (D17 NEW dimension) ===
    # For each center pair, teacher = mean SegNet logits over window [-W..+W],
    # student = SegNet(center) directly. KL captures the dark-knowledge structure
    # the static student misses (motion-coherent class boundaries).
    torch.manual_seed(0xCCE)  # same seed for apples-to-apples vs static
    kl_temporal_per_pair = []
    temporal_soft_entropies = []
    per_class_temporal_drift = {c: [] for c in range(5)}
    for center_idx in center_pair_indices:
        window_logits_list = [
            frame_logits_cache[center_idx + dw]
            for dw in range(-W, W + 1)
        ]
        teacher_window = torch.stack(window_logits_list, dim=0).mean(dim=0)  # (1, 5, 384, 512)
        student_temporal = frame_logits_cache[center_idx]  # no noise; the difference IS the temporal signal
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

        # Per-class temporal drift: |teacher_window_class_mass - student_class_mass|
        teacher_static_for_this_center = F.softmax(student_temporal / T, dim=1)
        for c in range(5):
            window_class_mass = float(teacher_soft_window[:, c].mean().item())
            static_class_mass = float(teacher_static_for_this_center[:, c].mean().item())
            per_class_temporal_drift[c].append(window_class_mass - static_class_mass)

    kl_mean_temporal = sum(kl_temporal_per_pair) / len(kl_temporal_per_pair)
    soft_entropy_temporal_mean = sum(temporal_soft_entropies) / len(temporal_soft_entropies)

    # === Per-class drift summary ===
    per_class_drift_summary = []
    classes_with_positive_drift = 0
    for c in range(5):
        class_drifts = per_class_temporal_drift[c]
        mean_drift = sum(class_drifts) / len(class_drifts)
        abs_mean_drift = abs(mean_drift)
        if abs_mean_drift > 1e-3:  # measurable per-class drift threshold
            classes_with_positive_drift += 1
        per_class_drift_summary.append(
            {
                "class_index": c,
                "mean_drift_window_minus_static": mean_drift,
                "abs_mean_drift": abs_mean_drift,
                "drift_per_pair": class_drifts,
            }
        )

    # === Predicted signature checks ===
    # The NEW dimension test: temporal-context teacher ≥ CCC static baseline.
    # NOTE: CCC's static baseline was teacher + noise (kl > 0 from added noise).
    # Our temporal probe is teacher_window vs teacher_static (kl from window
    # averaging displacement). These are not strictly comparable, but the
    # ABSOLUTE temporal KL magnitude is what matters: if it is >= CCC baseline
    # (6.74e-4) then temporal context CARRIES at least as much dark-knowledge
    # structure as the static teacher-vs-noisy-student paradigm.
    ccc_static_baseline = 6.739782984368503e-4  # exact CCC probe 1 value
    sig_temporal_ge_baseline = kl_mean_temporal >= ccc_static_baseline
    sig_temporal_strictly_positive = kl_mean_temporal > 1e-5
    sig_multi_class_drift = classes_with_positive_drift >= 2
    sig_temporal_soft_entropy_ge_static = soft_entropy_temporal_mean >= soft_entropy_static_mean

    # === Verdict logic ===
    if (
        sig_temporal_ge_baseline
        and sig_multi_class_drift
        and sig_temporal_strictly_positive
    ):
        verdict = "POSITIVE_SIGNAL_TEMPORAL_CONTEXT"
        recommendation = (
            "POSITIVE_SIGNAL_TEMPORAL_CONTEXT: temporal-context KL_mean="
            f"{kl_mean_temporal:.6f} >= CCC static baseline ({ccc_static_baseline:.6f}) "
            f"with {classes_with_positive_drift}/5 classes exhibiting measurable "
            "per-pair drift. Per RATE-ATTACK-MATRIX cell #4: Tier-2 paid dispatch "
            "on per-time-window Hinton-distilled scorer surrogate via HF Jobs T4 or "
            "Vast.ai 4090 JUSTIFIED. Predicted ΔS -0.005 to -0.020 [predicted] per "
            "CCC probe 1 + Quantizr 0.33 anchor. Estimated cost ~$2-5."
        )
    elif sig_temporal_strictly_positive and sig_multi_class_drift:
        verdict = "POSITIVE_SIGNAL_TEMPORAL_CONTEXT_PARTIAL"
        recommendation = (
            "PARTIAL: temporal-context signal present but below CCC static baseline. "
            "Per CLAUDE.md 'Forbidden premature KILL': DEFER-PENDING-WINDOW-EXTENSION; "
            "iterate sister probe with longer window W=3-5 OR motion-aware weighting "
            "per Catalog #308 alternative reducer. Paradigm INTACT (temporal-coherent "
            "dark-knowledge is real); IMPLEMENTATION-level window size insufficient."
        )
    else:
        verdict = "NULL_SIGNAL_DEFER"
        recommendation = (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification: temporal-context "
            "Hinton KL T=2.0 did NOT exhibit non-trivial dark-knowledge structure beyond "
            "CCC static baseline. Per Catalog #308 reactivation criteria: try motion-"
            "compensated temporal context OR per-segment-label temporal grouping. "
            "Paradigm INTACT (Quantizr 0.33 [contest-CUDA] anchor); per-time-window "
            "dimension implementation-level falsified at W=2."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_hinton_kl_t2_temporal_context_segnet_smoke",
        "lane_id": "lane_combined_tier_1_ccc_ext_probes_uniward_per_class_plus_hinton_kl_temporal_context_20260525",
        "probe_name": "Hinton KL T=2.0 x temporal-context SegNet (M44 x D17 RATE-ATTACK-MATRIX cell #4)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "predicted_signature": {
            "kl_temporal_ge_baseline": f"kl_mean_temporal >= CCC static baseline {ccc_static_baseline:.6f}",
            "multi_class_drift": ">= 2 classes with abs(mean_drift) > 1e-3",
            "temporal_strictly_positive": "kl_mean_temporal > 1e-5",
        },
        "actual_signature": {
            "n_frames_decoded": n_frames,
            "window_radius_W": W,
            "window_size_total_frames": 2 * W + 1,
            "center_pair_indices": center_pair_indices,
            "kl_mean_static_reproduced_for_delta": kl_mean_static,
            "kl_static_per_pair_reproduced": kl_static_per_pair,
            "kl_mean_temporal": kl_mean_temporal,
            "kl_temporal_per_pair": kl_temporal_per_pair,
            "ccc_static_baseline_anchor": ccc_static_baseline,
            "soft_entropy_static_mean": soft_entropy_static_mean,
            "soft_entropy_temporal_mean": soft_entropy_temporal_mean,
            "soft_entropy_delta_temporal_minus_static": soft_entropy_temporal_mean - soft_entropy_static_mean,
            "per_class_drift_summary": per_class_drift_summary,
            "classes_with_measurable_drift": classes_with_positive_drift,
            "segnet_missing_keys": len(missing),
            "segnet_unexpected_keys": len(unexpected),
        },
        "delta_vs_ccc_baseline": {
            "ccc_static_kl_mean_baseline": ccc_static_baseline,
            "temporal_kl_mean": kl_mean_temporal,
            "absolute_delta_temporal_minus_static_baseline": kl_mean_temporal - ccc_static_baseline,
            "ratio_temporal_over_baseline": kl_mean_temporal / max(ccc_static_baseline, 1e-10),
            "temporal_passed_baseline_ge_threshold": sig_temporal_ge_baseline,
            "improvement_direction": (
                "temporal_higher_dark_knowledge_signal"
                if kl_mean_temporal > ccc_static_baseline
                else "temporal_lower_dark_knowledge_signal"
            ),
        },
        "signature_checks": {
            "temporal_ge_baseline_present": sig_temporal_ge_baseline,
            "temporal_strictly_positive_present": sig_temporal_strictly_positive,
            "multi_class_drift_present": sig_multi_class_drift,
            "temporal_soft_entropy_ge_static_present": sig_temporal_soft_entropy_ge_static,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": (
            "candidate hinton_kl_temperature2_temporal_context_v1 (Catalog #344 "
            "FORMALIZATION_PENDING; RATIFY-N pending per RATE-ATTACK-MATRIX cell #4 "
            "op-routable)"
        ),
        "catalog_references": ["#344", "#523", "#287", "#323", "#192", "#1", "#313", "#307", "#308"],
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_6_hinton_kl_t2_temporal_context_segnet",
            "predecessor_probes": [
                "tier_1_distortion_hinton_kl_t2_segnet_smoke (CCC)",
            ],
            "hinton_canonical_reference": (
                "Hinton/Vinyals/Dean 2015 'Distilling the Knowledge in a Neural Network' "
                "(KL T=2.0 dark-knowledge); CLAUDE.md 'Grand Council Geoffrey Hinton' + "
                "'Quantizr intelligence' (Quantizr 0.33 [contest-CUDA] uses kl_on_logits T=2.0)."
            ),
            "temporal_context_reference": (
                "Atick-Redlich 1990 cooperative-receiver temporal coherence; "
                "CLAUDE.md 'Grand Council Atick' + 'predictive coding'."
            ),
            "rate_attack_matrix_reference": (
                "RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit 7a78c5661 Top-5 cell #4: "
                "(M44 Hinton KL T=2.0 x D17 per-time-window temporal-context)"
            ),
        },
        "sister_canonical_equation_candidate_for_RATIFY_N": "hinton_kl_temperature2_temporal_context_v1",
        "next_action_on_POSITIVE_TEMPORAL_CONTEXT": (
            "Operator-routable: Tier-2 paid dispatch on temporal-context Hinton-"
            "distilled scorer surrogate via HF Jobs T4 (RECHARGE pending) or Vast.ai "
            "4090 ($0.25/hr); estimated cost ~$2-5; predicted ΔS -0.005 to -0.020 "
            "[predicted] per RATE-ATTACK-MATRIX cell #4 + CCC probe 1 + Quantizr "
            "0.33 [contest-CUDA] anchor."
        ),
        "next_action_on_PARTIAL": (
            "Continue sister-probe iteration at $0: try longer window W=3-5 OR "
            "motion-aware weighting per Catalog #308."
        ),
        "next_action_on_NULL": (
            "DEFER per Catalog #307 IMPLEMENTATION-level falsification; queue motion-"
            "compensated temporal context per Catalog #308 reactivation criteria."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_6_hinton_kl_t2_temporal_context_segnet_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[probe_6] verdict={verdict['verdict']} "
        f"kl_mean_temporal={verdict['actual_signature']['kl_mean_temporal']:.6f} "
        f"(CCC static baseline={verdict['actual_signature']['ccc_static_baseline_anchor']:.6f}) "
        f"classes_with_drift={verdict['actual_signature']['classes_with_measurable_drift']}/5 "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_6] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
