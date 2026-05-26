# SPDX-License-Identifier: MIT
"""Cascade C' sensitivity sweep over Atick-Redlich asymmetric channel synthesis parameters.

CARMACK-DISSENT TEST per Catalog #307 paradigm-vs-implementation:
The smoke verdict depends on synthesis distribution choices. Sweep:
  - frame-1 seg_penalty mean (Atick-Redlich predicts O(boundary_pixels × disagreement_prob))
  - frame-1 pose_savings magnitude (Atick-Redlich predicts O(5x frame-0 PoseNet-null))
  - pose_avg_baseline operating point (frontier-pursuit vs plateau-adjacent)

Output: per-cell verdict (PARADIGM_VALIDATED / IMPLEMENTATION_LEVEL_NULL)
"""
import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_c_prime_atick_redlich_asymmetric_channel_smoke import (
    per_pair_lagrangian_dual_routing,
    compute_sidecar_bytes,
    synthesize_frame_0_per_pair_measurements,
    load_posenet_null_artifact,
    N_PAIRS,
    SEG_MULTIPLIER,
    POSE_SQRT_INNER,
    RATE_MULTIPLIER,
    RATE_DENOM_BYTES,
)


def synthesize_frame_1_parametric(seg_penalty_mean, pose_savings_mean, n_modes=8, seed=20260526):
    """Parametric synthesis of frame-1 menu."""
    rng = np.random.default_rng(seed=seed)
    seg_shape, seg_scale = 0.5, seg_penalty_mean / 0.5  # gamma mean = shape × scale
    frame_1_seg_penalty = rng.gamma(shape=seg_shape, scale=seg_scale, size=(N_PAIRS, n_modes))
    pose_penalty = rng.gamma(shape=0.3, scale=3e-7, size=(N_PAIRS, n_modes))
    pose_savings_shape, pose_savings_scale = 0.5, pose_savings_mean / 0.5
    pose_savings = -rng.gamma(shape=pose_savings_shape, scale=pose_savings_scale, size=(N_PAIRS, n_modes))
    return frame_1_seg_penalty, pose_penalty + pose_savings


def main():
    posenet_null = load_posenet_null_artifact()
    frame_0_seg_penalty, frame_0_pose_delta = synthesize_frame_0_per_pair_measurements(posenet_null)

    # Sweep grid
    seg_penalty_means = [5e-6, 1e-5, 5e-5, 1e-4]  # frame-1 seg cost
    pose_savings_means = [3e-7, 1.5e-6, 5e-6, 1e-5]  # frame-1 pose savings magnitude
    pose_avg_baselines = [3.4e-5, 1e-4, 1e-3]  # operating point

    results = []
    print(f"{'seg_mean':>10s} {'pose_save':>11s} {'pose_avg':>10s} {'frame_1_pct':>12s} {'net_score_delta_opt_b':>22s} {'verdict':>12s}")
    print("-" * 90)

    for seg_mean in seg_penalty_means:
        for pose_save_mean in pose_savings_means:
            for pose_avg in pose_avg_baselines:
                frame_1_seg, frame_1_pose = synthesize_frame_1_parametric(seg_mean, pose_save_mean)
                routing = per_pair_lagrangian_dual_routing(
                    frame_0_seg_penalty, frame_0_pose_delta,
                    frame_1_seg, frame_1_pose,
                    pose_avg,
                )
                rd = routing["routing_decision"]
                n_f1 = int(np.sum(rd == 1))
                total_score_delta = float(np.sum(routing["per_pair_score_delta"]))
                sidecar = compute_sidecar_bytes(rd)
                sidecar_cost = RATE_MULTIPLIER * sidecar / RATE_DENOM_BYTES
                net_b = total_score_delta + sidecar_cost
                verdict = "PARADIGM" if net_b < -0.001 else ("MARGINAL" if net_b < 0 else "NULL_OR_WORSE")
                print(f"{seg_mean:>10.1e} {pose_save_mean:>11.1e} {pose_avg:>10.1e} {n_f1:>12d} {net_b:>22.6f} {verdict:>12s}")
                results.append({
                    "frame_1_seg_penalty_mean": seg_mean,
                    "frame_1_pose_savings_mean": pose_save_mean,
                    "pose_avg_baseline": pose_avg,
                    "frame_1_selected_count": n_f1,
                    "total_score_delta_lagrangian": total_score_delta,
                    "sidecar_bytes_b": sidecar,
                    "net_score_delta_opt_b": net_b,
                    "verdict": verdict,
                })

    out_path = Path(__file__).resolve().parent / "cascade_c_prime_sensitivity_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "schema_version": "cascade_c_prime_sensitivity_v1_20260526",
            "axis_tag": "[macOS-MLX research-signal]",
            "evidence_grade": "macOS-MLX-research-signal",
            "promotable": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "sweep_dimensions": ["frame_1_seg_penalty_mean", "frame_1_pose_savings_mean", "pose_avg_baseline"],
            "n_combinations": len(results),
            "n_paradigm_validated": sum(1 for r in results if r["verdict"] == "PARADIGM"),
            "n_marginal": sum(1 for r in results if r["verdict"] == "MARGINAL"),
            "n_null_or_worse": sum(1 for r in results if r["verdict"] == "NULL_OR_WORSE"),
            "results": results,
        }, f, indent=2, sort_keys=True)
    print(f"\nWrote sweep: {out_path}")
    print(f"Verdicts: PARADIGM={sum(1 for r in results if r['verdict'] == 'PARADIGM')} | "
          f"MARGINAL={sum(1 for r in results if r['verdict'] == 'MARGINAL')} | "
          f"NULL={sum(1 for r in results if r['verdict'] == 'NULL_OR_WORSE')}")


if __name__ == "__main__":
    main()
