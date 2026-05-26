#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CASCADE B Path A SISTER WAVE 1 — production-scale convergence harness
(600 frames x 1000 epochs on the canonical fixture).

Thin wrapper around `tools/cascade_b_path_a_learnable_head_smoke.py::run_smoke`
that re-uses every canonical primitive (real SegNet teacher cache builder +
LearnableConv1x1StudentHead + MLX value_and_grad loop + identical fixture for
both arms) at production scale.

Operator-pre-approved per 2026-05-26 verbatim "all are approved + follow up
are approved + pursue other attacks as well + remember all MLX first +
individually fractally optimized" sister wave 1.

Sister landing memo `15b11c86e` PARADIGM-VALIDATED at 50f x 100ep
(deterministic 6.17->6.18 -0.2% noise; Path A 6.01->4.52 -24.8% MONOTONIC;
delta 1.66 nats). This wave validates production-scale convergence stability
BEFORE the 5th-order CATALYST cascade composition (P5 QAT + P10 BPR1 sister
wave) is spawned.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317:
[macOS-MLX research-signal] only; no contest-score authority; promotion gated
on paired CPU+CUDA verify per Catalog #205.

Acceptance verdicts (per Catalog #307):
  CONVERGES_CONSISTENTLY        : final KL < 1.5
  PARTIAL_CONVERGENCE_EXTENDED  : 1.5 <= final KL < 4.0
  STAGNATES_AT_SCAFFOLD_FLOOR   : 4.0 <= final KL < 5.0
  DIVERGES                      : final KL >= 5.0 or NaN/Inf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

# Re-use canonical sister primitives
import tools.cascade_b_path_a_learnable_head_smoke as sister_smoke
# Sister harness uses parents[2] for REPO_ROOT, which is one level too high
# when imported as a module from a sibling tools/ wrapper. Patch the sister
# module's REPO_ROOT to the canonical project root so its video_path +
# upstream_dir + default output_dir all resolve correctly. The sister's
# direct-CLI path (`python tools/cascade_b_path_a_learnable_head_smoke.py`)
# happens to work because `python tools/X.py` resolves `parents[2]` from
# the file's absolute path which lands on the parent dir of `pact/`; the
# upstream lookup then re-resolves via REPO_ROOT/'upstream'/... which is
# /Users/adpena/Projects/upstream — that path DOES NOT EXIST and the
# sister's run_smoke would fail there too. We override here for both the
# wrapper-import path AND restore the canonical project-root semantics.
sister_smoke.REPO_ROOT = REPO_ROOT
run_smoke = sister_smoke.run_smoke


PRODUCTION_VERDICT_THRESHOLDS = {
    "CONVERGES_CONSISTENTLY_MAX_KL": 1.5,
    "PARTIAL_CONVERGENCE_EXTENDED_MAX_KL": 4.0,
    "STAGNATES_AT_SCAFFOLD_FLOOR_MAX_KL": 5.0,
}


def classify_production_verdict(final_loss: float) -> str:
    """Apply the Catalog #307 4-verdict taxonomy for production scale.

    Distinct from sister scaffold's 3-verdict taxonomy because production
    scale (10x epochs) expects deeper convergence floor.
    """
    if final_loss != final_loss or final_loss == float("inf"):  # NaN or +inf
        return "DIVERGES"
    if final_loss < PRODUCTION_VERDICT_THRESHOLDS["CONVERGES_CONSISTENTLY_MAX_KL"]:
        return "CONVERGES_CONSISTENTLY"
    if final_loss < PRODUCTION_VERDICT_THRESHOLDS["PARTIAL_CONVERGENCE_EXTENDED_MAX_KL"]:
        return "PARTIAL_CONVERGENCE_EXTENDED"
    if final_loss < PRODUCTION_VERDICT_THRESHOLDS["STAGNATES_AT_SCAFFOLD_FLOOR_MAX_KL"]:
        return "STAGNATES_AT_SCAFFOLD_FLOOR"
    return "DIVERGES"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=600)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "results"
            / "cascade_b_path_a_sister_wave_1_production_scale_20260526"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["both", "learnable", "deterministic"],
        default="both",
        help="Both arms by default for apples-to-apples production-scale comparison.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[sister-wave-1] CASCADE B Path A production-scale convergence harness "
        f"n_frames={args.n_frames} n_epochs={args.n_epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"[sister-wave-1] output_dir={output_dir}", flush=True)

    t0_total = time.time()
    results = {}

    if args.mode in ("both", "deterministic"):
        print(
            "[sister-wave-1] running DETERMINISTIC arm "
            "(sister baseline; 0 trainable params; expected -0.2% noise)...",
            flush=True,
        )
        t0 = time.time()
        det_telemetry = run_smoke(
            student_head_mode="deterministic",
            n_frames=args.n_frames,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            seed=args.seed,
            lr=args.learning_rate,
            output_dir=output_dir,
            artifact_label="sister_deterministic_projection_production",
        )
        det_wall = time.time() - t0
        det_telemetry["wall_clock_seconds_arm"] = det_wall
        det_telemetry["production_verdict"] = classify_production_verdict(
            det_telemetry["final_loss"]
        )
        results["deterministic"] = det_telemetry
        print(
            f"[sister-wave-1] DETERMINISTIC arm complete in {det_wall:.1f}s "
            f"final_loss={det_telemetry['final_loss']:.4f} "
            f"production_verdict={det_telemetry['production_verdict']}",
            flush=True,
        )

    if args.mode in ("both", "learnable"):
        print(
            "[sister-wave-1] running LEARNABLE arm "
            "(Path A; 20 trainable params; expected monotonic descent past "
            "scaffold 4.52 plateau)...",
            flush=True,
        )
        t0 = time.time()
        learn_telemetry = run_smoke(
            student_head_mode="learnable",
            n_frames=args.n_frames,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            seed=args.seed,
            lr=args.learning_rate,
            output_dir=output_dir,
            artifact_label="cascade_b_path_a_learnable_head_production",
        )
        learn_wall = time.time() - t0
        learn_telemetry["wall_clock_seconds_arm"] = learn_wall
        learn_telemetry["production_verdict"] = classify_production_verdict(
            learn_telemetry["final_loss"]
        )
        results["learnable"] = learn_telemetry
        print(
            f"[sister-wave-1] LEARNABLE arm complete in {learn_wall:.1f}s "
            f"final_loss={learn_telemetry['final_loss']:.4f} "
            f"production_verdict={learn_telemetry['production_verdict']}",
            flush=True,
        )

    total_wall = time.time() - t0_total
    print(f"[sister-wave-1] TOTAL wall_clock={total_wall:.1f}s", flush=True)

    # Production-scale comparison summary
    if "deterministic" in results and "learnable" in results:
        det = results["deterministic"]
        learn = results["learnable"]
        print("\n=== CASCADE B PATH A SISTER WAVE 1 PRODUCTION-SCALE COMPARISON ===", flush=True)
        print(
            f"  Deterministic baseline: initial={det['initial_loss']:.4f} "
            f"final={det['final_loss']:.4f} min={det['min_loss_across_run']:.4f} "
            f"reduction={det['reduction_pct']:.1f}% verdict={det['production_verdict']}",
            flush=True,
        )
        print(
            f"  Path A learnable head:  initial={learn['initial_loss']:.4f} "
            f"final={learn['final_loss']:.4f} min={learn['min_loss_across_run']:.4f} "
            f"reduction={learn['reduction_pct']:.1f}% verdict={learn['production_verdict']}",
            flush=True,
        )
        print(
            f"  Delta (det - learn) final: {det['final_loss'] - learn['final_loss']:.4f}",
            flush=True,
        )
        print(f"  Path A trainable params: {learn['n_trainable_params']}", flush=True)
        summary = {
            "schema_version": "cascade_b_path_a_sister_wave_1_production_scale_comparison_v1_20260526",
            "lane_id": "lane_cascade_b_path_a_sister_wave_1_production_scale_convergence_20260526",
            "subagent_id": (
                "cascade-b-path-a-sister-wave-1-production-scale-convergence-"
                "600f-1000ep-canonical-fixture-mlx-first-individually-fractal-20260526"
            ),
            "production_verdict_thresholds": PRODUCTION_VERDICT_THRESHOLDS,
            "total_wall_clock_seconds": total_wall,
            "deterministic": det,
            "learnable_path_a": learn,
            "delta_det_minus_learn_final": det["final_loss"] - learn["final_loss"],
            "delta_det_minus_learn_min": det["min_loss_across_run"] - learn["min_loss_across_run"],
            "verdict_path_a_breaks_saturation_at_production_scale": (
                learn["final_loss"] < det["final_loss"] - 0.05
            ),
            "verdict_path_a_converges_consistently": (
                learn["production_verdict"] == "CONVERGES_CONSISTENTLY"
            ),
            "canonical_provenance": {
                "axis_tag": "[macOS-MLX research-signal]",
                "hardware_substrate": "macos_arm64",
                "evidence_grade": "macOS-MLX-research-signal",
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        }
        summary_path = output_dir / "sweep_results.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"  Summary written to {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
