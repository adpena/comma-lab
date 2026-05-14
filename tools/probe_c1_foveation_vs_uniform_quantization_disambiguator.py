"""Probe: foveation strategy vs uniform quantization for C1.

Per Catalog #125 hook #6 + the design-tension memo
``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md``:
when a design choice has 2+ defensible interpretations, ship BOTH modes via
callable interface + build a probe that returns the regime-conditional
verdict.

For C1's foveation strategy the design tension is:

1. **UNIFORM**: no foveation; every pixel gets the same bit budget.
   Control / ablation baseline.
2. **EGO_MOTION_RADIAL**: 2D Gaussian foveation centered on the
   ego-motion vanishing point (Atick-Redlich 1990). Geometric, no
   learned params.
3. **LEARNED_PER_PIXEL**: learned per-pixel attention from world-model
   latent. More expressive but pays a small foveation_meta byte cost.

The probe simulates the rate-distortion tradeoff on a synthetic radial
target (mimicking the dashcam's natural foveation -- camera-center detail
matters more than periphery for the contest scorer's frame-1 SegNet path
since the segmentation classes are denser at the lane center). It
measures the residual quality at a fixed bit budget for each strategy
and emits a typed verdict.

Verdict schema (JSON output)::

    {
      "uniform": {
        "residual_l2": float,
        "bits_used_proxy": float,
        "concentration_index": float  # entropy of the bit-allocation map
      },
      "ego_motion_radial": { ... },
      "learned_per_pixel": { ... },
      "verdict": "uniform" | "ego_motion_radial" | "learned_per_pixel" | "tie",
      "verdict_rationale": str,
      "evidence_grade": "proxy",
      "score_claim_valid": false,
      "ready_for_exact_eval_dispatch": false,
      "promotion_eligible": false,
      "rank_or_kill_eligible": false,
      "result_review_blockers": [...]
    }

Usage::

    .venv/bin/python tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py \\
        --image-h 24 --image-w 32 --bit-budget 1000 \\
        --output reports/raw/c1_foveation_probe_<utc>.json

The probe is intentionally lightweight (no scorer load; pure geometric +
quantization simulation). Production trainer dispatching the chosen
strategy still requires full contest-CUDA auth eval per Catalog #221.

Cross-ref:
  .omx/research/campaign_c1_world_model_foveation_20260514.md
  src/tac/substrates/c1_world_model_foveation/architecture.py (FoveationStrategy enum)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tac.substrates.c1_world_model_foveation.architecture import (  # noqa: E402
    FoveationMapModule,
    FoveationStrategy,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelRecurrenceMode,
)


def _make_radial_target(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """Synthetic target: high detail at center, low detail at periphery.

    This is the regime where ego-motion-radial foveation should win.
    """
    torch.manual_seed(seed)
    yy = torch.arange(h, dtype=torch.float32)
    xx = torch.arange(w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    cy, cx = h / 2.0, w / 2.0
    radial = torch.exp(
        -((grid_y - cy) ** 2 + (grid_x - cx) ** 2) / (2.0 * (min(h, w) / 4.0) ** 2)
    )
    # 3-channel "image" with center detail (high frequency) + periphery
    # smooth (low frequency).
    detail = radial.unsqueeze(0).expand(3, -1, -1)
    smooth = torch.full((3, h, w), 0.5)
    target = detail * 0.5 + smooth * 0.5
    return target.unsqueeze(0)  # (1, 3, H, W)


def _simulate_foveation_quantize(
    target: torch.Tensor,
    fov_map: torch.Tensor,
    bit_budget: float,
    base_attenuation: float = 0.5,
) -> tuple[float, float, float]:
    """Simulate quantization with foveation-modulated per-pixel bit allocation.

    Per the C1 design, per-pixel bit cost := base * (1 + attenuation * (1 - M)).
    Higher M (center) = lower bit cost = more bits allocated there.
    The simulation uniformly allocates bit_budget across the image and
    applies per-pixel quantization with step proportional to inverse bits.

    Returns (residual_l2, bits_used_proxy, concentration_index).
    """
    # Per-pixel bit allocation ~ M_t (normalized to sum to bit_budget).
    weights = fov_map.flatten() + 1e-3  # avoid zero
    weights = weights / weights.sum()
    bits_per_pixel = weights * bit_budget
    # Quantization step ~ 1 / (2 ** bits). For tiny bit counts (<1) we get
    # a coarse step; for high bit counts (>4) we get a fine step.
    step = 1.0 / (2.0 ** bits_per_pixel.clamp(min=0.5))  # (H*W,)
    step_map = step.reshape(*fov_map.shape)  # (B, 1, H, W)
    # Per-pixel residual: round(target / step) * step - target.
    quantized = torch.round(target / step_map) * step_map
    residual = (quantized - target).pow(2).mean().item()

    # Bits used proxy (for diagnostics)
    bits_used = bits_per_pixel.sum().item()

    # Concentration index: 1 - normalized entropy of bit allocation.
    # 0 = uniform; 1 = all bits on one pixel.
    p = weights
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    max_entropy = (torch.log(torch.tensor(weights.numel()))).item()
    concentration = 1.0 - entropy / max(max_entropy, 1e-6)

    return residual, bits_used, concentration


def run_probe(
    image_h: int = 24,
    image_w: int = 32,
    bit_budget: float = 1000.0,
    seed: int = 0,
) -> dict:
    """Run the probe and emit the verdict dict."""
    target = _make_radial_target(image_h, image_w, seed=seed)

    results: dict[str, dict[str, float]] = {}
    for strategy_name, strategy in (
        ("uniform", FoveationStrategy.UNIFORM),
        ("ego_motion_radial", FoveationStrategy.EGO_MOTION_RADIAL),
        ("learned_per_pixel", FoveationStrategy.LEARNED_PER_PIXEL),
    ):
        wm_cfg = WorldModelConfig(
            recurrence_mode=WorldModelRecurrenceMode.GRU,
            latent_dim=8,
            hidden_dim=8,
        )
        cfg = WorldModelFoveationConfig(
            world_model_cfg=wm_cfg,
            foveation_strategy=strategy,
            output_height=image_h,
            output_width=image_w,
            num_pairs=1,
        )
        fov_module = FoveationMapModule(cfg)
        # Random latent for learned-per-pixel (no training; just probe the
        # module's untrained foveation map -- the production trainer trains
        # the learned head end-to-end via the score-aware loss).
        torch.manual_seed(seed)
        z_t = torch.randn(1, wm_cfg.latent_dim)
        with torch.no_grad():
            fov_map = fov_module.map(z_t)

        residual, bits_used, concentration = _simulate_foveation_quantize(
            target, fov_map, bit_budget
        )
        results[strategy_name] = {
            "residual_l2": float(residual),
            "bits_used_proxy": float(bits_used),
            "concentration_index": float(concentration),
        }

    # Verdict: lowest residual_l2 at fixed bit budget wins. The probe's
    # purpose is to test whether geometric or learned foveation
    # outperforms uniform on the radial-target regime.
    by_residual = sorted(
        results.items(), key=lambda kv: kv[1]["residual_l2"]
    )
    best_name, best_metrics = by_residual[0]
    second_metrics = by_residual[1][1]
    margin = (
        (second_metrics["residual_l2"] - best_metrics["residual_l2"])
        / max(second_metrics["residual_l2"], 1e-9)
    )

    if margin < 0.05:
        verdict = "tie"
        rationale = (
            f"Top two within 5%: best={best_name} "
            f"residual={best_metrics['residual_l2']:.6f} vs runner-up "
            f"{by_residual[1][0]} residual={second_metrics['residual_l2']:.6f}; "
            f"margin {margin:.2%}."
        )
    else:
        verdict = best_name
        rationale = (
            f"Best: {best_name} residual={best_metrics['residual_l2']:.6f} "
            f"concentration={best_metrics['concentration_index']:.3f} vs "
            f"runner-up {by_residual[1][0]} "
            f"residual={second_metrics['residual_l2']:.6f}; "
            f"margin {margin:.2%}."
        )

    return {
        **results,
        "verdict": verdict,
        "verdict_rationale": rationale,
        "evidence_grade": "proxy",
        "score_claim_valid": False,
        "score_axis": "proxy_synthetic",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_proxy_synthetic_radial_not_contest_video",
            "no_scorer_load",
            "non_promotable_evidence_grade",
            "untrained_learned_head_per_pixel_baseline",
        ],
        "config": {
            "image_h": image_h,
            "image_w": image_w,
            "bit_budget": bit_budget,
            "seed": seed,
        },
        "lane_id": "lane_c1_world_model_foveation_campaign_l1_scaffold_20260514",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Probe-disambiguator for C1 foveation strategy. Catalog #125 hook #6."
        )
    )
    p.add_argument("--image-h", type=int, default=24)
    p.add_argument("--image-w", type=int, default=32)
    p.add_argument("--bit-budget", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    verdict = run_probe(
        image_h=args.image_h,
        image_w=args.image_w,
        bit_budget=args.bit_budget,
        seed=args.seed,
    )
    out_json = json.dumps(verdict, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json, encoding="utf-8")
        print(f"[c1-probe-fov] wrote {args.output}")
    print(out_json)
    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    sys.exit(main())
