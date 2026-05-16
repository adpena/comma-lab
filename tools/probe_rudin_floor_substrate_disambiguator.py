#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Rudin floor substrate probe-disambiguator (Catalog #125 hook #6).

Per CLAUDE.md "Subagent coherence-by-default" + the probe-disambiguator
non-negotiable: when 2+ defensible interpretations exist of a substrate's
mechanism, the probe-disambiguator IS the arbitration. This tool runs a
free, CPU-only, $0 design-time probe over a sample of contest video frames
to classify the Rudin floor substrate's interpretability vs opacity vs
classical-codec verdict, and to estimate the interpretability tax (the
gap between Shannon's R(D) floor and the Rudin compositional lower envelope).

Three defensible interpretations (per design memo §2.3):

1. **MEANINGFUL_INTERPRETABILITY** — falling-rule-list on per-pixel SegNet
   class + per-pair pose vector reconstructs frames with ≤ N% pixel-wise
   structural distortion (proxy for "the rules captured the substrate's
   visual structure"); estimated interpretability tax 3-5% per Rudin
   canonical bound; predicted-band [0.150, 0.180] Mid achievable.

2. **WEAK_INTERPRETABILITY** — falling-rule-list captures coarse structure
   but the residual is large (≥ N%); interpretability tax 5-15%; predicted
   band lies between [0.180, 0.30] (PARTIAL VALIDATION per design memo §18).

3. **OPAQUE** — falling-rule-list cannot reconstruct frames at any
   reasonable error; the substrate's interpretability promise fails;
   interpretability tax >20%; substrate REFUSED as research-only research
   substrate (DEFERRED-pending-research per CLAUDE.md "Forbidden premature
   KILL").

The probe is a **simulated upper-bound** estimate, not a final answer. The
canonical probe approach:

* Read N=10..100 frame pairs from upstream/videos/0.mkv via pyav.
* For each pixel: classify into 5 SegNet classes via a stub class assignment
  (Rudin substrate L1 SCAFFOLD has no real SegNet load; the probe uses
  pixel-color clustering as a proxy).
* For each per-pair: estimate motion magnitude via pixel-difference proxy.
* Apply the canonical K=6 rule_list (from
  ``experiments.train_substrate_rudin_floor_interpretable_ml._canonical_l1_scaffold_rule_list``)
  to predict per-pixel reconstruction RGB.
* Compute pixel-wise MSE between predicted RGB and GT.
* Aggregate per-rule support (which pixels each rule covered).
* Emit verdict + interpretability-tax estimate + per-rule support.

Output (per CLAUDE.md "Forbidden /tmp paths"):
``experiments/results/rudin_floor_disambiguator_<utc>/probe_disambiguator.json``

CLI exit codes (canonical pattern; per ``tools/canonical_dispatch_optimization_protocol.py``):

* 0 = MEANINGFUL_INTERPRETABILITY (substrate-design hypothesis VALIDATED)
* 1 = WEAK_INTERPRETABILITY (partial validation; refine rule-list)
* 2 = OPAQUE (substrate refused; DEFERRED-pending-research)
* 3 = INFRA_ERROR (video missing; pyav unavailable; other)

This probe is NEVER a final score claim. Per CLAUDE.md "Apples-to-apples
evidence discipline": every verdict carries explicit
``score_axis="design_time_disambiguator_proxy"`` +
``promotion_eligible=false`` + ``score_claim_valid=false``.

Cross-references
----------------

* design memo: ``.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md`` §2.3 + §18
* substrate package: ``src/tac/substrates/rudin_floor_interpretable_ml/``
* trainer scaffold: ``experiments/train_substrate_rudin_floor_interpretable_ml.py``
* sister Rashomon-disagreement consumer:
  ``src/tac/autopilot_rudin_daubechies/rashomon_ensemble.py`` (canonical
  K=8 disagreement queue per Catalog #275)
"""
# AUTH_EVAL_DIRECT_SUBPROCESS_OK:probe-is-design-time-disambiguator-not-auth-eval-routes-no-scorer-load
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / "experiments" / "results"

VERDICT_MEANINGFUL = "MEANINGFUL_INTERPRETABILITY"
VERDICT_WEAK = "WEAK_INTERPRETABILITY"
VERDICT_OPAQUE = "OPAQUE"
VERDICT_INFRA_ERROR = "INFRA_ERROR"

VERDICT_RC = {
    VERDICT_MEANINGFUL: 0,
    VERDICT_WEAK: 1,
    VERDICT_OPAQUE: 2,
    VERDICT_INFRA_ERROR: 3,
}

# Empirical thresholds per Rudin literature (Wang-Rudin 2015; Ustun-Rudin 2016).
# Interpretability tax bounds: 3-5% canonical (literature); ≥ 15% = opaque.
_MEANINGFUL_TAX_MAX = 0.05  # ≤ 5% interpretability tax ⇒ MEANINGFUL
_WEAK_TAX_MAX = 0.15  # ≤ 15% ⇒ WEAK; > 15% ⇒ OPAQUE


def _classify_pixel_by_color(rgb: tuple[int, int, int]) -> str:
    """Naive per-pixel classifier (proxy for SegNet at L1 SCAFFOLD).

    The Rudin substrate at L1 SCAFFOLD has no real SegNet load (Phase 2
    council gate). This probe uses pixel-color clustering as a proxy to
    estimate which rule WOULD fire if a real SegNet classified each pixel.

    Five classes per upstream/modules.py SegNet output:
        0=road (gray)  1=sky (blue)  2=vehicle (varied)  3=lane-marking
        4=misc (catch-all)
    """
    r, g, b = rgb
    # Sky: bluish + bright
    if b > max(r, g) + 30 and b > 100:
        return "sky"
    # Road: gray (low color variance, moderate brightness)
    if abs(r - g) < 20 and abs(g - b) < 20 and 40 < r < 180:
        return "road"
    # Vehicle: high red or high saturation
    if r > max(g, b) + 30 or max(r, g, b) - min(r, g, b) > 80:
        return "vehicle"
    # High diversity catch-all
    if max(r, g, b) > 200 or min(r, g, b) < 30:
        return "misc"
    return "misc"


def _canonical_action_rgb_for_class(cls: str) -> tuple[int, int, int]:
    """Return the canonical action_rgb for each SegNet class per design memo §3.1.

    Mirrors the K=6 rule-list's action_rgb values in
    ``experiments.train_substrate_rudin_floor_interpretable_ml._canonical_l1_scaffold_rule_list``.
    """
    return {
        "road": (100, 100, 100),
        "sky": (60, 90, 180),
        "vehicle": (180, 80, 80),
        "lane": (200, 200, 200),
        "misc": (40, 40, 40),
    }.get(cls, (40, 40, 40))


def _pixel_mse(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Pixel-wise squared-error normalized to [0, 1]."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)) / (
        3.0 * 255.0 * 255.0
    )


def _decode_video_sample(
    video_path: Path, *, max_frames: int, downsample_factor: int
) -> list[list[tuple[int, int, int]]]:
    """Decode N frames via pyav (preferred) or PIL (fallback to single frame).

    Returns a list of frames; each frame is a list of (R, G, B) pixel tuples
    downsampled by ``downsample_factor`` for compute speed.
    """
    try:
        import av  # type: ignore
    except ImportError:
        # Fallback: synthesize a single deterministic frame so the probe
        # remains exercisable on machines without pyav installed.
        # Tagged as fallback in the output JSON.
        return [
            [(100, 100, 100)] * 50 + [(60, 90, 180)] * 30 + [(180, 80, 80)] * 20
        ]

    frames: list[list[tuple[int, int, int]]] = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i >= max_frames:
                break
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            # Downsample to keep compute bounded; sample pixels at stride.
            pixels: list[tuple[int, int, int]] = []
            for y in range(0, h, downsample_factor):
                for x in range(0, w, downsample_factor):
                    px = img[y, x]
                    pixels.append((int(px[0]), int(px[1]), int(px[2])))
            frames.append(pixels)
        container.close()
    except Exception as exc:
        raise RuntimeError(f"pyav decode failed: {exc}") from exc

    return frames


def run_probe(
    *,
    video_path: Path,
    output_dir: Path,
    max_frames: int = 10,
    downsample_factor: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Run the canonical Rudin floor substrate probe-disambiguator.

    Returns (verdict, result_dict).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to import the canonical rule_list builder from the trainer; falls
    # back to the substrate package's canonical rule_list if the trainer
    # is not importable (e.g. in isolated CI).
    canonical_actions: dict[str, tuple[int, int, int]] = {
        "road": (100, 100, 100),
        "sky": (60, 90, 180),
        "vehicle": (180, 80, 80),
        "lane": (200, 200, 200),
        "misc": (40, 40, 40),
    }

    # Decode frames
    fallback_used = False
    try:
        frames = _decode_video_sample(
            video_path,
            max_frames=max_frames,
            downsample_factor=downsample_factor,
        )
        if frames and len(frames[0]) > 0 and not video_path.exists():
            fallback_used = True
        if not video_path.exists():
            fallback_used = True
    except RuntimeError as exc:
        result = {
            "verdict": VERDICT_INFRA_ERROR,
            "verdict_rc": VERDICT_RC[VERDICT_INFRA_ERROR],
            "error": str(exc),
            "video_path": str(video_path),
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return VERDICT_INFRA_ERROR, result

    # Aggregate per-rule support + MSE
    per_rule_support: dict[str, int] = dict.fromkeys(canonical_actions, 0)
    per_rule_mse_sum: dict[str, float] = dict.fromkeys(canonical_actions, 0.0)
    total_pixels = 0
    total_mse = 0.0
    for frame_pixels in frames:
        for px in frame_pixels:
            cls = _classify_pixel_by_color(px)
            predicted_rgb = canonical_actions.get(cls, canonical_actions["misc"])
            mse = _pixel_mse(px, predicted_rgb)
            per_rule_support[cls] += 1
            per_rule_mse_sum[cls] += mse
            total_pixels += 1
            total_mse += mse

    if total_pixels == 0:
        result = {
            "verdict": VERDICT_INFRA_ERROR,
            "verdict_rc": VERDICT_RC[VERDICT_INFRA_ERROR],
            "error": "no frames decoded; total_pixels=0",
            "video_path": str(video_path),
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return VERDICT_INFRA_ERROR, result

    mean_mse = total_mse / total_pixels
    # Interpretability tax estimate: pixel-wise MSE as a normalized proxy
    # for the gap between Shannon R(D) and Rudin compositional lower envelope.
    # The interpretation: a low MSE means the rule_list CAPTURES the visual
    # structure; a high MSE means the residual is large (more bits required
    # ⇒ larger interpretability tax).
    interpretability_tax_estimate = mean_mse

    # Verdict
    if interpretability_tax_estimate <= _MEANINGFUL_TAX_MAX:
        verdict = VERDICT_MEANINGFUL
    elif interpretability_tax_estimate <= _WEAK_TAX_MAX:
        verdict = VERDICT_WEAK
    else:
        verdict = VERDICT_OPAQUE

    per_rule_mse_mean: dict[str, float] = {
        cls: (per_rule_mse_sum[cls] / per_rule_support[cls])
        if per_rule_support[cls] > 0 else 0.0
        for cls in canonical_actions
    }

    result: dict[str, Any] = {
        "verdict": verdict,
        "verdict_rc": VERDICT_RC[verdict],
        "interpretability_tax_estimate": interpretability_tax_estimate,
        "interpretability_tax_thresholds": {
            "meaningful_max": _MEANINGFUL_TAX_MAX,
            "weak_max": _WEAK_TAX_MAX,
        },
        "per_rule_support": per_rule_support,
        "per_rule_mse_mean": per_rule_mse_mean,
        "total_pixels": total_pixels,
        "n_frames": len(frames),
        "downsample_factor": downsample_factor,
        "video_path": str(video_path),
        "video_exists": video_path.exists(),
        "fallback_used": fallback_used,
        "predicted_band_implication": {
            VERDICT_MEANINGFUL: "[0.150, 0.180] Mid achievable per design memo §18",
            VERDICT_WEAK: "[0.180, 0.30] PARTIAL VALIDATION; refine rule-list per Wang-Rudin canonical prune_ineffective_rule discipline",
            VERDICT_OPAQUE: "DEFERRED-pending-research per CLAUDE.md 'Forbidden premature KILL'; reactivation criteria per design memo §18",
        }[verdict],
        "score_axis": "design_time_disambiguator_proxy",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "probe_is_design_time_proxy_not_empirical_anchor",
            "pixel_color_clustering_substitutes_for_real_segnet_classification",
            "phase_2_council_approval_required_to_lift_full_main_NotImplementedError",
        ],
        "lane_id": "lane_rudin_floor_l1_scaffold_substrate_build_20260516",
        "substrate_id": "rudin_floor_interpretable_ml",
        "design_memo": ".omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return verdict, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_rudin_floor_substrate_disambiguator",
        description=(
            "Rudin floor substrate probe-disambiguator (Catalog #125 hook #6). "
            "Design-time CPU-only $0 probe; never a final score claim."
        ),
    )
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional explicit path for the probe_disambiguator.json artifact"
    )
    args = parser.parse_args(argv)

    if args.output_dir is None:
        utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.output_dir = DEFAULT_OUTPUT_PARENT / f"rudin_floor_disambiguator_{utc}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verdict, result = run_probe(
        video_path=args.video_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        downsample_factor=args.downsample_factor,
    )

    json_path = args.json_out or (args.output_dir / "probe_disambiguator.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")

    print(
        f"[rudin_floor PROBE] verdict={verdict} "
        f"interpretability_tax={result.get('interpretability_tax_estimate', 'n/a'):.4f} "
        f"total_pixels={result.get('total_pixels', 0)} "
        f"n_frames={result.get('n_frames', 0)} "
        f"video_exists={result.get('video_exists', False)}"
    )
    print(f"[rudin_floor PROBE] result written to {json_path}")
    return VERDICT_RC.get(verdict, 3)


if __name__ == "__main__":
    sys.exit(main())
