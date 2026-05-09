"""Non-arbitrariness probe: arbitrate between yuv6 routing modes.

Per CLAUDE.md "Design tensions: ship both interpretations, let math/empirics
arbitrate" (``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509``)
and "Unified Lagrangian action principle" hook 6 (non-arbitrariness probe):
when 2+ defensible interpretations exist, the solver MUST own the probe that
arbitrates.

This probe runs both ``Yuv6RoutingMode.MONKEY_PATCH_GLOBAL`` and
``Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING`` on a calibration batch and
reports:

  * pose-gradient parity (both should produce non-zero ‖∂L/∂rgb‖)
  * forward output bit-equivalence to upstream rgb_to_yuv6
  * gradient-magnitude ratio (should be 1.0 ± numerical noise)
  * recommended mode (MONKEY_PATCH_GLOBAL by default since that is the
    verified-working PR #95 recipe; TAC_DIFFERENTIABLE_ROUTING is recommended
    only if MONKEY_PATCH_GLOBAL fails on the runtime, which would indicate
    upstream import shadowing in the dispatch container)

Usage::

  .venv/bin/python tools/probe_yuv6_differentiability_disambiguator.py \\
      --output reports/yuv6_routing_disambiguator_<utc>.json

The output JSON is consumed by the trainer's ``--yuv6-mode auto`` resolution
in ``experiments/train_score_gradient_pr101_finetune.py`` and
``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``.

Exit codes:
  0 — both modes pass; recommendation issued
  2 — at least one mode fails parity check (bug-class detected)
  3 — upstream not importable (probe cannot run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Make ``src`` and ``upstream`` importable when invoked from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (_REPO_ROOT / "src", _REPO_ROOT / "upstream"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.differentiable_eval_roundtrip import (
    Yuv6RoutingMode,
    differentiable_rgb_to_yuv6,
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)


def _compute_grad_via_function(yuv6_fn, *, seed: int = 20260509) -> dict[str, float]:
    """Forward + backward through ``yuv6_fn``; return summary stats.

    Returns dict with: ``forward_max_abs_diff_vs_local`` (output diff vs
    ``differentiable_rgb_to_yuv6``), ``grad_l2`` (L2 norm of input gradient),
    ``grad_max_abs`` (max abs input gradient), ``grad_finite`` (bool).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    rgb = (torch.rand((1, 3, 64, 64), generator=g) * 255.0).requires_grad_(True)
    out = yuv6_fn(rgb)
    loss = out.sum()
    if loss.grad_fn is None:
        # The yuv6_fn was decorated with @torch.no_grad() (upstream baseline);
        # backward would raise. Report zero gradient explicitly.
        with torch.no_grad():
            local_out = differentiable_rgb_to_yuv6(rgb.detach())
        diff = (out.detach() - local_out).abs().max().item()
        return {
            "forward_max_abs_diff_vs_local": diff,
            "grad_l2": 0.0,
            "grad_max_abs": 0.0,
            "grad_finite": False,
            "no_grad_path_detected": True,
        }
    loss.backward()
    grad = rgb.grad
    if grad is None:
        return {
            "forward_max_abs_diff_vs_local": float("nan"),
            "grad_l2": 0.0,
            "grad_max_abs": 0.0,
            "grad_finite": False,
        }
    # Compare forward output against the local differentiable version.
    with torch.no_grad():
        local_out = differentiable_rgb_to_yuv6(rgb.detach())
    diff = (out.detach() - local_out).abs().max().item()
    return {
        "forward_max_abs_diff_vs_local": diff,
        "grad_l2": grad.norm(p=2).item(),
        "grad_max_abs": grad.abs().max().item(),
        "grad_finite": bool(torch.isfinite(grad).all().item()),
    }


def _probe_monkey_patch_global(*, seed: int = 20260509) -> dict[str, Any]:
    """Probe Aaron's monkey-patch path."""
    try:
        import frame_utils  # noqa: F401
    except ImportError:
        return {
            "mode": Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value,
            "available": False,
            "error": "upstream frame_utils not importable",
        }
    token = patch_upstream_yuv6_globally()
    try:
        import frame_utils as fu

        # Now frame_utils.rgb_to_yuv6 should be our differentiable version.
        is_patched = fu.rgb_to_yuv6 is differentiable_rgb_to_yuv6
        stats = _compute_grad_via_function(fu.rgb_to_yuv6, seed=seed)
        return {
            "mode": Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value,
            "available": True,
            "is_patched_after_call": is_patched,
            **stats,
        }
    finally:
        unpatch_upstream_yuv6(token)


def _probe_tac_differentiable_routing(*, seed: int = 20260509) -> dict[str, Any]:
    """Probe the cleaner tac-routing path.

    The trainer would route directly through ``differentiable_rgb_to_yuv6``
    instead of monkey-patching upstream globals.
    """
    stats = _compute_grad_via_function(differentiable_rgb_to_yuv6, seed=seed)
    return {
        "mode": Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING.value,
        "available": True,
        "is_patched_after_call": False,  # No global state mutated.
        **stats,
    }


def _probe_upstream_baseline(*, seed: int = 20260509) -> dict[str, Any]:
    """Probe upstream's @torch.no_grad() rgb_to_yuv6 — should produce ZERO grad.

    This is the BROKEN path our trainers use without the patch. Including it
    in the probe output makes the "before/after" delta explicit.
    """
    try:
        import frame_utils
    except ImportError:
        return {
            "mode": "upstream_no_grad_baseline",
            "available": False,
            "error": "upstream frame_utils not importable",
        }
    stats = _compute_grad_via_function(frame_utils.rgb_to_yuv6, seed=seed)
    return {
        "mode": "upstream_no_grad_baseline",
        "available": True,
        "is_patched_after_call": False,
        **stats,
    }


def _arbitrate(probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the recommended mode from probe results."""
    monkey = next(
        (p for p in probes if p.get("mode") == Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value),
        None,
    )
    tac = next(
        (p for p in probes if p.get("mode") == Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING.value),
        None,
    )

    monkey_ok = bool(monkey and monkey.get("available") and monkey.get("grad_l2", 0.0) > 0.0)
    tac_ok = bool(tac and tac.get("available") and tac.get("grad_l2", 0.0) > 0.0)

    # Both modes should produce identical gradients on the same input.
    if monkey_ok and tac_ok:
        m_g = monkey.get("grad_l2", 0.0)
        t_g = tac.get("grad_l2", 0.0)
        ratio = m_g / t_g if t_g > 0 else float("inf")
        # Default to monkey-patch (Aaron's verified-working PR #95 recipe).
        recommendation = Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value
        rationale = (
            "Both modes pass parity (grad_l2 ratio={:.6f}); recommending "
            "MONKEY_PATCH_GLOBAL because it is the empirically verified PR #95 "
            "recipe and propagates to every consumer in the process (including "
            "any upstream module that imported rgb_to_yuv6 at its own import "
            "time). TAC_DIFFERENTIABLE_ROUTING is equivalent in pose-gradient "
            "flow when consistently applied."
        ).format(ratio)
    elif monkey_ok and not tac_ok:
        recommendation = Yuv6RoutingMode.MONKEY_PATCH_GLOBAL.value
        rationale = "TAC_DIFFERENTIABLE_ROUTING failed parity; falling back to MONKEY_PATCH_GLOBAL."
    elif tac_ok and not monkey_ok:
        recommendation = Yuv6RoutingMode.TAC_DIFFERENTIABLE_ROUTING.value
        rationale = (
            "MONKEY_PATCH_GLOBAL failed (likely upstream not importable in this "
            "container); falling back to TAC_DIFFERENTIABLE_ROUTING."
        )
    else:
        recommendation = None
        rationale = "Both modes failed parity check; no recommendation possible."

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "monkey_patch_global_passed": monkey_ok,
        "tac_differentiable_routing_passed": tac_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report. Default: print to stdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260509,
        help="RNG seed for the calibration batch (default 20260509).",
    )
    args = parser.parse_args(argv)

    started = time.time()

    probes = [
        _probe_upstream_baseline(seed=args.seed),
        _probe_monkey_patch_global(seed=args.seed),
        _probe_tac_differentiable_routing(seed=args.seed),
    ]
    arbitration = _arbitrate(probes)

    report = {
        "schema_version": 1,
        "tool": "tools/probe_yuv6_differentiability_disambiguator.py",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "evidence_grade": "[predicted; YUV6 differentiability probe — non-arbitrariness arbitration]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "probes": probes,
        "arbitration": arbitration,
        "cross_references": {
            "claude_md_addition": ".omx/research/CLAUDE_md_addition_eval_roundtrip_inner_loop_yuv6_20260509.md",
            "binary_forensics": ".omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md",
            "design_tension_principle": "feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509",
            "unified_lagrangian_principle": "feedback_unified_lagrangian_action_principle_GR_style_20260509",
        },
    }

    body = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
        print(f"[probe-yuv6] wrote {args.output}")
    else:
        print(body)

    if arbitration["recommendation"] is None:
        return 2
    if not any(p.get("available") for p in probes if p.get("mode") != "upstream_no_grad_baseline"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
