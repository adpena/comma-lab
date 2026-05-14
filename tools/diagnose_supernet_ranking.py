#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnose a DARTS-SuperNet search provenance: which sub-architecture won?

Reads the provenance JSON produced by
``experiments/search_time_traveler_supernet.py`` (or directly by
``tac.composition.darts_supernet.write_provenance``) and prints:

  - Per-axis convergence verdict (KL nats vs uniform).
  - Per-axis softmax distribution (which candidate(s) the search prefers).
  - Top-k architecture ranking with byte+score estimates.
  - Cross-checks against the time-traveler memo §7 prediction
    (54 KB / 0.16-0.17 contest-CPU).
  - Substrate-engineering routing hint: which existing substrate lane is
    the best match for the discovered architecture.

Per CLAUDE.md "Score-claim discipline": the diagnose tool refuses any
provenance that carries ``score_claim=True``; the proxy SuperNet's
predictions are NOT authoritative.

Usage:
    .venv/bin/python tools/diagnose_supernet_ranking.py \\
        reports/raw/<timestamp>-darts-supernet-time-traveler/supernet_search_provenance.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.composition.darts_supernet import (  # noqa: E402
    SuperNetError,
    default_search_axes,
    load_provenance,
    reasonable_candidate_value,
)


# Time-traveler memo §7 prediction band (contest-CPU, realistic case).
_TT_PREDICTION_MIN = 0.150
_TT_PREDICTION_MAX = 0.180
_PR101_CONTEST_CUDA = 0.193
_PR101_CONTEST_CPU = None  # not authoritatively measured
_PUBLIC_GOLD_CONTEST_CPU = 0.195  # rem2 silver / EthanYangTW bronze cluster

# Existing substrate lane routing hints. Keys are (world_model_size,
# decoder_hidden_dim, quant_mode) tuples; values are best-match lane ids.
_SUBSTRATE_ROUTING_HINTS = {
    ("wm_25k", "hid_32", "ternary"): "lane_substrate_world_residual_predictive_coding (DESIGN)",
    ("wm_25k", "hid_32", "fp4"): "lane_quantizr_inspired_88k_fp4 (existing 0.33)",
    ("wm_50k", "hid_64", "fp4"): "lane_pr101_baseline_229k_fp4 (existing 0.193)",
    ("wm_50k", "hid_64", "int8"): "lane_pr101_baseline_229k_int8 (untested)",
    ("wm_75k", "hid_96", "fp4"): "lane_pr106_r2_film_pose_renderer (existing)",
    ("wm_100k", "hid_128", "fp4"): "lane_hnerv_family_bigger (DESIGN)",
}


def _verdict_label(kl_nats: float) -> str:
    if kl_nats >= 2.0:
        return "decisive"
    if kl_nats >= 1.0:
        return "moderate"
    return "inconclusive"


def _format_softmax_row(probs: Sequence[float], names: Sequence[str]) -> str:
    """One-line bar chart of softmax probs."""
    if not probs:
        return ""
    cells = []
    for p, n in zip(probs, names):
        bar_len = int(20.0 * p)
        bar = "#" * bar_len + "·" * (20 - bar_len)
        cells.append(f"{n:>10s} {bar} {100.0 * p:5.1f}%")
    return "\n          ".join(cells)


def _print_axis_diagnosis(
    axis: str,
    discovered_name: str,
    discovered_value: object,
    kl_nats: float,
    softmax_final: Sequence[float],
    candidate_names: Sequence[str],
) -> None:
    verdict = _verdict_label(kl_nats)
    print(f"  axis: {axis}")
    print(f"    discovered_value = {discovered_value}  ({discovered_name})")
    print(f"    KL nats to uniform = {kl_nats:.3f} [{verdict}]")
    print(f"    in_canonical_band = {reasonable_candidate_value(axis, discovered_value)}")
    print(f"    softmax distribution:")
    bar = _format_softmax_row(softmax_final, candidate_names)
    if bar:
        print(f"          {bar}")
    print()


def _compare_to_memo_prediction(final_score: float) -> str:
    """Return a one-line verdict comparing to time-traveler memo §7."""
    if final_score < _TT_PREDICTION_MIN:
        return (
            f"BELOW the time-traveler optimistic case "
            f"({_TT_PREDICTION_MIN:.3f}); proxy predicts the architecture "
            f"can beat the memo's realistic-case band. Strong DARTS signal."
        )
    if final_score <= _TT_PREDICTION_MAX:
        return (
            f"WITHIN the time-traveler memo §7 realistic band "
            f"[{_TT_PREDICTION_MIN:.3f}, {_TT_PREDICTION_MAX:.3f}]. "
            f"Search confirms the memo's prediction."
        )
    if final_score <= _PR101_CONTEST_CUDA:
        return (
            f"BETWEEN the time-traveler memo band and PR101's "
            f"{_PR101_CONTEST_CUDA:.3f}. Search did not fully commit; "
            f"may need more steps or sharper anneal."
        )
    return (
        f"ABOVE PR101 ({_PR101_CONTEST_CUDA:.3f}). The search did not "
        f"find a competitive architecture under the seeded prior. "
        f"Inspect the seeded anchors in default_search_axes()."
    )


def _routing_hint(arch: Mapping[str, str]) -> str:
    """Map a discovered architecture to its best-match substrate lane."""
    key = (
        arch.get("world_model_size", ""),
        arch.get("decoder_hidden_dim", ""),
        arch.get("quant_mode", ""),
    )
    if key in _SUBSTRATE_ROUTING_HINTS:
        return _SUBSTRATE_ROUTING_HINTS[key]
    # Fuzzy fallback: ternary + small → likely points at world-residual lane.
    if arch.get("quant_mode") == "ternary" and arch.get("world_model_size") == "wm_25k":
        return "lane_substrate_world_residual_predictive_coding (DESIGN; closest match)"
    return "no exact match; consider new substrate lane based on discovered axes"


def diagnose(provenance: Mapping[str, object]) -> int:
    print("=" * 72)
    print(" DARTS-SuperNet provenance diagnosis")
    print("=" * 72)
    print()
    print(f"  total_steps: {provenance.get('total_steps')}")
    print(f"  score_claim: {provenance.get('score_claim')}  (must be False)")
    print(f"  evidence_grade: {provenance.get('evidence_grade')}")
    print(f"  source_memos: {provenance.get('source_memos')}")
    print()

    final_score = float(provenance["final_score_predicted"])
    print(f"  final_score_predicted: {final_score:.4f}")
    print(f"  PR101 anchor:          {_PR101_CONTEST_CUDA:.4f} [contest-CUDA]")
    print(f"  Public gold cluster:   {_PUBLIC_GOLD_CONTEST_CPU:.4f} [contest-CPU]")
    print(f"  Time-traveler band:    [{_TT_PREDICTION_MIN:.3f}, {_TT_PREDICTION_MAX:.3f}] [time-traveler-prediction]")
    print()
    print("  Memo §7 comparison:")
    print(f"    {_compare_to_memo_prediction(final_score)}")
    print()

    # Per-axis diagnosis.
    print("-" * 72)
    print(" Per-axis convergence:")
    print("-" * 72)
    print()
    cfg = default_search_axes()
    discovered = provenance["discovered"]
    discovered_values = provenance["discovered_values"]
    kl_per_axis = provenance["kl_nats_per_axis"]
    softmax_per_axis = provenance["softmax_final_per_axis"]
    if not isinstance(discovered, dict):
        raise SuperNetError("provenance.discovered must be a dict")
    if not isinstance(discovered_values, dict):
        raise SuperNetError("provenance.discovered_values must be a dict")
    if not isinstance(kl_per_axis, dict):
        raise SuperNetError("provenance.kl_nats_per_axis must be a dict")
    if not isinstance(softmax_per_axis, dict):
        raise SuperNetError("provenance.softmax_final_per_axis must be a dict")
    for axis_name in cfg.axis_names():
        if axis_name not in discovered:
            raise SuperNetError(f"axis {axis_name} missing from discovered")
        _print_axis_diagnosis(
            axis=axis_name,
            discovered_name=str(discovered[axis_name]),
            discovered_value=discovered_values[axis_name],
            kl_nats=float(kl_per_axis[axis_name]),
            softmax_final=softmax_per_axis[axis_name],
            candidate_names=cfg.candidate_names(axis_name),
        )

    # Top-k routing hints.
    print("-" * 72)
    print(" Top-k architectures + substrate routing hints:")
    print("-" * 72)
    print()
    ranked = provenance.get("ranked_top_k", [])
    if not isinstance(ranked, list):
        raise SuperNetError("provenance.ranked_top_k must be a list")
    for i, entry in enumerate(ranked):
        if not isinstance(entry, dict):
            continue
        s = float(entry.get("predicted_score", 0.0))
        arch = entry.get("architecture", {})
        if not isinstance(arch, dict):
            continue
        arch_str = {str(k): str(v) for k, v in arch.items()}
        hint = _routing_hint(arch_str)
        print(f"  #{i + 1}  predicted_score={s:.4f}")
        for k, v in arch.items():
            print(f"         {k} = {v}")
        print(f"         best-match substrate lane: {hint}")
        print()

    # Operator-routable decisions.
    print("=" * 72)
    print(" Operator-routable decisions:")
    print("=" * 72)
    decisive = sum(1 for k in kl_per_axis.values() if float(k) >= 2.0)
    moderate = sum(1 for k in kl_per_axis.values() if 1.0 <= float(k) < 2.0)
    inconclusive = sum(1 for k in kl_per_axis.values() if float(k) < 1.0)
    print(f"  decisive axes:     {decisive}/5")
    print(f"  moderate axes:     {moderate}/5")
    print(f"  inconclusive axes: {inconclusive}/5")
    print()
    if inconclusive >= 3:
        print("  RECOMMENDATION: re-run with more steps (try --total-steps 1000)")
        print("    or sharper anneal (T_end=0.05) to drive α to commit. The")
        print("    current search has not committed to a clear architecture.")
    elif decisive >= 3:
        print("  RECOMMENDATION: dispatch the top-1 architecture as a smoke")
        print("    canary on macOS-CPU first ($0), then Modal A100 ($3-8) per")
        print("    CLAUDE.md smoke-before-full pattern (Catalog #167).")
        print()
        print("  Build a new substrate lane in src/tac/substrates/<lane_id>/")
        print("  per CLAUDE.md HNeRV parity discipline (13 lessons + Catalog")
        print("  #124 archive-grammar-at-design-time gate).")
    else:
        print("  RECOMMENDATION: the search is partially converged. Consider")
        print("    one of:")
        print("    - Re-run with seed sweep (3-5 seeds) to assess stability")
        print("    - Increase --total-steps to 1000-3000")
        print("    - Refine the AxisOp anchor seeds in default_search_axes()")
        print("      using empirical anchors from the continual-learning")
        print("      posterior (.omx/state/cost_band_posterior.jsonl)")
    print()
    print("  All deliverables remain DESIGN-SPACE SIGNAL only. No score")
    print("  claim, no promotion eligibility, no dispatch readiness until")
    print("  paired [contest-CUDA] + [contest-CPU] anchors land on the")
    print("  chosen substrate per CLAUDE.md \"Submission auth eval — BOTH")
    print("  CPU AND CUDA\" non-negotiable.")
    print()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "provenance",
        type=Path,
        help=(
            "Path to the provenance JSON produced by "
            "experiments/search_time_traveler_supernet.py."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.provenance.exists():
        print(f"ERROR: provenance file does not exist: {args.provenance}", file=sys.stderr)
        return 2
    try:
        provenance = load_provenance(args.provenance)
    except SuperNetError as exc:
        print(f"ERROR: provenance validation failed: {exc}", file=sys.stderr)
        return 3
    return diagnose(provenance)


if __name__ == "__main__":
    sys.exit(main())
