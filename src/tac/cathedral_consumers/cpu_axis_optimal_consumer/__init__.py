# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tools/cpu_axis_optimal_archive_selector.py`` (G1 Hotz binding).

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the orphan-signal-at-cathedral-autopilot bug class for the G1
canonical CPU-axis re-ranker per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE
2026-05-19.

The canonical helper ``tools/cpu_axis_optimal_archive_selector.py``
(landed by codex sister) re-ranks existing dual-eval data ON THE
``[contest-CPU]`` AXIS ONLY because the leaderboard ranks by CPU per
CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". The PR102 empirical
+0.033 CUDA-vs-CPU gap means CUDA-best != CPU-best per archive — a
purely re-rank-on-existing-data orphan-closure with **zero GPU cost**.

This consumer surfaces the G1 CPU-axis-min archive (per family) as a
non-promotable observability annotation. Promotion requires a paired
Linux x86_64 ``[contest-CPU]`` anchor per Catalog #192 + the canonical
submission PR gate.

Sister of:
- ``_example_consumer`` (canonical reference template)
- ``tac.cathedral_consumers.score_lagrangian_consumer`` (sister TOP-1
  arbitrariness-extinction consumer)
- ``tac.frontier_scan`` + ``tools/scan_best_anchor_per_axis.py``
  (Catalog #316 sister canonical helpers this consumer cites)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "cpu_axis_optimal_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    The G1 re-rank is purely structural — operates on existing dual-eval
    rows in ``.omx/state/continual_learning_posterior.jsonl`` +
    ``.omx/state/modal_call_id_ledger.jsonl``. As new paired CPU/CUDA
    anchors land, the canonical equation
    ``canonical_frontier_pointer_v1`` (per slot 14 + slot 22 review)
    auto-recalibrates via
    ``tac.canonical_equations.auto_recalibrate_from_continual_learning_posterior``.
    NO-OP here by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 + #6 — cathedral autopilot ranker + probe-disambiguator.

    Returns zero-adjustment observability annotation citing the canonical
    G1 helper. No score adjustment — re-rank discovery is ``[predicted]``
    until a paired Linux x86_64 ``[contest-CPU]`` empirical anchor confirms
    per CLAUDE.md "Submission auth eval" non-negotiable.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tools/cpu_axis_optimal_archive_selector.py canonical helper "
            "available (G1 Hotz binding PROCEED IMMEDIATELY; re-rank existing "
            "dual-eval data on [contest-CPU] axis; predicted ΔS [-0.010, "
            "-0.003] at $0 cost; sister Catalog #316 frontier-scan) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "update_from_anchor",
    "consume_candidate",
]
