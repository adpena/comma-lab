#!/usr/bin/env python3
"""Lane 17 — IMP 10-cycle iterative magnitude pruning orchestrator.

Per Phase 3 Lane 17 spec (memory project_phases_2_3_4_*) and Round 5 Council E
battleplan §5.1 ACCELERATE rank #4: orchestrator that drives 10 IMP cycles
in process. Each cycle: train_epoch → prune lowest-X% magnitude → rewind
survivors → repeat. Final output: sparse renderer with most weights = 0
(high entropy savings via sparse-CSR archive).

This file is the IN-PROCESS orchestrator complementing the existing
single-cycle helper ``experiments/train_imp_cycle.py`` and the remote-shell
bash wrapper ``scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh``.

Use the orchestrator when:
- You want to run all N cycles in one Python process (no shell loop / no
  intermediate-file serialisation between cycles)
- You want the cycle-loop to be unit-testable (this scaffold provides the
  test harness without dispatching real training)
- You want a deterministic sparsity tracker that fails fast on regression
  (sparsity must monotonically increase across cycles)

Use the existing ``experiments/train_imp_cycle.py`` when:
- You need to checkpoint between cycles for fault tolerance on remote GPU
- The shell wrapper is the canonical dispatcher for Lane J-IMP

CLAUDE.md compliance
--------------------
- Compress-time only (training); inflate path uses the sparse-CSR codec
  from tac.iterative_magnitude_pruning unchanged.
- No silent defaults — every CLI flag has explicit None or required behaviour
- All claims tagged [synthetic] / [prediction]
- No GPU dependency for the orchestrator itself; the train_step callback
  determines the device
- No scorer load (training uses a caller-provided train_step_fn)

Math foundation
---------------
Iterative Magnitude Pruning (Frankle & Carbin 2019, "The Lottery Ticket
Hypothesis"). After N cycles with per-cycle sparsity_increment p:
    cumulative_sparsity = 1 - (1 - p)^N

For p=0.20, N=10: cumulative ≈ 89.3% sparsity. The surviving 10.7% of
weights re-organise across cycles to match dense baseline performance
(within 0.05 score on classification; renderer behavior is empirical-pending).

Predicted Phase 2/3 EV: 30 KB renderer payload at <0.05 score regression
(kill criterion). [prediction] only — empirical confirm needed on a real
Lane G v3 anchor.

References
----------
* Frankle & Carbin 2019 "The Lottery Ticket Hypothesis"
* memory: project_phases_2_3_4_design_implementation_math_provenance §"Lane 17"
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# Path setup — match qat_finetune.py + train_distill.py conventions.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))


from tac.iterative_magnitude_pruning import (  # noqa: E402
    IMPState,
    apply_mask_to_model,
    compute_actual_sparsity,
    prune_lowest_magnitude,
    rewind_weights_to_early_epoch,
    snapshot_state_dict,
)


@dataclass
class CycleResult:
    """Per-cycle bookkeeping returned by run_imp_cycles."""

    cycle: int
    sparsity_after_prune: float
    expected_sparsity: float
    train_loss_after: float
    weight_count_total: int
    weight_count_kept: int


@dataclass
class ImpRunResult:
    """Aggregate result from run_imp_cycles."""

    final_state: IMPState
    cycle_results: list[CycleResult]
    monotone_sparsity: bool
    """True iff sparsity monotonically increased across cycles (sanity
    invariant: lottery-ticket pruning never RESURRECTS pruned weights)."""


def run_imp_cycles(
    model: nn.Module | None = None,
    *,
    num_cycles: int,
    sparsity_increment: float,
    train_step_fn: Callable[[nn.Module, int], float] | None = None,
    early_epoch_state: dict[str, torch.Tensor] | None = None,
    rewind_after_prune: bool = True,
) -> ImpRunResult:
    """Run N IMP cycles in process; returns aggregated CycleResults.

    Each cycle:
      1. prune_lowest_magnitude (sparsity_increment of CURRENTLY surviving
         weights pruned)
      2. apply_mask_to_model (zero pruned tensors)
      3. (optional) rewind_weights_to_early_epoch (Frankle-2019 stabilisation)
      4. train_step_fn(model, cycle_idx) — caller's train-one-epoch callback
      5. record CycleResult

    Args:
        model: nn.Module to prune. Required.
        num_cycles: number of IMP cycles to run. Required.
        sparsity_increment: per-cycle fraction to prune. Required (no silent
            default — Check 81 STRICT). Typical: 0.20.
        train_step_fn: callable (model, cycle_idx) -> train_loss. Required;
            no default. The orchestrator does NOT make assumptions about
            data, optimizer, or device — those live in the callback.
        early_epoch_state: state_dict snapshot for rewind. If None and
            rewind_after_prune=True, snapshot is taken from model BEFORE
            cycle 0 (Frankle-2019 alternative: rewind to init).
        rewind_after_prune: whether to apply the rewind step. Default True
            (canonical IMP). If False, weights survive pruning at their
            current values (more aggressive; converges faster but is less
            faithful to the lottery-ticket protocol).

    Returns:
        ImpRunResult with per-cycle stats + monotonicity sanity check.

    Raises:
        ValueError: bad inputs / pre-conditions.
    """
    if model is None:
        raise ValueError(
            "run_imp_cycles: model is required (no silent default — "
            "Check 81 STRICT)."
        )
    if num_cycles is None or num_cycles < 1:
        raise ValueError(
            f"run_imp_cycles: num_cycles must be >= 1; got {num_cycles}"
        )
    if sparsity_increment is None or not (0.0 < sparsity_increment < 1.0):
        raise ValueError(
            f"run_imp_cycles: sparsity_increment must be in (0, 1); "
            f"got {sparsity_increment}"
        )
    if train_step_fn is None:
        raise ValueError(
            "run_imp_cycles: train_step_fn is required (no silent default — "
            "the orchestrator does NOT assume an optimizer or dataloader)."
        )

    # Initialise / load early-epoch snapshot
    if early_epoch_state is None:
        early_epoch_state = snapshot_state_dict(model)

    state = IMPState(
        sparsity_target=1.0 - (1.0 - sparsity_increment) ** num_cycles,
        sparsity_increment=float(sparsity_increment),
        mask={},
        early_epoch_weights=early_epoch_state,
    )

    cycle_results: list[CycleResult] = []
    prev_sparsity = 0.0
    monotone = True

    for cycle_idx in range(num_cycles):
        # 1. Prune lowest magnitude (additive; updates state.mask)
        new_mask = prune_lowest_magnitude(
            model,
            sparsity_increment=sparsity_increment,
            current_mask=state.mask if state.mask else None,
        )
        state.mask = new_mask

        # 2. Apply mask (zero pruned weights)
        apply_mask_to_model(model, state.mask)

        # 3. Optionally rewind survivors to early-epoch snapshot
        if rewind_after_prune and state.early_epoch_weights:
            rewind_weights_to_early_epoch(
                model,
                early_epoch_weights=state.early_epoch_weights,
                mask=state.mask,
            )

        # 4. Caller's train-step callback (one epoch / one window of training)
        train_loss = float(train_step_fn(model, cycle_idx))

        # 5. Record cycle stats
        actual_sparsity = compute_actual_sparsity(model, state.mask)
        # Compute kept/total directly from state.mask
        total = 0
        kept = 0
        for _name, m in state.mask.items():
            total += int(m.numel())
            kept += int(m.sum().item())
        if actual_sparsity + 1e-9 < prev_sparsity:
            # Sparsity went down — masks resurrected weights, broken invariant
            monotone = False
        prev_sparsity = actual_sparsity
        cycle_results.append(
            CycleResult(
                cycle=int(cycle_idx),
                sparsity_after_prune=float(actual_sparsity),
                expected_sparsity=float(state.expected_sparsity_after_cycle(cycle_idx)),
                train_loss_after=float(train_loss),
                weight_count_total=int(total),
                weight_count_kept=int(kept),
            )
        )

        state.cycle_count = int(cycle_idx) + 1

    return ImpRunResult(
        final_state=state,
        cycle_results=cycle_results,
        monotone_sparsity=monotone,
    )


# ── CLI entry point (used by integration tests / dispatch scripts) ─────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI entry — for dispatch scripts that prefer a one-liner."""
    p = argparse.ArgumentParser(
        description="Lane 17 — IMP 10-cycle orchestrator (in-process)"
    )
    p.add_argument(
        "--num-cycles",
        type=int,
        required=True,
        help="Number of IMP cycles to run (Lane 17 default: 10).",
    )
    p.add_argument(
        "--sparsity-increment",
        type=float,
        required=True,
        help="Per-cycle prune fraction (Lane 17 typical: 0.20).",
    )
    p.add_argument(
        "--no-rewind",
        action="store_true",
        help="Skip the early-epoch rewind step (less faithful to "
        "Frankle-2019; converges faster).",
    )
    p.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Where to write the JSON manifest of CycleResults.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI main — fail-fast: this scaffold has NO default model loader.

    The CLI is provided so dispatch scripts can wire model + train_step_fn
    in a separate Python file and call ``run_imp_cycles`` directly.
    Standalone CLI use raises (no silent default model).
    """
    args = _parse_args(argv)
    raise NotImplementedError(
        "imp_cycle_runner.py CLI is a SCAFFOLD; it has no default model "
        "loader (Check 81 STRICT — no silent defaults). Use run_imp_cycles "
        "directly from a dispatcher that wires model + train_step_fn. "
        f"args={vars(args)}"
    )


def _write_manifest(result: ImpRunResult, path: str | Path) -> None:
    """Helper for dispatchers that want a JSON record of the cycle stats."""
    payload = {
        "cycle_results": [
            {
                "cycle": cr.cycle,
                "sparsity_after_prune": cr.sparsity_after_prune,
                "expected_sparsity": cr.expected_sparsity,
                "train_loss_after": cr.train_loss_after,
                "weight_count_total": cr.weight_count_total,
                "weight_count_kept": cr.weight_count_kept,
            }
            for cr in result.cycle_results
        ],
        "final_cycle_count": result.final_state.cycle_count,
        "final_sparsity_target": result.final_state.sparsity_target,
        "monotone_sparsity": result.monotone_sparsity,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


__all__ = [
    "CycleResult",
    "ImpRunResult",
    "run_imp_cycles",
    "_write_manifest",
]


if __name__ == "__main__":
    sys.exit(main())
