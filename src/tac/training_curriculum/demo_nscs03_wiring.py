# SPDX-License-Identifier: MIT
"""Demo wiring: NSCS03 + model soup checkpoint interpolation.

Demonstrates how the canonical pausing-exploits helpers compose with the
NSCS03 end-to-end Ballé joint codec (`experiments/train_substrate_nscs03_end_
to_end_balle_joint_codec.py`). NSCS03 implements Ballé 2018 hyperprior
end-to-end joint codec; the Phase 2 sextet council (per
`feedback_sextet_council_nscs03_phase_2_consensus_20260516.md` Revision #4)
recommends differentiated EMA decay (0.999 for hyperprior, 0.997 for main
codec) which is a NATURAL match for model soup checkpoint interpolation.

Why a separate demo module (not direct trainer edits)?
──────────────────────────────────────────────────────
Same Catalog #230 sister-subagent ownership discipline as
:mod:`tac.training_curriculum.demo_nscs01_wiring`. The trainer is shared;
the demo recipe is the safe wiring template.

NSCS03-specific recipe (per design memo + sextet council Revision #4):
──────────────────────────────────────────────────────────────────────
NSCS03 has 5 sub-networks (g_a / g_s / h_a / h_s / EntropyBottleneck) with
DIFFERENT optimal EMA decay rates. The sextet council Revision #4 deferred
differentiated EMA decay to Phase 2; the model-soup pattern is the cheaper
empirical probe BEFORE committing to differentiated EMA infrastructure.

The canonical model-soup recipe for NSCS03:

1. Train NSCS03 to convergence at λ_R₁ (e.g. 0.01 — low-rate target).
2. Pause; snapshot full state_dict; resume.
3. Re-train (FRESH OPTIMIZER per :class:`StageScheduler` reset policy) at
   λ_R₂ (e.g. 0.05 — mid-rate target). Snapshot.
4. Re-train at λ_R₃ (e.g. 0.1 — high-rate target). Snapshot.
5. **Greedy model soup** over the 3 checkpoints using contest-CPU score as
   the held-out metric.
6. The greedy soup may BEAT any single λ_R because the rate-distortion
   curve is non-convex and the soup interpolates between operating points.

Sister pattern: Wortsman 2022 §3.2 reports this kind of greedy soup beats
the best individual checkpoint on ImageNet by 1-2 points. Whether it
generalizes to NSCS03 + contest scorer is `[would-need-empirical]` and
pending Phase 2 paid smoke.

A1 BOLT-ON-on-A1 hook (T4 SYMPOSIUM Priority 1 BOLT-ON #1):
───────────────────────────────────────────────────────────
Per T4 SYMPOSIUM the FIRST Priority 1 BOLT-ON-on-A1 lane is "Ballé-2018
hyperprior on A1 per-pair latent". NSCS03 IS the end-to-end Ballé joint
codec; the demo wiring below shows how to (a) pause NSCS03 at end of Stage 2
to extract the trained hyperprior state, (b) attach the hyperprior to A1's
per-pair latent via the canonical adapter pattern (substrate-specific; not
landed here — pending the BOLT-ON-#1 design memo).

Apples-to-apples evidence discipline:
- The (λ_R₁, λ_R₂, λ_R₃) tuple below is `[derived]` from Ballé 2018 §5.2
  rate-distortion sweep recommendations. Substrate-specific tuning is
  `[would-need-empirical]`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tac.training_curriculum.model_soup_averaging import (
    GreedyModelSoup,
    ModelSoupResult,
    UniformModelSoup,
)
from tac.training_curriculum.pause_and_diagnose import (
    DiagnosticCheckpoint,
    pause_and_capture,
)


@dataclass(frozen=True)
class NSCS03SoupRecipe:
    """Operator-readable recipe for NSCS03 model-soup checkpoint interpolation.

    Args:
        lambda_r_grid: Tuple of rate-distortion λ_R targets to train. Default
            ``(0.01, 0.05, 0.10)`` per Ballé 2018 §5.2.
        soup_kind: ``"uniform"`` (simple 3-way average) or ``"greedy"``
            (Wortsman §3.2 greedy filter). Default ``"greedy"``.
        held_out_axis: One of ``"contest-CPU"`` / ``"contest-CUDA"``.
            Default ``"contest-CPU"`` per A1 anchor pattern.
    """

    lambda_r_grid: tuple[float, ...] = (0.01, 0.05, 0.10)
    soup_kind: str = "greedy"
    held_out_axis: str = "contest-CPU"

    def __post_init__(self) -> None:
        if not self.lambda_r_grid:
            raise ValueError("lambda_r_grid must be non-empty")
        for lr in self.lambda_r_grid:
            if lr <= 0:
                raise ValueError(
                    f"lambda_r={lr} must be > 0 (rate-distortion λ_R must "
                    "be positive)"
                )
        if self.soup_kind not in {"uniform", "greedy"}:
            raise ValueError(
                f"soup_kind={self.soup_kind!r} not in {{'uniform', 'greedy'}}"
            )
        if self.held_out_axis not in {"contest-CPU", "contest-CUDA"}:
            raise ValueError(
                f"held_out_axis={self.held_out_axis!r} not in canonical set"
            )


def apply_nscs03_soup_recipe(
    *,
    recipe: NSCS03SoupRecipe,
    per_lambda_checkpoints: dict[float, dict[str, torch.Tensor]],
    held_out_metric_fn: (
        Callable[[dict[str, torch.Tensor]], float] | None
    ) = None,
) -> ModelSoupResult:
    """Apply the recipe-defined soup over per-λ_R checkpoints.

    Args:
        recipe: :class:`NSCS03SoupRecipe`.
        per_lambda_checkpoints: Dict of ``{λ_R: state_dict}``; keys MUST
            cover ``recipe.lambda_r_grid``.
        held_out_metric_fn: REQUIRED if ``recipe.soup_kind == "greedy"``;
            takes a state_dict and returns a scalar metric (lower is
            better per contest scorer semantics).

    Returns:
        :class:`tac.training_curriculum.model_soup_averaging.ModelSoupResult`.

    Raises:
        :class:`ValueError` on recipe / checkpoint mismatch.
    """
    if set(per_lambda_checkpoints.keys()) != set(recipe.lambda_r_grid):
        raise ValueError(
            f"per_lambda_checkpoints keys {sorted(per_lambda_checkpoints.keys())} "
            f"!= recipe.lambda_r_grid {list(recipe.lambda_r_grid)}"
        )
    # Stringify keys for the soup helper (it operates on string keys).
    stringified = {
        f"lambda_r_{lr:.4f}": ckpt
        for lr, ckpt in per_lambda_checkpoints.items()
    }
    if recipe.soup_kind == "uniform":
        return UniformModelSoup()(stringified)
    # greedy
    if held_out_metric_fn is None:
        raise ValueError(
            "recipe.soup_kind='greedy' requires held_out_metric_fn"
        )
    return GreedyModelSoup()(
        stringified,
        held_out_metric_fn=held_out_metric_fn,
        minimize=True,  # contest score: lower is better
    )


def nscs03_per_subnet_l2_norms(
    state_dict: dict[str, torch.Tensor],
    sub_net_prefixes: tuple[str, ...] = (
        "g_a", "g_s", "h_a", "h_s", "entropy_bottleneck",
    ),
) -> dict[str, float]:
    """Compute L2 norm per NSCS03 sub-network.

    Useful as a diagnostic metric at pause-and-capture time; differentiated
    EMA decay would predict different norm trajectories per sub-network.
    """
    norms: dict[str, float] = {}
    for prefix in sub_net_prefixes:
        sq = torch.tensor(0.0)
        any_match = False
        for k, v in state_dict.items():
            if k.startswith(prefix) and v.is_floating_point():
                any_match = True
                sq = sq + (v.detach() ** 2).sum()
        norms[prefix] = float(sq.sqrt()) if any_match else 0.0
    return norms
