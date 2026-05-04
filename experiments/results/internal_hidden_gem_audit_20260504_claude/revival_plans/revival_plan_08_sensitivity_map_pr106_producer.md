# Revival plan: Lane Sensitivity-PR106: foundational producer artifact

**Gem**: `src/tac/sensitivity_map.py`
**ID**: `08_sensitivity_map_pr106_producer`

## Current state

Level-4 (β-Fisher 1.016 LANDED [contest-CUDA T4]). This is a producer artifact — it doesn't itself score, but is required input for water-fill, IMP, sensitivity-weighted hyperprior.

## Files touched

- experiments/build_sensitivity_map_for_pr106.py (new)
- src/tac/sensitivity_map.py (no changes)
- experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map.pt (new artifact)

## Integration sketch

1. Load PR106 HNeRV decoder via standard PR106 inflate path.
2. Compute β-Fisher: for each parameter θ_i, F_ii = E[(∂score/∂θ_i)^2] over contest video pairs.
3. Save sensitivity_map.pt with per-tensor sensitivities.
4. Document downstream consumers: water-fill v3, IMP-V2, sensitivity-weighted Ballé.

## Test plan

- Compute sensitivity map for at least the conv layers; sanity check non-uniform distribution.
- Verify integration with water-filling v2 codec via dry-run.
- Verify integration with IMP iterative pruning.

## Predicted score basis

Itself does not score. Downstream gain: when used with water-fill v3 (revival #1) → predicted -0.005 to -0.015. When used as IMP guide → marginal. When used as Ballé hyperprior input → predicted -0.005 to -0.001.

## What would change my mind

If β-Fisher is computationally infeasible on PR106 decoder (memory blow-up), use saliency-map approximation instead.

## Blockers resolved in plan

- Needs CUDA forward — deferred.

## Skunkworks council deliberation

Shannon/Dykstra/Fridrich/Selfcomp UNANIMOUS endorse (foundational producer). Treat as 1-hour priority-1 task.

**Verdict**: VOTE 10/10 GO. Foundational dependency for top revival lanes.
