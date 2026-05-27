<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Codex audit of repair-waterfill and autonomous-chain queue integration after Cascade C' full-frame runtime fix. -->

# Codex findings: repair-waterfill queue integration audit

**UTC:** 2026-05-27T00:05Z  
**Agent:** Codex  
**Primary artifact root:** `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z`

## Verdict

The repair/waterfill layer is implemented, queue-owned, and wired into the
autonomous many-op chain in the latest complete reference artifact. It is not a
score authority or dispatch authority surface, and it should not be described
as theoretically optimal yet. It is an executable local planning/actuation
surface that now has a parent autonomous queue, child repair queue, generated
work orders, materialization plans, materializer binding reports, and execution
audits.

## Queue evidence

Commands run:

```bash
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/repair_budget_waterfill_queue.json validate
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/autonomous_chain_optimization_queue.json validate
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/repair_budget_waterfill_queue.json status
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/autonomous_chain_optimization_queue.json status
```

Results:

- `repair_budget_waterfill_queue.json`
  - schema: `experiment_queue.v1`
  - queue id: `frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z_repair_budget_waterfill`
  - experiments: `3`
  - steps: `12`
  - validation: `valid=true`
  - state status: `12 succeeded`, `0 ready`, `0 orphaned`
- `autonomous_chain_optimization_queue.json`
  - schema: `experiment_queue.v1`
  - queue id: `frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z_autonomous_chain_optimization`
  - experiments: `3`
  - steps: `9`
  - validation: `valid=true`
  - state status: `9 succeeded`, `0 ready`, `0 orphaned`
- `autonomous_chain_optimization.json`
  - schema: `frontier_rate_attack_autonomous_chain_optimization.v1`
  - chain count: `3`
  - top chains:
    - `global_many_op_rate_distortion_receiver_campaign`
    - `portfolio_materializer_context_closure_campaign`
    - `segnet_posenet_geometry_drop_many_campaign`
  - target classes: `archive_section`, `byte_range`, `inverse_scorer`, `packet_member`, `tensor`
  - child queue action: `repair_budget_waterfill_queue`
- `receiver_closed_correction_budget.json`
  - schema: `frontier_rate_attack_receiver_closed_correction_budget.v1`
  - receiver-closed saved bytes total: `258`

## Action-functional integration status

The queue uses the existing action-functional lineage rather than spawning a
parallel theory layer:

- upstream rate-budget preservation action functional is referenced;
- operator action ledger terms are carried forward;
- repair allocation action terms are generated per accepted component-response row;
- objective is explicit:
  `minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes`;
- `new_parallel_action_functional_created=false`.

## Repair dynamics prior

The empirical K=16 palette signal is present in the queue/reference surface via
the repair-dynamics palette prior and materializer-binding commands:

- `none`
- `frame0_blue_chroma_amp_1`
- `frame0_blue_chroma_amp_3`
- `frame0_luma_bias_+1`
- `frame0_luma_bias_-1`
- `frame0_luma_bias_-2`
- `frame0_luma_bias_-4`
- `frame0_rgb_bias_m2_p1_p1`
- `frame0_rgb_bias_m4_p2_p2`
- `frame0_rgb_bias_p0_m1_p1`
- `frame0_rgb_bias_p0_m2_p2`
- `frame0_rgb_bias_p0_p1_m1`
- `frame0_rgb_bias_p0_p2_m2`
- `frame0_rgb_bias_p2_m1_m1`
- `frame0_rgb_bias_p4_m2_m2`
- `frame0_roll_dx+0_dy+1`

This matches the operator's observation: 15 of 16 modes are frame-0 modes, one
is identity, and there are zero frame-1 modes in this palette. That is useful
planning signal, not proof that frame-1 repair is bad.

## Authority posture

All audited surfaces remain fail-closed:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `budget_spend_allowed=false`
- local MLX/CPU waterfill is not score, dispatch, promotion, or kill authority

## Remaining non-optimal gaps

The current queue is coherent and executable, but not a theoretical optimum:

- exact-axis component response is still required before budget spend;
- receiver-runtime materialized repair candidates must be produced before any
  repair archive can be exact-eval eligible;
- Cascade C' now satisfies the full-frame raw-size contract, but its receiver
  renderer is still a scaffold until actual frame-0/frame-1 lookup and warp
  replay are wired;
- the K=16 frame-0-heavy palette probably leaves frame-1/P18/P19 repair
  opportunity unmeasured;
- Cascade C per-region selector codec remains a structural opportunity until
  selector payload bytes and receiver replay proof are landed.

## Next concrete action

Run the next MLX-local repair-dynamics tranche as a measured component-response
producer: PoseNet-null bottom-decile pairs, SegNet class-region waterfill,
frame-1 perturbation probes, and per-region selector payload estimates. Feed
those rows back into this exact repair-waterfill queue so the parent autonomous
chain can choose against the existing action-functional ledger rather than
operator intuition.
