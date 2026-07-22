---
schema: dag_feed.v1
date_utc: 2026-07-22T14:20:00Z
feed_id: FEED-DDM-STRUCTURED-CARRIERS-LAW-REGISTRATION-20260722
task: 540
master_task: 578
feeds_tasks: [603, 613]
lane_id: ddm_structured_carriers_law_registration
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# FEED - DDM structured-carriers measured bracket

## Dependency and result

`v3 bulk-only -> v4/v5 structured roles -> v6 rate amortization -> v7 exact-value rate wall -> v8 sparse-pixel ERF wall -> v9 structured in-box base`

The first six stages are historical inputs and remain settled. This landing registers their
measurement-line consolidation as `ddm_describe_line_rate_distortion_bracket_v1` with three active
anchor legs:

1. `v7_value_space_exactness`: evaluator-green and rate-dead at `43,112,153/171,332,654 B`.
2. `v8_sparse_posthoc_pixel_correction`: `4%-6%` selected sites and approximately `94%` byte
   collapse do not remove the `d_seg approximately 0.026-0.029` floor; FORMULATION scope.
3. `v9_structured_per_stratum_carrier`: `51,668/72,397 B` enters the rate box while
   `d_seg approximately 0.040-0.045`, `d_pose approximately 158`, and zero G2CS1 symbols leave the
   correction solve owed.

## Consumer trajectory

`ddm_describe_line_rate_distortion_bracket_v1`

`-> tac.optimization.v10_constructive_solver`

`-> joint Fisher-margin/curvature-ranked G2CS1 coefficients + xi birth/death events`

`-> tac.optimization.direct_description_entropy_priced_member (#613 exact-byte knee)`

`-> tac.witness_control.costate_organ_v2 (rate/distortion opportunity routing)`

`-> n600 receiver-closed empirical row -> contest CPU/CUDA only after normal authority gates`

The bracket's lower point is derived from measured v9 n64/n256 bytes; any n600 interpolation remains
labeled DERIVED until a receipt lands. The upper point is measured v7 exactness. No unseen curve
shape, score improvement, or promotion is inferred between them.

## Triality and no-orphan disposition

- DSL: N-A with rationale. No new launch lever or config value is authorized.
- DAG: this FEED routes the registered law into v10, #613, and the costate organ.
- Equations: typed callable + three SHA-bound anchors + append-only registry event.
- Sensitivity/Pareto/bit allocator: consume existing Fisher/margin, corrected inner-Jacobian,
  curvelet/shearlet, xi, and reverse-waterfill laws; do not create a second ranker.
- Cathedral/autopilot: the law exposes one measured bracket and the exact remaining DOF; it does not
  create a dispatch candidate.
- Continual learning: receipt verdicts and formulation scopes are machine-readable in the registry.
- Probe disambiguator: the future measured arbitration is G2CS1 coefficients alone versus
  G2CS1+xi events; arbitrary pixel values are structurally excluded by the v9 receiver.

No anti-pattern row is emitted because v8 remains FORMULATION-scoped. Optional Brenier companding
and Splay/Jones/MTF coding stay measurement entrants, not rate laws.

## Exit and pointer

This FEED is complete when the equation round-trips through Catalog #344, all nine receipt hashes
rederive, focused and existing registry tests are green, the branch is serializer-committed, and
MAIN reviews the landing.

`0.1910828242 [contest-CPU]` remains unchanged. No launch, scorer run, new measurement, or paid
dispatch occurred.

## STORES CONSULTED

STORES CONSULTED: delegated authority; governing manuals/specs; all ten DDM equation drafts; v7,
v8, and v9 receipts/memos/FEEDs; SegNet ERF factorization; Brenier and Splay crosswalks; canonical
equation/anti-pattern source and registries; current pointer/lane/task/subagent state; both inboxes.

