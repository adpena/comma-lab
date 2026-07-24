# DDM WS3 window completion and arbitration — build spec

## Objective

Complete the exact four-step W_seg and W_joint advisory histories required by
`ddm_ws1_warm_start_slope_falsifier_v1`, arbitrate exactly one warm start,
reseal through the landed J7 path, and run one bounded governed re-smoke. MAIN
retains campaign firing authority.

## Narrow code change

The existing `component_safe_exact_n600` opening path stops after the first
receiver-realized component rejection, even though its typed contract says
shrink and exact rollback. Repair that instance/class mismatch without changing
the campaign objective:

- add a typed `campaign_component_safe_exact_n600` policy requiring all of:
  exact priced joint delta below zero, component safety, and cumulative residual
  fire-gate safety;
- add a typed `seg_lexicographic_proxy_then_exact_component_gate` proposal
  ordering for W_seg;
- enumerate quarter-quantum-first rungs, rank the locally measured candidates
  seg-lexicographically, and continue to the next proposal/rung after an exact
  receiver-realized rejection;
- never admit a reformed opening that has no receiver-visible integer change.

The local ordering is only a proposal-source ranker. Exact n600
receiver-through-R Seg/Pose/rate and the unchanged campaign gates remain the
admission authority.

## Inputs and boundaries

- SHA-bound W_seg archive:
  `264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9`,
  138,031 bytes.
- SHA-bound W_joint archive:
  `5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e`,
  138,801 bytes.
- Reuse `tools/launch_ddm_joint_descent.py` and
  `tools/reseal_ddm_j7_366_ticket.py`; do not fork either path.
- Bounded runs only, Torch threads pinned to 4, governor and source-bound memory
  admission fail closed, SSD-first outputs.
- `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, pointer
  `0.1910828242 [contest-CPU]` unmoved.
- Do not build the E4-WS1 adapter and do not fire the campaign.

## Acceptance

1. `ruff check` is clean for changed Python surfaces.
2. Targeted typed-config, launcher, resealer, and slope-arbitration tests pass.
3. W_seg either reaches exact global step 4 under the reformed ladder or emits
   a formulation-scoped fail-closed receipt with the exhausted decomposition.
4. W_joint has exact receiver verdicts at steps `[0,1,2,3,4]`; no proxy fill.
5. The registered falsifier reports its measured ratio and one selected start,
   or an exact preregistration-permitted refusal.
6. The selected start is resealed by the landed resealer, passes governed
   dry-run plus full-run memory preflight, and receives one bounded exact
   re-smoke.
7. Findings, DAG feed, canonical-equations note, directive-consumption table,
   three clean passes, and `main_review_required=true` are committed on this
  isolated branch.
