# PR85 Pair-Action Lowering Worker

## Contract

- tool: `experiments/build_pr85_pair_action_candidates.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: false
- ready_for_exact_eval_after_lane_claim_count: 0
- blocker_class: `missing_pair_action_evidence`

## Source Evidence

- source archive bytes: 236328
- source archive sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- scorer plan: `/tmp/pr85_scorer_gradient_atoms_plan_baseline.json`
- scorer plan sha256: `367a4205efa6304712b9466519bd59168122b278ad784a69fc269a76fce567ff`
- action evidence: `None`

## Implementation Decision

The lowering surface emits pair-action candidate specs only when a grounded action evidence file provides stream, source value, candidate value, charged-byte proxy, and source-artifact custody. The current PR85 scorer-gradient/pair-atom artifacts provide ranked pairs and break-even bytes but no stream/value direction, so the default output records blocked unlowered specs instead of inventing candidate actions.

## Candidate Specs

- pr85_pair_0192_unlowered: status=blocked pairs=[192] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0060_unlowered: status=blocked pairs=[60] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0164_unlowered: status=blocked pairs=[164] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0197_unlowered: status=blocked pairs=[197] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0070_unlowered: status=blocked pairs=[70] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0496_unlowered: status=blocked pairs=[496] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0106_unlowered: status=blocked pairs=[106] blocker=missing_explicit_pair_action exact_ready=false
- pr85_pair_0522_unlowered: status=blocked pairs=[522] blocker=missing_explicit_pair_action exact_ready=false

## Blockers

- missing_pair_action_evidence: no grounded pair-action evidence JSON was supplied
- missing_explicit_pair_action: scorer-gradient pair ranking has no explicit stream/value delta, non-noop proof, or archive-changing runtime path

## Command Output

- command: `experiments/build_pr85_pair_action_candidates.py --out-json /tmp/pr85_pair_action_candidates_worker_20260504/candidate_specs.json --ledger .omx/research/pr85_pair_action_lowering_worker_20260504.md`
- emitted candidate_count: 8
- ready_for_pair_atom_archive_build_count: 0
- dispatch_unlocked_count: 0

## Exact Next Action

Generate measured, grounded pair-action evidence for at least one ranked pair, then run this lowering tool with `--action-evidence-json`. If it emits a non-noop `pair_atom_action_spec`, feed that spec plus a reviewed runtime contract into `experiments/build_pr85_pair_atom_candidates.py`; claim the lane before any exact CUDA auth eval.
