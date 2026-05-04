# PR85 Pair-Action Lowering Worker

## Contract

- tool: `experiments/build_pr85_pair_action_candidates.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: false
- ready_for_exact_eval_after_lane_claim_count: 0
- blocker_class: `no_archive_changing_path`

## Source Evidence

- source archive bytes: 236328
- source archive sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- scorer plan: `/tmp/pr85_scorer_gradient_atoms_plan_baseline.json`
- scorer plan sha256: `367a4205efa6304712b9466519bd59168122b278ad784a69fc269a76fce567ff`
- action evidence: `experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json`

## Implementation Decision

The lowering surface emits pair-action candidate specs only when a grounded action evidence file provides stream, source value, candidate value, charged-byte proxy, and source-artifact custody. The current PR85 scorer-gradient/pair-atom artifacts provide ranked pairs and break-even bytes but no stream/value direction, so the default output records blocked unlowered specs instead of inventing candidate actions.

## Candidate Specs

- pr85_qrgb_f2_randglobal_pair_0192: status=action_spec_emitted pairs=[192] blocker=no_archive_changing_path exact_ready=false
- pr85_qrgb_f1_bias_pair_0060: status=action_spec_emitted pairs=[60] blocker=no_archive_changing_path exact_ready=false
- pr85_qrgb_f1_bias_pair_0164: status=action_spec_emitted pairs=[164] blocker=no_archive_changing_path exact_ready=false
- pr85_qrgb_f1_region_pair_0197: status=action_spec_emitted pairs=[197] blocker=no_archive_changing_path exact_ready=false

## Blockers

- no_archive_changing_path: no built archive-changing path was supplied

## Command Output

- command: `experiments/build_pr85_pair_action_candidates.py --pair-atom-planning-json experiments/results/pr85_pair_atom_candidates_20260504_orchestrator/planning.json --action-evidence-json experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json --out-json experiments/results/pr85_qrgb_transfer_actions_20260504_orchestrator/pair_action_candidate_specs.json --ledger .omx/research/pr85_qrgb_pair_action_lowering_20260504_codex.md`
- emitted candidate_count: 4
- ready_for_pair_atom_archive_build_count: 4
- dispatch_unlocked_count: 0

## Exact Next Action

Generate measured, grounded pair-action evidence for at least one ranked pair, then run this lowering tool with `--action-evidence-json`. If it emits a non-noop `pair_atom_action_spec`, feed that spec plus a reviewed runtime contract into `experiments/build_pr85_pair_atom_candidates.py`; claim the lane before any exact CUDA auth eval.
