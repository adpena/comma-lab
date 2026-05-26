# Codex Findings: Byte-Range Stage Inputs Queue

## Verdict

The multisurface operation-chain compiler now carries `byte_range_entropy_recode_v1`
past advisory stage planning into a queue-owned local proof chain. This is still
not score, promotion, rank/kill, dispatch, or correction-budget-spend authority.

## Landed Change

- Added `frontier_rate_attack_byte_range_stage_inputs.v1`, which binds the
  payload-grammar stage to concrete PR103 schema, beam/global-combo probes,
  source runtime, source archive, member name, and byte-range chain output path.
- Added `tools/build_frontier_byte_range_stage_inputs.py`.
- Extended `build_frontier_operation_chain_compiler_queue(...)` so each
  multisurface chain experiment emits:
  1. operation chain stage plan;
  2. byte-range stage input packet;
  3. local `tools/run_byte_range_entropy_recode_chain.py` execution when the
     local context is complete.
- Added correction-budget semantics: realized rate savings are explicitly routed
  only to component-guarded SegNet/PoseNet correction planning, and spend remains
  blocked until receiver proof, exact-readiness bridge, component eval, and total
  Lagrangian improvement gates pass.

## Empirical Anchor

Executed the generated queue
`frontier_feedback_cycle_byte_range_stage_inputs_chain_compiler` locally with
three bounded steps:

- stage plan emitted;
- byte-range stage inputs emitted;
- byte-range entropy-recode chain emitted.

Smoke summary:

- source archive bytes: `178223`;
- candidate archive bytes: `178207`;
- realized saved bytes: `16`;
- receiver contract satisfied: `true`;
- exact/score authority: `false`;
- remaining blockers:
  - `candidate_inflate_output_parity_missing`;
  - `strict_pre_submission_compliance_json_missing`;
  - `lane_dispatch_claim_missing`;
  - `exact_cuda_auth_eval_missing`;
  - `full_frame_render_output_parity_missing`;
  - `shell_inflate_output_parity_missing`.

Durable artifacts:

- `.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_stage_inputs/initial_refresh/operation_chain_compiler_queue.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_stage_inputs/initial_refresh/byte_range_stage_inputs_smoke.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_stage_inputs/initial_refresh/byte_range_chain_manifest_smoke.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_stage_inputs/initial_refresh/byte_range_stage_inputs_smoke_summary.json`

## No-Signal-Loss Hooks

- Byte-range context is now queue-visible rather than operator hand-carried.
- Receiver proof remains distinct from full-frame/shell parity and exact eval.
- Rate savings are exposed as repair budget only through fail-closed policy.
- The queue validates and executes locally without cloud or score authority.

## Next Work

The next highest-EV closure is to feed the emitted chain manifest into
materializer chain harvest and exact-readiness bridge generation, then route
receiver-closed saved bytes into the targeted component correction queue only
after shell/full-frame parity and static submission closure are present.
