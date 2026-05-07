# Frontier roadmap post-floor refresh - 2026-05-07

Scope: local-only roadmap/status refresh after the active PR103-on-PR106 A++
rate-only floor guard. No GPU dispatch, no lane claim, and no score claim.

## Command

```text
.venv/bin/python tools/build_frontier_roadmap_status.py
  --json-out experiments/results/frontier_roadmap_status_post_floor_20260507_codex/status.json
  --md-out experiments/results/frontier_roadmap_status_post_floor_20260507_codex/status.md
  --operator-approved-exact-cuda
```

## Result

Artifacts:

- `experiments/results/frontier_roadmap_status_post_floor_20260507_codex/status.json`
- `experiments/results/frontier_roadmap_status_post_floor_20260507_codex/status.md`

The roadmap has `13` frontier rows and `0` dirty-blocked rows. The next
unblocked keys are:

- `categorical_qma9_clade_spade_openpilot`
- `joint_admm_balle_arithmetic_stack`
- `hnerv_per_tensor_context_entropy`
- `telescopic_foveation_field`
- `lapose_motion_atom_allocator`

Default HNeRV rate-only packets are not dispatch targets:

- `pr106_q10_151byte_brotli`: blocked by
  `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`
- `pr106x_lgblock16_1byte_brotli`: blocked by the same floor
- `wr01_apply_pr106x_half`: still strict-preflight refused

The generated `selected_candidate_packet` remains
`pr106_q10_151byte_brotli`, but its `selection_decision` is
`rate_only_candidate_above_active_pr103_pr106_floor`; this is an operator
status row, not dispatch authorization.

## Next Routing Decision

The next score-lowering tranche should not spend CUDA on above-floor rate-only
packets. It should focus on scorer-changing or parity-unblocking work:

1. PR91/HPM1 categorical semantic decode/re-encode parity.
2. Joint ADMM/Balle/arithmetic runtime consumption.
3. HNeRV entropy overhead reduction until a rate-only packet beats `185578`
   bytes, or a packet explicitly changes scorer behavior.
4. Telescopic/LA-pose/RAFT geometry only after a charged runtime consumer and
   component-risk gate exist.
