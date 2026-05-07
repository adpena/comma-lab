# WR01 exact-eval operator handoff

Scope: WR01/HNeRV exact-CUDA packet readiness only. This runbook does not claim
a score, claim a lane, or dispatch remote work by itself.

Packet authority:

```bash
.venv/bin/python tools/build_wr01_exact_eval_packet.py \
  --build-release-surface \
  --refresh-static-compliance \
  --json-out experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/wr01_exact_eval_packet.json
```

The packet field `operator_next_steps` is the copy-safe ordered sequence. Run
those steps in order. Step `submit_exact_cuda` is the first remote/GPU action;
all earlier steps are local checks, static packet refresh, or the required
Level-2 lane claim.

The submit step must remain blocked until all three non-static blockers clear:

- `LIGHTNING_*` environment is loaded and verified.
- `.omx/state/active_lane_dispatch_claims.md` has the matching active
  `wr01_apply_pr106x_half` claim for job
  `exact_eval_wr01_apply_pr106x_half_20260506`.
- The operator intentionally runs the packet refresh step containing
  `--operator-approved-exact-cuda`.

Before copying the submit command, run `assert_packet_ready_for_submit` from the
packet. It fails loudly unless `ready_for_submit=true` and the blocker list is
empty.
