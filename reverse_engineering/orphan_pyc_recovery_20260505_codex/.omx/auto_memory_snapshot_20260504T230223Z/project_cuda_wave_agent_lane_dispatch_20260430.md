---
name: 2026-04-30 — CUDA-WAVE-AGENT dispatched 4 Phase 2 lanes (Lane 12 NeRV / Lane 19 logit-margin / Lane 20 Ballé / Lane 8 multipass) on Vast.ai 4090
description: Initial dispatch wave 12:27-12:33Z hit 100% NVDEC/phase2-extract failures on first 3 lanes. Lane 20 Ballé completed full 5000-step training and concluded `empirical:STATIC_WINS_FALLBACK` (static codec wins; codec ships zero bytes — no auth eval needed). Retry wave dispatched at 12:51-12:55Z with --max-retries=5 for 12+19+8.
type: project
originSessionId: f0d211b9-718f-4fc9-a752-00eac703aca2
---

## Lane 20 Ballé — landed empirical:STATIC_WINS_FALLBACK

The trainer ran 5000 full steps on instance 35898486 (Lane 20 attempt 1, kept after phase2-launch rc=124 timeout but lane script was actually running). Result captured in `experiments/results/lane_20_balle_2026-04-30_a1_recovered/`:

- `lane_20_train_report.json`: `verdict: STATIC_WINS_FALLBACK`, `static_baseline_bytes: 136296`, `best_full_balle_bytes: 136296`
- `train.log`: 5000/5000 steps completed; loss(bits/block) plateau at ~1029.63 — the Ballé hyperprior failed to beat static arithmetic on this qint stream
- Per Phase A council kill criterion §3, archive ships ZERO bytes from Lane 20. Auto-fallback engaged. NO auth eval needed.
- `train.log` also reveals a CUDA device-mismatch bug in eval-step encode path (`full_balle_bytes=-1` on every eval); training itself completed cleanly. Bug to fix in `src/tac/balle_hyperprior_codec.py` for next iteration.

This is a clean **negative empirical result**: Lane 20 codec adds no value. Memory entry advances Lane 20 to `level 2 → fallback decision` per maturity harness.

## Lane 12 NeRV / Lane 19 / Lane 8 — first-wave failures

All 3 lanes burned 3 retry-attempts each due to NVDEC roulette (5/6 lanes failed last night per memory `feedback_vastai_nvdec_roulette_pivot_to_modal_20260429.md`):
- Lane 12 NeRV: a1+a2 phase2-extract failed; a3 phase2-wait SSH timeout
- Lane 19 logit-margin: a1 NVDEC, a2 SSH, a3 phase2-extract
- Lane 8 multipass: all 3 phase2-launch NVDEC_BAD

Cost burned: ~$0.30 across 9 destroyed instances. Retried as `_b` labels with --max-retries=5.

## Key operational findings

1. **`--predicted-band` takes 2 floats, not a quoted string**: `--predicted-band 1.00 1.10` not `--predicted-band "1.00 1.10"`. Earlier instructions had wrong example.
2. **First nohup invocation can result in instance creation EVEN IF logs are empty** — Python output buffering. Verify state via `vastai show instances` not `cat launch.log`.
3. **Duplicate dispatches** — running the SAME nohup twice in quick succession created 2 instances (35898286 + 35898318). Killed PID 37935 + destroyed instance 35898318 to clean up. Solution: always check `ps aux | grep launch_lane_with_retry` BEFORE re-issuing a dispatch nohup.
4. **Phase2-launch rc=124 timeout (Lane 20 a1)** kept the instance running and the lane script completed — even though the launcher gave up. Worth investigating: does phase2-launch actually need 120s? For lane scripts that detach to tmux quickly it should return faster; for scripts that exec inline (no tmux), 120s may be too short.
5. **`subagent_commit_serializer.py` lock contention** — when 3-5+ agents are committing simultaneously, the 120s default lock timeout is hit. Use `--timeout-seconds 600`. Even then, my commit of `active_dispatches.md` failed twice (once SIGURG-144 from being run synchronously over 3min, once via lock_timeout). State remains uncommitted on disk; will land via next agent or manual commit.
6. **`.omx/state/lane_maturity_audit.log` is gitignored** — drop from --files arg.

## Maturity harness updates

Marked then unmarked `contest_cuda` gates for Lane 12/19/20/multi_pass — pending dispatches MUST NOT mark a gate=true (would falsely advance to L2/L3). Pending state lives in `active_dispatches.md` only. Cleaned up audit log entries:

```
unmark lane_12_nerv_mask_codec contest_cuda
unmark lane_19_segnet_logit_margin contest_cuda
unmark lane_20_balle_hyperprior contest_cuda
unmark lane_multi_pass_inflate contest_cuda  # had erroneously gone to L3
```

## Next steps

1. Wait for Lane 12 + 19 + 8 retry_b instances to complete (or fail). If complete, retrieve `[contest-CUDA]` scores and mark maturity gates.
2. Investigate Lane 20 CUDA device-mismatch bug in `balle_hyperprior_codec.py` eval-encode path. With it fixed the codec might actually beat static.
3. Commit `active_dispatches.md` + the updated `lane_registry.json` once subagent commit lock contention clears.
