# SNeRV long-run blocker - fp16 decoder scale overflow

- schema: `snerv_long_run_blocker.v1`
- captured_utc: `2026-06-03T20:04:31Z`
- axis: `[macOS-MLX research-signal]`
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## Run

- parent_pid: `26531`
- worker_pid: `26532`
- output_dir: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_queue_launch_20260603T195128Z/snerv_snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_adbase_oms0p285_int8_symmetric_ceil216000_native_rate_aware_training`
- detached_log: `/Volumes/VertigoDataTier/pact/detached_launches/snerv_full600_scoreaware_long_queue_20260603T195223Z/run.log`
- planner_queue: `.omx/research/nerv_long_training_campaign_plan_20260603T195128Z_snerv_queue_launch_codex_queue.json`

## Blocker

The run loaded the pre-fix SNeRV decoder packet grammar and emitted:

```text
RuntimeWarning: overflow encountered in cast
  scale_payload = np.asarray(scales, dtype="<f2").tobytes()
RuntimeWarning: invalid value encountered in multiply
  values[start:stop] *= float(scale)
```

That means receiver decoder scales could become `inf`/`nan` before any
receiver-proven packet is promoted. This run is therefore contaminated for
archive/runtime custody and must not be used as a promotion candidate.

## Fix

The paired code landing makes quantized decoder scale payloads adaptive:
`float16_le` remains the compact path when safe, and `float32_le` is used when
fp16 would overflow, underflow positive scales to zero, or become non-finite.
Decoders remain backward compatible through the `scale_dtype` header.

## Next Action

Stop the contaminated local process after preserving this blocker note, commit
the grammar fix, then relaunch the same SNeRV queue row against the fixed code.
