# Active Dispatches

Tracks Vast.ai / Modal dispatches with Pattern A nohup detach. Updated on each new dispatch.

## Format

`| timestamp_utc | lane_label | instance_id | predicted_band | estimated_cost | ETA | kill_criteria | dispatch_log |`

## Active

| timestamp_utc | lane_label | instance_id | predicted_band | est_cost | ETA | kill_criteria | log_dir |
|---|---|---|---|---|---|---|---|
| 2026-04-30T12:51Z | lane_12_nerv_2026-04-30_b | (retry-3 attempts in flight; max-retries=5) | [1.00, 1.10] | $0.85 (dph $0.30 cap) | ~3.4h | seg+pose+rate via auth eval; KILL if no archive after 4h | /tmp/dispatch_lane_12_nerv_retry_20260430T125106Z |
| 2026-04-30T12:53Z | lane_19_logit_margin_2026-04-30_b | (retry-1 in flight; max-retries=5) | [0.85, 1.00] | $1.50 (dph $0.30 cap) | ~5h | seg+pose+rate via auth eval; KILL if no archive after 6h | /tmp/dispatch_lane_19_logit_margin_retry_20260430T125321Z |
| 2026-04-30T12:55Z | lane_8_multipass_2026-04-30_b | PENDING DISPATCH (60s spacing) | [1.04, 1.06] | $0.13 (dph $0.30 cap) | ~30min | seg+pose+rate via auth eval; KILL if no archive after 1h | /tmp/dispatch_lane_8_multipass_retry_<TS> |
| 2026-04-30T12:49Z | lane_17_imp_10cycle_2026-04-30T124951Z | 35899275 (_a1; round-2 success; round-1 burned 3 attempts on NVDEC roulette) | [0.95, 1.10] | $24.00 cap (dph $0.2957, 80h × dph = $23.66) | ~55-80h | revert-on-regression at 1.10× best [contest-CUDA] OR no archive after 80h OR cost > $25 | /tmp/dispatch_lane_17_imp_20260430T124951Z (other agent) |

## Completed (this session)

| timestamp_utc | lane_label | instance_id | result | notes |
|---|---|---|---|---|
| 2026-04-30T12:31Z | lane_20_balle_2026-04-30_a1 | 35898486 (DESTROYED 12:50Z) | empirical:STATIC_WINS_FALLBACK | Lane 20 trainer ran full 5000 steps; static codec wins (best=static=136296 bytes); archive ships ZERO bytes from Lane 20; auto-fallback engaged. Results harvested to experiments/results/lane_20_balle_2026-04-30_a1_recovered/. NO contest-CUDA auth eval needed (codec is no-op). Cost: ~$0.10 (3 NVDEC retries + 1 successful train). |

## Failed dispatches (re-attempting)

| timestamp_utc | lane_label | reason | retry_disp |
|---|---|---|---|
| 2026-04-30T12:27Z | lane_12_nerv_2026-04-30 (a1+a2+a3) | All 3 attempts: phase2-extract failed (a1, a2) + phase2-wait timeout (a3) | retry as `_b` with --max-retries=5 |
| 2026-04-30T12:30Z | lane_19_logit_margin_2026-04-30 (a1+a2+a3) | a1 NVDEC, a2 SSH timeout, a3 phase2-extract | retry as `_b` with --max-retries=5 |
| 2026-04-30T12:33Z | lane_8_multipass_2026-04-30 (a1+a2+a3) | All 3 attempts NVDEC_BAD on this host (5/6 lanes hit NVDEC roulette tonight, expected per memory feedback_vastai_nvdec_roulette_pivot_to_modal_20260429.md) | retry as `_b` with --max-retries=5 |
