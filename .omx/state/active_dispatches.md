# Active Dispatches

Tracks Vast.ai / Modal dispatches with Pattern A nohup detach. Updated on each new dispatch.

## Format

`| timestamp_utc | lane_label | instance_id | predicted_band | estimated_cost | ETA | kill_criteria | dispatch_log |`

## Active

| timestamp_utc | lane_label | instance_id | predicted_band | est_cost | ETA | kill_criteria | log_dir |
|---|---|---|---|---|---|---|---|
| 2026-04-30T12:27Z | lane_12_nerv_2026-04-30 | 35898555 (a3, after _a1+_a2 retries) | [1.00, 1.10] | $0.85 (dph $0.2777) | ~3.4h | seg+pose+rate via auth eval; KILL if no archive after 4h | /tmp/dispatch_lane_12_nerv_20260430T122716Z |
| 2026-04-30T12:30Z | lane_19_logit_margin_2026-04-30 | 35898546 (a2, after _a1 retry) | [0.85, 1.00] | $1.50 (dph $0.2497) | ~5h | seg+pose+rate via auth eval; KILL if no archive after 6h | /tmp/dispatch_lane_19_logit_margin_20260430T123010Z |
| 2026-04-30T12:31Z | lane_20_balle_2026-04-30 | 35898486 (a1) | [1.02, 1.07] | $0.50 (dph $0.2632) | ~2h | seg+pose+rate via auth eval; KILL if no archive after 3h | /tmp/dispatch_lane_20_balle_20260430T123145Z |
| 2026-04-30T12:33Z | lane_8_multipass_2026-04-30 | 35898590 (a1) | [1.04, 1.06] | $0.13 (dph $0.2897) | ~30min | seg+pose+rate via auth eval; KILL if no archive after 1h | /tmp/dispatch_lane_8_multipass_20260430T123320Z |
| 2026-04-30T12:39Z | lane_17_imp_10cycle_2026-04-30 | 35898876 (_a1, after duplicate-dispatch cleanup at 12:38Z) | [0.95, 1.10] | $24.00 cap (dph $0.2497, --max-dph 0.30 → 96h cap) | ~55-80h (10 cycles × 5-10h each) | revert-on-regression at 1.10× best [contest-CUDA] OR no archive after 80h OR cost > $25; per-cycle CUDA auth eval at cycles 0/2/4/6/8/9; KILL via `vastai destroy instance 35898876` | /tmp/dispatch_lane_17_imp_20260430T123933Z (PID 63047) |
