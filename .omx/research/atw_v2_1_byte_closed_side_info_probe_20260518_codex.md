# ATW V2-1 Byte-Closed Side-Info Probe

- observed_at_utc: `2026-05-18T08:11:56+00:00`
- axis_label: `[diagnostic-CPU; ATW V2-1 byte-closed side-info MI probe]`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`
- provider_spend_attempted: `false`
- source_reducer_json: `experiments/results/alternative_reducer_probes_20260516T225900Z/tishby_ib_pure_per_pair_reducer_outputs.json`
- output_dir: `experiments/results/atw_v2_1_sideinfo_probe_20260518T081156Z`
- side_info_budget_bytes: `2048`
- phase2_status: `byte_closed_channels_only_weak_conditioning`
- recommended_next_gate: `design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe`

## Channel Results

| Channel | Packet bytes | Budget ok | Unique values | MI bits/symbol | Threshold | Verdict | WZ ceiling | Phase 2 action |
|---|---:|---|---:|---:|---:|---|---:|---|
| per_pixel_histogram | 204 | true | 4 | 0.022656927447 | 0.500 | WEAK_CONDITIONING | 0.003218760133 | design_tighter_scorer_logit_sketch_or_trained_atw_residual_probe |
| per_region_histogram | 323 | true | 7 | 0.047381530305 | 1.000 | WEAK_CONDITIONING | 0.006731264914 | design_tighter_scorer_logit_sketch_or_trained_atw_residual_probe |
| per_pair_class_2_fraction | 127 | true | 2 | 0.009692520351 | 0.200 | INDEPENDENT | 0.001376969502 | do_not_dispatch_from_this_channel |
| per_frame_argmax | 117 | true | 1 | 0.000000000000 | 0.200 | INDEPENDENT | 0.000000000000 | do_not_dispatch_from_this_channel |

## Verdict

Best byte-closed channel: `per_region_histogram` with verdict `WEAK_CONDITIONING`, MI `0.047381530305` bits/symbol, packet bytes `323`.

This probe closes the byte-budget side of the ATW V2-1 question for
the currently available richer reducer artifacts: dictionary-coded
side-info packets fit the <=2KB archive sidecar budget, but Phase 2
dispatch still requires a MEANINGFUL_CONDITIONING result plus Wave
N+1 council. Diagnostic evidence here remains non-promotional.

## Reproduction

- command: `.venv/bin/python tools/probe_atw_v2_1_byte_closed_side_info_channel.py`
- packets: see per-channel `packet_path` entries in the JSON artifact

## Next Gate

`design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe`
