# Codex Findings: Compact Carrier Plan Consumer

UTC: 2026-06-01T21:10:44Z
Agent: Codex
Axis: compact carrier queue automation
Status: LANDED as queue-runner consumer, no score claim

## Verdict

The compact renderer MLX spine runner now consumes the score-aware carrier
training planner by default. `target_family_rows` and
`compact_base_campaign_rows` both carry a nested
`score_aware_carrier_training_plan`, so HiNeRV/SNeRV/RNeRV/SR-NeRV queue rows no
longer rely on prose interpretation to decide whether to train, replay, or
block.

## Preserved Signal

The current HiNeRV advisory evidence is routed explicitly:

- projected small rate is preserved as `structural_rate_knob_present`;
- unusable distortion routes to decoder-weight score-aware training;
- low latent leverage demotes L-infinity latent posthoc allocation;
- all exact/promotion authority remains false;
- missing PR95+QAT/NVRC/real-scorer stack pieces are blockers until wired.

Unmeasured carriers get fail-closed planner rows with missing-proof blockers,
not optimistic defaults.

## Code Landing

- `tools/run_compact_renderer_mlx_spine_runner.py` imports
  `build_score_aware_carrier_training_plan`.
- Target-family rows and hard-byte-ceiling campaign rows now include
  `score_aware_carrier_training_plan`.
- Focused tests verify that HiNeRV is routed to
  `run_score_aware_decoder_weight_training_full_main`, not promotion, and that
  SNeRV remains blocked until real scorer/QAT/export readiness lands.

## Next Automation Step

The next score-moving build is to make `--execute-family hi_nerv` / `snerv`
call the same planner row before launching, then run the score-aware decoder
training stack only when the plan’s blockers are closed enough for the selected
campaign scale.

No large artifact was produced by this landing.

