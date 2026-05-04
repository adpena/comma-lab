---
name: Lane I CRASHED at Stage 3 export — parametrize-strip mismatch on CCh1 checkpoint load
description: 2026-04-28 Lane I (Cool-Chic CCh1 renderer-replacement on Lane A masks/poses) trained successfully through epoch 999/1000 with best FP4 scorer 2.7196 at epoch 754. Stage 3 export crashed with RuntimeError: CCh1 load missing keys — parametrize.weight.original vs raw weight key mismatch. NO auth eval happened. Score unknown. Vast instance 35733831 (ssh1:13830) idle since heartbeat stopped 11:43Z (2+ hours of $0.28/hr burn = ~$0.60 wasted).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened

Vast.ai instance 35733831 ran Lane I (`scripts/remote_lane_i_coolchic_masks.sh`) for 5.7 hours starting 2026-04-28 ~08:01Z.

**Stages 1+2 (training) succeeded:**
- 1000 epochs (P1 + P2)
- Best FP4 scorer (proxy): 2.7196 at epoch 754
- Final train.log: `[train] Complete. Best FP4 scorer: 2.7196 at epoch 754`
- Saved `renderer_lane_i_coolchic_best_fp4.pt`

**Stage 3 (CCh1 export) FAILED:**
```
RuntimeError: CCh1 load missing keys: ['renderer.class_embed.weight', 'renderer.decoder.0.weight', 'renderer.decoder.2.weight', 'renderer.decoder.4.weight', 'motion.embedding.weight', 'motion.stem.0.weight']

[stage3] load mismatch: missing=[..., 'renderer.class_embed.weight', ...] unexpected=['renderer.class_embed.parametrizations.weight.original', 'renderer.class_embed.parametrizations.weight.0.codebook', ...]
```

**Root cause**: `torch.nn.utils.parametrize` was applied to the renderer convs (likely from CoolChic's quantization codec). The `state_dict()` then contains `parametrizations.weight.original` + `parametrizations.weight.0.codebook` keys instead of plain `weight` keys. The Stage 3 loader expects plain key names (model rebuilt from architecture spec, no parametrize applied) → key mismatch.

## Why Stage 3 was unable to handle this

Looking at `experiments/qat_finetune.py` history: there's a `parametrize-strip` fix (task #121, commit) but Lane I's Stage 3 inline export script (embedded in `scripts/remote_lane_i_coolchic_masks.sh`) doesn't import that helper — it has its own loader that doesn't strip.

## Score impact

- Best FP4 PROXY = 2.7196 at epoch 754
- Auth score = UNKNOWN (Stage 3 crashed before eval)
- Per `feedback_proxy_auth_math_useless`, proxy is meaningless without auth eval. Could be 1.5-3× higher (1.5 if optimistic, 3.0 if PoseNet sensitivity is bad)

## Operational waste

- Idle from 11:43Z to investigation at ~13:55Z = 2.2 hours × $0.28/hr = $0.62
- Total Lane I run cost ~$1.60
- Yielded ZERO contest-CUDA measurement

## Fix forward (TIER-1)

1. **Patch Lane I script**: Stage 3 export must use parametrize-strip helper from `qat_finetune.py`. Same fix as task #121 but applied to lane_i specifically.
2. **Watchdog**: stop wasting $0.28/hr on idle instances. Per `feedback_vastai_launch_returns_success_before_lane_starts`, build per-instance verify that destroys instances after N minutes of stale heartbeat.
3. **Re-launch Lane I** after patch (when launcher infra is fixed per Cycle 1 postmortem).

## Cross-references
- `feedback_cycle_1_launch_postmortem_20260428` — broader operational debt
- `feedback_proxy_auth_math_useless` — proxy score is meaningless
- `feedback_vastai_launch_returns_success_before_lane_starts` — silent failures
- task #121 — the parametrize-strip fix that exists but wasn't applied to Lane I
