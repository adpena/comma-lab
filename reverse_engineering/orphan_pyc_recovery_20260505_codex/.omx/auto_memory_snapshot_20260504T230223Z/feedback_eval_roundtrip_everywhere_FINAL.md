---
name: EVAL_ROUNDTRIP MUST BE ON EVERYWHERE — NO EXCEPTIONS — STOP LAUNCHING WITHOUT IT
description: Every training run, every TTO, every postfilter, every optimization that touches scorer loss MUST use eval_roundtrip. We have made this mistake on EVERY SINGLE COMPONENT. This is the #1 cause of wasted GPU hours in this project.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Pattern (repeated 9+ times — FIXED 2026-04-21)

1. train_distill.py: eval_roundtrip defaulted False → fixed
2. constrained_gen.py: eval_roundtrip defaulted False → fixed
3. optimize_poses.py: eval_roundtrip defaulted False → fixed
4. renderer_tto.py: eval_roundtrip defaulted False → fixed
5. training.py Trainer: HAD eval_roundtrip but GT cache bypassed it → FIXED
6. train_renderer.py: NO eval_roundtrip at all + GT cache unguarded → FIXED
7. train_renderer_fridrich.py: NO eval_roundtrip at all → FIXED
8. train_postfilter_on_renderer.py: BROKEN bilinear-only roundtrip → FIXED
9. train_lora_tto.py: BROKEN bilinear-only roundtrip → FIXED
10. optimize_latent_codes.py: BROKEN bilinear-only roundtrip → FIXED
11. quantize_distilled.py: BROKEN bilinear-only roundtrip → FIXED
12. pair_difficulty_map.py: renamed simulate_resize → eval_roundtrip → FIXED

## TWO Distinct Bugs

### Bug 1: GT Scorer Cache Bypass (3 files)
GT scorer outputs are pre-computed on RAW GT frames for efficiency. But when
eval_roundtrip is on, both pred and GT frames go through the roundtrip chain.
Using the cached (non-roundtripped) GT scorer outputs means comparing
roundtripped-pred against non-roundtripped-GT in scorer space — making
the proxy look ARTIFICIALLY GOOD. The fix: skip the cache when
eval_roundtrip=True, recompute GT scorer outputs through the roundtripped GT.

Impact: The proxy-auth gap appeared to be 1.2-1.5x but was actually hiding
a much larger distribution mismatch. Training converges to wrong optima.

### Bug 2: Bilinear-Only Roundtrip (4 files)
Several files had `simulate_resize` that did F.interpolate down/up WITHOUT
uint8 quantization (round+clamp) or noise injection. The actual eval chain
goes through uint8 PNG, so the quantization step is load-bearing.

Fixed by replacing with `simulate_eval_roundtrip()` from tac.renderer which
does: upscale→round→noise(STE)→clamp(0,255)→downscale.

## Why It Matters

Auth eval does: render at 384x512 → upscale to 874x1164 → uint8 quantize →
downscale to 384x512 → score. Without simulating this chain in training,
the model learns corrections that don't survive the roundtrip.

Proxy-auth gap WITHOUT roundtrip: 2-6x on PoseNet.
Proxy-auth gap WITH roundtrip: 1.2-1.5x on PoseNet.

## THE RULE — NON-NEGOTIABLE

Before launching ANY training, TTO, or optimization:
1. grep for "eval_roundtrip" or "simulate_eval_roundtrip" in the script
2. If it's not there or defaults to False: DO NOT LAUNCH
3. If the training framework doesn't support it: ADD IT FIRST
4. If there's a GT scorer cache: verify it's guarded with eval_roundtrip check
5. There are ZERO exceptions to this rule

## Components Status (ALL FIXED as of 2026-04-21)

- train_distill.py: HAS IT (fixed)
- constrained_gen.py: HAS IT (fixed)
- optimize_poses.py: HAS IT (fixed, local version)
- renderer_tto.py: HAS IT (fixed)
- training.py Trainer: HAS IT + GT cache guarded (fixed)
- train_renderer.py: HAS IT + GT cache guarded (fixed)
- train_renderer_fridrich.py: HAS IT in FridrichRendererConfig (fixed)
- train_postfilter_on_renderer.py: HAS IT (fixed, replaced broken bilinear)
- train_lora_tto.py: HAS IT (fixed, replaced broken bilinear)
- optimize_latent_codes.py: HAS IT (fixed, replaced broken bilinear)
- quantize_distilled.py: HAS IT (fixed, replaced broken bilinear)
- qat_finetune.py: HAS IT (was already correct)
- pair_difficulty_map.py: HAS IT (renamed from simulate_resize)
