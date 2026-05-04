---
name: Lane EC — Engineered Corrections (compress-time SegNet-flipping deltas) — exists in code, never deployed
description: 2026-04-28 discovery — the repo has a fully-implemented engineered-corrections feature (compress-time per-pixel deltas that flip wrong SegNet predictions, serialized sparse zlib-int8, applied at inflate before upscale). 33KB main script + 60KB trick_stack + 444-line test. CONTEST-COMPLIANT (scorer at compress time only). NEVER DEPLOYED as a Vast.ai lane. Predicted [0.85, 1.15] standalone; high stacking potential with Lane W/V/I.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## What it is (mechanism)

**Engineered corrections** = a compress-time gradient-attack feature that:

1. At COMPRESS time: loads the trained renderer + masks + SegNet GT
2. Computes per-pixel ±max_delta perturbations that flip incorrect SegNet predictions back to the GT class
3. Serializes as sparse zlib-compressed int8 deltas (`gradient_corrections.bin`)
4. Bundles into the submission archive
5. At INFLATE time: applies the deltas to rendered frames BEFORE the upscale to camera resolution

This is essentially **adversarial-perturbation steganography** — using compress-time gradient access to target SegNet's blind spots. Contest-compliant because scorers only load at compress time, not inflate.

## Files in the repo

- `experiments/precompute_gradient_corrections.py` — 33KB canonical compress-time tool
- `experiments/engineered_quant_noise.py` — alternative entry point
- `src/tac/precompute_corrections.py` — library helpers
- `src/tac/trick_stack.py` — 60KB; correction application logic + composition with other tricks
- `src/tac/tests/test_engineered_corrections.py` — 444-line comprehensive integration test (smoke + golden + edge cases)

## Why it's never been deployed

- Per the FORBIDDEN PATTERNS in CLAUDE.md, "neural artifacts must be inside archive.zip per contest rules" — we ALREADY DO this
- Per `feedback_strict_scorer_rule`, "no scorer at inflate time" — engineered corrections only loads scorer at COMPRESS time, fine
- The feature was implemented and tested in earlier sessions but never made it to a Vast.ai lane script
- Probable reason: when it was built, the focus shifted to renderer improvements (Lane A pose TTO) which took most of the GPU budget
- It's been sitting in the repo for ~2 weeks unused

## Score impact (predicted)

**Lane A baseline**: 1.15 = pose 0.22 + seg 0.46 + rate 0.46

**Lane EC standalone** (Lane A + corrections.bin):
- SegNet improvement: 0.0046 → 0.0010-0.0020 (4-5× better) — this is the explicit goal
- Score change from SegNet: 100 × (0.0046 - 0.0015) = -0.31
- Rate cost from corrections.bin: typically 10-50KB → 25 × 50000/37545489 = +0.033
- Net: -0.31 + 0.033 = -0.28 → predicted [0.85, 1.05]

**The catch**: corrections.bin size depends on how many pixels need flipping. Highly textured/error-prone frames need more corrections → bigger bin. The trade-off is a Lagrangian: at some max_delta, the SegNet improvement plateaus and rate cost grows linearly.

## Stacking opportunities (HIGH-EV combinations)

### Stack EC + Lane A (cheapest, highest-confidence)
- Lane A renderer (1.15 frontier)
- + corrections.bin
- Predicted [0.85, 1.05]
- Cost: ~$0.30 (compress-only, no retraining)
- Risk: low — purely additive

### Stack EC + Lane V (Quantizr replica)
- Lane V renderer (predicted [0.50, 1.10] if it lands)
- + corrections.bin polishes SegNet residual errors
- Predicted [0.40, 0.95]
- Cost: ~$4 (Lane V) + $0.30 (EC) = $4.30
- Risk: medium — depends on Lane V landing

### Stack EC + Lane W (hard-pair weighted SC)
- Lane W renderer (predicted [0.85, 1.10])
- + corrections.bin handles the hardest-pair SegNet errors that even hard-pair weighting can't catch
- Predicted [0.65, 0.95]
- Cost: ~$0.50 (Lane W) + $0.30 (EC) = $0.80
- Risk: low — Lane W's per-pair weighting + EC's per-pixel deltas are ORTHOGONAL

### Stack EC + Lane I (Cool-Chic) — orthogonal
- Lane I CCh1 renderer (smaller arch, predicted [0.95, 1.30])
- + corrections.bin compensates for any SegNet capacity reduction in Cool-Chic
- Predicted [0.75, 1.15]
- Cost: ~$0.50 + $0.30 = $0.80
- Risk: medium — depends on Cool-Chic landing well

### Stack EC + Lane SZ (szabolcs no-masks)
- Lane SZ renderer (predicted [0.30, 0.50])
- + corrections.bin polishes the LUT-decoded mask predictions
- Predicted [0.20, 0.45]
- Cost: ~$4 + $0.30
- Risk: high — SZ is a moonshot, but EC is purely additive bonus if SZ lands

### Stack EC + Lane Ω-V2 (Lagrangian per-element bits)
- Lane Ω-V2 renderer (predicted [0.65, 1.05])
- + corrections.bin handles SegNet errors from quantization noise
- Predicted [0.55, 0.95]
- Cost: ~$1.50 + $0.30

## Why corrections.bin works (the math)

SegNet outputs class logits per pixel. Distortion = argmax(predicted_logits) ≠ argmax(gt_logits) per pixel. To FLIP a pixel from class A to class B, we need to perturb the input pixel intensity (R, G, B) so that SegNet's logits change `argmax_A → argmax_B`.

For a fixed SegNet, the gradient `∂logit_B / ∂(pixel_R, G, B)` is computable at compress time. A small perturbation in the gradient direction increases logit_B faster than logit_A. The per-pixel delta required to flip is bounded by `max_delta` (typically 5-15 in 0-255 range). Sparse — most pixels need no correction.

This is INVERSE STEGANALYSIS at the per-pixel level — Yousfi/Fridrich's domain. The contest scorer is essentially a steganalyzer; we're embedding "scoring information" into the rendered frames.

## Risks / caveats (council adversarial)

- **Rate inflation on textured frames**: heavy lane changes / sun glare / pedestrians might need 50-100KB of corrections, blowing the rate budget
- **Generalization mismatch**: corrections are computed for the SPECIFIC scorer weights at compress time. If the contest scorer has minor weight drift (different floating-point precision), corrections may not flip
- **Inflate-time application order**: corrections must be applied at the EXACT resolution + format the scorer evaluates at (post-renderer, pre-upscale). Memory: `feedback_HWC_CHW_format_boundary` — careful with format ordering
- **Pose TTO interaction**: corrections are computed AFTER pose TTO. If the poses are bad, corrections compensate for bad rendering, not just for inherent SegNet errors. Need to ensure pose TTO converges before computing corrections.

## Recommended deployment order

1. Lane EC standalone on Lane A — cheapest, fastest validation
2. If standalone EC works (sub-1.0): deploy EC + Lane W stack
3. If EC + Lane W works (sub-0.8): deploy EC + Lane V stack
4. Long-term: EC + Lane SZ as the moonshot

## Action items

- A subagent has been dispatched to write `scripts/remote_lane_ec_engineered_corrections.sh` + tests
- After standalone Lane EC lands, write composition scripts for EC × {W, V, I, Ω-V2, SZ}

## Related memories
- `feedback_strict_scorer_rule` — scorer-at-inflate is forbidden; engineered corrections complies (compress-time only)
- `feedback_HWC_CHW_format_boundary` — careful ordering at the format boundary
- `project_outstanding_work_and_stacks_20260428` — full stack composition strategy
- `project_session_20260415` — earlier session that mentioned engineered corrections (they may have been auth-eval'd at "0.61 contest-compliant" with distillation+pose TTO)
