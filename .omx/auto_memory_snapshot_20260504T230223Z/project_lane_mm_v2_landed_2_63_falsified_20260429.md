---
name: Lane MM v2 LANDED 2.63 — encoder-only grayscale-LUT hypothesis FALSIFIED
description: 2026-04-29 PM. Lane MM (encoder-only grayscale-LUT mask + Lane A renderer) scored 2.63 [contest-CPU advisory]. Predicted [0.65, 0.85]. PoseNet 51x worse, archive 1.6x bigger (1.13MB vs 0.73MB). Hard-argmax grayscale-LUT does NOT preserve quality nor reduce rate when bolted onto a 3ch-trained renderer. Re-confirms Lane AL (SGD-optimized soft grayscale) is the correct path. NOTE: 2026-04-29 PM Council Round 7 §7.2 retagged from [contest-CPU advisory] (which conflated CUDA dispatch with CPU eval) to [contest-CPU advisory] — verdict needs CUDA confirm to be promoted from FALSIFIED-on-CPU to FALSIFIED-on-CUDA. Cost: ~$0.50 Vast.ai 4090 if anyone wants to formalize the kill.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Result**: Lane MM v2 = 2.63 [contest-CPU advisory], elapsed 832s. Predicted [0.65, 0.85].

**[Verdict-soundness retag, Round 7]**: per CLAUDE.md "MPS auth eval is NOISE"
+ feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md, ANY
kill/promote decision involving a neural-net forward pass MUST come from
contest-CUDA. The 2.63 score was computed via Modal CPU eval. CPU has
lower drift than MPS but is NOT contest-CUDA. The directional verdict
(FALSIFIED) STANDS — the architecture-mismatch is structural, not
measurement-noise — but the EXACT magnitude (51× PoseNet ratio) should
not be cited as `[contest-CUDA]`. Re-tag: `[contest-CPU advisory —
directional FALSIFIED, awaits contest-CUDA confirm]`. Cost to formalize:
~$0.50 Vast.ai 4090. NOT a kill-list reversal; just a tag-cleanup.

**Score breakdown** (vs Lane A 1.15):
| Component | Lane A | Lane MM v2 | Ratio |
|---|---|---|---|
| PoseNet dist | 0.0034 | **0.1737** | **51.1×** |
| SegNet dist | 0.0040 | 0.0056 | 1.4× |
| rate_unscaled | 0.0185 | 0.0302 | 1.6× |
| archive bytes | 729KB | 1,133,750B | 1.55× |

**HYPOTHESES FALSIFIED**:
1. ❌ "Grayscale-LUT mask cuts rate ~50% with no quality loss" → archive went UP 55%, PoseNet exploded 51×.
2. ❌ "AV1 monochrome at the spread targets {0,64,128,192,255} is more efficient than legacy {0,63,127,191,255}" → AV1 is LESS efficient on the spread, contrary to expectation.
3. ❌ "Existing 3ch-trained renderer can consume LUT-decoded one-hot input without retraining" → renderer trained on 3ch discrete masks doesn't accept a different mask distribution; produces broken RGB → PoseNet sees garbage.

**WHAT WE LEARNED (preserve signal)**:
- The Selfcomp paradigm REQUIRES the renderer to be JOINTLY TRAINED with grayscale-LUT input. Bolting it on top of a 3ch-trained renderer (Lane A) breaks the renderer.
- Hard-argmax grayscale encoding loses information at boundary pixels through the AV1 round-trip.
- The COUNCIL EUREKA insight stands: "optimize grayscale values directly via SGD as analog latent canvas, NOT compressed argmax". This is exactly Lane AL.
- Lane SC++ v3 trains SegMap WITH grayscale-LUT input from epoch 0; predicted [0.30, 0.40] is still valid.

**FALSIFICATION VALUE**:
- $0.30 spent
- Saves us from: any future "encoder-only swap mask format" experiment that doesn't co-train the renderer.
- Validates: Lane SC++ (joint-trained) and Lane AL (SGD-optimized soft grayscale).
- Updates: Lane MM v2 hypothesis is dead AS A QUICK WIN. Could still work as a probe IF the inflate path is fixed AND the renderer is fine-tuned for grayscale-LUT input (~1h adapter training).

**Next steps**:
- Lane MM v3 = MM v2 + 100-step adapter fine-tune of Lane A renderer to grayscale-LUT input. Cheap probe ($0.50 / 1h). Predict: rate stays ~0.030, but PoseNet should recover to ~0.005. Total ~0.95 instead of 2.63.
- OR: defer Lane MM entirely; focus on Lane SC++ (joint-trained from epoch 0) and Lane AL (SGD-optimized).

**Cross-refs**:
- project_codex_theoretical_floor_brutal_20260429 (codex says EUREKA lanes are bolt-ons not primary)
- project_council_kill_uniward_20260429 (similar pattern: encoder-only no-op)
- project_selfcomp_reverse_engineered_20260429 (the paradigm we're forking)

**Strategic note**: per user mandate "don't be too aggressive in killing", Lane MM v2 is NEGATIVE SIGNAL but the path stays alive. Lane MM v3 (with renderer adapter) is queued; Lane AL (SGD-optimized) is the proper EUREKA realization.
