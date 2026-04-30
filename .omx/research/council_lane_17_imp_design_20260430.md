---
name: Council Design Review — Lane 17 (IMP 10-cycle) Level-3 plan
description: 2026-04-30 council deliberation on Lane 17 IMP design choices before Level-3 push. Per CLAUDE.md "Council conduct" — non-conservative, mathematical/empirical only.
type: research
authoritative_for: lane_17_imp
---

## Convening

Inner council quintet pact: **Shannon (LEAD)** + **Dykstra (CO-LEAD)** + Yousfi + Fridrich + Contrarian.
Co-members: Quantizr + Hotz + Selfcomp + MacKay + Ballé.
Specialty calls (grand council): **Frankle** (Lane 17 OG, lottery-ticket hypothesis author).

Anchor: Lane G v3 = 1.05 [contest-CUDA], 88K-param dilated-h64 renderer, 290KB renderer.bin, 678KB total archive.

## The 7 design questions

### Q1: Per-cycle pruning rate — 20% / cycle × 10 cycles, or 10% / cycle × 22 cycles, or 30% / cycle × 6 cycles?

| Schedule | Final sparsity | Cycles | Total epochs | Compute cost |
|----------|----------------|--------|--------------|--------------|
| 30%/6 | 88.2% | 6 | 1200 | $15 |
| 20%/10 | 89.3% | 10 | 2000 | $25 |
| 10%/22 | 89.8% | 22 | 4400 | $55 |

**Frankle**: my 2019 stabilization paper showed 20%/cycle is near the sweet spot — 30% is too aggressive (kills too many lottery tickets in one shot, retrain can't recover) and 10% is wasted compute (each cycle barely changes the network, retrain re-converges fast). Choose 20%.

**Hotz**: 30% gets you 90% of the benefit at 60% of the cost. Why are we paying $10 extra to chase a marginal lottery-ticket signal? Just kill more per cycle and retrain longer.

**Quantizr**: I shipped 88K dense at 0.33. The question is whether sparsity actually buys score, not just bytes. At 89% sparse → 9.4K active → if the score holds, you're at ~5KB renderer instead of ~50KB FP4. That's -0.045 score from rate term alone. But the SCORE is ALSO going to move (whether up or down depends on whether the lottery ticket exists at our scale).

**Shannon**: from R(D) standpoint, 89% sparsity at the same distortion is a 10× rate reduction. But the THEORETICAL rate floor for the same distortion is determined by the model class capacity, not by sparsity directly. If 9.4K params CANNOT represent the renderer function, the distortion will increase regardless of how cleverly we encode the surviving weights.

**VOTE**: 8/10 for **20%/cycle × 10 cycles**. Hotz's 30%/6-cycle is the kill-criterion-trigger fallback (Q4).

### Q2: Global magnitude pruning vs per-layer magnitude pruning?

**Frankle**: the 2019 paper found global outperforms per-layer on ResNets. Global lets the network discover that some layers (e.g. early stem convs) need higher density than others (late 1×1 convs).

**Selfcomp**: my 88K-94K renderer has a heavy mid_ch=60 motion path that probably tolerates more pruning than the 6-channel embedding. Global is the right call.

**Contrarian**: per-layer guarantees no layer is fully zeroed out. Global at 89% sparsity could conceivably zero an entire small conv. Need a sanity check: assert min(per_layer_sparsity) < 0.99 to avoid catastrophic single-layer collapse.

**VOTE**: 9/10 for **GLOBAL magnitude pruning** with a per-layer safety check (no layer above 99% sparse). Add the check to `prune_lowest_magnitude` follow-up.

### Q3: Per-cycle CUDA auth eval ($3 extra) vs cycle-final-only ($0.50)?

**Yousfi**: my contest scorer is the source of truth. Per-cycle eval lets you DETECT a regression at cycle N=4 vs discovering it at cycle 10 = 2.5h of wasted retrain. That's worth $0.30/eval.

**Hotz**: $3 to detect early-kill is a great deal. Don't be cheap. Per-cycle.

**Quantizr**: I'd want eval at every cycle so I can see the proxy-auth gap evolution. The proxy-auth gap on PoseNet is 100-350x, so I cannot trust the per-cycle proxy loss to tell me if the score is regressing.

**Fridrich**: from a rate-distortion sequencing standpoint, the early cycles are where the lottery-ticket signal (if any) emerges. Eval at cycles 0, 2, 4, 6, 8, 9 (6 evals = $1.80) catches both early-kill and final-result.

**Shannon**: at the prune schedule (20%/cycle), cycles are non-uniform in their information-theoretic impact. Cycles 0-3 (0% → 59%) move the network into a substantially different basin; cycles 4-9 (59% → 89%) are progressive refinement. Allocate evals to where the signal is. **Evals at 0, 2, 4, 6, 8, 9** = 6 × $0.30 = $1.80.

**VOTE**: 7/10 for **6-eval schedule (cycles 0, 2, 4, 6, 8, 9)** + the final auth eval at Stage 4. Total: 7 × $0.30 = $2.10. Cheap insurance.

### Q4: Revert-on-regression — what threshold?

**Frankle**: in the 2019 paper, lottery tickets at high sparsity (>80%) are noisy. A single-cycle regression of 5% is signal noise; 15% is real failure. Anything between is ambiguous.

**Yousfi**: contest scoring is unforgiving — 10% regression on the score is a bigger deal than 10% regression on a per-cycle proxy. We're at 1.05 baseline; 10% = 0.105 = unacceptable archive.

**Contrarian**: revert-and-stop is the right kill criterion. But make sure the REVERT actually rolls back to the cycle-N-1 mask AND re-exports the cycle-N-1 renderer.bin AND uses the cycle-N-1 archive for the final report. Don't ship the regressed cycle-N artifacts by accident.

**Quantizr**: keep cycle 0's renderer.pt as a safety floor. If cycle 5 regresses >10% from cycle 0, kill — even cycle 4's mask might be on a bad trajectory.

**VOTE**: 9/10 for **revert-on-regression: kill if `cycle_N_score > 1.10 × min(cycle_0..N-1_score)`** AND ship the lowest-scoring cycle's archive as the final result. Cycle 0 score is the baseline floor; subsequent cycles must beat or hold within 10%.

### Q5: Final FP4 quantization on top of sparse-CSR?

**Selfcomp**: at 89% sparse you have 9.4K surviving weights. FP4 on those = 4 bits × 9.4K = 4.7KB. WITHOUT FP4 (FP32 on survivors) = 32 bits × 9.4K = 37.6KB. The 8× difference is huge.

**Quantizr**: FP4 is the de facto floor for quantization in this contest. The block-FP variant I use buys another ~10% over uniform FP4. Stack both: sparse + block-FP-on-survivors → ~4KB renderer.

**Shannon**: surviving-weight magnitude distribution after IMP is heavy-tailed (high-magnitude survivors dominate). Block-FP exploits this by allocating different per-block scales. Worth the implementation.

**Frankle**: in my LTH papers, FP4 on survivors didn't degrade the lottery ticket. The sparsity, not the precision, is the key signal.

**Decision deferred**: lane 17's primary deliverable is the SPARSE renderer; block-FP on survivors is a Lane Ω-W-V2 stack composition (Lane 17 + Lane Ω-W-V2). Council recommends the standalone Lane 17 result first, then the stack.

**VOTE**: 10/10 for **FP4 (uniform) on survivors as part of Lane 17**; defer block-FP stack to Phase 2 follow-up (separate dispatch).

### Q6: Inflate-side magic byte — `IMPS` (b"IMPS")?

**Hotz**: pick a magic byte, get it merged, move on. `IMPS` for "IMP Sparse". Done.

**Selfcomp**: my SCv1 / SZv1 / OWV2 magic-byte naming convention is `<lane-letter><variant><digit>`. So `IMPS` doesn't fit. But Lane 17 is special — it's not a renderer arch variant, it's a sparsity codec. `IMPS` is fine; it's consistent with `OWV2` (Lane Ω-W-V2 doesn't fit the convention either).

**Contrarian**: 4 bytes per the convention. `IMPS` is 4 bytes ASCII, valid. Document the format in the same docstring block as the other magic bytes (`_load_renderer` docstring).

**VOTE**: 10/10 for **magic byte b"IMPS"**.

### Q7: Stretch goal — combine Lane 17 + Lane Ω-W-V2 + Lane J-NWC?

**Dykstra**: from convex-feasibility, three independent rate savings stack ONLY if their gradients are non-overlapping. IMP zeros weights → reduces effective DOF. Ω-W-V2 quantizes survivors → no DOF reduction. J-NWC encodes the codebook itself → no DOF reduction. The three operate on DISJOINT axes. Stacking is feasible.

**Ballé**: Lane Ω-W-V2 hyperprior estimates per-element entropy on the surviving weights. With 89% sparsity, the surviving distribution narrows → hyperprior savings are 30-40% better than on the dense baseline. STACK.

**MacKay**: from MDL, the description-length cost of the SPARSITY MASK itself is non-trivial. 88K-bit mask = 11KB before any compression. RLE / arithmetic-coded mask is ~6-8KB. Don't forget to include this in the rate accounting.

**VOTE**: stretch is **out of scope for this Level-3 push**. Lane 17 ALONE first. Stack with Ω-W-V2 only if Lane 17 lands a clean [contest-CUDA] result.

## Summary of Council Verdict

| Q | Decision | Vote |
|---|----------|------|
| 1 | 20%/cycle × 10 cycles | 8/10 |
| 2 | Global magnitude prune + per-layer safety check (max 99% per layer) | 9/10 |
| 3 | Per-cycle CUDA auth eval at cycles 0, 2, 4, 6, 8, 9 (6 evals = $1.80) + final | 7/10 |
| 4 | Revert-and-stop on `cycle_N_score > 1.10 × min(cycle_0..N-1_score)`; ship lowest-scoring cycle's archive | 9/10 |
| 5 | FP4 on survivors (uniform) as part of Lane 17; defer block-FP stack | 10/10 |
| 6 | Magic byte `b"IMPS"` for inflate-side sparse-CSR handler | 10/10 |
| 7 | Stack with Ω-W-V2 / J-NWC: out of scope this push | (deferred) |

## Predicted score band [prediction]

- **Best case** (lottery ticket exists at 88K-param scale): score 0.90 - 1.00 (small improvement from rate term reduction; renderer holds at near-baseline distortion). Archive size: ~290KB → ~250KB (sparse-CSR saves ~21KB on renderer; mask itself adds ~7KB; net -14KB).
- **Most likely** (lottery ticket weakly exists): score 1.00 - 1.15 (renderer distortion increases ~5-10% due to capacity loss; rate savings partially offsets). Archive: ~270KB.
- **Worst case** (no lottery ticket at this scale, regression triggers revert): kill at cycle N=2-4; ship cycle N-1 archive ~ Lane G v3 baseline; sunk cost limited to cycles up to revert.

Predicted band per the dispatcher script: `[0.85, 1.00]`. Council narrows to `[0.92, 1.12]` central, `[0.85, 1.20]` 90% CI.

## Cost-stop conditions

- $5 (cycle 1 complete): if cycle-1 auth eval > 1.155 (10% of 1.05), STOP and report negative.
- $12 (cycle 5 complete): if cycle-5 auth eval > 1.05 × 1.10 = 1.155, STOP and report negative.
- $25 (cycle 9 complete): final result regardless.
- Any single cycle's CUDA auth eval log not appearing within 1h of cycle start: KILL the run (instance hang).

## Acknowledged risks

1. **Per-cycle CUDA auth eval reliability** — adds 6 inflate.sh + evaluate.py invocations. Each is ~5min on RTX 4090. Total +30min runtime. If auth eval crashes mid-run, dispatcher should log RESULT_JSON_FAIL and proceed to next cycle (not abort).
2. **Sparse-CSR header backwards-compat** — `IMPS` magic byte added to inflate path means archives built before 2026-04-30 cannot have it; safe (no false positives).
3. **EMA shadow at high sparsity** — at 89% sparse, the EMA shadow may have non-zero values at pruned positions due to initial-state averaging. The script at `experiments/train_imp_cycle.py:395` already re-applies the mask AFTER `ema.apply()` to zero those positions. Verified in audit.
4. **Mask serialization size** — 88K-bit mask = 11KB. Archive cost. RLE compression brings it to ~6-8KB. For Lane 17 standalone, the mask goes inside the renderer.bin payload; the inflate-side handler reconstructs the dense weight tensor from sparse-CSR + mask before normal model construction.

## Action items handed off to implementation phase

1. Wire revert-on-regression in `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`.
2. Add per-cycle smoke auth eval at cycles 0, 2, 4, 6, 8, 9.
3. Implement `IMPS` inflate handler in `submissions/robust_current/inflate_renderer.py`.
4. Add per-layer safety check to `prune_lowest_magnitude` (no layer > 99% sparse).
5. Add STRICT preflight Check 91: `check_imp_cycles_use_ema_and_auth_eval`.
6. Add real-archive empirical CPU smoke in tests dir.
7. Run 3-clean-pass adversarial review (rotating perspectives).
8. Pre-dispatch memo + STOP-and-ask user for $25 GPU work.

## Cross-refs

- `lane_17_imp_scaffold_audit_20260430.md` (Phase A audit)
- CLAUDE.md "Council conduct", "Recursive adversarial review protocol", "EMA — NON-NEGOTIABLE", "Auth eval EVERYWHERE"
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 17"
- `project_codec_stacking_composition_canonical_orders_20260429.md` (Q7 stacking discussion)
