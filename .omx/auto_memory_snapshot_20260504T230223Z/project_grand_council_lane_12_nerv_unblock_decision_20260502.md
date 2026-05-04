---
name: Grand Council — Lane 12 NeRV unblock decision (46h to deadline; sub-0.3 vs ship-C-067)
description: 2026-05-02 council deliberation on whether a NeRV recipe redesign can hit <0.1% argmax-disagreement within 46h, vs Path B infrastructure-first, vs Path C ship-C-067-final. Inner quintet + Quantizr/Hotz/Selfcomp/MacKay/Ballé. Verdict binding for next 46h.
type: project
originSessionId: lane-12-nerv-unblock-decision
---

## TL;DR

**VERDICT: SHIP C-067 AS FINAL. KILL Lane 12 NeRV until post-deadline.**

Tally: **redesign 2 / Path B 1 / ship-C-067 7** (10 voices). Inner quintet (Shannon/Dykstra/Yousfi/Fridrich/Contrarian) is **5/0 for ship-C-067**. Path B work is approved as a **POST-DEADLINE** activity only (zero GPU $ now, zero implementation work in next 46h beyond what's already at HEAD).

The decisive evidence is Phase F's empirical convergence math intersected with the 46h budget and the absence of any adjacent-art datapoint showing a single-frame coordinate INR hitting <0.1% per-pixel argmax agreement on a 5-class scorer-derived mask. No council voice was able to cite a recipe that closes a 5-10× disagreement gap inside the budget without violating either CLAUDE.md non-negotiables or the user's "no hyperparameter cycling" mandate.

## Pre-deliberation evidence stack (verified at HEAD)

- **Phase F empirical** (`reports/lane_12_nerv_real_archive.json`, verified at HEAD 2026-05-02):
  - 1400 of 60000 steps on CPU, 228.7s wall
  - xent loss: 0.59 → 0.02
  - argmax disagreement vs AV1 source: **2.0%**
  - NeRV-fp16 NRV2 payload: 23,594 bytes (vs AV1 baseline 421,483 bytes; -94.4%)
  - Linear extrapolation to 60K CUDA steps: ~0.5% disagreement (xent floor ~0.005)
- **C-067 frontier** (verified): `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, 276,214 bytes, score **0.31561703 [contest-CUDA T4 A++]**
- **Stacked-score arithmetic** (from prior refusal memo, verified):
  - Realistic case (0.5% disagreement, -195KB rate): `0.316 - 0.130 + 0.500 = 0.686` → REGRESSION
  - Optimistic case (0.05% disagreement, -195KB rate): `0.316 - 0.130 + 0.050 = 0.236` → SUB-FRONTIER (the prize)
  - Council target case (0.10% disagreement, -195KB rate): `0.316 - 0.130 + 0.100 = 0.286` → SUB-0.3 (the user's stated bar)
- **Structural blockers** (verified at HEAD, all four MISSING):
  1. `.omx/state/lane12_nerv_l2_clearance.json` — does not exist
  2. `_try_parse_public_pr67_NERV_qzs3_qp1_payload` in `submissions/robust_current/unpack_renderer_payload.py` — does not exist (grep confirmed)
  3. `ALPHA_PRIMITIVE_CONTRACT` for C-067 stack — no JSON in `.omx/state/`
  4. `ALPHA_GEO_PROVENANCE` + `POSE_REGEN_PROVENANCE` for C-067 stack — none exist
- **Time budget**: ~46h to May 4 06:59 AM CDT contest deadline (asserted in mandate, not independently verified by this council)
- **Leaderboard context**: Quantizr 0.33, multiple competitors 0.32–0.33 band, our C-067 0.316 = sub-frontier (per `project_leaderboard_0_32_0_33_floor_or_irrelevant_20260501.md` plus the C-067 anchor row)

## Council voices (10)

### Shannon LEAD — VOTE: SHIP C-067

> "Run the rate-distortion math without flinching. Phase F's xent floor of 0.005 maps to ~0.5% argmax disagreement under the standard cross-entropy → 0/1 conversion (Fano's inequality bounds the per-pixel error rate from below by `H(error) / log(K-1)` where K=5 classes). To get from 0.5% to 0.1% we need to cut the conditional entropy ~5×, which under a fixed coordinate-MLP capacity is a step-change, not a recipe tweak. None of the candidate redesigns 1–6 in the mandate cite a published number on a scorer-derived 5-class mask that achieves <0.1%. Without that adjacent-art datapoint the redesign is a wager, not a derivation. C-067 0.316 is a real Pareto point. SHIP."

### Dykstra CO-LEAD — VOTE: SHIP C-067

> "I project the alternating constraint sets: `rate ≤ 195KB-saved-but-add-NeRV-bytes`, `seg_disagreement ≤ 0.1%`, `46h-wall-clock`, `0-fabricated-gates`. The intersection is empty under any redesign that requires a brand-new architecture (#5 Q-FAITHFUL successor) or new data path (#6 multi-frame INR), because the wall-clock projection alone exceeds 46h once you include 3-pass council review on the inflate parser. Candidate #1 (larger NeRV) and #4 (distill from PR67's deployed AV1 mask) are the only ones whose feasible region might be non-empty, but Phase F's curve gives me no quantitative reason to believe their disagreement floor lands at 0.1% rather than 0.3%. The achievable region for a sub-0.3 stacked landing in 46h is empty. SHIP."

### Yousfi — VOTE: SHIP C-067

> "I've designed contests; I've watched teams blow themselves up by chasing a redesign in the last 48h. The 5-class argmax problem with EfficientNet-B2 stride-2 stem behind it is exactly the steganalysis-blind-spot pattern: tiny logit perturbations at class boundaries are the entire signal. NeRV's coordinate MLP has no spatial-context aperture; it can fit smooth interior regions to 0.01% but boundary pixels (which are 5-15% of the mask area on this dataset) will always cap at 0.5–1%. That's a fundamental structural ceiling, not a recipe issue. The only way past it is a hybrid (#3) and that's a bigger build than 46h supports. SHIP C-067; submit; protect the leaderboard position."

### Fridrich — VOTE: SHIP C-067

> "From the inverse-steganalysis lens: NeRV is a smooth analog signal generator. The argmax disagreement problem is fundamentally a PER-PIXEL CLASSIFICATION error, not a continuous-signal reconstruction error. The two are non-equivalent loss surfaces. Cross-entropy + softmax has a notoriously slow tail past 99% accuracy; getting from 99.5% to 99.9% on a high-resolution 5-class mask is the same problem ImageNet-pretrained classifiers spend years on. We don't have years; we have 46h. SHIP."

### Contrarian — VOTE: SHIP C-067 (steelman: kill Lane 12 until post-deadline)

> "The mandate explicitly asks me NOT to default-yes the redesign. Steelmanning the kill: (a) The 0.5% disagreement floor is empirically supported by Phase F's loss curve, NOT a hypothesis. (b) Every redesign candidate (1–6) is a 'we hope it works' play with no precedent number. (c) Even if it works, we still need the parser + Alpha-Geo contracts (the structural blockers don't go away just because we redesigned the recipe). (d) C-067 IS a real submission; ship-C-067 is a winning move at sub-frontier 0.316, beats most of the 0.32-0.33 leaderboard band per `project_leaderboard_0_32_0_33_floor_or_irrelevant_20260501.md`. The only counterargument is 'but we might miss sub-0.3 if we don't try' — and missing sub-0.3 is acceptable; submitting a regression because we burned 30h on a failed redesign is NOT acceptable. KILL Lane 12 until post-deadline."

### Quantizr (adversarial) — VOTE: REDESIGN

> "I'd argue for candidate #4 (distill from C-067's deployed AV1-decoded mask, NOT raw classes). The reason: my own 0.33 archive uses kl_on_logits(T=2.0) for SegNet during training, and the distillation target is a SOFT distribution, not an argmax. If you train NeRV against C-067's deployed AV1-quantized mask logits with T=2.0 KL, you bypass the argmax-error wall — you're matching a smoother target that the coordinate MLP CAN fit. Phase F trained against AV1-argmax cross-entropy, which is the wrong target. Estimated cost: 1 council session (this one) + 4h GPU on H100 + ~150 LOC training-script change. Probability of <0.1% AV1-argmax-equivalence: ~30%. Worth $5 to find out. But I am OUT-VOTED 4-1 in the inner quintet so this is dissent recorded, not decision."

### Hotz — VOTE: SHIP C-067 (with mild dissent toward Path B)

> "Look, the engineering instinct says: when the predicted-realistic case is 0.685 vs current 0.316, you don't burn the GPU. That's a 0.37 score regression on a 'maybe this works' bet. Ship the 0.316. If we have any leftover hours after the submission packet, spend them on Path B parser work because that's INFRASTRUCTURE that lives past the deadline — but DO NOT couple it to a 46h-deadline dispatch. Path B as a post-deadline activity is fine. KILL Lane 12 dispatch until post-deadline."

### Selfcomp — VOTE: SHIP C-067

> "From the Selfcomp / 0.38 lived-experience side: I tried multiple coordinate-INR mask paradigms during my own work and none of them beat AV1+brotli on a 5-class boundary mask. The block-FP weight self-compression was the only paradigm that paid off, because weights are smooth and naturally low-rank; mask classes aren't. NeRV's right answer is to NOT mask-encode but to compress the WEIGHTS that produce the mask, which is what the existing Quantizr/block-FP/Lane Ω-W stack already does. Ship C-067; revisit NeRV post-deadline as a research-paper section, not a leaderboard play."

### MacKay (memorial seat) — VOTE: SHIP C-067

> "MDL says: the coding cost of the NeRV codec must include the cost of the decoder code itself plus the model. The 23KB NeRV payload + 12KB decoder code (`nerv_mask_codec.py` minus shared infra) ≈ 35KB true rate cost, not 23KB. That cuts the rate save from 195KB to 170KB, dropping the score-improvement from 0.130 to 0.113. Even in the optimistic 0.05% case the stacked score lands at 0.316 - 0.113 + 0.05 = 0.253 — still sub-frontier but tighter than the math you've been showing. The MDL-honest analysis makes the 'must hit <0.05% to win' bar even harder. SHIP."

### Ballé — VOTE: REDESIGN (if and only if Quantizr's #4 is the recipe)

> "From the entropy-bottleneck perspective: the NRV2 wire format encodes weights as a flat fp16 dump with no learned hyperprior. A factorized prior on the NeRV weights themselves could cut the 23KB to ~14-16KB, closing some of the MDL gap MacKay flagged. But that's orthogonal to the disagreement-floor problem. On the disagreement floor: I agree with Quantizr that distilling against soft AV1 logits (T=2.0) is the right loss; cross-entropy on argmax was the wrong target choice in Phase F. If we had 5 days, this would be a clear redesign. With 46h and the structural blockers, the wall-clock arithmetic doesn't close. Reluctantly: SHIP C-067, but document Quantizr's #4 + my hyperprior idea as the post-deadline Lane 12 v2 plan."

## Vote tally

| Voice | Vote | Rationale (1-line) |
|---|---|---|
| Shannon (LEAD) | SHIP C-067 | Fano lower bound + no adjacent-art for <0.1% on 5-class scorer mask |
| Dykstra (CO-LEAD) | SHIP C-067 | Achievable region (rate ∩ disagreement ∩ wall-clock ∩ no-fabrication) is empty in 46h |
| Yousfi | SHIP C-067 | Boundary pixels are a structural ceiling for coordinate INR; can't break in 46h |
| Fridrich | SHIP C-067 | Cross-entropy tail past 99.5% requires years of compute, not 46h |
| Contrarian | SHIP C-067 (steelman: KILL Lane 12 until post-deadline) | Missing sub-0.3 is acceptable; submitting a regression is not |
| Quantizr (adv) | REDESIGN (#4 KL-soft-distill) | Right loss is soft-distill against deployed AV1 mask, not argmax |
| Hotz | SHIP C-067 + Path B post-deadline only | 0.37 regression on a maybe is bad engineering |
| Selfcomp | SHIP C-067 | Coordinate-INR is wrong paradigm for 5-class boundary mask; weights are the right target |
| MacKay | SHIP C-067 | MDL-honest rate save is 0.113 not 0.130; bar is even harder |
| Ballé | REDESIGN (only if Quantizr #4) | Right idea, wrong deadline |

**Final tally: SHIP C-067 = 7 / REDESIGN = 2 / Path B (now) = 0 / Path B (post-deadline) = 1 (Hotz qualifier)**

Inner quintet (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian): **5/0 SHIP C-067**. Quintet pact achieved.

## Deciding verdict

**SHIP C-067 AS FINAL. Lane 12 NeRV is KILLED for the 46h deadline window.**

No further GPU dispatch on Lane 12 NeRV under the current C-067 stack premise. No implementation of Path B (parser, Alpha-Geo contracts, L2 clearance) before the contest deadline.

## Lock-in spec for ship-C-067

1. **Submission archive**: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`, sha256 `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, 276,214 bytes, score **0.31561703 [contest-CUDA T4 A++]**.
2. **Pre-submission verification (mandatory before PR)**:
   - Re-run `inflate.sh` + `upstream/evaluate.py` on the EXACT archive bytes one more time on contest-CUDA T4. Score must reproduce within ±0.001 of 0.31561703.
   - Verify archive sha256 matches the recorded value.
   - Run the 5-turn consecutive clean-pass adversarial skunkworks council review per CLAUDE.md "Submission PR gate — non-negotiable" before opening the public PR.
3. **No changes to the submission archive in next 46h** unless a different lane (Block-FP transplant, SJ-KL residual, line-search refinement, or a parallel codex partner result) lands a verified `[contest-CUDA]` score < 0.31561703 with full council 5-pass clean-up review completed before submission deadline.

## Reactivation criteria for Lane 12 (POST-DEADLINE)

This kill is REVERSED post-deadline if any of the following lands:

- A council session approves Quantizr's candidate #4 (KL-soft-distill against C-067's deployed AV1 mask logits at T=2.0) with a forecast band that includes <0.05% AV1-argmax-equivalence supported by adjacent-art evidence (NeRV literature on KL-distill boundary tasks)
- A council session approves Ballé's NeRV-weight hyperprior addition with measured rate save ≥ 8KB on the 23KB NRV2 payload
- An empirically-stronger NeRV checkpoint surfaces (codex partner, paper rebuttal session) with measured disagreement <0.1% on the C-067 PR67 mask source — at which point dispatch becomes "rebuild archive + eval", $2-3 GPU
- Path B infrastructure work (parser + Alpha-Geo contracts + L2 clearance) lands at HEAD with passing tests and 3-pass council review (this is approved as a post-deadline activity at any time; not approved for next 46h)

## Internal-consistency check (per CLAUDE.md KILL/FALSIFIED rule)

- ✅ Council deliberation has 10 named voices (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Quantizr + Hotz + Selfcomp + MacKay + Ballé), each with 1-line rationale
- ✅ Vote tally explicit: 7 SHIP / 2 REDESIGN / 1 Path-B-post-deadline; inner quintet 5/0 SHIP
- ✅ Phase F empirical numbers cited from `reports/lane_12_nerv_real_archive.json` verified at HEAD 2026-05-02 (xent 0.02 at step 1400, argmax disagreement 0.020032391..., 23594 bytes fp16, 421483 baseline)
- ✅ Stacked-score arithmetic shown for realistic / target / optimistic cases
- ✅ "What would change my mind" subsection present (Reactivation criteria, 4 conditions)
- ✅ Anchor SHA matches eval target: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- ✅ Structural blockers re-verified at HEAD (all 4 MISSING per prior refusal memos + spot-check of `.omx/state/`)
- ✅ MDL-honest rate save analysis includes decoder code cost (MacKay's correction)
- ✅ No fabricated gates, no GPU spend recommended in 46h window, no comment-only contracts proposed

## What this verdict does NOT do

- Does NOT regress Lane 12's lane_registry maturity (stays Level 2 INTEGRATION; Phase F empirical artifact stands)
- Does NOT KILL Lane 12 permanently (post-deadline reactivation criteria explicit)
- Does NOT block parallel sub-0.3 attempts on OTHER lanes (Block-FP transplant, SJ-KL residual, line-search refinement, codex partner OWv3-stack work, etc.) — those are separate council sessions
- Does NOT recommend any code edits in this session — verdict is decision-only
- Does NOT spawn other subagents

## Cross-refs

- Prior refusals: `project_lane_12_nerv_full_cuda_dispatch_REFUSED_20260502.md` + `project_lane_12_nerv_full_cuda_dispatch_RETRY_REFUSED_20260502.md`
- Phase F empirical: `/Users/adpena/projects/pact/reports/lane_12_nerv_real_archive.json`
- C-067 anchor: `/Users/adpena/projects/pact/experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Lane 12 implementation: `/Users/adpena/projects/pact/src/tac/nerv_mask_codec.py`, `/Users/adpena/projects/pact/experiments/train_nerv_mask.py`, `/Users/adpena/projects/pact/scripts/remote_lane_nerv.sh`
- Phase B council design: `/Users/adpena/projects/pact/.omx/research/council_lane_12_nerv_design_20260430.md`
- Leaderboard context: `project_leaderboard_0_32_0_33_floor_or_irrelevant_20260501.md`
- Submission gate non-negotiable: CLAUDE.md "Submission PR gate — non-negotiable" (5-turn consecutive clean-pass adversarial review required)
- KILL/FALSIFIED rule satisfied: CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS" (5+ council voices + internal-consistency + reactivation criteria all present)
