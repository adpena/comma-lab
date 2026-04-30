# Council Round 1 — Lane 19 Logit-Margin Loss

**Date:** 2026-04-30
**Round:** 1 of 3 clean-pass gate
**Status:** REVIEW IN PROGRESS

---

## Files under review

- `src/tac/losses_logit_margin.py` (~340 LOC after Phase B additions)
- `src/tac/tests/test_losses_logit_margin.py` (~470 LOC, 16 tests)
- `src/tac/preflight.py` (Check 93 added near L14474; AST scanner)
- `src/tac/tests/test_preflight_logit_margin_threshold.py` (11 tests)
- `src/tac/experiments/train_renderer.py` (3 add_argument blocks; aux-loss block)
- `src/tac/profiles.py` (`LANE_19_LOGIT_MARGIN` profile + registry entry)
- `scripts/remote_lane_19_logit_margin.sh` (canonical 4-stage dispatch)
- `reports/lane_19_logit_margin_local_smoke.json` (smoke result)

---

## Rotating perspective: Yousfi (contest design)

**Concern:** The `compute_segnet_logit_margin_aux` calls `segnet(fs_in)` for student logits. The SegNet's `preprocess_input` resizes to (512, 384). Where in the call stack does the rendered_pair already match this? Looking at the train_renderer.py call site at the new aux block: `rendered_pair` shape is `(B, 2, H, W, 3)` where H/W comes from training resolution. SegNet's preprocess_input handles the resize internally so this should work — but the OUTPUT of segnet() is at the resized resolution (512, 384 → after stride-2 stem → 256, 192). The `logit_margin_loss` in the helper is computed on those output logits at 256×192. The teacher's argmax (also at 256×192) is the GT.

**Verdict:** OK. Both student and teacher run through identical preprocess_input + forward. The argmax is taken at the same resolution. The fragility weights are computed at the same resolution. No mismatch.

**Sub-concern:** The `segnet.preprocess_input(fx)` call for the student is OUTSIDE the `with torch.no_grad():` block (good — we want gradient). The teacher path is inside no_grad. But I notice in the existing JBL code (train_renderer.py L3141) the student path also has `fs_in = segnet.preprocess_input(fx)` OUTSIDE no_grad — same pattern. Consistent.

**Yousfi: GREEN.**

## Rotating perspective: Fridrich (steganalysis)

**Concern:** The fragility weight formula `(threshold - margin) / threshold` in fragility_weights() assumes the logit scale is roughly comparable to threshold. SegNet outputs raw logits — could be wildly different scale across runs / temperature settings.

**Counter:** This is exactly why threshold is configurable per-profile. The default 1.0 in profile is paired with no temperature scaling on logits. If a future profile adds temperature scaling (e.g., distill T=2.0), the threshold should be scaled accordingly.

**Action item:** Documentation in council memo §3 already mentions "typical values 0.5-2.0 depending on logit scale" but should also note that if logits are temperature-scaled, threshold should track. Decision: ADD a note in `LANE_19_LOGIT_MARGIN` profile comment to flag this for future profile authors.

**Fridrich: YELLOW** — fix above before round 2.

## Rotating perspective: Quantizr (empirical pragmatism)

**Concern:** I shipped 0.33 with `kl_on_logits(T=2.0)` — distillation. The Lane 19 design memo says "implement as SUPPLEMENT to KL distill, not a replacement". Confirmed in profile: `LANE_19_LOGIT_MARGIN` inherits `kl_distill_weight=0.002` from V3-annealed-kldistill — KL distill stays on alongside Lane 19. The Lane 19 weight is 0.1, KL distill is 0.002 — these two operate at very different scales. Need to verify they don't conflict.

**Sub-investigation:** The KL distill uses `kl_distill_segnet_only(rendered_pair, gt_pair, segnet, ...)` and divides per-pixel by H*W (per the bug-fix mentioned in train_renderer.py L2722-2727). So KL contributes ~0.025 at weight=1.0 → ~5e-5 at weight=0.002. Lane 19 logit_margin loss is computed via F.cross_entropy reduction="none" then mean-weighted by fragility — typical magnitude on partial-confidence batch ~0.1-0.5. At weight=0.1, it contributes ~0.01-0.05. So Lane 19 dominates KL distill by ~3 orders of magnitude. KL distill is essentially noise compared to Lane 19 in this stack.

**Action item:** Either (a) Lane 19 should DISABLE KL distill (set kl_distill_weight=0 in profile) since it's drowned out, OR (b) Lane 19 weight should be reduced so KL distill is still informative. Council recommends (a) — Lane 19 IS the SegNet-only distillation; turning KL distill off is cleaner. **DECISION: set kl_distill_weight=0.0 in LANE_19_LOGIT_MARGIN profile.**

**Quantizr: RED on current profile config.** Fix needed before round 2.

## Rotating perspective: Hotz (engineering simplicity)

**Concern:** The `compute_segnet_logit_margin_aux` does TWO SegNet forward passes per training step (student + teacher). The existing KL distill block ALSO does two forward passes. So when `kl_distill_weight > 0` AND `logit_margin_weight > 0` together, we do 4 SegNet forward passes per step.

**Optimization:** The teacher logits (gt_pair) are deterministic per epoch — could cache them. But this is profile-level optimization, not correctness. Skip for now; flag for Phase 3.

**Sub-concern:** The aux block uses `from tac.losses_logit_margin import compute_segnet_logit_margin_aux` INSIDE the training loop (per-step import). This is the same pattern as the JBL block (`from tac.losses_jbl import combined_jbl_distill_loss`). Python caches the import — no real perf cost — but it would be cleaner at module scope.

**Action item:** Move import to module top OR at least to top of train() function. Decision: leave inside the if-branch for now (matches existing JBL pattern; no urgency).

**Hotz: GREEN with minor performance flag.**

## Rotating perspective: Selfcomp (architectural fit)

**Concern:** The `LANE_19_LOGIT_MARGIN` profile inherits `DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL` which uses `mask_half_sim_prob=0.5` annealed + `use_zoom_flow=True`. Lane 19 is a TRAINING-TIME loss — should work with both half-frame and full-frame archives. The remote script Stage 2 builds a half-frame archive, which matches the profile.

**Counter:** Could Lane 19 also be tested standalone on a full-frame archive (cheaper test that isolates the wedge)? Yes — but the goal is to compete with Lane G v3 1.05, which IS half-frame. Half-frame is the right test bed.

**Sub-concern:** The training run will produce TWO different effects: (1) Lane 19 margin-loss aux → boundary improvement, (2) the inherited annealed half-frame curriculum from V3. If Lane 19 wedge is small (-0.05), the variance from half-frame annealing might mask it. Should we run a CONTROL with `loss_mode="standard"` and same seed=89 to isolate the Lane 19 contribution?

**Action item:** Add a SISTER profile `LANE_19_LOGIT_MARGIN_CONTROL` that's identical EXCEPT `loss_mode="standard"` and `logit_margin_weight=0.0`. Run both → diff isolates Lane 19. Cost: 2× $1.50 = $3 instead of $1.50.

**Decision:** DEFER control to a follow-up if first run looks promising. Otherwise we're spending double for an A/B that may not be needed (Lane G v3 1.05 is the implicit control — same arch + same anneal schedule + standard loss). The seed difference (89 vs 43) is a confound, but Yousfi's preferred way is to run multiple seeds anyway.

**Selfcomp: GREEN with deferred A/B note.**

## Rotating perspective: Contrarian (skeptical of every claim)

**Concern 1:** The council §2 said "Lane 19 silently abandons confident-wrong pixels — but underlying scorer_loss catches them". Verifying: `scorer_loss` in losses.py uses standard CE on logits (no fragility weighting). So yes, confident-wrong pixels contribute large CE there. But what's the WEIGHT of scorer_loss vs Lane 19 aux? Looking at profile: `segnet_weight` defaults from profile (Lane G v3 anchor; let me check).

**Counter (Yousfi):** segnet_weight in LANE_G_V3 profile = 100 (confirmed in profiles.py). Lane 19 aux = 0.1. So scorer_loss is 1000x stronger than Lane 19 — Lane 19 is a tiny additive nudge, NOT the dominant signal. This means for confident-wrong pixels, scorer_loss contributes ~100x while Lane 19 contributes 0. For boundary pixels, scorer_loss contributes ~100x (CE) while Lane 19 contributes ~0.1×CE = small extra nudge. **Lane 19 is barely felt by the optimizer.**

**Critical finding:** The Lane 19 weight 0.1 is too small relative to scorer_weight 100. To actually move SegNet behavior toward the boundary preference, Lane 19 weight needs to be at least 10-50% of effective scorer_loss contribution at boundary pixels. Current contribution: 0.1 / 100 = 0.001 = noise.

**Action item: RAISE LANE_19_LOGIT_MARGIN logit_margin_weight from 0.1 to 10.0** so the aux is meaningfully felt against scorer_loss at boundary pixels. (Lane 19 weight 10 × CE 0.5 = 5 vs scorer_loss ~100 × CE_avg 0.05 = 5 — comparable order of magnitude; the boundary preference is now actually expressed.)

**Concern 2:** What if Lane 19 weight 10 OVERWEIGHTS boundary at the cost of confident-wrong? The Contrarian's earlier concern was confident-wrong → 0 Lane 19 → optimizer might let it slide. But scorer_loss at weight 100 still applies. If Lane 19 = 10, total at confident-wrong pixel = 100*CE; at boundary = 100*CE + 10*CE_weighted ≈ 110*CE. So boundary gets +10% extra preference. Reasonable.

**Concern 3:** The threshold=1.0 default. If logits typically range -5 to +5, margin can be 0-10. Threshold 1.0 means "weight = 1 when margin ≤ 0; linearly to 0 at margin = 1". So only the most-ambiguous ~10% of pixels get any signal. Is this too narrow? Should be empirically tuned.

**Action item:** Document threshold tuning guidance (threshold should be the 10th-percentile margin observed at training start, NOT a fixed default). Add a note to the profile comment.

**Contrarian: RED.** Two fix items:
1. Raise `logit_margin_weight` from 0.1 to 10.0 (or document why 0.1 is correct after weight-balance analysis).
2. Disable KL distill in the Lane 19 profile (per Quantizr's finding).
3. Add threshold tuning guidance.

---

## Round 1 verdict

**3 issues found. Counter resets to 0.**

**Issues:**
1. **Quantizr: RED.** KL distill weight 0.002 is drowned out by Lane 19 weight 0.1 in the profile — set `kl_distill_weight=0.0` to make Lane 19 the SegNet-only distillation. (The KL distill's contribution is ~5e-5 vs Lane 19's ~0.01, three orders of magnitude smaller.)
2. **Contrarian: RED.** `logit_margin_weight=0.1` is too small relative to `segnet_weight=100`. Raise to 10.0 so the boundary-preference is actually felt. (At 0.1, the aux contributes 0.1% of scorer_loss — noise floor.)
3. **Fridrich: YELLOW.** Profile comment should flag threshold-temperature interaction for future profile authors.

**Action plan for fix-up:**
- Edit `src/tac/profiles.py` LANE_19_LOGIT_MARGIN: `kl_distill_weight=0.0`, `logit_margin_weight=10.0`, expanded comment.
- After fix, run Round 2.
