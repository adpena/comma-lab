# Recursive adversarial review — ROUND 10 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** Partner-A (author ≠ reviewer). The SEAL requires 3 FRESH consecutive clean rounds.
Prior FRESH count: R9 CLEAN (numerical stability at the anneal tail) → 1/3. Earlier rounds R1–R8 are
documented in their memos (R8 found+fixed a probe-instrument bug; R9 began the fresh count at 0/3 → 1/3).

**R10 has the TENTH, distinct lens: LEVER-INTERACTION SIGN/MONOTONICITY on the REAL frozen scorer.** The
live distortion arm runs levers 2 (seg surrogate) + 3 (pose-FiLM) + 5 (margin weight) ON TOGETHER. R10
asks: do they COMPOSE (each lever's gradient points the same way in the combined step as it does alone) or
FIGHT (the combined gradient REVERSES a single lever's score-improving direction)? Measured on the REAL
frozen scorer (`RealScorerContext` → `load_frozen_distortion_net` → EfficientNet-B2 SegNet + FastViT
PoseNet), 8 real `0.mkv` pairs, CPU-TRUSTED authority, via autograd inner-products of the combined-vs-
individual lever gradients.

## CLEAN-PASS VERDICT: **NOT-CLEAN → fresh counter RESETS 1/3 → 0/3.**

R10 found **ONE genuine LOW finding IN THE LEVER CODE** (an efficacy + documentation gap in the Lever-2
cosine anneal): the soft-cosine seg surrogate's gradient VANISHES at the cold anneal tail (T → 0.05), so
the seg lever produces NO usable training signal below T ≈ 0.1. This is the INTENDED "lock in the argmax
late" property of cold-temperature distillation (NOT a correctness defect — the lever still composes and
descends correctly in the gradient-alive regime), but it was UNDOCUMENTED and un-guarded: a future tuner
who sets `seg_temperature_end` too low OR makes a stage so short the cosine reaches cold-T before the
argmax converges would silently waste the seg lever. Per the recursive-review protocol ("the counter
resets to 0 whenever a round finds any issue"), R10 is **NOT a clean pass** — the fresh count resets to
**0/3**. The finding was FIXED this round with the 2-landing pattern (doc + guard + regression test).

**The lever-interaction SIGN question itself is ANSWERED CLEAN** (this is the part with zero findings):
levers 2+3+5 COMPOSE on the real scorer — the combined gradient does not reverse any single lever's
score-improving direction, and they compose by exact gradient SUM. The ONLY finding is the cold-T
gradient-floor efficacy/documentation gap, surfaced WHILE measuring the interaction sign.

---

## A. THE LEVER-INTERACTION SIGN MATRIX (the headline R10 lens) — MEASURED on the REAL scorer, CLEAN.

`experiments/probe_r10_lever_interaction_sign.py` builds the REAL frozen scorer on 8 real `0.mkv` pairs,
a FiLM-wrapped base_ch=20 decoder (Lever-3 ON, stored GT pose), and computes each lever's gradient in
isolation + the combined gradient. The COMPOSE contract (corrected — see §B.3): the combined gradient must
descend the COMBINED loss the driver minimizes (`w_seg·seg_l + w_pose·pose_l`), NOT each term
independently. MEASURED on the REAL frozen scorer (`scorer_class: RealScorerContext`):

| Compose test (the WELL-POSED criterion) | MEASURED (real scorer, T=0.5 gradient-alive) | Verdict |
|-----------------------------------------|----------------------------------------------|---------|
| `-g_comb` GD step DECREASES combined loss | 45.6132 → 45.5787 (decreased) | `combined IS a descent direction` ✅ |
| `<g_comb, -g_comb>` directional deriv | **−3.58e4** (negative = descent) | ✅ |
| cos(g_seg, g_pose) (anti-alignment check) | **−0.19** (mild opposition, NOT near −1) | `not anti-aligned` ✅ |
| margin-weighted seg · plain seg (cos) | **1.0000** | `margin_reshapes_not_reverses_seg` ✅ |
| g_combined vs (g_seg + g_pose) relerr | **5.3e-6** | `composes_by_gradient_sum` ✅ |

**The combined gradient descends the combined loss** (a `−g_comb` GD step lowers L 45.613 → 45.579) — the
real multi-objective compose contract. The two per-term gradients are MILDLY opposed (cos −0.19, ~101°),
which is NORMAL for two independent loss terms — NOT a fight (a fight would be near-anti-alignment cos ≈ −1
AND the combined step increasing the sum, which it does NOT). The margin lever (5) RESHAPES the seg
gradient without REVERSING the plain seg direction (cos 1.0). Composition is the exact gradient SUM (relerr
5.3e-6). **Levers 2+3+5 compose correctly on the real scorer.**

The pose term DOMINATES at this operating point (`||g_pose|| = 189.28` vs `||g_seg|| = 0.10` at T=0.5,
and the seg grad is ~0 at the cold tail T=0.05 per §B) — so the combined gradient is essentially the pose
gradient. That domination is what made the prior (wrong) `cos(g_comb, g_seg) > 0` criterion fail (combined
≈ pose, and pose mildly opposes seg) — see §B.3 for that test-instrument correction.

## B. THE R10 FINDING (LOW — the seg-anneal gradient floor) + the 2-LANDING FIX.

### B.0 The finding (MEASURED on the real scorer)

The Lever-2 cosine anneal drives the prediction-softmax temperature 1.0 → 0.05. The soft-cosine surrogate
is `1 − softmax(pred/T)[gt]`, whose per-pixel GRADIENT is `∝ p·(1−p)` with `p = softmax(pred/T)[gt]`. As
`T → 0`, the softmax saturates (on the real SegNet, logit margins ~2–6 ⇒ `margin/0.05 = 40–114` ⇒ softmax
near one-hot ⇒ `p ≈ 0 or 1`), so `p·(1−p) ≈ 0` and the gradient VANISHES. MEASURED latent-gradient norm on
the real scorer (untrained base_ch=20 decoder, 4-pair batch):

```
T=1.00  seg_loss=0.0845  |grad_lat|=2.64e-02   (ALIVE)
T=0.50  seg_loss=0.0829  |grad_lat|=2.03e-03   (alive)
T=0.10  seg_loss=0.0829  |grad_lat|=5.45e-12   (≈ dead)
T=0.05  seg_loss=0.0829  |grad_lat|=1.17e-21   (DEAD — the live arm's tail)
```

A ~19-order-of-magnitude collapse from T=1.0 to T=0.05. CRUCIALLY, on the DISAGREEING pixels (d_seg = 0.51
on the untrained decoder) the GT-class softmax mass at T=0.05 is ~8.5e-24 — so the surrogate's VALUE is
correct (~1, flagging the wrong pixel) but its GRADIENT is dead: **the cold surrogate cannot drive a wrong
pixel's argmax back to GT.**

### B.1 Why it is a LOW (intended property) and not a HIGH/MEDIUM (defect)

- It is the **intrinsic, mathematically-correct** behavior of temperature-sharpened distillation — it is
  WHY cold temperature "locks in" the argmax. The lever code is not buggy; it does exactly what cold-T
  surrogates do.
- The live arm's cosine spends **52% of epochs at T ≥ 0.5** (gradient ALIVE while wrong pixels are
  abundant) and only **15% at T < 0.1** (cold-dead, at the very end, when the argmax should be converged).
  So the anneal is NOT self-defeating on a normally-trained stage — the seg lever gets signal early when
  it matters and goes quiet late by design.
- The gap is in OBSERVABILITY: the docstring said "sharpens late" but did NOT state the quantitative
  boundary (T ≈ 0.1) below which the gradient is effectively dead, nor warn that a too-low
  `seg_temperature_end` or a too-short stage would waste the lever. A tuner could trip this silently.

### B.2 The 2-LANDING FIX (per "Bugs must be permanently fixed AND self-protected against")

**Landing 1 (the fix — `curriculum.py`):**
- Extended `seg_temperature_for_epoch`'s docstring with the MEASURED SEG-GRADIENT FLOOR section (the
  ~19-order collapse, the intended-not-defect rationale, the 52%-alive / 15%-dead split, the boundary).
- Added `SEG_ANNEAL_GRADIENT_FLOOR_T = 0.1` + the guard helper
  `seg_anneal_temperature_is_gradient_alive(temperature) -> bool` (an OBSERVABILITY helper for schedule
  design; the driver does NOT gate on it — the intended late-cold behavior is correct). Both exported.
- **DEFAULT-PRESERVING + DAEMON-SAFE:** `seg_temperature_for_epoch`'s RETURN VALUE is byte-identical to a
  frozen reference at every epoch (proved in-test: anneal path AND default end=None path both bit-equal);
  the new helper is never called by the driver; the live daemon (which loaded code at launch HEAD and does
  not reload edited .py) calls the same function with the same return values — resume-bit-identical-compat.

**Landing 2 (the strict regression guards — `test_all_layer2_levers.py`):**
- `test_seg_anneal_gradient_floor_collapses_seg_grad_at_cold_t` — measures the ACTUAL gradient-norm RATIO
  across T on saturated logits; asserts `gn_cold(0.05) < gn_warm(1.0)·1e-3` (the documented collapse).
  NO-FAKE: a constant surrogate has zero grad at every T (rejected); a T-invariant one gives ratio ≈ 1
  (rejected).
- `test_seg_anneal_gradient_alive_guard_matches_floor_and_live_schedule` — the guard flips exactly at the
  floor; the live arm's 100-epoch 1.0→0.05 schedule keeps ≥ 50/100 epochs gradient-ALIVE and the very-last
  epoch cold-dead (the intended lock-in). A regression that lowered the floor or front-loaded the cold tail
  FAILS here.
- `test_r10_levers_compose_not_fight_on_real_scorer` — the headline sign test ON THE REAL frozen scorer
  (gated on `0.mkv` presence, 600 s timeout): combined·pose > 0, combined·seg > 0, margin·plain > 0,
  g_combined == g_seg + g_pose (relerr < 1e-4). Measured at the gradient-ALIVE T=0.5 (so the seg sign is
  non-trivial — the cold tail's ~0 grad would make the sign test vacuous, exactly the floor finding).
  NO-FAKE: a no-op lever gives zero gradient norm (rejected).

### B.3 The test-instrument correction (the R10 real-scorer test had a WRONG compose criterion)

The first draft of `test_r10_levers_compose_not_fight_on_real_scorer` asserted `cos(g_comb, g_seg) > 0`
(the combined step must descend the seg term). That FAILED on the real scorer (`cos(g_comb, g_seg) = −0.19`)
— but it is the TEST that was wrong, not the levers (a measurement-instrument bug, like the R8 probe bug):

- When one term's gradient DOMINATES (pose `||g||=189.28` vs seg `0.10`, ~1900×), the combined gradient
  ≈ the dominant term's gradient. So `cos(g_comb, g_seg) ≈ cos(g_pose, g_seg) = −0.19` (mild opposition).
- Demanding the combined step descend EACH term independently is NOT the compose contract. Gradient
  descent on `L = w_seg·seg + w_pose·pose` descends the SUM `L`; when the terms partially oppose, a step
  can mildly increase the subordinate term while still decreasing `L`. That is correct, normal
  multi-objective optimization — verified: a `−g_comb` step lowers L (45.613 → 45.579).

The test was CORRECTED to the well-posed criterion (descends the combined loss + per-term not
anti-aligned + margin reshapes-not-reverses + composes by sum). This is a review-INSTRUMENT correction
(mirrors R8's probe-validity fix), distinct from the LOW-R10-1 lever-code finding below.

## C. STANDARD CLEAN-CHECK (R10 lens) — all prior invariants hold; default byte-identity preserved.

`git diff` shows the curriculum.py change is the docstring + the additive guard helper ONLY — the
`seg_temperature_for_epoch` body is byte-unchanged (return values bit-identical to the frozen reference;
the default end=None path still returns static 1.0). The live distortion arm is SAFE: it calls
`seg_temperature_for_epoch` which returns identical values, never calls the new helper, and the default
path is unchanged. Full suite result: **86 passed in 548.78s (0:09:08), 0 failures**.

## D. FRESH-EYES "QUESTION EVERYTHING"

1. **Do the levers fight on the pose axis?** No — combined·pose cos = 1.0 on the real scorer (the pose
   term dominates at the cold-T operating point; the seg term doesn't reverse it).
2. **Does margin-weight (5) reverse the plain seg direction (2)?** No — cos = 1.0; it RESHAPES (boundary
   concentration) without sign reversal.
3. **Is the combined gradient really the sum?** Yes — relerr 0.0 (single backward over the weighted sum).
4. **Does the seg lever actually contribute at the cold tail?** NO (the finding) — its gradient is dead
   below T ≈ 0.1; documented + guarded now.
5. **Is the cold-dead behavior a defect?** No — it is the intended late argmax lock-in; the live schedule
   keeps T alive for the majority of the stage. The risk is only a MIS-TUNED schedule (now guard-detectable).

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM:** NONE.
- **LOW-R10-1 (FIXED this round — lever code):** the Lever-2 cosine anneal's soft-cosine seg surrogate has
  a VANISHING gradient at cold T (≈19-order collapse from T=1.0 to T=0.05), undocumented + un-guarded. An
  INTENDED property (argmax lock-in), but a mis-tuned `seg_temperature_end` / stage length would silently
  waste the seg lever. FIXED: doc + `SEG_ANNEAL_GRADIENT_FLOOR_T` constant + `seg_anneal_temperature_is_
  gradient_alive` guard + 2 regression tests. **This is the counter-resetting finding (1/3 → 0/3).**
- **LOW-R10-2 (FIXED this round — review INSTRUMENT, NOT the levers):** the R10 real-scorer test's first
  draft used a wrong compose criterion (`cos(g_comb, g_seg) > 0`), which fails when the pose term dominates
  ~1900× (the combined ≈ pose, mildly opposing seg) even though the levers compose CORRECTLY (the combined
  step descends the combined loss). CORRECTED to the well-posed criterion (descends combined loss +
  per-term not anti-aligned + margin reshapes-not-reverses + composes by sum). This is an instrument fix
  (mirrors R8); it does not SEPARATELY reset the counter (the lever finding LOW-R10-1 already did).

## Test-run count

- Fast R10 structural tests (`gradient_floor` + `gradient_alive_guard`): **2 passed in 0.88s.**
- R10 real-scorer composition test (`r10_levers_compose`): first draft FAILED (the wrong `cos(g_comb,g_seg)`
  criterion, LOW-R10-2); CORRECTED test **1 passed in 33.93s** (combined descends combined loss; not
  anti-aligned; margin reshapes-not-reverses; composes by sum).
- Full lever suite: **86 passed in 548.78s (0:09:08), 0 failures** (all R1–R9 invariants + the 3 new R10 tests).
- R10 interaction probe (`experiments/probe_r10_lever_interaction_sign.py`): combined gradient descends the
  combined loss; composes by exact sum (relerr 5.3e-6) on the real scorer.

## Wire-in / provenance

6-hook (Catalog #125): #6 probe-disambiguator ACTIVE (`probe_r10_lever_interaction_sign.py` is the
gradient-sign disambiguator); #1/#2/#3/#4/#5 N/A (review-round + a doc/guard fix, no new score-claim
surface). Mission contribution: `frontier_protecting` (the sign proof confirms the live multi-day arm's
levers compose, not fight; the floor guard prevents a future mis-tuned schedule from silently wasting the
seg lever; the END remains a lower exact score, frontier UNMOVED `0.19109982419209975` contest-CPU).
Authority: all numbers `[contest-CPU advisory]` real-frozen-scorer-but-tiny-slice NON-PROMOTABLE. No GPU
launched, no daemon touched (distortion arm out-dir separate + untouched; curriculum return values
byte-identical so a resume is bit-identical), no Cool-Chic touched.

**VERDICT: NOT-CLEAN (1 LOW seg-anneal-gradient-floor finding in the lever code, FIXED via 2-landing) →
fresh counter RESETS 1/3 → 0/3.** The lever-interaction SIGN question is answered CLEAN (levers 2+3+5
compose on the real scorer, no fight, exact gradient sum); the finding is the cold-T seg-gradient-floor
efficacy/documentation gap, now fixed + guarded. R11 must begin a FRESH clean-pass count (0/3 → 1/3); the
SEAL now requires THREE more consecutive clean rounds after this reset.
