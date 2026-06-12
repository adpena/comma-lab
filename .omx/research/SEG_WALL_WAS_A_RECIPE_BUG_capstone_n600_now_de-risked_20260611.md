# THE TURNING POINT: the d_seg "capacity wall" was a RECIPE BUG — the capstone paid n600 (#90) is now a DE-RISKED, justified pointer-mover (2026-06-11)

**Authority:** two independent agents, corroborating; measured on the real `modules.py` SegNet (torch-CPU,
live AND EMA agree → no shadow-lag), `[macOS-CPU advisory]`, NO MPS, $0, no paid dispatch. Frontier UNMOVED
0.19109982 [contest-CPU], 177,169 B. This UNBLOCKS the retrain; it is not itself a pointer move.

## What was measured (the decisive controlled A/B — `a28f8a9c`, commits a9888191c + ff77c3419)

Architecture HELD FIXED (base_ch=20, tie_depth=2, n=8, stored_latent = the capstone basis = the #90 target).
Recipe varied ONLY by the muon_lr the bug controls:

| arm | stage | muon_lr | d_seg (live, real SegNet) | vs init 0.50727 |
|---|---|---:|---|---|
| BUGGY | stage 1 CE | 2e-4 + 100% clip | 0.50727 | **0% — FROZEN at init** |
| FIXED | stage 1 CE | 0.03 | 0.06647 | 7.6× descent |
| FIXED | stage 2 tau_softplus | 0.03 | 0.01646 | 30×, **still falling** |

## The bug (the thing we never internalized from PR95)

- **BUG-A (decisive):** `CapstoneTrainer.configure_stage` silently DROPPED the working `muon_lr=0.03`,
  rebuilding `opt_config` from the StageSpec's torch-tuned `muon_lr=2e-4` + `grad_clip_muon=1.0` → **150×
  smaller LR + 100%-clip-every-step → ~zero effective weight movement.** The buggy curriculum crawled
  (`c1prime` 0.0097) while the SAME arch at muon_lr=0.03 reached 0.0037 — "worse than no curriculum," the
  signature of a recipe bug.
- **BUG-B:** cosine `eta_min_ratio` floored LR at 0.1667 against `adamw_lr` (not muon_lr), prematurely
  halting descent → the "muon asymptotes ~0.0025 (capacity!)" reading was partly this LR-floor artifact.
- **BUG-C:** "smooth_disagreement raises d_seg" is a consequence of A+B (a vanishing-gradient surrogate
  mis-staged onto a throttled basis), not a standalone bug.

**Independent corroboration (`a173958b`, commit d52a91b0d):** a step-level diff of our MLX port vs the
runnable vendored PR95 original proved the port is **arithmetically faithful** (d_seg identical, gradients /
AdamW updates / latents match to fp32/bf16 epsilon; only Newton-Schulz bf16 epsilon diverges, faithful to
PR95's own cast). So the wall is NOT a port arithmetic bug — it is the recipe SCHEDULE/wiring. Two methods,
same verdict. (Also surfaced: #82's "torch-parity-GATED" was only COMPONENT-level, never a whole-step diff —
which is why it never caught this; #76's live-SegNet loop IS genuinely faithful.)

## What this FALSIFIES (Catalog #307, IMPLEMENTATION-LEVEL — paradigm intact)

The following verdicts ALL rested on the throttled recipe and are implementation-level falsified:
- the capstone base_ch=20 "capacity wall" / "muon asymptotes 0.0025" (capstone memo's adversarial correction);
- my "every $0 door routes through a retrain — and the retrain would ALSO wall / is a FAKE spend" framing
  (`post_hoc_levers_exhausted_...` + the #90 reframe). The retrain was being SABOTAGED by the recipe bug;
  with the fix, a faithful PR95-class basis DESCENDS toward the basin.

**Scope honesty:** this is proven for the CAPSTONE basis (base_ch=20, the #90 target). The Cool-Chic
COMPACT-synth 0.014 wall used a DIFFERENT harness/architecture (16-ch hinge) and is a separate question — do
NOT claim it falsified here. What IS proven: the #90 target basis is not capacity-walled; its wall was the
optimizer-LR wiring.

## What is still OPEN (honest — not overclaimed)

The fix unblocks DESCENT (0.507 → 0.066 → 0.0165, still falling at stage 2); it has NOT been shown to REACH
the 5.6e-4 basin. The full corrected multi-stage curriculum at n600 is needed to test basin-reaching, and
torch-CPU n48 throughput (~25 s/step) makes the full run infeasible locally — **it needs a paid GPU at n600.**

## The DECISION (operator-gated — the de-risked paid pointer-mover)

Task #90 (capstone paid n600 PR95-scale) is REFRAMED for the third time, now CORRECTLY: it is no longer a
"fake spend into a capacity wall" (that verdict was the recipe bug). It is the **MVP-first-de-risked,
justified paid pointer-mover** — the $0 A/B has PROVEN the fixed recipe unblocks the descent the wall was
hiding. Per GOAL "spend the Modal budget to BUY exact rows" + "a fail-closed paid eval on a real candidate
that beats the advisory bar is the RIGHT default (decide-don't-defer)": a paid n600 with the FIXED recipe
(full corrected AdamW-stages-1-7 + Muon-stage-8 curriculum, muon_lr=0.03) → does d_seg reach the basin at
scale → byte-close → paired CPU+CUDA exact eval is now the right spend. The $100 grant exists for exactly
this. Operator decision: commit the $100 to the de-risked capstone n600?

## Trajectory COMPLETED (`a28f8a9c` final, commit 5133fe82a) — strengthens the verdict

The fixed arm finished all 3 stages (el=1172s): **d_seg 0.50727 → 0.06647 → 0.01646 → 0.01197** (stage 1
CE → stage 2 tau_softplus → stage 3 smooth), 42× descent, still falling. **The decisive strengthening:
stage 3 `smooth_disagreement` REFINED d_seg (0.0165 → 0.0120) under the correct recipe — the SAME stage
that RAISED d_seg in the buggy `c1prime`.** This directly refutes BUG-C as anything but a throttle
consequence: the surrogate was never the problem; the dropped muon_lr + LR-floor were. Monotonic descent
across all three stages on a held-fixed small basis = the cleanest possible "not capacity-walled" signal.
Basin (5.6e-4) still not reached at n8 — that remains the paid n600 measurement.

### Compute-heavy probe reconciled (`a6b41ea9`, commit e3459719d) — confirms "recipe, not compute"

The parallel compute-heavy weight-tied-decoder probe (K inflate passes, hinge recipe, n8/35ep) found d_seg
**compute-INVARIANT** (best 0.0158 at K=4, degrades at K≥8) — more inflate-time compute does NOT lower
d_seg. Its own "hardens the capacity verdict / pivot to waterfiller" framing is OUTDATED (it ran in
parallel, never saw the recipe-bug result). Reconciled correctly: its 0.0158 plateau is the SAME
~0.012–0.016 band a28f8a9c's fixed-recipe single pass reached — so the honest reading is **compute is not
the lever, the recipe is.** The unified cross-agent verdict: the d_seg wall is **not capacity** (a28f8a9c
falsified), **not compute** (a6b41ea9 compute-invariant), but **recipe + scale** — the fixed recipe
unfreezes the descent; reaching the basin is the epoch-budget question (PR95 = 29,650 epochs at n600 = the
paid run). (Open lower-prior residual: implicit coordinate-MLP "capacity from compute" untested.)

## Bottom line (non-sycophantic)

For the first time this session, the d_seg wall has a specific, fixable name and the fix is MEASURED to
work at n8 — the small basis descends 30× where the buggy recipe froze it. The frontier is still UNMOVED at
0.19110 and this did NOT move a row; what it did is dissolve the false "capacity wall" that made the paid
pointer-mover look like a fake spend, and turn #90 into a de-risked, justified paid exact-row. The honest
next step is the operator's call on the $100 n600 — the $0 work that gates it is done.
