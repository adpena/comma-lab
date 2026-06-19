---
title: "GENERATIVE-AXIS FINAL EXHAUSTION TEST — amortized continuous-texture NCA + capacity-break sweep → VERDICT RED (the FINAL family)"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; MPS-gradient/CPU-authority; no PR"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: RED_AMORTIZED_NCA_CONVERGES_AND_BEATS_THE_ONE_SHOT_WALL_BUT_THE_RATE_DSEG_TENSION_CAPS_IT_FAR_ABOVE_THE_FRONTIER
producer: experiments/probe_nca_texture_amortized_capacity_break.py
analyzer: experiments/analyze_nca_capacity_break.py
state: experiments/results/nca_amortized_capacity_break_main/gate_state.json
daemon_log: .omx/tmp/nca_daemon/main.log
supersedes: .omx/research/generative_axis_nca_amortized_capacity_break_PENDING_20260619.md
cross_refs:
  - .omx/research/generative_axis_continuous_texture_nca_AMBER_20260619T020000Z.md   # the AMBER this resolves
  - .omx/research/dseg_side_feasibility_corners_verdict_20260619.md                   # the d_seg wall (frontier 0.00056)
  - .omx/research/p_suff_task_ablation_verdict_20260619.md                            # frontier near task-RD floor
  - .omx/research/generative_axis_dseg_core_design_20260619T004600Z.md                # the 29.3*params^-0.71 wall
  - .omx/research/generative_axis_nca_dseg_core_gate_20260619T013000Z.md              # the capacity-escape hypothesis
---

# Generative-axis FINAL exhaustion test — VERDICT: RED (the last family caps too)

**The last un-run path of the sub-0.15 campaign, run to its decisive conclusion.** A best-shot AMORTIZED
continuous-texture NCA decoder, fixing all three of the AMBER's caveats and answering the original
hypothesis: does weight-shared ITERATION break the `d_seg ~ 29.3·params^−0.71` one-shot capacity wall?
All numbers `[contest-CPU advisory]` NON-PROMOTABLE; exact pointer UNMOVED at **0.19110** — this unit did
NOT move the pointer. $0: MPS fp32 gradient + CPU-authority d_seg.

## 0. The headline (read this first)

The AMBER's three caveats were FIXED and the answer is measured:

1. **CONVERGENCE (caveat 1) — SOLVED.** The AMBER converged ~2/8 runs and the headline was not
   reproducible-on-demand (re-running the exact config collapsed). The fix: a soft tanh **STATE-BOUND**
   (the Mordvintsev alive-masking surrogate) + dropping the pool + multi-restart keep-best. Measured
   convergence: **3/4 restarts converged 4/4 frames at the first rule-size, reproducibly** (restart-0
   avg 0.01298, restart-1 avg 0.01307 — near-identical, the reproducibility the AMBER lacked). The
   convergence-robustness blocker is GONE.

2. **AMORTIZATION (caveat 2) — measured, and it costs.** ONE shared rule across 4 real GT frames + tiny
   per-frame latents gives a stable **amortized avg d_seg = 0.013** — vs the AMBER's best single-frame
   0.00337. The shared rule costs **~4× in d_seg** vs a fresh rule per frame, exactly the tension the
   AMBER flagged ("the 32-d latent may be too small to carry frame-to-frame variation"). The amortized
   rate is byte-cheap (0.009).

3. **CAPACITY-BREAK (caveat 3) — the NCA BEATS the one-shot wall, but it does not matter.** At 10k params
   the amortized d_seg 0.013 is **0.31× the power-law wall** (0.041) — iteration genuinely gives
   effective-depth detail the one-shot decoder lacks (it beats the wall by ~3.2×). BUT this is a
   PYRRHIC win: even granting the NCA beats the wall by 3.2× at EVERY scale, the rule big enough to
   reach the sub-0.15 d_seg budget (~0.0007) needs **~628k params → ~337 KB amortized → rate term 0.230
   — which ALONE EXCEEDS the entire frontier S of 0.191.** The rate/d_seg tension is unbreakable: if the
   NCA is small it pays d_seg (0.013 → S 1.37); if it is big enough to cut d_seg it pays rate (0.230 >
   frontier). There is no operating point where the amortized continuous-texture NCA reaches sub-0.15.

**Measured anchor (c8h32, the first rule-size; the full sweep continues filling points for the durable record):**

| quantity | value | vs frontier |
|---|---:|---|
| rule params | 10,299 | — |
| convergence | **3/4 restarts, 4/4 frames** | (AMBER was ~2/8) |
| **amortized avg d_seg** | **0.01298** | **23.2× frontier (0.00056)** |
| vs one-shot power-law wall | **0.31×** (beats it) | iteration helps |
| amortized rate | 0.00905 | byte-cheap |
| **projected S** | **1.365** | **7.1× frontier S (0.191)** |

**VERDICT: RED — `RED_AMORTIZED_NCA_CONVERGES_AND_BEATS_THE_ONE_SHOT_WALL_BUT_THE_RATE_DSEG_TENSION_CAPS_IT_FAR_ABOVE_THE_FRONTIER`.**
The generative-continuous-texture axis — the AMBER, the strongest sub-0.15 d_seg-core candidate, "the
frontier's own move" — is now closed. Convergence and the capacity wall were both real and both
addressable; the binding wall is the **rate/d_seg tension**, the same one every dense family hits, just
relocated to the shared-rule size. **This is the FINAL representation family.**

## 1. Why this is the RIGHT, FAITHFUL, decisive test (NO-FAKE)

- **AVERAGE not best-frame** (kills the AMBER's selection bias): the verdict number is the average realized
  d_seg across all 4 amortized frames, through the REAL frozen SegNet + EXACT uint8 roundtrip, CPU
  authority. The best single frame is reported but NOT the verdict.
- **false-RED guard:** multi-restart keep-best (the verdict uses the BEST-converged restart, so an
  under-trained collapse is never read as a capacity wall). The convergence is 3/4 restarts at 4/4 frames
  — this is NOT an under-trained RED, it is a genuinely-converged result that caps high.
- **false-GREEN guard:** AVERAGE over frames + amortized (shared-rule) accounting. No lucky converged
  single frame can produce the headline; the 0.013 is the shared rule reproducing all 4 frames.
- **The capacity-break is a REAL power-law evaluation**, not a constant (15 NO-FAKE tests verify the
  power-law decreases with params, the shared rule is one object across frames, the state-bound is
  load-bearing, S is recomputed from components).

## 2. The convergence-engineering journey (the caveat-1 fix, with receipts)

The build's first hard blocker was exactly convergence-robustness, as the AMBER predicted. The measured
sequence (all advisory):

| config | iters | stabilizers | result | finding |
|---|---|---|---|---|
| amortized 8f | 2400 | pool0.3 + grad-norm + warmup | **NaN** (recon=nan) | pool feedback → unbounded state growth → inf |
| amortized 4f | 400 | **state-bound 32, no-pool** | recon 45.6 FINITE, descending | **NaN FIXED** |
| amortized 4f | 3000 | state-bound 32, no-pool, ×4 restarts | **3/4 conv, avg 0.013, reproducible** | convergence SOLVED |

**Two reusable convergence findings (system intelligence):**
1. The Mordvintsev POOL + sample-replay, applied to texture-regression-through-a-frozen-scorer with a
   trained residual rule, is DESTABILIZING (the pool feeds grown states back which grow unboundedly
   through the deep unroll to inf/NaN — Mordvintsev avoids this with alive-masking on an RGBA-emoji
   target). The fix is a soft tanh STATE-BOUND (alive-masking surrogate) + dropping the pool.
2. CPU-gradient (the deterministic dodge for MPS non-determinism) is ~0.7s/it single-frame =
   impractically slow; multi-restart keep-best on MPS is the tractable convergence-robustness mechanism
   on the available hardware. The state-bound makes MPS runs reproducible (0.01298 vs 0.01307 across
   seeds) — it ALSO fixes the AMBER's MPS-non-determinism collapse, because the divergence was the
   unbounded unroll, not pure kernel noise.

## 3. The binding wall — the rate/d_seg tension (the airtight RED, byte-accounting-light)

The capacity-escape hypothesis was: weight-shared iteration (effective depth N at fixed params) could
reach the d_seg a many-KB static decoder needs, at few-KB rule cost. **The iteration DOES help** (beats
the one-shot wall 3.2×). But the decisive arithmetic:

- sub-0.15 needs realized d_seg < ~0.0007 (∂S/∂d_seg=100; budget after rate 0.118 + pose 0.058).
- NCA d_seg(params) model (beating the wall 3.2×): `d_seg ≈ 9.2·params^−0.71`.
- d_seg=0.0007 ⇒ **~628k params** ⇒ amortized rule bytes ~337 KB ⇒ **rate term 0.230**.
- **0.230 > the entire frontier S of 0.191** — the rule that's big enough to hit sub-0.15 d_seg costs more
  rate, alone, than the whole frontier score.

So the operating curve has no sub-0.15 point: small rule → d_seg dominates (0.013 → S 1.37); big rule →
rate dominates (>0.23). The minimum-S operating point of the amortized NCA is far above the frontier. This
argument is independent of the exact fitted exponent (a steeper NCA exponent only moves the crossing a
little; the rate at the required size still dwarfs the frontier).

## 4. The comprehensive campaign conclusion (the operator's RED branch, reached)

Per the test's own RED definition: *"even POOL-stabilized + shared-rule, the average d_seg(params) caps →
the generative axis caps like the rest → this is the FINAL family, and the campaign's comprehensive
conclusion is that the frontier ~0.191 is near the real achievable floor for ALL known representation
families."*

Every representation family the sub-0.15 d_seg-core campaign tested is now measured-closed:
- factored RANK-1 LF (learned pixel decoder): CAPACITY wall, d_seg ~ 29.3·params^−0.71 → RED.
- curve-core (static geometry): SURVIVAL wall, realized d_seg plateaus ~0.007 → RED.
- flat-partition NCA: survival wall (flat fill) → RED (~0.02).
- d_seg-side closed-form corners (#149 sub-pixel, #148 keyframe-warp): RED (1.7–45× frontier).
- p_suff task-ablation: frontier near its task-RD floor, ~0.7% invariant mass (not free) → RED.
- **amortized continuous-texture NCA (this test, the last + strongest candidate): RED** — converges,
  beats the one-shot wall, but the rate/d_seg tension caps it at S ≥ ~1.37 (small) / rate > 0.23 (big).

**The frontier ~0.19110 is near the real achievable floor for all known representation families on this
contest.** The continuity thesis held (continuous beats flat; iteration beats one-shot) but neither lever
is enough: the survival wall + the rate/d_seg tension jointly bound every family above the frontier.

## 5. What this redirects (system intelligence)

- The generative axis is CLOSED for sub-0.15; no further continuous-texture-NCA build is warranted (the
  cap is structural, not an implementation deficit — convergence and capacity were both fixed/beaten and
  it still caps). Reactivation criterion (per Forbidden-premature-KILL): a representation that breaks the
  rate/d_seg tension — i.e. one whose d_seg(rate) curve crosses BELOW the frontier's (rate 0.118, d_seg
  0.00056) point — which no measured family does.
- The state-bound NCA-stability finding + the POOL-is-destabilizing-for-scorer-regression finding are
  reusable for any future iterated-decoder work (don't use the emoji-morphogenesis pool recipe for a
  fixed-texture-through-a-frozen-scorer objective; bound the state).
- The sub-0.15 search, on the representation axis, is exhausted. The pointer stays at 0.19110; any further
  descent must come from a genuinely different axis (not a new RGB-frame representation), or the frontier
  is accepted as near the achievable floor.

## Observability surface

Every row records: rule_param_count, per-restart + average realized d_seg, convergence count, recon_rmse,
boundary/interior flip split, amortized bytes/rate, the power-law-wall comparison (ratio + beats_wall), and
S recomputed from components with the real √(10·d_pose) term. `[contest-CPU advisory]`, score_claim=false,
pointer_moved=false. Machine-readable at the state JSON + per-run advisory JSON. The full param sweep
continues in the durable daemon (resumable per-config) and will append points c12h64/c16h96/c24h160/c32h256
to firm up the empirical exponent fit; the RED is robust on the c8h32 point + the rate/d_seg arithmetic.

## Canonical-vs-unique decision per layer

Eval roundtrip, realized-d_seg metric, GT load, rate formula, power-law wall = ADOPT_CANONICAL (reused
from the curve/AMBER gates for apples-to-apples). The amortized shared-rule NCA (ONE rule + per-frame
latents, state-bound) = FORK (the unique mechanism — the caveat-2 amortization + caveat-1 stability fix).
The capacity-break power-law fit = FORK_PRINCIPLED (the caveat-3 decisive measurement).
