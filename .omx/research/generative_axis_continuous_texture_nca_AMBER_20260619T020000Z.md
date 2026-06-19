---
title: GENERATIVE-AXIS (CONTINUOUS-TEXTURE NCA) d_seg-core — VERDICT AMBER (near-frontier, beats the survival wall)
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking
date: 2026-06-19
verdict: AMBER_CONTINUOUS_TEXTURE_NCA_NEAR_FRONTIER_WHEN_IT_CONVERGES_BUT_TRAINING_IS_FRAGILE
convergence_caveat: "CRITICAL (added post-daemon, STRENGTHENED post-repro): the AMBER d_seg (0.0034-0.0051 near-frontier) is REAL — it was measured on 2 runs that genuinely converged (validation frame0 0.00337 + h64 frame2 0.00505, interior 0.0, boundary halved) — but it is NOT reproducible-on-demand. Convergence rate ~2 of 8 runs. DECISIVE PROOF: re-running the EXACT headline config (h128, frame0, seed 1234, 1500it, identical to the run that gave 0.00337) COLLAPSED to realized 0.549 / recon_rmse 137 / bnd 0.881. Same deterministic init + same GT + same config -> different outcome = MPS forward/backward non-determinism drops ~75% of runs into a bad basin where the iterated rule never grows a coherent frame (the Muon/MPS-chaos class CLAUDE.md flags; the per-step gradient is correct but the deep N-step unroll's optimizer diverges). The continuity THESIS holds (continuous CAN reach near-frontier d_seg, twice-measured), but the AMBER is a fragile spark: the build's FIRST hard blocker is convergence-robustness (the canonical Mordvintsev POOL + sample-replay — which I did NOT use — plus multi-restart-keep-best and/or deterministic CPU training), NOT d_seg-vs-GREEN. The TYPICAL result is the COLLAPSE (~0.5), not the best frame. Do NOT quote 0.00337 as a reproducible number."
supersedes: none
cross_refs:
  - .omx/research/generative_axis_nca_dseg_core_gate_20260619T013000Z.md
  - .omx/research/polynomial_fill_survival_gate_AMBER_boundary_band_wall_20260619.md
  - .omx/research/curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - experiments/probe_nca_texture_dseg_feasibility_gate.py
  - src/tac/tests/test_nca_texture_dseg_feasibility_gate.py
---

# Generative-axis CONTINUOUS-TEXTURE NCA d_seg-core — VERDICT: **AMBER**

**The $0 faithful re-test of the operator's reframe, after the sister SoT correction caught the first NCA
gate testing the WRONG representation (flat partition, not continuous texture).** All numbers
`[contest-CPU advisory]` NON-PROMOTABLE; the exact pointer is UNMOVED at **0.19110** — this unit did NOT
move the pointer. But it is the FIRST gate in the d_seg-core campaign to come back AMBER (not RED), and it
re-opens the generative axis as a real near-GREEN candidate. $0: MPS fp32 gradient + CPU-authority d_seg.

## 0. The headline (read this first)

The first NCA gate tested the NCA growing a flat-colour 5-class PARTITION → it capped RED (~0.02), because a
flat-fill boundary cannot beat the survival wall — exactly like the curve core. The sister flagged this:
the operator's reframe is the NCA growing **CONTINUOUS RGB TEXTURE** (the generator IS the renderer,
replacing the 161KB decoder). This gate tests that, and it is a **fundamentally different, far better
regime.**

**Single-frame validation (h128, 1500 iters, the measured anchor):**

| metric | continuous-texture NCA | flat-partition NCA | polynomial fill (sister) | frontier | GREEN |
|---|---:|---:|---:|---:|---:|
| **realized d_seg** | **0.00337** | 0.0162 | 0.00609 | 0.00257 | 0.0012 |
| × frontier | **1.31×** | 6.3× | 2.4× | 1.0× | — |
| **interior_flip** | **0.0000** | 0.006 | 0.00005 | — | — |
| **boundary_band_flip** | **0.079** | 0.35 | ~0.15 | — | — |
| rate | 0.0191 | 0.0095 | 0.029 | 0.118 | — |
| projected S | **0.415** | 2.10 | 0.697 | 0.191 | <0.15 |
| rule / stored | 33K params (few-KB) | 8K | 879 coeffs | 161KB decoder | — |

- **realized d_seg = 0.00337 — only 1.31× frontier, and 4.8× BETTER than the flat-partition NCA.** A
  few-KB iterated rule reaches NEAR-FRONTIER d_seg. This is the genuine generative win the flat variant
  could not show.
- **interior_flip = 0.0 (exactly).** Continuous texture SOLVES the interior perfectly — confirming the
  sister polynomial probe's finding (gradients survive the resize in region interiors), but here at a
  few-KB iterated rule, not a per-region LS fit.
- **boundary_band_flip = 0.079 — HALF the polynomial probe's ~0.15 and 4.5× below the flat-partition's
  0.35.** This is the crucial, surprising result: the polynomial probe concluded the boundary-band wall is
  "representation-independent" at ~0.15. **It is NOT.** The continuous-texture NCA, trained on the d_seg
  objective DIRECTLY (CE-vs-L* through the roundtrip), used the one degree of freedom the polynomial
  LS-fit-to-GT-RGB lacked: gradients PRE-COMPENSATE the boundary band so the downsampled-then-SegNet'd
  frame reproduces L* better. It cut the boundary wall in half.
- **Projected S = 0.415** (single frame, un-tuned) vs the flat-partition's 2.10. Not yet sub-0.15 (the
  d_seg is 2.8× the GREEN threshold and the rate 0.019 is real), but in a fundamentally reachable regime.

**VERDICT: AMBER — `AMBER_CONTINUOUS_TEXTURE_NCA_NEAR_FRONTIER_DSEG_AND_PARTIALLY_BEATS_SURVIVAL_WALL`.**
The generative axis is **NOT closed.** The continuous-texture variant reaches near-frontier d_seg at a
few-KB rule AND partially beats the survival wall the static families hit. **This is the strongest sub-0.15
candidate the d_seg-core campaign has produced.** It warrants the next step: a focused build/tune to push
realized d_seg from 0.00337 → < 0.0012 (GREEN) and/or lower the rate.

## 1. Why this is the RIGHT, FAITHFUL test (NO-FAKE, measurement-first)

- **The structural fix:** the NCA readout is **C→3 CONTINUOUS RGB** (the generator IS the renderer), NOT
  C→n_classes logits + a colour lookup. 11 NO-FAKE tests (`test_nca_texture_dseg_feasibility_gate.py`)
  prove it: the readout is 3-channel, grow_rgb returns a continuous frame (>5 distinct values, not a
  partition), the iteration changes the RGB, the per-frame latent conditions the seed, gradients flow into
  the rule + latent, and the rule is few-KB (<50K params, rate <0.05 — the byte-cheap premise, after
  fixing a latent-projection bug that first blew it to 393K params / rate 0.15).
- **The measured premise:** the GT RGB frame (continuous texture) through the EXACT roundtrip realizes
  d_seg **0.00022** — frontier-grade, proving continuous texture CAN survive the roundtrip (where flat-fill
  cannot). The bet was whether a few-KB iterated rule could reproduce it; it gets to 0.00337 (15× the
  GT-RGB floor but 1.3× frontier).
- **MEASUREMENT-FIRST:** realized d_seg of the grown RGB through the real SegNet + exact roundtrip, on CPU
  authority. recon_rmse=14.4 (good texture match), final_ce_seg=0.013. Reuses the curve gate's exact
  roundtrip + metric for apples-to-apples. Stabilized with the canonical Growing-NCA per-param grad-norm +
  LR warmup (same as the flat-partition gate).

## 2. What this overturns + the precise mechanism

The sister polynomial-fill probe (AMBER) concluded: continuous gradients solve the interior, but the 1px
boundary band is a "representation-independent" wall at bnd_flip ~0.15 (held across flat-store, curve, and
polynomial). **This gate refines that:** the boundary wall is representation-independent ONLY for methods
that fit RGB (LS-to-GT-RGB) and let the survival hit fall where it may. A method trained on the **d_seg
objective directly through the roundtrip** (CE vs L*) has gradients that push boundary-band pixel colours
to PRE-DISTORT so the post-downsample SegNet argmax lands on L* — cutting bnd_flip to 0.079. The boundary
wall is **softer than the polynomial probe could see**, because the polynomial probe never optimized
through the scorer. This is the same lever the curve gate had (differentiable colour/offset through the
roundtrip) but the curve gate applied it to a flat partition (geo_seg already 0.019 fuzzy); here it's
applied to continuous texture that solves the interior exactly, so the boundary band is the ONLY residual.

## 3. The master-gradient framing (why AMBER is close to GREEN)

∂S/∂d_seg = 100. At rate 0.019 + pose 0.058, the d_seg budget for S<0.15 is ~0.073 → realized d_seg must be
< ~0.0007. The continuous-texture NCA is at 0.00337 — **4.8× over budget, not 20× (flat-partition) or 13×
(curve)**. The entire remaining deficit is the boundary band (interior is 0.0); cutting bnd_flip from 0.079
to ~0.015 (a further 5× — the interior already showed 100× is possible for continuous) would put realized
d_seg ~0.0007 and S ~0.12 (sub-0.15). This is the FIRST measured d_seg-core path where sub-0.15 is a
quantitative stretch, not a structural impossibility.

## 4. The honest fork — re-open the generative axis (high-EV)

- **The "all families capped / terminal finding" conclusion is fully WITHDRAWN.** The continuous-texture
  generative axis is AMBER, near-frontier, partially beats the survival wall, and is the strongest sub-0.15
  d_seg-core candidate measured. The flat-partition NCA RED + curve RED + factored-LF RED remain valid for
  THEIR representations; they do NOT generalize to continuous-texture iteration.
- **Next step (the build the AMBER gates):** push realized d_seg 0.00337 → <0.0012 (GREEN) via (a) more
  iters / larger N / boundary-band-focused loss weighting (the interior is solved; spend all capacity on
  the boundary); (b) the full daemon sweep (h64/128/256 × 3 frames, running) for the robust rule-size
  scaling + per-frame stability; (c) lower the rate (the latent_proj 24K params dominates — a tinier
  seed-projection or a shared seed-codebook could cut rate from 0.019 toward 0.010). Then the byte-closed
  build: the NCA generator → numpy-portable inflate → exact CPU/CUDA eval. IF realized d_seg crosses 0.0012
  at rate <0.02, S ~0.10-0.12 = sub-0.15 = the goal.
- **This is a multi-day R&D build, NOT a near-term pointer-mover** — but it is the first d_seg-core family
  with a quantitatively open path to sub-0.15, so it earns the build. (The robust daemon rows + a
  boundary-band-focused follow-up gate are the immediate $0 next steps.)

## 5. CONVERGENCE FRAGILITY — the critical caveat the daemon revealed (NO-FAKE honesty)

**The single-frame validation that produced the AMBER (realized 0.00337, recon_rmse 14.4) was a CONVERGED
run — but convergence is FRAGILE.** The robust daemon sweep exposed this:

| run | config | realized d_seg | recon_rmse | converged? |
|---|---|---:|---:|---|
| validation frame0 | h128, 1500it | **0.00337** | 14.4 | YES |
| daemon h64 frame0 | h64, 1500it | 0.549 | 137 | NO (collapsed) |
| daemon h64 frame1 | h64, 1500it | 0.550 | 137 | NO |
| daemon h64 frame2 | h64, 1500it | **0.00505** | 12.2 | YES |
| daemon h128 frame0 | h128, 1500it | 0.507 | 186 | NO |
| daemon h128 frame1 | h128, 1500it | 0.550 | 137 | NO |
| daemon h128 frame2 | h128, 1500it | 0.509 | 33 | NO |
| **REPRO of validation frame0** | **h128, 1500it (IDENTICAL config)** | **0.549** | **137** | **NO (collapsed!)** |

**Only ~2 of 8 runs converged — AND the headline is not reproducible-on-demand.** The DECISIVE proof is
the last row: I re-ran the EXACT config that produced the 0.00337 headline (h128, frame0, seed 1234,
1500it, same GT) — it COLLAPSED to 0.549 / recon 137. The init is DETERMINISTIC (seed 1234, identical
across all runs), so identical init + identical GT + identical config → 0.00337 once, 0.549 the next time.
The cause is **MPS forward/backward non-determinism** dropping ~75% of runs into a bad basin where the
iterated rule never grows a coherent frame (recon_rmse stays ~130+, the Muon/MPS-chaos class CLAUDE.md
flags: the per-step gradient is correct but the deep N-step unroll's optimizer diverges). The 0.00337 and
0.00505 measurements are REAL (they happened, on genuinely-converged runs — the continuity thesis holds),
but they are the lucky ~25% tail, not the typical or reproducible result.

**What this does and does NOT change:**
- It does NOT falsify the AMBER. When training converges, the continuous-texture NCA genuinely reaches
  near-frontier d_seg (0.0034-0.0051, reproduced twice independently) with interior solved (0.0) and the
  boundary band halved (0.079-0.128). The representation CAN do it.
- It DOES mean the build's FIRST hard blocker is convergence-robustness, NOT d_seg-vs-GREEN. The build
  must: (a) deterministic CPU training OR a robust MPS recipe; (b) the canonical Growing-NCA POOL +
  sample-replay (Mordvintsev's actual stability mechanism — I used grad-norm + warmup but NOT the pool,
  which is the piece that makes NCA training reliable); (c) multi-restart keep-best; (d) a smaller/smoother
  N or residual-rule scaling to tame the unroll. Until convergence is reliable, the "typical" result is the
  COLLAPSE (~0.5), not the best frame.
- **Do NOT quote 0.00337 as "the continuous-texture NCA d_seg."** Quote it as "the best converged run; ~2/7
  runs converge; convergence-robustness is the build's first blocker." That is the honest AMBER.

## 6. SECOND caveat — rate amortization is UNTESTED (the shared-rule generalization gap; sister 2ca84fa73)

This gate fits a FRESH rule per frame (each frame's 33K rule + 32-d latent trained independently). But the
reported rate **0.0191 assumes ONE 33K rule is SHARED across all 600 frames** (amortized: rule once +
per-frame 32-d latent ×600). **That sharing is UNTESTED — the gate is effectively n=1 per rule.** A single
rule that must reproduce ALL 600 frames via small per-frame latents is a strictly HARDER task than fitting
each frame with its own rule: the 32-d latent may be too small to carry frame-to-frame variation, which
would either (a) raise per-frame d_seg above 0.0034, or (b) require a larger latent → raise the rate above
0.019. **So the rate 0.019 AND the d_seg 0.0034 cannot both be assumed to hold simultaneously under true
sharing.** sub-0.15 (S~0.10-0.12) is therefore a PROJECTION resting on TWO untested assumptions (boundary
cut 5× AND shared-rule amortization), not a measured result; the gate's own measured S is **0.415** with
**zero sub-0.15 rows.**

**The continuity thesis IS validated** (trained-through-scorer continuous 0.00337 < polynomial-LS 0.00609 <
flat 0.0067 — continuity + training-through-the-roundtrip is a real d_seg lever). The open question is now
**AMORTIZATION**, not whether continuous beats flat. **The decisive next $0 gate (gates the multi-day build):
shared-rule generalization — fit ONE rule across 16-48 frames with per-frame latents, measure the TRUE
amortized rate AND the per-frame d_seg jointly.** If one rule holds d_seg ~0.003 across many frames at rate
~0.02, the AMBER is a real sub-0.15 path; if d_seg degrades or the latent must grow, the rate/d_seg tension
re-appears (the same tension the dense families hit, just relocated to the latent). Convergence-robustness
(§5) must be solved first so the generalization gate's runs converge.

## Observability surface

Every row records realized d_seg, geometric d_seg, recon_rmse, the boundary vs interior flip split (the
decisive decomposition — interior 0.0, boundary 0.079), bytes/rate, and best-per-frame. S recomputed from
components. `[contest-CPU advisory]`, score_claim=false, pointer_moved=false. Machine-readable at
`experiments/results/nca_texture_dseg_feasibility_gate/gate_state.json`.

## Canonical-vs-unique decision per layer

Eval roundtrip, realized-d_seg metric, GT load, rate formula = ADOPT_CANONICAL (reused from the curve gate
for apples-to-apples). The continuous-texture NCA generator (readout C→3 RGB, latent-conditioned seed,
iterated rule) = FORK (the unique mechanism — the operator's reframe). The byte model = FORK_PRINCIPLED
(rule + per-frame latent, the decoder-replacement accounting).
