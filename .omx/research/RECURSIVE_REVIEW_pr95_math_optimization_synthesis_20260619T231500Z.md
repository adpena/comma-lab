---
title: "RECURSIVE REVIEW + MATH/ALGEBRA/CALCULUS/GEOMETRY OPTIMIZATION SYNTHESIS — 5-lens audit of the decisive LOCAL PR95 run + PR95 ITSELF; the optimal config + the FREE wins + the round-2 measurement plan"
authority: "[contest-CPU advisory / measured+derived synthesis] — pointer UNMOVED 0.19110; $0; NO paid dispatch; NO score claim"
score_claim: false
promotion_eligible: false
pointer_moved: false
date: 2026-06-19
operator_directive: "LOCAL but recursive adversarial review first and recursive pass for math/algebra/calculus/geometry optimization" + "Perhaps we can also start from PR95 measuring all the math … to reveal areas of non optimal"
lenses:
  - "A loss/d_seg-finishing geometry (a439e1fb) — MEASURED"
  - "B optimizer/convergence geometry (abe2967e) — DONE (MEASURED; reversed the floor-fix framing + found Muon-early@0.03)"
  - "C capacity/rate/architecture geometry (a1200898) — done"
  - "D adversarial correctness (a10a62b6) — done"
  - "E PR95-SOURCE math audit (a011036c) — done"
cross_refs:
  - .omx/research/lensA_dseg_optimal_loss_geometry_ce_vs_margin_hinge_20260619.md
  - .omx/research/INCORPORATION_adversarial_review_of_pr95_pivot_true_solution_20260619T223000Z.md
  - experiments/launch_bind_all_taper_ab.py            # the fully-bound round-2 actuator
  - experiments/results/capstone_capacity_ablation_2x2_20260611/
---

# Recursive review synthesis — the math-optimal decisive LOCAL run

**Operator directive (two parts):** (1) recursively adversarially review + math-optimize the decisive LOCAL run
BEFORE the ~6-day burn; (2) start from PR95 itself and measure all the math to reveal non-optimal areas. Five
council lenses ran measurement-first on the paused basin checkpoint. All `[contest-CPU advisory]`; pointer
UNMOVED 0.19110; $0; no paid spend. This memo consolidates them into ONE optimal config + the FREE wins + the
single round-2 measurement that gates the re-fire.

## 0. The headline
The operator's prior holds on both axes: **our run's config is sub-optimal AND PR95 itself is sub-optimal in
≥6 distinguishable places.** The biggest reframe: **lens E#1 — ~70% of PR95's 29,650 epochs are wasted**
(every late stage hits its EMA-best at 5-33% of its budget, then rides a near-zero cosine-tail LR doing ~0
work). So the "6-day burn" is mostly basin-sitting; early-stop + restart recovers it. The biggest ceiling:
**the seg loss is a fixed-τ PROXY, never the score's argmax-flip indicator** (E#2/lens A — measured
margin_hinge m=0.5 = 0.643× CE's residual d_seg). The cleanest free correctness fix: **the pose √ is inside
the minibatch reduction** (E#3, verified in our `driver.py:1813/2030`, bs=8 → Jensen gap mis-weights pairs
backwards). And the calculus the operator named: **PR95's fixed 100:1 seg:pose ignores that ∂S/∂d_pose=271 >
100 at the frontier's d_pose** (E#4).

## 1. The optimal config — confirmed/applied this unit
| lever | verdict | source | status |
|---|---|---|---|
| **bc20** (base_channels) | S-optimal sub-0.15 arch; bc24 needs d_seg<1.3e-4 (4.1× harder), bc28 mathematically impossible (rate 0.10+pose 0.0585>0.15) | C (measured arithmetic) | LOCKED |
| ~~muon_lr_floor_fix ON~~ → **OFF (CORRECTED by lens B)** | the floor-fix is correct-as-coded but a **PR95-faithfulness DEVIATION that benefits NO measured arm** (vendored applies ONE lr_lambda to both opts; prior capstone bugsweep = "BUG-B CLEAN, do NOT fix"). The apparatus audit's "it's the BUG-B fix" was OVER-CLAIMED. The flag + resume-guard CODE stays (opt-in, default-off, good hygiene) but is NOT enabled for the decisive run. The REAL lever is the muon_lr **PEAK** (below). | B (MEASURED A/B) | flag/guard kept (`0065a7b05`+`de2fbb4a8`); floor-fix NOT enabled |
| **Muon-early @ muon_lr=0.03** (NEW, lens B — the biggest optimizer lever) | Muon conditions the BULK descent; the vendored stage-8 `muon_lr=2e-4` is itself the throttle (M1=frozen 0.0304); `0.03` descends | **B MEASURED**: M3 → 2.4× lower d_seg @matched-ep, 32% lower basin, same wall-clock | NON-faithful (class-shift) → gate as its own vehicle, not a "faithful" tweak | `use_muon=True` + `muon_lr≈0.03` in early stages (curriculum override) |
| **ema-warmup** | the EMA-shadow-lag fix; default True in bind-all launcher; n600's 75 steps/ep fills the window in ~13 ep anyway | D | confirmed safe |
| **FiLM-v2 + trunk-stopgrad** | pose protection (∂S/∂d_pose = 86% of d_seg's marginal) | A + E#5 | keep ON |
| **the fully-bound launcher** = `launch_bind_all_taper_ab.py` NOT `launch_split_by_head_basin.py` | only the bind-all launcher carries taper + hinge + KD-warm + floor-fix-default-True | wiring audit | the re-fire switches launchers |

## 2. The MEASURED / DERIVED optimization levers (ranked by confidence × ΔS)
| # | lever | math | measured/derived | ΔS / Δd_seg | faithfulness | how to apply |
|---|---|---|---|---|---|---|
| **1** | **margin_hinge seg surrogate** (anneal target 1.0→0.5) replaces CE/soft_cosine | `relu(m−(z_g−z_2nd))` = −1 on every flip, 0 on correct-interior; CE untargeted; soft_cosine vanishes on deep flips (−6.9e-9 @Δ=6) | **MEASURED** (real frozen SegNet, basin forkpoint): m=0.5 → 0.643× CE residual d_seg, slope −0.313 vs CE −0.166; soft_cosine = WORST | −16% to −36% d_seg (advisory); memory l235 anchor −34% | MEDIUM (loss family; repo built it default-off) | `--seg-margin-hinge-throughout`; set `seg_margin_hinge_target` 1.0→0.5 anneal (small driver add) |
| **2** | **epoch early-stop + restart / advance through stages** (recover the ~70% wasted basin epochs) | **lens B#1 + E#1 are the SAME finding from two angles:** the cosine-to-zero tail does ~0 work because LR COLLAPSED below the useful floor (B measured: constant-LR beats full-cosine 26%/2.2×; full-cosine reproduces the basin's ep^−0.235 slope). The shallow d_seg slope is an LR-collapse artifact + the basin being STALLED mid-stage-1 — NOT a capacity/landscape wall. Fix = restart LR (warm restart re-opens descent), faithfully via stage transitions (stage 2 resets cosine to 1e-3) or explicitly via early-stop+restart | **MEASURED** (B 5-arm A/B + vendored best-ep fingerprint: stage5 2075/9000, stage8 250/5000) | 10-30% Δd_seg; FREE | LOW (faithful: just advance through stages so LR resets) | run FULL budget so stage transitions reset LR; or early-stop+restart |
| **3** | **byte-neutral d_seg-aware taper** (arm_b) reallocate the 69% low-res capacity → high-res boundary band | vendored taper puts 57.8K params (69%) in stages SegNet's stride-2 stem discards; only 7.76K at the 10-11ch high-res band where flips live; arm_b `[22,16,15,14,15,14,10]` ~doubles the boundary band at +0.05% bytes | **DERIVED** (byte-match measured; d_seg effect UNMEASURED — prior #121 found it weak via a *saliency* arg; this is a *spatial-band* arg → A/B is the arbiter) | byte-neutral; large IF it cuts d_seg | MEDIUM (arch change → KD-warm-start required) | `--arm arm_b --kd-warm-start-dir <basin>/best` |
| **4** | **equimarginal seg:pose schedule** (γ-up late once d_pose<2.5e-4) | crossover d_pose=2.5e-4; frontier d_pose~2-9e-5 → ∂S/∂d_pose=158-271 > ∂S/∂d_seg=100; fixed 100:1 is blind to this | **DERIVED** (exact calculus); repo has `equimarginal_pose_weight.py` | −0.002 to −0.01 | MEDIUM (per-stage weight schedule) | `--pose-equimarginal` / `--pose-dim-weights-auto` |
| **5** | **pose √ outside the batch reduction** (Jensen fix) | score = √(10·mean over ALL 600); code = Σ √(10·mean_batch) — concave √ inside bs=8 reduction weights easy batches 14×, hard 0.1× = backwards | **MEASURED bug** (verified `driver.py:1813/2030` + `common.py:193`); magnitude largest early, small at converged frontier | small but FREE | LOW (it's a latent bug, not a design) | accumulate linear 10·MSE, one √ over epoch mean (or √-readout-only); recalibrate pose_weight |
| **6** | **Muon earlier (stages 2+5) / sensitivity-routed partition** | Muon conditions bulk descent, not a 250-ep polish; RGB heads (where d_seg/pose decided) are on the LESS-conditioned AdamW; the score-aware QAT sensitivity EMA already exists to route by ‖∂S/∂w‖ | **DERIVED** (E#6); **B's measured A/B pending** | speculative; compounds with #2 | LOW (Muon-earlier = one flag/stage) | curriculum use_muon flags; (B confirms) |

**PR95 audited-and-~optimal (adversarial honesty, do NOT touch):** L17 sigma noise (correctly simulates the
uint8 Q-step, already in the gradient path); L20/L24/L25 archive + latent delta-uint8 codec (near entropy
floor; only ~217 B lever = ΔS −0.00014); L18 sin + bilinear-skip (defensible single-video memorization);
pose weight=1 (correct in units — its only defects are #4 schedule + #5 Jensen).

## 3. The honest expectation (anti-over-swing — preserved from the apparatus-audit review)
No single lever clears sub-0.15. Lens A: margin_hinge alone → ~0.24-0.25 at the 600-pair exponent (toward,
not across). The bet is the STACK: margin_hinge (#1) × recovered-epochs-as-restarts (#2) × boundary-band
capacity (#3) × equimarginal pose (#4), KD-warm-started from the basin. The independent-reviewer prior
(modal outcome ~60% EARNED ~0.19 wall, ~3% sub-0.15) stands until the stack is MEASURED to bend the d_seg
exponent materially. An ORIGINAL ours-trained vehicle that fixes PR95's measured non-optimalities is a
legitimate Innovation-Gate result even at ~0.19 (vs the BORROWED 0.191 frontier) — and the only honest path
to learn whether the stack crosses.

## 4. The round-2 measurement (the single decisive A/B that gates the re-fire)
Per the Recursive adversarial review protocol (3 clean passes) + the means/ends firewall (one decisive
measurement, not gate-volume): **ONE** head-to-head on the bind-all launcher, KD-warm-started from the basin,
short budget (~few thousand epochs at n600), measuring the d_seg LATE-exponent + d_pose stability:
- **CONTROL arm_a:** vendored taper + soft_cosine (the current config).
- **OPTIMAL arm_b:** solved byte-neutral taper + margin_hinge(0.5-anneal) + equimarginal + floor-fix + FiLM-v2.
Verdict: if arm_b bends the d_seg exponent materially (and d_pose stays bounded), the 6-day decisive run is
arm_b full; else iterate (per-lever ablation) until the stack is understood. Folds in #5 (pose-Jensen) as a
third arm if cheap. This is the OPTIMAL-FORM gate before the long LOCAL burn.

## 5. What was applied this unit (NO-FAKE)
- `--muon-lr-floor-fix` flag added to the basin launcher + the **resume-guard hardened** (manifest persists the
  flag; fail-closed on a toggled flag at a has_muon checkpoint; 12/12 resume tests green) — commits `0065a7b05`
  + `de2fbb4a8`.
- The decisive daemon is **PAUSED** (checkpoint intact at ~ep 2307) pending the round-2 verdict — so the re-fire
  is the OPTIMAL config (bind-all arm_b), not the sub-optimal split-by-head basin launcher.
- NOT yet applied (correctly gated on the round-2 measurement): the taper switch (arch change), the
  margin_hinge/equimarginal levers (loss/schedule changes), the pose-Jensen fix (loss change) — all MEDIUM-ish
  faithfulness, so they earn their place by the A/B, not by assertion.

## Observability surface
Every number cites a lens artifact + file:field. Daemon paused (out-dir torch_vehicle_full_mps_basin_bc20_n600).
Round-2 actuator: `launch_bind_all_taper_ab.py`. Axis `[contest-CPU advisory]`, score_claim=false, pointer 0.19110.

## Canonical-vs-unique decision per layer
This synthesis REUSES the bind-all launcher (all levers already wired — ADOPT). The only new code this unit is
the muon-floor-fix flag + resume guard (a fork of the resume-basis-guard pattern, justified by Lens-D). The
optimization levers fork the vendored PR95 recipe per UNIQUE-AND-COMPLETE — each earns its fork by measured ΔS.
