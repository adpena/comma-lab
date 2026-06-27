---
title: "DYNAMICAL-PARTITION FAMILY at OPTIMAL FORM — Morse-Smale + Neural-CA re-opened (operator RULE-6) → VERDICT: COMPOSE-AS-RATE-LEVER (witness-seeded residual-sharpener + free margin-located residual sidecar); DEFER both standalones (measured-dominated by the SDF witness)"
authority: "[macOS-CPU advisory] research-signal — NON-PROMOTABLE; exact pointer UNMOVED 0.19110"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-27
dag_feed: FEED-fh
verdict: COMPOSE_AS_RATE_LEVER_AND_RESIDUAL_SHARPENER__DEFER_BOTH_STANDALONES
producer_scripts:
  - scratchpad/feed_fh_separatrix_residual.py   # separatrix coding + free-margin-locator measurement
  - scratchpad/feed_fh_flipres.py               # flip-residual sidecar sizing
inputs:
  - experiments/results/mlx_fleet_gt_cache/gt_n96.npz        # lstars (argmax) + margins (FREE locator) + gt_poses
  - experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz  # full 600-frame GT partition
cross_refs:
  - .omx/research/morse_smale_partition_codec_feasibility_20260626.md          # #180 / FEED-cg — REVISE rate-dominated
  - .omx/research/generative_axis_continuous_texture_nca_AMBER_20260619T020000Z.md   # #146 AMBER 0.00337 fragile
  - .omx/research/generative_axis_nca_amortized_capacity_break_RED_20260619.md  # the optimal-form NCA RED (stabilized + amortized + capacity sweep)
  - .omx/research/generative_axis_nca_dseg_core_gate_20260619T013000Z.md        # #143 flat-partition RED
  - sub015_DAG... FEED-fg (margin field AUC 0.943 free flip-band locator; ringing FALSIFIED)
  - sub015_DAG... FEED-et (all-class flip decomposition 61% geometric / 39% Movable learned)
  - sub015_DAG... FEED-fd (sequence residual Lane+Movable HIGH-rank, amortization-saturation wall)
  - sub015_DAG... FEED-ff (coupled-oscillator DEFER; deterministic case = the measured AMBER NCA)
  - task #72 (margin-conditional residual coder) / #137 boundary sidecar / #138 lane prior / Lever-D
---

# Dynamical-partition family at OPTIMAL FORM — Morse-Smale + Neural-CA, re-opened

**Operator (2026-06-27): "Morse smale and neural CA are very interesting... I just love all
this math. Remember not to be pessimistic based on first results and sandbox playing mentality."**
This RE-OPENS the family at optimal form per the JANKY-PROTOTYPE-RE-OPEN discipline (prior verdicts
falsify IMPLEMENTATIONS, not the PARADIGM). Done in that spirit — generously, looking for what WOULD
make it win — but NO-FAKE honest about the measured walls. All numbers `[macOS-CPU advisory]`,
NON-PROMOTABLE; **this unit did NOT move the pointer — UNMOVED 0.19110; a design+feasibility verdict
is a MEANS, not the end.**

## 0. The headline (read first)

The dynamical-partition family does NOT beat the converging SDF level-set witness as a STANDALONE
(both Morse-Smale codec and from-scratch NCA are measured-dominated). But the deep-math convergence is
REAL and yields ONE genuinely non-dominated, sub-0.19-plausible re-opening: a **witness-seeded
residual operator** — Morse-Smale separatrix-residual coding (free-located by the margin field) and/or
an NCA residual-SHARPENER (not a from-scratch generator) — composed ON the witness, targeting the
rare-class (Lane+Movable) long-tail. Verdict: **COMPOSE-AS-RATE-LEVER + DEFER both standalones.**

## 1. The deep-math convergence — VERIFIED on data (the beautiful part)

A Morse-Smale complex partitions a domain by the gradient flow of a scalar field: basins of minima,
bounded by **separatrices** through saddles. Our SDF level-set witness's `argmax_k φ_k` IS a *soft
Morse-Smale partition*: basins = which field wins; boundaries = ties (`φ_top1 = φ_top2`) = separatrices.

**The FEED-fg correction is the keystone:** the separatrices are the **margin-ZERO set** of the
witness's OWN margin field `m = φ_top1 − φ_top2`, NOT the basis ringing (ringing FALSIFIED — it marks
confident sharp edges, anti-informative in the precarious band, AUC 0.417). So:

> **margin-zero level set = Morse-Smale separatrix graph = precarious flip-band = the FREE decode-side
> locator** — all the same set, and the witness already computes the margin field for free (rule-118).

**MEASURED (gt_n96, 48 frames, `feed_fh_separatrix_residual.py`):** mean margin **0.476 on the
separatrix vs 5.71 in the interior (12× contrast); separatrix-detection AUC = 0.9987.** The margin
field is a near-perfect FREE separatrix locator — the Morse-Smale graph is generated for zero bytes by
the witness forward pass. This VERIFIES the convergence law on the exact frozen-SegNet GT. (AUC 0.999
here = ALL separatrices; FEED-fg's 0.943 = the witness-flip subset — both confirm the identity.)

## 2. The three re-opened failure mechanisms — pinned (not vibes) + the optimal-form fix

| prior | what it actually was | measured cap | the real wall | optimal-form fix |
|---|---|---|---|---|
| **#180** (FEED-cg) Morse-Smale codec | cv2 conn-comp → contours → Douglas-Peucker (polygon shape coding, NOT a critical-point/separatrix GRAPH) | indep RD-opt eps0.5: d_seg 5.57e-4 (BELOW capstone!), **S 0.37**; temporal coding fails (DP verts incoherent, motion-comp 3.3%) | **pure RATE** — re-specs every frame; witness amortizes smooth structure into FREE shared weights, MS does not | don't code the full partition; code only the **residual the witness misses**, free-locate via margin |
| **#143** flat-partition NCA | NCA → 5-class logits → flat colour lookup | realized d_seg ~0.02, boundary_band 0.353 | flat-fill survives R **WORSE** (grown boundary fuzzier than polygons; "free detail" is the WRONG kind for d_seg) | continuous texture (→ #146) + train-through-R |
| **#146 AMBER → amortized RED** continuous-texture NCA | from-scratch RGB grower from a random seed | single-frame 0.00337 (1.31× frontier) but **amortized 0.013**; capacity curve **INVERTS** (bigger rule 10× worse); convergence FRAGILE then SOLVED | (a) survival wall (boundary band floors 0.079) + (b) rate/d_seg tension + (c) deep-unroll optimization difficulty | **don't grow from a seed** — SEED FROM THE WITNESS and only SHARPEN the residual (tiny basin near identity) |

**Crucial honesty:** the operator's "fragility is fixable" was ALREADY DONE — the amortized
capacity-break RED fixed convergence (soft tanh **state-bound** = Mordvintsev alive-masking surrogate;
**drop the pool** — the pool DEstabilizes texture-regression-through-a-frozen-scorer; multi-restart
keep-best) and STILL capped at 23× frontier d_seg, with the SDF witness's measured n96 **0.00124**
dominating the NCA's best faithful result. So a NEW from-scratch NCA build is NOT warranted. The
non-dominated re-opening is the *residual operator* role below, which the prior gates never tested.

## 3. Morse-Smale separatrix coding — byte estimate vs the witness (MEASURED)

`feed_fh_separatrix_residual.py` (eps0.5, the #180 capstone-fidelity setting), per-frame:

| coded set | B/frame | KB total | **S_rate** | verts/fr | regions/fr |
|---|---:|---:|---:|---:|---:|
| ALL classes (full partition) | 875 | 513 | **0.350** | 921 | 25 |
| RARE only (lane 1 + movable 3) | 377 | 221 | **0.151** | 375 | 22 |

Even the rare-class-ONLY arc set costs **S_rate 0.151 standalone** (~80% of the whole 0.191 frontier):
lane markings are thin+fragmented and Movable is multi-component, so the *full* rare-class geometry is
expensive (375 verts/fr). **Full-arc coding is the wrong sizing for COMPOSE** — the witness already
places most of this boundary (n96 d_seg 0.00124). The sidecar must code only the **flip residual**.

### The flip-residual sizing (`feed_fh_flipres.py`) — the decisive bound

The witness needs to fix only ~**118 px/frame** to halve d_seg toward frontier (0.00124→~0.0006). But
the margin-gated precarious set is far LARGER than the wrong subset:

| margin gate τ | precarious px/fr | boundary coverage | correction B/fr (full label) | S_rate |
|---|---:|---:|---:|---:|
| 0.25 | 1309 | 30% | 331 | 0.132 |
| 0.50 | 2598 | 57% | 653 | 0.261 |
| 1.0 | 5044 | 93% | 1245 | 0.498 |

**The honest wall:** the margin field is a perfect SEPARATRIX locator but NOT a flip-RESIDUAL locator —
it marks the whole separatrix (~1300–9200 px), of which the witness gets MOST right (only ~244 wrong).
To code only the ~244 wrong pixels you need the witness PREDICTION (free at decode) to define the
residual, then a subset-selection over the margin-gated candidates (~244 of 1309 → ~73 B positions +
~70 B labels ≈ **143 B/fr → S_rate ~0.057**). That is exactly the **#72/#137/#138 residual-coder
regime** — a sub-0.19 nudge, NOT the sub-0.15 path. The Morse-Smale framing SHARPENS that family (free
margin locator, rigorous "residual = margin-gated witness-wrong subset" definition, rule-118 separatrix
geometry) but does not change its order of magnitude.

## 4. NCA stabilization — the optimal-form design (the operator's ask) + the reframe that matters

The amortized RED's reusable convergence findings (system intelligence): (i) state-bound (tanh) is
load-bearing; (ii) the Mordvintsev POOL is DEstabilizing for scorer-regression (feeds grown states
back → unbounded → NaN); (iii) larger rules scatter (deep-N-step unroll harder to optimize → bigger =
worse). The optimal-form stabilizer stack:

1. **Identity/zero-init last layer** (rule starts as identity `dx≈0`) + **small residual step ε~0.1** —
   directly fixes the large-rule scatter: the deep unroll starts stable, capacity is a perturbation.
2. **Spectral / Lipschitz norm on the rule weights** — makes the update a contraction (converges by
   Banach) AND structurally enforces the 1-Lipschitz the SDF witness wants (FEED-ew Eikonal).
3. **State-bound tanh** (keep — RED-proven) + **per-step grad-norm + LR warmup** (keep — #143-proven).
4. **Train-through-R** (keep — AMBER cut boundary band 0.35→0.079) + **gradient checkpointing** to make
   deterministic CPU training tractable within the 10 GB floor (kills the MPS-non-determinism collapse).
5. **THE REFRAME (the non-dominated role):** the NCA is a **RESIDUAL-SHARPENER SEEDED BY THE WITNESS**,
   input = witness margin + class fields, output = a sharper margin on the precarious band — NOT a
   from-scratch RGB grower from a random seed. This dissolves the entire convergence-fragility class
   (the AMBER's collapse was "grow a coherent frame from a seed"; here the iteration starts from the
   witness's already-near-correct field, basin ≈ identity), spends all capacity on the Movable
   medial-axis residual a single φ-field cannot represent (FEED-ew), and COMPOSES with (does not
   replace) the witness. This is the FEED-ff(c) reactivation territory (learn within-region-sync /
   cross-separatrix-desync coupling) made concrete and on-vehicle.

## 5. The synthesis — compose vs replace (honest)

Both families CONVERGE to the same place: **a witness-seeded residual operator on the Lane+Movable
long-tail, free-located by the margin field.** Morse-Smale = the *coding* view (separatrix-residual
sidecar, #72-class); NCA = the *iteration* view (residual-sharpener of the Movable medial-axis).
Neither REPLACES the SDF witness — the witness's measured n96 0.00124 dominates every faithful
dynamical result (Morse-Smale rate-dominated, NCA 0.00337 single / 0.013 amortized). Both are
RATE-SIDE / d_seg-residual levers, in the **sub-0.19 incremental regime** on current evidence.

## 6. VERDICT + reactivation/build triggers

- **Morse-Smale STANDALONE codec → DEFER** (rate-dominated, S_rate 0.151 even rare-only; #180 confirmed
  + extended). Salvage = measurement instrument + the rigorous residual definition below.
- **NCA STANDALONE from-scratch generator → DEFER** (amortized RED already optimal-form; dominated by
  the SDF witness). Do NOT re-build a seed-grown frame NCA.
- **COMPOSE-AS-RATE-LEVER → PROCEED to the $0 design / sub-0.19-nudge build** (NOT the sub-0.15 path):
  fold the Morse-Smale identity into the existing #72/#137/#138 residual coder — drive it from the FREE
  margin field (AUC 0.999 locator, zero locator bytes), code only the margin-gated witness-wrong subset
  (~143 B/fr → S_rate ~0.057), rule-118-generate the separatrix geometry. Convergent with FEED-et
  (geometric priors offload 61%), FEED-fd (ego/φ_k offload frees the code), Lever-D.
- **BUILD TRIGGER for the residual-SHARPENER NCA (the genuine open question, RULE-6 "what would make
  it win"):** does a **witness-SEEDED** sharpener (starting from 0.00124, identity-init, Lipschitz-
  normed, train-through-R) break the survival wall the from-scratch NCA could not (boundary band
  floored 0.079)? It is non-dominated ONLY if it gets the witness's *realized* d_seg below the SDF's
  converged-n600 floor at near-zero added rate. Gate it AFTER the SDF witness's converged-n600 d_seg is
  measured (the live decoder, pid 8806) — if the witness already reaches the sub-0.15 budget
  (5.2–6.5e-4), the sharpener is unnecessary; if it walls on the realized boundary band, the
  witness-seeded sharpener is the next move (this is exactly FEED-ff(c)).

## Observability surface

Every row records B/frame, KB, S_rate, vertex/region counts, precarious-set size, boundary coverage,
correction bytes, and the margin separatrix-AUC, recomputed from components. `[macOS-CPU advisory]`,
score_claim=false, pointer_moved=false. Scripts deterministic (cv2 conn-comp + numpy, no RNG except a
seeded balanced AUC sample). Reproducible from the two cached npz inputs above.

## Canonical-vs-unique decision per layer

GT load + realized-partition + #180 rate model + S formula = ADOPT_CANONICAL (reused for
apples-to-apples with FEED-cg/fg). The separatrix-residual decomposition (rare-class arc split + free
margin locator measurement + flip-residual sizing) = FORK (the unique COMPOSE-sizing this unit adds).
The witness-seeded residual-sharpener NCA reframe = FORK_PRINCIPLED (the non-dominated role the prior
from-scratch gates never tested).
