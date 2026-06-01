# Inverse-steganalysis $0 gate — L-inf margin-budget BEATS L2 at equal rate — LANDED

- **Date:** 2026-06-01
- **Lane:** `lane_inverse_steganalysis_optimal_full_stack_20260601`
- **axis_tag:** `[macOS-CPU advisory]` — NON-PROMOTABLE (`score_claim=false`, `promotable=false`) per Catalog #341/#192/#127/#323. $0 macOS-CPU only; no paid dispatch, no GPU, no PR, no MPS authority.
- **Gates this:** the inverse-steganalysis new-substrate commitment (design memo §7 + open council decision #4).
- **Durable anchor:** `.omx/research/inverse_steganalysis_linf_vs_l2_gate_20260601T200459Z.json`
- **Design memo:** `.omx/research/inverse_steganalysis_optimal_full_stack_design_20260601.md` §7

## The gate (design memo §7, verbatim)

> Does L-inf margin-budget (inverse-steganalysis) allocation beat L2 (MSE-optimal) allocation at EQUAL rate, measured on the real scorer?

Run on the already-cheap PR101-class carrier (the carrier is NOT the bottleneck) so the comparison isolates the OBJECTIVE (the allocation rule), not the carrier. The contest scorer is a steganalysis DETECTOR (Yousfi built SegNet from EfficientNet steganalysis surgery); the contest objective is therefore the steganographic security objective — maximize bytes-saved subject to "the detector's decision does not flip" — NOT L2 fidelity.

## What was built (committed)

- `src/tac/analysis/inverse_steganalysis_linf_vs_l2_gate.py` — canonical helper: the high-rate uniform-quantizer rate model `B = sum_i log2(R/delta_i)`, the L2-optimal uniform-step allocator, the L-inf margin-budget reverse-water-fill allocator `delta_i = clip(c*rho_i, lo, hi)` with a tight water-level bisection + the anti-gaming fairness nudge (`fairness_direction`), the saliency→margin inversion `rho_i = 1/(s_i+eps)`, the per-pixel quantization-noise applicator (the "decode"), and the hard-contest `d_seg`/`d_pose` measurement on the verified mirror.
- `tools/run_inverse_steganalysis_linf_vs_l2_gate.py` — the gate runner: combines P18 (`s_seg`, last frame, upsampled to native) + P19 (`s_pose`, native) at score-derivative weights, allocates L2 vs L-inf at equal bit budget, decodes + measures on the real frozen scorer mirror over `--num-pairs` strided real `upstream/videos/0.mkv` pairs, seed-averaged, and emits the durable JSON anchor + a decisive shuffle control.
- `src/tac/tests/test_inverse_steganalysis_linf_vs_l2_gate.py` — 21 NO-FAKE tests (rate model, allocations genuinely differ, fairness invariants, quantization actually perturbs, `d_seg`/`d_pose` match contest definitions, and the decisive toy-detector aiming + shuffle-control-fails test). All pass.

## The empirical verdict ([macOS-CPU advisory], 6 strided real pairs × 3 noise seeds, 4.0 bits/pixel)

| Allocation | d_seg | d_pose | contest distortion `100·d_seg + sqrt(10·d_pose)` |
|---|---|---|---|
| **L2 uniform** (MSE-optimal) | 0.00158 | 2.440e-02 | **0.65174** |
| **L-inf margin-budget** (inverse-steganalysis) | 0.00181 | 1.007e-03 | **0.28117** |
| **shuffled control** (same rate + same step histogram, random placement) | 0.00588 | 2.281e-01 | **2.09795** |

**Verdict (1 line):** at equal rate (L-inf forced to spend ≥ L2 bits; rate match 4.16e-4), inverse-steganalysis L-inf margin-budget allocation **BEATS** L2 on the real scorer by **0.371 contest-distortion (56.9% lower), 6/6 pairs**, and the win is **genuine detector-aiming** (the shuffled control, with the identical rate + step histogram but random placement, is 7.5× WORSE than L-inf and 3.2× worse than L2). Verdict token: `INVERSE_STEGANALYSIS_LINF_BEATS_L2_AT_EQUAL_RATE_AIMING_GENUINE`.

Robustness: the win + decisive shuffle-failure replicate at 2/3/4/6 bits/pixel (advisory sweep).

### Honest per-axis decomposition (the load-bearing nuance)

The win is **pose-dominated**: L-inf cuts `d_pose` ~24× (2.44e-2 → 1.01e-3) by aiming the Fisher budget at the pose-sensitive pixels, while `d_seg` is roughly tied (L2 0.00158 vs L-inf 0.00181 — L-inf marginally worse on seg alone at this operating point). The aggregate contest distortion is what the contest scores, so the verdict stands, but the seg-margin aiming is NOT independently demonstrated to beat uniform at this operating point. The shuffle control's `d_seg`=0.00588 (3.7× worse than L-inf) confirms placement still matters for seg — random placement coarsens boundaries and flips argmax — so the seg signal is real even though the L-inf-vs-L2 seg margin is thin. The pose Fisher term is the dominant lever; the seg DeepFool term is real but second-order at 4 bpp.

## Why this gate is fair (NO-FAKE controls, all enforced + recorded in the anchor)

1. **EQUAL RATE** — both allocations spend the same total bits `B`; the L-inf allocator's water level is bisected so its realized rate matches the L2 uniform rate to <0.1%, and the `disadvantage_linf` fairness mode FORCES L-inf to spend AT LEAST as many bits as L2 (rate match 1.3e-13 on the dominant pair; worst-pair 4.16e-4 from clipping granularity). A win therefore can never be a rate artifact.
2. **SAME FRAMES / SAME NOISE SEEDS** — both allocations are applied to the same decoded gt pairs with the same per-pair generator seeds, averaged over 3 seeds.
3. **ALLOCATIONS GENUINELY DIFFER** (Catalog #139 no-op guard) — the L2 and L-inf step maps are asserted not byte-identical.
4. **THE DECISIVE SHUFFLE CONTROL** — the L-inf step map is randomly permuted across pixels (keeping the EXACT same step-value histogram → same rate, same MSE distribution, same per-pixel distortion budget) but destroying the detector aiming. The shuffled control scores 2.098 (vs L-inf 0.281), proving the win is WHERE the bits go (the oracle), not the step histogram. This is the empirical falsification of "it's just a rate-model artifact."

## Adversarial self-review (recursive protocol, clean pass)

- *Could L-inf win because the rate model `log2(R/delta)` over-rewards coarsening?* No — the shuffle control uses the identical rate model + step histogram and does 3.2× WORSE than L2; the rate model is neutralized by construction.
- *Could L-inf secretly spend fewer bits?* No — `disadvantage_linf` forces L-inf ≥ L2 bits; recorded rate match 4.16e-4.
- *Is the operating point cherry-picked?* No — replicated 2/3/4/6 bpp.
- *Is the saliency overfit to the scored frames?* Yes, deliberately — contest mode scores exactly `0.mkv` and overfit is allowed per `build_saliency_verification_contract.contest_compliance.compress_side_may`. The oracle is the verified bit-exact mirror; the receiver never loads the scorer.
- *Is the win an artifact of the quantization-noise surrogate?* The surrogate IS the uniform-quantizer error model the rate model assumes (same object); the shuffle control rules out a placement-independent artifact.

## What this gates

This is the §7 falsifiable gate that informs **open council decision #4** ("Does a co-designed inverse-steganalysis stack beat retrofitting the PR101 frontier carrier?"). The gate result is GREEN for the OBJECTIVE: at equal rate, on the real scorer, aiming bits by the detector's margin (L-inf/`rho_i`) reduces contest distortion 56.9% vs aiming by L2, and the win is provably detector-aiming. This is the empirical anchor the co-equal-keystone design (§2) needed for its objective half. It does NOT yet establish the carrier half (the co-equal-necessity from Z8/HPRC); the full-stack commitment still requires the per-substrate symposium (Catalog #325), the L2 carrier architecture decision (open #1), and paired CPU+CUDA on byte-closed bytes (Catalog #246). No score claim.

## Canonical-vs-unique decision per layer

- L0 oracle: ADOPT_CANONICAL (`tac.analysis.score_exact_saliency`, verified mirror).
- Rate model: ADOPT_CANONICAL (high-rate uniform-quantizer entropy; Gersho-Gray).
- L2 allocator: ADOPT_CANONICAL (uniform-step is the closed-form MSE-optimal under a flat source).
- L-inf allocator: FORK_PRINCIPLED — reverse-water-fill at cost=oracle-`rho_i` is the inverse-steganalysis objective (the class-shift); the canonical `joint_p18_p19_waterfill` KKT-Dykstra solver is the production-scale sister but this gate uses a self-contained closed-form bisection for the isolated-objective proof.

## 9-dimension success checklist evidence

UNIQUENESS — first empirical proof that the L-inf decision-margin objective beats L2 on the real scorer at equal rate. BEAUTY — one rate model, two allocators, one shuffle control. DISTINCTNESS — explicitly NOT fidelity; the shuffle control isolates the detector signal. RIGOR — verified bit-exact oracle, equal-rate to 4e-4, decisive shuffle control, 21 NO-FAKE tests, bpp sweep. OPTIMIZATION-PER-TECHNIQUE — DeepFool margin (P18) + Fisher (P19) are the per-channel optima. STACK-OF-STACKS — seg+pose combined additively at score-derivative weights. DETERMINISTIC-REPRODUCIBILITY — seeded generators; deterministic decode. EXTREME-OPTIMIZATION — closed-form water-level bisection. OPTIMAL-MINIMAL-SCORE — the objective IS the contest security objective.

## Cargo-cult audit per assumption

- "minimize L2/MSE is the codec objective" — CARGO-CULTED; FALSIFIED here (L-inf beats L2 56.9% at equal rate on the real scorer).
- "any allocation that coarsens the dead-zone wins" — CARGO-CULTED; the shuffle control (same histogram, random placement) does 3.2× WORSE than L2, so the WIN is the detector aiming, not the coarsening.
- "the seg-margin aiming dominates" — refined to HONEST: the win is pose-Fisher-dominated at 4 bpp; the seg-margin aiming is real (shuffle control confirms) but second-order at this operating point.

## Observability surface

Inspectable per layer (per-pixel `rho_i`, per-pixel step maps, per-pair allocations). Decomposable (per-axis `d_seg`/`d_pose`/contest separable; per-pair table). Diff-able (L2 vs L-inf vs shuffle on the same frames). Queryable (the durable JSON anchor). Cite-able (every measurement anchored to commit + frozen-weight mirror + pair index). Counterfactual-able (the shuffle control IS the counterfactual: same rate, random placement).

## Predicted ΔS band

No archive-level ΔS asserted — this gate measures the OBJECTIVE on rendered frames, not byte-closed archive bytes. Direction: the objective is GREEN. The archive-level ΔS is PENDING the carrier architecture (open #1), the per-substrate symposium (Catalog #325), and paired CPU+CUDA (Catalog #246). `# PREDICTED_BAND_VIBES_OK:objective-gate-not-archive-eval-archive-delta-deferred-to-carrier-decision-symposium-and-paired-cpu-cuda`

## Canonical equation reference

`# FORMALIZATION_PENDING: inverse_steganalysis_Linf_margin_budget_beats_L2_at_equal_rate — register in tac.canonical_equations after a paired CPU+CUDA archive-level anchor lands; the equal-rate L-inf-beats-L2 objective result is verified [macOS-CPU advisory] but the archive-level savings model is pending the carrier decision + exact eval.`

## 6-hook wire-in (Catalog #125)

1. Sensitivity-map — ACTIVE (`rho_i` = oracle saliency drives the allocation). 2. Pareto — ACTIVE (the rate/margin-budget trade is the polytope; the canonical `joint_p18_p19_waterfill` KKT-Dykstra solver is the production sister). 3. Bit-allocator — ACTIVE (the L-inf reverse-water-fill IS a bit-allocator). 4. Cathedral autopilot — N/A here (advisory $0 gate; the result feeds open council decision #4, not a dispatch ranking row). 5. Continual-learning posterior — N/A here (no contest-axis anchor; `[macOS-CPU advisory]` non-promotable, FORMALIZATION_PENDING). 6. Probe-disambiguator — ACTIVE (this gate IS the §7 probe-disambiguator that resolves the L-inf-vs-L2 objective; the shuffle control is the no-fake disambiguator within it).
