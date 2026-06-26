# Scaling-Law d_seg Capacity Fit + Rate-Optimal Witness Solve (2026-06-26)

**Agent:** SCALING-LAW APPLICATION (#170 math-optimal joint solver), grounded in Lilian Weng's
2026 scaling-laws framework (Hestness offset E set by basis; Sharma-Kaplan alpha=1/d, d set by
basis manifold dim; linear rate-in-N). **Authority:** `[contest-CPU advisory]` NON-PROMOTABLE
(no exact row moved; pointer UNMOVED contest-CPU **0.19109982**). **NO-FAKE:** fit on CLEAN
REALIZED-through-R CPU-torch d_seg ONLY (B3 harness == upstream oracle, 11-dec, git 8cceaa072);
proxy/generator-argmax/EMA-lagged/MPS/under-trained points DISCARDED from the fit (cited as
under-training evidence). means!=ends: this feeds the capstone launch config (optimal N + whether
basis-change is mandatory).

## THE MODEL
`d_seg(N) = E + A/N^alpha` ; `bytes(N) ≈ c0 + k_b·N` (linear, measured 2-pt) ;
`S(N) = 100·(E + A/N^alpha) + (25/37545489)·(c0 + k_b·N) + sqrt(10·d_pose)`.
Rate-optimal: `N_opt = (100·alpha·A / k)^(1/(alpha+1))`, `k = 25·k_b/37545489`.

## REALIZED POINTS GATHERED
| N (params) | bc | d_seg | d_pose | bytes(zip) | provenance | axis / use |
|---|---|---|---|---|---|---|
| 83,356 | 20 | **0.0025607** | 3.04e-4 | 89,211 | `reports/base_ch20_clean_verify_builder4_20260626.json` (+ headline 0.002601, g3 0.002399, n60 0.002510 — agree) | REALIZED-through-R, 600-pair. **FIT POINT 1** |
| 228,958 | 36 | **5.6e-4** (600pair) / 6.02e-4 (8pair) | 2.36e-5 | 178,417 | `reverse_engineer_pr95_prune_capacity_rd_20260623.md` (PR95 converged teacher, inflate→R→exact SegNet) | REALIZED-through-R, **CONVERGED**. **FIT POINT 2** |
| 231,559 | 36 | 0.009619 | 0.01048 | 230,009 | `CLEAN_pr95_bc36_faithful_20260623/best/best_meta.json` (stage-1 early-stop) | REALIZED but **UNDER-TRAINED** — DISCARD (Michaud evidence) |
| 114,710 | 24 | 0.002855 | 1.07e-4 | 108,300 | `capstone_capacity_ablation_2x2_20260611/bc24_p48` (n=48, the 2×2 arm re-audit flagged DIVERGED) | REALIZED but SUSPECT/under-trained — DISCARD |
| 83,356 | 20 | 0.023928 | 0.1087 | 82,601 | `reveng_pr95_prune/kd_bc20.json` | prune+KD subspace artifact — DISCARD |
| — | 24/28 | 0.016–0.024 | — | — | `reveng_pr95_prune/kd_bc28.json` | prune+KD cliff — DISCARD |

**Clean CONVERGED-state pair = {bc20 0.00256 @ 83K, bc36-converged 5.6e-4 @ 229K}.** The
under-trained bc24/bc36-stage1 points show **anti-scaling** (more params → worse d_seg) which is a
TRAINING artifact, NOT capacity (re-audit: "0 of 11 capacity-walls proven fundamental"). Fitting
through them would manufacture a fake flat/positive exponent → excluded per NO-FAKE.

## FIT (2 converged points → 2-param E=0 power law; 3rd param E underdetermined, handled by bound)
- **alpha = 1.504** (8-pair bc36 → 1.433). **Sharma-Kaplan manifold dim d = 1/alpha = 0.665.**
- **A = 6.48e4.** Residual = 0 (exact 2-pt). E underdetermined from 2 pts → **bounded by existence
  proof: E ≤ 5.6e-4** (the converged bc36 value; E=inf_N d_seg ≤ d_seg(229K)).
- Caveat: bc20 0.00256 may itself be above its converged floor (the same-N bc36 pair proves 17×
  training-state spread), so the TRUE converged alpha may be shallower; the verdict below is robust
  to this because it is bound by the CONVERGED bc36 point + the rate term.
- **Rate model (2-pt linear, total zip bytes):** `bytes(N) = 38,141 + 0.6127·N` ⇒ marginal
  **0.6127 B/param**, `k = 4.08e-7`/param, fixed_rate = 0.0254. (Task avg 1.07 B/param is the
  bc20 *average*; the *marginal* 0.61 is the correct N-slope — both reported.)

## N_opt + PREDICTED S
- **N_opt = 219,719 params ≈ bc36** (8-pair: 221,590). d_seg(N_opt)=5.96e-4, rate_term=0.115.
- **S(N_opt) = 0.1931** (pose=solved-sidecar 0.0184) / 0.2297 (pose=bc20 standalone 0.0548).
- **MIN S over ALL N in the HNeRV basis (E=0 best-case) = 0.1931 at N≈219.7K** — i.e. the
  rate-optimal point in this basis IS essentially the frontier. Pure capacity scaling **does NOT
  reach sub-0.15** (best = 0.193).

## DECISIVE VERDICT — E < 7.3e-4? YES, but NECESSARY-NOT-SUFFICIENT → BASIS-CHANGE MANDATORY
1. **E < 7.3e-4: YES (settled by existence proof).** bc36-converged realizes d_seg = 5.6e-4 <
   7.3e-4 at finite N=229K ⇒ E ≤ 5.6e-4 < 7.3e-4. The d_seg *number* IS reachable by capacity.
2. **BUT capacity CANNOT deliver sub-0.15.** The 7.3e-4 target was specified at **bc20's CHEAP rate
   (0.06)**. To reach d_seg≤7.3e-4 in the HNeRV basis needs **N≈192K → 155,800 B → rate 0.104**
   (NOT 0.06). The cheap-rate-AND-low-d_seg combination **does not exist** in the HNeRV/RGB basis.
   Rate-optimal S = 0.193. **This IS the measured trilemma resolution** (bc20 cheap-rate/high-d_seg,
   bc36 low-d_seg/expensive-rate; the rate-optimal interpolation is still ≈0.19 — the witness/
   basis-change is the only non-dominated arm).
3. **QUANTIFIED basis requirement for sub-0.15** (any ONE suffices; combine for margin):
   - **Hestness downshift:** drop d_seg by **3.55×** at the FIXED ~88KB bc20 budget
     (0.00256 → ≤7.2e-4 at SAME bytes). = the "capacity-ALLOCATION not -SHORTAGE" lever (prior
     DAG big-finding), now quantified: 89% of d_seg in 5% of pixels → re-route SAME bytes.
   - **Lower manifold dim d (steeper alpha):** holding the HNeRV carrier, min S crosses 0.15 at
     **alpha ≥ ~2.9 (d ≤ 0.34)** vs current d=0.67 — a ~2× steeper descent. (alpha=3.0→S 0.143;
     4.0→0.128; 5.0→0.119.) The **step-native / partition-indicator basis** (topology-matched to
     the piecewise-constant argmax, O(1) params/edge) is exactly the lower-d basis. **MANDATORY.**
   - **Cheaper carrier:** the task-space witness storing the ~8-dim sufficient statistic (not RGB
     rendering weights) cuts c0 (38KB→~15-20KB) and k_b → a **1.77× cheaper carrier** delivers
     d_seg=7.3e-4 within the ≤88KB sub-015 budget. **seg=pose fusion** lowers rate further.
   - Carrier floor note: even a perfect-descent basis ON the HNeRV carrier floors at
     fixed_rate+pose = **0.044**, so the binding lever is the DESCENT STEEPNESS (basis dim d),
     not the carrier — but the cheapest path combines lower-d basis + task-space carrier.

**VERDICT = (b) with the E-number caveat: capacity reaches the d_seg NUMBER (E<7.3e-4) but NOT
sub-0.15 S. The step-native lower-d basis + task-space carrier (+ seg=pose fusion) are MANDATORY,
not optional.** This is a quantitative confirmation of the established NON-RGB TASK-SPACE WITNESS
CAPSTONE frontier. Capstone launch config: do NOT scale N in the HNeRV basis (dominated, min S
0.19); launch the witness with (i) a lower-d step-native/directional basis targeting alpha≥3 /
3.55× fixed-byte d_seg downshift, (ii) margin-weighted capacity-routing onto the 5%-of-pixels
annulus, (iii) chroma active, (iv) composed with the stored-pose sidecar.

## SCALING-REGIME NOTES
- **Lovelace (memorization inversion):** this is single-video memorization (test=train), so the
  overfitting penalty is INVERTED — bigger N never hurts test d_seg in principle; the realized-vs-
  train gap is purely the deployment quantization (EMA shadow + int8), already in the realized
  harness. The observed anti-scaling (bc24>bc20, bc36-stage1≫bc36-converged) is therefore NOT a
  generalization gap; it is **under-training**, which Lovelace's regime says is the ONLY way bigger
  models look worse. Confirms the discard of bc24/bc36-stage1.
- **Michaud quantization (rare-chunk long-tail):** the bc36 SAME-N pair (5.6e-4 converged vs 0.0096
  stage-1, **17× by training state alone**) is direct evidence the descent is **training-limited**,
  not capacity-limited. The hard residual = the ~8-dim lane-orbit (the "rare chunk") whose quanta
  are learned LAST (slow tail) — the plateau is under-training, not a wall. Implication: the capstone
  needs CONVERGENCE (long resumable sweep) and a basis that puts the rare lane-quanta in cheap
  coordinates (step-native), not more raw N.
- **Basis-sets-d (Sharma-Kaplan):** measured d=0.665 for the HNeRV/RGB basis. The whole sub-0.15
  thesis is that the TASK-SPACE basis has a strictly LOWER d (it amortizes only the argmax partition,
  not RGB) → steeper alpha → reaches the same d_seg at far fewer bytes. The act_screen_stepbasis /
  directional-Fourier screens are the basis-d experiments to confirm this (next-slot: realize their
  d_seg through R to MEASURE their alpha vs 0.67).

## MEMO + DAG FEED + COMMIT
- Memo: this file. DAG: FEED-bq appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- Reproduce: the fit/solve is the two inline python blocks in the session (scipy/numpy, $0 CPU).
