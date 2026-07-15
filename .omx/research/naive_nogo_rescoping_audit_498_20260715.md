# Naive→NO-GO re-scoping audit (#498) — verdict-scope ladder — 2026-07-15

**Mode:** `research_only=true`. READ-ONLY on code. No launch, score, archive, trainer edit, new DSL
lever, or new canonical equation. Pointer UNMOVED: **0.19108** submittable / **0.18804** borrowed
non-submission bank. All below is MEANS.

**Task:** apply the verdict-scope ladder (INSTANCE < FORMULATION < FAMILY < PARADIGM;
`[[verdict_scope_ladder_formulation_level_one_failure_not_family_dead_20260708]]` +
`[[feedback_no_naive_implementations_binary_nogo_optimal_form_before_verdict_20260714]]`) to every
throughput / wall-clock / compute-substrate / convergence / basis / carrier NO-GO / DEAD / DOMINATED
/ REFUTED verdict of the recent campaign. Find the FEW real naive→binary mis-scopes that hide an
optimal-form win; leave the genuinely-exhausted walls closed.

## Headline (honest, adversarial)

**The corpus is ALREADY re-scoped — signal is NOT lost.** A sibling ran the program-wide rescope on
2026-07-14 (`throughput_nogo_naive_rescope_audit_DAG_FEED_20260714.md` +
`codex_findings_throughput_nogo_naive_rescope_audit_*`), and the reformulations are TRACKED as 17
reactivation rows **D41–D53** in `.omx/state/deferral_ledger.md` (VERIFIED present: D41 D42 D43 D44
D45 D46 D47 D48a/b/c D49 D50 D51 D52a/b/c D53). Every NO-GO certificate I spot-checked
(`fixedpoint_qdq_rung2_nogo_certificate_*`, `optimal_metric_p0_raw_cosine_audit_*`) **already carries
its own `verdict_scope` = INSTANCE/FORMULATION + an explicit reformulation queue** — the apparatus did
the right thing. This memo's value is (a) adversarial confirmation that the scoping is honest,
(b) a single consolidated ranked table in the requested format, (c) the top-3 reopenable +
$0 probes, (d) the genuinely-dead controls that must stay closed.

**No new naive→binary over-scope found that was not already caught.** The 07-14 audit was thorough;
I am confirming, not overturning. (MEASURED: 17/17 D-rows present; 2/2 certificates carry
verdict_scope + reformulation queue.)

## Ranked table — most-reopenable first

Scope column is my adversarial re-classification (defaulting narrowest). "Form measured" = was the
negative measured on the NAIVE/first-cut form or the OPTIMAL form.

| # | Verdict (as recorded) | Source | Form measured | Honest scope | Reformulation + $0/cheap probe | Tracked |
|---|---|---|---|---|---|---|
| 1 | `NO_ADMITTED_PRECISION_IN_LADDER` (fast low-bit SegNet forward not authority) | `fixedpoint_qdq_rung2_nogo_certificate_*` | **NAIVE** — single GLOBAL uniform fixed-scale WnAn QDQ | **INSTANCE** (n600, uniform global scale) | Margin-adaptive MIXED-precision + per-channel scales: reuse the EXISTING n600 fixedpoint probe artifact (`fixedpoint_scorer_forward_n600_fresh_*.json`) and apply the interval-arithmetic per-layer argmax certificate `L_top1 > max(U_other)` to drive margin-waterfill bit allocation (high bits on the annulus, Fisher=margin 0.978). **$0** — recompute from cached JSON. | D41 |
| 2 | raw-cosine surrogate `NOT-ADMITTED` (teacher/provider match) | `optimal_metric_p0_raw_cosine_audit_*`, `surrogate_vjp_fidelity_metric_*` | **NAIVE** — cosine (curved rep-divergence) as the decision predicate | **FORMULATION** (wrong metric locus) | Decision-quotient / categorical-Fisher / functional-flip-preservation: recompute the whole-teacher & on-policy surrogate decision on EXISTING n600 centered-logit/quotient receipts using `q_F=uᵀF(p)u` winner-rival `|t|≤√(8·δ_KL/C_wr)` (the RIPO route-now law) instead of `1−cos`. **$0** — reuse cached VJP/logit receipts. | D42 |
| 3 | custom sparse-adjoint `1.0x realized NO-GO` | throughput audit (D43) | **NAIVE** — DENSE execution timing (arithmetic, no kernel) | **INSTANCE** (dense-exec measured; 2.2086× DERIVED ceiling NEVER built) | Build the custom sparse Metal kernel; **cheap-local** M5 Metal micro-bench of realized wall vs the 2.2086× DERIVED ceiling. Dense-arithmetic timing is inadmissible for the kernel verdict. | D43 |
| 4 | taper `+18% NO-GO` | throughput audit (D44) | **NAIVE** — under-converged checkpoint | **INSTANCE** (under-converged) | Converged-checkpoint n600 ON/OFF with identical EMA + exact non-treatment custody; local annulus form reported separately from the settled global. **$0-ish** — needs a converged ckpt A/B. | D44 |
| 5 | costate raw-ZOH K2 `NOT-ADMITTED` | throughput audit (D47) | **NAIVE** — raw zero-order-hold reuse | **FORMULATION** | Transported / event-triggered costate provider with full scalar/quotient parity; time K2 only after admission. Probe = drift proxy on existing telemetry. | D47 |
| 6 | YOPO `no speedup` (validate every step) | throughput audit (D48a) | **NAIVE** — n=1 exact-validation cadence | **INSTANCE** (cadence) | Sparse-audit cadence: cheap drift proxy + sparse exact audits + alternate split. **$0** — cadence sim on logged drift. | D48a |
| 7 | feature-ball `0.98x` (n58 gate) | throughput audit (D48b) | **NAIVE** — n58 first-cut gate | **INSTANCE** | Exact/interval suffix bound OR learned risk proxy + sparse exact audit at n600. | D48b |
| 8 | INSTANT three-state `charged ratio 0.5889` | throughput audit (D48c) | **NAIVE** — n1, no-Metal projected | **INSTANCE** | Native low-rank / custom backward primitive + broader calibration at n600. | D48c |
| 9 | SPS `separation NO-GO` (disengaged) | throughput audit (D46) | **NAIVE** — measured while screw/phase DISENGAGED (uninformative) | **INSTANCE** (disengaged = no signal) | Re-measure with real temporal engagement (screw/phase-engaged n600 telemetry); scalarization/stratified batching before PCGrad. | D46 |
| 10 | historical micro-batch `2–4x overturned` | throughput audit (D49) | **NAIVE** — n24, non-ABBA, old SHA | **FORMULATION** | Same-SHA uncontended B1/B2 ABBA on current V9 with full semantics + memory/descent parity. | D49 |
| 11 | ANE/CoreML `no joint 10x row` | throughput audit (D50) | **NAIVE** — mis-calibrated decision; advisory win banked | **INSTANCE** | Prove device residency + per-op precision + real-n600 worst-pair + net economics. | D50 |
| 12 | halo685 finite spatial-crop `exact ideal 1.0x` | throughput audit (D41/D42 owners) | **NAIVE** — per-pixel / input-crop spatial form | **INSTANCE** | Per-channel precision + SE-aware channel pruning (dodges the halo); dense-SE/sparse-decoder or cached-SE local student. | D41/D50 |
| 13 | Apple CPU backend `no CPU speedup` | throughput audit | **OPTIMAL-ish** for the tested backend (no oneDNN) | **FORMULATION** (this backend) | A genuinely different oneDNN / x86 / ExecuTorch / custom fixed-point backend. Stands narrowly. | repoint HOLD |
| 14 | HOSC `activation death` (fixed β) | throughput audit (D52b) | **NAIVE** — no-init, fixed β | **INSTANCE** | SIREN init + β 1→4 anneal with trajectory custody. | D52b |
| 15 | median-freeze `does not converge` | throughput audit (D52a) | **APPARATUS confound** — not a physics verdict at all | **INSTANCE** (no physics verdict exists) | Liveness-proven clean-checkpoint A/B with emitted update counts. | D52a |

### Confirmed hard walls — stay closed AT THE SPECIFIC FORMULATION (do NOT reopen these forms)

| Wall | Source | Why genuinely closed (this form) | Where the family stays OPEN |
|---|---|---|---|
| **fp-reorder megakernel drop-in** | `#356` / `fp_reorder_transform_bit_identity_wall_megakernel_nogo` | MEASURED: whole-step `mx.compile` fp32 fusion is NEVER bit-identical (grad Δ 2.3e-7…2.3e-5) AND speed marginal (GPU 1.12–1.21× closure ≈ ~5% e2e; CPU slower). Forks the score-faithful lineage → un-adoptable as a drop-in. `verdict_scope: formulation`. | Explicit-order custom Metal kernels (fused-R, grouped-backward ~17×) + gradient-free caches — ALL ALREADY ON (#432/v752). Exact-integer/explicit-order megakernel tracked as **D51**. |
| **flicker floor (geometry-only-warp / label-smoothing witness)** | L85 / `witness_converged_to_flicker_floor_*` / `budget_gate_overturn_exactpose_*` | MEASURED: witness sits at temporal-majority floor 0.005318; sub-0.15 (0.0008–0.0012) is 4.5–7× below. Un-warped hood contributes 32% of flips → intrinsic per-frame frozen-SegNet jitter, AA-irremovable, geometry-only cannot reach it. | Appearance-PHASE endgame (spikes DETERMINISTIC, proof 0.00086) — already BUILT (L86, default-OFF, SEAL+A/B owed). The path below the floor is a different vehicle, not a reopening of the warp form. |
| **dual-metric no-solve shortcut** | `dual_metric_no_solve_is_squared_hessian_not_fisher_natural` | MEASURED geometric fact (err ~9e-13, 600/600 SPD): `‖Δη‖₂²=ΔθᵀH²Δθ` is a SQUARED-Hessian metric ≠ Fisher-natural `ΔθᵀHΔθ`. A no-solve dual is a name-preserving FAKE. | The metric family is open via the TYPED `H⁻¹` solve (Fisher-natural cotangent geometry). Routes #500/#501/#504; NOT a family kill — just "use the solve, not the shortcut." |

## Genuinely PARADIGM-dead count

**0 PARADIGM-dead.** No verdict in the recent throughput/convergence/basis/carrier campaign closes a
FAMILY or PARADIGM. The 3 confirmed hard walls above are **FORMULATION-closed, family-OPEN** — each
already routes its optimal-form reformulation elsewhere (explicit-order kernels / appearance-phase /
H⁻¹ solve). Everything else is INSTANCE/FORMULATION with a tracked D41–D53 reformulation.

## Top-3 reopenable reformulations (naive verdict → optimal form → $0 probe)

1. **Margin-adaptive mixed-precision SegNet forward** (naive verdict: `NO_ADMITTED_PRECISION_IN_LADDER`
   — a SINGLE GLOBAL uniform fixed-scale QDQ flipped boundary argmaxes at every 8..24 bit width).
   → **Optimal form:** per-channel scales + margin-waterfill bit allocation (high bits on the boundary
   annulus per Fisher=margin 0.978, low bits on the flat interior), certified by the interval-arithmetic
   per-layer argmax bound `L_top1 > max(U_other)`.
   → **$0 probe:** recompute the argmax-exact minimum bit-width from the ALREADY-CACHED n600 fixedpoint
   probe artifact (`experiments/results/throughput_authority_ladder_20260714/fixedpoint_scorer_forward_n600_fresh_*.json`)
   under a per-channel + margin-region schedule instead of one global scale. No launch. (D41)

2. **Quotient/categorical-Fisher surrogate fidelity metric** (naive verdict: raw cosine `NOT-ADMITTED`
   as the teacher/provider-match decision predicate — `1−cos` is a curved rep-divergence, the wrong locus).
   → **Optimal form:** decision-quotient / Fisher trust-region `q_F=uᵀF(p)u`, winner-rival
   `|t|≤√(8·δ_KL/C_wr)`, `C_wr=p_w+p_r−(p_w−p_r)²` (the RIPO route-now law that corrects the false
   `‖Δlogit‖≤√(δ/p1)` transfer).
   → **$0 probe:** recompute the whole-teacher / on-policy surrogate admission decision on the EXISTING
   n600 centered-logit + input-VJP receipts using the categorical-Fisher predicate instead of cosine;
   report winner/rival/tie. Cached receipts, no re-run. (D42 + RIPO $0 route-now)

3. **Custom sparse-adjoint Metal kernel** (naive verdict: sparse-adjoint `1.0x realized NO-GO` — measured
   on DENSE arithmetic execution; the sparse kernel was never built).
   → **Optimal form:** a real custom sparse-adjoint Metal kernel exploiting the separatrix/annulus sparsity
   (the 2.2086× speedup was DERIVED but never realized).
   → **Cheap-local probe:** M5 Metal micro-bench of the realized kernel wall vs the 2.2086× DERIVED
   ceiling on a representative annulus mask (dense-arithmetic timing is inadmissible for the kernel
   verdict). Local, no paid dispatch. (D43)

## Authority labels

- **MEASURED:** the 3 hard walls (fp-reorder Δ 2.3e-7…2.3e-5 + speed 1.12–1.21×; flicker floor 0.005318,
  32% un-warped-hood flips; dual-metric err ~9e-13); the QDQ n600 `minimum_argmax_exact_arm=null`; the
  raw-cosine source scan (36 files); 17/17 D41–D53 rows present.
- **DERIVED:** the 2.2086× sparse-adjoint ceiling; the categorical-Fisher trust-region law; margin-waterfill
  bit-budget from the interval certificate.
- **INFERRED:** that the top-3 $0 probes will re-admit their families — they test the untested optimal
  forms; they are not yet outcomes.
- MPS/MLX/macOS-CPU-torch rows remain non-score MEANS throughout.

## Bottom line

The recent campaign's negatives are honestly INSTANCE/FORMULATION-scoped and their reformulations are
already queued (D41–D53). **0 genuinely PARADIGM-dead; 3 confirmed hard walls (all formulation-closed,
family-open).** The highest-EV reopens are all **$0/cheap** because they recompute from cached artifacts:
margin-adaptive mixed-precision (D41), categorical-Fisher surrogate metric (D42 + RIPO), and the
custom sparse-adjoint kernel bench (D43). No signal loss; the value is confirming the apparatus held
and surfacing the three cheap reopens for the next execution window.

Pointer delta: 0.0000000000.
