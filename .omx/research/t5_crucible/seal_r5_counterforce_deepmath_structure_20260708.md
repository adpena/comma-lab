# SEAL R5 — v7.5 birth-stack counter-force + ramp + P0 phase-2 — DEEP-MATH + STRUCTURE lens

**Date:** 2026-07-08 · **Lens:** counter-force + ramp + P0-delta, DEEP-MATH re-derivation + STRUCTURE ·
**Delta:** `git diff 2fb876c43..HEAD` (22e4e8827 ramp · 3d63478fd P0 phase-2 · 4890cd36b/f0386efff/etc.
spec+ledger) · **This is effectively ROUND 1 of sealing a MAJOR new surface** (counter not inherited
from the older v7.4 micro-diff) · **NO launch · pid 63069 + run dirs UNTOUCHED · $0 · pointer 0.19110
UNMOVED (means).**

STORES CONSULTED: `v75_birth_counterforce_20260708.md` · `p0_forces_{derivation,phase2_build}_20260708.md`
· `SPEC_v75_optimal_single_trunk` §8/§9/§10 · `SPEC_v8_perclass_decomposition` · `road_anomaly_probe` ·
probe P-A · FEED-{missingforces,mergediff,roadfloorfix,v8risks} · CLAUDE.md · operating manual.

## VERDICT: **CLEAN** (pass 1/3 of the R5 counter; 2 pre-registered-owed A/B scales flagged, non-blocking)

---

## 1. Chan-Vese equilibrium RE-DERIVATION (independent, from the level-set energy — not confirmed off the memo)

**Energy → flow.** One-sided area energy `E_area,c = (λ_c/2)·max(0, A_c − A_c^GT)²` with the level-set
area functional `A_c(φ) = ∫ H(φ_c)`. Its first variation uses `∂A/∂φ = δ(φ)` (Heaviside → Dirac):

    ∂E/∂φ_c = λ_c·max(0, A_c − A_c^GT)·δ(φ_c)  ⇒  ∂φ_c/∂t|_area = −λ_c·max(0, A_c − A_c^GT)·δ(φ_c).

This IS the correct variational (Chan-Vese) area-constraint gradient flow: boundary-localized (the
`δ(φ)` support is the zero level set), inward (negative where `A_c > A_c^GT`), magnitude linear in the
overshoot. **The trainer's discrete twin is exact:** `L_area = Σ (λ_c/2)·relu(m_c − A_c^GT)²`,
`dL/d(logits) ∝ λ_c·relu(m_c − A_c^GT)·softmax_c(1−softmax_c)` — the softmax Jacobian peaks on the
codim-1 annulus = the discrete `δ(φ)`. FORCE linear in overshoot, POTENTIAL quadratic. ✓

**Equilibrium.** Compose with the (boundary-localized) birth force:
`∂φ_c/∂t = [F_birth − λ_c·max(0,A_c−A_c^GT)]·δ(φ_c)`. Bracket = 0 ⇒

    F_birth = λ_c·(A_c* − A_c^GT)  ⇒  **A_c* = A_c^GT + F_birth/λ_c**,  and with λ_c = F_birth/(δ·A_c^GT):
    **A_c* = (1 + δ)·A_c^GT = 1.25·A_c^GT** at δ=0.25.  ✓ (matches the memo/equation exactly)

**Stability (verified, not asserted).** For `A > A*`: retraction `λ(A−A_GT) > F_birth` ⇒ net inward
⇒ A ↓ toward A*. For `A_GT < A < A*`: retraction `< F_birth` ⇒ net outward ⇒ A ↑ toward A*. For
`A < A_GT`: relu ⇒ zero retraction, birth unopposed ⇒ free nucleation. So **A\* is a STABLE attractor**
and the constraint is a PRECISION CAP, not an annihilator. Robust to a state-dependent birth force as
long as `dF_birth/dA < λ` (birth recall DECREASES as the class fills ⇒ `dF_birth/dA ≤ 0 < λ` ⇒
even more stable). **The operator's "engineered annealing = the equilibrium, no schedule" is SOUND** —
the desired area is chosen by λ, and the quadratic well restores excursions.

**Dimensional consistency.** `[λ] = force/area`; `λ·(A−A_GT)` = force/area·area = force ✓, same units as
`F_birth` (a force on the boundary φ). The ONE approximation is `F_birth ≡ W_birth = 1.0` (the birth-loss
gradient magnitude on `m_c` proxied by the birth-loss weight) — HONESTLY tiered
ASSUMED_AWAITING_VERIFICATION for the absolute scale; the balance FORM + dominance are scale-independent.

**Dominance re-check.** retraction/birth at a runaway = `(A/A_GT − 1)/δ`. Lane ep125 ratio 13.76 ⇒
`(12.76)/0.25 = 51×`; movable `(3.58)/0.25 = 14×`. ✓ Matches the operator's "at 13.8× the retraction
MUST dominate." Area returned to Road at equilibrium ≈ 0.114 ≥ the measured 0.1189 Road+Undriv deficit
⇒ the ~9% Road-pixel theft is undone, the Road d_seg floor lifts. Arithmetic reproduced. ✓

## 2. Ramp `post_level = 1 − τ_persist` re-derivation

At completion `persistence = 1 − within_flip ≥ τ_persist` ⇒ a fraction `τ_persist` of GT support is
FORMED (held above the argmax margin); only the unformed tail `1 − τ_persist` still needs birth pressure
⇒ retain exactly `post_level = 1 − τ_persist` of `F_birth`. New Lever-1 equilibrium with the ramped
force: `A* = A_GT·(1 + post_level·δ)`. At τ=0.8, δ=0.25 ⇒ `A* = 1.05·A_GT`. ✓ Tight precision band,
strictly inside the completion band `[0.75,1.25]·GT`. `ramp_epochs=50 = curriculum_min_stage/3 (150/3)`
is a labeled DERIVED-AT-CONFIG heuristic (slow enough not to trip the spike-guard jump detector, fast
enough to free capacity in-stage); the exact fraction is ASSUMED_AWAITING_VERIFICATION. Sound.

## 3. The 3 P0 forces vs their derivation (each matches; none fights the τ-anneal or area-Lagrange)

- **FORCE 1 temporal-screw** (`L_temp` on GROUND {0,1,2}, warp f0 softmax by ξ, annulus-masked) matches
  §1.2–1.3: warp = `warp_frame0_native_mlx` (bit-checked), φ(f0) from the EVEN-index raw witness render
  (`c1−1`, NOT the carrier dispatch), Movable/MyCar never warped, `ground_gt` stop-grad DEFAULT (zero
  pose coupling) + `carrier_live` grad-to-dξ with the d_pose tripwire as **telemetry** (advisory
  stage-boundary revert, never per-step). Prob-space MSE, no area/margin target ⇒ orthogonal to the
  area-Lagrange (mild synergy). ✓
- **FORCE 2 satisfice** (`relu(m_safe − m_wit)` on the annulus) matches §2.2 and **preserves the
  τ-anneal by derivation**: `tau_softplus` IS a temperature-τ margin loss, this hinge is its τ→0 hard
  limit with ceiling `m_safe`; MASK-BY-STAGE at l7 (does NOT replace CE) so early region formation
  (which the area-Lagrange/island stack needs) is intact. Fails LOUD if `m_safe < δ_R` (factory +
  trainer). The `#2 ⊗ area-Lagrange` WATCH is resolved exactly as speced (start ≥ l7). ✓
- **FORCE 3 tie-locus** wraps the built subpix term + `W_e` STAMPED from the P-A artifact (uniform
  fallback + LOUD WARN, never a hardcoded guess) + `ref_domain` sub-option (FORCE 4 folds in;
  `camera874_dphase` inert for the through-R training loss = IDENTICAL to seg384, decode-consumer
  provenance only). Default `edge_weight_source=uniform` in the trainer ⇒ EXACT pre-existing mean ⇒
  byte-identical. ✓ No antagonism with the τ-anneal (a sub-pixel placement target on straddles).

**Interaction matrix cross-check:** the memo's SYNERGY/WATCH entries hold under re-derivation; the only
WATCH (satisfice ⊗ area-Lagrange, orthogonal quantities) is sequenced-safe (l7 gate). No built term
injects a systematically wrong target: FORCE 1 masks non-ground, FORCE 2 one-sided below-ceiling only,
FORCE 3 masked to genuine-V straddles.

## 4. STRUCTURE — unified level-set flow preserved

- **The counter-force IS the missing term, not a bolt-on:** Lever-1 is literally the area-constraint
  Lagrange term of the same Chan-Vese region energy the witness already minimizes (eikonal/length are
  the sisters). It restores the variational completeness the birth stack broke (recall pressure with no
  precision term). ✓
- **Birth → boundary hand-off is coherent with the curriculum:** `area_band == tolerance == 0.25`
  (birth_completion.py L82 ↔ chan_vese L149; autoconfig ties the compose) ⇒ the Lever-1 equilibrium
  (1.25·A_GT) sits EXACTLY at the Lever-2 completion-band upper edge (inclusive: `part_frac ≤ 1.25·g`
  fires) ⇒ a class reaches the cap, then the event fires and tightens to 1.05·A_GT. Two independent
  mechanisms composing MONOTONICALLY toward precision (defense-in-depth, #302). Elegant, not
  accidental. ✓
- **Island per-class ramp-split IDENTITY (re-derived):** with masks partitioning the island-weight
  support, `f_a·mean_a + f_b·mean_b = Σ(birth·w_a)/S + Σ(birth·w_b)/S = Σ(birth·weight)/S` = the single
  combined term when `mult=1` (EXACT). Per-class hand-off is therefore independent; the OFF/pre-fire
  path keeps the single combined term (gated on `amp_active`), so no byte-identity claim rides the
  fp32-summation-grouping ULPs. ✓ (precondition: the split masks must partition the ladder-grown
  island support — the memo/tests assert `|diff|=0.0`; rebuilt in lockstep with the ladder radii.)
- **Byte-identity when OFF:** every new term gated (`_area_lambda is None` / `ms_w=0` / `ts_w=0` /
  controller None / `edge_weight_source=uniform`); `_island_levers_on` correctly extended with
  `_area_lambda is not None` (so area-alone still computes the witness-alone forward, no silent skip).
  Confirmed by 128 green delta tests.

## 5. Value-provenance ladder (every new constant)

| constant | provenance | verdict |
|---|---|---|
| δ_R = 0.0196 | **MEASURED** (p95 uint8-at-camera margin perturb, `reports/delta_R_noise_floor.json`) | ✓ |
| λ_c | **DERIVED-LIVE** in trainer from GT areas (gold standard) | ✓ (scale owed, §6) |
| tolerance δ=0.25 | DERIVED design choice (loose-enough/tight-enough, docstring) | ✓ labeled |
| birth_force=1.0 | MEASURED-ANCHOR config-conditional (= amplify/recall weight; re-derive trigger) | ✓ |
| post_level=1−τ | DERIVED-AT-CONFIG (unformed-fraction argument) | ✓ (scale owed, §6) |
| ramp_epochs=50 | DERIVED-AT-CONFIG (min_stage/3) | ✓ (fraction owed) |
| τ_persist=0.8, area_band=0.25 | DERIVED (nucleus within_flip family; ties Chan-Vese tolerance) | ✓ labeled |
| m_safe=0.06=3·δ_R | DERIVED from δ_R (headroom 2) | ✓ |
| W_e matrix | **STAMPED** from P-A artifact (fail-loud/WARN, never hardcoded) | ✓ |
| w_t/w_s/w_tie=0.1/0.2/0.3 | cold-start activation weights (default-OFF, not composed; ramp owed) | ✓ |

**No bare literals.** Every constant carries a provenance tier.

## 6. Pre-registered-owed A/B scales (DESIGN-INTENT, NOT blocking findings — cannot be measured pre-launch)

1. **Chan-Vese λ absolute SCALE** — ASSUMED_AWAITING_VERIFICATION (equation anchor
   `chan_vese_area_constraint_lambda_scale_owed_v75_ab`). The FORM + balance + dominance are derived and
   scale-robust (any λ in a wide band caps the runaway); the A/B tunes the exact scale. Owed to the v7.5
   arm.
2. **Ramp `post_level` + `ramp_epochs` fraction** — ASSUMED_AWAITING_VERIFICATION (the unformed-fraction
   FORM is derived; the best absolute post_level/length is owed to the A/B). Sister of (1).

Both are explicitly labeled in code + memo + equation and pre-registered as owed measurements. Per the
lens instruction, these are DESIGN-INTENT owed to the run, NOT blocking.

## 7. Triality legs (all present + consistent)

- **DSL:** 5 factories present (`AreaConstraintBirth`, `BirthCompletionEvent` + P0
  `TemporalScrewConsistency`/`MarginBandSatisficing`/`TieLocusDisplacement`), all validated + fail-closed;
  P0 all DEFAULT-OFF and **NOT composed** in `_CRUCIBLE_V7_DSL_LEVERS` (only the 2 counter-force levers
  are composed) — scope discipline (activate one per increment) honored. ✓
- **DAG:** counter-force (FEED-roadfloorfix, 30 mentions) + P0 forces (present) appended. ✓
- **equations:** Chan-Vese `chan_vese_area_constraint_birth_balance_v1` **REGISTERED** (registry +1);
  P0 laws **FORMALIZATION_PENDING** (council-flagged, 0 registered — correct: designed-not-measured,
  register when the per-force A/B anchors land). ✓

## 8. LOW observation (non-blocking, doc-only)

**λ derived from GLOBAL area, penalty target is PER-PAIR area.** Trainer setup (~L4318) derives λ_c from
`_area_gt_global` (bincount over the whole L* stack), while the per-pair penalty (~L4884) uses the
per-pair `A_GT_c = mean(lstar_oh_c)`. So the per-pair equilibrium ratio is `1 + δ·(A_global/A_pair)`,
exactly `1+δ` only at the average pair. This is a **SOUND (indeed safer)** design — a global-stiffness λ
avoids `λ→∞` on a pair where the class is absent (then the per-pair target 0 correctly penalizes ANY
mass with a finite λ) — but the equation's `domain_of_validity` NOTE ("A_GT_c LIVE per-pair") reads as
implying a per-pair λ. Suggest a one-line clarification (λ = global stiffness, A_GT target = per-pair).
Moot for the launch since the λ scale is already owed to the A/B; **not a bug, not blocking.**

## Gates

- ruff F clean on all 7 touched Python surfaces.
- 128/128 delta tests green (`test_v75_birth_counterforce` 39 · `test_v75_birth_ramp_application` 19 ·
  `test_p0_forces_phase2_build` 19 · `test_crucible_v7_config` + neighbours).
- OUT OF SCOPE (pre-existing, tracked, not reported): #332 123-unmapped gap · #185 drift alarm.

**Pointer 0.19110 UNMOVED — MEANS. The END is a byte-closed `upstream/evaluate.py` n600 row < 0.19110
after the operator-GO v7.5 launch; the λ scale + ramp scales are the pre-registered owed A/B rows.**
