# R2 Council Ledger — Wave A 5 council-PROCEED commits, FRESH EYES

**Lane**: `lane_recursive_review_r2_wave_a_council_proceed_20260514` (L1)
**Subagent**: RECURSIVE-REVIEW-R2-WAVE-A-FRESH-EYES-SUBAGENT
**Date**: 2026-05-14
**Cycle**: R2 of 3 consecutive clean passes per CLAUDE.md "Recursive adversarial review protocol"
**Verdict**: **FINDINGS** — counter RESETS to 0/3 (NEW R2 findings + R1 META-1/META-3 partially still open)

## Scope

Same 5 commits as R1, with FRESH-EYES rotation per CLAUDE.md "Council conduct" + R1 CONTRARIAN-2 explicit challenge:

- `951858245` D9 per-class provider routing (omnibus D9 PROCEED B 11/11)
- `e84accd7c` + `8202dc0aa` Catalog #233 L1→L2 4-gate (D8 PROCEED D 11/11)
- `e54901d60` Z3 v2 latent-replacement (D3 PROCEED B 11/11)
- `dd17e6e2e` Tier C PR106 + DP1 extension (D7 PROCEED A→C 11/11)
- `0916332eb` C1 Z5 routing + F3 vq_vae + PDP (D13 PROCEED C 11/11) — body now backfilled via git notes per FIX-WAVE-R1

R2 council rotation per CLAUDE.md non-negotiable + R1 CONTRARIAN-2 dissent doctrine:

- **Selfcomp** (szabolcs-cs / PR #56 lead): working-implementation realism
- **MacKay** (memorial seat): MDL + arithmetic-coding + density-network framework
- **Hassabis** (DeepMind cross-domain): strategic breadth-vs-depth tradeoff
- **Boyd** (Stephen Boyd, convex optimization at operational level): ADMM/projection feasibility
- **Tao** (Terence Tao, pure-math): mathematical completeness + partition-set rigor

R3 will rotate to: Time-Traveler / Schmidhuber / Hinton / Karpathy / Carmack.

## FIX-WAVE-R1 status check

R1 produced 12 findings; FIX-WAVE-R1 has landed partial fixes:

| Finding | Severity | Pre-R1 state | Post-FIX-WAVE state | R2 verdict |
|---|---|---|---|---|
| META-1 | CRIT | 5 false catalog claims | 3 remain (#130, #158, #162) | **PARTIAL** |
| META-2 | CRIT | preflight_all() failing | wrapper-stage fixes landed | **CLOSED** |
| META-3 | CRIT | 0916332eb empty body | git notes backfill landed | **PARTIAL** (Catalog #234 strict-flip pending) |
| META-4 | CRIT | Catalog #117 prefix bug | 50→14 violations after fix | **CLOSED** |
| YOUSFI-1 | MED | Catalog #124 STRICT trip | lane_class added | **CLOSED** |
| QUANTIZR-1 | MED | Z3 v2 byte regression | assertion added (3 LOC test in 904470059) | **CLOSED** |
| CONTRARIAN-1 | MED | Catalog #171 latent | recipe updated | **CLOSED** |
| HOTZ-1 | LOW | Z3 v2 device duplication | inflate_v2.py refactored (20d9939eb) | **CLOSED** |
| HOTZ-2 | LOW | vq_vae waivers removed | argparse declarations confirmed | **CLOSED** (defense-in-depth not added but harmless) |
| QUANTIZR-2 | LOW | Catalog #228 text outdated | partial — text still says "live count at landing: 2" | **PARTIAL** |
| FRIDRICH-2 | LOW | PDP help text outdated | help text updated in 904470059 | **CLOSED** |
| CONTRARIAN-2 | MED | unanimous consensus | omnibus shows 13/15 unanimous | **STANDS** (R2 surfaces additional dissent) |

**Net FIX-WAVE-R1**: 6 of 12 findings fully closed, 4 partial, 2 stand. NEW Catalog gates #234 + #235 landed (in working tree per sister subagent). NEW R2 findings below.

## Per-commit verdict table — R2

| Commit | Lane / Subject | R2 verdict | New findings |
|---|---|---|---|
| `951858245` | D9 per-class provider routing | **FINDINGS** | BOYD-1 CRIT (fallback semantic overload); BOYD-2 LOW (transition-zone boundary asymmetry) |
| `e84accd7c` + `8202dc0aa` | Catalog #233 L1→L2 4-gate | **FINDINGS** | TAO-1 CRIT (token-substring 100ep gate false-positive class) |
| `e54901d60` | Z3 v2 latent-replacement | **FINDINGS** | SELFCOMP-1 MED (v1 still exported despite "retired" claim) |
| `dd17e6e2e` | Tier C PR106 + DP1 extension | **FINDINGS** | MACKAY-1 MED (per-tensor std assumes Gaussian); SELFCOMP-3 MED (per-pair renderer reconstruction wasteful) |
| `0916332eb` | C1 Z5 + F3 vq_vae + PDP | **FINDINGS** | SELFCOMP-2 MED (vq_vae lacks cached-vs-uncached equivalence test) |
| (cross-cutting) | Wave A as a 9-min landing window | **STRATEGIC FINDINGS** | HASSABIS-1 CRIT (8463 LOC infrastructure in 9 min, 0 contest-CUDA anchors, race-mode active) |

## CRITICAL findings

### TAO-1 (CRITICAL) — Catalog #233 4-gate token-substring false-positive class

**Voice**: Tao + MacKay unanimous CRITICAL.

`check_l1_to_l2_promotion_canonical_4_gate` (`src/tac/preflight.py:51789`) evaluates the 4 gates via case-insensitive substring membership against bag-of-tokens token sets. The bug class:

```python
# _CHECK_233_AUTH_EVAL_100EP_TOKENS contains "100ep"
# A lane evidence body that says "training duration was 100ep timeline; ..."
# trips gate 3 even when no 100ep auth-eval anchor exists.
```

Empirical proof:

```python
text = 'See discussion of 100ep vs 200ep tradeoffs in the smoke green report; '
       'tier c was measured; validate_custody verified; evidence_grade=contest_cuda'
g1, g2, g3, g4 = _check_233_evaluate_4_gates(text)
# Returns: (True, True, True, True) — gate PASSES on prose-only text!
```

**Tao's lens**: the 4-gate canonical is *meant* to be a partition of the L2-eligibility decision space. But token-substring matching makes the "smoke green" set and the "100ep auth-eval" set OVERLAP via prose contamination. Sister of Catalog #136 (`check_custody_gate_accept_tokens_concrete_only`) which already extincted this exact bug class for ANOTHER set of validator tokens — but #233's tokens were added without inheriting #136's discipline. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": this is the SAME bug class spreading 6-7× across the repo (META-meta finding from `a8bc7e79` proactive sweep).

**MacKay's lens**: from an MDL perspective, the 4 gates should each have INDEPENDENT description-length contributions. Token-substring overlap means the gates have correlated false-positive surface — the effective independent gates is < 4. This is the same MDL antipattern flagged in Catalog #143 / #144 / #145 docstrings.

**Fix**: refactor `_check_233_evaluate_4_gates` to require structured-field declarations (e.g., `gate_evidence:{auth_eval_anchor:{archive_sha256:..., score:..., axis:...}}`) instead of substring matching. Pattern to follow: Catalog #136's `_CUSTODY_VALIDATOR_TOKENS` filtered through AST-aware structural validation.

Live impact: 0 false-positive lanes today (no L2 lane has prose-only 100ep mention satisfying all 4 gates). But the bug class IS LIVE — any future lane registry edit by an in-flight subagent could trip the false-positive without #233 catching it.

**Status**: NEW finding from R2 — must be addressed before R3 can fire.

### BOYD-1 (CRITICAL) — D9 fallback semantic overload (cheaper-vs-capacity-overflow)

**Voice**: Boyd + Tao CRITICAL.

`tac.cost_band_calibration.FALLBACK_PROVIDERS_PER_CLASS` is OVERLOADED with two conflicting semantics:

1. **Time-Traveler amendment semantic**: fallback fires when "fallback is >25% CHEAPER than canonical" (`select_provider_for_class:294-313`).
2. **Capacity-overflow semantic** per the canonical doc `.omx/research/per_class_provider_routing_canonical_20260514.md` Section 3: `long_burn` fallback is `vastai/H100` with trigger condition "race-mode urgency (Lightning A100 saturated)".

Vast.ai H100 ($1.50–1.99/hr) is **MORE EXPENSIVE** than Lightning A100 (operator subscription, $0/hr). The Time-Traveler amendment will NEVER fire for `long_burn` because the cost-shift inequality fails. The capacity-overflow semantic is the ACTUAL intent for `long_burn`, but the routing helper has no code path for it.

**Boyd's lens**: this is a feasibility-region misspecification. The Time-Traveler dynamic re-routing forms a convex projection onto the cheaper-than-canonical feasible set, but `long_burn`'s fallback is OUTSIDE that set by construction. Either the fallback definition is wrong, OR the trigger logic is wrong, but they are mutually inconsistent.

**Tao's lens**: the data structure `dict[str, list[tuple[str, str]]]` carries no semantic discriminator. A future operator adding a new dispatch class will have no signal whether their fallback should be cheaper-alternative or capacity-overflow.

**Fix options**:
- (A) Split `FALLBACK_PROVIDERS_PER_CLASS` into `FALLBACK_CHEAPER_ALTERNATIVES_PER_CLASS` (Time-Traveler trigger) + `FALLBACK_CAPACITY_OVERFLOW_PER_CLASS` (manual operator escalation).
- (B) Tag each fallback tuple with a `FallbackReason` enum (`CHEAPER_ALTERNATIVE` | `CAPACITY_OVERFLOW`).
- (C) Document that `long_burn`'s H100 is currently DEAD (Time-Traveler will never fire) and accept the semantic asymmetry.

Council-grade decision per CLAUDE.md "Design decisions — non-negotiable" — this needs Decision 9 reconvening.

**Status**: NEW finding from R2 — must be addressed before R3 can fire.

### HASSABIS-1 (CRITICAL) — Wave A added 8,463 LOC in 9 min during ACTIVE race mode, 0 contest-CUDA anchors

**Voice**: Hassabis + Carmack (consultative) CRITICAL.

Empirical receipts:
- Wave A = 5 commits × 9-minute window (15:53:02 → 16:02:03 UTC)
- LOC change: 951858245 (1693+) + e84accd7c (1347+) + e54901d60 (1655+/61-) + dd17e6e2e (1743+/3-) + 0916332eb (2024+/33-) = **8,463 LOC added across 5 commits**
- Cost-band posterior anchors logged in same window: **11 successful_dispatch (all `provider=unknown`, all pre-existing from earlier dates)**
- New contest-CUDA empirical anchors landed in this window: **0**
- `.omx/state/RACE_MODE_ACTIVE.flag` exists (race mode IS active)

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable HIGHEST EMPHASIS:
> "If `.omx/state/RACE_MODE_ACTIVE.flag` exists, campaign actuation outranks new grand-council text unless that text directly writes launchable commands, hardens the actuator, or records the blocker preventing spend."

Wave A is exactly the failure pattern the rule warns against — building infrastructure (per-class routing API, 4-gate canonical, latent-replacement grammar, MDL extension, autopilot halve) instead of launching dispatches to harvest empirical anchors. Per the May 4 race postmortem (CLAUDE.md "May 4 2026 contest decided in 4 hour 8 minute race window... Silver medal was 241 lines of code in 2 files... PR #107 apogee landed at 0.229 — we had every primitive needed but spent the race window building meta-Lagrangian + predictor + sanity gates"), this is the same architectural-breadth-vs-empirical-depth misallocation.

**Hassabis's lens**: at AlphaFold scale, infrastructure-first work was tolerated *because* the dispatch surface was orders of magnitude more expensive than the engineering surface. Here, dispatches are $0.30 (smoke) to $5-15 (full). The cost-of-infrastructure is now COMPARABLE to the cost-of-dispatch. Strategically: 13 PROCEED unanimous decisions in 9 min consume ~$0 of GPU but ~8000 LOC of engineering attention — the same ~$15 budget as one full-class dispatch. Was this the right tradeoff?

**Carmack's lens** (consulted): "5-line diffs that get a real number landed > 8000-line diffs that prepare for a number." Sister of CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" applied at the strategic level: 5 council decisions all marked PROCEED with NO empirical anchor in same window is a verifiable-claim-without-evidence pattern at the meta level.

**Fix**: this is NOT a bug in the commits — each commit is independently CLEAN. This is a STRATEGIC finding about the 5-commit landing CADENCE. The recommendation is for the operator to pivot from infrastructure-build mode to dispatch mode IMMEDIATELY (per the race-mode non-negotiable) and run Z5/C1 contest-scale dispatches before the next council omnibus fires.

**Status**: STRATEGIC finding — operator-routable, not blocking R3.

## MEDIUM findings

### SELFCOMP-1 (MEDIUM) — Z3 v1 still exported despite "v1 retired as redundant" claim

**Voice**: Selfcomp.

Commit `e54901d60` message states "v1 retired as redundant per same verdict". But the v1 surfaces remain LIVE:

- `src/tac/substrates/z3_balle_hyperprior_bolton/archive.py` (16.2KB) — exported via `__init__.py` (lines 32-46)
- `src/tac/substrates/z3_balle_hyperprior_bolton/inflate.py` (5.3KB)
- `src/tac/substrates/z3_balle_hyperprior_bolton/score_aware_loss.py` (8.1KB)

The v1 path is the DEFAULT in `experiments/train_substrate_z3_balle_hyperprior_bolton.py` (line 119); v2 is gated behind `--enable-v2-latent-replacement`. So v1 is in fact NOT retired — it is the production default. This is documentation-vs-reality drift in the commit message, NOT a code bug.

**Selfcomp's lens**: when both v1 and v2 are exported, future trainer authors will import v1 by accident. Per CLAUDE.md "Beauty, simplicity": "delete dead fields, stale adapters, and duplicate one-offs once a canonical contract replaces them." Either:
- (A) Update commit message language to "v1 PRESERVED as reference; v2 is the score-affecting path",
- (B) Actually retire v1 by removing exports (and update the trainer to v2-only), OR
- (C) Add a deprecation warning to v1's `__init__.py` exports.

**Fix**: documentation correction in a follow-up memo + sister-subagent commit message style audit.

### SELFCOMP-2 (MEDIUM) — vq_vae lacks cached-vs-uncached equivalence test

**Voice**: Selfcomp + Yousfi (consultative).

The PDP F3 wire-in has a sister test `test_pdp_loss_cache_path_equivalent_to_gt_forward` in `src/tac/substrates/pretrained_driving_prior/tests/test_score_aware_loss_f3_kwargs.py` that asserts the cached path produces byte-equivalent (within tolerance) loss values to the uncached GT-forward path. This catches the cache-key bug class explicitly raised in the council Decision 2 deliberation: "cache HIT with wrong key → silently wrong score".

The vq_vae F3 wire-in tests (`src/tac/tests/test_f3_backport_vqvae_pdp_wired.py`) are TEXT-MATCHING ONLY:
- `test_vq_vae_calls_gt_cache_lookup_in_hot_loop` — checks `gt_cache.lookup` appears ≥2 times in trainer file text
- `test_vq_vae_threads_f3_kwargs_into_loss_fn` — checks kwargs are passed
- `test_vq_vae_uses_canonical_opt_ctx_pattern` — checks canonical pattern token

There is NO equivalent of `test_vq_vae_loss_cache_path_equivalent_to_gt_forward`. The cache-key bug class is unguarded for vq_vae.

**Selfcomp's lens**: this is the same regression class that PR #56's SegMap codec had — a "sister surface" guarantee broken because the test only exists on ONE of two surfaces. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": tests for the cache equivalence should be sister-pattern across BOTH surfaces.

**Fix**: add `test_vq_vae_loss_cache_path_equivalent_to_gt_forward` (~30 LOC; mirror the PDP test).

### MACKAY-1 (MEDIUM) — Tier C per-tensor std noise scaling assumes Gaussian distribution

**Voice**: MacKay + Boyd (consultative).

`tools/mdl_scorer_conditional_ablation.py::_run_tier_c_a1` (and sister `_run_tier_c_pr106`, `_run_tier_c_ibps1`, `_run_tier_c_dp1`) scales the noise as:

```python
rel_std = v_dev.std().clamp(min=1e-8)
noise = torch.randn_like(v_dev) * (rel_std * sigma)
```

This assumes the per-tensor weight distribution is approximately Gaussian. For TYPICAL trained NN weights with L2 regularization this is mostly true. But for substrates with sparse or long-tail weight distributions (DP1's coordinate-MLP at hidden=64 has very different distribution shape than A1's HNeRV decoder at 162KB), the per-tensor std UNDERESTIMATES the perturbation needed to reach a fixed fractional information loss.

**MacKay's lens**: from an MDL perspective, the canonical noise scale for fixed information loss is `sigma * sqrt(Var(W))` only when W is Gaussian. For long-tail W, the scale should be `sigma * MAD(W) * 1.4826` (median absolute deviation × Gaussian-equivalent constant) OR `sigma * IQR(W) / 1.349`. Cross-substrate Tier C density comparison is unreliable when the underlying weight distribution shape varies.

**Boyd's lens**: this is not a correctness bug — `Δscore(σ)` is well-defined for any noise scaling — but the SEMANTIC of "within-class density 0.95 vs 0.45" depends on the noise-scaling normalization. PR106's Tier C density is not directly comparable to DP1's Tier C density without controlling for distribution shape.

**Fix**: add a `noise_scaling_method` parameter to `_run_tier_c_*` (default `"gaussian_std"` for back-compat; alternatives `"mad_robust"`, `"iqr_robust"`). Document the cross-substrate comparison caveat in Tier C result schemas.

### SELFCOMP-3 (MEDIUM) — DP1 Tier C reconstructs renderer from scratch per pair (~2400× constructor calls)

**Voice**: Selfcomp.

`_run_tier_c_dp1` calls `DrivingPriorRenderer(cfg).to(device).eval()` inside `_render_pair_with_residual`, which is invoked ONCE PER PAIR per sigma per perturbation target. For 600 pairs × 4 sigmas × 2 targets (state_dict + latents) = **4800 renderer constructions per Tier C run**.

Empirical: the commit message reports 1.0s wall-clock for the synthetic-realistic test on macOS CPU. So per-construction cost is ~0.2ms — fast enough to not be a blocker. But this is a 4800× regression vs the canonical pattern (cache the renderer, swap state_dict in place).

**Selfcomp's lens**: this is engineering pragma, not a correctness bug. The pattern works but is wasteful. Future Modal/Vast.ai dispatches running this on a 600-frame archive at sigma_count=10 will see 12,000 constructor calls per Tier C run. At ~10ms per construct on a real GPU, that's 2 minutes of wasted overhead per Tier C invocation.

**Fix**: refactor `_render_pair_with_residual` to construct the renderer ONCE per `_run_tier_c_dp1` call, and call `renderer.load_state_dict(perturbed_sd, strict=False)` per pair. Sister pattern: `_run_tier_c_a1` already does this (single `decoder` instance constructed once, state_dict swapped per perturbation).

## LOW findings

### BOYD-2 (LOW) — D9 transition-zone boundary asymmetry

**Voice**: Boyd.

`classify_dispatch` (`tac.cost_band_calibration:962-1008`) uses `>=` for long_burn boundary and `<=` for smoke boundary:

```python
if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
    return "long_burn"   # 12.0h becomes long_burn (Lightning A100, $50+)
if estimated_wall_clock_sec <= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0:
    return "smoke"        # 0.5h becomes smoke
```

A borderline 12.0h dispatch routes to `long_burn` (Lightning A100, $50+ class) instead of `full` (Vast.ai 4090, $2-15 class). This is a 5-10× cost jump for a 1-second wall-clock difference at the boundary.

**Boyd's lens**: convex feasibility regions should have OPEN boundaries (strict `<` / `>`) when crossing them changes the cost class non-trivially. The existing CLOSED boundary (`>=`) is consistent with the dispatch class definitions in the canonical doc but creates a discontinuity in cost expectation at the boundary.

**Fix**: change `>=` to `>` for the long_burn check (or document that the boundary is canonical and operators should round-down for borderline cases).

### MACKAY-2 (LOW) — Catalog #233 token sets do not use frozenset

**Voice**: MacKay (style + immutability).

`_CHECK_233_SMOKE_GREEN_TOKENS`, `_CHECK_233_TIER_C_TOKENS`, `_CHECK_233_AUTH_EVAL_100EP_TOKENS`, `_CHECK_233_CUSTODY_VALIDATOR_TOKENS` are declared as `tuple[str, ...]` rather than `frozenset[str]`. This makes membership checks O(N) instead of O(1) and allows the gate's evidence sets to be mutated in tests. Sister gates (e.g., `_CHECK_233_WAIVER_PLACEHOLDER_REJECTS`) correctly use `frozenset`.

**Fix**: convert the 4 token sets to `frozenset[str]`. Per CLAUDE.md "Beauty, simplicity": consistent use of immutable data structures.

## Council adversarial cross-debate

**Selfcomp**: "v1 vs v2 dual-export and absent vq_vae cache-equivalence test are the implementation realism findings. The Tier C renderer reconstruction per-pair is wasteful but works. The Z3 v2 byte savings claim of 4842 B is empirically supported and now has a regression test (904470059)."

**MacKay**: "Tier C's per-tensor std noise scaling is the MDL-rigor concern. The 4-gate canonical's token-substring matching has the same overlap problem flagged in Catalog #136 and Catalog #143-#145 docstrings — sister-pattern bug class spreading. The fix is structural validation, not more tokens."

**Hassabis**: "Strategically, the most concerning finding is the cadence: 5 PROCEED unanimous decisions and 8,463 LOC in 9 minutes during active race mode with 0 contest-CUDA anchors landed in the same window. This is the May 4 race postmortem failure mode repeating. Operator should pivot from council mode to dispatch mode IMMEDIATELY."

**Boyd**: "Two structural findings on D9: (1) `FALLBACK_PROVIDERS_PER_CLASS` is semantically overloaded — Time-Traveler amendment's cheaper-alternative trigger conflicts with `long_burn`'s capacity-overflow fallback. (2) The classify_dispatch boundary at 12h uses `>=` which produces a discontinuous cost jump at the boundary. Both need council clarification, not code-only fixes."

**Tao**: "The 4-gate canonical (Catalog #233) is mathematically INCOMPLETE because the gate token sets have OVERLAP via prose contamination — a lane mentioning '100ep timeline' in any text trips gate 3 false-positive. The 'canonical' is supposed to partition the L2-eligibility space; it does not. Sister bug class to Catalog #136. Per CLAUDE.md 'META-meta finding: bug classes have 6-7× spread' — this is the canonical spread surface."

**CONTRARIAN-2 R2 carry-forward**: R1's CONTRARIAN-2 challenged "rapid unanimous consensus on infrastructure decisions." R2 confirmed: omnibus shows 13 of 15 decisions are 11/11 unanimous, with only D1 (10/11 with Contrarian addressed-then-joined) and D11 (10/11) showing visible dissent. The C1-reconvene Decision 6 was 4/11 explicit (NOT unanimous) and the implementation correctly cited it as "HALF-MEASURE" — partial mitigation. But the omnibus pattern stands. R2 surfaces 8 NEW findings across the 5 unanimously-PROCEED commits.

## R1 perspective blind spots (R2 critique of R1's rotation)

R1 used Yousfi/Fridrich/Hotz/Quantizr/Contrarian rotation. What did R1 MISS?

1. **Boyd's optimization-feasibility lens** — R1 had no convex-optimization voice. R1 missed the D9 fallback semantic overload (BOYD-1) because Yousfi/Fridrich/Hotz/Quantizr/Contrarian don't have ADMM/Dykstra-feasibility instinct.

2. **Tao's mathematical-completeness lens** — R1 had no pure-math voice. R1 missed the Catalog #233 token-substring false-positive class (TAO-1) because none of Yousfi/Fridrich/Hotz/Quantizr/Contrarian probe partition-set rigor as a default.

3. **Hassabis's strategic-research lens** — R1 had no cross-domain strategic voice. R1 missed the 8463-LOC-in-9-min-during-race-mode pattern (HASSABIS-1) because Quantizr-the-Contrarian focused on per-commit content, not cadence.

4. **MacKay's MDL-rigor lens** — R1 had no information-theoretic voice on the Tier C noise scaling. R1 didn't probe the Gaussian assumption (MACKAY-1).

5. **Selfcomp's lived-implementation lens** — R1's Hotz-1 (Z3 inflate device duplication) overlapped with Selfcomp's territory but missed v1-vs-v2 export confusion (SELFCOMP-1) and vq_vae cache-equivalence test gap (SELFCOMP-2). Hotz looks at engineering surface; Selfcomp looks at "what would I actually run into when implementing this?"

The R2 rotation produced 8 NEW findings in 5 commits R1 already reviewed. Per CLAUDE.md "Council conduct" + R1 CONTRARIAN-2: "rapid unanimous consensus on infrastructure decisions is itself a finding." R2 confirms the perspective-rotation discipline produces material new findings even on R1-CLEAN-CODE commits. R3 should expect similar yield.

## Recommended FIX-WAVE-R2 actions

Per CLAUDE.md "Recursive adversarial review protocol" + "Bugs must be permanently fixed AND self-protected against":

1. **FIX-CRIT-R2-1** (TAO-1): refactor `_check_233_evaluate_4_gates` to require structured-field declarations OR sister gate `check_catalog_233_4_gate_no_prose_token_overlap` that flags any `_CHECK_233_*_TOKENS` entry < 8 chars OR substring-of-another-gate's-token. Live-repo check that no current L2 lane satisfies all 4 gates via prose-only matches.

2. **FIX-CRIT-R2-2** (BOYD-1): council-reconvening on D9 fallback semantic. Recommended Option (B) — tag each fallback tuple with `FallbackReason` enum. Add tests that assert Time-Traveler amendment never fires when fallback is more expensive. Update canonical doc to remove the semantic conflict.

3. **FIX-STRATEGIC-R2-3** (HASSABIS-1): operator pivot to dispatch mode. Per CLAUDE.md "Race-mode rigor inversion": fan out Z5 contest-scale Modal A100 + at least 2 sister substrate full-runs in the next 2-hour window. Defer further council omnibus until ≥3 contest-CUDA anchors land.

4. **FIX-MED-R2-4** (SELFCOMP-2): add `test_vq_vae_loss_cache_path_equivalent_to_gt_forward` (~30 LOC) mirroring PDP's sister test. Sister-pattern protection across F3 surfaces.

5. **FIX-MED-R2-5** (SELFCOMP-1): commit-message correction memo stating Z3 v1 is PRESERVED-as-reference, not RETIRED. Optional: add deprecation warning to v1 `__init__.py` exports.

6. **FIX-MED-R2-6** (SELFCOMP-3): refactor `_render_pair_with_residual` to construct renderer once + swap state_dict per pair. ~10 LOC delta in `tools/mdl_scorer_conditional_ablation.py`.

7. **FIX-MED-R2-7** (MACKAY-1): add `noise_scaling_method` parameter to `_run_tier_c_*` with default `"gaussian_std"` for back-compat; document cross-substrate comparison caveat.

8. **FIX-LOW-R2-8** (BOYD-2): change `classify_dispatch` boundary `>=` to `>` for long_burn (or document the closed-boundary canonical decision). **CLOSED 2026-05-15** by R2-LOW-FIX-WAVE subagent (`lane_r2_low_fix_wave_boyd2_mackay2_20260515`): changed `>=` to `>` on lines 1208 (wallclock) + 1220 (cost) for the long_burn upgrade boundary; smoke `<=` downgrade boundaries kept (route to cheaper class). Self-protection landed via Catalog #239 STRICT preflight gate `check_classify_dispatch_no_raw_ge_boundaries` (refuses raw `>=` against UPGRADE-class ceilings inside `classify_dispatch`; same-line `# CLASSIFY_DISPATCH_GE_BOUNDARY_OK:<rationale>` waiver; placeholder rejected). 27 dedicated tests + 37/37 D9 regression tests pass. Live count at landing: 0.

9. **FIX-LOW-R2-9** (MACKAY-2): convert Catalog #233 token tuples to `frozenset[str]` for O(1) membership. **AUTO-CLOSED 2026-05-15** — premise verification by R2-LOW-FIX-WAVE subagent confirmed sister commit `3882468ef` ("preflight: harden l1 promotion evidence matching", landed earlier) had ALREADY converted all 4 cited token sets (`_CHECK_233_SMOKE_GREEN_TOKENS`, `_CHECK_233_TIER_C_TOKENS`, `_CHECK_233_AUTH_EVAL_100EP_TOKENS`, `_CHECK_233_CUSTODY_VALIDATOR_TOKENS`) from `tuple[str, ...]` to `frozenset[str]` as part of the L1->L2 promotion evidence hardening. No new self-protection gate needed because frozenset hygiene is mathematical/idiomatic discipline rather than a recurring bug class. The R2-LOW subagent skipped re-fix per CLAUDE.md "Premise verification before edit" pattern (`feedback_prompt_premise_verification_before_edit_pattern_20260514.md`).

10. **R1 META-1 / META-3 carry-forward**: Catalog #185 still reports 3 violations (#130, #158, #162). Catalog #234 (commit-body-not-empty) is in working tree; sister subagent needs to land + strict-flip atomically.

## R2 verdict

**FINDINGS — counter RESETS to 0/3.**

R3 trigger conditions: FIX-WAVE-R2 lands AND (Catalog #233 token-substring class extincted OR explicitly council-deferred) AND (D9 fallback semantic clarified) AND (CRITICAL operator pivot to dispatch mode acknowledged) AND R1 META-1 (#130/#158/#162) closed.

R3 council rotation per CLAUDE.md non-negotiable: **Time-Traveler / Schmidhuber / Hinton / Karpathy / Carmack** (or equivalent fresh-eyes set).

Per R12-D meta-finding (lens-coverage expansion outpacing Zipf-decay): if R3 also finds CRITICAL-class issues, consider operator-declared SEAL (D-1 protocol) under tight cool-down constraints. But per Hassabis-1 strategic finding, the higher-priority action is to PIVOT TO DISPATCH MODE, not to keep cycling council reviews on infrastructure deltas.

## Provenance

- Lane: `lane_recursive_review_r2_wave_a_council_proceed_20260514` (L1)
- Findings JSONL: `.omx/research/recursive_review_findings.jsonl` (R2 entries appended)
- Memory: `feedback_recursive_review_r2_wave_a_landed_20260514.md`
- Reproducers ran: `pytest src/tac/tests/test_d9_per_class_provider_routing.py src/tac/tests/test_d9_operator_authorize_routing_integration.py` (50 + 13 pass), `pytest src/tac/tests/test_check_233_l1_to_l2_promotion_canonical.py` (60 pass), `pytest src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py` (32 pass), `pytest src/tac/tests/test_mdl_ablation_tier_c_pr106.py src/tac/tests/test_mdl_ablation_tier_c_dp1.py` (41 pass), `pytest src/tac/substrates/c1_world_model_foveation/tests/test_c1_z5_routing_and_autopilot_halve.py src/tac/tests/test_f3_backport_vqvae_pdp_wired.py src/tac/substrates/pretrained_driving_prior/tests/test_score_aware_loss_f3_kwargs.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_z5_routed_latent_predictor.py` (71 pass).
- Strict gates run live: Catalog #117 (post-fix 14 viols, all true positives), #119 (21 viols — fix-wave-R1 commits themselves missing Co-Author trailer), #124 (0), #127 (0), #130 (2 — pre-existing carry-forward from R1 META-1), #158 (8 — pre-existing), #162 (1 — pre-existing), #185 (3 — META-1 partial), #220 (0), #227 (0), #233 (6 — known warn-only).
- Total dedicated tests passing for the 5 commits: **267 / 267**.

---

**Per CLAUDE.md "Council conduct"**: rapid unanimous consensus is itself a finding. R2 dissent is registered through 8 NEW findings in 5 R1-CLEAN-CODE commits. R3 must continue the rotation discipline. Operator: per HASSABIS-1, the NEXT action should be a dispatch-mode pivot, NOT another council omnibus.
