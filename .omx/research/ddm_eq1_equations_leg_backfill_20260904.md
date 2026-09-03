# ddm_eq1 — the equations leg, backfilled: 29 memos, two new laws, and 55% of the backlog was the gate

`arm: ddm_eq1` · `charter: .omx/research/charters/ddm_eq1_equations_leg_backfill_20260904.md` (14d90d3ad)
`utc: 2026-09-04` · `axis: [apparatus; no scorer, no Metal, no Modal]` · `score_claim: false` ·
`promotion_eligible: false` · `cost: $0` · `tokens: [no-triality] [p0-ledger-ok]`

## The finding, first

**The "week of arm output that drifted past the equations leg" was 55.2% instrument.**

MEASURED: of the 29 memos Catalog #344 reported live at commit `d3212bed1`, **16 (55.2%) tripped it
ONLY because `"ratified"` is a substring of `"stratified"`** — the word this campaign uses for the
seeded draws it takes INSTEAD of a contiguous prefix ([[m88]]). Corpus counts over all of
`.omx/research/*.md`: **704** `stratified` against **29** `unratified` and no other collision of any
kind. The gate was systematically flagging the memos that did their sampling **right**.

That is a finding about the instrument, not an excuse. All 29 memos still owed an equations leg, and
all 29 now carry one. Both halves landed in this batch:

| | before | after |
|---|---:|---:|
| Catalog #344 live count, strict | **29** | **0** |
| …of which were the `stratified` substring | 16 (55.2%) | 0 (fixed) |
| …of which were real triggers needing a backfill | 13 | 0 (backfilled) |
| memos carrying a dated equations-leg addendum | 0 | **29** |
| canonical equations registered by this arm | — | **2** |

## And the second finding: why nobody saw it for a week

**Catalog #344 has never once executed at commit time.** VERIFIED AT SOURCE, then executed:

- the gate is registered `strict=True` in `preflight_all()` at `src/tac/preflight.py:7510` —
  **inside the `if check_codebase:` block**;
- `tools/preflight_hook.py::_preflight_command` appends `--no-codebase` on its default branch
  (`:909`), and that mode examines **0 of 27** codebase gates;
- MEASURED, not inferred: `python -m tac.preflight --no-codebase --acknowledge-empty-scope` emits
  **zero** `[catalog-344]` lines.

So a STRICT gate ran at release time and never at the typing moment, and 29 memos accumulated
between 2026-08-27 and 2026-09-03 with nothing to say so. The hook's own docstrings already record
this cause three times — it is exactly why the `#184`, subset-selection and negative-verdict scans
were built as hook STEPS rather than preflight gates. **Cured in this batch, per the charter's
strict-flip-atomicity instruction:** `run_canonical_equation_reference_scan` is now step 1d2 in
`main()`, ahead of `run_preflight()` (which early-returns on failure, so a later placement would be
skipped exactly when it is needed).

The step **imports Catalog #344's own predicates** rather than restating them — two statements of one
law drift. What differs is only SCOPE: the gate walks every `.omx/research/*.md` (~0.96 s), the step
reads the staged ones (~ms), which is what makes it honestly STRICT from byte one. It fails OPEN and
loud on a broken guard, matching every sibling.

Shipping that step is also why the substring fix was not optional. A commit-path blocker with a
measured 55% false-positive rate against the campaign's own sampling discipline would refuse honest
commits. The cure is the narrowest one the measurement supports — `(?<!st)ratified`, applied to that
one token through `_CHECK_344_TOKEN_OVERRIDE_RE`; `unratified` (a real ratification-status claim)
still triggers, and every other token keeps plain substring semantics. Five tests pin it, including
one that pins the blast radius (`test_only_the_ratified_token_carries_an_override`).

## 1. The two laws registered

### `renderer_seg_pose_coupling_shipped_object_v1`
`tac.canonical_equations.renderer_seg_pose_coupling_20260903`

Two arms three weeks apart measured the same structural fact about the shipped SM3R renderer and
neither reached the equations leg:

    |Δd_pose| = k · |Δd_seg|,   k ∈ [166.81, 217.30]   (MEASURED, n = 2 arms, 1.303× apart)

| anchor | edit kind | k | label |
|---|---|---:|---|
| `rf1` `film_amortized_flat_w96` | un-retrained STRUCTURAL swap | **166.80837961844966** | DERIVED from the memo's four published components |
| `ft1` step-600 aligned fine-tune | TRAINED seg-only, realized through the shipped receiver | **217.30366224024704** | MEASURED, `retained/verdict_ft1_step600.json` |

**A finding the test caught:** `ddm_rf1` never prints `166.8` — it prints the four components. The
number first appears in print in `ft1`, which re-derived it exactly as this module does. My first
test asserted the memo carried the literal and failed; the test now pins the truth
(`test_rf1s_memo_carries_the_components_not_the_ratio`) so the next reader who greps rf1 and finds
nothing knows why.

The law carries the closing arithmetic, not just the constant. At ΔB = 0 on the afr1 object
(`d_seg 0.00020139`, `d_pose 6.37e-06`, 180,002 B, S 0.14797617125559104):

    d_pose_max = (√(10·d_pose_base) + 100·|Δd_seg|)² / 10

A 25% seg cut funds **1.694e-05** (2.66× base) and costs **≥ 8.40e-03** (1,318× base) at the
*smallest* measured coupling. After the best n600 carrier recovery ever measured (8.0×, jg5) it lands
**62× over** the ceiling at k = 166.81 and **81× over** at k = 217.30 — and a renderer weight moves
all 600 pairs at once, so there is no per-pair admission lever. `seg_only_move_is_payable` defaults to
the most favourable reading of every measured input, so its `False` is a closure that holds a
fortiori across the band.

Two things are stated in the domain, not smuggled: **JOINT (pose-priced) formulations are EXCLUDED**
and remain OPEN (w96b's 204× with pose in-loop vs 2,366× without is an order of magnitude, and that
gap is the whole remaining question); and **direction symmetry is an ASSUMPTION** — both anchors moved
d_seg *up*, and applying k to a seg *cut* assumes local linearity of the realized map. That is the
cheapest thing a future arm could falsify.

### `annulus_restricted_prefix_bias_detector_v1`
`tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904`

Checked first, as the charter required: no existing law covers it. `wallclock_fixed_cost_prefix_bias_v1`
is the TIME costume of [[m88]] (`r + F/n`); `seed_ensemble_falsifier_band_v1` is the SEED costume. The
RESTRICTED-STATISTIC costume was unregistered. dr1's n600 δ_R anchor was already appended to
`margin_band_satisficing_threshold_v1` by that arm, so it did not need re-appending; what was missing
was the detector.

MEASURED (dr1, same field, same pass, two cohorts):

| statistic of |m1 − m0| | n96 prefix | n600 | bias |
|---|---:|---:|---:|
| annulus-restricted p95 (δ_R) | 0.019590163230895963 | 0.021881818771362305 | **+11.698%** |
| all-pixel p95 | 0.038173675537109375 | 0.0383458137512207 | +0.451% |
| all-pixel mean | 0.01356075331568718 | 0.013560148887336254 | −0.004% |

**Amplification 25.94×.** The prefix positive control reproduces the independent n96 artifact at
relative difference **0.000e+00**, bit-identical across two months and two runs — so 100% of the
deviation is cohort and 0% is instrument. Band-robustness is banked too: narrowing the annulus 4×
moves δ_R only −8.28%, and even the narrowest n600 band sits +2.45% above the n96 band-1.0 value.

The exported rule is the reusable half: `global_check_is_blind` and `prefix_constant_is_suspect`. A
global-statistic agreement never clears a restricted-set constant measured on a contiguous prefix;
only a re-measure of the restricted statistic on the full population, or on a seeded random draw,
does. Honest boundary, in the domain: **n = 1 restriction measured**. The generic statement over other
restrictions is DERIVED from the dilution mechanism, not MEASURED.

There is a pleasing symmetry worth naming: this arm registered the detector for prefix bias, and then
measured that the gate it was sent to fix had been mis-firing on the very word the campaign uses to
*avoid* prefix bias.

## 2. The sweep — 29 memos, append-only

Every row is a dated addendum appended to the end of the memo. **No body was rewritten**
(Catalog #110/#113). 20 MEASURED memos name a law with its module and its relation; 9 review/process
memos carry a `# FORMALIZATION_PENDING` waiver whose rationale names the law it would need and why it
is not yet derivable. The 16 stratified-only memos additionally carry the misfire note, so the reader
knows the citation stands on its own merit.

| memo | kind | law | relation |
|---|---|---|---|
| `ddm_ft1_shipped_renderer_aligned_finetune` | MEASURED | `renderer_seg_pose_coupling_shipped_object_v1` | IN-DOMAIN ANCHOR (2 of 2) |
| `ddm_ar1_aa_render_price_on_born_field` | MEASURED | `aa_sdf_observation_footprint_render_dseg_v1` | REFINES (landed for it at d3212bed1) |
| `ddm_fcd1_field_for_coder_diagonal` | MEASURED | `field_change_bhw_decomposition_v1` | IN-DOMAIN ANCHOR (this memo IS the anchor) |
| `ddm_gf2_static_dynamic_generator_form` | MEASURED | `decoder_derivable_ideal_savings_ceiling_v1` | IN-DOMAIN ANCHOR (ceiling-first refusal) |
| `ddm_lc3_lane_carriage_rung` | MEASURED | `static_packet_custody_byte_delta_score_savings_v1` | IN-DOMAIN ANCHOR (rate-only ΔS) |
| `ddm_jc1_afr_rc64_joint_redesign` | MEASURED | `decoder_causal_condition_transport_v1` | IN-DOMAIN ANCHOR (refusal side) |
| `ddm_na11_negative_regrade` | MEASURED | `compensated_semantic_edit_exchange_v1` | IN-DOMAIN ANCHOR |
| `ddm_nx1_next_object_route` | MEASURED | `procedural_predictor_plus_residual_correction_savings_v1` | CONSULTED, NOT ANCHORED |
| `ddm_qbr1_born_fairform_burn_prep` | MEASURED | `ema_decay_run_geometry_v1` | IN-DOMAIN (consumes it) |
| `ddm_qn1_qbr1_n600_realization_ticket` | MEASURED | `gap_decomposition_against_demonstrated_floor_v1` | REFINES the m66 clause |
| `SPEC_ddm_qbflow_packet_schema_v1` | MEASURED | `static_packet_custody_byte_delta_score_savings_v1` | IN-DOMAIN (schema-surface gate) |
| `ddm_qbt1_r1_r2_qbflow_verdict` | MEASURED | `islands_necessity_floor_big3_only_v1` | DOMAIN-EXTENSION CANDIDATE |
| `ddm_qbt2b_r3_ce_birth_verdict` | MEASURED | `logit_adjustment_class_prior_law_v1` | DOMAIN-EXTENSION CANDIDATE |
| `ddm_qbt2b_r4_extended_ce_verdict` | MEASURED | `logit_adjustment_class_prior_law_v1` | the law's PREMISE, measured |
| `ddm_qbt2b_r5_balanced_ce_verdict` | MEASURED | `logit_adjustment_class_prior_law_v1` | a REFINEMENT (the cure has a cost) |
| `ddm_qbt2b_r6_born_field_margin_verdict` | MEASURED | `mcf_minority_erasure_inevitability_v1` | DOMAIN-EXTENSION CANDIDATE |
| `ddm_qbt2b_r7_constrained_margin_verdict` | MEASURED | `trajectory_derived_stopping_law_v1` | IN-DOMAIN (trajectory segment) |
| `ddm_qbt2b_r8_constrained_margin_verdict` | MEASURED | `trajectory_derived_stopping_law_v1` | IN-DOMAIN (first doubling) |
| `ddm_qbt2b_r9_constrained_margin_verdict` | MEASURED | `trajectory_derived_stopping_law_v1` | IN-DOMAIN (pre-registers the rule) |
| `ddm_qbt2b_r10_third_doubling_stop_verdict` | MEASURED | `trajectory_derived_stopping_law_v1` | IN-DOMAIN ANCHOR (STOP fired) |
| `ddm_fr2_final_fresh_eyes_pr_review` | WAIVER | — | owes a review-efficacy law |
| `ddm_pq13_pr_body_refresh_verdict` | WAIVER | — | owes a disclosure-staleness law |
| `ddm_ht1_red_debt_hygiene_verdict` | WAIVER | — | owes a debt-decay law |
| `ddm_hv3_done_arm_consumption` | WAIVER | — | owes a consumption-rate law |
| `ddm_hv4_recovery_consumption_sweep` | WAIVER | — | owes a consumption-rate law |
| `ddm_rc_precheck_folded_never_fired` | WAIVER | — | owes the activation-ledger law |
| `ddm_hp1_premise_lint_canonicalization` | WAIVER | — | owes a premise-decay law |
| `ddm_fpr1_falsified_premise_registrations` | WAIVER | — | owes a premise-decay law |
| `ddm_ql2_apparatus_debt` | WAIVER | — | owes a debt-decay law |

**The label I refused to fudge.** Eight of the qbt2b/qbt1 rows read **DOMAIN-EXTENSION CANDIDATE**,
not ANCHOR. Those laws declare `vehicle: softmax_of_sdf_levelset_witness` (or its trainer's lever) and
the QBFLOW born-field generator is a different vehicle. Citing them as in-domain anchors would be the
cross-regime-constant transfer ([[m143]]) those memos' own STORES CONSULTED warn about. The addendum
names the law, states the mechanism match, and says plainly that the transfer is a candidate awaiting a
matched re-measure. That is the honest equations leg; a forced in-domain claim would have been a fake.

## 3. Red debt found, NOT fixed — named with an owner

`src/tac/canonical_equations/tests/` is **RED on main at HEAD, before my change** (verified by
stashing). Six tests fail; five are one cause:

```
.venv/bin/python -m pytest src/tac/canonical_equations/tests/ -q
  test_margin_band_satisficing_threshold_20260712.py  5 failed
    Obtained: 0.04376363754272461   Expected: 0.039180326461791926 ± 3.9e-08
  test_custom_sparse_adjoint_achieved_ceiling_20260713.py::test_equation_is_honest_about_missing_metal_anchor
```

Cause: the dr1 n96→n600 repoint (dr1 addendum, MAIN 2026-09-04) moved the live `m_safe` default to
`0.04376363754272461`; its memo says three pinned tests were updated, and this module was not among
them. The tests now assert the superseded n96 value.

I did **not** fix it. It is another arm's landing, it pins a live lever default while the QBR1 burn is
resident, and absorbing a sister's work into my batch is the collision this repo's serializer
discipline exists to prevent. Cure and owner are in NEXT_IF_RESUMED.

## 4. A second gate gap, PINNED not fixed

MEASURED: Catalog #344's placeholder-rationale rejection is **exact-match only**. The waiver regex
captures `[^\n]+`, so inside the corpus's own canonical `<!-- # FORMALIZATION_PENDING:… -->` form the
captured rationale is `"<rationale> -->"` — neither an exact placeholder nor under the 4-char floor,
so a literal placeholder is ACCEPTED there. My own test asserted otherwise and failed; it now PINS the
behaviour (`test_catalog_344_placeholder_rejection_is_exact_match_only_MEASURED_GAP`) so the gap is a
recorded finding rather than an assumption. Widening the check changes the STRICT gate's semantics for
the whole corpus and belongs to a charter that owns it. All nine waivers I wrote carry substantive
rationales, so nothing in this batch depends on the gap.

## MEASURED / DERIVED / TRANSFERRED

- **MEASURED:** the 29→13→0 live counts; 16/29 = 55.2% stratified-only; 704 `stratified` vs 29
  `unratified` corpus-wide; the gate's 0 of 27 codebase gates under `--no-codebase`; the 6 pre-existing
  red tests; the exact-match placeholder gap; `k_ft1 = 217.30366224024704` from the retained receipt;
  every dr1 quantile.
- **DERIVED:** `k_rf1 = 166.80837961844966` from rf1's four published components; the band centre
  190.38926383452008 (geometric — the band is multiplicative) and both anchor residuals; every ceiling
  and overshoot in §1; the 25.94× amplification.
- **TRANSFERRED:** the afr1 object's `d_seg`/`d_pose`/bytes/S from its `[contest-CUDA T4 n600]`
  receipt; the 5.87×/8.0× carrier-recovery ceilings (fcd2/jg5).
- **ASSUMED:** direction symmetry in the coupling law's closing arithmetic — stated in the domain, not
  hidden.

## GESTALT-DELTA

The equations leg's failure was not neglect, it was **an unreadable instrument on the wrong side of
the commit boundary**. A STRICT gate that only runs at release time reports a backlog nobody can act
on at the moment they could fix it; and when more than half of what it reports is a substring artifact
against the campaign's own sampling discipline, the number stops being read at all. This is the
[[m50]] vacuity genus wearing a new costume: not "the check examined nothing", but "the check examined
the wrong thing, loudly, where nobody was standing." The cure is the same shape as the sisters that
came before it — put the law where the typing happens, make it state its denominator, and narrow the
token to what it actually means.

The coupling law adds a second, harder gestalt line: **the renderer is not a seg lever.** Two
independent arms, two mechanisms, one object, 1.303× apart — the seg-only direction on the shipped
renderer costs pose at ~170–220× and the arithmetic closes at both ends. What stays open is exactly
one thing, and it is now written where a future charter must read it before it launches: a joint,
pose-priced objective that refuses the seg-only direction and searches for a lower-coupling one.

## NEXT_IF_RESUMED

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **Green the 6 pre-existing red equation tests** — update `test_margin_band_satisficing_threshold_20260712.py` to the n600 `m_safe = 0.04376363754272461` the live law now resolves, plus the `custom_sparse_adjoint` honesty assertion | **QUEUED-WITH-FIRE-ORDER, fires FIRST** | MAIN (owner of the dr1 repoint) | fires now — it is stale test expectations behind a landed change, not a design question |
| 2 | **Measure the JOINT (pose-priced) coupling from the ft1 weights** | **QUEUED** | MAIN to assign | fires when a scorer slot frees; it is the one branch `renderer_seg_pose_coupling_shipped_object_v1` explicitly does not close |
| 3 | **Falsify direction symmetry** — one seg-DECREASING edit on the shipped renderer, k re-measured | **QUEUED, cheap** | unowned; MAIN to assign | fires with #2; it is the coupling law's only stated assumption |
| 4 | **Widen Catalog #344's placeholder rejection** to catch a placeholder inside the HTML-comment form | **QUEUED, no fire order** | unowned | fires only if a memo is found using the gap; the behaviour is pinned by a test either way |
| 5 | **A second restriction for the detector law** (per-class, per-margin-band, per-region prefix bias) | **QUEUED** | unowned | fires when any arm next measures a restricted-set constant; it would turn n = 1 into a family |

---

Pointer honesty: this arm registered two laws and repaired an instrument. It trained nothing, closed
no bytes, and could not move the frontier.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — UNMOVED.
