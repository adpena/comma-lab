---
arm: ddm_aa3
title: "Fresh-eyes adversarial audit of the 2026-08-16/17 arc. Fork resolves DEFECTS FOUND: 6 defects in 4 artifacts, all corrected at source in headline AND body. The largest is a noise floor quoted at the single step where it is 28x smaller than everywhere else -- 'well-powered, 23x headroom' becomes 0.81x at the end of the same window. Also corrected: a registered canonical equation whose model-comparison rows used two different denominators (and the wrong sign, in the direction that kills runs under its own named --walltime-cap-s consumer); an n-dependent over-pricing factor written as a constant; an evidence-table cell that switched aggregation exactly where it carried its rhetorical point; a re-pricing recipe that double-counts ~104 s; and a retired lint leg registered against another task's id. Eleven further claims re-derived from primary artifacts and found CLEAN, including one hole I predicted in the gitleaks entry and then failed to demonstrate under differential control."
utc: 2026-08-17
parent: "operator charter ddm_aa3 -- fresh-eyes adversarial audit of the 48h arc"
axis: "[audit] re-derivation from primary receipts -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_aa3 — fresh-eyes adversarial audit of the 2026-08-16/17 arc

STORES CONSULTED before any conclusion: `ddm_jr1_band_objective_judge_repair_20260817.md` ·
`ddm_l3000_no_descent_verdict_20260817.md` · `ddm_wallclock_prefix_bias_law_20260817.md` ·
`ddm_sf1_semantic_film_pose_map_20260817.md` · the registered canonical equation
`wallclock_fixed_cost_prefix_bias_v1`
(`tac.canonical_equations.wallclock_fixed_cost_prefix_bias_20260817`, queryable via
`tools/list_canonical_equations.py`) ·
`tools/codex_arm_queue.py` · `.gitleaks.toml` · all 9 pre-existing rows of
`.omx/research/falsified_premise_registry.jsonl` · the DAG FEED-20260817a block ·
CLAUDE.md (NO-FAKE, the verdict-scope ladder, "'off' is a tracked queue") · memories
[[m88]]/[[m96]] (prefix bias) · [[m89]] (task-ledger split: cite CONTENT, never a bare id) ·
[[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]] ·
[[measured_object_vs_named_object_20260816]] ·
[[the_denominator_and_the_falsifier_can_both_be_vacuous_20260816]].

**Method, and it is the whole method:** every number below was re-derived from the PRIMARY artifact
— run receipts, `run.log` history arrays, `result.json`, `safe_run_status.json`, the corrections
index itself, `git show` — never from a memo's summary of itself. That is exactly how `jr1` caught
MAIN's "in-sample residual" error (it opened the receipt instead of reading a heading), and it is
what found every defect here.

## OPTIMAL FORM

**Reference form** of an adversarial audit: re-derive each load-bearing claim from primary
artifacts under a named lens, with a pre-registered fork, and correct what fails AT SOURCE in both
headline and body.

| delta from that reference | class | why |
|---|---|---|
| ranked by blast radius; the unaudited remainder is NAMED in §6 | **SCOPE** | legal; silent truncation is the defect this arm exists to catch, so it is named rather than committed |
| — | **MECHANISM** | **none reduced.** Every claim was re-derived from the primary artifact, not from a memo. The gitleaks check was run as an executed differential (with-entry vs without-entry), not reasoned about. The determinism floor was measured from both runs' retained histories. |

No TOY-BRACKET is owed: no mechanism was reduced.

## PRE-REGISTERED FORK — resolved

Written in the charter before measurement. **Resolved: DEFECTS FOUND.** Six load-bearing claims in
four artifacts are wrong or over-scoped. All six are corrected at source. Eleven further claims
were checked and are CLEAN (§5) — listed with how each was checked, because a clean round is only
credible if the checking is visible.

## ANSWER FIRST

**The recurring shape across all six defects is the same one the arc kept catching in others: a
number is correct for the object it was measured on, and is then quoted for a wider object.** Not
one of the six is an arithmetic slip — every fit, every ratio, every census in the arc reproduces to
the digits. The failures are all at the boundary between *what was measured* and *what was claimed*.

**The largest, and the only one that would have mis-routed a decision:**

> `A2_repeat` was fired to establish the MPS run-to-run floor. It reported **72 flips, 0.2650% —
> against a 6.1% band effect, 23× headroom. The instrument can separate the arms.**
> That floor is measured **at step 100 and nowhere else.** The same A/A pair — identical config,
> identical seed — differs by **7.19%–12.48% at steps 200–600** (mean 9.51%). **At step 600 the
> floor is 7.52% and the headroom is 0.81×: the noise exceeds the effect.**

One re-run yields six floors spanning **48×**, and the one that was quoted is the smallest. Step 100
is the peak of a CE spike both runs hit hard and near-identically; it is the least representative
step in the window, not a summary of it.

*What survives:* jr1 §5.3's Leg C design compares **peaks at step 100** against A2's 27,170, so its
pre-registered bar is internally consistent and unaffected. jr1 §4 had already written the honest
sentence ("every single number here is inside the instrument's noise") — this audit supplies the
number that sentence lacked. And the L3000 no-descent verdict is **unaffected**: its end-state gap
(3,767 flips) is **6.2×** the measured A/A difference at step 600 (605 flips).

*What does not survive:* "the instrument can separate the arms" as a general claim, and
"well-powered" as a property of this trainer. Withdrawn in all three places they appear.

**Frontier UNMOVED: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.**
$0, no Modal, no launch, no scorer.

## §1 The six defects

| # | artifact | claim as published | measured | lens |
|---|---|---|---|---|
| D1 | `wallclock…v1` (registered equation) | `PREDICTION_SCORECARD["two_point_fit"] = (1463.0, **+0.061**)` | **−0.05735** — wrong sign AND wrong denominator | wrong denominator |
| D2 | wallclock memo title + equation docstring | "every window … priced **4.9×** too expensive" | 1.00× @ n=50 · 4.89× @ n=600 · **6.82× @ n=3,000** · 7.57× asymptote | scope over-claim |
| D3 | wallclock memo · L3000 verdict · DAG | "**72-flip floor**, 23× headroom, well-powered" | step-100 only; **7.19–12.48%** at steps 200–600 | wrong denominator / population |
| D4 | L3000 verdict · DAG | "the rate **7.4×** slower" | **3.52×–7.36×** — the numerator is one run's tail rate | units × level |
| D5 | L3000 verdict evidence table | step 1,800 `Δ/100 = **+320 (rises)**` | local rate is **−218 (falls)**; the rise is at 1,500→1,600 (+1,196) | aggregation |
| D6 | `tools/codex_arm_queue.py` | retirement registered against "task #1085" | #1085 is another task; **zero** ledger rows for this leg | wrong object / untracked "off" |

### D1 — a registered canonical equation reports a model comparison in two different units

`PREDICTION_SCORECARD` is labelled `(predicted_s, signed_relative_error)`. Recomputing from the
three stored predictions against the measured **1552.209 s**:

| row | predicted | stored | `(pred−meas)/meas` | verdict |
|---|---:|---:|---:|---|
| `b2e_single_point` | 9,972.0 | +5.424 | **+5.4253** | consistent |
| `naive_from_600_e2e` | 2,040.0 | +0.314 | **+0.3144** | consistent |
| `two_point_fit` | 1,463.0 | **+0.061** | **−0.05735** | **wrong sign + wrong denominator** |

`+0.061` is `(measured − predicted)/**predicted**` — the other denominator and the other sign. The
memo's "+6.1%" is a correct English sentence ("the measured total is 6.1% above the prediction");
it is simply not this field's definition, and the file's own `residual` field had already been set
to `0.05735` with a comment noting the two normalisations differ. The scorecard was not brought
along.

**The sign is load-bearing.** The equation names `tools/launch_detached_process.py --walltime-cap-s`
as a canonical consumer. The fit **under**-predicts by 89 s, so a cap set at
`predicted_total_seconds` is 5.7% *short* of the measured run and kills it. Stored as `+0.061` the
error reads as over-prediction — the safe direction — which is the opposite of the truth.

**Corrected**, with the convention now stated inside the container.

### D2 — the over-pricing factor is a function of `n`, written as a constant

The factor is `3.326·n / (F + r·n)`:

| n | 50 | 600 | 1,000 | 3,000 | → ∞ |
|---|---:|---:|---:|---:|---:|
| factor | **1.00×** | **4.89×** | 5.70× | **6.82×** | **7.57×** |

It is **1.00× at n=50 by construction** — that is where b2e's smoke was measured. The memo's own
re-pricing table already carried a 6.8× row for n=3,000 while its title said 4.9×. This is the
headline-vs-body genus with the body correct.

The `(F, r)` separation itself is sound and validated out-of-sample; only its single-number summary
was over-scoped. **Corrected** in the title, the body, the equation docstring, and the DAG, and the
re-pricing table now carries an explicit factor column.

### D3 — the noise floor, quoted at its most favourable step

A2 and `A2_repeat` are the same config and the same seed (`20260715`). Full argv diff:
`--band-objective-weight` explicit-`0.0` vs defaulted-`0.0`; `2e-5` vs `2.0e-5`; the `--save`
basename. Nothing else. So every difference below is pure MPS run-to-run nondeterminism:

| step | A2 | A2_repeat | Δ | Δ as % of A2 |
|---:|---:|---:|---:|---:|
| **100** | 27,170 | 27,098 | −72 | **0.26%** |
| 200 | 14,237 | 12,460 | −1,777 | **12.48%** |
| 300 | 12,009 | 11,146 | −863 | 7.19% |
| 400 | 9,607 | 10,415 | +808 | 8.41% |
| 500 | 8,415 | 9,419 | +1,004 | 11.93% |
| **600** | 8,049 | 8,654 | +605 | **7.52%** |

Headroom against the 6.1% band effect: **23.0× at step 100 · 0.81× at step 600.**

⚠ And the floor itself is **n = 1** — a single A/A pair is an *observed difference*, not an
estimated spread. jr1 §5.3's pre-registered `27,170 − 3 × spread` bar multiplies a one-sample
quantity by three; that is decorative, not a confidence statement. A second repeat costs 6.8 min by
the arc's own wall-clock law and is the cheap fix.

### D4 — a tail-rate ratio quoted from one of two identical runs

The "rate 7.4× slower" numerator is `A2_repeat`'s step-500→600 local rate, **−765/100**. A2's own,
same config and seed, is **−366/100**. Against L3000's −104/100 the ratio is **3.52×–7.36×** — a
2.1× span from nondeterminism alone.

The same fragility sat inside the *already-refuted* extrapolation: "~1,131 steps from parity" would
have been ~2,199 steps off A2's rate, so that estimate was `1,131–2,199` before the decaying-tail
error was even reached. Two independent defects in one sentence; only one had been caught.

### D5 — one cell of an evidence table switches aggregation

The L3000 verdict's `Δ per 100` column is the **local** 100-step rate at rows 100 / 900 / 1,500 /
2,400 / 3,000 — verified against all 31 history entries. The 1,800 row's **`+320 (rises)`** is the
**300-step average** over 1,500→1,800. Its local rate is **−218 — it falls.**

The *phenomenon* is real: the softplus transition does undo repayment. It happens one stage boundary
earlier, at **1,500 → 1,600 (+1,196)**. So the memo's qualitative point stands and its number and
step attribution do not — and the one cell that switched aggregation is the one carrying the
column's rhetorical weight. **Corrected**, with the true rise row shown instead.

### D6 — the retired lint leg is an untracked "off" pointing at another task's id

`tools/codex_arm_queue.py` retires `_lint_stale_numbers` "RETIRED-PENDING-STORE-REPAIR … task
#1085". In the repo ledger — **the only ledger arms can read** — `#1085` is
`1085_p0_always_keep_payload_retrofit_population_20260816`, owned by `next_retention_arm`:
unrelated work. And a search of all 565 ledger rows finds **zero** mentioning this leg, the
corrections store, or a `quantity` field. So a detector was switched off with no owner, no
fire-condition, and a citation that misroutes whoever follows it.

This is the recorded genus [[m89]] (harness TaskList ≠ repo ledger; **cite CONTENT, never a bare
id**) meeting the standing law that "off" must be a tracked queue. **Corrected** in the docstring
where the retirement lives, with owner and fire-condition stated inline.

## §2 The re-arm path — the charter's question, answered CLEAN

*"Does the fail-closed gate actually re-arm on an index rebuild, or is it permanently dark?"*

**It re-arms.** `_corrections_index_identifies_quantities()` probes only the first parseable row,
justified by "the index is homogeneous by construction". That justification holds *because*
`tools/au1_measurement_integrity_audit.py:820` writes via `write_jsonl`, which opens the path with
**mode `"w"`** — a rebuild replaces the file wholesale, so no stale first row can survive to pin the
gate dark. Had the builder appended, the probe would have been permanently blind. Sound, but for a
reason the docstring did not state; now stated.

## §3 The gitleaks entry — I predicted a hole and my own control refuted it

The charter asked whether `^expect_(?:bytes|sha)=[A-Z][A-Z0-9_]*$` is as tight as its comment
claims. My hypothesis: an AWS access key ID is exactly `[A-Z][A-Z0-9]*`, so `expect_sha=AKIA…`
should slip through a top-level allowlist that applies to every rule.

**Executed** (gitleaks 8.30.1, differential, faithful reproduction of the real `sf1:518-521`
custody block, with a non-canonical planted key per the entry's own control-design note):

| line | with entry | without entry |
|---|---|---|
| `expect_bytes=FRAMES * 2 * FRAME_BYTES` | **silent** | **generic-api-key FIRES** |
| `api_key = "<realistic non-canonical AWS-shaped key, 20 chars, AKIA-prefixed>"` | **FIRES** | FIRES |
| `expect_sha="ghp_…"` | **FIRES** | FIRES |
| `expect_sha=AKIA…` (bare) | silent | **silent** |
| `expect_sha="AKIA…"` (quoted) | silent | **silent** |

**The hypothesis is refuted.** The AWS variants are silent in **both** configs, so the entry is not
the cause — `generic-api-key` requires a credential-shaped keyword (`api_key`, `token`, `secret`)
that `expect_sha` does not supply, and this ruleset has no AWS-prefix rule firing independently.
Removing the entry changes exactly one finding: the `expect_bytes=FRAMES` false positive it was
written for. **The entry is verified tight and every claim in its comment reproduces.** CLEAN.

*(Recorded because a control that refutes the auditor is the only kind worth running. The residual —
`expect_sha="AKIA…"` silent — is a pre-existing property of the base ruleset, unrelated to this
entry, and I make no claim about it beyond naming it.)*

## §4 What I did NOT find — the arc's arithmetic is unusually sound

Across four memos, one registered equation and ~11,840 index rows I found **no arithmetic error**.
Every fit, ratio, census and break-even reproduces to the digits it claims. That is worth stating
plainly: the arc's failure mode is scope and aggregation, never calculation.

## §5 CLEAN — eleven claims re-derived, and how

| claim | how checked | result |
|---|---|---|
| jr1: "the charter's in-sample premise is FALSE" | opened `ddm_rg1b_lr1_refit_and_bar_20260816.json` | `fit_peak_vs_dw100 = {n: 4, dof: 2}`, 4 residuals — **jr1's correction is right** |
| jr1 free-judge floor 13.3% vs 6.1% effect | recomputed `1.96 × 0.072827 = 0.14274`, `exp → 1.15343`, `1 − 1/1.15343 = 13.28%`; effect `exp(0.06342) → 1.06547 → 6.145%` | exact; ratios 2.251× (log) / 2.161× (pct) both reproduce |
| jr1's own 15.3% → 13.3% correction | headline and §2.4 table both read 13.3% | correctly applied, body and headline agree |
| L3000 no-descent core | both `result.json`: `best_step = 0`, `improved_over_init = False`, `quantized_exact_seg == init` exactly (`0.00028616163465711804`) | exact |
| L3000 "43.5% closer, 3,767 of 8,654, rate decayed 32×" | recomputed from the 31-row history | exact — **and it clears the noise floor at 6.2×** |
| wall-clock two-point fit | re-solved from (50, 166.30) and (600, 408.0): `r = 0.439455`, `F = 144.327`, `n=3000 → 1462.7 s` | exact; reproduces b2e's 3.326 at n=50 |
| wall-clock elapsed values | `resource_safe_run_status.json` | 408.258 s and 1552.209 s — **receipts match** |
| lint-retirement census | parsed all 11,840 rows | 11,840 ✓ · `15,157` has **ZERO** rows ✓ · **no `quantity` field on any row** ✓ · artifact rates reproduce within 1–3 rows |
| gitleaks entry tightness | executed differential, 5 shapes | **entry verified tight**; my hypothesis refuted (§3) |
| sf1 end-to-end arithmetic | recomputed 14.23%/19.94%, net +0.062227 (keep75), 0.00078%, 76.8×, 855.6×, 192.9×, 5.65×, 2.96×, 3,942×, 184× | **every one reproduces to 4 digits** |
| pl1: "the complete result survived" | compared A2's recovered payload against its cohort | C0/A1/A3/W1 also lack `init_quantized_exact_seg`/`best_step`/`improved_over_init` — **field-complete for its cohort; claim holds** |

One incidental note, not a defect: the wallclock memo cited `verdict = PASS` as corroboration.
`ddm_pl1` had already measured that this trainer reports `PASS` on every arm regardless of descent —
including this one, which did not descend. Removed rather than repeated.

## §6 The unaudited remainder — NAMED, per this arm's own OPTIMAL FORM

Ranked by blast radius, these were **not** re-derived and no claim here covers them:

1. `jr1`'s Leg B rotation cosines (0.5244 → 0.4019) and Leg B2 AdamW reconstruction — the vectors
   are retained in `JR1_VECTORS.npz` (12.4 MB) but I did not recompute them from the checkpoints.
2. `jr1` §2.5's W1 `t = +12.87` and the END-law fit of §2.6.
3. `sf1`'s n120 group measurements themselves (I verified the arithmetic *over* them, not the
   renders under them) and its 8-row control table.
4. `ddm_cl3`, `ddm_a1s` FO-A, `ddm_gl2`, `ddm_pl1`'s gate, and the `ra2crr` row-2/3 corrections —
   inside the 48h window, outside what one pass could re-derive from primaries.

## §7 Verdict scope

`verdict_scope: INSTANCE` on all six defects — each is a specific claim in a specific artifact,
corrected at source. **No paradigm, family or method is refuted by this unit.** The L3000
no-descent FORMULATION verdict, the wall-clock `(F, r)` law, jr1's structural refusal of the
residual judge, and sf1's family closure all **survive** this audit; three of them are now stated
with a scope or a band they previously lacked.

`verdict_scope: FORMULATION` on one methodological finding: **a noise floor measured at one point in
a trajectory does not characterise the trajectory.** On this trainer one A/A re-run yields six
floors spanning 48×. Any future power claim on it must name its step.

## §8 Corrections landed at source (headline AND body, per the standing genus)

| file | what changed |
|---|---|
| `src/tac/canonical_equations/wallclock_fixed_cost_prefix_bias_20260817.py` | D1 scorecard `+0.061 → −0.05735` + convention stated in-container; D2 docstring scoped to n, with the factor-vs-n table; save count `6 → 7` per receipt; the vacuous `verdict = PASS` citation removed |
| `.omx/research/ddm_wallclock_prefix_bias_law_20260817.md` | **title rewritten** (D2); correction banner; D3 full six-step floor table replacing the one-step claim; factor column added to the re-pricing table; NEXT_IF_RESUMED row 1 struck and superseded (D4); row 2 gains the `14.8 s/save` double-count warning |
| `.omx/research/ddm_l3000_no_descent_verdict_20260817.md` | D5 evidence-table row corrected to the true local rate and true rise step; D4 band `7.4× → 3.5×–7.4×` with the noise context; NEXT_IF_RESUMED row 1 floor caveat; row 4 recipe de-double-counted |
| `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` | FEED-20260817a amended for D2, D3, D4 and (by reference) D1 |
| `tools/codex_arm_queue.py` | D6 citation corrected to CONTENT, owner + fire-condition stated, re-arm mechanism documented |
| `.omx/research/falsified_premise_registry.jsonl` | 3 new rows (9 → 12) with `claim_patterns`, so a future charter quoting `72-flip floor` / `23× headroom` / `4.90× too expensive` / `(1463.0, 0.061)` is caught at spawn |

## NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | **Fire a second `A2_repeat` (A/A/A).** The determinism floor is n=1 at every step; two pairs give a spread instead of a difference, and jr1's `3 × spread` bar needs one. **6.8 min** by the arc's own law. | unowned | before any future power claim on this trainer |
| 2 | **Re-fit the wall-clock model with save counts, from ≥3 points.** `(F, r)` absorbs ~7 saves into `F`; the `14.8 s/save` is back-solved from the residual it explains and does not close against n=50 (~107 s predicted vs 166.30 measured). Three points with recorded save counts separate all three terms honestly. | unowned | before quoting a per-save cost |
| 3 | **Add `quantity` to the au1 corrections index** and `_lint_stale_numbers` re-arms with no code change (§2). The `falsified_premise_registry` schema is the model. | unowned | any arm that rebuilds the au1 index |
| 4 | **Audit the §6 remainder**, starting with jr1's Leg B cosines — `JR1_VECTORS.npz` retains every vector, so it is a $0 re-derivation. | unowned | next audit pass |
| 5 | **The instrument question the arc never asked:** every step-100 value in this family is a CE spike peak that both runs reproduce to 0.26% while later steps diverge 9.5%. That is a strong hint the spike is *deterministic* and the divergence accumulates after it. If so, step-100 quantities are the trainer's most reliable observable and later ones its least — worth knowing before designing another comparison. | unowned | before the next matched-arm design |

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED by this unit.** Gap to 0.15: **−0.0095973**.

---

## ⚠ REDACTION (MAIN, 2026-08-17) — the planted control key is NOT committed

The differential control above was **executed as reported**; only the 20-character literal is
redacted. It was a SYNTHETIC key this arm constructed — never a live credential — planted per the
control-design note in `.gitleaks.toml` ("plant a realistic non-canonical key"), and the auto-push
hygiene gate correctly HELD the outgoing diff on it.

**The gate is right and this is not a false positive to wave through.** A scanner cannot
distinguish a documented fixture from a live key and must not try; an AWS-shaped literal in a
public repo trips GitHub secret scanning and pollutes history regardless of whether it unlocks
anything. The commit was unpushed, so amending keeps the literal out of published history entirely.

**The gap was in MAIN's guidance, not this arm's execution.** The note said *plant a realistic
key*; it never said *and do not commit the literal*. Both halves are now stated at the source
(`.gitleaks.toml`): run the control with a realistic key, record the VERDICT, commit a SHAPE
DESCRIPTION — never the characters. Genus: an instruction that is correct about the measurement
and silent about the artifact.
