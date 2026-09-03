---
arm: ddm_xr1_exchange_ratio_noise_floor
date: 2026-09-03
status: COMPLETE_INSTRUMENT_LANDED_POINTER_UNMOVED
axis: "[macOS-CPU advisory / scorer-free exact RC64 byte replay plus retained-score pair bootstrap; no scorer, Metal, Modal, or contest evaluation]"
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
canonical_equation: exchange_ratio_noise_floor_v1
verdict_scope: "INSTRUMENT: the noise floor of the byte<->distortion exchange ratio, measured on two named physical objects (JBP1 row A, FCD3 published tau_1e-6). The intervals do NOT transfer to other edit sets."
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_xr1 — the exchange ratio has a noise floor, and every near-win margin we quote sits inside it

## Result first

The exact pointer did not move. This arm ran no scorer, no Metal, no Modal, and no contest
evaluation. It is apparatus.

The campaign closes and re-opens rows on statements like "row X is 1.04× its bar". Until today
that ratio was a point with no measured dispersion. `ddm_rn1` screened **300** such near-win rows;
`ddm_ww1` §3.5 recorded plainly that "nobody has measured it". I measured it.

Three numbers, all new:

| statistic | object | measured 95% interval | half-width |
|---|---|---|---|
| σ_B, physical byte noise | afr1 field, null overlay, ×3 complete RC64 re-encodes | **0 B — three byte-identical archives** | **0.00 B** |
| ΔB, pair bootstrap | JBP1 row A (5,506 edits, 567 pairs, exact −2,950 B) | **[−3,159.27, −2,758.32] B** | **200.48 B (6.80%)** |
| ΔS, pair bootstrap | FCD3 published τ=1e-6 (exact −2,940 B, realized n600) | **[+0.00171513, +0.00219814] S** | **0.00024151 S (12.43%)** |

And the consequence that matters:

> **The FCD3 exchange ratio is r = −0.5018, 95% interval [−0.5311, −0.4733]. Its half-width is
> 5.76% of r. Every one of `rn1`'s top-20 near-win rows sits within 4.00% of its bar — that is,
> INSIDE the only exchange-ratio dispersion this campaign has ever measured.**

A 1.04× miss and a 0.98× clearance are the same measurement.

## 1. σ_B — the physical leg

I re-encoded the same field three times, complete and from frame zero, through the RXC1 exact
coder, with a null overlay (0 tokens changed). Each encode is a full n600 causal RC64 pass — not a
restart, not a resumed suffix.

| repeat | stream bytes | stream sha256 | archive bytes | archive sha256 | differing bytes vs repeat 0 | wall |
|---|---|---|---|---|---|---|
| 0 | 113,411 | `5601d6fd792c60c1…` | 180,002 | `cbb8d928a8ccdd3f…` | stream 0 · archive 0 | 1,383.0 s |
| 1 | 113,411 | `5601d6fd792c60c1…` | 180,002 | `cbb8d928a8ccdd3f…` | stream 0 · archive 0 | 2,050.2 s |
| 2 | 113,411 | `5601d6fd792c60c1…` | 180,002 | `cbb8d928a8ccdd3f…` | stream 0 · archive 0 | 1,635.1 s |

**σ_B = 0.0 B. Spread max−min = 0 B. Zero differing bytes, on the stream and on the archive.**
The charter's prior law held.

Two things this is stronger than a byte-count check.

First, `stream_delta_bytes = 0` on every repeat: the null re-encode reproduces the **shipped**
token stream, not merely a stream of the same length. Second, the archive sha256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` at 180,002 B is exactly
`our_local_frontier_contest_cuda.archive_sha256` and `archive_bytes` in
`.omx/state/canonical_frontier_pointer.json`. **The re-encode reconstructs the shipped frontier
archive bit-for-bit, three times, from scratch.**

Wall clock told the opposite story: 1,383 s, 2,050 s, 1,635 s — a 48% spread across identical
work, from machine contention alone. **Wall clock is not a byte proxy.** Bytes moved 0%.

Honest bound on this leg: three repeats cannot put a tight upper bound on a small *nonzero* σ_B.
What they can do, and did, is confirm bit-exact reproduction. On this evidence σ_B is exactly zero,
and a single differing byte would have been the finding — the summary records the falsifier rather
than raising it (see DEAD-ENDS #1).

**The consequence is the whole point of this arm: the exchange-ratio noise floor is ENTIRELY
statistical.** Re-encoding adds no dispersion to a byte claim. All the uncertainty in "row X is
1.04× its bar" comes from *which 600 pairs*, and that is what §2 and §3 measure.

## 2. ΔB — JBP1 row A clears its own bar comfortably

JBP1 row A is 5,506 XOV1 B/H/W edits across 567 of the 600 pairs, exact **−2,950 B** through the
RXC1 physical coder. I resampled the 600 retained per-pair codelength contributions 200 times with
replacement (seed 20260903) and held the sub-byte rounding residual fixed at **c = +0.4534 B**, so
the identity draw reproduces −2,950 B exactly. The interval is centred on a number we measured, not
on an ideal-codelength surrogate.

* 95% interval **[−3,159.27, −2,758.32] B**; half-width **200.48 B**; resample SD 105.23 B.
* Half-width / SD = 1.906, against 1.96 for a normal — the resample distribution is near-Gaussian.
* Monte-Carlo residual at 200 resamples: **+1.66 B** on −2,950 B (0.056%).

**The charter's prior law HELD.** It predicted an interval narrower than ±600 B; the measured
half-width is 200.48 B, a third of that bound. The rate credit is real: the whole interval is
negative and nowhere near zero.

Its standing against the live rate-corner demand (−42,016 B) is **7.02%**, and the interval puts
that standing between **6.57% and 7.52%**. The demand share is not the fragile part of that claim.

In S units at the live exchange (6.658589531221714e-7 S/B) the credit is **0.00196 S**, interval
[0.00184, 0.00210] S. The byte interval is worth **±0.000133 S**.

## 3. ΔS — FCD3's refusal is not a close call

FCD3 published τ=1e-6 is the one edit set with BOTH axes measured at n600 on retained receipts. I
drove bytes and both distortion axes from the **same** resample index vector — decoupling them
would destroy the pairing and the ratio would stop being a ratio.

Point (re-derived independently from the two retained receipts, exact match to the retained value):

```
dS_seg  = 100*(0.0003874630492646247 - 0.0003474002587608993) = +0.004006279
dS_pose = sqrt(10*0.00014620431466028094) - sqrt(10*0.0001470109127694741) = -0.000105329
dS_rate = 25*(-2940)/37,545,489                                             = -0.001957625
dS      =                                                                    +0.001943324
```

Bootstrap, 200 resamples:

* ΔS 95% interval **[+0.00171513, +0.00219814]**; half-width **0.00024151**; SD 0.000125289.
* **The interval EXCLUDES zero.** The charter's falsifier did NOT fire. The win-win cone stays
  REFUSED, now with an interval instead of a point.
* Monte-Carlo residual: +6.62e-06 S (0.34% of the point).
* ΔB 95% interval [−3,183.12, −2,709.38] B, half-width 236.87 B (8.06%).
* Δd_seg 95% interval [+3.662e-05, +4.340e-05]; Δd_pose 95% interval [−1.287e-06, −3.327e-07].

The exchange ratio r = ΔS_rate/ΔS_dist has a **sign-stable denominator** across all 200 resamples,
so its interval is defined:

* **r = −0.5018, 95% interval [−0.5311, −0.4733]**, half-width 0.02892 (5.76% of r).
* Break-even is r = −1. The measured r sits **17.2 half-widths** away from it.

Read plainly: the rate credit pays back about **half** the distortion it buys, and the interval
never comes close to break-even. FCD3 is not a marginal refusal that better statistics could
rescue.

**A useful conversion for the campaign.** A realized ΔS half-width of 0.00024151 S is worth
**≈363 B of archive** at the live exchange. Any lever whose whole realized effect is smaller than
about ±363 B-equivalent is inside the noise floor of a single n600 realized measurement.

## 4. The re-graded rn1 top-20

I ranked `rn1`'s 300 near-win rows by |closest ratio − 1| and re-graded the top 20. The grade is
**measured, not asserted**: for each row's `ddm_<slug>` store prefix, the probe scans every matching
retained store under both custody roots for (a) a (600,)-shaped `bits_per_frame` ledger and (b) a
JSON receipt carrying an **exact** `d_seg_per_pair` / `d_pose_per_pair` key whose list lengths cover
all 600 pairs.

The strictness is load-bearing. A substring probe reports `ddm_pa1r` as having per-pair distortion;
it does not. Its receipts carry `d_seg_per_pair_max`, a **scalar** summary. The naive probe would
have manufactured a gradable row that does not exist.

| grade | count | meaning |
|---|---|---|
| GRADABLE | **0** | matched n600 byte + d_seg + d_pose receipts |
| UNGRADABLE_RATE_ONLY | **1** | byte ledger, no distortion receipt → no denominator |
| UNGRADABLE_NO_PER_PAIR_DATA | **9** | store exists, holds neither |
| UNGRADABLE_NO_STORE | **10** | no retained store matches the slug |

| # | ratio | grade | store prefix | source |
|---|---|---|---|---|
| 1 | 1.0 | NO_STORE | ddm_cn5 | ddm_cn5_arc_consolidation:59 |
| 2 | 1.0 | NO_STORE | ddm_cn5 | ddm_cn5_arc_consolidation:62 |
| 3 | 1.0 | NO_PER_PAIR_DATA | ddm_da1 | ddm_da1_telemetry_decomposition:171 |
| 4 | 1.0 | NO_PER_PAIR_DATA | ddm_js1 | ddm_js1_staging_discriminator:27 |
| 5 | 1.0 | NO_STORE | ddm_na10 | ddm_na10_negative_audit_fresh_laws:550 |
| 6 | 1.0 | NO_STORE | ddm_wf2 | ddm_wf2_waterfill_reprice:379 |
| 7 | 1.0 | NO_STORE | ddm_wj1 | ddm_wj1_cost_error_position_join:158 |
| 8 | 1.0 | NO_STORE | ddm_wj1 | ddm_wj1_cost_error_position_join:293 |
| 9 | 0.99995 | NO_PER_PAIR_DATA | ddm_cw1 | ddm_cw1_win_family_canonicalization:44 |
| 10 | 1.000442 | NO_STORE | ddm_na11 | ddm_na11_negative_regrade:50 |
| 11 | 0.9959 | **RATE_ONLY** | ddm_fs3 | ddm_fs3_jg5_real_price_reopen:180 |
| 12 | 0.98 | NO_PER_PAIR_DATA | ddm_ck1 | ddm_ck1_composed_kneeA:52 |
| 13 | 0.98 | NO_STORE | ddm_cn3 | ddm_cn3_week_coherence_audit:450 |
| 14 | 0.98 | NO_STORE | ddm_deferral | ddm_deferral_queue_ledger:39 |
| 15 | 0.98 | NO_PER_PAIR_DATA | ddm_df1 | ddm_df1_retrain_contamination:301 |
| 16 | 1.02 | NO_STORE | ddm_eq1 | ddm_eq1_equations_lineages_vs_rc2:159 |
| 17 | 0.98 | NO_PER_PAIR_DATA | ddm_tw1 | ddm_tw1_token_waterfill_state_dependence:326 |
| 18 | 1.0213 | NO_PER_PAIR_DATA | ddm_tac1 | ddm_tac1_two_axis_composition:276 |
| 19 | 1.04 | NO_PER_PAIR_DATA | ddm_pa1r | ddm_pa1r_pool_a_race:193 |
| 20 | 1.04 | NO_PER_PAIR_DATA | ddm_qbt2b | ddm_qbt2b_r9_constrained_margin_verdict:35 |

**Not one of the 20 can be graded against its own interval.** Only `ddm_fs3` retains an n600 byte
ledger, and with no distortion receipt it has no denominator, so it yields a rate interval and never
a ΔS verdict.

Two honest boundaries on this table:

1. The probe keys on the `ddm_<slug>` store-naming convention. A store filed under another name is
   invisible to it. That is a recorded blind spot, not a proof of absence.
2. Several of these ratios are not byte↔distortion exchange ratios at all — rank 7 is a count
   enrichment, rank 16 an int12 lattice headroom. `rn1` extracted them as scalars near 1.0, which is
   the right screen for a sweep and the wrong unit for this instrument. Grading them against a ΔS
   interval would be a units error even with the data. This is why the custody grade, not the ratio,
   is the verdict.

**But the prior is strong and it points one way.** All 20 margins lie between 0.000% and 4.00% of
their bar. The only exchange-ratio dispersion we have ever measured is 5.76%. Every one of these
rows is inside it. That is a **prior, not a verdict** — the interval belongs to FCD3's object and
may not be transferred — but the campaign should stop treating a 2–4% margin as a result until the
row carries its own per-pair receipts.

## 5. What was registered

`exchange_ratio_noise_floor_v1` in `.omx/state/canonical_equations_registry.jsonl`, built by
`src/tac/canonical_equations/exchange_ratio_noise_floor_20260903.py`, registered through
`tools/register_exchange_ratio_noise_floor_equation_20260903.py`. It carries the estimand, the law,
the acceptance rule, and all three measurements as `EmpiricalAnchor` rows.

The equation states the law a second time in pure form. Two statements of one law drift, so
`src/tac/tests/test_exchange_ratio_noise_floor_equation.py` is a **drift guard**: it feeds the
equation module and the producer the same inputs and asserts identical draws, identical calibration
constants, identical bootstraps, and identical score arithmetic.

The domain of validity forbids what would break the estimand: site-level resampling (`ddm_fs3`
measured AVERAGE ≠ MARGINAL by 2.24×, so sites inside one pair are not exchangeable) and any
contiguous prefix (pose prefixes measure 2.54–4.21× harder than the population).

## 5a. Retained payload

Everything this arm materialised is on disk under
`/Volumes/VertigoDataTier/pact/ddm_xr1_exchange_ratio_noise_floor/` — 43 artifacts, 65,595,575 B.
The three complete RC64 streams, their archives, their per-frame ledgers and their terminal
checkpoints are all retained; nothing was measured and discarded.

| receipt | bytes | sha256 (16) |
|---|---:|---|
| `RESULT.json` | 4,151 | `096213601e681392` |
| `MANIFEST.json` | 12,060 | `ed7b888d1744c393` |
| `PHYSICAL_REPEATS.json` | 9,896 | `a4b227b2e9e1d918` |
| `BOOTSTRAP.json` | 4,333 | `904ea08cc8eb2319` |
| `RN1_TOP20_REGRADE.json` | 32,975 | `d88d26d1b9602202` |
| `PREFLIGHT.json` | 6,449 | `abbf07b8d3187676` |
| `retained/bootstrap_draw_indices.npy` | 240,128 | `dcf98bac242c35ad` |
| `retained/jbp1_row_a_bootstrap.npz` | 6,972 | `b0715c77ed4400ee` |
| `retained/fcd3_bootstrap.npz` | 42,296 | `ec918d7693e24075` |

The seeded draw matrix is retained as an artifact, so every interval in this memo is
reproducible from the same 200 × 600 index draw rather than from a re-seeded RNG.

## 6. RECALL EVIDENCE

Consulted before designing, not after:

* `ddm_rn1_n600_reopen_sweep_20260903.md` — rank-10 fire order; the 300-row near-win population and
  its `near_win_candidates.jsonl` are this arm's input, read at the pinned sha.
* `ddm_ww1_walls_that_werent_20260902.md` §3.5 — "exchange-ratio noise floor (#1248) — nobody has
  measured it". The gap statement this arm closes.
* `ddm_rxc1_gen3_gate1_verdict_20260901.md` — 64/64 byte-identical exact restarts. This is why σ_B's
  prior is zero and why a nonzero σ_B would have been the finding.
* `ddm_fs3` / memory `m166` — AVERAGE ≠ MARGINAL by 2.24×. This is why the bootstrap resamples
  PAIRS and never sites.
* memory `prefix_bias_sign_inverts_between_seg_and_pose_20260803` / `m88` / `m96` — pose prefixes
  measure 2.54–4.21× harder. This is why the population is the full n600 with seeded random draws.
* memory `[[⛔🎲 FALSIFIERS CONTROLLED BY THE DERIVATION THAT TESTS THEM]]` — see DEAD-ENDS #1.
* memory `[[⛔🔍 NEGATIVE-EXISTENCE = #1 false-claim class]]` — see DEAD-ENDS #2.
* `docs/operating_manual_craft_handoff.md` — verify by RE-DERIVING from primary artifacts. Applied:
  I recomputed FCD3's +0.0019433243907622244 from the two retained scorer receipts before accepting
  the inherited constant. It matched to the last digit.

## 7. DEAD-ENDS (defects found and cured in the inherited code)

A codex arm wrote 788 lines of this instrument before dying to a usage limit. The mechanism was
sound and I kept it. Three defects were not.

1. **The physical stage raised on its own falsifier.** The inherited code computed σ_B and then
   `raise`d if any repeat was not byte-identical. A non-identical repeat is precisely the finding
   the stage exists to catch; raising it deletes it. Cured: `summarize_physical_repeats` RECORDS
   `all_streams_byte_identical`, `prior_law_prediction_held`, and the per-repeat flags, and never
   raises. Two tests pin it, including the subtle case where byte COUNTS agree but bytes differ.
2. **The re-grade asserted "all 20 UNGRADABLE" without looking.** That is negative-existence by
   assertion. Cured: `probe_per_pair_custody` measures it, and the measurement produced a 3-way
   taxonomy the assertion had flattened — `ddm_fs3` is RATE_ONLY, not simply ungradable. The strict
   exact-key test also caught that `d_seg_per_pair_max` is a scalar, which a substring probe scores
   as a per-pair vector.
3. **The exchange ratio raised on a near-zero denominator.** Same class as #1: a sign-crossing
   denominator is a real property of an edit set. Cured: the ratio interval is reported as
   `null` with `exchange_ratio_undefined_reason` when the denominator changes sign.

One more, smaller: the inherited preflight pinned `.omx/tmp/codex_runs/_common_contract.md` as a
required input. `.omx/tmp/` is sanctioned ephemeral scratch, so that pin would have blocked every
future re-run once it was cleaned. Removed; the charter remains pinned.

## 8. LIVE-HYPOTHESES

1. **The floor scales with edit-set size, not with credit size.** JBP1 (5,506 edits) and FCD3
   (4,194 edits + carrier) produced half-widths of 200.5 B and 236.9 B on credits of −2,950 B and
   −2,940 B. Similar credits, similar dispersion. A one-parameter law
   `half_width ≈ 1.96 · sd(per-pair δ) · sqrt(600)` is directly testable against any third retained
   ledger. **Cheap: no new encode, only an existing per-pair ledger.**
2. **The 12.43% ΔS half-width is dominated by d_seg, not by bytes.** ΔS_dist half-width is
   0.000349 against ΔS_rate's 0.000158 — distortion carries 2.2× the dispersion of rate. If that
   holds on a second object, then realized-ΔS claims are limited by scorer-population variance, and
   buying more byte precision buys nothing.
3. **A 200-resample Monte-Carlo residual of 0.34% is the binding precision, not the interval.** For
   decisions at the third decimal of S, the resample count must rise. 2,000 resamples would cost
   seconds and shrink the MC residual roughly 3.2×.

## 9. NEXT_IF_RESUMED

1. **Re-grade the remaining 280 rn1 rows with the same custody probe.** It is mechanical and needs
   no new measurement. The 3-way grade taxonomy is more useful to the campaign than the top-20 slice.
2. **Test hypothesis 1** against `ddm_fs3`'s four retained n600 ledgers. That is the third and
   fourth object for free and would turn the noise floor from two point measurements into a law.
3. **Retro-fit the acceptance rule into the near-win screen.** `rn1`'s `near_win_candidates.jsonl`
   should carry the custody grade as a field, so a future sweep never re-surfaces an ungradable row
   as a candidate.
4. **Owed and NOT done here:** every future arm that reports a realized ΔS must retain the per-pair
   `d_seg` / `d_pose` vectors alongside the byte ledger. Nine of the twenty top rows have a store and
   still cannot be graded. That is retention debt, and it is cheap to stop accruing.

## 10. Pre-existing failures observed, not caused, not fixed

Two test failures exist on `main` independently of this arm and are reported for whoever owns them:

* `src/tac/tests/test_resize_exploit_flip_fix_frontier.py::test_builds_and_is_valid_canonical_equation`
  asserts 2 anchors; commit `74815c790` added a third without updating the test.
* `src/tac/tests/test_check_344_canonical_equation_referenced.py` reports **25** memos with
  empirical-finding tokens and no canonical-equation citation (bound is 5). None are this arm's —
  this memo cites `exchange_ratio_noise_floor_v1` in its frontmatter and in §5.

---

**Own-vehicle frontier: afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED
by this arm.** This is apparatus: it measures how much of a quoted margin is real, and the answer
for every near-win row currently on the board is "less than we have been assuming".
