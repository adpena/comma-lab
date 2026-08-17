# ddm_bu1 — the bank union was never held. It already fired, already won, and is already inside the frontier.

Date: 2026-08-17 · Arm: `ddm_bu1_bank_union_compile`
Axis: `[recall + exact arithmetic + custodied-byte sha re-verification]` · `score_claim: false`
Pointer moved: `false` · Modal dispatched: `false` · scorer run: `false` · spend: **$0**

## THE ANSWER

**The qs2 × re1 union does not need compiling. It was compiled on 2026-08-14 as `ddm_mc35`,
repaired as `ddm_mc36` Variant C, fired on T4, and PROMOTED at ΔS = −2.068040e-5. The current
hv1 ep0634 frontier carries it.** There is no candidate to build and no fire order to seal.

The charter's premise — *"admitted rows are sitting banked and unfired ... that hold has never
been revisited against the current base"* — is false at source. The hold WAS revisited, three
days ago, by the arm that owns the successor lane. This is the stale-headline genus
(`corrections_land_in_bodies_headlines_keep_the_stale_number_20260805`): the qs2 and re1 memos
still say **BANKED, HELD**, and the mc36 memos' bank note ("qs2 + re1 do NOT auto-compose onto
MC36") reads as *still available* when it means *do not blindly transfer the compensation
objects* — which is precisely what mc35 did not do.

## THE PROOF — subsumption at the EVENT ID, not merely the pair index

`.omx/research/ddm_mc35_micro35_union_build_20260814.md` names the built object outright:
*"The exact union contains the six QS2 token objects, RE1's admitted singleton at pair 96, and
the smallest distinct-pair sign-verified neighbor at pair 7."*

| bank | event id | pair | measured Seg gain | in the promoted mc36? |
|---|---|---:|---:|---|
| QS2 | `js6_0000_9fbf75d81c43` | 105 | +4 | yes (compensation fresh-solved) |
| QS2 | `js6_0072_f790b6493122` | 176 | +1 | yes |
| QS2 | `js6_0006_92685b3e3e44` | 178 | +3 | yes |
| QS2 | `js6_0004_06fc74e20d9e` | 517 | +12 | yes |
| QS2 | `js6_0001_da319a6b65d0` | 523 | +14 | yes |
| QS2 | `js6_0118_83f376603d6e` | 532 | **−2** | **no — measured harmful, dropped** |
| RE1 | `ec1_0164_3a4e239de5b9` | 96 | +2 | yes — **this IS re1 Round-1** |
| RE1 | `ec1_0004_3bc2b69c706c` | 7 | +1 | yes |

mc36 Variant C's runtime parse-back recovers compensation pairs `[7, 96, 105, 176, 178, 517, 523]`.
That set is exactly **(qs2's six MINUS pair 532) ∪ (re1's 96 and 7)**. The event ids match the
qs1 fourteen-pair census (`ddm_qs1_frame0_schur_coupled_solve_20260813.md:71` — "The six compiled
pair indices are 105, 176, 178, 517, 523, 532") and the re1 round table
(`ddm_re1_realization_engineered_candidate_20260813.md:14-15`) verbatim.

**Leg survival against the base change — the question the charter asked:**

| leg | survives onto hv1? | status |
|---|---|---|
| qs2, 5 of 6 events | n/a | **already in the base**, with better (joint) compensation |
| qs2, pair-532 event | no | measured NET HARMFUL: −2 flips. Dropping it MEASURED +2 flips and +21 B saved |
| re1 Round-1, pair 96 | n/a | **already in the base** |
| re1 Round-2, pair 7 | n/a | **already in the base** |
| re1 Round-2, pair 73 | no positive evidence | mc35 declined it as not sign-verified; re1 Round-2 with it projected −1.016e-6 vs Round-1's −1.228e-6 without it |
| qs5 in-compile Schur | n/a | **already consumed** — mc36 Variant A "first ran QS5's exact-object DLS/int12 solve" |

## hv1 CARRIES IT — now MEASURED, no longer assumed

The hv1 arm built a pure rate object on frozen distortion: it re-codes mc36's *fixed* token
stream with a better-trained HPAC model, admitted only under exact decoded-token identity plus
full raw-output byte identity. Its d_seg/d_pose were **inherited**, and the hv1 memo honestly
flagged the device-determinism carry-across as an assumption.

The T4 row closes it. `experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`
(call `fc-01M036FY225QC9A75CM0Y7X7NP`, `passed: true`):

```
avg_segnet_dist  = 0.00029611     <- identical to mc36's measured value
avg_posenet_dist = 6.88e-06       <- identical to mc36's measured value
archive_size_bytes = 182759
score_recomputed_from_components = 0.15959729295498598
```

Components reproduce the pointer exactly: `0.029611000 + 0.008294577 + 0.121691716 = 0.15959729295498598`.
Identical distortion at 3,510 fewer bytes. **hv1 is mc36's frames at a smaller rate**, so whatever
is inside mc36's frames is inside the frontier. Both archives re-verified on disk this turn:
mc36 `f0ba4bb4…` @ 186,269 B, hv1 `80d9c8c6…` @ 182,759 B — both match their memos.

## THE ARITHMETIC, DONE BEFORE ANY COMPUTE

Exact marginals: one net Seg flip = `100/117,964,800` = 8.477105034722222e-7 S;
one archive byte = `25/37,545,489` = 6.658589531221714e-7 S.

| composition | seg | pose | rate | net | vs 1e-5 bar |
|---|---:|---:|---:|---:|---|
| qs2 (cp135) | −2.712674e-5 | +1.126177e-7 | +34 B = +2.26392e-5 | **−4.374918e-6** | 43.7% |
| re1 (cp135) | −1.695421e-6 | +4.885472e-7 | 0 B | **−1.206874e-6** | 12.1% |
| naive union, non-interacting | −2.882216e-5 | +6.011649e-7 | +2.26392e-5 | **−5.581792e-6** | 55.8% — **FAILS** |
| union + qs5 Schur, pose fully cured | −2.882216e-5 | 0 | +2.26392e-5 | **−6.182957e-6** | 61.8% — **FAILS** |
| union, pose AND rate both free | −2.882216e-5 | 0 | 0 | −2.882216e-5 | 288% — physically unreachable |

**Granting every leg its best measured value, and granting qs5's proven in-compile compensation
in full, the union tops out at −6.18e-6 = 61.8% of the naming bar.** The charter's hold was
arithmetically correct. Firing it would have bought a sub-band row we could not name. The only
row in that table that clears the bar requires the compensation object to be free — and making
compensation cheap is exactly what mc36 attacked, by a different route.

## WHAT THE JOINT RE-SOLVE ACTUALLY BOUGHT — the transferable finding

mc36 beat the naive union by **3.705×** (−2.068040e-5 realized vs −5.581792e-6 projected). It did
not stack two independent compensation objects. It fresh-Schur-solved compensation **jointly** over
the final composed 7-object rendered stream, dropped the measured-harmful pair 532, and added
re1's pair 7.

| | qs2 alone | mc36 joint re-solve |
|---|---:|---:|
| net flips | 32 | **37** |
| archive delta | +34 B | **+17 B** |
| B/pair | 5.67 | **2.43** |
| flips/B | 0.941 | **2.18** |
| clears the 0.7855 flips/B breakeven by | 1.20× | **2.77×** |

**Law for the next micro-edit arm: never price a union as the sum of its legs' compensation
objects.** The compensation is a joint object over the composed frame; solving it once for the
union cost 2.3× fewer bytes per pair than carrying the legs' own objects, and gained flips at the
same time. This is the rate lever — not the HPAC token recode, which lives in a different archive
section (the −3,128 B hv1 saved came off the 115,238 B token stream, not off the 96 B residual
that holds the compensation overlay).

## WHAT THIS IS WORTH AGAINST THE GOAL — stated plainly

hv1 S = 0.15959729295498598. Gap to 0.15 = **0.00959729**.

- The union, at the value it actually realized: **0.2155% of the remaining gap.**
- The union, had it been available at the naive −5.58e-6 this arm was chartered to compile:
  **0.0582% of the remaining gap.**
- For scale, hv1's single rate recode was −2.337e-3 — **24× the entire union**, from re-coding a
  frozen distortion with a better probability model and zero new distortion risk.

The honest read: micro-edit unions on this vehicle are worth ~0.06–0.2% of the gap each and are
now exhausted on their measured support. Rate representation on frozen distortion is currently
paying two orders of magnitude better per unit of work.

## VERDICT

**`UNION_ALREADY_SUBSUMED_BY_THE_CURRENT_FRONTIER_NO_FIRE_ORDER`.**
`verdict_scope: INSTANCE` — the specific qs2 × re1 bank union named in this charter, against the
hv1 ep0634 base. This is **not** a family verdict on micro-edits: other supports, other edit
objects, and the proven joint-compensation mechanism all remain open. The residual bank items are
measured empty: pair 532 is harmful, pair 73 was never sign-verified.

No fire order is sealed, because no candidate exists. Rebuilding mc36 would re-buy a row we own.

## Payload law

This arm materialized **zero** payload bytes. No archive was compiled, no field was rendered, no
scorer ran. Nothing existed in memory to discard. `MEASURE_ONLY_OK: adjudication arm, zero-byte
materialization by construction.` The durable artifacts are this memo and
`/Volumes/APDataStore/pact/ddm_bu1/ADJUDICATION_RECEIPT.json`, which carries every event id, sha,
and arithmetic row above in machine-readable form. The two archives named here were re-verified
read-only by sha256 and were not moved or modified.

## Apparatus defect this exposed

Two memos still advertise **BANKED, HELD** for rows that were consumed three days later
(`ddm_qs2_r2_admitted_verdict_20260813.md`, `ddm_re1_round1_dual_axis_verdict_20260814.md`), and
the mc35 memo that consumed them does not back-reference them. A charter was then written on the
stale headline and an arm was spawned to rebuild something we already owned. The cure is the one
already named in `charter_recall_validation_is_apparatus_not_volition_20260816`: consumption must
be recorded at the CONSUMED row, not only at the consuming row. A bank entry needs a
`consumed_by` field that the spawn-site recall check reads.

## Recall evidence (read at source this turn)

`ddm_mc35_micro35_union_build_20260814.md` (event table, compensation pairs, custody) ·
`ddm_mc36_micro35_variants_20260814.md` (Variant A/B/C gates, parse-back pair set) ·
`ddm_mc36_promotion_complete_s_verdict_20260814.md` (promoted row) ·
`ddm_qs2_r2_admitted_verdict_20260813.md` · `ddm_qs2_compensation_rate_rung_20260813.md` ·
`ddm_qs1_frame0_schur_coupled_solve_20260813.md` (the six compiled pair indices) ·
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` (in-compile Schur proof, breakeven law) ·
`ddm_re1_round1_dual_axis_verdict_20260814.md` · `ddm_re1_realization_engineered_candidate_20260813.md`
(round 1/2 event ids and pairs) · `ddm_eu4_fresh_eyes_fractal_composition_20260813.md`
(exact marginals, "no compensation vector is a reusable asset across edit objects") ·
`ddm_hv1_harvest_compose_ep508_20260815.md` · `ddm_hv1_ep0634_t4_fire_execution_20260815.md` ·
`ddm_hv1_t4_sealed_fire_order_ep0634_20260815.json` · `.omx/state/canonical_frontier_pointer.json` ·
`experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`.

Own-vehicle frontier line: **hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4, n600]`,
UNMOVED this turn** (this arm ran no scorer and sought no move).
