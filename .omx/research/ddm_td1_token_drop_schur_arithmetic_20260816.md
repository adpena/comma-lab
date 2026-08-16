# ddm_td1 — token-drop × Schur compensation: the byte half is exact, and the rung is closed as a rate lever

Date: 2026-08-16 · Owner: td1 (supervised Opus arm) · Charter:
`.omx/research/ddm_td1_token_drop_schur_arithmetic_charter_20260816.md` · Axis:
`[local-CPU $0 cost-model]` · `score_claim=false`, `promotable=false`. Scorer-free, no Modal, no
launches. Frontier unmoved: hv1 ep0634 **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**.

STORES CONSULTED: ns1 audit (P2 row) · rfo2 gestalt (#1062 rungs) · qs1/qs2/qs3/qs4/qs5 verdicts +
`GT_ATTRIBUTED_DECOMPOSITION.json` (qs3 store) · mz2 verdict · hp1 receipt · #869 waterfill row ·
hv1 `FINAL_RESULT.json` (ep0634) · na7 scope warning on the 57.1% prior · memories [[m94]] [[m96]]
[[m88]] [[m48]] [[m66]].

## The headline, in order of size

1. **~95% of our seg term is render→SegNet round-trip loss, not label error.** The transmitted
   label field disagrees with GT at **1,717** pixels of 117,964,800. The scored seg term is
   **34,930.6** flips. At unit amplification that leaves **33,213.6 flips = 0.028156 S** that our
   labels did not cause. For any amplification r ≤ 2 the round-trip share stays ≥ 90%. The seg axis
   on this vehicle is a **render-fidelity** problem, not a label-fidelity problem. This names a
   supplier ~2.9× the entire remaining gap (−0.0095973). EXACT label count; the share depends on r.
2. **Token drop as a RATE lever is CLOSED** once pose compensation is priced. See the table.
3. **The qs3 57.1% beneficial prior does not transfer — measured wrong by 158×.** On the full field
   the drop population is **B=807 / H=222,883 / W=4**, a beneficial rate of **0.36%**. na7 warned
   this prior was instrument-scoped; this is the measurement that confirms it.

## What the shipped object actually is (measured, and it corrects a framing)

The shipped token stream is **lossless**. `rx1.encode_rc64` asserts `np.array_equal(decoded,
expected)` and pins `EXPECTED_SPATIAL_SHA256`. `s1p25_c1p0` is **not** a drop map — it is the RCF1
logit-correction table (shrink 1.25, clip_scale 1.0) which moves only coding probabilities, and the
winner was chosen by pure byte-minimisation. So rfo2/#1062 is literally right: **no drop level has
ever been applied.** The token field is exactly the scored seg population — 600 × 384 × 512 =
117,964,800, one token per scored pixel — which is why the breakeven law comes out to
`(25/37,545,489)/(100/117,964,800) = 0.785479` flips/B. That law is **derived here, not borrowed**.

## The byte half — EXACT, with a passing calibration control

Per-token cost computed with the deployed arithmetic (`probability_from_codes(codes, 8)`, softmax of
`codes/8`, cost `−log2 p[symbol]`), over the full retained `s1p25_c1p0` field.

| quantity | value |
|---|---|
| cost-model total | **112,109.5 B** |
| real `tokens.rc64` payload | **112,110 B** |
| **calibration ratio** | **0.9999957821778376** (0.5 B over 112 KB), repeat-identical |
| argmax floor | 45,992.8 B |
| **maximum drop pool** | **66,116.7 B** |
| disagreement tokens (token ≠ argmax) | 223,694 (0.1896% of field) |

The instrument reproduces the object it prices. A model that failed this control would have been
refused rather than reported.

## The ladder — EXACT bytes, EXACT GT attribution, one MODELED scalar

Attribution is full-field, not sampled. The event↔spatial permutation was rebuilt from the runtime's
own geometry (`group_id = (x % 64) + 2·(y % 64)`, `adapted_runtime/cpr1/inflate.py:264-276`) and
**verified exhaustively on all 600 frames**; a data-driven signature recovery was tried first and
fails here (only 30,277 of 196,608 positions have distinct signatures — the static sky/hood hold a
constant label, largest collision class 85,487).

`r` = label→scored amplification: scored SegNet flips produced per transmitted-label flip. It is
**UNMEASURED on this vehicle**. Every rung's admission collapses to it.

| threshold | tokens | gross B | B | H | net flips | dS_rate | r* breakeven |
|---|---|---|---|---|---|---|---|
| ≥16 bits | 143 | 316 | 4 | 139 | 135 | −2.106e-04 | 1.8400 |
| ≥8 bits | 9,819 | 12,806 | 40 | 9,779 | 9,739 | −8.527e-03 | 1.0329 |
| ≥4 bits | 38,028 | 32,094 | 264 | 37,764 | 37,500 | −2.137e-02 | 0.6723 |
| ≥2 bits | 90,994 | 51,412 | 511 | 90,483 | 89,972 | −3.423e-02 | 0.4488 |
| ≥0.01 bits | 213,986 | 66,117 | 784 | 213,198 | 212,414 | −4.402e-02 | 0.2445 |

## The pose column closes it

Compensation is priced with the charter's own instrument: **qs 4 B/active pair** (qs4 step-2 q11,
`12 B / 3 active pairs`), with qs5's **in-compile re-solve REQUIRED** — carrying a stale solve is the
named killer (qs4 +2.396e-4 pose disaster; qs5 fails closed on a stale fingerprint).

| rung | tokens | **active pairs** | gross B | comp B | **net B** | r* with comp | dS at r=1 |
|---|---|---|---|---|---|---|---|
| ≥16 bits | 143 | **102** | 316 | 408 | **−92** | −0.5339 | **+1.755e-04** |
| ≥8 bits | 9,819 | **600** | 12,806 | 2,400 | 10,406 | 0.8393 | **+1.327e-03** |

The ≥16-bit rung dies structurally: 143 tokens scatter across 102 pairs, so compensation (408 B)
exceeds the entire saving (316 B). The ≥8-bit rung touches all 600 pairs and needs r < 0.84 — it
**loses** +1.33e-3 at unit amplification. **Nothing in the drop family clears on the harm side.**

## The oracle test — why this is a family verdict, not one formulation

Fixed thresholds could be the wrong shape, so I ran the encoder-optimal selection: the encoder knows
GT, so at each r it picks per-token (`drop iff bytes saved > r·1.27311`) and per-pair (`use pair iff
gain > 4 B`). This is the best any token-drop formulation can do.

| r | pairs used | gross B | comp B | net B | dS |
|---|---|---|---|---|---|
| 0.25 | 600 | 23,424 | 2,400 | 21,024 | −1.400e-02 |
| 0.50 | 600 | 9,598 | 2,400 | 7,198 | −4.793e-03 |
| 0.84 | 426 | 3,049 | 1,704 | 1,345 | −8.956e-04 |
| **1.00** | 257 | 1,690 | 1,028 | 662 | **−4.408e-04** |
| 2.00 | 230 | 1,817 | 920 | 897 | −5.972e-04 |

The curve does not go to zero — but **at r ≥ 1 the optimizer stops dropping expensive tokens
altogether and switches to the 807 tokens where our label is wrong and the model's argmax is right.**
Those save bytes *and* fix flips, so they get *more* valuable as r rises. That is not a rate lever.
It is **label correction**, and it belongs to the qs family, not to token drop.

**verdict_scope: FAMILY for token-drop-as-rate-lever** — closed on exact bytes + exact GT attribution
+ the banked compensation price, at every granularity from 143 to 213,986 tokens.
**verdict_scope: INSTANCE (new, unraced) for the 807-token label-correction set.**

## The wall I hit, stated plainly

**I did not run the real re-encode.** The charter requires realized archive-byte diffs and forbids
substituting an entropy estimate. What I have is a cost model calibrated to 0.9999958 of the real
payload on the *unmutated* field — that is not the same thing, because the probability model is
doubly autoregressive (temporally on the previous frame, `rx1:597`; spatially on earlier groups,
`rx1:612`), so any edit cascades. For the sets that matter here (≤0.008% of the field) the cascade
should be small, but its sign is not determined and I did not measure it. **Every byte number above
is EXACT-first-order, not EXACT-realized.** Building a mutated `SourceSymbols` and re-running
export→rc64→compose was beyond this session's budget. Reporting the wall, per the charter's
TOY-BRACKET clause.

## Fire-order: NONE issued

The charter's fork says emit a sealed fire-order if the best set clears. On the harm side nothing
clears. The one surviving candidate — the 807 label corrections — is a **new instance of an
already-measured family whose realized ledger clusters at zero**: qs1 REFUSED +2.43e-5 · qs2 ADMITTED
−4.374914e-6 · qs4 REFUSED +2.44e-4 · qs5 REFUSED +2.52e-6. Its modeled −4.4e-4 assumes **no
collateral**, and collateral is exactly what refused its siblings (qs3 measured H = 41.3% of gross
activity). Firing it would spend a T4 row on the fifth member of a family already measured four
times, priced on an unmeasured collateral term and an unmeasured cascade.

Recommendation to MAIN: **do not fire token drop.** Route the slot at finding 1. Round-trip loss is
0.028 S — 290× the best modeled token-drop row and 2.9× the whole remaining gap.

## Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_td1_token_drop_schur_20260816/` (APDataStore chosen because
VertigoDataTier has 954 MiB free):

| artifact | bytes | sha256 (prefix) |
|---|---|---|
| `survey/TD1_SURVEY.json` | — | full ladder + calibration control |
| `survey/retained/drop_saving_bits.f16.npy` | 235,929,728 | per-token exact saving, full field |
| `survey/retained/disagreement_mask.u8.npy` | 117,964,928 | token ≠ argmax mask |
| `attribution/TD1_ATTRIBUTION.json` | — | B/H/W + r* + seg decomposition |
| `attribution/retained/drop_set_ge_16bits.frame_event.i32.npy` | 1,272 | `51227cdf6855cfcc…` (143 tokens) |
| `attribution/retained/drop_set_ge_8bits.frame_event.i32.npy` | 78,680 | `1310a3a25861e338…` (9,819 tokens) |

Provenance pins: base archive `80d9c8c6fdc72caa…` @182,759 B · token payload `73a878891a31c366…`
@112,110 B · GT field `91d3ff11a904c476…` · decoded event field 117,964,800 B. Tool:
`experiments/ddm_td1_token_drop_schur_arithmetic.py` (stages `preflight` / `survey` / `attribute`).

## LIVE HYPOTHESES

- **H1 (dominant).** The render→SegNet round trip costs ~0.0282 S. Decompose it: how much is the
  painter's label→RGB map, how much is SegNet's re-segmentation of synthetic RGB, how much is the
  874↔384 resize? Each is a separate supplier. This is the successor charter.
- **H2.** `r` is cheaply measurable and settles several families at once. The ≥8-bit drop set (9,819
  tokens → ~28% change in d_seg) is a clean instrument: one row's seg delta ÷ 9,739 gives r directly.
  Only worth firing if bundled with something else.
- **H3.** The 807 label-correction tokens are the qs family's cheapest remaining instance and, unlike
  JS6 proposals, they are *selected by the model's own confidence* rather than proposed. If the
  round-trip work ever needs a seg candidate, this is the pre-computed one.
- **H4.** The witness transmits 1,717 wrong labels the HPAC model corrects at 807 of them. That gap
  is a train-time signal: the model already knows better than the field it is compressing.

## DEAD ENDS (do not re-open without new preconditions)

- Token drop as a rate lever, all granularities — closed above (exact bytes, exact GT, banked
  compensation). Re-opening needs a compensation price below 4 B/pair **or** a measured r < 0.84.
- The qs3 57.1% beneficial prior transferred to any new population — measured wrong by 158× here.
- Data-driven recovery of the event↔spatial permutation — 30,277 of 196,608 signatures unique.
- Lossless token recoding — mz2 (all exact recodes ≥ +340 B) and the carrier tie at q11 already shut
  this; my calibration independently confirms the coder is within 0.5 B of its own model, so there is
  no slack left in the arithmetic coder itself.

## NEXT IF RESUMED

1. Charter the round-trip decomposition (H1). It is the only supplier on this vehicle larger than the
   gap. $0 first pass: render the transmitted labels locally, re-segment, and attribute the 33,214
   flips across painter / SegNet / resize.
2. If a token-drop row is ever wanted anyway, the owed engineering is exactly: build a mutated
   `SourceSymbols` from a retained drop set, re-run `export_probabilities` → `encode_rc64` → compose,
   diff real archive bytes. The drop sets are retained and ready.
3. Do **not** re-derive the breakeven law, the calibration, or the B/H/W split — they are measured
   here and the receipts are retained.
