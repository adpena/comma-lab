# ddm_rp1 ROUND 2 on the move-40 field — the K curve, measured, and the stop rule it fixes

Tokens: `[no-triality] [p0-ledger-ok]` · Arm ddm_rp1 · Base: **pointer move 40,
`ddm_sj1_t4_compose39_rp1_union_20260910`, S 0.13763861019288715 @ 180,233 B, sha
`986d536b…`** (the composition of this arm's move 39 with sj1's 42-pair pass-5 subset).
Gap to sub-0.12 re-derived at this base: **0.017638610 S = 26,490.7 B.**
Every number below states which field it is on. `score_claim=false`.

## 1. The base, verified rather than accepted

The composed field is **exactly this arm's move-39 field plus 78 tokens over 42 pairs** —
checked cell by cell, not inferred from the handoff. Move 40's sections: hpac 11,911 and
semantic 29,862 byte-identical to move 37; carrier 18,586 → **18,592** (sj1's re-solve);
tail 119,704 → **119,754**. Its TC1M rider carries a **119,618 B** stream, sha `648168db…`.

**Identity control on the new base PASSED**: this arm's instrumented encode of the move-40
field emitted **119,624 B**, byte-identical to the envelope reconstructed from move 40's own
shipped rider body. The prices below are the live coder's prices on the live field.

**Move-40 census** (n600, exact): 956,936.5 bits = 119,617.07 B; **99.8009 %** of tokens are
the coder's argmax; the **234,880 mispredicted tokens carry 83,104.5 B = 69.48 %** of the
tail; first-order ceiling ≥ 0.5 bits = **68,795.4 B** over 191,708 positions. The object is
the same shape as on move 37, one composition later.

## 2. THE K CURVE — measured on 24 seeded pairs, 6,144 realized proposals, cumulative mode

Round 1 measured neutrality at 4.36 % and FLAT over ranks 0–32 and asked what happens deeper.
This is the answer, on the move-40 field:

| rank band | tests | neutral | frac | bits | bits/test | cum bits/pair | n600 first-order |
|---|---|---|---|---|---|---|---|
| 0–8 | 192 | 9 | 4.69 % | 91.5 | **0.477** | 3.81 | 286 B |
| 8–16 | 192 | 10 | 5.21 % | 71.3 | 0.371 | 6.78 | 509 B |
| 16–32 | 384 | 12 | 3.12 % | 73.6 | 0.192 | 9.85 | 739 B |
| 32–64 | 768 | 26 | 3.39 % | 123.1 | 0.160 | 14.98 | 1,124 B |
| 64–128 | 1,536 | 63 | 4.10 % | 193.9 | 0.126 | 23.06 | 1,730 B |
| 128–192 | 1,536 | 69 | 4.49 % | 133.6 | 0.087 | 28.63 | **2,147 B** |
| 192–256 | 1,536 | 73 | 4.75 % | 86.3 | **0.056** | 32.23 | 2,417 B |

**Total neutral 262 / 6,144 = 4.26 %.**

**Finding 1 — neutrality is flat to rank 256, not just to 32.** It sits between 3.12 % and
5.21 % with no trend across an eight-fold extension of K. What decays is the PRIZE: bits per
realized test falls **0.477 → 0.056, a 8.5× decay**, entirely because the coder's expensive
tokens are ranked first. So raising K buys real bytes and buys them at a steadily worse rate;
it never hits a wall and it never gets cheap.

**Finding 2 — the in-loop joint check removed the failure it was built for.** In cumulative
mode the post-hoc composite disagreed on **0 of 24 pairs**, against **12.9 % of edited pairs**
needing a greedy narrowing in round 1's singles mode. Making the joint check the acceptance
does not merely detect the interaction, it prevents it.

## 3. PRE-REGISTERED STOP RULE (fixed before the n600 launch; commit order is the receipt)

Cost is 600 pairs x K x 0.72 s over 5 shards. Marginal first-order bytes per extra
shard-hour, and the same at round 1's measured selected realization ratio 0.2663:

| step | Δ first-order | Δ shard-hours | B/h first-order | **real B/h** | **ΔS per hour** |
|---|---|---|---|---|---|
| 32 → 64 | +385 B | 0.77 | 500 | 133 | 8.9e-05 |
| 64 → 128 | +606 B | 1.54 | 394 | 105 | 7.0e-05 |
| 128 → 192 | +417 B | 1.54 | 271 | 72 | **4.8e-05** |
| 192 → 256 | +270 B | 1.54 | 175 | 47 | **3.1e-05** |

**RULE: continue while the marginal band returns ≥ 2x the 2e-05 admit bar per shard-hour
(≥ 4e-05 S/h). K = 192 is the last band that clears it; 192 → 256 returns 3.1e-05 S/h and is
refused.** So the n600 runs at **K = 192**, projecting **2,147 B first-order → ≈ 572 B real
→ ΔS ≈ −3.81e-04** at 4.6 shard-hours. That is 2.3x round 1's realized move, on a base
2.5e-04 lower.

The rule is written here BEFORE the launch so the K choice cannot be rationalised afterwards
from whatever the run returns.

## 4. What is still owed

n600 acceptance at K=192 (cumulative), carrier re-solve on the changed pairs from move 40's
own coefficients, **frame-0 8-mode re-selection inside the admission** for every pair whose
resolved pose is still above base (encoder written and round-tripped: the shipped selector is
34 B for 24 non-identity pairs; marginal 2.00 B for the first adopted pair, 0.84 B/pair at
+44), the three-column pose print (stale / carrier-resolved / frame-0-reselected), the twin
re-encode of the selected subset against move 40's tail, and the seal.

## 5. Decode wall-clock — what this arm can honestly contribute to the budget gate

tc4's row did not score: `inflate.sh` timed out at 1,800 s on the T4, so decode wall-clock is
now a live gate. **This candidate does not change the receiver** — the stage step ASSERTS the
staged tree differs from the pointer in exactly `['archive.zip', 'inflate.py']`, the second
only in its two pins — so its decode work is the pointer's decode work.

Four CPU parse-backs already measured on this same receiver lineage, all with
`wall_clock_seconds` in their own receipts:

| run | seconds | vs the 1,800 s budget |
|---|---|---|
| sj1 pass 2a | **2,101.7** | **OVER** |
| sj1 pass 3 | 705.7 | under |
| ddm_rp1 move 39 | 1,036.4 | under |
| sj1 compose39 (move 40) | 1,047.8 | under |

**Two cautions the budget field has to carry or it will lie.**

1. **A 3× spread on one object.** 705.7 s and 2,101.7 s are the SAME receiver decoding
   near-identical archives on the same machine. The variance is concurrent load — these runs
   overlapped 4–6 other shards — not anything about the bytes. A single local timing is
   therefore not a verdict: a budget field measured under contention refuses valid candidates,
   and one measured on a quiet machine passes candidates that will not survive a loaded runner.
   It needs the concurrent-process count beside the seconds, or a quiesced measurement.
2. **It is the wrong device.** The contest budget is T4 CUDA. This receiver REFUSES the CPU
   path on purpose — `inflate.sh` raises "requires CUDA inflation on linux-nvidia-t4; the
   measured CPU path exceeded the 1,800-second contest budget" — which is exactly what this
   arm's public smoke records as `REACHED_CUDA_GATE` for candidate and frontier alike. So a
   macOS-CPU parse-back time is an advisory number on a different device, and quoting it
   against the 1,800 s budget compares two things that are not the same measurement.

What this arm will put in its seal notes is therefore the honest pair: its own parse-back
seconds WITH the concurrent-shard count, and the paired candidate/frontier public-entrypoint
outcomes showing both reach the same gate at the same time — never a bare number implying a
contest-budget verdict this instrument cannot give.

## 6. THE N600 RUN WAS VERIFIED ON THE WRONG BASE (respawn, 2026-09-10 ~10:00Z)

All five shards finished rc=0 and their arithmetic is internally consistent: **4,503 neutral
tokens of 115,200 tested (3.9089 %), 15,552.3 first-order bits = 1,944.0 B** over all 600
pairs, close to §3's pre-registered 2,147 B projection. But the acceptance was not measured
on move 40's field.

`ddm_rp1_sizing.py` takes its two objects from two different places:

| object | source | round 2's value |
|---|---|---|
| the RANKING | `--rank-dir` — a **flag** | built on `field_move40.u8` — correct |
| the BASE FIELD | `rp1.load_live_field()` — a **source constant** | sj1's pass-4 npz = **move 37's field** |

`sizing.py:65` reads `rp1.load_live_field()`, which is pinned to
`admission_pass4/field_admitted.npz` (sha `813bf1e6…`) behind a hardcoded `changed != 9_209`
refusal, and the `sizing` subcommand has no field flag at all. So every "argmax identical on
all 196,608 cells" verdict came from a body the pointer stopped shipping two moves ago.

**The receipts said so and nobody read them that way.** Each shard's
`pricing_field.tokens_changed_vs_pristine` reads 10,045–10,180 with
`tokens_changed_vs_live_field` 836–971 — the difference is 9,209 every time, and 9,209 is
pass-4's count. Move 40's field is pass-4 plus 473 tokens (rp1 move 39) plus 78 (sj1 pass 5)
over ~295 pairs.

**Law (new, general).** *A harness whose ranking is field-overridable and whose base field is
a source constant will silently verify on the wrong body.* A flag and a constant cannot
disagree loudly; only two flags can. Same genus as the carrier silent-revert and sj1's
subset-writer revert — the guard existed on one object and not on the one beside it.

## 7. WHY THE 4.6 SHARD-HOURS ARE RECOVERABLE, AND HOW

**What survives.** The rank computes `sj1_edit_mask = target != base` on the **override**
field and drops every masked position from the proposals (`order = order[~is_sj1_edit]`), so
no accepted proposal sits on a banked token. At every proposed position the two bases
therefore carry the **same** symbol, and `(pos, sym → best)` is the identical edit on either
body. That is now checked per edit against the `sym` the rank priced it against, not argued.

**What does not survive.** The render CONTEXT. The renderer's receptive field is 9 token
cells, so a banked edit near a proposal moves both the base argmax and the edited argmax.
Joint neutrality has to be re-measured on the shipping base.

**The repair** is `experiments/ddm_rp1_rebase.py` (commit `daadbf7a2`): re-verify the accepted
set on a named base field with the same cumulative bisection the acceptance used, so a set
that fails is NARROWED rather than dropped whole. Cost is ~2 realizations per pair instead of
192 — minutes, against the 4.6 shard-hours a re-run would take. It is **conservative by
construction**: it can lose yield, never invent it (a proposal rejected on pass-4 that would
be neutral on move 40 is not recovered), so its number is a floor on this pass's yield.

**It carries its own control group.** Pairs whose two base planes are identical must transfer
at fraction 1.0; the receipt reports that fraction separately from the exposed pairs'. If the
control is not 1.0 then the instrument moved, not the base, and the rebase is not a
measurement.

`ddm_rp1_sizing.py` is deliberately NOT patched: another arm may be running it, and an edit
to a live arm's script re-binds its checkpoints (`[[landing_a_gate_into_a_live_arms_script_
breaks_its_checkpoint_binding_20260910]]`). The structural cure belongs to whoever owns the
class: *a harness whose ranking is field-overridable must take its base field from the same
flag, or refuse when the two disagree.*

## 8. THE CHAIN FROM HERE, AND WHAT IT IS ALLOWED TO CLAIM

Base overlay reuse, verified not assumed: sj1's `compose39_price/pose/cand_overlay` is the
600-plane render of `field_composed.npz`, which is move 40's field — so it is this arm's BASE
overlay, and only the candidate's renders are owed. `verify-field` proves `field_move40.u8`
and that npz are the same 600 planes before either is used, because two files in two formats
cannot be compared by sha and a drift between them would put the pose leg on one body and the
rate leg on the other.

The pose base is still MEASURED here on the pointer's own carrier codes over that overlay
(the pose-base law); sj1's 4.886129e-06 is the number it must land near, never the number it
inherits.

Order: verify-field → rebase ×5 → merge → real twin encode of the rebased field (ledger +
stream) → render candidate overlay → pose base / stale → carrier re-solve on the changed
pairs → resolved → **frame-0 8-mode re-selection inside the admission**, on the pairs the
admission would drop for pose → admit on the RESOLVED pose → twin re-encode of the SELECTED
subset (the sum ranks, it does not charge) → stage → close → parse-back + seg identity → seal
with move 40's inherited decode-wall-clock leg.

**Re-derived at move 40** (binding numbers expire at every pointer move): gap to sub-0.12 is
0.01763861019288715 S = **26,490.0 B** at 6.658589531221714e-07 S/B; the rate corner is an
archive ≤ 153,743.0 B. At round 1's measured selected-realization ratio 0.2663 the pre-rebase
1,944.0 B first-order projects to ≈ 518 B, i.e. **≈ −3.45e-04 S, about 2.0 % of the corner** —
and the rebase can only take from that. This pass is a step down the rate axis, not a door.

## 8b. PRE-REGISTERED REBASE STOP RULE AND FALSIFIERS (written before the rebase ran)

Commit order is the receipt: none of the numbers below exist yet.

**Stop rule on the rebase's own output.**

| condition | verdict |
|---|---|
| `context_identical_transfer_fraction` < 1.0 | **STOP.** Pairs whose two base planes are identical must transfer every token. If they do not, the INSTRUMENT moved between the acceptance and the rebase, and neither run is a measurement. Report, do not narrow. |
| kept first-order ≥ 700 B | continue the full chain |
| 200 B ≤ kept < 700 B | continue; at ratio 0.2663 this is ≥ 53 B real ≈ −3.5e-05 S, still above the −2e-05 bar, but the pose leg now decides |
| kept < 200 B | continue ONLY if the resolved pose is at or below base; otherwise the pass cannot clear the bar and the rebase itself is the finding |
| kept < 75 B | **ABANDON** the chain. Report the base-mismatch law and the recovered floor; do not spend a T4 row on it |

**The exposure is known EXACTLY before the rebase runs, from receipts alone (no CPU).**
`ddm_sj1_pass5_price/compose39/COMPOSE.json` records `union_pairs 282`, `rp1_tokens 473`,
`my_tokens 78`, `position_collisions 0`, and — the useful part — `composed_u8_sha256
b50da438…`, which is **byte-identical to `ROUND2_BASE.json`'s `field_move40_u8.sha256`.**
So `field_move40.u8` IS sj1's `composed.u8`, and falsifier 4 is already carried by
provenance before `verify-field` confirms it by bytes.

Intersecting the two admissions' own `kept_pairs.json` (253 ∪ 42, overlap 13):

| | pairs |
|---|---|
| base plane DIFFERS from the acceptance base (exposed) | **282** |
| base plane IDENTICAL (the control group) | **318** |

So roughly 47 % of the field is exposed and 53 % must transfer perfectly. The rebase receipt
reports those two transfer fractions separately, and both numbers above were derived and
committed before it ran.

**Falsifiers, pre-registered for the seal.**

1. Pairs whose base plane is unchanged between the two bodies transfer at fraction **1.0**.
2. Every accepted edit's base symbol equals the `sym` the rank priced it against (refuses in-loop, per edit).
3. The merged field differs from the move-40 base at exactly `tokens_kept` positions — no collision, no no-op edit.
4. `field_move40.u8` and sj1's `compose39/field_composed.npz` are identical on all 600 planes (they cannot be compared by sha; the pose leg rides one and the rate leg the other).
5. NULL BUILD: this arm's own control body, re-packed into the pointer's member, reproduces move 40's member byte for byte.
6. Twin encodes of the same field emit byte-identical streams, so any byte delta is a measurement and not run-to-run variance.
7. The staged tree differs from the pointer in exactly `['archive.zip', 'inflate.py']`, the second only in its two pins — which is what makes move 40's decode-wall-clock leg inheritable.
8. Seg identity on the parse-back: 0 differing cells over 117,964,800, and the candidate's local flip count equals the base decode's local count (delta exactly 0 — the absolute local number is advisory and is NOT the T4 number).
9. The pose base, measured here on the pointer's own carrier codes, lands inside pm2's [1/3, 3]× band of the pointer's 4.89e-06 print.
10. The admission consumes the RESOLVED pose; stale is printed beside it and is never an input.
11. The shipped archive's bytes come from a REAL encode of the selected subset; the ledger-sum prediction is reported beside it with its gap (sj1 measured +19.6 B / +0.0108 % on its own subset — the sum ranks, it under-charges).

## 8c. THE REBASE, MEASURED (2026-09-10 ~12:00Z) — the control group came back exactly 1.0

`rebase/REBASE.json`, field `aae528e6…`, no pair hit the verify cap.

| group | pairs | tokens proposed | tokens kept | transfer |
|---|---|---|---|---|
| base plane IDENTICAL (**the control**) | 307 | 2,401 | 2,401 | **1.000000** |
| base plane DIFFERS (exposed) | 281 | 2,102 | 1,741 | 0.828259 |
| all | 588 | 4,503 | 4,142 | **0.919831** |

**The control group is the whole verdict.** 2,401 of 2,401 — every token on a pair whose two
base planes agree transferred, so the instrument did not move between the acceptance and the
rebase. The pre-registered STOP is not triggered, and the 361 lost tokens are attributable to
the base and to nothing else. The exposed/control split (281 / 307 among pairs that carry
edits) matches the 282 / 318 predicted from receipts alone in §8b.

**The cost of the defect, priced:** 361 tokens = 1,499.5 first-order bits = **187.4 B**, i.e.
17.2 % of the exposed group and 9.6 % of the pass. Kept: 4,142 tokens =
14,052.737 bits = **1,756.6 B first-order**, well above §8b's 700 B continue threshold.
422 of 588 pairs transferred whole.

**Two cross-arm controls passed on the way.**

1. `verify-field`: `field_move40.u8` and sj1's `field_composed.npz` are identical on all 600
   planes, **0 differing cells**.
2. This arm's re-render of move 40's odd frames reproduces sj1's recorded overlay sha
   **`37ea3842…` exactly** — two arms, two runs, byte-identical renders.

**A P0 custody finding, not this arm's:** control 2 was forced, not chosen. sj1's move-40
overlays were cold-stored at 04:56Z to
`/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/…/{base,cand}_overlay/odd_frames.u8`
and **those files do not exist** — the destination directories contain only ExFAT `._`
stubs, while `MOVE_LOG.jsonl` records the moves as `MOVED` with their shas. The same pattern
covers `parseback/0.raw` and three `ddm_sj1_multipass_token_predistortion` overlays
(1.83 GB each). Certify-or-block says a manifest without the bytes is a blocker, not a
cleanup. Here it cost 5 minutes because the render is deterministic and cheap; on an artifact
that is not reproducible it would have been unrecoverable.

**Pose base, measured on this arm's own instrument** over move 40's own carrier codes and
renders: **4.88709198352764e-06** (pose leg 0.006990774), against sj1's 4.886599506e-06 —
+0.010 %, comfortably inside pm2's [1/3, 3]× gate. Measured, never inherited.

## 8d. THE PRICE INVERTED — the K-curve's promise is not a charge, and at K=192 it flips SIGN

Twin encodes of the whole rebased field, byte-identical (120,244 B envelope / 120,238 B body,
sha `27813d91…`, both twins):

| | bytes |
|---|---|
| control (move 40's own stream) | 119,624 |
| candidate (all 4,142 kept edits) | 120,244 |
| **delta** | **+620** |

The first-order ranking promised **−1,756.6 B**. The real coder charged **+620 B**. Not a
decayed yield — an **inverted sign**.

Round 1 at K=32 returned −144.65 B on 838 changes (ratio 0.1445, still a saving). Round 2 at
K=192 returns +620 B on 4,142. **§2's K-curve measured only the first-order prize decaying
8.5×; what it could not see is that the realized price crosses zero somewhere above K=32.**
The deep ranks are not merely cheap, they are net-harmful: removing a low-surprise token
re-prices its neighbourhood by more than it saves.

Per-pair, from the two real ledgers: of 576 edited pairs, **146 are savers (−115.38 B) and
430 are losers (+734.79 B)**. Encoder drift on the 324 untouched pairs is +0.96 B and is
pinned out of the sweep rather than charged to this arm.

**So the pass survives only because it is a per-pair SELECTION, not a field.** This is the
sharpest confirmation yet of [[first_order_token_price_is_a_ranking_never_a_charge…]]: at
K=192 the ranking does not merely over-promise, it points the wrong way.

## 8e. THE ADMISSION on the RESOLVED pose

Three-column pose leg over all 600 pairs, this arm's own instrument:

| column | d_pose mean | vs base |
|---|---|---|
| base (move 40's carrier on move 40's renders) | 4.88709198e-06 | — |
| **stale** (candidate renders, shipped carrier) | **1.05512811e-03** | **216× worse** |
| **carrier-resolved** (576 pairs re-solved, all adopted) | **4.98084259e-06** | +1.9 % |

The re-solve recovers 99.99 % of the stale damage; the residual +1.9 % is what the sweep
then prices per pair.

**Sweep result (`admission/ADMISSION.json`), ledger-sum selection:**

| | value |
|---|---|
| pairs kept / offered | **160 / 576** |
| tokens kept | **1,054** of 4,142 |
| rate delta (ledger sum) | **−43.98 B** → −2.93e-05 S |
| d_pose mean | **4.64947689e-06** (BELOW base) → −1.721e-04 S |
| d_seg | 0 by construction |
| **net ΔS vs move 40** | **−2.0134828823e-04 = 10.1× the bar** |

The win is mostly **pose**, not rate: the carrier re-solve on the selected pairs lands the
pose *below* base, the same shape as round 1's −5.16e-5.

## 8f. FRAME 0, run INSIDE the admission — measured, priced, NOT adopted

Target: the post-re-solve residual tail among edited pairs, including all 35 pose-bound drops
(pairs whose rate is a saver and whose pose killed them, stranding −21.90 B = −1.458e-05 S).

- 120 pairs swept, 8 modes each, on the candidate's own renders with the re-solved codes.
- **38 of 120 (31.7 %) have a better mode** — against the 13 % population baseline sj1
  measured. **All 17 the greedy blob-price adopted fell OUTSIDE the kept set.** Both facts
  confirm sj1 §30's law exactly: frame 0 is a repair lever for pairs an edit BROKE, not a
  population win.
- Re-admitting with those pose values recovers **3 pairs**: 160 → 163, ΔS −2.0135e-04 →
  −2.0463e-04.
- The selector blob for exactly those 3, ENCODED and round-tripped through the receiver's own
  `decode_selector`: 34 B → 36 B, **+2 B = +1.332e-06 S**.
- **Frame-0 net effect: −1.950e-06 S = 0.0975× the bar.**

**Not adopted, and the reason is not that it failed.** It is 1 % of the candidate, and buying
it costs a fresh ~27-minute subset re-encode plus a carrier selector-splice path this arm has
not verified exists (`ddm_rp1_build.close` splices CODES only). Putting the seal at risk for
1 % is the wrong trade. Receipt: `frame0/FRAME0_VERDICT.json`.

## 9. Frontier line

`ddm_sj1 compose39+rp1 union S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]` (move 40)
